//! Bounded abstract memory admission and reservation accounting.
//!
//! This module tracks admission decisions only; it never allocates backend memory. Callers
//! serialize mutations by holding their own lock or by owning the ledger mutably. Snapshots are
//! supplied by the caller, and unknown telemetry rejects a constrained admission request.

/// Stable caller-assigned identity for one memory pool.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PcuMemoryPoolId(pub u32);

/// Telemetry for a memory usage measure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PcuMemoryUsage {
    Known(u64),
    Unknown,
}

/// Point-in-time capacity and usage facts for one pool.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PcuMemoryPoolSnapshot {
    pub id: PcuMemoryPoolId,
    pub capacity_bytes: Option<u64>,
    pub system_used_bytes: PcuMemoryUsage,
    pub process_used_bytes: PcuMemoryUsage,
    /// Ledger reservation bytes already included in `system_used_bytes` for this snapshot.
    pub system_ledger_reserved_bytes: u64,
    /// Ledger reservation bytes already included in `process_used_bytes` for this snapshot.
    pub process_ledger_reserved_bytes: u64,
}

/// Integer ratio `numerator / denominator`, with no floating-point rounding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PcuMemoryRatio {
    pub numerator: u64,
    pub denominator: u64,
}

impl PcuMemoryRatio {
    #[must_use]
    pub const fn new(numerator: u64, denominator: u64) -> Self {
        Self {
            numerator,
            denominator,
        }
    }

    const fn is_valid(self) -> bool {
        self.denominator != 0 && self.numerator <= self.denominator
    }

    fn permits(self, used: u64, capacity: u64) -> bool {
        u128::from(used) * u128::from(self.denominator)
            < u128::from(capacity) * u128::from(self.numerator)
    }
}

/// Usage measure constrained by a maximum fraction of pool capacity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PcuMemoryUsageMode {
    SystemUsed,
    ProcessUsed,
}

/// Hard upper bound applied to one usage measure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PcuMemoryLimit {
    pub mode: PcuMemoryUsageMode,
    pub max_fraction: PcuMemoryRatio,
}

/// Whether a failed admission should be retried later or treated as a hard rejection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PcuMemoryDisposition {
    Reject,
    Defer,
}

/// Specific reason an admission request could not be reserved.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PcuMemoryAdmissionReason {
    InvalidFraction,
    UnknownCapacity,
    UnknownUsage(PcuMemoryUsageMode),
    RepresentedReservationsExceedLedger {
        mode: PcuMemoryUsageMode,
        represented_bytes: u64,
        ledger_bytes: u64,
    },
    ReservationAccountingOverflow,
    LimitExceeded {
        mode: PcuMemoryUsageMode,
        projected_used_bytes: u64,
        maximum_used_bytes: u64,
    },
    ReservationSlotsExhausted,
    ReservationIdsExhausted,
}

/// Structured admission failure with its retry disposition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PcuMemoryAdmissionError {
    pub pool: PcuMemoryPoolId,
    pub disposition: PcuMemoryDisposition,
    pub reason: PcuMemoryAdmissionReason,
}

/// Opaque handle for releasing one successful reservation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PcuMemoryReservation {
    pool: PcuMemoryPoolId,
    slot: usize,
    generation: u64,
}

impl PcuMemoryReservation {
    #[must_use]
    pub const fn pool(self) -> PcuMemoryPoolId {
        self.pool
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Entry {
    pool: PcuMemoryPoolId,
    bytes: u64,
    generation: u64,
}

/// Fixed-capacity ledger for abstract reservations across memory pools.
///
/// Mutating methods require `&mut self`, so exclusive ownership or a caller-supplied lock can
/// serialize admission and release. Reservations are counted on top of the supplied snapshot
/// usage for both system and process limits, preventing this ledger's own concurrent clients
/// from each spending the same reported headroom. Snapshot overlap fields prevent reservations
/// already reflected in refreshed telemetry from being counted twice.
pub struct PcuMemoryReservationLedger<const N: usize> {
    entries: [Option<Entry>; N],
    next_generation: u64,
}

impl<const N: usize> PcuMemoryReservationLedger<N> {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            entries: [None; N],
            next_generation: 1,
        }
    }

    /// # Errors
    ///
    /// Returns an admission error when the snapshot or limit is invalid, capacity is exhausted,
    /// or the ledger cannot record another reservation.
    pub fn reserve(
        &mut self,
        snapshot: PcuMemoryPoolSnapshot,
        bytes: u64,
        limit: PcuMemoryLimit,
    ) -> Result<PcuMemoryReservation, PcuMemoryAdmissionError> {
        self.reserve_with_limits(snapshot, bytes, [Some(limit), None])
    }

    /// Atomically admits one allocation against zero, one, or both independent usage limits.
    ///
    /// A single reservation is recorded and therefore is counted once in the ledger, even when
    /// both system and process limits are enabled.
    ///
    /// # Errors
    ///
    /// Returns an admission error when telemetry is insufficient, a limit would be crossed, or
    /// reservation accounting cannot proceed.
    pub fn reserve_with_limits(
        &mut self,
        snapshot: PcuMemoryPoolSnapshot,
        bytes: u64,
        limits: [Option<PcuMemoryLimit>; 2],
    ) -> Result<PcuMemoryReservation, PcuMemoryAdmissionError> {
        let reject = |reason| PcuMemoryAdmissionError {
            pool: snapshot.id,
            disposition: PcuMemoryDisposition::Reject,
            reason,
        };
        let defer = |reason| PcuMemoryAdmissionError {
            pool: snapshot.id,
            disposition: PcuMemoryDisposition::Defer,
            reason,
        };

        let mut local_reserved = 0u64;
        for entry in self.entries.iter().flatten() {
            if entry.pool == snapshot.id {
                local_reserved = local_reserved.checked_add(entry.bytes).ok_or_else(|| {
                    reject(PcuMemoryAdmissionReason::ReservationAccountingOverflow)
                })?;
            }
        }
        let has_limits = limits.iter().any(Option::is_some);
        if has_limits {
            let Some(capacity) = snapshot.capacity_bytes else {
                return Err(reject(PcuMemoryAdmissionReason::UnknownCapacity));
            };
            for limit in limits.into_iter().flatten() {
                if !limit.max_fraction.is_valid() {
                    return Err(reject(PcuMemoryAdmissionReason::InvalidFraction));
                }
                let (reported, represented_reservations) = match limit.mode {
                    PcuMemoryUsageMode::SystemUsed => (
                        snapshot.system_used_bytes,
                        snapshot.system_ledger_reserved_bytes,
                    ),
                    PcuMemoryUsageMode::ProcessUsed => (
                        snapshot.process_used_bytes,
                        snapshot.process_ledger_reserved_bytes,
                    ),
                };
                let PcuMemoryUsage::Known(reported_used) = reported else {
                    return Err(reject(PcuMemoryAdmissionReason::UnknownUsage(limit.mode)));
                };
                if represented_reservations > local_reserved {
                    return Err(reject(
                        PcuMemoryAdmissionReason::RepresentedReservationsExceedLedger {
                            mode: limit.mode,
                            represented_bytes: represented_reservations,
                            ledger_bytes: local_reserved,
                        },
                    ));
                }
                let projected = reported_used
                    .checked_add(local_reserved - represented_reservations)
                    .and_then(|used| used.checked_add(bytes))
                    .ok_or_else(|| {
                        reject(PcuMemoryAdmissionReason::ReservationAccountingOverflow)
                    })?;
                let maximum = u64::try_from(
                    (u128::from(capacity) * u128::from(limit.max_fraction.numerator))
                        / u128::from(limit.max_fraction.denominator),
                )
                .map_err(|_| reject(PcuMemoryAdmissionReason::ReservationAccountingOverflow))?;
                if !limit.max_fraction.permits(projected, capacity) {
                    return Err(defer(PcuMemoryAdmissionReason::LimitExceeded {
                        mode: limit.mode,
                        projected_used_bytes: projected,
                        maximum_used_bytes: maximum,
                    }));
                }
            }
        }

        let Some(slot) = self.entries.iter().position(Option::is_none) else {
            return Err(defer(PcuMemoryAdmissionReason::ReservationSlotsExhausted));
        };
        let generation = self.next_generation;
        self.next_generation = self
            .next_generation
            .checked_add(1)
            .ok_or_else(|| reject(PcuMemoryAdmissionReason::ReservationIdsExhausted))?;
        self.entries[slot] = Some(Entry {
            pool: snapshot.id,
            bytes,
            generation,
        });
        Ok(PcuMemoryReservation {
            pool: snapshot.id,
            slot,
            generation,
        })
    }

    /// Releases a prior reservation and returns the number of bytes released.
    ///
    /// # Errors
    ///
    /// Returns `InvalidReservation` for a stale, foreign, or already released handle.
    pub fn release(
        &mut self,
        reservation: PcuMemoryReservation,
    ) -> Result<u64, PcuMemoryReleaseError> {
        let Some(entry) = self.entries.get_mut(reservation.slot) else {
            return Err(PcuMemoryReleaseError::InvalidReservation);
        };
        match *entry {
            Some(value)
                if value.pool == reservation.pool && value.generation == reservation.generation =>
            {
                *entry = None;
                Ok(value.bytes)
            }
            _ => Err(PcuMemoryReleaseError::InvalidReservation),
        }
    }

    #[must_use]
    pub fn reserved_bytes(&self, pool: PcuMemoryPoolId) -> Option<u64> {
        self.entries
            .iter()
            .flatten()
            .filter(|entry| entry.pool == pool)
            .try_fold(0u64, |total, entry| total.checked_add(entry.bytes))
    }
}

impl<const N: usize> Default for PcuMemoryReservationLedger<N> {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PcuMemoryReleaseError {
    InvalidReservation,
}

/// Access the consumer may perform through this allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PcuMemoryAccess {
    ReadOnly,
    WriteOnly,
    ReadWrite,
}

/// Host mapping and transfer operations requested by the consumer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PcuMemoryHostAccess {
    /// The consumer only needs provider-mediated transfers.
    TransferOnly,
    /// The consumer may request a scoped host mapping.
    Mapping,
}

/// Typed request to allocate one backend-owned memory resource.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PcuMemoryAllocationRequest {
    pub pool: PcuMemoryPoolId,
    pub size_bytes: u64,
    pub alignment_bytes: u64,
    pub access: PcuMemoryAccess,
    pub host_access: PcuMemoryHostAccess,
    /// If true, allocation must fail unless device-local placement is affirmatively established.
    pub require_device_local: bool,
}

impl PcuMemoryAllocationRequest {
    /// Checks the provider-independent constraints before admission or allocation.
    ///
    /// # Errors
    ///
    /// Returns `ZeroSize` or `InvalidAlignment` when the request violates those constraints.
    pub const fn validate(self) -> Result<(), PcuMemoryRequestError> {
        if self.size_bytes == 0 {
            return Err(PcuMemoryRequestError::ZeroSize);
        }
        if self.alignment_bytes == 0 || !self.alignment_bytes.is_power_of_two() {
            return Err(PcuMemoryRequestError::InvalidAlignment);
        }
        Ok(())
    }
}

/// Provider-independent allocation request validation failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PcuMemoryRequestError {
    ZeroSize,
    InvalidAlignment,
}

/// A byte range within a memory resource, expressed without pointer assumptions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PcuMemoryRange {
    pub offset_bytes: u64,
    pub size_bytes: u64,
}

impl PcuMemoryRange {
    /// Returns the exclusive end offset, or `None` if the range overflows.
    #[must_use]
    pub const fn checked_end(self) -> Option<u64> {
        self.offset_bytes.checked_add(self.size_bytes)
    }
}

/// Provider knowledge about whether two resource ranges refer to the same backing bytes.
///
/// `Unknown` must be treated as potentially overlapping wherever disjoint storage is required.
/// This matters for imported resources and views whose allocation identity is not exposed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PcuMemoryOverlap {
    Disjoint,
    Overlapping,
    Unknown,
}

/// Optional storage behavior a provider can affirm for a resource.
///
/// Capabilities are opt-in: a missing implementation never implies that an operation is safe.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PcuMemoryResourceCapability {
    /// Backing may be reused after the caller has established that prior users are quiescent.
    ReusableStorage,
    /// Provider can copy between resources without exposing their addresses.
    ResourceCopy,
}

/// Requirements for one member of a reusable bank; members may have different sizes/access.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PcuMemoryMemberRequirement {
    pub pool: PcuMemoryPoolId,
    pub minimum_size_bytes: u64,
    pub access: PcuMemoryAccess,
    pub require_device_local: bool,
}

/// Why a set of resources cannot satisfy a reusable bank requirement.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PcuMemoryBankError {
    Empty,
    WrongMemberCount {
        expected: usize,
        actual: usize,
    },
    WrongPool {
        index: usize,
        expected: PcuMemoryPoolId,
        actual: PcuMemoryPoolId,
    },
    TooSmall {
        index: usize,
        required: u64,
        actual: u64,
    },
    InsufficientAccess {
        index: usize,
        required: PcuMemoryAccess,
        actual: PcuMemoryAccess,
    },
    ReuseUnsupported {
        index: usize,
    },
    DeviceLocalNotEstablished {
        index: usize,
    },
    Overlapping {
        first: usize,
        second: usize,
    },
    OverlapUnknown {
        first: usize,
        second: usize,
    },
    InvalidExtent {
        index: usize,
    },
}

/// Validates a reusable bank with independent requirements for each member.
///
/// Unknown overlap is rejected: a provider must prove disjoint backing for every pair. The
/// caller remains responsible for ensuring prior device work is complete before reusing a slot.
///
/// # Errors
///
/// Returns a structured error for an empty or mismatched bank, an unsatisfied member
/// requirement, or overlap that is present or cannot be ruled out.
pub fn validate_reusable_memory_bank_members<R: PcuMemoryResource>(
    resources: &[R],
    requirements: &[PcuMemoryMemberRequirement],
) -> Result<(), PcuMemoryBankError> {
    validate_reusable_memory_bank_members_by(resources, requirements, |resource| resource)
}

/// Validates a bank whose entries contain a resource that can be borrowed through `resource_of`.
/// This lets wrappers keep their leases in place while the contract inspects provider metadata.
///
/// # Errors
///
/// Returns the same structured contract failures as
/// [`validate_reusable_memory_bank_members`].
pub fn validate_reusable_memory_bank_members_by<T, R: PcuMemoryResource>(
    members: &[T],
    requirements: &[PcuMemoryMemberRequirement],
    resource_of: impl Fn(&T) -> &R,
) -> Result<(), PcuMemoryBankError> {
    if members.is_empty() {
        return Err(PcuMemoryBankError::Empty);
    }
    if members.len() != requirements.len() {
        return Err(PcuMemoryBankError::WrongMemberCount {
            expected: requirements.len(),
            actual: members.len(),
        });
    }
    for (index, (member, requirement)) in members.iter().zip(requirements).enumerate() {
        let resource = resource_of(member);
        if resource.pool() != requirement.pool {
            return Err(PcuMemoryBankError::WrongPool {
                index,
                expected: requirement.pool,
                actual: resource.pool(),
            });
        }
        if resource.size_bytes() < requirement.minimum_size_bytes {
            return Err(PcuMemoryBankError::TooSmall {
                index,
                required: requirement.minimum_size_bytes,
                actual: resource.size_bytes(),
            });
        }
        if !memory_access_satisfies(resource.access(), requirement.access) {
            return Err(PcuMemoryBankError::InsufficientAccess {
                index,
                required: requirement.access,
                actual: resource.access(),
            });
        }
        if !resource.supports(PcuMemoryResourceCapability::ReusableStorage) {
            return Err(PcuMemoryBankError::ReuseUnsupported { index });
        }
        if requirement.require_device_local && resource.is_device_local() != Some(true) {
            return Err(PcuMemoryBankError::DeviceLocalNotEstablished { index });
        }
        if requirement.minimum_size_bytes == 0 {
            return Err(PcuMemoryBankError::InvalidExtent { index });
        }
    }
    for second in 1..members.len() {
        for first in 0..second {
            let first_resource = resource_of(&members[first]);
            let second_resource = resource_of(&members[second]);
            let first_range = PcuMemoryRange {
                offset_bytes: 0,
                size_bytes: requirements[first].minimum_size_bytes,
            };
            let second_range = PcuMemoryRange {
                offset_bytes: 0,
                size_bytes: requirements[second].minimum_size_bytes,
            };
            match first_resource.overlap(second_resource, first_range, second_range) {
                PcuMemoryOverlap::Disjoint => {}
                PcuMemoryOverlap::Overlapping => {
                    return Err(PcuMemoryBankError::Overlapping { first, second });
                }
                PcuMemoryOverlap::Unknown => {
                    return Err(PcuMemoryBankError::OverlapUnknown { first, second });
                }
            }
        }
    }
    Ok(())
}

const fn memory_access_satisfies(actual: PcuMemoryAccess, required: PcuMemoryAccess) -> bool {
    match (actual, required) {
        (_, PcuMemoryAccess::ReadOnly) => matches!(
            actual,
            PcuMemoryAccess::ReadOnly | PcuMemoryAccess::ReadWrite
        ),
        (_, PcuMemoryAccess::WriteOnly) => matches!(
            actual,
            PcuMemoryAccess::WriteOnly | PcuMemoryAccess::ReadWrite
        ),
        (PcuMemoryAccess::ReadWrite, PcuMemoryAccess::ReadWrite) => true,
        _ => false,
    }
}

/// Whether importing an external allocation transfers its ownership.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PcuMemoryImportOwnership {
    /// The importing provider retains a valid lease/reference and leaves the original owner intact.
    Borrowed,
    /// The import consumes the external handle according to the provider's documented protocol.
    Transferred,
}

/// Accounting and lifetime origin of a resource returned by a memory provider.
///
/// This describes who is responsible for the backing allocation's accounting and lifetime. It
/// does not assert that `size_bytes` corresponds to resident physical bytes: providers may return
/// lazy or framework-managed resources. An imported borrowed resource remains externally owned;
/// a transferred import is owned by the provider after successful import.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PcuMemoryResourceOrigin {
    /// The provider created or adopted the resource and is responsible for its lifetime.
    ProviderManaged,
    /// The resource remains owned by an external party whose import lease is retained.
    ExternallyOwnedBorrowed,
}

/// Backend-defined, already validated external memory import descriptor.
///
/// The descriptor must carry any OS/API ownership lease needed to keep the imported allocation
/// valid. A raw integer/pointer with no such lease is not sufficient for a safe implementation.
pub trait PcuMemoryImportDescriptor {
    fn pool(&self) -> PcuMemoryPoolId;
    fn size_bytes(&self) -> u64;
    fn alignment_bytes(&self) -> u64;
    fn access(&self) -> PcuMemoryAccess;
    fn ownership(&self) -> PcuMemoryImportOwnership;
}

/// Backend-owned resource returned by allocation or import.
///
/// Implementations own the native resource and must release it only when safe. If backend
/// completion is uncertain, the implementation must retain/quarantine the native allocation
/// rather than release storage that may still be in use. `is_device_local == false` means the
/// provider can affirm the resource is not device-local; `None` means it cannot establish that.
pub trait PcuMemoryResource {
    fn pool(&self) -> PcuMemoryPoolId;
    fn size_bytes(&self) -> u64;
    fn alignment_bytes(&self) -> u64;
    fn access(&self) -> PcuMemoryAccess;
    fn is_device_local(&self) -> Option<bool>;
    /// Identifies the accounting/lifetime owner for this resource. Implementers must state this
    /// explicitly so a borrowed import cannot silently appear provider-managed.
    fn origin(&self) -> PcuMemoryResourceOrigin;

    /// Whether the provider explicitly supports an optional storage behavior.
    fn supports(&self, _capability: PcuMemoryResourceCapability) -> bool {
        false
    }

    /// Classifies overlap between two byte ranges in resources of this provider's type.
    ///
    /// Implementations may return `Disjoint` only when they can prove distinct backing storage
    /// or nonintersecting ranges of the same storage. Out-of-bounds or overflowing ranges must
    /// return `Unknown`; callers must validate extents separately before use.
    fn overlap(
        &self,
        _other: &Self,
        _self_range: PcuMemoryRange,
        _other_range: PcuMemoryRange,
    ) -> PcuMemoryOverlap {
        PcuMemoryOverlap::Unknown
    }
}

/// Scoped mapped view. The view must not outlive the provider's mapping guard.
pub trait PcuMemoryMapping {
    fn as_bytes(&self) -> &[u8];
    fn as_bytes_mut(&mut self) -> Option<&mut [u8]>;
}

/// Stable operation tag for structured provider errors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PcuMemoryProviderOperation {
    Snapshot,
    Allocate,
    Import,
    Map,
    TransferTo,
    TransferFrom,
    CopyResource,
}

/// Reason a provider operation failed. Backends should map native failures to the closest honest
/// category; the IR does not infer retry safety from platform-specific status codes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PcuMemoryProviderFailure {
    /// The pool or resource is temporarily busy and a later attempt may succeed.
    Busy,
    Unsupported,
    InvalidRequest(PcuMemoryRequestError),
    PoolUnavailable,
    /// The device generation backing this pool/resource is gone. Existing handles are stale;
    /// callers must rediscover and re-import or reallocate rather than retry this handle.
    DeviceLost,
    DeviceLocalRequired,
    IncompatibleImport,
    OutOfMemory,
    RangeOutOfBounds,
    AccessDenied,
    MappingUnavailable,
    ResourceContractViolation,
    BackendFailure,
}

/// Structured memory-provider failure with an explicit scheduling disposition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PcuMemoryProviderError {
    pub pool: PcuMemoryPoolId,
    pub operation: PcuMemoryProviderOperation,
    pub disposition: PcuMemoryDisposition,
    pub failure: PcuMemoryProviderFailure,
}

/// Abstract backend memory service.
///
/// This contract exposes pool telemetry, allocation/import, scoped mappings and explicit byte
/// transfers. It does not promise physical contiguity, residency, cache coherence, synchronization,
/// or device-local placement beyond what the provider can affirm for a returned resource. The
/// caller performs ratio admission with [`PcuMemoryReservationLedger`] before allocation: reserve
/// using the matching pool snapshot, call the provider, then release the reservation if the
/// provider fails. On success, keep the reservation accounted until telemetry represents those
/// bytes or the resource is released. These reservations account for requested logical bytes;
/// they do not reserve externally owned backing or guarantee that a backend's hidden workspace is
/// included in telemetry. Providers must not silently convert a device-local request into a weaker
/// placement. For imported resources, borrowed imports remain charged to their external owner;
/// transferred imports become provider-managed after successful import. Callers that need ratio
/// admission for an import must explicitly reserve the imported size and retain that reservation
/// with the resource, since `import` itself has no ledger parameter.
///
/// `Defer` means retry may succeed after pressure/busy state changes; `Reject` means changing
/// timing alone is not expected to help. Providers should use `DeviceLost` with `Reject` for a
/// vanished device generation. A lost device invalidates pool snapshots and resource handles,
/// but does not prove that submitted work has stopped accessing backing memory. Implementations
/// must retain or quarantine resources until quiescence is known; ledger reservations likewise
/// remain held until the consumer can safely release them. Re-discovery creates a new identity or
/// generation and never revives an old resource handle.
pub trait PcuMemoryProvider {
    type Resource: PcuMemoryResource;
    type ImportDescriptor: PcuMemoryImportDescriptor;
    type Mapping<'a>: PcuMemoryMapping
    where
        Self: 'a;

    /// Reads telemetry for the requested stable pool identity.
    ///
    /// # Errors
    ///
    /// Returns a provider error if the pool is unavailable or telemetry cannot be read.
    fn snapshot(
        &self,
        pool: PcuMemoryPoolId,
    ) -> Result<PcuMemoryPoolSnapshot, PcuMemoryProviderError>;

    /// Allocates a provider-owned resource in the requested pool.
    ///
    /// # Errors
    ///
    /// Returns a provider error if the request cannot be satisfied.
    fn allocate(
        &mut self,
        request: PcuMemoryAllocationRequest,
    ) -> Result<Self::Resource, PcuMemoryProviderError>;

    /// Imports a provider-specific external allocation while retaining the descriptor's required
    /// ownership lease in the returned resource.
    ///
    /// # Errors
    ///
    /// Returns a provider error if the import is unsupported or incompatible.
    fn import(
        &mut self,
        descriptor: Self::ImportDescriptor,
    ) -> Result<Self::Resource, PcuMemoryProviderError>;

    /// Maps a checked range for the duration of the returned guard. Providers that cannot provide
    /// a safe scoped mapping return `MappingUnavailable`.
    ///
    /// # Errors
    ///
    /// Returns a provider error if mapping is unsupported, unsafe, or the range is invalid.
    fn map<'a>(
        &'a mut self,
        resource: &'a mut Self::Resource,
        range: PcuMemoryRange,
    ) -> Result<Self::Mapping<'a>, PcuMemoryProviderError>;

    /// Copies bytes into a resource without exposing its address space.
    ///
    /// Access to the borrowed host slice must be finished before this call returns, including on
    /// error. A provider that submits asynchronous DMA must stage the bytes into owned storage or
    /// use a separate owned-transfer API; it cannot retain this borrow after return.
    ///
    /// # Errors
    ///
    /// Returns a provider error if access is denied or the range cannot be transferred.
    fn transfer_to(
        &mut self,
        resource: &mut Self::Resource,
        offset_bytes: u64,
        bytes: &[u8],
    ) -> Result<(), PcuMemoryProviderError>;

    /// Copies bytes out of a resource without exposing its address space.
    ///
    /// Access to the borrowed host slice must be finished before this call returns, including on
    /// error. A provider that submits asynchronous DMA must establish completion before return or
    /// use a separate owned-transfer API; it cannot retain this borrow after return.
    ///
    /// # Errors
    ///
    /// Returns a provider error if access is denied or the range cannot be transferred.
    fn transfer_from(
        &mut self,
        resource: &Self::Resource,
        offset_bytes: u64,
        bytes: &mut [u8],
    ) -> Result<(), PcuMemoryProviderError>;

    /// Copies a prefix of one resource into another, completing before return.
    ///
    /// Providers must reject overlapping or uncertain alias relationships unless they can
    /// explicitly guarantee correct overlapping-copy semantics. A provider without this
    /// capability returns `Unsupported`; callers must not silently stage through host memory.
    ///
    /// # Errors
    ///
    /// Returns a structured provider error for unsupported copies, invalid resources or ranges,
    /// access violations, uncertain aliasing, or native copy/completion failure.
    fn copy_resource(
        &mut self,
        destination: &mut Self::Resource,
        source: &Self::Resource,
        size_bytes: u64,
    ) -> Result<(), PcuMemoryProviderError> {
        let _ = (destination, source, size_bytes);
        Err(PcuMemoryProviderError {
            pool: source.pool(),
            operation: PcuMemoryProviderOperation::CopyResource,
            disposition: PcuMemoryDisposition::Reject,
            failure: PcuMemoryProviderFailure::Unsupported,
        })
    }
}

/// Optional independent ratio limits for system and process usage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct PcuMemoryAdmissionPolicy {
    pub system_used: Option<PcuMemoryRatio>,
    pub process_used: Option<PcuMemoryRatio>,
}

/// Consumer-selected placement and admission requirements for a PCU allocation.
///
/// The pool and backend remain selected by the caller. This policy only adds requirements to an
/// individual allocation request; it does not choose a device, pool, or fallback provider.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct PcuMemoryResourcePolicy {
    /// Require the provider to affirm that the allocation is device-local.
    pub require_device_local: bool,
    /// Optional system and process usage limits enforced by the reservation ledger.
    pub admission: PcuMemoryAdmissionPolicy,
}

impl PcuMemoryResourcePolicy {
    /// Applies this policy to a provider-independent allocation request.
    #[must_use]
    pub const fn apply(
        self,
        mut request: PcuMemoryAllocationRequest,
    ) -> PcuMemoryAllocationRequest {
        request.require_device_local = self.require_device_local;
        request
    }
}

/// Set of up to two ledger reservations. A combined system/process ratio admission uses one
/// reservation so the pending allocation is counted only once.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PcuMemoryReservationSet {
    entries: [Option<PcuMemoryReservation>; 2],
}

impl PcuMemoryReservationSet {
    const fn new() -> Self {
        Self { entries: [None; 2] }
    }

    const fn push(&mut self, reservation: PcuMemoryReservation) {
        if self.entries[0].is_none() {
            self.entries[0] = Some(reservation);
        } else {
            self.entries[1] = Some(reservation);
        }
    }

    /// Returns the number of active reservation handles in this set.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.iter().flatten().count()
    }

    /// Returns whether this set contains no reservations.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Releases all reservations, retaining any handle whose release failed for retry.
    ///
    /// # Errors
    ///
    /// Returns the first invalid reservation error and keeps its handle for retry.
    pub fn release_all<const N: usize>(
        &mut self,
        ledger: &mut PcuMemoryReservationLedger<N>,
    ) -> Result<(), PcuMemoryReleaseError> {
        for entry in &mut self.entries {
            if let Some(reservation) = *entry {
                ledger.release(reservation)?;
                *entry = None;
            }
        }
        Ok(())
    }
}

/// Resource paired with its abstract admission reservations.
///
/// The reservation set remains live alongside the backend resource. Use [`Self::release`] after
/// the backend resource is safe to release, or [`Self::into_parts`] to take responsibility for
/// both explicitly. Dropping this wrapper drops the resource and reservation handles but cannot
/// mutate the external ledger; callers must therefore avoid dropping it before accounting has
/// been reconciled or explicitly transferred. Callers must also keep the reservation until all
/// aliases and in-flight work using the resource are quiescent: dropping this wrapper cannot
/// detect backend-specific cloned resource handles.
pub struct PcuAdmittedResource<R> {
    resource: Option<R>,
    reservations: PcuMemoryReservationSet,
}

impl<R> PcuAdmittedResource<R> {
    /// Borrows the resource while it remains admitted.
    ///
    /// # Panics
    ///
    /// Panics if the resource has already been removed during release.
    #[must_use]
    pub const fn resource(&self) -> &R {
        self.resource
            .as_ref()
            .expect("resource is present until release")
    }

    /// Mutably borrows the resource while it remains admitted.
    ///
    /// # Panics
    ///
    /// Panics if the resource has already been removed during release.
    pub const fn resource_mut(&mut self) -> &mut R {
        self.resource
            .as_mut()
            .expect("resource is present until release")
    }

    #[must_use]
    pub const fn reservations(&self) -> &PcuMemoryReservationSet {
        &self.reservations
    }

    /// Transfers the resource and reservation set to the caller.
    ///
    /// # Panics
    ///
    /// Panics if the resource has already been removed during release.
    pub fn into_parts(mut self) -> (R, PcuMemoryReservationSet) {
        let resource = self
            .resource
            .take()
            .expect("resource is present until release");
        (resource, self.reservations)
    }

    /// Drops the backend resource before releasing its admission reservations.
    ///
    /// If a ledger release fails, remaining reservation handles are returned so the caller can
    /// retry against the correct ledger. This only releases abstract accounting; the backend
    /// resource's own drop implementation defines physical deallocation behavior.
    ///
    /// # Errors
    ///
    /// Returns the failed ledger release and remaining reservation handles for recovery.
    pub fn release<const N: usize>(
        mut self,
        ledger: &mut PcuMemoryReservationLedger<N>,
    ) -> Result<(), PcuAdmittedResourceReleaseError> {
        drop(self.resource.take());
        self.reservations
            .release_all(ledger)
            .map_err(|error| PcuAdmittedResourceReleaseError {
                error,
                reservations: self.reservations,
            })
    }
}

/// Failure releasing an admitted resource's ledger accounting after its resource was dropped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PcuAdmittedResourceReleaseError {
    pub error: PcuMemoryReleaseError,
    pub reservations: PcuMemoryReservationSet,
}

/// Failure while sequencing validation, ratio admission, and backend allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PcuMemoryAllocateWithPolicyError {
    InvalidRequest(PcuMemoryRequestError),
    Snapshot(PcuMemoryProviderError),
    SnapshotPoolMismatch {
        requested: PcuMemoryPoolId,
        reported: PcuMemoryPoolId,
    },
    Admission(PcuMemoryAdmissionError),
    AdmissionRollback {
        admission: PcuMemoryAdmissionError,
        rollback: PcuMemoryReleaseError,
        reservations: PcuMemoryReservationSet,
    },
    Provider(PcuMemoryProviderError),
    ProviderRollback {
        provider: PcuMemoryProviderError,
        rollback: PcuMemoryReleaseError,
        reservations: PcuMemoryReservationSet,
    },
}

/// Takes a fresh snapshot, reserves configured system/process ratios, and allocates one resource.
///
/// The ledger mutations are serialized by the exclusive borrow of `ledger`; callers sharing a
/// ledger across threads must still supply synchronization around this call. A successful result
/// carries one reservation covering all enabled ratios. On allocation failure, that reservation
/// is rolled back before returning. The helper enforces only provider-reported metadata and
/// abstract admission; it makes no physical residency, contiguity, or completion guarantee.
///
/// # Errors
///
/// Returns validation, telemetry, admission, allocation, or rollback errors with their cause.
pub fn allocate_with_policy<P: PcuMemoryProvider, const N: usize>(
    provider: &mut P,
    ledger: &mut PcuMemoryReservationLedger<N>,
    request: PcuMemoryAllocationRequest,
    policy: PcuMemoryAdmissionPolicy,
) -> Result<PcuAdmittedResource<P::Resource>, PcuMemoryAllocateWithPolicyError> {
    request
        .validate()
        .map_err(PcuMemoryAllocateWithPolicyError::InvalidRequest)?;

    let mut reservations = PcuMemoryReservationSet::new();
    if policy.system_used.is_some() || policy.process_used.is_some() {
        let snapshot = provider
            .snapshot(request.pool)
            .map_err(PcuMemoryAllocateWithPolicyError::Snapshot)?;
        if snapshot.id != request.pool {
            return Err(PcuMemoryAllocateWithPolicyError::SnapshotPoolMismatch {
                requested: request.pool,
                reported: snapshot.id,
            });
        }
        let limits = [
            policy.system_used.map(|max_fraction| PcuMemoryLimit {
                mode: PcuMemoryUsageMode::SystemUsed,
                max_fraction,
            }),
            policy.process_used.map(|max_fraction| PcuMemoryLimit {
                mode: PcuMemoryUsageMode::ProcessUsed,
                max_fraction,
            }),
        ];
        match ledger.reserve_with_limits(snapshot, request.size_bytes, limits) {
            Ok(reservation) => reservations.push(reservation),
            Err(admission) => {
                if let Err(rollback) = reservations.release_all(ledger) {
                    return Err(PcuMemoryAllocateWithPolicyError::AdmissionRollback {
                        admission,
                        rollback,
                        reservations,
                    });
                }
                return Err(PcuMemoryAllocateWithPolicyError::Admission(admission));
            }
        }
    }

    let resource = match provider.allocate(request) {
        Ok(resource) => resource,
        Err(provider_error) => {
            if let Err(rollback) = reservations.release_all(ledger) {
                return Err(PcuMemoryAllocateWithPolicyError::ProviderRollback {
                    provider: provider_error,
                    rollback,
                    reservations,
                });
            }
            return Err(PcuMemoryAllocateWithPolicyError::Provider(provider_error));
        }
    };
    if !resource_matches_request(&resource, request) {
        drop(resource);
        let provider_error = PcuMemoryProviderError {
            pool: request.pool,
            operation: PcuMemoryProviderOperation::Allocate,
            disposition: PcuMemoryDisposition::Reject,
            failure: PcuMemoryProviderFailure::ResourceContractViolation,
        };
        if let Err(rollback) = reservations.release_all(ledger) {
            return Err(PcuMemoryAllocateWithPolicyError::ProviderRollback {
                provider: provider_error,
                rollback,
                reservations,
            });
        }
        return Err(PcuMemoryAllocateWithPolicyError::Provider(provider_error));
    }
    Ok(PcuAdmittedResource {
        resource: Some(resource),
        reservations,
    })
}

/// Allocates one resource using a typed placement and ratio-admission policy.
///
/// The returned admitted resource owns the reservation handles together with the backend
/// resource. Callers storing the resource beyond this call must retain that wrapper (or transfer
/// both parts to an owner that can explicitly release the reservations after use is quiescent).
///
/// # Errors
///
/// Returns the same validation, telemetry, admission, allocation, or rollback errors as
/// [`allocate_with_policy`].
pub fn allocate_with_resource_policy<P: PcuMemoryProvider, const N: usize>(
    provider: &mut P,
    ledger: &mut PcuMemoryReservationLedger<N>,
    request: PcuMemoryAllocationRequest,
    policy: PcuMemoryResourcePolicy,
) -> Result<PcuAdmittedResource<P::Resource>, PcuMemoryAllocateWithPolicyError> {
    allocate_with_policy(provider, ledger, policy.apply(request), policy.admission)
}

fn resource_matches_request<R: PcuMemoryResource>(
    resource: &R,
    request: PcuMemoryAllocationRequest,
) -> bool {
    resource.pool() == request.pool
        && resource.size_bytes() >= request.size_bytes
        && resource.alignment_bytes() >= request.alignment_bytes
        && resource
            .alignment_bytes()
            .is_multiple_of(request.alignment_bytes)
        && access_supports(resource.access(), request.access)
        && (!request.require_device_local || resource.is_device_local() == Some(true))
}

fn access_supports(actual: PcuMemoryAccess, required: PcuMemoryAccess) -> bool {
    match required {
        PcuMemoryAccess::ReadOnly => matches!(
            actual,
            PcuMemoryAccess::ReadOnly | PcuMemoryAccess::ReadWrite
        ),
        PcuMemoryAccess::WriteOnly => matches!(
            actual,
            PcuMemoryAccess::WriteOnly | PcuMemoryAccess::ReadWrite
        ),
        PcuMemoryAccess::ReadWrite => actual == PcuMemoryAccess::ReadWrite,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resource_policy_adds_locality_without_changing_pool_or_shape() {
        let request = PcuMemoryAllocationRequest {
            pool: PcuMemoryPoolId(9),
            size_bytes: 64,
            alignment_bytes: 16,
            access: PcuMemoryAccess::ReadWrite,
            host_access: PcuMemoryHostAccess::TransferOnly,
            require_device_local: false,
        };
        let policy = PcuMemoryResourcePolicy {
            require_device_local: true,
            admission: PcuMemoryAdmissionPolicy {
                system_used: Some(PcuMemoryRatio::new(9, 10)),
                process_used: None,
            },
        };
        let applied = policy.apply(request);
        assert_eq!(applied.pool, request.pool);
        assert_eq!(applied.size_bytes, request.size_bytes);
        assert_eq!(applied.alignment_bytes, request.alignment_bytes);
        assert!(applied.require_device_local);
    }

    fn snapshot(system: PcuMemoryUsage, process: PcuMemoryUsage) -> PcuMemoryPoolSnapshot {
        PcuMemoryPoolSnapshot {
            id: PcuMemoryPoolId(3),
            capacity_bytes: Some(100),
            system_used_bytes: system,
            process_used_bytes: process,
            system_ledger_reserved_bytes: 0,
            process_ledger_reserved_bytes: 0,
        }
    }

    const LIMIT_80_PERCENT_SYSTEM: PcuMemoryLimit = PcuMemoryLimit {
        mode: PcuMemoryUsageMode::SystemUsed,
        max_fraction: PcuMemoryRatio::new(4, 5),
    };

    struct BankResource {
        id: u8,
        pool: PcuMemoryPoolId,
        size: u64,
        access: PcuMemoryAccess,
        local: Option<bool>,
        reusable: bool,
        overlap: PcuMemoryOverlap,
    }

    impl PcuMemoryResource for BankResource {
        fn pool(&self) -> PcuMemoryPoolId {
            self.pool
        }
        fn size_bytes(&self) -> u64 {
            self.size
        }
        fn alignment_bytes(&self) -> u64 {
            1
        }
        fn access(&self) -> PcuMemoryAccess {
            self.access
        }
        fn is_device_local(&self) -> Option<bool> {
            self.local
        }
        fn origin(&self) -> PcuMemoryResourceOrigin {
            PcuMemoryResourceOrigin::ProviderManaged
        }
        fn supports(&self, capability: PcuMemoryResourceCapability) -> bool {
            capability == PcuMemoryResourceCapability::ReusableStorage && self.reusable
        }
        fn overlap(&self, other: &Self, _: PcuMemoryRange, _: PcuMemoryRange) -> PcuMemoryOverlap {
            if self.id == other.id {
                PcuMemoryOverlap::Overlapping
            } else {
                self.overlap
            }
        }
    }

    fn bank_resource(id: u8, size: u64, overlap: PcuMemoryOverlap) -> BankResource {
        BankResource {
            id,
            pool: PcuMemoryPoolId(3),
            size,
            access: PcuMemoryAccess::ReadWrite,
            local: Some(true),
            reusable: true,
            overlap,
        }
    }

    #[test]
    fn reusable_bank_checks_heterogeneous_contracts_and_requires_proven_disjointness() {
        let resources = [
            bank_resource(1, 64, PcuMemoryOverlap::Disjoint),
            bank_resource(2, 4, PcuMemoryOverlap::Disjoint),
        ];
        let requirements = [
            PcuMemoryMemberRequirement {
                pool: PcuMemoryPoolId(3),
                minimum_size_bytes: 32,
                access: PcuMemoryAccess::WriteOnly,
                require_device_local: true,
            },
            PcuMemoryMemberRequirement {
                pool: PcuMemoryPoolId(3),
                minimum_size_bytes: 4,
                access: PcuMemoryAccess::ReadOnly,
                require_device_local: true,
            },
        ];
        assert_eq!(
            validate_reusable_memory_bank_members(&resources, &requirements),
            Ok(())
        );

        let unknown = [
            bank_resource(1, 64, PcuMemoryOverlap::Unknown),
            bank_resource(2, 64, PcuMemoryOverlap::Unknown),
        ];
        assert_eq!(
            validate_reusable_memory_bank_members(&unknown, &requirements),
            Err(PcuMemoryBankError::OverlapUnknown {
                first: 0,
                second: 1
            })
        );
        let overlapping = [
            bank_resource(1, 64, PcuMemoryOverlap::Overlapping),
            bank_resource(2, 64, PcuMemoryOverlap::Overlapping),
        ];
        assert_eq!(
            validate_reusable_memory_bank_members(&overlapping, &requirements),
            Err(PcuMemoryBankError::Overlapping {
                first: 0,
                second: 1
            })
        );
    }

    #[test]
    fn reusable_bank_rejects_unproven_capabilities_and_locality() {
        let requirement = [PcuMemoryMemberRequirement {
            pool: PcuMemoryPoolId(3),
            minimum_size_bytes: 1,
            access: PcuMemoryAccess::ReadWrite,
            require_device_local: true,
        }];
        let mut resource = bank_resource(1, 1, PcuMemoryOverlap::Disjoint);
        resource.reusable = false;
        assert_eq!(
            validate_reusable_memory_bank_members(&[resource], &requirement),
            Err(PcuMemoryBankError::ReuseUnsupported { index: 0 })
        );
        let mut resource = bank_resource(1, 1, PcuMemoryOverlap::Disjoint);
        resource.local = None;
        assert_eq!(
            validate_reusable_memory_bank_members(&[resource], &requirement),
            Err(PcuMemoryBankError::DeviceLocalNotEstablished { index: 0 })
        );
    }

    struct MockResource {
        pool: PcuMemoryPoolId,
        size: u64,
        alignment: u64,
        access: PcuMemoryAccess,
        device_local: Option<bool>,
    }

    impl PcuMemoryResource for MockResource {
        fn pool(&self) -> PcuMemoryPoolId {
            self.pool
        }
        fn size_bytes(&self) -> u64 {
            self.size
        }
        fn alignment_bytes(&self) -> u64 {
            self.alignment
        }
        fn access(&self) -> PcuMemoryAccess {
            self.access
        }
        fn is_device_local(&self) -> Option<bool> {
            self.device_local
        }
        fn origin(&self) -> PcuMemoryResourceOrigin {
            PcuMemoryResourceOrigin::ProviderManaged
        }
    }

    #[test]
    fn resource_overlap_defaults_to_unknown_without_provider_proof() {
        let resource = MockResource {
            pool: PcuMemoryPoolId(3),
            size: 16,
            alignment: 4,
            access: PcuMemoryAccess::ReadWrite,
            device_local: Some(true),
        };
        let range = PcuMemoryRange {
            offset_bytes: 0,
            size_bytes: 16,
        };
        assert_eq!(
            resource.overlap(&resource, range, range),
            PcuMemoryOverlap::Unknown
        );
    }

    struct MockImport;

    impl PcuMemoryImportDescriptor for MockImport {
        fn pool(&self) -> PcuMemoryPoolId {
            PcuMemoryPoolId(3)
        }
        fn size_bytes(&self) -> u64 {
            1
        }
        fn alignment_bytes(&self) -> u64 {
            1
        }
        fn access(&self) -> PcuMemoryAccess {
            PcuMemoryAccess::ReadWrite
        }
        fn ownership(&self) -> PcuMemoryImportOwnership {
            PcuMemoryImportOwnership::Borrowed
        }
    }

    struct MockMapping([u8; 1]);

    impl PcuMemoryMapping for MockMapping {
        fn as_bytes(&self) -> &[u8] {
            &self.0
        }
        fn as_bytes_mut(&mut self) -> Option<&mut [u8]> {
            Some(&mut self.0)
        }
    }

    struct MockProvider {
        snapshot: PcuMemoryPoolSnapshot,
        fail_allocate: bool,
        allocate_calls: usize,
    }

    impl PcuMemoryProvider for MockProvider {
        type Resource = MockResource;
        type ImportDescriptor = MockImport;
        type Mapping<'a> = MockMapping;

        fn snapshot(
            &self,
            _: PcuMemoryPoolId,
        ) -> Result<PcuMemoryPoolSnapshot, PcuMemoryProviderError> {
            Ok(self.snapshot)
        }

        fn allocate(
            &mut self,
            request: PcuMemoryAllocationRequest,
        ) -> Result<Self::Resource, PcuMemoryProviderError> {
            self.allocate_calls += 1;
            if self.fail_allocate {
                return Err(PcuMemoryProviderError {
                    pool: request.pool,
                    operation: PcuMemoryProviderOperation::Allocate,
                    disposition: PcuMemoryDisposition::Defer,
                    failure: PcuMemoryProviderFailure::OutOfMemory,
                });
            }
            Ok(MockResource {
                pool: request.pool,
                size: request.size_bytes,
                alignment: request.alignment_bytes,
                access: request.access,
                device_local: Some(true),
            })
        }

        fn import(
            &mut self,
            _: Self::ImportDescriptor,
        ) -> Result<Self::Resource, PcuMemoryProviderError> {
            Err(provider_error(
                PcuMemoryPoolId(3),
                PcuMemoryProviderOperation::Import,
                PcuMemoryProviderFailure::Unsupported,
            ))
        }

        fn map<'a>(
            &'a mut self,
            _: &'a mut Self::Resource,
            _: PcuMemoryRange,
        ) -> Result<Self::Mapping<'a>, PcuMemoryProviderError> {
            Ok(MockMapping([0]))
        }

        fn transfer_to(
            &mut self,
            _: &mut Self::Resource,
            _: u64,
            _: &[u8],
        ) -> Result<(), PcuMemoryProviderError> {
            Ok(())
        }

        fn transfer_from(
            &mut self,
            _: &Self::Resource,
            _: u64,
            _: &mut [u8],
        ) -> Result<(), PcuMemoryProviderError> {
            Ok(())
        }
    }

    fn provider_error(
        pool: PcuMemoryPoolId,
        operation: PcuMemoryProviderOperation,
        failure: PcuMemoryProviderFailure,
    ) -> PcuMemoryProviderError {
        PcuMemoryProviderError {
            pool,
            operation,
            disposition: PcuMemoryDisposition::Reject,
            failure,
        }
    }

    fn mock_provider(system: u64, process: u64, fail_allocate: bool) -> MockProvider {
        MockProvider {
            snapshot: snapshot(
                PcuMemoryUsage::Known(system),
                PcuMemoryUsage::Known(process),
            ),
            fail_allocate,
            allocate_calls: 0,
        }
    }

    fn allocation_request(size_bytes: u64) -> PcuMemoryAllocationRequest {
        PcuMemoryAllocationRequest {
            pool: PcuMemoryPoolId(3),
            size_bytes,
            alignment_bytes: 8,
            access: PcuMemoryAccess::ReadWrite,
            host_access: PcuMemoryHostAccess::TransferOnly,
            require_device_local: true,
        }
    }

    #[test]
    fn provider_copy_requires_explicit_backend_support() {
        let mut provider = mock_provider(0, 0, false);
        let source = provider.allocate(allocation_request(8)).unwrap();
        let mut destination = provider.allocate(allocation_request(8)).unwrap();
        let error = provider
            .copy_resource(&mut destination, &source, 8)
            .unwrap_err();
        assert_eq!(error.operation, PcuMemoryProviderOperation::CopyResource);
        assert_eq!(error.failure, PcuMemoryProviderFailure::Unsupported);
    }

    #[test]
    fn allocate_with_policy_denies_ratio_before_calling_provider() {
        let mut provider = mock_provider(90, 0, false);
        let mut ledger = PcuMemoryReservationLedger::<2>::new();
        let result = allocate_with_policy(
            &mut provider,
            &mut ledger,
            allocation_request(10),
            PcuMemoryAdmissionPolicy {
                system_used: Some(PcuMemoryRatio::new(95, 100)),
                process_used: None,
            },
        );
        assert!(matches!(
            result,
            Err(PcuMemoryAllocateWithPolicyError::Admission(
                PcuMemoryAdmissionError {
                    disposition: PcuMemoryDisposition::Defer,
                    reason: PcuMemoryAdmissionReason::LimitExceeded {
                        mode: PcuMemoryUsageMode::SystemUsed,
                        ..
                    },
                    ..
                }
            ))
        ));
        assert_eq!(provider.allocate_calls, 0);
        assert_eq!(ledger.reserved_bytes(PcuMemoryPoolId(3)), Some(0));
    }

    #[test]
    fn allocate_failure_rolls_back_both_ratio_reservations() {
        let mut provider = mock_provider(10, 10, true);
        let mut ledger = PcuMemoryReservationLedger::<2>::new();
        let result = allocate_with_policy(
            &mut provider,
            &mut ledger,
            allocation_request(10),
            PcuMemoryAdmissionPolicy {
                system_used: Some(PcuMemoryRatio::new(95, 100)),
                process_used: Some(PcuMemoryRatio::new(95, 100)),
            },
        );
        assert!(matches!(
            result,
            Err(PcuMemoryAllocateWithPolicyError::Provider(
                PcuMemoryProviderError {
                    failure: PcuMemoryProviderFailure::OutOfMemory,
                    ..
                }
            ))
        ));
        assert_eq!(provider.allocate_calls, 1);
        assert_eq!(ledger.reserved_bytes(PcuMemoryPoolId(3)), Some(0));
    }

    #[test]
    fn admitted_resource_keeps_both_reservations_until_release() {
        let mut provider = mock_provider(10, 10, false);
        let mut ledger = PcuMemoryReservationLedger::<2>::new();
        let admitted = allocate_with_policy(
            &mut provider,
            &mut ledger,
            allocation_request(10),
            PcuMemoryAdmissionPolicy {
                system_used: Some(PcuMemoryRatio::new(95, 100)),
                process_used: Some(PcuMemoryRatio::new(95, 100)),
            },
        )
        .unwrap();
        assert_eq!(admitted.resource().size_bytes(), 10);
        assert_eq!(
            admitted.resource().origin(),
            PcuMemoryResourceOrigin::ProviderManaged
        );
        assert_eq!(admitted.reservations().len(), 1);
        assert_eq!(ledger.reserved_bytes(PcuMemoryPoolId(3)), Some(10));
        admitted.release(&mut ledger).unwrap();
        assert_eq!(ledger.reserved_bytes(PcuMemoryPoolId(3)), Some(0));
    }

    #[test]
    fn combined_system_and_process_limits_use_one_ledger_reservation() {
        let mut provider = mock_provider(40, 20, false);
        // A single slot is sufficient when one allocation must satisfy both limits.
        let mut ledger = PcuMemoryReservationLedger::<1>::new();
        let admitted = allocate_with_policy(
            &mut provider,
            &mut ledger,
            allocation_request(10),
            PcuMemoryAdmissionPolicy {
                system_used: Some(PcuMemoryRatio::new(95, 100)),
                process_used: Some(PcuMemoryRatio::new(50, 100)),
            },
        )
        .unwrap();

        assert_eq!(admitted.reservations().len(), 1);
        assert_eq!(ledger.reserved_bytes(PcuMemoryPoolId(3)), Some(10));
        admitted.release(&mut ledger).unwrap();
        assert_eq!(ledger.reserved_bytes(PcuMemoryPoolId(3)), Some(0));
    }

    #[test]
    fn serialized_concurrent_admissions_cannot_spend_the_same_headroom() {
        use std::sync::{
            Arc,
            Mutex,
        };

        let ledger = Arc::new(Mutex::new(PcuMemoryReservationLedger::<2>::new()));
        let pool = snapshot(PcuMemoryUsage::Known(89), PcuMemoryUsage::Known(0));
        let limit = PcuMemoryLimit {
            mode: PcuMemoryUsageMode::SystemUsed,
            max_fraction: PcuMemoryRatio::new(95, 100),
        };

        let workers = [0, 1].map(|_| {
            let ledger = Arc::clone(&ledger);
            std::thread::spawn(move || ledger.lock().unwrap().reserve(pool, 5, limit))
        });

        let results = workers
            .into_iter()
            .map(|worker| worker.join().unwrap())
            .collect::<std::vec::Vec<_>>();
        let reservations = results
            .iter()
            .filter_map(|result| result.as_ref().ok().copied())
            .collect::<std::vec::Vec<_>>();

        assert_eq!(reservations.len(), 1);
        assert_eq!(
            results
                .iter()
                .filter(|result| matches!(result, Err(error) if error.disposition == PcuMemoryDisposition::Defer))
                .count(),
            1
        );

        let mut ledger = ledger.lock().unwrap();
        assert_eq!(ledger.reserved_bytes(pool.id), Some(5));
        assert_eq!(ledger.release(reservations[0]), Ok(5));
        assert_eq!(ledger.reserved_bytes(pool.id), Some(0));
        drop(ledger);
    }

    #[test]
    fn reservation_remains_occupied_while_async_completion_holds_resource() {
        let mut provider = mock_provider(88, 20, false);
        let mut ledger = PcuMemoryReservationLedger::<1>::new();
        let policy = PcuMemoryAdmissionPolicy {
            system_used: Some(PcuMemoryRatio::new(95, 100)),
            process_used: None,
        };
        let mut completion = Some(
            allocate_with_policy(&mut provider, &mut ledger, allocation_request(6), policy)
                .unwrap(),
        );

        // Model an in-flight operation retaining its admitted resource until completion.
        assert_eq!(ledger.reserved_bytes(PcuMemoryPoolId(3)), Some(6));
        assert!(matches!(
            allocate_with_policy(&mut provider, &mut ledger, allocation_request(1), policy,),
            Err(PcuMemoryAllocateWithPolicyError::Admission(
                PcuMemoryAdmissionError {
                    disposition: PcuMemoryDisposition::Defer,
                    ..
                }
            ))
        ));
        assert_eq!(provider.allocate_calls, 1);

        completion.take().unwrap().release(&mut ledger).unwrap();
        assert_eq!(ledger.reserved_bytes(PcuMemoryPoolId(3)), Some(0));

        let next = allocate_with_policy(&mut provider, &mut ledger, allocation_request(6), policy)
            .unwrap();
        assert_eq!(provider.allocate_calls, 2);
        assert_eq!(ledger.reserved_bytes(PcuMemoryPoolId(3)), Some(6));
        next.release(&mut ledger).unwrap();
    }

    #[test]
    fn memory_request_checks_size_and_alignment_without_rounding_size() {
        let valid = PcuMemoryAllocationRequest {
            pool: PcuMemoryPoolId(3),
            size_bytes: 513,
            alignment_bytes: 256,
            access: PcuMemoryAccess::ReadWrite,
            host_access: PcuMemoryHostAccess::TransferOnly,
            require_device_local: true,
        };
        assert_eq!(valid.validate(), Ok(()));
        assert_eq!(
            PcuMemoryAllocationRequest {
                size_bytes: 0,
                ..valid
            }
            .validate(),
            Err(PcuMemoryRequestError::ZeroSize)
        );
        assert_eq!(
            PcuMemoryAllocationRequest {
                alignment_bytes: 0,
                ..valid
            }
            .validate(),
            Err(PcuMemoryRequestError::InvalidAlignment)
        );
        assert_eq!(
            PcuMemoryAllocationRequest {
                alignment_bytes: 24,
                ..valid
            }
            .validate(),
            Err(PcuMemoryRequestError::InvalidAlignment)
        );
    }

    #[test]
    fn memory_range_end_is_checked_for_overflow() {
        assert_eq!(
            PcuMemoryRange {
                offset_bytes: 12,
                size_bytes: 30,
            }
            .checked_end(),
            Some(42)
        );
        assert_eq!(
            PcuMemoryRange {
                offset_bytes: u64::MAX,
                size_bytes: 1,
            }
            .checked_end(),
            None
        );
    }

    #[test]
    fn defers_at_or_above_fraction_threshold_and_releases_reservations() {
        let mut ledger = PcuMemoryReservationLedger::<2>::new();
        let snap = snapshot(PcuMemoryUsage::Known(60), PcuMemoryUsage::Known(10));
        assert_eq!(
            ledger.reserve(snap, 20, LIMIT_80_PERCENT_SYSTEM),
            Err(PcuMemoryAdmissionError {
                pool: PcuMemoryPoolId(3),
                disposition: PcuMemoryDisposition::Defer,
                reason: PcuMemoryAdmissionReason::LimitExceeded {
                    mode: PcuMemoryUsageMode::SystemUsed,
                    projected_used_bytes: 80,
                    maximum_used_bytes: 80,
                },
            })
        );
        let below_threshold = snapshot(PcuMemoryUsage::Known(59), PcuMemoryUsage::Known(10));
        let reservation = ledger
            .reserve(below_threshold, 20, LIMIT_80_PERCENT_SYSTEM)
            .unwrap();
        assert_eq!(ledger.reserved_bytes(PcuMemoryPoolId(3)), Some(20));
        assert_eq!(
            ledger.reserve(below_threshold, 1, LIMIT_80_PERCENT_SYSTEM),
            Err(PcuMemoryAdmissionError {
                pool: PcuMemoryPoolId(3),
                disposition: PcuMemoryDisposition::Defer,
                reason: PcuMemoryAdmissionReason::LimitExceeded {
                    mode: PcuMemoryUsageMode::SystemUsed,
                    projected_used_bytes: 80,
                    maximum_used_bytes: 80,
                },
            })
        );
        assert_eq!(ledger.release(reservation), Ok(20));
        assert_eq!(ledger.reserved_bytes(PcuMemoryPoolId(3)), Some(0));
        assert!(ledger.reserve(snap, 19, LIMIT_80_PERCENT_SYSTEM).is_ok());
    }

    #[test]
    fn unknown_hard_limit_telemetry_is_rejected_and_process_limit_is_selected() {
        let mut ledger = PcuMemoryReservationLedger::<2>::new();
        assert_eq!(
            ledger.reserve(
                snapshot(PcuMemoryUsage::Known(0), PcuMemoryUsage::Unknown),
                1,
                PcuMemoryLimit {
                    mode: PcuMemoryUsageMode::ProcessUsed,
                    max_fraction: PcuMemoryRatio::new(1, 2),
                },
            ),
            Err(PcuMemoryAdmissionError {
                pool: PcuMemoryPoolId(3),
                disposition: PcuMemoryDisposition::Reject,
                reason: PcuMemoryAdmissionReason::UnknownUsage(PcuMemoryUsageMode::ProcessUsed),
            })
        );
        let reservation = ledger
            .reserve(
                snapshot(PcuMemoryUsage::Known(99), PcuMemoryUsage::Known(48)),
                1,
                PcuMemoryLimit {
                    mode: PcuMemoryUsageMode::ProcessUsed,
                    max_fraction: PcuMemoryRatio::new(1, 2),
                },
            )
            .unwrap();
        assert_eq!(ledger.release(reservation), Ok(1));
    }

    #[test]
    fn rejects_invalid_ratios_and_prevents_stale_or_duplicate_release() {
        let mut ledger = PcuMemoryReservationLedger::<1>::new();
        assert_eq!(
            ledger.reserve(
                snapshot(PcuMemoryUsage::Known(0), PcuMemoryUsage::Known(0)),
                0,
                PcuMemoryLimit {
                    mode: PcuMemoryUsageMode::SystemUsed,
                    max_fraction: PcuMemoryRatio::new(1, 0),
                },
            ),
            Err(PcuMemoryAdmissionError {
                pool: PcuMemoryPoolId(3),
                disposition: PcuMemoryDisposition::Reject,
                reason: PcuMemoryAdmissionReason::InvalidFraction,
            })
        );
        let snapshot = snapshot(PcuMemoryUsage::Known(0), PcuMemoryUsage::Known(0));
        let first = ledger
            .reserve(snapshot, 10, LIMIT_80_PERCENT_SYSTEM)
            .unwrap();
        assert_eq!(ledger.release(first), Ok(10));
        let second = ledger
            .reserve(snapshot, 10, LIMIT_80_PERCENT_SYSTEM)
            .unwrap();
        assert_eq!(
            ledger.release(first),
            Err(PcuMemoryReleaseError::InvalidReservation)
        );
        assert_eq!(ledger.release(second), Ok(10));
    }

    #[test]
    fn refreshed_snapshot_does_not_double_count_represented_reservations() {
        let mut ledger = PcuMemoryReservationLedger::<3>::new();
        let baseline = snapshot(PcuMemoryUsage::Known(10), PcuMemoryUsage::Known(10));
        let reservation = ledger
            .reserve(baseline, 20, LIMIT_80_PERCENT_SYSTEM)
            .unwrap();
        let refreshed = PcuMemoryPoolSnapshot {
            system_used_bytes: PcuMemoryUsage::Known(30),
            system_ledger_reserved_bytes: 20,
            ..baseline
        };
        // Reported usage includes the pending 20-byte reservation, so the projection is 79.
        // Counting that reservation twice would incorrectly push the total over the limit.
        let next = ledger
            .reserve(refreshed, 49, LIMIT_80_PERCENT_SYSTEM)
            .unwrap();
        assert_eq!(ledger.reserved_bytes(PcuMemoryPoolId(3)), Some(69));
        assert_eq!(
            ledger.reserve(
                PcuMemoryPoolSnapshot {
                    system_ledger_reserved_bytes: 70,
                    ..refreshed
                },
                1,
                LIMIT_80_PERCENT_SYSTEM,
            ),
            Err(PcuMemoryAdmissionError {
                pool: PcuMemoryPoolId(3),
                disposition: PcuMemoryDisposition::Reject,
                reason: PcuMemoryAdmissionReason::RepresentedReservationsExceedLedger {
                    mode: PcuMemoryUsageMode::SystemUsed,
                    represented_bytes: 70,
                    ledger_bytes: 69,
                },
            })
        );
        assert_eq!(ledger.release(reservation), Ok(20));
        assert_eq!(ledger.release(next), Ok(49));
    }

    #[test]
    fn ninety_five_percent_system_and_process_modes_are_distinct() {
        let pool = PcuMemoryPoolSnapshot {
            id: PcuMemoryPoolId(9),
            capacity_bytes: Some(1000),
            system_used_bytes: PcuMemoryUsage::Known(940),
            process_used_bytes: PcuMemoryUsage::Known(100),
            system_ledger_reserved_bytes: 0,
            process_ledger_reserved_bytes: 0,
        };
        let ratio = PcuMemoryRatio::new(95, 100);
        let mut system_ledger = PcuMemoryReservationLedger::<1>::new();
        let system = PcuMemoryLimit {
            mode: PcuMemoryUsageMode::SystemUsed,
            max_fraction: ratio,
        };
        assert_eq!(
            system_ledger
                .reserve(pool, 10, system)
                .unwrap_err()
                .disposition,
            PcuMemoryDisposition::Defer
        );

        let mut process_ledger = PcuMemoryReservationLedger::<1>::new();
        let process = PcuMemoryLimit {
            mode: PcuMemoryUsageMode::ProcessUsed,
            max_fraction: ratio,
        };
        let reservation = process_ledger.reserve(pool, 10, process).unwrap();
        assert_eq!(process_ledger.release(reservation), Ok(10));
        assert_eq!(
            process_ledger
                .reserve(
                    PcuMemoryPoolSnapshot {
                        process_used_bytes: PcuMemoryUsage::Known(940),
                        ..pool
                    },
                    10,
                    process
                )
                .unwrap_err()
                .disposition,
            PcuMemoryDisposition::Defer
        );
    }
}
