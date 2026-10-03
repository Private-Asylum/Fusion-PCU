//! Shared cold resource projection after family-specific checked-map validation.

#[rustfmt::skip]
use crate::{
    PcuBindingAccess,
    PcuBindingRef,
    PcuDispatchDataOp,
    PcuDispatchIndex,
    PcuDispatchKernelIr,
    PcuDispatchOp,
    PcuImplementationRequirements,
    PcuValueType,
};

/// One distinct binding accessed by the validated instruction body.
///
/// Read and write extents are independent: a binding read only at element zero
/// and written at every logical index needs one input element and a full output
/// extent. Those facts do not permit physical aliasing or early publication.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CheckedScalarMapResource {
    pub binding: PcuBindingRef,
    /// Original declared access, without weakening or inventing Rust permissions.
    pub declared_access: PcuBindingAccess,
    /// Zero means no load instruction accesses this binding.
    pub minimum_read_elements: u32,
    /// Original input contents needed by loads not preceded by a same-lane
    /// store in this flat validated body. This may be smaller than the required
    /// view capacity: a prior indexed store supplies subsequent indexed loads.
    /// Multi-lane element-zero loads remain conservative. This fact alone does
    /// not permit physical aliasing, omitted synchronization or early writes
    /// to externally visible backing; providers must separately prove safe
    /// private working storage and terminal publication.
    pub minimum_initial_read_elements: u32,
    /// At least one actual load selects element zero, even when another load
    /// from this binding also requires the full indexed span.
    pub reads_element_zero: bool,
    /// Zero means no store instruction accesses this binding.
    pub minimum_write_elements: u32,
}

impl CheckedScalarMapResource {
    const EMPTY: Self = Self {
        binding: PcuBindingRef::new(0, 0),
        declared_access: PcuBindingAccess::ReadOnly,
        minimum_read_elements: 0,
        minimum_initial_read_elements: 0,
        reads_element_zero: false,
        minimum_write_elements: 0,
    };

    /// Whether this binding's element-zero read overlaps writes at other indices.
    ///
    /// This is a structural dependency, not a proof of a physical race: logical
    /// invocations may share a physical worker and retain the same dependency.
    /// A provider must reject it or prove preservation of the original input
    /// (for example, a snapshot) under its execution contract. One logical
    /// element has no cross-index overlap. Distinct binding references may still
    /// physically alias; that requires independent backing/alias validation.
    #[must_use]
    pub const fn has_cross_index_read_write(self) -> bool {
        self.reads_element_zero && self.minimum_write_elements > 1
    }

    /// Minimum element capacity for one physical resource serving both roles.
    /// Byte size, alignment, backing identity and alias legality remain separate.
    #[must_use]
    pub const fn minimum_elements(self) -> u32 {
        if self.minimum_read_elements > self.minimum_write_elements {
            self.minimum_read_elements
        } else {
            self.minimum_write_elements
        }
    }
}

/// Actual unique resources in first-instruction-access order, detached from IR.
///
/// `CAPACITY` is chosen by the caller for its prepared representation; PCU adds
/// no universal binding limit. Unaccessed declarations occupy no slot. All
/// declared bindings still undergo the existing interface/type/SSA validation.
/// This description adds neither provider admission nor lifetime/alias rights.
/// First-access order is not an ABI: providers retain original binding references
/// when mapping declaration order, argument slots and independent output order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CheckedScalarMapResourceSchema<const CAPACITY: usize> {
    resources: [CheckedScalarMapResource; CAPACITY],
    resource_count: usize,
    pub value_type: PcuValueType,
    /// Complete original header tuple; lexical instruction policies stay in IR.
    pub requirements: PcuImplementationRequirements,
    pub submitted_invocations: u32,
    pub logical_extent: u32,
}

impl<const CAPACITY: usize> CheckedScalarMapResourceSchema<CAPACITY> {
    /// Actual unique reads/writes; declaration order and repeated access do not
    /// fabricate extra buffers. Loads whose SSA result is unused still count:
    /// removing instruction effects requires a separate legal optimization.
    #[must_use]
    pub fn resources(&self) -> &[CheckedScalarMapResource] {
        &self.resources[..self.resource_count]
    }

    /// Looks up an actual resource; an unread/unwritten declaration returns None.
    #[must_use]
    pub fn resource(&self, binding: PcuBindingRef) -> Option<CheckedScalarMapResource> {
        self.resources()
            .iter()
            .find(|resource| resource.binding == binding)
            .copied()
    }
}

/// Cold structural rejection or insufficient caller-chosen schema capacity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CheckedScalarMapResourceError<E> {
    InvalidMap(E),
    /// More actual unique bindings exist than the caller's representation holds.
    InsufficientCapacity {
        capacity: usize,
        required_at_least: usize,
    },
}

/// Internal projection requires complete family-specific validation first.
/// It adds no execution, fault, alias or publication permission.
pub(super) fn project<const CAPACITY: usize, E>(
    kernel: &PcuDispatchKernelIr<'_>,
    value_type: PcuValueType,
    undeclared: impl Fn(PcuBindingRef) -> E,
) -> Result<CheckedScalarMapResourceSchema<CAPACITY>, CheckedScalarMapResourceError<E>> {
    let submitted_invocations = kernel.entry.logical_shape[0];
    let (body, logical_extent) = match kernel.ops.first() {
        Some(PcuDispatchOp::GridStrideLoop { body, extent }) => (*body, *extent),
        _ => (kernel.ops, submitted_invocations),
    };
    project_body(kernel, value_type, body, logical_extent, undeclared)
}

/// Common cold access analysis, after the owning profile validated the body.
/// Representation-preserving transport and checked arithmetic deliberately share
/// incoming-content and view-capacity rules, without sharing numerical admission.
pub(super) fn project_body<const CAPACITY: usize, E>(
    kernel: &PcuDispatchKernelIr<'_>,
    value_type: PcuValueType,
    body: &[PcuDispatchOp<'_>],
    logical_extent: u32,
    undeclared: impl Fn(PcuBindingRef) -> E,
) -> Result<CheckedScalarMapResourceSchema<CAPACITY>, CheckedScalarMapResourceError<E>> {
    use CheckedScalarMapResourceError as Error;
    let mut schema = CheckedScalarMapResourceSchema {
        resources: [CheckedScalarMapResource::EMPTY; CAPACITY],
        resource_count: 0,
        value_type,
        requirements: kernel.numerical_requirements,
        submitted_invocations: kernel.entry.logical_shape[0],
        logical_extent,
    };
    for instruction in body {
        let (binding, index, reading) = match *instruction {
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad { binding, index, .. }) => {
                (binding, index, true)
            }
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore { binding, index, .. }) => {
                (binding, index, false)
            }
            _ => continue,
        };
        let slot = if let Some(slot) = schema
            .resources()
            .iter()
            .position(|resource| resource.binding == binding)
        {
            slot
        } else {
            let slot = schema.resource_count;
            if slot == CAPACITY {
                return Err(Error::InsufficientCapacity {
                    capacity: CAPACITY,
                    required_at_least: slot + 1,
                });
            }
            // Prior full validation guarantees each actual reference is declared.
            let declared = kernel
                .bindings
                .iter()
                .find(|declared| declared.reference() == binding)
                .ok_or(Error::InvalidMap(undeclared(binding)))?;
            schema.resources[slot] = CheckedScalarMapResource {
                binding,
                declared_access: declared.access,
                ..CheckedScalarMapResource::EMPTY
            };
            schema.resource_count += 1;
            slot
        };
        let extent = if index == PcuDispatchIndex::BindingElementZero {
            1
        } else {
            logical_extent
        };
        if reading && index == PcuDispatchIndex::BindingElementZero {
            schema.resources[slot].reads_element_zero = true;
        }
        if reading {
            let resource = &mut schema.resources[slot];
            let supplied_by_store = resource.minimum_write_elements != 0
                && (index != PcuDispatchIndex::BindingElementZero || logical_extent == 1);
            if !supplied_by_store {
                resource.minimum_initial_read_elements =
                    resource.minimum_initial_read_elements.max(extent);
            }
        }
        let required = if reading {
            &mut schema.resources[slot].minimum_read_elements
        } else {
            &mut schema.resources[slot].minimum_write_elements
        };
        *required = (*required).max(extent);
    }
    Ok(schema)
}
