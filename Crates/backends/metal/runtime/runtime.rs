//! Safe, session-affine ownership over the bounded Metal implementation.

#[rustfmt::skip]
use std::{
    fmt,
    rc::Rc,
};

#[rustfmt::skip]
use fusion_pcu::{
    PcuCheckedScalarFaultLaw,
    PcuExecutionFault,
    PcuExecutionFaultKind,
    PcuFloatUnderflowPolicy,
};

use crate::ffi;
#[path = "binary/binary.rs"]
mod binary;
#[path = "carrier/carrier.rs"]
pub mod carrier;
#[path = "composed/composed.rs"]
pub mod composed;
#[path = "fault/fault.rs"]
mod fault;
#[path = "integer/integer.rs"]
pub mod integer;
#[path = "transport/transport.rs"]
pub mod transport;
pub use integer::IntegerControl as MetalPreparedIntegerControl;
pub use carrier::Carrier as MetalPreparedCarrierControl;
#[path = "unary/unary.rs"]
mod unary;
#[rustfmt::skip]
pub use binary::{
    MetalPreparedF32Binary,
    MetalPreparedF64Binary,
    MetalPreparedFloatBinary,
};

/// Failure of discovery, admission, transport, compilation or terminal completion.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MetalError {
    /// This platform has no admitted Metal runtime.
    Unsupported,
    /// The requested allocation or map length is outside the bounded profile.
    InvalidExtent,
    /// A resource belongs to another independently opened session.
    ForeignSession,
    /// Metal or its compiler reported an operational failure.
    Runtime(String),
    /// Checked arithmetic failed; recovered faults retain completed borrowed-output payloads.
    Arithmetic(MetalFault),
}

impl fmt::Display for MetalError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for MetalError {}

/// Terminal numerical fault selected by increasing logical invocation index.
pub type MetalFault = PcuExecutionFault;

/// Admitted checked unsigned integer operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetalIntegerOp {
    /// Exact identity transport with no arithmetic.
    Identity,
    Add,
    Subtract,
    Multiply,
    Divide,
}

impl MetalIntegerOp {
    const fn code(self) -> u32 {
        match self {
            Self::Identity => 8,
            Self::Add => 0,
            Self::Subtract => 1,
            Self::Multiply => 2,
            Self::Divide => 3,
        }
    }
}

/// Physical facts returned by Metal, independent of checked-program capability claims.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MetalDeviceFacts {
    pub name: String,
    pub registry_id: u64,
    pub unified_memory: bool,
    pub max_buffer_bytes: u64,
}

struct Session {
    native: ffi::Session,
    facts: MetalDeviceFacts,
}

/// One device and command queue. Clones preserve the same resource affinity.
///
/// Sessions deliberately remain thread-local; no unchecked Send/Sync promise is made.
#[derive(Clone)]
pub struct MetalSession(Rc<Session>);

impl MetalSession {
    pub(crate) fn same_session(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.0, &other.0)
    }
    pub(crate) fn ensure_quiescent(&self) -> Result<(), MetalError> {
        self.0.native.ensure_quiescent()
    }
    /// Enumerates physical devices without compiling or executing a kernel.
    ///
    /// # Errors
    /// Returns Unsupported off macOS or the native discovery error.
    pub fn discover() -> Result<Vec<MetalDeviceFacts>, MetalError> {
        ffi::discover()
    }

    /// Opens the device at the current physical enumeration index.
    ///
    /// # Errors
    /// Returns Unsupported off macOS or a missing-device/queue error.
    pub fn open(index: usize) -> Result<Self, MetalError> {
        let (native, facts) = ffi::Session::open(index)?;
        Ok(Self(Rc::new(Session { native, facts })))
    }

    #[must_use]
    pub fn facts(&self) -> &MetalDeviceFacts {
        &self.0.facts
    }

    /// Allocates shared storage and transfers exact u32 words to it.
    ///
    /// # Errors
    /// Returns `InvalidExtent` or a native allocation/transport error.
    pub fn upload_u32(&self, words: &[u32]) -> Result<MetalBuffer, MetalError> {
        self.ensure_quiescent()?;
        let bytes = validate_extent(words.len(), self.0.facts.max_buffer_bytes)?;
        let native = self.0.native.allocate(bytes)?;
        native.write(words)?;
        Ok(MetalBuffer {
            session: self.clone(),
            native,
            bytes,
        })
    }

    /// Allocates owned Shared storage and copies the exact initialized byte slice.
    /// No caller pointer is retained; this is ordinary staging, not a no-copy import.
    ///
    /// # Errors
    /// Returns `InvalidExtent` for empty, overflowing or oversized extents,
    /// or a native allocation/transport error. Quarantined sessions reject before copying.
    pub fn upload_bytes(&self, bytes: &[u8]) -> Result<MetalBuffer, MetalError> {
        let buffer = self.allocate_zeroed_bytes(bytes.len())?;
        buffer.native.write_bytes(0, bytes)?;
        Ok(buffer)
    }

    #[cfg(all(test, target_os = "macos"))]
    pub(crate) fn allocate_zeroed(&self, words: usize) -> Result<MetalBuffer, MetalError> {
        self.ensure_quiescent()?;
        let bytes = validate_extent(words, self.0.facts.max_buffer_bytes)?;
        self.allocate_zeroed_bytes(bytes)
    }

    /// Allocates exact-byte initialized Shared storage in this retained session.
    ///
    /// Native Metal allocation establishes zero initialization without a host scratch vector.
    /// Logical bytes are never rounded up, including odd low-format payloads.
    ///
    /// # Errors
    /// Rejects zero/oversized extents, quarantined sessions or native allocation failure.
    pub fn allocate_zeroed_bytes(&self, bytes: usize) -> Result<MetalBuffer, MetalError> {
        self.ensure_quiescent()?;
        validate_byte_extent(bytes, self.0.facts.max_buffer_bytes)?;
        // Metal newBufferWithLength:options: clears the allocation to zero. This concrete
        // allocator therefore establishes initialization without a temporary host zero vector.
        let native = self.0.native.allocate(bytes)?;
        Ok(MetalBuffer {
            session: self.clone(),
            native,
            bytes,
        })
    }

    /// Compiles the fixed integer checker for one admitted operation.
    ///
    /// # Errors
    /// Returns a native compiler/pipeline error.
    pub fn prepare_integer_map(
        &self,
        operation: MetalIntegerOp,
    ) -> Result<MetalPreparedIntegerMap, MetalError> {
        let pipeline = self.0.native.compile(INTEGER_SOURCE, "pcu_checked_u32")?;
        Ok(MetalPreparedIntegerMap {
            session: self.clone(),
            pipeline,
            operation: operation.code(),
            fault_law: fault::integer(
                fusion_pcu::PcuScalarType::U32,
                operation,
                fusion_pcu::PcuRangePolicy::Reject,
            )?,
        })
    }
    pub(crate) fn prepare_typed_integer_map(
        &self,
        operation: MetalIntegerOp,
        scalar: fusion_pcu::PcuScalarType,
    ) -> Result<MetalPreparedIntegerMap, MetalError> {
        let code = match scalar {
            fusion_pcu::PcuScalarType::U32 => operation.code(),
            fusion_pcu::PcuScalarType::I32 => match operation {
                MetalIntegerOp::Add => 9,
                MetalIntegerOp::Subtract => 10,
                MetalIntegerOp::Multiply => 11,
                MetalIntegerOp::Identity => 8,
                MetalIntegerOp::Divide => return Err(MetalError::Unsupported),
            },
            fusion_pcu::PcuScalarType::U64 => match operation {
                MetalIntegerOp::Add => 16,
                MetalIntegerOp::Subtract => 17,
                MetalIntegerOp::Multiply => 18,
                MetalIntegerOp::Identity => 22,
                MetalIntegerOp::Divide => return Err(MetalError::Unsupported),
            },
            fusion_pcu::PcuScalarType::I64 => match operation {
                MetalIntegerOp::Add => 19,
                MetalIntegerOp::Subtract => 20,
                MetalIntegerOp::Multiply => 21,
                MetalIntegerOp::Identity => 22,
                MetalIntegerOp::Divide => return Err(MetalError::Unsupported),
            },
            _ => return Err(MetalError::Unsupported),
        };
        Ok(MetalPreparedIntegerMap {
            session: self.clone(),
            pipeline: self.0.native.compile(INTEGER_SOURCE, "pcu_checked_u32")?,
            operation: code,
            fault_law: fault::integer(scalar, operation, fusion_pcu::PcuRangePolicy::Reject)?,
        })
    }

    /// Prepares a checked signed I32 Add/Sub/Mul encoding map.
    /// All arithmetic uses unsigned encodings; exact negative range failures remain underflow.
    ///
    /// # Errors
    /// Rejects division before compilation, or returns native pipeline errors.
    pub fn prepare_i32_map(
        &self,
        operation: MetalIntegerOp,
    ) -> Result<MetalPreparedIntegerMap, MetalError> {
        self.prepare_typed_integer_map(operation, fusion_pcu::PcuScalarType::I32)
    }
    /// Prepares checked U64 identity/Add/Sub/Mul with bounded unsigned U32 limbs.
    ///
    /// # Errors
    /// Rejects division or returns native compiler/pipeline failures.
    pub fn prepare_u64_map(
        &self,
        operation: MetalIntegerOp,
    ) -> Result<MetalPreparedIntegerMap, MetalError> {
        self.prepare_typed_integer_map(operation, fusion_pcu::PcuScalarType::U64)
    }
    /// Prepares checked I64 identity/Add/Sub/Mul without signed-overflow arithmetic.
    ///
    /// # Errors
    /// Rejects division or returns native compiler/pipeline failures.
    pub fn prepare_i64_map(
        &self,
        operation: MetalIntegerOp,
    ) -> Result<MetalPreparedIntegerMap, MetalError> {
        self.prepare_typed_integer_map(operation, fusion_pcu::PcuScalarType::I64)
    }
}

/// Shared allocation retaining its originating session.
pub struct MetalBuffer {
    session: MetalSession,
    native: ffi::Buffer,
    bytes: usize,
}

impl MetalBuffer {
    pub(crate) const fn session(&self) -> &MetalSession {
        &self.session
    }
    #[allow(
        clippy::needless_pass_by_ref_mut,
        reason = "Exclusive host ownership is required to modify native shared storage."
    )]
    pub(crate) fn write_bytes(&mut self, offset: usize, bytes: &[u8]) -> Result<(), MetalError> {
        self.session.ensure_quiescent()?;
        self.native.write_bytes(offset, bytes)
    }
    pub(crate) fn read_pair_bytes(
        &self,
        first: &mut [u8],
        second: &mut [u8],
    ) -> Result<(), MetalError> {
        self.session.ensure_quiescent()?;
        self.native.read_pair_bytes(first, second)
    }
    pub(crate) fn read_bytes(&self, offset: usize, bytes: &mut [u8]) -> Result<(), MetalError> {
        self.session.ensure_quiescent()?;
        self.native.read_bytes(offset, bytes)
    }
    /// Number of complete U32 words in this allocation. Non-word scalar payloads retain their
    /// independent exact byte extent; use `byte_len` for byte transport and scalar admission.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.bytes / 4
    }

    /// Exact owned byte extent, including non-word-sized scalar storage.
    #[must_use]
    pub const fn byte_len(&self) -> usize {
        self.bytes
    }

    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.bytes == 0
    }

    /// Copies the complete owned Shared buffer into an exact initialized host byte slice.
    /// Quiescence and the entire nonempty byte extent are checked before any destination write.
    /// This API retains no host pointer and exposes no Shared mapping or borrowed device view.
    ///
    /// # Errors
    /// Returns `InvalidExtent` on a size mismatch or a native/quarantine error. On rejection
    /// no host bytes are modified. Checked callers must complete their numerical gate first.
    pub fn read_into_bytes(&self, bytes: &mut [u8]) -> Result<(), MetalError> {
        self.session.ensure_quiescent()?;
        if self.bytes != bytes.len() || bytes.is_empty() {
            return Err(MetalError::InvalidExtent);
        }
        self.native.read_bytes(0, bytes)
    }

    /// Transfers exact words after all synchronous submissions have completed.
    ///
    /// # Errors
    /// Returns a native transport error.
    pub fn download_u32(&self) -> Result<Vec<u32>, MetalError> {
        self.session.ensure_quiescent()?;
        if !self.bytes.is_multiple_of(4) {
            return Err(MetalError::InvalidExtent);
        }
        self.native.read(self.len())
    }
}

/// Fixed checked map executable retaining its queue, device and pipeline.
pub struct MetalPreparedIntegerMap {
    fault_law: Option<PcuCheckedScalarFaultLaw>,
    session: MetalSession,
    pipeline: ffi::Pipeline,
    operation: u32,
}

impl MetalPreparedIntegerMap {
    /// Executes with fresh output and private per-element status records.
    ///
    /// The command buffer reaches terminal completion before any record or output is read.
    /// Each lane owns one status record; increasing-index host selection requires no 64-bit
    /// atomic. Numerical failure discards output, and retry uses fresh status storage.
    ///
    /// # Errors
    /// Returns a shape/affinity error before work, a terminal runtime failure, or the first
    /// checked arithmetic fault. A failed call never returns partially useful output.
    pub fn execute(&self, lhs: &MetalBuffer, rhs: &MetalBuffer) -> Result<MetalBuffer, MetalError> {
        if lhs.len() != rhs.len() {
            return Err(MetalError::InvalidExtent);
        }
        self.execute_prefix(lhs, rhs, lhs.len())
    }
    pub(crate) fn execute_prefix(
        &self,
        lhs: &MetalBuffer,
        rhs: &MetalBuffer,
        words: usize,
    ) -> Result<MetalBuffer, MetalError> {
        execute_profile(
            &self.session,
            &self.pipeline,
            lhs,
            rhs,
            self.operation,
            words,
            self.fault_law,
        )
    }
}

/// Compatibility name for existing F32 prepared encoding maps.
pub type MetalPreparedF32Unary = MetalPreparedFloatUnary;
/// Compatibility name for F64 prepared encoding maps.
pub type MetalPreparedF64Unary = MetalPreparedFloatUnary;
/// Checked unary executable over exact encodings; no native floating arithmetic is executed.
pub struct MetalPreparedFloatUnary {
    fault_law: PcuCheckedScalarFaultLaw,
    session: MetalSession,
    pipeline: ffi::Pipeline,
    operation: u32,
    scalar: fusion_pcu::PcuScalarType,
}
impl MetalPreparedFloatUnary {
    pub(crate) fn with_broadcast(mut self, broadcast: bool) -> Self {
        self.operation |= u32::from(broadcast) << 16;
        self
    }
    fn input_bytes(&self, bytes: usize) -> [usize; 2] {
        [if self.operation & (1 << 16) != 0 {
            usize::from(self.scalar.bit_width()) / 8
        } else {
            bytes
        }; 2]
    }
    pub(crate) fn execute_completed(
        &self,
        input: &MetalBuffer,
        bytes: usize,
    ) -> Result<(MetalBuffer, Option<MetalFault>), MetalError> {
        execute_byte_profile_completed(
            &self.session,
            &self.pipeline,
            [input; 2],
            self.operation,
            bytes,
            self.logical_count(bytes)?,
            self.input_bytes(bytes),
            Some(self.fault_law),
        )
    }
    fn logical_count(&self, bytes: usize) -> Result<usize, MetalError> {
        let width = usize::from(self.scalar.bit_width()) / 8;
        if bytes == 0 || !bytes.is_multiple_of(width) {
            return Err(MetalError::InvalidExtent);
        }
        Ok(bytes / width)
    }
    /// Runs exact bit selection/sign inversion with nonfinite and underflow diagnostics.
    ///
    /// # Errors
    /// Returns affinity, extent, operational or checked numerical failure.
    pub fn execute(&self, input: &MetalBuffer) -> Result<MetalBuffer, MetalError> {
        self.execute_prefix(input, input.byte_len())
    }
    /// Writes the requested exact-byte prefix into a same-session borrowed output.
    ///
    /// Recovered range faults retain the terminal payload. Fatal execution faults may alter
    /// the physical prefix and require logical discard; no useful partial value is promised.
    ///
    /// # Errors
    /// Returns extent, affinity, operational or checked numerical failure.
    pub fn execute_into(
        &self,
        input: &MetalBuffer,
        output: &MetalBuffer,
        bytes: usize,
    ) -> Result<(), MetalError> {
        execute_byte_profile_into(
            &self.session,
            &self.pipeline,
            [input; 2],
            output,
            self.operation,
            bytes,
            self.logical_count(bytes)?,
            self.input_bytes(bytes),
            Some(self.fault_law),
        )
    }
    pub(crate) fn execute_prefix(
        &self,
        input: &MetalBuffer,
        bytes: usize,
    ) -> Result<MetalBuffer, MetalError> {
        execute_byte_profile(
            &self.session,
            &self.pipeline,
            [input; 2],
            self.operation,
            bytes,
            self.logical_count(bytes)?,
            self.input_bytes(bytes),
            Some(self.fault_law),
        )
    }
}
impl MetalSession {
    /// Prepares exact F32 sign inversion using integer encodings and neutral underflow policy.
    ///
    /// # Errors
    /// Returns a native compiler/pipeline failure. Nonfinite encodings fail during execution.
    pub fn prepare_f32_neg(
        &self,
        underflow: PcuFloatUnderflowPolicy,
    ) -> Result<MetalPreparedFloatUnary, MetalError> {
        Ok(MetalPreparedFloatUnary {
            fault_law: PcuCheckedScalarFaultLaw::float_unary(
                fusion_pcu::PcuScalarType::F32,
                fusion_pcu::PcuDispatchFloatUnaryOp::Neg,
                fusion_pcu::PcuRangePolicy::Reject,
                underflow,
            )
            .ok_or(MetalError::Unsupported)?,
            session: self.clone(),
            scalar: fusion_pcu::PcuScalarType::F32,
            pipeline: self.0.native.compile(INTEGER_SOURCE, "pcu_checked_u32")?,
            operation: if underflow == PcuFloatUnderflowPolicy::RejectSubnormalResult {
                7
            } else {
                6
            },
        })
    }
    /// Prepares checked F32 `ReLU` with the neutral underflow policy.
    ///
    /// # Errors
    /// Returns a native compiler/pipeline failure. Input/output use exact u32 float encodings.
    pub fn prepare_f32_relu(
        &self,
        underflow: PcuFloatUnderflowPolicy,
    ) -> Result<MetalPreparedFloatUnary, MetalError> {
        Ok(MetalPreparedFloatUnary {
            fault_law: PcuCheckedScalarFaultLaw::float_unary(
                fusion_pcu::PcuScalarType::F32,
                fusion_pcu::PcuDispatchFloatUnaryOp::Relu,
                fusion_pcu::PcuRangePolicy::Reject,
                underflow,
            )
            .ok_or(MetalError::Unsupported)?,
            session: self.clone(),
            scalar: fusion_pcu::PcuScalarType::F32,
            pipeline: self.0.native.compile(INTEGER_SOURCE, "pcu_checked_u32")?,
            operation: if underflow == PcuFloatUnderflowPolicy::RejectSubnormalResult {
                5
            } else {
                4
            },
        })
    }

    /// Prepares exact F64 sign inversion with paired unsigned encoding limbs.
    ///
    /// # Errors
    /// Returns native compilation failure; nonfinite inputs fault during execution.
    pub fn prepare_f64_neg(
        &self,
        underflow: PcuFloatUnderflowPolicy,
    ) -> Result<MetalPreparedFloatUnary, MetalError> {
        self.prepare_f64_encoding(
            if underflow == PcuFloatUnderflowPolicy::RejectSubnormalResult {
                13
            } else {
                12
            },
            fusion_pcu::PcuDispatchFloatUnaryOp::Neg,
            underflow,
        )
    }
    /// Prepares exact F64 `ReLU` with paired unsigned encoding limbs.
    ///
    /// # Errors
    /// Returns native compilation failure; nonfinite inputs fault during execution.
    pub fn prepare_f64_relu(
        &self,
        underflow: PcuFloatUnderflowPolicy,
    ) -> Result<MetalPreparedFloatUnary, MetalError> {
        self.prepare_f64_encoding(
            if underflow == PcuFloatUnderflowPolicy::RejectSubnormalResult {
                15
            } else {
                14
            },
            fusion_pcu::PcuDispatchFloatUnaryOp::Relu,
            underflow,
        )
    }
    fn prepare_f64_encoding(
        &self,
        operation: u32,
        kind: fusion_pcu::PcuDispatchFloatUnaryOp,
        underflow: PcuFloatUnderflowPolicy,
    ) -> Result<MetalPreparedFloatUnary, MetalError> {
        Ok(MetalPreparedFloatUnary {
            fault_law: PcuCheckedScalarFaultLaw::float_unary(
                fusion_pcu::PcuScalarType::F64,
                kind,
                fusion_pcu::PcuRangePolicy::Reject,
                underflow,
            )
            .ok_or(MetalError::Unsupported)?,
            session: self.clone(),
            scalar: fusion_pcu::PcuScalarType::F64,
            pipeline: self.0.native.compile(INTEGER_SOURCE, "pcu_checked_u32")?,
            operation,
        })
    }
}
fn execute_profile(
    session: &MetalSession,
    pipeline: &ffi::Pipeline,
    lhs: &MetalBuffer,
    rhs: &MetalBuffer,
    operation: u32,
    words: usize,
    fault_law: Option<PcuCheckedScalarFaultLaw>,
) -> Result<MetalBuffer, MetalError> {
    session.ensure_quiescent()?;
    if !Rc::ptr_eq(&session.0, &lhs.session.0) || !Rc::ptr_eq(&session.0, &rhs.session.0) {
        return Err(MetalError::ForeignSession);
    }
    if lhs.len() < words || rhs.len() < words {
        return Err(MetalError::InvalidExtent);
    }
    let bytes = validate_extent(words, session.0.facts.max_buffer_bytes)?;
    let output = session.0.native.allocate(bytes)?;
    let output = MetalBuffer {
        session: session.clone(),
        native: output,
        bytes,
    };
    execute_profile_into(
        session,
        pipeline,
        [lhs, rhs],
        &output,
        operation,
        words,
        fault_law,
    )?;
    Ok(output)
}

// The safe callers hold exclusive output ownership or allocate private scratch storage.
fn execute_profile_into(
    session: &MetalSession,
    pipeline: &ffi::Pipeline,
    inputs: [&MetalBuffer; 2],
    output: &MetalBuffer,
    operation: u32,
    words: usize,
    fault_law: Option<PcuCheckedScalarFaultLaw>,
) -> Result<(), MetalError> {
    session.ensure_quiescent()?;
    for buffer in [inputs[0], inputs[1], output] {
        if !session.same_session(buffer.session()) {
            return Err(MetalError::ForeignSession);
        }
        if buffer.len() < words {
            return Err(MetalError::InvalidExtent);
        }
    }
    let logical_count = if matches!(operation, 12..=22 | 32..=43) {
        if !words.is_multiple_of(2) {
            return Err(MetalError::InvalidExtent);
        }
        words / 2
    } else {
        words
    };
    execute_byte_profile_into(
        session,
        pipeline,
        inputs,
        output,
        operation,
        words.checked_mul(4).ok_or(MetalError::InvalidExtent)?,
        logical_count,
        [words.checked_mul(4).ok_or(MetalError::InvalidExtent)?; 2],
        fault_law,
    )
}

#[allow(clippy::too_many_arguments)] // Frozen physical input/output extents and diagnostic lane count differ.
fn execute_byte_profile(
    session: &MetalSession,
    pipeline: &ffi::Pipeline,
    inputs: [&MetalBuffer; 2],
    operation: u32,
    bytes: usize,
    logical_count: usize,
    input_bytes: [usize; 2],
    fault_law: Option<PcuCheckedScalarFaultLaw>,
) -> Result<MetalBuffer, MetalError> {
    let (output, fault) = execute_byte_profile_completed(
        session,
        pipeline,
        inputs,
        operation,
        bytes,
        logical_count,
        input_bytes,
        fault_law,
    )?;
    fault.map_or(Ok(output), |fault| Err(MetalError::Arithmetic(fault)))
}

#[allow(clippy::too_many_arguments)] // Detached payload, diagnostic count and physical input extents are independent.
fn execute_byte_profile_completed(
    session: &MetalSession,
    pipeline: &ffi::Pipeline,
    inputs: [&MetalBuffer; 2],
    operation: u32,
    bytes: usize,
    logical_count: usize,
    input_bytes: [usize; 2],
    fault_law: Option<PcuCheckedScalarFaultLaw>,
) -> Result<(MetalBuffer, Option<MetalFault>), MetalError> {
    session.ensure_quiescent()?;
    validate_byte_extent(bytes, session.0.facts.max_buffer_bytes)?;
    validate_extent(logical_count, session.0.facts.max_buffer_bytes)?;
    for (buffer, required) in inputs.into_iter().zip(input_bytes) {
        validate_byte_extent(required, session.0.facts.max_buffer_bytes)?;
        if !session.same_session(buffer.session()) {
            return Err(MetalError::ForeignSession);
        }
        if buffer.byte_len() < required {
            return Err(MetalError::InvalidExtent);
        }
    }
    let output = session.allocate_zeroed_bytes(bytes)?;
    let fault = match execute_byte_profile_into(
        session,
        pipeline,
        inputs,
        &output,
        operation,
        bytes,
        logical_count,
        input_bytes,
        fault_law,
    ) {
        Ok(()) => None,
        Err(MetalError::Arithmetic(fault)) if fault.recovered => Some(fault),
        Err(error) => return Err(error),
    };
    Ok((output, fault))
}

#[allow(clippy::too_many_arguments)] // Physical payload bytes and logical status lanes are independent.
fn execute_byte_profile_into(
    session: &MetalSession,
    pipeline: &ffi::Pipeline,
    inputs: [&MetalBuffer; 2],
    output: &MetalBuffer,
    operation: u32,
    payload_bytes: usize,
    logical_count: usize,
    input_bytes: [usize; 2],
    fault_law: Option<PcuCheckedScalarFaultLaw>,
) -> Result<(), MetalError> {
    session.ensure_quiescent()?;
    validate_byte_extent(payload_bytes, session.0.facts.max_buffer_bytes)?;
    for (buffer, required) in [inputs[0], inputs[1], output].into_iter().zip([
        input_bytes[0],
        input_bytes[1],
        payload_bytes,
    ]) {
        validate_byte_extent(required, session.0.facts.max_buffer_bytes)?;
        if !session.same_session(buffer.session()) {
            return Err(MetalError::ForeignSession);
        }
        if buffer.byte_len() < required {
            return Err(MetalError::InvalidExtent);
        }
    }
    let bytes = validate_extent(logical_count, session.0.facts.max_buffer_bytes)?;
    let records = session.0.native.allocate(bytes)?;
    records.fill_ones();
    let count = u32::try_from(logical_count).map_err(|_| MetalError::InvalidExtent)?;
    session.0.native.execute(
        pipeline,
        [
            &inputs[0].native,
            &inputs[1].native,
            &output.native,
            &records,
        ],
        [count, operation],
        logical_count,
    )?;
    records.inspect_words(logical_count, |records| {
        fault::validate(records, logical_count, fault_law, fault::Encoding::Scalar)?;
        select_fault(records)
    })
}

fn validate_byte_extent(bytes: usize, maximum: u64) -> Result<(), MetalError> {
    if bytes == 0 || u64::try_from(bytes).map_or(true, |bytes| bytes > maximum) {
        return Err(MetalError::InvalidExtent);
    }
    Ok(())
}

fn validate_extent(words: usize, maximum: u64) -> Result<usize, MetalError> {
    let bytes = words.checked_mul(4).ok_or(MetalError::InvalidExtent)?;
    if words == 0
        || u32::try_from(words).is_err()
        || u64::try_from(bytes).map_or(true, |bytes| bytes > maximum)
    {
        return Err(MetalError::InvalidExtent);
    }
    Ok(bytes)
}

fn select_fault(records: &[u32]) -> Result<(), MetalError> {
    // Bit 8 denotes a completed range recovery. Only overflow/underflow may carry it;
    // fatal operand/division failures and unwritten/unknown records never imply completion.
    if records
        .iter()
        .any(|&record| !matches!(record, 0..=4 | 0x101 | 0x103))
    {
        return Err(MetalError::Runtime("invalid checked status record".into()));
    }
    let selected = records
        .iter()
        .position(|&record| (1..=4).contains(&record))
        .or_else(|| records.iter().position(|&record| record != 0));
    if let Some(element) = selected {
        let record = records[element];
        return Err(MetalError::Arithmetic(MetalFault {
            invocation_id: u64::try_from(element).map_err(|_| MetalError::InvalidExtent)?,
            kind: match record & 0xff {
                1 => PcuExecutionFaultKind::ArithmeticOverflow,
                2 => PcuExecutionFaultKind::DivideByZero,
                3 => PcuExecutionFaultKind::ArithmeticUnderflow,
                _ => PcuExecutionFaultKind::InvalidFloatingOperand,
            },
            recovered: record & 0x100 != 0,
        }));
    }
    Ok(())
}

// Every path assigns its private record. Guard before division avoids undefined arithmetic;
// Signed operations use defined unsigned wrapping encodings to classify exact range failure.
// Signed multiply guards magnitude before multiplication; no signed overflow or floating ALU.
// F64 Neg implements the binary sign-bit operation (IEEE 754-2019 6.3); exact finite
// magnitude bits require no rounding. Rejecting nonfinite operands is a PCU policy,
// not an IEEE sign-operation requirement. ReLU is the neutral PCU selection contract.
const INTEGER_SOURCE: &str = r"
#include <metal_stdlib>
using namespace metal;
// Exact 32x32 -> 64 from bounded 16-bit products; every intermediate fits uint.
inline uint2 pcu_mul32(uint x, uint y) {
    uint x0 = x & 0xffffu, x1 = x >> 16, y0 = y & 0xffffu, y1 = y >> 16;
    uint w0 = x0 * y0, t = x1 * y0 + (w0 >> 16);
    uint w1 = (t & 0xffffu) + x0 * y1;
    return uint2((w1 << 16) | (w0 & 0xffffu), x1 * y1 + (t >> 16) + (w1 >> 16));
}
inline uint2 pcu_neg64(uint2 x) { return uint2(0u - x.x, 0u - x.y - uint(x.x != 0)); }
// Exact 64x64 -> 128: retain every upper limb for range classification.
inline uint4 pcu_mul64(uint2 x, uint2 y) {
    uint2 p00 = pcu_mul32(x.x, y.x), p01 = pcu_mul32(x.x, y.y);
    uint2 p10 = pcu_mul32(x.y, y.x), p11 = pcu_mul32(x.y, y.y);
    uint r1 = p00.y + p01.x, r2 = p01.y + uint(r1 < p00.y);
    uint next = r1 + p10.x, carry = next < r1; r1 = next;
    next = r2 + p10.y; uint r3 = next < r2;
    r2 = next + carry; r3 += uint(r2 < next);
    next = r2 + p11.x; r3 += p11.y + uint(next < r2);
    return uint4(p00.x, r1, next, r3);
}
kernel void pcu_checked_u32(device const uint* a [[buffer(0)]],
                            device const uint* b [[buffer(1)]],
                            device uint* out [[buffer(2)]],
                            device uint* fault [[buffer(3)]],
                            constant uint2& config [[buffer(4)]],
                            uint i [[thread_position_in_grid]]) {
    if (i >= config.x) return;
    if (config.y >= 12 && config.y <= 15) {
        uint low = a[2 * i], high = a[2 * i + 1], code = 0;
        if ((high & 0x7ff00000u) == 0x7ff00000u) {
            code = 4; low = 0; high = 0;
        } else {
            if (config.y <= 13) high ^= 0x80000000u;
            else if ((high & 0x80000000u) != 0) { low = 0; high = 0; }
            if ((config.y == 13 || config.y == 15) &&
                (high & 0x7ff00000u) == 0 && ((high & 0xfffffu) != 0 || low != 0)) code = 3;
        }
        out[2 * i] = low; out[2 * i + 1] = high; fault[i] = code;
        return;
    }
    if (config.y >= 16 && config.y <= 22) {
        uint2 x(a[2 * i], a[2 * i + 1]), y(b[2 * i], b[2 * i + 1]);
        uint2 value(0, 0); uint code = 0;
        if (config.y == 16 || config.y == 19) {
            value.x = x.x + y.x;
            uint carry = value.x < x.x;
            uint high = x.y + y.y;
            value.y = high + carry;
            if (config.y == 16) {
                if (high < x.y || value.y < high) code = 1;
            } else if (((x.y ^ value.y) & (y.y ^ value.y) & 0x80000000u) != 0)
                code = (x.y & 0x80000000u) != 0 ? 3 : 1;
        } else if (config.y == 17 || config.y == 20) {
            value.x = x.x - y.x;
            value.y = x.y - y.y - uint(x.x < y.x);
            if (config.y == 17) {
                if (x.y < y.y || (x.y == y.y && x.x < y.x)) code = 3;
            } else if (((x.y ^ y.y) & (x.y ^ value.y) & 0x80000000u) != 0)
                code = (x.y & 0x80000000u) != 0 ? 3 : 1;
        } else if (config.y == 18 || config.y == 21) {
            bool negative = config.y == 21 && ((x.y ^ y.y) & 0x80000000u) != 0;
            if (config.y == 21) {
                if ((x.y & 0x80000000u) != 0) x = pcu_neg64(x);
                if ((y.y & 0x80000000u) != 0) y = pcu_neg64(y);
            }
            uint4 product = pcu_mul64(x, y);
            bool outside = product.z != 0 || product.w != 0;
            if (config.y == 21) {
                uint2 limit = negative ? uint2(0, 0x80000000u) : uint2(0xffffffffu, 0x7fffffffu);
                outside = outside || product.y > limit.y || (product.y == limit.y && product.x > limit.x);
            }
            if (outside) code = negative ? 3 : 1;
            else { value = product.xy; if (negative) value = pcu_neg64(value); }
        } else value = x;
        out[2 * i] = value.x; out[2 * i + 1] = value.y; fault[i] = code;
        return;
    }
    uint x = a[i], y = b[i], value = 0, code = 0;
    switch (config.y) {
        case 0: if (x > 0xffffffffu - y) code = 1; else value = x + y; break;
        case 1: if (x < y) code = 3; else value = x - y; break;
        case 2: if (y != 0 && x > 0xffffffffu / y) code = 1; else value = x * y; break;
        case 3: if (y == 0) code = 2; else value = x / y; break;
        case 4: case 5:
            if ((x & 0x7f800000u) == 0x7f800000u) code = 4;
            else {
                value = (x & 0x80000000u) != 0 ? 0 : x;
                if (config.y == 5 && value != 0 && (value & 0x7f800000u) == 0) code = 3;
            }
            break;
        case 6: case 7:
            if ((x & 0x7f800000u) == 0x7f800000u) code = 4;
            else {
                value = x ^ 0x80000000u;
                if (config.y == 7 && (value & 0x7fffffffu) != 0 && (value & 0x7f800000u) == 0) code = 3;
            }
            break;
        case 8: value = x; break;
        case 9:
            value = x + y;
            if (((x ^ value) & (y ^ value) & 0x80000000u) != 0)
                code = (x & 0x80000000u) != 0 ? 3 : 1;
            break;
        case 10:
            value = x - y;
            if (((x ^ y) & (x ^ value) & 0x80000000u) != 0)
                code = (x & 0x80000000u) != 0 ? 3 : 1;
            break;
        case 11: {
            bool negative = ((x ^ y) & 0x80000000u) != 0;
            uint ax = (x & 0x80000000u) != 0 ? 0u - x : x;
            uint ay = (y & 0x80000000u) != 0 ? 0u - y : y;
            uint limit = negative ? 0x80000000u : 0x7fffffffu;
            if (ay != 0 && ax > limit / ay) code = negative ? 3 : 1;
            else { value = ax * ay; if (negative) value = 0u - value; }
            break;
        }
        default: code = 99; break;
    }
    out[i] = value;
    fault[i] = code;
}
";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extent_checks_precede_allocation() {
        assert_eq!(
            validate_byte_extent(0, u64::MAX),
            Err(MetalError::InvalidExtent)
        );
        assert_eq!(validate_byte_extent(1, 1), Ok(()));
        assert_eq!(validate_byte_extent(6, 6), Ok(()));
        assert_eq!(validate_byte_extent(6, 5), Err(MetalError::InvalidExtent));
        assert_eq!(validate_extent(0, u64::MAX), Err(MetalError::InvalidExtent));
        assert_eq!(validate_extent(2, 7), Err(MetalError::InvalidExtent));
        assert_eq!(validate_extent(2, 8), Ok(8));
        assert_eq!(
            validate_extent(usize::MAX, u64::MAX),
            Err(MetalError::InvalidExtent)
        );
    }

    #[test]
    fn records_select_first_logical_fault_and_preserve_kind() {
        assert_eq!(
            select_fault(&[0, 2, 1]),
            Err(MetalError::Arithmetic(MetalFault {
                invocation_id: 1,
                kind: PcuExecutionFaultKind::DivideByZero,
                recovered: false,
            }))
        );
        assert_eq!(select_fault(&[0, 0]), Ok(()));
        assert!(matches!(
            select_fault(&[u32::MAX]),
            Err(MetalError::Runtime(_))
        ));
    }

    #[test]
    fn recovered_records_require_complete_payload_and_fatal_priority() {
        let fault = |records| match select_fault(records) {
            Err(MetalError::Arithmetic(fault)) => fault,
            other => panic!("expected fault, got {other:?}"),
        };
        assert_eq!(
            fault(&[0x103, 0, 0x101]),
            MetalFault {
                invocation_id: 0,
                kind: PcuExecutionFaultKind::ArithmeticUnderflow,
                recovered: true
            }
        );
        assert_eq!(
            fault(&[0x103, 4, 2]),
            MetalFault {
                invocation_id: 1,
                kind: PcuExecutionFaultKind::InvalidFloatingOperand,
                recovered: false
            }
        );
        assert_eq!(
            fault(&[0, 0x101, 0x103]),
            MetalFault {
                invocation_id: 1,
                kind: PcuExecutionFaultKind::ArithmeticOverflow,
                recovered: true
            }
        );
        for invalid in [5, 0x100, 0x102, 0x104, 0x203, u32::MAX] {
            assert!(matches!(
                select_fault(&[0x103, invalid]),
                Err(MetalError::Runtime(_))
            ));
        }
    }
    #[cfg(not(target_os = "macos"))]
    #[test]
    fn unsupported_platform_never_opens_a_fallback() {
        assert_eq!(MetalSession::discover(), Err(MetalError::Unsupported));
        assert!(matches!(
            MetalSession::open(0),
            Err(MetalError::Unsupported)
        ));
    }
}

#[cfg(all(test, target_os = "macos"))]
mod hardware_tests {
    use super::*;

    #[test]
    #[ignore = "Requires an admitted macOS Metal device; runs actual compiler and GPU work."]
    fn checked_u32_maps_terminal_fault_retry_and_escaped_ownership() {
        let session = MetalSession::open(0).unwrap();
        assert!(!session.facts().name.is_empty());
        // Ordinary host byte staging may start at an unaligned address. Only the owned Shared
        // allocation requires word alignment; logical borrowed extents are never rounded up.
        let host = [91_u8, 1, 2, 3, 4, 92];
        let staged = session.upload_bytes(&host[1..5]).unwrap();
        let mut copied = [93_u8; 6];
        staged.read_into_bytes(&mut copied[1..5]).unwrap();
        assert_eq!(copied, [93, 1, 2, 3, 4, 93]);
        assert_eq!(
            staged.read_into_bytes(&mut copied),
            Err(MetalError::InvalidExtent)
        );
        assert_eq!(copied, [93, 1, 2, 3, 4, 93]);
        for length in 1..=5 {
            let staged = session.upload_bytes(&host[..length]).unwrap();
            assert_eq!(staged.byte_len(), length);
            let mut copied = vec![0; length];
            staged.read_into_bytes(&mut copied).unwrap();
            assert_eq!(copied, host[..length]);
            if !length.is_multiple_of(4) {
                assert!(matches!(
                    staged.download_u32(),
                    Err(MetalError::InvalidExtent)
                ));
            }
        }
        assert!(matches!(
            session.upload_bytes(&[]),
            Err(MetalError::InvalidExtent)
        ));
        let zeroed = session.allocate_zeroed(3).unwrap();
        let mut initialized = [94_u8; 12];
        zeroed.read_into_bytes(&mut initialized).unwrap();
        assert_eq!(initialized, [0; 12]);
        for (operation, lhs, rhs, expected) in [
            (
                MetalIntegerOp::Add,
                vec![0, 17, u32::MAX - 1],
                vec![0, 8, 1],
                vec![0, 25, u32::MAX],
            ),
            (
                MetalIntegerOp::Subtract,
                vec![0, 17, u32::MAX],
                vec![0, 8, 1],
                vec![0, 9, u32::MAX - 1],
            ),
            (
                MetalIntegerOp::Multiply,
                vec![0, 17, u32::MAX],
                vec![u32::MAX, 8, 1],
                vec![0, 136, u32::MAX],
            ),
            (
                MetalIntegerOp::Divide,
                vec![0, 17, u32::MAX],
                vec![1, 8, 1],
                vec![0, 2, u32::MAX],
            ),
        ] {
            let prepared = session.prepare_integer_map(operation).unwrap();
            let output = prepared
                .execute(
                    &session.upload_u32(&lhs).unwrap(),
                    &session.upload_u32(&rhs).unwrap(),
                )
                .unwrap();
            assert_eq!(output.download_u32().unwrap(), expected);
        }
        // Partial threadgroups and full-status transfers retain logical ordering.
        let prepared = session.prepare_integer_map(MetalIntegerOp::Add).unwrap();
        let mut left = vec![1; 257];
        left[3] = u32::MAX;
        left[256] = u32::MAX;
        let right = session.upload_u32(&vec![1; 257]).unwrap();
        assert!(matches!(
            prepared.execute(&session.upload_u32(&left).unwrap(), &right),
            Err(MetalError::Arithmetic(MetalFault {
                invocation_id: 3,
                kind: PcuExecutionFaultKind::ArithmeticOverflow,
                recovered: false
            }))
        ));
        let good = session.upload_u32(&vec![2; 257]).unwrap();
        let output = prepared.execute(&good, &right).unwrap();
        drop(prepared);
        drop(right);
        drop(good);
        drop(session);
        assert_eq!(output.download_u32().unwrap(), vec![3; 257]);
    }

    #[test]
    #[ignore = "Requires an admitted macOS Metal device; runs actual compiler and GPU work."]
    fn checked_u32_fault_kinds_and_session_affinity() {
        let session = MetalSession::open(0).unwrap();
        for (operation, lhs, rhs, zero) in [
            (
                MetalIntegerOp::Add,
                u32::MAX,
                1,
                PcuExecutionFaultKind::ArithmeticOverflow,
            ),
            (
                MetalIntegerOp::Subtract,
                0,
                1,
                PcuExecutionFaultKind::ArithmeticUnderflow,
            ),
            (
                MetalIntegerOp::Multiply,
                u32::MAX,
                2,
                PcuExecutionFaultKind::ArithmeticOverflow,
            ),
            (
                MetalIntegerOp::Divide,
                1,
                0,
                PcuExecutionFaultKind::DivideByZero,
            ),
        ] {
            let prepared = session.prepare_integer_map(operation).unwrap();
            assert!(
                matches!(prepared.execute(&session.upload_u32(&[lhs]).unwrap(), &session.upload_u32(&[rhs]).unwrap()),
                Err(MetalError::Arithmetic(MetalFault { invocation_id: 0, kind, recovered: false })) if kind == zero)
            );
        }
        let other = MetalSession::open(0).unwrap();
        let prepared = session.prepare_integer_map(MetalIntegerOp::Add).unwrap();
        assert!(matches!(
            prepared.execute(
                &session.upload_u32(&[1]).unwrap(),
                &other.upload_u32(&[1]).unwrap()
            ),
            Err(MetalError::ForeignSession)
        ));
    }
    #[test]
    #[ignore = "Requires actual macOS Metal device."]
    fn bit_only_f32_relu_nonfinite_underflow_signed_zero_and_retry() {
        let session = MetalSession::open(0).unwrap();
        let encodings = [
            0,
            0x8000_0000,
            1,
            0x8000_0001,
            0x007f_ffff,
            0x0080_0000,
            0x3f80_0000,
            0xbf80_0000,
            0x7f7f_ffff,
        ];
        let expected = [
            0,
            0,
            1,
            0,
            0x007f_ffff,
            0x0080_0000,
            0x3f80_0000,
            0,
            0x7f7f_ffff,
        ];
        for policy in [
            PcuFloatUnderflowPolicy::IeeeAfterRounding,
            PcuFloatUnderflowPolicy::AllowGradualUnderflow,
        ] {
            let prepared = session.prepare_f32_relu(policy).unwrap();
            assert_eq!(
                prepared
                    .execute(&session.upload_u32(&encodings).unwrap())
                    .unwrap()
                    .download_u32()
                    .unwrap(),
                expected
            );
            for special in [0x7f80_0000, 0xff80_0000, 0x7fc0_0001, 0x7f80_0001] {
                assert!(matches!(
                    prepared.execute(&session.upload_u32(&[special]).unwrap()),
                    Err(MetalError::Arithmetic(MetalFault {
                        kind: PcuExecutionFaultKind::InvalidFloatingOperand,
                        invocation_id: 0,
                        recovered: false
                    }))
                ));
            }
            assert_eq!(
                prepared
                    .execute(&session.upload_u32(&[0x3f80_0000]).unwrap())
                    .unwrap()
                    .download_u32()
                    .unwrap(),
                [0x3f80_0000]
            );
        }
        let tight = session
            .prepare_f32_relu(PcuFloatUnderflowPolicy::RejectSubnormalResult)
            .unwrap();
        assert!(matches!(
            tight.execute(&session.upload_u32(&[0, 1, 0x7fc0_0001]).unwrap()),
            Err(MetalError::Arithmetic(MetalFault {
                kind: PcuExecutionFaultKind::ArithmeticUnderflow,
                invocation_id: 1,
                recovered: false
            }))
        ));
        assert_eq!(
            tight
                .execute(&session.upload_u32(&[0x8000_0001]).unwrap())
                .unwrap()
                .download_u32()
                .unwrap(),
            [0]
        );
    }
    #[test]
    #[ignore = "Requires actual macOS Metal device."]
    fn bit_only_f32_neg_preserves_signed_zero_and_exact_subnormal_policy() {
        let session = MetalSession::open(0).unwrap();
        let words = [
            0,
            0x8000_0000,
            1,
            0x8000_0001,
            0x007f_ffff,
            0x0080_0000,
            0x3f80_0000,
            0xbf80_0000,
            0x7f7f_ffff,
        ];
        let expected: Vec<u32> = words.iter().map(|word| word ^ 0x8000_0000).collect();
        for policy in [
            PcuFloatUnderflowPolicy::IeeeAfterRounding,
            PcuFloatUnderflowPolicy::AllowGradualUnderflow,
        ] {
            let prepared = session.prepare_f32_neg(policy).unwrap();
            assert_eq!(
                prepared
                    .execute(&session.upload_u32(&words).unwrap())
                    .unwrap()
                    .download_u32()
                    .unwrap(),
                expected
            );
            for special in [0x7f80_0000, 0xff80_0000, 0x7fc0_0001, 0x7f80_0001] {
                assert!(matches!(
                    prepared.execute(&session.upload_u32(&[special]).unwrap()),
                    Err(MetalError::Arithmetic(MetalFault {
                        kind: PcuExecutionFaultKind::InvalidFloatingOperand,
                        invocation_id: 0,
                        recovered: false
                    }))
                ));
            }
        }
        let tight = session
            .prepare_f32_neg(PcuFloatUnderflowPolicy::RejectSubnormalResult)
            .unwrap();
        assert!(matches!(
            tight.execute(&session.upload_u32(&[0x8000_0000, 1, 0x7fc0_0001]).unwrap()),
            Err(MetalError::Arithmetic(MetalFault {
                kind: PcuExecutionFaultKind::ArithmeticUnderflow,
                invocation_id: 1,
                recovered: false
            }))
        ));
        assert_eq!(
            tight
                .execute(&session.upload_u32(&[0, 0x8000_0000, 0x0080_0000]).unwrap())
                .unwrap()
                .download_u32()
                .unwrap(),
            [0x8000_0000, 0, 0x8080_0000]
        );
    }
}

#[path = "div_rem/div_rem.rs"]
pub mod div_rem;
pub use div_rem::MetalPreparedDivRemControl;
