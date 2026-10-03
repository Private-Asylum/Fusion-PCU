//! Exact six-format checked floating arithmetic synthesized from unsigned integer encodings.
#[rustfmt::skip]
use super::{
    MetalBuffer,
    MetalError,
    MetalSession,
    PcuFloatUnderflowPolicy,
    execute_byte_profile,
    execute_byte_profile_completed,
    execute_byte_profile_into,
    ffi,
};
use fusion_pcu::PcuDispatchFloatBinaryOp;

/// One checked F16/BF16/F32/F64/OFP8 Add/Sub/Mul/Div with exact ties-even rounding and typed faults.
pub struct MetalPreparedFloatBinary {
    scalar: fusion_pcu::PcuScalarType,
    fault_law: fusion_pcu::PcuCheckedScalarFaultLaw,
    kind: PcuDispatchFloatBinaryOp,
    underflow: PcuFloatUnderflowPolicy,
    session: MetalSession,
    pipeline: ffi::Pipeline,
    operation: u32,
    broadcast: [bool; 2],
}
/// Compatibility name for exact binary64 preparation.
pub type MetalPreparedF64Binary = MetalPreparedFloatBinary;
/// Exact binary32 preparation with a separately proved U32 shader.
pub type MetalPreparedF32Binary = MetalPreparedFloatBinary;
impl MetalPreparedFloatBinary {
    /// Benchmark-only raw device boundary matching fresh completed host output/status work.
    ///
    /// # Errors
    /// Returns checked arithmetic, extent, session or terminal runtime failure.
    #[cfg(feature = "benchmark-control")]
    pub fn execute_completed_control(
        &self,
        inputs: [&MetalBuffer; 2],
        bytes: usize,
    ) -> Result<(MetalBuffer, Option<crate::MetalFault>), MetalError> {
        self.execute_completed(inputs, bytes)
    }
    fn with_range(mut self, range: fusion_pcu::PcuRangePolicy) -> Result<Self, MetalError> {
        self.operation |= u32::from(range == fusion_pcu::PcuRangePolicy::Clamp) << 8;
        self.fault_law = fusion_pcu::PcuCheckedScalarFaultLaw::float_binary(
            self.scalar,
            self.kind,
            range,
            self.underflow,
        )
        .ok_or(MetalError::Unsupported)?;
        Ok(self)
    }
    pub(crate) fn execute_completed(
        &self,
        inputs: [&MetalBuffer; 2],
        bytes: usize,
    ) -> Result<(MetalBuffer, Option<crate::MetalFault>), MetalError> {
        execute_byte_profile_completed(
            &self.session,
            &self.pipeline,
            inputs,
            self.operation,
            bytes,
            self.logical_count(bytes)?,
            self.input_bytes(bytes),
            Some(self.fault_law),
        )
    }
    pub(crate) fn with_broadcast(mut self, broadcast: [bool; 2]) -> Self {
        self.broadcast = broadcast;
        self.operation |= u32::from(broadcast[0]) << 16 | u32::from(broadcast[1]) << 17;
        self
    }
    fn input_bytes(&self, output_bytes: usize) -> [usize; 2] {
        let width = usize::from(self.scalar.bit_width()) / 8;
        self.broadcast
            .map(|broadcast| if broadcast { width } else { output_bytes })
    }

    fn logical_count(&self, bytes: usize) -> Result<usize, MetalError> {
        let width = usize::from(self.scalar.bit_width()) / 8;
        if bytes == 0 || !bytes.is_multiple_of(width) {
            return Err(MetalError::InvalidExtent);
        }
        Ok(bytes / width)
    }
    #[must_use]
    pub const fn scalar_type(&self) -> fusion_pcu::PcuScalarType {
        self.scalar
    }

    /// Runs two exact scalar encoding buffers with fresh terminal output.
    ///
    /// # Errors
    /// Returns shape/session, terminal runtime or checked numerical failure.
    pub fn execute(
        &self,
        left: &MetalBuffer,
        right: &MetalBuffer,
    ) -> Result<MetalBuffer, MetalError> {
        if left.byte_len() != right.byte_len() {
            return Err(MetalError::InvalidExtent);
        }
        self.execute_prefix([left, right], left.byte_len())
    }
    pub(crate) fn execute_prefix(
        &self,
        inputs: [&MetalBuffer; 2],
        bytes: usize,
    ) -> Result<MetalBuffer, MetalError> {
        execute_byte_profile(
            &self.session,
            &self.pipeline,
            inputs,
            self.operation,
            bytes,
            self.logical_count(bytes)?,
            self.input_bytes(bytes),
            Some(self.fault_law),
        )
    }
    /// Executes into a same-session borrowed exact-byte output prefix.
    /// Recovered Clamp faults preserve the complete terminal payload and return a structured fault.
    ///
    /// # Errors
    /// Returns extent/affinity, terminal/native failure or a checked arithmetic fault.
    pub fn execute_into(
        &self,
        inputs: [&MetalBuffer; 2],
        output: &MetalBuffer,
        bytes: usize,
    ) -> Result<(), MetalError> {
        execute_byte_profile_into(
            &self.session,
            &self.pipeline,
            inputs,
            output,
            self.operation,
            bytes,
            self.logical_count(bytes)?,
            self.input_bytes(bytes),
            Some(self.fault_law),
        )
    }
}
impl MetalSession {
    /// Compiles one six-format exact binary map with observable borrowed-output range recovery.
    ///
    /// # Errors
    /// Rejects other representations or returns a native compiler/pipeline failure.
    pub fn prepare_checked_float_binary_with_range(
        &self,
        scalar: fusion_pcu::PcuScalarType,
        operation: PcuDispatchFloatBinaryOp,
        underflow: PcuFloatUnderflowPolicy,
        range: fusion_pcu::PcuRangePolicy,
    ) -> Result<MetalPreparedFloatBinary, MetalError> {
        self.prepare_float_binary(operation, underflow, scalar)
            .and_then(|map| map.with_range(range))
    }
    /// Compiles a checked F16/BF16/E4M3FN/E5M2 integer realization for an exact scalar encoding.
    ///
    /// # Errors
    /// Rejects other scalar types before compilation or returns native pipeline failure.
    pub fn prepare_low_precision_binary(
        &self,
        scalar: fusion_pcu::PcuScalarType,
        operation: PcuDispatchFloatBinaryOp,
        underflow: PcuFloatUnderflowPolicy,
    ) -> Result<MetalPreparedFloatBinary, MetalError> {
        if !matches!(
            scalar,
            fusion_pcu::PcuScalarType::F16
                | fusion_pcu::PcuScalarType::BF16
                | fusion_pcu::PcuScalarType::F8E4M3FN
                | fusion_pcu::PcuScalarType::F8E5M2
        ) {
            return Err(MetalError::Unsupported);
        }
        self.prepare_float_binary(operation, underflow, scalar)
    }

    /// Compiles an exact checked binary64 integer realization; no native double ALU is used.
    ///
    /// # Errors
    /// Returns a native compilation/pipeline failure or unsupported operation.
    pub fn prepare_f64_binary(
        &self,
        operation: PcuDispatchFloatBinaryOp,
        underflow: PcuFloatUnderflowPolicy,
    ) -> Result<MetalPreparedFloatBinary, MetalError> {
        self.prepare_float_binary(operation, underflow, fusion_pcu::PcuScalarType::F64)
    }
    /// Compiles an exact checked binary32 U32 realization; no native float ALU is used.
    ///
    /// # Errors
    /// Returns a native compilation/pipeline failure.
    pub fn prepare_f32_binary(
        &self,
        operation: PcuDispatchFloatBinaryOp,
        underflow: PcuFloatUnderflowPolicy,
    ) -> Result<MetalPreparedF32Binary, MetalError> {
        self.prepare_float_binary(operation, underflow, fusion_pcu::PcuScalarType::F32)
    }
    fn prepare_float_binary(
        &self,
        operation: PcuDispatchFloatBinaryOp,
        underflow: PcuFloatUnderflowPolicy,
        scalar: fusion_pcu::PcuScalarType,
    ) -> Result<MetalPreparedFloatBinary, MetalError> {
        let kind = operation;
        let fault_law = fusion_pcu::PcuCheckedScalarFaultLaw::float_binary(
            scalar,
            kind,
            fusion_pcu::PcuRangePolicy::Reject,
            underflow,
        )
        .ok_or(MetalError::Unsupported)?;
        let operation = match operation {
            PcuDispatchFloatBinaryOp::Add => 0,
            PcuDispatchFloatBinaryOp::Sub => 1,
            PcuDispatchFloatBinaryOp::Mul => 2,
            PcuDispatchFloatBinaryOp::Div => 3,
        };
        let policy = match underflow {
            PcuFloatUnderflowPolicy::IeeeAfterRounding => 0,
            PcuFloatUnderflowPolicy::RejectSubnormalResult => 1,
            PcuFloatUnderflowPolicy::AllowGradualUnderflow => 2,
        };
        let (source, entry, offset) = match scalar {
            fusion_pcu::PcuScalarType::F64 => (
                include_str!("../../ffi/native/cpp/checked_f64/checked_f64.metal"),
                "pcu_checked_f64",
                32,
            ),
            fusion_pcu::PcuScalarType::F32 => (
                include_str!("../../ffi/native/cpp/checked_f32/checked_f32.metal"),
                "pcu_checked_f32",
                44,
            ),
            fusion_pcu::PcuScalarType::F16 => (
                include_str!(
                    "../../ffi/native/cpp/checked_low_precision/checked_low_precision.metal"
                ),
                "pcu_checked_f16",
                56,
            ),
            fusion_pcu::PcuScalarType::BF16 => (
                include_str!(
                    "../../ffi/native/cpp/checked_low_precision/checked_low_precision.metal"
                ),
                "pcu_checked_bf16",
                68,
            ),
            fusion_pcu::PcuScalarType::F8E4M3FN => (
                include_str!(
                    "../../ffi/native/cpp/checked_low_precision/checked_low_precision.metal"
                ),
                "pcu_checked_e4m3fn",
                80,
            ),
            fusion_pcu::PcuScalarType::F8E5M2 => (
                include_str!(
                    "../../ffi/native/cpp/checked_low_precision/checked_low_precision.metal"
                ),
                "pcu_checked_e5m2",
                92,
            ),
            _ => return Err(MetalError::Unsupported),
        };
        Ok(MetalPreparedFloatBinary {
            fault_law,
            kind,
            underflow,
            scalar,
            session: self.clone(),
            pipeline: self.0.native.compile(source, entry)?,
            operation: offset + operation + policy * 4,
            broadcast: [false; 2],
        })
    }
}

#[cfg(all(test, target_os = "macos"))]
#[path = "tests/tests.rs"]
mod tests;

#[cfg(all(test, target_os = "macos"))]
#[path = "tests/range.rs"]
mod range;
