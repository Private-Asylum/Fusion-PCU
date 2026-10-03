//! Checked low-format unary realization over exact initialized encoding bytes.
#[rustfmt::skip]
use super::{
    MetalError,
    MetalPreparedFloatUnary,
    MetalSession,
    PcuFloatUnderflowPolicy,
};
#[rustfmt::skip]
use fusion_pcu::{
    PcuDispatchFloatUnaryOp,
    PcuScalarType,
    PcuRangePolicy,
};
impl MetalSession {
    /// Prepares checked encoding Neg/ReLU over all six sealed checked-float formats.
    ///
    /// The F64 realization uses two U32 limbs; every format performs exact sign selection,
    /// with complete borrowed-output Clamp recovery and scalar-readonly broadcast support
    /// supplied by detached neutral admission. No native floating instruction is executed.
    ///
    /// # Errors
    /// Rejects other representations or returns a native compiler/pipeline error.
    pub fn prepare_checked_float_unary_with_range(
        &self,
        scalar: PcuScalarType,
        operation: PcuDispatchFloatUnaryOp,
        underflow: PcuFloatUnderflowPolicy,
        range: PcuRangePolicy,
    ) -> Result<MetalPreparedFloatUnary, MetalError> {
        let entry = match scalar {
            PcuScalarType::F32 => "pcu_unary_f32",
            PcuScalarType::F64 => "pcu_unary_f64",
            _ => {
                return self
                    .prepare_low_precision_unary_with_range(scalar, operation, underflow, range);
            }
        };
        let fault_law =
            fusion_pcu::PcuCheckedScalarFaultLaw::float_unary(scalar, operation, range, underflow)
                .ok_or(MetalError::Unsupported)?;
        let operation = u32::from(operation == PcuDispatchFloatUnaryOp::Relu);
        let policy = match underflow {
            PcuFloatUnderflowPolicy::IeeeAfterRounding => 0,
            PcuFloatUnderflowPolicy::RejectSubnormalResult => 1,
            PcuFloatUnderflowPolicy::AllowGradualUnderflow => 2,
        };
        Ok(MetalPreparedFloatUnary {
            fault_law,
            session: self.clone(),
            pipeline: self.0.native.compile(
                include_str!(
                    "../../ffi/native/cpp/checked_native_unary/checked_native_unary.metal"
                ),
                entry,
            )?,
            operation: operation | policy << 1 | u32::from(range == PcuRangePolicy::Clamp) << 8,
            scalar,
        })
    }
    /// Prepares exact finite-input F16/BF16/E4M3FN/E5M2 sign inversion or `ReLU`.
    ///
    /// Output subnormal rejection is independent of range checking. These exact selections
    /// cannot produce an inexact-tiny IEEE result. No native floating operation is used.
    ///
    /// # Errors
    /// Rejects other scalar formats or returns a native compilation/pipeline failure.
    pub fn prepare_low_precision_unary(
        &self,
        scalar: PcuScalarType,
        operation: PcuDispatchFloatUnaryOp,
        underflow: PcuFloatUnderflowPolicy,
    ) -> Result<MetalPreparedFloatUnary, MetalError> {
        self.prepare_low_precision_unary_with_range(
            scalar,
            operation,
            underflow,
            PcuRangePolicy::Reject,
        )
    }
    /// Prepares an exact low-format unary map with observable borrowed-output range recovery.
    ///
    /// # Errors
    /// Rejects other formats or returns a native compilation/pipeline failure.
    pub fn prepare_low_precision_unary_with_range(
        &self,
        scalar: PcuScalarType,
        operation: PcuDispatchFloatUnaryOp,
        underflow: PcuFloatUnderflowPolicy,
        range: PcuRangePolicy,
    ) -> Result<MetalPreparedFloatUnary, MetalError> {
        let entry = match scalar {
            PcuScalarType::F16 => "pcu_unary_f16",
            PcuScalarType::BF16 => "pcu_unary_bf16",
            PcuScalarType::F8E4M3FN => "pcu_unary_e4m3fn",
            PcuScalarType::F8E5M2 => "pcu_unary_e5m2",
            _ => return Err(MetalError::Unsupported),
        };
        let fault_law =
            fusion_pcu::PcuCheckedScalarFaultLaw::float_unary(scalar, operation, range, underflow)
                .ok_or(MetalError::Unsupported)?;
        let operation = match operation {
            PcuDispatchFloatUnaryOp::Neg => 0,
            PcuDispatchFloatUnaryOp::Relu => 1,
        };
        let policy = match underflow {
            PcuFloatUnderflowPolicy::IeeeAfterRounding => 0,
            PcuFloatUnderflowPolicy::RejectSubnormalResult => 1,
            PcuFloatUnderflowPolicy::AllowGradualUnderflow => 2,
        };
        Ok(MetalPreparedFloatUnary {
            fault_law,
            session: self.clone(),
            pipeline: self.0.native.compile(
                include_str!("../../ffi/native/cpp/checked_low_unary/checked_low_unary.metal"),
                entry,
            )?,
            operation: operation | policy << 1 | u32::from(range == PcuRangePolicy::Clamp) << 8,
            scalar,
        })
    }
}

#[cfg(all(test, target_os = "macos"))]
#[path = "tests/tests.rs"]
mod tests;

#[cfg(all(test, target_os = "macos"))]
#[path = "tests/native.rs"]
mod native_tests;
