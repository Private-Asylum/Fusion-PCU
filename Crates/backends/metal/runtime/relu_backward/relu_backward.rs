//! Detached finite-operand six-format backward selection, exact U32 encoding and private output.
use fusion_pcu::{PcuCheckedScalarFaultLaw, PcuFloatUnderflowPolicy, PcuRangePolicy, PcuScalarType};
use super::{MetalBuffer, MetalError, MetalFault, MetalSession, ffi, execute_byte_profile_completed};
/// Frozen backward control; selected tensor admission and discovery are independent scopes.
pub struct MetalPreparedReluBackward {
    session: MetalSession,
    pipeline: ffi::Pipeline,
    scalar: PcuScalarType,
    profile: u32,
    broadcast: [bool; 2],
    law: PcuCheckedScalarFaultLaw,
}
impl MetalSession {
    /// Prepare checked six-format ReLU-backward with exact finite checks and Reject disposition.
    /// # Errors
    /// Returns unsupported scalar/platform or actual native compilation failure.
    pub fn prepare_relu_backward(
        &self,
        scalar: PcuScalarType,
        policy: PcuFloatUnderflowPolicy,
        broadcast: [bool; 2],
    ) -> Result<MetalPreparedReluBackward, MetalError> {
        self.ensure_quiescent()?;
        if !matches!(
            scalar,
            PcuScalarType::F16
                | PcuScalarType::BF16
                | PcuScalarType::F8E4M3FN
                | PcuScalarType::F8E5M2
                | PcuScalarType::F32
                | PcuScalarType::F64
        ) {
            return Err(MetalError::Unsupported);
        }
        let law =
            PcuCheckedScalarFaultLaw::float_relu_backward(scalar, PcuRangePolicy::Reject, policy)
                .ok_or(MetalError::Unsupported)?;
        let format = match scalar {
            PcuScalarType::F16 => 1_u32,
            PcuScalarType::BF16 => 2,
            PcuScalarType::F8E4M3FN => 3,
            PcuScalarType::F8E5M2 => 4,
            _ => 0,
        };
        let profile = (format << 4)
            | u32::from(scalar == PcuScalarType::F64)
            | u32::from(policy == PcuFloatUnderflowPolicy::RejectSubnormalResult) << 1
            | u32::from(broadcast[0]) << 8
            | u32::from(broadcast[1]) << 9;
        Ok(MetalPreparedReluBackward {
            session: self.clone(),
            pipeline: self.0.native.compile(
                include_str!("../../ffi/native/cpp/relu_backward/relu_backward.metal"),
                "pcu_relu_backward",
            )?,
            scalar,
            profile,
            broadcast,
            law,
        })
    }
}
impl MetalPreparedReluBackward {
    /// Return one fresh owner after terminal status validation; fatal returns no owner.
    /// # Errors
    /// Rejects actual foreign/minimum extent mismatch or native/fatal arithmetic failure.
    pub fn execute_completed(
        &self,
        input: &MetalBuffer,
        upstream: &MetalBuffer,
        count: usize,
    ) -> Result<(MetalBuffer, Option<MetalFault>), MetalError> {
        let width = usize::from(self.scalar.bit_width()) / 8;
        let bytes = count.checked_mul(width).ok_or(MetalError::InvalidExtent)?;
        execute_byte_profile_completed(
            &self.session,
            &self.pipeline,
            [input, upstream],
            self.profile,
            bytes,
            count,
            [
                if self.broadcast[0] { width } else { bytes },
                if self.broadcast[1] { width } else { bytes },
            ],
            Some(self.law),
        )
    }
}
#[cfg(all(test, target_os = "macos"))]
#[path = "tests/tests.rs"]
mod tests;
