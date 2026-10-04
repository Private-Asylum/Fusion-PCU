//! Detached mixed-width conversion using U32 encodings and terminal private output.
#[rustfmt::skip]
use fusion_pcu::{PcuCheckedScalarFaultLaw,PcuDispatchCheckedFloatConversion,PcuFloatUnderflowPolicy,PcuRangePolicy};
use super::{MetalBuffer, MetalError, MetalFault, MetalSession, execute_byte_profile_completed, ffi};

/// Independently prepared conversion control; no source admission or capability is implied.
pub struct MetalPreparedFloatConversion {
    session: MetalSession,
    pipeline: ffi::Pipeline,
    conversion: PcuDispatchCheckedFloatConversion,
    operation: u32,
    broadcast: bool,
    law: PcuCheckedScalarFaultLaw,
}
impl MetalSession {
    /// Prepare exact checked F32/F64 conversion without native floating arithmetic.
    /// # Errors
    /// Returns unsupported platform or native compilation failure.
    pub fn prepare_float_conversion(
        &self,
        conversion: PcuDispatchCheckedFloatConversion,
        underflow: PcuFloatUnderflowPolicy,
        range: PcuRangePolicy,
        broadcast: bool,
    ) -> Result<MetalPreparedFloatConversion, MetalError> {
        self.ensure_quiescent()?;
        let direction = u32::from(conversion == PcuDispatchCheckedFloatConversion::F64ToF32);
        let policy = match underflow {
            PcuFloatUnderflowPolicy::IeeeAfterRounding => 0,
            PcuFloatUnderflowPolicy::RejectSubnormalResult => 1,
            PcuFloatUnderflowPolicy::AllowGradualUnderflow => 2,
        };
        Ok(MetalPreparedFloatConversion {
            session: self.clone(),
            pipeline: self.0.native.compile(
                include_str!("../../ffi/native/cpp/checked_conversion/checked_conversion.metal"),
                "pcu_checked_conversion",
            )?,
            conversion,
            operation: direction
                | policy << 1
                | u32::from(range == PcuRangePolicy::Clamp) << 8
                | u32::from(broadcast) << 16,
            broadcast,
            law: PcuCheckedScalarFaultLaw::float_conversion(conversion, range, underflow),
        })
    }
}
impl MetalPreparedFloatConversion {
    /// Complete a fresh private converted buffer and recovered notice after all status checks.
    /// # Errors
    /// Rejects session/extent mismatch before submission; fatal faults return no output owner.
    pub fn execute_completed(
        &self,
        input: &MetalBuffer,
        count: usize,
    ) -> Result<(MetalBuffer, Option<MetalFault>), MetalError> {
        let (source_width, output_width) = match self.conversion {
            PcuDispatchCheckedFloatConversion::F32ToF64 => (4, 8),
            PcuDispatchCheckedFloatConversion::F64ToF32 => (8, 4),
        };
        let input_bytes = if self.broadcast {
            source_width
        } else {
            count
                .checked_mul(source_width)
                .ok_or(MetalError::InvalidExtent)?
        };
        let bytes = count
            .checked_mul(output_width)
            .ok_or(MetalError::InvalidExtent)?;
        execute_byte_profile_completed(
            &self.session,
            &self.pipeline,
            [input, input],
            self.operation,
            bytes,
            count,
            [input_bytes; 2],
            Some(self.law),
        )
    }
}

#[cfg(all(test, target_os = "macos"))]
#[path = "tests/tests.rs"]
mod tests;
