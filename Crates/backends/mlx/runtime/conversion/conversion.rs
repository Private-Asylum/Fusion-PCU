//! Direct checked conversion control using immutable MLX-owned encoded arrays.
#[rustfmt::skip]
use fusion_pcu::{PcuDispatchCheckedFloatConversion,PcuFloatUnderflowPolicy,PcuRangePolicy};
use super::MlxSession;
use crate::{ffi, MlxEncodedArray, MlxEncodedCompletion, MlxError};
/// Frozen conversion primitive; ordinary source/aggregate offers remain independently gated.
pub struct MlxPreparedFloatConversion {
    native: ffi::Conversion,
    session: MlxSession,
}
impl MlxSession {
    /// Prepare exact F32/F64 conversion with the original range/underflow choice.
    /// # Errors
    /// Refuses unsupported platform, zero/overflowing shapes, native descriptor or stream failures.
    pub fn prepare_float_conversion(
        &self,
        conversion: PcuDispatchCheckedFloatConversion,
        underflow: PcuFloatUnderflowPolicy,
        range: PcuRangePolicy,
        count: usize,
        broadcast: bool,
    ) -> Result<MlxPreparedFloatConversion, MlxError> {
        let native = ffi::Conversion::prepare(
            &self.0.native,
            conversion,
            underflow,
            range,
            count,
            broadcast,
        )?;
        Ok(MlxPreparedFloatConversion {
            native,
            session: self.clone(),
        })
    }
    /// Freeze actual native input capacity cold while reading only the logical prefix.
    /// No input reshaping, host copy or kernel compilation occurs during resident replay.
    /// # Errors
    /// Rejects an input extent shorter than its logical read span or native descriptor failure.
    pub fn prepare_float_conversion_with_input_extent(
        &self,
        conversion: PcuDispatchCheckedFloatConversion,
        underflow: PcuFloatUnderflowPolicy,
        range: PcuRangePolicy,
        count: usize,
        broadcast: bool,
        input_count: usize,
    ) -> Result<MlxPreparedFloatConversion, MlxError> {
        let native = ffi::Conversion::prepare_with_input_extent(
            &self.0.native,
            conversion,
            underflow,
            range,
            count,
            broadcast,
            input_count,
        )?;
        Ok(MlxPreparedFloatConversion {
            native,
            session: self.clone(),
        })
    }
}
impl MlxPreparedFloatConversion {
    /// Complete a fresh converted owner plus observable recovered notice.
    /// # Errors
    /// Refuses scalar/count/actual-session mismatch; fatal arithmetic returns no owner.
    pub fn execute_resident(
        &self,
        input: &MlxEncodedArray,
    ) -> Result<MlxEncodedCompletion, MlxError> {
        if !input.same_session(&self.session) {
            return Err(MlxError::ForeignSession);
        }
        let (output, notice) = self.native.execute(input.native())?;
        Ok(self.session.complete_encoded(output, notice))
    }
}

#[cfg(test)]
#[path = "tests/tests.rs"]
mod tests;
