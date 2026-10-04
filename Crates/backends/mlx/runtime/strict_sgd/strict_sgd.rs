//! Ordered MLX checked SGD updates; source admission and discovery remain separate.
use fusion_pcu::{PcuFloatUnderflowPolicy, PcuScalarType};
use super::MlxSession;
use crate::{ffi, MlxEncodedArray, MlxEncodedCompletion, MlxError};
/// Frozen nonempty update control with ordered gradient/rate multiplication and weight subtraction.
pub struct MlxPreparedStrictSgd {
    native: ffi::StrictSgd,
    session: MlxSession,
}
impl MlxSession {
    /// Prepare ordered checked nonempty gradient-times-rate then weight-minus-product updates.
    /// # Errors
    /// Returns unsupported scalar/extent/platform or native contained descriptor failure.
    pub fn prepare_strict_sgd(
        &self,
        scalar: PcuScalarType,
        policy: PcuFloatUnderflowPolicy,
        count: usize,
        learning_rate: f32,
    ) -> Result<MlxPreparedStrictSgd, MlxError> {
        let native = ffi::StrictSgd::prepare(&self.0.native, scalar, policy, count, learning_rate)?;
        Ok(MlxPreparedStrictSgd {
            native,
            session: self.clone(),
        })
    }
}
impl MlxPreparedStrictSgd {
    /// Return a fresh typed private owner after all ordered arithmetic event records complete.
    /// # Errors
    /// Rejects actual scalar/count/session mismatch, terminal uncertainty or checked fatal fault.
    pub fn execute_resident(
        &self,
        input: &MlxEncodedArray,
        upstream: &MlxEncodedArray,
    ) -> Result<MlxEncodedCompletion, MlxError> {
        if !input.same_session(&self.session) || !upstream.same_session(&self.session) {
            return Err(MlxError::ForeignSession);
        }
        let (output, notice) = self.native.execute(input.native(), upstream.native())?;
        Ok(self.session.complete_encoded(output, notice))
    }
}

#[cfg(test)]
#[path = "tests/tests.rs"]
mod tests;
