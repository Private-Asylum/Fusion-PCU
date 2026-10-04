//! Ordered MLX checked bounded squared-error reduction; source admission and discovery remain separate.
use fusion_pcu::{PcuFloatUnderflowPolicy, PcuScalarType};
use super::MlxSession;
use crate::{ffi, MlxEncodedArray, MlxEncodedCompletion, MlxError};
/// Frozen nonempty control with ordered checked subtract/multiply/add events and final division.
pub struct MlxPreparedStrictMse {
    native: ffi::StrictMse,
    session: MlxSession,
}
impl MlxSession {
    /// Prepare a checked Strict F32/F64 nonempty squared-error reduction with Reject disposition.
    /// The private two-word receipt retains the full original event domain; count is bounded to 65535.
    /// # Errors
    /// Returns unsupported scalar/extent/platform or native contained descriptor failure.
    pub fn prepare_strict_mse(
        &self,
        scalar: PcuScalarType,
        policy: PcuFloatUnderflowPolicy,
        elements: usize,
    ) -> Result<MlxPreparedStrictMse, MlxError> {
        let native = ffi::StrictMse::prepare(&self.0.native, scalar, policy, elements)?;
        Ok(MlxPreparedStrictMse {
            native,
            session: self.clone(),
        })
    }
}
impl MlxPreparedStrictMse {
    /// Return a fresh typed private owner after the ordered original-event/status receipt completes.
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
