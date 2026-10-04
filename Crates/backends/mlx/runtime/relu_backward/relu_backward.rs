//! Actual MLX finite backward selection; tensor and source admission remain independent.
use fusion_pcu::{PcuFloatUnderflowPolicy, PcuScalarType};
use super::MlxSession;
use crate::{ffi, MlxEncodedArray, MlxEncodedCompletion, MlxError};
/// Frozen detached backward primitive with exact two-input immutable shapes.
pub struct MlxPreparedReluBackward {
    native: ffi::ReluBackward,
    session: MlxSession,
}
impl MlxSession {
    /// Prepare checked six-format finite backward selection, Reject policy and two broadcast roles.
    /// # Errors
    /// Returns unsupported scalar/extent/platform or native contained descriptor failure.
    pub fn prepare_relu_backward(
        &self,
        scalar: PcuScalarType,
        policy: PcuFloatUnderflowPolicy,
        count: usize,
        broadcast: [bool; 2],
    ) -> Result<MlxPreparedReluBackward, MlxError> {
        let native = ffi::ReluBackward::prepare(&self.0.native, scalar, policy, count, broadcast)?;
        Ok(MlxPreparedReluBackward {
            native,
            session: self.clone(),
        })
    }
}
impl MlxPreparedReluBackward {
    /// Return a fresh exact typed private owner after both finite operands and status complete.
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
