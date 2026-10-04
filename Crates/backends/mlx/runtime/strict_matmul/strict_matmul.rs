//! Ordered MLX checked dot products; source admission and discovery remain separate.
#[rustfmt::skip]
use fusion_pcu::{
    PcuFloatUnderflowPolicy,
    PcuScalarType,
};
use super::MlxSession;
#[rustfmt::skip]
use crate::{
    ffi,
    MlxEncodedArray,
    MlxEncodedCompletion,
    MlxError,
};
/// Frozen nonempty row-major 2D control with ordered checked multiply/add events.
pub struct MlxPreparedStrictMatMul {
    native: ffi::StrictMatMul,
    session: MlxSession,
}
impl MlxSession {
    /// Prepare a checked Strict F32/F64 nonempty 2D dot product with Reject disposition.
    /// Compact per-cell receipts bound the inner extent to 65535 and the full event domain to u32.
    /// # Errors
    /// Returns unsupported scalar/extent/platform or native contained descriptor failure.
    pub fn prepare_strict_matmul(
        &self,
        scalar: PcuScalarType,
        policy: PcuFloatUnderflowPolicy,
        shape: [usize; 3],
    ) -> Result<MlxPreparedStrictMatMul, MlxError> {
        let native = ffi::StrictMatMul::prepare(&self.0.native, scalar, policy, shape)?;
        Ok(MlxPreparedStrictMatMul {
            native,
            session: self.clone(),
        })
    }
}
impl MlxPreparedStrictMatMul {
    /// Full ordered semantic multiply/add fault domain, independent of receipt capacity.
    #[must_use]
    pub const fn fault_domain(&self) -> fusion_pcu::dialect::tensor::TensorStrictFaultDomain {
        self.native.fault_domain()
    }
    /// Actual compact native status word extent, independent of the semantic event domain.
    #[must_use]
    pub const fn status_word_extent(&self) -> usize {
        self.native.status_word_extent()
    }
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
