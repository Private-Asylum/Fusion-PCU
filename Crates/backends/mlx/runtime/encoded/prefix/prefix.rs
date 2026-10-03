//! Frozen immutable prefix publication through an exact retained MLX `UInt` carrier session.
#[rustfmt::skip]
use fusion_pcu::PcuScalarType;
#[rustfmt::skip]
use crate::{MlxEncodedArray,MlxError,MlxSession};

/// Cold checked and warmed immutable prefix merge. The previous owner is never mutated.
/// This plan borrows exact-session arrays and publishes a fresh completed full-length array.
pub struct MlxPreparedEncodedPrefix {
    session: MlxSession,
    scalar: PcuScalarType,
    prefix_count: usize,
    output_count: usize,
}
impl MlxPreparedEncodedPrefix {
    #[must_use]
    pub const fn scalar_type(&self) -> PcuScalarType {
        self.scalar
    }
    #[must_use]
    pub const fn prefix_count(&self) -> usize {
        self.prefix_count
    }
    #[must_use]
    pub const fn output_count(&self) -> usize {
        self.output_count
    }
    /// Borrows a completed logical prefix and previous longer owner without host materialization.
    /// A separate retained official holder prevents native donation of the previous backing.
    ///
    /// # Errors
    /// Rejects logical dtype, exact extent or retained-session mismatch before lazy construction;
    /// unknown device completion quarantines the real pending owners and their session.
    pub fn execute(
        &self,
        prefix: &MlxEncodedArray,
        previous: &MlxEncodedArray,
    ) -> Result<MlxEncodedArray, MlxError> {
        for input in [prefix, previous] {
            if input.scalar_type() != self.scalar {
                return Err(MlxError::UnsupportedScalar(input.scalar_type()));
            }
            if !input.same_session(&self.session) {
                return Err(MlxError::ForeignSession);
            }
            input.validate_access_available()?;
        }
        if prefix.element_count() != self.prefix_count
            || previous.element_count() != self.output_count
        {
            return Err(MlxError::InvalidExtent);
        }
        Ok(self
            .session
            .wrap_encoded(previous.native().merge_prefix(prefix.native())?))
    }
}
impl MlxSession {
    /// Freezes an exact byte-aligned `UInt` prefix merge and warms physical dtype/shape copies.
    /// Execution retains this same session; it does not reopen a device or stage host data.
    ///
    /// # Errors
    /// Rejects unsupported dtype, zero prefix, nonlarger output or oversized allocation;
    /// returns native staging/compilation/completion failures without exposing a prepared plan.
    pub fn prepare_encoded_prefix(
        &self,
        scalar: PcuScalarType,
        prefix_count: usize,
        output_count: usize,
    ) -> Result<MlxPreparedEncodedPrefix, MlxError> {
        if matches!(
            scalar,
            PcuScalarType::Bool | PcuScalarType::I4 | PcuScalarType::U4
        ) {
            return Err(MlxError::UnsupportedScalar(scalar));
        }
        let width = usize::from(scalar.bit_width()) / 8;
        let output_bytes = output_count
            .checked_mul(width)
            .ok_or(MlxError::InvalidExtent)?;
        if prefix_count == 0
            || prefix_count >= output_count
            || i32::try_from(output_bytes / width.min(4)).is_err()
            || isize::try_from(output_bytes).is_err()
        {
            return Err(MlxError::InvalidExtent);
        }
        let plan = MlxPreparedEncodedPrefix {
            session: self.clone(),
            scalar,
            prefix_count,
            output_count,
        };
        let prefix =
            self.upload_encoded_bytes(scalar, prefix_count, &vec![0; prefix_count * width])?;
        let previous = self.upload_encoded_bytes(scalar, output_count, &vec![0; output_bytes])?;
        drop(plan.execute(&prefix, &previous)?);
        Ok(plan)
    }
}
