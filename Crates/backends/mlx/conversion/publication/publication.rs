//! Retained private host publication shared by direct and ordinary mixed calls.

use super::MlxPreparedConversionHostKernel;
use crate::{MlxEncodedCompletion, MlxError};

impl MlxPreparedConversionHostKernel {
    /// Publish a terminal conversion through the already retained private buffer.
    ///
    /// Readback and checked native release both precede caller RAM publication.
    /// A recovered Clamp notice is returned after its useful prefix is copied;
    /// fatal failures preserve all caller bytes, and destination tails survive.
    /// No new buffer or native operation is prepared on this path.
    ///
    /// # Errors
    /// Refuses foreign/type/shape/short output, readback or native release failure,
    /// or reports the completed conversion's recovered arithmetic notice.
    pub fn publish_host_completion(
        &mut self,
        completion: MlxEncodedCompletion,
        output: &mut [u8],
    ) -> Result<(), MlxError> {
        self.may_have_written = false;
        let (completed, notice) = completion.into_parts();
        if !completed.same_session(&self.session)
            || completed.scalar_type() != self.plan.output_scalar()
            || completed.element_count() != self.plan.count
            || completed.byte_len() != self.readback.len()
            || output.len() < self.readback.len()
        {
            return Err(MlxError::InvalidExtent);
        }
        completed.read_bytes_into(&mut self.readback)?;
        completed.release()?;
        output[..self.readback.len()].copy_from_slice(&self.readback);
        self.may_have_written = true;
        notice.map_or(Ok(()), |fault| Err(MlxError::Arithmetic(fault)))
    }
}
