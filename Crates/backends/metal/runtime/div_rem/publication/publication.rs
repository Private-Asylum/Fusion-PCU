//! One terminal device publication command for both completed private quotient/remainder halves.
#[rustfmt::skip]
use super::super::{MetalSession,MetalBuffer,MetalError,ffi};
pub struct Publication {
    session: MetalSession,
    pipeline: ffi::Pipeline,
    count: usize,
    bytes: usize,
}
impl Publication {
    pub(crate) fn prepare(
        session: &MetalSession,
        bytes: usize,
        count: usize,
    ) -> Result<Self, MetalError> {
        let width = bytes
            .checked_div(count)
            .filter(|_| count > 0)
            .ok_or(MetalError::InvalidExtent)?;
        let source = format!(
            "#include <metal_stdlib>\nusing namespace metal;\nkernel void pcu_publish_div_rem(device const uchar* packed [[buffer(0)]],device const uchar* unused [[buffer(1)]],device uchar* q [[buffer(2)]],device uchar* r [[buffer(3)]],constant uint2& profile [[buffer(4)]],uint id [[thread_position_in_grid]]){{if(id>=profile.x)return;(void)unused;for(uint byte=0u;byte<{width}u;++byte){{uint index=id*{width}u+byte;q[index]=packed[index];r[index]=packed[profile.x*{width}u+index];}}}} "
        );
        let pipeline = session.0.native.compile(&source, "pcu_publish_div_rem")?;
        Ok(Self {
            session: session.clone(),
            pipeline,
            count,
            bytes,
        })
    }
    pub(crate) fn execute(
        &self,
        packed: &MetalBuffer,
        outputs: [&MetalBuffer; 2],
    ) -> Result<(), MetalError> {
        if std::ptr::eq(outputs[0], outputs[1]) {
            return Err(MetalError::Unsupported);
        }
        for (buffer, bytes) in [packed, outputs[0], outputs[1]].into_iter().zip([
            self.bytes * 2,
            self.bytes,
            self.bytes,
        ]) {
            if !self.session.same_session(buffer.session()) {
                return Err(MetalError::ForeignSession);
            }
            if buffer.byte_len() < bytes {
                return Err(MetalError::InvalidExtent);
            }
        }
        self.session.ensure_quiescent()?;
        self.session.0.native.execute(
            &self.pipeline,
            [
                &packed.native,
                &packed.native,
                &outputs[0].native,
                &outputs[1].native,
            ],
            [
                u32::try_from(self.count).map_err(|_| MetalError::InvalidExtent)?,
                0,
            ],
            self.count,
        )
    }
}
