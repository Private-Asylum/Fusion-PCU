//! Exact byte transport through the same terminal resource-retention protocol.
#[rustfmt::skip]
use super::{ffi,MetalSession,MetalBuffer,MetalError,execute_byte_profile,execute_byte_profile_into};
/// Direct exact-byte kernel control; scalar bits are never interpreted as arithmetic.
pub struct Carrier {
    session: MetalSession,
    pipeline: ffi::Pipeline,
    parameters: u32,
    count: usize,
    bytes: usize,
    input_bytes: usize,
}
impl Carrier {
    pub(crate) fn prepare(
        session: &MetalSession,
        count: usize,
        width: usize,
        broadcast: bool,
    ) -> Result<Self, MetalError> {
        let bytes = count.checked_mul(width).ok_or(MetalError::InvalidExtent)?;
        if count == 0
            || u32::try_from(bytes).is_err()
            || !matches!(width, 1 | 2 | 4 | 8 | 16 | 32 | 64)
        {
            return Err(MetalError::InvalidExtent);
        }
        super::validate_byte_extent(bytes, session.0.facts.max_buffer_bytes)?;
        super::validate_extent(count, session.0.facts.max_buffer_bytes)?;
        Ok(Self {
            session: session.clone(),
            pipeline: session.0.native.compile(
                include_str!("../../ffi/native/cpp/carrier/carrier.metal"),
                "pcu_carrier",
            )?,
            parameters: u32::try_from(width).map_err(|_| MetalError::InvalidExtent)?
                | (u32::from(broadcast) << 8),
            count,
            bytes,
            input_bytes: if broadcast { width } else { bytes },
        })
    }
    /// Executes into a fresh terminal output owner.
    /// # Errors
    /// Returns extent, affinity, quarantine or runtime failure.
    pub fn execute(&self, input: &MetalBuffer) -> Result<MetalBuffer, MetalError> {
        execute_byte_profile(
            &self.session,
            &self.pipeline,
            [input; 2],
            self.parameters,
            self.bytes,
            self.count,
            [self.input_bytes; 2],
            None,
        )
    }
    /// Executes into a distinct same-session borrowed owner.
    /// # Errors
    /// Returns alias, extent, affinity, quarantine or runtime failure.
    pub fn execute_into(
        &self,
        input: &MetalBuffer,
        output: &MetalBuffer,
    ) -> Result<(), MetalError> {
        if core::ptr::eq(input, output) {
            return Err(MetalError::Unsupported);
        }
        execute_byte_profile_into(
            &self.session,
            &self.pipeline,
            [input; 2],
            output,
            self.parameters,
            self.bytes,
            self.count,
            [self.input_bytes; 2],
            None,
        )
    }
}

impl MetalSession {
    /// Compiles a direct exact-byte scalar transport control without neutral/source admission.
    /// # Errors
    /// Rejects non-byte-addressed types, empty/overflowing/oversized spans or native failure.
    pub fn prepare_carrier_control(
        &self,
        scalar: fusion_pcu::PcuScalarType,
        count: usize,
        broadcast: bool,
    ) -> Result<Carrier, MetalError> {
        if scalar.bit_width() < 8 {
            return Err(MetalError::Unsupported);
        }
        Carrier::prepare(self, count, usize::from(scalar.bit_width()) / 8, broadcast)
    }
}
