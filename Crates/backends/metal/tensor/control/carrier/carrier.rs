//! Direct exact22 byte copy with the same fresh initialized owner as selected Input execution.
#[rustfmt::skip]
use super::super::{MetalSession,MetalPreparedCarrierControl,MetalTensorOwner,MetalError,PcuScalarType,PcuMemoryPoolId,Rc};
pub struct MetalNativeTensorCarrierControl {
    session: MetalSession,
    carrier: MetalPreparedCarrierControl,
    scalar: PcuScalarType,
    shape: Rc<[usize]>,
    count: usize,
    bytes: usize,
    pool: PcuMemoryPoolId,
}
impl MetalSession {
    /// Freezes a direct native byte-copy and fresh owned resource boundary without graph capture.
    /// # Errors
    /// Rejects packed types, byte order, invalid extent, quarantine or native compiler failure.
    pub fn prepare_native_tensor_carrier_control(
        &self,
        scalar: PcuScalarType,
        shape: &[usize],
        pool: PcuMemoryPoolId,
    ) -> Result<MetalNativeTensorCarrierControl, MetalError> {
        if cfg!(target_endian = "big") {
            return Err(MetalError::Unsupported);
        }
        let count = shape
            .iter()
            .try_fold(1_usize, |count, &dimension| count.checked_mul(dimension))
            .filter(|&n| n > 0)
            .ok_or(MetalError::InvalidExtent)?;
        let carrier = self.prepare_carrier_control(scalar, count, false)?;
        let bytes = count
            .checked_mul(usize::from(scalar.bit_width()) / 8)
            .ok_or(MetalError::InvalidExtent)?;
        Ok(MetalNativeTensorCarrierControl {
            session: self.clone(),
            carrier,
            scalar,
            shape: Rc::from(shape),
            count,
            bytes,
            pool,
        })
    }
}
impl MetalNativeTensorCarrierControl {
    /// Stages a fresh RAM span and terminally publishes an authentic exact-session owner.
    /// # Errors
    /// Rejects short input or native failure; no output owner escapes on failure.
    pub fn execute_host(&self, bytes: &[u8]) -> Result<MetalTensorOwner, MetalError> {
        if bytes.len() < self.bytes {
            return Err(MetalError::InvalidExtent);
        }
        let input = self.session.upload_bytes(&bytes[..self.bytes])?;
        let output = self.carrier.execute(&input)?;
        let resource = self
            .session
            .initialized_tensor_resource(self.pool, output)?;
        Ok(MetalTensorOwner {
            session: self.session.clone(),
            scalar: self.scalar,
            shape: Rc::clone(&self.shape),
            count: self.count,
            resource,
        })
    }
}
