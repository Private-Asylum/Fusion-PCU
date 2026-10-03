//! Direct native owned-control boundary; no fabricated neutral graph identifiers.
#[rustfmt::skip]
use super::{
    MetalSession,
    MetalPreparedFloatUnary,
    PcuScalarType,
    Rc,
    PcuMemoryPoolId,
    MetalError,
    MetalTensorOwner,
};
#[rustfmt::skip]
use fusion_pcu::{PcuFloatUnderflowPolicy,PcuDispatchFloatUnaryOp,PcuRangePolicy};
/// Cold direct unary/pool/shape control returning the same fresh owned resource as graph execution.
/// This benchmark-only API performs no source capture or neutral program assessment.
pub struct MetalNativeTensorReluControl {
    session: MetalSession,
    unary: MetalPreparedFloatUnary,
    scalar: PcuScalarType,
    shape: Rc<[usize]>,
    count: usize,
    bytes: usize,
    pool: PcuMemoryPoolId,
}
impl MetalSession {
    /// Retains an independently prepared native checked `ReLU` and fresh-owner boundary.
    /// # Errors
    /// Rejects unproved types, empty/overflowing/oversized extents, byte order or native failure.
    pub fn prepare_native_tensor_relu_control(
        &self,
        scalar: PcuScalarType,
        shape: &[usize],
        underflow: PcuFloatUnderflowPolicy,
        pool: PcuMemoryPoolId,
    ) -> Result<MetalNativeTensorReluControl, MetalError> {
        if cfg!(target_endian = "big") {
            return Err(MetalError::Unsupported);
        }
        let count = shape
            .iter()
            .try_fold(1_usize, |n, &dimension| n.checked_mul(dimension))
            .filter(|&n| n > 0)
            .ok_or(MetalError::InvalidExtent)?;
        let bytes = count
            .checked_mul(usize::from(scalar.bit_width()) / 8)
            .ok_or(MetalError::InvalidExtent)?;
        let status = count.checked_mul(4).ok_or(MetalError::InvalidExtent)?;
        for extent in [bytes, status] {
            if u32::try_from(extent).is_err()
                || u64::try_from(extent).map_err(|_| MetalError::InvalidExtent)?
                    > self.facts().max_buffer_bytes
            {
                return Err(MetalError::InvalidExtent);
            }
        }
        let unary = self.prepare_checked_float_unary_with_range(
            scalar,
            PcuDispatchFloatUnaryOp::Relu,
            underflow,
            PcuRangePolicy::Reject,
        )?;
        Ok(MetalNativeTensorReluControl {
            session: self.clone(),
            unary,
            scalar,
            shape: Rc::from(shape),
            count,
            bytes,
            pool,
        })
    }
}
impl MetalNativeTensorReluControl {
    /// Stages exact tagged host bytes, evaluates privately and returns only a terminal owner.
    /// # Errors
    /// Rejects short input or runtime/arithmetic failure; no result owner escapes on failure.
    pub fn execute_host(&self, bytes: &[u8]) -> Result<MetalTensorOwner, MetalError> {
        if bytes.len() < self.bytes {
            return Err(MetalError::InvalidExtent);
        }
        let input = self.session.upload_bytes(&bytes[..self.bytes])?;
        let output = self.unary.execute_prefix(&input, self.bytes)?;
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

#[path = "carrier/carrier.rs"]
mod carrier;
pub use carrier::MetalNativeTensorCarrierControl;
