//! Direct checked binary owner control without source capture or fabricated neutral identifiers.
#[rustfmt::skip]
use super::{
    MetalSession,
    MetalError,
    MetalTensorOwner,
    MetalIntegerOp,
    Kernel,
    Rc,
    PcuDispatchFloatBinaryOp,
    PcuMemoryPoolId,
};
#[rustfmt::skip]
use fusion_pcu::{
    PcuScalarType,
    PcuRangePolicy,
    PcuFloatUnderflowPolicy,
};
/// Retained exact native arithmetic and same fresh initialized owned-resource boundary.
pub struct MetalNativeTensorBinaryControl {
    session: MetalSession,
    kernel: Kernel,
    scalar: PcuScalarType,
    shape: Rc<[usize]>,
    count: usize,
    bytes: usize,
    pool: PcuMemoryPoolId,
}
impl MetalSession {
    /// Prepares the direct integer/floating implementation without graph construction.
    /// # Errors
    /// Rejects unproved type/op, empty/oversized extents, byte order or native compiler error.
    pub fn prepare_native_tensor_binary_control(
        &self,
        scalar: PcuScalarType,
        shape: &[usize],
        operation: PcuDispatchFloatBinaryOp,
        underflow: PcuFloatUnderflowPolicy,
        pool: PcuMemoryPoolId,
    ) -> Result<MetalNativeTensorBinaryControl, MetalError> {
        if cfg!(target_endian = "big") {
            return Err(MetalError::Unsupported);
        }
        let count = shape
            .iter()
            .try_fold(1_usize, |n, &d| n.checked_mul(d))
            .filter(|&n| n > 0)
            .ok_or(MetalError::InvalidExtent)?;
        let bytes = count
            .checked_mul(usize::from(scalar.bit_width()) / 8)
            .ok_or(MetalError::InvalidExtent)?;
        for extent in [
            bytes,
            count.checked_mul(4).ok_or(MetalError::InvalidExtent)?,
        ] {
            if u32::try_from(extent).is_err()
                || u64::try_from(extent).map_err(|_| MetalError::InvalidExtent)?
                    > self.facts().max_buffer_bytes
            {
                return Err(MetalError::InvalidExtent);
            }
        }
        let kernel = if scalar.binary_float_format().is_some() {
            Kernel::Float(self.prepare_checked_float_binary_with_range(
                scalar,
                operation,
                underflow,
                PcuRangePolicy::Reject,
            )?)
        } else {
            let op = match operation {
                PcuDispatchFloatBinaryOp::Add => MetalIntegerOp::Add,
                PcuDispatchFloatBinaryOp::Sub => MetalIntegerOp::Subtract,
                PcuDispatchFloatBinaryOp::Mul => MetalIntegerOp::Multiply,
                PcuDispatchFloatBinaryOp::Div => return Err(MetalError::Unsupported),
            };
            Kernel::Integer(self.prepare_checked_integer_map(
                scalar,
                op,
                PcuRangePolicy::Reject,
                count,
                [false; 2],
            )?)
        };
        Ok(MetalNativeTensorBinaryControl {
            session: self.clone(),
            kernel,
            scalar,
            shape: Rc::from(shape),
            count,
            bytes,
            pool,
        })
    }
}
impl MetalNativeTensorBinaryControl {
    /// Stages both exact host prefixes and publishes only the terminal successful private owner.
    /// # Errors
    /// Rejects either short input before staging, checked fatal fault or quarantined completion.
    pub fn execute_host(&self, inputs: [&[u8]; 2]) -> Result<MetalTensorOwner, MetalError> {
        self.session.ensure_quiescent()?;
        if inputs.iter().any(|bytes| bytes.len() < self.bytes) {
            return Err(MetalError::InvalidExtent);
        }
        let left = self.session.upload_bytes(&inputs[0][..self.bytes])?;
        let right = self.session.upload_bytes(&inputs[1][..self.bytes])?;
        let output = self.kernel.execute([&left, &right], self.bytes)?;
        Ok(MetalTensorOwner {
            session: self.session.clone(),
            scalar: self.scalar,
            shape: Rc::clone(&self.shape),
            count: self.count,
            resource: self
                .session
                .initialized_tensor_resource(self.pool, output)?,
        })
    }
}
