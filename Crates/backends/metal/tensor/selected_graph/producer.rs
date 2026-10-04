//! Private frozen immutable producer metadata; payloads remain in the original graph.
use std::rc::Rc;
#[rustfmt::skip]
use fusion_pcu::{
    PcuImplementationRequirements,
    PcuScalarType,
    PcuReproducibility,
};
#[rustfmt::skip]
use fusion_pcu::dialect::tensor::{
    OpDescriptor,
    TensorElement,
    TensorOwnedSelectedProgram,
    TensorScalarValue,
    TensorUnsupportedReason,
    TensorValue,
    ValueId,
};

#[derive(Clone)]
pub(super) struct Producer {
    pub value: ValueId,
    pub slot: usize,
    pub scalar: PcuScalarType,
    pub shape: Rc<[usize]>,
    pub count: usize,
    pub broadcast: bool,
}
impl Producer {
    pub fn assess(
        program: &TensorOwnedSelectedProgram,
        value: ValueId,
        request: PcuImplementationRequirements,
    ) -> Result<Option<Self>, TensorUnsupportedReason> {
        let node = program
            .graph()
            .node(value)
            .map_err(|_| TensorUnsupportedReason::Operation)?;
        let broadcast = match node.op {
            OpDescriptor::Constant(payload) => {
                if payload.scalar_type() != node.scalar_type || payload.shape() != node.shape {
                    return Err(TensorUnsupportedReason::ElementType);
                }
                false
            }
            OpDescriptor::Uniform { value: payload } => {
                if payload.scalar_type() != node.scalar_type {
                    return Err(TensorUnsupportedReason::ElementType);
                }
                true
            }
            _ => return Ok(None),
        };
        // Preserve the separately qualified pure producer envelope. Portable
        // transport headers require their own reciprocal producer qualification.
        if request.numerical_options.reproducibility != PcuReproducibility::Unspecified {
            return Err(TensorUnsupportedReason::Operation);
        }
        if !is_producer_float(node.scalar_type) {
            return Err(TensorUnsupportedReason::ElementType);
        }
        if node.numerical_mode.is_some()
            || node.float_underflow_policy.is_some()
            || node.numerical_options != request.numerical_options
        {
            return Err(TensorUnsupportedReason::Operation);
        }
        let count = node
            .shape
            .iter()
            .try_fold(1usize, |n, &d| n.checked_mul(d))
            .filter(|&n| n > 0)
            .ok_or(TensorUnsupportedReason::Shape)?;
        let bytes = count
            .checked_mul(usize::from(node.scalar_type.bit_width()) / 8)
            .ok_or(TensorUnsupportedReason::Shape)?;
        if i32::try_from(count).is_err()
            || u32::try_from(bytes).is_err()
            || i32::try_from(bytes.div_ceil(4)).is_err()
        {
            return Err(TensorUnsupportedReason::Shape);
        }
        let slot = program
            .operation_index_of(value)
            .ok_or(TensorUnsupportedReason::Operation)?;
        Ok(Some(Self {
            value,
            slot,
            scalar: node.scalar_type,
            shape: Rc::from(node.shape),
            count,
            broadcast,
        }))
    }
    /// Cold-only encoding of retained raw bits; no arithmetic or full uniform materialization.
    pub fn bytes(
        &self,
        program: &TensorOwnedSelectedProgram,
    ) -> Result<Vec<u8>, TensorUnsupportedReason> {
        let node = program
            .graph()
            .node(self.value)
            .map_err(|_| TensorUnsupportedReason::Operation)?;
        if node.scalar_type != self.scalar || node.shape != self.shape.as_ref() {
            return Err(TensorUnsupportedReason::Shape);
        }
        macro_rules! encode {
            ($ty:ty) => {
                match node.op {
                    OpDescriptor::Constant(value) => constant::<$ty>(value),
                    OpDescriptor::Uniform { value } => uniform::<$ty>(*value),
                    _ => Err(TensorUnsupportedReason::Operation),
                }
            };
        }
        match self.scalar {
            PcuScalarType::F16 => encode!(fusion_pcu::PcuF16Bits),
            PcuScalarType::BF16 => encode!(fusion_pcu::PcuBf16Bits),
            PcuScalarType::F8E4M3FN => encode!(fusion_pcu::PcuF8E4M3FnBits),
            PcuScalarType::F8E5M2 => encode!(fusion_pcu::PcuF8E5M2Bits),
            PcuScalarType::F32 => encode!(f32),
            PcuScalarType::F64 => encode!(f64),
            PcuScalarType::F128 => encode!(fusion_pcu::PcuF128Bits),
            PcuScalarType::F256 => encode!(fusion_pcu::PcuF256Bits),
            _ => Err(TensorUnsupportedReason::ElementType),
        }
    }
}
fn constant<T: TensorElement>(value: &TensorValue) -> Result<Vec<u8>, TensorUnsupportedReason> {
    let typed = value
        .as_typed::<T>()
        .map_err(|_| TensorUnsupportedReason::ElementType)?;
    let mut bytes = Vec::with_capacity(typed.len() * usize::from(T::TYPE.bit_width()) / 8);
    for &element in typed.data() {
        bytes.extend_from_slice(element.encode_le().as_ref());
    }
    Ok(bytes)
}
fn uniform<T: TensorElement>(value: TensorScalarValue) -> Result<Vec<u8>, TensorUnsupportedReason> {
    Ok(value
        .as_typed::<T>()
        .map_err(|_| TensorUnsupportedReason::ElementType)?
        .encode_le()
        .as_ref()
        .to_vec())
}
const fn is_producer_float(scalar: PcuScalarType) -> bool {
    matches!(
        scalar,
        PcuScalarType::F16
            | PcuScalarType::BF16
            | PcuScalarType::F8E4M3FN
            | PcuScalarType::F8E5M2
            | PcuScalarType::F32
            | PcuScalarType::F64
            | PcuScalarType::F128
            | PcuScalarType::F256
    )
}
