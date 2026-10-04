//! Immutable initialized literal views; public transfer completion remains synchronous.
#[rustfmt::skip]
use super::{
    NodeDescriptor,
    RocmPhysicalLayout,
    RocmTensorExecutionError,
};
use fusion_pcu::PcuHostArgument;
#[cfg(target_endian = "little")]
#[rustfmt::skip]
use fusion_pcu::{
    PcuBindingRef,
    PcuScalarType,
    dialect::tensor::TensorElement,
};
#[cfg(target_endian = "little")]
use super::OpDescriptor;

pub(super) enum Payload<'a> {
    #[cfg_attr(not(target_endian = "little"), allow(dead_code))]
    // Byte-order conversion requires owned encoding there.
    Borrowed(PcuHostArgument<'a>),
    Encoded(Vec<u8>),
}

impl Payload<'_> {
    pub(super) fn bytes(&self) -> &[u8] {
        match self {
            Self::Borrowed(argument) => argument.bytes(),
            Self::Encoded(bytes) => bytes,
        }
    }
}

pub(super) fn payload(
    node: NodeDescriptor<'_>,
    layout: RocmPhysicalLayout,
) -> Result<Payload<'_>, RocmTensorExecutionError> {
    #[cfg(target_endian = "little")]
    if matches!(node.op, OpDescriptor::Constant(_)) {
        // PcuHostArgument exposes only sealed padding-free initialized representations.
        // On this byte order, the native view equals the canonical little-endian encoding.
        // Borrowing never changes ownership, mutates the graph or aliases escaped outputs.
        macro_rules! scalar {
            ($($kind:ident => $ty:ty),+ $(,)?) => {
                match node.scalar_type {
                    $(PcuScalarType::$kind => borrowed::<$ty>(node, layout),)+
                    _ => Err(RocmTensorExecutionError::InvalidPlan(node.value)),
                }
            };
        }
        return scalar!(
            U8 => u8, I8 => i8, U16 => u16, I16 => i16,
            U32 => u32, I32 => i32, U64 => u64, I64 => i64,
            U128 => u128, I128 => i128,
            U256 => fusion_pcu::PcuU256, I256 => fusion_pcu::PcuI256,
            U512 => fusion_pcu::PcuU512, I512 => fusion_pcu::PcuI512,
            F64 => f64,
            F16 => fusion_pcu::PcuF16Bits, BF16 => fusion_pcu::PcuBf16Bits,
            F8E4M3FN => fusion_pcu::PcuF8E4M3FnBits,
            F8E5M2 => fusion_pcu::PcuF8E5M2Bits,
            F128 => fusion_pcu::PcuF128Bits, F256 => fusion_pcu::PcuF256Bits,
        );
    }
    let encoded = if node.scalar_type == fusion_pcu::PcuScalarType::F64 {
        super::encoded_f64(node, layout)?
    } else if super::super::is_raw_float_type(node.scalar_type) {
        super::raw_float::encoded(node, layout)?
    } else if super::super::is_low_float_type(node.scalar_type) {
        super::encoded_low_float(node, layout)?
    } else {
        super::encoded_integer(node, layout)?
    };
    Ok(Payload::Encoded(encoded))
}

#[cfg(target_endian = "little")]
fn borrowed<T: TensorElement>(
    node: NodeDescriptor<'_>,
    layout: RocmPhysicalLayout,
) -> Result<Payload<'_>, RocmTensorExecutionError> {
    if node.scalar_type != T::TYPE
        || layout.representation != super::super::RocmPhysicalRepresentation::Dense
        || T::HOST_SIZE != T::ENCODED_SIZE
    {
        return Err(RocmTensorExecutionError::InvalidPlan(node.value));
    }
    let OpDescriptor::Constant(value) = node.op else {
        return Err(RocmTensorExecutionError::InvalidPlan(node.value));
    };
    let tensor = value
        .as_typed::<T>()
        .map_err(|_| RocmTensorExecutionError::InvalidPlan(node.value))?;
    let count = node
        .shape
        .iter()
        .try_fold(1usize, |n, &d| n.checked_mul(d))
        .ok_or(RocmTensorExecutionError::SizeOverflow)?;
    let bytes = count
        .checked_mul(T::ENCODED_SIZE)
        .ok_or(RocmTensorExecutionError::SizeOverflow)?;
    if tensor.shape() != node.shape
        || tensor.data().len() != count
        || u64::try_from(bytes).ok() != Some(layout.physical_bytes)
    {
        return Err(RocmTensorExecutionError::InvalidPlan(node.value));
    }
    Ok(Payload::Borrowed(PcuHostArgument::read(
        PcuBindingRef::new(0, 0),
        tensor.data(),
    )))
}

#[cfg(all(test, target_endian = "little"))]
#[path = "tests/tests.rs"]
mod tests;
