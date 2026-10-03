//! Safe heterogeneous tensor values for graph constants and reference results.
//!
//! Each variant owns an ordinary typed vector through [`Tensor`]. A caller can inspect a value
//! through [`TensorValue::as_typed`] without interpreting bytes or changing scalar identity.

#[rustfmt::skip]
use alloc::vec::Vec;
#[rustfmt::skip]
use crate::{
    PcuBf16Bits,
    PcuF8E4M3FnBits,
    PcuF8E5M2Bits,
    PcuU256,
    PcuI256,
    PcuU512,
    PcuI512,
    PcuF128Bits,
    PcuF256Bits,
    PcuF16Bits,
    PcuScalar,
    PcuScalarType,
};

use super::Tensor;

mod sealed {
    pub trait Sealed {}

    impl Sealed for f32 {}
    impl Sealed for f64 {}
    impl Sealed for u8 {}
    impl Sealed for u16 {}
    impl Sealed for u32 {}
    impl Sealed for u64 {}
    impl Sealed for i8 {}
    impl Sealed for i16 {}
    impl Sealed for i32 {}
    impl Sealed for i64 {}
    impl Sealed for super::PcuF16Bits {}
    impl Sealed for super::PcuBf16Bits {}
    impl Sealed for super::PcuF8E4M3FnBits {}
    impl Sealed for super::PcuF8E5M2Bits {}
    impl Sealed for u128 {}
    impl Sealed for i128 {}
    impl Sealed for super::PcuU256 {}
    impl Sealed for super::PcuI256 {}
    impl Sealed for super::PcuU512 {}
    impl Sealed for super::PcuI512 {}
    impl Sealed for super::PcuF128Bits {}
    impl Sealed for super::PcuF256Bits {}
}

/// Closed scalar set that can be stored in a heterogeneous tensor value.
///
/// This trait is sealed and deliberately separate from [`PcuScalar`], whose contract does not
/// depend on the optional tensor dialect. It describes typed host storage only; it does not
/// promise arithmetic or backend support for a scalar.
#[allow(private_bounds)] // Sealing keeps each scalar identity paired with exactly one enum variant.
pub trait TensorElement: PcuScalar + sealed::Sealed {
    #[doc(hidden)]
    fn into_value(tensor: Tensor<Self>) -> TensorValue;

    #[doc(hidden)]
    fn as_value(value: &TensorValue) -> Option<&Tensor<Self>>;

    #[doc(hidden)]
    fn into_scalar(value: Self) -> TensorScalarValue;

    #[doc(hidden)]
    fn as_scalar(value: TensorScalarValue) -> Option<Self>;

    #[doc(hidden)]
    fn splat(shape: Vec<usize>, value: Self) -> Result<TensorValue, super::TensorError>;
}

/// Owned heterogeneous storage for the admitted host-reference scalar set.
///
/// `Bool`, `I4`, and `U4` have scalar identities but no `PcuScalar` host type in this crate, so
/// they intentionally have no variant here. Every sealed scalar representation otherwise has
/// exact host storage, including wide integer and floating carriers and named FP8 formats.
/// These variants preserve their storage bits and
/// do not imply arithmetic conversion. This is host/reference storage, not a device owner or a
/// promise that a backend supports every scalar or operation.
#[derive(Clone, Debug, PartialEq)]
pub enum TensorValue {
    F16(Tensor<PcuF16Bits>),
    Bf16(Tensor<PcuBf16Bits>),
    F32(Tensor<f32>),
    F64(Tensor<f64>),
    U8(Tensor<u8>),
    U16(Tensor<u16>),
    U32(Tensor<u32>),
    U64(Tensor<u64>),
    I8(Tensor<i8>),
    I16(Tensor<i16>),
    I32(Tensor<i32>),
    I64(Tensor<i64>),
    F8E4M3Fn(Tensor<PcuF8E4M3FnBits>),
    F8E5M2(Tensor<PcuF8E5M2Bits>),
    U128(Tensor<u128>),
    I128(Tensor<i128>),
    U256(Tensor<PcuU256>),
    I256(Tensor<PcuI256>),
    U512(Tensor<PcuU512>),
    I512(Tensor<PcuI512>),
    F128(Tensor<PcuF128Bits>),
    F256(Tensor<PcuF256Bits>),
}

/// One typed scalar value suitable for a heterogeneous graph uniform node.
///
/// This stores the scalar directly instead of allocating a one-element tensor. F16 and BF16
/// remain opaque storage bits; this enum does not convert them into floating-point values.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TensorScalarValue {
    F16(PcuF16Bits),
    Bf16(PcuBf16Bits),
    F32(f32),
    F64(f64),
    U8(u8),
    U16(u16),
    U32(u32),
    U64(u64),
    I8(i8),
    I16(i16),
    I32(i32),
    I64(i64),
    F8E4M3Fn(PcuF8E4M3FnBits),
    F8E5M2(PcuF8E5M2Bits),
    U128(u128),
    I128(i128),
    U256(PcuU256),
    I256(PcuI256),
    U512(PcuU512),
    I512(PcuI512),
    F128(PcuF128Bits),
    F256(PcuF256Bits),
}

impl TensorScalarValue {
    /// Scalar identity carried by this value.
    #[must_use]
    pub const fn scalar_type(self) -> PcuScalarType {
        match self {
            Self::F16(_) => PcuScalarType::F16,
            Self::F8E4M3Fn(_) => PcuScalarType::F8E4M3FN,
            Self::F8E5M2(_) => PcuScalarType::F8E5M2,
            Self::U128(_) => PcuScalarType::U128,
            Self::I128(_) => PcuScalarType::I128,
            Self::U256(_) => PcuScalarType::U256,
            Self::I256(_) => PcuScalarType::I256,
            Self::U512(_) => PcuScalarType::U512,
            Self::I512(_) => PcuScalarType::I512,
            Self::F128(_) => PcuScalarType::F128,
            Self::F256(_) => PcuScalarType::F256,
            Self::Bf16(_) => PcuScalarType::BF16,
            Self::F32(_) => PcuScalarType::F32,
            Self::F64(_) => PcuScalarType::F64,
            Self::U8(_) => PcuScalarType::U8,
            Self::U16(_) => PcuScalarType::U16,
            Self::U32(_) => PcuScalarType::U32,
            Self::U64(_) => PcuScalarType::U64,
            Self::I8(_) => PcuScalarType::I8,
            Self::I16(_) => PcuScalarType::I16,
            Self::I32(_) => PcuScalarType::I32,
            Self::I64(_) => PcuScalarType::I64,
        }
    }

    /// Returns the scalar with its exact Rust host type.
    ///
    /// # Errors
    ///
    /// Returns the requested and stored scalar identities when they differ.
    pub fn as_typed<T: TensorElement>(self) -> Result<T, TensorValueTypeMismatch> {
        T::as_scalar(self).ok_or_else(|| TensorValueTypeMismatch {
            expected: T::TYPE,
            actual: self.scalar_type(),
        })
    }

    /// Materializes a dense host tensor filled with this scalar without converting its value.
    ///
    /// The tensor records the uniform scalar when it is nonempty, allowing graph/reference
    /// logic to retain compact-uniform knowledge.
    ///
    /// # Errors
    ///
    /// Returns [`super::TensorError::ShapeOverflow`] when the element count overflows.
    pub fn splat(self, shape: impl Into<Vec<usize>>) -> Result<TensorValue, super::TensorError> {
        let shape = shape.into();
        match self {
            Self::F16(value) => PcuF16Bits::splat(shape, value),
            Self::F8E4M3Fn(value) => PcuF8E4M3FnBits::splat(shape, value),
            Self::F8E5M2(value) => PcuF8E5M2Bits::splat(shape, value),
            Self::U128(value) => u128::splat(shape, value),
            Self::I128(value) => i128::splat(shape, value),
            Self::U256(value) => PcuU256::splat(shape, value),
            Self::I256(value) => PcuI256::splat(shape, value),
            Self::U512(value) => PcuU512::splat(shape, value),
            Self::I512(value) => PcuI512::splat(shape, value),
            Self::F128(value) => PcuF128Bits::splat(shape, value),
            Self::F256(value) => PcuF256Bits::splat(shape, value),
            Self::Bf16(value) => PcuBf16Bits::splat(shape, value),
            Self::F32(value) => f32::splat(shape, value),
            Self::F64(value) => f64::splat(shape, value),
            Self::U8(value) => u8::splat(shape, value),
            Self::U16(value) => u16::splat(shape, value),
            Self::U32(value) => u32::splat(shape, value),
            Self::U64(value) => u64::splat(shape, value),
            Self::I8(value) => i8::splat(shape, value),
            Self::I16(value) => i16::splat(shape, value),
            Self::I32(value) => i32::splat(shape, value),
            Self::I64(value) => i64::splat(shape, value),
        }
    }
}

/// Requested tensor scalar did not match the value's stored representation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TensorValueTypeMismatch {
    /// Scalar type requested by the typed accessor.
    pub expected: PcuScalarType,
    /// Scalar type stored by the value.
    pub actual: PcuScalarType,
}

impl core::fmt::Display for TensorValueTypeMismatch {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            formatter,
            "tensor scalar type mismatch: requested {:?}, stored {:?}",
            self.expected, self.actual
        )
    }
}

impl core::error::Error for TensorValueTypeMismatch {}

impl TensorValue {
    /// Moves typed tensor storage into the corresponding closed enum variant.
    #[must_use]
    pub fn from_tensor<T: TensorElement>(tensor: Tensor<T>) -> Self {
        T::into_value(tensor)
    }

    /// Scalar identity recorded by this value.
    #[must_use]
    pub const fn scalar_type(&self) -> PcuScalarType {
        match self {
            Self::F16(_) => PcuScalarType::F16,
            Self::F8E4M3Fn(_) => PcuScalarType::F8E4M3FN,
            Self::F8E5M2(_) => PcuScalarType::F8E5M2,
            Self::U128(_) => PcuScalarType::U128,
            Self::I128(_) => PcuScalarType::I128,
            Self::U256(_) => PcuScalarType::U256,
            Self::I256(_) => PcuScalarType::I256,
            Self::U512(_) => PcuScalarType::U512,
            Self::I512(_) => PcuScalarType::I512,
            Self::F128(_) => PcuScalarType::F128,
            Self::F256(_) => PcuScalarType::F256,
            Self::Bf16(_) => PcuScalarType::BF16,
            Self::F32(_) => PcuScalarType::F32,
            Self::F64(_) => PcuScalarType::F64,
            Self::U8(_) => PcuScalarType::U8,
            Self::U16(_) => PcuScalarType::U16,
            Self::U32(_) => PcuScalarType::U32,
            Self::U64(_) => PcuScalarType::U64,
            Self::I8(_) => PcuScalarType::I8,
            Self::I16(_) => PcuScalarType::I16,
            Self::I32(_) => PcuScalarType::I32,
            Self::I64(_) => PcuScalarType::I64,
        }
    }

    /// Dense shape of the stored tensor.
    #[must_use]
    pub fn shape(&self) -> &[usize] {
        match self {
            Self::F16(tensor) => tensor.shape(),
            Self::F8E4M3Fn(tensor) => tensor.shape(),
            Self::F8E5M2(tensor) => tensor.shape(),
            Self::U128(tensor) => tensor.shape(),
            Self::I128(tensor) => tensor.shape(),
            Self::U256(tensor) => tensor.shape(),
            Self::I256(tensor) => tensor.shape(),
            Self::U512(tensor) => tensor.shape(),
            Self::I512(tensor) => tensor.shape(),
            Self::F128(tensor) => tensor.shape(),
            Self::F256(tensor) => tensor.shape(),
            Self::Bf16(tensor) => tensor.shape(),
            Self::F32(tensor) => tensor.shape(),
            Self::F64(tensor) => tensor.shape(),
            Self::U8(tensor) => tensor.shape(),
            Self::U16(tensor) => tensor.shape(),
            Self::U32(tensor) => tensor.shape(),
            Self::U64(tensor) => tensor.shape(),
            Self::I8(tensor) => tensor.shape(),
            Self::I16(tensor) => tensor.shape(),
            Self::I32(tensor) => tensor.shape(),
            Self::I64(tensor) => tensor.shape(),
        }
    }

    /// Number of stored elements.
    #[must_use]
    pub const fn len(&self) -> usize {
        match self {
            Self::F16(tensor) => tensor.len(),
            Self::F8E4M3Fn(tensor) => tensor.len(),
            Self::F8E5M2(tensor) => tensor.len(),
            Self::U128(tensor) => tensor.len(),
            Self::I128(tensor) => tensor.len(),
            Self::U256(tensor) => tensor.len(),
            Self::I256(tensor) => tensor.len(),
            Self::U512(tensor) => tensor.len(),
            Self::I512(tensor) => tensor.len(),
            Self::F128(tensor) => tensor.len(),
            Self::F256(tensor) => tensor.len(),
            Self::Bf16(tensor) => tensor.len(),
            Self::F32(tensor) => tensor.len(),
            Self::F64(tensor) => tensor.len(),
            Self::U8(tensor) => tensor.len(),
            Self::U16(tensor) => tensor.len(),
            Self::U32(tensor) => tensor.len(),
            Self::U64(tensor) => tensor.len(),
            Self::I8(tensor) => tensor.len(),
            Self::I16(tensor) => tensor.len(),
            Self::I32(tensor) => tensor.len(),
            Self::I64(tensor) => tensor.len(),
        }
    }

    /// Whether this tensor has no elements.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Borrows the value as its exact typed tensor representation.
    ///
    /// # Errors
    ///
    /// Returns the requested and stored scalar identities when they differ.
    pub fn as_typed<T: TensorElement>(&self) -> Result<&Tensor<T>, TensorValueTypeMismatch> {
        T::as_value(self).ok_or_else(|| TensorValueTypeMismatch {
            expected: T::TYPE,
            actual: self.scalar_type(),
        })
    }
}

macro_rules! tensor_element {
    ($ty:ty, $variant:ident) => {
        impl TensorElement for $ty {
            fn into_value(tensor: Tensor<Self>) -> TensorValue {
                TensorValue::$variant(tensor)
            }

            fn as_value(value: &TensorValue) -> Option<&Tensor<Self>> {
                match value {
                    TensorValue::$variant(tensor) => Some(tensor),
                    _ => None,
                }
            }

            fn into_scalar(value: Self) -> TensorScalarValue {
                TensorScalarValue::$variant(value)
            }

            fn as_scalar(value: TensorScalarValue) -> Option<Self> {
                match value {
                    TensorScalarValue::$variant(value) => Some(value),
                    _ => None,
                }
            }

            fn splat(shape: Vec<usize>, value: Self) -> Result<TensorValue, super::TensorError> {
                Tensor::<Self>::splat(shape, value).map(TensorValue::$variant)
            }
        }

        impl From<Tensor<$ty>> for TensorValue {
            fn from(tensor: Tensor<$ty>) -> Self {
                Self::$variant(tensor)
            }
        }

        impl From<$ty> for TensorScalarValue {
            fn from(value: $ty) -> Self {
                TensorScalarValue::$variant(value)
            }
        }
    };
}

tensor_element!(PcuF16Bits, F16);
tensor_element!(PcuBf16Bits, Bf16);
tensor_element!(PcuF8E4M3FnBits, F8E4M3Fn);
tensor_element!(PcuF8E5M2Bits, F8E5M2);
tensor_element!(u128, U128);
tensor_element!(i128, I128);
tensor_element!(PcuU256, U256);
tensor_element!(PcuI256, I256);
tensor_element!(PcuU512, U512);
tensor_element!(PcuI512, I512);
tensor_element!(PcuF128Bits, F128);
tensor_element!(PcuF256Bits, F256);
tensor_element!(f32, F32);
tensor_element!(f64, F64);
tensor_element!(u8, U8);
tensor_element!(u16, U16);
tensor_element!(u32, U32);
tensor_element!(u64, U64);
tensor_element!(i8, I8);
tensor_element!(i16, I16);
tensor_element!(i32, I32);
tensor_element!(i64, I64);

#[cfg(test)]
mod tests {
    use super::{TensorScalarValue, TensorValue, TensorValueTypeMismatch};
    #[rustfmt::skip]
    use alloc::{
        vec,
        vec::Vec,
    };
    #[rustfmt::skip]
    use crate::{
        PcuBf16Bits,
        PcuF16Bits,
        PcuScalarType,
    };

    use super::super::Tensor;

    #[test]
    fn heterogeneous_values_retain_shape_type_and_typed_borrows() {
        let single = TensorValue::from_tensor(Tensor::new([2], vec![1.25_f32, -3.5]).unwrap());
        let double = TensorValue::from_tensor(
            Tensor::new(
                [1, 2],
                vec![f64::MAX, f64::from_bits(0x8000_0000_0000_0000)],
            )
            .unwrap(),
        );

        assert_eq!(single.scalar_type(), PcuScalarType::F32);
        assert_eq!(single.shape(), [2]);
        assert_eq!(single.len(), 2);
        assert_eq!(single.as_typed::<f32>().unwrap().data(), [1.25, -3.5]);

        assert_eq!(double.scalar_type(), PcuScalarType::F64);
        assert_eq!(double.shape(), [1, 2]);
        let values = double.as_typed::<f64>().unwrap().data();
        assert_eq!(values[0].to_bits(), f64::MAX.to_bits());
        assert_eq!(values[1].to_bits(), 0x8000_0000_0000_0000);
        assert_eq!(
            double.as_typed::<f32>().unwrap_err(),
            TensorValueTypeMismatch {
                expected: PcuScalarType::F32,
                actual: PcuScalarType::F64,
            }
        );
    }

    #[test]
    fn half_storage_preserves_all_bits_without_numeric_conversion() {
        let half = [PcuF16Bits::from_bits(0x8000), PcuF16Bits::from_bits(0x7c01)];
        let bfloat = [
            PcuBf16Bits::from_bits(0xff80),
            PcuBf16Bits::from_bits(0x7f81),
        ];
        let half_value = TensorValue::from_tensor(Tensor::new([2], half.to_vec()).unwrap());
        let bfloat_value = TensorValue::from_tensor(Tensor::new([2], bfloat.to_vec()).unwrap());

        assert_eq!(
            half_value
                .as_typed::<PcuF16Bits>()
                .unwrap()
                .data()
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            [0x8000, 0x7c01]
        );
        assert_eq!(
            bfloat_value
                .as_typed::<PcuBf16Bits>()
                .unwrap()
                .data()
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            [0xff80, 0x7f81]
        );
    }

    #[test]
    fn integer_extrema_and_conversions_remain_exact() {
        let unsigned =
            TensorValue::from_tensor(Tensor::new([2], vec![u64::MIN, u64::MAX]).unwrap());
        let signed = TensorValue::from_tensor(Tensor::new([2], vec![i64::MIN, i64::MAX]).unwrap());

        assert_eq!(
            unsigned.as_typed::<u64>().unwrap().data(),
            [u64::MIN, u64::MAX]
        );
        assert_eq!(
            signed.as_typed::<i64>().unwrap().data(),
            [i64::MIN, i64::MAX]
        );
        assert_eq!(
            unsigned.as_typed::<i64>().unwrap_err().actual,
            PcuScalarType::U64
        );
        assert_eq!(
            signed.as_typed::<u64>().unwrap_err().actual,
            PcuScalarType::I64
        );
    }

    #[test]
    fn scalar_values_preserve_float_bits_and_check_requested_types() {
        let nan = f64::from_bits(0xfff8_0000_0000_1234);
        let scalar = TensorScalarValue::from(nan);
        assert_eq!(scalar.scalar_type(), PcuScalarType::F64);
        assert_eq!(scalar.as_typed::<f64>().unwrap().to_bits(), nan.to_bits());
        assert_eq!(
            scalar.as_typed::<f32>().unwrap_err(),
            TensorValueTypeMismatch {
                expected: PcuScalarType::F32,
                actual: PcuScalarType::F64,
            }
        );

        let negative_zero = TensorScalarValue::from(-0.0_f32);
        assert_eq!(
            negative_zero.as_typed::<f32>().unwrap().to_bits(),
            0x8000_0000
        );
    }

    #[test]
    fn half_scalars_splat_without_conversion_and_empty_shapes_stay_empty() {
        let half = TensorScalarValue::from(PcuF16Bits::from_bits(0x7c01));
        let value = half.splat([3]).unwrap();
        let tensor = value.as_typed::<PcuF16Bits>().unwrap();
        assert_eq!(tensor.known_uniform_value().unwrap().to_bits(), 0x7c01);
        assert_eq!(
            tensor
                .data()
                .iter()
                .map(|item| item.to_bits())
                .collect::<Vec<_>>(),
            [0x7c01; 3]
        );

        let empty = half.splat([0, 4]).unwrap();
        let empty_tensor = empty.as_typed::<PcuF16Bits>().unwrap();
        assert_eq!(empty_tensor.shape(), [0, 4]);
        assert!(empty_tensor.is_empty());
        assert_eq!(empty_tensor.known_uniform_value(), None);

        let bf16 = TensorScalarValue::from(PcuBf16Bits::from_bits(0x7f81));
        let bfloat_tensor = bf16.splat([2]).unwrap();
        assert_eq!(
            bfloat_tensor
                .as_typed::<PcuBf16Bits>()
                .unwrap()
                .data()
                .iter()
                .map(|item| item.to_bits())
                .collect::<Vec<_>>(),
            [0x7f81; 2]
        );
    }
}
