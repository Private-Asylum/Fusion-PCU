//! Portable scalar identities and lossless canonical encodings.
//!
//! This contract describes supported Rust values, not a promise that a particular backend can
//! execute every operation on them or map host memory directly into a device address space.

#[rustfmt::skip]
use crate::{
    PcuBinding,
    PcuBindingAccess,
    PcuBindingStorageClass,
    PcuScalarType,
    PcuValueType,
};

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
}

/// Lossless binary16 storage bits, without an implied arithmetic or conversion policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct PcuF16Bits(u16);

impl PcuF16Bits {
    #[must_use]
    pub const fn from_bits(bits: u16) -> Self {
        Self(bits)
    }

    #[must_use]
    pub const fn to_bits(self) -> u16 {
        self.0
    }

    /// Widens IEEE 754 binary16 to binary32 without performing arithmetic.
    ///
    /// Finite values, signed zero, infinities, and NaN sign/payload/quiet bits are preserved
    /// exactly where binary32 has room for them. In particular, signaling NaNs remain signaling.
    #[must_use]
    pub const fn to_f32(self) -> f32 {
        f32::from_bits(f16_to_f32_bits(self.0))
    }

    /// Narrows binary32 to IEEE 754 binary16 using round-to-nearest, ties-to-even.
    ///
    /// Overflow becomes signed infinity and underflow rounds to signed zero or a subnormal.
    /// NaNs preserve sign and the most significant payload bits, and are always quieted.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)] // The rounded word is explicitly narrowed to its high 16 bits.
    pub const fn from_f32(value: f32) -> Self {
        Self(f32_to_f16_bits(value.to_bits()))
    }
}

/// Lossless bfloat16 storage bits, without an implied arithmetic or conversion policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct PcuBf16Bits(u16);

impl PcuBf16Bits {
    #[must_use]
    pub const fn from_bits(bits: u16) -> Self {
        Self(bits)
    }

    #[must_use]
    pub const fn to_bits(self) -> u16 {
        self.0
    }

    /// Widens bfloat16 to binary32 by placing its bits in the high half of the binary32 word.
    ///
    /// This exactly preserves finite values, signed zero, infinities, and NaN sign/payload/quiet
    /// bits. Signaling NaNs remain signaling.
    #[must_use]
    pub const fn to_f32(self) -> f32 {
        f32::from_bits((self.0 as u32) << 16)
    }

    /// Narrows binary32 to bfloat16 using round-to-nearest, ties-to-even.
    ///
    /// Overflow becomes signed infinity and underflow rounds to signed zero or a subnormal.
    /// NaNs preserve sign and the most significant payload bits, and are always quieted.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)] // The rounded word is explicitly narrowed to its high 16 bits.
    pub const fn from_f32(value: f32) -> Self {
        let bits = value.to_bits();
        let exponent = bits & 0x7f80_0000;
        let fraction = bits & 0x007f_ffff;
        if exponent == 0x7f80_0000 && fraction != 0 {
            // Keep the available high payload bits, force a nonzero payload, and quiet the NaN.
            let upper = (bits >> 16) as u16;
            return Self((upper & 0x8000) | (upper & 0x7fff) | 0x0040);
        }

        let upper = bits >> 16;
        let lower = bits & 0xffff;
        let tie_lsb = upper & 1;
        let rounded = upper + (lower > 0x8000 || (lower == 0x8000 && tie_lsb != 0)) as u32;
        Self(rounded as u16)
    }
}

const fn round_shift_right_ties_even(value: u32, shift: u32) -> u32 {
    if shift == 0 {
        return value;
    }
    if shift >= 32 {
        return 0;
    }
    let truncated = value >> shift;
    let mask = (1_u32 << shift) - 1;
    let discarded = value & mask;
    let halfway = 1_u32 << (shift - 1);
    truncated + (discarded > halfway || (discarded == halfway && truncated & 1 != 0)) as u32
}

// Bitfield casts below are bounded by exponent and fraction masks and implement IEEE field
// extraction. Using checked integer conversion here would obscure those bit-level invariants.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss
)]
const fn f32_to_f16_bits(bits: u32) -> u16 {
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32;
    let fraction = bits & 0x007f_ffff;

    if exponent == 0xff {
        if fraction == 0 {
            return sign | 0x7c00;
        }
        let payload = (fraction >> 13) as u16;
        return sign | 0x7c00 | payload | 0x0200;
    }
    if exponent == 0 {
        // Every binary32 subnormal is far below half the least binary16 subnormal.
        return sign;
    }

    let unbiased = exponent - 127;
    let significand = 0x0080_0000 | fraction;
    if unbiased > 15 {
        return sign | 0x7c00;
    }
    if unbiased >= -14 {
        let rounded = round_shift_right_ties_even(significand, 13);
        let half_exponent = if rounded == 0x0800 {
            (unbiased + 16) as u32
        } else {
            (unbiased + 15) as u32
        };
        let half_fraction = if rounded == 0x0800 {
            0
        } else {
            rounded & 0x03ff
        };
        if half_exponent >= 31 {
            return sign | 0x7c00;
        }
        return sign | ((half_exponent as u16) << 10) | (half_fraction as u16);
    }
    if unbiased < -25 {
        return sign;
    }

    let rounded = round_shift_right_ties_even(significand, (-unbiased - 1) as u32);
    sign | (rounded as u16)
}

// A half subnormal's normalized exponent is always in [-24, -15], so this conversion cannot
// lose a sign. The cast simply places the validated exponent into the binary32 exponent field.
#[allow(clippy::cast_sign_loss)]
const fn f16_to_f32_bits(bits: u16) -> u32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let exponent = (bits >> 10) & 0x1f;
    let fraction = (bits & 0x03ff) as u32;

    if exponent == 0 {
        if fraction == 0 {
            return sign;
        }
        let mut normalized = fraction;
        let mut unbiased = -14_i32;
        while normalized & 0x0400 == 0 {
            normalized <<= 1;
            unbiased -= 1;
        }
        normalized &= 0x03ff;
        return sign | (((unbiased + 127) as u32) << 23) | (normalized << 13);
    }
    if exponent == 0x1f {
        return sign | 0x7f80_0000 | (fraction << 13);
    }
    sign | (((exponent + 112) as u32) << 23) | (fraction << 13)
}

impl PcuScalar for PcuF16Bits {
    const TYPE: PcuScalarType = PcuScalarType::F16;
    const HOST_SIZE: usize = core::mem::size_of::<Self>();
    const HOST_ALIGNMENT: usize = core::mem::align_of::<Self>();
    const ENCODED_SIZE: usize = 2;
    type Encoded = [u8; 2];

    fn encode_le(self) -> Self::Encoded {
        self.0.to_le_bytes()
    }

    fn decode_le(bytes: Self::Encoded) -> Self {
        Self(u16::from_le_bytes(bytes))
    }
}

impl PcuScalar for PcuBf16Bits {
    const TYPE: PcuScalarType = PcuScalarType::BF16;
    const HOST_SIZE: usize = core::mem::size_of::<Self>();
    const HOST_ALIGNMENT: usize = core::mem::align_of::<Self>();
    const ENCODED_SIZE: usize = 2;
    type Encoded = [u8; 2];

    fn encode_le(self) -> Self::Encoded {
        self.0.to_le_bytes()
    }

    fn decode_le(bytes: Self::Encoded) -> Self {
        Self(u16::from_le_bytes(bytes))
    }
}

/// Scalar with a verified PCU identity and a lossless little-endian transfer encoding.
///
/// The trait is sealed until custom scalar representations have an explicit unsafe contract.
/// Backend admission still decides whether the type, operations, and physical layout are
/// supported. The host layout constants are not device ABI guarantees.
/// Every sealed implementation is padding-free and admits all bit patterns. Typed host-borrow
/// byte views rely on this invariant; types such as bool or enums cannot be added without changing
/// that byte-view contract first.
#[allow(private_bounds)] // A private supertrait prevents unchecked downstream implementations.
pub trait PcuScalar: Copy + sealed::Sealed + 'static {
    /// Semantic scalar identity in PCU IR.
    const TYPE: PcuScalarType;
    /// Size of the Rust value in host memory.
    const HOST_SIZE: usize;
    /// Alignment of the Rust value in host memory.
    const HOST_ALIGNMENT: usize;
    /// Size of the canonical little-endian encoding.
    const ENCODED_SIZE: usize;
    /// Fixed-size byte representation used for canonical transfer.
    type Encoded: AsRef<[u8]> + Copy;

    /// Encodes this value without changing its bit pattern.
    fn encode_le(self) -> Self::Encoded;

    /// Decodes an encoded value without changing its bit pattern.
    fn decode_le(bytes: Self::Encoded) -> Self;
}

impl PcuScalar for f32 {
    const TYPE: PcuScalarType = PcuScalarType::F32;
    const HOST_SIZE: usize = core::mem::size_of::<Self>();
    const HOST_ALIGNMENT: usize = core::mem::align_of::<Self>();
    const ENCODED_SIZE: usize = 4;
    type Encoded = [u8; 4];

    fn encode_le(self) -> Self::Encoded {
        self.to_bits().to_le_bytes()
    }

    fn decode_le(bytes: Self::Encoded) -> Self {
        Self::from_bits(u32::from_le_bytes(bytes))
    }
}

impl PcuScalar for f64 {
    const TYPE: PcuScalarType = PcuScalarType::F64;
    const HOST_SIZE: usize = core::mem::size_of::<Self>();
    const HOST_ALIGNMENT: usize = core::mem::align_of::<Self>();
    const ENCODED_SIZE: usize = 8;
    type Encoded = [u8; 8];

    fn encode_le(self) -> Self::Encoded {
        self.to_bits().to_le_bytes()
    }

    fn decode_le(bytes: Self::Encoded) -> Self {
        Self::from_bits(u64::from_le_bytes(bytes))
    }
}

impl PcuScalar for u8 {
    const TYPE: PcuScalarType = PcuScalarType::U8;
    const HOST_SIZE: usize = core::mem::size_of::<Self>();
    const HOST_ALIGNMENT: usize = core::mem::align_of::<Self>();
    const ENCODED_SIZE: usize = 1;
    type Encoded = [Self; 1];

    fn encode_le(self) -> Self::Encoded {
        [self]
    }

    fn decode_le(bytes: Self::Encoded) -> Self {
        bytes[0]
    }
}

impl PcuScalar for u16 {
    const TYPE: PcuScalarType = PcuScalarType::U16;
    const HOST_SIZE: usize = core::mem::size_of::<Self>();
    const HOST_ALIGNMENT: usize = core::mem::align_of::<Self>();
    const ENCODED_SIZE: usize = 2;
    type Encoded = [u8; 2];

    fn encode_le(self) -> Self::Encoded {
        self.to_le_bytes()
    }

    fn decode_le(bytes: Self::Encoded) -> Self {
        Self::from_le_bytes(bytes)
    }
}

impl PcuScalar for u32 {
    const TYPE: PcuScalarType = PcuScalarType::U32;
    const HOST_SIZE: usize = core::mem::size_of::<Self>();
    const HOST_ALIGNMENT: usize = core::mem::align_of::<Self>();
    const ENCODED_SIZE: usize = 4;
    type Encoded = [u8; 4];

    fn encode_le(self) -> Self::Encoded {
        self.to_le_bytes()
    }

    fn decode_le(bytes: Self::Encoded) -> Self {
        Self::from_le_bytes(bytes)
    }
}

impl PcuScalar for u64 {
    const TYPE: PcuScalarType = PcuScalarType::U64;
    const HOST_SIZE: usize = core::mem::size_of::<Self>();
    const HOST_ALIGNMENT: usize = core::mem::align_of::<Self>();
    const ENCODED_SIZE: usize = 8;
    type Encoded = [u8; 8];

    fn encode_le(self) -> Self::Encoded {
        self.to_le_bytes()
    }
    fn decode_le(bytes: Self::Encoded) -> Self {
        Self::from_le_bytes(bytes)
    }
}

impl PcuScalar for i8 {
    const TYPE: PcuScalarType = PcuScalarType::I8;
    const HOST_SIZE: usize = core::mem::size_of::<Self>();
    const HOST_ALIGNMENT: usize = core::mem::align_of::<Self>();
    const ENCODED_SIZE: usize = 1;
    type Encoded = [u8; 1];

    fn encode_le(self) -> Self::Encoded {
        self.to_le_bytes()
    }

    fn decode_le(bytes: Self::Encoded) -> Self {
        Self::from_le_bytes(bytes)
    }
}

impl PcuScalar for i16 {
    const TYPE: PcuScalarType = PcuScalarType::I16;
    const HOST_SIZE: usize = core::mem::size_of::<Self>();
    const HOST_ALIGNMENT: usize = core::mem::align_of::<Self>();
    const ENCODED_SIZE: usize = 2;
    type Encoded = [u8; 2];

    fn encode_le(self) -> Self::Encoded {
        self.to_le_bytes()
    }

    fn decode_le(bytes: Self::Encoded) -> Self {
        Self::from_le_bytes(bytes)
    }
}

impl PcuScalar for i32 {
    const TYPE: PcuScalarType = PcuScalarType::I32;
    const HOST_SIZE: usize = core::mem::size_of::<Self>();
    const HOST_ALIGNMENT: usize = core::mem::align_of::<Self>();
    const ENCODED_SIZE: usize = 4;
    type Encoded = [u8; 4];

    fn encode_le(self) -> Self::Encoded {
        self.to_le_bytes()
    }
    fn decode_le(bytes: Self::Encoded) -> Self {
        Self::from_le_bytes(bytes)
    }
}

impl PcuScalar for i64 {
    const TYPE: PcuScalarType = PcuScalarType::I64;
    const HOST_SIZE: usize = core::mem::size_of::<Self>();
    const HOST_ALIGNMENT: usize = core::mem::align_of::<Self>();
    const ENCODED_SIZE: usize = 8;
    type Encoded = [u8; 8];

    fn encode_le(self) -> Self::Encoded {
        self.to_le_bytes()
    }
    fn decode_le(bytes: Self::Encoded) -> Self {
        Self::from_le_bytes(bytes)
    }
}

impl<'a> PcuBinding<'a> {
    /// Declares a scalar resource with its IR element type derived from `T`.
    ///
    /// This does not establish device placement or a zero-copy host representation.
    #[must_use]
    pub const fn scalar<T: PcuScalar>(
        name: Option<&'a str>,
        set: u32,
        binding: u32,
        storage: PcuBindingStorageClass,
        access: PcuBindingAccess,
    ) -> Self {
        Self::value(
            name,
            set,
            binding,
            storage,
            access,
            PcuValueType::Scalar(T::TYPE),
        )
    }
}

#[cfg(test)]
mod tests {
    #[rustfmt::skip]
    use super::{
        PcuBf16Bits,
        PcuF16Bits,
        PcuScalar,
    };
    #[rustfmt::skip]
    use crate::{
        PcuBinding,
        PcuBindingAccess,
        PcuBindingStorageClass,
        PcuScalarType,
        PcuValueType,
    };

    #[test]
    fn i8_scalar_encoding_preserves_twos_complement_bits() {
        let value = i8::MIN + 37;
        assert_eq!(i8::decode_le(value.encode_le()), value);
        assert_eq!(<i8 as PcuScalar>::TYPE, PcuScalarType::I8);
        assert_eq!(<i8 as PcuScalar>::ENCODED_SIZE, 1);
        assert_eq!(<i8 as PcuScalar>::HOST_SIZE, 1);
    }

    #[test]
    fn scalar_encodings_preserve_integer_and_float_bit_patterns() {
        let integer = 0x1234_56ab_u32;
        assert_eq!(integer.encode_le(), [0xab, 0x56, 0x34, 0x12]);
        assert_eq!(u32::decode_le(integer.encode_le()), integer);
        let wide = 0x0123_4567_89ab_cdef_u64;
        assert_eq!(
            wide.encode_le(),
            [0xef, 0xcd, 0xab, 0x89, 0x67, 0x45, 0x23, 0x01]
        );
        assert_eq!(u64::decode_le(wide.encode_le()), wide);
        let signed32 = i32::MIN + 0x1234;
        assert_eq!(i32::decode_le(signed32.encode_le()), signed32);
        assert_eq!(<i32 as PcuScalar>::TYPE, PcuScalarType::I32);
        let signed = i64::MIN + 0x0123_4567;
        assert_eq!(i64::decode_le(signed.encode_le()), signed);
        assert_eq!(<i64 as PcuScalar>::TYPE, PcuScalarType::I64);
        assert_eq!(<u64 as PcuScalar>::TYPE, PcuScalarType::U64);

        let nan = f32::from_bits(0x7fc0_1234);
        assert_eq!(f32::decode_le(nan.encode_le()).to_bits(), nan.to_bits());
        assert_eq!(<f32 as PcuScalar>::TYPE, PcuScalarType::F32);
        let wide_nan = f64::from_bits(0x7ff8_0000_0000_1234);
        assert_eq!(
            f64::decode_le(wide_nan.encode_le()).to_bits(),
            wide_nan.to_bits()
        );
        assert_eq!(<f64 as PcuScalar>::TYPE, PcuScalarType::F64);
        assert_eq!(<f64 as PcuScalar>::ENCODED_SIZE, 8);
        assert_eq!(<u32 as PcuScalar>::TYPE, PcuScalarType::U32);
        assert_eq!(<f32 as PcuScalar>::ENCODED_SIZE, 4);
        assert_eq!(
            <u32 as PcuScalar>::HOST_ALIGNMENT,
            core::mem::align_of::<u32>()
        );

        let scalar = PcuBinding::scalar::<u32>(
            Some("values"),
            0,
            0,
            PcuBindingStorageClass::Storage,
            PcuBindingAccess::ReadOnly,
        );
        assert_eq!(scalar.value_type(), Some(PcuValueType::u32()));
    }

    #[test]
    fn narrow_scalar_encodings_preserve_bits_and_type_identity() {
        let signed16 = i16::MIN + 0x1234;
        assert_eq!(i16::decode_le(signed16.encode_le()), signed16);
        assert_eq!(<i16 as PcuScalar>::TYPE, PcuScalarType::I16);
        let byte = 0xb2_u8;
        assert_eq!(u8::decode_le(byte.encode_le()), byte);
        assert_eq!(<u8 as PcuScalar>::TYPE, PcuScalarType::U8);
        assert_eq!(<u8 as PcuScalar>::ENCODED_SIZE, 1);
        let narrow = 0xa1b2_u16;
        assert_eq!(u16::decode_le(narrow.encode_le()), narrow);
        assert_eq!(<u16 as PcuScalar>::TYPE, PcuScalarType::U16);
        assert_eq!(<u16 as PcuScalar>::ENCODED_SIZE, 2);
    }

    #[test]
    fn half_storage_transport_preserves_all_bits_without_arithmetic_claims() {
        for bits in [0, 1, 0x8000, 0x7c00, 0x7e01, u16::MAX] {
            let f16 = PcuF16Bits::from_bits(bits);
            let bf16 = PcuBf16Bits::from_bits(bits);
            assert_eq!(PcuF16Bits::decode_le(f16.encode_le()).to_bits(), bits);
            assert_eq!(PcuBf16Bits::decode_le(bf16.encode_le()).to_bits(), bits);
        }
        assert_eq!(<PcuF16Bits as PcuScalar>::TYPE, PcuScalarType::F16);
        assert_eq!(<PcuBf16Bits as PcuScalar>::TYPE, PcuScalarType::BF16);
        assert_eq!(<PcuF16Bits as PcuScalar>::HOST_SIZE, 2);
        assert_eq!(<PcuBf16Bits as PcuScalar>::HOST_SIZE, 2);
    }

    #[test]
    fn f16_conversion_uses_ties_even_and_preserves_ieee_special_values() {
        // Exactly representable values and the smallest/largest finite boundaries.
        for (single, half) in [
            (0x0000_0000, 0x0000),
            (0x8000_0000, 0x8000),
            (0x3f80_0000, 0x3c00),
            (0x477f_e000, 0x7bff),
            (0x3880_0000, 0x0400),
            (0x3380_0000, 0x0001),
            (0x7f80_0000, 0x7c00),
            (0xff80_0000, 0xfc00),
        ] {
            assert_eq!(PcuF16Bits::from_f32(f32::from_bits(single)).to_bits(), half);
            assert_eq!(PcuF16Bits::from_bits(half).to_f32().to_bits(), single);
        }

        // Halfway values choose the even destination significand, including subnormals.
        assert_eq!(
            PcuF16Bits::from_f32(f32::from_bits(0x3f80_1000)).to_bits(),
            0x3c00
        );
        assert_eq!(
            PcuF16Bits::from_f32(f32::from_bits(0x3f80_3000)).to_bits(),
            0x3c02
        );
        assert_eq!(
            PcuF16Bits::from_f32(f32::from_bits(0x3300_0000)).to_bits(),
            0x0000
        );
        assert_eq!(
            PcuF16Bits::from_f32(f32::from_bits(0x3380_0000)).to_bits(),
            0x0001
        );

        // Narrowing quiets NaNs and preserves sign plus retained payload; widening preserves
        // the resulting quiet bit and payload exactly.
        let negative_nan = PcuF16Bits::from_f32(f32::from_bits(0xff80_1234));
        assert_eq!(negative_nan.to_bits(), 0xfe00);
        assert_eq!(negative_nan.to_f32().to_bits(), 0xffc0_0000);
        assert_eq!(
            PcuF16Bits::from_bits(0x7c01).to_f32().to_bits(),
            0x7f80_2000
        );
    }

    #[test]
    fn bf16_conversion_uses_ties_even_and_preserves_ieee_special_values() {
        for (single, half) in [
            (0x0000_0000, 0x0000),
            (0x8000_0000, 0x8000),
            (0x3f80_0000, 0x3f80),
            (0x7f80_0000, 0x7f80),
            (0xff80_0000, 0xff80),
        ] {
            assert_eq!(
                PcuBf16Bits::from_f32(f32::from_bits(single)).to_bits(),
                half
            );
            assert_eq!(PcuBf16Bits::from_bits(half).to_f32().to_bits(), single);
        }

        assert_eq!(
            PcuBf16Bits::from_f32(f32::from_bits(0x3f80_8000)).to_bits(),
            0x3f80
        );
        assert_eq!(
            PcuBf16Bits::from_f32(f32::from_bits(0x3f81_8000)).to_bits(),
            0x3f82
        );
        let negative_nan = PcuBf16Bits::from_f32(f32::from_bits(0xff80_1234));
        assert_eq!(negative_nan.to_bits(), 0xffc0);
        assert_eq!(negative_nan.to_f32().to_bits(), 0xffc0_0000);
        assert_eq!(
            PcuBf16Bits::from_bits(0x7f81).to_f32().to_bits(),
            0x7f81_0000
        );
    }

    #[test]
    fn widening_then_narrowing_preserves_every_non_nan_half_encoding() {
        for bits in 0..=u16::MAX {
            let f16 = PcuF16Bits::from_bits(bits);
            let bf16 = PcuBf16Bits::from_bits(bits);
            let f16_round_trip = PcuF16Bits::from_f32(f16.to_f32()).to_bits();
            let bf16_round_trip = PcuBf16Bits::from_f32(bf16.to_f32()).to_bits();
            let f16_is_nan = bits & 0x7c00 == 0x7c00 && bits & 0x03ff != 0;
            let bf16_is_nan = bits & 0x7f80 == 0x7f80 && bits & 0x007f != 0;

            assert_eq!(
                f16_round_trip,
                if f16_is_nan { bits | 0x0200 } else { bits },
                "binary16 bits {bits:#06x}"
            );
            assert_eq!(
                bf16_round_trip,
                if bf16_is_nan { bits | 0x0040 } else { bits },
                "bfloat16 bits {bits:#06x}"
            );
        }
    }
}
