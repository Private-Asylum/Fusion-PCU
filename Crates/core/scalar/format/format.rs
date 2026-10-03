//! Named floating storage formats, separate from compute/accumulation permissions.
//!
//! IEEE binary16/32/64/128 field sizes follow IEEE Std 754-2019 clause 3.6.
//! Binary256 uses the clause 3.6 generalized binary interchange construction
//! (237-bit precision, 19-bit exponent), not a new basic IEEE format. BF16 is
//! not an IEEE basic format. The named FP8 encodings are OCP OFP8 revision 1.0:
//! <https://www.opencompute.org/documents/ocp-8-bit-floating-point-specification-ofp8-revision-1-0-2023-06-20-pdf>.
//! None of this metadata permits TF32, FTZ, approximate arithmetic or native
//! conversion. Those require independent implementation/precision contracts.
use crate::PcuScalarType;

/// Exact fields of one binary storage encoding. Precision includes the hidden bit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PcuBinaryFloatFormat {
    pub storage_bits: u16,
    pub exponent_bits: u8,
    pub precision_bits: u16,
    pub exponent_bias: i32,
    pub has_infinity: bool,
}

impl PcuScalarType {
    /// Returns representation metadata, not execution support or rounding policy.
    #[allow(clippy::similar_names)] // Field names distinguish exponent width from bias.
    #[must_use]
    pub const fn binary_float_format(self) -> Option<PcuBinaryFloatFormat> {
        let (storage_bits, exponent_bits, precision_bits, exponent_bias, has_infinity) = match self
        {
            Self::F16 => (16, 5, 11, 15, true),
            Self::BF16 => (16, 8, 8, 127, true),
            Self::F32 => (32, 8, 24, 127, true),
            Self::F64 => (64, 11, 53, 1023, true),
            Self::F128 => (128, 15, 113, 16_383, true),
            Self::F256 => (256, 19, 237, 262_143, true),
            Self::F8E4M3FN => (8, 4, 4, 7, false),
            Self::F8E5M2 => (8, 5, 3, 15, true),
            _ => return None,
        };
        Some(PcuBinaryFloatFormat {
            storage_bits,
            exponent_bits,
            precision_bits,
            exponent_bias,
            has_infinity,
        })
    }
}

/// Representation classification without floating hardware operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PcuFloatClass {
    Zero,
    Subnormal,
    Normal,
    Infinite,
    QuietNaN,
    SignalingNaN,
}

pub(super) fn classify_words<const N: usize>(words: [u64; N], exponent_bits: u8) -> PcuFloatClass {
    let fraction_bits = 63 - exponent_bits;
    let high = words[N - 1];
    let exponent_mask = (1_u64 << exponent_bits) - 1;
    let exponent = (high >> fraction_bits) & exponent_mask;
    let fraction = (high & ((1_u64 << fraction_bits) - 1)) != 0
        || words[..N - 1].iter().any(|word| *word != 0);
    match (exponent, fraction) {
        (0, false) => PcuFloatClass::Zero,
        (0, true) => PcuFloatClass::Subnormal,
        (exponent, false) if exponent == exponent_mask => PcuFloatClass::Infinite,
        (exponent, true) if exponent == exponent_mask => {
            if high & (1_u64 << (fraction_bits - 1)) != 0 {
                PcuFloatClass::QuietNaN
            } else {
                PcuFloatClass::SignalingNaN
            }
        }
        _ => PcuFloatClass::Normal,
    }
}

macro_rules! fp8 {
    ($name:ident, $kind:ident) => {
        #[doc = concat!("Lossless ", stringify!($kind), " storage; no implicit arithmetic conversion.")]
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        #[repr(transparent)]
        pub struct $name(u8);
        impl $name {
            #[must_use]
            pub const fn from_bits(bits: u8) -> Self {
                Self(bits)
            }
            #[must_use]
            pub const fn to_bits(self) -> u8 {
                self.0
            }
        }
        impl super::sealed::Sealed for $name {}
        impl crate::PcuScalar for $name {
            const TYPE: PcuScalarType = PcuScalarType::$kind;
            const HOST_SIZE: usize = 1;
            const HOST_ALIGNMENT: usize = 1;
            const ENCODED_SIZE: usize = 1;
            type Encoded = [u8; 1];
            fn encode_le(self) -> Self::Encoded {
                [self.0]
            }
            fn decode_le(bytes: Self::Encoded) -> Self {
                Self(bytes[0])
            }
        }
    };
}
fp8!(PcuF8E4M3FnBits, F8E4M3FN);
fp8!(PcuF8E5M2Bits, F8E5M2);

impl PcuF8E4M3FnBits {
    /// Widens OFP8 E4M3FN exactly; its two NaN encodings become signed quiet NaNs.
    #[must_use]
    pub const fn to_f32(self) -> f32 {
        widen_fp8(self.0, 3, 7, 0x7e)
    }
    /// OFP8 E4M3's exponent-all-ones encodings remain finite except fraction 111.
    #[must_use]
    pub const fn classify(self) -> PcuFloatClass {
        let magnitude = self.0 & 0x7f;
        if magnitude == 0x7f {
            PcuFloatClass::QuietNaN
        } else if magnitude == 0 {
            PcuFloatClass::Zero
        } else if magnitude < 8 {
            PcuFloatClass::Subnormal
        } else {
            PcuFloatClass::Normal
        }
    }
}
impl PcuF8E5M2Bits {
    /// Widens OFP8 E5M2 exactly, retaining zero sign and available NaN payload/quiet bits.
    #[must_use]
    pub const fn to_f32(self) -> f32 {
        widen_fp8(self.0, 2, 15, 0x7b)
    }
    #[must_use]
    pub const fn classify(self) -> PcuFloatClass {
        let magnitude = self.0 & 0x7f;
        match magnitude {
            0 => PcuFloatClass::Zero,
            1..=3 => PcuFloatClass::Subnormal,
            0x7c => PcuFloatClass::Infinite,
            0x7d => PcuFloatClass::SignalingNaN,
            0x7e..=0x7f => PcuFloatClass::QuietNaN,
            _ => PcuFloatClass::Normal,
        }
    }
}

// All casts are bounded masked fields or finite FP8 normalized exponents.
// This is bitfield construction, including subnormals; host FTZ is irrelevant.
#[allow(clippy::cast_possible_wrap, clippy::cast_sign_loss)]
const fn widen_fp8(bits: u8, fraction_bits: u32, exponent_bias: i32, max_finite: u8) -> f32 {
    let sign = ((bits & 0x80) as u32) << 24;
    let magnitude = bits & 0x7f;
    let fraction = magnitude as u32 & ((1 << fraction_bits) - 1);
    if magnitude > max_finite {
        return f32::from_bits(sign | 0x7f80_0000 | (fraction << (23 - fraction_bits)));
    }
    let exponent = (magnitude as u32 >> fraction_bits) as i32;
    let significand = fraction | if exponent == 0 { 0 } else { 1 << fraction_bits };
    if significand == 0 {
        return f32::from_bits(sign);
    }
    let leading = significand.ilog2(); // Zero was handled above; no panic is possible.
    let scale = if exponent == 0 { 1 } else { exponent } - exponent_bias - fraction_bits as i32;
    let destination_exponent = (127 + scale + leading as i32) as u32;
    f32::from_bits(
        sign | (destination_exponent << 23) | ((significand << (23 - leading)) & 0x007f_ffff),
    )
}
impl super::PcuF128Bits {
    #[must_use]
    pub fn classify(self) -> PcuFloatClass {
        classify_words(self.to_limbs_le(), 15)
    }
}
impl super::PcuF256Bits {
    #[must_use]
    pub fn classify(self) -> PcuFloatClass {
        classify_words(self.to_limbs_le(), 19)
    }
}
#[cfg(test)]
mod tests;
