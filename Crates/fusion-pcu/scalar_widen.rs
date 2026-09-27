//! Exact scalar widening helpers whose semantics are defined over IEEE bit patterns.

/// Return the binary64 encoding of the exact widening of an IEEE binary32 encoding.
///
/// This is bit based so signaling NaNs never pass through floating point arithmetic. It preserves
/// the sign, all 23 source payload bits, and the quiet/signaling bit, as well as all finite values.
#[must_use]
#[allow(clippy::cast_lossless)]
pub const fn f32_bits_to_f64_bits(bits: u32) -> u64 {
    let sign = ((bits >> 31) as u64) << 63;
    let exponent = (bits >> 23) & 0xff;
    let fraction = bits & 0x007f_ffff;
    if exponent == 0xff {
        return sign | (0x7ff_u64 << 52) | ((fraction as u64) << 29);
    }
    if exponent != 0 {
        let widened_exponent = (exponent + (1023 - 127)) as u64;
        return sign | (widened_exponent << 52) | ((fraction as u64) << 29);
    }
    if fraction == 0 {
        return sign;
    }

    let mut leading_bit = 22_u32;
    while (fraction & (1_u32 << leading_bit)) == 0 {
        leading_bit -= 1;
    }
    let significand = (fraction as u64) << (52 - leading_bit);
    // A nonzero binary32 subnormal's top fraction bit is in 0..=22, so its binary64 exponent
    // field is 874..=896. This integer form avoids signed exponent casts in this const helper.
    let widened_exponent = (leading_bit + 874) as u64;
    sign | (widened_exponent << 52) | (significand & 0x000f_ffff_ffff_ffff)
}

/// Widen an `f32` value exactly, preserving NaN payload and signaling state.
#[must_use]
pub const fn widen_f32_exact(value: f32) -> f64 {
    f64::from_bits(f32_bits_to_f64_bits(value.to_bits()))
}

#[cfg(test)]
mod tests {
    use super::f32_bits_to_f64_bits;

    #[test]
    #[allow(clippy::verbose_bit_mask)] // IEEE exponent/fraction fields are explicitly classified by mask.
    fn preserves_binary32_edges_and_nan_payloads() {
        for bits in [
            0x0000_0000,
            0x8000_0000,
            0x0000_0001,
            0x007f_ffff,
            0x0080_0000,
            0x3f80_0000,
            0x7f7f_ffff,
            0x7f80_0000,
            0xff80_0000,
            0x7f80_0001,
            0xffc1_2345,
            0x7fc0_0000,
        ] {
            let expected = if bits & 0x7f80_0000 == 0x7f80_0000 {
                (u64::from(bits & 0x8000_0000) << 32)
                    | (0x7ff_u64 << 52)
                    | (u64::from(bits & 0x007f_ffff) << 29)
            } else if bits & 0x7f80_0000 == 0 {
                if bits & 0x007f_ffff == 0 {
                    u64::from(bits & 0x8000_0000) << 32
                } else {
                    let value = f32::from_bits(bits);
                    f64::from(value).to_bits()
                }
            } else {
                f64::from(f32::from_bits(bits)).to_bits()
            };
            assert_eq!(f32_bits_to_f64_bits(bits), expected, "{bits:#010x}");
        }
    }
}
