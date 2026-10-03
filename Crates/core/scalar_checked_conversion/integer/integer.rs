//! Finite unsigned-integer conversion with fixed IEEE nearest-even rounding.
//!
//! IEEE Std 754-2019 clauses 4.3.1 and 5.4.2: round once into the destination
//! precision with ties to even. All u64 values fit within binary32/binary64's
//! finite exponent range. Integer packing ignores the host floating environment;
//! this utility does not declare device conversion-instruction support.

/// Converts an unsigned 64-bit integer to binary32, nearest with ties to even.
///
/// Zero maps to positive zero; even `u64::MAX` rounds to a finite binary32 value.
#[must_use]
#[allow(clippy::cast_possible_truncation)] // Packing is bounded to binary32's 32-bit fields.
pub const fn u64_to_f32_nearest_even(value: u64) -> f32 {
    f32::from_bits(pack_count(value, 23, 127) as u32)
}

/// Converts an unsigned 64-bit integer to binary64, nearest with ties to even.
///
/// Zero maps to positive zero; even `u64::MAX` rounds to a finite binary64 value.
#[must_use]
pub const fn u64_to_f64_nearest_even(value: u64) -> f64 {
    f64::from_bits(pack_count(value, 52, 1023))
}

const fn pack_count(count: u64, fraction_bits: u32, bias: u32) -> u64 {
    if count == 0 {
        return 0;
    }
    let mut exponent = count.ilog2();
    let mut significand = if exponent <= fraction_bits {
        count << (fraction_bits - exponent)
    } else {
        let shift = exponent - fraction_bits;
        let quotient = count >> shift;
        let remainder = count & ((1_u64 << shift) - 1);
        let halfway = 1_u64 << (shift - 1);
        let increment = remainder > halfway || (remainder == halfway && quotient & 1 != 0);
        quotient + increment as u64
    };
    if significand == 1_u64 << (fraction_bits + 1) {
        significand >>= 1;
        exponent += 1;
    }
    ((exponent + bias) as u64) << fraction_bits | (significand & ((1_u64 << fraction_bits) - 1))
}

#[cfg(test)]
mod tests;
