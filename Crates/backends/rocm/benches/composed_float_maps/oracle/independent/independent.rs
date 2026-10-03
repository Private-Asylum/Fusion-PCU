//! Independent binary64/exact-dyadic search for the two-step low-format chain.
//! Destination precision is <=11 bits. Self-add and the subsequent finite product
//! have <=22 exact significand bits and exponent range inside binary64 normality,
//! so host rounding/flush state cannot change these exact reference operations.
//! Packing uses explicit midpoint/ties-even search, separately from core arithmetic.
use super::Format;
#[rustfmt::skip]
use fusion_pcu::{
    PcuFloatUnderflowPolicy,
    PcuRangePolicy,
};
fn unpack<T: Format>(bits: u64) -> f64 {
    let fraction = T::MIN_NORMAL.ilog2();
    let exponent = (bits & (T::SIGN - 1)) >> fraction;
    let significand =
        (bits & ((1 << fraction) - 1)) | if exponent == 0 { 0 } else { 1 << fraction };
    let scale = i32::try_from(exponent.max(1)).unwrap()
        - i32::try_from(T::ONE >> fraction).unwrap()
        - i32::try_from(fraction).unwrap();
    <f64 as From<u32>>::from(u32::try_from(significand).unwrap())
        * 2f64.powi(scale)
        * if bits & T::SIGN == 0 { 1.0 } else { -1.0 }
}
#[allow(clippy::float_cmp)] // All compared dyadics/midpoints are represented exactly in binary64.
fn pack<T: Format>(result: f64, uf: PcuFloatUnderflowPolicy) -> (T, u64) {
    let sign = if result.is_sign_negative() {
        T::SIGN
    } else {
        0
    };
    let magnitude = result.abs();
    if magnitude == 0.0 {
        return (T::from(sign), 0);
    }
    let fraction = T::MIN_NORMAL.ilog2();
    let bias = i32::try_from(T::ONE >> fraction).unwrap();
    let maximum_exponent = i32::try_from(T::MAX >> fraction).unwrap() - bias;
    let spacing = 2f64.powi(maximum_exponent - i32::try_from(fraction).unwrap());
    let overflow_midpoint = unpack::<T>(T::MAX).midpoint(unpack::<T>(T::MAX) + spacing);
    if magnitude > overflow_midpoint || (magnitude == overflow_midpoint && T::MAX & 1 != 0) {
        return (T::from(sign | T::MAX), 3);
    }
    let mut low = 0;
    let mut high = T::MAX;
    while low < high {
        let middle = low + (high - low).div_ceil(2);
        if unpack::<T>(middle) <= magnitude {
            low = middle;
        } else {
            high = middle - 1;
        }
    }
    let magnitude_bits = if low == T::MAX {
        low
    } else {
        let midpoint = unpack::<T>(low).midpoint(unpack::<T>(low + 1));
        low + u64::from(magnitude > midpoint || (magnitude == midpoint && low & 1 != 0))
    };
    let rounded = T::from(sign | magnitude_bits);
    let exponent = i32::try_from((magnitude.to_bits() >> 52) & 0x7ff).unwrap() - 1023;
    let units =
        (magnitude / 2f64.powi(exponent - i32::try_from(fraction).unwrap())).round_ties_even();
    let carried = units == <f64 as From<u32>>::from(1u32 << (fraction + 1));
    let tiny = exponent + i32::from(carried) < 1 - bias;
    let inexact = unpack::<T>(rounded.bits()) != result;
    let rejected = match uf {
        PcuFloatUnderflowPolicy::IeeeAfterRounding => tiny && inexact,
        PcuFloatUnderflowPolicy::RejectSubnormalResult => {
            (tiny && inexact) || (magnitude_bits != 0 && magnitude_bits < T::MIN_NORMAL)
        }
        PcuFloatUnderflowPolicy::AllowGradualUnderflow => false,
    };
    (rounded, if rejected { 4 } else { 0 })
}
pub fn expected<T: Format>(
    input: T,
    uf: PcuFloatUnderflowPolicy,
    range: PcuRangePolicy,
) -> (u64, Option<T>) {
    assert!(
        u16::try_from(T::MAX).is_ok(),
        "independent model is bounded to the four low formats"
    );
    if input.bits() & (T::SIGN - 1) > T::MAX {
        return (5, None);
    }
    let (sum, first) = pack::<T>(unpack::<T>(input.bits()) + unpack::<T>(input.bits()), uf);
    if first != 0 && range == PcuRangePolicy::Reject {
        return (first, None);
    }
    let (product, second) = pack::<T>(unpack::<T>(sum.bits()) * unpack::<T>(input.bits()), uf);
    if second != 0 && range == PcuRangePolicy::Reject {
        return (second, None);
    }
    let notice = if first != 0 { first } else { second };
    (
        if notice == 0 {
            u64::MAX
        } else {
            (1 << 63) | notice
        },
        Some(product),
    )
}

pub fn corpus<T: Format>() -> Vec<u64> {
    let maximum_bits = (T::SIGN << 1) - 1;
    if maximum_bits <= 255 {
        return (0..=maximum_bits).collect();
    }
    let mut values = vec![
        0,
        T::SIGN,
        1,
        T::SIGN | 1,
        T::MIN_NORMAL - 1,
        T::MIN_NORMAL,
        T::MAX,
        T::SIGN | T::MAX,
        T::MAX + 1,
        T::ONE,
        T::SIGN | T::ONE,
    ];
    values.extend((0..1024u64).map(|value| value.wrapping_mul(40503) & maximum_bits));
    values
}
