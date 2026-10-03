#[rustfmt::skip]
use super::{
    BF16,
    F16,
    Format,
    narrow_f64,
};
#[rustfmt::skip]
use crate::{
    PcuBf16Bits,
    PcuClampedError,
    PcuExecutionFaultKind as Fault,
    PcuF16Bits,
    PcuFloatUnderflowPolicy as Policy,
};

fn value(bits: u16, format: Format) -> f64 {
    if format.fraction == 10 {
        f64::from(PcuF16Bits::from_bits(bits).to_f32())
    } else {
        f64::from(PcuBf16Bits::from_bits(bits).to_f32())
    }
}

// Independent nearest-neighbor search. Every destination value and midpoint is
// exactly representable in f64; comparing the input to that midpoint is exact.
#[allow(clippy::float_cmp)] // Exact dyadic midpoint equality defines ties; tolerance is incorrect.
fn oracle(source: f64, format: Format, policy: Policy) -> Result<u16, Fault> {
    if !source.is_finite() {
        return Err(Fault::InvalidFloatingOperand);
    }
    let sign = if source.is_sign_negative() { 0x8000 } else { 0 };
    let source = source.abs();
    let max = ((format.exponent_mask - 1) << format.fraction) | ((1 << format.fraction) - 1);
    let max_value = value(max, format);
    let overflow_midpoint = max_value + (max_value - value(max - 1, format)) / 2.0;
    if source >= overflow_midpoint {
        return Err(Fault::ArithmeticOverflow);
    }
    let mut lower = 0;
    let mut upper = max;
    while lower < upper {
        let midpoint = lower + (upper - lower).div_ceil(2);
        if value(midpoint, format) <= source {
            lower = midpoint;
        } else {
            upper = midpoint - 1;
        }
    }
    let rounded = if lower == max {
        max
    } else {
        let midpoint = value(lower, format).midpoint(value(lower + 1, format));
        lower + u16::from(source > midpoint || (source == midpoint && lower & 1 != 0))
    };
    let normal = 1 << format.fraction;
    let tininess_boundary = value(normal, format) - value(1, format) / 4.0;
    let tiny_inexact =
        source != 0.0 && source < tininess_boundary && source != value(rounded, format);
    if (policy != Policy::AllowGradualUnderflow && tiny_inexact)
        || (policy == Policy::RejectSubnormalResult && rounded != 0 && rounded < normal)
    {
        return Err(Fault::ArithmeticUnderflow);
    }
    Ok(sign | rounded)
}

fn compare(source: f64, format: Format) {
    for policy in [
        Policy::IeeeAfterRounding,
        Policy::RejectSubnormalResult,
        Policy::AllowGradualUnderflow,
    ] {
        assert_eq!(
            narrow_f64(source.to_bits(), format, policy).map_err(|fault| fault.kind()),
            oracle(source, format, policy),
            "source bits={:#018x}, destination fraction={}, policy={policy:?}",
            source.to_bits(),
            format.fraction,
        );
    }
}

#[test]
fn every_half_encoding_round_trips_and_every_midpoint_uses_destination_even_parity() {
    for format in [F16, BF16] {
        let max = ((format.exponent_mask - 1) << format.fraction) | ((1 << format.fraction) - 1);
        for bits in 0..=max {
            for sign in [0, 0x8000] {
                compare(value(bits | sign, format), format);
            }
            if bits == max {
                continue;
            }
            let midpoint = value(bits, format).midpoint(value(bits + 1, format));
            for source in [midpoint.next_down(), midpoint, midpoint.next_up()] {
                compare(source, format);
                compare(-source, format);
            }
        }
    }
}

#[test]
fn random_full_exponent_inputs_match_independent_destination_midpoint_oracle() {
    let mut state = 0x7a36_958d_d42e_abc1_u64;
    for _ in 0..100_000 {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        for format in [F16, BF16] {
            compare(f64::from_bits(state), format);
            // The low word is intentionally a separately distributed binary32 encoding.
            let low_word = u32::from_le_bytes(state.to_le_bytes()[..4].try_into().unwrap());
            let source = f32::from_bits(low_word);
            let actual = if format.fraction == 10 {
                PcuF16Bits::pcu_checked_from_f32_with_policy(source, Policy::AllowGradualUnderflow)
                    .map(PcuF16Bits::to_bits)
            } else {
                PcuBf16Bits::pcu_checked_from_f32_with_policy(source, Policy::AllowGradualUnderflow)
                    .map(PcuBf16Bits::to_bits)
            };
            assert_eq!(
                actual,
                oracle(f64::from(source), format, Policy::AllowGradualUnderflow)
            );
        }
    }
}

#[test]
#[allow(clippy::cast_possible_truncation)] // A deliberate rounded F32 intermediate is the rejected double-rounding control.
fn direct_binary64_narrowing_avoids_a_binary32_double_rounding() {
    let half_midpoint = 1.0 + 2.0_f64.powi(-11);
    let source = half_midpoint.next_up();
    assert_eq!(
        PcuF16Bits::pcu_checked_from_f64(source).unwrap().to_bits(),
        0x3c01
    );
    assert_eq!(PcuF16Bits::from_f32(source as f32).to_bits(), 0x3c00);
    let bfloat_midpoint = 1.0 + 2.0_f64.powi(-8);
    let source = bfloat_midpoint.next_up();
    assert_eq!(
        PcuBf16Bits::pcu_checked_from_f64(source).unwrap().to_bits(),
        0x3f81
    );
    assert_eq!(PcuBf16Bits::from_f32(source as f32).to_bits(), 0x3f80);
}

#[test]
fn widening_preserves_all_finite_bits_and_rejects_specials_without_quieting() {
    for bits in 0..=u16::MAX {
        let half = PcuF16Bits::from_bits(bits);
        let bfloat = PcuBf16Bits::from_bits(bits);
        for (encoded, to_f32, to_f64) in [
            (
                half.to_f32(),
                half.pcu_checked_to_f32(),
                half.pcu_checked_to_f64(),
            ),
            (
                bfloat.to_f32(),
                bfloat.pcu_checked_to_f32(),
                bfloat.pcu_checked_to_f64(),
            ),
        ] {
            if encoded.is_finite() {
                assert_eq!(to_f32.unwrap().to_bits(), encoded.to_bits());
                assert_eq!(to_f64.unwrap().to_bits(), f64::from(encoded).to_bits());
            } else {
                assert_eq!(to_f32.map(f32::to_bits), Err(Fault::InvalidFloatingOperand));
                assert_eq!(to_f64.map(f64::to_bits), Err(Fault::InvalidFloatingOperand));
            }
        }
    }
}

#[test]
fn fatal_and_observable_range_recovery_remain_distinct() {
    let overflow = PcuF16Bits::pcu_clamped_from_f64(-f64::MAX).unwrap_err();
    let PcuClampedError::Range(fault) = overflow else {
        panic!("overflow must retain recovery");
    };
    assert_eq!(fault.kind(), Fault::ArithmeticOverflow);
    assert_eq!(fault.clamped_value().to_bits(), 0xfbff);
    let tiny = PcuBf16Bits::pcu_clamped_from_f64(-f64::from_bits(1)).unwrap_err();
    let PcuClampedError::Range(fault) = tiny else {
        panic!("underflow must retain recovery");
    };
    assert_eq!(fault.kind(), Fault::ArithmeticUnderflow);
    assert_eq!(fault.clamped_value().to_bits(), 0x8000);
    for source in [f64::INFINITY, f64::NEG_INFINITY, f64::NAN] {
        assert_eq!(
            PcuF16Bits::pcu_clamped_from_f64(source),
            Err(PcuClampedError::Fatal(Fault::InvalidFloatingOperand))
        );
    }
}
