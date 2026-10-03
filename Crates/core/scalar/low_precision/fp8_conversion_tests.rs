#[rustfmt::skip]
use super::{
    E4M3FN,
    E5M2,
    Format,
    round_value,
    value,
};
#[rustfmt::skip]
use crate::{
    PcuCheckedFloat,
    PcuClampedError,
    PcuExecutionFaultKind,
    PcuF8E4M3FnBits,
    PcuF8E5M2Bits,
    PcuFloatUnderflowPolicy,
};

#[allow(clippy::float_cmp)] // Exact representability/ties define errors; tolerances would mask contract failures.
fn expected(
    source: f64,
    format: Format,
    policy: PcuFloatUnderflowPolicy,
) -> Result<u16, PcuExecutionFaultKind> {
    let bits = round_value(source, format)?;
    let magnitude = bits & 0x7f;
    let minimum_normal = 1 << format.fraction;
    let tiny_boundary = value(minimum_normal, format) - value(1, format) / 4.0;
    let tiny_inexact =
        source != 0.0 && source.abs() < tiny_boundary && source != value(bits, format);
    let reject = match policy {
        PcuFloatUnderflowPolicy::AllowGradualUnderflow => false,
        PcuFloatUnderflowPolicy::IeeeAfterRounding => tiny_inexact,
        PcuFloatUnderflowPolicy::RejectSubnormalResult => {
            tiny_inexact || (magnitude != 0 && magnitude < minimum_normal)
        }
    };
    if reject {
        Err(PcuExecutionFaultKind::ArithmeticUnderflow)
    } else {
        Ok(bits)
    }
}

fn actual(
    source: f64,
    format: Format,
    policy: PcuFloatUnderflowPolicy,
) -> Result<u16, PcuClampedError<u16>> {
    // map_err below preserves the payload without referring to the reference's packer.
    macro_rules! narrow {
        ($ty:ty) => {
            <$ty>::pcu_clamped_from_f64_with_policy(source, policy)
                .map(|bits| u16::from(bits.to_bits()))
                .map_err(|error| match error {
                    PcuClampedError::Fatal(kind) => PcuClampedError::Fatal(kind),
                    PcuClampedError::Range(fault) => {
                        PcuClampedError::Range(crate::PcuClampedFault::new(
                            fault.kind(),
                            u16::from(fault.clamped_value().to_bits()),
                        ))
                    }
                })
        };
    }
    if format.fraction == 3 {
        narrow!(PcuF8E4M3FnBits)
    } else {
        narrow!(PcuF8E5M2Bits)
    }
}

fn compare(source: f64, format: Format) {
    for policy in [
        PcuFloatUnderflowPolicy::IeeeAfterRounding,
        PcuFloatUnderflowPolicy::AllowGradualUnderflow,
        PcuFloatUnderflowPolicy::RejectSubnormalResult,
    ] {
        let observed = actual(source, format, policy);
        assert_eq!(
            observed.as_ref().copied().map_err(PcuClampedError::kind),
            expected(source, format, policy),
            "source={:#018x}, fraction={}, policy={policy:?}",
            source.to_bits(),
            format.fraction
        );
        if let Err(PcuClampedError::Range(fault)) = observed {
            let recovery = match fault.kind() {
                PcuExecutionFaultKind::ArithmeticOverflow => {
                    format.max_finite | if source.is_sign_negative() { 0x80 } else { 0 }
                }
                PcuExecutionFaultKind::ArithmeticUnderflow => round_value(source, format).unwrap(),
                other => panic!("unexpected recovery kind {other:?}"),
            };
            assert_eq!(fault.clamped_value(), recovery);
        }
    }
}

#[test]
fn every_fp8_midpoint_signed_neighbor_and_encoding_converts_once() {
    for format in [E4M3FN, E5M2] {
        for lower in 0..format.max_finite {
            let midpoint = value(lower, format).midpoint(value(lower + 1, format));
            for source in [
                value(lower, format),
                midpoint.next_down(),
                midpoint,
                midpoint.next_up(),
            ] {
                compare(source, format);
                compare(-source, format);
            }
        }
        let max = value(format.max_finite, format);
        let threshold = max + (max - value(format.max_finite - 1, format)) / 2.0;
        for source in [
            max,
            threshold.next_down(),
            threshold,
            threshold.next_up(),
            f64::MAX,
            f64::MIN_POSITIVE,
            f64::from_bits(1),
            0.0,
            f64::INFINITY,
            f64::NAN,
        ] {
            compare(source, format);
            compare(-source, format);
        }
    }
}

#[test]
fn random_full_exponent_binary64_and_binary32_narrow_without_intermediate_rounding() {
    let mut state = 0x12fe_e4e5_900d_7013_u64;
    for _ in 0..100_000 {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let source = f64::from_bits(state);
        for format in [E4M3FN, E5M2] {
            compare(source, format);
        }
        let bytes = state.to_le_bytes();
        let source32 = f32::from_bits(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]));
        assert_eq!(
            PcuF8E4M3FnBits::pcu_checked_from_f32(source32).map(|bits| u16::from(bits.to_bits())),
            expected(
                f64::from(source32),
                E4M3FN,
                PcuFloatUnderflowPolicy::default()
            )
        );
        assert_eq!(
            PcuF8E5M2Bits::pcu_checked_from_f32(source32).map(|bits| u16::from(bits.to_bits())),
            expected(
                f64::from(source32),
                E5M2,
                PcuFloatUnderflowPolicy::default()
            )
        );
    }
}

#[test]
fn every_fp8_encoding_widens_exactly_and_rejects_nonfinite_checked_inputs() {
    for bits in 0..=u8::MAX {
        for (format, wide32, checked32, checked64) in [
            (
                E4M3FN,
                PcuF8E4M3FnBits::from_bits(bits).to_f32(),
                PcuF8E4M3FnBits::from_bits(bits).pcu_checked_to_f32(),
                PcuF8E4M3FnBits::from_bits(bits).pcu_checked_to_f64(),
            ),
            (
                E5M2,
                PcuF8E5M2Bits::from_bits(bits).to_f32(),
                PcuF8E5M2Bits::from_bits(bits).pcu_checked_to_f32(),
                PcuF8E5M2Bits::from_bits(bits).pcu_checked_to_f64(),
            ),
        ] {
            let exact = value(u16::from(bits), format);
            if exact.is_finite() {
                #[allow(clippy::cast_possible_truncation)]
                // Every finite FP8 value is exactly representable in f32.
                let expected32 = exact as f32;
                assert_eq!(wide32.to_bits(), expected32.to_bits());
                assert_eq!(checked32.unwrap().to_bits(), expected32.to_bits());
                assert_eq!(checked64.unwrap().to_bits(), exact.to_bits());
            } else {
                assert_eq!(
                    checked32,
                    Err(PcuExecutionFaultKind::InvalidFloatingOperand)
                );
                assert_eq!(
                    checked64,
                    Err(PcuExecutionFaultKind::InvalidFloatingOperand)
                );
            }
        }
    }
    assert_eq!(
        PcuF8E5M2Bits::from_bits(0x7c).to_f32().to_bits(),
        f32::INFINITY.to_bits()
    );
    assert_eq!(
        PcuF8E5M2Bits::from_bits(0x7d).to_f32().to_bits(),
        0x7fa0_0000
    );
    assert_eq!(
        PcuF8E4M3FnBits::from_bits(0xff).to_f32().to_bits(),
        0xfff0_0000
    );
    assert_eq!(
        generic_identity(
            PcuF8E4M3FnBits::from_bits(0x38),
            PcuF8E4M3FnBits::from_bits(0)
        ),
        Ok(PcuF8E4M3FnBits::from_bits(0x38))
    );
    assert_eq!(
        generic_identity(PcuF8E5M2Bits::from_bits(0x3c), PcuF8E5M2Bits::from_bits(0)),
        Ok(PcuF8E5M2Bits::from_bits(0x3c))
    );
}

fn generic_identity<T: PcuCheckedFloat>(one: T, zero: T) -> Result<T, PcuExecutionFaultKind> {
    one.pcu_checked_add(zero)
}
