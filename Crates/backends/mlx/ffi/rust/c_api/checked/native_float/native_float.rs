//! Independent raw encoding unary oracle and core cross-check through actual MLX siblings.
#[rustfmt::skip]
use fusion_pcu::{
    PcuScalarType,
    PcuDispatchFloatUnaryOp as Op,
    PcuFloatUnderflowPolicy as Policy,
    PcuRangePolicy as Range,
    PcuExecutionFaultKind as Kind,
    PcuClampedFloat,
    PcuClampedError,
};
use super::super::CheckedUnary;
fn expected(
    bits: u64,
    sign: u64,
    fraction: u32,
    maximum: u64,
    op: Op,
    policy: Policy,
    range: Range,
) -> (u64, u32) {
    let magnitude = bits & (sign - 1);
    if magnitude > maximum {
        return (0, 4);
    }
    let result = match op {
        Op::Neg => bits ^ sign,
        Op::Relu if bits & sign != 0 || magnitude == 0 => 0,
        Op::Relu => bits,
    };
    let magnitude = result & (sign - 1);
    if policy == Policy::RejectSubnormalResult && magnitude != 0 && magnitude < 1 << fraction {
        if range == Range::Reject {
            (0, 3)
        } else {
            (result, 0x103)
        }
    } else {
        (result, 0)
    }
}
fn core<T: PcuClampedFloat>(
    value: T,
    bits: impl Fn(T) -> u64,
    op: Op,
    policy: Policy,
    range: Range,
) -> (u64, u32) {
    let result = match op {
        Op::Neg => value.pcu_clamped_neg_with_policy(policy),
        Op::Relu => value.pcu_clamped_relu_with_policy(policy),
    };
    match result {
        Ok(value) => (bits(value), 0),
        Err(PcuClampedError::Range(fault)) => {
            assert_eq!(fault.kind(), Kind::ArithmeticUnderflow);
            if range == Range::Reject {
                (0, 3)
            } else {
                (bits(fault.clamped_value()), 0x103)
            }
        }
        Err(PcuClampedError::Fatal(kind)) => {
            assert_eq!(kind, Kind::InvalidFloatingOperand);
            (0, 4)
        }
    }
}
fn samples(sign: u64, fraction: u32, maximum: u64) -> Vec<u64> {
    let mask = sign | (sign - 1);
    let mut result = vec![
        0,
        1,
        2,
        3,
        (1 << fraction) - 1,
        1 << fraction,
        (1 << fraction) + 1,
        maximum,
        maximum - 1,
        maximum + 1,
        sign - 1,
        sign - (1 << fraction),
        1 << (fraction + 1),
    ];
    result.extend(result.clone().into_iter().map(|bits| bits | sign));
    let mut random = 0x7261_7762_6974_7365_u64;
    while result.len() < 131_584 {
        random ^= random << 13;
        random ^= random >> 7;
        random ^= random << 17;
        result.push(random & mask);
    }
    result
}
fn qualify(
    session: &super::super::super::session::Session,
    scalar: PcuScalarType,
    sign: u64,
    fraction: u32,
    maximum: u64,
) {
    let values = samples(sign, fraction, maximum);
    let width = usize::from(scalar.bit_width()) / 8;
    let bytes: Vec<_> = values
        .iter()
        .flat_map(|bits| bits.to_le_bytes().into_iter().take(width))
        .collect();
    for op in [Op::Neg, Op::Relu] {
        for policy in [
            Policy::IeeeAfterRounding,
            Policy::RejectSubnormalResult,
            Policy::AllowGradualUnderflow,
        ] {
            for range in [Range::Reject, Range::Clamp] {
                let map =
                    CheckedUnary::prepare(session, scalar, op, policy, range, values.len(), false)
                        .unwrap();
                let (payload, status) = super::diagnostics(&map, &bytes);
                for (index, &input) in values.iter().enumerate() {
                    let wanted = expected(input, sign, fraction, maximum, op, policy, range);
                    let shared = if scalar == PcuScalarType::F32 {
                        core(
                            f32::from_bits(u32::try_from(input).unwrap()),
                            |value| u64::from(value.to_bits()),
                            op,
                            policy,
                            range,
                        )
                    } else {
                        core(f64::from_bits(input), f64::to_bits, op, policy, range)
                    };
                    assert_eq!(
                        shared, wanted,
                        "core {scalar:?}/{op:?}/{policy:?}/{range:?}/{input:x}"
                    );
                    let mut actual = [0; 8];
                    actual[..width].copy_from_slice(&payload[index * width..(index + 1) * width]);
                    assert_eq!(
                        (u64::from_le_bytes(actual), status[index]),
                        wanted,
                        "MLX {scalar:?}/{op:?}/{policy:?}/{range:?}/{input:x}"
                    );
                }
                // One actual input carrier is repeated into a distinct 1024-lane output.
                for input in [
                    0,
                    sign,
                    1,
                    sign | 1,
                    1 << fraction,
                    maximum,
                    maximum + 1,
                    sign - 1,
                ] {
                    let map = CheckedUnary::prepare(session, scalar, op, policy, range, 1024, true)
                        .unwrap();
                    let (payload, status) = super::diagnostics(&map, &input.to_le_bytes()[..width]);
                    let wanted = expected(input, sign, fraction, maximum, op, policy, range);
                    for (index, record) in status.iter().enumerate() {
                        let mut actual = [0; 8];
                        actual[..width]
                            .copy_from_slice(&payload[index * width..(index + 1) * width]);
                        assert_eq!((u64::from_le_bytes(actual), *record), wanted);
                    }
                }
            }
        }
    }
}
#[test]
#[ignore = "Requires actual pinned MLX F32/F64 integer-only unary kernels and sibling completion."]
fn f32_f64_native_unary_bit_and_fault_oracle() {
    let api = super::super::super::api::Api::load_default().unwrap();
    let session = api.open(0).unwrap();
    qualify(&session, PcuScalarType::F32, 0x8000_0000, 23, 0x7f7f_ffff);
    qualify(
        &session,
        PcuScalarType::F64,
        0x8000_0000_0000_0000,
        52,
        0x7fef_ffff_ffff_ffff,
    );
}
