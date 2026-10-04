//! Checked casts share mixed-width schemas, transactional faults and IEEE rounding.
#[rustfmt::skip]
use fusion_pcu::{
    global,
    pcu,
    PcuCompoundArithmeticPolicy,
    PcuExecutionError,
    PcuExecutionFaultKind,
    PcuFloatUnderflowPolicy,
    PcuNumericalMode,
    PcuNumericalOptions,
    PcuPrecisionPolicy,
};
#[rustfmt::skip]
use super::support::{
    bits,
    POLICY_LOCK,
};

#[cfg(all(feature = "tensor", any(feature = "metal", feature = "mlx")))]
#[path = "mixed/mixed.rs"]
mod mixed;

#[pcu(invocations = N)]
fn narrow<const N: usize>(input: &[f64; N], output: &mut [f32]) {
    let id = pcu::context::global_invocation_id();
    output[id] = input[id] as f32;
}
#[pcu(invocations = N)]
fn widen<const N: usize>(input: &[f32; N], output: &mut [f64]) {
    let id = pcu::context::global_invocation_id();
    output[id] = input[id] as f64;
}
#[pcu(invocations = N)]
fn broadcast<const N: usize>(input: &f64, output: &mut [f32]) {
    let id = pcu::context::global_invocation_id();
    output[id] = *input as f32;
}
#[pcu(invocations = 3)]
fn grid(input: &[f64; 7], output: &mut [f32]) {
    let mut id = pcu::context::global_invocation_id();
    let stride = pcu::context::invocation_count();
    while id < 7 {
        output[id] = input[id] as f32;
        id += stride;
    }
}
#[pcu(invocations = N, flag(clamp_range))]
fn clamp<const N: usize>(input: &[f64; N], output: &mut [f32]) {
    let id = pcu::context::global_invocation_id();
    output[id] = input[id] as f32;
}

fn fault(result: Result<(), PcuExecutionError>, kind: PcuExecutionFaultKind, recovered: bool) {
    let error = result.unwrap_err();
    let actual = error
        .arithmetic_fault()
        .unwrap_or_else(|| panic!("expected {kind:?}: {error:?}"));
    assert_eq!(actual.kind, kind);
    assert_eq!(actual.invocation_id, 2);
    assert_eq!(actual.recovered, recovered);
}

fn finite_rounding() {
    // IEEE 754-2019 nearest-even: exact half-ULPs go to the even significand.
    // Direct expected encodings avoid using the same host cast as an oracle.
    let input = [
        0.0,
        -0.0,
        f64::from_bits(0x3ff0_0000_1000_0000), // 1 + 2^-24.
        f64::from_bits(0x3ff0_0000_3000_0000), // 1 + 3*2^-24.
        -1.5,
        2.0,
        4.0,
    ];
    let expected = [0.0, -0.0, 1.0, f32::from_bits(0x3f80_0002), -1.5, 2.0, 4.0];
    let mut output = [23.0_f32; 9];
    narrow(&input, &mut output).unwrap();
    bits(&output[..7], &expected);
    bits(&output[7..], &[23.0; 2]);
    grid(&input, &mut output).unwrap();
    bits(&output[..7], &expected);
    broadcast::<7>(&-0.0, &mut output).unwrap();
    bits(&output[..7], &[-0.0; 7]);
    bits(&output[7..], &[23.0; 2]);
    let mut widened = [29.0_f64; 9];
    widen(&expected, &mut widened).unwrap();
    bits(&widened[..7], &expected.map(f64::from));
    bits(&widened[7..], &[29.0; 2]);
    for invalid in [f32::INFINITY, f32::NAN] {
        let mut input = expected;
        input[2] = invalid;
        let before = widened;
        fault(
            widen(&input, &mut widened),
            PcuExecutionFaultKind::InvalidFloatingOperand,
            false,
        );
        bits(&widened, &before);
    }
}

fn exceptional_ranges(policy: PcuFloatUnderflowPolicy) {
    let tiny = f64::from(f32::from_bits(1));
    let mut output = [19.0_f32; 9];
    let mut input = [1.0_f64; 7];
    input[2] = tiny;
    input[6] = tiny;
    if policy == PcuFloatUnderflowPolicy::RejectSubnormalResult {
        fault(
            narrow(&input, &mut output),
            PcuExecutionFaultKind::ArithmeticUnderflow,
            false,
        );
        bits(&output, &[19.0; 9]);
    } else {
        narrow(&input, &mut output).unwrap();
        let mut expected = [1.0; 7];
        expected[2] = f32::from_bits(1);
        expected[6] = f32::from_bits(1);
        bits(&output[..7], &expected);
    }
    let before = output;
    input[2] = tiny * 0.5;
    input[6] = -tiny * 0.5;
    if policy == PcuFloatUnderflowPolicy::AllowGradualUnderflow {
        narrow(&input, &mut output).unwrap();
        bits(&output[2..3], &[0.0]);
        bits(&output[6..7], &[-0.0]);
    } else {
        fault(
            narrow(&input, &mut output),
            PcuExecutionFaultKind::ArithmeticUnderflow,
            false,
        );
        bits(&output, &before);
    }
    input[2] = f64::MAX;
    input[6] = -f64::MAX;
    let before = output;
    fault(
        narrow(&input, &mut output),
        PcuExecutionFaultKind::ArithmeticOverflow,
        false,
    );
    bits(&output, &before);
    fault(
        clamp(&input, &mut output),
        PcuExecutionFaultKind::ArithmeticOverflow,
        true,
    );
    bits(&output[2..3], &[f32::MAX]);
    bits(&output[6..7], &[-f32::MAX]);
    bits(&output[7..], &[19.0; 2]);
    // A later fatal operand wins over recoverable overflow and forbids publication.
    input[6] = f64::INFINITY;
    let before = output;
    let error = clamp(&input, &mut output).unwrap_err();
    let actual = error.arithmetic_fault().unwrap();
    assert_eq!(actual.kind, PcuExecutionFaultKind::InvalidFloatingOperand);
    assert_eq!(actual.invocation_id, 6);
    assert!(!actual.recovered);
    bits(&output, &before);
    narrow(&[2.0; 7], &mut output).unwrap();
    bits(&output[..7], &[2.0; 7]);
}

pub fn verify(backend: global::PcuBackendChoice) {
    let _guard = POLICY_LOCK.lock().unwrap();
    for numerical_mode in [PcuNumericalMode::Boundary, PcuNumericalMode::Strict] {
        for compound_arithmetic in [
            PcuCompoundArithmeticPolicy::Checked,
            PcuCompoundArithmeticPolicy::BackendDefined,
        ] {
            for precision in [
                PcuPrecisionPolicy::Preserve,
                PcuPrecisionPolicy::BackendOptimized,
            ] {
                for float_underflow in [
                    PcuFloatUnderflowPolicy::IeeeAfterRounding,
                    PcuFloatUnderflowPolicy::AllowGradualUnderflow,
                    PcuFloatUnderflowPolicy::RejectSubnormalResult,
                ] {
                    global::configure(global::PcuExecutionPolicy {
                        backend,
                        numerical_mode,
                        float_underflow,
                        numerical_options: PcuNumericalOptions {
                            compound_arithmetic,
                            precision,
                            ..Default::default()
                        },
                        ..Default::default()
                    })
                    .unwrap();
                    global::clear_thread_cache().unwrap();
                    finite_rounding();
                    exceptional_ranges(float_underflow);
                    #[cfg(all(feature = "tensor", any(feature = "metal", feature = "mlx")))]
                    mixed::verify(backend);
                }
            }
        }
    }
    global::clear_thread_cache().unwrap();
    global::use_defaults().unwrap();
}
