//! Sign-bit oracles check unary policy parity without borrowing provider arithmetic.
use core::sync::atomic::AtomicUsize;
use core::sync::atomic::Ordering;

#[rustfmt::skip]
use fusion_pcu::{
    global,
    pcu,
    PcuBf16Bits,
    PcuCheckedFloat,
    PcuExecutionFaultKind,
    PcuF16Bits,
    PcuF8E4M3FnBits,
    PcuF8E5M2Bits,
    PcuFloatUnderflowPolicy,
    PcuNumericalMode,
    PcuNumericalOptions,
    PcuRangePolicy,
};

#[path = "portable/portable.rs"]
pub mod portable;

#[path = "roles/roles.rs"]
pub mod roles;

#[cfg(feature = "tensor")]
#[path = "prefix/prefix.rs"]
pub mod prefix;

#[rustfmt::skip]
use super::support::{
    bits,
    POLICY_LOCK,
};

#[pcu(invocations = N)]
fn neg<T: PcuCheckedFloat, const N: usize>(input: &[T], output: &mut [T]) {
    let id = pcu::context::global_invocation_id();
    output[id] = -input[id];
}

#[pcu(invocations = N)]
fn relu<T: PcuCheckedFloat, const N: usize>(input: &[T], output: &mut [T]) {
    let id = pcu::context::global_invocation_id();
    output[id] = pcu::relu(input[id]);
}

#[pcu(invocations = N, flag(clamp_range), flag(reject_subnormal_result))]
fn local_clamp<T: PcuCheckedFloat, const N: usize>(input: &[T], output: &mut [T]) {
    let id = pcu::context::global_invocation_id();
    output[id] = -input[id];
}

trait Sample: PcuCheckedFloat {
    const SIGN: u64;
    const NORMAL: u64;
    const NONFINITE: u64;
    fn from_raw(value: u64) -> Self;
}

macro_rules! sample {
    ($ty:ty, $bits:ty, $sign:expr, $normal:expr, $nonfinite:expr) => {
        impl Sample for $ty {
            const SIGN: u64 = $sign;
            const NORMAL: u64 = $normal;
            const NONFINITE: u64 = $nonfinite;
            fn from_raw(value: u64) -> Self {
                Self::from_bits(<$bits>::try_from(value).unwrap())
            }
        }
    };
}

sample!(PcuF16Bits, u16, 0x8000, 0x0400, 0x7c00);
sample!(PcuBf16Bits, u16, 0x8000, 0x0080, 0x7f80);
sample!(PcuF8E4M3FnBits, u8, 0x80, 0x08, 0x7f);
sample!(PcuF8E5M2Bits, u8, 0x80, 0x04, 0x7c);
sample!(f32, u32, 0x8000_0000, 0x0080_0000, 0x7f80_0000);
sample!(
    f64,
    u64,
    0x8000_0000_0000_0000,
    0x0010_0000_0000_0000,
    0x7ff0_0000_0000_0000
);

static SCORES: AtomicUsize = AtomicUsize::new(0);

fn score(_: &global::PcuInvocationCandidate<'_>) -> i128 {
    SCORES.fetch_add(1, Ordering::Relaxed);
    1
}

fn outcome(
    result: Result<(), global::PcuExecutionError>,
    underflow: PcuFloatUnderflowPolicy,
    range: PcuRangePolicy,
) -> bool {
    if underflow != PcuFloatUnderflowPolicy::RejectSubnormalResult {
        result.unwrap();
        return true;
    }
    let error = result.unwrap_err();
    assert!(
        matches!(error, global::PcuExecutionError::ArithmeticFault(fault)
        if fault.kind == PcuExecutionFaultKind::ArithmeticUnderflow
            && fault.invocation_id == 2 && fault.recovered == (range == PcuRangePolicy::Clamp))
    );
    range == PcuRangePolicy::Clamp
}

type Execute<T> = fn(&[T], &mut [T]) -> Result<(), global::PcuExecutionError>;

fn exercise<T: Sample>(
    underflow: PcuFloatUnderflowPolicy,
    range: PcuRangePolicy,
    negate: Execute<T>,
    activate: Execute<T>,
) {
    let sentinel = T::from_raw(T::NORMAL);
    for phase in 0..3 {
        // Raw encoding input changes on every replay, including signed exact subnormals.
        let raw = [
            0,
            T::SIGN,
            1 + phase,
            T::SIGN | (1 + phase),
            T::NORMAL + phase,
            T::SIGN | (T::NORMAL + phase),
            2 + phase,
        ];
        let input = raw.map(T::from_raw);
        let mut output = [sentinel; 9];
        let published = outcome(negate(&input, &mut output), underflow, range);
        if published {
            bits(&output[..7], &raw.map(|v| T::from_raw(v ^ T::SIGN)));
        } else {
            bits(&output[..7], &[sentinel; 7]);
        }
        bits(&output[7..], &[sentinel; 2]);
        output.fill(sentinel);
        let published = outcome(activate(&input, &mut output), underflow, range);
        if published {
            bits(
                &output[..7],
                &raw.map(|v| T::from_raw(if v & T::SIGN == 0 { v } else { 0 })),
            );
        } else {
            bits(&output[..7], &[sentinel; 7]);
        }
        bits(&output[7..], &[sentinel; 2]);

        // An inactive negative subnormal produces +0 and must not trigger tight underflow.
        let inactive = [T::from_raw(T::SIGN | (1 + phase)); 7];
        activate(&inactive, &mut output).unwrap();
        bits(&output[..7], &[T::from_raw(0); 7]);

        // Fatal invalid input outranks the earlier recovered range lane. Never publish a prefix.
        let mut invalid = input;
        invalid[4] = T::from_raw(T::NONFINITE);
        output.fill(sentinel);
        for execute in [negate, activate] {
            let error = execute(&invalid, &mut output).unwrap_err();
            let expected_id = if underflow == PcuFloatUnderflowPolicy::RejectSubnormalResult
                && range == PcuRangePolicy::Reject
            {
                2
            } else {
                4
            };
            let expected_kind = if expected_id == 2 {
                PcuExecutionFaultKind::ArithmeticUnderflow
            } else {
                PcuExecutionFaultKind::InvalidFloatingOperand
            };
            assert!(
                matches!(error, global::PcuExecutionError::ArithmeticFault(fault)
                if fault.kind == expected_kind && fault.invocation_id == expected_id
                    && !fault.recovered)
            );
            bits(&output, &[sentinel; 9]);
        }
        // Reuse the exact prepared source entry after failure, with fresh values and no rescore.
        outcome(negate(&input, &mut output), underflow, range);
    }
}

fn format<T: Sample>(underflow: PcuFloatUnderflowPolicy, range: PcuRangePolicy) {
    exercise(underflow, range, neg::<T, 7>, relu::<T, 7>);
}

fn all_formats(underflow: PcuFloatUnderflowPolicy, range: PcuRangePolicy) {
    format::<PcuF16Bits>(underflow, range);
    format::<PcuBf16Bits>(underflow, range);
    format::<PcuF8E4M3FnBits>(underflow, range);
    format::<PcuF8E5M2Bits>(underflow, range);
    format::<f32>(underflow, range);
    format::<f64>(underflow, range);
}

pub fn verify(backend: global::PcuBackendChoice) {
    verify_options(backend, PcuNumericalOptions::default());
}

fn verify_options(backend: global::PcuBackendChoice, numerical_options: PcuNumericalOptions) {
    let _guard = POLICY_LOCK.lock().unwrap();
    for numerical_mode in [PcuNumericalMode::Boundary, PcuNumericalMode::Strict] {
        for float_underflow in [
            PcuFloatUnderflowPolicy::IeeeAfterRounding,
            PcuFloatUnderflowPolicy::AllowGradualUnderflow,
            PcuFloatUnderflowPolicy::RejectSubnormalResult,
        ] {
            for range_policy in [PcuRangePolicy::Reject, PcuRangePolicy::Clamp] {
                global::configure(global::PcuExecutionPolicy {
                    backend,
                    numerical_mode,
                    float_underflow,
                    range_policy,
                    numerical_options,
                    score_invocation: Some(score),
                    ..Default::default()
                })
                .unwrap();
                global::clear_thread_cache().unwrap();
                all_formats(float_underflow, range_policy);
                let cold = SCORES.load(Ordering::Relaxed);
                all_formats(float_underflow, range_policy);
                assert_eq!(SCORES.load(Ordering::Relaxed), cold);
            }
        }
    }
    global::configure(global::PcuExecutionPolicy {
        backend,
        numerical_options,
        ..Default::default()
    })
    .unwrap();
    // Explicit local flags override these default globals without changing an inherited entry.
    let input = [f32::from_bits(1), 2.0];
    let mut output = [17.0; 2];
    let error = local_clamp::<f32, 2>(&input, &mut output).unwrap_err();
    assert!(error.recovered_range_fault().unwrap().recovered);
    bits(&output, &[-f32::from_bits(1), -2.0]);
    neg::<f32, 2>(&input, &mut output).unwrap();
    bits(&output, &[-f32::from_bits(1), -2.0]);
    global::clear_thread_cache().unwrap();
    global::use_defaults().unwrap();
}
