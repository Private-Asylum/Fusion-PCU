//! Explicit CPU feature participates in the ordinary per-function source call spine.
#![cfg(feature = "cpu")]

use core::sync::atomic::AtomicUsize;
use core::sync::atomic::Ordering;
#[rustfmt::skip]
use fusion_pcu::{
    global,
    pcu,
    PcuDeviceClass,
    PcuExecutionFault,
    PcuExecutionFaultKind,
    PcuNumericalMode,
};

static SCORES: AtomicUsize = AtomicUsize::new(0);
fn score(candidate: &global::PcuInvocationCandidate<'_>) -> i128 {
    assert_eq!(candidate.device.class, PcuDeviceClass::Cpu);
    assert_eq!(candidate.device.reference.provider.0, 0x4350_5531);
    assert_eq!(candidate.total_memory_bytes, None);
    assert_eq!(candidate.facts.compute_unit_count, None);
    SCORES.fetch_add(1, Ordering::Relaxed);
    1
}

macro_rules! source_cases {
    ($($module:ident : $ty:ty),* $(,)?) => { $(
        mod $module {
            use fusion_pcu::pcu;
            #[pcu(invocations = N)]
            pub fn add<const N: usize>(lhs: &[$ty; N], rhs: &[$ty; N], output: &mut [$ty]) {
                let id = pcu::context::global_invocation_id();
                output[id] = lhs[id] + rhs[id];
            }
            #[pcu(invocations = N)]
            pub fn sub<const N: usize>(lhs: &[$ty; N], rhs: &[$ty; N], output: &mut [$ty]) {
                let id = pcu::context::global_invocation_id();
                output[id] = lhs[id] - rhs[id];
            }
            #[pcu(invocations = N)]
            pub fn mul<const N: usize>(lhs: &[$ty; N], rhs: &[$ty; N], output: &mut [$ty]) {
                let id = pcu::context::global_invocation_id();
                output[id] = lhs[id] * rhs[id];
            }
            pub fn verify() {
                let mut output: [$ty; 20] = [19; 20];
                for phase in [1, 2, 3] {
                    let lhs: [$ty; 17] = [6 + phase; 17];
                    let rhs: [$ty; 17] = [2; 17];
                    add(&lhs, &rhs, &mut output).unwrap();
                    assert_eq!(output[..17], [lhs[0] + rhs[0]; 17]);
                    sub(&lhs, &rhs, &mut output).unwrap();
                    assert_eq!(output[..17], [lhs[0] - rhs[0]; 17]);
                    mul(&lhs, &rhs, &mut output).unwrap();
                    assert_eq!(output[..17], [lhs[0] * rhs[0]; 17]);
                    assert_eq!(output[17..], [19; 3]);
                }
            }
        }
    )* };
}
source_cases!(s8:i8, u8_case:u8, s16:i16, u16_case:u16, s32:i32, u32_case:u32, s64:i64, u64_case:u64);

#[pcu(invocations = N)]
fn negate<const N: usize>(input: &[f32; N], output: &mut [f32; N]) {
    let id = pcu::context::global_invocation_id();
    output[id] = -input[id];
}

#[pcu(invocations = N)]
fn negate_f64<const N: usize>(input: &[f64; N], output: &mut [f64]) {
    let id = pcu::context::global_invocation_id();
    output[id] = -input[id];
}

fn configure() {
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Cpu,
        numerical_mode: PcuNumericalMode::Strict,
        score_invocation: Some(score),
        ..Default::default()
    })
    .unwrap();
}

#[test]
fn ordinary_cpu_source_reuses_selection_and_preserves_fault_outputs() {
    configure();
    global::clear_thread_cache().unwrap();
    SCORES.store(0, Ordering::Relaxed);
    for verify in [
        s8::verify,
        u8_case::verify,
        s16::verify,
        u16_case::verify,
        s32::verify,
        u32_case::verify,
        s64::verify,
        u64_case::verify,
    ] {
        verify();
    }
    let cold = SCORES.load(Ordering::Relaxed);
    assert_eq!(cold, 24);
    for verify in [
        s8::verify,
        u8_case::verify,
        s16::verify,
        u16_case::verify,
        s32::verify,
        u32_case::verify,
        s64::verify,
        u64_case::verify,
    ] {
        verify();
    }
    assert_eq!(SCORES.load(Ordering::Relaxed), cold);
    let input = [f32::from_bits(1), 0.0, -0.0, 1.0, -2.0];
    let mut output = [0.0_f32; 5];
    negate(&input, &mut output).unwrap();
    for (value, actual) in input.iter().zip(output) {
        assert_eq!(actual.to_bits(), value.to_bits() ^ 0x8000_0000);
    }
    let mut input = input;
    input[2] = f32::NAN;
    let previous = output.map(f32::to_bits);
    assert!(matches!(
        negate(&input, &mut output),
        Err(global::PcuExecutionError::ArithmeticFault(
            PcuExecutionFault {
                recovered: false,
                kind: PcuExecutionFaultKind::InvalidFloatingOperand,
                invocation_id: 2
            }
        ))
    ));
    assert_eq!(output.map(f32::to_bits), previous);
    binary64_source();
    integer_faults();
    let before = SCORES.load(Ordering::Relaxed);
    configure(); // No explicit cache clear: the generation itself must invalidate selection.
    s8::verify();
    assert_eq!(SCORES.load(Ordering::Relaxed), before + 3);
    global::clear_thread_cache().unwrap();
    global::use_defaults().unwrap();
}

fn binary64_source() {
    let mut input = [1.0_f64; 17];
    input[..5].copy_from_slice(&[f64::from_bits(1), 0.0, -0.0, f64::MAX, -2.0]);
    let mut output = [23.0_f64; 20];
    negate_f64(&input, &mut output).unwrap();
    for (input, output) in input.iter().zip(&output) {
        assert_eq!(output.to_bits(), input.to_bits() ^ 0x8000_0000_0000_0000);
    }
    assert_eq!(
        output[17..].iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        [23.0_f64.to_bits(); 3]
    );
    let cold = SCORES.load(Ordering::Relaxed);
    let previous = output.map(f64::to_bits);
    input[3] = f64::NAN;
    assert!(matches!(
        negate_f64(&input, &mut output),
        Err(global::PcuExecutionError::ArithmeticFault(
            PcuExecutionFault {
                recovered: false,
                kind: PcuExecutionFaultKind::InvalidFloatingOperand,
                invocation_id: 3
            }
        ))
    ));
    assert_eq!(output.map(f64::to_bits), previous);
    input[3] = 3.0;
    negate_f64(&input, &mut output).unwrap();
    assert_eq!(output[3].to_bits(), (-3.0_f64).to_bits());
    assert_eq!(SCORES.load(Ordering::Relaxed), cold);
}

fn integer_faults() {
    let mut lhs = [1_u64; 17];
    lhs[3] = u64::MAX;
    lhs[16] = u64::MAX;
    let rhs = [1_u64; 17];
    let mut output = [29_u64; 20];
    assert!(matches!(
        u64_case::add(&lhs, &rhs, &mut output),
        Err(global::PcuExecutionError::ArithmeticFault(
            PcuExecutionFault {
                recovered: false,
                kind: PcuExecutionFaultKind::ArithmeticOverflow,
                invocation_id: 3
            }
        ))
    ));
    assert_eq!(output, [29; 20]);
    lhs.fill(2);
    u64_case::add(&lhs, &rhs, &mut output).unwrap();
    assert_eq!(output[..17], [3; 17]);
    assert_eq!(output[17..], [29; 3]);
    let mut output = [31_u8; 20];
    assert!(matches!(
        u8_case::sub(&[0; 17], &[1; 17], &mut output),
        Err(global::PcuExecutionError::ArithmeticFault(
            PcuExecutionFault {
                recovered: false,
                kind: PcuExecutionFaultKind::ArithmeticUnderflow,
                invocation_id: 0
            }
        ))
    ));
    assert_eq!(output, [31; 20]);
}
