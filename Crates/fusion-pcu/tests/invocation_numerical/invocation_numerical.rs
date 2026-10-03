//! The ordinary source boundary freezes independent policies; warm calls do not rescore.
use core::sync::atomic::AtomicUsize;
use core::sync::atomic::Ordering;
use fusion_pcu::pcu;
#[rustfmt::skip]
use fusion_pcu::{
    PcuCompoundArithmeticPolicy,
    PcuFloatUnderflowPolicy,
    PcuImplementationRequirements,
    PcuNumericalMode,
    PcuNumericalOptions,
    PcuPrecisionPolicy,
    PcuRangePolicy,
    PcuReproducibility,
};
#[rustfmt::skip]
use fusion_pcu::global::{
    configure,
    PcuBackendChoice,
    PcuExecutionError,
    PcuExecutionPolicy,
    PcuInvocationCandidate,
};

#[pcu(invocations = 3)]
fn inherited(left: &[f32], right: &[f32], output: &mut [f32]) {
    let id = context.global_invocation_id;
    output[id] = left[id] * right[id];
}
#[pcu(
    invocations = 3,
    flag(strict),
    flag(native_compound),
    flag(backend_precision),
    flag(non_deterministic)
)]
fn permitted(left: &[f32], right: &[f32], output: &mut [f32]) {
    let id = context.global_invocation_id;
    output[id] = left[id] * right[id];
}
#[pcu(
    invocations = 3,
    flag(non_strict),
    flag(checked_compound),
    flag(preserve_precision),
    flag(non_deterministic),
    flag(ieee_underflow)
)]
fn overrides(left: &[f32], right: &[f32], output: &mut [f32]) {
    let id = context.global_invocation_id;
    output[id] = left[id] * right[id];
}
#[pcu(invocations = 3, flag(deterministic))]
fn portable(left: &[f32], right: &[f32], output: &mut [f32]) {
    let id = context.global_invocation_id;
    output[id] = left[id] * right[id];
}
static SCORES: AtomicUsize = AtomicUsize::new(0);
static EXPECTED: AtomicUsize = AtomicUsize::new(0);
fn score(candidate: &PcuInvocationCandidate<'_>) -> i128 {
    let expected = match EXPECTED.load(Ordering::Relaxed) {
        0 => PcuImplementationRequirements::DEFAULT,
        1 => PcuImplementationRequirements {
            numerical_mode: PcuNumericalMode::Strict,
            numerical_options: PcuNumericalOptions {
                compound_arithmetic: PcuCompoundArithmeticPolicy::BackendDefined,
                precision: PcuPrecisionPolicy::BackendOptimized,
                reproducibility: PcuReproducibility::Unspecified,
            },
            ..PcuImplementationRequirements::DEFAULT
        },
        2 => PcuImplementationRequirements {
            numerical_mode: PcuNumericalMode::Strict,
            float_underflow: PcuFloatUnderflowPolicy::RejectSubnormalResult,
            ..PcuImplementationRequirements::DEFAULT
        },
        _ => panic!("unexpected policy fixture"),
    };
    assert_eq!(candidate.kernel.numerical_requirements, expected);
    SCORES.fetch_add(1, Ordering::Relaxed);
    1
}
fn policy() -> PcuExecutionPolicy {
    PcuExecutionPolicy {
        backend: PcuBackendChoice::Cpu,
        score_invocation: Some(score),
        ..PcuExecutionPolicy::default()
    }
}

#[test]
fn source_scopes_cache_and_protection_preserve_the_complete_tuple() {
    // The IR companion is independently inspectable and includes the explicit function flags.
    let bindings = permitted_bindings();
    let builder = permitted_ir(&bindings).unwrap();
    let requirements = builder.ir().numerical_requirements;
    assert_eq!(requirements.numerical_mode, PcuNumericalMode::Strict);
    assert_eq!(
        requirements.numerical_options.compound_arithmetic,
        PcuCompoundArithmeticPolicy::BackendDefined
    );
    assert_eq!(
        requirements.numerical_options.precision,
        PcuPrecisionPolicy::BackendOptimized
    );
    assert_eq!(requirements.range_policy, PcuRangePolicy::Reject);
    configure(policy()).unwrap();
    let mut output = [99.0; 5];
    inherited(&[1.0; 3], &[2.0; 3], &mut output).unwrap();
    assert_eq!(
        output.map(f32::to_bits),
        [2.0_f32, 2.0, 2.0, 99.0, 99.0].map(f32::to_bits)
    );
    let cold = SCORES.load(Ordering::Relaxed);
    assert!(cold > 0);
    inherited(&[3.0; 3], &[2.0; 3], &mut output).unwrap();
    assert_eq!(SCORES.load(Ordering::Relaxed), cold);

    EXPECTED.store(1, Ordering::Relaxed);
    permitted(&[1.0; 3], &[2.0; 3], &mut output).unwrap();
    let previous = output.map(f32::to_bits);
    let fault = permitted(&[1.0, f32::MAX, 1.0], &[2.0; 3], &mut output).unwrap_err();
    assert!(matches!(fault, PcuExecutionError::ArithmeticFault(f) if f.invocation_id == 1));
    assert_eq!(output.map(f32::to_bits), previous); // Compound/precision permissions do not uncheck scalar math.

    EXPECTED.store(2, Ordering::Relaxed);
    configure(PcuExecutionPolicy {
        numerical_mode: PcuNumericalMode::Strict,
        float_underflow: PcuFloatUnderflowPolicy::RejectSubnormalResult,
        ..policy()
    })
    .unwrap();
    inherited(&[1.0; 3], &[2.0; 3], &mut output).unwrap();
    assert!(SCORES.load(Ordering::Relaxed) > cold); // Changed defaults invalidate the prepared route.
    let before = output.map(f32::to_bits);
    assert!(inherited(&[f32::from_bits(1); 3], &[2.0; 3], &mut output).is_err());
    assert_eq!(output.map(f32::to_bits), before);
    EXPECTED.store(0, Ordering::Relaxed);
    overrides(&[f32::from_bits(1); 3], &[2.0; 3], &mut output).unwrap();
    assert_eq!(output[0].to_bits(), 2); // Explicit IEEE permits exact subnormals.

    configure(PcuExecutionPolicy {
        numerical_options: PcuNumericalOptions {
            reproducibility: PcuReproducibility::PortableV1,
            ..PcuNumericalOptions::default()
        },
        ..policy()
    })
    .unwrap();
    let before = output.map(f32::to_bits);
    let before_scores = SCORES.load(Ordering::Relaxed);
    assert!(matches!(
        inherited(&[1.0; 3], &[2.0; 3], &mut output),
        Err(PcuExecutionError::UnsupportedNumericalOptions(_))
    ));
    assert_eq!(output.map(f32::to_bits), before);
    assert_eq!(SCORES.load(Ordering::Relaxed), before_scores); // Reject before discovery or scoring.
    overrides(&[1.0; 3], &[2.0; 3], &mut output).unwrap(); // Explicit local override wins.
    configure(policy()).unwrap();
    assert!(matches!(
        portable(&[1.0; 3], &[2.0; 3], &mut output),
        Err(PcuExecutionError::UnsupportedNumericalOptions(_))
    ));
}
