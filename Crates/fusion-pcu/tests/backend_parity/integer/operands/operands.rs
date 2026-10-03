//! Shared ordinary-call integer operand gate; provider opt-in requires native proof.
#[rustfmt::skip]
use core::sync::atomic::{
    AtomicUsize,
    Ordering,
};
#[rustfmt::skip]
use fusion_pcu::{
    global,
    PcuCompoundArithmeticPolicy,
    PcuExecutionFaultKind,
    PcuFloatUnderflowPolicy,
    PcuI256,
    PcuI512,
    PcuNumericalMode,
    PcuNumericalOptions,
    PcuPrecisionPolicy,
    PcuRangePolicy,
    PcuReproducibility,
    PcuU256,
    PcuU512,
};
use super::Sample;
#[rustfmt::skip]
use super::super::support::{
    bits,
    POLICY_LOCK,
};

#[path = "faults/faults.rs"]
mod faults;
#[path = "source/source.rs"]
mod source;

static SCORES: AtomicUsize = AtomicUsize::new(0);

fn score(_: &global::PcuInvocationCandidate<'_>) -> i128 {
    SCORES.fetch_add(1, Ordering::Relaxed);
    1
}

fn format<T: Sample>(range: PcuRangePolicy) {
    // Every specialization fits the bounded cache. Distinct numerical policies
    // remain separately admitted; this test does not authorize realization reuse.
    global::clear_thread_cache().unwrap();
    let sentinel = T::small(17);
    let empty: [T; 0] = [];
    let mut output = [sentinel; 9];
    let mut warm_scores = None;
    for phase in [1_u8, 2, 3] {
        let values = [2_u8, 3, 4, 5, 6, 7, 8].map(|value| value + phase);
        let input = values.map(T::small);
        source::doubled::<T, 7>(&mut output, &input).unwrap();
        bits(&output[..7], &values.map(|value| T::small(value * 2)));
        source::squared::<T, 7>(&empty, &mut output, &input).unwrap();
        bits(&output[..7], &values.map(|value| T::small(value * value)));
        source::reordered::<T, 7>(&[T::small(phase); 7], &mut output, &input).unwrap();
        bits(&output[..7], &[2, 3, 4, 5, 6, 7, 8].map(T::small));
        source::independent::<T, 7>(&input, &mut output).unwrap();
        bits(&output[..7], &[0, 1, 2, 3, 4, 5, 6].map(T::small));
        source::grid::<T, 7>(
            &empty,
            &mut output,
            &values.map(|value| T::small(12 + phase - value)),
        )
        .unwrap();
        bits(&output[..7], &[0, 1, 2, 3, 4, 5, 6].map(T::small));
        bits(&output[7..], &[sentinel; 2]);

        faults::repeated::<T>(range, &input, &mut output);
        faults::broadcast::<T>(range, &input, &mut output);
        bits(&output[7..], &[sentinel; 2]);
        let scores = SCORES.load(Ordering::Relaxed);
        if let Some(cold) = warm_scores {
            assert_eq!(scores, cold, "warm integer roles must not reselect");
        } else {
            warm_scores = Some(scores);
        }
    }
}

fn all_formats(range: PcuRangePolicy) {
    format::<u8>(range);
    format::<i8>(range);
    format::<u16>(range);
    format::<i16>(range);
    format::<u32>(range);
    format::<i32>(range);
    format::<u64>(range);
    format::<i64>(range);
    format::<u128>(range);
    format::<i128>(range);
    format::<PcuU256>(range);
    format::<PcuI256>(range);
    format::<PcuU512>(range);
    format::<PcuI512>(range);
}

pub fn verify(backend: global::PcuBackendChoice) {
    verify_options(
        backend,
        PcuReproducibility::Unspecified,
        &[PcuFloatUnderflowPolicy::IeeeAfterRounding],
    );
}

pub fn verify_portable(backend: global::PcuBackendChoice) {
    verify_options(
        backend,
        PcuReproducibility::PortableV1,
        &[
            PcuFloatUnderflowPolicy::IeeeAfterRounding,
            PcuFloatUnderflowPolicy::RejectSubnormalResult,
            PcuFloatUnderflowPolicy::AllowGradualUnderflow,
        ],
    );
}

fn verify_options(
    backend: global::PcuBackendChoice,
    reproducibility: PcuReproducibility,
    underflow_policies: &[PcuFloatUnderflowPolicy],
) {
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
                for range_policy in [PcuRangePolicy::Reject, PcuRangePolicy::Clamp] {
                    for &float_underflow in underflow_policies {
                        global::configure(global::PcuExecutionPolicy {
                            backend,
                            numerical_mode,
                            numerical_options: PcuNumericalOptions {
                                compound_arithmetic,
                                precision,
                                reproducibility,
                            },
                            range_policy,
                            float_underflow,
                            score_invocation: Some(score),
                            ..Default::default()
                        })
                        .unwrap();
                        all_formats(range_policy);
                    }
                }
            }
        }
    }
    global::clear_thread_cache().unwrap();
    global::use_defaults().unwrap();
}
