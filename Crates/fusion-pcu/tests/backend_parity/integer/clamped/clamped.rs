//! Integer recovery publishes complete saturated output and an observable range Result.
#[rustfmt::skip]
use core::sync::atomic::{
    AtomicUsize,
    Ordering,
};

#[rustfmt::skip]
use fusion_pcu::{
    global,
    pcu,
    PcuCheckedInteger,
    PcuExecutionFaultKind,
    PcuI256,
    PcuI512,
    PcuNumericalMode,
    PcuRangePolicy,
    PcuU256,
    PcuU512,
};
use super::Sample;
#[rustfmt::skip]
use super::super::support::{
    bits,
    POLICY_LOCK,
};

static SCORES: AtomicUsize = AtomicUsize::new(0);

fn score(_: &global::PcuInvocationCandidate<'_>) -> i128 {
    SCORES.fetch_add(1, Ordering::Relaxed);
    1
}

#[pcu(invocations = N, flag(clamp_range))]
fn add<T: PcuCheckedInteger, const N: usize>(lhs: &[T], rhs: &[T], out: &mut [T]) {
    let id = pcu::context::global_invocation_id();
    out[id] = lhs[id] + rhs[id];
}

// This function inherits global Clamp; source does not need a different arithmetic API.
#[pcu(invocations = N)]
fn sub<T: PcuCheckedInteger, const N: usize>(lhs: &[T], rhs: &[T], out: &mut [T]) {
    let id = pcu::context::global_invocation_id();
    out[id] = lhs[id] - rhs[id];
}

#[pcu(invocations = 3, flag(clamp_range))]
fn mul<T: PcuCheckedInteger, const N: usize>(lhs: &[T], rhs: &T, out: &mut [T]) {
    let mut id = pcu::context::global_invocation_id();
    let stride = pcu::context::invocation_count();
    while id < N {
        out[id] = lhs[id] * *rhs;
        id += stride;
    }
}

fn recovered(error: &global::PcuExecutionError, kind: PcuExecutionFaultKind) {
    assert!(
        matches!(error,
            global::PcuExecutionError::ArithmeticFault(fault)
            if fault.recovered && fault.kind == kind && fault.invocation_id == 2
        ),
        "{error:?}"
    );
}

fn format<T: Sample>() {
    // Isolate each type's cold preparations; do not force eviction of warm entries
    // by combining all specializations into a cohort larger than the cache budget.
    global::clear_thread_cache().unwrap();
    let sentinel = T::small(17);
    let mut output = [sentinel; 9];
    let one = [T::small(1); 7];
    let mut warm_scores = None;
    for phase in [1_u8, 2, 3] {
        let values = [2_u8, 3, 4, 5, 6, 7, 8].map(|value| value + phase);
        let mut lhs = values.map(T::small);
        lhs[2] = T::MAX;
        lhs[6] = T::MAX;
        let mut expected = values.map(|value| T::small(value + 1));
        expected[2] = T::MAX;
        expected[6] = T::MAX;
        recovered(
            &add::<T, 7>(&lhs, &one, &mut output).unwrap_err(),
            PcuExecutionFaultKind::ArithmeticOverflow,
        );
        bits(&output[..7], &expected);
        bits(&output[7..], &[sentinel; 2]);

        lhs[2] = T::MIN;
        lhs[6] = T::MIN;
        expected = values.map(|value| T::small(value - 1));
        expected[2] = T::MIN;
        expected[6] = T::MIN;
        recovered(
            &sub::<T, 7>(&lhs, &one, &mut output).unwrap_err(),
            PcuExecutionFaultKind::ArithmeticUnderflow,
        );
        bits(&output[..7], &expected);

        lhs[2] = T::MAX;
        lhs[6] = T::MAX;
        expected = values.map(|value| T::small(value * 2));
        expected[2] = T::MAX;
        expected[6] = T::MAX;
        recovered(
            &mul::<T, 7>(&lhs, &T::small(2), &mut output).unwrap_err(),
            PcuExecutionFaultKind::ArithmeticOverflow,
        );
        bits(&output[..7], &expected);
        bits(&output[7..], &[sentinel; 2]);

        // A recovered execution does not poison the next ordinary successful call.
        mul::<T, 7>(&values.map(T::small), &T::small(2), &mut output).unwrap();
        bits(&output[..7], &values.map(|value| T::small(value * 2)));
        let scores = SCORES.load(Ordering::Relaxed);
        if let Some(expected) = warm_scores {
            assert_eq!(
                scores, expected,
                "integer Clamp must not reselect a warm provider"
            );
        } else {
            warm_scores = Some(scores);
        }
    }
}

pub fn verify(backend: global::PcuBackendChoice) {
    let _guard = POLICY_LOCK.lock().unwrap();
    for numerical_mode in [PcuNumericalMode::Boundary, PcuNumericalMode::Strict] {
        global::configure(global::PcuExecutionPolicy {
            backend,
            numerical_mode,
            range_policy: PcuRangePolicy::Clamp,
            score_invocation: Some(score),
            ..Default::default()
        })
        .unwrap();
        format::<u8>();
        format::<i8>();
        format::<u16>();
        format::<i16>();
        format::<u32>();
        format::<i32>();
        format::<u64>();
        format::<i64>();
        format::<u128>();
        format::<i128>();
        format::<PcuU256>();
        format::<PcuI256>();
        format::<PcuU512>();
        format::<PcuI512>();
    }
    global::clear_thread_cache().unwrap();
    global::use_defaults().unwrap();
}
