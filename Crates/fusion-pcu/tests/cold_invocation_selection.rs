//! The real annotated route scores cold work and never reranks cached changing-input calls.
#![cfg(feature = "rocm")]

use core::sync::atomic::AtomicUsize;
use core::sync::atomic::Ordering;
#[rustfmt::skip]
use fusion_pcu::{
    global,
    pcu,
};

static SCORES: AtomicUsize = AtomicUsize::new(0);

fn score(candidate: &global::PcuInvocationCandidate<'_>) -> i128 {
    assert_eq!(candidate.kernel.entry.logical_shape, [17, 1, 1]);
    assert!(!candidate.kernel.ops.is_empty());
    assert!(candidate.total_memory_bytes.is_some_and(|bytes| bytes > 0));
    assert!(candidate.facts.compute_unit_count.is_some());
    SCORES.fetch_add(1, Ordering::Relaxed);
    // Consumer preference only: actual ROCm preparation must still admit the full kernel.
    i128::from(candidate.total_memory_bytes.unwrap())
}

#[pcu(invocations = N)]
fn negate<const N: usize>(input: &[f32; N], output: &mut [f32; N]) {
    let id = context.global_invocation_id;
    output[id] = -input[id];
}

#[test]
#[ignore = "requires idle ROCm hardware; run serially"]
fn fresh_inputs_reuse_cold_ranking_and_policy_changes_invalidate_it() {
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Rocm,
        device: Some(0),
        score_invocation: Some(score),
        ..Default::default()
    })
    .unwrap();
    global::clear_thread_cache().unwrap();
    SCORES.store(0, Ordering::Relaxed);
    let mut output = [0.0_f32; 17];
    let first = core::array::from_fn(|index| f32::from(u16::try_from(index).unwrap()));
    negate(&first, &mut output).unwrap();
    let cold_scores = SCORES.load(Ordering::Relaxed);
    assert!(cold_scores > 0);
    for phase in 1..4_u16 {
        let input = core::array::from_fn(|index| f32::from(u16::try_from(index).unwrap() + phase));
        negate(&input, &mut output).unwrap();
        for (actual, expected) in output.iter().zip(input) {
            assert_eq!(actual.to_bits(), expected.to_bits() ^ 0x8000_0000);
        }
        assert_eq!(SCORES.load(Ordering::Relaxed), cold_scores);
    }
    // A new generation invalidates prepared selection. The hook itself is a cold policy field.
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Rocm,
        device: Some(0),
        score_invocation: Some(score),
        ..Default::default()
    })
    .unwrap();
    negate(&first, &mut output).unwrap();
    assert!(SCORES.load(Ordering::Relaxed) > cold_scores);
    global::clear_thread_cache().unwrap();
    global::configure(global::PcuExecutionPolicy::default()).unwrap();
}
