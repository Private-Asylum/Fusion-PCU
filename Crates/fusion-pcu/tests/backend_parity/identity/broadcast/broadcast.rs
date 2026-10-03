//! A borrowed scalar broadcasts exact representation bits without arithmetic eligibility.

#[rustfmt::skip]
use fusion_pcu::{
    global,
    pcu,
    PcuScalar,
};
#[rustfmt::skip]
use super::{
    PcuBf16Bits,
    PcuF16Bits,
    PcuF128Bits,
    PcuF256Bits,
    PcuF8E4M3FnBits,
    PcuF8E5M2Bits,
    PcuI256,
    PcuI512,
    PcuNumericalMode,
    PcuU256,
    PcuU512,
    Sample,
    score,
    SCORES,
    Ordering,
};
#[rustfmt::skip]
use super::super::support::{
    bits,
    POLICY_LOCK,
};

#[pcu(invocations = N)]
fn direct<T: PcuScalar, const N: usize>(input: &T, output: &mut [T]) {
    let id = pcu::context::global_invocation_id();
    output[id] = *input;
}

#[pcu(invocations = 17)]
fn grid<T: PcuScalar, const N: usize>(input: &T, output: &mut [T]) {
    let mut id = pcu::context::global_invocation_id();
    let stride = pcu::context::invocation_count();
    while id < N {
        output[id] = *input;
        id += stride;
    }
}

#[pcu(invocations = N)]
fn flat<T: PcuScalar, const N: usize>(input: &[T], output: &mut [T]) {
    let id = pcu::context::global_invocation_id();
    output[id] = input[0];
}

#[pcu(invocations = 17)]
fn flat_grid<T: PcuScalar, const N: usize>(input: &[T], output: &mut [T]) {
    let mut id = pcu::context::global_invocation_id();
    let stride = pcu::context::invocation_count();
    while id < N {
        output[id] = input[0usize];
        id += stride;
    }
}

fn format<T: Sample>() {
    global::clear_thread_cache().unwrap();
    let mut warm_scores = None;
    for seed in [0_u8, 73, 191] {
        let input = T::pattern(seed);
        let sentinel = T::pattern(251);
        let mut output = [sentinel; 68];
        for execute in [direct::<T, 65>, grid::<T, 65>] {
            execute(&input, &mut output).unwrap();
            bits(&output[..65], &[input; 65]);
            bits(&output[65..], &[sentinel; 3]);
            output.fill(sentinel);
        }
        // Only element zero is read, even when the supplied host slice has a tail.
        // Arbitrary representation bits in the tail must never become arithmetic inputs.
        for execute in [flat::<T, 65>, flat_grid::<T, 65>] {
            for source in [&[input][..], &[input, sentinel][..]] {
                execute(source, &mut output).unwrap();
                bits(&output[..65], &[input; 65]);
                bits(&output[65..], &[sentinel; 3]);
                output.fill(sentinel);
            }
            if seed == 0 {
                assert!(execute(&[], &mut output).is_err());
                bits(&output, &[sentinel; 68]);
            }
        }
        let scores = SCORES.load(Ordering::Relaxed);
        if let Some(expected) = warm_scores {
            assert_eq!(
                scores, expected,
                "warm scalar broadcast must not rerank providers"
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
        format::<PcuF16Bits>();
        format::<PcuBf16Bits>();
        format::<PcuF8E4M3FnBits>();
        format::<PcuF8E5M2Bits>();
        format::<f32>();
        format::<f64>();
        format::<PcuF128Bits>();
        format::<PcuF256Bits>();
    }
    global::clear_thread_cache().unwrap();
    global::use_defaults().unwrap();
}
