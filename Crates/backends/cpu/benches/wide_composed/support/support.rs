//! Separate handwritten transactional orchestration reuses the core per-operation numerical law.
use pcu_facade::{PcuCheckedInteger, PcuClampedFault, PcuExecutionFault, PcuRangePolicy};
use std::hint::black_box;

pub trait Wide: PcuCheckedInteger {
    fn small(value: u64) -> Self;
    fn maximum() -> Self;
}
macro_rules! widths {
    ($($ty:ty, $limbs:literal, $signed:literal;)+) => {$(
        impl Wide for $ty {
            fn small(value: u64) -> Self {
                let mut limbs = [0; $limbs]; limbs[0] = value;
                Self::from_limbs_le(limbs)
            }
            fn maximum() -> Self {
                let mut limbs = [u64::MAX; $limbs];
                if $signed { limbs[$limbs - 1] >>= 1; }
                Self::from_limbs_le(limbs)
            }
        }
    )+};
}
widths! {
    pcu_facade::PcuI256, 4, true;
    pcu_facade::PcuU256, 4, false;
    pcu_facade::PcuI512, 8, true;
    pcu_facade::PcuU512, 8, false;
}
fn apply<T: Wide>(
    result: Result<T, PcuClampedFault<T>>,
    lane: usize,
    range: PcuRangePolicy,
    first: &mut Option<PcuExecutionFault>,
) -> Result<T, PcuExecutionFault> {
    match result {
        Ok(value) => Ok(value),
        Err(error) => {
            let fault = PcuExecutionFault {
                invocation_id: lane as u64,
                kind: error.kind(),
                recovered: range == PcuRangePolicy::Clamp,
            };
            if fault.recovered {
                first.get_or_insert(fault);
                Ok(error.clamped_value())
            } else {
                Err(fault)
            }
        }
    }
}
pub fn native<T: Wide>(
    input: &[T],
    seed: T,
    stage: &mut [T],
    output: &mut [T],
    shadows: &mut [T],
    range: PcuRangePolicy,
) -> Result<(), PcuExecutionFault> {
    let n = input.len();
    let (next_stage, next_output) = shadows.split_at_mut(n);
    let mut first = None;
    for (lane, &original) in input.iter().enumerate() {
        let intermediate = apply(original.pcu_clamped_add(seed), lane, range, &mut first)?;
        next_stage[lane] = intermediate;
        let product = apply(
            intermediate.pcu_clamped_mul(original),
            lane,
            range,
            &mut first,
        )?;
        next_output[lane] = apply(product.pcu_clamped_sub(original), lane, range, &mut first)?;
    }
    stage[..n].copy_from_slice(next_stage);
    output[..n].copy_from_slice(next_output);
    first.map_or(Ok(()), Err)
}
fn equal<T: Wide>(left: &[T], right: &[T]) {
    for (left, right) in left.iter().zip(right) {
        assert_eq!(left.encode_le().as_ref(), right.encode_le().as_ref());
    }
}
fn change<T: Wide>(input: &mut [T], phase: usize, edge: bool) {
    for (lane, value) in input.iter_mut().enumerate() {
        *value = T::small(((lane + phase) % 7 + 1) as u64);
    }
    if edge {
        *input.last_mut().unwrap() = T::maximum();
    }
}
pub fn compare<T: Wide, const N: usize>(
    criterion: &mut criterion::Criterion,
    label: &str,
    range: PcuRangePolicy,
    edge: bool,
    mut entry: impl FnMut(&[T], &T, &mut [T], &mut [T]) -> Result<(), PcuExecutionFault>,
) {
    let seed = T::small(1);
    let sentinel = T::small(17);
    let mut input = vec![seed; N];
    let mut stage = vec![sentinel; N + 2];
    let mut output = vec![sentinel; N + 3];
    let mut expected_stage = stage.clone();
    let mut expected_output = output.clone();
    let mut shadows = vec![seed; N * 2];
    for phase in 0..3 {
        change(&mut input, phase, edge);
        let expected = native(
            &input,
            seed,
            &mut expected_stage,
            &mut expected_output,
            &mut shadows,
            range,
        );
        assert_eq!(entry(&input, &seed, &mut stage, &mut output), expected);
        equal(&stage, &expected_stage);
        equal(&output, &expected_output);
    }
    let scores = crate::SCORES.load(std::sync::atomic::Ordering::Relaxed);
    let counts = crate::ffi::count_heap(|| {
        for phase in 0..64 {
            change(&mut input, phase, edge);
            let expected = native(
                &input,
                seed,
                &mut expected_stage,
                &mut expected_output,
                &mut shadows,
                range,
            );
            assert_eq!(entry(&input, &seed, &mut stage, &mut output), expected);
            equal(&stage, &expected_stage);
            equal(&output, &expected_output);
        }
    });
    assert_eq!(
        (counts.allocations, counts.reallocations, counts.frees),
        (0, 0, 0)
    );
    assert_eq!(
        crate::SCORES.load(std::sync::atomic::Ordering::Relaxed),
        scores
    );
    println!(
        "wide composed {:?}/{N}/{range:?}/{edge}/{label}: 64 changing calls, zero warm Rust heap/rescore",
        T::TYPE
    );
    let mut phase = 0_usize;
    criterion.bench_function(
        &format!(
            "cpu_wide_composed/{:?}/{N}/{range:?}/{edge}/{label}",
            T::TYPE
        ),
        |bench| {
            bench.iter(|| {
                phase = phase.wrapping_add(1);
                change(&mut input, phase, edge);
                black_box(entry(
                    black_box(&input),
                    &seed,
                    black_box(&mut stage),
                    black_box(&mut output),
                ))
                .ok();
                black_box((&stage, &output));
            });
        },
    );
}
