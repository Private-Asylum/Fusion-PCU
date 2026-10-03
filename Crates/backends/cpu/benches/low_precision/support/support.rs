//! Matched complete-map boundaries, independent native controls and warm allocation census.
use std::hint::black_box;
#[rustfmt::skip]
use criterion::{
    Criterion,
    Throughput,
};
use super::oracle::Low;
#[rustfmt::skip]
use pcu_facade::{
    PcuDispatchFloatBinaryOp,
    PcuExecutionFault,
    PcuFloatUnderflowPolicy,
};
pub fn native<T: Low, const N: usize>(
    left: &[T],
    right: &[T],
    output: &mut [T],
    op: PcuDispatchFloatBinaryOp,
) -> Result<(), PcuExecutionFault> {
    assert!(left.len() >= N && right.len() >= N && output.len() >= N);
    let evaluate = |index: usize| {
        T::FORMAT.evaluate(
            left[index].bits(),
            right[index].bits(),
            op,
            PcuFloatUnderflowPolicy::IeeeAfterRounding,
        )
    };
    for index in 0..N {
        evaluate(index).map_err(|kind| PcuExecutionFault {
            recovered: false,
            kind,
            invocation_id: u64::try_from(index).unwrap(),
        })?;
    }
    for (index, value) in output.iter_mut().take(N).enumerate() {
        *value = T::from_bits(evaluate(index).expect("whole-map preflight succeeded"));
    }
    Ok(())
}
pub fn compare<T: Low, const N: usize>(
    criterion: &mut Criterion,
    op: PcuDispatchFloatBinaryOp,
    mut prepared: impl FnMut(&[T], &[T], &mut [T]),
    mut ordinary: impl FnMut(&[T], &[T], &mut [T]),
    mut graph: impl FnMut(&[T], &[T], &mut [T]),
) {
    let one = T::from_bits(u16::try_from(T::FORMAT.bias).unwrap() << T::FORMAT.fraction);
    let two = T::from_bits(one.bits() + (1 << T::FORMAT.fraction));
    let sentinel = T::from_bits(T::FORMAT.max);
    let mut left = vec![two; N];
    let right = vec![one; N];
    let mut output = vec![sentinel; N + 3];
    let mut expected = output.clone();
    native::<T, N>(&left, &right, &mut expected, op).unwrap();
    let mut invoke = |route, left: &[T], right: &[T], output: &mut [T]| match route {
        0 => prepared(left, right, output),
        1 => ordinary(left, right, output),
        2 => graph(left, right, output),
        _ => native::<T, N>(left, right, output, op).unwrap(),
    };
    for route in 0..4 {
        invoke(route, &left, &right, &mut output);
        assert_eq!(output, expected);
    }
    let scores = super::COLD_SCORES.load(std::sync::atomic::Ordering::Relaxed);
    for route in 0..4 {
        let counts = super::ffi::count_heap(|| {
            for _ in 0..256 {
                left[0] = if left[0] == one { two } else { one };
                invoke(
                    route,
                    black_box(&left),
                    black_box(&right),
                    black_box(&mut output),
                );
                black_box(&output);
            }
        });
        assert_eq!(
            (counts.allocations, counts.reallocations, counts.frees),
            (0, 0, 0)
        );
        assert_eq!(
            super::COLD_SCORES.load(std::sync::atomic::Ordering::Relaxed),
            scores
        );
    }
    let mut invalid = right.clone();
    invalid[N - 1] = T::from_bits(T::FORMAT.sign - 1);
    let before = output.clone();
    assert!(native::<T, N>(&left, &invalid, &mut output, op).is_err());
    assert_eq!(output, before);
    {
        let mut group = criterion.benchmark_group(format!(
            "{}/{}/{op:?}/N{N}",
            super::PROFILE,
            core::any::type_name::<T>()
        ));
        group.throughput(Throughput::Elements(u64::try_from(N).unwrap()));
        for (route, name) in [
            "source_prepared",
            "source_ordinary",
            "explicit_graph_diagnostic",
            "native_binary64_midpoints",
        ]
        .into_iter()
        .enumerate()
        {
            group.bench_function(name, |bencher| {
                bencher.iter(|| {
                    left[0] = if left[0] == one { two } else { one };
                    invoke(
                        route,
                        black_box(&left),
                        black_box(&right),
                        black_box(&mut output),
                    );
                    black_box(&output);
                });
            });
        }
        group.finish();
    }
    assert_eq!(
        super::COLD_SCORES.load(std::sync::atomic::Ordering::Relaxed),
        scores
    );
}
