//! Matched transactional boundaries with fresh inputs, complete output proof and warm census.
#[rustfmt::skip]
use std::{
    fmt::Debug,
    hint::black_box,
};
#[rustfmt::skip]
use criterion::{
    Criterion,
    Throughput,
};
#[rustfmt::skip]
use pcu_facade::{
    PcuExecutionFault,
    PcuExecutionFaultKind,
    PcuScalar,
};

pub fn native<T: PcuScalar, const N: usize>(
    lhs: &[T],
    rhs: &[T],
    q: &mut [T],
    r: &mut [T],
    evaluate: impl Fn(T, T) -> Result<(T, T), PcuExecutionFaultKind>,
    domain: impl Fn(T, T) -> Result<(), PcuExecutionFaultKind>,
) -> Result<(), PcuExecutionFault> {
    assert!(lhs.len() >= N && rhs.len() >= N && q.len() >= N && r.len() >= N);
    for index in 0..N {
        domain(lhs[index], rhs[index]).map_err(|kind| PcuExecutionFault {
            recovered: false,
            kind,
            invocation_id: u64::try_from(index).unwrap(),
        })?;
    }
    for index in 0..N {
        (q[index], r[index]) =
            evaluate(lhs[index], rhs[index]).expect("whole-map preflight succeeded");
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)] // The four actual invocation routes and native oracle are independent peers.
pub fn compare<T: PcuScalar + Debug + PartialEq, const N: usize>(
    criterion: &mut Criterion,
    mut prepared: impl FnMut(&[T], &[T], &mut [T], &mut [T]),
    mut ordinary: impl FnMut(&[T], &[T], &mut [T], &mut [T]),
    mut graph: impl FnMut(&[T], &[T], &mut [T], &mut [T]),
    evaluate: impl Fn(T, T) -> Result<(T, T), PcuExecutionFaultKind>,
    domain: impl Fn(T, T) -> Result<(), PcuExecutionFaultKind>,
    values: [T; 5],
) {
    let mut invoke = |route: usize, lhs: &[T], rhs: &[T], q: &mut [T], r: &mut [T]| match route {
        0 => prepared(lhs, rhs, q, r),
        1 => ordinary(lhs, rhs, q, r),
        2 => graph(lhs, rhs, q, r),
        _ => native::<T, N>(lhs, rhs, q, r, &evaluate, &domain).unwrap(),
    };
    let [zero, two, three, seven, tail] = values;
    let mut lhs = vec![seven; N];
    let rhs = vec![three; N];
    let mut q = vec![tail; N + 3];
    let mut r = q.clone();
    let mut expected_q = q.clone();
    let mut expected_r = r.clone();
    native::<T, N>(
        &lhs,
        &rhs,
        &mut expected_q,
        &mut expected_r,
        &evaluate,
        &domain,
    )
    .unwrap();
    for route in 0..4 {
        invoke(route, &lhs, &rhs, &mut q, &mut r);
        assert_eq!((&q, &r), (&expected_q, &expected_r));
    }
    let warm_scores = super::COLD_SCORES.load(std::sync::atomic::Ordering::Relaxed);
    for route in 0..4 {
        let counts = super::ffi::count_heap(|| {
            for _ in 0..256 {
                lhs[0] = if lhs[0] == two { seven } else { two };
                invoke(
                    route,
                    black_box(&lhs),
                    black_box(&rhs),
                    black_box(&mut q),
                    black_box(&mut r),
                );
                black_box((&q, &r));
            }
        });
        assert_eq!(
            (counts.allocations, counts.reallocations, counts.frees),
            (0, 0, 0)
        );
        assert_eq!(
            super::COLD_SCORES.load(std::sync::atomic::Ordering::Relaxed),
            warm_scores
        );
    }
    // The independent native oracle also checks failed-map rollback before measurements.
    let mut invalid_rhs = rhs.clone();
    invalid_rhs[N - 1] = zero;
    let before = (q.clone(), r.clone());
    assert!(native::<T, N>(&lhs, &invalid_rhs, &mut q, &mut r, &evaluate, &domain).is_err());
    assert_eq!((&q, &r), (&before.0, &before.1));
    {
        let mut group = criterion.benchmark_group(format!(
            "cpu_checked_div_rem/{}/N{N}",
            core::any::type_name::<T>()
        ));
        group.throughput(Throughput::Elements(N as u64));
        for (route, label) in [
            "source_prepared",
            "source_ordinary",
            "explicit_graph_diagnostic",
            "native_checked",
        ]
        .into_iter()
        .enumerate()
        {
            group.bench_function(label, |bencher| {
                bencher.iter(|| {
                    lhs[0] = if lhs[0] == two { seven } else { two };
                    invoke(
                        route,
                        black_box(&lhs),
                        black_box(&rhs),
                        black_box(&mut q),
                        black_box(&mut r),
                    );
                    black_box((&q, &r));
                });
            });
        }
        group.finish();
    }
    assert_eq!(
        super::COLD_SCORES.load(std::sync::atomic::Ordering::Relaxed),
        warm_scores
    );
}
