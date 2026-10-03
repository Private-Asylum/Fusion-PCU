//! Actual operand routing and independent domain/pair work at matched transaction boundaries.
use std::hint::black_box;
use criterion::Criterion;
#[rustfmt::skip]
use super::oracle::{
    self,
    Wide,
};
use pcu_facade::PcuExecutionFault;

pub fn native<T: Wide>(
    kind: usize,
    left: &[T],
    right: &[T],
    q: &mut [T],
    r: &mut [T],
) -> Result<(), PcuExecutionFault> {
    for lane in 0..5 {
        let (a, b) = operands(kind, left, right, lane);
        oracle::domain(a, b).map_err(|kind| PcuExecutionFault {
            kind,
            invocation_id: lane as u64,
            recovered: false,
        })?;
    }
    for lane in 0..5 {
        let (a, b) = operands(kind, left, right, lane);
        (q[lane], r[lane]) = oracle::evaluate(a, b).expect("whole-map domain succeeded");
    }
    Ok(())
}
const fn operands<T: Copy>(kind: usize, left: &[T], right: &[T], lane: usize) -> (T, T) {
    match kind {
        0 | 1 => (left[lane], left[lane]),
        2 => (left[lane], right[lane]),
        3 => (left[lane], left[0]),
        4 => (left[0], left[lane]),
        _ => (left[0], right[0]),
    }
}
fn check<T: Wide>(kind: usize, left: &[T], right: &[T], q: &[T], r: &[T]) {
    for lane in 0..5 {
        let (a, b) = operands(kind, left, right, lane);
        assert_eq!((q[lane], r[lane]), oracle::evaluate(a, b).unwrap());
    }
    assert_eq!(
        (&q[5..], &r[5..]),
        (
            &[oracle::small::<T>(99); 3][..],
            &[oracle::small::<T>(99); 3][..]
        )
    );
}
pub fn compare<T: Wide>(
    criterion: &mut Criterion,
    kind: usize,
    mut prepared: impl FnMut(&[T], &[T], &mut [T], &mut [T]),
    mut ordinary: impl FnMut(&[T], &[T], &mut [T], &mut [T]),
    mut graph: impl FnMut(&[T], &[T], &mut [T], &mut [T]),
) {
    let mut invoke = |route: usize, left: &[T], right: &[T], q: &mut [T], r: &mut [T]| match route {
        0 => prepared(left, right, q, r),
        1 => ordinary(left, right, q, r),
        2 => graph(left, right, q, r),
        _ => native(kind, left, right, q, r).unwrap(),
    };
    let seven = oracle::small::<T>(7);
    let mut left = [seven; 5];
    left[0] = oracle::small(3);
    left[2] = oracle::maximum::<T>();
    let right = [oracle::small::<T>(3); 5];
    let mut q = [oracle::small::<T>(99); 8];
    let mut r = q;
    for route in 0..4 {
        invoke(route, &left, &right, &mut q, &mut r);
        check(kind, &left, &right, &q, &r);
    }
    let scores = super::SCORES.load(std::sync::atomic::Ordering::Relaxed);
    for route in 0..4 {
        let counts = super::ffi::count_heap(|| {
            for _ in 0..64 {
                left[0] = if left[0] == seven {
                    oracle::small(3)
                } else {
                    seven
                };
                left[3] = if left[3] == seven {
                    oracle::maximum::<T>()
                } else {
                    seven
                };
                invoke(
                    route,
                    black_box(&left),
                    black_box(&right),
                    black_box(&mut q),
                    black_box(&mut r),
                );
                check(kind, &left, &right, &q, &r);
            }
        });
        assert_eq!(
            (counts.allocations, counts.reallocations, counts.frees),
            (0, 0, 0)
        );
        assert_eq!(
            super::SCORES.load(std::sync::atomic::Ordering::Relaxed),
            scores
        );
        println!(
            "DivRem roles census type={} kind{kind} route{route} calls64 alloc0 realloc0 free0",
            core::any::type_name::<T>()
        );
    }
    {
        let mut group = criterion.benchmark_group(format!(
            "cpu_div_rem_roles/{}/kind{kind}",
            core::any::type_name::<T>()
        ));
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
                    left[0] = if left[0] == seven {
                        oracle::small(3)
                    } else {
                        seven
                    };
                    left[3] = if left[3] == seven {
                        oracle::maximum::<T>()
                    } else {
                        seven
                    };
                    invoke(
                        route,
                        black_box(&left),
                        black_box(&right),
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
        super::SCORES.load(std::sync::atomic::Ordering::Relaxed),
        scores
    );
}
