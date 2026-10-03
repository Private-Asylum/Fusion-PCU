//! Matched complete wide maps, changing inputs, no-rescore/zero-heap warm census.
use std::hint::black_box;
#[rustfmt::skip]
use criterion::{Criterion,Throughput};
use super::oracle::Wide;
#[rustfmt::skip]
use pcu_facade::{PcuDispatchIntegerBinaryOp,PcuExecutionFault};
fn native<T: Wide, const N: usize>(
    left: &[T],
    right: &[T],
    output: &mut [T],
    op: PcuDispatchIntegerBinaryOp,
) -> Result<(), PcuExecutionFault> {
    assert!(left.len() >= N && right.len() >= N && output.len() >= N);
    for index in 0..N {
        super::oracle::evaluate(left[index], right[index], op).map_err(|kind| {
            PcuExecutionFault {
                recovered: false,
                kind,
                invocation_id: u64::try_from(index).unwrap(),
            }
        })?;
    }
    for (index, value) in output.iter_mut().take(N).enumerate() {
        *value = super::oracle::evaluate(left[index], right[index], op).unwrap();
    }
    Ok(())
}
#[allow(clippy::too_many_lines)] // Frozen workload proof, matched route census and semantic loop stay adjacent.
pub fn compare<T: Wide, const N: usize>(
    criterion: &mut Criterion,
    op: PcuDispatchIntegerBinaryOp,
    mut prepared: impl FnMut(&[T], &[T], &mut [T]) -> Result<(), PcuExecutionFault>,
    mut ordinary: impl FnMut(&[T], &[T], &mut [T]) -> Result<(), PcuExecutionFault>,
    mut graph: impl FnMut(&[T], &[T], &mut [T]) -> Result<(), PcuExecutionFault>,
) {
    let mut bytes = [0; 64];
    bytes[0] = 2;
    bytes[T::BYTES / 2] = 1;
    let first = T::from_bytes(bytes);
    bytes[0] = 3;
    let second = T::from_bytes(bytes);
    let one = super::oracle::small::<T>(1);
    let sentinel = super::oracle::small::<T>(77);
    let mut left = vec![first; N];
    let right = vec![one; N];
    let mut output = vec![sentinel; N + 3];
    let mut expected = output.clone();
    let mut invoke = |route, left: &[T], right: &[T], output: &mut [T]| match route {
        0 => prepared(left, right, output),
        1 => ordinary(left, right, output),
        2 => graph(left, right, output),
        _ => native::<T, N>(left, right, output, op),
    };
    for initial in [first, second] {
        left[0] = initial;
        native::<T, N>(&left, &right, &mut expected, op).unwrap();
        for route in 0..4 {
            invoke(route, &left, &right, &mut output).unwrap();
            assert_eq!(output, expected);
        }
    }
    let scores = super::COLD_SCORES.load(std::sync::atomic::Ordering::Relaxed);
    for route in 0..4 {
        let counts = super::ffi::count_heap(|| {
            for _ in 0..256 {
                left[0] = if left[0] == first { second } else { first };
                invoke(
                    route,
                    black_box(&left),
                    black_box(&right),
                    black_box(&mut output),
                )
                .unwrap();
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
        assert_eq!(output[N..], [sentinel; 3]);
    }
    {
        let mut group = criterion.benchmark_group(format!(
            "cpu_wide_integer/{}/{op:?}/N{N}",
            core::any::type_name::<T>()
        ));
        group.throughput(Throughput::Elements(u64::try_from(N).unwrap()));
        for (route, name) in [
            "source_prepared",
            "source_ordinary",
            "explicit_graph_diagnostic",
            "native_base256_full_product",
        ]
        .into_iter()
        .enumerate()
        {
            group.bench_function(name, |bencher| {
                bencher.iter(|| {
                    left[0] = if left[0] == first { second } else { first };
                    invoke(
                        route,
                        black_box(&left),
                        black_box(&right),
                        black_box(&mut output),
                    )
                    .unwrap();
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
