//! Complete saturated outputs with observable notices, independent arithmetic and matched warm census.
use std::hint::black_box;
use criterion::{Criterion, Throughput};
use super::oracle::Wide;
#[rustfmt::skip]
use pcu_facade::{PcuDispatchIntegerBinaryOp,PcuExecutionFault,PcuExecutionFaultKind};
fn native<T: Wide, const N: usize>(
    left: &[T],
    right: &[T],
    output: &mut [T],
    op: PcuDispatchIntegerBinaryOp,
) -> Result<(), PcuExecutionFault> {
    assert!(left.len() >= N && right.len() >= N && output.len() >= N);
    let mut notice = None;
    for lane in 0..N {
        output[lane] = match super::oracle::evaluate(left[lane], right[lane], op) {
            Ok(value) => value,
            Err(
                kind @ (PcuExecutionFaultKind::ArithmeticUnderflow
                | PcuExecutionFaultKind::ArithmeticOverflow),
            ) => {
                notice.get_or_insert_with(|| PcuExecutionFault {
                    recovered: true,
                    invocation_id: u64::try_from(lane).unwrap(),
                    kind,
                });
                if kind == PcuExecutionFaultKind::ArithmeticUnderflow {
                    super::oracle::minimum()
                } else {
                    super::oracle::maximum()
                }
            }
            Err(other) => panic!("independent range-only oracle {other:?}"),
        };
    }
    notice.map_or(Ok(()), Err)
}
#[allow(clippy::too_many_lines)] // One matched workload includes cold proof, changing inputs and four caller censuses.
pub fn compare<T: Wide, const N: usize>(
    criterion: &mut Criterion,
    op: PcuDispatchIntegerBinaryOp,
    mut prepared: impl FnMut(&[T], &[T], &mut [T]) -> Result<(), PcuExecutionFault>,
    mut ordinary: impl FnMut(&[T], &[T], &mut [T]) -> Result<(), PcuExecutionFault>,
    mut graph: impl FnMut(&[T], &[T], &mut [T]) -> Result<(), PcuExecutionFault>,
) {
    let endpoint = if op == PcuDispatchIntegerBinaryOp::Sub {
        super::oracle::minimum::<T>()
    } else {
        super::oracle::maximum::<T>()
    };
    let first = endpoint;
    let mut bytes = first.bytes();
    bytes[0] ^= 1;
    let second = T::from_bytes(bytes);
    let two = super::oracle::small(2);
    let sentinel = super::oracle::small(17);
    let mut left = vec![first; N];
    let right = vec![two; N];
    let mut output = vec![sentinel; N + 3];
    let mut expected = output.clone();
    let expected_result = native::<T, N>(&left, &right, &mut expected, op);
    let mut invoke = |route, left: &[T], output: &mut [T]| match route {
        0 => prepared(left, &right, output),
        1 => ordinary(left, &right, output),
        2 => graph(left, &right, output),
        _ => native::<T, N>(left, &right, output, op),
    };
    for changed in [first, second] {
        left[0] = changed;
        for route in 0..4 {
            assert_eq!(invoke(route, &left, &mut output), expected_result);
            assert_eq!(output, expected);
        }
    }
    let scores = super::COLD_SCORES.load(std::sync::atomic::Ordering::Relaxed);
    for route in 0..4 {
        let counts = super::ffi::count_heap(|| {
            for _ in 0..64 {
                left[0] = if left[0] == first { second } else { first };
                assert_eq!(invoke(route, &left, &mut output), expected_result);
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
        println!(
            "census {:?} {op:?} N{N} route{route} calls64 allocations0 reallocations0 frees0",
            T::TYPE
        );
    }
    {
        let mut group =
            criterion.benchmark_group(format!("cpu_integer_clamp/{:?}/{op:?}/N{N}", T::TYPE));
        group.throughput(Throughput::Elements(u64::try_from(N).unwrap()));
        for (route, label) in [
            "prepared_annotated_source",
            "ordinary_annotated_source",
            "explicit_graph_diagnostic",
            "native_base256_integer",
        ]
        .into_iter()
        .enumerate()
        {
            group.bench_function(label, |bench| {
                bench.iter(|| {
                    left[0] = if left[0] == first { second } else { first };
                    assert_eq!(
                        invoke(route, black_box(&left), black_box(&mut output)),
                        expected_result
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
