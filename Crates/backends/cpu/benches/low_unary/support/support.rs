//! Matched caller-owned unary boundaries with independent finite encoding arithmetic.
use std::hint::black_box;
#[rustfmt::skip]
use criterion::{Criterion,Throughput};
#[rustfmt::skip]
use pcu_facade::{PcuDispatchFloatUnaryOp,PcuFloatUnderflowPolicy,PcuRangePolicy,PcuExecutionFault};
use super::{oracle, ffi};
use oracle::Low;
#[allow(clippy::too_many_lines)] // Four matched warm routes and their independent ownership/fault/census checks stay adjacent.
pub fn compare<T: Low, const N: usize>(
    criterion: &mut Criterion,
    op: PcuDispatchFloatUnaryOp,
    policy: PcuFloatUnderflowPolicy,
    range: PcuRangePolicy,
    mut source: impl FnMut(&[T], &mut [T]) -> Result<(), PcuExecutionFault>,
    mut ordinary: impl FnMut(&[T], &mut [T]) -> Result<(), PcuExecutionFault>,
    mut graph: impl FnMut(&[T], &mut [T]) -> Result<(), PcuExecutionFault>,
) {
    let base = if range == PcuRangePolicy::Clamp
        && policy == PcuFloatUnderflowPolicy::RejectSubnormalResult
    {
        2
    } else {
        1 << T::FRACTION
    };
    let mut input = vec![T::from_bits(base); N];
    let mut output = vec![T::from_bits(17); N + 3];
    let mut expected = output.clone();
    let mut native =
        |input: &[T], output: &mut [T]| oracle::native::<T, N>(input, output, op, policy, range);
    for entry in [
        &mut source as &mut dyn FnMut(&[T], &mut [T]) -> Result<(), PcuExecutionFault>,
        &mut ordinary,
        &mut graph,
        &mut native,
    ] {
        for value in [base, base + 1] {
            input[0] = T::from_bits(value);
            let result = oracle::native::<T, N>(&input, &mut expected, op, policy, range);
            assert_eq!(entry(&input, &mut output), result);
            assert_eq!(output, expected);
        }
        let before = output.clone();
        input[N - 1] = T::from_bits(T::SIGN - 1);
        assert!(entry(&input, &mut output).is_err());
        assert_eq!(output, before);
        input[N - 1] = T::from_bits(base);
        let result = oracle::native::<T, N>(&input, &mut expected, op, policy, range);
        assert_eq!(entry(&input, &mut output), result);
        assert_eq!(output, expected);
    }
    let expected_result = oracle::native::<T, N>(&input, &mut expected, op, policy, range);
    let warm_scores = super::COLD_SCORES.load(std::sync::atomic::Ordering::Relaxed);
    for entry in [
        &mut source as &mut dyn FnMut(&[T], &mut [T]) -> Result<(), PcuExecutionFault>,
        &mut ordinary,
        &mut graph,
        &mut native,
    ] {
        let counts = ffi::count_heap(|| {
            for _ in 0..256 {
                input[0] = T::from_bits(input[0].bits() ^ 1);
                assert_eq!(
                    entry(black_box(&input), black_box(&mut output)),
                    expected_result
                );
                black_box(&output);
            }
        });
        assert_eq!(
            (counts.allocations, counts.reallocations, counts.frees),
            (0, 0, 0)
        );
    }
    assert_eq!(
        super::COLD_SCORES.load(std::sync::atomic::Ordering::Relaxed),
        warm_scores
    );
    println!(
        "unary {:?} {op:?}/{policy:?}/{range:?} N{N}: four routes x256 warm calls, zero Rust heap",
        T::TYPE
    );
    {
        let mut group = criterion.benchmark_group(format!(
            "cpu_low_unary/{:?}/{op:?}/{policy:?}/{range:?}/{N}",
            T::TYPE
        ));
        group.throughput(Throughput::Elements(N as u64));
        for (entry, name) in [
            (
                &mut source as &mut dyn FnMut(&[T], &mut [T]) -> Result<(), PcuExecutionFault>,
                "pcu_source_prepared",
            ),
            (&mut ordinary, "pcu_source_ordinary"),
            (&mut graph, "explicit_graph_diagnostic"),
            (&mut native, "native_independent_bits"),
        ] {
            group.bench_function(name, |bench| {
                bench.iter(|| {
                    input[0] = T::from_bits(input[0].bits() ^ 1);
                    assert_eq!(
                        entry(black_box(&input), black_box(&mut output)),
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
        warm_scores
    );
}
