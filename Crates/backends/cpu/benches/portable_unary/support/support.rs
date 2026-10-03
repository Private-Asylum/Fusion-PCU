//! Matched caller-owned unary boundaries with independent finite encoding arithmetic.
use std::hint::black_box;
#[rustfmt::skip]
use criterion::{
    Criterion,
    Throughput,
};
#[rustfmt::skip]
use pcu_facade::{
    PcuDispatchFloatUnaryOp,
    PcuFloatUnderflowPolicy,
    PcuRangePolicy,
    PcuExecutionFault,
};
use super::{oracle, ffi};
use oracle::Native;
#[allow(clippy::too_many_lines)] // Four matched warm routes and their independent ownership/fault/census checks stay adjacent.
pub fn compare<T: Native, const N: usize, const BROADCAST: bool>(
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
    let input_len = if BROADCAST { 1 } else { N };
    let mut input = vec![T::from_bits(base); input_len];
    let mut output = vec![T::from_bits(17); N + 3];
    let mut expected = output.clone();
    let mut native = |input: &[T], output: &mut [T]| {
        native_map::<T, N, BROADCAST>(input, output, op, policy, range)
    };
    for entry in [
        &mut source as &mut dyn FnMut(&[T], &mut [T]) -> Result<(), PcuExecutionFault>,
        &mut ordinary,
        &mut graph,
        &mut native,
    ] {
        for value in [base, base + 1] {
            input[0] = T::from_bits(value);
            let result = native_map::<T, N, BROADCAST>(&input, &mut expected, op, policy, range);
            assert_eq!(entry(&input, &mut output), result);
            assert_eq!(oracle::bits(&output), oracle::bits(&expected));
        }
        let before = output.clone();
        input[input_len - 1] = T::from_bits(T::SIGN - 1);
        assert!(entry(&input, &mut output).is_err());
        assert_eq!(oracle::bits(&output), oracle::bits(&before));
        input[input_len - 1] = T::from_bits(base);
        let result = native_map::<T, N, BROADCAST>(&input, &mut expected, op, policy, range);
        assert_eq!(entry(&input, &mut output), result);
        assert_eq!(oracle::bits(&output), oracle::bits(&expected));
    }
    let expected_result = native_map::<T, N, BROADCAST>(&input, &mut expected, op, policy, range);
    let warm_scores = super::COLD_SCORES.load(std::sync::atomic::Ordering::Relaxed);
    for entry in [
        &mut source as &mut dyn FnMut(&[T], &mut [T]) -> Result<(), PcuExecutionFault>,
        &mut ordinary,
        &mut graph,
        &mut native,
    ] {
        let counts = ffi::count_heap(|| {
            for _ in 0..64 {
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
        "unary {:?} {op:?}/{policy:?}/{range:?} N{N}: four routes x64 warm calls, zero Rust heap",
        T::TYPE
    );
    {
        let mut group = criterion.benchmark_group(format!(
            "cpu_portable_unary/{:?}/{op:?}/{policy:?}/{range:?}/broadcast{BROADCAST}/{N}",
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

fn native_map<T: Native, const N: usize, const BROADCAST: bool>(
    input: &[T],
    output: &mut [T],
    op: PcuDispatchFloatUnaryOp,
    policy: PcuFloatUnderflowPolicy,
    range: PcuRangePolicy,
) -> Result<(), PcuExecutionFault> {
    if BROADCAST {
        oracle::native_broadcast::<T, N>(input, output, op, policy, range)
    } else {
        oracle::native::<T, N>(input, output, op, policy, range)
    }
}
