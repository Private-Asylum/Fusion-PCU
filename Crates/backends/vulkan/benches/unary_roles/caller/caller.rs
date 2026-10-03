//! Static matched caller boundaries; semantic checks and 64-changing-call heap census are untimed.
#[rustfmt::skip]
use criterion::{
    Criterion,
    Throughput,
};
#[rustfmt::skip]
use pcu_facade::{
    PcuExecutionFault,
    PcuDispatchFloatUnaryOp,
    PcuImplementationRequirements,
};
#[rustfmt::skip]
use super::{
    ffi,
    oracle,
    COLD_SCORES,
};
use oracle::Native;
use std::sync::atomic::Ordering;

pub fn compare<T: Native, const BROADCAST: bool>(
    criterion: &mut Criterion,
    role: &str,
    profile: (PcuDispatchFloatUnaryOp, PcuImplementationRequirements),
    source: impl FnMut(&[T], &mut [T]) -> Result<(), PcuExecutionFault>,
    ordinary: impl FnMut(&[T], &mut [T]) -> Result<(), PcuExecutionFault>,
    graph: impl FnMut(&[T], &mut [T]) -> Result<(), PcuExecutionFault>,
    control: impl FnMut(&[T], &mut [T]) -> Result<(), PcuExecutionFault>,
) {
    let (operation, requirements) = profile;
    let name = format!(
        "vulkan_unary_roles/{:?}/{role}/{:?}/{:?}/{:?}/{:?}/{:?}",
        T::TYPE,
        requirements.numerical_mode,
        requirements.numerical_options.compound_arithmetic,
        requirements.numerical_options.precision,
        requirements.float_underflow,
        requirements.range_policy
    );
    let mut group = criterion.benchmark_group(&name);
    group.throughput(Throughput::Elements(7));
    entry::<T, BROADCAST>(
        &mut group,
        &name,
        "source_prepared",
        source,
        operation,
        requirements,
    );
    entry::<T, BROADCAST>(
        &mut group,
        &name,
        "source_ordinary",
        ordinary,
        operation,
        requirements,
    );
    entry::<T, BROADCAST>(
        &mut group,
        &name,
        "graph_prepared",
        graph,
        operation,
        requirements,
    );
    entry::<T, BROADCAST>(
        &mut group,
        &name,
        "native_vulkan",
        control,
        operation,
        requirements,
    );
    group.finish();
}
fn native<T: Native, const BROADCAST: bool>(
    input: &[T],
    output: &mut [T],
    operation: PcuDispatchFloatUnaryOp,
    requirements: PcuImplementationRequirements,
) -> Result<(), PcuExecutionFault> {
    if BROADCAST {
        oracle::native_broadcast::<T, 7>(
            input,
            output,
            operation,
            requirements.float_underflow,
            requirements.range_policy,
        )
    } else {
        oracle::native::<T, 7>(
            input,
            output,
            operation,
            requirements.float_underflow,
            requirements.range_policy,
        )
    }
}
fn equal<T: Native>(left: &[T], right: &[T]) {
    assert_eq!(left.len(), right.len());
    for (left, right) in left.iter().zip(right) {
        assert_eq!(left.bits(), right.bits());
    }
}
fn entry<T: Native, const BROADCAST: bool>(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    name: &str,
    route: &str,
    mut entry: impl FnMut(&[T], &mut [T]) -> Result<(), PcuExecutionFault>,
    operation: PcuDispatchFloatUnaryOp,
    requirements: PcuImplementationRequirements,
) {
    let base = if requirements.range_policy == pcu_facade::PcuRangePolicy::Clamp
        && requirements.float_underflow
            == pcu_facade::PcuFloatUnderflowPolicy::RejectSubnormalResult
    {
        2
    } else {
        1 << T::FRACTION
    };
    let mut input = [T::from_bits(base); 7];
    let mut output = [T::from_bits(17); 10];
    let mut expected = output;
    for phase in 0..3 {
        input.fill(T::from_bits(base + phase));
        assert_eq!(
            entry(&input, &mut output),
            native::<T, BROADCAST>(&input, &mut expected, operation, requirements)
        );
        equal(&output, &expected);
    }
    assert_eq!(oracle::bits(&output), oracle::bits(&expected));
    let before = output;
    input[if BROADCAST { 0 } else { 5 }] = T::from_bits(T::SIGN - 1);
    assert_eq!(
        entry(&input, &mut output),
        native::<T, BROADCAST>(&input, &mut expected, operation, requirements)
    );
    equal(&output, &before);
    input.fill(T::from_bits(base));
    assert_eq!(
        entry(&input, &mut output),
        native::<T, BROADCAST>(&input, &mut expected, operation, requirements)
    );
    equal(&output, &expected);
    let scores = COLD_SCORES.load(Ordering::Relaxed);
    let counts = ffi::count_heap(|| {
        for phase in 0..64 {
            input.fill(T::from_bits(base + phase % 3));
            let result = native::<T, BROADCAST>(&input, &mut expected, operation, requirements);
            assert_eq!(
                entry(
                    std::hint::black_box(&input),
                    std::hint::black_box(&mut output)
                ),
                result
            );
            equal(&output, &expected);
        }
    });
    assert_eq!(
        (counts.allocations, counts.reallocations, counts.frees),
        (0, 0, 0)
    );
    assert_eq!(COLD_SCORES.load(Ordering::Relaxed), scores);
    println!("CENSUS {name}/{route} calls64 heap0 score0");
    group.bench_function(route, |bench| {
        bench.iter(|| {
            input[0] = T::from_bits(input[0].bits() ^ 1);
            let result = native::<T, BROADCAST>(&input, &mut expected, operation, requirements);
            assert_eq!(
                entry(
                    std::hint::black_box(&input),
                    std::hint::black_box(&mut output)
                ),
                result
            );
            equal(&output, &expected);
        });
    });
}
