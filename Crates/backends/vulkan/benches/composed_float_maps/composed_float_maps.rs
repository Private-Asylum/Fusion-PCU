//! Static ordinary/prepared/hand-IR callers versus independent native Vulkan lifecycle.
extern crate pcu_facade as fusion_pcu;
#[path = "ffi/ffi.rs"]
mod ffi;
#[path = "../../prepared/composed/tests/graph/graph.rs"]
mod graph;
#[path = "../../tests/composed_float_maps/oracle/oracle.rs"]
#[allow(dead_code)] // Complete low encoding/fault models are exercised separately.
mod oracle;
#[path = "../../tests/composed_float_maps/source/source.rs"]
#[allow(dead_code)] // Grid/broadcast profiles are qualified by the source fixture.
mod source;
#[rustfmt::skip]
use criterion::{
    Criterion,
    criterion_group,
    criterion_main,
};
#[rustfmt::skip]
use pcu_facade::{
    global,
    PcuBindingRef,
    PcuExecutionFault,
    PcuFloatUnderflowPolicy,
    PcuHostArgument,
    PcuPreparedHostKernel,
    PcuRangePolicy,
};
#[rustfmt::skip]
use std::{
    hint::black_box,
    sync::atomic::{AtomicUsize,Ordering},
};
#[rustfmt::skip]
use fusion_pcu_vulkan::{
    PcuVulkanBackend,
    PcuVulkanError,
};
#[path = "../../tests/scalar_transport/device/device.rs"]
mod device;
fn fault(error: PcuVulkanError) -> PcuExecutionFault {
    match error {
        PcuVulkanError::Fault(fault) => fault,
        other => panic!("unexpected native/provider error: {other:?}"),
    }
}
#[global_allocator]
static ALLOCATOR: ffi::CountingAllocator = ffi::CountingAllocator;
static SCORES: AtomicUsize = AtomicUsize::new(0);
fn score(candidate: &global::PcuInvocationCandidate<'_>) -> i128 {
    one_effect::assert_candidate(candidate.kernel);
    SCORES.fetch_add(1, Ordering::Relaxed);
    0
}
fn compare_entry<T: oracle::Format, const N: usize>(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    route: &str,
    name: &str,
    mut entry: impl FnMut(&[T], &mut [T]) -> Result<(), PcuExecutionFault>,
    input: &mut [T],
    output: &mut [T],
    sentinel: T,
) {
    for phase in 0..3 {
        for (lane, value) in input.iter_mut().enumerate() {
            *value = oracle::dyadic::<T>(lane + phase).0;
        }
        entry(input, output).unwrap();
        for (lane, value) in output[..N].iter().enumerate() {
            assert_eq!(value.bits(), oracle::dyadic::<T>(lane + phase).1.bits());
        }
    }
    for value in &output[N..] {
        assert_eq!(value.bits(), sentinel.bits());
    }
    let scores = SCORES.load(Ordering::Relaxed);
    let counts = ffi::count_heap(|| {
        for phase in 0..64 {
            for (lane, value) in input.iter_mut().enumerate() {
                *value = oracle::dyadic::<T>(lane + phase).0;
            }
            entry(black_box(&*input), black_box(&mut *output)).unwrap();
            for (lane, value) in output[..N].iter().enumerate() {
                assert_eq!(value.bits(), oracle::dyadic::<T>(lane + phase).1.bits());
            }
        }
    });
    assert_eq!(
        (counts.allocations, counts.reallocations, counts.frees),
        (0, 0, 0)
    );
    assert_eq!(SCORES.load(Ordering::Relaxed), scores);
    println!(
        "vulkan_composed {:?}/{name}/{N}/{route}: 64 changing calls, zero warm Rust heap/rescore",
        T::TYPE
    );
    let mut phase = 0_usize;
    group.bench_function(route, |bench| {
        bench.iter(|| {
            phase = phase.wrapping_add(1);
            for (lane, value) in input.iter_mut().enumerate() {
                *value = oracle::dyadic::<T>(lane + phase).0;
            }
            entry(black_box(&*input), black_box(&mut *output)).unwrap();
            black_box(&output);
        });
    });
}
fn compare<T: oracle::Format, const N: usize>(
    criterion: &mut Criterion,
    name: &str,
    source: impl FnMut(&[T], &mut [T]) -> Result<(), PcuExecutionFault>,
    ordinary: impl FnMut(&[T], &mut [T]) -> Result<(), PcuExecutionFault>,
    graph: impl FnMut(&[T], &mut [T]) -> Result<(), PcuExecutionFault>,
    native: impl FnMut(&[T], &mut [T]) -> Result<(), PcuExecutionFault>,
) {
    let mut input = vec![T::from(T::ONE); N];
    let sentinel = T::from(T::ONE + 1);
    let mut output = vec![sentinel; N + 3];
    let mut group = criterion.benchmark_group(format!("vulkan_composed/{:?}/{name}/{N}", T::TYPE));
    // Each caller is monomorphized; no function-trait-object dispatch in measured work.
    compare_entry::<T, N>(
        &mut group,
        "pcu_source_prepared",
        name,
        source,
        &mut input,
        &mut output,
        sentinel,
    );
    compare_entry::<T, N>(
        &mut group,
        "pcu_source_ordinary",
        name,
        ordinary,
        &mut input,
        &mut output,
        sentinel,
    );
    compare_entry::<T, N>(
        &mut group,
        "independent_graph_diagnostic",
        name,
        graph,
        &mut input,
        &mut output,
        sentinel,
    );
    compare_entry::<T, N>(
        &mut group,
        "native_independent_u32_vulkan",
        name,
        native,
        &mut input,
        &mut output,
        sentinel,
    );
    group.finish();
}
fn composed_requirements(
    uf: PcuFloatUnderflowPolicy,
    range: PcuRangePolicy,
) -> pcu_facade::PcuImplementationRequirements {
    pcu_facade::PcuImplementationRequirements {
        numerical_mode: if uf == PcuFloatUnderflowPolicy::RejectSubnormalResult {
            pcu_facade::PcuNumericalMode::Strict
        } else {
            pcu_facade::PcuNumericalMode::Boundary
        },
        float_underflow: uf,
        range_policy: range,
        ..Default::default()
    }
}
fn width<T: oracle::Format, const N: usize>(
    criterion: &mut Criterion,
    backend: &PcuVulkanBackend,
    identity: pcu_facade::PcuStableDeviceIdentity,
) {
    macro_rules! profile {
        ($entry:ident,$prepare:ident,$ir:ident,$policy:ident,$range:ident) => {{
            let requirements =
                composed_requirements(PcuFloatUnderflowPolicy::$policy, PcuRangePolicy::$range);
            let verified = one_effect::verified(backend, requirements, 2);
            let mut prepared = source::$prepare::<T, N, _>(&verified).unwrap();
            let mut graph = graph::prepare::<T, N, _>(
                &verified,
                PcuFloatUnderflowPolicy::$policy,
                PcuRangePolicy::$range,
            );
            let mut native = ffi::NativeComposed::new(
                identity,
                u32::try_from(N).unwrap(),
                match PcuFloatUnderflowPolicy::$policy {
                    PcuFloatUnderflowPolicy::IeeeAfterRounding => 0,
                    PcuFloatUnderflowPolicy::RejectSubnormalResult => 1,
                    PcuFloatUnderflowPolicy::AllowGradualUnderflow => 2,
                },
                u32::from(PcuRangePolicy::$range == PcuRangePolicy::Clamp),
                T::TYPE,
            )
            .unwrap();
            native.assert_policy(
                T::TYPE,
                PcuFloatUnderflowPolicy::$policy,
                PcuRangePolicy::$range,
            );
            let name = concat!(stringify!($policy), "/", stringify!($range));
            compare::<T, N>(
                criterion,
                name,
                move |input, output| prepared(output, input).map_err(fault),
                |input, output| {
                    source::$entry::<T, N>(output, input)
                        .map_err(|error| error.arithmetic_fault().unwrap())
                },
                move |input, output| {
                    graph
                        .call(&mut [
                            PcuHostArgument::read_write(PcuBindingRef::new(0, 0), output),
                            PcuHostArgument::read(PcuBindingRef::new(0, 1), input),
                        ])
                        .map_err(fault)
                },
                move |input, output| {
                    native
                        .call(ffi::bytes(input), ffi::bytes_mut(output))
                        .unwrap()
                        .map_or(Ok(()), Err)
                },
            );
            one_effect::clear_expected();
        }};
    }
    // Names preserve the exact source range/underflow tuple of all three PCU routes.
    profile!(
        reject_ieee,
        reject_ieee_prepare,
        reject_ieee_ir,
        IeeeAfterRounding,
        Reject
    );
    profile!(
        reject_gradual,
        reject_gradual_prepare,
        reject_gradual_ir,
        AllowGradualUnderflow,
        Reject
    );
    profile!(
        reject_tight,
        reject_tight_prepare,
        reject_tight_ir,
        RejectSubnormalResult,
        Reject
    );
    profile!(
        clamp_ieee,
        clamp_ieee_prepare,
        clamp_ieee_ir,
        IeeeAfterRounding,
        Clamp
    );
    profile!(
        clamp_gradual,
        clamp_gradual_prepare,
        clamp_gradual_ir,
        AllowGradualUnderflow,
        Clamp
    );
    profile!(
        clamp_tight,
        clamp_tight_prepare,
        clamp_tight_ir,
        RejectSubnormalResult,
        Clamp
    );
}
#[path = "one_effect/one_effect.rs"]
mod one_effect;
fn benchmark(criterion: &mut Criterion) {
    let (backend, identity) = device::selected();
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Vulkan,
        score_invocation: Some(score),
        ..Default::default()
    })
    .unwrap();
    width::<pcu_facade::PcuF16Bits, 1>(criterion, &backend, identity);
    width::<pcu_facade::PcuF16Bits, 65>(criterion, &backend, identity);
    width::<pcu_facade::PcuBf16Bits, 1>(criterion, &backend, identity);
    width::<pcu_facade::PcuBf16Bits, 65>(criterion, &backend, identity);
    width::<pcu_facade::PcuF8E4M3FnBits, 1>(criterion, &backend, identity);
    width::<pcu_facade::PcuF8E4M3FnBits, 65>(criterion, &backend, identity);
    width::<pcu_facade::PcuF8E5M2Bits, 1>(criterion, &backend, identity);
    width::<pcu_facade::PcuF8E5M2Bits, 65>(criterion, &backend, identity);
    width::<f32, 1>(criterion, &backend, identity);
    width::<f32, 65>(criterion, &backend, identity);
    width::<f64, 1>(criterion, &backend, identity);
    width::<f64, 65>(criterion, &backend, identity);
    one_effect::width::<pcu_facade::PcuF16Bits, 1>(criterion, &backend, identity);
    one_effect::width::<pcu_facade::PcuF16Bits, 65>(criterion, &backend, identity);
    one_effect::width::<pcu_facade::PcuBf16Bits, 1>(criterion, &backend, identity);
    one_effect::width::<pcu_facade::PcuBf16Bits, 65>(criterion, &backend, identity);
    one_effect::width::<pcu_facade::PcuF8E4M3FnBits, 1>(criterion, &backend, identity);
    one_effect::width::<pcu_facade::PcuF8E4M3FnBits, 65>(criterion, &backend, identity);
    one_effect::width::<pcu_facade::PcuF8E5M2Bits, 1>(criterion, &backend, identity);
    one_effect::width::<pcu_facade::PcuF8E5M2Bits, 65>(criterion, &backend, identity);
    one_effect::width::<f32, 1>(criterion, &backend, identity);
    one_effect::width::<f32, 65>(criterion, &backend, identity);
    one_effect::width::<f64, 1>(criterion, &backend, identity);
    one_effect::width::<f64, 65>(criterion, &backend, identity);
    global::clear_thread_cache().unwrap();
    global::use_defaults().unwrap();
}
criterion_group!(benches, benchmark);
criterion_main!(benches);
