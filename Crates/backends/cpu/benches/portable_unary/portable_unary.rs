//! Six-format Portable exact unary: genuine annotated source, explicit graph, independent native peers.
#[rustfmt::skip]
use std::sync::atomic::{
    AtomicUsize,
    Ordering,
};
#[rustfmt::skip]
use criterion::{
    Criterion,
    criterion_group,
    criterion_main,
};
#[rustfmt::skip]
use pcu_facade::{
    global,
    PcuDispatchFloatUnaryOp,
    PcuFloatUnderflowPolicy,
    PcuRangePolicy,
    PcuBindingRef,
    PcuHostArgument,
    PcuHostKernelBackend,
    PcuPreparedHostKernel,
};
use fusion_pcu_cpu::PcuCpuHostBackend;
#[path = "../low_precision/ffi/ffi.rs"]
mod ffi;
#[path = "../../tests/low_unary/graph/graph.rs"]
mod graph;
#[path = "../../tests/portable_unary/oracle/oracle.rs"]
mod oracle;
#[path = "../../tests/portable_unary/source/source.rs"]
#[allow(dead_code)] // Grid/broadcast schemas are qualified in the source fixture.
mod source;
#[path = "support/support.rs"]
mod support;
#[global_allocator]
static ALLOCATOR: ffi::CountingAllocator = ffi::CountingAllocator;
static COLD_SCORES: AtomicUsize = AtomicUsize::new(0);
fn score(_: &global::PcuInvocationCandidate<'_>) -> i128 {
    COLD_SCORES.fetch_add(1, Ordering::Relaxed);
    0
}
#[allow(clippy::too_many_lines)] // Twelve genuine policy specializations share matched cold setup.
fn width<T: oracle::Native, const N: usize>(criterion: &mut Criterion) {
    macro_rules! operation {
        ($entry:ident,$prepare:ident,$op:ident,$range:ident,$policy:ident) => {{
            let mut source = source::$prepare::<T, N, _>(&PcuCpuHostBackend::detect()).unwrap();
            let mut graph = graph::Graph::new(
                T::TYPE,
                u32::try_from(N).unwrap(),
                PcuDispatchFloatUnaryOp::$op,
                PcuFloatUnderflowPolicy::$policy,
                PcuRangePolicy::$range,
            )
            .with(|kernel| {
                let mut kernel = *kernel;
                kernel
                    .numerical_requirements
                    .numerical_options
                    .reproducibility = pcu_facade::PcuReproducibility::PortableV1;
                if PcuFloatUnderflowPolicy::$policy
                    == PcuFloatUnderflowPolicy::RejectSubnormalResult
                {
                    kernel.numerical_requirements.numerical_mode =
                        pcu_facade::PcuNumericalMode::Strict;
                }
                PcuCpuHostBackend::detect().prepare_host_kernel(&kernel)
            })
            .unwrap();
            support::compare::<T, N, false>(
                criterion,
                PcuDispatchFloatUnaryOp::$op,
                PcuFloatUnderflowPolicy::$policy,
                PcuRangePolicy::$range,
                move |input, output| source(input, output).map_err(|error| error.fault().unwrap()),
                |input, output| {
                    source::$entry::<T, N>(input, output).map_err(|error| match error {
                        global::PcuExecutionError::ArithmeticFault(fault) => fault,
                        other => panic!("unexpected source error {other:?}"),
                    })
                },
                move |input, output| {
                    graph
                        .call(&mut [
                            PcuHostArgument::read(PcuBindingRef::new(0, 0), input),
                            PcuHostArgument::read_write(PcuBindingRef::new(0, 1), output),
                        ])
                        .map_err(|error| error.fault().unwrap())
                },
            );
        }};
    }
    operation!(
        neg_reject_ieee,
        neg_reject_ieee_prepare,
        Neg,
        Reject,
        IeeeAfterRounding
    );
    operation!(
        neg_reject_gradual,
        neg_reject_gradual_prepare,
        Neg,
        Reject,
        AllowGradualUnderflow
    );
    operation!(
        neg_reject_strict,
        neg_reject_strict_prepare,
        Neg,
        Reject,
        RejectSubnormalResult
    );
    operation!(
        neg_clamp_ieee,
        neg_clamp_ieee_prepare,
        Neg,
        Clamp,
        IeeeAfterRounding
    );
    operation!(
        neg_clamp_gradual,
        neg_clamp_gradual_prepare,
        Neg,
        Clamp,
        AllowGradualUnderflow
    );
    operation!(
        neg_clamp_strict,
        neg_clamp_strict_prepare,
        Neg,
        Clamp,
        RejectSubnormalResult
    );
    operation!(
        relu_reject_ieee,
        relu_reject_ieee_prepare,
        Relu,
        Reject,
        IeeeAfterRounding
    );
    operation!(
        relu_reject_gradual,
        relu_reject_gradual_prepare,
        Relu,
        Reject,
        AllowGradualUnderflow
    );
    operation!(
        relu_reject_strict,
        relu_reject_strict_prepare,
        Relu,
        Reject,
        RejectSubnormalResult
    );
    operation!(
        relu_clamp_ieee,
        relu_clamp_ieee_prepare,
        Relu,
        Clamp,
        IeeeAfterRounding
    );
    operation!(
        relu_clamp_gradual,
        relu_clamp_gradual_prepare,
        Relu,
        Clamp,
        AllowGradualUnderflow
    );
    operation!(
        relu_clamp_strict,
        relu_clamp_strict_prepare,
        Relu,
        Clamp,
        RejectSubnormalResult
    );
}

#[allow(clippy::too_many_lines)] // Twelve genuine broadcast specializations have matched cold owner construction.
fn benchmark(criterion: &mut Criterion) {
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Cpu,
        score_invocation: Some(score),
        ..Default::default()
    })
    .unwrap();
    global::clear_thread_cache().unwrap();
    width::<pcu_facade::PcuF16Bits, 1>(criterion);
    width::<pcu_facade::PcuF16Bits, 65>(criterion);
    width::<pcu_facade::PcuBf16Bits, 1>(criterion);
    width::<pcu_facade::PcuBf16Bits, 65>(criterion);
    width::<pcu_facade::PcuF8E4M3FnBits, 1>(criterion);
    width::<pcu_facade::PcuF8E4M3FnBits, 65>(criterion);
    width::<pcu_facade::PcuF8E5M2Bits, 1>(criterion);
    width::<pcu_facade::PcuF8E5M2Bits, 65>(criterion);
    width::<f32, 1>(criterion);
    width::<f32, 65>(criterion);
    width::<f64, 1>(criterion);
    width::<f64, 65>(criterion);
    global::clear_thread_cache().unwrap();
    global::use_defaults().unwrap();
}
criterion_group!(benches, benchmark);
criterion_main!(benches);
