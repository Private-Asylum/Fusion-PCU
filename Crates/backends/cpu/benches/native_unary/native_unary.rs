//! Four-format Neg/ReLU source, explicit graph, independent native controls and warm census.
#[rustfmt::skip]
use std::sync::atomic::{AtomicUsize,Ordering};
#[rustfmt::skip]
use criterion::{Criterion,criterion_group,criterion_main};
#[rustfmt::skip]
use pcu_facade::{global,PcuDispatchFloatUnaryOp,PcuFloatUnderflowPolicy,PcuRangePolicy,PcuBindingRef,PcuHostArgument,PcuHostKernelBackend,PcuPreparedHostKernel};
use fusion_pcu_cpu::PcuCpuHostBackend;
#[path = "../low_precision/ffi/ffi.rs"]
mod ffi;
#[path = "../../tests/low_unary/graph/graph.rs"]
mod graph;
#[path = "../../tests/native_unary/oracle/oracle.rs"]
mod oracle;
#[path = "../low_unary/policy/policy.rs"]
mod policy;
#[path = "../../tests/native_unary/source/source.rs"]
#[allow(dead_code)] // Grid/broadcast schemas are qualified in the source fixture.
mod source;
#[path = "support/support.rs"]
mod support;
#[global_allocator]
static ALLOCATOR: ffi::CountingAllocator = ffi::CountingAllocator;
static COLD_SCORES: AtomicUsize = AtomicUsize::new(0);
fn score(candidate: &global::PcuInvocationCandidate<'_>) -> i128 {
    policy::assert_expected(candidate.kernel);
    COLD_SCORES.fetch_add(1, Ordering::Relaxed);
    0
}
#[allow(clippy::too_many_lines)] // Twelve genuine policy specializations share matched cold setup.
fn width<T: oracle::Native, const N: usize>(criterion: &mut Criterion) {
    macro_rules! operation {
        ($entry:ident,$prepare:ident,$op:ident,$range:ident,$policy:ident) => {{
            let expected = policy::Policy::new(
                PcuDispatchFloatUnaryOp::$op,
                PcuFloatUnderflowPolicy::$policy,
                PcuRangePolicy::$range,
            );
            expected.expect();
            let mut source = source::$prepare::<T, N, _>(&policy::Verified(
                PcuCpuHostBackend::detect(),
                expected,
            ))
            .unwrap();
            let mut description = graph::Graph::new(
                T::TYPE,
                u32::try_from(N).unwrap(),
                PcuDispatchFloatUnaryOp::$op,
                PcuFloatUnderflowPolicy::$policy,
                PcuRangePolicy::$range,
            );
            description.numerical_mode = expected.requirements.numerical_mode;
            let mut graph = description
                .with(|kernel| {
                    policy::Verified(PcuCpuHostBackend::detect(), expected)
                        .prepare_host_kernel(kernel)
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
fn broadcasts<T: oracle::Native, const N: usize>(criterion: &mut Criterion) {
    macro_rules! operation {
        ($entry:ident,$prepare:ident,$op:ident,$range:ident,$policy:ident) => {{
            let expected = policy::Policy::new(
                PcuDispatchFloatUnaryOp::$op,
                PcuFloatUnderflowPolicy::$policy,
                PcuRangePolicy::$range,
            );
            expected.expect();
            let mut source = source::$prepare::<T, N, _>(&policy::Verified(
                PcuCpuHostBackend::detect(),
                expected,
            ))
            .unwrap();
            let mut description = graph::Graph::new(
                T::TYPE,
                u32::try_from(N).unwrap(),
                PcuDispatchFloatUnaryOp::$op,
                PcuFloatUnderflowPolicy::$policy,
                PcuRangePolicy::$range,
            );
            description.broadcast = true;
            description.numerical_mode = expected.requirements.numerical_mode;
            let mut graph = description
                .with(|kernel| {
                    policy::Verified(PcuCpuHostBackend::detect(), expected)
                        .prepare_host_kernel(kernel)
                })
                .unwrap();
            support::compare::<T, N, true>(
                criterion,
                PcuDispatchFloatUnaryOp::$op,
                PcuFloatUnderflowPolicy::$policy,
                PcuRangePolicy::$range,
                move |input, output| {
                    source(&input[0], output).map_err(|error| error.fault().unwrap())
                },
                |input, output| {
                    source::$entry::<T, N>(&input[0], output).map_err(|error| match error {
                        global::PcuExecutionError::ArithmeticFault(fault) => fault,
                        other => panic!("unexpected broadcast error {other:?}"),
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
        neg_broadcast_reject_ieee,
        neg_broadcast_reject_ieee_prepare,
        Neg,
        Reject,
        IeeeAfterRounding
    );
    operation!(
        neg_broadcast_reject_gradual,
        neg_broadcast_reject_gradual_prepare,
        Neg,
        Reject,
        AllowGradualUnderflow
    );
    operation!(
        neg_broadcast_reject_strict,
        neg_broadcast_reject_strict_prepare,
        Neg,
        Reject,
        RejectSubnormalResult
    );
    operation!(
        neg_broadcast_clamp_ieee,
        neg_broadcast_clamp_ieee_prepare,
        Neg,
        Clamp,
        IeeeAfterRounding
    );
    operation!(
        neg_broadcast_clamp_gradual,
        neg_broadcast_clamp_gradual_prepare,
        Neg,
        Clamp,
        AllowGradualUnderflow
    );
    operation!(
        neg_broadcast_clamp_strict,
        neg_broadcast_clamp_strict_prepare,
        Neg,
        Clamp,
        RejectSubnormalResult
    );
    operation!(
        relu_broadcast_reject_ieee,
        relu_broadcast_reject_ieee_prepare,
        Relu,
        Reject,
        IeeeAfterRounding
    );
    operation!(
        relu_broadcast_reject_gradual,
        relu_broadcast_reject_gradual_prepare,
        Relu,
        Reject,
        AllowGradualUnderflow
    );
    operation!(
        relu_broadcast_reject_strict,
        relu_broadcast_reject_strict_prepare,
        Relu,
        Reject,
        RejectSubnormalResult
    );
    operation!(
        relu_broadcast_clamp_ieee,
        relu_broadcast_clamp_ieee_prepare,
        Relu,
        Clamp,
        IeeeAfterRounding
    );
    operation!(
        relu_broadcast_clamp_gradual,
        relu_broadcast_clamp_gradual_prepare,
        Relu,
        Clamp,
        AllowGradualUnderflow
    );
    operation!(
        relu_broadcast_clamp_strict,
        relu_broadcast_clamp_strict_prepare,
        Relu,
        Clamp,
        RejectSubnormalResult
    );
}
fn benchmark(criterion: &mut Criterion) {
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Cpu,
        score_invocation: Some(score),
        ..Default::default()
    })
    .unwrap();
    global::clear_thread_cache().unwrap();
    width::<f32, 1>(criterion);
    width::<f32, 4096>(criterion);
    width::<f64, 1>(criterion);
    width::<f64, 4096>(criterion);
    broadcasts::<f32, 1>(criterion);
    broadcasts::<f32, 4096>(criterion);
    broadcasts::<f64, 1>(criterion);
    broadcasts::<f64, 4096>(criterion);
    global::clear_thread_cache().unwrap();
    global::use_defaults().unwrap();
}
criterion_group!(benches, benchmark);
criterion_main!(benches);
