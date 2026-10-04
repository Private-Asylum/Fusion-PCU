//! Genuine four-declaration unary source paired with explicit graph and independent bit controls.
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
    pcu,
    PcuBindingRef,
    PcuCheckedFloat,
    PcuCompoundArithmeticPolicy,
    PcuDispatchFloatUnaryOp,
    PcuFloatUnderflowPolicy,
    PcuHostArgument,
    PcuHostKernelBackend,
    PcuImplementationRequirements,
    PcuNumericalMode,
    PcuPrecisionPolicy,
    PcuPreparedHostKernel,
    PcuRangePolicy,
};
use fusion_pcu_vulkan::PcuVulkanError;
#[path = "bytes/bytes.rs"]
mod bytes;
#[path = "caller/caller.rs"]
mod caller;
#[path = "../../tests/scalar_transport/device/device.rs"]
mod device;
#[path = "../checked_unary/ffi/ffi.rs"]
#[allow(dead_code)]
mod ffi;
#[path = "../../../cpu/tests/unary_roles/graph/graph.rs"]
mod graph;
#[path = "../../../cpu/tests/portable_unary/oracle/oracle.rs"]
mod oracle;
#[path = "../../../cpu/benches/unary_roles/source/source.rs"]
mod source;
#[global_allocator]
static ALLOCATOR: ffi::CountingAllocator = ffi::CountingAllocator;
static COLD_SCORES: AtomicUsize = AtomicUsize::new(0);
static SELECTED_IDENTITY: std::sync::OnceLock<pcu_facade::PcuStableDeviceIdentity> =
    std::sync::OnceLock::new();
fn fault(error: PcuVulkanError) -> pcu_facade::PcuExecutionFault {
    match error {
        PcuVulkanError::Fault(fault) => fault,
        other => panic!("unexpected native profile failure {other:?}"),
    }
}
fn score(candidate: &global::PcuInvocationCandidate<'_>) -> i128 {
    assert_eq!(
        candidate.facts.stable_identity.as_ref(),
        SELECTED_IDENTITY.get()
    );
    COLD_SCORES.fetch_add(1, Ordering::Relaxed);
    0
}
macro_rules! shape {
    ($T:ident,$criterion:ident,$requirements:ident,$entry:ident,$ir:ident,$bindings:ident,$op:ident,$grid:expr,$broadcast:expr) => {{
        let (backend, identity) = device::selected();
        assert_eq!(*SELECTED_IDENTITY.get_or_init(|| identity), identity);
        let bindings = source::$bindings::<$T>();
        let mut prepared = source::$ir::<$T>(&bindings)
            .unwrap()
            .with_numerical_requirements($requirements)
            .with_ir(|kernel| backend.prepare_host_kernel(kernel))
            .unwrap();
        let mut graph = graph::Graph::new(
            $T::TYPE,
            7,
            PcuDispatchFloatUnaryOp::$op,
            $requirements.float_underflow,
            $requirements.range_policy,
        );
        graph.grid = $grid;
        graph.broadcast = $broadcast;
        let mut graph = graph
            .with(|kernel| {
                let mut kernel = *kernel;
                kernel.numerical_requirements = $requirements;
                backend.prepare_host_kernel(&kernel)
            })
            .unwrap();
        let format = match $T::TYPE {
            pcu_facade::PcuScalarType::F16 => 0,
            pcu_facade::PcuScalarType::BF16 => 1,
            pcu_facade::PcuScalarType::F8E4M3FN => 2,
            pcu_facade::PcuScalarType::F8E5M2 => 3,
            pcu_facade::PcuScalarType::F32 => 4,
            pcu_facade::PcuScalarType::F64 => 5,
            _ => unreachable!(),
        };
        let operation = match PcuDispatchFloatUnaryOp::$op {
            PcuDispatchFloatUnaryOp::Neg => 0,
            PcuDispatchFloatUnaryOp::Relu => 1,
        };
        let policy = match $requirements.float_underflow {
            PcuFloatUnderflowPolicy::IeeeAfterRounding => 0,
            PcuFloatUnderflowPolicy::RejectSubnormalResult => 1,
            PcuFloatUnderflowPolicy::AllowGradualUnderflow => 2,
        };
        let mut native = if $broadcast {
            ffi::NativeUnary::new_broadcast(identity, 7, format, operation, policy)
        } else {
            ffi::NativeUnary::new(identity, 7, format, operation, policy)
        }
        .unwrap();
        let ghost = $T::from_bits($T::SIGN - 1);
        caller::compare::<$T, $broadcast>(
            $criterion,
            stringify!($entry),
            (PcuDispatchFloatUnaryOp::$op, $requirements),
            |input, output| {
                prepared
                    .call(&mut [
                        PcuHostArgument::read(PcuBindingRef::new(0, 0), &[] as &[$T]),
                        PcuHostArgument::read_write(PcuBindingRef::new(0, 1), output),
                        PcuHostArgument::read(PcuBindingRef::new(0, 2), input),
                        PcuHostArgument::read(PcuBindingRef::new(0, 3), &[] as &[$T]),
                    ])
                    .map_err(fault)
            },
            |input, output| {
                source::$entry::<$T>(&[], output, input, &ghost)
                    .map_err(|e| e.arithmetic_fault().unwrap())
            },
            |input, output| {
                graph
                    .call(&mut [
                        PcuHostArgument::read(PcuBindingRef::new(0, 0), &[] as &[$T]),
                        PcuHostArgument::read_write(PcuBindingRef::new(0, 1), output),
                        PcuHostArgument::read(PcuBindingRef::new(0, 2), input),
                        PcuHostArgument::read(PcuBindingRef::new(0, 3), &[] as &[$T]),
                    ])
                    .map_err(fault)
            },
            |input, output| {
                native
                    .call(
                        bytes::input(input),
                        bytes::output(output),
                        $requirements.range_policy == PcuRangePolicy::Clamp,
                    )
                    .unwrap()
                    .map_or(Ok(()), Err)
            },
        );
    }};
}

fn reject_ieeeafterrounding<T: oracle::Native>(
    criterion: &mut Criterion,
    requirements: PcuImplementationRequirements,
) {
    shape!(
        T,
        criterion,
        requirements,
        direct_neg_reject_ieee,
        direct_neg_reject_ieee_ir,
        direct_neg_reject_ieee_bindings,
        Neg,
        false,
        false
    );
    shape!(
        T,
        criterion,
        requirements,
        direct_relu_reject_ieee,
        direct_relu_reject_ieee_ir,
        direct_relu_reject_ieee_bindings,
        Relu,
        false,
        false
    );
    reject_ieeeafterrounding_grid::<T>(criterion, requirements);
    reject_ieeeafterrounding_broadcast::<T>(criterion, requirements);
}
fn reject_ieeeafterrounding_grid<T: oracle::Native>(
    criterion: &mut Criterion,
    requirements: PcuImplementationRequirements,
) {
    shape!(
        T,
        criterion,
        requirements,
        grid_neg_reject_ieee,
        grid_neg_reject_ieee_ir,
        grid_neg_reject_ieee_bindings,
        Neg,
        true,
        false
    );
    shape!(
        T,
        criterion,
        requirements,
        grid_relu_reject_ieee,
        grid_relu_reject_ieee_ir,
        grid_relu_reject_ieee_bindings,
        Relu,
        true,
        false
    );
}
fn reject_ieeeafterrounding_broadcast<T: oracle::Native>(
    criterion: &mut Criterion,
    requirements: PcuImplementationRequirements,
) {
    shape!(
        T,
        criterion,
        requirements,
        broadcast_neg_reject_ieee,
        broadcast_neg_reject_ieee_ir,
        broadcast_neg_reject_ieee_bindings,
        Neg,
        false,
        true
    );
    shape!(
        T,
        criterion,
        requirements,
        broadcast_relu_reject_ieee,
        broadcast_relu_reject_ieee_ir,
        broadcast_relu_reject_ieee_bindings,
        Relu,
        false,
        true
    );
}
fn reject_allowgradualunderflow<T: oracle::Native>(
    criterion: &mut Criterion,
    requirements: PcuImplementationRequirements,
) {
    shape!(
        T,
        criterion,
        requirements,
        direct_neg_reject_gradual,
        direct_neg_reject_gradual_ir,
        direct_neg_reject_gradual_bindings,
        Neg,
        false,
        false
    );
    shape!(
        T,
        criterion,
        requirements,
        direct_relu_reject_gradual,
        direct_relu_reject_gradual_ir,
        direct_relu_reject_gradual_bindings,
        Relu,
        false,
        false
    );
    reject_allowgradualunderflow_grid::<T>(criterion, requirements);
    reject_allowgradualunderflow_broadcast::<T>(criterion, requirements);
}
fn reject_allowgradualunderflow_grid<T: oracle::Native>(
    criterion: &mut Criterion,
    requirements: PcuImplementationRequirements,
) {
    shape!(
        T,
        criterion,
        requirements,
        grid_neg_reject_gradual,
        grid_neg_reject_gradual_ir,
        grid_neg_reject_gradual_bindings,
        Neg,
        true,
        false
    );
    shape!(
        T,
        criterion,
        requirements,
        grid_relu_reject_gradual,
        grid_relu_reject_gradual_ir,
        grid_relu_reject_gradual_bindings,
        Relu,
        true,
        false
    );
}
fn reject_allowgradualunderflow_broadcast<T: oracle::Native>(
    criterion: &mut Criterion,
    requirements: PcuImplementationRequirements,
) {
    shape!(
        T,
        criterion,
        requirements,
        broadcast_neg_reject_gradual,
        broadcast_neg_reject_gradual_ir,
        broadcast_neg_reject_gradual_bindings,
        Neg,
        false,
        true
    );
    shape!(
        T,
        criterion,
        requirements,
        broadcast_relu_reject_gradual,
        broadcast_relu_reject_gradual_ir,
        broadcast_relu_reject_gradual_bindings,
        Relu,
        false,
        true
    );
}
fn reject_rejectsubnormalresult<T: oracle::Native>(
    criterion: &mut Criterion,
    requirements: PcuImplementationRequirements,
) {
    shape!(
        T,
        criterion,
        requirements,
        direct_neg_reject_tight,
        direct_neg_reject_tight_ir,
        direct_neg_reject_tight_bindings,
        Neg,
        false,
        false
    );
    shape!(
        T,
        criterion,
        requirements,
        direct_relu_reject_tight,
        direct_relu_reject_tight_ir,
        direct_relu_reject_tight_bindings,
        Relu,
        false,
        false
    );
    reject_rejectsubnormalresult_grid::<T>(criterion, requirements);
    reject_rejectsubnormalresult_broadcast::<T>(criterion, requirements);
}
fn reject_rejectsubnormalresult_grid<T: oracle::Native>(
    criterion: &mut Criterion,
    requirements: PcuImplementationRequirements,
) {
    shape!(
        T,
        criterion,
        requirements,
        grid_neg_reject_tight,
        grid_neg_reject_tight_ir,
        grid_neg_reject_tight_bindings,
        Neg,
        true,
        false
    );
    shape!(
        T,
        criterion,
        requirements,
        grid_relu_reject_tight,
        grid_relu_reject_tight_ir,
        grid_relu_reject_tight_bindings,
        Relu,
        true,
        false
    );
}
fn reject_rejectsubnormalresult_broadcast<T: oracle::Native>(
    criterion: &mut Criterion,
    requirements: PcuImplementationRequirements,
) {
    shape!(
        T,
        criterion,
        requirements,
        broadcast_neg_reject_tight,
        broadcast_neg_reject_tight_ir,
        broadcast_neg_reject_tight_bindings,
        Neg,
        false,
        true
    );
    shape!(
        T,
        criterion,
        requirements,
        broadcast_relu_reject_tight,
        broadcast_relu_reject_tight_ir,
        broadcast_relu_reject_tight_bindings,
        Relu,
        false,
        true
    );
}
fn clamp_ieeeafterrounding<T: oracle::Native>(
    criterion: &mut Criterion,
    requirements: PcuImplementationRequirements,
) {
    shape!(
        T,
        criterion,
        requirements,
        direct_neg_clamp_ieee,
        direct_neg_clamp_ieee_ir,
        direct_neg_clamp_ieee_bindings,
        Neg,
        false,
        false
    );
    shape!(
        T,
        criterion,
        requirements,
        direct_relu_clamp_ieee,
        direct_relu_clamp_ieee_ir,
        direct_relu_clamp_ieee_bindings,
        Relu,
        false,
        false
    );
    clamp_ieeeafterrounding_grid::<T>(criterion, requirements);
    clamp_ieeeafterrounding_broadcast::<T>(criterion, requirements);
}
fn clamp_ieeeafterrounding_grid<T: oracle::Native>(
    criterion: &mut Criterion,
    requirements: PcuImplementationRequirements,
) {
    shape!(
        T,
        criterion,
        requirements,
        grid_neg_clamp_ieee,
        grid_neg_clamp_ieee_ir,
        grid_neg_clamp_ieee_bindings,
        Neg,
        true,
        false
    );
    shape!(
        T,
        criterion,
        requirements,
        grid_relu_clamp_ieee,
        grid_relu_clamp_ieee_ir,
        grid_relu_clamp_ieee_bindings,
        Relu,
        true,
        false
    );
}
fn clamp_ieeeafterrounding_broadcast<T: oracle::Native>(
    criterion: &mut Criterion,
    requirements: PcuImplementationRequirements,
) {
    shape!(
        T,
        criterion,
        requirements,
        broadcast_neg_clamp_ieee,
        broadcast_neg_clamp_ieee_ir,
        broadcast_neg_clamp_ieee_bindings,
        Neg,
        false,
        true
    );
    shape!(
        T,
        criterion,
        requirements,
        broadcast_relu_clamp_ieee,
        broadcast_relu_clamp_ieee_ir,
        broadcast_relu_clamp_ieee_bindings,
        Relu,
        false,
        true
    );
}
fn clamp_allowgradualunderflow<T: oracle::Native>(
    criterion: &mut Criterion,
    requirements: PcuImplementationRequirements,
) {
    shape!(
        T,
        criterion,
        requirements,
        direct_neg_clamp_gradual,
        direct_neg_clamp_gradual_ir,
        direct_neg_clamp_gradual_bindings,
        Neg,
        false,
        false
    );
    shape!(
        T,
        criterion,
        requirements,
        direct_relu_clamp_gradual,
        direct_relu_clamp_gradual_ir,
        direct_relu_clamp_gradual_bindings,
        Relu,
        false,
        false
    );
    clamp_allowgradualunderflow_grid::<T>(criterion, requirements);
    clamp_allowgradualunderflow_broadcast::<T>(criterion, requirements);
}
fn clamp_allowgradualunderflow_grid<T: oracle::Native>(
    criterion: &mut Criterion,
    requirements: PcuImplementationRequirements,
) {
    shape!(
        T,
        criterion,
        requirements,
        grid_neg_clamp_gradual,
        grid_neg_clamp_gradual_ir,
        grid_neg_clamp_gradual_bindings,
        Neg,
        true,
        false
    );
    shape!(
        T,
        criterion,
        requirements,
        grid_relu_clamp_gradual,
        grid_relu_clamp_gradual_ir,
        grid_relu_clamp_gradual_bindings,
        Relu,
        true,
        false
    );
}
fn clamp_allowgradualunderflow_broadcast<T: oracle::Native>(
    criterion: &mut Criterion,
    requirements: PcuImplementationRequirements,
) {
    shape!(
        T,
        criterion,
        requirements,
        broadcast_neg_clamp_gradual,
        broadcast_neg_clamp_gradual_ir,
        broadcast_neg_clamp_gradual_bindings,
        Neg,
        false,
        true
    );
    shape!(
        T,
        criterion,
        requirements,
        broadcast_relu_clamp_gradual,
        broadcast_relu_clamp_gradual_ir,
        broadcast_relu_clamp_gradual_bindings,
        Relu,
        false,
        true
    );
}
fn clamp_rejectsubnormalresult<T: oracle::Native>(
    criterion: &mut Criterion,
    requirements: PcuImplementationRequirements,
) {
    shape!(
        T,
        criterion,
        requirements,
        direct_neg_clamp_tight,
        direct_neg_clamp_tight_ir,
        direct_neg_clamp_tight_bindings,
        Neg,
        false,
        false
    );
    shape!(
        T,
        criterion,
        requirements,
        direct_relu_clamp_tight,
        direct_relu_clamp_tight_ir,
        direct_relu_clamp_tight_bindings,
        Relu,
        false,
        false
    );
    clamp_rejectsubnormalresult_grid::<T>(criterion, requirements);
    clamp_rejectsubnormalresult_broadcast::<T>(criterion, requirements);
}
fn clamp_rejectsubnormalresult_grid<T: oracle::Native>(
    criterion: &mut Criterion,
    requirements: PcuImplementationRequirements,
) {
    shape!(
        T,
        criterion,
        requirements,
        grid_neg_clamp_tight,
        grid_neg_clamp_tight_ir,
        grid_neg_clamp_tight_bindings,
        Neg,
        true,
        false
    );
    shape!(
        T,
        criterion,
        requirements,
        grid_relu_clamp_tight,
        grid_relu_clamp_tight_ir,
        grid_relu_clamp_tight_bindings,
        Relu,
        true,
        false
    );
}
fn clamp_rejectsubnormalresult_broadcast<T: oracle::Native>(
    criterion: &mut Criterion,
    requirements: PcuImplementationRequirements,
) {
    shape!(
        T,
        criterion,
        requirements,
        broadcast_neg_clamp_tight,
        broadcast_neg_clamp_tight_ir,
        broadcast_neg_clamp_tight_bindings,
        Neg,
        false,
        true
    );
    shape!(
        T,
        criterion,
        requirements,
        broadcast_relu_clamp_tight,
        broadcast_relu_clamp_tight_ir,
        broadcast_relu_clamp_tight_bindings,
        Relu,
        false,
        true
    );
}
fn width<T: oracle::Native>(
    criterion: &mut Criterion,
    requirements: PcuImplementationRequirements,
) {
    match (requirements.float_underflow, requirements.range_policy) {
        (PcuFloatUnderflowPolicy::IeeeAfterRounding, PcuRangePolicy::Reject) => {
            reject_ieeeafterrounding::<T>(criterion, requirements);
        }
        (PcuFloatUnderflowPolicy::AllowGradualUnderflow, PcuRangePolicy::Reject) => {
            reject_allowgradualunderflow::<T>(criterion, requirements);
        }
        (PcuFloatUnderflowPolicy::RejectSubnormalResult, PcuRangePolicy::Reject) => {
            reject_rejectsubnormalresult::<T>(criterion, requirements);
        }
        (PcuFloatUnderflowPolicy::IeeeAfterRounding, PcuRangePolicy::Clamp) => {
            clamp_ieeeafterrounding::<T>(criterion, requirements);
        }
        (PcuFloatUnderflowPolicy::AllowGradualUnderflow, PcuRangePolicy::Clamp) => {
            clamp_allowgradualunderflow::<T>(criterion, requirements);
        }
        (PcuFloatUnderflowPolicy::RejectSubnormalResult, PcuRangePolicy::Clamp) => {
            clamp_rejectsubnormalresult::<T>(criterion, requirements);
        }
    }
}

fn comparisons(criterion: &mut Criterion) {
    for numerical_mode in [PcuNumericalMode::Boundary, PcuNumericalMode::Strict] {
        for compound_arithmetic in [
            PcuCompoundArithmeticPolicy::Checked,
            PcuCompoundArithmeticPolicy::BackendDefined,
        ] {
            for precision in [
                PcuPrecisionPolicy::Preserve,
                PcuPrecisionPolicy::BackendOptimized,
            ] {
                for float_underflow in [
                    PcuFloatUnderflowPolicy::IeeeAfterRounding,
                    PcuFloatUnderflowPolicy::RejectSubnormalResult,
                    PcuFloatUnderflowPolicy::AllowGradualUnderflow,
                ] {
                    for range_policy in [PcuRangePolicy::Reject, PcuRangePolicy::Clamp] {
                        let requirements = PcuImplementationRequirements {
                            numerical_mode,
                            float_underflow,
                            range_policy,
                            numerical_options: pcu_facade::PcuNumericalOptions {
                                compound_arithmetic,
                                precision,
                                ..Default::default()
                            },
                        };
                        global::configure(global::PcuExecutionPolicy {
                            backend: global::PcuBackendChoice::Vulkan,
                            numerical_mode,
                            numerical_options: requirements.numerical_options,
                            float_underflow,
                            range_policy,
                            score_invocation: Some(score),
                            ..Default::default()
                        })
                        .unwrap();
                        global::clear_thread_cache().unwrap();
                        width::<pcu_facade::PcuF16Bits>(criterion, requirements);
                        width::<pcu_facade::PcuBf16Bits>(criterion, requirements);
                        width::<pcu_facade::PcuF8E4M3FnBits>(criterion, requirements);
                        width::<pcu_facade::PcuF8E5M2Bits>(criterion, requirements);
                        width::<f32>(criterion, requirements);
                        width::<f64>(criterion, requirements);
                    }
                }
            }
        }
    }
    global::clear_thread_cache().unwrap();
    global::use_defaults().unwrap();
}
criterion_group!(benches, comparisons);
criterion_main!(benches);
