//! Matched detached source graph and test/benchmark policy fixtures.

#[rustfmt::skip]
use pcu_facade::{
    PcuBinding,
    PcuBindingAccess,
    PcuBindingRef,
    PcuBindingStorageClass,
    PcuDispatchControlOp,
    PcuDispatchDataOp,
    PcuDispatchEntryPoint,
    PcuDispatchFeatureCaps,
    PcuDispatchFloatUnaryOp,
    PcuDispatchIndex,
    PcuDispatchKernelIr,
    PcuDispatchOp,
    PcuDispatchValueId,
    PcuFloatUnderflowPolicy,
    PcuHostKernelBackend,
    PcuKernelId,
    PcuRangePolicy,
    PcuScalarType,
    PcuValueType,
    PcuValueTypeCaps,
};
#[rustfmt::skip]
use fusion_pcu_vulkan::{
    PcuVulkanBackend,
    PcuVulkanError,
    PcuVulkanPreparedBitMap,
};

pub fn prepare_graph(
    backend: &PcuVulkanBackend,
    extent: u32,
    underflow: PcuFloatUnderflowPolicy,
    range: PcuRangePolicy,
    grid: bool,
) -> Result<PcuVulkanPreparedBitMap, PcuVulkanError> {
    with_graph(extent, underflow, range, grid, |kernel| {
        backend.prepare_host_kernel(kernel)
    })
}

pub fn with_graph<R>(
    extent: u32,
    underflow: PcuFloatUnderflowPolicy,
    range: PcuRangePolicy,
    grid: bool,
    run: impl FnOnce(&PcuDispatchKernelIr<'_>) -> R,
) -> R {
    with_typed_graph(extent, underflow, range, grid, PcuScalarType::F32, run)
}

pub fn with_typed_graph<R>(
    extent: u32,
    underflow: PcuFloatUnderflowPolicy,
    range: PcuRangePolicy,
    grid: bool,
    scalar: PcuScalarType,
    run: impl FnOnce(&PcuDispatchKernelIr<'_>) -> R,
) -> R {
    let bindings = [
        PcuBinding::value(
            Some("input"),
            0,
            0,
            PcuBindingStorageClass::Storage,
            PcuBindingAccess::ReadOnly,
            PcuValueType::Scalar(scalar),
        ),
        PcuBinding::value(
            Some("output"),
            0,
            1,
            PcuBindingStorageClass::Storage,
            PcuBindingAccess::WriteOnly,
            PcuValueType::Scalar(scalar),
        ),
    ];
    let index = if grid {
        PcuDispatchIndex::GridStrideId
    } else {
        PcuDispatchIndex::InvocationId
    };
    let body = [
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
            result: PcuDispatchValueId(1),
            binding: PcuBindingRef::new(0, 0),
            index,
        }),
        PcuDispatchOp::Data(PcuDispatchDataOp::CheckedFloatUnary {
            value_type: PcuValueType::Scalar(scalar),
            op: PcuDispatchFloatUnaryOp::Neg,
            underflow_policy: underflow,
            range_policy: range,
            result: PcuDispatchValueId(2),
            value: PcuDispatchValueId(1),
        }),
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
            binding: PcuBindingRef::new(0, 1),
            index,
            value: PcuDispatchValueId(2),
        }),
    ];
    let direct = [
        body[0],
        body[1],
        body[2],
        PcuDispatchOp::Control(PcuDispatchControlOp::Return),
    ];
    let grid_ops = [
        PcuDispatchOp::GridStrideLoop {
            extent,
            body: &body,
        },
        PcuDispatchOp::Control(PcuDispatchControlOp::Return),
    ];
    run(&PcuDispatchKernelIr {
        id: PcuKernelId(90),
        entry: PcuDispatchEntryPoint {
            name: "checked_neg",
            logical_shape: [if grid { 2 } else { extent }, 1, 1],
        },
        bindings: &bindings,
        ports: &[],
        parameters: &[],
        ops: if grid { &grid_ops } else { &direct },
        type_caps: if scalar == PcuScalarType::F64 {
            PcuValueTypeCaps::FLOAT64
        } else {
            PcuValueTypeCaps::FLOAT32
        },
        feature_caps: PcuDispatchFeatureCaps::default(),
    })
}
