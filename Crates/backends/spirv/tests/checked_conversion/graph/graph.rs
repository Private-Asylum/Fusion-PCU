//! Independent mixed-width SSA diagnostic with arbitrary original binding references.
#[rustfmt::skip]
use fusion_pcu_core::{PcuBinding,PcuBindingRef,PcuBindingAccess,PcuBindingStorageClass,PcuDispatchKernelIr,PcuDispatchEntryPoint,PcuDispatchFeatureCaps,PcuValueTypeCaps,PcuValueType,PcuScalarType,PcuDispatchOp,PcuDispatchDataOp,PcuDispatchControlOp,PcuDispatchCheckedFloatConversion,PcuDispatchIndex,PcuDispatchValueId,PcuKernelId,PcuRangePolicy,PcuFloatUnderflowPolicy,PcuImplementationRequirements};
pub const INPUT: PcuBindingRef = PcuBindingRef::new(7, 9);
pub const OUTPUT: PcuBindingRef = PcuBindingRef::new(9, 5);
pub fn with<R>(
    conversion: PcuDispatchCheckedFloatConversion,
    extent: u32,
    range: PcuRangePolicy,
    policy: PcuFloatUnderflowPolicy,
    broadcast: bool,
    grid: bool,
    run: impl FnOnce(&PcuDispatchKernelIr<'_>) -> R,
) -> R {
    let (source, destination) = match conversion {
        PcuDispatchCheckedFloatConversion::F32ToF64 => (PcuScalarType::F32, PcuScalarType::F64),
        PcuDispatchCheckedFloatConversion::F64ToF32 => (PcuScalarType::F64, PcuScalarType::F32),
    };
    let binding = |reference: PcuBindingRef, access, scalar| {
        PcuBinding::value(
            None,
            reference.set,
            reference.binding,
            PcuBindingStorageClass::Storage,
            access,
            PcuValueType::Scalar(scalar),
        )
    };
    let bindings = [
        binding(OUTPUT, PcuBindingAccess::ReadWrite, destination),
        binding(INPUT, PcuBindingAccess::ReadOnly, source),
    ];
    let index = if grid {
        PcuDispatchIndex::GridStrideId
    } else {
        PcuDispatchIndex::InvocationId
    };
    let body = [
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
            result: PcuDispatchValueId(11),
            binding: INPUT,
            index: if broadcast {
                PcuDispatchIndex::BindingElementZero
            } else {
                index
            },
        }),
        PcuDispatchOp::Data(PcuDispatchDataOp::CheckedFloatConvert {
            result: PcuDispatchValueId(7),
            value: PcuDispatchValueId(11),
            conversion,
            range_policy: range,
            underflow_policy: policy,
        }),
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
            binding: OUTPUT,
            index,
            value: PcuDispatchValueId(7),
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
        numerical_requirements: PcuImplementationRequirements {
            range_policy: range,
            float_underflow: policy,
            ..PcuDispatchKernelIr::DEFAULT_REQUIREMENTS
        },
        id: PcuKernelId(1),
        entry: PcuDispatchEntryPoint {
            name: "conversion_diagnostic",
            logical_shape: [if grid { 1 } else { extent }, 1, 1],
        },
        bindings: &bindings,
        ops: if grid { &grid_ops } else { &direct },
        ports: &[],
        parameters: &[],
        type_caps: PcuValueTypeCaps::FLOAT32 | PcuValueTypeCaps::FLOAT64,
        feature_caps: if range == PcuRangePolicy::Clamp {
            PcuDispatchFeatureCaps::RANGE_CLAMP
        } else {
            PcuDispatchFeatureCaps::empty()
        },
    })
}
