use super::*;
use fusion_pcu::{
    PcuBinding, PcuBindingAccess, PcuBindingStorageClass, PcuDispatchEntryPoint,
    PcuDispatchControlOp, PcuImplementationRequirements, PcuKernelId, PcuValueType,
    PcuValueTypeCaps, PcuDispatchFeatureCaps, PcuScalarType,
};
fn kernel<'a>(
    bindings: &'a [PcuBinding<'a>],
    ops: &'a [PcuDispatchOp<'a>],
) -> PcuDispatchKernelIr<'a> {
    PcuDispatchKernelIr {
        numerical_requirements: PcuImplementationRequirements::default(),
        id: PcuKernelId(19),
        entry: PcuDispatchEntryPoint {
            name: "cyclic_cold_refusal",
            logical_shape: [7, 1, 1],
        },
        bindings,
        ops,
        ports: &[],
        parameters: &[],
        type_caps: PcuValueTypeCaps::FLOAT32 | PcuValueTypeCaps::FLOAT64,
        feature_caps: PcuDispatchFeatureCaps::empty(),
    }
}
#[test]
fn cyclic_dispatch_public_cold_profiles_refuse_without_device_or_recursive_scan() {
    static CYCLE: [PcuDispatchOp<'static>; 1] = [PcuDispatchOp::GridStrideLoop {
        extent: 7,
        body: &CYCLE,
    }];
    let bindings = [
        PcuBinding::value(
            None,
            0,
            0,
            PcuBindingStorageClass::Storage,
            PcuBindingAccess::ReadOnly,
            PcuValueType::f64(),
        ),
        PcuBinding::value(
            None,
            0,
            1,
            PcuBindingStorageClass::Storage,
            PcuBindingAccess::ReadWrite,
            PcuValueType::f32(),
        ),
    ];
    let wrapped = [
        PcuDispatchOp::GridStrideLoop {
            extent: 7,
            body: &CYCLE,
        },
        PcuDispatchOp::Control(PcuDispatchControlOp::Return),
    ];
    for ops in [&CYCLE[..], &wrapped[..]] {
        let kernel = kernel(&bindings, ops);
        assert!(require_non_nested(&kernel).is_err());
        assert!(crate::MetalCheckedConversionPlan::assess(&kernel).is_err());
        assert!(crate::MetalCheckedMapPlan::assess(&kernel, PcuScalarType::F32).is_err());
        assert!(crate::MetalTransportPlan::assess(&kernel, PcuScalarType::F32).is_err());
        assert!(crate::MetalCheckedUnaryPlan::assess(&kernel).is_err());
        assert!(crate::MetalPortableUnaryPlan::assess(&kernel).is_err());
        assert!(crate::MetalDivRemRolePlan::assess(&kernel).is_err());
    }
}
#[test]
fn non_nested_query_preserves_direct_and_multiple_finite_outer_bodies() {
    let leaf = [PcuDispatchOp::Control(PcuDispatchControlOp::Return)];
    let loops = [
        PcuDispatchOp::GridStrideLoop {
            extent: 7,
            body: &leaf,
        },
        PcuDispatchOp::GridStrideLoop {
            extent: 7,
            body: &leaf,
        },
    ];
    for ops in [&leaf[..], &loops[..]] {
        assert!(require_non_nested(&kernel(&[], ops)).is_ok());
    }
}
