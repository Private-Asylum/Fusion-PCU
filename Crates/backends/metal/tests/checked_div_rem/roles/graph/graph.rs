//! Independent four-binding direct/canonical-grid quotient/remainder IR.
#[path = "roles/roles.rs"]
pub mod roles;
#[rustfmt::skip]
use pcu_facade::{
    PcuBinding,
    PcuBindingAccess,
    PcuBindingStorageClass,
    PcuDispatchControlOp,
    PcuDispatchDataOp,
    PcuDispatchEntryPoint,
    PcuDispatchFeatureCaps,
    PcuDispatchIndex,
    PcuDispatchKernelIr,
    PcuDispatchOp,
    PcuDispatchValueId,
    PcuImplementationRequirements,
    PcuKernelId,
    PcuScalar,
    PcuValueType,
    PcuValueTypeCaps,
};
use fusion_pcu_core::model::PcuIntegerDivFlags;
pub fn fixture<T: PcuScalar, R>(
    count: u32,
    grid: bool,
    reverse: bool,
    request: PcuImplementationRequirements,
    visit: impl FnOnce(&PcuDispatchKernelIr<'_>) -> R,
) -> R {
    let binding = |name, number, access| {
        PcuBinding::scalar::<T>(
            Some(name),
            2,
            number,
            PcuBindingStorageClass::Storage,
            access,
        )
    };
    let bindings = [
        binding("left", 3, PcuBindingAccess::ReadOnly),
        binding("right", 7, PcuBindingAccess::ReadOnly),
        binding("quotient", 9, PcuBindingAccess::ReadWrite),
        binding("remainder", 1, PcuBindingAccess::ReadWrite),
    ];
    let index = if grid {
        PcuDispatchIndex::GridStrideId
    } else {
        PcuDispatchIndex::InvocationId
    };
    let body = [
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
            result: PcuDispatchValueId(1),
            binding: bindings[0].reference(),
            index,
        }),
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
            result: PcuDispatchValueId(2),
            binding: bindings[1].reference(),
            index,
        }),
        PcuDispatchOp::Data(PcuDispatchDataOp::CheckedDivRem {
            value_type: PcuValueType::Scalar(T::TYPE),
            flags: PcuIntegerDivFlags::CHECKED,
            quotient: PcuDispatchValueId(3),
            remainder: PcuDispatchValueId(4),
            lhs: PcuDispatchValueId(if reverse { 2 } else { 1 }),
            rhs: PcuDispatchValueId(if reverse { 1 } else { 2 }),
        }),
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
            binding: bindings[2].reference(),
            index,
            value: PcuDispatchValueId(3),
        }),
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
            binding: bindings[3].reference(),
            index,
            value: PcuDispatchValueId(4),
        }),
    ];
    let direct = [
        body[0],
        body[1],
        body[2],
        body[3],
        body[4],
        PcuDispatchOp::Control(PcuDispatchControlOp::Return),
    ];
    let looped = [
        PcuDispatchOp::GridStrideLoop {
            extent: count,
            body: &body,
        },
        PcuDispatchOp::Control(PcuDispatchControlOp::Return),
    ];
    visit(&PcuDispatchKernelIr {
        id: PcuKernelId(139),
        entry: PcuDispatchEntryPoint {
            name: "checked-div-rem",
            logical_shape: [if grid { 1 } else { count }, 1, 1],
        },
        numerical_requirements: request,
        bindings: &bindings,
        ports: &[],
        parameters: &[],
        ops: if grid { &looped } else { &direct },
        type_caps: PcuValueTypeCaps::for_scalar(T::TYPE),
        feature_caps: PcuDispatchFeatureCaps::empty(),
    })
}
