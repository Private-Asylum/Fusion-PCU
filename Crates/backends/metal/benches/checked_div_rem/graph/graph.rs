//! Independent four-binding typed SSA, preserving numerical mode and direct/grid extents.
#[rustfmt::skip]
use fusion_pcu_core::{PcuScalar,PcuBinding,PcuBindingStorageClass,PcuBindingAccess,PcuDispatchOp,PcuDispatchDataOp,
 PcuDispatchValueId,PcuDispatchIndex,PcuValueType,PcuDispatchControlOp,PcuDispatchKernelIr,
 PcuKernelId,PcuDispatchEntryPoint,PcuValueTypeCaps,PcuDispatchFeatureCaps,PcuImplementationRequirements,PcuNumericalMode};
use fusion_pcu_core::model::PcuIntegerDivFlags;
#[allow(clippy::too_many_lines)] // Keep independent four-binding SSA and both frozen indexing regions together.
pub fn fixture<T: PcuScalar, R>(
    count: u32,
    grid: bool,
    strict: bool,
    visit: impl FnOnce(&PcuDispatchKernelIr<'_>) -> R,
) -> R {
    let bindings = [
        PcuBinding::scalar::<T>(
            Some("left"),
            2,
            3,
            PcuBindingStorageClass::Storage,
            PcuBindingAccess::ReadOnly,
        ),
        PcuBinding::scalar::<T>(
            Some("right"),
            2,
            7,
            PcuBindingStorageClass::Storage,
            PcuBindingAccess::ReadOnly,
        ),
        PcuBinding::scalar::<T>(
            Some("quotient"),
            4,
            1,
            PcuBindingStorageClass::Storage,
            PcuBindingAccess::WriteOnly,
        ),
        PcuBinding::scalar::<T>(
            Some("remainder"),
            4,
            6,
            PcuBindingStorageClass::Storage,
            PcuBindingAccess::WriteOnly,
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
            lhs: PcuDispatchValueId(1),
            rhs: PcuDispatchValueId(2),
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
        numerical_requirements: PcuImplementationRequirements {
            numerical_mode: if strict {
                PcuNumericalMode::Strict
            } else {
                PcuNumericalMode::Boundary
            },
            ..PcuDispatchKernelIr::DEFAULT_REQUIREMENTS
        },
        id: PcuKernelId(118),
        entry: PcuDispatchEntryPoint {
            name: "checked-div-rem",
            logical_shape: [if grid { 3 } else { count }, 1, 1],
        },
        bindings: &bindings,
        ports: &[],
        parameters: &[],
        ops: if grid { &looped } else { &direct },
        type_caps: PcuValueTypeCaps::for_scalar(T::TYPE),
        feature_caps: PcuDispatchFeatureCaps::empty(),
    })
}
