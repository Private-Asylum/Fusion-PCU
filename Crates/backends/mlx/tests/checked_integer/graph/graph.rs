//! Independent hand-built cold integer graph, not source-emitted IR reuse.
#[rustfmt::skip]
use fusion_pcu_core::{
    PcuBinding,
    PcuBindingAccess,
    PcuBindingStorageClass,
    PcuDispatchControlOp,
    PcuDispatchDataOp,
    PcuDispatchEntryPoint,
    PcuDispatchFeatureCaps,
    PcuDispatchIntegerBinaryOp,
    PcuDispatchIndex,
    PcuDispatchKernelIr,
    PcuDispatchOp,
    PcuDispatchValueId,
    PcuImplementationRequirements,
    PcuKernelId,
    PcuNumericalMode,
    PcuRangePolicy,
    PcuScalar,
    PcuValueType,
    PcuValueTypeCaps,
};
#[allow(clippy::too_many_arguments)] // Exact operation/policy/shape are independent cold fixture dimensions.
pub fn fixture_profile<T: PcuScalar, R>(
    count: u32,
    op: PcuDispatchIntegerBinaryOp,
    range: PcuRangePolicy,
    grid: bool,
    broadcast: [bool; 2],
    visit: impl FnOnce(&PcuDispatchKernelIr<'_>) -> R,
) -> R {
    fixture_roles::<T, R>(count, op, range, grid, broadcast, false, visit)
}
#[allow(clippy::too_many_arguments)] // Actual repeated binding identity is independent of both index roles.
pub fn fixture_roles<T: PcuScalar, R>(
    count: u32,
    op: PcuDispatchIntegerBinaryOp,
    range: PcuRangePolicy,
    grid: bool,
    broadcast: [bool; 2],
    repeated: bool,
    visit: impl FnOnce(&PcuDispatchKernelIr<'_>) -> R,
) -> R {
    let bindings = [
        PcuBinding::scalar::<T>(
            Some("input"),
            2,
            3,
            PcuBindingStorageClass::Storage,
            PcuBindingAccess::ReadOnly,
        ),
        PcuBinding::scalar::<T>(
            Some("right"),
            3,
            2,
            PcuBindingStorageClass::Storage,
            PcuBindingAccess::ReadOnly,
        ),
        PcuBinding::scalar::<T>(
            Some("output"),
            4,
            1,
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
            index: if broadcast[0] {
                PcuDispatchIndex::BindingElementZero
            } else {
                index
            },
        }),
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
            result: PcuDispatchValueId(2),
            binding: bindings[usize::from(!repeated)].reference(),
            index: if broadcast[1] {
                PcuDispatchIndex::BindingElementZero
            } else {
                index
            },
        }),
        PcuDispatchOp::Data(PcuDispatchDataOp::CheckedIntegerBinary {
            result: PcuDispatchValueId(3),
            lhs: PcuDispatchValueId(1),
            rhs: PcuDispatchValueId(2),
            value_type: PcuValueType::Scalar(T::TYPE),
            op,
            range_policy: range,
        }),
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
            binding: bindings[2].reference(),
            index,
            value: PcuDispatchValueId(3),
        }),
    ];
    let direct = [
        body[0],
        body[1],
        body[2],
        body[3],
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
            range_policy: range,
            numerical_mode: PcuNumericalMode::Strict,
            ..PcuDispatchKernelIr::DEFAULT_REQUIREMENTS
        },
        id: PcuKernelId(112),
        entry: PcuDispatchEntryPoint {
            name: "checked-integer-binary",
            logical_shape: [if grid { 1 } else { count }, 1, 1],
        },
        bindings: &bindings,
        ports: &[],
        parameters: &[],
        ops: if grid { &looped } else { &direct },
        type_caps: PcuValueTypeCaps::for_scalar(T::TYPE),
        feature_caps: PcuDispatchFeatureCaps::empty(),
    })
}
