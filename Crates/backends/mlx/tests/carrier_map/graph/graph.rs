//! Independent hand-built copy/repetition IR, with nonordinal binding roles.
#[rustfmt::skip]
use pcu_facade::{PcuScalar,PcuBinding,PcuBindingAccess,PcuBindingStorageClass,
    PcuDispatchDataOp,PcuDispatchOp,PcuDispatchControlOp,PcuDispatchValueId,PcuDispatchIndex,
    PcuDispatchKernelIr,PcuDispatchEntryPoint,PcuKernelId,PcuValueTypeCaps,PcuDispatchFeatureCaps};
pub fn fixture<T: PcuScalar, R>(
    count: u32,
    broadcast: bool,
    grid: bool,
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
            index: if broadcast {
                PcuDispatchIndex::BindingElementZero
            } else {
                index
            },
        }),
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
            binding: bindings[1].reference(),
            index,
            value: PcuDispatchValueId(1),
        }),
    ];
    let direct = [
        body[0],
        body[1],
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
        numerical_requirements: PcuDispatchKernelIr::DEFAULT_REQUIREMENTS,
        id: PcuKernelId(127),
        entry: PcuDispatchEntryPoint {
            name: "carrier-copy",
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
