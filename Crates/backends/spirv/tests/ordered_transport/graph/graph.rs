//! Explicit graph independent of source emitter ordering/SSA IDs.
#[rustfmt::skip]
use fusion_pcu_core::{
    PcuBinding,
    PcuBindingAccess,
    PcuBindingRef,
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
    PcuScalarType,
    PcuValueType,
    PcuValueTypeCaps,
};
pub fn with<R>(
    scalar: PcuScalarType,
    extent: u32,
    grid: bool,
    requirements: PcuImplementationRequirements,
    run: impl FnOnce(&PcuDispatchKernelIr<'_>) -> R,
) -> R {
    let ty = PcuValueType::Scalar(scalar);
    let bindings = core::array::from_fn::<_, 5, _>(|index| {
        PcuBinding::value(
            None,
            0,
            u32::try_from(index).unwrap(),
            PcuBindingStorageClass::Storage,
            if index < 2 {
                PcuBindingAccess::ReadOnly
            } else {
                PcuBindingAccess::ReadWrite
            },
            ty,
        )
    });
    let index = if grid {
        PcuDispatchIndex::GridStrideId
    } else {
        PcuDispatchIndex::InvocationId
    };
    let body = [
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
            result: PcuDispatchValueId(7),
            binding: PcuBindingRef::new(0, 0),
            index,
        }),
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
            binding: PcuBindingRef::new(0, 3),
            index,
            value: PcuDispatchValueId(7),
        }),
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
            result: PcuDispatchValueId(31),
            binding: PcuBindingRef::new(0, 3),
            index,
        }),
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
            result: PcuDispatchValueId(65),
            binding: PcuBindingRef::new(0, 1),
            index: PcuDispatchIndex::BindingElementZero,
        }),
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
            binding: PcuBindingRef::new(0, 3),
            index,
            value: PcuDispatchValueId(65),
        }),
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
            binding: PcuBindingRef::new(0, 4),
            index,
            value: PcuDispatchValueId(31),
        }),
    ];
    let direct = [
        body[0],
        body[1],
        body[2],
        body[3],
        body[4],
        body[5],
        PcuDispatchOp::Control(PcuDispatchControlOp::Return),
    ];
    let loop_ops = [
        PcuDispatchOp::GridStrideLoop {
            extent,
            body: &body,
        },
        PcuDispatchOp::Control(PcuDispatchControlOp::Return),
    ];
    run(&PcuDispatchKernelIr {
        numerical_requirements: requirements,
        id: PcuKernelId(19456),
        entry: PcuDispatchEntryPoint {
            name: "raw_saved_ssa",
            logical_shape: [if grid { 3 } else { extent }, 1, 1],
        },
        bindings: &bindings,
        ports: &[],
        parameters: &[],
        ops: if grid { &loop_ops } else { &direct },
        type_caps: PcuValueTypeCaps::empty(),
        feature_caps: PcuDispatchFeatureCaps::empty(),
    })
}
