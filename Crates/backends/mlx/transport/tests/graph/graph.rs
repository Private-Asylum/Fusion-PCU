//! Independent handwritten transport graph, preserving a saved value across overwrite.
#[rustfmt::skip]
use fusion_pcu_core::{
    PcuBinding,
    PcuBindingAccess as Access,
    PcuBindingRef,
    PcuBindingStorageClass as Storage,
    PcuDispatchControlOp as Control,
    PcuDispatchDataOp as Data,
    PcuDispatchEntryPoint,
    PcuDispatchFeatureCaps,
    PcuDispatchIndex as Index,
    PcuDispatchKernelIr,
    PcuDispatchOp as Op,
    PcuDispatchValueId as Id,
    PcuImplementationRequirements,
    PcuKernelId,
    PcuScalarType,
    PcuValueType,
    PcuValueTypeCaps,
};
pub const INPUT: PcuBindingRef = PcuBindingRef::new(2, 9);
pub const SEED: PcuBindingRef = PcuBindingRef::new(0, 5);
pub const STAGE: PcuBindingRef = PcuBindingRef::new(8, 1);
pub const OUTPUT: PcuBindingRef = PcuBindingRef::new(3, 7);
pub const GHOST: PcuBindingRef = PcuBindingRef::new(6, 4);

pub fn visit<R>(
    scalar: PcuScalarType,
    count: u32,
    grid: bool,
    requirements: PcuImplementationRequirements,
    visit: impl FnOnce(&PcuDispatchKernelIr<'_>) -> R,
) -> R {
    let declarations = [
        (GHOST, Access::ReadWrite),
        (OUTPUT, Access::ReadWrite),
        (SEED, Access::ReadOnly),
        (STAGE, Access::ReadWrite),
        (INPUT, Access::ReadOnly),
    ]
    .map(|(binding, access)| {
        PcuBinding::value(
            None,
            binding.set,
            binding.binding,
            Storage::Storage,
            access,
            PcuValueType::Scalar(scalar),
        )
    });
    let index = if grid {
        Index::GridStrideId
    } else {
        Index::InvocationId
    };
    let body = [
        Op::Data(Data::BindingLoad {
            result: Id(0),
            binding: INPUT,
            index,
        }),
        Op::Data(Data::BindingStore {
            binding: STAGE,
            index,
            value: Id(0),
        }),
        Op::Data(Data::BindingLoad {
            result: Id(1),
            binding: STAGE,
            index,
        }),
        Op::Data(Data::BindingLoad {
            result: Id(2),
            binding: SEED,
            index: Index::BindingElementZero,
        }),
        Op::Data(Data::BindingStore {
            binding: STAGE,
            index,
            value: Id(2),
        }),
        Op::Data(Data::BindingStore {
            binding: OUTPUT,
            index,
            value: Id(1),
        }),
    ];
    let mut direct = body.to_vec();
    direct.push(Op::Control(Control::Return));
    let wrapped = [
        Op::GridStrideLoop {
            extent: count,
            body: &body,
        },
        Op::Control(Control::Return),
    ];
    visit(&PcuDispatchKernelIr {
        id: PcuKernelId(941),
        entry: PcuDispatchEntryPoint {
            name: "saved-byte-transport",
            logical_shape: [if grid { 3 } else { count }, 1, 1],
        },
        bindings: &declarations,
        ops: if grid { &wrapped } else { &direct },
        ports: &[],
        parameters: &[],
        numerical_requirements: requirements,
        type_caps: PcuValueTypeCaps::empty(),
        feature_caps: PcuDispatchFeatureCaps::empty(),
    })
}

/// Four actual original snapshots, including both writable banks before any store.
pub fn visit_four<R>(
    scalar: PcuScalarType,
    count: u32,
    grid: bool,
    requirements: PcuImplementationRequirements,
    visit: impl FnOnce(&PcuDispatchKernelIr<'_>) -> R,
) -> R {
    let declarations = [
        (GHOST, Access::ReadWrite),
        (OUTPUT, Access::ReadWrite),
        (SEED, Access::ReadOnly),
        (STAGE, Access::ReadWrite),
        (INPUT, Access::ReadOnly),
    ]
    .map(|(binding, access)| {
        PcuBinding::value(
            None,
            binding.set,
            binding.binding,
            Storage::Storage,
            access,
            PcuValueType::Scalar(scalar),
        )
    });
    let index = if grid {
        Index::GridStrideId
    } else {
        Index::InvocationId
    };
    let body = [
        Op::Data(Data::BindingLoad {
            result: Id(0),
            binding: INPUT,
            index,
        }),
        Op::Data(Data::BindingLoad {
            result: Id(1),
            binding: SEED,
            index: Index::BindingElementZero,
        }),
        Op::Data(Data::BindingLoad {
            result: Id(2),
            binding: STAGE,
            index,
        }),
        Op::Data(Data::BindingLoad {
            result: Id(3),
            binding: OUTPUT,
            index,
        }),
        Op::Data(Data::BindingStore {
            binding: STAGE,
            index,
            value: Id(1),
        }),
        Op::Data(Data::BindingStore {
            binding: OUTPUT,
            index,
            value: Id(2),
        }),
    ];
    let mut direct = body.to_vec();
    direct.push(Op::Control(Control::Return));
    let wrapped = [
        Op::GridStrideLoop {
            extent: count,
            body: &body,
        },
        Op::Control(Control::Return),
    ];
    visit(&PcuDispatchKernelIr {
        id: PcuKernelId(942),
        entry: PcuDispatchEntryPoint {
            name: "four-initial-byte-transport",
            logical_shape: [if grid { 3 } else { count }, 1, 1],
        },
        bindings: &declarations,
        ops: if grid { &wrapped } else { &direct },
        ports: &[],
        parameters: &[],
        numerical_requirements: requirements,
        type_caps: PcuValueTypeCaps::empty(),
        feature_caps: PcuDispatchFeatureCaps::empty(),
    })
}
