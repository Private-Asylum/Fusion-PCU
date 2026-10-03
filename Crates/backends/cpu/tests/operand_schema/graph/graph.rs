//! Independent typed diagnostic for one loaded binding, unused declarations and per-operand indices.
#[rustfmt::skip]
use pcu_facade::{PcuBinding,PcuBindingRef,PcuBindingAccess,PcuBindingStorageClass,PcuDispatchKernelIr,PcuDispatchEntryPoint,PcuDispatchFeatureCaps,PcuValueTypeCaps,PcuValueType,PcuScalarType,PcuDispatchOp,PcuDispatchDataOp,PcuDispatchControlOp,PcuDispatchFloatBinaryOp,PcuDispatchIndex,PcuDispatchValueId,PcuKernelId};
pub const INPUT: PcuBindingRef = PcuBindingRef::new(7, 9);
pub const UNUSED: PcuBindingRef = PcuBindingRef::new(8, 3);
pub const OUTPUT: PcuBindingRef = PcuBindingRef::new(9, 5);
pub fn with<R>(
    scalar: PcuScalarType,
    extent: u32,
    schema: usize,
    run: impl FnOnce(&PcuDispatchKernelIr<'_>) -> R,
) -> R {
    let value_type = PcuValueType::Scalar(scalar);
    let binding = |reference: PcuBindingRef, access| {
        PcuBinding::value(
            None,
            reference.set,
            reference.binding,
            PcuBindingStorageClass::Storage,
            access,
            value_type,
        )
    };
    let bindings = match schema {
        0 | 3 => [
            binding(OUTPUT, PcuBindingAccess::ReadWrite),
            binding(INPUT, PcuBindingAccess::ReadOnly),
            binding(UNUSED, PcuBindingAccess::ReadOnly),
        ],
        1 => [
            binding(INPUT, PcuBindingAccess::ReadOnly),
            binding(UNUSED, PcuBindingAccess::ReadOnly),
            binding(OUTPUT, PcuBindingAccess::ReadWrite),
        ],
        _ => [
            binding(UNUSED, PcuBindingAccess::ReadOnly),
            binding(OUTPUT, PcuBindingAccess::ReadWrite),
            binding(INPUT, PcuBindingAccess::ReadOnly),
        ],
    };
    let grid = schema == 4;
    let index = if grid {
        PcuDispatchIndex::GridStrideId
    } else {
        PcuDispatchIndex::InvocationId
    };
    let requirements = PcuDispatchKernelIr::DEFAULT_REQUIREMENTS;
    let body = [
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
            result: PcuDispatchValueId(11),
            binding: INPUT,
            index,
        }),
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
            result: PcuDispatchValueId(7),
            binding: INPUT,
            index: if schema == 3 {
                PcuDispatchIndex::BindingElementZero
            } else {
                index
            },
        }),
        PcuDispatchOp::Data(PcuDispatchDataOp::CheckedFloatBinary {
            result: PcuDispatchValueId(15),
            op: operation(schema),
            value_type,
            underflow_policy: requirements.float_underflow,
            range_policy: requirements.range_policy,
            lhs: PcuDispatchValueId(11),
            rhs: PcuDispatchValueId(7),
        }),
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
            binding: OUTPUT,
            index,
            value: PcuDispatchValueId(15),
        }),
    ];
    let direct = [
        body[0],
        body[1],
        body[2],
        body[3],
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
        numerical_requirements: requirements,
        id: PcuKernelId(1),
        entry: PcuDispatchEntryPoint {
            name: "operand_schema",
            logical_shape: [if grid { 1 } else { extent }, 1, 1],
        },
        bindings: if schema == 0 || schema == 3 {
            &bindings[..2]
        } else {
            &bindings
        },
        ops: if grid { &grid_ops } else { &direct },
        ports: &[],
        parameters: &[],
        type_caps: PcuValueTypeCaps::for_scalar(scalar),
        feature_caps: PcuDispatchFeatureCaps::empty(),
    })
}

const fn operation(schema: usize) -> PcuDispatchFloatBinaryOp {
    match schema {
        0 => PcuDispatchFloatBinaryOp::Add,
        1 => PcuDispatchFloatBinaryOp::Mul,
        2 => PcuDispatchFloatBinaryOp::Sub,
        _ => PcuDispatchFloatBinaryOp::Div,
    }
}
