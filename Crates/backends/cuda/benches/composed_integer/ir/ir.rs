//! Explicit typed staged/dead IR authored independently of the procedural source lowering.
#[rustfmt::skip]
use fusion_pcu::{
 PcuBinding,PcuBindingRef,PcuDispatchKernelIr,PcuDispatchEntryPoint,PcuDispatchOp,
 PcuDispatchDataOp,PcuDispatchControlOp,PcuDispatchIndex,PcuDispatchIntegerBinaryOp,
 PcuDispatchValueId,PcuValueType,PcuScalarType,PcuKernelId,PcuValueTypeCaps,
 PcuDispatchFeatureCaps,PcuImplementationRequirements,
};
pub fn body<'a>(
    scalar: PcuScalarType,
    grid: bool,
    dead: bool,
    request: PcuImplementationRequirements,
) -> Vec<PcuDispatchOp<'a>> {
    let index = if grid {
        PcuDispatchIndex::GridStrideId
    } else {
        PcuDispatchIndex::InvocationId
    };
    let load = |v, b, index| {
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
            result: PcuDispatchValueId(v),
            binding: PcuBindingRef::new(0, b),
            index,
        })
    };
    let store = |b, v| {
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
            binding: PcuBindingRef::new(0, b),
            index,
            value: PcuDispatchValueId(v),
        })
    };
    let checked = |v, a, b, op| {
        PcuDispatchOp::Data(PcuDispatchDataOp::CheckedIntegerBinary {
            result: PcuDispatchValueId(v),
            lhs: PcuDispatchValueId(a),
            rhs: PcuDispatchValueId(b),
            value_type: PcuValueType::Scalar(scalar),
            op,
            range_policy: request.range_policy,
        })
    };
    let mut ops = vec![
        load(0, 0, index),
        load(1, 1, PcuDispatchIndex::BindingElementZero),
        checked(2, 0, 1, PcuDispatchIntegerBinaryOp::Add),
    ];
    if dead {
        ops.push(store(2, 0));
    } else {
        ops.extend([
            store(2, 2),
            load(3, 2, index),
            checked(4, 3, 0, PcuDispatchIntegerBinaryOp::Mul),
            checked(5, 4, 0, PcuDispatchIntegerBinaryOp::Sub),
            store(3, 5),
        ]);
    }
    ops
}
pub const fn kernel<'a>(
    scalar: PcuScalarType,
    n: u32,
    bindings: &'a [PcuBinding<'a>],
    ops: &'a [PcuDispatchOp<'a>],
    request: PcuImplementationRequirements,
) -> PcuDispatchKernelIr<'a> {
    PcuDispatchKernelIr {
        numerical_requirements: request,
        id: PcuKernelId(0x4349_5242),
        entry: PcuDispatchEntryPoint {
            name: "explicit_composed_integer",
            logical_shape: [n, 1, 1],
        },
        bindings,
        ports: &[],
        parameters: &[],
        ops,
        type_caps: PcuValueTypeCaps::for_scalar(scalar),
        feature_caps: PcuDispatchFeatureCaps::READ_ONLY_RESOURCES
            .union(PcuDispatchFeatureCaps::MUTABLE_RESOURCES),
    }
}
pub const RETURN: PcuDispatchOp<'static> = PcuDispatchOp::Control(PcuDispatchControlOp::Return);
