#[rustfmt::skip]
use super::{
    validate_checked_float_map_kernel,
    CheckedFloatMapValidationError as Error,
};
#[rustfmt::skip]
use crate::{
    PcuBinding,
    PcuBindingAccess,
    PcuBindingRef,
    PcuBindingStorageClass,
    PcuDispatchControlOp,
    PcuDispatchDataOp as Data,
    PcuDispatchEntryPoint,
    PcuDispatchFloatBinaryOp as Binary,
    PcuDispatchFloatUnaryOp as Unary,
    PcuDispatchIndex,
    PcuDispatchKernelIr,
    PcuDispatchOp as Op,
    PcuDispatchValueId as Id,
    PcuFloatUnderflowPolicy as Policy,
    PcuKernelId,
    PcuParameterValue,
    PcuValueType,
    PcuValueTypeCaps,
};

fn fixture<'a>(bindings: &'a [PcuBinding], ops: &'a [Op<'a>]) -> PcuDispatchKernelIr<'a> {
    PcuDispatchKernelIr {
        id: PcuKernelId(3),
        entry: PcuDispatchEntryPoint {
            name: "checked_float_map",
            logical_shape: [1, 1, 1],
        },
        bindings,
        ports: &[],
        parameters: &[],
        ops,
        type_caps: PcuValueTypeCaps::empty(),
        feature_caps: crate::PcuDispatchFeatureCaps::empty(),
    }
}

fn bindings(value_type: PcuValueType) -> [PcuBinding<'static>; 3] {
    [
        PcuBinding::value(
            Some("input"),
            0,
            0,
            PcuBindingStorageClass::Storage,
            PcuBindingAccess::ReadOnly,
            value_type,
        ),
        PcuBinding::value(
            Some("broadcast"),
            0,
            1,
            PcuBindingStorageClass::Storage,
            PcuBindingAccess::ReadOnly,
            value_type,
        ),
        PcuBinding::value(
            Some("output"),
            0,
            2,
            PcuBindingStorageClass::Storage,
            PcuBindingAccess::WriteOnly,
            value_type,
        ),
    ]
}

fn accepted_body(value_type: PcuValueType, index: PcuDispatchIndex) -> [Op<'static>; 6] {
    let constant = if value_type == PcuValueType::f32() {
        PcuParameterValue::F32(1.0_f32.to_bits())
    } else {
        PcuParameterValue::F64(1.0_f64.to_bits())
    };
    [
        Op::Data(Data::BindingLoad {
            result: Id(0),
            binding: PcuBindingRef::new(0, 0),
            index,
        }),
        Op::Data(Data::BindingLoad {
            result: Id(1),
            binding: PcuBindingRef::new(0, 1),
            index: PcuDispatchIndex::BindingElementZero,
        }),
        Op::Data(Data::Constant {
            result: Id(2),
            value: constant,
        }),
        Op::Data(Data::CheckedFloatBinary {
            value_type,
            op: Binary::Add,
            underflow_policy: Policy::IeeeAfterRounding,
            range_policy: crate::PcuRangePolicy::Reject,
            result: Id(3),
            lhs: Id(0),
            rhs: Id(1),
        }),
        Op::Data(Data::CheckedFloatBinary {
            value_type,
            op: Binary::Div,
            underflow_policy: Policy::AllowGradualUnderflow,
            range_policy: crate::PcuRangePolicy::Reject,
            result: Id(4),
            lhs: Id(3),
            rhs: Id(2),
        }),
        Op::Data(Data::BindingStore {
            binding: PcuBindingRef::new(0, 2),
            index,
            value: Id(4),
        }),
    ]
}

#[test]
fn admits_multiple_checked_nodes_constants_and_per_node_policies_for_both_widths() {
    for (value_type, caps) in [
        (PcuValueType::f32(), PcuValueTypeCaps::FLOAT32),
        (PcuValueType::f64(), PcuValueTypeCaps::FLOAT64),
    ] {
        let bindings = bindings(value_type);
        let body = accepted_body(value_type, PcuDispatchIndex::InvocationId);
        let mut ops = body.to_vec();
        ops.push(Op::Control(PcuDispatchControlOp::Return));
        assert_eq!(
            validate_checked_float_map_kernel(&fixture(&bindings, &ops), value_type, caps),
            Ok(())
        );

        let grid_body = accepted_body(value_type, PcuDispatchIndex::GridStrideId);
        let grid_ops = [
            Op::GridStrideLoop {
                extent: 32,
                body: &grid_body,
            },
            Op::Control(PcuDispatchControlOp::Return),
        ];
        assert_eq!(
            validate_checked_float_map_kernel(&fixture(&bindings, &grid_ops), value_type, caps),
            Ok(())
        );
    }
}

#[test]
fn admits_checked_relu_for_direct_and_grid_stride_maps_at_both_widths() {
    for (value_type, caps) in [
        (PcuValueType::f32(), PcuValueTypeCaps::FLOAT32),
        (PcuValueType::f64(), PcuValueTypeCaps::FLOAT64),
    ] {
        let bindings = bindings(value_type);
        let direct = [
            Op::Data(Data::BindingLoad {
                result: Id(1),
                binding: PcuBindingRef::new(0, 0),
                index: PcuDispatchIndex::InvocationId,
            }),
            Op::Data(Data::CheckedFloatUnary {
                value_type,
                op: Unary::Relu,
                underflow_policy: Policy::RejectSubnormalResult,
                range_policy: crate::PcuRangePolicy::Clamp,
                result: Id(2),
                value: Id(1),
            }),
            Op::Data(Data::BindingStore {
                binding: PcuBindingRef::new(0, 2),
                index: PcuDispatchIndex::InvocationId,
                value: Id(2),
            }),
            Op::Control(PcuDispatchControlOp::Return),
        ];
        assert_eq!(
            validate_checked_float_map_kernel(&fixture(&bindings, &direct), value_type, caps),
            Ok(())
        );

        let grid_body = [
            Op::Data(Data::BindingLoad {
                result: Id(1),
                binding: PcuBindingRef::new(0, 0),
                index: PcuDispatchIndex::GridStrideId,
            }),
            Op::Data(Data::CheckedFloatUnary {
                value_type,
                op: Unary::Relu,
                underflow_policy: Policy::AllowGradualUnderflow,
                range_policy: crate::PcuRangePolicy::Reject,
                result: Id(2),
                value: Id(1),
            }),
            Op::Data(Data::BindingStore {
                binding: PcuBindingRef::new(0, 2),
                index: PcuDispatchIndex::GridStrideId,
                value: Id(2),
            }),
        ];
        let grid = [
            Op::GridStrideLoop {
                extent: 8,
                body: &grid_body,
            },
            Op::Control(PcuDispatchControlOp::Return),
        ];
        assert_eq!(
            validate_checked_float_map_kernel(&fixture(&bindings, &grid), value_type, caps),
            Ok(())
        );
    }
}

#[test]
fn rejects_unchecked_float_arithmetic_and_noncanonical_indices() {
    let bindings = bindings(PcuValueType::f32());
    let mut body = accepted_body(PcuValueType::f32(), PcuDispatchIndex::InvocationId);
    body[4] = Op::Data(Data::Alu {
        value_type: PcuValueType::f32(),
        op: crate::PcuDispatchAluOp::Mul,
        result: Id(4),
        lhs: Id(3),
        rhs: Id(2),
    });
    let mut ops = body.to_vec();
    ops.push(Op::Control(PcuDispatchControlOp::Return));
    assert_eq!(
        validate_checked_float_map_kernel(
            &fixture(&bindings, &ops),
            PcuValueType::f32(),
            PcuValueTypeCaps::FLOAT32 | PcuValueTypeCaps::FLOAT64
        ),
        Err(Error::UnsupportedOperation(4)),
    );

    let mut body = accepted_body(PcuValueType::f32(), PcuDispatchIndex::InvocationId);
    body[5] = Op::Data(Data::BindingStore {
        binding: PcuBindingRef::new(0, 2),
        index: PcuDispatchIndex::BindingElementZero,
        value: Id(4),
    });
    let mut ops = body.to_vec();
    ops.push(Op::Control(PcuDispatchControlOp::Return));
    assert_eq!(
        validate_checked_float_map_kernel(
            &fixture(&bindings, &ops),
            PcuValueType::f32(),
            PcuValueTypeCaps::FLOAT32 | PcuValueTypeCaps::FLOAT64
        ),
        Err(Error::InvalidIndex(5)),
    );
}

#[test]
fn rejects_type_mismatch_and_invalid_ssa() {
    let bindings = bindings(PcuValueType::f32());
    let mut body = accepted_body(PcuValueType::f32(), PcuDispatchIndex::InvocationId);
    body[4] = Op::Data(Data::CheckedFloatBinary {
        value_type: PcuValueType::f64(),
        op: Binary::Div,
        underflow_policy: Policy::AllowGradualUnderflow,
        range_policy: crate::PcuRangePolicy::Reject,
        result: Id(4),
        lhs: Id(3),
        rhs: Id(2),
    });
    let mut ops = body.to_vec();
    ops.push(Op::Control(PcuDispatchControlOp::Return));
    assert_eq!(
        validate_checked_float_map_kernel(
            &fixture(&bindings, &ops),
            PcuValueType::f32(),
            PcuValueTypeCaps::FLOAT32 | PcuValueTypeCaps::FLOAT64
        ),
        Err(Error::UnsupportedOperation(4)),
    );

    let mut body = accepted_body(PcuValueType::f32(), PcuDispatchIndex::InvocationId);
    body[4] = Op::Data(Data::CheckedFloatBinary {
        value_type: PcuValueType::f32(),
        op: Binary::Div,
        underflow_policy: Policy::AllowGradualUnderflow,
        range_policy: crate::PcuRangePolicy::Reject,
        result: Id(4),
        lhs: Id(99),
        rhs: Id(2),
    });
    let mut ops = body.to_vec();
    ops.push(Op::Control(PcuDispatchControlOp::Return));
    assert_eq!(
        validate_checked_float_map_kernel(
            &fixture(&bindings, &ops),
            PcuValueType::f32(),
            PcuValueTypeCaps::FLOAT32
        ),
        Err(Error::InvalidSsa),
    );
}

#[test]
fn requires_positive_one_dimensional_logical_shape() {
    let bindings = bindings(PcuValueType::f32());
    let body = accepted_body(PcuValueType::f32(), PcuDispatchIndex::InvocationId);
    let mut ops = body.to_vec();
    ops.push(Op::Control(PcuDispatchControlOp::Return));
    let mut kernel = fixture(&bindings, &ops);

    kernel.entry.logical_shape = [0, 1, 1];
    assert_eq!(
        validate_checked_float_map_kernel(&kernel, PcuValueType::f32(), PcuValueTypeCaps::FLOAT32),
        Err(Error::InvalidLogicalShape),
    );

    kernel.entry.logical_shape = [8, 2, 1];
    assert_eq!(
        validate_checked_float_map_kernel(&kernel, PcuValueType::f32(), PcuValueTypeCaps::FLOAT32),
        Err(Error::InvalidLogicalShape),
    );

    kernel.entry.logical_shape = [8, 1, 3];
    assert_eq!(
        validate_checked_float_map_kernel(&kernel, PcuValueType::f32(), PcuValueTypeCaps::FLOAT32),
        Err(Error::InvalidLogicalShape),
    );
}
