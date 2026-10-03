#[rustfmt::skip]
use super::{
    validate_checked_float_binary_kernel,
    CheckedFloatBinaryMapValidationError as Error,
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
    PcuDispatchFloatBinaryOp as BinaryOp,
    PcuDispatchIndex,
    PcuDispatchKernelIr,
    PcuDispatchOp as Op,
    PcuDispatchValueId as Id,
    PcuFloatUnderflowPolicy,
    PcuKernelId,
    PcuValueType,
    PcuValueTypeCaps,
};

mod operand_schema;

fn validate(
    kernel: &PcuDispatchKernelIr<'_>,
    op: BinaryOp,
    policy: PcuFloatUnderflowPolicy,
    caps: PcuValueTypeCaps,
) -> Result<(), Error> {
    validate_checked_float_binary_kernel(kernel, PcuValueType::f32(), op, policy, caps)
}

fn fixture<'a>(
    bindings: &'a [PcuBinding],
    ops: &'a [Op<'a>],
    parameters: &'a [crate::PcuParameter],
) -> PcuDispatchKernelIr<'a> {
    PcuDispatchKernelIr {
        numerical_requirements: PcuDispatchKernelIr::DEFAULT_REQUIREMENTS,
        id: PcuKernelId(7),
        entry: PcuDispatchEntryPoint {
            name: "checked_f32",
            logical_shape: [1, 1, 1],
        },
        bindings,
        ports: &[],
        parameters,
        ops,
        type_caps: PcuValueTypeCaps::empty(),
        feature_caps: crate::PcuDispatchFeatureCaps::empty(),
    }
}

fn make_bindings(value_type: PcuValueType) -> [PcuBinding<'static>; 3] {
    [
        PcuBinding::value(
            Some("lhs"),
            0,
            0,
            PcuBindingStorageClass::Storage,
            PcuBindingAccess::ReadOnly,
            value_type,
        ),
        PcuBinding::value(
            Some("rhs"),
            0,
            1,
            PcuBindingStorageClass::Storage,
            PcuBindingAccess::ReadOnly,
            value_type,
        ),
        PcuBinding::value(
            Some("out"),
            0,
            2,
            PcuBindingStorageClass::Storage,
            PcuBindingAccess::WriteOnly,
            value_type,
        ),
    ]
}

fn body(index: PcuDispatchIndex, result: Id, policy: PcuFloatUnderflowPolicy) -> [Op<'static>; 4] {
    [
        Op::Data(Data::BindingLoad {
            result: Id(1),
            binding: PcuBindingRef::new(0, 0),
            index,
        }),
        Op::Data(Data::BindingLoad {
            result: Id(2),
            binding: PcuBindingRef::new(0, 1),
            index,
        }),
        Op::Data(Data::CheckedFloatBinary {
            value_type: PcuValueType::f32(),
            op: BinaryOp::Add,
            underflow_policy: policy,
            range_policy: crate::PcuRangePolicy::Reject,
            result,
            lhs: Id(1),
            rhs: Id(2),
        }),
        Op::Data(Data::BindingStore {
            binding: PcuBindingRef::new(0, 2),
            index,
            value: result,
        }),
    ]
}

#[test]
fn accepts_checked_f32_direct_and_grid_stride_profiles() {
    let bindings = make_bindings(PcuValueType::f32());
    let direct_body = body(
        PcuDispatchIndex::InvocationId,
        Id(3),
        PcuFloatUnderflowPolicy::IeeeAfterRounding,
    );
    let direct_ops = [
        direct_body[0],
        direct_body[1],
        direct_body[2],
        direct_body[3],
        Op::Control(PcuDispatchControlOp::Return),
    ];
    let direct = fixture(&bindings, &direct_ops, &[]);
    assert_eq!(
        validate(
            &direct,
            BinaryOp::Add,
            PcuFloatUnderflowPolicy::IeeeAfterRounding,
            PcuValueTypeCaps::FLOAT32
        ),
        Ok(())
    );

    let grid_body = body(
        PcuDispatchIndex::GridStrideId,
        Id(3),
        PcuFloatUnderflowPolicy::AllowGradualUnderflow,
    );
    let grid_ops = [
        Op::GridStrideLoop {
            extent: 19,
            body: &grid_body,
        },
        Op::Control(PcuDispatchControlOp::Return),
    ];
    let grid = fixture(&bindings, &grid_ops, &[]);
    assert_eq!(
        validate(
            &grid,
            BinaryOp::Add,
            PcuFloatUnderflowPolicy::AllowGradualUnderflow,
            PcuValueTypeCaps::FLOAT32
        ),
        Ok(())
    );
}

#[test]
fn rejects_interface_type_capability_and_policy_mismatches() {
    let bindings = make_bindings(PcuValueType::f32());
    let body = body(
        PcuDispatchIndex::InvocationId,
        Id(3),
        PcuFloatUnderflowPolicy::IeeeAfterRounding,
    );
    let ops = [
        body[0],
        body[1],
        body[2],
        body[3],
        Op::Control(PcuDispatchControlOp::Return),
    ];
    let parameter = [crate::PcuParameter::named(
        crate::PcuParameterSlot(0),
        "unused",
        PcuValueType::f32(),
    )];
    assert_eq!(
        validate(
            &fixture(&bindings, &ops, &parameter),
            BinaryOp::Add,
            PcuFloatUnderflowPolicy::IeeeAfterRounding,
            PcuValueTypeCaps::FLOAT32
        ),
        Err(Error::UnsupportedInterface)
    );
    assert_eq!(
        validate(
            &fixture(&bindings, &ops, &[]),
            BinaryOp::Add,
            PcuFloatUnderflowPolicy::AllowGradualUnderflow,
            PcuValueTypeCaps::FLOAT32
        ),
        Err(Error::UnsupportedOperation(2))
    );
    assert_eq!(
        validate(
            &fixture(&bindings, &ops, &[]),
            BinaryOp::Add,
            PcuFloatUnderflowPolicy::IeeeAfterRounding,
            PcuValueTypeCaps::FLOAT64
        ),
        Err(Error::UnsupportedRequirements)
    );

    let f64_bindings = make_bindings(PcuValueType::Scalar(crate::PcuScalarType::F64));
    assert_eq!(
        validate(
            &fixture(&f64_bindings, &ops, &[]),
            BinaryOp::Add,
            PcuFloatUnderflowPolicy::IeeeAfterRounding,
            PcuValueTypeCaps::FLOAT32
        ),
        Err(Error::UnsupportedRequirements)
    );

    let f64_ops = [
        Op::Data(Data::BindingLoad {
            result: Id(1),
            binding: PcuBindingRef::new(0, 0),
            index: PcuDispatchIndex::InvocationId,
        }),
        Op::Data(Data::BindingLoad {
            result: Id(2),
            binding: PcuBindingRef::new(0, 1),
            index: PcuDispatchIndex::InvocationId,
        }),
        Op::Data(Data::CheckedFloatBinary {
            value_type: PcuValueType::f64(),
            op: BinaryOp::Add,
            underflow_policy: PcuFloatUnderflowPolicy::IeeeAfterRounding,
            range_policy: crate::PcuRangePolicy::Reject,
            result: Id(3),
            lhs: Id(1),
            rhs: Id(2),
        }),
        Op::Data(Data::BindingStore {
            binding: PcuBindingRef::new(0, 2),
            index: PcuDispatchIndex::InvocationId,
            value: Id(3),
        }),
        Op::Control(PcuDispatchControlOp::Return),
    ];
    assert!(
        validate(
            &fixture(&f64_bindings, &f64_ops, &[]),
            BinaryOp::Add,
            PcuFloatUnderflowPolicy::IeeeAfterRounding,
            PcuValueTypeCaps::FLOAT32 | PcuValueTypeCaps::FLOAT64,
        )
        .is_err()
    );
}

#[test]
fn rejects_bad_ssa_and_grid_stride_shapes() {
    let bindings = make_bindings(PcuValueType::f32());
    let body = body(
        PcuDispatchIndex::InvocationId,
        Id(3),
        PcuFloatUnderflowPolicy::IeeeAfterRounding,
    );
    let mut bad_body = body;
    bad_body[2] = Op::Data(Data::CheckedFloatBinary {
        value_type: PcuValueType::f32(),
        op: BinaryOp::Mul,
        underflow_policy: PcuFloatUnderflowPolicy::IeeeAfterRounding,
        range_policy: crate::PcuRangePolicy::Reject,
        result: Id(1),
        lhs: Id(1),
        rhs: Id(2),
    });
    bad_body[3] = Op::Data(Data::BindingStore {
        binding: PcuBindingRef::new(0, 2),
        index: PcuDispatchIndex::InvocationId,
        value: Id(1),
    });
    let bad_ops = [
        bad_body[0],
        bad_body[1],
        bad_body[2],
        bad_body[3],
        Op::Control(PcuDispatchControlOp::Return),
    ];
    assert_eq!(
        validate(
            &fixture(&bindings, &bad_ops, &[]),
            BinaryOp::Mul,
            PcuFloatUnderflowPolicy::IeeeAfterRounding,
            PcuValueTypeCaps::FLOAT32
        ),
        Err(Error::InvalidSsa)
    );

    let grid_ops = [
        Op::GridStrideLoop {
            extent: 0,
            body: &body,
        },
        Op::Control(PcuDispatchControlOp::Return),
    ];
    assert_eq!(
        validate(
            &fixture(&bindings, &grid_ops, &[]),
            BinaryOp::Add,
            PcuFloatUnderflowPolicy::IeeeAfterRounding,
            PcuValueTypeCaps::FLOAT32
        ),
        Err(Error::UnsupportedOperation(0))
    );
}

#[test]
fn accepts_binary64_profiles_and_requires_matching_type_floor() {
    let policies = [
        PcuFloatUnderflowPolicy::IeeeAfterRounding,
        PcuFloatUnderflowPolicy::RejectSubnormalResult,
        PcuFloatUnderflowPolicy::AllowGradualUnderflow,
    ];
    for policy in policies {
        for binary_op in [BinaryOp::Add, BinaryOp::Sub, BinaryOp::Mul] {
            for index in [
                PcuDispatchIndex::InvocationId,
                PcuDispatchIndex::GridStrideId,
            ] {
                let bindings = make_bindings(PcuValueType::f64());
                let mut body = body(index, Id(3), policy);
                body[2] = Op::Data(Data::CheckedFloatBinary {
                    value_type: PcuValueType::f64(),
                    op: binary_op,
                    underflow_policy: policy,
                    range_policy: crate::PcuRangePolicy::Reject,
                    result: Id(3),
                    lhs: Id(1),
                    rhs: Id(2),
                });
                let direct = [
                    body[0],
                    body[1],
                    body[2],
                    body[3],
                    Op::Control(PcuDispatchControlOp::Return),
                ];
                let grid = [
                    Op::GridStrideLoop {
                        extent: 19,
                        body: &body,
                    },
                    Op::Control(PcuDispatchControlOp::Return),
                ];
                let ops: &[Op<'_>] = if index == PcuDispatchIndex::InvocationId {
                    &direct
                } else {
                    &grid
                };
                let kernel = fixture(&bindings, ops, &[]);
                assert_eq!(
                    validate_checked_float_binary_kernel(
                        &kernel,
                        PcuValueType::f64(),
                        binary_op,
                        policy,
                        PcuValueTypeCaps::FLOAT64
                    ),
                    Ok(())
                );
                assert_eq!(
                    validate_checked_float_binary_kernel(
                        &kernel,
                        PcuValueType::f64(),
                        binary_op,
                        policy,
                        PcuValueTypeCaps::FLOAT32
                    ),
                    Err(Error::UnsupportedRequirements)
                );
                assert!(
                    validate_checked_float_binary_kernel(
                        &kernel,
                        PcuValueType::f32(),
                        binary_op,
                        policy,
                        PcuValueTypeCaps::FLOAT32 | PcuValueTypeCaps::FLOAT64
                    )
                    .is_err()
                );
                assert_eq!(
                    validate_checked_float_binary_kernel(
                        &kernel,
                        PcuValueType::u32(),
                        binary_op,
                        policy,
                        PcuValueTypeCaps::FLOAT32 | PcuValueTypeCaps::FLOAT64
                    ),
                    Err(Error::UnsupportedRequirements)
                );
            }
        }
    }
}

#[test]
fn named_low_precision_contracts_require_exact_format_and_capabilities() {
    for scalar in [
        crate::PcuScalarType::F16,
        crate::PcuScalarType::BF16,
        crate::PcuScalarType::F8E4M3FN,
        crate::PcuScalarType::F8E5M2,
    ] {
        let value_type = PcuValueType::Scalar(scalar);
        let bindings = make_bindings(value_type);
        for policy in [
            PcuFloatUnderflowPolicy::IeeeAfterRounding,
            PcuFloatUnderflowPolicy::AllowGradualUnderflow,
            PcuFloatUnderflowPolicy::RejectSubnormalResult,
        ] {
            for op in [BinaryOp::Add, BinaryOp::Sub, BinaryOp::Mul, BinaryOp::Div] {
                for index in [
                    PcuDispatchIndex::InvocationId,
                    PcuDispatchIndex::GridStrideId,
                ] {
                    let mut body = body(index, Id(3), policy);
                    body[2] = Op::Data(Data::CheckedFloatBinary {
                        value_type,
                        op,
                        underflow_policy: policy,
                        range_policy: crate::PcuRangePolicy::Reject,
                        result: Id(3),
                        lhs: Id(1),
                        rhs: Id(2),
                    });
                    let direct = [
                        body[0],
                        body[1],
                        body[2],
                        body[3],
                        Op::Control(PcuDispatchControlOp::Return),
                    ];
                    let grid = [
                        Op::GridStrideLoop {
                            extent: 17,
                            body: &body,
                        },
                        Op::Control(PcuDispatchControlOp::Return),
                    ];
                    let ops: &[Op<'_>] = if index == PcuDispatchIndex::InvocationId {
                        &direct
                    } else {
                        &grid
                    };
                    let kernel = fixture(&bindings, ops, &[]);
                    assert_eq!(
                        validate_checked_float_binary_kernel(
                            &kernel,
                            value_type,
                            op,
                            policy,
                            PcuValueTypeCaps::for_scalar(scalar)
                        ),
                        Ok(())
                    );
                    assert_eq!(
                        validate_checked_float_binary_kernel(
                            &kernel,
                            value_type,
                            op,
                            policy,
                            PcuValueTypeCaps::FLOAT32
                        ),
                        Err(Error::UnsupportedRequirements)
                    );
                    assert!(
                        validate_checked_float_binary_kernel(
                            &kernel,
                            PcuValueType::f32(),
                            op,
                            policy,
                            PcuValueTypeCaps::FLOAT32 | PcuValueTypeCaps::for_scalar(scalar)
                        )
                        .is_err()
                    );
                    for wide in [crate::PcuScalarType::F128, crate::PcuScalarType::F256] {
                        assert_eq!(
                            validate_checked_float_binary_kernel(
                                &kernel,
                                PcuValueType::Scalar(wide),
                                op,
                                policy,
                                PcuValueTypeCaps::for_scalar(wide)
                            ),
                            Err(Error::UnsupportedRequirements)
                        );
                    }
                }
            }
        }
    }
}
