#[rustfmt::skip]
use super::{
    validate_integer_checked_binary_kernel,
    IntegerMapValidationError as Error,
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
    PcuDispatchIndex,
    PcuDispatchIntegerBinaryOp as BinaryOp,
    PcuDispatchKernelIr,
    PcuDispatchOp as Op,
    PcuDispatchValueId as Id,
    PcuKernelId,
    PcuScalarType,
    PcuValueType,
    PcuValueTypeCaps,
};

#[allow(clippy::too_many_arguments)] // Fixture dimensions let tests vary index and SSA failures independently.
fn exercise(
    scalar: PcuScalarType,
    op: BinaryOp,
    lhs_index: PcuDispatchIndex,
    rhs_index: PcuDispatchIndex,
    result_id: Id,
    checked_lhs: Id,
    scalar_caps: PcuValueTypeCaps,
    grid: bool,
) -> Result<(), Error> {
    exercise_policies(
        scalar,
        op,
        lhs_index,
        rhs_index,
        result_id,
        checked_lhs,
        scalar_caps,
        grid,
        (crate::PcuRangePolicy::Reject, crate::PcuRangePolicy::Reject),
    )
}

#[allow(clippy::too_many_arguments)] // Independent fixture axes exercise malformed SSA and policy headers.
fn exercise_policies(
    scalar: PcuScalarType,
    op: BinaryOp,
    lhs_index: PcuDispatchIndex,
    rhs_index: PcuDispatchIndex,
    result_id: Id,
    checked_lhs: Id,
    scalar_caps: PcuValueTypeCaps,
    grid: bool,
    policies: (crate::PcuRangePolicy, crate::PcuRangePolicy),
) -> Result<(), Error> {
    let value_type = PcuValueType::Scalar(scalar);
    let bindings = [
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
    ];
    let index = if grid {
        PcuDispatchIndex::GridStrideId
    } else {
        PcuDispatchIndex::InvocationId
    };
    let body = [
        Op::Data(Data::BindingLoad {
            result: Id(1),
            binding: PcuBindingRef::new(0, 0),
            index: lhs_index,
        }),
        Op::Data(Data::BindingLoad {
            result: Id(2),
            binding: PcuBindingRef::new(0, 1),
            index: rhs_index,
        }),
        Op::Data(Data::CheckedIntegerBinary {
            value_type,
            op,
            range_policy: policies.0,
            result: result_id,
            lhs: checked_lhs,
            rhs: Id(2),
        }),
        Op::Data(Data::BindingStore {
            binding: PcuBindingRef::new(0, 2),
            index,
            value: result_id,
        }),
    ];
    let loop_ops = [
        Op::GridStrideLoop {
            extent: 17,
            body: &body,
        },
        Op::Control(PcuDispatchControlOp::Return),
    ];
    let direct_ops = [
        body[0],
        body[1],
        body[2],
        body[3],
        Op::Control(PcuDispatchControlOp::Return),
    ];
    let kernel = PcuDispatchKernelIr {
        numerical_requirements: crate::PcuImplementationRequirements {
            range_policy: policies.1,
            ..PcuDispatchKernelIr::DEFAULT_REQUIREMENTS
        },
        id: PcuKernelId(7),
        entry: PcuDispatchEntryPoint {
            name: "checked-map",
            logical_shape: [17, 1, 1],
        },
        bindings: &bindings,
        ports: &[],
        parameters: &[],
        ops: if grid { &loop_ops } else { &direct_ops },
        type_caps: PcuValueTypeCaps::empty(),
        feature_caps: crate::PcuDispatchFeatureCaps::empty(),
    };
    validate_integer_checked_binary_kernel(&kernel, value_type, op, scalar_caps)
}

#[test]
fn range_policy_matches_header_in_direct_and_grid_broadcast_maps() {
    use crate::PcuRangePolicy;
    for scalar in [
        PcuScalarType::I8,
        PcuScalarType::U8,
        PcuScalarType::I16,
        PcuScalarType::U16,
        PcuScalarType::I32,
        PcuScalarType::U32,
        PcuScalarType::I64,
        PcuScalarType::U64,
        PcuScalarType::I128,
        PcuScalarType::U128,
        PcuScalarType::I256,
        PcuScalarType::U256,
        PcuScalarType::I512,
        PcuScalarType::U512,
    ] {
        for op in [BinaryOp::Add, BinaryOp::Sub, BinaryOp::Mul] {
            for grid in [false, true] {
                let index = if grid {
                    PcuDispatchIndex::GridStrideId
                } else {
                    PcuDispatchIndex::InvocationId
                };
                for instruction in [PcuRangePolicy::Reject, PcuRangePolicy::Clamp] {
                    for header in [PcuRangePolicy::Reject, PcuRangePolicy::Clamp] {
                        assert_eq!(
                            exercise_policies(
                                scalar,
                                op,
                                index,
                                PcuDispatchIndex::BindingElementZero,
                                Id(3),
                                Id(1),
                                PcuValueTypeCaps::for_scalar(scalar),
                                grid,
                                (instruction, header),
                            ),
                            if instruction == header {
                                Ok(())
                            } else {
                                Err(Error::RangePolicyMismatch)
                            },
                            "{scalar:?} {op:?} grid={grid} instruction={instruction:?} header={header:?}",
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn admits_all_fourteen_integer_representations_in_direct_and_grid_stride_maps() {
    for (scalar, caps) in [
        (PcuScalarType::I8, PcuValueTypeCaps::INT8),
        (PcuScalarType::U8, PcuValueTypeCaps::UINT8),
        (PcuScalarType::I16, PcuValueTypeCaps::INT16),
        (PcuScalarType::U16, PcuValueTypeCaps::UINT16),
        (PcuScalarType::I32, PcuValueTypeCaps::INT32),
        (PcuScalarType::U32, PcuValueTypeCaps::UINT32),
        (PcuScalarType::I64, PcuValueTypeCaps::INT64),
        (PcuScalarType::U64, PcuValueTypeCaps::UINT64),
        (PcuScalarType::I128, PcuValueTypeCaps::INT128),
        (PcuScalarType::U128, PcuValueTypeCaps::UINT128),
        (PcuScalarType::I256, PcuValueTypeCaps::INT256),
        (PcuScalarType::U256, PcuValueTypeCaps::UINT256),
        (PcuScalarType::I512, PcuValueTypeCaps::INT512),
        (PcuScalarType::U512, PcuValueTypeCaps::UINT512),
    ] {
        assert_eq!(
            exercise(
                scalar,
                BinaryOp::Add,
                PcuDispatchIndex::InvocationId,
                PcuDispatchIndex::InvocationId,
                Id(3),
                Id(1),
                caps,
                false
            ),
            Ok(())
        );
        assert_eq!(
            exercise(
                scalar,
                BinaryOp::Mul,
                PcuDispatchIndex::GridStrideId,
                PcuDispatchIndex::GridStrideId,
                Id(3),
                Id(1),
                caps,
                true
            ),
            Ok(())
        );
    }
}

#[test]
fn admits_broadcast_and_rejects_invalid_type_ssa_and_capabilities() {
    assert_eq!(
        exercise(
            PcuScalarType::I32,
            BinaryOp::Sub,
            PcuDispatchIndex::BindingElementZero,
            PcuDispatchIndex::InvocationId,
            Id(3),
            Id(1),
            PcuValueTypeCaps::INT32,
            false
        ),
        Ok(())
    );
    assert_eq!(
        exercise(
            PcuScalarType::F32,
            BinaryOp::Add,
            PcuDispatchIndex::InvocationId,
            PcuDispatchIndex::InvocationId,
            Id(3),
            Id(1),
            PcuValueTypeCaps::FLOAT32,
            false
        ),
        Err(Error::UnsupportedInterface)
    );
    assert_eq!(
        exercise(
            PcuScalarType::U32,
            BinaryOp::Add,
            PcuDispatchIndex::InvocationId,
            PcuDispatchIndex::InvocationId,
            Id(3),
            Id(1),
            PcuValueTypeCaps::INT32,
            false
        ),
        Err(Error::UnsupportedRequirements)
    );
    assert_eq!(
        exercise(
            PcuScalarType::U32,
            BinaryOp::Add,
            PcuDispatchIndex::InvocationId,
            PcuDispatchIndex::InvocationId,
            Id(1),
            Id(1),
            PcuValueTypeCaps::UINT32,
            false
        ),
        Err(Error::InvalidValue(Id(1)))
    );
    assert_eq!(
        exercise(
            PcuScalarType::U32,
            BinaryOp::Add,
            PcuDispatchIndex::InvocationId,
            PcuDispatchIndex::InvocationId,
            Id(3),
            Id(9),
            PcuValueTypeCaps::UINT32,
            false
        ),
        Err(Error::InvalidValue(Id(9)))
    );
}

#[test]
fn checked_binary_roles_follow_access_and_ssa_instead_of_parameter_order() {
    let ty = PcuValueType::Scalar(PcuScalarType::U128);
    let schema = [
        PcuBinding::value(
            Some("left"),
            4,
            7,
            PcuBindingStorageClass::Storage,
            PcuBindingAccess::ReadOnly,
            ty,
        ),
        PcuBinding::value(
            Some("right"),
            2,
            9,
            PcuBindingStorageClass::Storage,
            PcuBindingAccess::ReadOnly,
            ty,
        ),
        PcuBinding::value(
            Some("output"),
            1,
            3,
            PcuBindingStorageClass::Storage,
            PcuBindingAccess::ReadWrite,
            ty,
        ),
    ];
    for order in [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ] {
        let mut bindings = order.map(|index| schema[index]);
        let actual = super::validate_checked_binary_interface(
            &schema_kernel(&bindings),
            ty,
            PcuValueTypeCaps::UINT128,
            false,
        )
        .unwrap();
        assert_eq!(actual[2], schema[2].reference());
        assert!(actual[..2].contains(&schema[0].reference()));
        assert!(actual[..2].contains(&schema[1].reference()));
        for binding in &mut bindings {
            binding.access = PcuBindingAccess::ReadOnly;
        }
        assert_eq!(
            super::validate_checked_binary_interface(
                &schema_kernel(&bindings),
                ty,
                PcuValueTypeCaps::UINT128,
                false,
            ),
            Err(Error::UnsupportedInterface)
        );
        bindings[0].access = PcuBindingAccess::ReadWrite;
        bindings[1].access = PcuBindingAccess::ReadWrite;
        assert_eq!(
            super::validate_checked_binary_interface(
                &schema_kernel(&bindings),
                ty,
                PcuValueTypeCaps::UINT128,
                false,
            ),
            Err(Error::UnsupportedInterface)
        );
    }
}

fn schema_kernel<'a>(bindings: &'a [PcuBinding<'a>]) -> PcuDispatchKernelIr<'a> {
    PcuDispatchKernelIr {
        id: PcuKernelId(1),
        entry: PcuDispatchEntryPoint {
            name: "role-assessment",
            logical_shape: [5, 1, 1],
        },
        numerical_requirements: PcuDispatchKernelIr::DEFAULT_REQUIREMENTS,
        bindings,
        ports: &[],
        parameters: &[],
        ops: &[],
        type_caps: PcuValueTypeCaps::empty(),
        feature_caps: crate::PcuDispatchFeatureCaps::empty(),
    }
}
