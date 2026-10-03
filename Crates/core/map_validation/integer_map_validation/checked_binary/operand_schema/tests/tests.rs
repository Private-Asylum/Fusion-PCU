//! Detached roles are mathematical SSA roles, never Rust declaration positions.
#[rustfmt::skip]
use crate::{
    assess_checked_integer_binary_operands,
    validate_integer_checked_binary_kernel,
    IntegerMapValidationError as Error,
    PcuBinding,
    PcuBindingAccess,
    PcuBindingRef,
    PcuBindingStorageClass,
    PcuDispatchControlOp,
    PcuDispatchDataOp as Data,
    PcuDispatchEntryPoint,
    PcuDispatchFeatureCaps,
    PcuDispatchIndex as Index,
    PcuDispatchIntegerBinaryOp as Binary,
    PcuDispatchKernelIr,
    PcuDispatchOp as Op,
    PcuDispatchValueId as Id,
    PcuKernelId,
    PcuRangePolicy,
    PcuScalarType,
    PcuValueType,
    PcuValueTypeCaps,
};

fn declarations(scalar: PcuScalarType) -> [PcuBinding<'static>; 3] {
    core::array::from_fn(|slot| {
        PcuBinding::value(
            None,
            0,
            u32::try_from(slot).unwrap(),
            PcuBindingStorageClass::Storage,
            if slot == 2 {
                PcuBindingAccess::ReadWrite
            } else {
                PcuBindingAccess::ReadOnly
            },
            PcuValueType::Scalar(scalar),
        )
    })
}

fn instructions(
    scalar: PcuScalarType,
    op: Binary,
    range: PcuRangePolicy,
    index: Index,
) -> [Op<'static>; 4] {
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
        Op::Data(Data::CheckedIntegerBinary {
            value_type: PcuValueType::Scalar(scalar),
            op,
            range_policy: range,
            result: Id(3),
            lhs: Id(1),
            rhs: Id(2),
        }),
        Op::Data(Data::BindingStore {
            binding: PcuBindingRef::new(0, 2),
            index,
            value: Id(3),
        }),
    ]
}

fn kernel<'a>(
    bindings: &'a [PcuBinding<'a>],
    ops: &'a [Op<'a>],
    range: PcuRangePolicy,
) -> PcuDispatchKernelIr<'a> {
    PcuDispatchKernelIr {
        numerical_requirements: crate::PcuImplementationRequirements {
            range_policy: range,
            ..PcuDispatchKernelIr::DEFAULT_REQUIREMENTS
        },
        id: PcuKernelId(41),
        entry: PcuDispatchEntryPoint {
            name: "integer-roles",
            logical_shape: [19, 1, 1],
        },
        bindings,
        ports: &[],
        parameters: &[],
        ops,
        type_caps: PcuValueTypeCaps::empty(),
        feature_caps: PcuDispatchFeatureCaps::empty(),
    }
}

#[test]
fn fourteen_types_retain_repeated_reversed_and_independent_broadcast_roles() {
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
        for op in [Binary::Add, Binary::Sub, Binary::Mul] {
            for range in [PcuRangePolicy::Reject, PcuRangePolicy::Clamp] {
                for grid in [false, true] {
                    let original = declarations(scalar);
                    for order in [
                        [0, 1, 2],
                        [0, 2, 1],
                        [1, 0, 2],
                        [1, 2, 0],
                        [2, 0, 1],
                        [2, 1, 0],
                    ] {
                        let declared = order.map(|slot| original[slot]);
                        // Load roles are independent of mathematical SSA operand roles.
                        assert_distinct_operand_roles(scalar, op, range, grid, &declared);
                        assert_repeated_operand_roles(scalar, op, range, grid, &declared);
                    }
                }
            }
        }
    }
}

fn assert_distinct_operand_roles(
    scalar: PcuScalarType,
    op: Binary,
    range: PcuRangePolicy,
    grid: bool,
    declared: &[PcuBinding<'_>; 3],
) {
    let index = if grid {
        Index::GridStrideId
    } else {
        Index::InvocationId
    };
    let original = declarations(scalar);
    for reversed in [false, true] {
        let mut body = instructions(scalar, op, range, index);
        let Op::Data(Data::CheckedIntegerBinary { lhs, rhs, .. }) = &mut body[2] else {
            unreachable!()
        };
        if reversed {
            core::mem::swap(lhs, rhs);
        }
        let direct = [
            body[0],
            body[1],
            body[2],
            body[3],
            Op::Control(PcuDispatchControlOp::Return),
        ];
        let loop_ops = [
            Op::GridStrideLoop {
                extent: 19,
                body: &body,
            },
            Op::Control(PcuDispatchControlOp::Return),
        ];
        let ir = kernel(declared, if grid { &loop_ops } else { &direct }, range);
        let roles = assess_checked_integer_binary_operands(
            &ir,
            PcuValueType::Scalar(scalar),
            op,
            PcuValueTypeCaps::for_scalar(scalar),
        )
        .unwrap();
        assert_eq!(
            roles.input_bindings(),
            &[original[0].reference(), original[1].reference()]
        );
        assert_eq!(
            roles.operand_inputs(),
            if reversed { [1, 0] } else { [0, 1] }
        );
        assert_eq!(roles.input_element_counts(19), [19, 19]);
        assert_eq!(roles.output_binding(), original[2].reference());
        // Existing distinct-source admission is unchanged.
        validate_integer_checked_binary_kernel(
            &ir,
            PcuValueType::Scalar(scalar),
            op,
            PcuValueTypeCaps::for_scalar(scalar),
        )
        .unwrap();
    }
}

fn assert_repeated_operand_roles(
    scalar: PcuScalarType,
    op: Binary,
    range: PcuRangePolicy,
    grid: bool,
    declared: &[PcuBinding<'_>; 3],
) {
    let index = if grid {
        Index::GridStrideId
    } else {
        Index::InvocationId
    };
    let original = declarations(scalar);
    for zero_first in [false, true] {
        let mut body = instructions(scalar, op, range, index);
        let Op::Data(Data::BindingLoad {
            binding,
            index: second,
            ..
        }) = &mut body[1]
        else {
            unreachable!()
        };
        *binding = original[0].reference();
        *second = if zero_first {
            index
        } else {
            Index::BindingElementZero
        };
        if zero_first {
            let Op::Data(Data::BindingLoad { index: first, .. }) = &mut body[0] else {
                unreachable!()
            };
            *first = Index::BindingElementZero;
        }
        let direct = [
            body[0],
            body[1],
            body[2],
            body[3],
            Op::Control(PcuDispatchControlOp::Return),
        ];
        let loop_ops = [
            Op::GridStrideLoop {
                extent: 19,
                body: &body,
            },
            Op::Control(PcuDispatchControlOp::Return),
        ];
        let single = [original[2], original[0]];
        for bindings in [&declared[..], &single[..]] {
            let ir = kernel(bindings, if grid { &loop_ops } else { &direct }, range);
            let roles = assess_checked_integer_binary_operands(
                &ir,
                PcuValueType::Scalar(scalar),
                op,
                PcuValueTypeCaps::for_scalar(scalar),
            )
            .unwrap();
            assert_eq!(roles.input_bindings(), &[original[0].reference()]);
            assert_eq!(roles.operand_inputs(), [0, 0]);
            assert_eq!(roles.input_element_counts(19), [19, 0]);
            assert_eq!(
                roles.operand_indices(),
                if zero_first {
                    [Index::BindingElementZero, index]
                } else {
                    [index, Index::BindingElementZero]
                }
            );
            assert!(
                validate_integer_checked_binary_kernel(
                    &ir,
                    PcuValueType::Scalar(scalar),
                    op,
                    PcuValueTypeCaps::for_scalar(scalar)
                )
                .is_err()
            );
        }
    }
}

#[test]
fn broadcast_only_and_repeated_ssa_keep_actual_reads() {
    let declared = declarations(PcuScalarType::U512);
    let mut body = instructions(
        PcuScalarType::U512,
        Binary::Mul,
        PcuRangePolicy::Reject,
        Index::InvocationId,
    );
    for instruction in &mut body[..2] {
        let Op::Data(Data::BindingLoad { index, .. }) = instruction else {
            unreachable!()
        };
        *index = Index::BindingElementZero;
    }
    let Op::Data(Data::CheckedIntegerBinary { rhs, .. }) = &mut body[2] else {
        unreachable!()
    };
    *rhs = Id(1);
    let ops = [
        body[0],
        body[1],
        body[2],
        body[3],
        Op::Control(PcuDispatchControlOp::Return),
    ];
    let roles = assess_checked_integer_binary_operands(
        &kernel(&declared, &ops, PcuRangePolicy::Reject),
        PcuValueType::Scalar(PcuScalarType::U512),
        Binary::Mul,
        PcuValueTypeCaps::UINT512,
    )
    .unwrap();
    assert_eq!(
        roles.input_bindings(),
        &[declared[0].reference(), declared[1].reference()]
    );
    assert_eq!(roles.operand_inputs(), [0, 0]);
    assert_eq!(roles.input_element_counts(19), [1, 1]);
}

#[test]
fn malformed_roles_ssa_indices_capabilities_and_range_headers_reject() {
    let scalar = PcuScalarType::I32;
    let declared = declarations(scalar);
    let body = instructions(
        scalar,
        Binary::Sub,
        PcuRangePolicy::Reject,
        Index::InvocationId,
    );
    for (slot, instruction, expected) in [
        (
            1,
            Op::Data(Data::BindingLoad {
                result: Id(1),
                binding: declared[1].reference(),
                index: Index::InvocationId,
            }),
            Error::InvalidValue(Id(1)),
        ),
        (
            1,
            Op::Data(Data::BindingLoad {
                result: Id(2),
                binding: declared[2].reference(),
                index: Index::InvocationId,
            }),
            Error::InvalidBinding(declared[2].reference()),
        ),
        (
            2,
            Op::Data(Data::CheckedIntegerBinary {
                value_type: PcuValueType::Scalar(scalar),
                op: Binary::Sub,
                range_policy: PcuRangePolicy::Reject,
                result: Id(3),
                lhs: Id(99),
                rhs: Id(2),
            }),
            Error::InvalidValue(Id(99)),
        ),
        (
            2,
            Op::Data(Data::CheckedIntegerBinary {
                value_type: PcuValueType::Scalar(scalar),
                op: Binary::Sub,
                range_policy: PcuRangePolicy::Clamp,
                result: Id(3),
                lhs: Id(1),
                rhs: Id(2),
            }),
            Error::RangePolicyMismatch,
        ),
        (
            3,
            Op::Data(Data::BindingStore {
                binding: declared[2].reference(),
                index: Index::BindingElementZero,
                value: Id(3),
            }),
            Error::InvalidIndex(3),
        ),
    ] {
        let mut changed = body;
        changed[slot] = instruction;
        let ops = [
            changed[0],
            changed[1],
            changed[2],
            changed[3],
            Op::Control(PcuDispatchControlOp::Return),
        ];
        assert_eq!(
            assess_checked_integer_binary_operands(
                &kernel(&declared, &ops, PcuRangePolicy::Reject),
                PcuValueType::Scalar(scalar),
                Binary::Sub,
                PcuValueTypeCaps::INT32
            ),
            Err(expected)
        );
    }
}

#[test]
fn malformed_interfaces_capabilities_and_grid_indices_reject() {
    let scalar = PcuScalarType::I32;
    let declared = declarations(scalar);
    let body = instructions(
        scalar,
        Binary::Sub,
        PcuRangePolicy::Reject,
        Index::InvocationId,
    );
    let ops = [
        body[0],
        body[1],
        body[2],
        body[3],
        Op::Control(PcuDispatchControlOp::Return),
    ];
    let ir = kernel(&declared, &ops, PcuRangePolicy::Reject);
    assert_eq!(
        assess_checked_integer_binary_operands(
            &ir,
            PcuValueType::Scalar(scalar),
            Binary::Sub,
            PcuValueTypeCaps::empty()
        ),
        Err(Error::UnsupportedRequirements)
    );
    let duplicate = [declared[0], declared[0], declared[2]];
    assert_eq!(
        assess_checked_integer_binary_operands(
            &kernel(&duplicate, &ops, PcuRangePolicy::Reject),
            PcuValueType::Scalar(scalar),
            Binary::Sub,
            PcuValueTypeCaps::INT32
        ),
        Err(Error::DuplicateBinding(declared[0].reference()))
    );
    let multiple_writes = [declared[0], declared[2], declared[2]];
    assert!(
        assess_checked_integer_binary_operands(
            &kernel(&multiple_writes, &ops, PcuRangePolicy::Reject),
            PcuValueType::Scalar(scalar),
            Binary::Sub,
            PcuValueTypeCaps::INT32
        )
        .is_err()
    );
    let bad_grid = [
        Op::GridStrideLoop {
            extent: 19,
            body: &body,
        },
        Op::Control(PcuDispatchControlOp::Return),
    ];
    assert_eq!(
        assess_checked_integer_binary_operands(
            &kernel(&declared, &bad_grid, PcuRangePolicy::Reject),
            PcuValueType::Scalar(scalar),
            Binary::Sub,
            PcuValueTypeCaps::INT32
        ),
        Err(Error::InvalidIndex(0))
    );
}
