//! Exact role proofs; backend execution remains a separate qualification.

#[rustfmt::skip]
use crate::{
    assess_checked_integer_div_rem_operands as assess,
    validate_integer_checked_div_rem_kernel as legacy,
    CheckedIntegerDivRemOperandSchema,
    IntegerMapValidationError as Error,
    PcuBinding,
    PcuBindingAccess as Access,
    PcuBindingRef as Ref,
    PcuBindingStorageClass,
    PcuDispatchControlOp,
    PcuDispatchDataOp as Data,
    PcuDispatchEntryPoint,
    PcuDispatchFeatureCaps,
    PcuDispatchIndex as Index,
    PcuDispatchKernelIr as Kernel,
    PcuDispatchOp as Op,
    PcuDispatchValueId as Id,
    PcuKernelId,
    PcuRangePolicy,
    PcuScalarType as Scalar,
    PcuValueType as Type,
    PcuValueTypeCaps as Caps,
};

use crate::model::PcuIntegerDivFlags;

const SCALARS: [Scalar; 14] = [
    Scalar::I8,
    Scalar::U8,
    Scalar::I16,
    Scalar::U16,
    Scalar::I32,
    Scalar::U32,
    Scalar::I64,
    Scalar::U64,
    Scalar::I128,
    Scalar::U128,
    Scalar::I256,
    Scalar::U256,
    Scalar::I512,
    Scalar::U512,
];

fn declarations(scalar: Scalar) -> [PcuBinding<'static>; 4] {
    core::array::from_fn(|slot| {
        PcuBinding::value(
            None,
            5,
            u32::try_from(slot).unwrap(),
            PcuBindingStorageClass::Storage,
            if slot < 2 {
                Access::ReadOnly
            } else {
                Access::ReadWrite
            },
            Type::Scalar(scalar),
        )
    })
}

fn body(scalar: Scalar, index: Index) -> [Op<'static>; 5] {
    [
        Op::Data(Data::BindingLoad {
            result: Id(600),
            binding: Ref::new(5, 0),
            index,
        }),
        Op::Data(Data::BindingLoad {
            result: Id(601),
            binding: Ref::new(5, 1),
            index,
        }),
        Op::Data(Data::CheckedDivRem {
            value_type: Type::Scalar(scalar),
            flags: PcuIntegerDivFlags::CHECKED,
            quotient: Id(65534),
            remainder: Id(65535),
            lhs: Id(600),
            rhs: Id(601),
        }),
        Op::Data(Data::BindingStore {
            binding: Ref::new(5, 2),
            index,
            value: Id(65534),
        }),
        Op::Data(Data::BindingStore {
            binding: Ref::new(5, 3),
            index,
            value: Id(65535),
        }),
    ]
}

fn kernel<'a>(bindings: &'a [PcuBinding<'a>], ops: &'a [Op<'a>]) -> Kernel<'a> {
    Kernel {
        numerical_requirements: Kernel::DEFAULT_REQUIREMENTS,
        id: PcuKernelId(71),
        entry: PcuDispatchEntryPoint {
            name: "div-rem-roles",
            logical_shape: [19, 1, 1],
        },
        bindings,
        ports: &[],
        parameters: &[],
        ops,
        type_caps: Caps::empty(),
        feature_caps: PcuDispatchFeatureCaps::empty(),
    }
}

fn assessment(
    scalar: Scalar,
    declarations: &[PcuBinding<'_>],
    body: &[Op<'_>; 5],
    grid: bool,
) -> Result<CheckedIntegerDivRemOperandSchema, Error> {
    let direct = [
        body[0],
        body[1],
        body[2],
        body[3],
        body[4],
        Op::Control(PcuDispatchControlOp::Return),
    ];
    let loops = [
        Op::GridStrideLoop { extent: 19, body },
        Op::Control(PcuDispatchControlOp::Return),
    ];
    assess(
        &kernel(declarations, if grid { &loops } else { &direct }),
        Type::Scalar(scalar),
        Caps::for_scalar(scalar),
    )
}

#[test]
fn all_fourteen_widths_permuted_declarations_stores_and_math_roles() {
    for scalar in SCALARS {
        let original = declarations(scalar);
        for first in 0..4 {
            for second in 0..4 {
                for third in 0..4 {
                    for fourth in 0..4 {
                        let order = [first, second, third, fourth];
                        if order
                            .iter()
                            .enumerate()
                            .any(|(slot, item)| order[..slot].contains(item))
                        {
                            continue;
                        }
                        let declared = order.map(|slot| original[slot]);
                        for grid in [false, true] {
                            let index = if grid {
                                Index::GridStrideId
                            } else {
                                Index::InvocationId
                            };
                            for reverse_math in [false, true] {
                                for reverse_store in [false, true] {
                                    let mut ops = body(scalar, index);
                                    if reverse_math {
                                        let Op::Data(Data::CheckedDivRem { lhs, rhs, .. }) =
                                            &mut ops[2]
                                        else {
                                            unreachable!()
                                        };
                                        core::mem::swap(lhs, rhs);
                                    }
                                    if reverse_store {
                                        ops.swap(3, 4);
                                    }
                                    let roles = assessment(scalar, &declared, &ops, grid).unwrap();
                                    assert_eq!(
                                        roles.input_bindings(),
                                        &[original[0].reference(), original[1].reference()]
                                    );
                                    assert_eq!(
                                        roles.operand_inputs(),
                                        if reverse_math { [1, 0] } else { [0, 1] }
                                    );
                                    assert_eq!(roles.operand_indices(), [index; 2]);
                                    assert_eq!(roles.input_element_counts(19), [19, 19]);
                                    assert_eq!(
                                        roles.output_bindings(),
                                        [original[2].reference(), original[3].reference()]
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn repeated_resources_unused_declarations_and_independent_broadcasts() {
    for scalar in SCALARS {
        let declared = declarations(scalar);
        let single = [declared[3], declared[0], declared[2]];
        for grid in [false, true] {
            let index = if grid {
                Index::GridStrideId
            } else {
                Index::InvocationId
            };
            for indices in [
                [index, index],
                [Index::BindingElementZero, index],
                [index, Index::BindingElementZero],
                [Index::BindingElementZero; 2],
            ] {
                for repeated_binding in [false, true] {
                    for repeated_ssa in [false, true] {
                        let mut ops = body(scalar, index);
                        for (slot, instruction) in ops[..2].iter_mut().enumerate() {
                            let Op::Data(Data::BindingLoad {
                                binding,
                                index: got,
                                ..
                            }) = instruction
                            else {
                                unreachable!()
                            };
                            *got = indices[slot];
                            if repeated_binding {
                                *binding = declared[0].reference();
                            }
                        }
                        if repeated_ssa {
                            let Op::Data(Data::CheckedDivRem { rhs, .. }) = &mut ops[2] else {
                                unreachable!()
                            };
                            *rhs = Id(600);
                        }
                        for declarations in [&declared[..], &single[..]] {
                            if declarations.len() == 3 && !repeated_binding {
                                continue;
                            }
                            let roles = assessment(scalar, declarations, &ops, grid).unwrap();
                            let expected = [declared[0].reference(), declared[1].reference()];
                            assert_eq!(
                                roles.input_bindings(),
                                &expected[..if repeated_binding { 1 } else { 2 }]
                            );
                            assert_eq!(
                                roles.operand_inputs(),
                                if repeated_binding || repeated_ssa {
                                    [0, 0]
                                } else {
                                    [0, 1]
                                }
                            );
                            assert_eq!(
                                roles.operand_indices(),
                                if repeated_ssa {
                                    [indices[0]; 2]
                                } else {
                                    indices
                                }
                            );
                            let count = |got| {
                                if got == Index::BindingElementZero {
                                    1
                                } else {
                                    19
                                }
                            };
                            assert_eq!(
                                roles.input_element_counts(19),
                                if repeated_binding {
                                    [count(indices[0]).max(count(indices[1])), 0]
                                } else {
                                    [count(indices[0]), count(indices[1])]
                                }
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn existing_distinct_profile_stays_narrow_and_original_ids_are_accepted() {
    for scalar in SCALARS {
        let declared = declarations(scalar);
        let mut ops = body(scalar, Index::InvocationId);
        for operation in &mut ops {
            let Op::Data(data) = operation else {
                unreachable!()
            };
            match data {
                Data::BindingLoad { result, .. } => *result = Id(result.0 - 599),
                Data::CheckedDivRem {
                    quotient,
                    remainder,
                    lhs,
                    rhs,
                    ..
                } => {
                    *quotient = Id(3);
                    *remainder = Id(4);
                    *lhs = Id(1);
                    *rhs = Id(2);
                }
                Data::BindingStore { value, .. } => *value = Id(value.0 - 65531),
                _ => unreachable!(),
            }
        }
        let direct = [
            ops[0],
            ops[1],
            ops[2],
            ops[3],
            ops[4],
            Op::Control(PcuDispatchControlOp::Return),
        ];
        let ir = kernel(&declared, &direct);
        legacy(&ir, Type::Scalar(scalar), Caps::for_scalar(scalar)).unwrap();
        assess(&ir, Type::Scalar(scalar), Caps::for_scalar(scalar)).unwrap();
        let reordered = [declared[2], declared[1], declared[3], declared[0]];
        assert!(
            legacy(
                &kernel(&reordered, &direct),
                Type::Scalar(scalar),
                Caps::for_scalar(scalar)
            )
            .is_err()
        );
        assess(
            &kernel(&reordered, &direct),
            Type::Scalar(scalar),
            Caps::for_scalar(scalar),
        )
        .unwrap();
    }
}

#[test]
fn malformed_values_indices_and_dual_writes_reject() {
    let scalar = Scalar::I256;
    let declared = declarations(scalar);
    let original = body(scalar, Index::InvocationId);
    let bad_load = |id, binding, index| {
        Op::Data(Data::BindingLoad {
            result: id,
            binding,
            index,
        })
    };
    let bad_store = |binding, value, index| {
        Op::Data(Data::BindingStore {
            binding,
            value,
            index,
        })
    };
    for (slot, replacement, expected) in [
        (
            0,
            bad_load(Id(0), Ref::new(5, 0), Index::InvocationId),
            Error::InvalidValue(Id(0)),
        ),
        (
            1,
            bad_load(Id(600), Ref::new(5, 1), Index::InvocationId),
            Error::InvalidValue(Id(600)),
        ),
        (
            1,
            bad_load(Id(601), Ref::new(5, 2), Index::InvocationId),
            Error::InvalidBinding(Ref::new(5, 2)),
        ),
        (
            1,
            bad_load(Id(601), Ref::new(5, 1), Index::GridStrideId),
            Error::InvalidIndex(1),
        ),
        (
            3,
            bad_store(Ref::new(5, 2), Id(99), Index::InvocationId),
            Error::InvalidValue(Id(99)),
        ),
        (
            4,
            bad_store(Ref::new(5, 3), Id(65534), Index::InvocationId),
            Error::UnsupportedOperation(4),
        ),
        (
            4,
            bad_store(Ref::new(5, 2), Id(65535), Index::InvocationId),
            Error::InvalidBinding(Ref::new(5, 2)),
        ),
        (
            4,
            bad_store(Ref::new(5, 0), Id(65535), Index::InvocationId),
            Error::InvalidBinding(Ref::new(5, 0)),
        ),
        (
            4,
            bad_store(Ref::new(5, 3), Id(65535), Index::BindingElementZero),
            Error::InvalidIndex(4),
        ),
    ] {
        let mut ops = original;
        ops[slot] = replacement;
        assert_eq!(assessment(scalar, &declared, &ops, false), Err(expected));
    }
    for (lhs, rhs, quotient, remainder, expected) in [
        (Id(99), Id(601), Id(65534), Id(65535), Id(99)),
        (Id(600), Id(99), Id(65534), Id(65535), Id(99)),
        (Id(600), Id(601), Id(600), Id(65535), Id(600)),
        (Id(600), Id(601), Id(65534), Id(65534), Id(65534)),
        (Id(600), Id(601), Id(65534), Id(0), Id(0)),
    ] {
        let mut ops = original;
        ops[2] = Op::Data(Data::CheckedDivRem {
            value_type: Type::Scalar(scalar),
            flags: PcuIntegerDivFlags::CHECKED,
            lhs,
            rhs,
            quotient,
            remainder,
        });
        assert_eq!(
            assessment(scalar, &declared, &ops, false),
            Err(Error::InvalidValue(expected))
        );
    }
}

#[test]
fn interfaces_types_geometry_flags_and_requirements_are_bounded() {
    let scalar = Scalar::U512;
    let declared = declarations(scalar);
    let body = body(scalar, Index::InvocationId);
    let direct = [
        body[0],
        body[1],
        body[2],
        body[3],
        body[4],
        Op::Control(PcuDispatchControlOp::Return),
    ];
    let original = kernel(&declared, &direct);
    let duplicate = [declared[0], declared[0], declared[2], declared[3]];
    assert_eq!(
        assess(
            &kernel(&duplicate, &direct),
            Type::Scalar(scalar),
            Caps::UINT512
        ),
        Err(Error::DuplicateBinding(declared[0].reference()))
    );
    for shape in [[0, 1, 1], [19, 2, 1], [19, 1, 2]] {
        let mut changed = original;
        changed.entry.logical_shape = shape;
        assert_eq!(
            assess(&changed, Type::Scalar(scalar), Caps::UINT512),
            Err(Error::UnsupportedInterface)
        );
    }
    for bad in [Scalar::F32, Scalar::Bool] {
        assert_eq!(
            assess(&original, Type::Scalar(bad), Caps::for_scalar(bad)),
            Err(Error::UnsupportedInterface)
        );
    }
    assert_eq!(
        assess(&original, Type::Scalar(scalar), Caps::empty()),
        Err(Error::UnsupportedRequirements)
    );
    let mut clamped = original;
    clamped.numerical_requirements.range_policy = PcuRangePolicy::Clamp;
    assert_eq!(
        assess(&clamped, Type::Scalar(scalar), Caps::UINT512),
        Err(Error::UnsupportedRequirements)
    );
    let loops = [
        Op::GridStrideLoop {
            extent: 0,
            body: &body,
        },
        Op::Control(PcuDispatchControlOp::Return),
    ];
    assert_eq!(
        assess(
            &kernel(&declared, &loops),
            Type::Scalar(scalar),
            Caps::UINT512
        ),
        Err(Error::UnsupportedOperation(0))
    );
    let mut total_body = body;
    let Op::Data(Data::CheckedDivRem { flags, .. }) = &mut total_body[2] else {
        unreachable!()
    };
    *flags = PcuIntegerDivFlags::DIV_OR_ZERO;
    assert_eq!(
        assessment(scalar, &declared, &total_body, false),
        Err(Error::UnsupportedOperation(2))
    );
    assert_eq!(
        assess(
            &kernel(&declared, &direct[..5]),
            Type::Scalar(scalar),
            Caps::UINT512
        ),
        Err(Error::MissingReturn)
    );
    let short = [body[0], Op::Control(PcuDispatchControlOp::Return)];
    assert_eq!(
        assess(
            &kernel(&declared, &short),
            Type::Scalar(scalar),
            Caps::UINT512
        ),
        Err(Error::UnsupportedOperation(1))
    );
}
