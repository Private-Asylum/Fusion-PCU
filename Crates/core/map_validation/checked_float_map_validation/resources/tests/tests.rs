use super::*;
#[rustfmt::skip]
use crate::{
    PcuBindingAccess,
    PcuBindingRef,
    PcuBinding,
    PcuBindingStorageClass,
    PcuCompoundArithmeticPolicy,
    PcuDispatchControlOp,
    PcuDispatchDataOp as Data,
    PcuDispatchEntryPoint,
    PcuDispatchFeatureCaps,
    PcuDispatchFloatBinaryOp as Binary,
    PcuDispatchFloatUnaryOp as Unary,
    PcuDispatchIndex as Index,
    PcuDispatchOp as Op,
    PcuDispatchValueId as Id,
    PcuFloatUnderflowPolicy as Underflow,
    PcuKernelId,
    PcuNumericalMode,
    PcuPrecisionPolicy,
    PcuRangePolicy as Range,
    PcuReproducibility,
    PcuScalarType as Scalar,
};

const SHARED: PcuBindingRef = PcuBindingRef::new(4, 2);
const BROADCAST: PcuBindingRef = PcuBindingRef::new(6, 11);
const FIRST_OUTPUT: PcuBindingRef = PcuBindingRef::new(1, 3);
const SECOND_OUTPUT: PcuBindingRef = PcuBindingRef::new(7, 9);
const UNUSED_READ: PcuBindingRef = PcuBindingRef::new(9, 5);
const UNUSED_WRITE: PcuBindingRef = PcuBindingRef::new(8, 4);
const FORMATS: [Scalar; 6] = [
    Scalar::F16,
    Scalar::BF16,
    Scalar::F32,
    Scalar::F64,
    Scalar::F8E4M3FN,
    Scalar::F8E5M2,
];

fn bindings(scalar: Scalar) -> [PcuBinding<'static>; 6] {
    [
        (UNUSED_READ, PcuBindingAccess::ReadOnly),
        (SECOND_OUTPUT, PcuBindingAccess::WriteOnly),
        (SHARED, PcuBindingAccess::ReadWrite),
        (BROADCAST, PcuBindingAccess::ReadOnly),
        (FIRST_OUTPUT, PcuBindingAccess::ReadWrite),
        (UNUSED_WRITE, PcuBindingAccess::ReadWrite),
    ]
    .map(|(reference, access)| {
        PcuBinding::value(
            None,
            reference.set,
            reference.binding,
            PcuBindingStorageClass::Storage,
            access,
            PcuValueType::Scalar(scalar),
        )
    })
}

fn body(scalar: Scalar, index: Index, range: Range, underflow: Underflow) -> [Op<'static>; 9] {
    [
        Op::Data(Data::BindingLoad {
            result: Id(0),
            binding: SHARED,
            index: Index::BindingElementZero,
        }),
        Op::Data(Data::BindingLoad {
            result: Id(1),
            binding: SHARED,
            index,
        }),
        Op::Data(Data::BindingLoad {
            result: Id(2),
            binding: BROADCAST,
            index: Index::BindingElementZero,
        }),
        Op::Data(Data::CheckedFloatUnary {
            result: Id(3),
            value: Id(0),
            value_type: PcuValueType::Scalar(scalar),
            op: Unary::Neg,
            range_policy: range,
            underflow_policy: underflow,
        }),
        Op::Data(Data::CheckedFloatBinary {
            result: Id(4),
            lhs: Id(1),
            rhs: Id(3),
            value_type: PcuValueType::Scalar(scalar),
            op: Binary::Add,
            range_policy: Range::Reject,
            underflow_policy: Underflow::AllowGradualUnderflow,
        }),
        Op::Data(Data::BindingStore {
            binding: FIRST_OUTPUT,
            value: Id(4),
            index,
        }),
        Op::Data(Data::BindingStore {
            binding: SHARED,
            value: Id(3),
            index,
        }),
        Op::Data(Data::BindingStore {
            binding: SECOND_OUTPUT,
            value: Id(4),
            index,
        }),
        Op::Control(PcuDispatchControlOp::Return),
    ]
}

fn kernel<'a>(bindings: &'a [PcuBinding<'a>], ops: &'a [Op<'a>]) -> PcuDispatchKernelIr<'a> {
    PcuDispatchKernelIr {
        numerical_requirements: PcuDispatchKernelIr::DEFAULT_REQUIREMENTS,
        id: PcuKernelId(101),
        entry: PcuDispatchEntryPoint {
            name: "composed-actual-resources",
            logical_shape: [23, 1, 1],
        },
        bindings,
        ops,
        ports: &[],
        parameters: &[],
        type_caps: PcuValueTypeCaps::empty(),
        feature_caps: PcuDispatchFeatureCaps::empty(),
    }
}

fn assess<const N: usize>(
    ir: &PcuDispatchKernelIr<'_>,
    scalar: Scalar,
) -> Result<
    CheckedScalarMapResourceSchema<N>,
    CheckedScalarMapResourceError<CheckedFloatMapValidationError>,
> {
    assess_checked_float_map_resources(
        ir,
        PcuValueType::Scalar(scalar),
        PcuValueTypeCaps::for_scalar(scalar),
    )
}

fn check_projection<const N: usize>(schema: &CheckedScalarMapResourceSchema<N>) {
    assert_eq!(schema.resources().len(), 4);
    assert_eq!(
        schema
            .resources()
            .iter()
            .map(|resource| resource.binding)
            .collect::<std::vec::Vec<_>>(),
        [SHARED, BROADCAST, FIRST_OUTPUT, SECOND_OUTPUT]
    );
    let shared = schema.resource(SHARED).unwrap();
    assert_eq!(shared.declared_access, PcuBindingAccess::ReadWrite);
    assert_eq!(shared.minimum_read_elements, schema.logical_extent);
    assert_eq!(shared.minimum_initial_read_elements, schema.logical_extent);
    assert_eq!(shared.minimum_write_elements, schema.logical_extent);
    assert_eq!(shared.minimum_elements(), schema.logical_extent);
    assert!(shared.reads_element_zero);
    assert!(shared.has_cross_index_read_write());
    let broadcast = schema.resource(BROADCAST).unwrap();
    assert_eq!(broadcast.minimum_read_elements, 1);
    assert_eq!(broadcast.minimum_initial_read_elements, 1);
    assert_eq!(broadcast.minimum_write_elements, 0);
    assert_eq!(broadcast.minimum_elements(), 1);
    assert!(broadcast.reads_element_zero);
    assert!(!broadcast.has_cross_index_read_write());
    for output in [FIRST_OUTPUT, SECOND_OUTPUT] {
        let resource = schema.resource(output).unwrap();
        assert_eq!(resource.minimum_read_elements, 0);
        assert_eq!(resource.minimum_initial_read_elements, 0);
        assert_eq!(resource.minimum_write_elements, schema.logical_extent);
        assert_eq!(resource.minimum_elements(), schema.logical_extent);
        assert!(!resource.reads_element_zero);
        assert!(!resource.has_cross_index_read_write());
    }
    assert!(schema.resource(UNUSED_READ).is_none());
    assert!(schema.resource(UNUSED_WRITE).is_none());
}

fn ordered_body(
    scalar: Scalar,
    index: Index,
    loaded_index: Index,
    read_before_store: bool,
) -> std::vec::Vec<Op<'static>> {
    let mut body = std::vec![
        Op::Data(Data::BindingLoad {
            result: Id(0),
            binding: BROADCAST,
            index: Index::BindingElementZero
        }),
        Op::Data(Data::CheckedFloatUnary {
            result: Id(1),
            value: Id(0),
            value_type: PcuValueType::Scalar(scalar),
            op: Unary::Neg,
            range_policy: Range::Reject,
            underflow_policy: Underflow::IeeeAfterRounding,
        }),
    ];
    if read_before_store {
        // An unused load still requires incoming contents; no dead-read
        // optimization or alias permission is inferred by the descriptor.
        body.push(Op::Data(Data::BindingLoad {
            result: Id(2),
            binding: FIRST_OUTPUT,
            index: loaded_index,
        }));
    }
    body.extend([
        Op::Data(Data::BindingStore {
            binding: FIRST_OUTPUT,
            value: Id(1),
            index,
        }),
        Op::Data(Data::BindingLoad {
            result: Id(3),
            binding: FIRST_OUTPUT,
            index: loaded_index,
        }),
        Op::Data(Data::CheckedFloatBinary {
            result: Id(4),
            lhs: Id(3),
            rhs: Id(1),
            value_type: PcuValueType::Scalar(scalar),
            op: Binary::Add,
            range_policy: Range::Reject,
            underflow_policy: Underflow::IeeeAfterRounding,
        }),
        Op::Data(Data::BindingStore {
            binding: SECOND_OUTPUT,
            value: Id(4),
            index,
        }),
    ]);
    body
}

#[test]
fn ordered_stores_separate_original_contents_from_required_view_capacity() {
    for scalar in FORMATS {
        for logical_extent in [1, 23] {
            for grid in [false, true] {
                for broadcast in [false, true] {
                    for read_before_store in [false, true] {
                        let index = if grid {
                            Index::GridStrideId
                        } else {
                            Index::InvocationId
                        };
                        let loaded_index = if broadcast {
                            Index::BindingElementZero
                        } else {
                            index
                        };
                        let declarations = bindings(scalar);
                        let body = ordered_body(scalar, index, loaded_index, read_before_store);
                        let mut outer = std::vec![Op::Control(PcuDispatchControlOp::Return)];
                        if grid {
                            outer.insert(
                                0,
                                Op::GridStrideLoop {
                                    body: &body,
                                    extent: logical_extent,
                                },
                            );
                        } else {
                            outer.splice(0..0, body.iter().copied());
                        }
                        let mut ir = kernel(&declarations, &outer);
                        ir.entry.logical_shape = [if grid { 3 } else { logical_extent }, 1, 1];
                        let schema = assess::<3>(&ir, scalar).unwrap();
                        let working = schema.resource(FIRST_OUTPUT).unwrap();
                        let read_extent = if broadcast { 1 } else { logical_extent };
                        assert_eq!(working.minimum_read_elements, read_extent);
                        assert_eq!(working.minimum_write_elements, logical_extent);
                        assert_eq!(working.minimum_elements(), logical_extent);
                        let original_needed =
                            read_before_store || (broadcast && logical_extent > 1);
                        assert_eq!(
                            working.minimum_initial_read_elements,
                            if original_needed { read_extent } else { 0 }
                        );
                        // The generic host-staging fact must agree with the
                        // validated descriptor without shrinking view capacity.
                        assert_eq!(
                            ir.fully_written_binding_elements(
                                FIRST_OUTPUT,
                                ir.entry.logical_shape[0],
                            ),
                            if original_needed {
                                None
                            } else {
                                Some(logical_extent)
                            }
                        );
                        assert_eq!(
                            ir.fully_written_binding_elements(
                                SECOND_OUTPUT,
                                ir.entry.logical_shape[0],
                            ),
                            Some(logical_extent)
                        );
                        assert_eq!(
                            ir.fully_written_binding_elements(
                                BROADCAST,
                                ir.entry.logical_shape[0],
                            ),
                            None
                        );
                        assert_eq!(
                            working.has_cross_index_read_write(),
                            broadcast && logical_extent > 1
                        );
                        assert_eq!(
                            schema
                                .resource(BROADCAST)
                                .unwrap()
                                .minimum_initial_read_elements,
                            1
                        );
                        assert_eq!(
                            schema
                                .resource(SECOND_OUTPUT)
                                .unwrap()
                                .minimum_initial_read_elements,
                            0
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn all_formats_project_actual_roles_independent_of_declarations_and_scoped_policies() {
    let mut cases = 0;
    for scalar in FORMATS {
        for range in [Range::Reject, Range::Clamp] {
            for underflow in [
                Underflow::IeeeAfterRounding,
                Underflow::RejectSubnormalResult,
                Underflow::AllowGradualUnderflow,
            ] {
                for grid in [false, true] {
                    for reversed in [false, true] {
                        let mut declarations = bindings(scalar);
                        if reversed {
                            declarations.reverse();
                        }
                        let index = if grid {
                            Index::GridStrideId
                        } else {
                            Index::InvocationId
                        };
                        let body = body(scalar, index, range, underflow);
                        let outer = [
                            Op::GridStrideLoop {
                                body: &body[..8],
                                extent: 23,
                            },
                            Op::Control(PcuDispatchControlOp::Return),
                        ];
                        let mut ir = kernel(&declarations, if grid { &outer } else { &body });
                        ir.numerical_requirements.numerical_mode = PcuNumericalMode::Strict;
                        ir.numerical_requirements.numerical_options.reproducibility =
                            PcuReproducibility::PortableV1;
                        ir.numerical_requirements
                            .numerical_options
                            .compound_arithmetic = PcuCompoundArithmeticPolicy::BackendDefined;
                        ir.numerical_requirements.numerical_options.precision =
                            PcuPrecisionPolicy::BackendOptimized;
                        ir.numerical_requirements.range_policy = range;
                        ir.numerical_requirements.float_underflow = underflow;
                        if grid {
                            ir.entry.logical_shape[0] = 3;
                        }
                        let schema = assess::<4>(&ir, scalar).unwrap();
                        assert_eq!(schema.value_type, PcuValueType::Scalar(scalar));
                        assert_eq!(schema.requirements, ir.numerical_requirements);
                        assert_eq!(schema.submitted_invocations, if grid { 3 } else { 23 });
                        assert_eq!(schema.logical_extent, 23);
                        check_projection(&schema);
                        cases += 1;
                    }
                }
            }
        }
    }
    assert_eq!(cases, 144);
}

#[test]
fn broadcast_read_and_indexed_write_remain_distinct_roles_without_alias_permission() {
    let declarations = bindings(Scalar::F32);
    let mut body = body(
        Scalar::F32,
        Index::InvocationId,
        Range::Reject,
        Underflow::IeeeAfterRounding,
    );
    let Op::Data(Data::BindingLoad { index, .. }) = &mut body[1] else {
        unreachable!()
    };
    *index = Index::BindingElementZero;
    let ir = kernel(&declarations, &body);
    let schema = assess::<4>(&ir, Scalar::F32).unwrap();
    let shared = schema.resource(SHARED).unwrap();
    assert_eq!(shared.minimum_read_elements, 1);
    assert_eq!(shared.minimum_write_elements, 23);
    assert_eq!(shared.minimum_elements(), 23);
}

#[test]
fn capacity_counts_unique_actual_accesses_including_dead_loads_but_not_unused_arguments() {
    let declarations = bindings(Scalar::F64);
    let body = body(
        Scalar::F64,
        Index::InvocationId,
        Range::Reject,
        Underflow::IeeeAfterRounding,
    );
    let ir = kernel(&declarations, &body);
    assert_eq!(
        assess::<3>(&ir, Scalar::F64),
        Err(CheckedScalarMapResourceError::InsufficientCapacity {
            capacity: 3,
            required_at_least: 4
        })
    );
    assert_eq!(
        assess::<0>(&ir, Scalar::F64),
        Err(CheckedScalarMapResourceError::InsufficientCapacity {
            capacity: 0,
            required_at_least: 1
        })
    );
    check_projection(&assess::<4>(&ir, Scalar::F64).unwrap());
    check_projection(&assess::<16>(&ir, Scalar::F64).unwrap());
    // Id(2) has no SSA consumer, but its load remains an actual instruction read.
    assert_eq!(
        assess::<4>(&ir, Scalar::F64)
            .unwrap()
            .resource(BROADCAST)
            .unwrap()
            .minimum_read_elements,
        1
    );
}

#[test]
fn invalid_unused_metadata_access_and_ssa_are_rejected_before_capacity_projection() {
    let mut declarations = bindings(Scalar::F32);
    let mut body = body(
        Scalar::F32,
        Index::InvocationId,
        Range::Reject,
        Underflow::IeeeAfterRounding,
    );
    declarations[0].binding_type = crate::PcuBindingType::Value(PcuValueType::Scalar(Scalar::Bool));
    assert_eq!(
        assess::<0>(&kernel(&declarations, &body), Scalar::F32),
        Err(CheckedScalarMapResourceError::InvalidMap(
            CheckedFloatMapValidationError::UnsupportedRequirements
        ))
    );
    declarations = bindings(Scalar::F32);
    declarations[3].access = PcuBindingAccess::WriteOnly;
    assert_eq!(
        assess::<0>(&kernel(&declarations, &body), Scalar::F32),
        Err(CheckedScalarMapResourceError::InvalidMap(
            CheckedFloatMapValidationError::InvalidBinding(BROADCAST)
        ))
    );
    declarations = bindings(Scalar::F32);
    let Op::Data(Data::BindingStore { value, .. }) = &mut body[7] else {
        unreachable!()
    };
    *value = Id(31);
    assert_eq!(
        assess::<0>(&kernel(&declarations, &body), Scalar::F32),
        Err(CheckedScalarMapResourceError::InvalidMap(
            CheckedFloatMapValidationError::InvalidSsa
        ))
    );
}

#[test]
fn store_before_later_read_retains_first_access_order_and_independent_spans() {
    let declarations = bindings(Scalar::F32);
    let original = body(
        Scalar::F32,
        Index::InvocationId,
        Range::Reject,
        Underflow::IeeeAfterRounding,
    );
    let later_load = Op::Data(Data::BindingLoad {
        result: Id(5),
        binding: FIRST_OUTPUT,
        index: Index::BindingElementZero,
    });
    let mut body = original[..8].to_vec();
    body.push(later_load);
    body.push(Op::Control(PcuDispatchControlOp::Return));
    let schema = assess::<4>(&kernel(&declarations, &body), Scalar::F32).unwrap();
    assert_eq!(schema.resources()[2].binding, FIRST_OUTPUT);
    let first = schema.resource(FIRST_OUTPUT).unwrap();
    assert_eq!(first.minimum_read_elements, 1);
    assert_eq!(first.minimum_write_elements, 23);
}

#[test]
fn constant_only_checked_map_needs_only_its_real_destination() {
    for (scalar, value) in [
        (
            Scalar::F32,
            crate::PcuParameterValue::F32(1.0_f32.to_bits()),
        ),
        (
            Scalar::F64,
            crate::PcuParameterValue::F64(1.0_f64.to_bits()),
        ),
    ] {
        let declarations = bindings(scalar);
        let body = [
            Op::Data(Data::Constant {
                result: Id(0),
                value,
            }),
            Op::Data(Data::CheckedFloatUnary {
                result: Id(1),
                value: Id(0),
                value_type: PcuValueType::Scalar(scalar),
                op: Unary::Neg,
                range_policy: Range::Reject,
                underflow_policy: Underflow::IeeeAfterRounding,
            }),
            Op::Data(Data::BindingStore {
                binding: SECOND_OUTPUT,
                value: Id(1),
                index: Index::InvocationId,
            }),
            Op::Control(PcuDispatchControlOp::Return),
        ];
        let schema = assess::<1>(&kernel(&declarations, &body), scalar).unwrap();
        assert_eq!(schema.resources().len(), 1);
        assert_eq!(schema.resources()[0].binding, SECOND_OUTPUT);
        assert_eq!(schema.resources()[0].minimum_read_elements, 0);
        assert_eq!(schema.resources()[0].minimum_write_elements, 23);
        assert!(schema.resource(SHARED).is_none());
    }
}

#[test]
fn maximum_logical_grid_extent_is_described_without_physical_backend_limits() {
    let declarations = bindings(Scalar::F32);
    let body = body(
        Scalar::F32,
        Index::GridStrideId,
        Range::Reject,
        Underflow::IeeeAfterRounding,
    );
    let outer = [
        Op::GridStrideLoop {
            body: &body[..8],
            extent: u32::MAX,
        },
        Op::Control(PcuDispatchControlOp::Return),
    ];
    let mut ir = kernel(&declarations, &outer);
    ir.entry.logical_shape = [1, 1, 1];
    let schema = assess::<4>(&ir, Scalar::F32).unwrap();
    assert_eq!(schema.submitted_invocations, 1);
    assert_eq!(schema.logical_extent, u32::MAX);
    check_projection(&schema);
}

#[test]
fn larger_caller_capacity_supports_more_than_sixteen_real_resources() {
    let mut declarations = bindings(Scalar::F32).to_vec();
    let mut ops = body(
        Scalar::F32,
        Index::InvocationId,
        Range::Reject,
        Underflow::IeeeAfterRounding,
    )[..8]
        .to_vec();
    for binding in 0..17 {
        let reference = PcuBindingRef::new(20, binding);
        declarations.push(PcuBinding::value(
            None,
            reference.set,
            reference.binding,
            PcuBindingStorageClass::Storage,
            PcuBindingAccess::WriteOnly,
            PcuValueType::f32(),
        ));
        ops.push(Op::Data(Data::BindingStore {
            binding: reference,
            value: Id(4),
            index: Index::InvocationId,
        }));
    }
    ops.push(Op::Control(PcuDispatchControlOp::Return));
    let ir = kernel(&declarations, &ops);
    assert_eq!(
        assess::<16>(&ir, Scalar::F32),
        Err(CheckedScalarMapResourceError::InsufficientCapacity {
            capacity: 16,
            required_at_least: 17
        })
    );
    let schema = assess::<21>(&ir, Scalar::F32).unwrap();
    assert_eq!(schema.resources().len(), 21);
    let last = schema.resource(PcuBindingRef::new(20, 16)).unwrap();
    assert_eq!(last.minimum_read_elements, 0);
    assert_eq!(last.minimum_write_elements, 23);
}

#[test]
fn cross_index_dependency_uses_logical_extent_and_survives_indexed_read_merging() {
    for scalar in FORMATS {
        let declarations = bindings(scalar);
        for logical_extent in [1, 2, 23] {
            for grid in [false, true] {
                let index = if grid {
                    Index::GridStrideId
                } else {
                    Index::InvocationId
                };
                let instructions = body(scalar, index, Range::Reject, Underflow::IeeeAfterRounding);
                let outer = [
                    Op::GridStrideLoop {
                        body: &instructions[..8],
                        extent: logical_extent,
                    },
                    Op::Control(PcuDispatchControlOp::Return),
                ];
                let mut ir = kernel(&declarations, if grid { &outer } else { &instructions });
                ir.entry.logical_shape[0] = if grid { 1 } else { logical_extent };
                let schema = assess_checked_float_map_resources::<4>(
                    &ir,
                    PcuValueType::Scalar(scalar),
                    PcuValueTypeCaps::for_scalar(scalar),
                )
                .unwrap();
                let shared = schema.resource(SHARED).unwrap();
                assert_eq!(shared.minimum_read_elements, logical_extent);
                assert!(shared.reads_element_zero);
                assert_eq!(shared.has_cross_index_read_write(), logical_extent > 1);
                assert!(
                    !schema
                        .resource(BROADCAST)
                        .unwrap()
                        .has_cross_index_read_write()
                );
                // Replacing the element-zero read by an indexed read gives
                // same-lane input/output behavior, with unchanged minimum spans.
                let mut indexed = instructions;
                if let Op::Data(Data::BindingLoad { index: actual, .. }) = &mut indexed[0] {
                    *actual = index;
                }
                let indexed_outer = [
                    Op::GridStrideLoop {
                        body: &indexed[..8],
                        extent: logical_extent,
                    },
                    Op::Control(PcuDispatchControlOp::Return),
                ];
                ir.ops = if grid { &indexed_outer } else { &indexed };
                let safe = assess_checked_float_map_resources::<4>(
                    &ir,
                    PcuValueType::Scalar(scalar),
                    PcuValueTypeCaps::for_scalar(scalar),
                )
                .unwrap()
                .resource(SHARED)
                .unwrap();
                assert_eq!(safe.minimum_read_elements, shared.minimum_read_elements);
                assert_eq!(safe.minimum_write_elements, shared.minimum_write_elements);
                assert!(!safe.reads_element_zero);
                assert!(!safe.has_cross_index_read_write());
            }
        }
    }
}

#[test]
fn overwritten_single_checked_result_retains_actual_inputs_and_complete_writer() {
    for scalar in FORMATS {
        let declarations = bindings(scalar);
        for grid in [false, true] {
            let index = if grid {
                Index::GridStrideId
            } else {
                Index::InvocationId
            };
            let instructions = [
                Op::Data(Data::BindingLoad {
                    result: Id(0),
                    binding: SHARED,
                    index,
                }),
                Op::Data(Data::BindingLoad {
                    result: Id(1),
                    binding: BROADCAST,
                    index,
                }),
                Op::Data(Data::CheckedFloatBinary {
                    result: Id(2),
                    lhs: Id(0),
                    rhs: Id(1),
                    value_type: PcuValueType::Scalar(scalar),
                    op: Binary::Div,
                    range_policy: Range::Reject,
                    underflow_policy: Underflow::IeeeAfterRounding,
                }),
                Op::Data(Data::BindingLoad {
                    result: Id(3),
                    binding: SHARED,
                    index,
                }),
                Op::Data(Data::BindingStore {
                    binding: FIRST_OUTPUT,
                    value: Id(3),
                    index,
                }),
                Op::Control(PcuDispatchControlOp::Return),
            ];
            let outer = [
                Op::GridStrideLoop {
                    body: &instructions[..5],
                    extent: 23,
                },
                Op::Control(PcuDispatchControlOp::Return),
            ];
            let mut ir = kernel(&declarations, if grid { &outer } else { &instructions });
            ir.entry.logical_shape[0] = if grid { 3 } else { 23 };
            let schema = assess::<3>(&ir, scalar).unwrap();
            assert_eq!(schema.resources().len(), 3);
            assert_eq!(schema.logical_extent, 23);
            // The unused SSA result does not make its division operands unread.
            for reference in [SHARED, BROADCAST] {
                let resource = schema.resource(reference).unwrap();
                assert_eq!(resource.minimum_read_elements, 23);
                assert_eq!(resource.minimum_initial_read_elements, 23);
                assert_eq!(resource.minimum_write_elements, 0);
            }
            let output = schema.resource(FIRST_OUTPUT).unwrap();
            assert_eq!(output.minimum_read_elements, 0);
            assert_eq!(output.minimum_initial_read_elements, 0);
            assert_eq!(output.minimum_write_elements, 23);
            assert_eq!(
                ir.fully_written_binding_elements(FIRST_OUTPUT, ir.entry.logical_shape[0]),
                Some(23)
            );
            assert!(schema.resource(UNUSED_READ).is_none());
            assert!(schema.resource(UNUSED_WRITE).is_none());
        }
    }
}
