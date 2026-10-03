use super::*;
#[rustfmt::skip]
use crate::{
    validate_checked_float_map_kernel,
    CheckedFloatMapValidationError,
    PcuBinding,
    PcuBindingAccess as Access,
    PcuBindingStorageClass as Storage,
    PcuCompoundArithmeticPolicy,
    PcuDispatchControlOp as Control,
    PcuDispatchDataOp as Data,
    PcuDispatchEntryPoint,
    PcuDispatchFeatureCaps,
    PcuDispatchFloatUnaryOp,
    PcuDispatchIndex as Index,
    PcuDispatchOp as Op,
    PcuDispatchValueId as Id,
    PcuFloatUnderflowPolicy,
    PcuKernelId,
    PcuNumericalMode,
    PcuPrecisionPolicy,
    PcuRangePolicy,
    PcuReproducibility,
    PcuValueTypeCaps,
};

const INPUT: PcuBindingRef = PcuBindingRef::new(4, 7);
const STAGE: PcuBindingRef = PcuBindingRef::new(1, 3);
const OUTPUT: PcuBindingRef = PcuBindingRef::new(8, 2);
const UNUSED: PcuBindingRef = PcuBindingRef::new(9, 11);

fn declarations(scalar: PcuScalarType) -> [PcuBinding<'static>; 4] {
    [
        (UNUSED, Access::ReadWrite),
        (OUTPUT, Access::WriteOnly),
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
    })
}

fn body(index: Index, reloaded: Index) -> [Op<'static>; 5] {
    [
        Op::Data(Data::BindingLoad {
            result: Id(0),
            binding: INPUT,
            index: Index::BindingElementZero,
        }),
        Op::Data(Data::BindingStore {
            binding: STAGE,
            index,
            value: Id(0),
        }),
        Op::Data(Data::BindingLoad {
            result: Id(1),
            binding: STAGE,
            index: reloaded,
        }),
        Op::Data(Data::BindingStore {
            binding: OUTPUT,
            index,
            value: Id(1),
        }),
        Op::Control(Control::Return),
    ]
}

fn kernel<'a>(bindings: &'a [PcuBinding<'a>], ops: &'a [Op<'a>]) -> PcuDispatchKernelIr<'a> {
    PcuDispatchKernelIr {
        id: PcuKernelId(314),
        entry: PcuDispatchEntryPoint {
            name: "ordered-transport",
            logical_shape: [23, 1, 1],
        },
        bindings,
        ports: &[],
        parameters: &[],
        ops,
        type_caps: PcuValueTypeCaps::empty(),
        feature_caps: PcuDispatchFeatureCaps::empty(),
        numerical_requirements: PcuDispatchKernelIr::DEFAULT_REQUIREMENTS,
    }
}

#[test]
fn all_scalar_identities_keep_actual_roles_and_original_request() {
    for scalar in PcuScalarType::ALL {
        let bindings = declarations(scalar);
        for grid in [false, true] {
            let index = if grid {
                Index::GridStrideId
            } else {
                Index::InvocationId
            };
            let body = body(index, index);
            let wrapped = [
                Op::GridStrideLoop {
                    extent: 65,
                    body: &body[..4],
                },
                Op::Control(Control::Return),
            ];
            let mut ir = kernel(&bindings, if grid { &wrapped } else { &body });
            for mode in [PcuNumericalMode::Boundary, PcuNumericalMode::Strict] {
                for compound in [
                    PcuCompoundArithmeticPolicy::Checked,
                    PcuCompoundArithmeticPolicy::BackendDefined,
                ] {
                    for precision in [
                        PcuPrecisionPolicy::Preserve,
                        PcuPrecisionPolicy::BackendOptimized,
                    ] {
                        for reproducibility in [
                            PcuReproducibility::Unspecified,
                            PcuReproducibility::PortableV1,
                        ] {
                            ir.numerical_requirements.numerical_mode = mode;
                            ir.numerical_requirements
                                .numerical_options
                                .compound_arithmetic = compound;
                            ir.numerical_requirements.numerical_options.precision = precision;
                            ir.numerical_requirements.numerical_options.reproducibility =
                                reproducibility;
                            check_request_policies(&mut ir, scalar, grid);
                        }
                    }
                }
            }
        }
    }
}

fn check_request_policies(ir: &mut PcuDispatchKernelIr<'_>, scalar: PcuScalarType, grid: bool) {
    for underflow in [
        PcuFloatUnderflowPolicy::IeeeAfterRounding,
        PcuFloatUnderflowPolicy::AllowGradualUnderflow,
        PcuFloatUnderflowPolicy::RejectSubnormalResult,
    ] {
        for range in [PcuRangePolicy::Reject, PcuRangePolicy::Clamp] {
            ir.numerical_requirements.float_underflow = underflow;
            ir.numerical_requirements.range_policy = range;
            let description = describe_scalar_transport_map::<3>(ir, scalar).unwrap();
            assert_eq!(description.scalar, scalar);
            assert_eq!(description.requirements, ir.numerical_requirements);
            assert_eq!(description.submitted_invocations, 23);
            let extent = if grid { 65 } else { 23 };
            assert_eq!(description.logical_extent, extent);
            assert_eq!(
                description
                    .resources()
                    .iter()
                    .map(|r| r.binding)
                    .collect::<std::vec::Vec<_>>(),
                [INPUT, STAGE, OUTPUT]
            );
            assert!(description.resource(UNUSED).is_none());
            let input = description.resource(INPUT).unwrap();
            assert_eq!(input.declared_access, Access::ReadOnly);
            assert_eq!(input.minimum_elements(), 1);
            assert_eq!(input.minimum_initial_read_elements, 1);
            let stage = description.resource(STAGE).unwrap();
            assert_eq!(stage.declared_access, Access::ReadWrite);
            assert_eq!(stage.minimum_read_elements, extent);
            assert_eq!(stage.minimum_write_elements, extent);
            assert_eq!(stage.minimum_initial_read_elements, 0);
            assert!(!stage.has_cross_index_read_write());
            let output = description.resource(OUTPUT).unwrap();
            assert_eq!(output.declared_access, Access::WriteOnly);
            assert_eq!(output.minimum_read_elements, 0);
            assert_eq!(output.minimum_write_elements, extent);
            assert_eq!(ir.fully_written_binding_elements(OUTPUT, 23), Some(extent));
        }
    }
}

#[test]
fn ordered_contents_distinguish_view_capacity_and_cross_index_dependencies() {
    let scalar = PcuScalarType::U512;
    let bindings = declarations(scalar);
    for extent in [1, 23] {
        let ops = body(Index::InvocationId, Index::BindingElementZero);
        let mut ir = kernel(&bindings, &ops);
        ir.entry.logical_shape[0] = extent;
        let description = describe_scalar_transport_map::<3>(&ir, scalar).unwrap();
        let stage = description.resource(STAGE).unwrap();
        assert_eq!(stage.minimum_elements(), extent);
        assert_eq!(stage.minimum_read_elements, 1);
        assert_eq!(stage.minimum_initial_read_elements, u32::from(extent != 1));
        assert_eq!(stage.has_cross_index_read_write(), extent != 1);
    }
    // An unused early load is still an actual original-content read.
    let ordered = body(Index::InvocationId, Index::InvocationId);
    let mut ops = std::vec![Op::Data(Data::BindingLoad {
        result: Id(2),
        binding: STAGE,
        index: Index::InvocationId,
    })];
    ops.extend(ordered);
    let ir = kernel(&bindings, &ops);
    let description = describe_scalar_transport_map::<3>(&ir, scalar).unwrap();
    assert_eq!(
        description
            .resource(STAGE)
            .unwrap()
            .minimum_initial_read_elements,
        23
    );
    assert_eq!(description.resources()[0].binding, STAGE);
}

#[test]
fn schema_capacity_and_ssa_scratch_are_caller_owned_not_ir_limits() {
    let scalar = PcuScalarType::F256;
    let bindings = declarations(scalar);
    let mut ops = body(Index::InvocationId, Index::InvocationId);
    let ir = kernel(&bindings, &ops);
    assert_eq!(
        describe_scalar_transport_map::<2>(&ir, scalar),
        Err(PcuScalarTransportError::InsufficientCapacity {
            capacity: 2,
            required_at_least: 3
        })
    );
    ops[0] = Op::Data(Data::BindingLoad {
        result: Id(500),
        binding: INPUT,
        index: Index::BindingElementZero,
    });
    ops[1] = Op::Data(Data::BindingStore {
        binding: STAGE,
        value: Id(500),
        index: Index::InvocationId,
    });
    let ir = kernel(&bindings, &ops);
    assert_eq!(
        describe_scalar_transport_map::<3>(&ir, scalar),
        Err(PcuScalarTransportError::InvalidValues(
            PcuTypedDispatchValidationError::ValueOutOfRange(Id(500))
        ))
    );
    let mut scratch = [Some(PcuValueType::Scalar(PcuScalarType::U32)); 501];
    let description =
        describe_scalar_transport_map_with_scratch::<3>(&ir, scalar, &mut scratch).unwrap();
    assert_eq!(description.resources().len(), 3);
    assert_eq!(scratch[500], Some(PcuValueType::Scalar(scalar)));
    assert_eq!(scratch[499], None);
}

#[test]
fn unused_declarations_have_no_universal_count_limit_but_keep_access_metadata() {
    let scalar = PcuScalarType::I512;
    let mut bindings = std::vec::Vec::from(declarations(scalar));
    for binding in 100..160 {
        bindings.push(PcuBinding::value(
            None,
            3,
            binding,
            Storage::Storage,
            Access::ReadWrite,
            PcuValueType::Scalar(scalar),
        ));
    }
    assert_eq!(bindings.len(), 64);
    let ops = body(Index::InvocationId, Index::InvocationId);
    let ir = kernel(&bindings, &ops);
    let description = describe_scalar_transport_map::<3>(&ir, scalar).unwrap();
    assert_eq!(description.resources().len(), 3);
    assert!(description.resource(UNUSED).is_none());
    assert!(description.resource(PcuBindingRef::new(3, 159)).is_none());
    assert_eq!(ir.bindings[63].access, Access::ReadWrite);
    // An invalid unused declaration still refuses the entire interface.
    bindings[63].storage = Storage::Shared;
    assert!(describe_scalar_transport_map::<3>(&kernel(&bindings, &ops), scalar).is_err());
}

#[test]
fn malformed_metadata_indices_and_values_are_rejected_even_for_unused_arguments() {
    let scalar = PcuScalarType::F16;
    let bindings = declarations(scalar);
    let ops = body(Index::InvocationId, Index::InvocationId);
    let mut ir = kernel(&bindings, &ops);
    ir.entry.logical_shape = [0, 1, 1];
    assert_eq!(
        describe_scalar_transport_map::<3>(&ir, scalar),
        Err(PcuScalarTransportError::InvalidLogicalShape)
    );
    ir.entry.logical_shape = [23, 2, 1];
    assert!(describe_scalar_transport_map::<3>(&ir, scalar).is_err());
    ir.entry.logical_shape = [23, 1, 1];
    ir.feature_caps = PcuDispatchFeatureCaps::COOPERATIVE_SCRATCHPAD;
    assert_eq!(
        describe_scalar_transport_map::<3>(&ir, scalar),
        Err(PcuScalarTransportError::UnsupportedRequirements)
    );
    let mut duplicates = bindings;
    duplicates[0] = duplicates[1];
    assert_eq!(
        describe_scalar_transport_map::<3>(&kernel(&duplicates, &ops), scalar),
        Err(PcuScalarTransportError::DuplicateBinding(OUTPUT))
    );
    let mut wrong_type = bindings;
    wrong_type[0] = declarations(PcuScalarType::BF16)[0];
    assert!(describe_scalar_transport_map::<3>(&kernel(&wrong_type, &ops), scalar).is_err());
    let mut forbidden = bindings;
    forbidden[3].access = Access::WriteOnly;
    assert_eq!(
        describe_scalar_transport_map::<3>(&kernel(&forbidden, &ops), scalar),
        Err(PcuScalarTransportError::InvalidBinding(INPUT))
    );
    forbidden = bindings;
    forbidden[2].access = Access::ReadOnly;
    assert_eq!(
        describe_scalar_transport_map::<3>(&kernel(&forbidden, &ops), scalar),
        Err(PcuScalarTransportError::InvalidBinding(STAGE))
    );
    let mut wrong = ops;
    wrong[1] = Op::Data(Data::BindingStore {
        binding: STAGE,
        value: Id(0),
        index: Index::BindingElementZero,
    });
    assert_eq!(
        describe_scalar_transport_map::<3>(&kernel(&bindings, &wrong), scalar),
        Err(PcuScalarTransportError::InvalidIndex(1))
    );
    wrong = ops;
    wrong[2] = Op::Data(Data::BindingLoad {
        binding: STAGE,
        result: Id(0),
        index: Index::InvocationId,
    });
    assert_eq!(
        describe_scalar_transport_map::<3>(&kernel(&bindings, &wrong), scalar),
        Err(PcuScalarTransportError::InvalidValues(
            PcuTypedDispatchValidationError::DuplicateValue(Id(0))
        ))
    );
    wrong = ops;
    wrong[3] = Op::Data(Data::BindingStore {
        binding: OUTPUT,
        value: Id(77),
        index: Index::InvocationId,
    });
    assert_eq!(
        describe_scalar_transport_map::<3>(&kernel(&bindings, &wrong), scalar),
        Err(PcuScalarTransportError::InvalidValues(
            PcuTypedDispatchValidationError::UndefinedValue(Id(77))
        ))
    );
}

#[test]
fn transport_cannot_bypass_checked_arithmetic_or_acquire_numeric_admission() {
    let scalar = PcuScalarType::F32;
    let bindings = declarations(scalar);
    let ops = body(Index::InvocationId, Index::InvocationId);
    let ir = kernel(&bindings, &ops);
    assert!(describe_scalar_transport_map::<3>(&ir, scalar).is_ok());
    assert_eq!(
        validate_checked_float_map_kernel(
            &ir,
            PcuValueType::Scalar(scalar),
            PcuValueTypeCaps::for_scalar(scalar)
        ),
        Err(CheckedFloatMapValidationError::MissingCheckedOperation)
    );
    let mut numeric = ops;
    numeric[2] = Op::Data(Data::CheckedFloatUnary {
        result: Id(1),
        value: Id(0),
        value_type: PcuValueType::Scalar(scalar),
        op: PcuDispatchFloatUnaryOp::Neg,
        range_policy: PcuRangePolicy::Reject,
        underflow_policy: PcuFloatUnderflowPolicy::IeeeAfterRounding,
    });
    assert_eq!(
        describe_scalar_transport_map::<3>(&kernel(&bindings, &numeric), scalar),
        Err(PcuScalarTransportError::UnsupportedOperation(2))
    );
    let empty = [Op::Control(Control::Return)];
    assert_eq!(
        describe_scalar_transport_map::<3>(&kernel(&bindings, &empty), scalar),
        Err(PcuScalarTransportError::MissingLoadOrStore)
    );
    let nested = [
        Op::GridStrideLoop {
            extent: 0,
            body: &ops[..4],
        },
        Op::Control(Control::Return),
    ];
    assert!(describe_scalar_transport_map::<3>(&kernel(&bindings, &nested), scalar).is_err());
    assert_eq!(
        describe_scalar_transport_map::<3>(&kernel(&bindings, &ops[..4]), scalar),
        Err(PcuScalarTransportError::MissingReturn)
    );
}
