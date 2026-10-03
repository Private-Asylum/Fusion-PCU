//! Eligibility does not advertise any independently unqualified execution provider.
use super::*;
#[rustfmt::skip]
use crate::{
    PcuBinding,
    PcuBindingAccess,
    PcuBindingRef,
    PcuBindingStorageClass,
    PcuCompoundArithmeticPolicy,
    PcuDispatchEntryPoint,
    PcuDispatchFeatureCaps,
    PcuDispatchIndex,
    PcuDispatchValueId,
    PcuFloatUnderflowPolicy,
    PcuKernelId,
    PcuNumericalMode,
    PcuPrecisionPolicy,
};

const INTEGERS: [PcuScalarType; 14] = [
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
];

fn bindings(scalar: PcuScalarType) -> [PcuBinding<'static>; 3] {
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

fn body(
    scalar: PcuScalarType,
    operation: PcuDispatchIntegerBinaryOp,
    range: PcuRangePolicy,
    grid: bool,
) -> [PcuDispatchOp<'static>; 4] {
    let index = if grid {
        PcuDispatchIndex::GridStrideId
    } else {
        PcuDispatchIndex::InvocationId
    };
    [
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
            result: PcuDispatchValueId(1),
            binding: PcuBindingRef::new(0, 0),
            index,
        }),
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
            result: PcuDispatchValueId(2),
            binding: PcuBindingRef::new(0, 0),
            index: PcuDispatchIndex::BindingElementZero,
        }),
        PcuDispatchOp::Data(PcuDispatchDataOp::CheckedIntegerBinary {
            value_type: PcuValueType::Scalar(scalar),
            op: operation,
            range_policy: range,
            result: PcuDispatchValueId(3),
            lhs: PcuDispatchValueId(2),
            rhs: PcuDispatchValueId(1),
        }),
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
            binding: PcuBindingRef::new(0, 2),
            index,
            value: PcuDispatchValueId(3),
        }),
    ]
}

fn kernel<'a>(
    bindings: &'a [PcuBinding<'a>],
    ops: &'a [PcuDispatchOp<'a>],
    range: PcuRangePolicy,
) -> PcuDispatchKernelIr<'a> {
    let mut requirements = PcuDispatchKernelIr::DEFAULT_REQUIREMENTS;
    requirements.range_policy = range;
    requirements.numerical_options.reproducibility = PcuReproducibility::PortableV1;
    PcuDispatchKernelIr {
        id: PcuKernelId(71),
        entry: PcuDispatchEntryPoint {
            name: "portable-integer",
            logical_shape: [3, 1, 1],
        },
        bindings,
        ports: &[],
        parameters: &[],
        ops,
        type_caps: PcuValueTypeCaps::empty(),
        feature_caps: PcuDispatchFeatureCaps::empty(),
        numerical_requirements: requirements,
    }
}

#[test]
fn all_widths_policies_and_request_axes_keep_exact_independent_operand_roles() {
    for scalar in INTEGERS {
        // Declaration order is unrelated to mathematical or unique-input order.
        let declared = bindings(scalar);
        let declared = [declared[2], declared[1], declared[0]];
        for operation in [
            PcuDispatchIntegerBinaryOp::Add,
            PcuDispatchIntegerBinaryOp::Sub,
            PcuDispatchIntegerBinaryOp::Mul,
        ] {
            for range in [PcuRangePolicy::Reject, PcuRangePolicy::Clamp] {
                for grid in [false, true] {
                    let body = body(scalar, operation, range, grid);
                    let direct = [
                        body[0],
                        body[1],
                        body[2],
                        body[3],
                        PcuDispatchOp::Control(PcuDispatchControlOp::Return),
                    ];
                    let loop_ops = [
                        PcuDispatchOp::GridStrideLoop {
                            extent: 19,
                            body: &body,
                        },
                        PcuDispatchOp::Control(PcuDispatchControlOp::Return),
                    ];
                    let mut kernel =
                        kernel(&declared, if grid { &loop_ops } else { &direct }, range);
                    for mode in [PcuNumericalMode::Boundary, PcuNumericalMode::Strict] {
                        for compound in [
                            PcuCompoundArithmeticPolicy::Checked,
                            PcuCompoundArithmeticPolicy::BackendDefined,
                        ] {
                            for precision in [
                                PcuPrecisionPolicy::Preserve,
                                PcuPrecisionPolicy::BackendOptimized,
                            ] {
                                for underflow in [
                                    PcuFloatUnderflowPolicy::IeeeAfterRounding,
                                    PcuFloatUnderflowPolicy::AllowGradualUnderflow,
                                    PcuFloatUnderflowPolicy::RejectSubnormalResult,
                                ] {
                                    kernel.numerical_requirements.numerical_mode = mode;
                                    kernel
                                        .numerical_requirements
                                        .numerical_options
                                        .compound_arithmetic = compound;
                                    kernel.numerical_requirements.numerical_options.precision =
                                        precision;
                                    kernel.numerical_requirements.float_underflow = underflow;
                                    let result = describe_portable_v1_integer_map(&kernel).unwrap();
                                    assert_eq!(result.scalar, scalar);
                                    assert_eq!(result.operation, operation);
                                    assert_eq!(result.range_policy, range);
                                    assert_eq!(result.submitted_invocations, 3);
                                    assert_eq!(result.logical_extent, if grid { 19 } else { 3 });
                                    assert_eq!(
                                        result.operands.input_bindings(),
                                        &[PcuBindingRef::new(0, 0)]
                                    );
                                    assert_eq!(result.operands.operand_inputs(), [0, 0]);
                                    assert_eq!(
                                        result.operands.operand_indices()[0],
                                        PcuDispatchIndex::BindingElementZero
                                    );
                                    assert_eq!(
                                        result.operands.output_binding(),
                                        PcuBindingRef::new(0, 2)
                                    );
                                    assert_eq!(
                                        result
                                            .operands
                                            .input_element_counts(result.logical_extent as usize),
                                        [result.logical_extent as usize, 0]
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
fn eligibility_rejects_nonintegers_missing_request_geometry_and_malformed_ssa() {
    let declared = bindings(PcuScalarType::U64);
    let good = body(
        PcuScalarType::U64,
        PcuDispatchIntegerBinaryOp::Sub,
        PcuRangePolicy::Reject,
        false,
    );
    let mut ops = [
        good[0],
        good[1],
        good[2],
        good[3],
        PcuDispatchOp::Control(PcuDispatchControlOp::Return),
    ];
    let mut candidate = kernel(&declared, &ops, PcuRangePolicy::Reject);
    candidate
        .numerical_requirements
        .numerical_options
        .reproducibility = PcuReproducibility::Unspecified;
    assert_eq!(
        describe_portable_v1_integer_map(&candidate),
        Err(PcuPortableV1IntegerMapError::NotRequested)
    );
    candidate
        .numerical_requirements
        .numerical_options
        .reproducibility = PcuReproducibility::PortableV1;
    for shape in [[0, 1, 1], [3, 2, 1], [3, 1, 2]] {
        candidate.entry.logical_shape = shape;
        assert_eq!(
            describe_portable_v1_integer_map(&candidate),
            Err(PcuPortableV1IntegerMapError::InvalidLogicalShape(shape))
        );
    }
    ops[2] = PcuDispatchOp::Data(PcuDispatchDataOp::CheckedIntegerBinary {
        value_type: PcuValueType::u64(),
        op: PcuDispatchIntegerBinaryOp::Sub,
        range_policy: PcuRangePolicy::Clamp,
        result: PcuDispatchValueId(3),
        lhs: PcuDispatchValueId(2),
        rhs: PcuDispatchValueId(1),
    });
    assert!(matches!(
        describe_portable_v1_integer_map(&kernel(&declared, &ops, PcuRangePolicy::Reject)),
        Err(PcuPortableV1IntegerMapError::InvalidOperands(_))
    ));
    ops[2] = PcuDispatchOp::Data(PcuDispatchDataOp::CheckedIntegerBinary {
        value_type: PcuValueType::u64(),
        op: PcuDispatchIntegerBinaryOp::Sub,
        range_policy: PcuRangePolicy::Reject,
        result: PcuDispatchValueId(3),
        lhs: PcuDispatchValueId(99),
        rhs: PcuDispatchValueId(1),
    });
    assert!(matches!(
        describe_portable_v1_integer_map(&kernel(&declared, &ops, PcuRangePolicy::Reject)),
        Err(PcuPortableV1IntegerMapError::InvalidOperands(_))
    ));
    for scalar in [
        PcuScalarType::Bool,
        PcuScalarType::I4,
        PcuScalarType::U4,
        PcuScalarType::F16,
        PcuScalarType::BF16,
        PcuScalarType::F8E4M3FN,
        PcuScalarType::F8E5M2,
        PcuScalarType::F32,
        PcuScalarType::F64,
        PcuScalarType::F128,
        PcuScalarType::F256,
    ] {
        let declared = bindings(scalar);
        let invalid = body(
            scalar,
            PcuDispatchIntegerBinaryOp::Add,
            PcuRangePolicy::Reject,
            false,
        );
        let invalid = [
            invalid[0],
            invalid[1],
            invalid[2],
            invalid[3],
            PcuDispatchOp::Control(PcuDispatchControlOp::Return),
        ];
        assert_eq!(
            describe_portable_v1_integer_map(&kernel(&declared, &invalid, PcuRangePolicy::Reject)),
            Err(PcuPortableV1IntegerMapError::UnsupportedScalar)
        );
    }
}

#[test]
fn one_or_two_actual_inputs_never_require_a_fabricated_second_resource() {
    for scalar in INTEGERS {
        let declared = bindings(scalar);
        let original = body(
            scalar,
            PcuDispatchIntegerBinaryOp::Sub,
            PcuRangePolicy::Reject,
            false,
        );
        let mut ops = [
            original[0],
            original[1],
            original[2],
            original[3],
            PcuDispatchOp::Control(PcuDispatchControlOp::Return),
        ];
        let minimal = [declared[2], declared[0]];
        let repeated =
            describe_portable_v1_integer_map(&kernel(&minimal, &ops, PcuRangePolicy::Reject))
                .unwrap();
        assert_eq!(
            repeated.operands.input_bindings(),
            &[PcuBindingRef::new(0, 0)]
        );
        assert_eq!(repeated.operands.input_element_counts(3), [3, 0]);
        ops[1] = PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
            result: PcuDispatchValueId(2),
            binding: PcuBindingRef::new(0, 1),
            index: PcuDispatchIndex::BindingElementZero,
        });
        let separate =
            describe_portable_v1_integer_map(&kernel(&declared, &ops, PcuRangePolicy::Reject))
                .unwrap();
        assert_eq!(
            separate.operands.input_bindings(),
            &[PcuBindingRef::new(0, 0), PcuBindingRef::new(0, 1)]
        );
        assert_eq!(separate.operands.operand_inputs(), [1, 0]);
        assert_eq!(separate.operands.input_element_counts(3), [3, 1]);
    }
}
