use super::*;
#[rustfmt::skip]
use crate::{
    PcuBinding,
    PcuBindingAccess,
    PcuBindingStorageClass,
    PcuCompoundArithmeticPolicy,
    PcuDispatchControlOp,
    PcuDispatchEntryPoint,
    PcuDispatchFeatureCaps,
    PcuDispatchIndex,
    PcuDispatchIntegerBinaryOp,
    PcuDispatchValueId,
    PcuFloatUnderflowPolicy,
    PcuKernelId,
    PcuNumericalMode,
    PcuPrecisionPolicy,
    PcuRangePolicy,
    PcuScalarType,
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
fn bindings(scalar: PcuScalarType) -> [PcuBinding<'static>; 4] {
    core::array::from_fn(|slot| {
        PcuBinding::value(
            None,
            0,
            u32::try_from(slot).unwrap(),
            PcuBindingStorageClass::Storage,
            if slot < 2 {
                PcuBindingAccess::ReadOnly
            } else {
                PcuBindingAccess::ReadWrite
            },
            PcuValueType::Scalar(scalar),
        )
    })
}
fn body(scalar: PcuScalarType, range: PcuRangePolicy, grid: bool) -> [PcuDispatchOp<'static>; 9] {
    let index = if grid {
        PcuDispatchIndex::GridStrideId
    } else {
        PcuDispatchIndex::InvocationId
    };
    let load = |result, slot, index| {
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
            result: PcuDispatchValueId(result),
            binding: PcuBindingRef::new(0, slot),
            index,
        })
    };
    let binary = |result, lhs, rhs, op| {
        PcuDispatchOp::Data(PcuDispatchDataOp::CheckedIntegerBinary {
            result: PcuDispatchValueId(result),
            lhs: PcuDispatchValueId(lhs),
            rhs: PcuDispatchValueId(rhs),
            value_type: PcuValueType::Scalar(scalar),
            range_policy: range,
            op,
        })
    };
    let store = |slot, value| {
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
            binding: PcuBindingRef::new(0, slot),
            value: PcuDispatchValueId(value),
            index,
        })
    };
    [
        load(0, 0, index),
        load(1, 1, PcuDispatchIndex::BindingElementZero),
        binary(2, 0, 1, PcuDispatchIntegerBinaryOp::Add),
        store(2, 2),
        load(3, 2, index),
        binary(4, 3, 0, PcuDispatchIntegerBinaryOp::Mul),
        binary(5, 4, 0, PcuDispatchIntegerBinaryOp::Sub),
        store(3, 5),
        PcuDispatchOp::Control(PcuDispatchControlOp::Return),
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
        id: PcuKernelId(719),
        entry: PcuDispatchEntryPoint {
            name: "portable-composed",
            logical_shape: [3, 1, 1],
        },
        bindings,
        ops,
        ports: &[],
        parameters: &[],
        type_caps: PcuValueTypeCaps::empty(),
        feature_caps: PcuDispatchFeatureCaps::empty(),
        numerical_requirements: requirements,
    }
}
#[test]
fn exact_resources_headers_and_store_reload_for_all_fourteen_widths() {
    for scalar in INTEGERS {
        for range in [PcuRangePolicy::Reject, PcuRangePolicy::Clamp] {
            for grid in [false, true] {
                let bindings = bindings(scalar);
                let body = body(scalar, range, grid);
                let wrapped = [
                    PcuDispatchOp::GridStrideLoop {
                        extent: 19,
                        body: &body[..8],
                    },
                    PcuDispatchOp::Control(PcuDispatchControlOp::Return),
                ];
                let mut kernel = kernel(&bindings, if grid { &wrapped } else { &body }, range);
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
                                let result =
                                    describe_portable_v1_checked_integer_composed_map::<4>(&kernel)
                                        .unwrap();
                                assert_eq!(result.requirements, kernel.numerical_requirements);
                                assert_eq!(result.value_type, PcuValueType::Scalar(scalar));
                                assert_eq!(result.submitted_invocations, 3);
                                assert_eq!(result.logical_extent, if grid { 19 } else { 3 });
                                assert_eq!(result.resources().len(), 4);
                                assert_eq!(
                                    result
                                        .resource(PcuBindingRef::new(0, 1))
                                        .unwrap()
                                        .minimum_initial_read_elements,
                                    1
                                );
                                assert_eq!(
                                    result
                                        .resource(PcuBindingRef::new(0, 2))
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
    }
}
#[test]
fn missing_request_range_mismatch_and_capacity_are_not_admission() {
    let bindings = bindings(PcuScalarType::U256);
    let ops = body(PcuScalarType::U256, PcuRangePolicy::Reject, false);
    let mut kernel = kernel(&bindings, &ops, PcuRangePolicy::Reject);
    assert!(matches!(
        describe_portable_v1_checked_integer_composed_map::<3>(&kernel),
        Err(PcuPortableV1IntegerComposedMapError::InvalidResources(
            CheckedScalarMapResourceError::InsufficientCapacity { .. }
        ))
    ));
    kernel.numerical_requirements.range_policy = PcuRangePolicy::Clamp;
    assert_eq!(
        describe_portable_v1_checked_integer_composed_map::<4>(&kernel),
        Err(PcuPortableV1IntegerComposedMapError::RangeMismatch(2))
    );
    kernel
        .numerical_requirements
        .numerical_options
        .reproducibility = PcuReproducibility::Unspecified;
    assert_eq!(
        describe_portable_v1_checked_integer_composed_map::<4>(&kernel),
        Err(PcuPortableV1IntegerComposedMapError::NotRequested)
    );
}
#[test]
fn cross_index_reads_and_bad_ssa_remain_refused() {
    let bindings = bindings(PcuScalarType::I512);
    let mut ops = body(PcuScalarType::I512, PcuRangePolicy::Reject, false);
    let PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad { index, .. }) = &mut ops[4] else {
        panic!()
    };
    *index = PcuDispatchIndex::BindingElementZero;
    assert_eq!(
        describe_portable_v1_checked_integer_composed_map::<4>(&kernel(
            &bindings,
            &ops,
            PcuRangePolicy::Reject
        )),
        Err(PcuPortableV1IntegerComposedMapError::CrossIndexDependency(
            PcuBindingRef::new(0, 2)
        ))
    );
    let PcuDispatchOp::Data(PcuDispatchDataOp::CheckedIntegerBinary { lhs, .. }) = &mut ops[5]
    else {
        panic!()
    };
    *lhs = PcuDispatchValueId(999);
    assert!(matches!(
        describe_portable_v1_checked_integer_composed_map::<4>(&kernel(
            &bindings,
            &ops,
            PcuRangePolicy::Reject
        )),
        Err(PcuPortableV1IntegerComposedMapError::InvalidResources(_))
    ));
}

#[test]
fn cyclic_and_nested_grid_bodies_refuse_before_recursive_scanners() {
    static CYCLE: [PcuDispatchOp<'static>; 1] = [PcuDispatchOp::GridStrideLoop {
        extent: 19,
        body: &CYCLE,
    }];
    let integer_bindings = bindings(PcuScalarType::U512);
    let ops = [
        PcuDispatchOp::GridStrideLoop {
            extent: 19,
            body: &CYCLE,
        },
        PcuDispatchOp::Control(PcuDispatchControlOp::Return),
    ];
    assert_eq!(
        describe_portable_v1_checked_integer_composed_map::<4>(&kernel(
            &integer_bindings,
            &ops,
            PcuRangePolicy::Reject
        )),
        Err(PcuPortableV1IntegerComposedMapError::UnsupportedStructure)
    );
    let mut direct = kernel(&integer_bindings, &ops, PcuRangePolicy::Reject);
    direct.ops = &CYCLE;
    assert_eq!(
        describe_portable_v1_checked_integer_composed_map::<4>(&direct),
        Err(PcuPortableV1IntegerComposedMapError::UnsupportedStructure)
    );
    // The ordinary map admission must refuse independently of cache or policy.
    assert_eq!(
        crate::validate_checked_integer_map_kernel(
            &kernel(&integer_bindings, &ops, PcuRangePolicy::Reject),
            PcuValueType::Scalar(PcuScalarType::U512),
            PcuValueTypeCaps::for_scalar(PcuScalarType::U512),
        ),
        Err(CheckedIntegerMapValidationError::UnsupportedOperation(0))
    );
    let float_bindings = bindings(PcuScalarType::F64);
    assert_eq!(
        crate::validate_checked_float_map_kernel(
            &kernel(&float_bindings, &ops, PcuRangePolicy::Reject),
            PcuValueType::Scalar(PcuScalarType::F64),
            PcuValueTypeCaps::for_scalar(PcuScalarType::F64),
        ),
        Err(crate::CheckedFloatMapValidationError::UnsupportedOperation(
            0
        ))
    );
}

#[test]
fn cyclic_ir_refuses_transport_conversion_and_primitive_admission() {
    static CYCLE: [PcuDispatchOp<'static>; 1] = [PcuDispatchOp::GridStrideLoop {
        extent: 19,
        body: &CYCLE,
    }];
    let ops = [
        PcuDispatchOp::GridStrideLoop {
            extent: 19,
            body: &CYCLE,
        },
        PcuDispatchOp::Control(PcuDispatchControlOp::Return),
    ];
    for scalar in [
        PcuScalarType::U32,
        PcuScalarType::U64,
        PcuScalarType::U512,
        PcuScalarType::F16,
        PcuScalarType::BF16,
        PcuScalarType::F64,
    ] {
        let declarations = bindings(scalar);
        let identity = kernel(&declarations[..2], &ops, PcuRangePolicy::Reject);
        let arithmetic = kernel(&declarations[..3], &ops, PcuRangePolicy::Reject);
        assert!(identity.has_nested_grid_stride_loop());
        assert!(crate::describe_scalar_transport_map::<4>(&identity, scalar).is_err());
        assert!(crate::validate_scalar_identity_kernel(&identity, scalar).is_err());
        assert!(crate::validate_scalar_broadcast_kernel(&identity, scalar).is_err());
        assert!(crate::validate_u32_identity_kernel(&identity).is_err());
        assert!(crate::validate_u64_identity_kernel(&identity).is_err());
        assert!(crate::validate_f16_identity_kernel(&identity).is_err());
        assert!(crate::validate_bf16_identity_kernel(&identity).is_err());
        assert!(
            crate::validate_checked_float_conversion_map_kernel(
                &arithmetic,
                PcuValueTypeCaps::FLOAT32 | PcuValueTypeCaps::FLOAT64
            )
            .is_err()
        );
        assert!(
            crate::validate_checked_float_binary_kernel(
                &arithmetic,
                PcuValueType::Scalar(PcuScalarType::F64),
                crate::PcuDispatchFloatBinaryOp::Add,
                PcuFloatUnderflowPolicy::IeeeAfterRounding,
                PcuValueTypeCaps::FLOAT64
            )
            .is_err()
        );
        assert!(
            crate::validate_integer_checked_binary_kernel(
                &arithmetic,
                PcuValueType::Scalar(PcuScalarType::U512),
                PcuDispatchIntegerBinaryOp::Add,
                PcuValueTypeCaps::for_scalar(PcuScalarType::U512)
            )
            .is_err()
        );
        assert!(
            crate::validate_integer_checked_div_rem_kernel(
                &arithmetic,
                PcuValueType::Scalar(PcuScalarType::U512),
                PcuValueTypeCaps::for_scalar(PcuScalarType::U512)
            )
            .is_err()
        );
    }
}
