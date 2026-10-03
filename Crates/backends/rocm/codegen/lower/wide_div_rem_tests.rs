//! Cold joint-output lowering tests; no wider device offer is inferred from source emission.
#[rustfmt::skip]
use fusion_pcu::{
    PcuBinding,
    PcuBindingAccess,
    PcuBindingRef,
    PcuBindingStorageClass,
    PcuCompoundArithmeticPolicy,
    PcuDispatchControlOp,
    PcuDispatchDataOp,
    PcuDispatchEntryPoint,
    PcuDispatchFeatureCaps,
    PcuDispatchIndex,
    PcuDispatchKernelIr,
    PcuDispatchOp,
    PcuDispatchValueId,
    PcuFloatUnderflowPolicy,
    PcuImplementationRequirements,
    PcuKernelId,
    PcuNumericalMode,
    PcuPrecisionPolicy,
    PcuRangePolicy,
    PcuReproducibility,
    PcuScalarType,
    PcuValueType,
    PcuValueTypeCaps,
};
use fusion_pcu::model::PcuIntegerDivFlags;
const WIDE: [PcuScalarType; 6] = [
    PcuScalarType::I128,
    PcuScalarType::U128,
    PcuScalarType::I256,
    PcuScalarType::U256,
    PcuScalarType::I512,
    PcuScalarType::U512,
];
fn body(scalar: PcuScalarType, index: PcuDispatchIndex) -> [PcuDispatchOp<'static>; 5] {
    [
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
            result: PcuDispatchValueId(1),
            binding: PcuBindingRef::new(0, 0),
            index,
        }),
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
            result: PcuDispatchValueId(2),
            binding: PcuBindingRef::new(0, 1),
            index,
        }),
        PcuDispatchOp::Data(PcuDispatchDataOp::CheckedDivRem {
            value_type: PcuValueType::Scalar(scalar),
            flags: PcuIntegerDivFlags::empty(),
            quotient: PcuDispatchValueId(3),
            remainder: PcuDispatchValueId(4),
            lhs: PcuDispatchValueId(1),
            rhs: PcuDispatchValueId(2),
        }),
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
            binding: PcuBindingRef::new(0, 2),
            index,
            value: PcuDispatchValueId(3),
        }),
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
            binding: PcuBindingRef::new(0, 3),
            index,
            value: PcuDispatchValueId(4),
        }),
    ]
}
fn bindings(scalar: PcuScalarType) -> [PcuBinding<'static>; 4] {
    std::array::from_fn(|i| {
        PcuBinding::value(
            None,
            0,
            u32::try_from(i).unwrap(),
            PcuBindingStorageClass::Storage,
            if i < 2 {
                PcuBindingAccess::ReadOnly
            } else {
                PcuBindingAccess::WriteOnly
            },
            PcuValueType::Scalar(scalar),
        )
    })
}
fn kernel<'a>(
    ops: &'a [PcuDispatchOp<'a>],
    bindings: &'a [PcuBinding<'a>],
    grid: bool,
) -> PcuDispatchKernelIr<'a> {
    PcuDispatchKernelIr {
        numerical_requirements: PcuDispatchKernelIr::DEFAULT_REQUIREMENTS,
        id: PcuKernelId(1),
        entry: PcuDispatchEntryPoint {
            name: "joint_wide_div_rem",
            logical_shape: [if grid { 3 } else { 65 }, 1, 1],
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
fn wide_joint_div_rem_emits_one_pass_with_both_exact_outputs_for_all_permitted_tuples() {
    for scalar in WIDE {
        for grid in [false, true] {
            let index = if grid {
                PcuDispatchIndex::GridStrideId
            } else {
                PcuDispatchIndex::InvocationId
            };
            let body = body(scalar, index);
            let bindings = bindings(scalar);
            let ops = if grid {
                vec![
                    PcuDispatchOp::GridStrideLoop {
                        extent: 65,
                        body: &body,
                    },
                    PcuDispatchOp::Control(PcuDispatchControlOp::Return),
                ]
            } else {
                body.into_iter()
                    .chain([PcuDispatchOp::Control(PcuDispatchControlOp::Return)])
                    .collect()
            };
            for mode in [PcuNumericalMode::Boundary, PcuNumericalMode::Strict] {
                for compound in [
                    PcuCompoundArithmeticPolicy::Checked,
                    PcuCompoundArithmeticPolicy::BackendDefined,
                ] {
                    for precision in [
                        PcuPrecisionPolicy::Preserve,
                        PcuPrecisionPolicy::BackendOptimized,
                    ] {
                        for uf in [
                            PcuFloatUnderflowPolicy::IeeeAfterRounding,
                            PcuFloatUnderflowPolicy::AllowGradualUnderflow,
                            PcuFloatUnderflowPolicy::RejectSubnormalResult,
                        ] {
                            let mut ir = kernel(&ops, &bindings, grid);
                            ir.numerical_requirements = PcuImplementationRequirements {
                                numerical_mode: mode,
                                float_underflow: uf,
                                ..PcuImplementationRequirements::DEFAULT
                            };
                            ir.numerical_requirements
                                .numerical_options
                                .compound_arithmetic = compound;
                            ir.numerical_requirements.numerical_options.precision = precision;
                            let source = super::lower_dispatch_to_hip_rtc_source(&ir).unwrap();
                            assert_eq!(
                                source
                                    .matches(" = fusion_checked_wide_div_rem(v1, v2,")
                                    .count(),
                                1
                            );
                            assert!(source.contains("v3 = fusion_joint_3.quotient;"));
                            assert!(source.contains("v4 = fusion_joint_3.remainder;"));
                            assert!(source.contains("minimum && minus_one"));
                            assert!(source.contains("return {quotient, remainder, 2u}"));
                            assert!(!source.contains("__int128"));
                        }
                    }
                }
            }
        }
    }
}
#[test]
fn wide_joint_div_rem_rejects_reserved_flags_clamp_and_output_aliasing() {
    for scalar in WIDE {
        let bindings = bindings(scalar);
        let mut body = body(scalar, PcuDispatchIndex::InvocationId);
        let mut ops = body
            .into_iter()
            .chain([PcuDispatchOp::Control(PcuDispatchControlOp::Return)])
            .collect::<Vec<_>>();
        let mut ir = kernel(&ops, &bindings, false);
        ir.numerical_requirements.range_policy = PcuRangePolicy::Clamp;
        assert!(super::lower_dispatch_to_hip_rtc_source(&ir).is_err());
        if let PcuDispatchOp::Data(PcuDispatchDataOp::CheckedDivRem { flags, .. }) = &mut ops[2] {
            *flags = PcuIntegerDivFlags::DIV_OR_ZERO;
        }
        assert!(super::lower_dispatch_to_hip_rtc_source(&kernel(&ops, &bindings, false)).is_err());
        if let PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore { binding, .. }) = &mut body[4] {
            *binding = PcuBindingRef::new(0, 2);
        }
        ops = body
            .into_iter()
            .chain([PcuDispatchOp::Control(PcuDispatchControlOp::Return)])
            .collect();
        assert!(super::lower_dispatch_to_hip_rtc_source(&kernel(&ops, &bindings, false)).is_err());
    }
}

#[test]
fn legacy_eight_div_rem_rejects_clamp_header_before_compilation() {
    for scalar in [
        PcuScalarType::I8,
        PcuScalarType::U8,
        PcuScalarType::I16,
        PcuScalarType::U16,
        PcuScalarType::I32,
        PcuScalarType::U32,
        PcuScalarType::I64,
        PcuScalarType::U64,
    ] {
        let bindings = bindings(scalar);
        let ops = body(scalar, PcuDispatchIndex::InvocationId)
            .into_iter()
            .chain([PcuDispatchOp::Control(PcuDispatchControlOp::Return)])
            .collect::<Vec<_>>();
        let mut ir = kernel(&ops, &bindings, false);
        ir.numerical_requirements.range_policy = PcuRangePolicy::Clamp;
        assert!(
            super::lower_dispatch_to_hip_rtc_source(&ir).is_err(),
            "division Clamp has no admitted quotient/remainder contract"
        );
    }
}

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

fn role_body(scalar: PcuScalarType, grid: bool, role: usize) -> Vec<PcuDispatchOp<'static>> {
    let index = if grid {
        PcuDispatchIndex::GridStrideId
    } else {
        PcuDispatchIndex::InvocationId
    };
    let mut body = body(scalar, index);
    if matches!(role, 0 | 1 | 3 | 4)
        && let PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad { binding, index, .. }) =
            &mut body[1]
    {
        *binding = PcuBindingRef::new(0, 0);
        if role == 3 {
            *index = PcuDispatchIndex::BindingElementZero;
        }
    }
    if role == 4
        && let PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad { index, .. }) = &mut body[0]
    {
        *index = PcuDispatchIndex::BindingElementZero;
    }
    if role == 2 {
        if let PcuDispatchOp::Data(PcuDispatchDataOp::CheckedDivRem { lhs, rhs, .. }) = &mut body[2]
        {
            std::mem::swap(lhs, rhs);
        }
        body.swap(3, 4);
    }
    body.into()
}

#[test]
#[allow(clippy::too_many_lines)] // Complete exact request matrix shares one detached fixture.
fn fourteen_joint_operand_schemas_and_portable_headers_preserve_unique_resources() {
    for scalar in INTEGERS {
        for grid in [false, true] {
            for role in 0..5 {
                let mut bindings = bindings(scalar).to_vec();
                if role == 0 {
                    bindings.remove(1);
                }
                if role == 2 {
                    bindings.rotate_left(2);
                }
                if role == 4 {
                    bindings.swap(0, 1);
                }
                let body = role_body(scalar, grid, role);
                let ops = if grid {
                    vec![
                        PcuDispatchOp::GridStrideLoop {
                            extent: 65,
                            body: &body,
                        },
                        PcuDispatchOp::Control(PcuDispatchControlOp::Return),
                    ]
                } else {
                    body.iter()
                        .copied()
                        .chain([PcuDispatchOp::Control(PcuDispatchControlOp::Return)])
                        .collect()
                };
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
                                for reproducibility in [
                                    PcuReproducibility::Unspecified,
                                    PcuReproducibility::PortableV1,
                                ] {
                                    let mut ir = kernel(&ops, &bindings, grid);
                                    ir.numerical_requirements = PcuImplementationRequirements {
                                        numerical_mode: mode,
                                        numerical_options: fusion_pcu::PcuNumericalOptions {
                                            compound_arithmetic: compound,
                                            precision,
                                            reproducibility,
                                        },
                                        float_underflow: underflow,
                                        range_policy: PcuRangePolicy::Reject,
                                    };
                                    let expected =
                                        fusion_pcu::assess_checked_integer_div_rem_operands(
                                            &ir,
                                            PcuValueType::Scalar(scalar),
                                            PcuValueTypeCaps::for_scalar(scalar),
                                        )
                                        .unwrap();
                                    let projection = super::map_binding_projection(&ir).unwrap();
                                    assert!(matches!(
                                        projection,
                                        super::MapBindingProjection::Joint(_)
                                    ));
                                    assert_eq!(
                                        projection.input_bindings(),
                                        expected.input_bindings()
                                    );
                                    for output in expected.output_bindings() {
                                        assert!(projection.contains_output(output));
                                    }
                                    assert_eq!(
                                        expected.output_bindings(),
                                        [PcuBindingRef::new(0, 2), PcuBindingRef::new(0, 3)]
                                    );
                                    assert_eq!(
                                        expected.input_element_counts(65),
                                        if role == 2 { [65, 65] } else { [65, 0] }
                                    );
                                    if reproducibility == PcuReproducibility::PortableV1 {
                                        assert_eq!(
                                            fusion_pcu::describe_portable_v1_integer_div_rem_map(
                                                &ir
                                            )
                                            .unwrap()
                                            .operands,
                                            expected
                                        );
                                    }
                                    let source =
                                        super::lower_dispatch_to_hip_rtc_source(&ir).unwrap();
                                    let declaration = source
                                        .split("void fusion_kernel(")
                                        .nth(1)
                                        .unwrap()
                                        .split(')')
                                        .next()
                                        .unwrap();
                                    for binding in expected.output_bindings() {
                                        assert!(declaration.contains(&format!(
                                            "binding_{}_{}",
                                            binding.set, binding.binding
                                        )));
                                    }
                                    assert_eq!(declaration.contains("binding_0_1"), role == 2);
                                    let mut malformed = ir;
                                    malformed.numerical_requirements.range_policy =
                                        PcuRangePolicy::Clamp;
                                    assert!(
                                        super::lower_dispatch_to_hip_rtc_source(&malformed)
                                            .is_err()
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
