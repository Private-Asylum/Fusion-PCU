//! Actual-resource ABI projection and same-resource broadcast hazard boundaries.
use super::*;
#[rustfmt::skip]
use fusion_pcu::{
    PcuBinding,
    PcuDispatchEntryPoint,
    PcuDispatchFloatBinaryOp,
    PcuFloatUnderflowPolicy,
    PcuKernelId,
    PcuRangePolicy,
};

fn body(
    scalar: PcuScalarType,
    index: PcuDispatchIndex,
    broadcast: bool,
) -> Vec<PcuDispatchOp<'static>> {
    vec![
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
            result: PcuDispatchValueId(1),
            binding: PcuBindingRef::new(0, 0),
            index: if broadcast {
                PcuDispatchIndex::BindingElementZero
            } else {
                index
            },
        }),
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
            result: PcuDispatchValueId(2),
            binding: PcuBindingRef::new(0, 1),
            index,
        }),
        PcuDispatchOp::Data(PcuDispatchDataOp::CheckedFloatBinary {
            result: PcuDispatchValueId(3),
            lhs: PcuDispatchValueId(1),
            rhs: PcuDispatchValueId(2),
            value_type: PcuValueType::Scalar(scalar),
            op: PcuDispatchFloatBinaryOp::Add,
            range_policy: PcuRangePolicy::Reject,
            underflow_policy: PcuFloatUnderflowPolicy::IeeeAfterRounding,
        }),
        PcuDispatchOp::Data(PcuDispatchDataOp::CheckedFloatBinary {
            result: PcuDispatchValueId(4),
            lhs: PcuDispatchValueId(3),
            rhs: PcuDispatchValueId(3),
            value_type: PcuValueType::Scalar(scalar),
            op: PcuDispatchFloatBinaryOp::Mul,
            range_policy: PcuRangePolicy::Reject,
            underflow_policy: PcuFloatUnderflowPolicy::AllowGradualUnderflow,
        }),
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
            binding: PcuBindingRef::new(0, 0),
            index,
            value: PcuDispatchValueId(4),
        }),
    ]
}

fn fixture<'a>(
    scalar: PcuScalarType,
    n: u32,
    bindings: &'a [PcuBinding<'a>],
    ops: &'a [PcuDispatchOp<'a>],
) -> PcuDispatchKernelIr<'a> {
    PcuDispatchKernelIr {
        numerical_requirements: PcuDispatchKernelIr::DEFAULT_REQUIREMENTS,
        id: PcuKernelId(0x434f_4d50),
        entry: PcuDispatchEntryPoint {
            name: "composed_resources",
            logical_shape: [n, 1, 1],
        },
        bindings,
        ports: &[],
        parameters: &[],
        ops,
        type_caps: PcuValueTypeCaps::for_scalar(scalar),
        feature_caps: PcuDispatchFeatureCaps::READ_ONLY_RESOURCES
            .union(PcuDispatchFeatureCaps::MUTABLE_RESOURCES),
    }
}
fn bindings(scalar: PcuScalarType) -> [PcuBinding<'static>; 3] {
    let binding = |slot, access| {
        PcuBinding::value(
            None,
            0,
            slot,
            PcuBindingStorageClass::Storage,
            access,
            PcuValueType::Scalar(scalar),
        )
    };
    [
        binding(2, PcuBindingAccess::ReadWrite),
        binding(1, PcuBindingAccess::ReadOnly),
        binding(0, PcuBindingAccess::ReadWrite),
    ]
}

#[test]
fn composed_same_resource_broadcast_write_requires_single_lane_or_snapshot() {
    for scalar in [
        PcuScalarType::F16,
        PcuScalarType::BF16,
        PcuScalarType::F8E4M3FN,
        PcuScalarType::F8E5M2,
        PcuScalarType::F32,
        PcuScalarType::F64,
    ] {
        let bindings = bindings(scalar);
        for grid in [false, true] {
            let index = if grid {
                PcuDispatchIndex::GridStrideId
            } else {
                PcuDispatchIndex::InvocationId
            };
            let mut body = body(scalar, index, true);
            for n in [1, 4] {
                let ops = if grid {
                    vec![
                        PcuDispatchOp::GridStrideLoop {
                            extent: n,
                            body: &body,
                        },
                        PcuDispatchOp::Control(PcuDispatchControlOp::Return),
                    ]
                } else {
                    body.push(PcuDispatchOp::Control(PcuDispatchControlOp::Return));
                    body.clone()
                };
                let kernel = fixture(scalar, if grid { 1 } else { n }, &bindings, &ops);
                assert_eq!(
                    lower_dispatch_to_hip_source(&kernel).is_ok(),
                    n == 1,
                    "{scalar:?} grid={grid} extent={n}"
                );
                if !grid {
                    body.pop();
                }
            }
        }
    }
}

#[test]
fn composed_merged_indexed_span_preserves_zero_read_hazard() {
    for scalar in [
        PcuScalarType::F16,
        PcuScalarType::BF16,
        PcuScalarType::F8E4M3FN,
        PcuScalarType::F8E5M2,
        PcuScalarType::F32,
        PcuScalarType::F64,
    ] {
        let bindings = bindings(scalar);
        for grid in [false, true] {
            let index = if grid {
                PcuDispatchIndex::GridStrideId
            } else {
                PcuDispatchIndex::InvocationId
            };
            let mut body = body(scalar, index, true);
            if let PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad { binding, .. }) =
                &mut body[1]
            {
                *binding = PcuBindingRef::new(0, 0);
            }
            for n in [1, 4] {
                let ops = if grid {
                    vec![
                        PcuDispatchOp::GridStrideLoop {
                            extent: n,
                            body: &body,
                        },
                        PcuDispatchOp::Control(PcuDispatchControlOp::Return),
                    ]
                } else {
                    let mut ops = body.clone();
                    ops.push(PcuDispatchOp::Control(PcuDispatchControlOp::Return));
                    ops
                };
                let kernel = fixture(scalar, if grid { 1 } else { n }, &bindings, &ops);
                let schema = fusion_pcu::assess_checked_float_map_resources::<4>(
                    &kernel,
                    PcuValueType::Scalar(scalar),
                    PcuValueTypeCaps::for_scalar(scalar),
                )
                .unwrap();
                let shared = schema.resource(PcuBindingRef::new(0, 0)).unwrap();
                assert_eq!(shared.minimum_read_elements, n);
                assert_eq!(shared.minimum_write_elements, n);
                assert!(shared.reads_element_zero);
                assert_eq!(shared.has_cross_index_read_write(), n > 1);
                assert_eq!(super::composed::project(&kernel).is_some(), n == 1);
                assert_eq!(
                    lower_dispatch_to_hip_source(&kernel).is_ok(),
                    n == 1,
                    "{scalar:?} grid={grid} extent={n}"
                );
            }
        }
    }
}

#[test]
fn composed_projection_keeps_actual_rw_reads_and_omits_only_unaccessed_resources() {
    for scalar in [
        PcuScalarType::F16,
        PcuScalarType::BF16,
        PcuScalarType::F8E4M3FN,
        PcuScalarType::F8E5M2,
        PcuScalarType::F32,
        PcuScalarType::F64,
    ] {
        let bindings = bindings(scalar);
        let mut body = body(scalar, PcuDispatchIndex::InvocationId, false);
        body.push(PcuDispatchOp::Control(PcuDispatchControlOp::Return));
        let kernel = fixture(scalar, 4, &bindings, &body);
        let projection = map_binding_projection(&kernel).unwrap();
        assert_eq!(
            projection.input_bindings(),
            &[PcuBindingRef::new(0, 0), PcuBindingRef::new(0, 1)]
        );
        assert!(projection.contains_output(PcuBindingRef::new(0, 0)));
        assert!(!projection.contains_output(PcuBindingRef::new(0, 2)));
        let source = lower_dispatch_to_hip_source(&kernel).unwrap();
        let args = source
            .split("fusion_kernel(")
            .nth(1)
            .unwrap()
            .split(')')
            .next()
            .unwrap();
        assert!(!args.contains("binding_0_2"));
        assert!(args.find("binding_0_1").unwrap() < args.find("binding_0_0").unwrap());
        assert!(source.contains("binding_0_0"));
        assert!(crate::owned_dispatch::checked_scalar_fault_law(&kernel).is_some());
        let mut portable = kernel;
        portable
            .numerical_requirements
            .numerical_options
            .reproducibility = fusion_pcu::PcuReproducibility::PortableV1;
        assert!(lower_dispatch_to_hip_source(&portable).is_err());
    }
}

#[test]
fn composed_distinct_resource_broadcast_remains_lawful() {
    for scalar in [
        PcuScalarType::F16,
        PcuScalarType::BF16,
        PcuScalarType::F8E4M3FN,
        PcuScalarType::F8E5M2,
        PcuScalarType::F32,
        PcuScalarType::F64,
    ] {
        let mut bindings = bindings(scalar);
        bindings[2].access = PcuBindingAccess::ReadOnly;
        let mut body = body(scalar, PcuDispatchIndex::InvocationId, true);
        if let PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore { binding, .. }) = &mut body[4] {
            *binding = PcuBindingRef::new(0, 2);
        }
        body.push(PcuDispatchOp::Control(PcuDispatchControlOp::Return));
        let kernel = fixture(scalar, 4, &bindings, &body);
        let projection = super::composed::project(&kernel).unwrap();
        assert_eq!(
            projection.input_bindings(),
            &[PcuBindingRef::new(0, 0), PcuBindingRef::new(0, 1)]
        );
        assert!(projection.contains_output(PcuBindingRef::new(0, 2)));
        lower_dispatch_to_hip_source(&kernel).unwrap();
        let resource = fusion_pcu::assess_checked_float_map_resources::<4>(
            &kernel,
            PcuValueType::Scalar(scalar),
            PcuValueTypeCaps::for_scalar(scalar),
        )
        .unwrap();
        assert_eq!(
            resource
                .resource(PcuBindingRef::new(0, 0))
                .unwrap()
                .minimum_read_elements,
            1
        );
        assert_eq!(
            resource
                .resource(PcuBindingRef::new(0, 2))
                .unwrap()
                .minimum_write_elements,
            4
        );
        let law = crate::owned_dispatch::checked_scalar_fault_law(&kernel).unwrap();
        // IEEE Add cannot tiny-inexact fault; this multiplication permits gradual underflow.
        assert!(!law.allows(
            fusion_pcu::PcuExecutionFaultKind::ArithmeticUnderflow,
            false
        ));
    }
}

#[test]
fn composed_provider_capacity_does_not_limit_existing_broader_float_maps() {
    for scalar in [
        PcuScalarType::F16,
        PcuScalarType::BF16,
        PcuScalarType::F8E4M3FN,
        PcuScalarType::F8E5M2,
        PcuScalarType::F32,
        PcuScalarType::F64,
    ] {
        let mut bindings = bindings(scalar).to_vec();
        for slot in [3, 4] {
            bindings.push(PcuBinding::value(
                None,
                0,
                slot,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::ReadOnly,
                PcuValueType::Scalar(scalar),
            ));
        }
        bindings[0].access = PcuBindingAccess::ReadOnly;
        let mut body = body(scalar, PcuDispatchIndex::InvocationId, false);
        let store = body.pop().unwrap();
        for (slot, result) in [(2, 5), (3, 6), (4, 7)] {
            body.push(PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result: PcuDispatchValueId(result),
                binding: PcuBindingRef::new(0, slot),
                index: PcuDispatchIndex::InvocationId,
            }));
        }
        for (lhs, rhs, result) in [(4, 5, 8), (8, 6, 9), (9, 7, 10)] {
            body.push(PcuDispatchOp::Data(PcuDispatchDataOp::CheckedFloatBinary {
                result: PcuDispatchValueId(result),
                lhs: PcuDispatchValueId(lhs),
                rhs: PcuDispatchValueId(rhs),
                value_type: PcuValueType::Scalar(scalar),
                op: PcuDispatchFloatBinaryOp::Add,
                range_policy: PcuRangePolicy::Reject,
                underflow_policy: PcuFloatUnderflowPolicy::IeeeAfterRounding,
            }));
        }
        let PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore { binding, index, .. }) = store
        else {
            unreachable!()
        };
        body.push(PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
            binding,
            index,
            value: PcuDispatchValueId(10),
        }));
        body.push(PcuDispatchOp::Control(PcuDispatchControlOp::Return));
        let kernel = fixture(scalar, 4, &bindings, &body);
        assert!(super::composed::project(&kernel).is_none());
        assert_eq!(
            lower_dispatch_to_hip_source(&kernel).is_ok(),
            matches!(scalar, PcuScalarType::F32 | PcuScalarType::F64)
        );
    }
}

#[test]
fn overwritten_checked_division_retains_composed_admission_and_effect() {
    for scalar in [
        PcuScalarType::F16,
        PcuScalarType::BF16,
        PcuScalarType::F8E4M3FN,
        PcuScalarType::F8E5M2,
        PcuScalarType::F32,
        PcuScalarType::F64,
    ] {
        let bindings = bindings(scalar);
        for grid in [false, true] {
            let index = if grid {
                PcuDispatchIndex::GridStrideId
            } else {
                PcuDispatchIndex::InvocationId
            };
            let mut body = body(scalar, index, false);
            body.truncate(3);
            if let PcuDispatchOp::Data(PcuDispatchDataOp::CheckedFloatBinary { op, .. }) =
                &mut body[2]
            {
                *op = PcuDispatchFloatBinaryOp::Div;
            }
            body.extend([
                PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                    result: PcuDispatchValueId(4),
                    binding: PcuBindingRef::new(0, 0),
                    index,
                }),
                PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                    binding: PcuBindingRef::new(0, 0),
                    index,
                    value: PcuDispatchValueId(4),
                }),
            ]);
            let mut ops = if grid {
                vec![PcuDispatchOp::GridStrideLoop {
                    extent: 7,
                    body: &body,
                }]
            } else {
                body.clone()
            };
            ops.push(PcuDispatchOp::Control(PcuDispatchControlOp::Return));
            let kernel = fixture(scalar, if grid { 3 } else { 7 }, &bindings, &ops);
            assert!(checked_float_binary_operand_schema(&kernel).is_none());
            assert!(matches!(
                map_binding_projection(&kernel),
                Some(MapBindingProjection::Composed(_))
            ));
            assert!(
                lower_dispatch_to_hip_source(&kernel).is_ok(),
                "{scalar:?} grid={grid}"
            );
            // The retained division has a dead SSA result, but remains an observable
            // checked effect. Primitive matching must fall through to composed
            // admission instead of inferring a primitive map from the operation count.
            assert!(matches!(
                body[2],
                PcuDispatchOp::Data(PcuDispatchDataOp::CheckedFloatBinary {
                    result: PcuDispatchValueId(3),
                    op: PcuDispatchFloatBinaryOp::Div,
                    ..
                })
            ));
        }
    }
}

fn integer_body(
    scalar: PcuScalarType,
    index: PcuDispatchIndex,
    broadcast: bool,
) -> Vec<PcuDispatchOp<'static>> {
    body(scalar, index, broadcast)
        .into_iter()
        .map(|operation| match operation {
            PcuDispatchOp::Data(PcuDispatchDataOp::CheckedFloatBinary {
                value_type,
                result,
                lhs,
                rhs,
                op,
                range_policy,
                ..
            }) => PcuDispatchOp::Data(PcuDispatchDataOp::CheckedIntegerBinary {
                value_type,
                result,
                lhs,
                rhs,
                range_policy,
                op: match op {
                    PcuDispatchFloatBinaryOp::Add => fusion_pcu::PcuDispatchIntegerBinaryOp::Add,
                    PcuDispatchFloatBinaryOp::Mul => fusion_pcu::PcuDispatchIntegerBinaryOp::Mul,
                    _ => unreachable!(),
                },
            }),
            other => other,
        })
        .collect()
}
#[test]
fn fourteen_structural_integer_maps_and_ten_provider_composed_profiles() {
    for scalar in [
        PcuScalarType::U8,
        PcuScalarType::I8,
        PcuScalarType::U16,
        PcuScalarType::I16,
        PcuScalarType::U32,
        PcuScalarType::I32,
        PcuScalarType::U64,
        PcuScalarType::I64,
        PcuScalarType::U128,
        PcuScalarType::I128,
        PcuScalarType::U256,
        PcuScalarType::I256,
        PcuScalarType::U512,
        PcuScalarType::I512,
    ] {
        let bindings = bindings(scalar);
        for grid in [false, true] {
            let index = if grid {
                PcuDispatchIndex::GridStrideId
            } else {
                PcuDispatchIndex::InvocationId
            };
            let mut body = integer_body(scalar, index, false);
            if !grid {
                body.push(PcuDispatchOp::Control(PcuDispatchControlOp::Return));
            }
            let outer = [
                PcuDispatchOp::GridStrideLoop {
                    extent: 19,
                    body: &body,
                },
                PcuDispatchOp::Control(PcuDispatchControlOp::Return),
            ];
            let kernel = fixture(
                scalar,
                if grid { 3 } else { 19 },
                &bindings,
                if grid { &outer } else { &body },
            );
            assert!(
                fusion_pcu::assess_checked_integer_map_resources::<4>(
                    &kernel,
                    PcuValueType::Scalar(scalar),
                    PcuValueTypeCaps::for_scalar(scalar)
                )
                .is_ok()
            );
            if matches!(
                scalar,
                PcuScalarType::U256
                    | PcuScalarType::I256
                    | PcuScalarType::U512
                    | PcuScalarType::I512
            ) {
                assert!(super::composed::project(&kernel).is_none());
                assert!(super::lower_dispatch_to_hip_source(&kernel).is_err());
                continue;
            }
            assert!(
                super::composed::project(&kernel).is_some(),
                "genuine composed integer structural profile {scalar:?} grid={grid}"
            );
            assert!(super::lower_dispatch_to_hip_source(&kernel).is_ok());
            let projection = super::map_binding_projection(&kernel).unwrap();
            assert_eq!(
                projection.input_bindings(),
                [PcuBindingRef::new(0, 0), PcuBindingRef::new(0, 1)]
            );
            assert!(projection.contains_output(PcuBindingRef::new(0, 0)));
            assert!(
                !projection.contains_output(PcuBindingRef::new(0, 2)),
                "untouched mutable declaration is not a physical resource"
            );
        }
    }
}

#[test]
fn integer_primitive_projection_stays_distinct_from_composition() {
    let scalar = PcuScalarType::U32;
    let bindings = [
        PcuBinding::value(
            None,
            0,
            0,
            PcuBindingStorageClass::Storage,
            PcuBindingAccess::ReadOnly,
            PcuValueType::Scalar(scalar),
        ),
        PcuBinding::value(
            None,
            0,
            1,
            PcuBindingStorageClass::Storage,
            PcuBindingAccess::ReadOnly,
            PcuValueType::Scalar(scalar),
        ),
        PcuBinding::value(
            None,
            0,
            2,
            PcuBindingStorageClass::Storage,
            PcuBindingAccess::WriteOnly,
            PcuValueType::Scalar(scalar),
        ),
    ];
    let mut ops = integer_body(scalar, PcuDispatchIndex::InvocationId, false);
    ops.remove(3);
    ops[3] = PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
        binding: PcuBindingRef::new(0, 2),
        index: PcuDispatchIndex::InvocationId,
        value: PcuDispatchValueId(3),
    });
    ops.push(PcuDispatchOp::Control(PcuDispatchControlOp::Return));
    let kernel = fixture(scalar, 19, &bindings, &ops);
    assert!(matches!(
        super::map_binding_projection(&kernel),
        Some(MapBindingProjection::Integer(_))
    ));
}

#[test]
fn integer_composed_hazard_header_and_ssa_rejections_remain_cold() {
    let scalar = PcuScalarType::U64;
    let bindings = bindings(scalar);
    let mut ops = integer_body(scalar, PcuDispatchIndex::InvocationId, true);
    ops.push(PcuDispatchOp::Control(PcuDispatchControlOp::Return));
    let single = fixture(scalar, 1, &bindings, &ops);
    assert!(super::composed::project(&single).is_some());
    let multi = fixture(scalar, 19, &bindings, &ops);
    assert!(super::composed::project(&multi).is_none());
    assert!(super::lower_dispatch_to_hip_source(&multi).is_err());
    ops[0] = PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
        result: PcuDispatchValueId(1),
        binding: PcuBindingRef::new(0, 0),
        index: PcuDispatchIndex::InvocationId,
    });
    let mut kernel = fixture(scalar, 19, &bindings, &ops);
    kernel
        .numerical_requirements
        .numerical_options
        .reproducibility = fusion_pcu::PcuReproducibility::PortableV1;
    assert!(
        super::lower_dispatch_to_hip_source(&kernel).is_err(),
        "no composed Portable descriptor or provider grant"
    );
    kernel
        .numerical_requirements
        .numerical_options
        .reproducibility = fusion_pcu::PcuReproducibility::Unspecified;
    kernel.numerical_requirements.range_policy = PcuRangePolicy::Clamp;
    assert!(
        super::composed::project(&kernel).is_none(),
        "shared packed disposition cannot erase local Reject"
    );
    ops[3] = PcuDispatchOp::Data(PcuDispatchDataOp::CheckedIntegerBinary {
        value_type: PcuValueType::Scalar(scalar),
        op: fusion_pcu::PcuDispatchIntegerBinaryOp::Mul,
        range_policy: PcuRangePolicy::Reject,
        result: PcuDispatchValueId(4),
        lhs: PcuDispatchValueId(99),
        rhs: PcuDispatchValueId(3),
    });
    assert!(super::composed::project(&fixture(scalar, 19, &bindings, &ops)).is_none());
}
