//! Independent byte-transport projection, exact requests and snapshot refusal.
use super::*;
#[rustfmt::skip]
use fusion_pcu::{
    PcuBinding,
    PcuCompoundArithmeticPolicy,
    PcuDispatchEntryPoint,
    PcuFloatUnderflowPolicy,
    PcuImplementationRequirements,
    PcuKernelId,
    PcuNumericalMode,
    PcuPrecisionPolicy,
    PcuRangePolicy,
    PcuReproducibility,
};

fn transport_declarations(scalar: PcuScalarType) -> [PcuBinding<'static>; 4] {
    [
        (3, PcuBindingAccess::ReadWrite),
        (0, PcuBindingAccess::WriteOnly),
        (1, PcuBindingAccess::ReadWrite),
        (2, PcuBindingAccess::ReadOnly),
    ]
    .map(|(slot, access)| {
        PcuBinding::value(
            None,
            0,
            slot,
            PcuBindingStorageClass::Storage,
            access,
            PcuValueType::Scalar(scalar),
        )
    })
}
fn transport_body(
    index: PcuDispatchIndex,
    broadcast: bool,
    reload_zero: bool,
) -> [PcuDispatchOp<'static>; 5] {
    [
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
            result: PcuDispatchValueId(1),
            binding: PcuBindingRef::new(0, 2),
            index: if broadcast {
                PcuDispatchIndex::BindingElementZero
            } else {
                index
            },
        }),
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
            binding: PcuBindingRef::new(0, 1),
            index,
            value: PcuDispatchValueId(1),
        }),
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
            result: PcuDispatchValueId(2),
            binding: PcuBindingRef::new(0, 1),
            index: if reload_zero {
                PcuDispatchIndex::BindingElementZero
            } else {
                index
            },
        }),
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
            binding: PcuBindingRef::new(0, 0),
            index,
            value: PcuDispatchValueId(2),
        }),
        PcuDispatchOp::Control(PcuDispatchControlOp::Return),
    ]
}
fn transport_kernel<'a>(
    scalar: PcuScalarType,
    bindings: &'a [PcuBinding<'a>],
    ops: &'a [PcuDispatchOp<'a>],
    n: u32,
) -> PcuDispatchKernelIr<'a> {
    PcuDispatchKernelIr {
        id: PcuKernelId(0x5452_4e53),
        entry: PcuDispatchEntryPoint {
            name: "ordered_transport",
            logical_shape: [n, 1, 1],
        },
        bindings,
        ops,
        ports: &[],
        parameters: &[],
        type_caps: PcuValueTypeCaps::for_scalar(scalar),
        feature_caps: PcuDispatchFeatureCaps::READ_ONLY_RESOURCES
            .union(PcuDispatchFeatureCaps::MUTABLE_RESOURCES),
        numerical_requirements: PcuImplementationRequirements::DEFAULT,
    }
}

fn transport_requests() -> Vec<PcuImplementationRequirements> {
    let mut requests = Vec::new();
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
                    for range in [PcuRangePolicy::Reject, PcuRangePolicy::Clamp] {
                        let mut request = PcuImplementationRequirements {
                            numerical_mode: mode,
                            float_underflow: uf,
                            range_policy: range,
                            ..PcuImplementationRequirements::DEFAULT
                        };
                        request.numerical_options.compound_arithmetic = compound;
                        request.numerical_options.precision = precision;
                        requests.push(request);
                    }
                }
            }
        }
    }
    requests
}
#[test]
fn byte_transport_keeps_all22_actual_resources_and_exact_requests() {
    for scalar in PcuScalarType::ALL {
        if matches!(
            scalar,
            PcuScalarType::Bool | PcuScalarType::I4 | PcuScalarType::U4
        ) {
            continue;
        }
        let bindings = transport_declarations(scalar);
        for grid in [false, true] {
            let index = if grid {
                PcuDispatchIndex::GridStrideId
            } else {
                PcuDispatchIndex::InvocationId
            };
            for broadcast in [false, true] {
                let body = transport_body(index, broadcast, false);
                let wrapped = [
                    PcuDispatchOp::GridStrideLoop {
                        extent: 65,
                        body: &body[..4],
                    },
                    PcuDispatchOp::Control(PcuDispatchControlOp::Return),
                ];
                let mut ir = transport_kernel(
                    scalar,
                    &bindings,
                    if grid { &wrapped } else { &body },
                    if grid { 17 } else { 65 },
                );
                for request in transport_requests() {
                    ir.numerical_requirements = request;
                    let description =
                        fusion_pcu::describe_scalar_transport_map::<4>(&ir, scalar).unwrap();
                    assert_eq!(description.requirements, request);
                    let projection = map_binding_projection(&ir).unwrap();
                    assert!(matches!(projection, MapBindingProjection::Transport(_)));
                    assert_eq!(
                        projection.input_bindings(),
                        &[PcuBindingRef::new(0, 2), PcuBindingRef::new(0, 1)]
                    );
                    assert!(projection.contains_output(PcuBindingRef::new(0, 0)));
                    assert!(projection.contains_output(PcuBindingRef::new(0, 1)));
                    assert!(!projection.contains_output(PcuBindingRef::new(0, 3)));
                    assert!(
                        lower_dispatch_to_cuda_source(&ir).is_ok(),
                        "{scalar:?} grid={grid} broadcast={broadcast}"
                    );
                }
                ir.numerical_requirements.numerical_options.reproducibility =
                    PcuReproducibility::PortableV1;
                assert!(
                    lower_dispatch_to_cuda_source(&ir).is_err(),
                    "transport does not invent a Portable offer"
                );
            }
        }
    }
}

#[test]
fn byte_transport_cross_lane_own_read_requires_single_lane_or_snapshot() {
    for scalar in [PcuScalarType::F16, PcuScalarType::F32, PcuScalarType::I512] {
        let bindings = transport_declarations(scalar);
        for grid in [false, true] {
            let index = if grid {
                PcuDispatchIndex::GridStrideId
            } else {
                PcuDispatchIndex::InvocationId
            };
            let body = transport_body(index, false, true);
            for n in [1, 7] {
                let wrapped = [
                    PcuDispatchOp::GridStrideLoop {
                        extent: n,
                        body: &body[..4],
                    },
                    PcuDispatchOp::Control(PcuDispatchControlOp::Return),
                ];
                let ir = transport_kernel(
                    scalar,
                    &bindings,
                    if grid { &wrapped } else { &body },
                    if grid { 3 } else { n },
                );
                assert_eq!(transport::project(&ir).is_some(), n == 1);
                assert_eq!(
                    lower_dispatch_to_cuda_source(&ir).is_ok(),
                    n == 1,
                    "{scalar:?} grid={grid} N{n}"
                );
            }
        }
    }
}

#[test]
fn byte_transport_limits_actual_resources_and_sealed_carriers() {
    for scalar in [PcuScalarType::Bool, PcuScalarType::I4, PcuScalarType::U4] {
        let bindings = transport_declarations(scalar);
        let body = transport_body(PcuDispatchIndex::InvocationId, false, false);
        let kernel = transport_kernel(scalar, &bindings, &body, 7);
        assert!(super::transport::project(&kernel).is_none());
        assert!(lower_dispatch_to_cuda_source(&kernel).is_err());
    }
    let scalar = PcuScalarType::F128;
    let mut bindings = transport_declarations(scalar).to_vec();
    for slot in [4, 5] {
        bindings.push(PcuBinding::value(
            None,
            0,
            slot,
            PcuBindingStorageClass::Storage,
            PcuBindingAccess::ReadWrite,
            PcuValueType::Scalar(scalar),
        ));
    }
    let mut body = transport_body(PcuDispatchIndex::InvocationId, false, false).to_vec();
    let kernel = transport_kernel(scalar, &bindings, &body, 7);
    // Six declarations are lawful: only three resources are actually accessed.
    assert!(super::transport::project(&kernel).is_some());
    assert!(lower_dispatch_to_cuda_source(&kernel).is_ok());
    body.pop();
    for slot in [3, 4] {
        body.push(PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
            binding: PcuBindingRef::new(0, slot),
            index: PcuDispatchIndex::InvocationId,
            value: PcuDispatchValueId(2),
        }));
    }
    body.push(PcuDispatchOp::Control(PcuDispatchControlOp::Return));
    let kernel = transport_kernel(scalar, &bindings, &body, 7);
    // This provider profile permits four actual resources, not five.
    assert!(super::transport::project(&kernel).is_none());
    assert!(lower_dispatch_to_cuda_source(&kernel).is_err());
}
