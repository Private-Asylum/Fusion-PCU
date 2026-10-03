#[rustfmt::skip]
use fusion_pcu::{
    PcuBinding,
    PcuBindingAccess,
    PcuBindingRef,
    PcuBindingStorageClass,
    PcuCostBoundary,
    PcuDeviceIdentity,
    PcuDispatchControlOp,
    PcuDispatchDataOp,
    PcuDispatchEntryPoint,
    PcuDispatchFeatureCaps,
    PcuDispatchFloatUnaryOp,
    PcuDispatchIndex,
    PcuDispatchKernelIr,
    PcuDispatchOp,
    PcuDispatchValueId,
    PcuExecutorId,
    PcuFloatUnderflowPolicy,
    PcuImplementationCost,
    PcuImplementationMechanism,
    PcuImplementationOffers,
    PcuImplementationRequest,
    PcuImplementationRequirements,
    PcuKernelId,
    PcuNumericalMode,
    PcuObjectKind,
    PcuObjectRef,
    PcuProviderId,
    PcuRangePolicy,
    PcuReproducibility,
    PcuValueType,
    PcuValueTypeCaps,
};
#[rustfmt::skip]
use crate::{
    PcuCpuCheckedNeg,
    PcuCpuNegOfferError,
    PcuCpuNegOffers,
};
#[cfg(feature = "std")]
#[rustfmt::skip]
use crate::{
    PcuCpuImplementation,
    PcuCpuProcessor,
};

fn identity(generation: u64) -> PcuDeviceIdentity {
    PcuDeviceIdentity::from_device_ref(PcuObjectRef {
        provider: PcuProviderId(3),
        generation,
        kind: PcuObjectKind::Device,
        id: 0,
    })
    .unwrap()
}

fn provider() -> PcuCpuNegOffers {
    PcuCpuNegOffers::new(PcuCpuCheckedNeg::scalar(), identity(1), PcuExecutorId(0))
}

fn with_request(
    grid: bool,
    underflow: PcuFloatUnderflowPolicy,
    inspect: impl FnOnce(&mut PcuImplementationRequest<'_, PcuDispatchKernelIr<'_>>),
) {
    let bindings = [
        PcuBinding::scalar::<f32>(
            None,
            0,
            0,
            PcuBindingStorageClass::Storage,
            PcuBindingAccess::ReadOnly,
        ),
        PcuBinding::scalar::<f32>(
            None,
            0,
            1,
            PcuBindingStorageClass::Storage,
            PcuBindingAccess::WriteOnly,
        ),
    ];
    let index = if grid {
        PcuDispatchIndex::GridStrideId
    } else {
        PcuDispatchIndex::InvocationId
    };
    let body = [
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
            result: PcuDispatchValueId(1),
            binding: PcuBindingRef::new(0, 0),
            index,
        }),
        PcuDispatchOp::Data(PcuDispatchDataOp::CheckedFloatUnary {
            value_type: PcuValueType::f32(),
            op: PcuDispatchFloatUnaryOp::Neg,
            underflow_policy: underflow,
            range_policy: PcuRangePolicy::Reject,
            result: PcuDispatchValueId(2),
            value: PcuDispatchValueId(1),
        }),
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
            binding: PcuBindingRef::new(0, 1),
            index,
            value: PcuDispatchValueId(2),
        }),
    ];
    let direct = [
        body[0],
        body[1],
        body[2],
        PcuDispatchOp::Control(PcuDispatchControlOp::Return),
    ];
    let grid_ops = [
        PcuDispatchOp::GridStrideLoop {
            extent: 17,
            body: &body,
        },
        PcuDispatchOp::Control(PcuDispatchControlOp::Return),
    ];
    let kernel = PcuDispatchKernelIr {
        numerical_requirements: PcuImplementationRequirements {
            float_underflow: underflow,
            ..PcuImplementationRequirements::default()
        },
        id: PcuKernelId(0),
        entry: PcuDispatchEntryPoint {
            name: "neg_offer",
            logical_shape: [if grid { 2 } else { 17 }, 1, 1],
        },
        bindings: &bindings,
        ports: &[],
        parameters: &[],
        ops: if grid { &grid_ops } else { &direct },
        type_caps: PcuValueTypeCaps::FLOAT32,
        feature_caps: PcuDispatchFeatureCaps::default(),
    };
    let mut request = PcuImplementationRequest {
        device: identity(1),
        executor: PcuExecutorId(0),
        requirements: PcuImplementationRequirements {
            float_underflow: underflow,
            ..PcuImplementationRequirements::default()
        },
        boundary: PcuCostBoundary::Host,
        operation: &kernel,
    };
    inspect(&mut request);
}

#[test]
fn cold_offer_preserves_exact_requirements_and_has_no_invented_cost() {
    for grid in [false, true] {
        for underflow in [
            PcuFloatUnderflowPolicy::IeeeAfterRounding,
            PcuFloatUnderflowPolicy::AllowGradualUnderflow,
            PcuFloatUnderflowPolicy::RejectSubnormalResult,
        ] {
            with_request(grid, underflow, |request| {
                for mode in [PcuNumericalMode::Boundary, PcuNumericalMode::Strict] {
                    request.requirements.numerical_mode = mode;
                    let kernel = PcuDispatchKernelIr {
                        numerical_requirements: request.requirements,
                        ..*request.operation
                    };
                    let request = &mut PcuImplementationRequest {
                        operation: &kernel,
                        ..*request
                    };
                    let mut output = [None; 2];
                    assert_eq!(
                        provider().implementation_offers(request, &mut output),
                        Ok(1)
                    );
                    let offer = output[0].unwrap();
                    assert_eq!(offer.validate_request(request), Ok(()));
                    assert_eq!(offer.kind, PcuImplementationMechanism::NativeKernel);
                    assert_eq!(offer.implementation.local_id, 0);
                    assert_eq!(offer.implementation.revision, 1);
                    assert_eq!(offer.workspace_bytes, Some(0));
                    assert_eq!(
                        offer.cost,
                        PcuImplementationCost::unknown(PcuCostBoundary::Host)
                    );
                    assert_eq!(output[1], None);
                    assert_eq!(provider().implementation_offers(request, &mut []), Ok(1));
                }
            });
        }
    }
}

#[test]
fn cold_offer_rejects_stale_targets_and_mismatched_operation_policy() {
    with_request(
        false,
        PcuFloatUnderflowPolicy::IeeeAfterRounding,
        |request| {
            let mut output = [None];
            request.device = identity(2);
            assert_eq!(
                provider().implementation_offers(request, &mut output),
                Err(PcuCpuNegOfferError::DeviceMismatch)
            );
            request.device = identity(1);
            request.executor = PcuExecutorId(1);
            assert_eq!(
                provider().implementation_offers(request, &mut output),
                Err(PcuCpuNegOfferError::ExecutorMismatch)
            );
            request.executor = PcuExecutorId(0);
            request.requirements.float_underflow = PcuFloatUnderflowPolicy::RejectSubnormalResult;
            let kernel = PcuDispatchKernelIr {
                numerical_requirements: request.requirements,
                ..*request.operation
            };
            let request = &mut PcuImplementationRequest {
                operation: &kernel,
                ..*request
            };
            assert_eq!(
                provider().implementation_offers(request, &mut output),
                Err(PcuCpuNegOfferError::UnderflowMismatch)
            );
            assert_eq!(output, [None]);
        },
    );
}

#[test]
fn cold_offer_has_no_resident_clamp_or_portable_admission() {
    with_request(
        false,
        PcuFloatUnderflowPolicy::IeeeAfterRounding,
        |request| {
            let mut output = [None];
            request.boundary = PcuCostBoundary::Resident;
            assert_eq!(
                provider().implementation_offers(request, &mut output),
                Ok(0)
            );
            request.boundary = PcuCostBoundary::Host;
            request.requirements.range_policy = PcuRangePolicy::Clamp;
            assert_eq!(
                provider().implementation_offers(request, &mut output),
                Ok(0)
            );
            request.requirements.range_policy = PcuRangePolicy::Reject;
            request.requirements.numerical_options.reproducibility = PcuReproducibility::PortableV1;
            for mode in [PcuNumericalMode::Boundary, PcuNumericalMode::Strict] {
                request.requirements.numerical_mode = mode;
                assert_eq!(
                    provider().implementation_offers(request, &mut output),
                    Ok(0)
                );
            }
            assert_eq!(output, [None]);
        },
    );
}

#[cfg(feature = "std")]
#[test]
fn detected_instruction_offers_have_distinct_implementation_identities() {
    let processor = PcuCpuProcessor::detect();
    for (implementation, local_id) in [
        (PcuCpuImplementation::Scalar, 0),
        (PcuCpuImplementation::Sse2, 1),
        (PcuCpuImplementation::Avx2, 2),
        (PcuCpuImplementation::Neon, 3),
    ] {
        let Ok(backend) = PcuCpuCheckedNeg::new(processor, implementation) else {
            continue;
        };
        let offers = PcuCpuNegOffers::new(backend, identity(1), PcuExecutorId(0));
        with_request(
            false,
            PcuFloatUnderflowPolicy::IeeeAfterRounding,
            |request| {
                let mut output = [None];
                assert_eq!(offers.implementation_offers(request, &mut output), Ok(1));
                assert_eq!(output[0].unwrap().implementation.local_id, local_id);
            },
        );
    }
}

#[test]
fn another_checked_float_operation_does_not_acquire_a_neg_offer() {
    with_request(
        false,
        PcuFloatUnderflowPolicy::IeeeAfterRounding,
        |request| {
            let ops = [
                request.operation.ops[0],
                PcuDispatchOp::Data(PcuDispatchDataOp::CheckedFloatUnary {
                    value_type: PcuValueType::f32(),
                    op: PcuDispatchFloatUnaryOp::Relu,
                    underflow_policy: PcuFloatUnderflowPolicy::IeeeAfterRounding,
                    range_policy: PcuRangePolicy::Reject,
                    result: PcuDispatchValueId(2),
                    value: PcuDispatchValueId(1),
                }),
                request.operation.ops[2],
                request.operation.ops[3],
            ];
            let kernel = PcuDispatchKernelIr {
                ops: &ops,
                ..*request.operation
            };
            let unsupported = PcuImplementationRequest {
                operation: &kernel,
                device: request.device,
                executor: request.executor,
                requirements: request.requirements,
                boundary: request.boundary,
            };
            let mut output = [None];
            assert_eq!(
                provider().implementation_offers(&unsupported, &mut output),
                Ok(0)
            );
            assert_eq!(output, [None]);
        },
    );
}

#[test]
fn malformed_neg_retains_value_flow_shape_and_binding_errors() {
    with_request(
        false,
        PcuFloatUnderflowPolicy::IeeeAfterRounding,
        |request| {
            let mut ops = request.operation.ops.to_vec();
            if let PcuDispatchOp::Data(PcuDispatchDataOp::CheckedFloatUnary { value, .. }) =
                &mut ops[1]
            {
                *value = PcuDispatchValueId(99);
            }
            let kernel = PcuDispatchKernelIr {
                ops: &ops,
                ..*request.operation
            };
            let malformed = PcuImplementationRequest {
                operation: &kernel,
                ..*request
            };
            assert!(matches!(
                provider().implementation_offers(&malformed, &mut []),
                Err(PcuCpuNegOfferError::Provider(
                    crate::PcuCpuPreparedNegError::InvalidValueFlow(_)
                ))
            ));
            let mut kernel = *request.operation;
            kernel.entry.logical_shape = [0, 1, 1];
            let malformed = PcuImplementationRequest {
                operation: &kernel,
                ..*request
            };
            assert!(matches!(
                provider().implementation_offers(&malformed, &mut []),
                Err(PcuCpuNegOfferError::Provider(
                    crate::PcuCpuPreparedNegError::InvalidKernel(
                        fusion_pcu::CheckedFloatMapValidationError::InvalidLogicalShape
                    )
                ))
            ));
            let bindings = [request.operation.bindings[0], request.operation.bindings[0]];
            let kernel = PcuDispatchKernelIr {
                bindings: &bindings,
                ..*request.operation
            };
            let malformed = PcuImplementationRequest {
                operation: &kernel,
                ..*request
            };
            assert!(matches!(
                provider().implementation_offers(&malformed, &mut []),
                Err(PcuCpuNegOfferError::Provider(_))
            ));
        },
    );
}
