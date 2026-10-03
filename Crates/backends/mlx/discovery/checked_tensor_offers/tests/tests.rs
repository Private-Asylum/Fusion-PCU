//! Pure detached offer law; synthetic provenance here never substitutes for runtime discovery.
#[rustfmt::skip]
use super::{
    integer_offer,
    MlxTensorIntegerRequest,
    PcuCostBoundary,
    PcuImplementationRequest,
    PcuImplementationMechanism,
    PcuImplementationCost,
    PcuObjectRef,
    PcuObjectKind,
    EXECUTOR,
};
#[rustfmt::skip]
use fusion_pcu::{
    PcuProviderId,
    PcuDeviceIdentity,
    PcuScalarType,
    PcuImplementationRequirements,
    PcuNumericalMode,
    PcuFloatUnderflowPolicy,
    PcuRangePolicy,
    PcuReproducibility,
};
#[rustfmt::skip]
use fusion_pcu::dialect::tensor::{
    Graph,
    TensorOwnedSelectedProgram,
    TensorArithmeticRewritePolicy,
    TensorArithmeticCapability,
    TensorPointwiseGroupingPolicy,
};
fn program(scalar: PcuScalarType, repeated: bool, operation: usize) -> TensorOwnedSelectedProgram {
    let mut graph = Graph::try_new().unwrap();
    let left = graph.input([5], scalar).unwrap();
    let right = if repeated {
        left
    } else {
        graph.input([5], scalar).unwrap()
    };
    let effect = match operation {
        0 => graph.add(left, right),
        1 => graph.sub(right, left),
        _ => graph.mul(left, right),
    }
    .unwrap();
    graph
        .into_selected_program(
            &[effect],
            TensorArithmeticRewritePolicy::Disabled,
            TensorArithmeticCapability::Strict,
            TensorPointwiseGroupingPolicy::Disabled,
        )
        .unwrap()
}
#[test]
fn fourteen_width_integer_owner_offers_preserve_requirements_and_exact_boundary() {
    let device = PcuDeviceIdentity::from_device_ref(PcuObjectRef {
        provider: PcuProviderId(7),
        generation: 11,
        kind: PcuObjectKind::Device,
        id: 13,
    })
    .unwrap();
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
        for operation in 0..3 {
            for repeated in [false, true] {
                let program = program(scalar, repeated, operation);
                let operation = MlxTensorIntegerRequest { program: &program };
                for mode in [PcuNumericalMode::Boundary, PcuNumericalMode::Strict] {
                    for underflow in [
                        PcuFloatUnderflowPolicy::IeeeAfterRounding,
                        PcuFloatUnderflowPolicy::RejectSubnormalResult,
                        PcuFloatUnderflowPolicy::AllowGradualUnderflow,
                    ] {
                        let requirements = PcuImplementationRequirements {
                            numerical_mode: mode,
                            float_underflow: underflow,
                            ..PcuImplementationRequirements::default()
                        };
                        let plan = crate::MlxCheckedTensorIntegerPlan::assess_program(
                            &program,
                            requirements,
                        )
                        .unwrap();
                        for boundary in [
                            PcuCostBoundary::Host,
                            PcuCostBoundary::Resident,
                            PcuCostBoundary::HostInputsResidentOutput,
                            PcuCostBoundary::MixedInputsResidentOutput,
                        ] {
                            let offer = integer_offer(&plan, device, boundary);
                            let admitted = boundary != PcuCostBoundary::Host
                                && !(repeated
                                    && boundary == PcuCostBoundary::MixedInputsResidentOutput);
                            assert_eq!(offer.is_some(), admitted);
                            if let Some(offer) = offer {
                                let request = PcuImplementationRequest {
                                    device,
                                    executor: EXECUTOR,
                                    operation: &operation,
                                    requirements,
                                    boundary,
                                };
                                offer.validate_request(&request).unwrap();
                                assert_eq!(offer.implementation, plan.implementation_id(device));
                                assert_eq!(offer.implementation.revision, 0x0003_0020_0003_0b00);
                                assert_eq!(
                                    offer.kind,
                                    PcuImplementationMechanism::DelegatedRuntime
                                );
                                assert_eq!(offer.cost, PcuImplementationCost::unknown(boundary));
                                assert_eq!(offer.workspace_bytes, None);
                                let mismatch = PcuImplementationRequest {
                                    boundary: PcuCostBoundary::Host,
                                    ..request
                                };
                                assert!(offer.validate_request(&mismatch).is_err());
                            }
                        }
                        let clamp = PcuImplementationRequirements {
                            range_policy: PcuRangePolicy::Clamp,
                            ..requirements
                        };
                        assert!(
                            crate::MlxCheckedTensorIntegerPlan::assess_program(&program, clamp)
                                .is_err()
                        );
                        let mut portable = requirements;
                        portable.numerical_options.reproducibility = PcuReproducibility::PortableV1;
                        assert!(
                            crate::MlxCheckedTensorIntegerPlan::assess_program(&program, portable)
                                .is_err()
                        );
                    }
                }
            }
        }
    }
}
