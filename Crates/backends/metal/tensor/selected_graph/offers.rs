use super::*;
use super::tests::{source, requests, bytes};
use fusion_pcu::{
    PcuRuntimeDiscovery, PcuDeviceActivation, PcuImplementationOffers, PcuCostBoundary,
    PcuImplementationRequest, PcuImplementationCost, PcuExecutorId,
};
pub(super) fn device_identity(discovery: &crate::MetalDiscovery) -> fusion_pcu::PcuDeviceIdentity {
    let readiness = fusion_pcu::PcuProviderReadiness {
        status: fusion_pcu::PcuProviderStatus::Unavailable,
        reason: None,
    };
    let empty = fusion_pcu::PcuObjectRef {
        provider: fusion_pcu::PcuProviderId(0),
        generation: 0,
        kind: fusion_pcu::PcuObjectKind::Device,
        id: 0,
    };
    let mut providers = [fusion_pcu::PcuProviderDescriptor {
        id: empty.provider,
        generation: 0,
        backend: "",
        readiness,
    }; 1];
    discovery.providers(&mut providers).unwrap();
    let mut targets = [fusion_pcu::PcuTargetDescriptor {
        reference: empty,
        name: "",
        readiness,
    }; 1];
    discovery
        .targets(providers[0].id, providers[0].generation, &mut targets)
        .unwrap();
    let mut devices = [fusion_pcu::PcuDeviceDescriptor {
        reference: empty,
        target: empty,
        name: "",
        class: fusion_pcu::PcuDeviceClass::Gpu,
        vendor: None,
        architecture: None,
        generation: None,
        location: None,
    }; 1];
    discovery
        .devices(targets[0].reference, &mut devices)
        .unwrap();
    fusion_pcu::PcuDeviceIdentity::from_device_ref(devices[0].reference).unwrap()
}

fn single_input(
    scalar: PcuScalarType,
    request: PcuImplementationRequirements,
) -> Arc<TensorOwnedSelectedProgram> {
    use fusion_pcu::dialect::tensor::Graph;
    let mut graph = Graph::default();
    graph.set_numerical_options(request.numerical_options);
    let input = graph.input([4], scalar).unwrap();
    let first = graph.relu(input).unwrap();
    let second = graph.relu(first).unwrap();
    for effect in [first, second] {
        graph
            .set_value_float_underflow_policy(effect, request.float_underflow)
            .unwrap();
    }
    super::tests::freeze(graph, input)
}
#[test]
#[ignore = "Requires actual native graph inventory, authentic prepared receipt and exact offers."]
fn selected_graph_native_offers_full_tuples_actual_roles_and_receipts() {
    let discovery = crate::MetalDiscovery::discover().unwrap();
    let device = device_identity(&discovery);
    let session = discovery
        .open_device(fusion_pcu::PcuObjectRef {
            provider: device.provider(),
            generation: device.generation(),
            kind: fusion_pcu::PcuObjectKind::Device,
            id: device.device_id(),
        })
        .unwrap();
    for requirements in requests() {
        for scalar in GRAPH_SCALARS {
            for profile in 0..5 {
                let program = offer_program(profile, scalar, requirements);
                let plan =
                    MetalSelectedTensorGraphPlan::assess_program(program, requirements).unwrap();
                let prepared = session
                    .prepare_selected_tensor_graph(plan.clone(), PcuMemoryPoolId(186))
                    .unwrap();
                assert!(prepared.session.same_session(&session));
                assert_eq!(plan.implementation_id(device).local_id, 0x7100);
                assert_eq!(
                    plan.implementation_id(device).revision,
                    0x0000_0008_0000_0404
                );
                let operation = crate::MetalSelectedTensorGraphRequest { plan: &plan };
                for boundary in [
                    PcuCostBoundary::HostInputsResidentOutput,
                    PcuCostBoundary::Resident,
                    PcuCostBoundary::MixedInputsResidentOutput,
                    PcuCostBoundary::Host,
                ] {
                    let request = PcuImplementationRequest {
                        device,
                        executor: PcuExecutorId(0),
                        requirements,
                        boundary,
                        operation: &operation,
                    };
                    let mut offers = [None; 2];
                    let expected = usize::from(
                        boundary != PcuCostBoundary::Host
                            && (boundary != PcuCostBoundary::MixedInputsResidentOutput
                                || plan.input_values().len() >= 2),
                    );
                    assert_eq!(
                        discovery
                            .implementation_offers(&request, &mut offers)
                            .unwrap(),
                        expected
                    );
                    assert_eq!(
                        discovery.implementation_offers(&request, &mut []).unwrap(),
                        expected
                    );
                    assert_eq!(offers[1], None);
                    if expected == 1 {
                        let offer = offers[0].unwrap();
                        assert_eq!(offer.implementation, plan.implementation_id(device));
                        assert_eq!(offer.requirements, requirements);
                        assert_eq!(offer.workspace_bytes, None);
                        assert_eq!(offer.cost, PcuImplementationCost::unknown(boundary));
                    } else {
                        assert_eq!(offers, [None; 2]);
                    }
                    let mut wrong = request;
                    wrong.requirements.float_underflow = if requirements.float_underflow
                        == fusion_pcu::PcuFloatUnderflowPolicy::RejectSubnormalResult
                    {
                        fusion_pcu::PcuFloatUnderflowPolicy::IeeeAfterRounding
                    } else {
                        fusion_pcu::PcuFloatUnderflowPolicy::RejectSubnormalResult
                    };
                    assert!(
                        discovery
                            .implementation_offers(&wrong, &mut offers)
                            .is_err()
                    );
                    assert_eq!(offers, [None; 2]);
                    wrong.requirements = requirements;
                    wrong.executor = PcuExecutorId(99);
                    assert!(
                        discovery
                            .implementation_offers(&wrong, &mut offers)
                            .is_err()
                    );
                    assert_eq!(offers, [None; 2]);
                }
                if profile == 4 {
                    single_scalar_execution(&prepared, scalar);
                }
            }
        }
    }
}
fn single_execution<T: fusion_pcu::PcuCheckedFloat>(
    prepared: &MetalPreparedSelectedTensorGraph,
    data: &[T],
    invalid: T,
) {
    let input = prepared.plan.input_values()[0];
    let spec = &prepared.plan.inputs[0];
    let raw = bytes(data);
    let owner = prepared
        .owner(
            prepared.session.upload_bytes(&raw).unwrap(),
            spec.scalar,
            Rc::clone(&spec.shape),
            spec.count,
        )
        .unwrap();
    let resident = [(
        input,
        MetalTensorInput::Resident {
            scalar: T::TYPE,
            elements: data.len(),
            resource: owner.resource(),
        },
    )];
    let old = prepared.execute_mixed(&resident).unwrap();
    let mut actual = data.to_vec();
    old.read_into(&mut actual).unwrap();
    assert_eq!(bytes(&actual), raw);
    let host = [(
        input,
        MetalTensorInput::HostBytes {
            scalar: T::TYPE,
            elements: data.len(),
            bytes: &raw,
        },
    )];
    let host_output = prepared.execute_mixed(&host).unwrap();
    host_output.read_into(&mut actual).unwrap();
    assert_eq!(bytes(&actual), raw);
    let mut changed = data.to_vec();
    changed[3] = invalid;
    let raw_changed = bytes(&changed);
    let bad = [(
        input,
        MetalTensorInput::HostBytes {
            scalar: T::TYPE,
            elements: data.len(),
            bytes: &raw_changed,
        },
    )];
    let failure = prepared.execute_mixed(&bad).err().unwrap();
    assert_eq!(failure.effect, Some(prepared.plan.stages[0].effect));
    assert!(matches!(failure.cause,MetalError::Arithmetic(fault) if fault.invocation_id==3));
    old.read_into(&mut actual).unwrap();
    assert_eq!(bytes(&actual), raw);
    drop(host_output);
    drop(old);
}

fn single_scalar_execution(prepared: &MetalPreparedSelectedTensorGraph, scalar: PcuScalarType) {
    match scalar {
        PcuScalarType::F32 => single_execution(prepared, &[1.0_f32, -2.0, 3.0, -4.0], f32::NAN),
        PcuScalarType::F64 => single_execution(prepared, &[1.0_f64, -2.0, 3.0, -4.0], f64::NAN),
        PcuScalarType::F16 => single_execution(
            prepared,
            &[15360, 49152, 16896, 50176].map(fusion_pcu::PcuF16Bits::from_bits),
            fusion_pcu::PcuF16Bits::from_bits(32257),
        ),
        PcuScalarType::BF16 => single_execution(
            prepared,
            &[16256, 49152, 16448, 49280].map(fusion_pcu::PcuBf16Bits::from_bits),
            fusion_pcu::PcuBf16Bits::from_bits(32705),
        ),
        PcuScalarType::F8E4M3FN => single_execution(
            prepared,
            &[56, 192, 68, 200].map(fusion_pcu::PcuF8E4M3FnBits::from_bits),
            fusion_pcu::PcuF8E4M3FnBits::from_bits(127),
        ),
        PcuScalarType::F8E5M2 => single_execution(
            prepared,
            &[60, 192, 66, 196].map(fusion_pcu::PcuF8E5M2Bits::from_bits),
            fusion_pcu::PcuF8E5M2Bits::from_bits(127),
        ),
        _ => unreachable!(),
    }
}

fn offer_program(
    profile: usize,
    scalar: PcuScalarType,
    requirements: PcuImplementationRequirements,
) -> Arc<TensorOwnedSelectedProgram> {
    if profile == 4 {
        single_input(scalar, requirements)
    } else if matches!(scalar, PcuScalarType::F32 | PcuScalarType::F64) {
        source(profile, scalar, requirements)
    } else {
        super::tests::low_source(profile, scalar, requirements)
    }
}

const GRAPH_SCALARS: [PcuScalarType; 6] = [
    PcuScalarType::F16,
    PcuScalarType::BF16,
    PcuScalarType::F8E4M3FN,
    PcuScalarType::F8E5M2,
    PcuScalarType::F32,
    PcuScalarType::F64,
];
