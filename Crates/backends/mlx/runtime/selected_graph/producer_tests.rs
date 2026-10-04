//! Independent raw-bit producer fixtures match the ordinary immutable carrier source.
use super::super::*;
use super::requests;
#[rustfmt::skip]
use fusion_pcu::dialect::tensor::{
    Graph,
    Tensor,
    TensorElement,
    TensorArithmeticRewritePolicy,
    TensorArithmeticCapability,
    TensorPointwiseGroupingPolicy,
};
#[rustfmt::skip]
use fusion_pcu::{
    PcuScalar,
    PcuNumericalMode,
    PcuImplementationOffers,
    PcuDeviceActivation,
};
fn source<T: TensorElement>(
    request: PcuImplementationRequirements,
    payload: [T; 3],
    uniform: bool,
    matrix: bool,
) -> Arc<TensorOwnedSelectedProgram> {
    let mut graph = Graph::default();
    graph.set_numerical_mode(request.numerical_mode);
    graph.set_numerical_options(request.numerical_options);
    let shape = if matrix { vec![2, 3] } else { vec![3] };
    let output = if uniform {
        graph.uniform_typed(shape, payload[0]).unwrap().erase()
    } else {
        let values = if matrix {
            vec![
                payload[0], payload[1], payload[2], payload[2], payload[1], payload[0],
            ]
        } else {
            payload.to_vec()
        };
        graph
            .constant_typed(Tensor::new(shape, values).unwrap())
            .erase()
    };
    Arc::new(
        graph
            .into_selected_program(
                &[output],
                TensorArithmeticRewritePolicy::Disabled,
                TensorArithmeticCapability::Strict,
                TensorPointwiseGroupingPolicy::Disabled,
            )
            .unwrap(),
    )
}
fn headers() -> Vec<PcuImplementationRequirements> {
    requests()
        .into_iter()
        .flat_map(|request| {
            [
                request,
                PcuImplementationRequirements {
                    numerical_mode: PcuNumericalMode::Boundary,
                    ..request
                },
            ]
        })
        .collect()
}
fn expected<T: PcuScalar>(payload: [T; 3], uniform: bool, matrix: bool) -> Vec<T> {
    if uniform {
        vec![payload[0]; if matrix { 6 } else { 3 }]
    } else if matrix {
        vec![
            payload[0], payload[1], payload[2], payload[2], payload[1], payload[0],
        ]
    } else {
        payload.to_vec()
    }
}
fn bytes<T: PcuScalar>(values: &[T]) -> Vec<u8> {
    let mut bytes = Vec::new();
    for &value in values {
        bytes.extend_from_slice(value.encode_le().as_ref());
    }
    bytes
}
macro_rules! formats {
    ($function:ident $(,$arg:expr)*) => {{
        $function($($arg,)* [f32::from_bits(0x7f80_0042), f32::from_bits(0x8000_0000), f32::from_bits(1)]);
        $function($($arg,)* [f64::from_bits(0x7ff0_0000_0000_0042), f64::from_bits(0x8000_0000_0000_0000), f64::from_bits(1)]);
        $function($($arg,)* [0x7c42,0x8000,1].map(fusion_pcu::PcuF16Bits::from_bits));
        $function($($arg,)* [0x7f82,0x8000,1].map(fusion_pcu::PcuBf16Bits::from_bits));
        $function($($arg,)* [0x7f,0x80,1].map(fusion_pcu::PcuF8E4M3FnBits::from_bits));
        $function($($arg,)* [0x7d,0x80,1].map(fusion_pcu::PcuF8E5M2Bits::from_bits));
        $function($($arg,)* [[0x42,0x7fff_0000_0000_0000],[0,0x8000_0000_0000_0000],[1,0]].map(fusion_pcu::PcuF128Bits::from_limbs_le));
        $function($($arg,)* [[0x42,0,0,0x7fff_f000_0000_0000],[0,0,0,0x8000_0000_0000_0000],[1,0,0,0]].map(fusion_pcu::PcuF256Bits::from_limbs_le));
    }};
}
fn pure<T: TensorElement>(payload: [T; 3]) {
    for request in headers() {
        for uniform in [false, true] {
            for matrix in [false, true] {
                let original = source(request, payload, uniform, matrix);
                let plan =
                    MlxSelectedTensorGraphPlan::assess_program(Arc::clone(&original), request)
                        .unwrap();
                assert!(Arc::ptr_eq(&plan.program_owner(), &original));
                assert!(plan.input_values().is_empty());
                let mut portable = request;
                portable.numerical_options.reproducibility =
                    fusion_pcu::PcuReproducibility::PortableV1;
                let unqualified = source(portable, payload, uniform, matrix);
                assert!(MlxSelectedTensorGraphPlan::assess_program(unqualified, portable).is_err());
                assert!(plan.stages.is_empty());
                assert_eq!(plan.producers.len(), 1);
                assert_eq!(plan.requirements(), request);
                assert_eq!(plan.scalar_type(), T::TYPE);
                assert_eq!(
                    plan.shape(),
                    if matrix {
                        [2, 3].as_slice()
                    } else {
                        [3].as_slice()
                    }
                );
                let producer = &plan.producers[0];
                let expected = if uniform {
                    bytes(&payload[..1])
                } else {
                    bytes(&expected(payload, uniform, matrix))
                };
                assert_eq!(producer.bytes(plan.program()).unwrap(), expected);
                let node = original.graph().node(producer.value).unwrap();
                assert!(node.numerical_mode.is_none());
                assert!(node.float_underflow_policy.is_none());
                let mut wrong = request;
                wrong.numerical_options.compound_arithmetic =
                    match request.numerical_options.compound_arithmetic {
                        fusion_pcu::PcuCompoundArithmeticPolicy::Checked => {
                            fusion_pcu::PcuCompoundArithmeticPolicy::BackendDefined
                        }
                        fusion_pcu::PcuCompoundArithmeticPolicy::BackendDefined => {
                            fusion_pcu::PcuCompoundArithmeticPolicy::Checked
                        }
                    };
                assert!(
                    MlxSelectedTensorGraphPlan::assess_program(Arc::clone(&original), wrong)
                        .is_err()
                );
            }
        }
    }
}
#[test]
fn immutable_producer_eight_formats_exact_headers_shapes_and_original_payloads() {
    formats!(pure);
}
fn verify_offers(
    discovery: &crate::MlxDiscovery,
    device: fusion_pcu::PcuDeviceIdentity,
    plan: &MlxSelectedTensorGraphPlan,
) {
    let request = plan.requirements();
    let operation = crate::MlxSelectedTensorGraphRequest { plan };
    for boundary in [
        fusion_pcu::PcuCostBoundary::HostInputsResidentOutput,
        fusion_pcu::PcuCostBoundary::Resident,
        fusion_pcu::PcuCostBoundary::MixedInputsResidentOutput,
        fusion_pcu::PcuCostBoundary::Host,
    ] {
        let request_offer = fusion_pcu::PcuImplementationRequest {
            device,
            executor: fusion_pcu::PcuExecutorId(0),
            requirements: request,
            boundary,
            operation: &operation,
        };
        let expected_offers = usize::from(matches!(
            boundary,
            fusion_pcu::PcuCostBoundary::HostInputsResidentOutput
                | fusion_pcu::PcuCostBoundary::Resident
        ));
        let mut offers = [None; 2];
        assert_eq!(
            discovery
                .implementation_offers(&request_offer, &mut offers)
                .unwrap(),
            expected_offers
        );
        assert_eq!(
            discovery
                .implementation_offers(&request_offer, &mut [])
                .unwrap(),
            expected_offers
        );
        assert_eq!(offers[1], None);
        if expected_offers == 1 {
            let offer = offers[0].unwrap();
            assert_eq!(offer.implementation, plan.implementation_id(device));
            assert_eq!(offer.requirements, request);
            assert_eq!(offer.workspace_bytes, None);
            assert_eq!(
                offer.cost,
                fusion_pcu::PcuImplementationCost::unknown(boundary)
            );
        } else {
            assert_eq!(offers, [None; 2]);
        }
        let mut wrong = request_offer;
        wrong.requirements.numerical_mode = match request.numerical_mode {
            PcuNumericalMode::Boundary => PcuNumericalMode::Strict,
            PcuNumericalMode::Strict => PcuNumericalMode::Boundary,
        };
        assert!(
            discovery
                .implementation_offers(&wrong, &mut offers)
                .is_err()
        );
    }
}
fn native<T: TensorElement>(
    session: &crate::MlxSession,
    discovery: &crate::MlxDiscovery,
    device: fusion_pcu::PcuDeviceIdentity,
    payload: [T; 3],
) {
    for request in headers() {
        for uniform in [false, true] {
            for matrix in [false, true] {
                let original = source(request, payload, uniform, matrix);
                let plan = MlxSelectedTensorGraphPlan::assess_program(original, request).unwrap();
                let kernel = session.prepare_selected_tensor_graph(plan).unwrap();
                verify_offers(discovery, device, kernel.plan());
                assert_eq!(
                    kernel.implementation_id(),
                    Some(kernel.plan().implementation_id(device))
                );
                let expected = expected(payload, uniform, matrix);
                // Detached single-output SDK carrier: same logical read/write spans,
                // with no PCU graph assessment or source lowering.
                let control = session
                    .prepare_carrier_native(T::TYPE, expected.len(), uniform)
                    .unwrap();
                let dense = session.upload_encoded(&expected).unwrap();
                let seed = session.upload_encoded(&payload[..1]).unwrap();
                let input = if uniform { &seed } else { &dense };
                let direct = session.wrap_encoded(control.execute(input.native()).unwrap());
                let old = kernel.execute_mixed(&[]).unwrap();
                for _ in 0..3 {
                    let output = kernel.execute_mixed(&[]).unwrap();
                    for owner in [&output, &old, &direct] {
                        let mut read = vec![payload[1]; expected.len() + 2];
                        owner.read_into(&mut read).unwrap();
                        assert_eq!(bytes(&read[..expected.len()]), bytes(&expected));
                        assert_eq!(bytes(&read[expected.len()..]), bytes(&[payload[1]; 2]));
                    }
                    output.release().unwrap();
                }
                let invalid = [(
                    kernel.plan().output(),
                    crate::MlxCheckedProgramInput::Resident(&dense),
                )];
                assert!(kernel.execute_mixed(&invalid).is_err());
                drop(kernel);
                let mut read = vec![payload[1]; expected.len() + 2];
                old.read_into(&mut read).unwrap();
                assert_eq!(bytes(&read[..expected.len()]), bytes(&expected));
                old.release().unwrap();
                direct.release().unwrap();
                dense.release().unwrap();
                seed.release().unwrap();
            }
        }
    }
}
#[test]
#[ignore = "Required actual M4 private cold seeds and fresh immutable output publication."]
fn immutable_producer_native_eight_formats_all_headers_shapes_and_old_owners() {
    let discovery = crate::MlxDiscovery::discover_default().unwrap();
    let device = super::super::offers::device_identity(&discovery);
    let session = discovery
        .open_device(discovery.device_reference(0).unwrap())
        .unwrap();
    formats!(native, &session, &discovery, device);
}
