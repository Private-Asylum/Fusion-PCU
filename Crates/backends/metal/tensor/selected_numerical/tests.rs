use super::*;
use fusion_pcu::{PcuNumericalMode, PcuCompoundArithmeticPolicy, PcuFloatUnderflowPolicy};
use fusion_pcu::dialect::tensor::{
    Graph, TensorArithmeticRewritePolicy, TensorArithmeticCapability, TensorPointwiseGroupingPolicy,
};
fn request() -> PcuImplementationRequirements {
    let mut request = PcuImplementationRequirements {
        numerical_mode: PcuNumericalMode::Strict,
        ..Default::default()
    };
    request.numerical_options.compound_arithmetic = PcuCompoundArithmeticPolicy::Checked;
    request
}
fn strict_requests() -> Vec<PcuImplementationRequirements> {
    let mut requests = Vec::new();
    for precision in [
        fusion_pcu::PcuPrecisionPolicy::Preserve,
        fusion_pcu::PcuPrecisionPolicy::BackendOptimized,
    ] {
        for compound in [
            PcuCompoundArithmeticPolicy::Checked,
            PcuCompoundArithmeticPolicy::BackendDefined,
        ] {
            for policy in [
                PcuFloatUnderflowPolicy::IeeeAfterRounding,
                PcuFloatUnderflowPolicy::RejectSubnormalResult,
                PcuFloatUnderflowPolicy::AllowGradualUnderflow,
            ] {
                let mut request = request();
                request.numerical_options.precision = precision;
                request.numerical_options.compound_arithmetic = compound;
                request.float_underflow = policy;
                requests.push(request);
            }
        }
    }
    requests
}
fn source(
    operation: u8,
    identity: bool,
    request: PcuImplementationRequirements,
    scalar: PcuScalarType,
) -> Arc<TensorOwnedSelectedProgram> {
    source_with_mse_count(operation, identity, request, scalar, 6)
}
fn source_with_mse_count(
    operation: u8,
    identity: bool,
    request: PcuImplementationRequirements,
    scalar: PcuScalarType,
    count: usize,
) -> Arc<TensorOwnedSelectedProgram> {
    let mut graph = Graph::default();
    graph.set_numerical_mode(request.numerical_mode);
    graph.set_numerical_options(request.numerical_options);
    let shape = if operation == 3 && count != 6 {
        vec![count]
    } else {
        vec![2, 3]
    };
    let left = graph.input(shape.clone(), scalar).unwrap();
    let right = if operation == 1 {
        graph.input([3, 4], scalar).unwrap()
    } else {
        graph.input(shape, scalar).unwrap()
    };
    let effect = match operation {
        0 => graph.relu_backward(left, right).unwrap(),
        1 => graph.matmul(left, right).unwrap(),
        2 => graph.sgd_update(left, right, -0.0).unwrap(),
        _ => graph.mean_squared_error(left, right).unwrap(),
    };
    graph
        .set_value_float_underflow_policy(effect, request.float_underflow)
        .unwrap();
    Arc::new(
        graph
            .into_selected_program(
                &[if identity { right } else { effect }],
                TensorArithmeticRewritePolicy::Disabled,
                TensorArithmeticCapability::Strict,
                TensorPointwiseGroupingPolicy::Disabled,
            )
            .unwrap(),
    )
}
#[test]
fn numerical_aggregate_retains_original_owner_roles_and_effect_policies() {
    let request = request();
    for operation in 0..4 {
        for identity in [false, true] {
            let source = source(operation, identity, request, PcuScalarType::F64);
            let plan =
                MetalSelectedNumericalTensorPlan::assess_program(Arc::clone(&source), request)
                    .unwrap();
            assert!(Arc::ptr_eq(&plan.program_owner(), &source));
            assert_eq!(plan.requirements(), request);
            assert_eq!(plan.input_shape(0), Some([2, 3].as_slice()));
            assert_eq!(
                plan.input_shape(1),
                Some(if operation == 1 {
                    [3, 4].as_slice()
                } else {
                    [2, 3].as_slice()
                })
            );
            assert_eq!(
                plan.input_element_count(1),
                Some(if operation == 1 { 12 } else { 6 })
            );
            assert_eq!(
                plan.input_byte_len(1),
                Some(if operation == 1 { 96 } else { 48 })
            );
            assert_eq!(plan.input_shape(2), None);
            assert_eq!(plan.input_element_count(2), None);
            assert_eq!(plan.selected_input(), identity.then_some(1));
            assert_eq!(
                plan.learning_rate_bits(),
                (operation == 2).then_some((-0.0f32).to_bits())
            );
            assert_eq!(
                plan.matmul_dimensions(),
                (operation == 1).then_some([2, 3, 4])
            );
            assert_eq!(
                plan.shape(),
                if identity && operation == 1 {
                    [3, 4].as_slice()
                } else if operation == 1 {
                    [2, 4].as_slice()
                } else if operation == 3 && !identity {
                    &[]
                } else {
                    [2, 3].as_slice()
                }
            );
            assert_eq!(
                plan.element_count(),
                if operation == 3 && !identity {
                    1
                } else if operation == 1 {
                    if identity { 12 } else { 8 }
                } else {
                    6
                }
            );
            let mut mismatched = request;
            mismatched.float_underflow = PcuFloatUnderflowPolicy::RejectSubnormalResult;
            assert!(
                MetalSelectedNumericalTensorPlan::assess_program(Arc::clone(&source), mismatched)
                    .is_err()
            );
            mismatched = request;
            mismatched.numerical_options.compound_arithmetic =
                PcuCompoundArithmeticPolicy::BackendDefined;
            assert!(
                MetalSelectedNumericalTensorPlan::assess_program(Arc::clone(&source), mismatched)
                    .is_err()
            );
        }
    }
}
#[test]
fn numerical_aggregate_never_hides_discarded_effect_policy() {
    for operation in 0..4 {
        let source = source(
            operation,
            true,
            PcuImplementationRequirements {
                float_underflow: PcuFloatUnderflowPolicy::RejectSubnormalResult,
                ..request()
            },
            PcuScalarType::F64,
        );
        assert!(MetalSelectedNumericalTensorPlan::assess_program(source, request()).is_err());
    }
}

fn bytes<T: fusion_pcu::PcuCheckedFloat>(values: &[T]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.encode_le().as_ref().to_vec())
        .collect()
}
fn oracle<T: fusion_pcu::PcuCheckedFloat>(
    operation: u8,
    left: &[T],
    right: &[T],
    zero: T,
    rate: T,
    denominator: T,
    policy: PcuFloatUnderflowPolicy,
) -> Vec<T> {
    match operation {
        0 => left
            .iter()
            .zip(right)
            .map(|(&activation, &gradient)| {
                activation
                    .pcu_checked_relu_backward_with_policy(gradient, policy)
                    .unwrap()
            })
            .collect(),
        1 => (0..8)
            .map(|index| {
                let mut sum = zero;
                for inner in 0..3 {
                    let product = left[(index / 4) * 3 + inner]
                        .pcu_checked_mul_with_policy(right[inner * 4 + index % 4], policy)
                        .unwrap();
                    sum = sum.pcu_checked_add_with_policy(product, policy).unwrap();
                }
                sum
            })
            .collect(),
        2 => left
            .iter()
            .zip(right)
            .map(|(&weight, &gradient)| {
                weight
                    .pcu_checked_sub_with_policy(
                        gradient.pcu_checked_mul_with_policy(rate, policy).unwrap(),
                        policy,
                    )
                    .unwrap()
            })
            .collect(),
        _ => {
            let mut sum = zero;
            for (&prediction, &target) in left.iter().zip(right) {
                let difference = prediction
                    .pcu_checked_sub_with_policy(target, policy)
                    .unwrap();
                let squared = difference
                    .pcu_checked_mul_with_policy(difference, policy)
                    .unwrap();
                sum = sum.pcu_checked_add_with_policy(squared, policy).unwrap();
            }
            vec![
                sum.pcu_checked_div_with_policy(denominator, policy)
                    .unwrap(),
            ]
        }
    }
}

fn owner<T: fusion_pcu::PcuCheckedFloat>(session: &MetalSession, values: &[T]) -> MetalTensorOwner {
    let mut graph = Graph::default();
    let input = graph.input([values.len()], T::TYPE).unwrap();
    let source = graph
        .into_selected_program(
            &[input],
            TensorArithmeticRewritePolicy::Disabled,
            TensorArithmeticCapability::Strict,
            TensorPointwiseGroupingPolicy::Disabled,
        )
        .unwrap();
    let plan =
        crate::MetalTensorPlan::assess_program(&source, PcuImplementationRequirements::default())
            .unwrap();
    session
        .prepare_tensor_program(plan, fusion_pcu::PcuMemoryPoolId(145))
        .unwrap()
        .execute(MetalTensorInput::HostBytes {
            scalar: T::TYPE,
            elements: values.len(),
            bytes: &bytes(values),
        })
        .unwrap()
}
#[allow(clippy::too_many_lines)] // One reciprocal mixed-owner matrix checks all sealed variants without changing the leaf proof.
fn native<T: fusion_pcu::PcuCheckedFloat>(
    session: &MetalSession,
    foreign: &MetalSession,
    values: &[T],
    zero: T,
    rate: T,
    denominator: T,
    invalid: T,
) {
    for request in strict_requests() {
        for operation in aggregate_operations(T::TYPE, values.len() > 12) {
            for identity in [false, true] {
                let count = if values.len() > 12 {
                    values.len() / 2
                } else {
                    6
                };
                let source = source_with_mse_count(operation, identity, request, T::TYPE, count);
                let plan =
                    MetalSelectedNumericalTensorPlan::assess_program(Arc::clone(&source), request)
                        .unwrap();
                let kernel = session
                    .prepare_selected_numerical_tensor_program(
                        plan,
                        fusion_pcu::PcuMemoryPoolId(146),
                    )
                    .unwrap();
                let left = &values[..count];
                let right = if operation == 3 {
                    &values[count..count * 2]
                } else {
                    &values[..if operation == 1 { 12 } else { 6 }]
                };
                let a = owner(session, left);
                let b = owner(session, right);
                let wrong = owner(foreign, left);
                let a_bytes = bytes(left);
                let b_bytes = bytes(right);
                let expected = if identity {
                    right.to_vec()
                } else {
                    oracle(
                        operation,
                        left,
                        right,
                        zero,
                        rate,
                        denominator,
                        request.float_underflow,
                    )
                };
                for role in 0..4 {
                    let inputs = [
                        if role & 1 == 0 {
                            MetalTensorInput::HostBytes {
                                scalar: T::TYPE,
                                elements: left.len(),
                                bytes: &a_bytes,
                            }
                        } else {
                            MetalTensorInput::Resident {
                                scalar: T::TYPE,
                                elements: left.len(),
                                resource: a.resource(),
                            }
                        },
                        if role & 2 == 0 {
                            MetalTensorInput::HostBytes {
                                scalar: T::TYPE,
                                elements: right.len(),
                                bytes: &b_bytes,
                            }
                        } else {
                            MetalTensorInput::Resident {
                                scalar: T::TYPE,
                                elements: right.len(),
                                resource: b.resource(),
                            }
                        },
                    ];
                    let output = kernel.execute(&inputs).unwrap();
                    let mut actual = vec![zero; expected.len() + 1];
                    output.read_into(&mut actual).unwrap();
                    assert_eq!(bytes(&actual[..expected.len()]), bytes(&expected));
                    assert_eq!(bytes(&actual[expected.len()..]), bytes(&[zero]));
                }
                let inputs = [
                    MetalTensorInput::Resident {
                        scalar: T::TYPE,
                        elements: left.len(),
                        resource: wrong.resource(),
                    },
                    MetalTensorInput::Resident {
                        scalar: T::TYPE,
                        elements: right.len(),
                        resource: b.resource(),
                    },
                ];
                assert!(kernel.execute(&inputs).is_err());
                let mut bad = left.to_vec();
                let position = if left.len() > 10 { left.len() - 1 } else { 0 };
                bad[position] = invalid;
                let bad_bytes = bytes(&bad);
                let inputs = [
                    MetalTensorInput::HostBytes {
                        scalar: T::TYPE,
                        elements: left.len(),
                        bytes: &bad_bytes,
                    },
                    MetalTensorInput::Resident {
                        scalar: T::TYPE,
                        elements: right.len(),
                        resource: b.resource(),
                    },
                ];
                assert!(matches!(
                    kernel.execute(&inputs),
                    Err(MetalError::Arithmetic(fault)) if operation != 3 || (fault.invocation_id == 3 * u64::try_from(position).unwrap() && fault.kind == fusion_pcu::PcuExecutionFaultKind::InvalidFloatingOperand && !fault.recovered)
                ));
                let mut old = vec![zero; left.len()];
                a.read_into(&mut old).unwrap();
                assert_eq!(bytes(&old), a_bytes);
                let mut old = vec![zero; right.len()];
                b.read_into(&mut old).unwrap();
                assert_eq!(bytes(&old), b_bytes);
                assert!(Arc::ptr_eq(&kernel.plan().program_owner(), &source));
            }
        }
    }
}
#[test]
#[ignore = "Requires actual Metal mixed numerical aggregate and immutable owners."]
fn numerical_aggregate_native_mixed_roles_and_discarded_faults() {
    let session = MetalSession::open(0).unwrap();
    let foreign = MetalSession::open(0).unwrap();
    native(
        &session,
        &foreign,
        &[1f32, 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.],
        0.,
        -0.,
        6.,
        f32::NAN,
    );
    native(
        &session,
        &foreign,
        &[1f64, 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.],
        0.,
        -0.,
        6.,
        f64::NAN,
    );
}

use fusion_pcu::{PcuImplementationOffers, PcuRuntimeDiscovery};
fn device_identity(discovery: &crate::MetalDiscovery) -> fusion_pcu::PcuDeviceIdentity {
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

#[test]
#[ignore = "Requires actual Apple inventory for exact numerical aggregate offers."]
fn numerical_aggregate_native_offers_preserve_full_request_and_boundary() {
    let discovery = crate::MetalDiscovery::discover().unwrap();
    let device = device_identity(&discovery);
    for requirements in strict_requests() {
        for scalar in AGGREGATE_SCALARS {
            for operation in aggregate_operations(scalar, false) {
                for identity in [false, true] {
                    let source = source(operation, identity, requirements, scalar);
                    let plan =
                        MetalSelectedNumericalTensorPlan::assess_program(source, requirements)
                            .unwrap();
                    assert_eq!(
                        plan.implementation_id(device).revision,
                        expected_revision(operation)
                    );
                    let operation = crate::MetalSelectedNumericalTensorRequest { plan: &plan };
                    for boundary in [
                        fusion_pcu::PcuCostBoundary::HostInputsResidentOutput,
                        fusion_pcu::PcuCostBoundary::Resident,
                        fusion_pcu::PcuCostBoundary::MixedInputsResidentOutput,
                        fusion_pcu::PcuCostBoundary::Host,
                    ] {
                        let request = fusion_pcu::PcuImplementationRequest {
                            device,
                            executor: fusion_pcu::PcuExecutorId(0),
                            requirements,
                            boundary,
                            operation: &operation,
                        };
                        let mut offers = [None];
                        let expected = usize::from(boundary != fusion_pcu::PcuCostBoundary::Host);
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
                        if expected == 1 {
                            let offer = offers[0].unwrap();
                            assert_eq!(offer.requirements, requirements);
                            assert_eq!(
                                offer.cost,
                                fusion_pcu::PcuImplementationCost::unknown(boundary)
                            );
                            assert_eq!(offer.workspace_bytes, None);
                        }
                        let mut wrong = request;
                        wrong.requirements.float_underflow = if requirements.float_underflow
                            == PcuFloatUnderflowPolicy::RejectSubnormalResult
                        {
                            PcuFloatUnderflowPolicy::IeeeAfterRounding
                        } else {
                            PcuFloatUnderflowPolicy::RejectSubnormalResult
                        };
                        assert!(
                            discovery
                                .implementation_offers(&wrong, &mut offers)
                                .is_err()
                        );
                        assert_eq!(offers, [None]);
                        wrong.requirements = requirements;
                        wrong.executor = fusion_pcu::PcuExecutorId(99);
                        assert!(
                            discovery
                                .implementation_offers(&wrong, &mut offers)
                                .is_err()
                        );
                        assert_eq!(offers, [None]);
                    }
                }
            }
        }
    }
}

#[test]
fn numerical_aggregate_repeated_mse_has_one_actual_input_and_separate_output_shape() {
    for request in strict_requests() {
        for scalar in [PcuScalarType::F32, PcuScalarType::F64] {
            for identity in [false, true] {
                let mut graph = Graph::default();
                graph.set_numerical_mode(request.numerical_mode);
                graph.set_numerical_options(request.numerical_options);
                let input = graph.input([2, 3], scalar).unwrap();
                let effect = graph.mean_squared_error(input, input).unwrap();
                graph
                    .set_value_float_underflow_policy(effect, request.float_underflow)
                    .unwrap();
                let source = Arc::new(
                    graph
                        .into_selected_program(
                            &[if identity { input } else { effect }],
                            TensorArithmeticRewritePolicy::Disabled,
                            TensorArithmeticCapability::Strict,
                            TensorPointwiseGroupingPolicy::Disabled,
                        )
                        .unwrap(),
                );
                let plan =
                    MetalSelectedNumericalTensorPlan::assess_program(Arc::clone(&source), request)
                        .unwrap();
                assert!(Arc::ptr_eq(&source, &plan.program_owner()));
                assert_eq!(
                    plan.operation(),
                    MetalSelectedNumericalTensorOperation::StrictMse
                );
                assert_eq!(plan.requirements(), request);
                assert_eq!(plan.input_values(), [input]);
                assert_eq!(plan.operand_inputs(), [0, 0]);
                assert_eq!(plan.input_shape(0), Some([2, 3].as_slice()));
                assert_eq!(plan.input_shape(1), None);
                assert_eq!(plan.input_element_count(0), Some(6));
                assert_eq!(plan.input_element_count(1), None);
                assert_eq!(plan.selected_input(), identity.then_some(0));
                assert_eq!(plan.shape(), if identity { [2, 3].as_slice() } else { &[] });
                assert_eq!(plan.element_count(), if identity { 6 } else { 1 });
                assert_eq!(plan.learning_rate_bits(), None);
                assert_eq!(plan.matmul_dimensions(), None);
            }
        }
    }
}

#[test]
#[ignore = "Requires actual native inventory for large compact MSE exact offers and authentic receipts."]
fn numerical_aggregate_compact_mse_native_exact_large_offers() {
    let discovery = crate::MetalDiscovery::discover().unwrap();
    let device = device_identity(&discovery);
    for requirements in strict_requests() {
        for scalar in [PcuScalarType::F32, PcuScalarType::F64] {
            for count in [11, 1024, 65535] {
                let operation = 3;
                for identity in [false, true] {
                    let source =
                        source_with_mse_count(operation, identity, requirements, scalar, count);
                    let plan =
                        MetalSelectedNumericalTensorPlan::assess_program(source, requirements)
                            .unwrap();
                    assert_eq!(plan.input_element_count(0), Some(count));
                    assert_eq!(
                        plan.implementation_id(device).revision,
                        0x0000_0008_0000_0301
                    );
                    let operation = crate::MetalSelectedNumericalTensorRequest { plan: &plan };
                    for boundary in [
                        fusion_pcu::PcuCostBoundary::HostInputsResidentOutput,
                        fusion_pcu::PcuCostBoundary::Resident,
                        fusion_pcu::PcuCostBoundary::MixedInputsResidentOutput,
                        fusion_pcu::PcuCostBoundary::Host,
                    ] {
                        let request = fusion_pcu::PcuImplementationRequest {
                            device,
                            executor: fusion_pcu::PcuExecutorId(0),
                            requirements,
                            boundary,
                            operation: &operation,
                        };
                        let mut offers = [None];
                        let expected = usize::from(boundary != fusion_pcu::PcuCostBoundary::Host);
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
                        if expected == 1 {
                            let offer = offers[0].unwrap();
                            assert_eq!(offer.requirements, requirements);
                            assert_eq!(
                                offer.cost,
                                fusion_pcu::PcuImplementationCost::unknown(boundary)
                            );
                            assert_eq!(offer.workspace_bytes, None);
                        }
                        let mut wrong = request;
                        wrong.requirements.float_underflow = if requirements.float_underflow
                            == PcuFloatUnderflowPolicy::RejectSubnormalResult
                        {
                            PcuFloatUnderflowPolicy::IeeeAfterRounding
                        } else {
                            PcuFloatUnderflowPolicy::RejectSubnormalResult
                        };
                        assert!(
                            discovery
                                .implementation_offers(&wrong, &mut offers)
                                .is_err()
                        );
                        assert_eq!(offers, [None]);
                        wrong.requirements = requirements;
                        wrong.executor = fusion_pcu::PcuExecutorId(99);
                        assert!(
                            discovery
                                .implementation_offers(&wrong, &mut offers)
                                .is_err()
                        );
                        assert_eq!(offers, [None]);
                    }
                }
            }
        }
    }
}

#[test]
#[ignore = "Requires actual Metal numerical aggregate large MSE mixed roles, late faults and receipt revision."]
fn numerical_aggregate_compact_mse_native_large_roles_and_faults() {
    let session = MetalSession::open(0).unwrap();
    let foreign = MetalSession::open(0).unwrap();
    for count in [11, 1024, 65535] {
        let mut values = vec![1f32; count * 2];
        values[count..].fill(2.);
        native(
            &session,
            &foreign,
            &values,
            0.,
            -0.,
            f32::from(u16::try_from(count).unwrap()),
            f32::NAN,
        );
        let mut values = vec![1f64; count * 2];
        values[count..].fill(2.);
        native(
            &session,
            &foreign,
            &values,
            0.,
            -0.,
            f64::from(u16::try_from(count).unwrap()),
            f64::NAN,
        );
    }
}

const AGGREGATE_SCALARS: [PcuScalarType; 6] = [
    PcuScalarType::F16,
    PcuScalarType::BF16,
    PcuScalarType::F8E4M3FN,
    PcuScalarType::F8E5M2,
    PcuScalarType::F32,
    PcuScalarType::F64,
];
fn aggregate_operations(scalar: PcuScalarType, large: bool) -> std::ops::Range<u8> {
    if !matches!(scalar, PcuScalarType::F32 | PcuScalarType::F64) {
        0..1
    } else if large {
        3..4
    } else {
        0..4
    }
}
#[test]
#[ignore = "Requires actual low-format aggregate backward exact roles, discarded effects and owner faults."]
fn numerical_aggregate_native_four_low_formats_backward() {
    let session = MetalSession::open(0).unwrap();
    let foreign = MetalSession::open(0).unwrap();
    macro_rules! low {
        ($ty:ty, $word:ty, $one:expr, $negative_one:expr, $negative_zero:expr, $two:expr, $negative_two:expr, $nan:expr) => {
            native(
                &session,
                &foreign,
                &[$one, $negative_one, 0, $negative_zero, $two, $negative_two]
                    .map(|bits| <$ty>::from_bits(<$word>::try_from(bits).unwrap())),
                <$ty>::from_bits(0),
                <$ty>::from_bits($negative_zero),
                <$ty>::from_bits($one),
                <$ty>::from_bits($nan),
            );
        };
    }
    low!(
        fusion_pcu::PcuF16Bits,
        u16,
        0x3c00,
        0xbc00,
        0x8000,
        0x4000,
        0xc000,
        0x7e01
    );
    low!(
        fusion_pcu::PcuBf16Bits,
        u16,
        0x3f80,
        0xbf80,
        0x8000,
        0x4000,
        0xc000,
        0x7fc1
    );
    low!(
        fusion_pcu::PcuF8E4M3FnBits,
        u8,
        0x38,
        0xb8,
        0x80,
        0x40,
        0xc0,
        0x7f
    );
    low!(
        fusion_pcu::PcuF8E5M2Bits,
        u8,
        0x3c,
        0xbc,
        0x80,
        0x40,
        0xc0,
        0x7f
    );
}

const fn expected_revision(operation: u8) -> u64 {
    match operation {
        0 => 0x0000_0008_0000_0302,
        3 => 0x0000_0008_0000_0301,
        _ => 0x0000_0008_0000_0300,
    }
}
