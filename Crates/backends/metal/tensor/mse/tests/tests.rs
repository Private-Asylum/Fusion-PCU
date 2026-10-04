use std::sync::Arc;
use fusion_pcu::{
    PcuScalarType, PcuImplementationRequirements, PcuNumericalMode, PcuCompoundArithmeticPolicy,
    PcuPrecisionPolicy, PcuFloatUnderflowPolicy, PcuReproducibility, PcuRangePolicy,
    PcuCheckedFloat, PcuExecutionFault,
};
use fusion_pcu::dialect::tensor::{
    Graph, TensorOwnedSelectedProgram, TensorArithmeticRewritePolicy, TensorArithmeticCapability,
    TensorPointwiseGroupingPolicy,
};
fn tuples() -> Vec<PcuImplementationRequirements> {
    let mut requests = Vec::new();
    for precision in [
        PcuPrecisionPolicy::Preserve,
        PcuPrecisionPolicy::BackendOptimized,
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
                let mut request = PcuImplementationRequirements {
                    numerical_mode: PcuNumericalMode::Strict,
                    float_underflow: policy,
                    ..Default::default()
                };
                request.numerical_options.compound_arithmetic = compound;
                request.numerical_options.precision = precision;
                requests.push(request);
            }
        }
    }
    requests
}

fn repeated(profile: u8) -> bool {
    matches!(profile, 0 | 7)
}
fn program(
    scalar: PcuScalarType,
    request: PcuImplementationRequirements,
    profile: u8,
) -> Arc<TensorOwnedSelectedProgram> {
    program_with_count(
        scalar,
        request,
        profile,
        if repeated(profile) { 4 } else { 6 },
    )
}
fn program_with_count(
    scalar: PcuScalarType,
    request: PcuImplementationRequirements,
    profile: u8,
    count: usize,
) -> Arc<TensorOwnedSelectedProgram> {
    let mut graph = Graph::default();
    graph.set_numerical_mode(request.numerical_mode);
    graph.set_numerical_options(request.numerical_options);
    let left = graph
        .input(
            if count == 4 {
                vec![2, 2]
            } else if count == 6 {
                vec![2, 3]
            } else {
                vec![count]
            },
            scalar,
        )
        .unwrap();
    let right = if repeated(profile) {
        left
    } else {
        graph
            .input(if count == 6 { vec![2, 3] } else { vec![count] }, scalar)
            .unwrap()
    };
    let effect = if profile >= 4 {
        graph.mean_squared_error(right, left)
    } else {
        graph.mean_squared_error(left, right)
    }
    .unwrap();
    graph
        .set_value_float_underflow_policy(effect, request.float_underflow)
        .unwrap();
    let output = match profile {
        2 | 6 | 7 => left,
        3 | 5 => right,
        _ => effect,
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
#[test]
fn mse_selected_schema_retains_shapes_full_request_and_discarded_effect() {
    for scalar in [PcuScalarType::F32, PcuScalarType::F64] {
        for request in tuples() {
            for profile in 0..8 {
                let source = program(scalar, request, profile);
                let plan = MetalTensorMsePlan::assess_program(&source, request).unwrap();
                assert_eq!(plan.requirements(), request);
                assert_eq!(
                    plan.operand_inputs(),
                    if repeated(profile) {
                        [0, 0]
                    } else if profile >= 4 {
                        [1, 0]
                    } else {
                        [0, 1]
                    }
                );
                assert_eq!(
                    plan.input_element_count(0),
                    Some(if repeated(profile) { 4 } else { 6 })
                );
                assert_eq!(
                    plan.element_count(),
                    if matches!(profile, 2 | 3 | 5 | 6 | 7) {
                        if repeated(profile) { 4 } else { 6 }
                    } else {
                        1
                    }
                );
                assert_eq!(
                    plan.shape(),
                    if matches!(profile, 2 | 3 | 5 | 6 | 7) {
                        if repeated(profile) {
                            [2, 2].as_slice()
                        } else {
                            [2, 3].as_slice()
                        }
                    } else {
                        &[]
                    }
                );
                let mut other = request;
                other.numerical_mode = PcuNumericalMode::Boundary;
                assert!(MetalTensorMsePlan::assess_program(&source, other).is_err());
                other = request;
                other.range_policy = PcuRangePolicy::Clamp;
                assert!(MetalTensorMsePlan::assess_program(&source, other).is_err());
                other = request;
                other.numerical_options.compound_arithmetic =
                    if request.numerical_options.compound_arithmetic
                        == PcuCompoundArithmeticPolicy::Checked
                    {
                        PcuCompoundArithmeticPolicy::BackendDefined
                    } else {
                        PcuCompoundArithmeticPolicy::Checked
                    };
                assert!(MetalTensorMsePlan::assess_program(&source, other).is_err());
                other = request;
                other.float_underflow =
                    if request.float_underflow == PcuFloatUnderflowPolicy::RejectSubnormalResult {
                        PcuFloatUnderflowPolicy::IeeeAfterRounding
                    } else {
                        PcuFloatUnderflowPolicy::RejectSubnormalResult
                    };
                assert!(MetalTensorMsePlan::assess_program(&source, other).is_err());
                other = request;
                other.numerical_options.precision =
                    if request.numerical_options.precision == PcuPrecisionPolicy::Preserve {
                        PcuPrecisionPolicy::BackendOptimized
                    } else {
                        PcuPrecisionPolicy::Preserve
                    };
                assert!(MetalTensorMsePlan::assess_program(&source, other).is_err());
                other = request;
                other.numerical_options.reproducibility = PcuReproducibility::PortableV1;
                assert!(MetalTensorMsePlan::assess_program(&source, other).is_err());
            }
        }
    }
}
fn bytes<T: PcuCheckedFloat>(values: &[T]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.encode_le().as_ref().to_vec())
        .collect()
}
fn oracle<T: PcuCheckedFloat>(
    prediction: &[T],
    target: &[T],
    count: T,
    policy: PcuFloatUnderflowPolicy,
    zero: T,
) -> Result<Vec<T>, PcuExecutionFault> {
    let fault = |kind, invocation_id| PcuExecutionFault {
        kind,
        invocation_id,
        recovered: false,
    };
    let mut sum = zero;
    for (index, (&prediction, &target)) in prediction.iter().zip(target).enumerate() {
        let ordinal = u64::try_from(index * 3).unwrap();
        let difference = prediction
            .pcu_checked_sub_with_policy(target, policy)
            .map_err(|kind| fault(kind, ordinal))?;
        let squared = difference
            .pcu_checked_mul_with_policy(difference, policy)
            .map_err(|kind| fault(kind, ordinal + 1))?;
        sum = sum
            .pcu_checked_add_with_policy(squared, policy)
            .map_err(|kind| fault(kind, ordinal + 2))?;
    }
    sum.pcu_checked_div_with_policy(count, policy)
        .map(|value| vec![value])
        .map_err(|kind| fault(kind, u64::try_from(prediction.len() * 3).unwrap()))
}

use super::*;
use fusion_pcu::PcuMemoryPoolId;
fn owner<T: PcuCheckedFloat>(session: &MetalSession, values: &[T]) -> MetalTensorOwner {
    let mut graph = Graph::default();
    let input = graph.input([values.len()], T::TYPE).unwrap();
    let selected = graph
        .into_selected_program(
            &[input],
            TensorArithmeticRewritePolicy::Disabled,
            TensorArithmeticCapability::Strict,
            TensorPointwiseGroupingPolicy::Disabled,
        )
        .unwrap();
    let plan = super::super::MetalTensorPlan::assess_program(
        &selected,
        PcuImplementationRequirements::default(),
    )
    .unwrap();
    let map = session
        .prepare_tensor_program(plan, PcuMemoryPoolId(122))
        .unwrap();
    map.execute(MetalTensorInput::HostBytes {
        scalar: T::TYPE,
        elements: values.len(),
        bytes: &bytes(values),
    })
    .unwrap()
}

fn host<T: PcuCheckedFloat>(bytes: &[u8], elements: usize) -> MetalTensorInput<'_> {
    MetalTensorInput::HostBytes {
        scalar: T::TYPE,
        elements,
        bytes,
    }
}
fn resident<T: PcuCheckedFloat>(owner: &MetalTensorOwner, elements: usize) -> MetalTensorInput<'_> {
    MetalTensorInput::Resident {
        scalar: T::TYPE,
        elements,
        resource: owner.resource(),
    }
}
fn fault_cases<T: PcuCheckedFloat>(
    prepared: &MetalPreparedTensorMseProgram,
    profile: u8,
    left: &[T],
    right: &[T],
    values: [T; 3],
    denominator: T,
) {
    let [zero, invalid, tiny] = values;
    let count = left.len();
    for (value, gradient_change) in [
        (invalid, false),
        (tiny, false),
        (invalid, true),
        (tiny, true),
    ] {
        let mut changed = left.to_vec();
        let mut changed_gradient = right.to_vec();
        if gradient_change {
            changed_gradient[if count > 10 { count - 1 } else { 0 }] = value;
        } else {
            changed[if count > 10 { count - 1 } else { 0 }] = value;
        }
        let changed_right = if repeated(profile) {
            if gradient_change {
                changed[if count > 10 { count - 1 } else { 0 }] = value;
            }
            changed.as_slice()
        } else {
            changed_gradient.as_slice()
        };
        let inputs = [changed.as_slice(), changed_right];
        let slots = prepared.plan().operand_inputs();
        let expected = oracle(
            inputs[slots[0]],
            inputs[slots[1]],
            denominator,
            prepared.plan().requirements().float_underflow,
            zero,
        );
        let a_bytes = bytes(&changed);
        let b_bytes = bytes(changed_right);
        let input = |bytes| MetalTensorInput::HostBytes {
            scalar: T::TYPE,
            elements: count,
            bytes,
        };
        let inputs = if repeated(profile) {
            vec![input(&a_bytes)]
        } else {
            vec![input(&a_bytes), input(&b_bytes)]
        };
        let actual = prepared.execute(&inputs);
        match expected {
            Err(fault) => {
                assert!(matches!(actual,Err(MetalError::Arithmetic(value)) if value==fault));
            }
            Ok(output) => {
                let expected = if profile == 2 || profile == 6 || profile == 7 {
                    changed.clone()
                } else if profile == 3 || profile == 5 {
                    changed_right.to_vec()
                } else {
                    output
                };
                let result = actual.unwrap();
                let mut values = vec![zero; expected.len()];
                result.read_into(&mut values).unwrap();
                assert_eq!(bytes(&values), bytes(&expected));
            }
        }
    }
}
#[allow(clippy::too_many_arguments, clippy::too_many_lines)] // One independent full request/denominator/ownership matrix retains both actual sessions and typed oracle factories.
fn native<T: PcuCheckedFloat>(
    session: &MetalSession,
    foreign: &MetalSession,
    zero: T,
    values: &[T],
    upstream: &[T],
    invalid: T,
    tiny: T,
    from_count: impl Fn(u16) -> T,
) {
    for request in tuples() {
        for profile in 0..8 {
            let count = if values.len() == 6 && repeated(profile) {
                4
            } else {
                values.len()
            };
            let denominator = from_count(u16::try_from(count).unwrap());
            let left = &values[..count];
            let right = if repeated(profile) { left } else { upstream };
            let source = program_with_count(T::TYPE, request, profile, count);
            let prepared = session
                .prepare_tensor_mse_program(Arc::clone(&source), request, PcuMemoryPoolId(144))
                .unwrap();
            assert!(std::ptr::eq(prepared.program(), source.as_ref()));
            let a = owner(session, left);
            let b = owner(session, right);
            let wrong = owner(foreign, right);
            let left_bytes = bytes(left);
            let right_bytes = bytes(right);
            let mut retained = Vec::new();
            for layout in 0..4 {
                let inputs = if repeated(profile) {
                    vec![if layout & 1 == 0 {
                        host::<T>(&left_bytes, count)
                    } else {
                        resident::<T>(&a, count)
                    }]
                } else {
                    vec![
                        if layout & 1 == 0 {
                            host::<T>(&left_bytes, count)
                        } else {
                            resident::<T>(&a, count)
                        },
                        if layout & 2 == 0 {
                            host::<T>(&right_bytes, count)
                        } else {
                            resident::<T>(&b, count)
                        },
                    ]
                };
                let operands = [left, right];
                let slots = prepared.plan().operand_inputs();
                let expected = oracle(
                    operands[slots[0]],
                    operands[slots[1]],
                    denominator,
                    request.float_underflow,
                    zero,
                );
                let result = prepared.execute(&inputs);
                match expected {
                    Err(fault) => {
                        assert!(
                            matches!(result,Err(MetalError::Arithmetic(value)) if value==fault)
                        );
                    }
                    Ok(output) => {
                        let expected = if profile == 2 || profile == 6 || profile == 7 {
                            left.to_vec()
                        } else if profile == 3 || profile == 5 {
                            right.to_vec()
                        } else {
                            output
                        };
                        let result = result.unwrap();
                        assert_eq!(result.shape(), prepared.plan().shape());
                        let mut actual = vec![zero; expected.len() + 2];
                        result.read_into(&mut actual).unwrap();
                        assert_eq!(bytes(&actual[..expected.len()]), bytes(&expected));
                        assert_eq!(bytes(&actual[expected.len()..]), bytes(&[zero; 2]));
                        retained.push((result, expected));
                    }
                }
            }
            fault_cases(
                &prepared,
                profile,
                left,
                right,
                [zero, invalid, tiny],
                denominator,
            );
            for (owner, expected) in &retained {
                let mut actual = vec![zero; expected.len() + 2];
                owner.read_into(&mut actual).unwrap();
                assert_eq!(bytes(&actual[..expected.len()]), bytes(expected));
                assert_eq!(bytes(&actual[expected.len()..]), bytes(&[zero; 2]));
            }

            let inputs = if repeated(profile) {
                vec![MetalTensorInput::Resident {
                    scalar: T::TYPE,
                    elements: count,
                    resource: wrong.resource(),
                }]
            } else {
                vec![
                    MetalTensorInput::Resident {
                        scalar: T::TYPE,
                        elements: count,
                        resource: a.resource(),
                    },
                    MetalTensorInput::Resident {
                        scalar: T::TYPE,
                        elements: count,
                        resource: wrong.resource(),
                    },
                ]
            };
            assert!(matches!(
                prepared.execute(&inputs),
                Err(MetalError::ForeignSession)
            ));
            let short = host::<T>(&left_bytes[..1], count);
            let short_inputs = if repeated(profile) {
                vec![short]
            } else {
                vec![short, resident::<T>(&b, count)]
            };
            assert!(matches!(
                prepared.execute(&short_inputs),
                Err(MetalError::InvalidExtent)
            ));
            let mut old = vec![zero; count];
            a.read_into(&mut old).unwrap();
            assert_eq!(bytes(&old), bytes(left));
            b.read_into(&mut old).unwrap();
            assert_eq!(bytes(&old), bytes(right));
        }
    }
}
#[test]
#[ignore = "Requires native Metal selected Strict MSE ownership and exact event arbitration."]
fn mse_selected_graph_host_resident_full_request_discarded_faults_and_old_owners() {
    let session = MetalSession::open(0).unwrap();
    let foreign = MetalSession::open(0).unwrap();
    native(
        &session,
        &foreign,
        0f32,
        &[1., 2., 3., 4., 5., 6.],
        &[7., 8., 9., 10., 11., 12.],
        f32::NAN,
        f32::from_bits(1),
        f32::from,
    );
    native(
        &session,
        &foreign,
        0f64,
        &[1., 2., 3., 4., 5., 6.],
        &[7., 8., 9., 10., 11., 12.],
        f64::NAN,
        f64::from_bits(1),
        f64::from,
    );
}

#[test]
fn mse_selected_schema_refuses_empty_and_above_compact_bound() {
    let request = tuples()[0];
    for count in [0, 65536] {
        let mut graph = Graph::default();
        graph.set_numerical_mode(request.numerical_mode);
        graph.set_numerical_options(request.numerical_options);
        let input = graph.input([count], PcuScalarType::F64).unwrap();
        let result = graph.mean_squared_error(input, input);
        if count == 0 {
            assert!(result.is_err());
            continue;
        }
        let effect = result.unwrap();
        graph
            .set_value_float_underflow_policy(effect, request.float_underflow)
            .unwrap();
        let source = graph
            .into_selected_program(
                &[effect],
                TensorArithmeticRewritePolicy::Disabled,
                TensorArithmeticCapability::Strict,
                TensorPointwiseGroupingPolicy::Disabled,
            )
            .unwrap();
        assert!(MetalTensorMsePlan::assess_program(&source, request).is_err());
    }
}

#[test]
fn mse_selected_compact_schema_large_extent_full_roles() {
    for scalar in [PcuScalarType::F32, PcuScalarType::F64] {
        for request in tuples() {
            for count in [11, 1024, 65535] {
                for profile in 0..8 {
                    let source = program_with_count(scalar, request, profile, count);
                    let plan = MetalTensorMsePlan::assess_program(&source, request).unwrap();
                    assert_eq!(plan.requirements(), request);
                    assert_eq!(plan.input_element_count(0), Some(count));
                    assert_eq!(
                        plan.element_count(),
                        if matches!(profile, 2 | 3 | 5 | 6 | 7) {
                            count
                        } else {
                            1
                        }
                    );
                }
            }
        }
    }
}
#[test]
#[ignore = "Requires native Metal compact selected MSE large mixed roles, discarded effects and original late faults."]
fn mse_selected_compact_large_full_request_roles_faults_and_old_owners() {
    let session = MetalSession::open(0).unwrap();
    let foreign = MetalSession::open(0).unwrap();
    for count in [11, 1024, 65535] {
        native(
            &session,
            &foreign,
            0f32,
            &vec![1f32; count],
            &vec![2f32; count],
            f32::NAN,
            f32::from_bits(1),
            f32::from,
        );
        native(
            &session,
            &foreign,
            0f64,
            &vec![1f64; count],
            &vec![2f64; count],
            f64::NAN,
            f64::from_bits(1),
            f64::from,
        );
    }
}
