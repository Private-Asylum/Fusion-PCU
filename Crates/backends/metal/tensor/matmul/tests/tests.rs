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

fn program(
    scalar: PcuScalarType,
    request: PcuImplementationRequirements,
    profile: u8,
) -> Arc<TensorOwnedSelectedProgram> {
    let mut graph = Graph::default();
    graph.set_numerical_mode(request.numerical_mode);
    graph.set_numerical_options(request.numerical_options);
    let left = graph
        .input(if profile == 0 { [2, 2] } else { [2, 3] }, scalar)
        .unwrap();
    let right = if profile == 0 {
        left
    } else {
        graph.input([3, 2], scalar).unwrap()
    };
    let effect = graph.matmul(left, right).unwrap();
    graph
        .set_value_float_underflow_policy(effect, request.float_underflow)
        .unwrap();
    let output = match profile {
        2 => left,
        3 => right,
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
fn matmul_selected_schema_retains_shapes_full_request_and_discarded_effect() {
    for scalar in [PcuScalarType::F32, PcuScalarType::F64] {
        for request in tuples() {
            for profile in 0..4 {
                let source = program(scalar, request, profile);
                let plan = MetalTensorMatMulPlan::assess_program(&source, request).unwrap();
                assert_eq!(plan.requirements(), request);
                assert_eq!(
                    plan.dimensions(),
                    if profile == 0 { [2, 2, 2] } else { [2, 3, 2] }
                );
                assert_eq!(
                    plan.input_element_count(0),
                    Some(if profile == 0 { 4 } else { 6 })
                );
                assert_eq!(plan.element_count(), if profile < 2 { 4 } else { 6 });
                assert_eq!(
                    plan.shape(),
                    if profile == 2 {
                        &[2, 3]
                    } else if profile == 3 {
                        &[3, 2]
                    } else {
                        &[2, 2]
                    }
                );
                let mut other = request;
                other.numerical_mode = PcuNumericalMode::Boundary;
                assert!(MetalTensorMatMulPlan::assess_program(&source, other).is_err());
                other = request;
                other.range_policy = PcuRangePolicy::Clamp;
                assert!(MetalTensorMatMulPlan::assess_program(&source, other).is_err());
                other = request;
                other.numerical_options.compound_arithmetic =
                    if request.numerical_options.compound_arithmetic
                        == PcuCompoundArithmeticPolicy::Checked
                    {
                        PcuCompoundArithmeticPolicy::BackendDefined
                    } else {
                        PcuCompoundArithmeticPolicy::Checked
                    };
                assert!(MetalTensorMatMulPlan::assess_program(&source, other).is_err());
                other = request;
                other.float_underflow =
                    if request.float_underflow == PcuFloatUnderflowPolicy::RejectSubnormalResult {
                        PcuFloatUnderflowPolicy::IeeeAfterRounding
                    } else {
                        PcuFloatUnderflowPolicy::RejectSubnormalResult
                    };
                assert!(MetalTensorMatMulPlan::assess_program(&source, other).is_err());
                other = request;
                other.numerical_options.precision =
                    if request.numerical_options.precision == PcuPrecisionPolicy::Preserve {
                        PcuPrecisionPolicy::BackendOptimized
                    } else {
                        PcuPrecisionPolicy::Preserve
                    };
                assert!(MetalTensorMatMulPlan::assess_program(&source, other).is_err());
                other = request;
                other.numerical_options.reproducibility = PcuReproducibility::PortableV1;
                assert!(MetalTensorMatMulPlan::assess_program(&source, other).is_err());
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
    left: &[T],
    right: &[T],
    zero: T,
    inner: usize,
    policy: PcuFloatUnderflowPolicy,
) -> Result<Vec<T>, PcuExecutionFault> {
    let mut output = Vec::new();
    for cell in 0..4 {
        let mut accumulator = zero;
        for reduction in 0..inner {
            let ordinal = u64::try_from((cell * inner + reduction) * 2).unwrap();
            let fault = |kind, invocation_id| PcuExecutionFault {
                kind,
                invocation_id,
                recovered: false,
            };
            let product = left[cell / 2 * inner + reduction]
                .pcu_checked_mul_with_policy(right[reduction * 2 + cell % 2], policy)
                .map_err(|kind| fault(kind, ordinal))?;
            accumulator = accumulator
                .pcu_checked_add_with_policy(product, policy)
                .map_err(|kind| fault(kind, ordinal + 1))?;
        }
        output.push(accumulator);
    }
    Ok(output)
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
    prepared: &MetalPreparedTensorMatMulProgram,
    profile: u8,
    left: &[T],
    right: &[T],
    values: [T; 3],
) {
    let [zero, invalid, tiny] = values;
    let count = left.len();
    for value in [invalid, tiny] {
        let mut changed = left.to_vec();
        changed[0] = value;
        let changed_right = if profile == 0 {
            changed.as_slice()
        } else {
            right
        };
        let expected = oracle(
            &changed,
            changed_right,
            zero,
            prepared.plan().dimensions()[1],
            prepared.plan().requirements().float_underflow,
        );
        let a_bytes = bytes(&changed);
        let b_bytes = bytes(changed_right);
        let input = |bytes| MetalTensorInput::HostBytes {
            scalar: T::TYPE,
            elements: count,
            bytes,
        };
        let inputs = if profile == 0 {
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
                let expected = if profile == 2 {
                    changed.clone()
                } else if profile == 3 {
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
fn native<T: PcuCheckedFloat>(
    session: &MetalSession,
    foreign: &MetalSession,
    zero: T,
    values: &[T; 6],
    upstream: &[T; 6],
    invalid: T,
    tiny: T,
) {
    for request in tuples() {
        for profile in 0..4 {
            let count = if profile == 0 { 4 } else { 6 };
            let left = &values[..count];
            let right = if profile == 0 {
                left
            } else {
                upstream.as_slice()
            };
            let source = program(T::TYPE, request, profile);
            let prepared = session
                .prepare_tensor_matmul_program(Arc::clone(&source), request, PcuMemoryPoolId(144))
                .unwrap();
            assert!(std::ptr::eq(prepared.program(), source.as_ref()));
            let a = owner(session, left);
            let b = owner(session, right);
            let wrong = owner(foreign, right);
            let left_bytes = bytes(left);
            let right_bytes = bytes(right);
            for layout in 0..4 {
                let inputs = if profile == 0 {
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
                let result = prepared.execute(&inputs).unwrap();
                assert_eq!(result.shape(), prepared.plan().shape());
                let expected = if profile == 2 {
                    left.to_vec()
                } else if profile == 3 {
                    right.to_vec()
                } else {
                    oracle(
                        left,
                        right,
                        zero,
                        prepared.plan().dimensions()[1],
                        request.float_underflow,
                    )
                    .unwrap()
                };
                let mut actual = vec![zero; expected.len() + 2];
                result.read_into(&mut actual).unwrap();
                assert_eq!(bytes(&actual[..expected.len()]), bytes(&expected));
                assert_eq!(bytes(&actual[expected.len()..]), bytes(&[zero; 2]));
            }
            fault_cases(&prepared, profile, left, right, [zero, invalid, tiny]);

            let inputs = if profile == 0 {
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
            let mut old = vec![zero; count];
            a.read_into(&mut old).unwrap();
            assert_eq!(bytes(&old), bytes(left));
            b.read_into(&mut old).unwrap();
            assert_eq!(bytes(&old), bytes(right));
        }
    }
}
#[test]
#[ignore = "Requires native Metal selected Strict MatMul ownership and exact event arbitration."]
fn matmul_selected_graph_host_resident_full_request_discarded_faults_and_old_owners() {
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
    );
    native(
        &session,
        &foreign,
        0f64,
        &[1., 2., 3., 4., 5., 6.],
        &[7., 8., 9., 10., 11., 12.],
        f64::NAN,
        f64::from_bits(1),
    );
}
