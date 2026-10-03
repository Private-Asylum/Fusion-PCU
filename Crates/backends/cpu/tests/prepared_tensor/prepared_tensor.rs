//! Selected checked tensor parity, fault ordering, storage reuse and publication contracts.
#![allow(clippy::float_cmp)] // Exact integer-valued sentinels and results, never approximate comparisons.
#[rustfmt::skip]
use fusion_pcu_core::{
    PcuCompoundArithmeticPolicy,
    PcuExecutionFaultKind,
    PcuNumericalMode,
    PcuNumericalOptions,
    PcuScalar,
    PcuScalarType,
    PcuReproducibility,
};
#[rustfmt::skip]
use fusion_pcu_core::dialect::tensor::{
    Graph,
    Tensor,
    TensorArithmeticStep,
    TensorError,
    TensorValue,
};
use fusion_pcu_cpu::PcuCpuPreparedTensorGraph;
#[path = "allocation/allocation.rs"]
mod allocation;

macro_rules! training {
    ($name:ident, $ty:ty, $variant:ident) => {
        #[test]
        fn $name() {
            let mut graph = Graph::default();
            graph.set_numerical_mode(PcuNumericalMode::Strict);
            let input = graph.input([2, 2], <$ty>::TYPE).unwrap();
            let weights = graph.input([2, 2], <$ty>::TYPE).unwrap();
            let target = graph.input([2, 2], <$ty>::TYPE).unwrap();
            let product = graph.matmul(input, weights).unwrap();
            let prediction = graph.relu(product).unwrap();
            let loss = graph.mean_squared_error(prediction, target).unwrap();
            let gradients = graph.backward_mse(loss).unwrap();
            let weight_gradient =
                gradients[graph.execution_plan().index_of(weights).unwrap()].unwrap();
            let updated = graph.sgd_update(weights, weight_gradient, 0.125).unwrap();
            let outputs = [loss, weight_gradient, updated];
            let mut prepared = PcuCpuPreparedTensorGraph::<$ty>::prepare(&graph, &outputs).unwrap();
            assert!(prepared.scratch_slot_count() > 0);
            let x: [$ty; 4] = [1.0, -2.0, 3.0, 4.0];
            let w: [$ty; 4] = [0.5, -0.25, 1.0, 0.75];
            let y: [$ty; 4] = [0.0, 1.0, 2.0, -1.0];
            let reference = graph
                .evaluate_checked(&[
                    (
                        input,
                        TensorValue::$variant(Tensor::new([2, 2], x.to_vec()).unwrap()),
                    ),
                    (
                        weights,
                        TensorValue::$variant(Tensor::new([2, 2], w.to_vec()).unwrap()),
                    ),
                    (
                        target,
                        TensorValue::$variant(Tensor::new([2, 2], y.to_vec()).unwrap()),
                    ),
                ])
                .unwrap();
            let mut loss_output = [99.0 as $ty; 2];
            let mut gradient_output = [99.0 as $ty; 5];
            let mut updated_output = [99.0 as $ty; 5];
            let count = allocation::count(|| {
                prepared
                    .call(
                        &[&x, &w, &y],
                        &mut [&mut loss_output, &mut gradient_output, &mut updated_output],
                    )
                    .unwrap()
            });
            assert_eq!(count, 0, "warm tensor executor allocated");
            for (index, output) in outputs.iter().enumerate() {
                let expected = reference.value_typed::<$ty>(*output).unwrap();
                assert_eq!(
                    prepared
                        .output(index)
                        .unwrap()
                        .iter()
                        .map(|v| v.to_bits())
                        .collect::<Vec<_>>(),
                    expected
                        .data()
                        .iter()
                        .map(|v| v.to_bits())
                        .collect::<Vec<_>>()
                );
            }
            assert_eq!(loss_output[1], 99.0);
            assert_eq!(gradient_output[4], 99.0);
            assert_eq!(updated_output[4], 99.0);
            let bad = [<$ty>::NAN, x[1], x[2], x[3]];
            let before = (loss_output, gradient_output, updated_output);
            assert!(
                prepared
                    .call(
                        &[&bad, &w, &y],
                        &mut [&mut loss_output, &mut gradient_output, &mut updated_output]
                    )
                    .is_err()
            );
            assert_eq!((loss_output, gradient_output, updated_output), before);
            assert!(prepared.output(0).is_none());
            prepared.execute(&[&x, &w, &y]).unwrap();
            assert!(prepared.output(2).is_some());
        }
    };
}
training!(f32_forward_loss_backward_sgd, f32, F32);
training!(f64_forward_loss_backward_sgd, f64, F64);

macro_rules! transpose {
    ($name:ident, $ty:ty, $variant:ident) => {
        #[test]
        fn $name() {
            for transpose_left in [false, true] {
                for transpose_right in [false, true] {
                    let mut graph = Graph::default();
                    graph.set_numerical_mode(PcuNumericalMode::Strict);
                    let left_shape = if transpose_left { [3, 2] } else { [2, 3] };
                    let right_shape = if transpose_right { [4, 3] } else { [3, 4] };
                    let left = graph.input(left_shape, <$ty>::TYPE).unwrap();
                    let right = graph.input(right_shape, <$ty>::TYPE).unwrap();
                    let output = graph
                        .matmul_transposed(left, right, transpose_left, transpose_right)
                        .unwrap();
                    let mut prepared =
                        PcuCpuPreparedTensorGraph::<$ty>::prepare(&graph, &[output]).unwrap();
                    let l: [$ty; 6] = [1.0, -2.0, 4.0, 8.0, 2.0, -0.5];
                    let r: [$ty; 12] = [
                        0.25, 2.0, -1.0, 0.5, 4.0, 2.0, 1.0, -1.0, 0.5, 0.25, 0.125, 8.0,
                    ];
                    let reference = graph
                        .evaluate_checked(&[
                            (
                                left,
                                TensorValue::$variant(Tensor::new(left_shape, l.to_vec()).unwrap()),
                            ),
                            (
                                right,
                                TensorValue::$variant(
                                    Tensor::new(right_shape, r.to_vec()).unwrap(),
                                ),
                            ),
                        ])
                        .unwrap();
                    let actual = prepared.execute_owned(&[&l, &r]).unwrap();
                    assert_eq!(actual, *reference.value_typed::<$ty>(output).unwrap());
                }
            }
        }
    };
}
transpose!(f32_all_transpose_combinations, f32, F32);
transpose!(f64_all_transpose_combinations, f64, F64);

#[test]
fn elementwise_repeated_operands_constants_and_reuse() {
    let mut graph = Graph::default();
    let input = graph.input_typed::<f64>([4]).unwrap();
    let two = graph.uniform_typed([4], 2.0_f64).unwrap();
    let a = graph.add_typed(input, input).unwrap();
    let b = graph.sub_typed(a, two).unwrap();
    let c = graph.mul_typed(b, two).unwrap();
    let d = graph.div_typed(c, two).unwrap();
    let output = graph.relu_typed(d).unwrap();
    let mut prepared =
        PcuCpuPreparedTensorGraph::<f64>::prepare(&graph, &[output.erase()]).unwrap();
    assert_eq!(prepared.scratch_slot_count(), 2);
    let input = [-1.0, -0.0, 2.0, 8.0];
    let mut output = [99.0; 5];
    assert_eq!(
        allocation::count(|| prepared.call(&[&input], &mut [&mut output]).unwrap()),
        0
    );
    assert_eq!(output, [0.0, 0.0, 2.0, 14.0, 99.0]);
}

#[test]
fn matmul_first_multiply_fault_is_exact_and_transactional() {
    let mut graph = Graph::default();
    graph.set_numerical_mode(PcuNumericalMode::Strict);
    let left = graph.input([1, 2], PcuScalarType::F32).unwrap();
    let right = graph.input([2, 2], PcuScalarType::F32).unwrap();
    let output = graph.matmul(left, right).unwrap();
    let mut prepared = PcuCpuPreparedTensorGraph::<f32>::prepare(&graph, &[output]).unwrap();
    let mut out = [7.0, 8.0];
    let error = prepared
        .call(
            &[&[f32::MAX, f32::NAN], &[2.0, 1.0, 0.0, f32::NAN]],
            &mut [&mut out],
        )
        .unwrap_err();
    assert_eq!(
        error,
        TensorError::CompoundArithmeticFault {
            value: output,
            element_index: 0,
            reduction_index: 0,
            step: TensorArithmeticStep::Multiply,
            kind: PcuExecutionFaultKind::ArithmeticOverflow
        }
    );
    assert_eq!(out, [7.0, 8.0]);
}

#[test]
fn matmul_add_fault_matches_reference() {
    let mut graph = Graph::default();
    graph.set_numerical_mode(PcuNumericalMode::Strict);
    let left = graph.input([1, 2], PcuScalarType::F64).unwrap();
    let right = graph.input([2, 1], PcuScalarType::F64).unwrap();
    let output = graph.matmul(left, right).unwrap();
    let mut prepared = PcuCpuPreparedTensorGraph::<f64>::prepare(&graph, &[output]).unwrap();
    let error = prepared
        .execute(&[&[f64::MAX, f64::MAX], &[1.0, 1.0]])
        .unwrap_err();
    assert_eq!(
        error,
        TensorError::CompoundArithmeticFault {
            value: output,
            element_index: 0,
            reduction_index: 1,
            step: TensorArithmeticStep::Add,
            kind: PcuExecutionFaultKind::ArithmeticOverflow
        }
    );
}

#[test]
fn mse_first_subtract_fault() {
    let mut graph = Graph::default();
    graph.set_numerical_mode(PcuNumericalMode::Strict);
    let left = graph.input([1], PcuScalarType::F64).unwrap();
    let right = graph.input([1], PcuScalarType::F64).unwrap();
    let output = graph.mean_squared_error(left, right).unwrap();
    let mut prepared = PcuCpuPreparedTensorGraph::<f64>::prepare(&graph, &[output]).unwrap();
    assert_eq!(
        prepared.execute(&[&[f64::MAX], &[-f64::MAX]]).unwrap_err(),
        TensorError::CompoundArithmeticFault {
            value: output,
            element_index: 0,
            reduction_index: 0,
            step: TensorArithmeticStep::Subtract,
            kind: PcuExecutionFaultKind::ArithmeticOverflow
        }
    );
}

#[test]
fn complete_output_schema_precedes_arithmetic_and_preserves_all_outputs() {
    let mut graph = Graph::default();
    let input = graph.input([2], PcuScalarType::F32).unwrap();
    let output = graph.add(input, input).unwrap();
    let mut prepared = PcuCpuPreparedTensorGraph::<f32>::prepare(&graph, &[input, output]).unwrap();
    let mut first = [9.0; 2];
    let mut second = [8.0; 1];
    assert_eq!(
        prepared
            .call(&[&[f32::NAN; 2]], &mut [&mut first, &mut second])
            .unwrap_err(),
        TensorError::DataLength {
            expected: 2,
            actual: 1
        }
    );
    assert_eq!(first, [9.0; 2]);
    assert_eq!(second, [8.0; 1]);
}

#[test]
fn boundary_and_portable_reject_while_strict_permissions_preserve_checks() {
    let mut graph = Graph::default();
    let left = graph.input([1, 1], PcuScalarType::F32).unwrap();
    let right = graph.input([1, 1], PcuScalarType::F32).unwrap();
    let output = graph.matmul(left, right).unwrap();
    assert!(matches!(
        PcuCpuPreparedTensorGraph::<f32>::prepare(&graph, &[output]),
        Err(TensorError::UnsupportedNumericalMode { .. })
    ));
    graph
        .set_value_numerical_mode(output, PcuNumericalMode::Strict)
        .unwrap();
    graph
        .set_value_numerical_options(
            output,
            PcuNumericalOptions {
                compound_arithmetic: PcuCompoundArithmeticPolicy::BackendDefined,
                ..PcuNumericalOptions::default()
            },
        )
        .unwrap();
    let mut prepared = PcuCpuPreparedTensorGraph::<f32>::prepare(&graph, &[output]).unwrap();
    let mut observed = [17.0_f32; 2];
    prepared
        .call(&[&[2.0], &[3.0]], &mut [&mut observed])
        .unwrap();
    assert_eq!(
        observed.map(f32::to_bits),
        [6.0_f32, 17.0].map(f32::to_bits)
    );
    graph
        .set_value_numerical_options(
            output,
            PcuNumericalOptions {
                reproducibility: PcuReproducibility::PortableV1,
                ..PcuNumericalOptions::default()
            },
        )
        .unwrap();
    assert!(matches!(
        PcuCpuPreparedTensorGraph::<f32>::prepare(&graph, &[output]),
        Err(TensorError::UnsupportedNumericalOptions { .. })
    ));
}

#[test]
fn unselected_unsupported_branch_is_not_admitted_or_executed() {
    let mut graph = Graph::default();
    let input = graph.input([2], PcuScalarType::F32).unwrap();
    let output = graph.relu(input).unwrap();
    let _ignored = graph.input([1, 1], PcuScalarType::I32).unwrap();
    let mut prepared = PcuCpuPreparedTensorGraph::<f32>::prepare(&graph, &[output]).unwrap();
    assert_eq!(prepared.input_bindings().len(), 1);
    drop(graph);
    assert_eq!(
        prepared.execute_owned(&[&[-1.0, 2.0]]).unwrap().data(),
        [0.0, 2.0]
    );
}

#[test]
fn disconnected_checked_fault_remains_observable() {
    let mut graph = Graph::default();
    let input = graph.input([1], PcuScalarType::F32).unwrap();
    let output = graph.relu(input).unwrap();
    let disconnected = graph.input([1], PcuScalarType::F32).unwrap();
    let faulting = graph.div(disconnected, disconnected).unwrap();
    let mut prepared = PcuCpuPreparedTensorGraph::<f32>::prepare(&graph, &[output]).unwrap();
    let mut out = [7.0];
    assert_eq!(
        prepared
            .call(&[&[1.0], &[0.0]], &mut [&mut out])
            .unwrap_err(),
        TensorError::ArithmeticFault {
            value: faulting,
            element_index: 0,
            kind: PcuExecutionFaultKind::DivideByZero
        }
    );
    assert_eq!(out, [7.0]);
}

#[test]
fn zero_inner_matmul_has_positive_zero_and_no_input_reads() {
    let mut graph = Graph::default();
    graph.set_numerical_mode(PcuNumericalMode::Strict);
    let left = graph.input([2, 0], PcuScalarType::F64).unwrap();
    let right = graph.input([0, 3], PcuScalarType::F64).unwrap();
    let output = graph.matmul(left, right).unwrap();
    let mut prepared = PcuCpuPreparedTensorGraph::<f64>::prepare(&graph, &[output]).unwrap();
    prepared.execute(&[&[], &[]]).unwrap();
    assert!(
        prepared
            .output(0)
            .unwrap()
            .iter()
            .all(|value| value.to_bits() == 0)
    );
}

#[test]
fn sgd_fails_at_multiply_before_subtraction() {
    let mut graph = Graph::default();
    graph.set_numerical_mode(PcuNumericalMode::Strict);
    let weight = graph.input([2], PcuScalarType::F64).unwrap();
    let gradient = graph.input([2], PcuScalarType::F64).unwrap();
    let output = graph.sgd_update(weight, gradient, 2.0).unwrap();
    let mut prepared = PcuCpuPreparedTensorGraph::<f64>::prepare(&graph, &[output]).unwrap();
    assert_eq!(
        prepared
            .execute(&[&[f64::NAN, 1.0], &[f64::MAX, f64::NAN]])
            .unwrap_err(),
        TensorError::CompoundArithmeticFault {
            value: output,
            element_index: 0,
            reduction_index: 0,
            step: TensorArithmeticStep::Multiply,
            kind: PcuExecutionFaultKind::ArithmeticOverflow
        }
    );
}

#[test]
fn masked_relu_gradient_still_rejects_nonfinite_upstream() {
    let mut graph = Graph::default();
    let input = graph.input([2], PcuScalarType::F32).unwrap();
    let upstream = graph.input([2], PcuScalarType::F32).unwrap();
    let output = graph.relu_backward(input, upstream).unwrap();
    let mut prepared = PcuCpuPreparedTensorGraph::<f32>::prepare(&graph, &[output]).unwrap();
    assert_eq!(
        prepared
            .execute(&[&[-1.0, 0.0], &[f32::NAN, 1.0]])
            .unwrap_err(),
        TensorError::ArithmeticFault {
            value: output,
            element_index: 0,
            kind: PcuExecutionFaultKind::InvalidFloatingOperand
        }
    );
}

#[test]
fn compound_underflow_policy_is_frozen_and_independent() {
    use fusion_pcu_core::PcuFloatUnderflowPolicy;
    let mut graph = Graph::default();
    graph.set_numerical_mode(PcuNumericalMode::Strict);
    let left = graph.input([1, 1], PcuScalarType::F32).unwrap();
    let right = graph.input([1, 1], PcuScalarType::F32).unwrap();
    let output = graph.matmul(left, right).unwrap();
    for policy in [
        PcuFloatUnderflowPolicy::IeeeAfterRounding,
        PcuFloatUnderflowPolicy::RejectSubnormalResult,
        PcuFloatUnderflowPolicy::AllowGradualUnderflow,
    ] {
        graph
            .set_value_float_underflow_policy(output, policy)
            .unwrap();
        let mut prepared = PcuCpuPreparedTensorGraph::<f32>::prepare(&graph, &[output]).unwrap();
        let actual = prepared.execute(&[&[f32::MIN_POSITIVE], &[0.5]]);
        if policy == PcuFloatUnderflowPolicy::RejectSubnormalResult {
            assert!(matches!(
                actual,
                Err(TensorError::CompoundArithmeticFault {
                    kind: PcuExecutionFaultKind::ArithmeticUnderflow,
                    step: TensorArithmeticStep::Multiply,
                    ..
                })
            ));
        } else {
            actual.unwrap();
            assert_eq!(
                prepared.output(0).unwrap()[0].to_bits(),
                (f32::MIN_POSITIVE * 0.5).to_bits()
            );
        }
    }
}

#[test]
fn f16_leaf_identity_is_exact_without_compound_admission() {
    use fusion_pcu_core::PcuF16Bits;
    let mut graph = Graph::default();
    let input = graph.input_typed::<PcuF16Bits>([2]).unwrap();
    let mut plan =
        PcuCpuPreparedTensorGraph::<PcuF16Bits>::prepare(&graph, &[input.erase()]).unwrap();
    let values = [PcuF16Bits::from_bits(0x8000), PcuF16Bits::from_bits(1)];
    plan.execute(&[&values]).unwrap();
    assert_eq!(plan.output(0).unwrap(), &values);
}

#[test]
fn bf16_leaf_identity_is_exact_without_compound_admission() {
    use fusion_pcu_core::PcuBf16Bits;
    let mut graph = Graph::default();
    let input = graph.input_typed::<PcuBf16Bits>([2]).unwrap();
    let mut plan =
        PcuCpuPreparedTensorGraph::<PcuBf16Bits>::prepare(&graph, &[input.erase()]).unwrap();
    let values = [PcuBf16Bits::from_bits(0x8000), PcuBf16Bits::from_bits(1)];
    plan.execute(&[&values]).unwrap();
    assert_eq!(plan.output(0).unwrap(), &values);
}

#[path = "permissions/permissions.rs"]
mod permissions;
