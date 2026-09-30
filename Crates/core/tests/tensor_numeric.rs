#![cfg(feature = "tensor")]

#[rustfmt::skip]
use fusion_pcu_core::{
    PcuBf16Bits,
    PcuF16Bits,
    core::{PcuExecutionFaultKind, PcuScalarType},
    dialect::tensor::{
        Graph,
        Tensor,
        TensorElement,
        TensorError,
        TensorValue,
    },
};

trait SameBits: Copy {
    fn same_bits(left: Self, right: Self) -> bool;
}

macro_rules! same_bits_by_equality {
    ($($ty:ty),+ $(,)?) => {
        $(
        impl SameBits for $ty {
            fn same_bits(left: Self, right: Self) -> bool {
                left == right
            }
        }
        )+
    };
}

same_bits_by_equality!(
    u8,
    u16,
    u32,
    u64,
    i8,
    i16,
    i32,
    i64,
    PcuF16Bits,
    PcuBf16Bits,
);

impl SameBits for f32 {
    fn same_bits(left: Self, right: Self) -> bool {
        left.to_bits() == right.to_bits()
    }
}

impl SameBits for f64 {
    fn same_bits(left: Self, right: Self) -> bool {
        left.to_bits() == right.to_bits()
    }
}

fn assert_values_keep_bits<T: SameBits>(actual: &[T], expected: &[T]) {
    assert_eq!(actual.len(), expected.len());
    for (&actual, &expected) in actual.iter().zip(expected) {
        assert!(T::same_bits(actual, expected), "value bits changed");
    }
}

fn assert_scalar_transport<T>(values: [T; 3], uniform: T)
where
    T: TensorElement + SameBits,
{
    let mut graph = Graph::default();
    let input = graph.input_typed::<T>([3]).unwrap();
    let constant = graph.constant_typed(Tensor::new([3], values.to_vec()).unwrap());
    let uniform_value = graph.uniform_typed([3], uniform).unwrap();
    let execution = graph
        .evaluate(&[(
            input.erase(),
            TensorValue::from_tensor(Tensor::new([3], values.to_vec()).unwrap()),
        )])
        .unwrap();

    for value in [input.erase(), constant.erase()] {
        let tensor = execution.value_typed::<T>(value).unwrap();
        assert_values_keep_bits(tensor.data(), &values);
        assert_eq!(tensor.shape(), &[3]);
    }
    let uniform_tensor = execution.value_typed::<T>(uniform_value.erase()).unwrap();
    assert_values_keep_bits(uniform_tensor.data(), &[uniform; 3]);
    assert_eq!(uniform_tensor.shape(), &[3]);
}

#[test]
fn every_dense_host_scalar_transports_through_inputs_constants_and_uniforms() {
    assert_scalar_transport(
        [
            PcuF16Bits::from_bits(0xfc00),
            PcuF16Bits::from_bits(0x0001),
            PcuF16Bits::from_bits(0x7d55),
        ],
        PcuF16Bits::from_bits(0x8000),
    );
    assert_scalar_transport(
        [
            PcuBf16Bits::from_bits(0xff80),
            PcuBf16Bits::from_bits(0x0001),
            PcuBf16Bits::from_bits(0x7f93),
        ],
        PcuBf16Bits::from_bits(0x8000),
    );
    assert_scalar_transport(
        [f32::NEG_INFINITY, -0.0, f32::from_bits(0x7fc0_1234)],
        f32::from_bits(0x7fa0_5678),
    );
    assert_scalar_transport(
        [
            f64::NEG_INFINITY,
            -0.0,
            f64::from_bits(0x7ff8_0000_0000_1234),
        ],
        f64::from_bits(0x7ff0_0000_0000_5678),
    );
    assert_scalar_transport([u8::MIN, u8::MAX, 0], 17);
    assert_scalar_transport([u16::MIN, u16::MAX, 0], 17);
    assert_scalar_transport([u32::MIN, u32::MAX, 0], 17);
    assert_scalar_transport([u64::MIN, u64::MAX, 0], 17);
    assert_scalar_transport([i8::MIN, i8::MAX, 0], 17);
    assert_scalar_transport([i16::MIN, i16::MAX, 0], 17);
    assert_scalar_transport([i32::MIN, i32::MAX, 0], 17);
    assert_scalar_transport([i64::MIN, i64::MAX, 0], 17);
}

#[test]
fn independent_f32_and_f64_graph_branches_evaluate_without_coercion() {
    let mut graph = Graph::default();
    let f32_input = graph.input_typed::<f32>([2]).unwrap();
    let f64_input = graph.input_typed::<f64>([2]).unwrap();
    let f32_output = graph.relu_typed(f32_input).unwrap();
    let f64_output = graph.add_typed(f64_input, f64_input).unwrap();

    let execution = graph
        .evaluate(&[
            (
                f32_input.erase(),
                TensorValue::from(Tensor::new([2], vec![-2.0_f32, 3.0]).unwrap()),
            ),
            (
                f64_input.erase(),
                TensorValue::from(Tensor::new([2], vec![1.25_f64, -4.5]).unwrap()),
            ),
        ])
        .unwrap();

    assert_eq!(
        execution
            .value_typed::<f32>(f32_output.erase())
            .unwrap()
            .data(),
        &[0.0, 3.0]
    );
    assert_eq!(
        execution
            .value_typed::<f64>(f64_output.erase())
            .unwrap()
            .data(),
        &[2.5, -9.0]
    );
}

#[test]
fn tensor_relu_uses_checked_finite_selection_and_reports_nonfinite_faults() {
    let mut graph = Graph::default();
    let input = graph.input_typed::<f32>([4]).unwrap();
    let output = graph.relu_typed(input).unwrap();
    let execution = graph
        .evaluate(&[(
            input.erase(),
            TensorValue::from(
                Tensor::new(
                    [4],
                    vec![f32::from_bits(1), -0.0, f32::from_bits(0x8000_0001), 2.0],
                )
                .unwrap(),
            ),
        )])
        .unwrap();
    let values = execution.value_typed::<f32>(output.erase()).unwrap();
    assert_eq!(values.data()[0].to_bits(), 1);
    assert_eq!(values.data()[1].to_bits(), 0);
    assert_eq!(values.data()[2].to_bits(), 0);
    assert_eq!(values.data()[3].to_bits(), 2.0_f32.to_bits());

    let mut graph = Graph::default();
    let input = graph.input_typed::<f32>([1]).unwrap();
    let output = graph.relu_typed(input).unwrap();
    assert_eq!(
        graph
            .evaluate(&[(
                input.erase(),
                TensorValue::from(Tensor::new([1], vec![f32::NAN]).unwrap()),
            )])
            .unwrap_err(),
        TensorError::ArithmeticFault {
            value: output.erase(),
            element_index: 0,
            kind: PcuExecutionFaultKind::InvalidFloatingOperand,
        }
    );
}

#[test]
fn f64_transposed_matmul_matches_hand_calculated_row_major_result() {
    let mut graph = Graph::default();
    let left = graph.input_typed::<f64>([3, 2]).unwrap();
    let right = graph.input_typed::<f64>([4, 3]).unwrap();
    let output = graph
        .matmul_transposed_typed(left, right, true, true)
        .unwrap();
    let execution = graph
        .evaluate(&[
            (
                left.erase(),
                TensorValue::from(
                    Tensor::new([3, 2], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
                ),
            ),
            (
                right.erase(),
                TensorValue::from(
                    Tensor::new(
                        [4, 3],
                        vec![
                            1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
                        ],
                    )
                    .unwrap(),
                ),
            ),
        ])
        .unwrap();

    let result = execution.value_typed::<f64>(output.erase()).unwrap();
    assert_eq!(result.shape(), &[2, 4]);
    assert_eq!(
        result.data(),
        &[22.0, 49.0, 76.0, 103.0, 28.0, 64.0, 100.0, 136.0]
    );
}

#[test]
fn f64_reference_arithmetic_preserves_precision_beyond_f32_integer_range() {
    let mut graph = Graph::default();
    let input = graph.input_typed::<f64>([1]).unwrap();
    let zero = graph.uniform_typed([1], 0.0_f64).unwrap();
    let output = graph.add_typed(input, zero).unwrap();
    let exact = 16_777_217.0_f64;
    let execution = graph
        .evaluate(&[(
            input.erase(),
            TensorValue::from(Tensor::new([1], vec![exact]).unwrap()),
        )])
        .unwrap();
    let actual = execution.value_typed::<f64>(output.erase()).unwrap().data()[0];

    assert_eq!(actual.to_bits(), exact.to_bits());
    assert_ne!(actual.to_bits(), f64::from(16_777_216.0_f32).to_bits());
}

#[test]
fn typed_and_dynamic_graph_boundaries_report_dtype_mismatches() {
    let mut graph = Graph::default();
    let f32_value = graph.input_typed::<f32>([1]).unwrap();
    let f64_value = graph.input_typed::<f64>([1]).unwrap();

    assert!(matches!(
        graph.add(f32_value.erase(), f64_value.erase()),
        Err(TensorError::ScalarTypeMismatch {
            value,
            expected: PcuScalarType::F32,
            actual: PcuScalarType::F64,
        }) if value == f64_value.erase()
    ));
    assert!(matches!(
        graph.typed_view::<f32>(f64_value.erase()),
        Err(TensorError::ScalarTypeMismatch {
            value,
            expected: PcuScalarType::F32,
            actual: PcuScalarType::F64,
        }) if value == f64_value.erase()
    ));
    assert!(matches!(
        graph.evaluate(&[(
            f64_value.erase(),
            TensorValue::from(Tensor::new([1], vec![1.0_f32]).unwrap()),
        )]),
        Err(TensorError::ScalarTypeMismatch {
            value,
            expected: PcuScalarType::F64,
            actual: PcuScalarType::F32,
        }) if value == f64_value.erase()
    ));
}

#[test]
fn typed_identity_rejects_a_handle_from_another_graph() {
    let mut first = Graph::default();
    let foreign = first.input_typed::<f32>([1]).unwrap();
    let second = Graph::default();

    assert_eq!(
        second.identity_typed(foreign),
        Err(TensorError::UnknownValue(foreign.erase()))
    );
}

#[test]
fn f32_mse_gradients_ignore_an_unrelated_f64_branch() {
    let mut graph = Graph::default();
    let f64_input = graph.input_typed::<f64>([2]).unwrap();
    let f64_side = graph.relu_typed(f64_input).unwrap();
    let f32_input = graph.input_typed::<f32>([1]).unwrap();
    let target = graph.uniform_typed([1], 0.0_f32).unwrap();
    let loss = graph
        .mean_squared_error(f32_input.erase(), target.erase())
        .unwrap();
    let input_position = graph
        .nodes()
        .position(|node| node.value == f32_input.erase())
        .unwrap();
    let gradient_values = graph.backward_mse(loss).unwrap();
    let input_gradient = gradient_values[input_position].unwrap();

    let execution = graph
        .evaluate(&[
            (
                f64_input.erase(),
                TensorValue::from(Tensor::new([2], vec![-1.0_f64, 2.0]).unwrap()),
            ),
            (
                f32_input.erase(),
                TensorValue::from(Tensor::new([1], vec![3.0_f32]).unwrap()),
            ),
        ])
        .unwrap();

    assert_eq!(
        execution
            .value_typed::<f64>(f64_side.erase())
            .unwrap()
            .data(),
        &[0.0, 2.0]
    );
    assert_eq!(
        execution.value_typed::<f32>(input_gradient).unwrap().data(),
        &[6.0]
    );
}
