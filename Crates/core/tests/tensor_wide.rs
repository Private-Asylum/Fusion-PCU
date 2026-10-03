#![cfg(feature = "tensor")]
//! Wide references retain exact type/range contracts without enabling a device provider.
#[rustfmt::skip]
use fusion_pcu_core::{
    PcuCheckedInteger,
    PcuExecutionFaultKind,
    PcuF128Bits,
    PcuF256Bits,
    PcuI256,
    PcuI512,
    PcuU256,
    PcuU512,
    dialect::tensor::{
        Graph,
        Tensor,
        TensorCheckedReferenceAssessor,
        TensorElement,
        TensorError,
        TensorOperationAssessor,
        TensorOperationSupport,
        TensorValue,
    },
};

fn value<T: TensorElement>(values: [T; 2]) -> TensorValue {
    TensorValue::from_tensor(Tensor::new([2], values.to_vec()).unwrap())
}

fn arithmetic<T: TensorElement + PcuCheckedInteger + PartialEq + core::fmt::Debug>(
    one: T,
    two: T,
    three: T,
    four: T,
    minimum: T,
    maximum: T,
) {
    let mut graph = Graph::try_new().unwrap();
    let left = graph.input_typed::<T>([2]).unwrap();
    let right = graph.input_typed::<T>([2]).unwrap();
    let sum = graph.add_typed(left, right).unwrap();
    let difference = graph.sub_typed(left, right).unwrap();
    let product = graph.mul_typed(left, right).unwrap();
    for node in graph.nodes() {
        assert!(matches!(
            TensorCheckedReferenceAssessor.assess_node(&graph, node),
            TensorOperationSupport::Supported { .. }
        ));
    }
    let execution = graph
        .evaluate_checked(&[
            (left.erase(), value([two, three])),
            (right.erase(), value([one, one])),
        ])
        .unwrap();
    assert_eq!(
        execution.value_typed::<T>(sum.erase()).unwrap().data(),
        [three, four]
    );
    assert_eq!(
        execution
            .value_typed::<T>(difference.erase())
            .unwrap()
            .data(),
        [one, two]
    );
    assert_eq!(
        execution.value_typed::<T>(product.erase()).unwrap().data(),
        [two, three]
    );
    // Independent branches report their first logical lane; no partial execution is returned.
    for (a, b, expected_kind) in [
        (
            [one, maximum],
            [one, one],
            PcuExecutionFaultKind::ArithmeticOverflow,
        ),
        (
            [one, minimum],
            [one, one],
            PcuExecutionFaultKind::ArithmeticUnderflow,
        ),
    ] {
        let mut failing = Graph::try_new().unwrap();
        let a_id = failing.input_typed::<T>([2]).unwrap();
        let b_id = failing.input_typed::<T>([2]).unwrap();
        let output = if expected_kind == PcuExecutionFaultKind::ArithmeticOverflow {
            failing.add_typed(a_id, b_id).unwrap()
        } else {
            failing.sub_typed(a_id, b_id).unwrap()
        };
        assert!(matches!(failing.evaluate_checked(&[
            (a_id.erase(), value(a)), (b_id.erase(), value(b)),
        ]), Err(TensorError::ArithmeticFault { value, element_index: 1, kind })
            if value == output.erase() && kind == expected_kind));
        let retry = failing
            .evaluate_checked(&[
                (a_id.erase(), value([two; 2])),
                (b_id.erase(), value([one; 2])),
            ])
            .unwrap();
        assert_eq!(
            retry.value_typed::<T>(output.erase()).unwrap().data(),
            if expected_kind == PcuExecutionFaultKind::ArithmeticOverflow {
                [three; 2]
            } else {
                [one; 2]
            }
        );
    }
    // This dialect has no implicit integer-division or floating accumulation contract.
    assert!(matches!(
        graph.div_typed(left, right),
        Err(TensorError::UnsupportedScalarType { .. })
    ));
    assert!(matches!(
        graph.sgd_update(left.erase(), right.erase(), 0.5),
        Err(TensorError::UnsupportedScalarType { .. })
    ));
}

#[test]
fn native128_and_limb256_512_arithmetic_preserve_range_and_retry() {
    arithmetic(1_u128, 2, 3, 4, u128::MIN, u128::MAX);
    arithmetic(1_i128, 2, 3, 4, i128::MIN, i128::MAX);
    arithmetic(
        PcuU256::from_limbs_le([1, 0, 0, 0]),
        PcuU256::from_limbs_le([2, 0, 0, 0]),
        PcuU256::from_limbs_le([3, 0, 0, 0]),
        PcuU256::from_limbs_le([4, 0, 0, 0]),
        PcuU256::ZERO,
        PcuU256::MAX,
    );
    arithmetic(
        PcuI256::from_limbs_le([1, 0, 0, 0]),
        PcuI256::from_limbs_le([2, 0, 0, 0]),
        PcuI256::from_limbs_le([3, 0, 0, 0]),
        PcuI256::from_limbs_le([4, 0, 0, 0]),
        PcuI256::MIN,
        PcuI256::MAX,
    );
    arithmetic(
        PcuU512::from_limbs_le([1, 0, 0, 0, 0, 0, 0, 0]),
        PcuU512::from_limbs_le([2, 0, 0, 0, 0, 0, 0, 0]),
        PcuU512::from_limbs_le([3, 0, 0, 0, 0, 0, 0, 0]),
        PcuU512::from_limbs_le([4, 0, 0, 0, 0, 0, 0, 0]),
        PcuU512::ZERO,
        PcuU512::MAX,
    );
    arithmetic(
        PcuI512::from_limbs_le([1, 0, 0, 0, 0, 0, 0, 0]),
        PcuI512::from_limbs_le([2, 0, 0, 0, 0, 0, 0, 0]),
        PcuI512::from_limbs_le([3, 0, 0, 0, 0, 0, 0, 0]),
        PcuI512::from_limbs_le([4, 0, 0, 0, 0, 0, 0, 0]),
        PcuI512::MIN,
        PcuI512::MAX,
    );
}

#[test]
fn limb_products_and_carries_remain_wide_in_tensor_execution() {
    let mut graph = Graph::try_new().unwrap();
    let left = graph.input_typed::<PcuU512>([2]).unwrap();
    let right = graph.input_typed::<PcuU512>([2]).unwrap();
    let sum = graph.add_typed(left, right).unwrap();
    let product = graph.mul_typed(left, right).unwrap();
    let execution = graph
        .evaluate_checked(&[
            (
                left.erase(),
                value([
                    PcuU512::from_limbs_le([u64::MAX, u64::MAX, 0, 0, 0, 0, 0, 0]),
                    PcuU512::from_limbs_le([0, 0, 0, 1, 0, 0, 0, 0]),
                ]),
            ),
            (
                right.erase(),
                value([
                    PcuU512::from_limbs_le([1, 0, 0, 0, 0, 0, 0, 0]),
                    PcuU512::from_limbs_le([0, 0, 0, 0, 1, 0, 0, 0]),
                ]),
            ),
        ])
        .unwrap();
    assert_eq!(
        execution
            .value_typed::<PcuU512>(sum.erase())
            .unwrap()
            .data()[0],
        PcuU512::from_limbs_le([0, 0, 1, 0, 0, 0, 0, 0])
    );
    assert_eq!(
        execution
            .value_typed::<PcuU512>(product.erase())
            .unwrap()
            .data()[1],
        PcuU512::from_limbs_le([0, 0, 0, 0, 0, 0, 0, 1])
    );
}

fn float_storage_only<T: TensorElement>() {
    let mut graph = Graph::try_new().unwrap();
    let left = graph.input_typed::<T>([1]).unwrap();
    let right = graph.input_typed::<T>([1]).unwrap();
    assert!(matches!(
        graph.add_typed(left, right),
        Err(TensorError::UnsupportedScalarType { .. })
    ));
    assert!(matches!(
        graph.div_typed(left, right),
        Err(TensorError::UnsupportedScalarType { .. })
    ));
    assert!(matches!(
        graph.relu_typed(left),
        Err(TensorError::UnsupportedScalarType { .. })
    ));
}

#[test]
fn wide_float_transport_does_not_admit_unimplemented_arithmetic() {
    float_storage_only::<PcuF128Bits>();
    float_storage_only::<PcuF256Bits>();
}
