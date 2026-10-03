//! Weaker permissions do not require weakening the ordered checked reference.
use alloc::vec;

#[rustfmt::skip]
use crate::{
    PcuCompoundArithmeticPolicy,
    PcuExecutionFaultKind,
    PcuFloatUnderflowPolicy,
    PcuNumericalMode,
    PcuNumericalOptions,
    PcuPrecisionPolicy,
    PcuReproducibility,
};
#[rustfmt::skip]
use crate::dialect::tensor::{
    Graph,
    Tensor,
    TensorArithmeticStep,
    TensorCheckedReferenceAssessor,
    TensorElement,
    TensorError,
};

fn strict(options: PcuNumericalOptions) -> Graph {
    let mut graph = Graph::default();
    graph.set_numerical_mode(PcuNumericalMode::Strict);
    graph.set_numerical_options(options);
    graph
}

fn bits<T: TensorElement>(actual: &[T], expected: &[T]) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert_eq!(actual.encode_le().as_ref(), expected.encode_le().as_ref());
    }
}

fn finite<T: TensorElement>(
    options: PcuNumericalOptions,
    policy: PcuFloatUnderflowPolicy,
    value: impl Fn(f32) -> T + Copy,
) {
    let mut graph = strict(options);
    let left = graph
        .constant_typed(Tensor::new([2, 2], [1.0, 2.0, 3.0, 4.0].map(value).to_vec()).unwrap());
    let right = graph
        .constant_typed(Tensor::new([2, 2], [1.0, 0.0, 0.0, 1.0].map(value).to_vec()).unwrap());
    let zero = graph.constant_typed(Tensor::new([2, 2], vec![value(0.0); 4]).unwrap());
    let product = graph.matmul_typed(left, right).unwrap();
    let loss = graph.mean_squared_error_typed(product, zero).unwrap();
    let update = graph.sgd_update_typed(product, product, 0.5).unwrap();
    for output in [product.erase(), loss.erase(), update.erase()] {
        graph
            .set_value_float_underflow_policy(output, policy)
            .unwrap();
    }
    assert!(
        graph
            .assess_with(&TensorCheckedReferenceAssessor)
            .is_supported()
    );
    let result = graph.evaluate_checked(&[]).unwrap();
    bits(
        result.value_typed::<T>(product.erase()).unwrap().data(),
        &[1.0, 2.0, 3.0, 4.0].map(value),
    );
    bits(
        result.value_typed::<T>(loss.erase()).unwrap().data(),
        &[value(7.5)],
    );
    bits(
        result.value_typed::<T>(update.erase()).unwrap().data(),
        &[0.5, 1.0, 1.5, 2.0].map(value),
    );

    // Permission remains metadata, rather than rewriting the chosen algorithm.
    assert_eq!(
        graph.node(product.erase()).unwrap().numerical_options,
        options
    );
    graph
        .set_value_numerical_mode(product.erase(), PcuNumericalMode::Boundary)
        .unwrap();
    assert!(
        !graph
            .assess_with(&TensorCheckedReferenceAssessor)
            .is_supported()
    );
    assert!(matches!(
        graph.evaluate_checked(&[]),
        Err(TensorError::UnsupportedNumericalMode { .. })
    ));
    graph
        .set_value_numerical_mode(product.erase(), PcuNumericalMode::Strict)
        .unwrap();
    let mut portable = options;
    portable.reproducibility = PcuReproducibility::PortableV1;
    graph
        .set_value_numerical_options(loss.erase(), portable)
        .unwrap();
    assert!(
        !graph
            .assess_with(&TensorCheckedReferenceAssessor)
            .is_supported()
    );
    assert!(matches!(
        graph.evaluate_checked(&[]),
        Err(TensorError::UnsupportedNumericalOptions { .. })
    ));
}

fn tiny<T: TensorElement>(
    options: PcuNumericalOptions,
    policy: PcuFloatUnderflowPolicy,
    minimum: T,
    value: impl Fn(f32) -> T,
) {
    let mut graph = strict(options);
    let left = graph.constant_typed(Tensor::new([1, 2], vec![minimum, value(1.0)]).unwrap());
    let right = graph.constant_typed(Tensor::new([2, 1], vec![value(0.5), value(1.0)]).unwrap());
    let product = graph.matmul_typed(left, right).unwrap();
    graph
        .set_value_float_underflow_policy(product.erase(), policy)
        .unwrap();
    assert!(
        graph
            .assess_with(&TensorCheckedReferenceAssessor)
            .is_supported()
    );
    match policy {
        PcuFloatUnderflowPolicy::AllowGradualUnderflow => {
            let output = graph.evaluate_checked(&[]).unwrap();
            bits(
                output.value_typed::<T>(product.erase()).unwrap().data(),
                &[value(1.0)],
            );
        }
        PcuFloatUnderflowPolicy::IeeeAfterRounding
        | PcuFloatUnderflowPolicy::RejectSubnormalResult => {
            assert!(
                matches!(graph.evaluate_checked(&[]), Err(TensorError::CompoundArithmeticFault {
                value: output,
                element_index: 0,
                reduction_index: 0,
                step: TensorArithmeticStep::Multiply,
                kind: PcuExecutionFaultKind::ArithmeticUnderflow,
            }) if output == product.erase())
            );
        }
    }
}

#[test]
fn ordered_compounds_keep_exact_results_and_faults_under_all_permissions() {
    for compound_arithmetic in [
        PcuCompoundArithmeticPolicy::Checked,
        PcuCompoundArithmeticPolicy::BackendDefined,
    ] {
        for precision in [
            PcuPrecisionPolicy::Preserve,
            PcuPrecisionPolicy::BackendOptimized,
        ] {
            let options = PcuNumericalOptions {
                compound_arithmetic,
                precision,
                ..Default::default()
            };
            for policy in [
                PcuFloatUnderflowPolicy::IeeeAfterRounding,
                PcuFloatUnderflowPolicy::AllowGradualUnderflow,
                PcuFloatUnderflowPolicy::RejectSubnormalResult,
            ] {
                finite::<f32>(options, policy, |v| v);
                finite::<f64>(options, policy, f64::from);
                tiny::<f32>(options, policy, f32::from_bits(1), |v| v);
                tiny::<f64>(options, policy, f64::from_bits(1), f64::from);
            }
        }
    }
}
