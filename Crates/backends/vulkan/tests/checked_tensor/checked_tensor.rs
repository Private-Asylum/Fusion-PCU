//! Actual same-session native pointwise plans; detached graph, faults and owner publication.
extern crate pcu_facade as fusion_pcu;
#[path = "../scalar_transport/sample/sample.rs"]
#[allow(dead_code)] // The bit-only comparison is shared with the existing full transport corpus.
mod bits;
#[path = "../scalar_transport/device/device.rs"]
mod device;
#[path = "faults/faults.rs"]
mod faults;
#[path = "../../../cpu/tests/wide_integer/oracle/oracle.rs"]
#[allow(dead_code)] // Arithmetic helpers serve the full independent CPU integer corpus too.
mod integer_oracle;
#[path = "../../../rocm/benches/low_tensor/oracle/oracle.rs"]
mod low_oracle;
#[path = "profiles/profiles.rs"]
mod profiles;
#[path = "sample/sample.rs"]
mod sample;
#[path = "source/source.rs"]
mod source;
#[path = "../../../rocm/benches/wide_integer/oracle/oracle.rs"]
#[allow(dead_code)]
// The golden decoder is reused without relabeling the original oracle provenance.
mod wide_oracle;
#[rustfmt::skip]
use pcu_facade::{
    PcuScalar,
    PcuI512,
    PcuExecutionFaultKind,
    PcuNumericalMode,
    PcuNumericalOptions,
    PcuCompoundArithmeticPolicy,
    PcuPrecisionPolicy,
};
#[rustfmt::skip]
use pcu_facade::dialect::tensor::{
    Graph,
    Tensor,
    TensorError,
};
#[rustfmt::skip]
use fusion_pcu_vulkan::{
    PcuVulkanPreparedTensorGraph,
    PcuVulkanTensorInput,
    PcuVulkanTensorError,
};

#[test]
#[ignore = "requires an actual Vulkan compute GPU"]
fn native_f32_chain_borrowed_inputs_effect_fault_and_retry() {
    let (backend, _) = device::selected();
    let mut graph = Graph::default();
    graph.set_numerical_mode(PcuNumericalMode::Strict);
    graph.set_numerical_options(PcuNumericalOptions {
        compound_arithmetic: PcuCompoundArithmeticPolicy::BackendDefined,
        precision: PcuPrecisionPolicy::BackendOptimized,
        ..Default::default()
    });
    let input = graph.input([5, 13], f32::TYPE).unwrap();
    let constant = graph.constant_typed(Tensor::new([5, 13], vec![2.0_f32; 65]).unwrap());
    let sum = graph.add(input, constant.erase()).unwrap();
    let output = graph.relu(sum).unwrap();
    let mut plan =
        PcuVulkanPreparedTensorGraph::<f32>::prepare(&backend, &graph, &[output]).unwrap();
    drop(graph);
    let values: Vec<_> = (0_i16..65).map(|index| f32::from(index - 20)).collect();
    let owner = backend.upload_owned(&values).unwrap();
    let mut observed = [117.0_f32; 68];
    for input in [
        PcuVulkanTensorInput::Host(&values),
        PcuVulkanTensorInput::Owned(&owner),
    ] {
        let result = plan.execute_owned(&[input]).unwrap();
        result.read_into(&mut observed).unwrap();
        for (actual, value) in observed[..65].iter().zip(&values) {
            assert_eq!(actual.to_bits(), (value + 2.0).max(0.0).to_bits());
        }
        assert_eq!(observed[65..], [117.0; 3]);
    }
    assert!(
        plan.execute_owned(&[PcuVulkanTensorInput::Host(&values[..64])])
            .is_err()
    );
    owner.read_into(&mut observed).unwrap();
    assert_eq!(&observed[..65], &values);

    let mut graph = Graph::default();
    let a = graph.input([65], f32::TYPE).unwrap();
    let b = graph.input([65], f32::TYPE).unwrap();
    let checked_unused = graph.div(a, b).unwrap();
    let mut effect = PcuVulkanPreparedTensorGraph::<f32>::prepare(&backend, &graph, &[a]).unwrap();
    let mut right = [2.0; 65];
    right[7] = 0.0;
    assert!(
        matches!(effect.execute_owned(&[PcuVulkanTensorInput::Owned(&owner), PcuVulkanTensorInput::Host(&right)]),
        Err(PcuVulkanTensorError::Graph(TensorError::ArithmeticFault {
            value, element_index: 7, kind: PcuExecutionFaultKind::DivideByZero,
        })) if value == checked_unused)
    );
    owner.read_into(&mut observed).unwrap();
    assert_eq!(&observed[..65], &values);
    right[7] = 2.0;
    effect
        .execute_owned(&[
            PcuVulkanTensorInput::Owned(&owner),
            PcuVulkanTensorInput::Host(&right),
        ])
        .unwrap()
        .read_into(&mut observed)
        .unwrap();
    assert_eq!(&observed[..65], &values);
}

#[test]
#[ignore = "requires an actual Vulkan compute GPU"]
fn native_i512_chain_repeated_operands_fatal_rollback_and_retry() {
    let (backend, _) = device::selected();
    let one = PcuI512::from_limbs_le([1, 0, 0, 0, 0, 0, 0, 0]);
    let two = PcuI512::from_limbs_le([2, 0, 0, 0, 0, 0, 0, 0]);
    let four = PcuI512::from_limbs_le([4, 0, 0, 0, 0, 0, 0, 0]);
    let mut graph = Graph::default();
    let input = graph.input([65], PcuI512::TYPE).unwrap();
    let uniform = graph.uniform_typed([65], one).unwrap();
    let sum = graph.add(input, uniform.erase()).unwrap();
    let product = graph.mul(sum, sum).unwrap();
    let mut plan =
        PcuVulkanPreparedTensorGraph::<PcuI512>::prepare(&backend, &graph, &[product]).unwrap();
    drop(graph);
    let values = [one; 65];
    let owner = backend.upload_owned(&values).unwrap();
    let escaped = plan
        .execute_owned(&[PcuVulkanTensorInput::Owned(&owner)])
        .unwrap();
    let mut observed = [two; 68];
    escaped.read_into(&mut observed).unwrap();
    assert_eq!(observed[..65], [four; 65]);
    assert_eq!(observed[65..], [two; 3]);
    let mut bad = values;
    bad[3] = PcuI512::MAX;
    assert!(
        matches!(plan.execute_owned(&[PcuVulkanTensorInput::Host(&bad)]),
        Err(PcuVulkanTensorError::Graph(TensorError::ArithmeticFault {
            value, element_index: 3, kind: PcuExecutionFaultKind::ArithmeticOverflow,
        })) if value == sum)
    );
    escaped.read_into(&mut observed).unwrap();
    assert_eq!(observed[..65], [four; 65]);
    plan.execute_owned(&[PcuVulkanTensorInput::Owned(&owner)])
        .unwrap()
        .read_into(&mut observed)
        .unwrap();
    assert_eq!(observed[..65], [four; 65]);
}

#[test]
fn cold_pointwise_assessment_has_exact_type_operation_and_portable_boundary() {
    let mut graph = Graph::default();
    let input = graph.input([65], f64::TYPE).unwrap();
    let output = graph.relu(input).unwrap();
    PcuVulkanPreparedTensorGraph::<f64>::assess(&graph, &[output]).unwrap();
    assert!(PcuVulkanPreparedTensorGraph::<f32>::assess(&graph, &[output]).is_err());
    let mut graph = Graph::default();
    graph.set_numerical_options(PcuNumericalOptions {
        reproducibility: pcu_facade::PcuReproducibility::PortableV1,
        ..Default::default()
    });
    let input = graph.input([65], pcu_facade::PcuF16Bits::TYPE).unwrap();
    let output = graph.relu(input).unwrap();
    assert!(
        PcuVulkanPreparedTensorGraph::<pcu_facade::PcuF16Bits>::assess(&graph, &[output]).is_err()
    );
    let mut graph = Graph::default();
    let input = graph.input([65], pcu_facade::PcuF256Bits::TYPE).unwrap();
    PcuVulkanPreparedTensorGraph::<pcu_facade::PcuF256Bits>::assess(&graph, &[input]).unwrap();
    assert!(graph.add(input, input).is_err());
    let mut graph = Graph::default();
    let input = graph.input([2, 2], f64::TYPE).unwrap();
    let output = graph.matmul(input, input).unwrap();
    assert!(PcuVulkanPreparedTensorGraph::<f64>::assess(&graph, &[output]).is_err());
    let mut graph = Graph::default();
    let input = graph.input([65], f64::TYPE).unwrap();
    let output = graph.relu_backward(input, input).unwrap();
    PcuVulkanPreparedTensorGraph::<f64>::assess(&graph, &[output]).unwrap();
}
