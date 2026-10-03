//! Detached selected leaf plans on actual Vulkan native ownership.
#[path = "../scalar_transport/device/device.rs"]
mod device;
#[path = "../scalar_transport/sample/sample.rs"]
mod sample;
#[rustfmt::skip]
use pcu_facade::{
    PcuBf16Bits,
    PcuF16Bits,
    PcuF128Bits,
    PcuF256Bits,
    PcuF8E4M3FnBits,
    PcuF8E5M2Bits,
    PcuI256,
    PcuI512,
    PcuU256,
    PcuU512,
    PcuNumericalOptions,
    PcuReproducibility,
};
#[rustfmt::skip]
use pcu_facade::dialect::tensor::{Graph,Tensor,TensorElement,TensorError};
#[rustfmt::skip]
use fusion_pcu_vulkan::{
    PcuVulkanBackend,
    PcuVulkanPreparedScalarTensorGraph,
    PcuVulkanTensorError,
    PcuVulkanTensorInput,
};
use sample::{same, Sample};

fn carrier<T: Sample + TensorElement>(backend: &PcuVulkanBackend) {
    const N: usize = 65;
    let mut graph = Graph::default();
    let input = graph.input([N], T::TYPE).unwrap();
    let constants: Vec<T> = (0..N).map(T::pattern).collect();
    let constant = graph.constant_typed(Tensor::new([N], constants.clone()).unwrap());
    let uniform = T::pattern(1);
    let uniform_value = graph.uniform_typed([N], uniform).unwrap();
    let mut input_plan =
        PcuVulkanPreparedScalarTensorGraph::<T>::prepare(backend, &graph, &[input]).unwrap();
    let mut constant_plan =
        PcuVulkanPreparedScalarTensorGraph::<T>::prepare(backend, &graph, &[constant.erase()])
            .unwrap();
    let mut uniform_plan =
        PcuVulkanPreparedScalarTensorGraph::<T>::prepare(backend, &graph, &[uniform_value.erase()])
            .unwrap();
    drop(graph);
    assert_eq!(input_plan.input_bindings().len(), 1);
    assert_eq!(constant_plan.input_bindings().len(), 0);
    assert_eq!(uniform_plan.input_bindings().len(), 0);
    assert_eq!(input_plan.output_shape().as_ref(), &[N]);
    let sentinel = T::pattern(17);
    let mut observed = [sentinel; N + 3];
    let mut values = [sentinel; N];
    for phase in 0..64 {
        for (index, value) in values.iter_mut().enumerate() {
            *value = T::pattern(phase * 79 + index);
        }
        let first = input_plan
            .execute_owned(&[PcuVulkanTensorInput::Host(&values)])
            .unwrap();
        let second = input_plan
            .execute_owned(&[PcuVulkanTensorInput::Owned(&first)])
            .unwrap();
        drop(first);
        second.read_into(&mut observed).unwrap();
        same(&observed[..N], &values);
        same(&observed[N..], &[sentinel; 3]);
        let constant = constant_plan.execute_owned(&[]).unwrap();
        constant.read_into(&mut observed).unwrap();
        same(&observed[..N], &constants);
        let uniform_owner = uniform_plan.execute_owned(&[]).unwrap();
        uniform_owner.read_into(&mut observed).unwrap();
        same(&observed[..N], &[uniform; N]);
        assert!(input_plan.execute_owned(&[]).is_err());
        assert!(
            input_plan
                .execute_owned(&[PcuVulkanTensorInput::Host(&values[..N - 1])])
                .is_err()
        );
        second.read_into(&mut observed).unwrap();
        same(&observed[..N], &values); // Invalid later calls never invalidate escaped siblings.
    }
}

#[test]
#[ignore = "requires an actual Vulkan compute GPU"]
fn twenty_two_leaf_plans_detach_constants_and_native_owner_lifetimes() {
    let (backend, _) = device::selected();
    macro_rules! carriers {($($ty:ty),+$(,)?)=>{$(carrier::<$ty>(&backend);)+};}
    carriers!(
        i8,
        u8,
        i16,
        u16,
        i32,
        u32,
        i64,
        u64,
        i128,
        u128,
        PcuI256,
        PcuU256,
        PcuI512,
        PcuU512,
        PcuF16Bits,
        PcuBf16Bits,
        PcuF8E4M3FnBits,
        PcuF8E5M2Bits,
        f32,
        f64,
        PcuF128Bits,
        PcuF256Bits
    );
}

#[test]
#[ignore = "requires an actual Vulkan compute GPU"]
fn leaf_admission_never_grants_arithmetic_or_portable() {
    let (backend, _) = device::selected();
    let mut graph = Graph::default();
    let left = graph.input([3], pcu_facade::PcuScalarType::I32).unwrap();
    let right = graph.input([3], pcu_facade::PcuScalarType::I32).unwrap();
    let output = graph.add(left, right).unwrap();
    assert!(matches!(
        PcuVulkanPreparedScalarTensorGraph::<i32>::prepare(&backend, &graph, &[output]),
        Err(PcuVulkanTensorError::UnsupportedNode(_))
    ));
    // Checked unused arithmetic remains part of the old graph's effect closure. A fresh graph
    // isolates the Portable leaf guard without changing or bypassing that dependency law.
    let mut graph = Graph::default();
    graph.set_numerical_options(PcuNumericalOptions {
        reproducibility: PcuReproducibility::PortableV1,
        ..Default::default()
    });
    let input = graph.input([3], pcu_facade::PcuScalarType::F128).unwrap();
    assert!(matches!(
        PcuVulkanPreparedScalarTensorGraph::<PcuF128Bits>::prepare(&backend, &graph, &[input]),
        Err(PcuVulkanTensorError::Graph(
            TensorError::UnsupportedNumericalOptions { .. }
        ))
    ));
    assert!(matches!(
        PcuVulkanPreparedScalarTensorGraph::<f64>::prepare(&backend, &graph, &[input]),
        Err(PcuVulkanTensorError::Graph(
            TensorError::ScalarTypeMismatch { .. }
        ))
    ));
    assert!(matches!(
        PcuVulkanPreparedScalarTensorGraph::<PcuF128Bits>::prepare(&backend, &graph, &[]),
        Err(PcuVulkanTensorError::OutputCount(0))
    ));
}
