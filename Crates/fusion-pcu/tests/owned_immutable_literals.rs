//! Real immutable producer source preserves selected inputs and escaped storage.
#![cfg(all(feature = "cpu", feature = "tensor"))]
#[rustfmt::skip]
use fusion_pcu::{
    global,
    pcu,
    PcuExecutionError,
    PcuNumericalMode,
    PcuNumericalOptions,
    PcuTensor,
    PcuU512,
};
use fusion_pcu::dialect::tensor::OpDescriptor;

#[path = "owned_immutable_literals/carriers/carriers.rs"]
mod carriers;
#[path = "owned_immutable_literals/matrices/matrices.rs"]
mod matrices;

#[pcu]
fn pipeline<const N: usize>(input: &[u32; N]) -> Result<PcuTensor<u32>, PcuExecutionError> {
    let constant = pcu::constant(const { [7_u32; N] })?;
    let uniform = pcu::uniform_like(input, const { 2_u32 })?;
    let sum = pcu::add(input, &constant)?;
    pcu::mul(&sum, &uniform)
}
#[pcu]
fn selected(_unused: &[PcuU512]) -> Result<PcuTensor<PcuU512>, PcuExecutionError> {
    pcu::constant(
        const { [PcuU512::from_limbs_le([1, 2, 3, 4, 5, 6, 7, 0x8000_0000_0000_0000]); 3] },
    )
}
#[pcu]
fn shaped_uniform(input: &[[u32; 3]; 2]) -> Result<PcuTensor<u32>, PcuExecutionError> {
    pcu::uniform_like(input, const { 19_u32 })
}
#[pcu]
fn shaped_consumed(input: PcuTensor<u32>) -> Result<PcuTensor<u32>, PcuExecutionError> {
    let uniform = pcu::uniform_like(input, const { 2_u32 })?;
    pcu::add(input, &uniform)
}

#[pcu]
fn consume(input: PcuTensor<PcuU512>) -> Result<PcuTensor<PcuU512>, PcuExecutionError> {
    pcu::identity(input)
}
#[pcu(invocations = 3)]
fn overwrite<T: fusion_pcu::PcuScalar>(input: &[T], output: &mut [T]) {
    let id = pcu::context::global_invocation_id();
    output[id] = input[0];
}
fn configure_cpu() {
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Cpu,
        ..Default::default()
    })
    .unwrap();
    global::clear_thread_cache().unwrap();
}
#[test]
fn genuine_producer_nodes_select_only_real_data_and_preserve_full_policy() {
    let captured = global::__pcu_capture_tensor_program::<u32, 1, _>(
        [global::PcuSourceShape::FixedArray { length: 3 }],
        fusion_pcu::PcuFloatUnderflowPolicy::IeeeAfterRounding,
        PcuNumericalMode::Strict,
        PcuNumericalOptions::default(),
        pipeline::__pcu_capture_entry::<3>,
    )
    .unwrap();
    assert_eq!(captured.argument_indices(), [0]);
    assert_eq!(captured.program().input_values().len(), 1);
    let graph = captured.program().graph();
    let mut constants = 0;
    let mut uniforms = 0;
    for node in graph.nodes() {
        if let OpDescriptor::Constant(value) = node.op {
            constants += 1;
            assert_eq!(value.as_typed::<u32>().unwrap().data(), [7; 3]);
            assert_eq!(node.numerical_options, PcuNumericalOptions::default());
        }
        if let OpDescriptor::Uniform { value } = node.op {
            uniforms += 1;
            assert_eq!(value.as_typed::<u32>().unwrap(), 2);
            assert_eq!(node.shape, [3]);
        }
    }
    assert_eq!((constants, uniforms), (1, 1));
    configure_cpu();
    let output = pipeline::<3>(&[1, 2, 3]).unwrap();
    let mut bits = [0; 5];
    output.read_into(&mut bits).unwrap();
    assert_eq!(bits, [16, 18, 20, 0, 0]);
    let mut second = [0; 3];
    pipeline::<3>(&[4, 5, 6])
        .unwrap()
        .read_into(&mut second)
        .unwrap();
    assert_eq!(second, [22, 24, 26]);
}
#[test]
fn literal_only_source_has_zero_bindings_and_escaped_mutation_cannot_poison_replay() {
    let captured = global::__pcu_capture_tensor_program::<PcuU512, 1, _>(
        [global::PcuSourceShape::Slice { length: 0 }],
        fusion_pcu::PcuFloatUnderflowPolicy::IeeeAfterRounding,
        PcuNumericalMode::Boundary,
        PcuNumericalOptions::default(),
        selected::__pcu_capture_entry,
    )
    .unwrap();
    assert!(captured.argument_indices().is_empty());
    assert!(captured.input_values().is_empty());
    assert!(captured.program().input_values().is_empty());
    configure_cpu();
    let mut escaped = selected(&[]).unwrap();
    let expected = PcuU512::from_limbs_le([1, 2, 3, 4, 5, 6, 7, 0x8000_0000_0000_0000]);
    let replacement = PcuU512::from_limbs_le([42, 0, 0, 0, 0, 0, 0, 0]);
    overwrite(&[replacement], &mut escaped).unwrap();
    let consumed = consume(escaped).unwrap();
    let mut observed = [expected; 3];
    consumed.read_into(&mut observed).unwrap();
    assert_eq!(observed, [replacement; 3]);
    let original = selected(&[]).unwrap();
    original.read_into(&mut observed).unwrap();
    assert_eq!(observed, [expected; 3]);
    global::clear_thread_cache().unwrap();
    consumed.read_into(&mut observed).unwrap();
    assert_eq!(observed, [replacement; 3]);
}
#[test]
fn uniform_shape_anchor_is_selected_out_as_data() {
    let captured = global::__pcu_capture_tensor_program::<u32, 1, _>(
        [global::PcuSourceShape::FixedMatrix {
            rows: 2,
            columns: 3,
        }],
        fusion_pcu::PcuFloatUnderflowPolicy::IeeeAfterRounding,
        PcuNumericalMode::Boundary,
        PcuNumericalOptions::default(),
        shaped_uniform::__pcu_capture_entry,
    )
    .unwrap();
    assert!(captured.input_values().is_empty());
    let output = captured.program().output_values()[0];
    assert_eq!(captured.program().graph().shape(output).unwrap(), [2, 3]);
    configure_cpu();
    let output = shaped_uniform(&[[1, 2, 3], [4, 5, 6]]).unwrap();
    let mut values = [0; 6];
    output.read_into(&mut values).unwrap();
    assert_eq!(values, [19; 6]);
}

#[test]
fn shape_only_inspection_borrows_consumed_owner_before_real_data_use() {
    configure_cpu();
    let input = pipeline::<3>(&[1, 2, 3]).unwrap();
    let output = shaped_consumed(input).unwrap();
    let mut observed = [0; 3];
    output.read_into(&mut observed).unwrap();
    assert_eq!(observed, [18, 20, 22]);
}

#[pcu]
fn specialized<const VALUE: usize>(_unused: &[u32]) -> Result<PcuTensor<u32>, PcuExecutionError> {
    pcu::constant(const { [if VALUE == 7 { 7_u32 } else { 11_u32 }; 3] })
}

#[test]
fn immutable_const_specializations_keep_distinct_cache_identity_without_data_inputs() {
    configure_cpu();
    let mut observed = [0; 3];
    specialized::<7>(&[])
        .unwrap()
        .read_into(&mut observed)
        .unwrap();
    assert_eq!(observed, [7; 3]);
    specialized::<11>(&[])
        .unwrap()
        .read_into(&mut observed)
        .unwrap();
    assert_eq!(observed, [11; 3]);
    specialized::<7>(&[])
        .unwrap()
        .read_into(&mut observed)
        .unwrap();
    assert_eq!(observed, [7; 3]);
}

#[test]
fn eight_float_carriers_preserve_raw_literal_bits_across_policies_and_ownership() {
    carriers::run(global::PcuBackendChoice::Cpu);
}
