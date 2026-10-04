//! Genuine no-argument producers retain type/const cache identity and initialized owners.
#![cfg(all(feature = "cpu", feature = "tensor"))]
#[rustfmt::skip]
use fusion_pcu::{
    global,
    pcu,
    PcuExecutionError,
    PcuTensor,
    PcuU512,
};
#[pcu]
fn literal<const VALUE: usize>() -> Result<PcuTensor<u32>, PcuExecutionError> {
    pcu::constant(const { [if VALUE == 7 { 7_u32 } else { 11_u32 }; 3] })
}
#[pcu]
fn nested() -> Result<PcuTensor<u32>, PcuExecutionError> {
    literal::<7>()
}
#[pcu]
fn splat() -> Result<PcuTensor<u32>, PcuExecutionError> {
    let shape = pcu::constant(const { [0_u32; 3] })?;
    pcu::uniform_like(&shape, const { 19_u32 })
}
#[pcu]
fn wide() -> Result<PcuTensor<PcuU512>, PcuExecutionError> {
    pcu::constant(const { [PcuU512::from_limbs_le([1, 2, 3, 4, 5, 6, 7, u64::MAX]); 3] })
}
#[pcu]
fn empty() -> Result<PcuTensor<u32>, PcuExecutionError> {
    pcu::constant(const { [0_u32; 0] })
}
fn configure() {
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Cpu,
        ..Default::default()
    })
    .unwrap();
    global::clear_thread_cache().unwrap();
}
#[test]
fn no_argument_producers_and_nested_helpers_execute_with_exact_cache_identity() {
    configure();
    let mut output = [0; 5];
    literal::<7>().unwrap().read_into(&mut output).unwrap();
    assert_eq!(output, [7, 7, 7, 0, 0]);
    literal::<11>().unwrap().read_into(&mut output).unwrap();
    assert_eq!(output, [11, 11, 11, 0, 0]);
    nested().unwrap().read_into(&mut output).unwrap();
    assert_eq!(output, [7, 7, 7, 0, 0]);
    splat().unwrap().read_into(&mut output).unwrap();
    assert_eq!(output, [19, 19, 19, 0, 0]);
    let owner = wide().unwrap();
    let expected = PcuU512::from_limbs_le([1, 2, 3, 4, 5, 6, 7, u64::MAX]);
    let mut bits = [PcuU512::ZERO; 3];
    owner.read_into(&mut bits).unwrap();
    assert_eq!(bits, [expected; 3]);
    global::clear_thread_cache().unwrap();
    owner.read_into(&mut bits).unwrap();
    assert_eq!(bits, [expected; 3]);
}
#[test]
fn empty_cpu_literal_preserves_constructor_law_and_does_not_write_destination_tail() {
    configure();
    let owner = empty().unwrap();
    assert!(owner.is_empty());
    assert_eq!(owner.shape(), [0]);
    let mut destination = [7_u32, 11, 19];
    owner.read_into(&mut destination).unwrap();
    assert_eq!(destination, [7, 11, 19]);
}

#[pcu(flag(ieee_underflow))]
fn float_literal() -> Result<PcuTensor<f32>, PcuExecutionError> {
    pcu::constant(const { [f32::from_bits(0x7fc1_2345), -0.0, f32::from_bits(1)] })
}
#[test]
fn explicit_float_policy_checks_return_type_without_changing_payload_bits() {
    configure();
    let captured = global::__pcu_capture_tensor_program::<f32, 0, _>(
        [],
        fusion_pcu::PcuFloatUnderflowPolicy::RejectSubnormalResult,
        fusion_pcu::PcuNumericalMode::Strict,
        fusion_pcu::PcuNumericalOptions::default(),
        float_literal::__pcu_capture_entry,
    )
    .unwrap();
    assert!(captured.input_values().is_empty());
    assert!(captured.argument_indices().is_empty());
    let mut output = [0.0; 3];
    float_literal().unwrap().read_into(&mut output).unwrap();
    assert_eq!(output.map(f32::to_bits), [0x7fc1_2345, 0x8000_0000, 1]);
}
#[pcu]
fn real_input(input: &[u32]) -> Result<PcuTensor<u32>, PcuExecutionError> {
    pcu::identity(input)
}
#[test]
fn selected_empty_input_is_still_refused() {
    configure();
    assert!(matches!(
        real_input(&[]),
        Err(PcuExecutionError::EmptyTensorInput)
    ));
}
