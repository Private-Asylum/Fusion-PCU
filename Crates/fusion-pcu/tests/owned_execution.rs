//! Hardware conformance for source-level owned tensor composition.
#![cfg(all(feature = "rocm", feature = "tensor"))]

use fusion_pcu::pcu;
#[rustfmt::skip]
use fusion_pcu::{
    global,
    PcuExecutionError,
    PcuTensor,
};

static POLICY_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[pcu]
fn relu(input: &[f32]) -> Result<PcuTensor<f32>, PcuExecutionError> {
    Ok(pcu::relu(input)?)
}

#[pcu]
fn identity(input: &[f32]) -> Result<PcuTensor<f32>, PcuExecutionError> {
    Ok(pcu::identity(input)?)
}

#[pcu]
fn compose(input: &[f32]) -> Result<PcuTensor<f32>, PcuExecutionError> {
    let first = pcu::relu(input)?;
    let _dead = pcu::identity(input)?;
    let _dead_branch = pcu::identity(&first)?;
    let live_branch = pcu::relu(first)?;
    Ok(pcu::identity(live_branch)?)
}

#[pcu]
fn activate_then_copy(input: &[f32]) -> Result<PcuTensor<f32>, PcuExecutionError> {
    let activated = relu(input)?;
    Ok(identity(&activated)?)
}

mod tensor_helpers {
    use super::*;

    #[pcu]
    pub(super) fn positive_part(input: &[f32]) -> Result<PcuTensor<f32>, PcuExecutionError> {
        Ok(pcu::relu(input)?)
    }
}

#[pcu]
fn qualified_helper(input: &[f32]) -> Result<PcuTensor<f32>, PcuExecutionError> {
    Ok(tensor_helpers::positive_part(input)?)
}

use tensor_helpers::positive_part as imported_positive;

#[pcu]
fn aliased_helper(input: &[f32]) -> Result<PcuTensor<f32>, PcuExecutionError> {
    Ok(imported_positive(input)?)
}

#[pcu]
fn self_qualified_helper(input: &[f32]) -> Result<PcuTensor<f32>, PcuExecutionError> {
    Ok(self::tensor_helpers::positive_part(input)?)
}

// Neither the wrapper nor its companion may survive the source function's cfg gate.
// The unresolved helper makes a leaked companion a compilation failure.
#[pcu]
#[cfg(any())]
fn disabled_helper(input: &[f32]) -> Result<PcuTensor<f32>, PcuExecutionError> {
    Ok(helper_that_does_not_exist(input)?)
}

#[pcu]
#[cfg_attr(all(), cfg(any()))]
fn cfg_attr_disabled_helper(input: &[f32]) -> Result<PcuTensor<f32>, PcuExecutionError> {
    Ok(helper_that_does_not_exist(input)?)
}

#[pcu]
fn direct_recursive(input: &[f32]) -> Result<PcuTensor<f32>, PcuExecutionError> {
    let nested = direct_recursive(input)?;
    Ok(nested)
}

#[pcu]
fn mutual_recursive_a(input: &[f32]) -> Result<PcuTensor<f32>, PcuExecutionError> {
    let nested = mutual_recursive_b(input)?;
    Ok(nested)
}

#[pcu]
fn mutual_recursive_b(input: &[f32]) -> Result<PcuTensor<f32>, PcuExecutionError> {
    let nested = mutual_recursive_a(input)?;
    Ok(nested)
}

#[pcu]
fn add_pair(lhs: &[f32], rhs: &[f32]) -> Result<PcuTensor<f32>, PcuExecutionError> {
    Ok(pcu::add(lhs, rhs)?)
}

#[pcu]
fn add_after_activation(lhs: &[f32], rhs: &[f32]) -> Result<PcuTensor<f32>, PcuExecutionError> {
    let sum = add_pair(lhs, rhs)?;
    Ok(pcu::relu(sum)?)
}

#[pcu]
fn arithmetic(lhs: &[f32], rhs: &[f32]) -> Result<PcuTensor<f32>, PcuExecutionError> {
    let sum = lhs + rhs;
    let difference = pcu::sub(lhs, rhs)?;
    let product = difference * rhs;
    Ok(pcu::mul(sum, product)?)
}

#[pcu]
fn nested_arithmetic(lhs: &[f32], rhs: &[f32]) -> Result<PcuTensor<f32>, PcuExecutionError> {
    Ok(pcu::relu(&(lhs + rhs) * rhs)?)
}

#[pcu]
fn nested_helper(lhs: &[f32], rhs: &[f32]) -> Result<PcuTensor<f32>, PcuExecutionError> {
    Ok(pcu::relu(add_pair(lhs, rhs)?)?)
}

fn assert_bits(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert_eq!(actual.to_bits(), expected.to_bits());
    }
}

#[test]
fn generated_owned_entry_preserves_slice_reference_fn_pointer_shape() {
    let _: for<'input> fn(&'input [f32]) -> Result<PcuTensor<f32>, PcuExecutionError> = relu;
}

#[test]
#[ignore = "requires a working ROCm device"]
fn mutable_slice_reference_is_reborrowed_without_consuming_it() {
    let _guard = POLICY_TEST_LOCK.lock().unwrap();
    global::use_defaults().unwrap();
    let mut source = [-4.0_f32, 2.0, -7.0];
    let (first, second) = {
        let input = &mut source;
        (identity(input).unwrap(), identity(input).unwrap())
    };
    let mut first_values = [f32::NAN; 3];
    let mut second_values = [f32::NAN; 3];
    first.read_into(&mut first_values).unwrap();
    second.read_into(&mut second_values).unwrap();
    assert_bits(&first_values, &source);
    assert_bits(&second_values, &source);
    global::clear_thread_cache().unwrap();
}

#[test]
#[ignore = "requires a working ROCm device"]
fn owned_result_can_be_borrowed_by_the_same_generated_function() {
    let _guard = POLICY_TEST_LOCK.lock().unwrap();
    global::use_defaults().unwrap();

    let original = [-2.0_f32, -0.0, 0.5, 3.0];
    let first = relu(&original).unwrap();
    let second = relu(&first).unwrap();
    let copied = identity(&second).unwrap();

    let mut first_values = [f32::NAN; 4];
    let mut copied_values = [f32::NAN; 4];
    first.read_into(&mut first_values).unwrap();
    copied.read_into(&mut copied_values).unwrap();
    assert_bits(&first_values, &[0.0, 0.0, 0.5, 3.0]);
    assert_bits(&copied_values, &[0.0, 0.0, 0.5, 3.0]);
    global::clear_thread_cache().unwrap();
}

#[test]
#[ignore = "requires a working ROCm device"]
fn escaped_results_remain_independent_across_shape_specialization_and_cache_clear() {
    let _guard = POLICY_TEST_LOCK.lock().unwrap();
    global::use_defaults().unwrap();

    let first_source = [-1.0_f32, 2.0, -3.0];
    let first = relu(&first_source).unwrap();
    let other_shape = [4.0_f32, -5.0, 6.0, -7.0, 8.0];
    let other = relu(&other_shape).unwrap();
    let changed_source = [9.0_f32, -10.0, 11.0];
    let changed = relu(&changed_source).unwrap();

    global::clear_thread_cache().unwrap();
    let after_clear = relu(&[-12.0_f32, 13.0, -14.0]).unwrap();

    let mut first_values = [f32::NAN; 3];
    let mut other_values = [f32::NAN; 5];
    let mut changed_values = [f32::NAN; 3];
    let mut after_clear_values = [f32::NAN; 3];
    first.read_into(&mut first_values).unwrap();
    other.read_into(&mut other_values).unwrap();
    changed.read_into(&mut changed_values).unwrap();
    after_clear.read_into(&mut after_clear_values).unwrap();
    assert_bits(&first_values, &[0.0, 2.0, 0.0]);
    assert_bits(&other_values, &[4.0, 0.0, 6.0, 0.0, 8.0]);
    assert_bits(&changed_values, &[9.0, 0.0, 11.0]);
    assert_bits(&after_clear_values, &[0.0, 13.0, 0.0]);
    global::clear_thread_cache().unwrap();
}

#[test]
#[ignore = "requires a working ROCm device"]
fn captured_chain_fanout_and_dead_steps_execute_as_one_owned_program() {
    let _guard = POLICY_TEST_LOCK.lock().unwrap();
    global::use_defaults().unwrap();

    let source = [-2.0_f32, 1.0, 4.0, -8.0];
    let held = compose(&source).unwrap();
    let later = relu(&[7.0_f32, -9.0, 2.0, -3.0]).unwrap();

    let mut held_values = [f32::NAN; 4];
    let mut later_values = [f32::NAN; 4];
    held.read_into(&mut held_values).unwrap();
    later.read_into(&mut later_values).unwrap();
    assert_bits(&held_values, &[0.0, 1.0, 4.0, 0.0]);
    assert_bits(&later_values, &[7.0, 0.0, 2.0, 0.0]);
    global::clear_thread_cache().unwrap();
}

#[test]
#[ignore = "requires a working ROCm device"]
fn helper_composition_captures_multiple_functions_into_one_result() {
    let _guard = POLICY_TEST_LOCK.lock().unwrap();
    global::use_defaults().unwrap();
    let source = [-3.0_f32, 2.0, -1.0, 4.0];
    let output = activate_then_copy(&source).unwrap();
    let qualified = qualified_helper(&source).unwrap();
    let aliased = aliased_helper(&source).unwrap();
    let self_qualified = self_qualified_helper(&source).unwrap();
    let mut output_values = [f32::NAN; 4];
    let mut qualified_values = [f32::NAN; 4];
    output.read_into(&mut output_values).unwrap();
    qualified.read_into(&mut qualified_values).unwrap();
    assert_bits(&output_values, &[0.0, 2.0, 0.0, 4.0]);
    assert_bits(&qualified_values, &[0.0, 2.0, 0.0, 4.0]);
    for owner in [&aliased, &self_qualified] {
        let mut observed = [f32::NAN; 4];
        owner.read_into(&mut observed).unwrap();
        assert_bits(&observed, &[0.0, 2.0, 0.0, 4.0]);
    }
    global::clear_thread_cache().unwrap();
}

#[test]
fn direct_and_mutual_helper_recursion_are_rejected_during_capture() {
    let _guard = POLICY_TEST_LOCK.lock().unwrap();
    global::use_defaults().unwrap();
    let source = [1.0_f32];
    assert!(matches!(
        direct_recursive(&source),
        Err(PcuExecutionError::RecursiveTensorSource)
    ));
    assert!(matches!(
        mutual_recursive_a(&source),
        Err(PcuExecutionError::RecursiveTensorSource)
    ));
    global::clear_thread_cache().unwrap();
}

#[test]
#[ignore = "requires a working ROCm device"]
fn multiple_host_inputs_use_one_captured_add_graph() {
    let _guard = POLICY_TEST_LOCK.lock().unwrap();
    global::use_defaults().unwrap();
    let lhs = [-3.0_f32, 2.0, 1.0];
    let rhs = [5.0_f32, -4.0, 8.0];
    let output = add_pair(&lhs, &rhs).unwrap();
    let mut values = [f32::NAN; 3];
    output.read_into(&mut values).unwrap();
    assert_bits(&values, &[2.0, -2.0, 9.0]);
    assert!(add_pair(&lhs, &[1.0_f32, 2.0]).is_err());
    global::clear_thread_cache().unwrap();
}

#[test]
#[ignore = "requires a working ROCm device"]
fn multi_input_helper_composition_accepts_two_resident_owners() {
    let _guard = POLICY_TEST_LOCK.lock().unwrap();
    global::use_defaults().unwrap();
    let lhs = relu(&[-3.0_f32, 2.0, 1.0]).unwrap();
    let rhs = relu(&[5.0_f32, -4.0, 8.0]).unwrap();
    let output = add_after_activation(&lhs, &rhs).unwrap();
    let mut values = [f32::NAN; 3];
    output.read_into(&mut values).unwrap();
    assert_bits(&values, &[5.0, 2.0, 9.0]);
    global::clear_thread_cache().unwrap();
}

#[test]
#[ignore = "requires a working ROCm device"]
fn multi_input_calls_accept_mixed_and_repeated_resident_sources() {
    let _guard = POLICY_TEST_LOCK.lock().unwrap();
    global::use_defaults().unwrap();
    let resident = relu(&[-3.0_f32, 2.0, -1.0]).unwrap();
    let host = [5.0_f32, -4.0, 8.0];
    let mixed = add_after_activation(&host, &resident).unwrap();
    let repeated = add_pair(&resident, &resident).unwrap();
    let mut mixed_values = [f32::NAN; 3];
    let mut repeated_values = [f32::NAN; 3];
    mixed.read_into(&mut mixed_values).unwrap();
    repeated.read_into(&mut repeated_values).unwrap();
    assert_bits(&mixed_values, &[5.0, 0.0, 8.0]);
    assert_bits(&repeated_values, &[0.0, 4.0, 0.0]);
    global::clear_thread_cache().unwrap();
}

#[test]
#[ignore = "requires a working ROCm device"]
fn multi_input_rejects_resident_sources_from_different_retained_roots() {
    let _guard = POLICY_TEST_LOCK.lock().unwrap();
    global::use_defaults().unwrap();
    let first = relu(&[1.0_f32, 2.0]).unwrap();
    global::clear_thread_cache().unwrap();
    global::use_defaults().unwrap();
    let second = relu(&[3.0_f32, 4.0]).unwrap();
    assert!(matches!(
        add_pair(&first, &second),
        Err(PcuExecutionError::Argument(
            global::PcuArgumentError::SessionMismatch
        ))
    ));
    global::clear_thread_cache().unwrap();
}

#[test]
#[ignore = "requires a working ROCm device"]
fn elementwise_add_subtract_and_multiply_capture_with_named_or_operator_forms() {
    let _guard = POLICY_TEST_LOCK.lock().unwrap();
    global::use_defaults().unwrap();
    let lhs = [2.0_f32, 3.0];
    let rhs = [1.0_f32, -2.0];
    let output = arithmetic(&lhs, &rhs).unwrap();
    let mut values = [f32::NAN; 2];
    output.read_into(&mut values).unwrap();
    assert_bits(&values, &[3.0, -10.0]);
    global::clear_thread_cache().unwrap();
}

#[test]
#[ignore = "requires a working ROCm device"]
fn nested_expressions_and_helpers_lower_to_capture_temporaries() {
    let _guard = POLICY_TEST_LOCK.lock().unwrap();
    global::use_defaults().unwrap();
    let lhs = [2.0_f32, 3.0];
    let rhs = [1.0_f32, -2.0];
    let nested = nested_arithmetic(&lhs, &rhs).unwrap();
    let helper = nested_helper(&lhs, &rhs).unwrap();
    let mut nested_values = [f32::NAN; 2];
    let mut helper_values = [f32::NAN; 2];
    nested.read_into(&mut nested_values).unwrap();
    helper.read_into(&mut helper_values).unwrap();
    assert_bits(&nested_values, &[3.0, 0.0]);
    assert_bits(&helper_values, &[3.0, 1.0]);
    global::clear_thread_cache().unwrap();
}
