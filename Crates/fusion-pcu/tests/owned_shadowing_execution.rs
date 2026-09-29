//! Immutable source shadowing keeps each alias attached to its original owner.
#![cfg(feature = "tensor")]

use fusion_pcu::pcu;
#[cfg(feature = "rocm")]
use fusion_pcu::global;
#[rustfmt::skip]
use fusion_pcu::{
    PcuExecutionError,
    PcuScalar,
    PcuTensor,
};

#[pcu]
fn seed<T: PcuScalar, const N: usize>(input: &[T; N]) -> Result<PcuTensor<T>, PcuExecutionError> {
    Ok(pcu::identity(input)?)
}

#[pcu]
fn owner_operation_shadow<T: PcuScalar>(
    input: PcuTensor<T>,
) -> Result<PcuTensor<T>, PcuExecutionError> {
    let input = pcu::relu(input)?;
    let input = pcu::identity(input)?;
    Ok(input)
}

#[pcu]
fn view_of_view_then_shadow<T: PcuScalar>(
    input: PcuTensor<T>,
) -> Result<PcuTensor<T>, PcuExecutionError> {
    let view = &input;
    let view = &view;
    let copied = pcu::identity(view)?;
    let input = pcu::relu(input)?;
    Ok(pcu::add(copied, input)?)
}

#[pcu]
fn shadow_owner_keeps_old_view<T: PcuScalar>(
    input: PcuTensor<T>,
    replacement: PcuTensor<T>,
) -> Result<PcuTensor<T>, PcuExecutionError> {
    let owner = pcu::identity(input)?;
    let old_view = &owner;
    let owner = pcu::relu(replacement)?;
    let preserved = pcu::identity(old_view)?;
    Ok(pcu::add(preserved, owner)?)
}

#[pcu]
fn shadow_view_with_owner<T: PcuScalar>(
    input: PcuTensor<T>,
) -> Result<PcuTensor<T>, PcuExecutionError> {
    let view = &input;
    let view = pcu::relu(input)?;
    let copied = pcu::identity(&view)?;
    Ok(copied)
}

#[pcu]
fn borrowed_input_shadowed_by_owner<T: PcuScalar, const N: usize>(
    input: &[T; N],
    replacement: PcuTensor<T>,
) -> Result<PcuTensor<T>, PcuExecutionError> {
    let input = pcu::identity(input)?;
    Ok(pcu::add(input, replacement)?)
}

#[pcu]
fn hidden_names_do_not_capture_generated_locals<T: PcuScalar>(
    __pcu_capture: PcuTensor<T>,
) -> Result<PcuTensor<T>, PcuExecutionError> {
    let __pcu_input = pcu::identity(__pcu_capture)?;
    let __pcu_graph_value = pcu::relu(__pcu_input)?;
    let __pcu_owner = pcu::identity(__pcu_graph_value)?;
    Ok(__pcu_owner)
}

#[test]
#[allow(clippy::type_complexity)] // Exact signatures verify that source shadowing does not alter the API.
fn shadowing_profiles_keep_the_authored_public_signatures() {
    let _: fn(PcuTensor<f32>) -> Result<PcuTensor<f32>, PcuExecutionError> = owner_operation_shadow;
    let _: fn(PcuTensor<f64>) -> Result<PcuTensor<f64>, PcuExecutionError> =
        view_of_view_then_shadow;
    let _: fn(PcuTensor<f32>, PcuTensor<f32>) -> Result<PcuTensor<f32>, PcuExecutionError> =
        shadow_owner_keeps_old_view;
    let _: fn(PcuTensor<f64>) -> Result<PcuTensor<f64>, PcuExecutionError> = shadow_view_with_owner;
    let _: fn(&[f32; 4], PcuTensor<f32>) -> Result<PcuTensor<f32>, PcuExecutionError> =
        borrowed_input_shadowed_by_owner;
    let _: fn(PcuTensor<f32>) -> Result<PcuTensor<f32>, PcuExecutionError> =
        hidden_names_do_not_capture_generated_locals;
}

#[test]
#[cfg(feature = "rocm")]
#[ignore = "requires ROCm hardware"]
fn immutable_shadowing_preserves_old_views_and_escaped_results() {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let _guard = LOCK.lock().unwrap();
    global::use_defaults().unwrap();
    global::clear_thread_cache().unwrap();

    let input = [-2.0_f32, 3.0, -0.0, 8.5];
    let replacement = [1.0_f32, -4.0, -0.0, 0.5];
    let input_owner = seed(&input).unwrap();
    let replacement_owner = seed(&replacement).unwrap();
    let first = shadow_owner_keeps_old_view(input_owner, replacement_owner).unwrap();

    let changed_input = [4.0_f32, -5.0, 0.25, -1.0];
    let changed_replacement = [-3.0_f32, 2.0, -1.0, 5.0];
    let changed_owner = seed(&changed_input).unwrap();
    let changed_replacement_owner = seed(&changed_replacement).unwrap();
    let changed = shadow_owner_keeps_old_view(changed_owner, changed_replacement_owner).unwrap();

    let f64_input = [-16_777_217.0_f64, 16_777_219.0, -0.0, 0.125];
    let f64_replacement = [1.0_f64, -2.0, -0.0, 0.125];
    let f64_input_owner = seed(&f64_input).unwrap();
    let f64_replacement_owner = seed(&f64_replacement).unwrap();
    let f64_result = shadow_owner_keeps_old_view(f64_input_owner, f64_replacement_owner).unwrap();

    let self_shadow_owner = seed(&[-2.0_f32, 3.0, -0.0, 8.5]).unwrap();
    let self_shadow = owner_operation_shadow(self_shadow_owner).unwrap();
    let view_chain_owner = seed(&[-2.0_f32, 3.0, -0.0, 8.5]).unwrap();
    let view_chain = view_of_view_then_shadow(view_chain_owner).unwrap();
    let shadowed_view_owner = seed(&[-2.0_f32, 3.0, -0.0, 8.5]).unwrap();
    let shadowed_view = shadow_view_with_owner(shadowed_view_owner).unwrap();

    let borrowed_source = [2.0_f32, -3.0, 0.25, -0.5];
    let borrowed_replacement = seed(&[-1.0_f32, 4.0, 0.75, 0.5]).unwrap();
    let borrowed_shadow =
        borrowed_input_shadowed_by_owner(&borrowed_source, borrowed_replacement).unwrap();

    let f64_shadow_owner = seed(&[-2.0_f64, 3.0, -0.0, 8.5]).unwrap();
    let f64_shadowed_view = shadow_view_with_owner(f64_shadow_owner).unwrap();
    let hidden_name_owner = seed(&[-2.0_f32, 3.0, -0.0, 8.5]).unwrap();
    let hidden_name_result =
        hidden_names_do_not_capture_generated_locals(hidden_name_owner).unwrap();

    global::clear_thread_cache().unwrap();
    assert_f32(&first, [-1.0, 3.0, 0.0, 9.0]);
    assert_f32(&changed, [4.0, -3.0, 0.25, 4.0]);
    assert_f64(&f64_result, [-16_777_216.0, 16_777_219.0, 0.0, 0.25]);
    assert_f32(&self_shadow, [0.0, 3.0, 0.0, 8.5]);
    // The preserved view contributes input; the shadowed owner contributes ReLU(input).
    assert_f32(&view_chain, [-2.0, 6.0, 0.0, 17.0]);
    assert_f32(&shadowed_view, [0.0, 3.0, 0.0, 8.5]);
    assert_f32(&borrowed_shadow, [1.0, 1.0, 1.0, 0.0]);
    assert_f64(&f64_shadowed_view, [0.0, 3.0, 0.0, 8.5]);
    assert_f32(&hidden_name_result, [0.0, 3.0, 0.0, 8.5]);
    global::clear_thread_cache().unwrap();
}

#[cfg(feature = "rocm")]
fn assert_f32(owner: &PcuTensor<f32>, expected: [f32; 4]) {
    let mut actual = [0.0_f32; 4];
    owner.read_into(&mut actual).unwrap();
    assert_eq!(actual.map(f32::to_bits), expected.map(f32::to_bits));
}

#[cfg(feature = "rocm")]
fn assert_f64(owner: &PcuTensor<f64>, expected: [f64; 4]) {
    let mut actual = [0.0_f64; 4];
    owner.read_into(&mut actual).unwrap();
    assert_eq!(actual.map(f64::to_bits), expected.map(f64::to_bits));
}
