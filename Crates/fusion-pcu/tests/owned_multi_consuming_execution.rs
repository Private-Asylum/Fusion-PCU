//! Mixed source orchestration with multiple consumed resident owners.
#![cfg(feature = "tensor")]

use fusion_pcu::pcu;
#[rustfmt::skip]
use fusion_pcu::{
    PcuExecutionError,
    PcuScalar,
    PcuTensor,
};
#[cfg(feature = "rocm")]
use fusion_pcu::global;

#[pcu]
fn sum_two_owned<T: PcuScalar>(
    lhs: PcuTensor<T>,
    rhs: PcuTensor<T>,
) -> Result<PcuTensor<T>, PcuExecutionError> {
    Ok(pcu::add(lhs, rhs)?)
}

#[pcu]
fn sum_two_owned_and_ram<T: PcuScalar>(
    lhs: PcuTensor<T>,
    rhs: PcuTensor<T>,
    tail: &[T],
) -> Result<PcuTensor<T>, PcuExecutionError> {
    let partial = pcu::add(lhs, rhs)?;
    Ok(pcu::add(partial, tail)?)
}

#[pcu]
fn sum_owned_and_readonly_resident<T: PcuScalar>(
    owner: PcuTensor<T>,
    resident: &PcuTensor<T>,
) -> Result<PcuTensor<T>, PcuExecutionError> {
    Ok(pcu::add(owner, resident)?)
}

#[pcu]
fn ignore_second_owner<T: PcuScalar>(
    retained: PcuTensor<T>,
    _unused: PcuTensor<T>,
) -> Result<PcuTensor<T>, PcuExecutionError> {
    Ok(pcu::identity(retained)?)
}

#[pcu]
fn seed<T: PcuScalar, const N: usize>(input: &[T; N]) -> Result<PcuTensor<T>, PcuExecutionError> {
    Ok(pcu::identity(input)?)
}

#[cfg(feature = "rocm")]
static POLICY_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[test]
#[allow(clippy::type_complexity)] // The exact generated HRTB signatures are the acceptance contract.
fn mixed_multiowner_profiles_preserve_native_reference_signatures() {
    let _: for<'tail> fn(
        PcuTensor<f32>,
        PcuTensor<f32>,
        &'tail [f32],
    ) -> Result<PcuTensor<f32>, PcuExecutionError> = sum_two_owned_and_ram::<f32>;
    let _: for<'resident> fn(
        PcuTensor<f64>,
        &'resident PcuTensor<f64>,
    ) -> Result<PcuTensor<f64>, PcuExecutionError> = sum_owned_and_readonly_resident::<f64>;
}

#[test]
#[cfg(feature = "rocm")]
#[ignore = "requires ROCm hardware"]
fn multiowner_ram_and_borrowed_resident_inputs_keep_results_and_sources() {
    let _guard = POLICY_TEST_LOCK.lock().unwrap();
    global::use_defaults().unwrap();
    global::clear_thread_cache().unwrap();

    let lhs = seed(&[-2.0_f32, 3.0, -0.5, 4.0]).unwrap();
    let rhs = seed(&[1.0_f32, -5.0, 2.0, 0.5]).unwrap();
    let tail = [2.0_f32, 4.0, 1.0, -3.0];
    let first = sum_two_owned_and_ram(lhs, rhs, &tail).unwrap();

    let changed_lhs = seed(&[4.0_f32, -3.0, 1.25, -2.0]).unwrap();
    let changed_rhs = seed(&[-1.0_f32, 5.0, 0.75, 6.0]).unwrap();
    let changed_tail = [-4.0_f32, 2.0, 3.0, 1.0];
    let changed = sum_two_owned_and_ram(changed_lhs, changed_rhs, &changed_tail).unwrap();

    let unused_left = seed(&[-4.0_f32, 3.0, 2.0, -1.0]).unwrap();
    let unused_right = seed(&[99.0_f32, 98.0, 97.0, 96.0]).unwrap();
    let unused_owner_output = ignore_second_owner(unused_left, unused_right).unwrap();

    let resident_values = [16_777_217.0_f64, -16_777_219.0, -0.5, 4.0];
    let resident = seed(&resident_values).unwrap();
    let owner = seed(&[-16_777_216.0_f64, 16_777_220.0, 0.25, -2.0]).unwrap();
    let resident_result = sum_owned_and_readonly_resident(owner, &resident).unwrap();

    let old_root = seed(&[1.0_f32, 2.0, 3.0, 4.0]).unwrap();
    global::clear_thread_cache().unwrap();
    let new_root = seed(&[10.0_f32, 20.0, 30.0, 40.0]).unwrap();
    let cross_root_error = sum_two_owned(old_root, new_root).unwrap_err();
    assert!(matches!(
        cross_root_error,
        PcuExecutionError::Argument(global::PcuArgumentError::SessionMismatch)
    ));

    let wrong_shape_owner = seed(&[1.0_f32, 2.0, 3.0, 4.0]).unwrap();
    let matching_owner = seed(&[4.0_f32, 3.0, 2.0, 1.0]).unwrap();
    let short_tail = [1.0_f32, 2.0, 3.0];
    let shape_error =
        sum_two_owned_and_ram(wrong_shape_owner, matching_owner, &short_tail).unwrap_err();
    assert!(matches!(
        shape_error,
        PcuExecutionError::TensorBuild(
            fusion_pcu::dialect::tensor::TensorError::ShapeMismatch { left, right }
        ) if left == [4] && right == [3]
    ));

    global::clear_thread_cache().unwrap();
    let mut observed = [f32::NAN; 4];
    first.read_into(&mut observed).unwrap();
    assert_eq!(
        observed.map(f32::to_bits),
        [1.0, 2.0, 2.5, 1.5].map(f32::to_bits)
    );
    changed.read_into(&mut observed).unwrap();
    assert_eq!(
        observed.map(f32::to_bits),
        [-1.0, 4.0, 5.0, 5.0].map(f32::to_bits)
    );
    unused_owner_output.read_into(&mut observed).unwrap();
    assert_eq!(
        observed.map(f32::to_bits),
        [-4.0, 3.0, 2.0, -1.0].map(f32::to_bits)
    );

    let mut observed_f64 = [f64::NAN; 4];
    resident_result.read_into(&mut observed_f64).unwrap();
    assert_eq!(
        observed_f64.map(f64::to_bits),
        [1.0, 1.0, -0.25, 2.0].map(f64::to_bits)
    );
    let mut resident_after = [f64::NAN; 4];
    resident.read_into(&mut resident_after).unwrap();
    assert_eq!(
        resident_after.map(f64::to_bits),
        resident_values.map(f64::to_bits)
    );
    assert_eq!(
        tail.map(f32::to_bits),
        [2.0, 4.0, 1.0, -3.0].map(f32::to_bits)
    );
}

#[test]
#[cfg(feature = "rocm")]
#[ignore = "requires ROCm hardware"]
fn unused_resident_owners_do_not_affect_selected_root_admission() {
    let _guard = POLICY_TEST_LOCK.lock().unwrap();
    global::use_defaults().unwrap();
    global::clear_thread_cache().unwrap();

    let initial_selected = seed(&[1.0_f32, 2.0, 3.0, 4.0]).unwrap();
    let unused_a_warm = seed(&[-1.0_f32, -2.0, -3.0, -4.0]).unwrap();
    let selected_a_after_switch = seed(&[5.0_f32, 6.0, 7.0, 8.0]).unwrap();
    let unused_a_for_new_root = seed(&[90.0_f32, 91.0, 92.0, 93.0]).unwrap();
    let unused_a_for_warm_new_root = seed(&[80.0_f32, 81.0, 82.0, 83.0]).unwrap();
    let cross_root_a = seed(&[11.0_f32, 12.0, 13.0, 14.0]).unwrap();
    let warmed_a = ignore_second_owner(initial_selected, unused_a_warm).unwrap();

    global::clear_thread_cache().unwrap();
    let selected_b = seed(&[-5.0_f32, -6.0, -7.0, -8.0]).unwrap();
    let unused_b_for_old_root = seed(&[70.0_f32, 71.0, 72.0, 73.0]).unwrap();
    let revisited_selected = seed(&[-9.0_f32, -10.0, -11.0, -12.0]).unwrap();
    let cross_root_b = seed(&[21.0_f32, 22.0, 23.0, 24.0]).unwrap();

    // First call warms the new root while carrying an unused old-root owner.
    let new_selected_old_unused = ignore_second_owner(selected_b, unused_a_for_new_root).unwrap();
    // The retained old root can receive a new cache entry after the thread cache was cleared.
    let old_selected_new_unused =
        ignore_second_owner(selected_a_after_switch, unused_b_for_old_root).unwrap();
    // Repeat on the new warm root with another unused owner from the old session.
    let warm_new_selected_old_unused =
        ignore_second_owner(revisited_selected, unused_a_for_warm_new_root).unwrap();

    let selected_cross_root = sum_two_owned(cross_root_a, cross_root_b).unwrap_err();
    assert!(matches!(
        selected_cross_root,
        PcuExecutionError::Argument(global::PcuArgumentError::SessionMismatch)
    ));

    global::clear_thread_cache().unwrap();
    let mut observed = [f32::NAN; 4];
    warmed_a.read_into(&mut observed).unwrap();
    assert_eq!(
        observed.map(f32::to_bits),
        [1.0, 2.0, 3.0, 4.0].map(f32::to_bits)
    );
    new_selected_old_unused.read_into(&mut observed).unwrap();
    assert_eq!(
        observed.map(f32::to_bits),
        [-5.0, -6.0, -7.0, -8.0].map(f32::to_bits)
    );
    old_selected_new_unused.read_into(&mut observed).unwrap();
    assert_eq!(
        observed.map(f32::to_bits),
        [5.0, 6.0, 7.0, 8.0].map(f32::to_bits)
    );
    warm_new_selected_old_unused
        .read_into(&mut observed)
        .unwrap();
    assert_eq!(
        observed.map(f32::to_bits),
        [-9.0, -10.0, -11.0, -12.0].map(f32::to_bits)
    );
}
