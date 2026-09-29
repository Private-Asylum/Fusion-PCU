//! Source ownership permits reuse; borrowed operands retain their original values.
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
fn add_left<T: PcuScalar>(
    donor: PcuTensor<T>,
    other: &PcuTensor<T>,
) -> Result<PcuTensor<T>, PcuExecutionError> {
    Ok(pcu::add(donor, other)?)
}

#[pcu]
fn subtract_right<T: PcuScalar>(
    other: &PcuTensor<T>,
    donor: PcuTensor<T>,
) -> Result<PcuTensor<T>, PcuExecutionError> {
    Ok(pcu::sub(other, donor)?)
}

#[pcu]
fn multiply_ram<T: PcuScalar, const N: usize>(
    donor: PcuTensor<T>,
    other: &[T; N],
) -> Result<PcuTensor<T>, PcuExecutionError> {
    Ok(pcu::mul(donor, other)?)
}

#[pcu]
fn extra_operation<T: PcuScalar>(
    donor: PcuTensor<T>,
    other: &PcuTensor<T>,
) -> Result<PcuTensor<T>, PcuExecutionError> {
    let sum = pcu::add(donor, other)?;
    Ok(pcu::relu(sum)?)
}

#[pcu]
fn pruned_donor<T: PcuScalar>(
    _donor: PcuTensor<T>,
    other: &PcuTensor<T>,
) -> Result<PcuTensor<T>, PcuExecutionError> {
    Ok(pcu::identity(other)?)
}

#[pcu]
fn add_owners<T: PcuScalar>(
    lhs: PcuTensor<T>,
    rhs: PcuTensor<T>,
) -> Result<PcuTensor<T>, PcuExecutionError> {
    Ok(lhs + rhs)
}

#[pcu]
fn retain_second_owner<T: PcuScalar>(
    _unused: PcuTensor<T>,
    selected: PcuTensor<T>,
) -> Result<PcuTensor<T>, PcuExecutionError> {
    Ok(selected)
}

#[test]
#[allow(clippy::type_complexity)] // Exact source signatures are the contract under test.
fn binary_owner_signatures_remain_ordinary_rust() {
    let _: fn(PcuTensor<f32>, &PcuTensor<f32>) -> Result<PcuTensor<f32>, PcuExecutionError> =
        add_left;
    let _: fn(&PcuTensor<f64>, PcuTensor<f64>) -> Result<PcuTensor<f64>, PcuExecutionError> =
        subtract_right;
    let _: fn(PcuTensor<f32>, &[f32; 4]) -> Result<PcuTensor<f32>, PcuExecutionError> =
        multiply_ram;
    let _: fn(PcuTensor<f64>, PcuTensor<f64>) -> Result<PcuTensor<f64>, PcuExecutionError> =
        add_owners;
}

#[cfg(feature = "rocm")]
fn read_f32(owner: &PcuTensor<f32>, expected: [f32; 4]) {
    let mut actual = [0.0_f32; 4];
    owner.read_into(&mut actual).unwrap();
    assert_eq!(actual.map(f32::to_bits), expected.map(f32::to_bits));
}

#[cfg(feature = "rocm")]
fn read_f64(owner: &PcuTensor<f64>, expected: [f64; 4]) {
    let mut actual = [0.0_f64; 4];
    owner.read_into(&mut actual).unwrap();
    assert_eq!(actual.map(f64::to_bits), expected.map(f64::to_bits));
}

#[test]
#[cfg(feature = "rocm")]
#[ignore = "requires ROCm hardware"]
fn moved_binary_inputs_and_fallbacks_preserve_borrowed_values() {
    global::use_defaults().unwrap();
    global::clear_thread_cache().unwrap();
    let values = [2.0_f32, -3.0, 0.5, 8.0];
    let other = seed(&values).unwrap();
    let result = add_left(seed(&[1.0_f32, 4.0, -0.5, -2.0]).unwrap(), &other).unwrap();
    let changed = add_left(seed(&[-1.0_f32, -4.0, 1.5, 2.0]).unwrap(), &other).unwrap();
    let product = multiply_ram(seed(&values).unwrap(), &[3.0_f32, 2.0, 4.0, 0.5]).unwrap();
    let fallback = extra_operation(seed(&[-3.0_f32, 1.0, 2.0, -9.0]).unwrap(), &other).unwrap();
    let pruned = pruned_donor(seed(&[99.0_f32; 4]).unwrap(), &other).unwrap();
    global::clear_thread_cache().unwrap();
    read_f32(&result, [3.0, 1.0, 0.0, 6.0]);
    read_f32(&changed, [1.0, -7.0, 2.0, 10.0]);
    read_f32(&product, [6.0, -6.0, 2.0, 4.0]);
    read_f32(&fallback, [0.0, 0.0, 2.5, 0.0]);
    read_f32(&pruned, values);
    read_f32(&other, values);

    let other = seed(&[16_777_219.0_f64, -16_777_217.0, 0.125, 8.5]).unwrap();
    let result = subtract_right(&other, seed(&[1.0_f64, -2.0, 0.25, 10.0]).unwrap()).unwrap();
    let sum = add_left(seed(&[1.0_f64, -2.0, 0.25, 10.0]).unwrap(), &other).unwrap();
    let product = multiply_ram(
        seed(&[16_777_219.0_f64, -16_777_217.0, 0.125, 8.5]).unwrap(),
        &[1.0_f64, -2.0, 0.25, 10.0],
    )
    .unwrap();
    global::clear_thread_cache().unwrap();
    read_f64(&result, [16_777_218.0, -16_777_215.0, -0.125, -1.5]);
    read_f64(&sum, [16_777_220.0, -16_777_219.0, 0.375, 18.5]);
    read_f64(&product, [16_777_219.0, 33_554_434.0, 0.03125, 85.0]);
    read_f64(&other, [16_777_219.0, -16_777_217.0, 0.125, 8.5]);
}

#[test]
#[cfg(feature = "rocm")]
#[ignore = "requires ROCm hardware"]
fn two_moved_owners_support_ordinary_operators_and_selected_input_pruning() {
    global::use_defaults().unwrap();
    global::clear_thread_cache().unwrap();
    let first = add_owners(
        seed(&[1.0_f32, 4.0, -0.5, -2.0]).unwrap(),
        seed(&[2.0_f32, -3.0, 0.5, 8.0]).unwrap(),
    )
    .unwrap();
    let second = add_owners(
        seed(&[-1.0_f32, -4.0, 1.5, 2.0]).unwrap(),
        seed(&[2.0_f32, -3.0, 0.5, 8.0]).unwrap(),
    )
    .unwrap();
    let precise = add_owners(
        seed(&[1.0_f64, -2.0, 0.25, 10.0]).unwrap(),
        seed(&[16_777_219.0_f64, -16_777_217.0, 0.125, 8.5]).unwrap(),
    )
    .unwrap();
    let selected = retain_second_owner(
        seed(&[99.0_f32; 2]).unwrap(),
        seed(&[2.0_f32, -3.0, 0.5, 8.0]).unwrap(),
    )
    .unwrap();
    global::clear_thread_cache().unwrap();
    read_f32(&first, [3.0, 1.0, 0.0, 6.0]);
    read_f32(&second, [1.0, -7.0, 2.0, 10.0]);
    read_f64(&precise, [16_777_220.0, -16_777_219.0, 0.375, 18.5]);
    read_f32(&selected, [2.0, -3.0, 0.5, 8.0]);
}
