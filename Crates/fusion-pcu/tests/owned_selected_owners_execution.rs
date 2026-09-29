//! Selected-input ownership is independent of the number or position of authored formals.
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
fn seed<T: PcuScalar, const N: usize>(input: &[T; N]) -> Result<PcuTensor<T>, PcuExecutionError> {
    Ok(pcu::identity(input)?)
}

#[pcu]
fn selected_pair<T: PcuScalar>(
    _unused: PcuTensor<T>,
    lhs: PcuTensor<T>,
    rhs: PcuTensor<T>,
) -> Result<PcuTensor<T>, PcuExecutionError> {
    Ok(lhs - rhs)
}

#[pcu]
fn selected_last<T: PcuScalar>(
    _first: PcuTensor<T>,
    _second: PcuTensor<T>,
    last: PcuTensor<T>,
) -> Result<PcuTensor<T>, PcuExecutionError> {
    Ok(last)
}

#[pcu]
fn all_selected<T: PcuScalar>(
    first: PcuTensor<T>,
    second: PcuTensor<T>,
    third: PcuTensor<T>,
) -> Result<PcuTensor<T>, PcuExecutionError> {
    let sum = first + second;
    Ok(sum * third)
}

#[test]
#[allow(clippy::type_complexity)] // Verify the generated signature matches three ordinary moves.
fn three_owner_signature_is_native_rust() {
    let _: fn(
        PcuTensor<f64>,
        PcuTensor<f64>,
        PcuTensor<f64>,
    ) -> Result<PcuTensor<f64>, PcuExecutionError> = selected_pair;
}

#[test]
#[cfg(feature = "rocm")]
#[ignore = "requires ROCm hardware"]
fn selected_nonprefix_owners_ignore_unused_roots_and_retain_results() {
    global::use_defaults().unwrap();
    global::clear_thread_cache().unwrap();
    let unused = seed(&[99.0_f64; 2]).unwrap();
    global::clear_thread_cache().unwrap();
    let pair = selected_pair(
        unused,
        seed(&[16_777_219.0_f64, -16_777_217.0, 0.125, 8.5]).unwrap(),
        seed(&[1.0_f64, -2.0, 0.25, 10.0]).unwrap(),
    )
    .unwrap();
    let changed = selected_pair(
        seed(&[-99.0_f64; 2]).unwrap(),
        seed(&[4.0_f64, -8.0, 0.5, 3.0]).unwrap(),
        seed(&[-2.0_f64, 1.0, 0.25, -7.0]).unwrap(),
    )
    .unwrap();
    let old_first = seed(&[88.0_f64; 2]).unwrap();
    let old_second = seed(&[77.0_f64; 3]).unwrap();
    global::clear_thread_cache().unwrap();
    let last = selected_last(
        old_first,
        old_second,
        seed(&[-0.0_f64, 16_777_219.0, 0.125, -8.5]).unwrap(),
    )
    .unwrap();
    let fresh = all_selected(
        seed(&[1.0_f64, 2.0, -3.0, 0.5]).unwrap(),
        seed(&[2.0_f64, -4.0, 1.0, 0.5]).unwrap(),
        seed(&[3.0_f64, 0.5, -2.0, 4.0]).unwrap(),
    )
    .unwrap();
    let old_selected = seed(&[1.0_f64; 4]).unwrap();
    global::clear_thread_cache().unwrap();
    let rejected = selected_pair(
        seed(&[99.0_f64; 2]).unwrap(),
        old_selected,
        seed(&[2.0_f64; 4]).unwrap(),
    )
    .unwrap_err();
    assert!(matches!(
        rejected,
        PcuExecutionError::Argument(global::PcuArgumentError::SessionMismatch)
    ));
    global::clear_thread_cache().unwrap();
    for (owner, expected) in [
        (pair, [16_777_218.0_f64, -16_777_215.0, -0.125, -1.5]),
        (changed, [6.0_f64, -9.0, 0.25, 10.0]),
        (last, [-0.0_f64, 16_777_219.0, 0.125, -8.5]),
        (fresh, [9.0_f64, -1.0, 4.0, 4.0]),
    ] {
        let mut actual = [0.0_f64; 4];
        owner.read_into(&mut actual).unwrap();
        assert_eq!(actual.map(f64::to_bits), expected.map(f64::to_bits));
    }
}
