//! Mixed ownership uses the selected graph inputs, rather than authored argument count.
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
    pcu::identity(input)
}

#[pcu]
fn right_donor<T: PcuScalar>(
    _unused: &PcuTensor<T>,
    left: &PcuTensor<T>,
    right: PcuTensor<T>,
) -> Result<PcuTensor<T>, PcuExecutionError> {
    Ok(left - right)
}

#[pcu]
fn sparse_four<T: PcuScalar>(
    _owner: PcuTensor<T>,
    _ram: &[T; 2],
    left: &PcuTensor<T>,
    right: PcuTensor<T>,
) -> Result<PcuTensor<T>, PcuExecutionError> {
    Ok(left - right)
}

#[pcu]
fn selected_pair<T: PcuScalar>(
    _unused: &PcuTensor<T>,
    left: PcuTensor<T>,
    right: PcuTensor<T>,
) -> Result<PcuTensor<T>, PcuExecutionError> {
    Ok(left + right)
}

#[pcu]
fn ram_peer<T: PcuScalar, const N: usize>(
    _unused: &PcuTensor<T>,
    left: PcuTensor<T>,
    right: &[T; N],
) -> Result<PcuTensor<T>, PcuExecutionError> {
    Ok(left * right)
}

#[pcu]
fn selected_owner<T: PcuScalar>(
    _unused: &PcuTensor<T>,
    _ram: &[T; 2],
    selected: PcuTensor<T>,
) -> Result<PcuTensor<T>, PcuExecutionError> {
    Ok(selected)
}

#[pcu]
fn selected_borrows<T: PcuScalar>(
    _unused: PcuTensor<T>,
    left: &PcuTensor<T>,
    right: &PcuTensor<T>,
) -> Result<PcuTensor<T>, PcuExecutionError> {
    Ok(left + right)
}

#[pcu]
fn three_selected<T: PcuScalar>(
    owner: PcuTensor<T>,
    peer: &PcuTensor<T>,
    tail: &[T; 4],
) -> Result<PcuTensor<T>, PcuExecutionError> {
    let sum = owner + peer;
    Ok(sum * tail)
}

#[test]
#[allow(clippy::type_complexity)] // Exact lifetimes and ownership are the source API contract.
fn mixed_signatures_keep_native_borrows_and_moves() {
    let _: fn(
        &PcuTensor<f64>,
        &PcuTensor<f64>,
        PcuTensor<f64>,
    ) -> Result<PcuTensor<f64>, PcuExecutionError> = right_donor;
    let _: fn(
        &PcuTensor<f32>,
        PcuTensor<f32>,
        &[f32; 4],
    ) -> Result<PcuTensor<f32>, PcuExecutionError> = ram_peer;
}

#[test]
#[cfg(feature = "rocm")]
#[ignore = "requires ROCm hardware"]
fn mixed_selected_inputs_preserve_readonly_peers_and_escaped_results() {
    global::use_defaults().unwrap();
    global::clear_thread_cache().unwrap();
    let unused = seed(&[99.0_f64; 2]).unwrap();
    let unused_moved = seed(&[88.0_f64; 3]).unwrap();
    global::clear_thread_cache().unwrap();
    let values = [16_777_219.0_f64, -16_777_217.0, 0.125, 8.5];
    let peer = seed(&values).unwrap();
    let right = right_donor(&unused, &peer, seed(&[1.0_f64, -2.0, 0.25, 10.0]).unwrap()).unwrap();
    let sparse = sparse_four(
        unused_moved,
        &[0.0_f64; 2],
        &peer,
        seed(&[1.0_f64, -2.0, 0.25, 10.0]).unwrap(),
    )
    .unwrap();
    let changed = right_donor(
        &unused,
        &peer,
        seed(&[-1.0_f64, 2.0, -0.25, -10.0]).unwrap(),
    )
    .unwrap();
    let ram = ram_peer(
        &unused,
        seed(&[2.0_f64, -3.0, 0.5, 8.0]).unwrap(),
        &[3.0_f64, 2.0, 4.0, 0.5],
    )
    .unwrap();
    let ram_changed = ram_peer(
        &unused,
        seed(&[4.0_f64, -2.0, 1.5, 0.25]).unwrap(),
        &[2.0_f64, 3.0, 0.5, 4.0],
    )
    .unwrap();
    let last = selected_owner(
        &unused,
        &[0.0_f64; 2],
        seed(&[-0.0_f64, 16_777_219.0, 0.125, -8.5]).unwrap(),
    )
    .unwrap();
    let pair = selected_pair(
        &unused,
        seed(&[1.0_f64, 4.0, -0.5, -2.0]).unwrap(),
        seed(&[2.0_f64, -3.0, 0.5, 8.0]).unwrap(),
    )
    .unwrap();
    let shape_error = ram_peer(&unused, seed(&[1.0_f64; 4]).unwrap(), &[2.0_f64; 3]).unwrap_err();
    assert!(matches!(shape_error, PcuExecutionError::TensorBuild(
        fusion_pcu::dialect::tensor::TensorError::ShapeMismatch { left, right }
    ) if left == [4] && right == [3]));
    let discarded = seed(&[77.0_f64; 3]).unwrap();
    global::clear_thread_cache().unwrap();
    let left = seed(&[1.0_f64, 2.0, 3.0, 4.0]).unwrap();
    let other = seed(&[4.0_f64, 3.0, 2.0, 1.0]).unwrap();
    let borrowed = selected_borrows(discarded, &left, &other).unwrap();
    let fresh = three_selected(
        seed(&[2.0_f64, -3.0, 0.5, 8.0]).unwrap(),
        &left,
        &[3.0_f64, 2.0, 4.0, 0.5],
    )
    .unwrap();
    let rejected = right_donor(&unused, &peer, seed(&[2.0_f64; 4]).unwrap()).unwrap_err();
    assert!(matches!(
        rejected,
        PcuExecutionError::Argument(global::PcuArgumentError::SessionMismatch)
    ));
    global::clear_thread_cache().unwrap();
    for (owner, expected) in [
        (right, [16_777_218.0_f64, -16_777_215.0, -0.125, -1.5]),
        (sparse, [16_777_218.0_f64, -16_777_215.0, -0.125, -1.5]),
        (changed, [16_777_220.0_f64, -16_777_219.0, 0.375, 18.5]),
        (ram, [6.0_f64, -6.0, 2.0, 4.0]),
        (ram_changed, [8.0_f64, -6.0, 0.75, 1.0]),
        (pair, [3.0_f64, 1.0, 0.0, 6.0]),
        (last, [-0.0_f64, 16_777_219.0, 0.125, -8.5]),
        (borrowed, [5.0_f64; 4]),
        (fresh, [9.0_f64, -2.0, 14.0, 6.0]),
    ] {
        read(&owner, expected);
    }
    read(&peer, values);
    read(&left, [1.0_f64, 2.0, 3.0, 4.0]);
    read(&other, [4.0_f64, 3.0, 2.0, 1.0]);
    read(&unused, [99.0_f64; 2]);
}

#[cfg(feature = "rocm")]
fn read<const N: usize>(owner: &PcuTensor<f64>, expected: [f64; N]) {
    let mut actual = [0.0_f64; N];
    owner.read_into(&mut actual).unwrap();
    assert_eq!(actual.map(f64::to_bits), expected.map(f64::to_bits));
}
