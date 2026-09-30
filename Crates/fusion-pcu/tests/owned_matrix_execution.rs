//! Hardware conformance for fixed-shape owned matrix source entries.
#![cfg(all(feature = "rocm", feature = "tensor"))]

use fusion_pcu::pcu;
#[rustfmt::skip]
use fusion_pcu::{
    global,
    PcuArgumentError,
    PcuCheckedFloat,
    PcuExecutionError,
    PcuSourceShape,
    PcuTensor,
};

static POLICY_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[pcu]
fn copy_matrix<const R: usize, const C: usize>(
    input: &[[f32; C]; R],
) -> Result<PcuTensor<f32>, PcuExecutionError> {
    Ok(pcu::identity(input)?)
}

#[pcu]
fn copy_vector(input: &[f32]) -> Result<PcuTensor<f32>, PcuExecutionError> {
    Ok(pcu::identity(input)?)
}

#[pcu(flag(strict))]
fn gemm<const M: usize, const K: usize, const N: usize>(
    lhs: &[[f32; K]; M],
    rhs: &[[f32; N]; K],
) -> Result<PcuTensor<f32>, PcuExecutionError> {
    Ok(pcu::matmul(lhs, rhs)?)
}

#[pcu]
fn call_wrong_matrix_helper(input: &[[f32; 3]; 2]) -> Result<PcuTensor<f32>, PcuExecutionError> {
    let copied = copy_matrix::<3, 2>(input)?;
    Ok(copied)
}

#[pcu]
fn call_wrong_rank_helper(input: &[[f32; 3]; 2]) -> Result<PcuTensor<f32>, PcuExecutionError> {
    let copied = copy_vector(input)?;
    Ok(copied)
}

fn assert_bits(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert_eq!(actual.to_bits(), expected.to_bits());
    }
}

fn assert_gemm<const M: usize, const K: usize, const N: usize>(
    actual: &[f32],
    lhs: &[[f32; K]; M],
    rhs: &[[f32; N]; K],
) {
    assert_eq!(actual.len(), M * N);
    for (row, lhs_row) in lhs.iter().enumerate() {
        let mut expected = [0.0_f32; N];
        for (lhs_value, rhs_row) in lhs_row.iter().zip(rhs) {
            for (total, rhs_value) in expected.iter_mut().zip(rhs_row) {
                let product = lhs_value.pcu_checked_mul(*rhs_value).unwrap();
                *total = total.pcu_checked_add(product).unwrap();
            }
        }
        let actual_row = &actual[row * N..(row + 1) * N];
        for (actual, expected) in actual_row.iter().zip(expected) {
            assert_eq!(actual.to_bits(), expected.to_bits());
        }
    }
}

#[test]
fn helper_shape_contracts_fail_before_backend_discovery() {
    let _guard = POLICY_TEST_LOCK.lock().unwrap();
    global::use_defaults().unwrap();
    global::clear_thread_cache().unwrap();
    let input = [[1.0_f32, 2.0, 3.0], [4.0, 5.0, 6.0]];

    assert!(matches!(
        call_wrong_matrix_helper(&input),
        Err(PcuExecutionError::TensorSourceShapeMismatch {
            expected: PcuSourceShape::FixedMatrix {
                rows: 3,
                columns: 2,
            },
            actual,
        }) if actual == [2, 3]
    ));
    assert!(matches!(
        call_wrong_rank_helper(&input),
        Err(PcuExecutionError::TensorSourceRankMismatch {
            expected: 1,
            actual: 2,
        })
    ));
    global::clear_thread_cache().unwrap();
}

#[test]
#[ignore = "requires a working ROCm device and HIPRTC"]
fn matrix_shapes_survive_host_resident_and_mixed_capture() {
    let _guard = POLICY_TEST_LOCK.lock().unwrap();
    global::use_defaults().unwrap();
    global::clear_thread_cache().unwrap();

    let lhs = [[1.0_f32, 2.0, 3.0], [4.0, 5.0, 6.0]];
    let rhs = [[7.0_f32, 8.0], [9.0, 10.0], [11.0, 12.0]];
    let direct_product = gemm::<2, 3, 2>(&lhs, &rhs).unwrap();
    let changed_lhs = [[2.0_f32, 0.0, -1.0], [1.0, 3.0, 2.0]];
    let changed_rhs = [[1.0_f32, 0.0], [2.0, 1.0], [0.0, 2.0]];
    let warm_product = gemm::<2, 3, 2>(&changed_lhs, &changed_rhs).unwrap();
    let host_copy = copy_matrix::<2, 3>(&lhs).unwrap();
    assert_eq!(host_copy.shape(), [2, 3]);
    let copy_lhs = [[-1.0_f32, -2.0, -3.0], [9.0, 8.0, 7.0]];
    let warm_copy = copy_matrix::<2, 3>(&copy_lhs).unwrap();

    let boxed = std::boxed::Box::new(lhs);
    let boxed_copy = copy_matrix::<2, 3>(&boxed).unwrap();
    let shared = std::rc::Rc::new(lhs);
    let shared_copy = copy_matrix::<2, 3>(&shared).unwrap();
    let shared_arc = std::sync::Arc::new(lhs);
    let arc_copy = copy_matrix::<2, 3>(&shared_arc).unwrap();

    let right_owner = copy_matrix::<3, 2>(&rhs).unwrap();
    let resident_copy = copy_matrix::<3, 2>(&right_owner).unwrap();
    assert_eq!(resident_copy.shape(), [3, 2]);

    let mixed = gemm::<2, 3, 2>(&host_copy, &rhs).unwrap();
    assert_eq!(mixed.shape(), [2, 2]);
    let resident = gemm::<2, 3, 2>(&host_copy, &right_owner).unwrap();
    assert_eq!(resident.shape(), [2, 2]);

    let mut boxed_values = [f32::NAN; 6];
    let mut shared_values = [f32::NAN; 6];
    let mut arc_values = [f32::NAN; 6];
    let mut warm_values = [f32::NAN; 6];
    boxed_copy.read_into(&mut boxed_values).unwrap();
    shared_copy.read_into(&mut shared_values).unwrap();
    arc_copy.read_into(&mut arc_values).unwrap();
    warm_copy.read_into(&mut warm_values).unwrap();
    assert_bits(&boxed_values, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    assert_bits(&shared_values, &boxed_values);
    assert_bits(&arc_values, &boxed_values);
    assert_bits(&warm_values, &[-1.0, -2.0, -3.0, 9.0, 8.0, 7.0]);

    let mut product_values = [f32::NAN; 4];
    let mut direct_product_values = [f32::NAN; 4];
    let mut warm_product_values = [f32::NAN; 4];
    mixed.read_into(&mut product_values).unwrap();
    assert_bits(&product_values, &[58.0, 64.0, 139.0, 154.0]);
    direct_product
        .read_into(&mut direct_product_values)
        .unwrap();
    warm_product.read_into(&mut warm_product_values).unwrap();
    assert_gemm(&direct_product_values, &lhs, &rhs);
    assert_gemm(&warm_product_values, &changed_lhs, &changed_rhs);
    let mut resident_values = [f32::NAN; 4];
    resident.read_into(&mut resident_values).unwrap();
    assert_bits(&resident_values, &product_values);

    let transposed = copy_matrix::<3, 2>(&rhs).unwrap();
    assert!(matches!(
        copy_matrix::<2, 3>(&transposed),
        Err(PcuExecutionError::Argument(
            PcuArgumentError::ResidentShapeMismatch {
                expected: PcuSourceShape::FixedMatrix {
                    rows: 2,
                    columns: 3,
                },
            }
        ))
    ));
    let vector = copy_vector(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    assert!(matches!(
        copy_matrix::<2, 3>(&vector),
        Err(PcuExecutionError::Argument(
            PcuArgumentError::ResidentShapeMismatch { .. }
        ))
    ));

    global::clear_thread_cache().unwrap();
}
