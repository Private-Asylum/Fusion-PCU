//! Marked helper calls preserve the source-level distinction between moving and borrowing owners.
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
fn borrow_helper<T: PcuScalar>(input: &PcuTensor<T>) -> Result<PcuTensor<T>, PcuExecutionError> {
    Ok(pcu::identity(input)?)
}

#[pcu]
fn consume_helper<T: PcuScalar>(input: PcuTensor<T>) -> Result<PcuTensor<T>, PcuExecutionError> {
    Ok(pcu::relu(input)?)
}

#[pcu]
fn borrow_then_consume<T: PcuScalar>(
    input: PcuTensor<T>,
) -> Result<PcuTensor<T>, PcuExecutionError> {
    let borrowed = borrow_helper(&input)?;
    let consumed = consume_helper(input)?;
    Ok(pcu::add(borrowed, consumed)?)
}

#[pcu]
fn renamed_owner_then_consume<T: PcuScalar>(
    input: PcuTensor<T>,
) -> Result<PcuTensor<T>, PcuExecutionError> {
    let owner = input;
    let alias = owner;
    Ok(consume_helper(alias)?)
}

#[pcu]
fn borrowed_view_last_use_then_consume<T: PcuScalar>(
    input: PcuTensor<T>,
) -> Result<PcuTensor<T>, PcuExecutionError> {
    let owner = input;
    let view = &owner;
    let borrowed = borrow_helper(view)?;
    let consumed = consume_helper(owner)?;
    Ok(pcu::add(borrowed, consumed)?)
}

#[pcu]
fn borrow_operation_rvalue<T: PcuScalar>(
    input: PcuTensor<T>,
) -> Result<PcuTensor<T>, PcuExecutionError> {
    Ok(borrow_helper(&pcu::relu(&input)?)?)
}

#[pcu]
fn seed<T: PcuScalar, const N: usize>(input: &[T; N]) -> Result<PcuTensor<T>, PcuExecutionError> {
    Ok(pcu::identity(input)?)
}

#[test]
fn generated_helper_modes_keep_native_owner_signatures() {
    let _: fn(PcuTensor<f32>) -> Result<PcuTensor<f32>, PcuExecutionError> = borrow_then_consume;
    let _: fn(PcuTensor<f64>) -> Result<PcuTensor<f64>, PcuExecutionError> = consume_helper;
    let _: fn(&PcuTensor<f32>) -> Result<PcuTensor<f32>, PcuExecutionError> = borrow_helper;
    let _: fn(PcuTensor<f32>) -> Result<PcuTensor<f32>, PcuExecutionError> =
        renamed_owner_then_consume;
    let _: fn(PcuTensor<f64>) -> Result<PcuTensor<f64>, PcuExecutionError> =
        borrowed_view_last_use_then_consume;
    let _: fn(PcuTensor<f32>) -> Result<PcuTensor<f32>, PcuExecutionError> =
        borrow_operation_rvalue::<f32>;
}

#[test]
#[cfg(feature = "rocm")]
#[ignore = "requires ROCm hardware"]
fn marked_borrow_then_consume_helpers_preserve_results_and_roots() {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let _guard = LOCK.lock().unwrap();
    global::use_defaults().unwrap();
    global::clear_thread_cache().unwrap();

    let first_values = [-3.0_f32, 2.0, -0.0, 7.5];
    let first_owner = seed(&first_values).unwrap();
    let first = borrow_then_consume(first_owner).unwrap();

    let changed_values = [4.0_f32, -5.0, 0.25, -1.0];
    let changed_owner = seed(&changed_values).unwrap();
    let changed = borrow_then_consume(changed_owner).unwrap();

    let f64_values = [-16_777_217.0_f64, 16_777_219.0, -0.0, 0.125];
    let f64_owner = seed(&f64_values).unwrap();
    let f64_result = borrow_then_consume(f64_owner).unwrap();

    let renamed_owner = seed(&[-4.0_f32, -2.0, 3.0, -0.0]).unwrap();
    let renamed_result = renamed_owner_then_consume(renamed_owner).unwrap();
    let borrow_last_owner = seed(&[1.0_f64, -2.0, 3.0, -0.0]).unwrap();
    let borrow_last_result = borrowed_view_last_use_then_consume(borrow_last_owner).unwrap();

    let rvalue_owner = seed(&[-4.0_f64, 2.0, -0.0, 0.125]).unwrap();
    let rvalue_result = borrow_operation_rvalue(rvalue_owner).unwrap();

    global::clear_thread_cache().unwrap();
    let mut observed = [f32::NAN; 4];
    first.read_into(&mut observed).unwrap();
    assert_eq!(
        observed.map(f32::to_bits),
        [-3.0, 4.0, 0.0, 15.0].map(f32::to_bits)
    );
    changed.read_into(&mut observed).unwrap();
    assert_eq!(
        observed.map(f32::to_bits),
        [8.0, -5.0, 0.5, -1.0].map(f32::to_bits)
    );

    let mut observed_f64 = [f64::NAN; 4];
    f64_result.read_into(&mut observed_f64).unwrap();
    assert_eq!(
        observed_f64.map(f64::to_bits),
        [-16_777_217.0, 33_554_438.0, 0.0, 0.25].map(f64::to_bits)
    );
    renamed_result.read_into(&mut observed).unwrap();
    assert_eq!(
        observed.map(f32::to_bits),
        [0.0, 0.0, 3.0, 0.0].map(f32::to_bits)
    );
    rvalue_result.read_into(&mut observed_f64).unwrap();
    assert_eq!(
        observed_f64.map(f64::to_bits),
        [0.0, 2.0, 0.0, 0.125].map(f64::to_bits)
    );
    borrow_last_result.read_into(&mut observed_f64).unwrap();
    assert_eq!(
        observed_f64.map(f64::to_bits),
        [2.0, -2.0, 6.0, 0.0].map(f64::to_bits)
    );
}
