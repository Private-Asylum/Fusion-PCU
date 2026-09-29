//! Lexical move/borrow behavior for captured operations on one consumed resident source.
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
fn borrowed_seed<T: PcuScalar, const N: usize>(
    input: &[T; N],
) -> Result<PcuTensor<T>, PcuExecutionError> {
    Ok(pcu::identity(input)?)
}

#[pcu]
fn borrow_then_move<T: PcuScalar>(input: PcuTensor<T>) -> Result<PcuTensor<T>, PcuExecutionError> {
    let borrowed = pcu::identity(&input)?;
    let consumed = pcu::relu(input)?;
    Ok(pcu::add(borrowed, consumed)?)
}

#[pcu]
fn consumed_fanout<T: PcuScalar>(input: PcuTensor<T>) -> Result<PcuTensor<T>, PcuExecutionError> {
    let activated = pcu::relu(input)?;
    let copied = pcu::identity(&activated)?;
    let relu_again = pcu::relu(&activated)?;
    let joined = pcu::add(copied, relu_again)?;
    Ok(consume_produced(joined)?)
}

#[pcu]
fn consume_produced<T: PcuScalar>(input: PcuTensor<T>) -> Result<PcuTensor<T>, PcuExecutionError> {
    Ok(pcu::relu(input)?)
}

#[pcu]
fn produced_owner_pipeline<T: PcuScalar>(
    input: PcuTensor<T>,
) -> Result<PcuTensor<T>, PcuExecutionError> {
    let multiplied = pcu::mul(&input, &input)?;
    let added = pcu::add(multiplied, input)?;
    let activated = pcu::relu(added)?;
    Ok(consume_produced(activated)?)
}

#[pcu]
fn consumed_return<T: PcuScalar>(input: PcuTensor<T>) -> Result<PcuTensor<T>, PcuExecutionError> {
    Ok(input)
}

#[pcu]
fn matrix_seed<const R: usize, const C: usize>(
    input: &[[f32; C]; R],
) -> Result<PcuTensor<f32>, PcuExecutionError> {
    Ok(pcu::identity(input)?)
}

#[test]
#[cfg(not(feature = "rocm"))]
fn lexical_consumed_composition_compiles_without_a_host_fallback() {
    let _: fn(PcuTensor<f32>) -> Result<PcuTensor<f32>, PcuExecutionError> = borrow_then_move;
    let _: fn(PcuTensor<f64>) -> Result<PcuTensor<f64>, PcuExecutionError> = consumed_fanout;
    let _: fn(PcuTensor<f32>) -> Result<PcuTensor<f32>, PcuExecutionError> = consumed_return;
    let _: fn(PcuTensor<f64>) -> Result<PcuTensor<f64>, PcuExecutionError> =
        produced_owner_pipeline;
}

#[test]
#[cfg(feature = "rocm")]
#[ignore = "requires ROCm hardware"]
fn consumed_source_borrows_then_moves_and_ssa_fanout_retains_old_outputs() {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let _guard = LOCK.lock().unwrap();
    global::use_defaults().unwrap();
    global::clear_thread_cache().unwrap();

    let first_source = [-2.0_f32, 3.0, -0.0, 8.5];
    let first_owner = borrowed_seed(&first_source).unwrap();
    let first = borrow_then_move(first_owner).unwrap();

    let changed_source = [4.0_f32, -5.0, 0.25, -1.0];
    let changed_owner = borrowed_seed(&changed_source).unwrap();
    let changed = borrow_then_move(changed_owner).unwrap();

    let fanout_source = [-3.0_f32, 2.0, -0.0, 4.5];
    let fanout_owner = borrowed_seed(&fanout_source).unwrap();
    let fanout = consumed_fanout(fanout_owner).unwrap();

    let pipeline_source = [-2.0_f32, 3.0, -0.0, 0.125];
    let pipeline_owner = borrowed_seed(&pipeline_source).unwrap();
    let pipeline = produced_owner_pipeline(pipeline_owner).unwrap();

    let returned_source = [
        f32::from_bits(0x8000_0000),
        f32::from_bits(0x7fc1_2345),
        5.0,
        -7.0,
    ];
    let returned_owner = borrowed_seed(&returned_source).unwrap();
    let returned = consumed_return(returned_owner).unwrap();

    let f64_source = [-16_777_217.0_f64, 16_777_219.0, -0.0, 0.125];
    let f64_owner = borrowed_seed(&f64_source).unwrap();
    let f64_output = borrow_then_move(f64_owner).unwrap();
    let f64_pipeline_source = [-2.0_f64, 3.0, -0.0, 0.125];
    let f64_pipeline_owner = borrowed_seed(&f64_pipeline_source).unwrap();
    let f64_pipeline = produced_owner_pipeline(f64_pipeline_owner).unwrap();

    global::clear_thread_cache().unwrap();
    let mut observed = [0.0_f32; 4];
    first.read_into(&mut observed).unwrap();
    assert_eq!(
        observed.map(f32::to_bits),
        [-2.0, 6.0, 0.0, 17.0].map(f32::to_bits)
    );
    changed.read_into(&mut observed).unwrap();
    assert_eq!(
        observed.map(f32::to_bits),
        [8.0, -5.0, 0.5, -1.0].map(f32::to_bits)
    );
    fanout.read_into(&mut observed).unwrap();
    assert_eq!(
        observed.map(f32::to_bits),
        [0.0, 4.0, 0.0, 9.0].map(f32::to_bits)
    );
    pipeline.read_into(&mut observed).unwrap();
    assert_eq!(
        observed.map(f32::to_bits),
        [2.0, 12.0, 0.0, 0.140_625].map(f32::to_bits)
    );
    returned.read_into(&mut observed).unwrap();
    assert_eq!(
        observed.map(f32::to_bits),
        returned_source.map(f32::to_bits)
    );

    let mut observed_f64 = [0.0_f64; 4];
    f64_output.read_into(&mut observed_f64).unwrap();
    assert_eq!(
        observed_f64.map(f64::to_bits),
        [-16_777_217.0, 33_554_438.0, 0.0, 0.25].map(f64::to_bits)
    );
    f64_pipeline.read_into(&mut observed_f64).unwrap();
    assert_eq!(
        observed_f64.map(f64::to_bits),
        [2.0, 12.0, 0.0, 0.140_625].map(f64::to_bits)
    );

    let matrix = [[1.0_f32, 2.0], [3.0, 4.0]];
    let rank_two = matrix_seed(&matrix).unwrap();
    let rank_two_result = consumed_return(rank_two).unwrap();
    assert_eq!(rank_two_result.shape(), [2, 2]);
    global::clear_thread_cache().unwrap();
    rank_two_result.read_into(&mut observed).unwrap();
    assert_eq!(
        observed.map(f32::to_bits),
        [1.0, 2.0, 3.0, 4.0].map(f32::to_bits)
    );
}
