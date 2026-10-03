//! Automatic RAM staging and ordinary ownership of low-format MLX resident results.
#[rustfmt::skip]
use fusion_pcu::{
    global,
    pcu,
    PcuExecutionError,
    PcuF16Bits,
    PcuScalar,
    PcuTensor,
};

#[pcu]
fn retain<T: PcuScalar>(input: &[T]) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::identity(input)
}

#[pcu]
fn activate<T: PcuScalar>(input: &[T]) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::relu(input)
}

#[pcu]
fn finish<T: PcuScalar>(input: PcuTensor<T>) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::identity(input)
}

fn main() -> Result<(), PcuExecutionError> {
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Mlx,
        ..Default::default()
    })?;
    let input = [1.0, -2.0, 0.0, 3.0, -4.0].map(PcuF16Bits::from_f32);
    // PCU stages RAM into an MLX-owned encoded allocation; there is no consumer upload call.
    let retained = retain(&input)?;
    // Borrowing pins the existing allocation. ReLU executes against that resident input.
    let activated = activate::<PcuF16Bits>(&retained)?;
    drop(retained); // Its logical claim ends; the completed activation remains independently live.
    let result = finish(activated)?; // Move transfers ownership; identity needs no RAM round trip.
    global::clear_thread_cache()?; // Escaped owners retain the actual runtime outside the cache.
    let mut stack = [PcuF16Bits::from_bits(0); 5];
    result.read_into(&mut stack)?; // This explicit boundary materializes the result in caller RAM.
    drop(result); // Device ownership ends; the stack result remains ordinary Rust memory.
    let values = stack.map(PcuF16Bits::to_f32);
    assert_eq!(
        values.map(f32::to_bits),
        [1.0_f32, 0.0, 0.0, 3.0, 0.0].map(f32::to_bits)
    );
    println!("Completed MLX result in stack RAM: {values:?}");
    Ok(())
}
