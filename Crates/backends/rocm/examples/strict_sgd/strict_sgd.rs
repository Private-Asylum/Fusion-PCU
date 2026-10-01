//! Ordinary authored checked `ROCm` optimizer with resident output and explicit errors.
extern crate pcu_facade as fusion_pcu;
#[rustfmt::skip]
use fusion_pcu::{
    global,
    pcu,
    PcuExecutionError,
    PcuScalar,
    PcuTensor,
};
#[pcu(flag(strict))]
fn update<T: PcuScalar, const N: usize>(
    weights: &[T; N],
    gradient: &[T; N],
) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::sgd_update(weights, gradient, 0.5_f32)
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Rocm,
        device: Some(0),
        ..Default::default()
    })?;
    let result = update(&[2.0_f32, 4.0], &[1.0, 2.0])?;
    let mut f32_output = [0.0_f32; 2];
    result.read_into(&mut f32_output)?;
    assert_eq!(
        f32_output.map(f32::to_bits),
        [1.5_f32, 3.0].map(f32::to_bits)
    );
    println!("F32 checked SGD: {f32_output:?}");
    let result = update(&[2.0_f64, 4.0], &[1.0, 2.0])?;
    let mut f64_output = [0.0_f64; 2];
    result.read_into(&mut f64_output)?;
    assert_eq!(
        f64_output.map(f64::to_bits),
        [1.5_f64, 3.0].map(f64::to_bits)
    );
    println!("F64 checked SGD: {f64_output:?}");
    let error = update(&[1.0_f32], &[f32::from_bits(1)]).unwrap_err();
    println!("Checked tiny-inexact multiply: {error}");
    let result = update(&[2.0_f32], &[1.0])?;
    let mut retry = [0.0_f32];
    result.read_into(&mut retry)?;
    assert_eq!(retry.map(f32::to_bits), [1.5_f32.to_bits()]);
    global::clear_thread_cache()?;
    Ok(())
}
