//! An ordinary F64 optimizer function with explicit native numerical permission.
extern crate pcu_facade as fusion_pcu;
#[rustfmt::skip]
use fusion_pcu::{global, pcu, PcuExecutionError, PcuTensor};
#[pcu(flag(native_compound), flag(preserve_precision))]
fn update(weights: &[f64; 2], gradient: &[f64; 2]) -> Result<PcuTensor<f64>, PcuExecutionError> {
    pcu::sgd_update(weights, gradient, -0.25_f32)
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Cuda,
        ..Default::default()
    })?;
    let updated = update(&[1.0_f64, -0.5], &[0.5_f64, 1.0])?;
    let mut actual = [0.0_f64; 2];
    updated.read_into(&mut actual)?;
    assert_eq!(
        actual.map(f64::to_bits),
        [1.125_f64.to_bits(), (-0.25_f64).to_bits()]
    );
    println!("Native F64 updated weights: {actual:?}");
    Ok(())
}
