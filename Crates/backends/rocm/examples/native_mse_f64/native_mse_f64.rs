//! Ordinary F64 projection/loss under explicitly native numerical permission.
extern crate pcu_facade as fusion_pcu;
#[rustfmt::skip]
use fusion_pcu::{
    global,
    pcu,
    PcuExecutionError,
    PcuTensor,
};
#[pcu(flag(native_compound), flag(preserve_precision))]
fn loss(
    left: &[[f64; 2]; 2],
    right: &[[f64; 2]; 2],
    target: &[[f64; 2]; 2],
) -> Result<PcuTensor<f64>, PcuExecutionError> {
    let prediction = pcu::matmul(left, right)?;
    pcu::mean_squared_error(&prediction, target)
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Rocm,
        ..Default::default()
    })?;
    let output = loss(
        &[[1.0, 2.0], [3.0, 4.0]],
        &[[1.0, 0.0], [0.0, 1.0]],
        &[[0.0; 2]; 2],
    )?;
    let mut actual = [0.0_f64];
    output.read_into(&mut actual)?;
    assert!(output.shape().is_empty());
    assert_eq!(actual[0].to_bits(), 7.5_f64.to_bits());
    println!("Native F64 projection loss: {actual:?}");
    Ok(())
}
