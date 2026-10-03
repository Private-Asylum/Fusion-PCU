//! Ordinary immutable owned `ReLU` with host staging and escaped lifetime.
#[rustfmt::skip]
use pcu_facade::{
    global,
    pcu,
    PcuTensor,
    PcuExecutionError,
};
#[pcu(crate_path=::pcu_facade)]
fn activate(input: &[f64]) -> Result<PcuTensor<f64>, PcuExecutionError> {
    pcu::relu(input)
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Mlx,
        ..Default::default()
    })?;
    let input = [-3.0, -0.0, 1.5, 2.0];
    let result = activate(&input)?;
    global::clear_thread_cache()?;
    let mut output = [19.0; 6];
    result.read_into(&mut output)?;
    // Checked ReLU selects canonical positive zero for both signed-zero inputs.
    assert_eq!(
        output.map(f64::to_bits),
        [0.0_f64, 0.0, 1.5, 2.0, 19.0, 19.0].map(f64::to_bits)
    );
    println!("MLX owned `ReLU` shape {:?}: {output:?}", result.shape());
    global::use_defaults()?;
    Ok(())
}
