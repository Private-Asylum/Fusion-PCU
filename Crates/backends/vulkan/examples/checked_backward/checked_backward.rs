//! Genuine source derivative preserves exact F64 representations without native floating math.
#[rustfmt::skip]
use pcu_facade::{
    global,
    pcu,
    PcuExecutionError,
    PcuTensor,
};
#[pcu(crate_path=::pcu_facade,flag(strict),flag(native_compound),flag(backend_precision),flag(allow_gradual_underflow))]
fn derivative(input: &[f64], upstream: &[f64]) -> Result<PcuTensor<f64>, PcuExecutionError> {
    pcu::relu_backward(input, upstream)
}
fn main() -> Result<(), PcuExecutionError> {
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Vulkan,
        ..Default::default()
    })?;
    let tiny = f64::from_bits(1);
    let owner = derivative(
        &[0.0, -0.0, 1.0, -1.0, 1.0],
        &[-0.0, -0.0, -0.0, f64::MAX, tiny],
    )?;
    global::clear_thread_cache()?;
    let mut observed = [17.0; 7];
    owner.read_into(&mut observed)?;
    assert_eq!(
        observed[..5]
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        [0.0_f64, 0.0, -0.0, 0.0, tiny].map(f64::to_bits)
    );
    assert_eq!(observed[5..], [17.0; 2]);
    println!(
        "Vulkan checked F64 ReLU derivative: both signed-zero laws and exact selected subnormal survived cache clear."
    );
    global::use_defaults()?;
    Ok(())
}
