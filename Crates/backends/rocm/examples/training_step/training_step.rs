//! Ordinary source forward, checked loss, backward and optimizer on an explicit provider.
extern crate pcu_facade as fusion_pcu;
#[allow(dead_code)] // Reuse the exact source acceptance function; other policy entrypoints are fixtures.
#[path = "../../benches/strict_mse/source/source.rs"]
mod source;
use fusion_pcu::global;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Rocm,
        device: Some(0),
        ..Default::default()
    })?;
    let input = [[1.0_f32, 0.0], [0.0, 1.0]];
    let weights = [[2.0_f32], [-1.0]];
    let target = [[1.0_f32], [0.0]];
    let updated = source::training_step(&input, &input, &weights, &target)?;
    let mut actual = [0.0_f32; 2];
    updated.read_into(&mut actual)?;
    assert_eq!(
        actual.map(f32::to_bits),
        [1.5_f32.to_bits(), (-1.0_f32).to_bits()]
    );
    println!(
        "{:?} weights after forward/loss/backward/SGD: {actual:?}",
        global::PcuBackendChoice::Rocm
    );
    Ok(())
}
