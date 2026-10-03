//! Ordinary strict checked CPU training: every intermediate uses the same frozen plan.
extern crate pcu_facade as fusion_pcu;
#[path = "../../benches/checked_tensor/source/source.rs"]
#[allow(dead_code)] // Policy variants are exercised by the paired benchmark/source test.
mod source;
#[rustfmt::skip]
use fusion_pcu::{
    global,
    global::PcuBackendChoice,
    global::PcuExecutionPolicy,
};
fn main() {
    global::configure(PcuExecutionPolicy {
        backend: PcuBackendChoice::Cpu,
        ..PcuExecutionPolicy::default()
    })
    .unwrap();
    let output = source::training(
        &[[1.0_f64, 2.0], [3.0, 4.0]],
        &[[1.0, 3.0], [2.0, 4.0]],
        &[[0.5], [0.25]],
        &[[0.0], [1.0]],
    )
    .unwrap();
    let mut weights = [0.0_f64; 2];
    output.read_into(&mut weights).unwrap();
    assert_eq!(
        weights.map(f64::to_bits),
        [-2.25_f64, -3.75].map(f64::to_bits)
    );
    println!("Checked F64 CPU training weights: {weights:?}");
}
