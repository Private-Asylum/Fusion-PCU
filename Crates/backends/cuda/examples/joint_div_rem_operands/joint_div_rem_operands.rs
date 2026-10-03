//! Ordinary full-width joint division with exact requested Portable policy and both results.
extern crate pcu_facade as fusion_pcu;
#[path = "../../benches/joint_div_rem_operands/oracle/oracle.rs"]
mod oracle;
#[path = "../../benches/joint_div_rem_operands/source/source.rs"]
#[allow(dead_code)] // Canonical source companions share direct and grid profiles.
mod source;
#[rustfmt::skip]
use fusion_pcu::{
    global,
    PcuI512,
    PcuNumericalMode,
    PcuNumericalOptions,
    PcuReproducibility,
};
fn main() {
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Cuda,
        device: Some(0),
        numerical_mode: PcuNumericalMode::Strict,
        numerical_options: PcuNumericalOptions {
            reproducibility: PcuReproducibility::PortableV1,
            ..Default::default()
        },
        ..Default::default()
    })
    .unwrap();
    let mut quotient = vec![<PcuI512 as oracle::Format>::SENTINEL; 67];
    let mut remainder = quotient.clone();
    for phase in [1, 17] {
        let (left, right, want_q, want_r) = oracle::inputs::<PcuI512>(65, phase, 2);
        source::reordered::<PcuI512, 65>(&right, &mut remainder, &left, &mut quotient).unwrap();
        oracle::verify(&want_q, &want_r, &quotient, &remainder);
    }
    println!(
        "Two changing I512 quotient/remainder source calls match independent full-bit results; tails preserved."
    );
}
