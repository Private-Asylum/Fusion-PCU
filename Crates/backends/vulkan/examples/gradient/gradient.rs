//! Actual target-selective differentiation; the caller supplies no gradient tensor.
extern crate pcu_facade as fusion_pcu;
#[rustfmt::skip]
use pcu_facade::{
    global,
};
#[path = "../../benches/gradient/source/source.rs"]
mod source;
use source::train;
fn main() {
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Vulkan,
        ..Default::default()
    })
    .unwrap();
    let input = [[1.0_f64, 0.0], [0.0, 1.0]];
    let target = [[1.0_f64], [0.0]];
    let first = train(&input, &[[3.0_f64], [-1.0]], &target).unwrap();
    let second = train(&input, &first, &target).unwrap();
    global::clear_thread_cache().unwrap();
    drop(first);
    let mut output = [117.0_f64; 5];
    second.read_into(&mut output).unwrap();
    assert_eq!(
        output.map(f64::to_bits),
        [1.5_f64, -1.0, 117.0, 117.0, 117.0].map(f64::to_bits)
    );
    println!(
        "Vulkan actual target-selective F64 MSE gradient and two Strict SGD steps retained exact output after cache clear."
    );
}
