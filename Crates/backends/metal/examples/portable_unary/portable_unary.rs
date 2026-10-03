//! Actual deterministic checked `ReLU` keeps positive subnormal bits and canonicalizes -0.
#[rustfmt::skip]
use pcu_facade::{
    global,
    pcu,
};
#[pcu(invocations = 3, flag(deterministic), crate_path = ::pcu_facade)]
fn relu(input: &[f64], output: &mut [f64]) {
    let id = pcu::context::global_invocation_id();
    output[id] = pcu::relu(input[id]);
}
fn main() {
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Metal,
        ..Default::default()
    })
    .unwrap();
    let mut output = [91.0; 5];
    relu(&[f64::from_bits(1), -0.0, -2.0], &mut output).unwrap();
    assert_eq!(
        output.map(f64::to_bits),
        [1, 0, 0, 91.0_f64.to_bits(), 91.0_f64.to_bits()]
    );
    global::clear_thread_cache().unwrap();
    println!(
        "Metal Portable unary exact bits {:?}",
        output.map(f64::to_bits)
    );
}
