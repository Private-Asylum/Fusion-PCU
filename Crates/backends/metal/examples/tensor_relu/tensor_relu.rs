//! Ordinary source owned checked `ReLU` with retained-device composition and tails.
#[path = "../../benches/tensor_relu/source/source.rs"]
mod source;
use pcu_facade::global;
fn main() {
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Metal,
        ..Default::default()
    })
    .unwrap();
    let input = [-1.0_f64, -0.0, 1.0];
    let old = source::relu(&input).unwrap();
    let owner = source::relu::<f64>(&old).unwrap();
    drop(old);
    global::clear_thread_cache().unwrap();
    let mut output = [91.0; 5];
    owner.read_into(&mut output).unwrap();
    assert_eq!(
        output.map(f64::to_bits),
        [0.0_f64, 0.0, 1.0, 91.0, 91.0].map(f64::to_bits)
    );
    println!("Ordinary source terminal Metal output: {output:?}");
    global::use_defaults().unwrap();
}
