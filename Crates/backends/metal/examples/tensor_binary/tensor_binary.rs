//! Ordinary checked source retains independently owned binary results across cache clearing.
#[path = "../../tests/tensor_binary/source/source.rs"]
mod source;
use pcu_facade::global;
fn main() {
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Metal,
        ..Default::default()
    })
    .unwrap();
    let left = [1.0_f64, 2.0, 3.0];
    let right = [3.0_f64, 2.0, 1.0];
    let first = source::add(&left, &right).unwrap();
    let second = source::add::<f64>(&first, &right).unwrap();
    drop(first);
    global::clear_thread_cache().unwrap();
    let mut output = [91.0; 5];
    second.read_into(&mut output).unwrap();
    assert_eq!(
        output.map(f64::to_bits),
        [7.0_f64, 6.0, 5.0, 91.0, 91.0].map(f64::to_bits)
    );
    println!("Ordinary source checked binary owner: {output:?}");
    global::use_defaults().unwrap();
}
