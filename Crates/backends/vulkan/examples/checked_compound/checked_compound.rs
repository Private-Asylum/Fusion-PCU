//! Genuine annotated ordered MatMul/MSE/SGD chain with retained source owners.
extern crate pcu_facade as fusion_pcu;
#[path = "../../benches/checked_compound/source/source.rs"]
#[allow(dead_code)] // Individual source peers are qualified by the paired test/benchmark targets.
mod source;
#[rustfmt::skip]
use pcu_facade::{
    global,
    PcuNumericalOptions,
    PcuCompoundArithmeticPolicy,
    PcuPrecisionPolicy,
};
fn main() {
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Vulkan,
        numerical_options: PcuNumericalOptions {
            compound_arithmetic: PcuCompoundArithmeticPolicy::BackendDefined,
            precision: PcuPrecisionPolicy::BackendOptimized,
            ..Default::default()
        },
        ..Default::default()
    })
    .unwrap();
    let left = source::retain_left(&[[-3.0_f64, -2.0, -1.0], [0.0, 1.0, 2.0]]).unwrap();
    let right = source::retain_right(&[[2.0_f64, -1.0], [-3.0, 2.0], [1.0, 4.0]]).unwrap();
    let target = source::identity(&[[1.0_f64, 2.0], [-1.0, 3.0]]).unwrap();
    let updated = source::chain(&left, &right, &target).unwrap();
    global::clear_thread_cache().unwrap();
    drop(left);
    drop(right);
    drop(target);
    let mut data = [117.0_f64; 7];
    updated.read_into(&mut data).unwrap();
    assert_eq!(
        data.map(f64::to_bits),
        [-1.5_f64, -6.0, -0.5, 8.5, 117.0, 117.0, 117.0].map(f64::to_bits)
    );
    println!(
        "Vulkan Strict F64 MatMul, checked-unused ordered MSE and SGD retained exact output after input drops and cache clear."
    );
}
