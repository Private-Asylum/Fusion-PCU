//! Strict ordered compounds remain checked when other flags permit optimizations.
#[path = "../../benches/compound_permissions/source/source.rs"]
#[allow(dead_code)]
mod source;
#[rustfmt::skip]
use pcu_facade::{global,PcuCompoundArithmeticPolicy,PcuNumericalOptions,PcuPrecisionPolicy};
fn main() {
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Cpu,
        numerical_options: PcuNumericalOptions {
            compound_arithmetic: PcuCompoundArithmeticPolicy::BackendDefined,
            precision: PcuPrecisionPolicy::BackendOptimized,
            ..Default::default()
        },
        ..Default::default()
    })
    .unwrap();
    let input = [[1.0_f64, 2.0], [3.0, 4.0]];
    let identity = [[1.0, 0.0], [0.0, 1.0]];
    let product = source::product(&input, &identity).unwrap();
    let loss = source::loss::<f64>(&product, &[[0.0; 2]; 2]).unwrap();
    let update = source::update::<f64>(&product, &product).unwrap();
    let mut value = [17.0; 5];
    update.read_into(&mut value).unwrap();
    assert_eq!(
        value.map(f64::to_bits),
        [0.5_f64, 1.0, 1.5, 2.0, 17.0].map(f64::to_bits)
    );
    let mut error = [0.0];
    loss.read_into(&mut error).unwrap();
    assert_eq!(error[0].to_bits(), 7.5_f64.to_bits());
    println!("Strict MatMul → MSE / SGD: loss=7.5, update={value:?}");
    global::clear_thread_cache().unwrap();
    global::use_defaults().unwrap();
}
