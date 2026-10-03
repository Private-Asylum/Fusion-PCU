//! Borrowed completed Clamp payload plus a structured recovery fault.
#[rustfmt::skip]
use pcu_facade::{pcu,global::{self,PcuBackendChoice,PcuExecutionPolicy}};
#[pcu(invocations=3,flag(strict),flag(clamp_range),flag(reject_subnormal_result),crate_path=::pcu_facade)]
fn add(left: &[f64], right: &[f64], out: &mut [f64]) {
    let id = pcu::context::global_invocation_id();
    out[id] = left[id] + right[id];
}
fn main() {
    if !cfg!(target_os = "macos") {
        println!("SKIP: requires Metal hardware");
        return;
    }
    global::configure(PcuExecutionPolicy {
        backend: PcuBackendChoice::Metal,
        ..PcuExecutionPolicy::default()
    })
    .unwrap();
    let mut output = [91.0; 5];
    let result = add(
        &[f64::from_bits(1), f64::MAX, 1.0],
        &[0.0, f64::MAX, 1.0],
        &mut output,
    );
    assert!(
        matches!(result,Err(pcu_facade::PcuExecutionError::ArithmeticFault(f)) if f.recovered&&f.invocation_id==0)
    );
    assert_eq!(
        output.map(f64::to_bits),
        [f64::from_bits(1), f64::MAX, 2.0, 91.0, 91.0].map(f64::to_bits)
    );
    println!(
        "Completed binary Clamp: {result:?}; bits={:?}",
        output.map(f64::to_bits)
    );
    global::use_defaults().unwrap();
}
