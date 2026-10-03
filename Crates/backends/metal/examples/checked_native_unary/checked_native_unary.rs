//! Observable F64 Clamp output, with exact paired-U32 GPU unary encoding selection.
#[rustfmt::skip]
use pcu_facade::{pcu,global::{configure,PcuExecutionPolicy,PcuBackendChoice}};
#[pcu(invocations=3,flag(strict),flag(clamp_range),flag(reject_subnormal_result),crate_path=::pcu_facade)]
fn negate(input: &[f64], output: &mut [f64]) {
    let id = pcu::context::global_invocation_id();
    output[id] = -input[id];
}
fn main() {
    if !cfg!(target_os = "macos") {
        println!("SKIP: Metal GPU is unavailable");
        return;
    }
    configure(PcuExecutionPolicy {
        backend: PcuBackendChoice::Metal,
        ..PcuExecutionPolicy::default()
    })
    .unwrap();
    let mut output = [0.0_f64; 3];
    let fault = negate(&[f64::from_bits(1), 2.0, -0.0], &mut output).unwrap_err();
    println!(
        "Recovered fault: {fault:?}; terminal exact bits: {:?}",
        output.map(f64::to_bits)
    );
}
