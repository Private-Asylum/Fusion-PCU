//! Ordinary source calls with the opt-in CPU provider selected at runtime.
//!
//! ```toml
//! [dependencies]
//! fusion-pcu = { version = "0.0.5", features = ["cpu"] }
//! ```
//! Run: cargo run -p fusion-pcu --features cpu --example cpu-transform

#[rustfmt::skip]
use fusion_pcu::{
    global,
    pcu,
    PcuExecutionError,
};

#[pcu(invocations: N)]
fn transform<const N: usize>(input: &[f32; N], output: &mut [f32; N]) {
    let id = pcu::context::global_invocation_id();
    output[id] = -input[id];
}

fn main() -> Result<(), PcuExecutionError> {
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Cpu,
        ..Default::default()
    })?;
    let input = [1.0_f32, -2.0, 0.0, f32::from_bits(1)];
    let mut output = [0.0_f32; 4];
    // Cold: discover the CPU, detect supported instructions, and prepare this function.
    // Input and output remain in caller-owned stack RAM. This synchronous operation
    // borrows them directly; no device upload, staging allocation, or readback occurs.
    // Checked validation completes before any output is written.
    transform(&input, &mut output)?;
    println!("Results in stack RAM: {output:?}");
    // Warm: fresh values reuse the prepared implementation without detecting or scoring.
    transform(&[3.0, 4.0, 5.0, 6.0], &mut output)?;
    println!("Fresh results in stack RAM: {output:?}");
    // The cached executable retains neither host borrow. Clearing it drops preparation;
    // input and output stay available until their ordinary Rust lifetimes end.
    global::clear_thread_cache()?;
    Ok(())
}
