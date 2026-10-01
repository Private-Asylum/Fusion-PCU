//! Ordinary source calls with Vulkan selected at runtime.
//!
//! ```toml
//! [dependencies]
//! fusion-pcu = { version = "0.0.6", features = ["vulkan"] }
//! ```
//! Run: cargo run -p fusion-pcu --features vulkan --example vulkan-transform

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
        backend: global::PcuBackendChoice::Vulkan,
        ..Default::default()
    })?;
    let input = [1.0_f32, -2.0, 0.0, f32::from_bits(1)];
    let mut output = [0.0_f32; 4];
    // Cold: discover a compatible physical GPU and retain its admitted prepared executable.
    // Current RAM input is copied into owned mapped Vulkan storage. Device work completes
    // before checked results are copied into the caller's stack output; no borrow escapes.
    transform(&input, &mut output)?;
    println!("Results in stack RAM: {output:?}");
    // Warm: the same preparation sees fresh input values without discovery or scoring.
    transform(&[3.0, 4.0, 5.0, 6.0], &mut output)?;
    println!("Fresh results in stack RAM: {output:?}");
    // Prepared backing is quiescent and retained for reuse until cache eviction or this drop.
    global::clear_thread_cache()?;
    Ok(())
}
