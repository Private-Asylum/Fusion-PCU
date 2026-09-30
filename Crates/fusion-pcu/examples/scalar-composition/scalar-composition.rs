//! Ordinary typed host calls using lazy runtime selection and reusable prepared state.
//!
//! Run with `cargo run -p fusion-pcu --features rocm --example scalar-composition`.
//! The feature makes `ROCm` available; compatible device selection happens at runtime.

use fusion_pcu::pcu;
use fusion_pcu::global::PcuExecutionError;

// Only individually annotated functions participate; this ordinary module is not captured.
mod scalar {
    use fusion_pcu::pcu;

    #[pcu]
    fn double(value: f32) -> f32 {
        value * 2.0
    }

    #[pcu]
    pub fn affine(value: f32) -> f32 {
        double(value) + 1.0
    }
}

#[pcu(invocations: N)]
fn transform<const N: usize>(seed: &f32, input: &[f32; N], output: &mut [f32; N]) {
    let id = pcu::context::global_invocation_id();
    output[id] = scalar::affine(input[id] * *seed);
}

fn main() -> Result<(), PcuExecutionError> {
    // No setup is required. Runtime policy can optionally be changed before subsequent calls:
    // fusion_pcu::global::use_defaults()?;
    let input = [1.0_f32, 2.0, 3.0, 4.0];
    let seed = 1.0_f32;
    let mut output = [0.0_f32; 4];
    // Cold: discover compatible devices, select one, compile and retain prepared state.
    // Every call: current seed/input RAM -> device, compute, wait, output -> RAM.
    // This kernel fully writes output without reading it, so its initial RAM contents need no
    // upload. Read/modify/write kernels still upload the current mutable input automatically.
    // The readonly scalar is broadcast to every invocation; it needs no invocation-sized buffer.
    // Prepared backing stays owned by the thread cache for reuse; no host borrow escapes.
    transform(&seed, &input, &mut output)?;
    println!("Results in stack RAM: {output:?}");
    // Optional deterministic release of this thread's cached, quiescent prepared allocations.
    fusion_pcu::global::clear_thread_cache()?;
    Ok(())
}
