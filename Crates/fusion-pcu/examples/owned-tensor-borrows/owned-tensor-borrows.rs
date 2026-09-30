//! Ordinary Rust borrows retain owned device results; RAM readback is explicit.
//!
//! Run with `cargo run -p fusion-pcu --features rocm,tensor --example owned-tensor-borrows`.
#[rustfmt::skip]
use fusion_pcu::{
    pcu,
    PcuExecutionError,
    PcuTensor,
};

#[pcu]
fn activate(input: &[f32]) -> Result<PcuTensor<f32>, PcuExecutionError> {
    let activated = pcu::relu(input)?;
    Ok(activated)
}

// Calling another individually marked function here contributes to the same prepared graph.
// `activated` is a logical intermediate, not an independently published device owner.
#[pcu]
fn transform(input: &[f32], bias: &[f32]) -> Result<PcuTensor<f32>, PcuExecutionError> {
    let activated = activate(input)?;
    let adjusted = &activated + bias;
    Ok(adjusted)
}

#[pcu(invocations: N)]
fn add<const N: usize>(input: &[f32], output: &mut [f32]) {
    let id = pcu::context::global_invocation_id();
    output[id] = output[id] + input[id];
}

fn main() -> Result<(), PcuExecutionError> {
    // Defaults already discover compiled providers and select a compatible device at runtime.
    // fusion_pcu::global::use_defaults()?;
    let input = [-2.0_f32, 1.0, -3.0, 4.0];
    let bias = [1.0_f32, 2.0, 3.0, 4.0];
    let mut output = [0.0_f32; 4];
    {
        // A normal call stages current RAM and returns an initialized device owner.
        let device_bias = activate(&bias)?;
        // Mixed RAM/resident arguments need no upload/binding API. Input is staged; bias stays
        // device-local. The internal activate/add chain is one prepared graph, not eager calls.
        let first = transform(&input, &device_bias)?;
        // Explicit ordinary Drop releases bias early; first owns independent initialized bytes.
        drop(device_bias);
        // A resident borrow binds the existing storage: no device -> RAM -> device round trip.
        let mut second = activate(&first)?;
        // The same invocation kernel accepts resident borrows, including exclusive mutation.
        add::<4>(&first, &mut second)?;
        // This is the explicit boundary back into stack RAM; second still owns its device value.
        second.read_into(&mut output)?;
        // Scope exit drops second and first. Their logical owners no longer retain the bytes.
        // Quiescent backing can be released/recycled; prepared cache storage has its own lifetime.
    }
    println!("Results in stack RAM: {output:?}");
    fusion_pcu::global::clear_thread_cache()?;
    Ok(())
}
