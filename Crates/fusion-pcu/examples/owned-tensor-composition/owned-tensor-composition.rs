//! Const-specialized matrix composition with ordinary RAM and resident borrows.
//!
//! Run with `cargo run -p fusion-pcu --features rocm,tensor --example owned-tensor-composition`.

#[rustfmt::skip]
use fusion_pcu::{
    pcu,
    PcuExecutionError,
    PcuTensor,
};

#[pcu]
fn matrix_identity<const R: usize, const C: usize>(
    input: &[[f32; C]; R],
) -> Result<PcuTensor<f32>, PcuExecutionError> {
    pcu::identity(input)
}

// This compound uses the increasing-K checked reference contract.
#[pcu(flag(strict))]
fn linear<const R: usize, const K: usize, const C: usize>(
    input: &[[f32; K]; R],
    weights: &[[f32; C]; K],
    bias: &[[f32; C]; R],
) -> Result<PcuTensor<f32>, PcuExecutionError> {
    Ok(pcu::matmul(input, weights)? + bias)
}

#[pcu]
fn activate(input: PcuTensor<f32>) -> Result<PcuTensor<f32>, PcuExecutionError> {
    Ok(pcu::relu(input)?)
}

fn main() -> Result<(), PcuExecutionError> {
    // Compiled providers are discovered lazily under configurable runtime defaults.
    // fusion_pcu::global::use_defaults()?;
    let input = [[1.0_f32, 2.0, 3.0], [-1.0, 0.0, 2.0]];
    let weights = [[1.0_f32, -2.0], [0.0, 1.0], [2.0, 1.0]];
    let bias = [[-8.0_f32, 2.0], [1.0, -5.0]];
    let mut output = [[0.0_f32; 2]; 2];
    {
        // RAM is staged automatically. This successful call returns an initialized [3, 2]
        // owner; the caller supplies neither device allocation nor transfer machinery.
        let device_weights = matrix_identity::<3, 2>(&weights)?;
        // Dynamic resident owners retain their dimensions at runtime. Explicit const arguments
        // provide compile-time extents here; those extents are checked against resident storage.
        // Input and bias move from RAM; weights bind directly to their existing device storage.
        let result = linear::<2, 3, 2>(&input, &device_weights, &bias)?;
        // MatMul and Add were captured into one graph. Internal values never escaped as
        // independent owners; graph liveness governs their backing. The result owns [2, 2].
        // This ordinary move carries the actual matrix dimensions into the next PCU function.
        // No RAM transfer occurs. Eligible exclusive backing may be reused for ReLU; the
        // planner checks that permission and otherwise provides a fresh initialized output.
        let result = activate(result)?;
        assert_eq!(result.shape(), &[2, 2]);
        // The synchronous call has quiesced. Ordinary Drop can now release the weights owner;
        // the result owns independent initialized bytes and remains valid.
        drop(device_weights);
        // Explicit device -> RAM boundary, into a matrix on the stack. Flattening the physical
        // destination for readback does not change the result's logical rank or dimensions.
        result.read_into(output.as_flattened_mut())?;
        // Scope exit releases the result owner. Prepared cache storage has its own lifetime.
    }
    println!("Matrix in stack RAM: {output:?}");
    assert_eq!(output, [[0.0, 5.0], [4.0, 0.0]]);
    fusion_pcu::global::clear_thread_cache()?;
    Ok(())
}
