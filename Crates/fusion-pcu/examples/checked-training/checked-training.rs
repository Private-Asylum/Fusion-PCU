//! Ordinary authored forward/loss/backward/SGD using automatic device residency.
//! Run: cargo run -p fusion-pcu --features rocm,tensor --example checked-training
//! Or use cuda,tensor on an NVIDIA host. CPU is not an implicit substitute.
#[rustfmt::skip]
use fusion_pcu::{
    pcu,
    PcuExecutionError,
    PcuTensor,
};

#[pcu(flag(strict))]
fn loss(
    input: &[[f32; 2]; 2],
    weights: &[[f32; 1]; 2],
    target: &[[f32; 1]; 2],
) -> Result<PcuTensor<f32>, PcuExecutionError> {
    let prediction = pcu::relu(pcu::matmul(input, weights)?)?;
    pcu::mean_squared_error(&prediction, target)
}

#[pcu(flag(strict))]
fn training_step(
    input: &[[f32; 2]; 2],
    weights: &[[f32; 1]; 2],
    target: &[[f32; 1]; 2],
) -> Result<PcuTensor<f32>, PcuExecutionError> {
    let preactivation = pcu::matmul(input, weights)?;
    let prediction = pcu::relu(&preactivation)?;
    let loss = pcu::mean_squared_error(&prediction, target)?;
    // Cold capture differentiates this actual loss closure, including ReLU and
    // the transposed MatMul derivative. The warm call runs its prepared graph.
    let gradient = pcu::gradient(&loss, weights)?;
    pcu::sgd_update(weights, &gradient, 0.5_f32)
}

fn main() -> Result<(), PcuExecutionError> {
    // This optional CPU-only build explicitly chooses CPU; GPU builds never
    // replace a missing arithmetic contract with a hidden CPU fallback.
    #[cfg(all(
        feature = "cpu",
        not(any(
            feature = "rocm",
            feature = "cuda",
            feature = "metal",
            feature = "mlx",
            feature = "vulkan"
        ))
    ))]
    fusion_pcu::global::configure(fusion_pcu::global::PcuExecutionPolicy {
        backend: fusion_pcu::global::PcuBackendChoice::Cpu,
        ..Default::default()
    })?;
    let input = [[1.0, 0.0], [0.0, 1.0]];
    let weights = [[2.0], [-1.0]];
    let target = [[1.0], [0.0]];
    let mut host_loss = [0.0];
    let mut host_weights = [0.0; 2];
    {
        // Ordinary RAM borrows are staged by PCU. Inside each captured function,
        // intermediates stay on the selected device; graph liveness permits reuse.
        // Strict checks the prescribed operations individually. It does not request
        // PortableV1 or claim repeatable vendor-library/whole-model execution.
        let before = loss(&input, &weights, &target)?;
        before.read_into(&mut host_loss)?; // Explicit device-to-stack boundary.
        let updated = training_step(&input, &weights, &target)?;
        updated.read_into(&mut host_weights)?;
        assert_eq!(host_loss.map(f32::to_bits), [0.5_f32.to_bits()]);
        assert_eq!(
            host_weights.map(f32::to_bits),
            [1.5_f32.to_bits(), (-1.0_f32).to_bits()]
        );
        // Escaped owners retain backing until Drop. Drop releases logical ownership;
        // PCU may retain physical allocations in a prepared storage bank for reuse.
    }
    println!("RAM loss: {host_loss:?}; RAM updated weights: {host_weights:?}");
    Ok(())
}
