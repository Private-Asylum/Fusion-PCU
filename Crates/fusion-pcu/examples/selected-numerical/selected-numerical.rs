//! Single-effect Strict functions compose through ordinary escaped device owners.
//! Run with an explicitly compiled provider, for example:
//! cargo run -p fusion-pcu --example selected-numerical --features cpu,tensor -- cpu
//! cargo run -p fusion-pcu --example selected-numerical --features metal,tensor -- metal
//! cargo run -p fusion-pcu --example selected-numerical --features mlx,tensor -- mlx
//! cargo run -p fusion-pcu --example selected-numerical --features rocm,tensor -- rocm
//! cargo run -p fusion-pcu --example selected-numerical --features cuda,tensor -- cuda
//! cargo run -p fusion-pcu --example selected-numerical --features vulkan,tensor -- vulkan
//! Eligibility is checked for the actual compiled provider and numerical tuple.
#[rustfmt::skip]
use fusion_pcu::{global,pcu,PcuExecutionError,PcuTensor};
#[pcu(flag(strict))]
fn project(
    input: &[[f32; 2]; 2],
    weights: &[[f32; 2]; 2],
) -> Result<PcuTensor<f32>, PcuExecutionError> {
    pcu::matmul(input, weights)
}
#[pcu(flag(strict))]
fn backward(
    input: &[[f32; 2]; 2],
    upstream: &[[f32; 2]; 2],
) -> Result<PcuTensor<f32>, PcuExecutionError> {
    pcu::relu_backward(input, upstream)
}
#[pcu(flag(strict))]
fn update(
    weights: &[[f32; 2]; 2],
    gradient: &[[f32; 2]; 2],
) -> Result<PcuTensor<f32>, PcuExecutionError> {
    pcu::sgd_update(weights, gradient, 0.5)
}
#[pcu(flag(strict))]
fn loss(
    prediction: &[[f32; 2]; 2],
    target: &[[f32; 2]; 2],
) -> Result<PcuTensor<f32>, PcuExecutionError> {
    pcu::mean_squared_error(prediction, target)
}
fn select(name: &str) -> Result<global::PcuBackendChoice, Box<dyn std::error::Error>> {
    match name {
        #[cfg(feature = "cpu")]
        "cpu" => Ok(global::PcuBackendChoice::Cpu),
        #[cfg(feature = "rocm")]
        "rocm" => Ok(global::PcuBackendChoice::Rocm),
        #[cfg(feature = "cuda")]
        "cuda" => Ok(global::PcuBackendChoice::Cuda),
        #[cfg(feature = "vulkan")]
        "vulkan" => Ok(global::PcuBackendChoice::Vulkan),
        #[cfg(feature = "metal")]
        "metal" => Ok(global::PcuBackendChoice::Metal),
        #[cfg(feature = "mlx")]
        "mlx" => Ok(global::PcuBackendChoice::Mlx),
        _ => Err("provider not compiled; enable its Cargo feature".into()),
    }
}
#[cfg_attr(
    not(any(
        feature = "cpu",
        feature = "rocm",
        feature = "cuda",
        feature = "vulkan",
        feature = "metal",
        feature = "mlx"
    )),
    allow(
        clippy::drop_non_drop,
        reason = "Backendless builds reject selection before execution; explicit drops illustrate actual escaped-owner lifetime when a provider is compiled."
    )
)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let name = std::env::args()
        .nth(1)
        .ok_or("select an explicitly compiled provider: cpu, rocm, cuda, vulkan, metal, mlx")?;
    let backend = select(&name)?;
    global::configure(global::PcuExecutionPolicy {
        backend,
        ..Default::default()
    })?;
    let input = [[1.0, 2.0], [3.0, 4.0]];
    let weights = [[1.0, 0.0], [0.0, 1.0]];
    // First ordinary host borrows are staged by PCU. The returned owner escapes
    // the graph and retains its allocation until Rust ownership releases it.
    let projected = project(&input, &weights)?;
    // The projected borrow stays device-resident; only the host weights need staging.
    let measured_loss = loss(&projected, &[[0.0; 2]; 2])?;
    let gradient = backward(&projected, &weights)?;
    let updated = update(&projected, &gradient)?;
    drop(gradient);
    drop(projected);
    // A host demand is explicit. Copy into caller-owned stack RAM, then release
    // the escaped device result. Terminal completion controls physical reclamation.
    let mut scalar_stack = [0.0_f32];
    measured_loss.read_into(&mut scalar_stack)?;
    drop(measured_loss);
    let mut stack = [0.0_f32; 4];
    updated.read_into(&mut stack)?;
    drop(updated);
    global::clear_thread_cache()?;
    global::use_defaults()?;
    println!("MSE in stack RAM: {}", scalar_stack[0]);
    println!("Updated values in stack RAM: {stack:?}");
    Ok(())
}
