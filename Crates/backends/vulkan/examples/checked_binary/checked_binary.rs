//! Actual checked F32 source preparation and ordinary Vulkan provider execution.
#[rustfmt::skip]
use pcu_facade::{
    global,
    pcu,
};
use fusion_pcu_vulkan::PcuVulkanBackend;

#[pcu(invocations = N, crate_path = ::pcu_facade)]
fn divide<const N: usize>(left: &[f32], right: &[f32], output: &mut [f32]) {
    let id = context.global_invocation_id;
    output[id] = left[id] / right[id];
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Vulkan,
        ..Default::default()
    })?;
    let backend = PcuVulkanBackend::new()?;
    let left = [6.0, -12.0, -0.0];
    let right = [2.0, 3.0, 1.0];
    let mut output = [17.0; 5];
    let mut prepared = divide_prepare::<3, _>(&backend)?;
    prepared(&left, &right, &mut output)?;
    assert_eq!(
        output.map(f32::to_bits),
        [3.0_f32, -4.0, -0.0, 17.0, 17.0].map(f32::to_bits)
    );
    divide::<3>(&left, &right, &mut output)?;
    println!(
        "{} exact checked F32 division: {:?}",
        backend.name(),
        output
    );
    global::clear_thread_cache()?;
    global::use_defaults()?;
    Ok(())
}
