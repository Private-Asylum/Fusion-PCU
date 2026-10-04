//! Ordinary annotated calls with runtime RAM/disk shader artifact preferences.
//!
//! Run with `--features vulkan --example vulkan-shader-cache`; pass `--memory`
//! for RAM-only caching. `--external` reads exported composed template packages
//! beside the executable and also works with default features disabled.
//! Package export: the Vulkan backend's `export-shader-packages` example.

#[rustfmt::skip]
use fusion_pcu::{
    global,
    global::vulkan::{
        PcuVulkanShaderCachePolicy,
        PcuVulkanShaderDiskConfig,
        PcuVulkanShaderOptions,
        PcuVulkanShaderSource,
    },
    pcu,
};

#[pcu(invocations: N)]
fn transform<const N: usize>(input: &[f32; N], output: &mut [f32; N]) {
    let id = pcu::context::global_invocation_id();
    output[id] = (input[id] + input[id]) * input[id];
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let memory_only = std::env::args().any(|argument| argument == "--memory");
    let external = std::env::args().any(|argument| argument == "--external");
    let source = if external {
        PcuVulkanShaderSource::beside_executable()?
    } else {
        PcuVulkanShaderSource::Embedded
    };
    let cache = if memory_only {
        PcuVulkanShaderCachePolicy::MemoryOnly
    } else {
        PcuVulkanShaderCachePolicy::Disk(PcuVulkanShaderDiskConfig::beside_executable()?)
    };
    println!("Shader source: {source:?}; generated artifacts: {cache:?}");
    global::vulkan::configure(PcuVulkanShaderOptions { source, cache })?;
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Vulkan,
        ..Default::default()
    })?;
    let input = [0.5_f32, 1.0, -2.0, 0.0];
    let mut output = [0.0_f32; 4];
    // Cold: compile admitted source to SPIR-V, validate/cache its artifact, then
    // create the native pipeline. RAM borrows stage into owned Vulkan resources.
    transform(&input, &mut output)?;
    assert_eq!(
        output.map(f32::to_bits),
        [0.5_f32, 2.0, 8.0, 0.0].map(f32::to_bits)
    );
    println!("Results in stack RAM: {output:?}");
    // Warm: execute retained native preparation, without consulting source files
    // or caches. Completion precedes publication into the caller's stack array.
    transform(&[1.0, 2.0, 3.0, 4.0], &mut output)?;
    assert_eq!(
        output.map(f32::to_bits),
        [2.0_f32, 8.0, 18.0, 32.0].map(f32::to_bits)
    );
    // Release this thread's prepared pipeline/resources after terminal completion.
    // Persisted .spv artifacts remain available to a later process/cold prepare.
    global::clear_thread_cache()?;
    Ok(())
}
