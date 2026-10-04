//! Persist, reload and remove generated shaders while retained native pipelines remain usable.
#[rustfmt::skip]
use fusion_pcu_vulkan::{
    PcuVulkanBackend,
    PcuVulkanShaderArtifactKey,
    PcuVulkanShaderCachePolicy,
    PcuVulkanShaderDiskConfig,
    unload_shader_artifact,
};
use pcu_facade::pcu;
#[pcu(invocations = 65, crate_path = ::pcu_facade)]
fn identity(input: &[u32], output: &mut [u32]) {
    let id = context.global_invocation_id;
    output[id] = input[id];
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let backend = PcuVulkanBackend::new()?;
    let mut config = PcuVulkanShaderDiskConfig::beside_executable()?;
    if let Some(directory) = std::env::args_os().nth(1) {
        config.directory = directory.into();
    }
    config.retain_in_memory = false;
    backend.configure_shader_cache(PcuVulkanShaderCachePolicy::Disk(config.clone()));
    let mut prepared = identity_prepare(&backend)?;
    let input = [0xfedc_ba98; 65];
    let mut output = [7; 68];
    prepared(&input, &mut output)?;
    assert_eq!(output[..65], input);
    assert_eq!(output[65..], [7; 3]);
    let mut reloaded = identity_prepare(&backend)?;
    assert!(backend.shader_cache_stats().disk_hits > 0);
    // The explicit unload key comes from independent lowering of this exact typed request.
    let bindings = identity_bindings();
    let builder = identity_ir(&bindings)?;
    let mut expected = Vec::new();
    fusion_pcu_spirv::lower_scalar_transport_to_spirv(
        &builder.ir(),
        fusion_pcu_spirv::PcuSpirvLoweringOptions::minimal_shader(),
        &mut expected,
    )
    .map_err(|error| fusion_pcu_vulkan::PcuVulkanError::SpirvLowering { error })?;
    let key = PcuVulkanShaderArtifactKey::for_kernel(&builder.ir(), backend.caps())?
        .with_module(&expected);
    unload_shader_artifact(&config.directory, &key)?;
    prepared(&input, &mut output)?;
    reloaded(&input, &mut output)?;
    assert_eq!(output[..65], input);
    println!(
        "Vulkan generated and reloaded .spv; both retained pipelines survived artifact removal"
    );
    Ok(())
}
