//! Deployment-time export of the exact GLSL-derived composed shader packages.
fn main() -> Result<(), fusion_pcu_vulkan::PcuVulkanError> {
    let directory = std::env::args_os().nth(1).map_or_else(
        || std::path::PathBuf::from("pcu-shader-packages"),
        Into::into,
    );
    fusion_pcu_vulkan::write_composed_shader_package(&directory)?;
    println!(
        "exact known composed templates exported to {}",
        directory.display()
    );
    Ok(())
}
