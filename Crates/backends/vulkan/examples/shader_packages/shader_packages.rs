//! Ordinary preparation consumes exact external GLSL-derived helpers without embedding them.
#[rustfmt::skip]
use fusion_pcu_vulkan::{PcuVulkanBackend,PcuVulkanShaderSource};
use pcu_facade::{pcu, PcuCheckedFloat};
#[pcu(invocations = 65, crate_path = ::pcu_facade)]
fn expression<T: PcuCheckedFloat>(input: &[T], output: &mut [T]) {
    let id = context.global_invocation_id;
    output[id] = (input[id] + input[id]) * input[id];
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let backend = PcuVulkanBackend::new()?;
    let source = if let Some(directory) = std::env::args_os().nth(1) {
        PcuVulkanShaderSource::ExternalComposed {
            directory: directory.into(),
            retain_in_memory: false,
        }
    } else {
        PcuVulkanShaderSource::beside_executable()?
    };
    backend.configure_shader_source(source);
    let mut prepared = expression_prepare::<f32, _>(&backend)?;
    let input = [0.5_f32; 65];
    let mut output = [17_f32; 68];
    prepared(&input, &mut output)?;
    assert_eq!(output[..65], input);
    assert_eq!(output[65..], [17_f32; 3]);
    backend.clear_shader_source_cache();
    // Retained native owners no longer require the source package during a warm call.
    prepared(&input, &mut output)?;
    println!(
        "Vulkan annotated composition consumed an exact external GLSL template; warm owner retained"
    );
    Ok(())
}
