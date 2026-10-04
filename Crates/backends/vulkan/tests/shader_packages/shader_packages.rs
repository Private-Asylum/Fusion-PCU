//! Native source ownership survives external package removal and cold errors remain explicit.
#[rustfmt::skip]
use fusion_pcu_vulkan::{PcuVulkanBackend,PcuVulkanShaderSource,PcuVulkanShaderCachePolicy};
use pcu_facade::pcu;
#[pcu(invocations = 65, crate_path = ::pcu_facade)]
fn expression(input: &[f32], output: &mut [f32]) {
    let id = context.global_invocation_id;
    output[id] = (input[id] + input[id]) * input[id];
}
#[test]
#[ignore = "requires actual Vulkan device and exported PCU_COMPOSED_PACKAGES"]
fn external_source_unload_and_refusal_preserve_owned_pipeline() {
    let original = std::path::PathBuf::from(std::env::var_os("PCU_COMPOSED_PACKAGES").unwrap());
    let directory = std::env::temp_dir().join(format!("pcu-native-package-{}", std::process::id()));
    let family = directory.join("composed-f32");
    std::fs::create_dir_all(&family).unwrap();
    for file in ["template.spv", "rewrite.bin"] {
        std::fs::copy(original.join("composed-f32").join(file), family.join(file)).unwrap();
    }
    let backend = PcuVulkanBackend::new().unwrap();
    backend.configure_shader_cache(PcuVulkanShaderCachePolicy::Disabled);
    backend.configure_shader_source(PcuVulkanShaderSource::ExternalComposed {
        directory: directory.clone(),
        retain_in_memory: true,
    });
    let mut prepared = expression_prepare(&backend).unwrap();
    let input = [0.5; 65];
    let mut output = [17.0; 68];
    prepared(&input, &mut output).unwrap();
    assert_eq!(output[..65], input);
    assert_eq!(output[65..], [17.0; 3]);
    std::fs::remove_dir_all(&directory).unwrap();
    let mut cached_source = expression_prepare(&backend).unwrap();
    backend.clear_shader_source_cache();
    assert!(expression_prepare(&backend).is_err());
    prepared(&input, &mut output).unwrap();
    cached_source(&input, &mut output).unwrap();
    assert_eq!(output[..65], input);
    assert_eq!(output[65..], [17.0; 3]);
    let before = output;
    assert!(cached_source(&input[..64], &mut output).is_err());
    assert_eq!(output.map(f32::to_bits), before.map(f32::to_bits));
    let mut overflow = input;
    overflow[2] = f32::MAX;
    assert!(prepared(&overflow, &mut output).is_err());
    assert_eq!(output.map(f32::to_bits), before.map(f32::to_bits));
    prepared(&input, &mut output).unwrap();
    assert!(!directory.exists());
}

#[pcu(invocations = 65, crate_path = ::pcu_facade)]
fn integer_expression<T: pcu_facade::PcuCheckedInteger>(input: &[T], output: &mut [T]) {
    let id = context.global_invocation_id;
    output[id] = (input[id] + input[id]) * input[id];
}
#[test]
#[ignore = "requires actual Vulkan device and exported PCU_COMPOSED_PACKAGES"]
fn external_integer_source_unload_preserves_owned_pipeline() {
    use pcu_facade::{PcuU512, PcuScalar};
    let original = std::path::PathBuf::from(std::env::var_os("PCU_COMPOSED_PACKAGES").unwrap());
    let directory =
        std::env::temp_dir().join(format!("pcu-native-integer-package-{}", std::process::id()));
    let family = directory.join("composed-integer-wide");
    std::fs::create_dir_all(&family).unwrap();
    for file in ["template.spv", "rewrite.bin"] {
        std::fs::copy(
            original.join("composed-integer-wide").join(file),
            family.join(file),
        )
        .unwrap();
    }
    let backend = PcuVulkanBackend::new().unwrap();
    backend.configure_shader_cache(PcuVulkanShaderCachePolicy::Disabled);
    backend.configure_shader_source(PcuVulkanShaderSource::ExternalComposed {
        directory: directory.clone(),
        retain_in_memory: true,
    });
    let mut prepared = integer_expression_prepare::<PcuU512, _>(&backend).unwrap();
    let mut bytes = [0; 64];
    bytes[0] = 1;
    let one = PcuU512::decode_le(bytes);
    bytes[0] = 2;
    let two = PcuU512::decode_le(bytes);
    bytes[0] = 17;
    let sentinel = PcuU512::decode_le(bytes);
    let input = [one; 65];
    let mut output = [sentinel; 68];
    prepared(&input, &mut output).unwrap();
    assert_eq!(output[..65], [two; 65]);
    assert_eq!(output[65..], [sentinel; 3]);
    std::fs::remove_dir_all(&directory).unwrap();
    let mut cached_source = integer_expression_prepare::<PcuU512, _>(&backend).unwrap();
    backend.clear_shader_source_cache();
    assert!(integer_expression_prepare::<PcuU512, _>(&backend).is_err());
    prepared(&input, &mut output).unwrap();
    cached_source(&input, &mut output).unwrap();
    let before = output;
    assert!(cached_source(&input[..64], &mut output).is_err());
    assert_eq!(output, before);
    prepared(&input, &mut output).unwrap();
    assert!(!directory.exists());
}

#[test]
#[ignore = "requires actual Vulkan device and exported PCU_COMPOSED_PACKAGES"]
fn external_narrow_integer_source_unload_preserves_owned_pipeline() {
    use pcu_facade::PcuScalar;
    let original = std::path::PathBuf::from(std::env::var_os("PCU_COMPOSED_PACKAGES").unwrap());
    let directory = std::env::temp_dir().join(format!(
        "pcu-native-narrow-integer-package-{}",
        std::process::id()
    ));
    let family = directory.join("composed-integer-narrow");
    std::fs::create_dir_all(&family).unwrap();
    for file in ["template.spv", "rewrite.bin"] {
        std::fs::copy(
            original.join("composed-integer-narrow").join(file),
            family.join(file),
        )
        .unwrap();
    }
    let backend = PcuVulkanBackend::new().unwrap();
    backend.configure_shader_cache(PcuVulkanShaderCachePolicy::Disabled);
    backend.configure_shader_source(PcuVulkanShaderSource::ExternalComposed {
        directory: directory.clone(),
        retain_in_memory: true,
    });
    let mut prepared = integer_expression_prepare::<u8, _>(&backend).unwrap();
    let mut bytes = [0; 1];
    bytes[0] = 1;
    let one = u8::decode_le(bytes);
    bytes[0] = 2;
    let two = u8::decode_le(bytes);
    bytes[0] = 17;
    let sentinel = u8::decode_le(bytes);
    let input = [one; 65];
    let mut output = [sentinel; 68];
    prepared(&input, &mut output).unwrap();
    assert_eq!(output[..65], [two; 65]);
    assert_eq!(output[65..], [sentinel; 3]);
    std::fs::remove_dir_all(&directory).unwrap();
    let mut cached_source = integer_expression_prepare::<u8, _>(&backend).unwrap();
    backend.clear_shader_source_cache();
    assert!(integer_expression_prepare::<u8, _>(&backend).is_err());
    prepared(&input, &mut output).unwrap();
    cached_source(&input, &mut output).unwrap();
    let before = output;
    assert!(cached_source(&input[..64], &mut output).is_err());
    assert_eq!(output, before);
    let mut overflow = input;
    overflow[7] = u8::MAX;
    assert!(prepared(&overflow, &mut output).is_err());
    assert_eq!(output, before);
    prepared(&input, &mut output).unwrap();
    assert!(!directory.exists());
}
