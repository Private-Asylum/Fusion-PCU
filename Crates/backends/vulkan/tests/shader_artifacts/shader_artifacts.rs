//! Actual device lifecycle for genuine annotated source, without global backend fallback.
#[rustfmt::skip]
use fusion_pcu_vulkan::{PcuVulkanBackend,PcuVulkanShaderCachePolicy,PcuVulkanShaderDiskConfig};
use pcu_facade::pcu;
#[pcu(invocations = 65, crate_path = ::pcu_facade)]
fn identity(input: &[u32], output: &mut [u32]) {
    let id = context.global_invocation_id;
    output[id] = input[id];
}
#[test]
#[ignore = "requires actual Vulkan compute device"]
fn disk_reload_corruption_and_unload_preserve_native_owners() {
    let root = std::env::temp_dir().join(format!("pcu-native-artifacts-{}", std::process::id()));
    assert!(!root.exists());
    let backend = PcuVulkanBackend::new().unwrap();
    println!(
        "artifact device={} api={}",
        backend.name(),
        backend.caps().api_version
    );
    let config = PcuVulkanShaderDiskConfig {
        directory: root.clone(),
        retain_in_memory: false,
        rebuild_invalid: false,
    };
    backend.configure_shader_cache(PcuVulkanShaderCachePolicy::Disk(config.clone()));
    let mut first = identity_prepare(&backend).unwrap();
    assert_eq!(backend.shader_cache_stats().disk_misses, 1);
    let input: Vec<_> = (0_u32..65).map(|v| v.wrapping_mul(0xa5a5_0123)).collect();
    let mut output = [17; 68];
    let stats = backend.shader_cache_stats();
    first(&input, &mut output).unwrap();
    first(&input, &mut output).unwrap();
    assert_eq!(backend.shader_cache_stats(), stats);
    assert_eq!(output[..65], input);
    assert_eq!(output[65..], [17; 3]);
    let mut second = identity_prepare(&backend).unwrap();
    assert_eq!(backend.shader_cache_stats().disk_hits, 1);
    let directory = std::fs::read_dir(&root)
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .path();
    let path = directory.join("shader.spv");
    std::fs::write(&path, [0; 4]).unwrap();
    assert!(identity_prepare(&backend).is_err());
    // Preparation failure does not retire either previous native owner or publish output.
    first(&input, &mut output).unwrap();
    backend.configure_shader_cache(PcuVulkanShaderCachePolicy::Disk(
        PcuVulkanShaderDiskConfig {
            rebuild_invalid: true,
            ..config
        },
    ));
    let mut rebuilt = identity_prepare(&backend).unwrap();
    assert_eq!(backend.shader_cache_stats().rebuilds, 1);
    std::fs::remove_dir_all(&root).unwrap();
    first(&input, &mut output).unwrap();
    second(&input, &mut output).unwrap();
    rebuilt(&input, &mut output).unwrap();
    assert_eq!(output[..65], input);
    assert_eq!(output[65..], [17; 3]);
    let before = output;
    assert!(rebuilt(&input[..64], &mut output).is_err());
    assert_eq!(output, before);
    rebuilt(&input, &mut output).unwrap();
    assert!(!root.exists());
}

#[test]
#[ignore = "requires actual Vulkan compute device"]
fn cyclic_ir_is_refused_before_artifact_hashing() {
    use pcu_facade::{PcuDispatchOp, PcuHostKernelBackend};
    use fusion_pcu_vulkan::{PcuVulkanError, PcuVulkanShaderArtifactError, PcuVulkanShaderArtifactKey};
    static CYCLIC: [PcuDispatchOp<'static>; 1] = [PcuDispatchOp::GridStrideLoop {
        extent: 65,
        body: &CYCLIC,
    }];
    let backend = PcuVulkanBackend::new().unwrap();
    let bindings = identity_bindings();
    let builder = identity_ir(&bindings).unwrap();
    let mut kernel = builder.ir();
    kernel.ops = &CYCLIC;
    assert!(matches!(
        PcuVulkanShaderArtifactKey::for_kernel(&kernel, backend.caps()),
        Err(PcuVulkanShaderArtifactError::InvalidRequest)
    ));
    let root = std::env::temp_dir().join(format!("pcu-cyclic-refusal-{}", std::process::id()));
    assert!(!root.exists());
    for policy in [
        PcuVulkanShaderCachePolicy::Disabled,
        PcuVulkanShaderCachePolicy::MemoryOnly,
        PcuVulkanShaderCachePolicy::Disk(PcuVulkanShaderDiskConfig {
            directory: root.clone(),
            retain_in_memory: false,
            rebuild_invalid: false,
        }),
    ] {
        backend.configure_shader_cache(policy);
        assert!(matches!(
            backend.prepare_host_kernel(&kernel),
            Err(PcuVulkanError::UnsupportedPreparedProfile)
        ));
        assert_eq!(
            backend.shader_cache_stats(),
            fusion_pcu_vulkan::PcuVulkanShaderCacheStats::default()
        );
        assert!(!root.exists());
    }
    backend.configure_shader_cache(PcuVulkanShaderCachePolicy::MemoryOnly);
    let mut valid = identity_prepare(&backend).unwrap();
    let input = [71; 65];
    let mut output = [19; 68];
    valid(&input, &mut output).unwrap();
    assert_eq!(output[..65], input);
    assert_eq!(output[65..], [19; 3]);
}
