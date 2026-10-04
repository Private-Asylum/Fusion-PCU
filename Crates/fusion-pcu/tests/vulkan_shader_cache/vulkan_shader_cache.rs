//! Ordinary call generation, disk refusal and warm pipeline ownership.
#[rustfmt::skip]
use fusion_pcu::{
    global,
    global::vulkan::{
        PcuVulkanShaderCachePolicy,
        PcuVulkanShaderDiskConfig,
        PcuVulkanShaderOptions,
    },
    pcu,
};

#[pcu(invocations: 65)]
fn identity(input: &[u32], output: &mut [u32]) {
    let id = pcu::context::global_invocation_id();
    output[id] = input[id];
}

struct Reset;
impl Drop for Reset {
    fn drop(&mut self) {
        global::clear_thread_cache().unwrap();
        global::vulkan::use_defaults().unwrap();
        global::use_defaults().unwrap();
    }
}

#[test]
#[ignore = "requires actual Vulkan compute hardware"]
fn ordinary_calls_use_disk_policy_and_keep_warm_ownership() {
    let _reset = Reset;
    let directory =
        std::env::temp_dir().join(format!("pcu-global-shader-cache-{}", std::process::id()));
    assert!(!directory.exists());
    let disk = PcuVulkanShaderDiskConfig {
        directory: directory.clone(),
        retain_in_memory: false,
        rebuild_invalid: false,
    };
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Vulkan,
        ..Default::default()
    })
    .unwrap();
    global::vulkan::configure(PcuVulkanShaderOptions {
        cache: PcuVulkanShaderCachePolicy::Disk(disk.clone()),
        ..Default::default()
    })
    .unwrap();
    let input: Vec<_> = (0_u32..65)
        .map(|value| value.wrapping_mul(0xa5a5_0123))
        .collect();
    let mut output = [17; 68];
    identity(&input, &mut output).unwrap();
    assert_eq!(output[..65], input);
    assert_eq!(output[65..], [17; 3]);
    let artifacts: Vec<_> = std::fs::read_dir(&directory)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .collect();
    assert_eq!(artifacts.len(), 1);
    let module = artifacts[0].join("shader.spv");
    let valid = std::fs::read(&module).unwrap();
    assert_eq!(&valid[..4], &0x0723_0203_u32.to_le_bytes());
    std::fs::write(&module, [0; 4]).unwrap();
    // A warm execution is wholly independent of its on-disk artifact.
    identity(&input, &mut output).unwrap();
    global::clear_thread_cache().unwrap();
    let before = output;
    assert!(identity(&input, &mut output).is_err());
    assert_eq!(output, before);
    // Explicit repair creates a new cold generation; checked behavior is unchanged.
    global::vulkan::configure(PcuVulkanShaderOptions {
        cache: PcuVulkanShaderCachePolicy::Disk(PcuVulkanShaderDiskConfig {
            rebuild_invalid: true,
            ..disk
        }),
        ..Default::default()
    })
    .unwrap();
    identity(&input, &mut output).unwrap();
    assert_eq!(std::fs::read(&module).unwrap(), valid);
    std::fs::remove_dir_all(&directory).unwrap();
    identity(&input, &mut output).unwrap();
    assert!(!directory.exists());
    assert!(identity(&input[..64], &mut output).is_err());
    assert_eq!(output, before);
    // Switching RAM-only invalidates selection and prepares without recreating disk assets.
    global::vulkan::use_defaults().unwrap();
    identity(&input, &mut output).unwrap();
    assert!(!directory.exists());
    assert_eq!(output[..65], input);
    assert_eq!(output[65..], [17; 3]);
}
