//! Shared process policy and generation for all compiled hosted providers.

use core::sync::atomic::AtomicU64;
use core::sync::atomic::Ordering;
use std::sync::RwLock;
use super::PcuExecutionError;
use super::PcuExecutionPolicy;

#[derive(Clone, Copy)]
pub(super) struct PolicySnapshot {
    pub(super) generation: u64,
    pub(super) policy: PcuExecutionPolicy,
}

struct PolicyState {
    snapshot: PolicySnapshot,
    #[cfg(feature = "vulkan")]
    vulkan_shaders: super::vulkan::PcuVulkanShaderOptions,
}

#[derive(Clone, Copy)]
#[cfg(any(
    feature = "cuda",
    feature = "metal",
    feature = "vulkan",
    feature = "cpu",
    feature = "mlx"
))]
pub(super) struct PolicyRoute {
    pub(super) generation: u64,
    pub(super) backend: super::PcuBackendChoice,
}

const fn backend_tag(backend: super::PcuBackendChoice) -> u64 {
    match backend {
        super::PcuBackendChoice::Automatic => 0,
        super::PcuBackendChoice::Rocm => 1,
        #[cfg(feature = "cuda")]
        super::PcuBackendChoice::Cuda => 2,
        #[cfg(feature = "metal")]
        super::PcuBackendChoice::Metal => 3,
        #[cfg(feature = "vulkan")]
        super::PcuBackendChoice::Vulkan => 4,
        #[cfg(feature = "cpu")]
        super::PcuBackendChoice::Cpu => 5,
        #[cfg(feature = "mlx")]
        super::PcuBackendChoice::Mlx => 6,
    }
}

// Reserve four route bits for compiled providers; the packed word remains one warm load.
const ROUTE_TAG_BITS: u32 = 4;
#[cfg(any(
    feature = "cuda",
    feature = "metal",
    feature = "vulkan",
    feature = "cpu",
    feature = "mlx"
))]
const ROUTE_TAG_MASK: u64 = (1 << ROUTE_TAG_BITS) - 1;

const fn encode_route(generation: u64, backend: super::PcuBackendChoice) -> u64 {
    (generation << ROUTE_TAG_BITS) | backend_tag(backend)
}

#[cfg(any(
    feature = "cuda",
    feature = "metal",
    feature = "vulkan",
    feature = "cpu",
    feature = "mlx"
))]
const fn decode_backend(tag: u64) -> super::PcuBackendChoice {
    match tag {
        1 => super::PcuBackendChoice::Rocm,
        #[cfg(feature = "cuda")]
        2 => super::PcuBackendChoice::Cuda,
        #[cfg(feature = "metal")]
        3 => super::PcuBackendChoice::Metal,
        #[cfg(feature = "vulkan")]
        4 => super::PcuBackendChoice::Vulkan,
        #[cfg(feature = "cpu")]
        5 => super::PcuBackendChoice::Cpu,
        #[cfg(feature = "mlx")]
        6 => super::PcuBackendChoice::Mlx,
        _ => super::PcuBackendChoice::Automatic,
    }
}

static ROUTE: AtomicU64 = AtomicU64::new(encode_route(1, super::PcuBackendChoice::Automatic));
static POLICY: RwLock<PolicyState> = RwLock::new(PolicyState {
    snapshot: PolicySnapshot {
        generation: 1,
        policy: PcuExecutionPolicy {
            backend: super::PcuBackendChoice::Automatic,
            device: None,
            cache_capacity: 64,
            block_size: 256,
            float_underflow: crate::PcuFloatUnderflowPolicy::IeeeAfterRounding,
            range_policy: crate::PcuRangePolicy::Reject,
            numerical_mode: crate::PcuNumericalMode::Boundary,
            numerical_options: crate::PcuNumericalOptions {
                compound_arithmetic: crate::PcuCompoundArithmeticPolicy::Checked,
                precision: crate::PcuPrecisionPolicy::Preserve,
                reproducibility: crate::PcuReproducibility::Unspecified,
            },
            score_device: super::default_device_score,
            score_invocation: None,
        },
    },
    #[cfg(feature = "vulkan")]
    vulkan_shaders: super::vulkan::PcuVulkanShaderOptions {
        source: super::vulkan::PcuVulkanShaderSource::Embedded,
        cache: super::vulkan::PcuVulkanShaderCachePolicy::MemoryOnly,
    },
});

#[cfg(any(
    feature = "rocm",
    feature = "cuda",
    feature = "metal",
    all(feature = "vulkan", feature = "tensor"),
    feature = "cpu",
    feature = "mlx"
))]
pub(super) fn snapshot() -> Result<PolicySnapshot, PcuExecutionError> {
    POLICY
        .read()
        .map(|state| state.snapshot)
        .map_err(|_| PcuExecutionError::PolicyUnavailable)
}

#[cfg(any(feature = "rocm", all(feature = "cuda", feature = "tensor")))]
pub(super) fn generation() -> u64 {
    ROUTE.load(Ordering::Acquire) >> ROUTE_TAG_BITS
}

#[cfg(any(
    feature = "cuda",
    feature = "metal",
    feature = "vulkan",
    feature = "cpu",
    feature = "mlx"
))]
pub(super) fn route() -> PolicyRoute {
    let packed = ROUTE.load(Ordering::Acquire);
    PolicyRoute {
        generation: packed >> ROUTE_TAG_BITS,
        backend: decode_backend(packed & ROUTE_TAG_MASK),
    }
}

pub(super) fn configure(policy: PcuExecutionPolicy) -> Result<(), PcuExecutionError> {
    let mut state = POLICY
        .write()
        .map_err(|_| PcuExecutionError::PolicyUnavailable)?;
    let generation = state
        .snapshot
        .generation
        .checked_add(1)
        .filter(|generation| *generation <= (u64::MAX >> ROUTE_TAG_BITS))
        .ok_or(PcuExecutionError::PolicyUnavailable)?;
    state.snapshot.policy = policy;
    state.snapshot.generation = generation;
    ROUTE.store(encode_route(generation, policy.backend), Ordering::Release);
    drop(state);
    Ok(())
}

#[cfg(feature = "vulkan")]
pub(super) fn snapshot_with_vulkan_shaders()
-> Result<(PolicySnapshot, super::vulkan::PcuVulkanShaderOptions), PcuExecutionError> {
    // A single cold lock keeps source/cache options and the selection generation
    // coherent even when another thread replaces process preferences.
    POLICY
        .read()
        .map(|state| (state.snapshot, state.vulkan_shaders.clone()))
        .map_err(|_| PcuExecutionError::PolicyUnavailable)
}

#[cfg(feature = "vulkan")]
pub(super) fn configure_vulkan_shaders(
    options: super::vulkan::PcuVulkanShaderOptions,
) -> Result<(), PcuExecutionError> {
    let mut state = POLICY
        .write()
        .map_err(|_| PcuExecutionError::PolicyUnavailable)?;
    let generation = state
        .snapshot
        .generation
        .checked_add(1)
        .filter(|generation| *generation <= (u64::MAX >> ROUTE_TAG_BITS))
        .ok_or(PcuExecutionError::PolicyUnavailable)?;
    state.vulkan_shaders = options;
    state.snapshot.generation = generation;
    ROUTE.store(
        encode_route(generation, state.snapshot.policy.backend),
        Ordering::Release,
    );
    drop(state);
    Ok(())
}

#[cfg(test)]
pub(super) static TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[cfg(test)]
mod tests {
    #[cfg(feature = "vulkan")]
    #[test]
    fn shader_configuration_advances_generation_without_changing_execution_permissions() {
        let _guard = super::TEST_LOCK.lock().unwrap();
        let (initial, original) = super::snapshot_with_vulkan_shaders().unwrap();
        let directory = std::path::PathBuf::from("nonexistent-cold-shader-cache");
        super::configure_vulkan_shaders(crate::global::vulkan::PcuVulkanShaderOptions {
            source: crate::global::vulkan::PcuVulkanShaderSource::ExternalComposed {
                directory: directory.clone(),
                retain_in_memory: false,
            },
            cache: crate::global::vulkan::PcuVulkanShaderCachePolicy::Disk(
                crate::global::vulkan::PcuVulkanShaderDiskConfig {
                    directory: directory.clone(),
                    retain_in_memory: false,
                    rebuild_invalid: false,
                },
            ),
        })
        .unwrap();
        let (changed, options) = super::snapshot_with_vulkan_shaders().unwrap();
        assert!(changed.generation > initial.generation);
        assert_eq!(changed.policy.backend, initial.policy.backend);
        assert_eq!(changed.policy.range_policy, initial.policy.range_policy);
        assert_eq!(
            changed.policy.numerical_options,
            initial.policy.numerical_options
        );
        assert_eq!(
            changed.policy.float_underflow,
            initial.policy.float_underflow
        );
        assert_eq!(changed.policy.numerical_mode, initial.policy.numerical_mode);
        assert_eq!(super::route().generation, changed.generation);
        assert_eq!(super::route().backend, changed.policy.backend);
        assert!(matches!(options.source,
            crate::global::vulkan::PcuVulkanShaderSource::ExternalComposed {
                directory: actual, retain_in_memory: false,
            } if actual == directory));
        assert!(matches!(options.cache,
            crate::global::vulkan::PcuVulkanShaderCachePolicy::Disk(
                crate::global::vulkan::PcuVulkanShaderDiskConfig {
                    directory: actual, retain_in_memory: false, rebuild_invalid: false,
                }
            ) if actual == directory));
        // Replacing execution preferences preserves separately configured asset policy.
        super::configure(initial.policy).unwrap();
        let (_, retained) = super::snapshot_with_vulkan_shaders().unwrap();
        assert!(matches!(
            retained.cache,
            crate::global::vulkan::PcuVulkanShaderCachePolicy::Disk(_)
        ));
        super::configure_vulkan_shaders(original).unwrap();
    }

    #[test]
    fn numerical_mode_configuration_advances_cache_generation() {
        let _guard = super::TEST_LOCK.lock().unwrap();
        let initial = super::POLICY.read().unwrap().snapshot;
        let mode = match initial.policy.numerical_mode {
            crate::PcuNumericalMode::Boundary => crate::PcuNumericalMode::Strict,
            crate::PcuNumericalMode::Strict => crate::PcuNumericalMode::Boundary,
        };
        super::configure(super::PcuExecutionPolicy {
            numerical_mode: mode,
            ..initial.policy
        })
        .unwrap();
        let selected = super::POLICY.read().unwrap().snapshot;
        assert_eq!(selected.policy.numerical_mode, mode);
        assert!(selected.generation > initial.generation);
        assert_eq!(
            super::ROUTE.load(core::sync::atomic::Ordering::Acquire) >> super::ROUTE_TAG_BITS,
            selected.generation
        );
        let options = crate::PcuNumericalOptions {
            compound_arithmetic: crate::PcuCompoundArithmeticPolicy::BackendDefined,
            precision: crate::PcuPrecisionPolicy::BackendOptimized,
            reproducibility: crate::PcuReproducibility::PortableV1,
        };
        super::configure(super::PcuExecutionPolicy {
            numerical_options: options,
            ..selected.policy
        })
        .unwrap();
        let changed = super::POLICY.read().unwrap().snapshot;
        assert_eq!(changed.policy.numerical_options, options);
        assert_eq!(changed.policy.numerical_mode, mode);
        assert!(changed.generation > selected.generation);
        super::configure(initial.policy).unwrap();
        assert!(super::POLICY.read().unwrap().snapshot.generation > changed.generation);
    }
}
