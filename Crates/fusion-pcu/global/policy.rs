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

#[derive(Clone, Copy)]
#[cfg(any(feature = "cuda", feature = "metal"))]
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
    }
}

const fn encode_route(generation: u64, backend: super::PcuBackendChoice) -> u64 {
    (generation << 2) | backend_tag(backend)
}

#[cfg(any(feature = "cuda", feature = "metal"))]
const fn decode_backend(tag: u64) -> super::PcuBackendChoice {
    match tag {
        1 => super::PcuBackendChoice::Rocm,
        #[cfg(feature = "cuda")]
        2 => super::PcuBackendChoice::Cuda,
        #[cfg(feature = "metal")]
        3 => super::PcuBackendChoice::Metal,
        _ => super::PcuBackendChoice::Automatic,
    }
}

static ROUTE: AtomicU64 = AtomicU64::new(encode_route(1, super::PcuBackendChoice::Automatic));
static POLICY: RwLock<PolicySnapshot> = RwLock::new(PolicySnapshot {
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
    },
});

#[cfg(any(feature = "rocm", feature = "cuda", feature = "metal"))]
pub(super) fn snapshot() -> Result<PolicySnapshot, PcuExecutionError> {
    POLICY
        .read()
        .map(|state| *state)
        .map_err(|_| PcuExecutionError::PolicyUnavailable)
}

#[cfg(any(feature = "rocm", all(feature = "cuda", feature = "tensor")))]
pub(super) fn generation() -> u64 {
    ROUTE.load(Ordering::Acquire) >> 2
}

#[cfg(any(feature = "cuda", feature = "metal"))]
pub(super) fn route() -> PolicyRoute {
    let packed = ROUTE.load(Ordering::Acquire);
    PolicyRoute {
        generation: packed >> 2,
        backend: decode_backend(packed & 0b11),
    }
}

pub(super) fn configure(policy: PcuExecutionPolicy) -> Result<(), PcuExecutionError> {
    let mut state = POLICY
        .write()
        .map_err(|_| PcuExecutionError::PolicyUnavailable)?;
    let generation = state
        .generation
        .checked_add(1)
        .filter(|generation| *generation <= (u64::MAX >> 2))
        .ok_or(PcuExecutionError::PolicyUnavailable)?;
    state.policy = policy;
    state.generation = generation;
    ROUTE.store(encode_route(generation, policy.backend), Ordering::Release);
    drop(state);
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn numerical_mode_configuration_advances_cache_generation() {
        let initial = *super::POLICY.read().unwrap();
        let mode = match initial.policy.numerical_mode {
            crate::PcuNumericalMode::Boundary => crate::PcuNumericalMode::Strict,
            crate::PcuNumericalMode::Strict => crate::PcuNumericalMode::Boundary,
        };
        super::configure(super::PcuExecutionPolicy {
            numerical_mode: mode,
            ..initial.policy
        })
        .unwrap();
        let selected = *super::POLICY.read().unwrap();
        assert_eq!(selected.policy.numerical_mode, mode);
        assert!(selected.generation > initial.generation);
        assert_eq!(
            super::ROUTE.load(core::sync::atomic::Ordering::Acquire) >> 2,
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
        let changed = *super::POLICY.read().unwrap();
        assert_eq!(changed.policy.numerical_options, options);
        assert_eq!(changed.policy.numerical_mode, mode);
        assert!(changed.generation > selected.generation);
        super::configure(initial.policy).unwrap();
        assert!(super::POLICY.read().unwrap().generation > changed.generation);
    }
}
