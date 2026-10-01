//! Shared statically selected host/resident invocation path for compiled providers. Automatic selection
//! ranks candidates from every compiled runtime; resident affinity pins borrowed device storage.

use core::any::TypeId;
use core::sync::atomic::Ordering;
use std::cell::RefCell;
use std::ffi::OsString;
#[cfg(any(
    feature = "rocm",
    feature = "cuda",
    feature = "metal",
    feature = "vulkan"
))]
use std::rc::Rc;
#[cfg(any(feature = "rocm", feature = "cuda", feature = "metal"))]
use smallvec::SmallVec;
#[path = "affinity/affinity.rs"]
mod affinity;
#[path = "discovery/discovery.rs"]
mod discovery;
use affinity::ResidentAffinity;
#[cfg(feature = "cpu")]
#[rustfmt::skip]
use fusion_pcu_cpu::{
    PcuCpuDiscovery,
    PcuCpuDiscoveryError,
    PcuCpuHostBackend,
    PcuCpuHostError,
    PcuCpuHostArgumentError,
    PcuCpuPreparedHost,
};
#[cfg(feature = "vulkan")]
#[rustfmt::skip]
use fusion_pcu_vulkan::{
    PcuVulkanBackend,
    PcuVulkanDiscovery,
    PcuVulkanError,
    PcuVulkanPreparedBitMap,
};
#[cfg(feature = "cuda")]
#[rustfmt::skip]
use fusion_pcu_cuda::{
    CudaDiscovery,
    CudaHostKernelError,
    CudaOwnedDispatchBackend,
    CudaPreparedHostKernel,
};
#[cfg(feature = "metal")]
#[rustfmt::skip]
use fusion_pcu_metal::{
    MetalDiscovery,
    MetalOwnedDispatchBackend,
    MetalPreparedHostKernel,
};
#[cfg(feature = "rocm")]
#[rustfmt::skip]
use fusion_pcu_rocm::{
    RocmDiscovery,
    RocmHostKernelError,
    RocmOwnedDispatchBackend,
    RocmPreparedHostKernel,
};
#[rustfmt::skip]
use crate::{
    PcuDeviceClass,
    PcuDeviceDescriptor,
    PcuHostArgument,
    PcuHostKernelBackend,
    PcuObjectKind,
    PcuObjectRef,
    PcuPreparedHostKernel,
    PcuProviderDescriptor,
    PcuProviderId,
    PcuProviderReadiness,
    PcuProviderStatus,
    PcuRuntimeDiscovery,
    PcuTargetDescriptor,
};
#[rustfmt::skip]
use super::{
    policy,
    PcuBackendChoice,
    PcuExecutionError,
    PcuExecutionPolicy,
    PcuHostCallSite,
    PcuHostPreparation,
};

#[rustfmt::skip]
#[cfg(any(feature = "rocm", feature = "cuda", feature = "metal"))]
use super::arguments::{
    PcuCallArgumentKind,
    ResidentWriteGuard,
};
#[cfg(any(feature = "rocm", feature = "cuda", feature = "metal"))]
use super::resident::DeviceArgument;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Provider {
    #[cfg(feature = "metal")]
    Metal,
    #[cfg(feature = "cuda")]
    Cuda,
    #[cfg(feature = "rocm")]
    Rocm,
    #[cfg(feature = "vulkan")]
    Vulkan,
    #[cfg(feature = "cpu")]
    Cpu,
}

#[derive(Clone)]
pub(super) struct Candidate {
    pub(super) provider: Provider,
    pub(super) device: PcuObjectRef,
    score: i128,
}

#[derive(Clone)]
enum Session {
    #[cfg(feature = "metal")]
    Metal(Rc<MetalOwnedDispatchBackend>),
    #[cfg(feature = "cuda")]
    Cuda(Rc<CudaOwnedDispatchBackend>),
    #[cfg(any(feature = "rocm", feature = "cuda", feature = "metal"))]
    Resident(Rc<super::resident::Session>),
    #[cfg(feature = "vulkan")]
    Vulkan(Rc<PcuVulkanBackend>),
    #[cfg(feature = "cpu")]
    Cpu(PcuCpuHostBackend),
    #[cfg(feature = "rocm")]
    Rocm(Rc<RocmOwnedDispatchBackend>),
}

pub(super) enum Prepared {
    #[cfg(feature = "metal")]
    Metal(MetalPreparedHostKernel),
    #[cfg(feature = "cuda")]
    Cuda(CudaPreparedHostKernel),
    #[cfg(feature = "rocm")]
    Rocm(RocmPreparedHostKernel),
    #[cfg(feature = "vulkan")]
    Vulkan(PcuVulkanPreparedBitMap),
    #[cfg(feature = "cpu")]
    Cpu(PcuCpuPreparedHost),
}

impl Prepared {
    fn call(&mut self, args: &mut [PcuHostArgument<'_>]) -> Result<(), PcuExecutionError> {
        match self {
            #[cfg(feature = "metal")]
            Self::Metal(prepared) => prepared.call(args).map_err(map_metal_error),
            #[cfg(feature = "cuda")]
            Self::Cuda(prepared) => prepared.call(args).map_err(map_cuda_error),
            #[cfg(feature = "rocm")]
            Self::Rocm(prepared) => prepared.call(args).map_err(map_rocm_error),
            #[cfg(feature = "vulkan")]
            Self::Vulkan(prepared) => prepared.call(args).map_err(map_vulkan_error),
            #[cfg(feature = "cpu")]
            Self::Cpu(prepared) => prepared.call(args).map_err(map_cpu_error),
        }
    }
    fn call_arguments<const N: usize>(
        &mut self,
        arguments: [super::arguments::PcuCallArgument<'_>; N],
    ) -> Result<(), PcuExecutionError> {
        match self {
            #[cfg(feature = "metal")]
            Self::Metal(prepared) => call_metal_arguments(prepared, arguments),
            #[cfg(feature = "cuda")]
            Self::Cuda(prepared) => call_cuda_arguments(prepared, arguments),
            #[cfg(feature = "rocm")]
            Self::Rocm(prepared) => call_rocm_arguments(prepared, arguments),
            #[cfg(feature = "vulkan")]
            Self::Vulkan(prepared) => call_vulkan_arguments(prepared, arguments),
            #[cfg(feature = "cpu")]
            Self::Cpu(prepared) => call_cpu_arguments(prepared, arguments),
        }
    }
}

pub(super) struct Preparation {
    policy: PcuExecutionPolicy,
    result: Option<(Session, Prepared)>,
    affinity: Option<ResidentAffinity>,
}

impl Preparation {
    pub(super) const fn new(policy: PcuExecutionPolicy) -> Self {
        Self {
            policy,
            result: None,
            affinity: None,
        }
    }
    pub(super) const fn float_underflow_policy(&self) -> crate::PcuFloatUnderflowPolicy {
        self.policy.float_underflow
    }
    pub(super) const fn range_policy(&self) -> crate::PcuRangePolicy {
        self.policy.range_policy
    }

    #[cfg(any(feature = "rocm", feature = "cuda", feature = "metal"))]
    fn prepare_affinity(
        &mut self,
        kernel: &crate::PcuDispatchKernelIr<'_>,
    ) -> Result<bool, PcuExecutionError> {
        let Some(root) = &self.affinity else {
            return Ok(false);
        };
        affinity::validate(root, self.policy)?;
        let prepared = root.prepare_host_kernel(kernel)?;
        self.result = Some((Session::Resident(Rc::clone(root)), prepared));
        Ok(true)
    }
    pub(super) fn prepare(
        &mut self,
        kernel: &crate::PcuDispatchKernelIr<'_>,
    ) -> Result<(), PcuExecutionError> {
        if self.policy.numerical_options.reproducibility != crate::PcuReproducibility::Unspecified {
            return Err(PcuExecutionError::UnsupportedNumericalOptions(
                self.policy.numerical_options,
            ));
        }
        #[cfg(any(feature = "rocm", feature = "cuda", feature = "metal"))]
        if self.prepare_affinity(kernel)? {
            return Ok(());
        }
        let providers = discovery::Discoveries::new(self.policy);
        let (candidates, discovery_errors) = providers.candidates(self.policy, kernel);
        let policy = self.policy;
        self.prepare_candidates(kernel, candidates, discovery_errors, |candidate| {
            providers.open(candidate, policy)
        })
    }

    fn prepare_candidates(
        &mut self,
        kernel: &crate::PcuDispatchKernelIr<'_>,
        candidates: Vec<Candidate>,
        discovery_errors: Vec<String>,
        mut open: impl FnMut(&Candidate) -> Result<Session, PcuExecutionError>,
    ) -> Result<(), PcuExecutionError> {
        let mut rejected = Vec::new();
        for candidate in candidates {
            let session = match open(&candidate) {
                Ok(session) => session,
                Err(error) => {
                    rejected.push((candidate.device, error));
                    continue;
                }
            };
            let prepared = prepare_session(&session, kernel);
            match prepared {
                Ok(prepared) => {
                    self.result = Some((session, prepared));
                    return Ok(());
                }
                Err(error) => rejected.push((candidate.device, error)),
            }
        }
        Err(rejected_candidates(rejected, discovery_errors))
    }
}

struct Entry {
    specialization: TypeId,
    _session: Session,
    affinity: Option<ResidentAffinity>,
    prepared: Prepared,
}
#[derive(Default)]
struct State {
    generation: u64,
    realm: Option<RuntimeRealm>,
    entries: Vec<Entry>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct RuntimeRealm {
    hip_runtime: Option<OsString>,
    hipcc: Option<OsString>,
    cuda_runtime: Option<OsString>,
    cuda_driver: Option<OsString>,
    nvcc: Option<OsString>,
    cuda_nvcc: Option<OsString>,
    nvrtc: Option<OsString>,
    hiprtc: Option<OsString>,
    cuda_home: Option<OsString>,
    path: Option<OsString>,
}

impl RuntimeRealm {
    fn current() -> Self {
        Self {
            hip_runtime: std::env::var_os("HIP_RUNTIME_LIBRARY"),
            hipcc: std::env::var_os("HIPCC"),
            cuda_runtime: std::env::var_os("CUDA_RUNTIME_LIBRARY"),
            cuda_driver: std::env::var_os("CUDA_DRIVER_LIBRARY"),
            nvcc: std::env::var_os("NVCC"),
            cuda_nvcc: std::env::var_os("CUDA_NVCC"),
            nvrtc: std::env::var_os("NVRTC_LIBRARY"),
            hiprtc: std::env::var_os("HIPRTC_LIBRARY"),
            cuda_home: std::env::var_os("CUDA_HOME").or_else(|| std::env::var_os("CUDA_PATH")),
            path: std::env::var_os("PATH"),
        }
    }
}
std::thread_local! { static STATE: RefCell<State> = RefCell::new(State::default()); }

pub(super) fn call_host(
    site: &PcuHostCallSite,
    specialization: TypeId,
    arguments: &mut [PcuHostArgument<'_>],
    prepare: impl FnOnce(&mut PcuHostPreparation) -> Result<(), PcuExecutionError>,
) -> Result<(), PcuExecutionError> {
    with_entry(site, specialization, None, prepare, |prepared| {
        prepared.call(arguments)
    })
}

fn with_entry<R>(
    site: &PcuHostCallSite,
    specialization: TypeId,
    affinity: Option<&ResidentAffinity>,
    prepare: impl FnOnce(&mut PcuHostPreparation) -> Result<(), PcuExecutionError>,
    execute: impl FnOnce(&mut Prepared) -> Result<R, PcuExecutionError>,
) -> Result<R, PcuExecutionError> {
    let route = policy::route();
    STATE
        .try_with(|state| {
            let mut state = state
                .try_borrow_mut()
                .map_err(|_| PcuExecutionError::ReentrantCall)?;
            let hint = site.provider_slot.load(Ordering::Relaxed);
            let matches = |entry: &Entry| {
                entry.specialization == specialization
                    && match (&entry.affinity, affinity) {
                        (None, None) => true,
                        (Some(retained), Some(requested)) => affinity::same(retained, requested),
                        _ => false,
                    }
            };
            if state.generation == route.generation {
                let slot = if state.entries.get(hint).is_some_and(matches) {
                    Some(hint)
                } else {
                    state.entries.iter().position(matches)
                };
                if let Some(slot) = slot {
                    site.provider_slot.store(slot, Ordering::Relaxed);
                    return execute(&mut state.entries[slot].prepared);
                }
            }

            // Policy locks and environment inspection are cold-path work only. A warm entry stays
            // pinned to its prepared runtime even if process environment variables later change.
            let policy_snapshot = policy::snapshot()?;
            let policy = policy_snapshot.policy;
            if let Some(root) = affinity {
                affinity::validate(root, policy)?;
            }
            if matches!(policy.backend, PcuBackendChoice::Rocm) && !cfg!(feature = "rocm") {
                return Err(PcuExecutionError::NoBackendEnabled);
            }
            let generation = policy_snapshot.generation;
            let runtime_realm = RuntimeRealm::current();
            if state.generation != generation || state.realm.as_ref() != Some(&runtime_realm) {
                state.entries.clear();
                state.generation = generation;
                state.realm = Some(runtime_realm);
            }

            let mut context = PcuHostPreparation {
                #[cfg(feature = "rocm")]
                inner: None,
                shared: Some(Preparation::new(policy)),
            };
            if let Some(preparation) = context.shared.as_mut() {
                preparation.affinity = affinity.cloned();
            }
            prepare(&mut context)?;
            let (session, prepared) = context
                .shared
                .and_then(|mut context| context.result.take())
                .ok_or(PcuExecutionError::PreparationDidNotProduceKernel)?;
            if policy.cache_capacity == 0 {
                let mut prepared = prepared;
                return execute(&mut prepared);
            }
            let slot = if state.entries.len() == policy.cache_capacity {
                let victim = hint.min(state.entries.len() - 1);
                state.entries[victim] = Entry {
                    specialization,
                    _session: session,
                    affinity: affinity.cloned(),
                    prepared,
                };
                victim
            } else {
                state.entries.push(Entry {
                    specialization,
                    _session: session,
                    affinity: affinity.cloned(),
                    prepared,
                });
                state.entries.len() - 1
            };
            site.provider_slot.store(slot, Ordering::Relaxed);
            execute(&mut state.entries[slot].prepared)
        })
        .map_err(|_| PcuExecutionError::ThreadUnavailable)?
}

pub(super) fn clear_thread_cache() -> Result<(), PcuExecutionError> {
    STATE
        .try_with(|state| {
            let mut state = state
                .try_borrow_mut()
                .map_err(|_| PcuExecutionError::ReentrantCall)?;
            state.entries.clear();
            state.realm = None;
            Ok(())
        })
        .map_err(|_| PcuExecutionError::ThreadUnavailable)?
}

pub(super) fn call_arguments<const N: usize>(
    site: &PcuHostCallSite,
    specialization: TypeId,
    arguments: [super::arguments::PcuCallArgument<'_>; N],
    prepare: impl FnOnce(&mut PcuHostPreparation) -> Result<(), PcuExecutionError>,
) -> Result<(), PcuExecutionError> {
    let affinity = affinity::from_arguments(&arguments)?;
    with_entry(site, specialization, affinity, prepare, |prepared| {
        prepared.call_arguments(arguments)
    })
}

#[cfg(feature = "cuda")]
fn map_cuda_error(error: CudaHostKernelError) -> PcuExecutionError {
    match error {
        CudaHostKernelError::CheckedExecutionFault(fault) => {
            PcuExecutionError::ArithmeticFault(fault)
        }
        other => PcuExecutionError::BackendFailure(format!("CUDA execution: {other}")),
    }
}
#[cfg(feature = "rocm")]
fn map_rocm_error(error: RocmHostKernelError) -> PcuExecutionError {
    match error {
        RocmHostKernelError::CheckedExecutionFault(fault) => {
            PcuExecutionError::ArithmeticFault(fault)
        }
        other => PcuExecutionError::BackendFailure(format!("ROCm execution: {other}")),
    }
}

const fn provider_id(provider: Provider) -> u32 {
    match provider {
        #[cfg(feature = "cuda")]
        Provider::Cuda => 0x4355_4441,
        #[cfg(feature = "metal")]
        Provider::Metal => 0x4d54_4c31,
        #[cfg(feature = "rocm")]
        Provider::Rocm => 0x524f_434d,
        #[cfg(feature = "vulkan")]
        Provider::Vulkan => 0x564b_4c31,
        #[cfg(feature = "cpu")]
        Provider::Cpu => 0x4350_5531,
    }
}

pub(super) fn rank_candidates(candidates: &mut [Candidate]) {
    candidates.sort_by(|left, right| {
        right
            .score
            .cmp(&left.score)
            .then_with(|| provider_id(left.provider).cmp(&provider_id(right.provider)))
            .then_with(|| left.device.id.cmp(&right.device.id))
    });
}

const EMPTY_REF: PcuObjectRef = PcuObjectRef {
    provider: PcuProviderId(0),
    generation: 0,
    kind: PcuObjectKind::Target,
    id: 0,
};
const EMPTY_READY: PcuProviderReadiness<'static> = PcuProviderReadiness {
    status: PcuProviderStatus::Unavailable,
    reason: None,
};

#[cfg(feature = "cuda")]
pub(super) fn collect_cuda_candidates(
    discovery: &CudaDiscovery,
    policy: PcuExecutionPolicy,
    kernel: Option<&crate::PcuDispatchKernelIr<'_>>,
    out: &mut Vec<Candidate>,
) -> Result<(), fusion_pcu_cuda::CudaError> {
    let mut providers = [PcuProviderDescriptor {
        id: PcuProviderId(0),
        generation: 0,
        backend: "",
        readiness: EMPTY_READY,
    }];
    discovery.providers(&mut providers)?;
    let mut targets = [PcuTargetDescriptor {
        reference: EMPTY_REF,
        name: "",
        readiness: EMPTY_READY,
    }];
    discovery.targets(providers[0].id, providers[0].generation, &mut targets)?;
    let count = discovery.devices(targets[0].reference, &mut [])?;
    let blank = PcuDeviceDescriptor {
        reference: EMPTY_REF,
        target: EMPTY_REF,
        name: "",
        class: PcuDeviceClass::Other,
        vendor: None,
        architecture: None,
        generation: None,
        location: None,
    };
    let mut devices = vec![blank; count];
    discovery.devices(targets[0].reference, &mut devices)?;
    for descriptor in devices {
        if policy
            .device
            .is_some_and(|id| id != descriptor.reference.id)
        {
            continue;
        }
        let memory = discovery
            .device_info(descriptor.reference)
            .ok()
            .map(|info| info.total_memory);
        out.push(Candidate {
            provider: Provider::Cuda,
            device: descriptor.reference,
            score: super::selection::score_candidate(policy, descriptor, memory, kernel, || {
                discovery.device_facts(descriptor.reference)
            })?,
        });
    }
    Ok(())
}

#[cfg(feature = "rocm")]
pub(super) fn collect_rocm_candidates(
    discovery: &RocmDiscovery,
    policy: PcuExecutionPolicy,
    kernel: Option<&crate::PcuDispatchKernelIr<'_>>,
    out: &mut Vec<Candidate>,
) -> Result<(), fusion_pcu_rocm::HipError> {
    let mut providers = [PcuProviderDescriptor {
        id: PcuProviderId(0),
        generation: 0,
        backend: "",
        readiness: EMPTY_READY,
    }];
    discovery.providers(&mut providers)?;
    let mut targets = [PcuTargetDescriptor {
        reference: EMPTY_REF,
        name: "",
        readiness: EMPTY_READY,
    }];
    discovery.targets(providers[0].id, providers[0].generation, &mut targets)?;
    let count = discovery.devices(targets[0].reference, &mut [])?;
    let blank = PcuDeviceDescriptor {
        reference: EMPTY_REF,
        target: EMPTY_REF,
        name: "",
        class: PcuDeviceClass::Other,
        vendor: None,
        architecture: None,
        generation: None,
        location: None,
    };
    let mut devices = vec![blank; count];
    discovery.devices(targets[0].reference, &mut devices)?;
    for descriptor in devices {
        if policy
            .device
            .is_some_and(|id| id != descriptor.reference.id)
        {
            continue;
        }
        let memory = discovery
            .device_info(descriptor.reference)
            .ok()
            .map(|info| info.total_memory);
        out.push(Candidate {
            provider: Provider::Rocm,
            device: descriptor.reference,
            score: super::selection::score_candidate(policy, descriptor, memory, kernel, || {
                discovery.device_facts(descriptor.reference)
            })?,
        });
    }
    Ok(())
}

#[cfg(feature = "metal")]
fn open_metal(
    discovery: Option<&Result<MetalDiscovery, fusion_pcu_metal::MetalError>>,
    device: PcuObjectRef,
) -> Result<Session, PcuExecutionError> {
    let discovery = discovery
        .ok_or(PcuExecutionError::NoBackendEnabled)?
        .as_ref()
        .map_err(|error| PcuExecutionError::BackendFailure(error.to_string()))?;
    discovery
        .open_owned_device(device)
        .map(|backend| Session::Metal(Rc::new(backend)))
        .map_err(|error| {
            PcuExecutionError::BackendFailure(format!("Metal initialization: {error}"))
        })
}
#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;

    #[test]
    fn unproved_portable_invocation_rejects_before_provider_discovery() {
        let options = crate::PcuNumericalOptions {
            reproducibility: crate::PcuReproducibility::PortableV1,
            ..crate::PcuNumericalOptions::default()
        };
        let mut preparation = Preparation::new(PcuExecutionPolicy {
            numerical_options: options,
            ..PcuExecutionPolicy::default()
        });
        let kernel = crate::PcuDispatchKernelIr {
            id: crate::PcuKernelId(1),
            entry: crate::PcuDispatchEntryPoint {
                name: "portable_probe",
                logical_shape: [1, 1, 1],
            },
            bindings: &[],
            ports: &[],
            parameters: &[],
            ops: &[],
            type_caps: crate::PcuValueTypeCaps::default(),
            feature_caps: crate::PcuDispatchFeatureCaps::default(),
        };
        assert!(matches!(
            preparation.prepare(&kernel),
            Err(PcuExecutionError::UnsupportedNumericalOptions(rejected)) if rejected == options
        ));
        assert!(preparation.result.is_none());
    }

    #[test]
    fn provider_candidate_order_uses_score_then_stable_provider_and_ordinal() {
        let device = |provider, ordinal| PcuObjectRef {
            provider: PcuProviderId(provider),
            generation: 1,
            kind: PcuObjectKind::Device,
            id: ordinal,
        };
        let mut candidates = vec![
            Candidate {
                provider: Provider::Cuda,
                device: device(provider_id(Provider::Cuda), 2),
                score: 9,
            },
            Candidate {
                provider: Provider::Cuda,
                device: device(provider_id(Provider::Cuda), 0),
                score: 10,
            },
            Candidate {
                provider: Provider::Cuda,
                device: device(provider_id(Provider::Cuda), 1),
                score: 10,
            },
            #[cfg(feature = "rocm")]
            Candidate {
                provider: Provider::Rocm,
                device: device(provider_id(Provider::Rocm), 0),
                score: 10,
            },
        ];
        rank_candidates(&mut candidates);
        assert_eq!(candidates[0].device.id, 0);
        assert_eq!(candidates[1].device.id, 1);
        #[cfg(feature = "rocm")]
        {
            assert_eq!(candidates[2].provider, Provider::Rocm);
            assert_eq!(candidates[3].device.id, 2);
        }
        #[cfg(not(feature = "rocm"))]
        assert_eq!(candidates[2].device.id, 2);
    }
}

#[cfg(feature = "metal")]
fn map_metal_error(error: fusion_pcu_metal::MetalHostKernelError) -> PcuExecutionError {
    match error {
        crate::PcuHostDispatchError::Backend(fusion_pcu_metal::MetalError::Arithmetic(fault)) => {
            PcuExecutionError::ArithmeticFault(fault)
        }
        other => PcuExecutionError::BackendFailure(format!("Metal source execution: {other:?}")),
    }
}
#[cfg(feature = "metal")]
pub(super) fn collect_metal_candidates(
    discovery: &MetalDiscovery,
    policy: PcuExecutionPolicy,
    kernel: Option<&crate::PcuDispatchKernelIr<'_>>,
    out: &mut Vec<Candidate>,
) -> Result<(), fusion_pcu_metal::MetalError> {
    let mut providers = [PcuProviderDescriptor {
        id: PcuProviderId(0),
        generation: 0,
        backend: "",
        readiness: EMPTY_READY,
    }];
    discovery.providers(&mut providers)?;
    let mut targets = [PcuTargetDescriptor {
        reference: EMPTY_REF,
        name: "",
        readiness: EMPTY_READY,
    }];
    discovery.targets(providers[0].id, providers[0].generation, &mut targets)?;
    let count = discovery.devices(targets[0].reference, &mut [])?;
    let blank = PcuDeviceDescriptor {
        reference: EMPTY_REF,
        target: EMPTY_REF,
        name: "",
        class: PcuDeviceClass::Other,
        vendor: None,
        architecture: None,
        generation: None,
        location: None,
    };
    let mut devices = vec![blank; count];
    discovery.devices(targets[0].reference, &mut devices)?;
    for descriptor in devices {
        if policy
            .device
            .is_some_and(|id| id != descriptor.reference.id)
        {
            continue;
        }
        // Metal exposes max single allocation, not physical capacity; report unknown capacity.
        out.push(Candidate {
            provider: Provider::Metal,
            device: descriptor.reference,
            score: super::selection::score_candidate(policy, descriptor, None, kernel, || {
                discovery.device_facts(descriptor.reference)
            })?,
        });
    }
    Ok(())
}

#[cfg(feature = "metal")]
fn call_metal_arguments<'a, const N: usize>(
    prepared: &mut fusion_pcu_metal::MetalPreparedHostKernel,
    arguments: [super::arguments::PcuCallArgument<'a>; N],
) -> Result<(), PcuExecutionError> {
    let mut guards: [Option<ResidentWriteGuard<'a>>; N] = core::array::from_fn(|_| None);

    let mut bindings: SmallVec<[fusion_pcu_metal::MetalMixedHostArgument<'_>; 8]> = SmallVec::new();
    for (index, argument) in arguments.into_iter().enumerate() {
        let (_, kind) = argument.into_parts();
        let binding = match kind {
            PcuCallArgumentKind::Host(host) => fusion_pcu_metal::MetalMixedHostArgument::Host(host),
            PcuCallArgumentKind::ResidentRead(resident) => {
                #[allow(irrefutable_let_patterns)]
                let DeviceArgument::Metal(argument) = resident.argument else {
                    return Err(PcuExecutionError::Argument(
                        super::PcuArgumentError::SessionMismatch,
                    ));
                };
                fusion_pcu_metal::MetalMixedHostArgument::Resident(argument)
            }
            PcuCallArgumentKind::ResidentWrite(resident) => {
                #[allow(irrefutable_let_patterns)]
                let DeviceArgument::Metal(argument) = resident.argument else {
                    return Err(PcuExecutionError::Argument(
                        super::PcuArgumentError::SessionMismatch,
                    ));
                };
                guards[index] = Some(resident.guard);
                fusion_pcu_metal::MetalMixedHostArgument::Resident(argument)
            }
        };
        bindings.push(binding);
    }
    for guard in guards.iter_mut().flatten() {
        guard.mark_may_have_written();
    }
    let result = prepared.call_mixed(&mut bindings);
    if result.is_ok() {
        for guard in guards.iter_mut().flatten() {
            guard.mark_complete();
        }
    } else if !prepared.last_call_completion_uncertain() {
        for guard in guards.iter_mut().flatten() {
            guard.mark_known_partial();
        }
    }
    result.map_err(map_metal_error)
}

#[cfg(feature = "cuda")]
fn call_cuda_arguments<'a, const N: usize>(
    prepared: &mut fusion_pcu_cuda::CudaPreparedHostKernel,
    arguments: [super::arguments::PcuCallArgument<'a>; N],
) -> Result<(), PcuExecutionError> {
    let mut guards: [Option<ResidentWriteGuard<'a>>; N] = core::array::from_fn(|_| None);

    let mut bindings: SmallVec<[fusion_pcu_cuda::CudaMixedHostArgument<'_>; 8]> = SmallVec::new();
    for (index, argument) in arguments.into_iter().enumerate() {
        let (_, kind) = argument.into_parts();
        let binding = match kind {
            PcuCallArgumentKind::Host(host) => fusion_pcu_cuda::CudaMixedHostArgument::Host(host),
            PcuCallArgumentKind::ResidentRead(resident) => {
                #[allow(irrefutable_let_patterns)]
                let DeviceArgument::Cuda(argument) = resident.argument else {
                    return Err(PcuExecutionError::Argument(
                        super::PcuArgumentError::SessionMismatch,
                    ));
                };
                fusion_pcu_cuda::CudaMixedHostArgument::Resident(argument)
            }
            PcuCallArgumentKind::ResidentWrite(resident) => {
                #[allow(irrefutable_let_patterns)]
                let DeviceArgument::Cuda(argument) = resident.argument else {
                    return Err(PcuExecutionError::Argument(
                        super::PcuArgumentError::SessionMismatch,
                    ));
                };
                guards[index] = Some(resident.guard);
                fusion_pcu_cuda::CudaMixedHostArgument::Resident(argument)
            }
        };
        bindings.push(binding);
    }
    for guard in guards.iter_mut().flatten() {
        guard.mark_may_have_written();
    }
    let result = prepared.call_mixed(&mut bindings);
    if result.is_ok() {
        for guard in guards.iter_mut().flatten() {
            guard.mark_complete();
        }
    } else if !prepared.last_call_completion_uncertain() {
        for guard in guards.iter_mut().flatten() {
            guard.mark_known_partial();
        }
    }
    result.map_err(map_cuda_error)
}

#[cfg(feature = "rocm")]
fn call_rocm_arguments<'a, const N: usize>(
    prepared: &mut fusion_pcu_rocm::RocmPreparedHostKernel,
    arguments: [super::arguments::PcuCallArgument<'a>; N],
) -> Result<(), PcuExecutionError> {
    let mut guards: [Option<ResidentWriteGuard<'a>>; N] = core::array::from_fn(|_| None);

    let mut bindings: SmallVec<[fusion_pcu_rocm::RocmMixedHostArgument<'_>; 8]> = SmallVec::new();
    for (index, argument) in arguments.into_iter().enumerate() {
        let (_, kind) = argument.into_parts();
        let binding = match kind {
            PcuCallArgumentKind::Host(host) => fusion_pcu_rocm::RocmMixedHostArgument::Host(host),
            PcuCallArgumentKind::ResidentRead(resident) => {
                #[allow(irrefutable_let_patterns)]
                let DeviceArgument::Rocm(argument) = resident.argument else {
                    return Err(PcuExecutionError::Argument(
                        super::PcuArgumentError::SessionMismatch,
                    ));
                };
                fusion_pcu_rocm::RocmMixedHostArgument::Resident(argument)
            }
            PcuCallArgumentKind::ResidentWrite(resident) => {
                #[allow(irrefutable_let_patterns)]
                let DeviceArgument::Rocm(argument) = resident.argument else {
                    return Err(PcuExecutionError::Argument(
                        super::PcuArgumentError::SessionMismatch,
                    ));
                };
                guards[index] = Some(resident.guard);
                fusion_pcu_rocm::RocmMixedHostArgument::Resident(argument)
            }
        };
        bindings.push(binding);
    }
    for guard in guards.iter_mut().flatten() {
        guard.mark_may_have_written();
    }
    let result = prepared.call_mixed(&mut bindings);
    if result.is_ok() {
        for guard in guards.iter_mut().flatten() {
            guard.mark_complete();
        }
    } else if !prepared.last_call_completion_uncertain() {
        for guard in guards.iter_mut().flatten() {
            guard.mark_known_partial();
        }
    }
    result.map_err(map_rocm_error)
}

fn prepare_session(
    session: &Session,
    kernel: &crate::PcuDispatchKernelIr<'_>,
) -> Result<Prepared, PcuExecutionError> {
    match session {
        #[cfg(feature = "metal")]
        Session::Metal(backend) => backend
            .prepare_host_kernel(kernel)
            .map(Prepared::Metal)
            .map_err(|error| {
                PcuExecutionError::BackendFailure(format!("Metal preparation: {error:?}"))
            }),
        #[cfg(any(feature = "rocm", feature = "cuda", feature = "metal"))]
        Session::Resident(root) => root.prepare_host_kernel(kernel),
        #[cfg(feature = "cpu")]
        Session::Cpu(backend) => backend
            .prepare_host_kernel(kernel)
            .map(Prepared::Cpu)
            .map_err(map_cpu_error),
        #[cfg(feature = "vulkan")]
        Session::Vulkan(backend) => backend
            .prepare_host_kernel(kernel)
            .map(Prepared::Vulkan)
            .map_err(map_vulkan_error),
        #[cfg(feature = "cuda")]
        Session::Cuda(backend) => backend
            .prepare_host_kernel(kernel)
            .map(Prepared::Cuda)
            .map_err(|error| {
                PcuExecutionError::BackendFailure(format!("CUDA preparation: {error}"))
            }),
        #[cfg(feature = "rocm")]
        Session::Rocm(backend) => backend
            .prepare_host_kernel(kernel)
            .map(Prepared::Rocm)
            .map_err(|error| {
                PcuExecutionError::BackendFailure(format!("ROCm preparation: {error}"))
            }),
    }
}

fn rejected_candidates(
    rejected: Vec<(PcuObjectRef, PcuExecutionError)>,
    discovery: Vec<String>,
) -> PcuExecutionError {
    if rejected.is_empty() && discovery.is_empty() {
        return PcuExecutionError::NoBackendEnabled;
    }
    PcuExecutionError::NoCompatibleInvocationDevice {
        rejected,
        discovery,
    }
}

#[cfg(feature = "vulkan")]
fn map_vulkan_error(error: PcuVulkanError) -> PcuExecutionError {
    match error {
        PcuVulkanError::Fault(fault) => PcuExecutionError::ArithmeticFault(fault),
        other => PcuExecutionError::VulkanExecution(other),
    }
}

#[cfg(feature = "vulkan")]
fn open_vulkan(
    discovery: Option<&Result<PcuVulkanDiscovery, PcuVulkanError>>,
    device: PcuObjectRef,
) -> Result<Session, PcuExecutionError> {
    let discovery = discovery.ok_or(PcuExecutionError::NoBackendEnabled)?;
    discovery.as_ref().map_or_else(
        |_| Err(PcuExecutionError::NoBackendEnabled),
        |discovery| {
            PcuVulkanBackend::open(discovery, device)
                .map(|backend| Session::Vulkan(Rc::new(backend)))
                .map_err(map_vulkan_error)
        },
    )
}

#[cfg(feature = "vulkan")]
fn collect_vulkan_candidates(
    discovery: &PcuVulkanDiscovery,
    policy: PcuExecutionPolicy,
    kernel: &crate::PcuDispatchKernelIr<'_>,
    out: &mut Vec<Candidate>,
) -> Result<(), PcuVulkanError> {
    let mut providers = [PcuProviderDescriptor {
        id: PcuProviderId(0),
        generation: 0,
        backend: "",
        readiness: EMPTY_READY,
    }];
    discovery.providers(&mut providers)?;
    let mut targets = [PcuTargetDescriptor {
        reference: EMPTY_REF,
        name: "",
        readiness: EMPTY_READY,
    }];
    discovery.targets(providers[0].id, providers[0].generation, &mut targets)?;
    let count = discovery.devices(targets[0].reference, &mut [])?;
    let blank = PcuDeviceDescriptor {
        reference: EMPTY_REF,
        target: EMPTY_REF,
        name: "",
        class: PcuDeviceClass::Other,
        vendor: None,
        architecture: None,
        generation: None,
        location: None,
    };
    let mut devices = vec![blank; count];
    discovery.devices(targets[0].reference, &mut devices)?;
    for descriptor in devices {
        if policy
            .device
            .is_some_and(|id| id != descriptor.reference.id)
        {
            continue;
        }
        // Vulkan memory heaps are not yet exposed as neutral physical capacity. Keep it unknown.
        let score =
            super::selection::score_candidate(policy, descriptor, None, Some(kernel), || {
                discovery.device_facts(descriptor.reference)
            })?;
        out.push(Candidate {
            provider: Provider::Vulkan,
            device: descriptor.reference,
            score,
        });
    }
    Ok(())
}

#[cfg(feature = "vulkan")]
fn call_vulkan_arguments<const N: usize>(
    prepared: &mut PcuVulkanPreparedBitMap,
    arguments: [super::arguments::PcuCallArgument<'_>; N],
) -> Result<(), PcuExecutionError> {
    // This profile admits exactly two host bindings. A fixed stack array adds no warm heap work.
    // Other providers' resident arguments are never silently read back or reinterpreted.
    if N != 2 {
        return Err(PcuExecutionError::VulkanExecution(
            PcuVulkanError::InvalidArguments,
        ));
    }
    let mut source = arguments.into_iter();
    let mut host = || match source
        .next()
        .ok_or(PcuExecutionError::VulkanExecution(
            PcuVulkanError::InvalidArguments,
        ))?
        .into_parts()
        .1
    {
        super::arguments::PcuCallArgumentKind::Host(argument) => Ok(argument),
        #[cfg(any(feature = "rocm", feature = "cuda", feature = "metal"))]
        _ => Err(PcuExecutionError::Argument(
            super::PcuArgumentError::SessionMismatch,
        )),
    };
    prepared
        .call(&mut [host()?, host()?])
        .map_err(map_vulkan_error)
}

#[cfg(feature = "cpu")]
fn map_cpu_error(error: PcuCpuHostError) -> PcuExecutionError {
    error.fault().map_or(
        PcuExecutionError::CpuExecution(error),
        PcuExecutionError::ArithmeticFault,
    )
}

#[cfg(feature = "cpu")]
fn collect_cpu_candidates(
    discovery: &PcuCpuDiscovery,
    policy: PcuExecutionPolicy,
    kernel: &crate::PcuDispatchKernelIr<'_>,
    out: &mut Vec<Candidate>,
) -> Result<(), PcuCpuDiscoveryError> {
    let mut providers = [PcuProviderDescriptor {
        id: PcuProviderId(0),
        generation: 0,
        backend: "",
        readiness: EMPTY_READY,
    }];
    discovery.providers(&mut providers)?;
    let mut targets = [PcuTargetDescriptor {
        reference: EMPTY_REF,
        name: "",
        readiness: EMPTY_READY,
    }];
    discovery.targets(providers[0].id, providers[0].generation, &mut targets)?;
    let mut devices = [PcuDeviceDescriptor {
        reference: EMPTY_REF,
        target: EMPTY_REF,
        name: "",
        class: PcuDeviceClass::Other,
        vendor: None,
        architecture: None,
        generation: None,
        location: None,
    }];
    discovery.devices(targets[0].reference, &mut devices)?;
    let descriptor = devices[0];
    if policy.device.is_none_or(|id| id == descriptor.reference.id) {
        let score =
            super::selection::score_candidate(policy, descriptor, None, Some(kernel), || {
                discovery.device_facts(descriptor.reference)
            })?;
        out.push(Candidate {
            provider: Provider::Cpu,
            device: descriptor.reference,
            score,
        });
    }
    Ok(())
}

#[cfg(feature = "cpu")]
fn call_cpu_arguments<const N: usize>(
    prepared: &mut PcuCpuPreparedHost,
    arguments: [super::arguments::PcuCallArgument<'_>; N],
) -> Result<(), PcuExecutionError> {
    let expected = if matches!(prepared, PcuCpuPreparedHost::Neg(_)) {
        2
    } else {
        3
    };
    if N != expected {
        return Err(PcuExecutionError::CpuExecution(PcuCpuHostError::Arguments(
            PcuCpuHostArgumentError::Count {
                expected,
                actual: N,
            },
        )));
    }
    let mut source = arguments.into_iter();
    let mut host = || match source
        .next()
        .ok_or(PcuExecutionError::CpuExecution(PcuCpuHostError::Arguments(
            PcuCpuHostArgumentError::Count {
                expected,
                actual: N,
            },
        )))?
        .into_parts()
        .1
    {
        super::arguments::PcuCallArgumentKind::Host(argument) => Ok(argument),
        #[cfg(any(feature = "rocm", feature = "cuda", feature = "metal"))]
        _ => Err(PcuExecutionError::Argument(
            super::PcuArgumentError::SessionMismatch,
        )),
    };
    // Profile counts are frozen at preparation. Arguments stay on the stack at warm submission.
    if expected == 2 {
        prepared.call(&mut [host()?, host()?])
    } else {
        prepared.call(&mut [host()?, host()?, host()?])
    }
    .map_err(map_cpu_error)
}
