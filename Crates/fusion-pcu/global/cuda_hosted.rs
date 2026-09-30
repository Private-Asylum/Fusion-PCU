//! Shared typed host/resident invocation path used when CUDA is compiled. Automatic selection
//! ranks candidates from every compiled runtime; resident affinity pins borrowed device storage.

use core::any::TypeId;
use core::sync::atomic::Ordering;
use std::cell::RefCell;
use std::ffi::OsString;
use std::rc::Rc;
use smallvec::SmallVec;
#[rustfmt::skip]
use fusion_pcu_cuda::{
    CudaDiscovery,
    CudaHostKernelError,
    CudaOwnedDispatchBackend,
    CudaPreparedHostKernel,
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
use super::arguments::{
    PcuCallArgumentKind,
    ResidentWriteGuard,
};
use super::resident::DeviceArgument;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Provider {
    Cuda,
    #[cfg(feature = "rocm")]
    Rocm,
}

#[derive(Clone)]
pub(super) struct Candidate {
    pub(super) provider: Provider,
    pub(super) device: PcuObjectRef,
    score: i128,
}

#[derive(Clone)]
enum Session {
    Cuda(Rc<CudaOwnedDispatchBackend>),
    Resident(Rc<super::resident::Session>),
    #[cfg(feature = "rocm")]
    Rocm(Rc<RocmOwnedDispatchBackend>),
}

pub(super) enum Prepared {
    Cuda(CudaPreparedHostKernel),
    #[cfg(feature = "rocm")]
    Rocm(RocmPreparedHostKernel),
}

impl Prepared {
    fn call(&mut self, args: &mut [PcuHostArgument<'_>]) -> Result<(), PcuExecutionError> {
        match self {
            Self::Cuda(prepared) => prepared.call(args).map_err(map_cuda_error),
            #[cfg(feature = "rocm")]
            Self::Rocm(prepared) => prepared.call(args).map_err(map_rocm_error),
        }
    }
    fn call_arguments<'a, const N: usize>(
        &mut self,
        arguments: [super::arguments::PcuCallArgument<'a>; N],
    ) -> Result<(), PcuExecutionError> {
        let mut guards: [Option<ResidentWriteGuard<'a>>; N] = core::array::from_fn(|_| None);
        match self {
            #[cfg(feature = "cuda")]
            Self::Cuda(prepared) => {
                let mut bindings: SmallVec<[fusion_pcu_cuda::CudaMixedHostArgument<'_>; 8]> =
                    SmallVec::new();
                for (index, argument) in arguments.into_iter().enumerate() {
                    let (_, kind) = argument.into_parts();
                    let binding = match kind {
                        PcuCallArgumentKind::Host(host) => {
                            fusion_pcu_cuda::CudaMixedHostArgument::Host(host)
                        }
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
            Self::Rocm(prepared) => {
                let mut bindings: SmallVec<[fusion_pcu_rocm::RocmMixedHostArgument<'_>; 8]> =
                    SmallVec::new();
                for (index, argument) in arguments.into_iter().enumerate() {
                    let (_, kind) = argument.into_parts();
                    let binding = match kind {
                        PcuCallArgumentKind::Host(host) => {
                            fusion_pcu_rocm::RocmMixedHostArgument::Host(host)
                        }
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
        }
    }
}

pub(super) struct Preparation {
    policy: PcuExecutionPolicy,
    result: Option<(Session, Prepared)>,
    affinity: Option<Rc<super::resident::Session>>,
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

    pub(super) fn prepare(
        &mut self,
        kernel: &crate::PcuDispatchKernelIr<'_>,
    ) -> Result<(), PcuExecutionError> {
        if let Some(root) = &self.affinity {
            root.validate_policy(self.policy)?;
            let prepared = root.prepare_host_kernel(kernel)?;
            self.result = Some((Session::Resident(Rc::clone(root)), prepared));
            return Ok(());
        }
        let cuda_discovery = CudaDiscovery::new();
        #[cfg(feature = "rocm")]
        let rocm_discovery = RocmDiscovery::new();
        let mut candidates = Vec::new();
        let mut discovery_errors = Vec::new();

        if matches!(
            self.policy.backend,
            PcuBackendChoice::Automatic | PcuBackendChoice::Cuda
        ) && let Err(error) =
            collect_cuda_candidates(&cuda_discovery, self.policy, &mut candidates)
        {
            discovery_errors.push(format!("CUDA discovery: {error}"));
        }
        #[cfg(feature = "rocm")]
        if matches!(
            self.policy.backend,
            PcuBackendChoice::Automatic | PcuBackendChoice::Rocm
        ) && let Err(error) =
            collect_rocm_candidates(&rocm_discovery, self.policy, &mut candidates)
        {
            discovery_errors.push(format!("ROCm discovery: {error}"));
        }
        rank_candidates(&mut candidates);

        let mut rejected = Vec::new();
        for candidate in candidates {
            let opened = match candidate.provider {
                Provider::Cuda => CudaOwnedDispatchBackend::open(
                    &cuda_discovery,
                    candidate.device,
                    self.policy.block_size,
                )
                .map(|backend| Session::Cuda(Rc::new(backend)))
                .map_err(|error| format!("CUDA initialization: {error}")),
                #[cfg(feature = "rocm")]
                Provider::Rocm => RocmOwnedDispatchBackend::open(
                    &rocm_discovery,
                    candidate.device,
                    self.policy.block_size,
                )
                .map(|backend| Session::Rocm(Rc::new(backend)))
                .map_err(|error| format!("ROCm initialization: {error}")),
            };
            let session = match opened {
                Ok(session) => session,
                Err(error) => {
                    rejected.push((candidate.device.id, error));
                    continue;
                }
            };
            let prepared = match &session {
                Session::Resident(root) => root
                    .prepare_host_kernel(kernel)
                    .map_err(|error| error.to_string()),
                Session::Cuda(backend) => backend
                    .prepare_host_kernel(kernel)
                    .map(Prepared::Cuda)
                    .map_err(|error| format!("CUDA preparation: {error}")),
                #[cfg(feature = "rocm")]
                Session::Rocm(backend) => backend
                    .prepare_host_kernel(kernel)
                    .map(Prepared::Rocm)
                    .map_err(|error| format!("ROCm preparation: {error}")),
            };
            match prepared {
                Ok(prepared) => {
                    self.result = Some((session, prepared));
                    return Ok(());
                }
                Err(error) => rejected.push((candidate.device.id, error)),
            }
        }
        if rejected.is_empty() && discovery_errors.is_empty() {
            return Err(PcuExecutionError::NoBackendEnabled);
        }
        Err(PcuExecutionError::BackendFailure(format!(
            "no compatible device; rejected={rejected:?}; discovery={discovery_errors:?}"
        )))
    }
}

struct Entry {
    specialization: TypeId,
    _session: Session,
    affinity: Option<Rc<super::resident::Session>>,
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
    affinity: Option<&Rc<super::resident::Session>>,
    prepare: impl FnOnce(&mut PcuHostPreparation) -> Result<(), PcuExecutionError>,
    execute: impl FnOnce(&mut Prepared) -> Result<R, PcuExecutionError>,
) -> Result<R, PcuExecutionError> {
    let route = policy::route();
    STATE
        .try_with(|state| {
            let mut state = state
                .try_borrow_mut()
                .map_err(|_| PcuExecutionError::ReentrantCall)?;
            let hint = site.cuda_slot.load(Ordering::Relaxed);
            let matches = |entry: &Entry| {
                entry.specialization == specialization
                    && match (&entry.affinity, affinity) {
                        (None, None) => true,
                        (Some(retained), Some(requested)) => Rc::ptr_eq(retained, requested),
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
                    site.cuda_slot.store(slot, Ordering::Relaxed);
                    return execute(&mut state.entries[slot].prepared);
                }
            }

            // Policy locks and environment inspection are cold-path work only. A warm entry stays
            // pinned to its prepared runtime even if process environment variables later change.
            let policy_snapshot = policy::snapshot()?;
            let policy = policy_snapshot.policy;
            if let Some(root) = affinity {
                root.validate_policy(policy)?;
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
                cuda: Some(Preparation::new(policy)),
            };
            if let Some(preparation) = context.cuda.as_mut() {
                preparation.affinity = affinity.map(Rc::clone);
            }
            prepare(&mut context)?;
            let (session, prepared) = context
                .cuda
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
                    affinity: affinity.map(Rc::clone),
                    prepared,
                };
                victim
            } else {
                state.entries.push(Entry {
                    specialization,
                    _session: session,
                    affinity: affinity.map(Rc::clone),
                    prepared,
                });
                state.entries.len() - 1
            };
            site.cuda_slot.store(slot, Ordering::Relaxed);
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
    let mut affinity = None;
    for argument in &arguments {
        let session = match argument.kind() {
            super::arguments::PcuCallArgumentKind::Host(_) => continue,
            super::arguments::PcuCallArgumentKind::ResidentRead(resident) => resident.session,
            super::arguments::PcuCallArgumentKind::ResidentWrite(resident) => resident.session,
        };
        if affinity.is_some_and(|root| !Rc::ptr_eq(root, session)) {
            return Err(PcuExecutionError::Argument(
                super::PcuArgumentError::SessionMismatch,
            ));
        }
        affinity = Some(session);
    }
    with_entry(site, specialization, affinity, prepare, |prepared| {
        prepared.call_arguments(arguments)
    })
}

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
        Provider::Cuda => 0x4355_4441,
        #[cfg(feature = "rocm")]
        Provider::Rocm => 0x524f_434d,
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

pub(super) fn collect_cuda_candidates(
    discovery: &CudaDiscovery,
    policy: PcuExecutionPolicy,
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
            .map_or(0, |info| info.total_memory);
        out.push(Candidate {
            provider: Provider::Cuda,
            device: descriptor.reference,
            score: (policy.score_device)(&descriptor, memory),
        });
    }
    Ok(())
}

#[cfg(feature = "rocm")]
pub(super) fn collect_rocm_candidates(
    discovery: &RocmDiscovery,
    policy: PcuExecutionPolicy,
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
            .map_or(0, |info| info.total_memory);
        out.push(Candidate {
            provider: Provider::Rocm,
            device: descriptor.reference,
            score: (policy.score_device)(&descriptor, memory),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

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
