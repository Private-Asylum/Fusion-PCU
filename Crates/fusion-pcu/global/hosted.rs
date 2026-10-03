//! Thread-owned hosted state. Cold paths own discovery, ranking and compilation.

use core::any::TypeId;
use std::ffi::OsString;
use core::sync::atomic::Ordering;
use std::cell::RefCell;
use std::rc::Rc;
use super::session::RocmSession;
use super::policy;
#[cfg(feature = "tensor")]
#[cfg(all(feature = "tensor", not(feature = "cuda")))]
use super::policy::PolicySnapshot;

#[rustfmt::skip]
use fusion_pcu_rocm::{
    RocmDiscovery,
    RocmOwnedDispatchBackend,
    RocmPreparedHostKernel,
};
#[rustfmt::skip]
use crate::{
    PcuDeviceClass,
    PcuDeviceDescriptor,
    PcuDispatchKernelIr,
    PcuHostArgument,
    PcuHostKernelBackend,
    PcuObjectKind,
    PcuObjectRef,
    PcuOwnedDispatchBackend,
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
    PcuExecutionError,
    PcuExecutionPolicy,
    PcuHostCallSite,
    PcuHostPreparation,
};

struct Entry {
    specialization: TypeId,
    prepared: RocmPreparedHostKernel,
    // Keep the execution domain alive independently of the bounded discovery/session cache.
    session: Rc<RocmSession>,
    resident_affinity: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct RuntimeRealm {
    hip_library: Option<OsString>,
    hipcc: Option<OsString>,
    path: Option<OsString>,
    hiprtc_library: Option<OsString>,
    #[cfg(feature = "tensor")]
    rocblas_library: Option<OsString>,
}

impl RuntimeRealm {
    fn current() -> Self {
        Self {
            hip_library: std::env::var_os("HIP_RUNTIME_LIBRARY"),
            hipcc: std::env::var_os("HIPCC"),
            path: std::env::var_os("PATH"),
            hiprtc_library: std::env::var_os("HIPRTC_LIBRARY"),
            #[cfg(feature = "tensor")]
            rocblas_library: std::env::var_os("ROCBLAS_LIBRARY"),
        }
    }
}

struct SessionEntry {
    key: SessionKey,
    session: Rc<RocmSession>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct SessionKey {
    device: PcuObjectRef,
    pci_bus_id: Option<String>,
    runtime_realm: RuntimeRealm,
}
#[derive(Default)]
struct SessionArena {
    runtime_realm: Option<RuntimeRealm>,
    discovery: Option<RocmDiscovery>,
    sessions: Vec<SessionEntry>,
}
#[derive(Default)]
struct ThreadState {
    generation: u64,
    entries: Vec<Entry>,
    arena: SessionArena,
}
std::thread_local! { static STATE: RefCell<ThreadState> = RefCell::new(ThreadState::default()); }

pub(super) struct Preparation {
    policy: PcuExecutionPolicy,
    prepared: Option<RocmPreparedHostKernel>,
    session: Option<Rc<RocmSession>>,
    affinity: Option<Rc<RocmSession>>,
    arena: SessionArena,
}

impl Preparation {
    pub(super) const fn numerical_requirements(&self) -> crate::PcuImplementationRequirements {
        crate::PcuImplementationRequirements {
            numerical_mode: self.policy.numerical_mode,
            numerical_options: self.policy.numerical_options,
            float_underflow: self.policy.float_underflow,
            range_policy: self.policy.range_policy,
        }
    }

    pub(super) fn prepare(
        &mut self,
        kernel: &PcuDispatchKernelIr<'_>,
    ) -> Result<(), PcuExecutionError> {
        super::numerical::validate_invocation_contract(kernel)?;
        let (session, prepared) = prepare_in_arena(
            &mut self.arena,
            self.policy,
            self.affinity.as_ref(),
            Some(kernel),
            |session| {
                session
                    .backend()
                    .prepare_host_kernel(kernel)
                    .map_err(PcuExecutionError::from)
            },
        )?;
        self.session = Some(session);
        self.prepared = Some(prepared);
        Ok(())
    }
}

fn prepare_in_arena<R>(
    arena: &mut SessionArena,
    policy: PcuExecutionPolicy,
    affinity: Option<&Rc<RocmSession>>,
    kernel: Option<&PcuDispatchKernelIr<'_>>,
    mut prepare: impl FnMut(&Rc<RocmSession>) -> Result<R, PcuExecutionError>,
) -> Result<(Rc<RocmSession>, R), PcuExecutionError> {
    if let Some(session) = affinity {
        if policy
            .device
            .is_some_and(|device| device != session.backend().device_identity().device_id())
            || policy.block_size != session.block_size()
        {
            return Err(PcuExecutionError::ResidentPolicyConflict);
        }
        let prepared = prepare(session)?;
        return Ok((Rc::clone(session), prepared));
    }
    let discovery = arena.discovery.get_or_insert_with(RocmDiscovery::new);
    let runtime_realm = arena
        .runtime_realm
        .as_ref()
        .expect("cold preparation records its runtime realm")
        .clone();
    let mut rejected = Vec::new();
    for device in candidates(discovery, policy, kernel)? {
        let key = SessionKey {
            device,
            pci_bus_id: discovery
                .device_info(device)
                .ok()
                .and_then(|info| info.pci_bus_id.clone()),
            runtime_realm: runtime_realm.clone(),
        };
        let session = if let Some(existing) = arena.sessions.iter().find(|entry| entry.key == key) {
            Rc::clone(&existing.session)
        } else {
            match RocmOwnedDispatchBackend::open(discovery, device, policy.block_size) {
                Ok(backend) => {
                    let session = Rc::new(RocmSession::new(backend, policy.block_size));
                    retain_session(
                        &mut arena.sessions,
                        SessionEntry {
                            key,
                            session: Rc::clone(&session),
                        },
                        policy.cache_capacity,
                    );
                    session
                }
                Err(error) => {
                    rejected.push((device, PcuExecutionError::BackendInitialization(error)));
                    continue;
                }
            }
        };
        match prepare(&session) {
            Ok(prepared) => return Ok((session, prepared)),
            Err(error) => rejected.push((device, error)),
        }
    }
    Err(PcuExecutionError::NoCompatibleDevice {
        rejected,
        discovery: Vec::new(),
    })
}

/// Cold typed graph preparation shares the same selection and retained roots as Dispatch.
#[cfg(all(feature = "tensor", not(feature = "cuda")))]
pub(super) fn prepare_tensor<R>(
    snapshot: PolicySnapshot,
    affinity: Option<&Rc<RocmSession>>,
    prepare: impl FnMut(&Rc<RocmSession>) -> Result<R, PcuExecutionError>,
) -> Result<(Rc<RocmSession>, R, u64, usize), PcuExecutionError> {
    let policy = snapshot.policy;
    let generation = snapshot.generation;
    STATE
        .try_with(|state| {
            let mut state = state
                .try_borrow_mut()
                .map_err(|_| PcuExecutionError::ReentrantCall)?;
            if state.generation != generation {
                state.entries.clear();
                state.arena = SessionArena::default();
                state.generation = generation;
            }
            let runtime_realm = RuntimeRealm::current();
            if state.arena.runtime_realm.as_ref() != Some(&runtime_realm) {
                state.entries.clear();
                state.arena = SessionArena {
                    runtime_realm: Some(runtime_realm),
                    ..SessionArena::default()
                };
            }
            let (session, prepared) =
                prepare_in_arena(&mut state.arena, policy, affinity, None, prepare)?;
            Ok((session, prepared, generation, policy.cache_capacity))
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
            state.arena = SessionArena::default();
            Ok(())
        })
        .map_err(|_| PcuExecutionError::ThreadUnavailable)?
}

pub(super) fn call_host(
    site: &PcuHostCallSite,
    specialization: TypeId,
    arguments: &mut [PcuHostArgument<'_>],
    prepare: impl FnOnce(&mut PcuHostPreparation) -> Result<(), PcuExecutionError>,
) -> Result<(), PcuExecutionError> {
    with_entry(site, specialization, None, prepare, |entry| {
        entry
            .prepared
            .call(arguments)
            .map_err(PcuExecutionError::from)
    })
}

/// Convert one fixed source argument set without a heap-backed projection vector.
#[cfg(not(any(
    feature = "cuda",
    feature = "metal",
    feature = "vulkan",
    feature = "cpu",
    feature = "mlx"
)))]
pub(super) fn call_arguments<'a, const N: usize>(
    site: &PcuHostCallSite,
    specialization: TypeId,
    arguments: [super::PcuCallArgument<'a>; N],
    prepare: impl FnOnce(&mut PcuHostPreparation) -> Result<(), PcuExecutionError>,
) -> Result<(), PcuExecutionError> {
    #[rustfmt::skip]
    use super::arguments::{
        finish_resident_writes,
        PcuCallArgumentKind,
        ResidentWriteGuard,
    };
    use fusion_pcu_rocm::RocmMixedHostArgument;
    let mut affinity: Option<&'a Rc<RocmSession>> = None;
    for argument in &arguments {
        let root = match argument.kind() {
            PcuCallArgumentKind::Host(_) => continue,
            PcuCallArgumentKind::ResidentRead(resident) => resident.session,
            PcuCallArgumentKind::ResidentWrite(resident) => resident.session,
        };
        let super::resident::Session::Rocm(root) = root.as_ref();
        if affinity.is_some_and(|selected| !Rc::ptr_eq(selected, root)) {
            return Err(super::argument_error(
                super::PcuArgumentError::SessionMismatch,
            ));
        }
        affinity = Some(root);
    }
    let mut guards: [Option<ResidentWriteGuard<'a>>; N] = core::array::from_fn(|_| None);
    let mut index = 0;
    let mut bindings = arguments.map(|argument| {
        let (_, kind) = argument.into_parts();
        let binding = match kind {
            PcuCallArgumentKind::Host(host) => RocmMixedHostArgument::Host(host),
            PcuCallArgumentKind::ResidentRead(resident) => {
                let super::resident::DeviceArgument::Rocm(argument) = resident.argument;
                RocmMixedHostArgument::Resident(argument)
            }
            PcuCallArgumentKind::ResidentWrite(resident) => {
                guards[index] = Some(resident.guard);
                let super::resident::DeviceArgument::Rocm(argument) = resident.argument;
                RocmMixedHostArgument::Resident(argument)
            }
        };
        index += 1;
        binding
    });
    with_entry(site, specialization, affinity, prepare, |entry| {
        for guard in guards.iter_mut().flatten() {
            guard.mark_may_have_written();
        }
        let result = entry
            .prepared
            .call_mixed(&mut bindings)
            .map_err(PcuExecutionError::from);
        finish_resident_writes(
            &mut guards,
            entry.prepared.last_call_may_have_written(),
            entry.prepared.last_call_completion_uncertain(),
            &result,
        );
        result
    })
}

fn with_entry<R>(
    site: &PcuHostCallSite,
    specialization: TypeId,
    affinity: Option<&Rc<RocmSession>>,
    prepare: impl FnOnce(&mut PcuHostPreparation) -> Result<(), PcuExecutionError>,
    execute: impl FnOnce(&mut Entry) -> Result<R, PcuExecutionError>,
) -> Result<R, PcuExecutionError> {
    STATE
        .try_with(|state| {
            let mut state = state
                .try_borrow_mut()
                .map_err(|_| PcuExecutionError::ReentrantCall)?;
            let generation = policy::generation();
            if state.generation != generation {
                state.entries.clear();
                state.arena = SessionArena::default();
                state.generation = generation;
            }
            let hint = site.slot.load(Ordering::Relaxed);
            let cached = find_slot(&state.entries, hint, specialization, affinity);
            let slot = if let Some(slot) = cached {
                slot
            } else {
                // Only cold misses take a policy lock; the snapshot pins preparation preferences.
                let snapshot = policy::snapshot()?;
                let policy = snapshot.policy;
                let snapshot_generation = snapshot.generation;
                if state.generation != snapshot_generation {
                    state.entries.clear();
                    state.arena = SessionArena::default();
                    state.generation = snapshot_generation;
                }
                // Environment changes are observed on cold misses. A warm specialization remains
                // pinned to the runtime it was prepared against without checking process env.
                let runtime_realm = RuntimeRealm::current();
                if state.arena.runtime_realm.as_ref() != Some(&runtime_realm) {
                    state.entries.clear();
                    state.arena = SessionArena {
                        runtime_realm: Some(runtime_realm),
                        ..SessionArena::default()
                    };
                }
                let mut context = PcuHostPreparation {
                    #[cfg(any(
                        feature = "cuda",
                        feature = "metal",
                        feature = "vulkan",
                        feature = "cpu",
                        feature = "mlx"
                    ))]
                    shared: None,
                    inner: Some(Preparation {
                        policy,
                        prepared: None,
                        session: None,
                        affinity: affinity.map(Rc::clone),
                        arena: core::mem::take(&mut state.arena),
                    }),
                };
                let preparation_result = prepare(&mut context);
                let prepared = context
                    .inner
                    .as_mut()
                    .expect("ROCm preparation context")
                    .prepared
                    .take()
                    .ok_or(PcuExecutionError::PreparationDidNotProduceKernel);
                let inner = context.inner.as_mut().expect("ROCm preparation context");
                let session = inner.session.take();
                state.arena = core::mem::take(&mut inner.arena);
                drop(context);
                preparation_result?;
                let prepared = prepared?;
                let entry = Entry {
                    specialization,
                    prepared,
                    session: session.expect("successful preparation retains its execution domain"),
                    resident_affinity: affinity.is_some(),
                };
                if state.entries.len() == policy.cache_capacity {
                    // Synchronous host calls have no escaping executable/resource owner here.
                    // Replacing one slot bounds retention and invalidates other callsite hints safely.
                    let victim = hint.min(state.entries.len() - 1);
                    state.entries[victim] = entry;
                    victim
                } else {
                    state.entries.push(entry);
                    state.entries.len() - 1
                }
            };
            site.slot.store(slot, Ordering::Relaxed);
            execute(&mut state.entries[slot])
        })
        .map_err(|_| PcuExecutionError::ThreadUnavailable)?
}

fn find_slot(
    entries: &[Entry],
    hint: usize,
    specialization: TypeId,
    affinity: Option<&Rc<RocmSession>>,
) -> Option<usize> {
    let matches = |entry: &Entry| {
        entry.specialization == specialization
            && affinity.map_or(!entry.resident_affinity, |root| {
                entry.resident_affinity && Rc::ptr_eq(root, &entry.session)
            })
    };
    if entries.get(hint).is_some_and(matches) {
        Some(hint)
    } else {
        entries.iter().position(matches)
    }
}

fn retain_session(sessions: &mut Vec<SessionEntry>, session: SessionEntry, capacity: usize) {
    if capacity == 0 {
        return;
    }
    if sessions.len() == capacity {
        // This cache owns only backend/session roots. Prepared host entries retain the runtime
        // they need, and synchronous host calls expose no resource owner beyond this thread.
        sessions.remove(0);
    }
    sessions.push(session);
}

const EMPTY_REF: PcuObjectRef = PcuObjectRef {
    provider: PcuProviderId(0),
    generation: 0,
    kind: PcuObjectKind::Target,
    id: 0,
};
const EMPTY_READINESS: PcuProviderReadiness<'static> = PcuProviderReadiness {
    status: PcuProviderStatus::Unavailable,
    reason: None,
};

fn candidates(
    discovery: &RocmDiscovery,
    policy: PcuExecutionPolicy,
    kernel: Option<&PcuDispatchKernelIr<'_>>,
) -> Result<Vec<PcuObjectRef>, PcuExecutionError> {
    let mut providers = [PcuProviderDescriptor {
        id: PcuProviderId(0),
        generation: 0,
        backend: "",
        readiness: EMPTY_READINESS,
    }];
    discovery
        .providers(&mut providers)
        .map_err(PcuExecutionError::Discovery)?;
    let mut targets = [PcuTargetDescriptor {
        reference: EMPTY_REF,
        name: "",
        readiness: EMPTY_READINESS,
    }];
    discovery
        .targets(providers[0].id, providers[0].generation, &mut targets)
        .map_err(PcuExecutionError::Discovery)?;
    let count = discovery
        .devices(targets[0].reference, &mut [])
        .map_err(PcuExecutionError::Discovery)?;
    let empty = PcuDeviceDescriptor {
        reference: EMPTY_REF,
        target: EMPTY_REF,
        name: "",
        class: PcuDeviceClass::Other,
        vendor: None,
        architecture: None,
        generation: None,
        location: None,
    };
    let mut devices = vec![empty; count];
    discovery
        .devices(targets[0].reference, &mut devices)
        .map_err(PcuExecutionError::Discovery)?;
    devices.retain(|device| policy.device.is_none_or(|id| device.reference.id == id));
    // Score once per cold candidate, never inside sort or a cached warm invocation.
    // Concrete preparation still hard-filters capability and numerical admission.
    let mut scored = devices
        .into_iter()
        .map(|descriptor| {
            let memory = discovery
                .device_info(descriptor.reference)
                .ok()
                .map(|info| info.total_memory);
            let score =
                super::selection::score_candidate(policy, descriptor, memory, kernel, || {
                    discovery.device_facts(descriptor.reference)
                })
                .map_err(PcuExecutionError::Discovery)?;
            Ok((descriptor.reference, score))
        })
        .collect::<Result<Vec<_>, PcuExecutionError>>()?;
    scored.sort_by(|(left, left_score), (right, right_score)| {
        right_score
            .cmp(left_score)
            .then_with(|| left.id.cmp(&right.id))
    });
    Ok(scored.into_iter().map(|(device, _)| device).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unproved_portable_invocation_rejects_before_provider_discovery() {
        let options = crate::PcuNumericalOptions {
            reproducibility: crate::PcuReproducibility::PortableV1,
            ..crate::PcuNumericalOptions::default()
        };
        let mut preparation = Preparation {
            policy: PcuExecutionPolicy {
                numerical_options: options,
                ..PcuExecutionPolicy::default()
            },
            prepared: None,
            session: None,
            affinity: None,
            arena: SessionArena::default(),
        };
        let kernel = crate::PcuDispatchKernelIr {
            numerical_requirements: crate::PcuImplementationRequirements {
                numerical_options: options,
                ..crate::PcuDispatchKernelIr::DEFAULT_REQUIREMENTS
            },
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
        assert!(preparation.prepared.is_none());
        assert!(preparation.session.is_none());
        assert!(preparation.arena.sessions.is_empty());
    }

    #[test]
    fn invalid_hints_never_select_another_specialization() {
        // No hardware is needed to verify an empty/evicted callsite hint.
        assert_eq!(find_slot(&[], usize::MAX, TypeId::of::<u32>(), None), None);
        assert_eq!(find_slot(&[], 0, TypeId::of::<u64>(), None), None);
    }

    #[test]
    fn session_keys_bind_stable_device_discovery_and_runtime_realm() {
        let device = PcuObjectRef {
            provider: PcuProviderId(7),
            generation: 11,
            kind: PcuObjectKind::Device,
            id: 2,
        };
        let realm = RuntimeRealm {
            hip_library: Some("/opt/hip/libamdhip64.so".into()),
            hipcc: Some("/opt/hip/bin/hipcc".into()),
            path: Some("/opt/hip/bin".into()),
            hiprtc_library: Some("/opt/hip/libhiprtc.so".into()),
            #[cfg(feature = "tensor")]
            rocblas_library: Some("/opt/rocm/lib/librocblas.so".into()),
        };
        let key = SessionKey {
            device,
            pci_bus_id: Some("0000:03:00.0".into()),
            runtime_realm: realm.clone(),
        };
        assert_eq!(
            key,
            SessionKey {
                device,
                pci_bus_id: Some("0000:03:00.0".into()),
                runtime_realm: realm.clone(),
            }
        );
        assert_ne!(
            key,
            SessionKey {
                device: PcuObjectRef {
                    generation: 12,
                    ..device
                },
                pci_bus_id: Some("0000:03:00.0".into()),
                runtime_realm: realm.clone(),
            }
        );
        assert_ne!(
            key,
            SessionKey {
                device,
                pci_bus_id: Some("0000:04:00.0".into()),
                runtime_realm: realm.clone(),
            }
        );
        #[cfg(feature = "tensor")]
        assert_ne!(
            key,
            SessionKey {
                device,
                pci_bus_id: Some("0000:03:00.0".into()),
                runtime_realm: RuntimeRealm {
                    rocblas_library: Some("/different/librocblas.so".into()),
                    ..realm.clone()
                },
            }
        );
        assert_ne!(
            key,
            SessionKey {
                device,
                pci_bus_id: Some("0000:03:00.0".into()),
                runtime_realm: RuntimeRealm {
                    hip_library: Some("/different/hip/libamdhip64.so".into()),
                    ..realm
                },
            }
        );
    }

    #[test]
    fn thread_teardown_reports_unavailable_instead_of_panicking() {
        use std::sync::atomic::AtomicBool;
        static REPORTED: AtomicBool = AtomicBool::new(false);
        struct Callback;
        impl Drop for Callback {
            fn drop(&mut self) {
                REPORTED.store(
                    matches!(
                        clear_thread_cache(),
                        Err(PcuExecutionError::ThreadUnavailable)
                    ),
                    Ordering::Relaxed,
                );
            }
        }
        std::thread_local! { static BEFORE: Callback = const { Callback }; }
        std::thread::spawn(|| {
            BEFORE.with(|_| {});
            clear_thread_cache().unwrap();
        })
        .join()
        .unwrap();
        assert!(REPORTED.load(Ordering::Relaxed));
    }

    #[test]
    fn nested_cache_access_returns_error_instead_of_panicking() {
        STATE.with(|state| {
            let _active = state.borrow_mut();
            assert!(matches!(
                clear_thread_cache(),
                Err(PcuExecutionError::ReentrantCall)
            ));
        });
    }
}
