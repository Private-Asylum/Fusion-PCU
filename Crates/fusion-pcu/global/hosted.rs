//! Thread-owned hosted state. Cold paths own discovery, ranking and compilation.

use core::any::TypeId;
use std::ffi::OsString;
#[rustfmt::skip]
use core::sync::atomic::{
    AtomicU64,
    Ordering,
};
use std::cell::RefCell;
use std::rc::Rc;
use super::session::RocmSession;
use std::sync::RwLock;

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

struct PolicySnapshot {
    generation: u64,
    policy: PcuExecutionPolicy,
}
static GENERATION: AtomicU64 = AtomicU64::new(1);
static POLICY: RwLock<PolicySnapshot> = RwLock::new(PolicySnapshot {
    generation: 1,
    policy: PcuExecutionPolicy {
        backend: super::PcuBackendChoice::Automatic,
        device: None,
        cache_capacity: 64,
        block_size: 256,
        score_device: super::default_device_score,
    },
});

pub(super) fn configure(policy: PcuExecutionPolicy) -> Result<(), PcuExecutionError> {
    let mut state = POLICY
        .write()
        .map_err(|_| PcuExecutionError::PolicyUnavailable)?;
    state.generation = state
        .generation
        .checked_add(1)
        .ok_or(PcuExecutionError::PolicyUnavailable)?;
    state.policy = policy;
    GENERATION.store(state.generation, Ordering::Release);
    drop(state);
    Ok(())
}

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
    pub(super) fn prepare(
        &mut self,
        kernel: &PcuDispatchKernelIr<'_>,
    ) -> Result<(), PcuExecutionError> {
        let (session, prepared) = prepare_in_arena(
            &mut self.arena,
            self.policy,
            self.affinity.as_ref(),
            |session| {
                session
                    .backend()
                    .prepare_host_kernel(kernel)
                    .map_err(PcuExecutionError::Execution)
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
    for device in candidates(discovery, policy)? {
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
                    rejected.push((device.id, PcuExecutionError::BackendInitialization(error)));
                    continue;
                }
            }
        };
        match prepare(&session) {
            Ok(prepared) => return Ok((session, prepared)),
            Err(error) => rejected.push((device.id, error)),
        }
    }
    Err(PcuExecutionError::NoCompatibleDevice(rejected))
}

#[cfg(feature = "tensor")]
pub(super) fn current_generation() -> u64 {
    GENERATION.load(Ordering::Acquire)
}

/// Cold typed graph preparation shares the same selection and retained roots as Dispatch.
#[cfg(feature = "tensor")]
pub(super) fn prepare_tensor<R>(
    affinity: Option<&Rc<RocmSession>>,
    prepare: impl FnMut(&Rc<RocmSession>) -> Result<R, PcuExecutionError>,
) -> Result<(Rc<RocmSession>, R, u64, usize), PcuExecutionError> {
    let snapshot = POLICY
        .read()
        .map_err(|_| PcuExecutionError::PolicyUnavailable)?;
    let policy = snapshot.policy;
    let generation = snapshot.generation;
    drop(snapshot);
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
                prepare_in_arena(&mut state.arena, policy, affinity, prepare)?;
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
            .map_err(PcuExecutionError::Execution)
    })
}

/// Convert one fixed source argument set without a heap-backed projection vector.
pub(super) fn call_arguments<'a, const N: usize>(
    site: &PcuHostCallSite,
    specialization: TypeId,
    arguments: [super::PcuCallArgument<'a>; N],
    prepare: impl FnOnce(&mut PcuHostPreparation) -> Result<(), PcuExecutionError>,
) -> Result<(), PcuExecutionError> {
    #[rustfmt::skip]
    use super::arguments::{
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
                RocmMixedHostArgument::Resident(resident.argument)
            }
            PcuCallArgumentKind::ResidentWrite(resident) => {
                guards[index] = Some(resident.guard);
                RocmMixedHostArgument::Resident(resident.argument)
            }
        };
        index += 1;
        binding
    });
    with_entry(site, specialization, affinity, prepare, |entry| {
        for guard in guards.iter_mut().flatten() {
            guard.mark_may_have_written();
        }
        let result = entry.prepared.call_mixed(&mut bindings);
        if result.is_ok() {
            for guard in guards.iter_mut().flatten() {
                guard.mark_complete();
            }
        } else if !entry.prepared.last_call_completion_uncertain() {
            for guard in guards.iter_mut().flatten() {
                guard.mark_known_partial();
            }
        }
        result.map_err(PcuExecutionError::Execution)
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
            let generation = GENERATION.load(Ordering::Acquire);
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
                let snapshot = POLICY
                    .read()
                    .map_err(|_| PcuExecutionError::PolicyUnavailable)?;
                let policy = snapshot.policy;
                let snapshot_generation = snapshot.generation;
                drop(snapshot);
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
                    inner: Preparation {
                        policy,
                        prepared: None,
                        session: None,
                        affinity: affinity.map(Rc::clone),
                        arena: core::mem::take(&mut state.arena),
                    },
                };
                let preparation_result = prepare(&mut context);
                let prepared = context
                    .inner
                    .prepared
                    .take()
                    .ok_or(PcuExecutionError::PreparationDidNotProduceKernel);
                let session = context.inner.session.take();
                state.arena = core::mem::take(&mut context.inner.arena);
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
    // Concrete kernel preparation hard-filters each candidate before it can execute.
    devices.sort_by(|a, b| {
        let memory = |device: &PcuDeviceDescriptor<'_>| {
            discovery
                .device_info(device.reference)
                .map_or(0, |info| info.total_memory)
        };
        (policy.score_device)(b, memory(b))
            .cmp(&(policy.score_device)(a, memory(a)))
            .then_with(|| a.reference.id.cmp(&b.reference.id))
    });
    Ok(devices.into_iter().map(|device| device.reference).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

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
