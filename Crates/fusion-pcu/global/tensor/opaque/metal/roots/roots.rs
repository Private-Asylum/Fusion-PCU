//! Cold weak roots preserve a discovered Metal session across escaped shapes and calls.
#[rustfmt::skip]
use std::{
    cell::RefCell,
    rc::{
        Rc,
        Weak,
    },
};
#[rustfmt::skip]
use crate::{
    PcuRuntimeDiscovery,
    PcuObjectRef,
    PcuObjectKind,
    PcuProviderId,
    PcuProviderDescriptor,
    PcuProviderReadiness,
    PcuProviderStatus,
    PcuTargetDescriptor,
    PcuDeviceDescriptor,
    PcuDeviceClass,
};
#[rustfmt::skip]
use crate::global::{
    PcuExecutionError,
    resident::Session,
};
use fusion_pcu_metal::MetalDiscovery;

std::thread_local! {
    static ROOTS: RefCell<Vec<(u32, Weak<Session>)>> = const { RefCell::new(Vec::new()) };
}

pub(super) fn get(ordinal: u32, block_size: u32) -> Result<Rc<Session>, PcuExecutionError> {
    ROOTS
        .try_with(|roots| {
            let mut roots = roots
                .try_borrow_mut()
                .map_err(|_| PcuExecutionError::ReentrantCall)?;
            roots.retain(|(_, root)| root.strong_count() != 0);
            if let Some(root) = roots
                .iter()
                .filter(|(device, _)| *device == ordinal)
                .filter_map(|(_, root)| root.upgrade())
                .find(|root| root.block_size() == block_size)
            {
                return Ok(root);
            }
            // Geometry hints belong to facade preparation, not native allocation affinity.
            // Preserve the actual device session when only that metadata changes.
            let retained_backend = roots
                .iter()
                .filter(|(device, _)| *device == ordinal)
                .filter_map(|(_, root)| root.upgrade())
                .find_map(|root| root.metal_backend().cloned());
            let backend = if let Some(backend) = retained_backend {
                backend
            } else {
                open(ordinal)?
            };
            let root = Rc::new(Session::from_metal_backend(backend, block_size));
            roots.push((ordinal, Rc::downgrade(&root)));
            Ok(root)
        })
        .map_err(|_| PcuExecutionError::ThreadUnavailable)?
}

fn open(ordinal: u32) -> Result<fusion_pcu_metal::MetalOwnedDispatchBackend, PcuExecutionError> {
    let discovery = MetalDiscovery::discover().map_err(super::map_error)?;
    let empty = PcuObjectRef {
        provider: PcuProviderId(0),
        generation: 0,
        kind: PcuObjectKind::Target,
        id: 0,
    };
    let ready = PcuProviderReadiness {
        status: PcuProviderStatus::Unavailable,
        reason: None,
    };
    let mut providers = [PcuProviderDescriptor {
        id: PcuProviderId(0),
        generation: 0,
        backend: "",
        readiness: ready,
    }];
    discovery
        .providers(&mut providers)
        .map_err(super::map_error)?;
    let mut targets = [PcuTargetDescriptor {
        reference: empty,
        name: "",
        readiness: ready,
    }];
    discovery
        .targets(providers[0].id, providers[0].generation, &mut targets)
        .map_err(super::map_error)?;
    let count = discovery
        .devices(targets[0].reference, &mut [])
        .map_err(super::map_error)?;
    let mut devices = vec![
        PcuDeviceDescriptor {
            reference: empty,
            target: empty,
            name: "",
            class: PcuDeviceClass::Other,
            vendor: None,
            architecture: None,
            generation: None,
            location: None
        };
        count
    ];
    discovery
        .devices(targets[0].reference, &mut devices)
        .map_err(super::map_error)?;
    let device = devices
        .iter()
        .find(|device| device.reference.id == ordinal)
        .ok_or(PcuExecutionError::ResidentPolicyConflict)?;
    discovery
        .open_owned_device(device.reference)
        .map_err(super::map_error)
}
