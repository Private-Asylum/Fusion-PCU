//! Retained private host readbacks; public output copies occur only after every transfer succeeds.
#[rustfmt::skip]
use super::{
    HostBindingSlot,
    HostKernelCallArgument,
    PcuBindingAccess,
    PcuOwnedBindingRequirement,
};

pub(super) fn clear(slots: &mut [HostBindingSlot]) {
    for slot in slots {
        slot.readback = None;
    }
}

pub(super) fn stage_and_publish<A: HostKernelCallArgument, E>(
    slots: &mut [HostBindingSlot],
    requirements: &[PcuOwnedBindingRequirement],
    arguments: &mut [A],
    mut transfer: impl FnMut(usize, &mut HostBindingSlot, usize) -> Result<(), E>,
) -> Result<(), E> {
    for (index, (slot, requirement)) in slots.iter_mut().zip(requirements).enumerate() {
        let argument = arguments
            .iter()
            .find(|arg| arg.target() == requirement.target)
            .expect("coverage validated before submission");
        if argument.access() == PcuBindingAccess::ReadOnly {
            continue;
        }
        let Some(bytes) = argument.host_bytes() else {
            continue;
        };
        transfer(
            index,
            slot,
            slot.fully_written_prefix.unwrap_or(bytes.len()),
        )?;
    }
    // Every device read has succeeded. These validated same-length copies cannot fail and
    // execute no SDK calls. Resident destinations keep their independent discard contract.
    for (slot, requirement) in slots.iter().zip(requirements) {
        let argument = arguments
            .iter_mut()
            .find(|arg| arg.target() == requirement.target)
            .expect("coverage validated before submission");
        let Some(bytes) = argument.host_bytes_mut() else {
            continue;
        };
        let span = slot.fully_written_prefix.unwrap_or(bytes.len());
        slot.readback
            .as_ref()
            .expect("all private reads completed")
            .publish_to(&mut bytes[..span]);
    }
    clear(slots);
    Ok(())
}

/// Returns whether completion is unknown. Availability is inspected after the synchronous
/// transfer has returned and its completion guard has either dropped or been quarantined.
pub(super) fn retire_after_transfer_failure(slot: &mut HostBindingSlot) -> bool {
    let uncertain = slot.resource.as_ref().is_some_and(|resource| {
        resource
            .device_buffer()
            .validate_access_available()
            .is_err()
    });
    // Unknown synchronous completion already forgot the native lease, retaining the allocation
    // and its owned host Vec together. No separate RAM destination or caller reference exists.
    slot.readback = None;
    slot.resource = None;
    uncertain
}

#[cfg(all(test, feature = "tensor"))]
#[path = "native/native.rs"]
mod native;
