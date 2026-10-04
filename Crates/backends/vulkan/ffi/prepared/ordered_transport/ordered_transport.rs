//! Private native raw-carrier shadows. Complete zero-only status scan precedes all RAM publication.
use std::rc::Rc;
use fusion_pcu::PcuHostArgument;
use fusion_pcu_spirv::PcuSpirvOrderedTransportProfile as Profile;
#[rustfmt::skip]
use super::{
    mem,
    ptr,
    submit_and_wait_measured,
    vk_try,
    StatusPolicy,
    VulkanDevice,
    VulkanPreparedMap,
    PcuVulkanCallMeasurements,
    PcuVulkanComposedMemoryRealizations,
    PcuVulkanError,
};
pub struct VulkanPreparedOrderedTransport(Owners);
enum Owners {
    One(VulkanPreparedMap<2, true, 1>),
    Two(VulkanPreparedMap<3, true, 2>),
    Three(VulkanPreparedMap<4, true, 3>),
    Four(VulkanPreparedMap<5, true, 4>),
}
impl VulkanPreparedOrderedTransport {
    pub(crate) fn new(
        device: Rc<VulkanDevice>,
        words: &[u32],
        profile: &Profile,
    ) -> Result<Self, PcuVulkanError> {
        let owners = match profile.resources().len() {
            1 => build(device, words, profile).map(Owners::One),
            2 => build(device, words, profile).map(Owners::Two),
            3 => build(device, words, profile).map(Owners::Three),
            4 => build(device, words, profile).map(Owners::Four),
            _ => Err(PcuVulkanError::UnsupportedPreparedProfile),
        }?;
        Ok(Self(owners))
    }
    pub(crate) fn memory_realizations(&self) -> Option<PcuVulkanComposedMemoryRealizations> {
        match &self.0 {
            Owners::One(map) => map
                .memory_realizations()
                .map(PcuVulkanComposedMemoryRealizations::One),
            Owners::Two(map) => map
                .memory_realizations()
                .map(PcuVulkanComposedMemoryRealizations::Two),
            Owners::Three(map) => map
                .memory_realizations()
                .map(PcuVulkanComposedMemoryRealizations::Three),
            Owners::Four(map) => map
                .memory_realizations()
                .map(PcuVulkanComposedMemoryRealizations::Four),
        }
    }
    pub(crate) fn call(
        &mut self,
        profile: &Profile,
        arguments: &mut [PcuHostArgument<'_>],
        positions: [usize; 4],
        measurements: Option<&mut PcuVulkanCallMeasurements>,
    ) -> Result<(), PcuVulkanError> {
        match &mut self.0 {
            Owners::One(map) => execute(map, profile, arguments, positions, measurements),
            Owners::Two(map) => execute(map, profile, arguments, positions, measurements),
            Owners::Three(map) => execute(map, profile, arguments, positions, measurements),
            Owners::Four(map) => execute(map, profile, arguments, positions, measurements),
        }
    }
}
fn build<const N: usize, const OUTPUTS: usize>(
    device: Rc<VulkanDevice>,
    words: &[u32],
    profile: &Profile,
) -> Result<VulkanPreparedMap<N, true, OUTPUTS>, PcuVulkanError> {
    let count = profile.resources().len();
    let mut lengths = [0; N];
    for (slot, resource) in profile.resources().iter().enumerate() {
        lengths[slot] = usize::try_from(resource.read_elements.max(resource.write_elements))
            .ok()
            .and_then(|n| n.checked_mul(profile.element_bytes()))
            .ok_or(PcuVulkanError::BufferTooLarge)?;
    }
    lengths[N - 1] = usize::try_from(profile.dispatch_extent())
        .ok()
        .and_then(|n| n.checked_mul(4))
        .ok_or(PcuVulkanError::BufferTooLarge)?;
    let descriptors = core::array::from_fn::<_, 5, _>(|slot| {
        if slot == 4 {
            count
        } else if slot < count {
            slot
        } else {
            0
        }
    });
    VulkanPreparedMap::new_with_descriptors(
        device,
        words,
        profile.dispatch_extent(),
        profile.dispatch_extent(),
        [64, 1, 1],
        lengths,
        StatusPolicy::ZERO_ONLY,
        descriptors,
    )
}
fn execute<const N: usize, const OUTPUTS: usize>(
    map: &mut VulkanPreparedMap<N, true, OUTPUTS>,
    profile: &Profile,
    arguments: &mut [PcuHostArgument<'_>],
    positions: [usize; 4],
    mut measurements: Option<&mut PcuVulkanCallMeasurements>,
) -> Result<(), PcuVulkanError> {
    if map.device.poisoned.get() {
        return Err(PcuVulkanError::Quarantined);
    }
    let resources = map.resources.as_ref().ok_or(PcuVulkanError::Quarantined)?;
    let reads = prefix_lengths(profile, false)?;
    let writes = prefix_lengths(profile, true)?;
    let started = measurements.as_ref().map(|_| std::time::Instant::now());
    for (slot, &bytes) in reads[..profile.resources().len()].iter().enumerate() {
        if bytes != 0 {
            unsafe {
                // SAFETY: Complete typed preflight proved every readable prefix and prior calls are
                // terminal. Native mapped capacity covers this conservative read-view copy. Caller RAM
                // is never submitted to a GPU; all subsequent work owns these retained private bytes.
                ptr::copy_nonoverlapping(
                    arguments[positions[slot]].bytes().as_ptr(),
                    resources.buffers[slot].mapped,
                    bytes,
                );
            }
        }
    }
    let uploaded = started.map(|_| std::time::Instant::now());
    vk_try("reset retained Vulkan transport fence", unsafe {
        // SAFETY: Every previous call completed or poisoned the actual retained device.
        vk_api_owner!(
            map.device,
            ResetFences,
            map.device.device.reset_fences(&[resources.fence])
        )
    })?;
    if let Err(error) = submit_and_wait_measured(
        &map.device,
        resources.command,
        resources.fence,
        measurements.as_deref_mut(),
    ) {
        if matches!(error, PcuVulkanError::CompletionUnknown) {
            map.device.poisoned.set(true);
            map.resources.take();
            mem::forget(Rc::clone(&map.device));
        }
        return Err(error);
    }
    let completed = started.map(|_| std::time::Instant::now());
    map.status_policy.scan(map.extent, |word| {
        Ok(unsafe {
            // SAFETY: Every complete output-word owner initialized one zero-only protocol record.
            // Compute-to-host dependency and terminal fence precede this retained mapped read.
            resources.buffers[N - 1]
                .mapped
                .add(word * 4)
                .cast::<u32>()
                .read_unaligned()
        })
    })?;
    for (slot, &bytes) in writes[..profile.resources().len()].iter().enumerate() {
        if bytes != 0 {
            let output = arguments[positions[slot]]
                .bytes_mut()
                .expect("complete typed preflight proved every writable transport resource");
            unsafe {
                // SAFETY: All protocol records and all exclusive caller extents were checked before
                // publication. Every transfer is terminal, these copies cannot fail, and each exact
                // logical prefix preserves odd packed tails. No fallible sibling SDK readback follows.
                ptr::copy_nonoverlapping(
                    resources.buffers[slot].mapped,
                    output.as_mut_ptr(),
                    bytes,
                );
            }
        }
    }
    if let (Some(measurements), Some(started), Some(uploaded), Some(completed)) =
        (measurements, started, uploaded, completed)
    {
        let end = std::time::Instant::now();
        measurements.upload = uploaded - started;
        measurements.submission = (completed - uploaded).saturating_sub(measurements.completion);
        measurements.diagnostic_publication = end - completed;
        measurements.wall = end - started;
    }
    Ok(())
}

fn prefix_lengths(profile: &Profile, write: bool) -> Result<[usize; 4], PcuVulkanError> {
    let mut lengths = [0; 4];
    for (slot, resource) in profile.resources().iter().enumerate() {
        let elements = if write {
            resource.write_elements
        } else {
            resource.read_elements
        };
        lengths[slot] = usize::try_from(elements)
            .ok()
            .and_then(|n| n.checked_mul(profile.element_bytes()))
            .ok_or(PcuVulkanError::BufferTooLarge)?;
    }
    Ok(lengths)
}
