//! Retained private resource shadows; only a complete validated call publishes RAM.
use std::rc::Rc;
use fusion_pcu::PcuHostArgument;
use fusion_pcu_spirv::PcuSpirvComposedFloatProfile;
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
    PcuVulkanError,
    PcuVulkanMemoryRealization,
};
#[path = "status/status.rs"]
mod status;

/// Actual private resource allocations followed by the ordered status allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PcuVulkanComposedMemoryRealizations {
    One([PcuVulkanMemoryRealization; 2]),
    Two([PcuVulkanMemoryRealization; 3]),
    Three([PcuVulkanMemoryRealization; 4]),
    Four([PcuVulkanMemoryRealization; 5]),
}

pub struct VulkanPreparedComposed(Owners);

enum Owners {
    One(VulkanPreparedMap<2, true, 1>),
    Two(VulkanPreparedMap<3, true, 2>),
    Three(VulkanPreparedMap<4, true, 3>),
    Four(VulkanPreparedMap<5, true, 4>),
}

impl VulkanPreparedComposed {
    pub(crate) fn new(
        device: Rc<VulkanDevice>,
        words: &[u32],
        profile: &PcuSpirvComposedFloatProfile,
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
        profile: &PcuSpirvComposedFloatProfile,
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
    profile: &PcuSpirvComposedFloatProfile,
) -> Result<VulkanPreparedMap<N, true, OUTPUTS>, PcuVulkanError> {
    let count = profile.resources().len();
    let mut lengths = [0; N];
    for (slot, resource) in profile.resources().iter().enumerate() {
        lengths[slot] = (resource.read_elements.max(resource.write_elements) as usize)
            .checked_mul(profile.element_bytes())
            .ok_or(PcuVulkanError::BufferTooLarge)?;
    }
    lengths[N - 1] = (profile.extent() as usize)
        .checked_mul(8)
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
    // Unused banks have no native loads or stores; their descriptors alias bank zero.
    // Actual resources remain unique and the status descriptor always owns separate memory.
    VulkanPreparedMap::new_with_descriptors(
        device,
        words,
        profile.extent(),
        profile.dispatch_extent(),
        [64, 1, 1],
        lengths,
        StatusPolicy::ZERO_ONLY,
        descriptors,
    )
}

fn execute<const N: usize, const OUTPUTS: usize>(
    map: &mut VulkanPreparedMap<N, true, OUTPUTS>,
    profile: &PcuSpirvComposedFloatProfile,
    arguments: &mut [PcuHostArgument<'_>],
    positions: [usize; 4],
    mut measurements: Option<&mut PcuVulkanCallMeasurements>,
) -> Result<(), PcuVulkanError> {
    if map.device.poisoned.get() {
        return Err(PcuVulkanError::Quarantined);
    }
    let resources = map.resources.as_ref().ok_or(PcuVulkanError::Quarantined)?;
    let started = measurements.as_ref().map(|_| std::time::Instant::now());
    for (slot, resource) in profile.resources().iter().enumerate() {
        let bytes = resource.read_elements as usize * profile.element_bytes();
        if bytes != 0 {
            unsafe {
                // SAFETY: Typed preflight proves the complete readable prefix and retained
                // mapping capacity. Earlier calls are terminal; one unique resource is seeded once.
                ptr::copy_nonoverlapping(
                    arguments[positions[slot]].bytes().as_ptr(),
                    resources.buffers[slot].mapped,
                    bytes,
                );
            }
        }
    }
    let uploaded = started.map(|_| std::time::Instant::now());
    vk_try("reset retained Vulkan composed fence", unsafe {
        // SAFETY: All prior calls completed or poisoned the entire retained device.
        map.device.device.reset_fences(&[resources.fence])
    })?;
    if let Err(error) = submit_and_wait_measured(
        &map.device.device,
        map.device.queue,
        resources.command,
        resources.fence,
        measurements.as_deref_mut(),
    ) {
        if matches!(error, PcuVulkanError::CompletionUnknown) {
            map.device.poisoned.set(true);
            map.resources.take(); // Raw native handles remain mapped and alive.
            mem::forget(Rc::clone(&map.device)); // Retain loader, queue and session roots too.
        }
        return Err(error);
    }
    let completed = started.map(|_| std::time::Instant::now());
    let notice = status::scan(profile, |lane| unsafe {
        // SAFETY: Every logical lane writes both U32 fields, followed by a compute-to-host
        // dependency and known terminal fence. The cold status allocation spans extent * 8.
        let record = resources.buffers[N - 1]
            .mapped
            .add(lane * 8)
            .cast::<[u8; 8]>()
            .read_unaligned();
        (
            u32::from_ne_bytes([record[0], record[1], record[2], record[3]]),
            u32::from_ne_bytes([record[4], record[5], record[6], record[7]]),
        )
    })?;
    for (slot, resource) in profile.resources().iter().enumerate() {
        let bytes = resource.write_elements as usize * profile.element_bytes();
        if bytes != 0 {
            let output = arguments[positions[slot]]
                .bytes_mut()
                .expect("complete typed preflight proved every writable resource");
            unsafe {
                // SAFETY: All records are valid and no fatal occurred. Typed preflight proves
                // every exclusive destination, including disjoint active prefixes. Publish the
                // exact logical bytes, preserving odd packed tails and every unwritten resource.
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
    notice.map_or(Ok(()), |fault| Err(PcuVulkanError::Fault(fault)))
}
