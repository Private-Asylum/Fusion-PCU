//! Native physical inventory without logical device, queue or resource activation.

#[rustfmt::skip]
use fusion_pcu::{
    PcuDeviceClass,
    PcuDeviceFacts,
    PcuStableDeviceIdentity,
};
#[rustfmt::skip]
use super::{
    choose_instance_api_version,
    create_instance,
    find_compute_queue_family,
    physical_device_name,
    query_backend_caps,
    vk_try,
    vk,
    PcuVulkanCaps,
    PcuVulkanDescriptorHeapBudget,
    PcuVulkanError,
    SelectedPhysicalDevice,
    VulkanProbeInstance,
};

/// Raw inventory facts have no resource interoperability or execution admission meaning.
pub struct VulkanNativeDevice {
    pub ordinal: u32,
    pub name: String,
    pub class: PcuDeviceClass,
    pub vendor: Option<&'static str>,
    pub facts: PcuDeviceFacts,
    pub caps: PcuVulkanCaps,
    vendor_id: u32,
    device_id: u32,
}

impl VulkanNativeDevice {
    pub fn discover() -> Result<Vec<Self>, PcuVulkanError> {
        let entry = unsafe {
            // SAFETY: ash resolves process-local Vulkan loader entry points.
            ash::Entry::load()
        }
        .map_err(PcuVulkanError::Loader)?;
        let api = choose_instance_api_version(&entry)?;
        let instance = VulkanProbeInstance {
            handle: create_instance(&entry, api)?,
        };
        let devices = vk_try("enumerate Vulkan discovery devices", unsafe {
            // SAFETY: The probe instance remains live through all physical queries.
            instance.handle.enumerate_physical_devices()
        })?;
        let mut inventory = Vec::new();
        for (ordinal, physical) in devices.into_iter().enumerate() {
            if find_compute_queue_family(&instance.handle, physical).is_none() {
                continue;
            }
            let device = describe(
                &instance.handle,
                physical,
                u32::try_from(ordinal).map_err(|_| PcuVulkanError::BufferTooLarge)?,
                api,
            )?;
            if device.class != PcuDeviceClass::Cpu {
                inventory.push(device);
            }
        }
        Ok(inventory)
    }
}

fn describe(
    instance: &ash::Instance,
    physical: vk::PhysicalDevice,
    ordinal: u32,
    api: u32,
) -> Result<VulkanNativeDevice, PcuVulkanError> {
    let properties = unsafe {
        // SAFETY: physical belongs to this live probe instance.
        instance.get_physical_device_properties(physical)
    };
    let caps = query_backend_caps(
        instance,
        physical,
        &properties,
        api,
        PcuVulkanDescriptorHeapBudget::empty(),
    )?;
    let mut facts = PcuDeviceFacts {
        max_workgroup_invocations: Some(properties.limits.max_compute_work_group_invocations),
        max_workgroup_dimensions: Some(properties.limits.max_compute_work_group_size),
        max_workgroup_count: Some(properties.limits.max_compute_work_group_count),
        local_memory_bytes_per_workgroup: Some(u64::from(
            properties.limits.max_compute_shared_memory_size,
        )),
        ..PcuDeviceFacts::default()
    };
    if caps.api_version >= vk::API_VERSION_1_1 {
        let mut id = vk::PhysicalDeviceIDProperties::default();
        let mut subgroup = vk::PhysicalDeviceSubgroupProperties::default();
        let mut extended = vk::PhysicalDeviceProperties2::default()
            .push_next(&mut id)
            .push_next(&mut subgroup);
        unsafe {
            // SAFETY: Vulkan 1.1 core properties use valid output extension structs.
            instance.get_physical_device_properties2(physical, &mut extended);
        }
        facts.subgroup_width = Some(subgroup.subgroup_size);
        if id.device_uuid != [0; vk::UUID_SIZE] {
            facts.stable_identity = Some(
                PcuStableDeviceIdentity::new("vulkan.deviceUUID", &id.device_uuid)
                    .map_err(|_| PcuVulkanError::InvalidDiscoveryReference)?,
            );
        }
    }
    Ok(VulkanNativeDevice {
        ordinal,
        name: physical_device_name(&properties),
        class: match properties.device_type {
            vk::PhysicalDeviceType::INTEGRATED_GPU | vk::PhysicalDeviceType::DISCRETE_GPU => {
                PcuDeviceClass::Gpu
            }
            vk::PhysicalDeviceType::CPU => PcuDeviceClass::Cpu,
            _ => PcuDeviceClass::Other,
        },
        vendor: match properties.vendor_id {
            0x1002 => Some("AMD"),
            0x10de => Some("NVIDIA"),
            0x8086 => Some("Intel"),
            0x13b5 => Some("Arm"),
            _ => None,
        },
        facts,
        caps,
        vendor_id: properties.vendor_id,
        device_id: properties.device_id,
    })
}

pub(super) fn select_native_device(
    instance: &ash::Instance,
    expected: &VulkanNativeDevice,
    api: u32,
) -> Result<SelectedPhysicalDevice, PcuVulkanError> {
    let devices = vk_try("enumerate Vulkan selected device", unsafe {
        // SAFETY: Native activation owns this live instance.
        instance.enumerate_physical_devices()
    })?;
    let physical_device = *devices
        .get(expected.ordinal as usize)
        .ok_or(PcuVulkanError::InvalidDiscoveryReference)?;
    let actual = describe(instance, physical_device, expected.ordinal, api)?;
    // An absent native stable identifier cannot prove a reopened physical inventory.
    if expected.facts.stable_identity.is_none()
        || actual.facts.stable_identity != expected.facts.stable_identity
        || actual.vendor_id != expected.vendor_id
        || actual.device_id != expected.device_id
    {
        return Err(PcuVulkanError::InvalidDiscoveryReference);
    }
    let queue_family_index = find_compute_queue_family(instance, physical_device)
        .ok_or(PcuVulkanError::NoComputeQueueFamily)?;
    let properties = unsafe {
        // SAFETY: The selected physical handle was returned by this instance.
        instance.get_physical_device_properties(physical_device)
    };
    Ok(SelectedPhysicalDevice {
        physical_device,
        queue_family_index,
        properties,
    })
}
