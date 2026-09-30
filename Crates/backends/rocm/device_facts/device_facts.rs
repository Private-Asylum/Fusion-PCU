//! Optional cold hardware facts, separate from executable admission and runtime initialization.

#[rustfmt::skip]
use crate::{
    HipDeviceInfo,
    HipError,
    ffi,
    ffi::hip::{
        ATTRIBUTE_ASYNC_ENGINE_COUNT,
        ATTRIBUTE_MAX_BLOCK_DIMENSIONS,
        ATTRIBUTE_MAX_GRID_DIMENSIONS,
        ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK,
        ATTRIBUTE_MAX_THREADS_PER_BLOCK,
        ATTRIBUTE_MULTIPROCESSOR_COUNT,
        ATTRIBUTE_WARP_SIZE,
        HIP_SUCCESS,
    },
    query_pci_bus_id,
};
#[rustfmt::skip]
use std::{
    ffi::c_int,
    ptr,
};
use fusion_pcu::PcuDeviceFacts;

pub fn query(info: &HipDeviceInfo) -> Result<PcuDeviceFacts, HipError> {
    let expected = info
        .pci_bus_id
        .as_deref()
        .ok_or(HipError::MissingStableDeviceIdentity)?;
    let candidates = std::env::var_os("HIP_RUNTIME_LIBRARY").map_or_else(
        || vec!["libamdhip64.so".into(), "libamdhip64.so.6".into()],
        |path| vec![path],
    );
    let mut last_error = None;
    for candidate in candidates {
        let library = match ffi::load_library(&candidate) {
            Ok(library) => library,
            Err(error) => {
                last_error = Some(error);
                continue;
            }
        };
        let actual =
            query_pci_bus_id(&library, info.index).ok_or(HipError::MissingStableDeviceIdentity)?;
        if !expected.eq_ignore_ascii_case(&actual) {
            return Err(HipError::DeviceIdentityChanged {
                expected: expected.to_owned(),
                actual,
            });
        }
        let Some(function) = ffi::device_get_attribute(&library) else {
            return Ok(PcuDeviceFacts::default());
        };
        return Ok(map_attributes(|attribute| {
            let mut value = 0;
            // SAFETY: HIP writes one int to a valid output pointer. The retained library backs
            // the documented function ABI, and discovery validated this device before querying.
            let status = unsafe {
                ffi::raw_hipDeviceGetAttribute(
                    &function,
                    ptr::from_mut(&mut value),
                    attribute,
                    info.index,
                )
            };
            (status == HIP_SUCCESS).then_some(value)
        }));
    }
    Err(HipError::RuntimeUnavailable(
        last_error.unwrap_or_else(|| "no runtime library candidates".into()),
    ))
}

fn positive(value: Option<c_int>) -> Option<u32> {
    value
        .and_then(|value| u32::try_from(value).ok())
        .filter(|value| *value > 0)
}

fn dimensions(
    attributes: [c_int; 3],
    query: &mut impl FnMut(c_int) -> Option<c_int>,
) -> Option<[u32; 3]> {
    let [x, y, z] = attributes.map(|attribute| positive(query(attribute)));
    Some([x?, y?, z?])
}

fn map_attributes(mut query: impl FnMut(c_int) -> Option<c_int>) -> PcuDeviceFacts {
    PcuDeviceFacts {
        // HIP's UUID API is Beta. PCI location and marketing name are not stable UUIDs.
        stable_identity: None,
        // Native multiprocessor units: HIP may report WGPs rather than physical AMD CUs.
        compute_unit_count: positive(query(ATTRIBUTE_MULTIPROCESSOR_COUNT)),
        subgroup_width: positive(query(ATTRIBUTE_WARP_SIZE)),
        max_workgroup_invocations: positive(query(ATTRIBUTE_MAX_THREADS_PER_BLOCK)),
        max_workgroup_dimensions: dimensions(ATTRIBUTE_MAX_BLOCK_DIMENSIONS, &mut query),
        max_workgroup_count: dimensions(ATTRIBUTE_MAX_GRID_DIMENSIONS, &mut query),
        local_memory_bytes_per_workgroup: positive(query(ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK))
            .map(u64::from),
        // Zero is a meaningful report of no asynchronous copy engines.
        async_copy_engine_count: query(ATTRIBUTE_ASYNC_ENGINE_COUNT)
            .and_then(|value| u32::try_from(value).ok()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_native_attributes_without_inventing_identity() {
        let facts = map_attributes(|attribute| {
            Some(match attribute {
                ATTRIBUTE_MULTIPROCESSOR_COUNT => 40,
                ATTRIBUTE_WARP_SIZE => 32,
                ATTRIBUTE_MAX_THREADS_PER_BLOCK | 26..=28 => 1024,
                ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK => 65_536,
                ATTRIBUTE_ASYNC_ENGINE_COUNT => 0,
                29..=31 => i32::MAX,
                _ => panic!("unexpected attribute {attribute}"),
            })
        });
        assert_eq!(facts.stable_identity, None);
        assert_eq!(facts.compute_unit_count, Some(40));
        assert_eq!(facts.subgroup_width, Some(32));
        assert_eq!(facts.max_workgroup_invocations, Some(1024));
        assert_eq!(facts.max_workgroup_dimensions, Some([1024; 3]));
        assert_eq!(facts.max_workgroup_count, Some([2_147_483_647; 3]));
        assert_eq!(facts.local_memory_bytes_per_workgroup, Some(65_536));
        assert_eq!(facts.async_copy_engine_count, Some(0));
    }

    #[test]
    fn unavailable_negative_and_zero_limits_stay_unknown() {
        assert_eq!(map_attributes(|_| None), PcuDeviceFacts::default());
        assert_eq!(map_attributes(|_| Some(-1)), PcuDeviceFacts::default());
        let zero = map_attributes(|_| Some(0));
        assert_eq!(
            zero,
            PcuDeviceFacts {
                async_copy_engine_count: Some(0),
                ..PcuDeviceFacts::default()
            }
        );
        let partial = map_attributes(|attribute| (attribute != 27).then_some(1));
        assert_eq!(partial.max_workgroup_dimensions, None);
    }

    #[test]
    #[ignore = "requires RX 6900 XT and ROCm device access"]
    fn rx_6900_xt_hardware_facts() {
        let info = crate::HipRuntime::enumerate_devices()
            .expect("HIP enumeration")
            .into_iter()
            .find(|info| info.name.contains("6900 XT"))
            .expect("RX 6900 XT");
        let facts = query(&info).expect("cold hardware query");
        eprintln!("{}: {facts:?}", info.name);
        assert_eq!(facts.stable_identity, None);
        assert!(facts.compute_unit_count.is_some_and(|value| value > 0));
        assert_eq!(facts.subgroup_width, Some(32));
        assert_eq!(facts.max_workgroup_invocations, Some(1024));
        assert_eq!(facts.local_memory_bytes_per_workgroup, Some(65_536));
        assert!(facts.max_workgroup_dimensions.is_some());
        assert!(facts.max_workgroup_count.is_some());
    }
}
