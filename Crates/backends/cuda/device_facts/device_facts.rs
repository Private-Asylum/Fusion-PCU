//! Explicit physical device facts for selection and diagnostic consumers.
//!
//! These hardware limits and flags never advertise neutral executable operations. Querying them
//! is opt-in and does not add metadata calls to ordinary preparation, dispatch, or facade paths.

mod discovery;
pub use discovery::query_snapshot;

use std::fmt;
#[rustfmt::skip]
use crate::{
    CudaError,
    CudaRuntime,
    ffi::driver::{
        DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT,
        DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
        DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
        DEVICE_ATTRIBUTE_CONCURRENT_KERNELS,
        DEVICE_ATTRIBUTE_INTEGRATED,
        DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X,
        DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y,
        DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z,
        DEVICE_ATTRIBUTE_MAX_GRID_DIM_X,
        DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y,
        DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z,
        DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK,
        DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
        DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR,
        DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
        DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED,
        DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING,
        DEVICE_ATTRIBUTE_WARP_SIZE,
        DriverUuid,
    },
};

/// Stable Driver API 16-octet identity, preserving UUID bytes without marketing-name inference.
/// CUDA's v2 identity distinguishes a subscribed MIG compute instance when MIG is active.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CudaDeviceUuid(pub [u8; 16]);

impl fmt::Display for CudaDeviceUuid {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (index, byte) in self.0.iter().enumerate() {
            if matches!(index, 4 | 6 | 8 | 10) {
                formatter.write_str("-")?;
            }
            write!(formatter, "{byte:02x}")?;
        }
        Ok(())
    }
}

/// Independent physical flags, separate from neutral executable capability masks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CudaPhysicalDeviceFlags(u8);

impl CudaPhysicalDeviceFlags {
    pub const INTEGRATED: Self = Self(1);
    pub const CONCURRENT_KERNELS: Self = Self(2);
    pub const UNIFIED_ADDRESSING: Self = Self(4);
    pub const STREAM_PRIORITIES: Self = Self(8);

    #[must_use]
    pub const fn contains(self, flag: Self) -> bool {
        self.0 & flag.0 == flag.0
    }
}

/// Physical limits and hardware flags, independent from PCU operation admission/capabilities.
/// In particular, unified addressing is not a managed-memory ownership or host-access promise.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CudaPhysicalDeviceFacts {
    pub uuid: CudaDeviceUuid,
    pub compute_capability: [u32; 2],
    pub multiprocessor_count: u32,
    pub warp_size: u32,
    pub max_threads_per_block: u32,
    pub max_threads_per_multiprocessor: u32,
    pub max_block_dimensions: [u32; 3],
    pub max_grid_dimensions: [u32; 3],
    pub max_shared_memory_per_block: u64,
    pub asynchronous_engine_count: u32,
    pub hardware_flags: CudaPhysicalDeviceFlags,
}

fn checked_nonnegative(value: i32) -> Result<u32, CudaError> {
    u32::try_from(value).map_err(|_| CudaError::Runtime {
        operation: "cuDeviceGetAttribute",
        code: -1,
        detail: Some(format!("negative physical-device attribute {value}")),
    })
}

fn checked_boolean(value: i32) -> Result<bool, CudaError> {
    match value {
        0 => Ok(false),
        1 => Ok(true),
        _ => Err(CudaError::Runtime {
            operation: "cuDeviceGetAttribute",
            code: -1,
            detail: Some(format!("nonboolean physical-device attribute {value}")),
        }),
    }
}

impl CudaRuntime {
    /// Query stable Driver UUID and selected physical-device limits/flags on explicit request.
    /// These facts are not PCU executable capabilities or a throughput score. They do not infer
    /// cooperative launch, Tensor Core use, managed-memory access, or vendor math admission.
    ///
    /// # Errors
    /// Returns a missing stable Driver symbol, CUDA query error, or malformed attribute value.
    pub fn physical_device_facts(&self) -> Result<CudaPhysicalDeviceFacts, CudaError> {
        let mut uuid = DriverUuid { bytes: [0; 16] };
        unsafe { crate::ffi::invoke_cuDeviceGetUuid_v2(self, &raw mut uuid, self.0.device) }?;
        let numeric = |attribute| checked_nonnegative(self.physical_attribute(attribute)?);
        let mut flags = 0;
        for (attribute, flag) in [
            (
                DEVICE_ATTRIBUTE_INTEGRATED,
                CudaPhysicalDeviceFlags::INTEGRATED,
            ),
            (
                DEVICE_ATTRIBUTE_CONCURRENT_KERNELS,
                CudaPhysicalDeviceFlags::CONCURRENT_KERNELS,
            ),
            (
                DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING,
                CudaPhysicalDeviceFlags::UNIFIED_ADDRESSING,
            ),
            (
                DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED,
                CudaPhysicalDeviceFlags::STREAM_PRIORITIES,
            ),
        ] {
            if checked_boolean(self.physical_attribute(attribute)?)? {
                flags |= flag.0;
            }
        }
        Ok(CudaPhysicalDeviceFacts {
            uuid: CudaDeviceUuid(uuid.bytes),
            compute_capability: [
                numeric(DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)?,
                numeric(DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)?,
            ],
            multiprocessor_count: numeric(DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)?,
            warp_size: numeric(DEVICE_ATTRIBUTE_WARP_SIZE)?,
            max_threads_per_block: numeric(DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)?,
            max_threads_per_multiprocessor: numeric(
                DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR,
            )?,
            max_block_dimensions: [
                numeric(DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X)?,
                numeric(DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y)?,
                numeric(DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z)?,
            ],
            max_grid_dimensions: [
                numeric(DEVICE_ATTRIBUTE_MAX_GRID_DIM_X)?,
                numeric(DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y)?,
                numeric(DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z)?,
            ],
            max_shared_memory_per_block: u64::from(numeric(
                DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK,
            )?),
            asynchronous_engine_count: numeric(DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT)?,
            hardware_flags: CudaPhysicalDeviceFlags(flags),
        })
    }

    fn physical_attribute(&self, attribute: i32) -> Result<i32, CudaError> {
        let mut value = 0;
        unsafe {
            crate::ffi::invoke_cuDeviceGetAttribute(self, &raw mut value, attribute, self.0.device)
        }?;
        Ok(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uuid_format_preserves_all_octets_in_canonical_order() {
        assert_eq!(
            CudaDeviceUuid(std::array::from_fn(|index| u8::try_from(index).unwrap())).to_string(),
            "00010203-0405-0607-0809-0a0b0c0d0e0f"
        );
        assert_eq!(size_of::<DriverUuid>(), 16);
        assert_eq!(align_of::<DriverUuid>(), 1);
    }

    #[test]
    fn physical_fact_validation_does_not_guess_unknown_values() {
        assert_eq!(checked_nonnegative(0).unwrap(), 0);
        assert_eq!(
            checked_nonnegative(i32::MAX).unwrap(),
            i32::MAX.unsigned_abs()
        );
        assert!(checked_nonnegative(-1).is_err());
        assert!(!checked_boolean(0).unwrap());
        assert!(checked_boolean(1).unwrap());
        assert!(checked_boolean(2).is_err());
        assert!(checked_boolean(-1).is_err());
        let flags = CudaPhysicalDeviceFlags(5);
        assert!(flags.contains(CudaPhysicalDeviceFlags::INTEGRATED));
        assert!(flags.contains(CudaPhysicalDeviceFlags::UNIFIED_ADDRESSING));
        assert!(!flags.contains(CudaPhysicalDeviceFlags::CONCURRENT_KERNELS));
        assert!(!flags.contains(CudaPhysicalDeviceFlags::STREAM_PRIORITIES));
    }

    #[test]
    #[ignore = "requires CUDA Driver device for UUID and physical attributes"]
    fn physical_device_facts_are_stable_and_match_selected_architecture() {
        let runtime = CudaRuntime::new(0).unwrap();
        let first = runtime.physical_device_facts().unwrap();
        assert_eq!(runtime.physical_device_facts().unwrap(), first);
        assert_ne!(first.uuid.0, [0; 16]);
        assert!(first.multiprocessor_count > 0);
        assert!(first.warp_size > 0);
        assert!(first.max_threads_per_block > 0);
        assert!(first.max_threads_per_multiprocessor >= first.max_threads_per_block);
        assert!(
            first
                .max_block_dimensions
                .into_iter()
                .all(|value| value > 0)
        );
        assert!(first.max_grid_dimensions.into_iter().all(|value| value > 0));
        assert!(first.max_shared_memory_per_block > 0);
        let [major, minor] = first.compute_capability;
        assert_eq!(
            runtime.device_info().unwrap().architecture,
            Some(format!("sm_{major}{minor}"))
        );
    }
}
