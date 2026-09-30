//! Driver-only cold neutral facts queries for a validated discovery snapshot.

#[rustfmt::skip]
use crate::{
    CudaDeviceInfo,
    CudaError,
    ffi::{
        Library,
        load_library,
        symbol,
        driver::{
            CUDA_ERROR_INVALID_VALUE,
            CUDA_ERROR_NOT_SUPPORTED,
            DriverDeviceGetAttribute,
            DriverDeviceGetUuid,
            DriverGetDevice,
            DriverInit,
            DriverUuid,
            driver_symbol,
        },
    },
    CUDA_SUCCESS,
    query_pci_bus_id,
    raw_driver_error,
};
#[rustfmt::skip]
use fusion_pcu::{
    PcuDeviceFacts,
    PcuStableDeviceIdentity,
};
#[rustfmt::skip]
use super::{
    DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT,
    DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X,
    DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y,
    DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z,
    DEVICE_ATTRIBUTE_MAX_GRID_DIM_X,
    DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y,
    DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z,
    DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK,
    DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
    DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
    DEVICE_ATTRIBUTE_WARP_SIZE,
    checked_nonnegative,
};

const UUID_NAMESPACE: &str = "cuda-driver-uuid-v2";

fn driver_call<T: Copy>(
    library: &Library,
    operation: &'static str,
    invoke: impl FnOnce(T) -> i32,
) -> Result<(), CudaError> {
    let name = driver_symbol(operation).ok_or_else(|| CudaError::MissingSymbol {
        symbol: operation,
        detail: "facts symbol is not declared in the Driver ABI table".into(),
    })?;
    // SAFETY: all private callers use the declared Driver API ABI and retain this library.
    let function =
        unsafe { symbol::<T>(library, name) }.map_err(|error| CudaError::MissingSymbol {
            symbol: operation,
            detail: error.to_string(),
        })?;
    let status = invoke(*function);
    if status == CUDA_SUCCESS {
        Ok(())
    } else {
        Err(raw_driver_error(library, operation, status))
    }
}

fn optional_fact<T>(result: Result<T, CudaError>) -> Result<Option<T>, CudaError> {
    match result {
        Ok(value) => Ok(Some(value)),
        Err(
            CudaError::MissingSymbol { .. }
            | CudaError::Runtime {
                code: CUDA_ERROR_INVALID_VALUE | CUDA_ERROR_NOT_SUPPORTED,
                ..
            },
        ) => Ok(None),
        Err(error) => Err(error),
    }
}

fn map_facts(
    uuid: Option<[u8; 16]>,
    mut attribute: impl FnMut(i32) -> Result<Option<i32>, CudaError>,
) -> Result<PcuDeviceFacts, CudaError> {
    let stable_identity = uuid.map(|bytes| {
        // Fixed namespace and fixed nonempty 16-octet value satisfy the bounded neutral identity.
        PcuStableDeviceIdentity::new(UUID_NAMESPACE, &bytes)
            .expect("bounded CUDA Driver UUID identity")
    });
    let mut numeric = |kind| attribute(kind)?.map(checked_nonnegative).transpose();
    let compute_unit_count = numeric(DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)?;
    let subgroup_width = numeric(DEVICE_ATTRIBUTE_WARP_SIZE)?;
    let max_workgroup_invocations = numeric(DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)?;
    let max_workgroup_dimensions = dimensions([
        numeric(DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X)?,
        numeric(DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y)?,
        numeric(DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z)?,
    ]);
    let max_workgroup_count = dimensions([
        numeric(DEVICE_ATTRIBUTE_MAX_GRID_DIM_X)?,
        numeric(DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y)?,
        numeric(DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z)?,
    ]);
    let local_memory_bytes_per_workgroup =
        numeric(DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK)?.map(u64::from);
    let async_copy_engine_count = numeric(DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT)?;
    Ok(PcuDeviceFacts {
        stable_identity,
        compute_unit_count,
        subgroup_width,
        max_workgroup_invocations,
        max_workgroup_dimensions,
        max_workgroup_count,
        local_memory_bytes_per_workgroup,
        async_copy_engine_count,
    })
}

const fn dimensions(values: [Option<u32>; 3]) -> Option<[u32; 3]> {
    match values {
        [Some(x), Some(y), Some(z)] => Some([x, y, z]),
        _ => None,
    }
}

/// Query neutral facts without opening a runtime session, creating a context, or allocating a
/// stream. The discovery owner validates its full device reference before calling this helper.
/// The PCI check prevents an ordinal remapping from silently scoring a different device.
pub fn query_snapshot(snapshot: &CudaDeviceInfo) -> Result<PcuDeviceFacts, CudaError> {
    let expected = snapshot
        .pci_bus_id
        .as_ref()
        .ok_or(CudaError::MissingStableDeviceIdentity)?;
    let candidate =
        std::env::var_os("CUDA_DRIVER_LIBRARY").unwrap_or_else(|| "libcuda.so.1".into());
    let library = load_library(&candidate).map_err(CudaError::RuntimeUnavailable)?;
    driver_call(&library, "cuInit", |f: DriverInit| unsafe { f(0) })?;
    let mut device = 0;
    driver_call(&library, "cuDeviceGet", |f: DriverGetDevice| unsafe {
        f(&raw mut device, snapshot.index)
    })?;
    let actual =
        query_pci_bus_id(&library, device).ok_or(CudaError::MissingStableDeviceIdentity)?;
    if actual != *expected {
        return Err(CudaError::DeviceIdentityChanged {
            expected: expected.clone(),
            actual,
        });
    }
    let mut uuid = DriverUuid { bytes: [0; 16] };
    let uuid = optional_fact(
        driver_call(
            &library,
            "cuDeviceGetUuid_v2",
            |f: DriverDeviceGetUuid| unsafe { f(&raw mut uuid, device) },
        )
        .map(|()| uuid.bytes),
    )?;
    map_facts(uuid, |kind| {
        let mut value = 0;
        optional_fact(
            driver_call(
                &library,
                "cuDeviceGetAttribute",
                |f: DriverDeviceGetAttribute| unsafe { f(&raw mut value, kind, device) },
            )
            .map(|()| value),
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_optional_queries_stay_unknown_and_real_errors_propagate() {
        let missing = CudaError::MissingSymbol {
            symbol: "test",
            detail: String::new(),
        };
        assert_eq!(optional_fact::<u32>(Err(missing)).unwrap(), None);
        for code in [1, 801] {
            assert_eq!(
                optional_fact::<u32>(Err(CudaError::Runtime {
                    operation: "cuDeviceGetAttribute",
                    code,
                    detail: None,
                }))
                .unwrap(),
                None
            );
        }
        let error = CudaError::Runtime {
            operation: "cuDeviceGetAttribute",
            code: 700,
            detail: None,
        };
        assert_eq!(optional_fact::<u32>(Err(error.clone())), Err(error));
        assert_eq!(
            map_facts(None, |_| Ok(None)).unwrap(),
            PcuDeviceFacts::default()
        );
    }

    #[test]
    fn cuda_limits_map_to_neutral_facts_without_inventing_partial_dimensions() {
        let mapped = map_facts(Some([19; 16]), |kind| {
            Ok(match kind {
                DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT => Some(70),
                DEVICE_ATTRIBUTE_WARP_SIZE => Some(32),
                DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK
                | DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X
                | DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y => Some(1024),
                DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z => Some(64),
                DEVICE_ATTRIBUTE_MAX_GRID_DIM_X => None,
                DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y | DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z => Some(65535),
                DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK => Some(49152),
                DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT => Some(0),
                _ => unreachable!(),
            })
        })
        .unwrap();
        let identity = mapped.stable_identity.unwrap();
        assert_eq!(identity.namespace(), UUID_NAMESPACE);
        assert_eq!(identity.value(), &[19; 16]);
        assert_eq!(mapped.compute_unit_count, Some(70));
        assert_eq!(mapped.subgroup_width, Some(32));
        assert_eq!(mapped.max_workgroup_invocations, Some(1024));
        assert_eq!(mapped.max_workgroup_dimensions, Some([1024, 1024, 64]));
        assert_eq!(mapped.max_workgroup_count, None);
        assert_eq!(mapped.local_memory_bytes_per_workgroup, Some(49152));
        assert_eq!(mapped.async_copy_engine_count, Some(0));
        assert!(map_facts(None, |_| Ok(Some(-1))).is_err());
    }
}
