//! Exact HIP ABI entries resolved once and retained with the owning runtime library.
#[rustfmt::skip]
use super::{hip, Library};
#[derive(Clone)]
pub struct RetainedApi {
    pub(super) get_device_count: Result<hip::GetDeviceCount, crate::HipError>,
    pub(super) device_total_mem: Result<hip::DeviceTotalMem, crate::HipError>,
    pub(super) mem_get_info: Result<hip::MemGetInfo, crate::HipError>,
    pub(super) malloc: Result<hip::Malloc, crate::HipError>,
    pub(super) module_load_data: Result<hip::ModuleLoadData, crate::HipError>,
    pub(super) stream_create: Result<hip::StreamCreate, crate::HipError>,
    pub(super) event_create_with_flags: Result<hip::EventCreate, crate::HipError>,
    pub(super) event_elapsed_time: Result<hip::EventElapsedTime, crate::HipError>,
    pub(super) device_get_name: Result<hip::GetDeviceName, crate::HipError>,
    pub(super) device_get: Result<hip::GetDevice, crate::HipError>,
    pub(super) set_device: Result<hip::SetDevice, crate::HipError>,
    pub(super) free: Result<hip::Free, crate::HipError>,
    pub(super) memcpy: Result<hip::Memcpy, crate::HipError>,
    pub(super) device_synchronize: Result<hip::HipNoArgStatus, crate::HipError>,
    pub(super) stream_destroy: Result<hip::StreamDestroy, crate::HipError>,
    pub(super) stream_synchronize: Result<hip::StreamSynchronize, crate::HipError>,
    pub(super) event_record: Result<hip::EventRecord, crate::HipError>,
    pub(super) event_destroy: Result<hip::EventDestroy, crate::HipError>,
    pub(super) event_synchronize: Result<hip::EventSynchronize, crate::HipError>,
    pub(super) module_unload: Result<hip::ModuleUnload, crate::HipError>,
    pub(super) module_get_function: Result<hip::ModuleGetFunction, crate::HipError>,
    pub(super) memcpy_async: Result<hip::MemcpyAsync, crate::HipError>,
    pub(super) stream_wait_event: Result<hip::StreamWaitEvent, crate::HipError>,
    pub(super) module_launch_kernel: Result<hip::ModuleLaunchKernel, crate::HipError>,
    pub(super) device_get_pci_bus_id: Result<hip::GetDevicePciBusId, crate::HipError>,
    pub(super) get_error_string: Result<hip::GetErrorString, crate::HipError>,
}
impl RetainedApi {
    #[allow(clippy::too_many_lines)] // One cold table pairs every exact SDK name and ABI.
    pub(crate) fn load(library: &Library) -> Self {
        Self {
            get_device_count: resolve::<hip::GetDeviceCount>(
                library,
                "hipGetDeviceCount",
                b"hipGetDeviceCount\0",
            ),
            device_total_mem: resolve::<hip::DeviceTotalMem>(
                library,
                "hipDeviceTotalMem",
                b"hipDeviceTotalMem\0",
            ),
            mem_get_info: resolve::<hip::MemGetInfo>(library, "hipMemGetInfo", b"hipMemGetInfo\0"),
            malloc: resolve::<hip::Malloc>(library, "hipMalloc", b"hipMalloc\0"),
            module_load_data: resolve::<hip::ModuleLoadData>(
                library,
                "hipModuleLoadData",
                b"hipModuleLoadData\0",
            ),
            stream_create: resolve::<hip::StreamCreate>(
                library,
                "hipStreamCreate",
                b"hipStreamCreate\0",
            ),
            event_create_with_flags: resolve::<hip::EventCreate>(
                library,
                "hipEventCreateWithFlags",
                b"hipEventCreateWithFlags\0",
            ),
            event_elapsed_time: resolve::<hip::EventElapsedTime>(
                library,
                "hipEventElapsedTime",
                b"hipEventElapsedTime\0",
            ),
            device_get_name: resolve::<hip::GetDeviceName>(
                library,
                "hipDeviceGetName",
                b"hipDeviceGetName\0",
            ),
            device_get: resolve::<hip::GetDevice>(library, "hipDeviceGet", b"hipDeviceGet\0"),
            set_device: resolve::<hip::SetDevice>(library, "hipSetDevice", b"hipSetDevice\0"),
            free: resolve::<hip::Free>(library, "hipFree", b"hipFree\0"),
            memcpy: resolve::<hip::Memcpy>(library, "hipMemcpy", b"hipMemcpy\0"),
            device_synchronize: resolve::<hip::HipNoArgStatus>(
                library,
                "hipDeviceSynchronize",
                b"hipDeviceSynchronize\0",
            ),
            stream_destroy: resolve::<hip::StreamDestroy>(
                library,
                "hipStreamDestroy",
                b"hipStreamDestroy\0",
            ),
            stream_synchronize: resolve::<hip::StreamSynchronize>(
                library,
                "hipStreamSynchronize",
                b"hipStreamSynchronize\0",
            ),
            event_record: resolve::<hip::EventRecord>(
                library,
                "hipEventRecord",
                b"hipEventRecord\0",
            ),
            event_destroy: resolve::<hip::EventDestroy>(
                library,
                "hipEventDestroy",
                b"hipEventDestroy\0",
            ),
            event_synchronize: resolve::<hip::EventSynchronize>(
                library,
                "hipEventSynchronize",
                b"hipEventSynchronize\0",
            ),
            module_unload: resolve::<hip::ModuleUnload>(
                library,
                "hipModuleUnload",
                b"hipModuleUnload\0",
            ),
            module_get_function: resolve::<hip::ModuleGetFunction>(
                library,
                "hipModuleGetFunction",
                b"hipModuleGetFunction\0",
            ),
            memcpy_async: resolve::<hip::MemcpyAsync>(
                library,
                "hipMemcpyAsync",
                b"hipMemcpyAsync\0",
            ),
            stream_wait_event: resolve::<hip::StreamWaitEvent>(
                library,
                "hipStreamWaitEvent",
                b"hipStreamWaitEvent\0",
            ),
            module_launch_kernel: resolve::<hip::ModuleLaunchKernel>(
                library,
                "hipModuleLaunchKernel",
                b"hipModuleLaunchKernel\0",
            ),
            device_get_pci_bus_id: resolve::<hip::GetDevicePciBusId>(
                library,
                "hipDeviceGetPCIBusId",
                b"hipDeviceGetPCIBusId\0",
            ),
            get_error_string: resolve::<hip::GetErrorString>(
                library,
                "hipGetErrorString",
                b"hipGetErrorString\0",
            ),
        }
    }
}
fn resolve<T: Copy>(
    library: &Library,
    name: &'static str,
    bytes: &[u8],
) -> Result<T, crate::HipError> {
    // SAFETY: Only the exact name/type pairs in the table above call this helper. RuntimeInner
    // retains the same Library through every use, including cloned cross-thread runtimes.
    unsafe { super::symbol::<T>(library, bytes) }
        .map(|entry| *entry)
        .map_err(|error| crate::HipError::MissingSymbol {
            symbol: name,
            detail: error.to_string(),
        })
}

#[cfg(all(test, feature = "allocation-census", target_os = "linux"))]
mod tests {
    #[test]
    fn missing_entries_are_retained_without_warm_relookup_or_submission() {
        // SAFETY: libc is the trusted platform C runtime. It deliberately exports no HIP
        // entries; no function pointer from this library is ever invoked by this test.
        let library =
            unsafe { super::super::load_uncached_library(std::ffi::OsStr::new("libc.so.6")) }
                .unwrap();
        let api = super::RetainedApi::load(&library);
        let runtime = crate::HipRuntime(std::sync::Arc::new(crate::RuntimeInner {
            library: std::sync::Arc::new(library),
            api,
            device: 0,
            name: String::new(),
        }));
        super::super::reset_rocm_api_census();
        for _ in 0..2 {
            let result =
                super::super::hip_call(&runtime, "hipMalloc", &runtime.0.api.malloc, |_function| {
                    panic!("unresolved entry cannot be called")
                });
            assert!(matches!(
                result,
                Err(crate::HipError::MissingSymbol {
                    symbol: "hipMalloc",
                    ..
                })
            ));
            assert!(matches!(
                super::super::hip_error(&runtime, "retained-status", 17),
                crate::HipError::Runtime {
                    code: 17,
                    detail: None,
                    ..
                }
            ));
        }
        let census = super::super::rocm_api_census();
        assert_eq!(census.symbol_resolutions, 0);
        assert_eq!(census.runtime_calls, 0);
    }
}
