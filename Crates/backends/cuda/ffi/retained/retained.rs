//! Cold-resolved Runtime and Driver entrypoints retained by one runtime.

#[rustfmt::skip]
use super::{
    driver,
    runtime,
    Library,
};

#[derive(Clone)]
pub struct RetainedApi {
    pub(super) cu_device_get: Result<driver::DriverGetDevice, crate::CudaError>,
    pub(super) cu_device_get_attribute: Result<driver::DriverDeviceGetAttribute, crate::CudaError>,
    pub(super) cu_device_get_name: Result<driver::DriverDeviceGetName, crate::CudaError>,
    pub(super) cu_device_get_uuid_v2: Result<driver::DriverDeviceGetUuid, crate::CudaError>,
    pub(super) cu_device_total_mem_v2: Result<driver::DriverDeviceTotalMem, crate::CudaError>,
    pub(super) cu_get_error_string: Result<driver::DriverGetErrorString, crate::CudaError>,
    pub(super) cu_launch_kernel: Result<driver::DriverModuleLaunchKernel, crate::CudaError>,
    pub(super) cu_module_get_function: Result<driver::DriverModuleGetFunction, crate::CudaError>,
    pub(super) cu_module_load_data: Result<driver::DriverModuleLoadData, crate::CudaError>,
    pub(super) cu_module_unload: Result<driver::DriverModuleUnload, crate::CudaError>,
    pub(super) cuda_device_get_stream_priority_range:
        Result<runtime::DeviceGetStreamPriorityRange, crate::CudaError>,
    pub(super) cuda_device_synchronize: Result<runtime::DeviceSynchronize, crate::CudaError>,
    pub(super) cuda_event_create_with_flags: Result<runtime::EventCreate, crate::CudaError>,
    pub(super) cuda_event_destroy: Result<runtime::EventDestroy, crate::CudaError>,
    pub(super) cuda_event_elapsed_time: Result<runtime::EventElapsedTime, crate::CudaError>,
    pub(super) cuda_event_query: Result<runtime::EventQuery, crate::CudaError>,
    pub(super) cuda_event_record: Result<runtime::EventRecord, crate::CudaError>,
    pub(super) cuda_event_synchronize: Result<runtime::EventSynchronize, crate::CudaError>,
    pub(super) cuda_free: Result<runtime::RuntimeFree, crate::CudaError>,
    pub(super) cuda_free_host: Result<runtime::FreeHost, crate::CudaError>,
    pub(super) cuda_get_device_count: Result<runtime::GetDeviceCount, crate::CudaError>,
    pub(super) cuda_get_error_string: Result<runtime::GetErrorString, crate::CudaError>,
    pub(super) cuda_graph_destroy: Result<runtime::GraphDestroy, crate::CudaError>,
    pub(super) cuda_graph_exec_destroy: Result<runtime::GraphExecDestroy, crate::CudaError>,
    pub(super) cuda_graph_instantiate_with_flags:
        Result<runtime::GraphInstantiateWithFlags, crate::CudaError>,
    pub(super) cuda_graph_launch: Result<runtime::GraphLaunch, crate::CudaError>,
    pub(super) cuda_malloc: Result<runtime::Malloc, crate::CudaError>,
    pub(super) cuda_malloc_host: Result<runtime::MallocHost, crate::CudaError>,
    pub(super) cuda_mem_get_info: Result<runtime::MemGetInfo, crate::CudaError>,
    pub(super) cuda_memcpy: Result<runtime::Memcpy, crate::CudaError>,
    pub(super) cuda_memcpy_async: Result<runtime::MemcpyAsync, crate::CudaError>,
    pub(super) cuda_memset_async: Result<runtime::MemsetAsync, crate::CudaError>,
    pub(super) cuda_set_device: Result<runtime::SetDevice, crate::CudaError>,
    pub(super) cuda_stream_begin_capture: Result<runtime::StreamBeginCapture, crate::CudaError>,
    pub(super) cuda_stream_create: Result<runtime::StreamCreate, crate::CudaError>,
    pub(super) cuda_stream_create_with_priority:
        Result<runtime::StreamCreateWithPriority, crate::CudaError>,
    pub(super) cuda_stream_destroy: Result<runtime::StreamDestroy, crate::CudaError>,
    pub(super) cuda_stream_end_capture: Result<runtime::StreamEndCapture, crate::CudaError>,
    pub(super) cuda_stream_get_flags: Result<runtime::StreamGetFlags, crate::CudaError>,
    pub(super) cuda_stream_get_priority: Result<runtime::StreamGetPriority, crate::CudaError>,
    pub(super) cuda_stream_query: Result<runtime::StreamQuery, crate::CudaError>,
    pub(super) cuda_stream_synchronize: Result<runtime::StreamSynchronize, crate::CudaError>,
    pub(super) cuda_stream_wait_event: Result<runtime::StreamWaitEvent, crate::CudaError>,
}

impl RetainedApi {
    #[allow(clippy::too_many_lines)] // One cold ABI table keeps exact symbol/type pairings reviewable.
    pub(crate) fn load(runtime_library: &Library, driver_library: &Library) -> Self {
        Self {
            cu_device_get: resolve::<driver::DriverGetDevice>(
                driver_library,
                "cuDeviceGet",
                b"cuDeviceGet\0",
            ),
            cu_device_get_attribute: resolve::<driver::DriverDeviceGetAttribute>(
                driver_library,
                "cuDeviceGetAttribute",
                b"cuDeviceGetAttribute\0",
            ),
            cu_device_get_name: resolve::<driver::DriverDeviceGetName>(
                driver_library,
                "cuDeviceGetName",
                b"cuDeviceGetName\0",
            ),
            cu_device_get_uuid_v2: resolve::<driver::DriverDeviceGetUuid>(
                driver_library,
                "cuDeviceGetUuid_v2",
                b"cuDeviceGetUuid_v2\0",
            ),
            cu_device_total_mem_v2: resolve::<driver::DriverDeviceTotalMem>(
                driver_library,
                "cuDeviceTotalMem_v2",
                b"cuDeviceTotalMem_v2\0",
            ),
            cu_get_error_string: resolve::<driver::DriverGetErrorString>(
                driver_library,
                "cuGetErrorString",
                b"cuGetErrorString\0",
            ),
            cu_launch_kernel: resolve::<driver::DriverModuleLaunchKernel>(
                driver_library,
                "cuLaunchKernel",
                b"cuLaunchKernel\0",
            ),
            cu_module_get_function: resolve::<driver::DriverModuleGetFunction>(
                driver_library,
                "cuModuleGetFunction",
                b"cuModuleGetFunction\0",
            ),
            cu_module_load_data: resolve::<driver::DriverModuleLoadData>(
                driver_library,
                "cuModuleLoadData",
                b"cuModuleLoadData\0",
            ),
            cu_module_unload: resolve::<driver::DriverModuleUnload>(
                driver_library,
                "cuModuleUnload",
                b"cuModuleUnload\0",
            ),
            cuda_device_get_stream_priority_range: resolve::<runtime::DeviceGetStreamPriorityRange>(
                runtime_library,
                "cudaDeviceGetStreamPriorityRange",
                b"cudaDeviceGetStreamPriorityRange\0",
            ),
            cuda_device_synchronize: resolve::<runtime::DeviceSynchronize>(
                runtime_library,
                "cudaDeviceSynchronize",
                b"cudaDeviceSynchronize\0",
            ),
            cuda_event_create_with_flags: resolve::<runtime::EventCreate>(
                runtime_library,
                "cudaEventCreateWithFlags",
                b"cudaEventCreateWithFlags\0",
            ),
            cuda_event_destroy: resolve::<runtime::EventDestroy>(
                runtime_library,
                "cudaEventDestroy",
                b"cudaEventDestroy\0",
            ),
            cuda_event_elapsed_time: resolve::<runtime::EventElapsedTime>(
                runtime_library,
                "cudaEventElapsedTime",
                b"cudaEventElapsedTime\0",
            ),
            cuda_event_query: resolve::<runtime::EventQuery>(
                runtime_library,
                "cudaEventQuery",
                b"cudaEventQuery\0",
            ),
            cuda_event_record: resolve::<runtime::EventRecord>(
                runtime_library,
                "cudaEventRecord",
                b"cudaEventRecord\0",
            ),
            cuda_event_synchronize: resolve::<runtime::EventSynchronize>(
                runtime_library,
                "cudaEventSynchronize",
                b"cudaEventSynchronize\0",
            ),
            cuda_free: resolve::<runtime::RuntimeFree>(runtime_library, "cudaFree", b"cudaFree\0"),
            cuda_free_host: resolve::<runtime::FreeHost>(
                runtime_library,
                "cudaFreeHost",
                b"cudaFreeHost\0",
            ),
            cuda_get_device_count: resolve::<runtime::GetDeviceCount>(
                runtime_library,
                "cudaGetDeviceCount",
                b"cudaGetDeviceCount\0",
            ),
            cuda_get_error_string: resolve::<runtime::GetErrorString>(
                runtime_library,
                "cudaGetErrorString",
                b"cudaGetErrorString\0",
            ),
            cuda_graph_destroy: resolve::<runtime::GraphDestroy>(
                runtime_library,
                "cudaGraphDestroy",
                b"cudaGraphDestroy\0",
            ),
            cuda_graph_exec_destroy: resolve::<runtime::GraphExecDestroy>(
                runtime_library,
                "cudaGraphExecDestroy",
                b"cudaGraphExecDestroy\0",
            ),
            cuda_graph_instantiate_with_flags: resolve::<runtime::GraphInstantiateWithFlags>(
                runtime_library,
                "cudaGraphInstantiateWithFlags",
                b"cudaGraphInstantiateWithFlags\0",
            ),
            cuda_graph_launch: resolve::<runtime::GraphLaunch>(
                runtime_library,
                "cudaGraphLaunch",
                b"cudaGraphLaunch\0",
            ),
            cuda_malloc: resolve::<runtime::Malloc>(runtime_library, "cudaMalloc", b"cudaMalloc\0"),
            cuda_malloc_host: resolve::<runtime::MallocHost>(
                runtime_library,
                "cudaMallocHost",
                b"cudaMallocHost\0",
            ),
            cuda_mem_get_info: resolve::<runtime::MemGetInfo>(
                runtime_library,
                "cudaMemGetInfo",
                b"cudaMemGetInfo\0",
            ),
            cuda_memcpy: resolve::<runtime::Memcpy>(runtime_library, "cudaMemcpy", b"cudaMemcpy\0"),
            cuda_memcpy_async: resolve::<runtime::MemcpyAsync>(
                runtime_library,
                "cudaMemcpyAsync",
                b"cudaMemcpyAsync\0",
            ),
            cuda_memset_async: resolve::<runtime::MemsetAsync>(
                runtime_library,
                "cudaMemsetAsync",
                b"cudaMemsetAsync\0",
            ),
            cuda_set_device: resolve::<runtime::SetDevice>(
                runtime_library,
                "cudaSetDevice",
                b"cudaSetDevice\0",
            ),
            cuda_stream_begin_capture: resolve::<runtime::StreamBeginCapture>(
                runtime_library,
                "cudaStreamBeginCapture",
                b"cudaStreamBeginCapture\0",
            ),
            cuda_stream_create: resolve::<runtime::StreamCreate>(
                runtime_library,
                "cudaStreamCreate",
                b"cudaStreamCreate\0",
            ),
            cuda_stream_create_with_priority: resolve::<runtime::StreamCreateWithPriority>(
                runtime_library,
                "cudaStreamCreateWithPriority",
                b"cudaStreamCreateWithPriority\0",
            ),
            cuda_stream_destroy: resolve::<runtime::StreamDestroy>(
                runtime_library,
                "cudaStreamDestroy",
                b"cudaStreamDestroy\0",
            ),
            cuda_stream_end_capture: resolve::<runtime::StreamEndCapture>(
                runtime_library,
                "cudaStreamEndCapture",
                b"cudaStreamEndCapture\0",
            ),
            cuda_stream_get_flags: resolve::<runtime::StreamGetFlags>(
                runtime_library,
                "cudaStreamGetFlags",
                b"cudaStreamGetFlags\0",
            ),
            cuda_stream_get_priority: resolve::<runtime::StreamGetPriority>(
                runtime_library,
                "cudaStreamGetPriority",
                b"cudaStreamGetPriority\0",
            ),
            cuda_stream_query: resolve::<runtime::StreamQuery>(
                runtime_library,
                "cudaStreamQuery",
                b"cudaStreamQuery\0",
            ),
            cuda_stream_synchronize: resolve::<runtime::StreamSynchronize>(
                runtime_library,
                "cudaStreamSynchronize",
                b"cudaStreamSynchronize\0",
            ),
            cuda_stream_wait_event: resolve::<runtime::StreamWaitEvent>(
                runtime_library,
                "cudaStreamWaitEvent",
                b"cudaStreamWaitEvent\0",
            ),
        }
    }
}

fn resolve<T: Copy>(
    library: &Library,
    name: &'static str,
    bytes: &[u8],
) -> Result<T, crate::CudaError> {
    // SAFETY: every table entry pairs the SDK symbol with its exact declared C ABI.
    // RuntimeInner retains both originating libraries for every copied pointer's lifetime.
    unsafe { super::symbol::<T>(library, bytes) }
        .map(|pointer| *pointer)
        .map_err(|error| crate::CudaError::MissingSymbol {
            symbol: name,
            detail: error.to_string(),
        })
}

#[cfg(all(test, feature = "allocation-census"))]
mod tests {
    #[test]
    #[ignore = "requires authorized idle CUDA GPU; run serially"]
    fn warm_runtime_calls_and_thread_device_selection_resolve_no_symbols() {
        let runtime = crate::CudaRuntime::new(0).unwrap();
        // Opening/configuring another host thread must still select this runtime's device;
        // retaining pointers does not turn CUDA's thread-local current device into global state.
        std::thread::spawn(move || {
            super::super::reset_cuda_api_census();
            assert!(runtime.device_count().unwrap() > 0);
            assert!(runtime.memory_info().unwrap().total_bytes > 0);
            let mut buffer = runtime.allocate(16).unwrap();
            buffer.copy_from(&[3_u8; 16]).unwrap();
            let mut actual = [0_u8; 16];
            buffer.copy_to(&mut actual).unwrap();
            assert_eq!(actual, [3_u8; 16]);
            let stream = runtime.create_stream().unwrap();
            stream.synchronize().unwrap();
            drop(stream);
            drop(buffer);
            let census = super::super::cuda_api_census();
            assert!(census.runtime_driver_calls >= 8);
            assert_eq!(census.symbol_resolutions, 0);
            assert_eq!(census.allocations, 1);
            assert_eq!(census.frees, 1);
        })
        .join()
        .unwrap();
    }
}
