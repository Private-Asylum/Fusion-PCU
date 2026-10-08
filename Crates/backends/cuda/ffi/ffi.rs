//! Backend-private CUDA ABI declarations and dynamic symbol loading.
//! Safe execution and resource ownership live outside this module.

#[rustfmt::skip]
pub use libloading::{
    Library,
    Symbol,
};

/// Resolve a symbol while preserving its borrowed library lifetime.
///
/// # Safety
///
/// `T` must match the named symbol's exact ABI and signature. The symbol must not outlive `library`.
pub unsafe fn symbol<'library, T>(
    library: &'library Library,
    name: &[u8],
) -> Result<Symbol<'library, T>, libloading::Error> {
    #[cfg(feature = "allocation-census")]
    census::symbol();
    // SAFETY: the caller establishes the requested ABI and retained-library lifetime.
    unsafe { library.get(name) }
}

#[path = "loader.rs"]
mod loader;
#[rustfmt::skip]
pub use loader::{
    load_library,
    load_uncached_library,
};

#[cfg(feature = "allocation-census")]
#[path = "census/census.rs"]
mod census;
pub mod cublas;
pub mod cublaslt;
pub mod driver;
pub mod nvrtc;
#[path = "retained/retained.rs"]
mod retained;
pub mod runtime;
pub use retained::RetainedApi;
#[cfg(feature = "allocation-census")]
pub use census::{CudaApiCensus, cuda_api_census, reset_cuda_api_census};

#[rustfmt::skip]
use std::ffi::{
    c_char,
    c_int,
    c_void,
};
#[rustfmt::skip]
use crate::{
    CudaError,
    CudaRuntime,
};

/// Invoke the declared `cudaFree` ABI using the retained selected runtime.
///
/// # Safety
/// `arg0` must be null or a live allocation from this runtime, with no remaining queued users. It must be released exactly once.
#[allow(non_snake_case)] // Preserve SDK operation names at the private FFI boundary.
pub unsafe fn invoke_cudaFree(runtime: &CudaRuntime, arg0: *mut c_void) -> Result<(), CudaError> {
    #[cfg(feature = "allocation-census")]
    census::call("cudaFree", None);
    runtime.call(
        "cudaFree",
        runtime
            .0
            .api
            .cuda_free
            .as_ref()
            .copied()
            .map_err(Clone::clone)?,
        |f: runtime::RuntimeFree| unsafe { f(arg0) },
    )
}

/// Invoke the declared `cuDeviceGet` ABI using the retained selected runtime.
///
/// # Safety
/// Any result pointers must address writable storage of the declared pointee type. All handles must belong to the selected device/context and remain live for this call.
#[allow(non_snake_case)] // Preserve SDK operation names at the private FFI boundary.
pub unsafe fn invoke_cuDeviceGet(
    runtime: &CudaRuntime,
    arg0: *mut CudaDevice,
    arg1: c_int,
) -> Result<(), CudaError> {
    #[cfg(feature = "allocation-census")]
    census::call("cuDeviceGet", None);
    runtime.driver_call(
        "cuDeviceGet",
        runtime
            .0
            .api
            .cu_device_get
            .as_ref()
            .copied()
            .map_err(Clone::clone)?,
        |f: driver::DriverGetDevice| unsafe { f(arg0, arg1) },
    )
}

/// Invoke the declared `cudaGetDeviceCount` ABI using the retained selected runtime.
///
/// # Safety
/// Any result pointers must address writable storage of the declared pointee type. All handles must belong to the selected device/context and remain live for this call.
#[allow(non_snake_case)] // Preserve SDK operation names at the private FFI boundary.
pub unsafe fn invoke_cudaGetDeviceCount(
    runtime: &CudaRuntime,
    arg0: *mut c_int,
) -> Result<(), CudaError> {
    #[cfg(feature = "allocation-census")]
    census::call("cudaGetDeviceCount", None);
    runtime.call(
        "cudaGetDeviceCount",
        runtime
            .0
            .api
            .cuda_get_device_count
            .as_ref()
            .copied()
            .map_err(Clone::clone)?,
        |f: runtime::GetDeviceCount| unsafe { f(arg0) },
    )
}

/// Invoke the declared `cuDeviceTotalMem_v2` ABI using the retained selected runtime.
///
/// # Safety
/// Any result pointers must address writable storage of the declared pointee type. All handles must belong to the selected device/context and remain live for this call.
#[allow(non_snake_case)] // Preserve SDK operation names at the private FFI boundary.
pub unsafe fn invoke_cuDeviceTotalMem_v2(
    runtime: &CudaRuntime,
    arg0: *mut usize,
    arg1: CudaDevice,
) -> Result<(), CudaError> {
    #[cfg(feature = "allocation-census")]
    census::call("cuDeviceTotalMem_v2", None);
    runtime.driver_call(
        "cuDeviceTotalMem_v2",
        runtime
            .0
            .api
            .cu_device_total_mem_v2
            .as_ref()
            .copied()
            .map_err(Clone::clone)?,
        |f: driver::DriverDeviceTotalMem| unsafe { f(arg0, arg1) },
    )
}

/// Invoke the declared `cudaMemGetInfo` ABI using the retained selected runtime.
///
/// # Safety
/// Any result pointers must address writable storage of the declared pointee type. All handles must belong to the selected device/context and remain live for this call.
#[allow(non_snake_case)] // Preserve SDK operation names at the private FFI boundary.
pub unsafe fn invoke_cudaMemGetInfo(
    runtime: &CudaRuntime,
    arg0: *mut usize,
    arg1: *mut usize,
) -> Result<(), CudaError> {
    #[cfg(feature = "allocation-census")]
    census::call("cudaMemGetInfo", None);
    runtime.call(
        "cudaMemGetInfo",
        runtime
            .0
            .api
            .cuda_mem_get_info
            .as_ref()
            .copied()
            .map_err(Clone::clone)?,
        |f: runtime::MemGetInfo| unsafe { f(arg0, arg1) },
    )
}

/// Invoke the declared `cudaMalloc` ABI using the retained selected runtime.
///
/// # Safety
/// Any result pointers must address writable storage of the declared pointee type. All handles must belong to the selected device/context and remain live for this call.
#[allow(non_snake_case)] // Preserve SDK operation names at the private FFI boundary.
pub unsafe fn invoke_cudaMalloc(
    runtime: &CudaRuntime,
    arg0: *mut *mut c_void,
    arg1: usize,
) -> Result<(), CudaError> {
    #[cfg(feature = "allocation-census")]
    census::call("cudaMalloc", None);
    runtime.call(
        "cudaMalloc",
        runtime
            .0
            .api
            .cuda_malloc
            .as_ref()
            .copied()
            .map_err(Clone::clone)?,
        |f: runtime::Malloc| unsafe { f(arg0, arg1) },
    )
}

/// Invoke the declared `cuModuleLoadData` ABI using the retained selected runtime.
///
/// # Safety
/// `arg0` must be writable module-handle storage. `arg1` must address a complete, valid CUDA image (including a terminator for PTX).
#[allow(non_snake_case)] // Preserve SDK operation names at the private FFI boundary.
pub unsafe fn invoke_cuModuleLoadData(
    runtime: &CudaRuntime,
    arg0: *mut driver::ModuleHandle,
    arg1: *const c_void,
) -> Result<(), CudaError> {
    #[cfg(feature = "allocation-census")]
    census::call("cuModuleLoadData", None);
    runtime.driver_call(
        "cuModuleLoadData",
        runtime
            .0
            .api
            .cu_module_load_data
            .as_ref()
            .copied()
            .map_err(Clone::clone)?,
        |f: driver::DriverModuleLoadData| unsafe { f(arg0, arg1) },
    )
}

/// Invoke the declared `cudaStreamCreate` ABI using the retained selected runtime.
///
/// # Safety
/// Any result pointers must address writable storage of the declared pointee type. All handles must belong to the selected device/context and remain live for this call.
#[allow(non_snake_case)] // Preserve SDK operation names at the private FFI boundary.
pub unsafe fn invoke_cudaStreamCreate(
    runtime: &CudaRuntime,
    arg0: *mut runtime::CudaStream,
) -> Result<(), CudaError> {
    #[cfg(feature = "allocation-census")]
    census::call("cudaStreamCreate", None);
    runtime.call(
        "cudaStreamCreate",
        runtime
            .0
            .api
            .cuda_stream_create
            .as_ref()
            .copied()
            .map_err(Clone::clone)?,
        |f: runtime::StreamCreate| unsafe { f(arg0) },
    )
}

/// Invoke the declared `cudaEventCreateWithFlags` ABI using the retained selected runtime.
///
/// # Safety
/// Any result pointers must address writable storage of the declared pointee type. All handles must belong to the selected device/context and remain live for this call.
#[allow(non_snake_case)] // Preserve SDK operation names at the private FFI boundary.
pub unsafe fn invoke_cudaEventCreateWithFlags(
    runtime: &CudaRuntime,
    arg0: *mut runtime::CudaEvent,
    arg1: c_int,
) -> Result<(), CudaError> {
    #[cfg(feature = "allocation-census")]
    census::call("cudaEventCreateWithFlags", None);
    runtime.call(
        "cudaEventCreateWithFlags",
        runtime
            .0
            .api
            .cuda_event_create_with_flags
            .as_ref()
            .copied()
            .map_err(Clone::clone)?,
        |f: runtime::EventCreate| unsafe { f(arg0, arg1) },
    )
}

/// Invoke the declared `cudaEventElapsedTime` ABI using the retained selected runtime.
///
/// # Safety
/// `arg0` must be writable; both live events must have timing enabled and completed records.
#[allow(non_snake_case)] // Preserve SDK operation names at the private FFI boundary.
pub unsafe fn invoke_cudaEventElapsedTime(
    runtime: &CudaRuntime,
    arg0: *mut f32,
    arg1: runtime::CudaEvent,
    arg2: runtime::CudaEvent,
) -> Result<(), CudaError> {
    #[cfg(feature = "allocation-census")]
    census::call("cudaEventElapsedTime", None);
    runtime.call(
        "cudaEventElapsedTime",
        runtime
            .0
            .api
            .cuda_event_elapsed_time
            .as_ref()
            .copied()
            .map_err(Clone::clone)?,
        |f: runtime::EventElapsedTime| unsafe { f(arg0, arg1, arg2) },
    )
}

/// Invoke the declared `cuDeviceGetName` ABI using the retained selected runtime.
///
/// # Safety
/// `arg0` must cover `arg1` writable bytes and `arg2` must identify an initialized driver device.
#[allow(non_snake_case)] // Preserve SDK operation names at the private FFI boundary.
pub unsafe fn invoke_cuDeviceGetName(
    runtime: &CudaRuntime,
    arg0: *mut c_char,
    arg1: c_int,
    arg2: CudaDevice,
) -> Result<(), CudaError> {
    #[cfg(feature = "allocation-census")]
    census::call("cuDeviceGetName", None);
    runtime.driver_call(
        "cuDeviceGetName",
        runtime
            .0
            .api
            .cu_device_get_name
            .as_ref()
            .copied()
            .map_err(Clone::clone)?,
        |f: driver::DriverDeviceGetName| unsafe { f(arg0, arg1, arg2) },
    )
}

/// Invoke the declared `cudaSetDevice` ABI using the retained selected runtime.
#[allow(non_snake_case)] // Preserve SDK operation names at the private FFI boundary.
pub fn invoke_cudaSetDevice(
    runtime: &CudaRuntime,
    arg0: runtime::CudaDevice,
) -> Result<(), CudaError> {
    #[cfg(feature = "allocation-census")]
    census::call("cudaSetDevice", None);
    runtime.call(
        "cudaSetDevice",
        runtime
            .0
            .api
            .cuda_set_device
            .as_ref()
            .copied()
            .map_err(Clone::clone)?,
        |f: runtime::SetDevice| unsafe { f(arg0) },
    )
}

/// Invoke the declared `cudaMemcpy` ABI using the retained selected runtime.
///
/// # Safety
/// Both pointers must cover `arg2` bytes, match the transfer direction `arg3`, and have nonoverlapping ranges.
#[allow(non_snake_case)] // Preserve SDK operation names at the private FFI boundary.
pub unsafe fn invoke_cudaMemcpy(
    runtime: &CudaRuntime,
    arg0: *mut c_void,
    arg1: *const c_void,
    arg2: usize,
    arg3: c_int,
) -> Result<(), CudaError> {
    #[cfg(feature = "allocation-census")]
    census::call("cudaMemcpy", Some(arg3));
    runtime.call(
        "cudaMemcpy",
        runtime
            .0
            .api
            .cuda_memcpy
            .as_ref()
            .copied()
            .map_err(Clone::clone)?,
        |f: runtime::Memcpy| unsafe { f(arg0, arg1, arg2, arg3) },
    )
}

/// Invoke the declared `cudaDeviceSynchronize` ABI using the retained selected runtime.
#[allow(non_snake_case)] // Preserve SDK operation names at the private FFI boundary.
pub fn invoke_cudaDeviceSynchronize(runtime: &CudaRuntime) -> Result<(), CudaError> {
    #[cfg(feature = "allocation-census")]
    census::call("cudaDeviceSynchronize", None);
    #[cfg(feature = "allocation-census")]
    let started = std::time::Instant::now();
    let result = runtime.call(
        "cudaDeviceSynchronize",
        runtime
            .0
            .api
            .cuda_device_synchronize
            .as_ref()
            .copied()
            .map_err(Clone::clone)?,
        |f: runtime::DeviceSynchronize| unsafe { f() },
    );
    #[cfg(feature = "allocation-census")]
    census::completion_time(started.elapsed());
    result
}

/// Invoke the declared `cudaStreamDestroy` ABI using the retained selected runtime.
///
/// # Safety
/// `arg0` must be a live owned stream, released exactly once after its users finish.
#[allow(non_snake_case)] // Preserve SDK operation names at the private FFI boundary.
pub unsafe fn invoke_cudaStreamDestroy(
    runtime: &CudaRuntime,
    arg0: runtime::CudaStream,
) -> Result<(), CudaError> {
    #[cfg(feature = "allocation-census")]
    census::call("cudaStreamDestroy", None);
    runtime.call(
        "cudaStreamDestroy",
        runtime
            .0
            .api
            .cuda_stream_destroy
            .as_ref()
            .copied()
            .map_err(Clone::clone)?,
        |f: runtime::StreamDestroy| unsafe { f(arg0) },
    )
}

/// Invoke the declared `cudaStreamSynchronize` ABI using the retained selected runtime.
///
/// # Safety
/// Any result pointers must address writable storage of the declared pointee type. All handles must belong to the selected device/context and remain live for this call.
#[allow(non_snake_case)] // Preserve SDK operation names at the private FFI boundary.
pub unsafe fn invoke_cudaStreamSynchronize(
    runtime: &CudaRuntime,
    arg0: runtime::CudaStream,
) -> Result<(), CudaError> {
    #[cfg(feature = "allocation-census")]
    census::call("cudaStreamSynchronize", None);
    #[cfg(feature = "allocation-census")]
    let started = std::time::Instant::now();
    let result = runtime.call(
        "cudaStreamSynchronize",
        runtime
            .0
            .api
            .cuda_stream_synchronize
            .as_ref()
            .copied()
            .map_err(Clone::clone)?,
        |f: runtime::StreamSynchronize| unsafe { f(arg0) },
    );
    #[cfg(feature = "allocation-census")]
    census::completion_time(started.elapsed());
    result
}

/// Invoke the declared `cudaEventRecord` ABI using the retained selected runtime.
///
/// # Safety
/// Any result pointers must address writable storage of the declared pointee type. All handles must belong to the selected device/context and remain live for this call.
#[allow(non_snake_case)] // Preserve SDK operation names at the private FFI boundary.
pub unsafe fn invoke_cudaEventRecord(
    runtime: &CudaRuntime,
    arg0: runtime::CudaEvent,
    arg1: runtime::CudaStream,
) -> Result<(), CudaError> {
    #[cfg(feature = "allocation-census")]
    census::call("cudaEventRecord", None);
    runtime.call(
        "cudaEventRecord",
        runtime
            .0
            .api
            .cuda_event_record
            .as_ref()
            .copied()
            .map_err(Clone::clone)?,
        |f: runtime::EventRecord| unsafe { f(arg0, arg1) },
    )
}

/// Invoke the declared `cudaEventDestroy` ABI using the retained selected runtime.
///
/// # Safety
/// `arg0` must be a live owned event, released exactly once after its users finish.
#[allow(non_snake_case)] // Preserve SDK operation names at the private FFI boundary.
pub unsafe fn invoke_cudaEventDestroy(
    runtime: &CudaRuntime,
    arg0: runtime::CudaEvent,
) -> Result<(), CudaError> {
    #[cfg(feature = "allocation-census")]
    census::call("cudaEventDestroy", None);
    runtime.call(
        "cudaEventDestroy",
        runtime
            .0
            .api
            .cuda_event_destroy
            .as_ref()
            .copied()
            .map_err(Clone::clone)?,
        |f: runtime::EventDestroy| unsafe { f(arg0) },
    )
}

/// Invoke the declared `cudaEventSynchronize` ABI using the retained selected runtime.
///
/// # Safety
/// A non-null event must belong to the selected device/context and remain live for this call.
/// A null event is permitted for the SDK-defined invalid-resource-handle diagnostic.
#[allow(non_snake_case)] // Preserve SDK operation names at the private FFI boundary.
pub unsafe fn invoke_cudaEventSynchronize(
    runtime: &CudaRuntime,
    arg0: runtime::CudaEvent,
) -> Result<(), CudaError> {
    #[cfg(feature = "allocation-census")]
    census::call("cudaEventSynchronize", None);
    runtime.call(
        "cudaEventSynchronize",
        runtime
            .0
            .api
            .cuda_event_synchronize
            .as_ref()
            .copied()
            .map_err(Clone::clone)?,
        |f: runtime::EventSynchronize| unsafe { f(arg0) },
    )
}

/// Invoke the declared `cuModuleUnload` ABI using the retained selected runtime.
///
/// # Safety
/// `arg0` must be a live module with no remaining queued kernel users and must be unloaded exactly once.
#[allow(non_snake_case)] // Preserve SDK operation names at the private FFI boundary.
pub unsafe fn invoke_cuModuleUnload(
    runtime: &CudaRuntime,
    arg0: driver::ModuleHandle,
) -> Result<(), CudaError> {
    #[cfg(feature = "allocation-census")]
    census::call("cuModuleUnload", None);
    runtime.driver_call(
        "cuModuleUnload",
        runtime
            .0
            .api
            .cu_module_unload
            .as_ref()
            .copied()
            .map_err(Clone::clone)?,
        |f: driver::DriverModuleUnload| unsafe { f(arg0) },
    )
}

/// Invoke the declared `cuModuleGetFunction` ABI using the retained selected runtime.
///
/// # Safety
/// `arg0` must be writable function-handle storage; `arg1` must be live and `arg2` must address a terminated kernel name.
#[allow(non_snake_case)] // Preserve SDK operation names at the private FFI boundary.
pub unsafe fn invoke_cuModuleGetFunction(
    runtime: &CudaRuntime,
    arg0: *mut driver::KernelHandle,
    arg1: driver::ModuleHandle,
    arg2: *const c_char,
) -> Result<(), CudaError> {
    #[cfg(feature = "allocation-census")]
    census::call("cuModuleGetFunction", None);
    runtime.driver_call(
        "cuModuleGetFunction",
        runtime
            .0
            .api
            .cu_module_get_function
            .as_ref()
            .copied()
            .map_err(Clone::clone)?,
        |f: driver::DriverModuleGetFunction| unsafe { f(arg0, arg1, arg2) },
    )
}

/// Invoke the declared `cudaMemcpyAsync` ABI using the retained selected runtime.
///
/// # Safety
/// Both pointers must cover `arg2` bytes, match direction `arg3`, and have nonoverlapping ranges. Their owners must survive completion on `arg4`.
#[allow(non_snake_case)] // Preserve SDK operation names at the private FFI boundary.
pub unsafe fn invoke_cudaMemcpyAsync(
    runtime: &CudaRuntime,
    arg0: *mut c_void,
    arg1: *const c_void,
    arg2: usize,
    arg3: c_int,
    arg4: runtime::CudaStream,
) -> Result<(), CudaError> {
    #[cfg(feature = "allocation-census")]
    census::call("cudaMemcpyAsync", Some(arg3));
    runtime.call(
        "cudaMemcpyAsync",
        runtime
            .0
            .api
            .cuda_memcpy_async
            .as_ref()
            .copied()
            .map_err(Clone::clone)?,
        |f: runtime::MemcpyAsync| unsafe { f(arg0, arg1, arg2, arg3, arg4) },
    )
}

/// Invoke the declared `cudaStreamWaitEvent` ABI using the retained selected runtime.
///
/// # Safety
/// Any result pointers must address writable storage of the declared pointee type. All handles must belong to the selected device/context and remain live for this call.
#[allow(non_snake_case)] // Preserve SDK operation names at the private FFI boundary.
pub unsafe fn invoke_cudaStreamWaitEvent(
    runtime: &CudaRuntime,
    arg0: runtime::CudaStream,
    arg1: runtime::CudaEvent,
    arg2: u32,
) -> Result<(), CudaError> {
    #[cfg(feature = "allocation-census")]
    census::call("cudaStreamWaitEvent", None);
    runtime.call(
        "cudaStreamWaitEvent",
        runtime
            .0
            .api
            .cuda_stream_wait_event
            .as_ref()
            .copied()
            .map_err(Clone::clone)?,
        |f: runtime::StreamWaitEvent| unsafe { f(arg0, arg1, arg2) },
    )
}

/// Invoke the declared `cuLaunchKernel` ABI using the retained selected runtime.
///
/// # Safety
/// The kernel and stream must belong to the selected context. Argument pointers must match the kernel ABI; referenced buffers must survive stream completion.
#[allow(non_snake_case)] // Preserve SDK operation names at the private FFI boundary.
#[allow(clippy::too_many_arguments)] // Exact SDK ABI; the owner validates its arguments.
pub unsafe fn invoke_cuLaunchKernel(
    runtime: &CudaRuntime,
    kernel: driver::KernelHandle,
    blocks_x: u32,
    blocks_y: u32,
    blocks_z: u32,
    threads_x: u32,
    threads_y: u32,
    threads_z: u32,
    shared_bytes: u32,
    stream: CudaStream,
    parameters: *mut *mut c_void,
    extra: *mut *mut c_void,
) -> Result<(), CudaError> {
    #[cfg(feature = "allocation-census")]
    census::call("cuLaunchKernel", None);
    runtime.driver_call(
        "cuLaunchKernel",
        runtime
            .0
            .api
            .cu_launch_kernel
            .as_ref()
            .copied()
            .map_err(Clone::clone)?,
        |f: driver::DriverModuleLaunchKernel| unsafe {
            f(
                kernel,
                blocks_x,
                blocks_y,
                blocks_z,
                threads_x,
                threads_y,
                threads_z,
                shared_bytes,
                stream,
                parameters,
                extra,
            )
        },
    )
}

/// Invoke the declared `cudaStreamCreateWithPriority` ABI using the retained selected runtime.
///
/// # Safety
/// Any result pointers must address writable storage of the declared pointee type. All handles must belong to the selected device/context and remain live for this call.
#[allow(non_snake_case)] // Preserve SDK operation names at the private FFI boundary.
pub unsafe fn invoke_cudaStreamCreateWithPriority(
    runtime: &CudaRuntime,
    arg0: *mut runtime::CudaStream,
    arg1: u32,
    arg2: c_int,
) -> Result<(), CudaError> {
    #[cfg(feature = "allocation-census")]
    census::call("cudaStreamCreateWithPriority", None);
    runtime.call(
        "cudaStreamCreateWithPriority",
        runtime
            .0
            .api
            .cuda_stream_create_with_priority
            .as_ref()
            .copied()
            .map_err(Clone::clone)?,
        |f: runtime::StreamCreateWithPriority| unsafe { f(arg0, arg1, arg2) },
    )
}

/// Invoke the declared `cudaDeviceGetStreamPriorityRange` ABI using the retained selected runtime.
///
/// # Safety
/// Any result pointers must address writable storage of the declared pointee type. All handles must belong to the selected device/context and remain live for this call.
#[allow(non_snake_case)] // Preserve SDK operation names at the private FFI boundary.
pub unsafe fn invoke_cudaDeviceGetStreamPriorityRange(
    runtime: &CudaRuntime,
    arg0: *mut c_int,
    arg1: *mut c_int,
) -> Result<(), CudaError> {
    #[cfg(feature = "allocation-census")]
    census::call("cudaDeviceGetStreamPriorityRange", None);
    runtime.call(
        "cudaDeviceGetStreamPriorityRange",
        runtime
            .0
            .api
            .cuda_device_get_stream_priority_range
            .as_ref()
            .copied()
            .map_err(Clone::clone)?,
        |f: runtime::DeviceGetStreamPriorityRange| unsafe { f(arg0, arg1) },
    )
}

/// Invoke the declared `cudaStreamGetFlags` ABI using the retained selected runtime.
///
/// # Safety
/// Any result pointers must address writable storage of the declared pointee type. All handles must belong to the selected device/context and remain live for this call.
#[allow(non_snake_case)] // Preserve SDK operation names at the private FFI boundary.
pub unsafe fn invoke_cudaStreamGetFlags(
    runtime: &CudaRuntime,
    arg0: runtime::CudaStream,
    arg1: *mut u32,
) -> Result<(), CudaError> {
    #[cfg(feature = "allocation-census")]
    census::call("cudaStreamGetFlags", None);
    runtime.call(
        "cudaStreamGetFlags",
        runtime
            .0
            .api
            .cuda_stream_get_flags
            .as_ref()
            .copied()
            .map_err(Clone::clone)?,
        |f: runtime::StreamGetFlags| unsafe { f(arg0, arg1) },
    )
}

/// Invoke the declared `cudaStreamGetPriority` ABI using the retained selected runtime.
///
/// # Safety
/// Any result pointers must address writable storage of the declared pointee type. All handles must belong to the selected device/context and remain live for this call.
#[allow(non_snake_case)] // Preserve SDK operation names at the private FFI boundary.
pub unsafe fn invoke_cudaStreamGetPriority(
    runtime: &CudaRuntime,
    arg0: runtime::CudaStream,
    arg1: *mut c_int,
) -> Result<(), CudaError> {
    #[cfg(feature = "allocation-census")]
    census::call("cudaStreamGetPriority", None);
    runtime.call(
        "cudaStreamGetPriority",
        runtime
            .0
            .api
            .cuda_stream_get_priority
            .as_ref()
            .copied()
            .map_err(Clone::clone)?,
        |f: runtime::StreamGetPriority| unsafe { f(arg0, arg1) },
    )
}

/// Invoke the declared `cudaStreamQuery` ABI using the retained selected runtime.
///
/// # Safety
/// Any result pointers must address writable storage of the declared pointee type. All handles must belong to the selected device/context and remain live for this call.
#[allow(non_snake_case)] // Preserve SDK operation names at the private FFI boundary.
pub unsafe fn invoke_cudaStreamQuery(
    runtime: &CudaRuntime,
    arg0: runtime::CudaStream,
) -> Result<crate::control::CudaReadiness, CudaError> {
    #[cfg(feature = "allocation-census")]
    census::call("cudaStreamQuery", None);
    runtime.query_readiness(
        "cudaStreamQuery",
        runtime
            .0
            .api
            .cuda_stream_query
            .as_ref()
            .copied()
            .map_err(Clone::clone)?,
        |f: runtime::StreamQuery| unsafe { f(arg0) },
    )
}

/// Invoke the declared `cudaEventQuery` ABI using the retained selected runtime.
///
/// # Safety
/// Any result pointers must address writable storage of the declared pointee type. All handles must belong to the selected device/context and remain live for this call.
#[allow(non_snake_case)] // Preserve SDK operation names at the private FFI boundary.
pub unsafe fn invoke_cudaEventQuery(
    runtime: &CudaRuntime,
    arg0: runtime::CudaEvent,
) -> Result<crate::control::CudaReadiness, CudaError> {
    #[cfg(feature = "allocation-census")]
    census::call("cudaEventQuery", None);
    runtime.query_readiness(
        "cudaEventQuery",
        runtime
            .0
            .api
            .cuda_event_query
            .as_ref()
            .copied()
            .map_err(Clone::clone)?,
        |f: runtime::EventQuery| unsafe { f(arg0) },
    )
}

/// Invoke the declared `cuDeviceGetUuid_v2` ABI using the retained selected runtime.
///
/// # Safety
/// Any result pointers must address writable storage of the declared pointee type. All handles must belong to the selected device/context and remain live for this call.
#[allow(non_snake_case)] // Preserve SDK operation names at the private FFI boundary.
pub unsafe fn invoke_cuDeviceGetUuid_v2(
    runtime: &CudaRuntime,
    arg0: *mut driver::DriverUuid,
    arg1: CudaDevice,
) -> Result<(), CudaError> {
    #[cfg(feature = "allocation-census")]
    census::call("cuDeviceGetUuid_v2", None);
    runtime.driver_call(
        "cuDeviceGetUuid_v2",
        runtime
            .0
            .api
            .cu_device_get_uuid_v2
            .as_ref()
            .copied()
            .map_err(Clone::clone)?,
        |f: driver::DriverDeviceGetUuid| unsafe { f(arg0, arg1) },
    )
}

/// Invoke the declared `cuDeviceGetAttribute` ABI using the retained selected runtime.
///
/// # Safety
/// Any result pointers must address writable storage of the declared pointee type. All handles must belong to the selected device/context and remain live for this call.
#[allow(non_snake_case)] // Preserve SDK operation names at the private FFI boundary.
pub unsafe fn invoke_cuDeviceGetAttribute(
    runtime: &CudaRuntime,
    arg0: *mut c_int,
    arg1: c_int,
    arg2: CudaDevice,
) -> Result<(), CudaError> {
    #[cfg(feature = "allocation-census")]
    census::call("cuDeviceGetAttribute", None);
    runtime.driver_call(
        "cuDeviceGetAttribute",
        runtime
            .0
            .api
            .cu_device_get_attribute
            .as_ref()
            .copied()
            .map_err(Clone::clone)?,
        |f: driver::DriverDeviceGetAttribute| unsafe { f(arg0, arg1, arg2) },
    )
}

/// Invoke the declared `cudaFreeHost` ABI using the retained selected runtime.
///
/// # Safety
/// `arg0` must be a live pinned allocation, with no remaining queued users. It must be released exactly once.
#[allow(non_snake_case)] // Preserve SDK operation names at the private FFI boundary.
pub unsafe fn invoke_cudaFreeHost(
    runtime: &CudaRuntime,
    arg0: *mut c_void,
) -> Result<(), CudaError> {
    #[cfg(feature = "allocation-census")]
    census::call("cudaFreeHost", None);
    runtime.call(
        "cudaFreeHost",
        runtime
            .0
            .api
            .cuda_free_host
            .as_ref()
            .copied()
            .map_err(Clone::clone)?,
        |f: runtime::FreeHost| unsafe { f(arg0) },
    )
}

/// Invoke the declared `cudaMallocHost` ABI using the retained selected runtime.
///
/// # Safety
/// Any result pointers must address writable storage of the declared pointee type. All handles must belong to the selected device/context and remain live for this call.
#[allow(non_snake_case)] // Preserve SDK operation names at the private FFI boundary.
pub unsafe fn invoke_cudaMallocHost(
    runtime: &CudaRuntime,
    arg0: *mut *mut c_void,
    arg1: usize,
) -> Result<(), CudaError> {
    #[cfg(feature = "allocation-census")]
    census::call("cudaMallocHost", None);
    runtime.call(
        "cudaMallocHost",
        runtime
            .0
            .api
            .cuda_malloc_host
            .as_ref()
            .copied()
            .map_err(Clone::clone)?,
        |f: runtime::MallocHost| unsafe { f(arg0, arg1) },
    )
}

/// Invoke the declared `cudaStreamBeginCapture` ABI using the retained selected runtime.
///
/// # Safety
/// The live stream must be eligible for capture in `arg1` mode; its capture must be completed or aborted before reuse.
#[allow(non_snake_case)] // Preserve SDK operation names at the private FFI boundary.
pub unsafe fn invoke_cudaStreamBeginCapture(
    runtime: &CudaRuntime,
    arg0: runtime::CudaStream,
    arg1: c_int,
) -> Result<(), CudaError> {
    #[cfg(feature = "allocation-census")]
    census::call("cudaStreamBeginCapture", None);
    runtime.call(
        "cudaStreamBeginCapture",
        runtime
            .0
            .api
            .cuda_stream_begin_capture
            .as_ref()
            .copied()
            .map_err(Clone::clone)?,
        |f: runtime::StreamBeginCapture| unsafe { f(arg0, arg1) },
    )
}

/// Invoke the declared `cudaStreamEndCapture` ABI using the retained selected runtime.
///
/// # Safety
/// The live stream must have an active capture and `arg1` must be writable graph-handle storage.
#[allow(non_snake_case)] // Preserve SDK operation names at the private FFI boundary.
pub unsafe fn invoke_cudaStreamEndCapture(
    runtime: &CudaRuntime,
    arg0: runtime::CudaStream,
    arg1: *mut runtime::CudaGraph,
) -> Result<(), CudaError> {
    #[cfg(feature = "allocation-census")]
    census::call("cudaStreamEndCapture", None);
    runtime.call(
        "cudaStreamEndCapture",
        runtime
            .0
            .api
            .cuda_stream_end_capture
            .as_ref()
            .copied()
            .map_err(Clone::clone)?,
        |f: runtime::StreamEndCapture| unsafe { f(arg0, arg1) },
    )
}

/// Invoke the declared `cudaGraphInstantiateWithFlags` ABI using the retained selected runtime.
///
/// # Safety
/// `arg0` must be writable executable-handle storage and `arg1` must be a live graph.
#[allow(non_snake_case)] // Preserve SDK operation names at the private FFI boundary.
pub unsafe fn invoke_cudaGraphInstantiateWithFlags(
    runtime: &CudaRuntime,
    arg0: *mut runtime::CudaGraphExec,
    arg1: runtime::CudaGraph,
    arg2: u64,
) -> Result<(), CudaError> {
    #[cfg(feature = "allocation-census")]
    census::call("cudaGraphInstantiateWithFlags", None);
    runtime.call(
        "cudaGraphInstantiateWithFlags",
        runtime
            .0
            .api
            .cuda_graph_instantiate_with_flags
            .as_ref()
            .copied()
            .map_err(Clone::clone)?,
        |f: runtime::GraphInstantiateWithFlags| unsafe { f(arg0, arg1, arg2) },
    )
}

/// Invoke the declared `cudaGraphDestroy` ABI using the retained selected runtime.
///
/// # Safety
/// `arg0` must be a live owned graph, released exactly once.
#[allow(non_snake_case)] // Preserve SDK operation names at the private FFI boundary.
pub unsafe fn invoke_cudaGraphDestroy(
    runtime: &CudaRuntime,
    arg0: runtime::CudaGraph,
) -> Result<(), CudaError> {
    #[cfg(feature = "allocation-census")]
    census::call("cudaGraphDestroy", None);
    runtime.call(
        "cudaGraphDestroy",
        runtime
            .0
            .api
            .cuda_graph_destroy
            .as_ref()
            .copied()
            .map_err(Clone::clone)?,
        |f: runtime::GraphDestroy| unsafe { f(arg0) },
    )
}

/// Invoke the declared `cudaGraphLaunch` ABI using the retained selected runtime.
///
/// # Safety
/// The executable and stream must be live. Captured buffers and external owners must survive launch completion.
#[allow(non_snake_case)] // Preserve SDK operation names at the private FFI boundary.
pub unsafe fn invoke_cudaGraphLaunch(
    runtime: &CudaRuntime,
    arg0: runtime::CudaGraphExec,
    arg1: runtime::CudaStream,
) -> Result<(), CudaError> {
    #[cfg(feature = "allocation-census")]
    census::call("cudaGraphLaunch", None);
    runtime.call(
        "cudaGraphLaunch",
        runtime
            .0
            .api
            .cuda_graph_launch
            .as_ref()
            .copied()
            .map_err(Clone::clone)?,
        |f: runtime::GraphLaunch| unsafe { f(arg0, arg1) },
    )
}

/// Invoke the declared `cudaMemsetAsync` ABI using the retained selected runtime.
///
/// # Safety
/// `arg0` must cover `arg2` writable device bytes and remain live until completion on `arg3`.
#[allow(non_snake_case)] // Preserve SDK operation names at the private FFI boundary.
pub unsafe fn invoke_cudaMemsetAsync(
    runtime: &CudaRuntime,
    arg0: *mut c_void,
    arg1: c_int,
    arg2: usize,
    arg3: runtime::CudaStream,
) -> Result<(), CudaError> {
    #[cfg(feature = "allocation-census")]
    census::call("cudaMemsetAsync", None);
    runtime.call(
        "cudaMemsetAsync",
        runtime
            .0
            .api
            .cuda_memset_async
            .as_ref()
            .copied()
            .map_err(Clone::clone)?,
        |f: runtime::MemsetAsync| unsafe { f(arg0, arg1, arg2, arg3) },
    )
}

/// Invoke the declared `cudaGraphExecDestroy` ABI using the retained selected runtime.
///
/// # Safety
/// `arg0` must be a live executable graph, released exactly once after all launches finish.
#[allow(non_snake_case)] // Preserve SDK operation names at the private FFI boundary.
pub unsafe fn invoke_cudaGraphExecDestroy(
    runtime: &CudaRuntime,
    arg0: runtime::CudaGraphExec,
) -> Result<(), CudaError> {
    #[cfg(feature = "allocation-census")]
    census::call("cudaGraphExecDestroy", None);
    runtime.call(
        "cudaGraphExecDestroy",
        runtime
            .0
            .api
            .cuda_graph_exec_destroy
            .as_ref()
            .copied()
            .map_err(Clone::clone)?,
        |f: runtime::GraphExecDestroy| unsafe { f(arg0) },
    )
}

thread_local! {
    // Valid only during private backend execution with no caller callbacks. Every new scope
    // selects explicitly, so foreign CUDA clients can change the current context between calls.
    static SELECTED_RUNTIME_SCOPE: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

pub struct RuntimeScope {
    _runtime: CudaRuntime,
    _selection: SelectedRuntimeScope,
}

struct SelectedRuntimeScope {
    _thread: std::marker::PhantomData<std::rc::Rc<()>>,
}

impl SelectedRuntimeScope {
    fn enter(identity: usize) -> Self {
        SELECTED_RUNTIME_SCOPE.set(identity);
        Self {
            _thread: std::marker::PhantomData,
        }
    }
}

impl Drop for SelectedRuntimeScope {
    fn drop(&mut self) {
        // A nested different runtime may have changed the native context. Clearing instead of
        // restoring an assumed selection makes subsequent outer operations select explicitly.
        SELECTED_RUNTIME_SCOPE.set(0);
    }
}

impl CudaRuntime {
    /// Enter a private synchronous call boundary. No foreign/user callback may run in the scope.
    pub(crate) fn enter_private_scope(&self) -> Result<RuntimeScope, CudaError> {
        self.cuda_set_device(self.0.ordinal)?;
        Ok(RuntimeScope {
            _runtime: self.clone(),
            _selection: SelectedRuntimeScope::enter(std::sync::Arc::as_ptr(&self.0) as usize),
        })
    }

    fn private_scope_selected(&self) -> bool {
        SELECTED_RUNTIME_SCOPE.get() == std::sync::Arc::as_ptr(&self.0) as usize
    }

    fn driver_call<T: Copy>(
        &self,
        symbol: &'static str,
        function: T,
        invoke: impl FnOnce(T) -> CudaResult,
    ) -> Result<(), CudaError> {
        if !self.private_scope_selected() {
            self.cuda_set_device(self.0.ordinal)?;
        }
        let status = invoke(function);
        if status == CUDA_SUCCESS {
            Ok(())
        } else {
            Err(self.driver_error(symbol, status))
        }
    }
    fn driver_error(&self, operation: &'static str, code: CudaResult) -> CudaError {
        let mut message = ptr::null();
        let detail = self
            .0
            .api
            .cu_get_error_string
            .as_ref()
            .ok()
            .filter(|f| unsafe { f(code, &raw mut message) } == CUDA_SUCCESS)
            .and_then(|_| {
                (!message.is_null()).then(|| {
                    unsafe { CStr::from_ptr(message) }
                        .to_string_lossy()
                        .into_owned()
                })
            });
        CudaError::Runtime {
            operation,
            code,
            detail,
        }
    }
    fn call<T: Copy>(
        &self,
        symbol: &'static str,
        function: T,
        invoke: impl FnOnce(T) -> CudaResult,
    ) -> Result<(), CudaError> {
        if symbol == "cudaSetDevice" {
            // Explicit selection, including a nested operation on another runtime, invalidates
            // the enclosing scope before touching native state, even when selection fails.
            SELECTED_RUNTIME_SCOPE.set(0);
        }
        if symbol != "cudaSetDevice" && !self.private_scope_selected() {
            // CUDA's current device is thread-local, so select this runtime's device before each
            // operation. This keeps cloned handles valid when used from another host thread.
            let setter = self
                .0
                .api
                .cuda_set_device
                .as_ref()
                .copied()
                .map_err(Clone::clone)?;
            #[cfg(feature = "allocation-census")]
            census::call("cudaSetDevice", None);
            SELECTED_RUNTIME_SCOPE.set(0);
            let status = unsafe { setter(self.0.ordinal) };
            if status != CUDA_SUCCESS {
                return Err(self.error("cudaSetDevice", status));
            }
        }
        let status = invoke(function);
        if status == CUDA_SUCCESS {
            Ok(())
        } else {
            Err(self.error(symbol, status))
        }
    }
    #[allow(clippy::redundant_pub_crate)] // Keep the inherent method private to this backend crate.
    pub(super) fn error(&self, operation: &'static str, code: CudaResult) -> CudaError {
        // Error-string lookup is optional; preserve numeric status even if the symbol is absent.
        let detail = self
            .0
            .api
            .cuda_get_error_string
            .as_ref()
            .ok()
            .map(|f| unsafe { f(code) })
            .filter(|p| !p.is_null())
            .map(|p| unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned());
        CudaError::Runtime {
            operation,
            code,
            detail,
        }
    }
    fn query_readiness<T: Copy>(
        &self,
        operation: &'static str,
        function: T,
        invoke: impl FnOnce(T) -> CudaResult,
    ) -> Result<CudaReadiness, CudaError> {
        if !self.private_scope_selected() {
            self.cuda_set_device(self.0.ordinal)?;
        }
        let status = invoke(function);
        classify_readiness(status).ok_or_else(|| self.error(operation, status))
    }
}

/// Query PCI location without depending on the CUDA device-property structure ABI.
pub fn query_pci_bus_id(library: &Library, ordinal: c_int) -> Option<String> {
    let function = unsafe {
        crate::ffi::symbol::<DriverDeviceGetPciBusId>(
            library,
            driver_symbol("cuDeviceGetPCIBusId").unwrap_or(&[]),
        )
    }
    .ok()?;
    let mut buffer = [0_i8; 64];
    let capacity = c_int::try_from(buffer.len()).expect("fixed PCI bus buffer fits c_int");
    let status = unsafe { function(buffer.as_mut_ptr(), capacity, ordinal) };
    if status != CUDA_SUCCESS {
        return None;
    }
    let value = bounded_device_name(&buffer);
    (!value.is_empty()).then_some(value)
}
pub fn driver_architecture(library: &Library, device: CudaDevice) -> Option<String> {
    let get_attribute = unsafe {
        crate::ffi::symbol::<DriverDeviceGetAttribute>(
            library,
            driver_symbol("cuDeviceGetAttribute").unwrap_or(&[]),
        )
    }
    .ok()?;
    let mut major = 0;
    let mut minor = 0;
    if unsafe {
        get_attribute(
            &raw mut major,
            DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
            device,
        )
    } != CUDA_SUCCESS
        || unsafe {
            get_attribute(
                &raw mut minor,
                DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                device,
            )
        } != CUDA_SUCCESS
        || major <= 0
        || minor < 0
    {
        return None;
    }
    Some(format!("sm_{major}{minor}"))
}
pub fn enumerate_driver_devices(
    driver: &Library,
    count: c_int,
    get_device: &DriverGetDevice,
    get_name: &DriverDeviceGetName,
    total_mem: &DriverDeviceTotalMem,
) -> Result<Vec<CudaDeviceInfo>, CudaError> {
    let mut devices = Vec::new();
    for index in 0..count.max(0) {
        let mut device = 0;
        let status = unsafe { get_device(ptr::from_mut(&mut device), index) };
        if status != CUDA_SUCCESS {
            return Err(raw_driver_error(driver, "cuDeviceGet", status));
        }
        let mut name = [0_i8; 256];
        let status = unsafe { get_name(name.as_mut_ptr(), 256, device) };
        if status != CUDA_SUCCESS {
            return Err(raw_driver_error(driver, "cuDeviceGetName", status));
        }
        let name = bounded_device_name(&name);
        let mut total = 0_usize;
        let status = unsafe { total_mem(&raw mut total, device) };
        if status != CUDA_SUCCESS {
            return Err(raw_driver_error(driver, "cuDeviceTotalMem_v2", status));
        }
        let pci_bus_id = query_pci_bus_id(driver, device);
        let architecture = driver_architecture(driver, device);
        devices.push(CudaDeviceInfo {
            index,
            name,
            vendor: "NVIDIA".into(),
            architecture,
            generation: None,
            pci_bus_id,
            total_memory: total as u64,
        });
    }
    Ok(devices)
}
pub fn raw_cuda_error(library: &Library, operation: &'static str, code: CudaResult) -> CudaError {
    let detail = unsafe {
        crate::ffi::symbol::<GetErrorString>(
            library,
            runtime_symbol("cudaGetErrorString").unwrap_or(&[]),
        )
    }
    .ok()
    .map(|function| unsafe { function(code) })
    .filter(|pointer| !pointer.is_null())
    .map(|pointer| {
        unsafe { CStr::from_ptr(pointer) }
            .to_string_lossy()
            .into_owned()
    });
    CudaError::Runtime {
        operation,
        code,
        detail,
    }
}
pub fn raw_driver_error(library: &Library, operation: &'static str, code: CudaResult) -> CudaError {
    let mut message = ptr::null();
    let detail = unsafe {
        crate::ffi::symbol::<DriverGetErrorString>(
            library,
            driver_symbol("cuGetErrorString").unwrap_or(&[]),
        )
    }
    .ok()
    .filter(|function| unsafe { function(code, &raw mut message) } == CUDA_SUCCESS)
    .and_then(|_| {
        (!message.is_null()).then(|| {
            unsafe { CStr::from_ptr(message) }
                .to_string_lossy()
                .into_owned()
        })
    });
    CudaError::Runtime {
        operation,
        code,
        detail,
    }
}

#[rustfmt::skip]
use crate::{
    CudaDeviceInfo,
    bounded_device_name,
};

#[rustfmt::skip]
use crate::ffi::{
    driver::{
        DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
        DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
        DriverDeviceGetAttribute,
        DriverDeviceGetName,
        DriverDeviceGetPciBusId,
        DriverDeviceTotalMem,
        DriverGetDevice,
        DriverGetErrorString,
        DriverInit,
    },
    runtime::{
        CUDA_SUCCESS,
        CudaDevice,
        CudaResult,
        CudaStream,
        GetErrorString,
    },
};

#[rustfmt::skip]
use std::{
    ffi::CStr,
    ptr,
};

#[rustfmt::skip]
use crate::control::{
    CudaReadiness,
    classify_readiness,
};

use crate::codegen::rtc::CudaRtcError;
use std::ffi::CString;
pub fn device_compute_capability(ordinal: c_int) -> Result<(i32, i32), CudaRtcError> {
    let library =
        rtc_load_library(&["libcuda.so.1", "libcuda.so"]).map_err(CudaRtcError::Library)?;
    let init = rtc_symbol::<DriverInit>(&library, "cuInit", nvrtc::CU_INIT)?;
    let get_device = rtc_symbol::<DriverGetDevice>(&library, "cuDeviceGet", nvrtc::CU_DEVICE_GET)?;
    let get_attribute = rtc_symbol::<DriverDeviceGetAttribute>(
        &library,
        "cuDeviceGetAttribute",
        nvrtc::CU_DEVICE_GET_ATTRIBUTE,
    )?;
    rtc_driver_check("cuInit", unsafe { init(0) })?;
    let mut device = 0;
    rtc_driver_check("cuDeviceGet", unsafe {
        get_device(&raw mut device, ordinal)
    })?;
    let mut major = 0;
    let mut minor = 0;
    rtc_driver_check("cuDeviceGetAttribute(compute capability major)", unsafe {
        get_attribute(
            &raw mut major,
            DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
            device,
        )
    })?;
    rtc_driver_check("cuDeviceGetAttribute(compute capability minor)", unsafe {
        get_attribute(
            &raw mut minor,
            DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
            device,
        )
    })?;
    if !(1..=12).contains(&major) || !(0..=9).contains(&minor) {
        return Err(CudaRtcError::InvalidComputeCapability { major, minor });
    }
    Ok((major, minor))
}

pub fn compile_for_architecture(
    source: &CStr,
    architecture: &str,
) -> Result<Vec<u8>, CudaRtcError> {
    let library = rtc_load_nvrtc()?;
    let create = rtc_symbol::<nvrtc::CreateProgram>(
        &library,
        "nvrtcCreateProgram",
        nvrtc::NVRTC_CREATE_PROGRAM,
    )?;
    let destroy = rtc_symbol::<nvrtc::DestroyProgram>(
        &library,
        "nvrtcDestroyProgram",
        nvrtc::NVRTC_DESTROY_PROGRAM,
    )?;
    let compile = rtc_symbol::<nvrtc::CompileProgram>(
        &library,
        "nvrtcCompileProgram",
        nvrtc::NVRTC_COMPILE_PROGRAM,
    )?;
    let log_size = rtc_symbol::<nvrtc::GetProgramLogSize>(
        &library,
        "nvrtcGetProgramLogSize",
        nvrtc::NVRTC_GET_PROGRAM_LOG_SIZE,
    )?;
    let get_log = rtc_symbol::<nvrtc::GetProgramLog>(
        &library,
        "nvrtcGetProgramLog",
        nvrtc::NVRTC_GET_PROGRAM_LOG,
    )?;
    let ptx_size =
        rtc_symbol::<nvrtc::GetPtxSize>(&library, "nvrtcGetPTXSize", nvrtc::NVRTC_GET_PTX_SIZE)?;
    let get_ptx = rtc_symbol::<nvrtc::GetPtx>(&library, "nvrtcGetPTX", nvrtc::NVRTC_GET_PTX)?;
    let error_string = rtc_symbol::<nvrtc::GetErrorString>(
        &library,
        "nvrtcGetErrorString",
        nvrtc::NVRTC_GET_ERROR_STRING,
    )?;
    let name = c"fusion-kernel.cu";
    let options = [
        format!("--gpu-architecture={architecture}"),
        "--fmad=false".to_owned(),
        "--ftz=false".to_owned(),
        "--prec-div=true".to_owned(),
        "--prec-sqrt=true".to_owned(),
    ];
    let options = options
        .iter()
        .map(|option| CString::new(option.as_str()).expect("compiler option has no NUL"))
        .collect::<Vec<_>>();
    let option_ptrs = options
        .iter()
        .map(|option| option.as_ptr())
        .collect::<Vec<_>>();
    let mut program = ptr::null_mut();
    let status = unsafe {
        create(
            &raw mut program,
            source.as_ptr(),
            name.as_ptr(),
            0,
            ptr::null(),
            ptr::null(),
        )
    };
    rtc_check("nvrtcCreateProgram", status, error_string)?;
    let result = rtc_compile_program(
        program,
        &option_ptrs,
        compile,
        log_size,
        get_log,
        ptx_size,
        get_ptx,
        error_string,
    );
    let destroy_status = unsafe { destroy(&raw mut program) };
    match result {
        Ok(ptx) => {
            rtc_check("nvrtcDestroyProgram", destroy_status, error_string)?;
            Ok(ptx)
        }
        Err(error) => Err(error),
    }
}

#[allow(clippy::too_many_arguments)]
fn rtc_compile_program(
    program: nvrtc::Program,
    options: &[*const c_char],
    compile: nvrtc::CompileProgram,
    log_size: nvrtc::GetProgramLogSize,
    get_log: nvrtc::GetProgramLog,
    ptx_size: nvrtc::GetPtxSize,
    get_ptx: nvrtc::GetPtx,
    error_string: nvrtc::GetErrorString,
) -> Result<Vec<u8>, CudaRtcError> {
    let status = unsafe {
        compile(
            program,
            c_int::try_from(options.len()).expect("fixed compile option count fits c_int"),
            options.as_ptr(),
        )
    };
    if status != 0 {
        let mut size = 0;
        let _ = unsafe { log_size(program, &raw mut size) };
        let mut log = vec![0_u8; size.max(1)];
        let _ = unsafe { get_log(program, log.as_mut_ptr().cast()) };
        let log = CStr::from_bytes_until_nul(&log).map_or_else(
            |_| String::from_utf8_lossy(&log).into_owned(),
            |s| s.to_string_lossy().into_owned(),
        );
        return Err(CudaRtcError::Compilation { log });
    }
    let mut size = 0;
    rtc_check(
        "nvrtcGetPTXSize",
        unsafe { ptx_size(program, &raw mut size) },
        error_string,
    )?;
    if size <= 1 {
        return Err(CudaRtcError::EmptyPtx);
    }
    let mut ptx = vec![0_u8; size];
    rtc_check(
        "nvrtcGetPTX",
        unsafe { get_ptx(program, ptx.as_mut_ptr().cast()) },
        error_string,
    )?;
    // The driver module loader consumes the terminating NUL as part of the PTX image.
    Ok(ptx)
}

fn rtc_load_nvrtc() -> Result<Library, CudaRtcError> {
    let mut candidates = Vec::new();
    if let Some(path) = std::env::var_os("NVRTC_LIBRARY") {
        candidates.push(path);
    }
    for variable in ["CUDA_HOME", "CUDA_PATH"] {
        if let Some(root) = std::env::var_os(variable) {
            let root = std::path::PathBuf::from(root);
            candidates.push(root.join("lib64/libnvrtc.so").into_os_string());
            candidates.push(
                root.join("targets/x86_64-linux/lib/libnvrtc.so")
                    .into_os_string(),
            );
        }
    }
    candidates.extend(
        [
            "libnvrtc.so",
            "libnvrtc.so.13",
            "libnvrtc.so.12",
            "/usr/local/cuda/lib64/libnvrtc.so",
        ]
        .into_iter()
        .map(Into::into),
    );
    let mut details = Vec::new();
    for candidate in candidates {
        match unsafe { crate::ffi::load_uncached_library(&candidate) } {
            Ok(library) => return Ok(library),
            Err(error) => details.push(format!("{}: {error}", candidate.to_string_lossy())),
        }
    }
    Err(CudaRtcError::Library(details.join("; ")))
}

fn rtc_load_library(candidates: &[&str]) -> Result<Library, String> {
    let mut details = Vec::new();
    for candidate in candidates {
        match unsafe { crate::ffi::load_uncached_library(candidate) } {
            Ok(library) => return Ok(library),
            Err(error) => details.push(format!("{candidate}: {error}")),
        }
    }
    Err(details.join("; "))
}

/// Check that NVRTC and all symbols needed for PTX compilation can be loaded.
///
/// # Errors
///
/// Returns a typed library or symbol error when NVRTC cannot be used.
pub fn nvrtc_available() -> Result<(), CudaRtcError> {
    let library = rtc_load_nvrtc()?;
    let _ = rtc_symbol::<nvrtc::CreateProgram>(
        &library,
        "nvrtcCreateProgram",
        nvrtc::NVRTC_CREATE_PROGRAM,
    )?;
    let _ = rtc_symbol::<nvrtc::DestroyProgram>(
        &library,
        "nvrtcDestroyProgram",
        nvrtc::NVRTC_DESTROY_PROGRAM,
    )?;
    let _ = rtc_symbol::<nvrtc::CompileProgram>(
        &library,
        "nvrtcCompileProgram",
        nvrtc::NVRTC_COMPILE_PROGRAM,
    )?;
    let _ = rtc_symbol::<nvrtc::GetProgramLogSize>(
        &library,
        "nvrtcGetProgramLogSize",
        nvrtc::NVRTC_GET_PROGRAM_LOG_SIZE,
    )?;
    let _ = rtc_symbol::<nvrtc::GetProgramLog>(
        &library,
        "nvrtcGetProgramLog",
        nvrtc::NVRTC_GET_PROGRAM_LOG,
    )?;
    let _ =
        rtc_symbol::<nvrtc::GetPtxSize>(&library, "nvrtcGetPTXSize", nvrtc::NVRTC_GET_PTX_SIZE)?;
    let _ = rtc_symbol::<nvrtc::GetPtx>(&library, "nvrtcGetPTX", nvrtc::NVRTC_GET_PTX)?;
    let _ = rtc_symbol::<nvrtc::GetErrorString>(
        &library,
        "nvrtcGetErrorString",
        nvrtc::NVRTC_GET_ERROR_STRING,
    )?;
    Ok(())
}

fn rtc_symbol<T: Copy>(
    library: &Library,
    name: &'static str,
    symbol: &'static [u8],
) -> Result<T, CudaRtcError> {
    unsafe { crate::ffi::symbol::<T>(library, symbol) }
        .map(|symbol| *symbol)
        .map_err(|error| CudaRtcError::MissingSymbol {
            symbol: name,
            detail: error.to_string(),
        })
}

fn rtc_driver_check(operation: &'static str, code: nvrtc::ResultCode) -> Result<(), CudaRtcError> {
    if code == 0 {
        return Ok(());
    }
    Err(CudaRtcError::Driver {
        operation,
        code,
        detail: format!("driver error code {code}"),
    })
}

fn rtc_check(
    operation: &'static str,
    code: nvrtc::ResultCode,
    error_string: nvrtc::GetErrorString,
) -> Result<(), CudaRtcError> {
    if code == 0 {
        return Ok(());
    }
    let detail = unsafe {
        let pointer = error_string(code);
        if pointer.is_null() {
            "unknown NVRTC error".into()
        } else {
            CStr::from_ptr(pointer).to_string_lossy().into_owned()
        }
    };
    Err(CudaRtcError::Api {
        operation,
        code,
        detail,
    })
}

/// Resolve the fixed `cudaGetDeviceCount` entrypoint from its retained SDK library.
pub fn resolve_cuda_get_device_count(
    library: &Library,
) -> Result<Symbol<'_, runtime::GetDeviceCount>, libloading::Error> {
    unsafe {
        symbol::<runtime::GetDeviceCount>(
            library,
            runtime_symbol("cudaGetDeviceCount").unwrap_or(&[]),
        )
    }
}

/// Resolve the fixed `cuInit` entrypoint from its retained SDK library.
pub fn resolve_cu_init(
    library: &Library,
) -> Result<Symbol<'_, driver::DriverInit>, libloading::Error> {
    unsafe { symbol::<driver::DriverInit>(library, driver_symbol("cuInit").unwrap_or(&[])) }
}

/// Resolve the fixed `cuDeviceGet` entrypoint from its retained SDK library.
pub fn resolve_cu_device_get(
    library: &Library,
) -> Result<Symbol<'_, driver::DriverGetDevice>, libloading::Error> {
    unsafe {
        symbol::<driver::DriverGetDevice>(library, driver_symbol("cuDeviceGet").unwrap_or(&[]))
    }
}

/// Resolve the fixed `cuDeviceGetName` entrypoint from its retained SDK library.
pub fn resolve_cu_device_get_name(
    library: &Library,
) -> Result<Symbol<'_, driver::DriverDeviceGetName>, libloading::Error> {
    unsafe {
        symbol::<driver::DriverDeviceGetName>(
            library,
            driver_symbol("cuDeviceGetName").unwrap_or(&[]),
        )
    }
}

/// Resolve the fixed `cuDeviceTotalMem_v2` entrypoint from its retained SDK library.
pub fn resolve_cu_device_total_mem_v2(
    library: &Library,
) -> Result<Symbol<'_, driver::DriverDeviceTotalMem>, libloading::Error> {
    unsafe {
        symbol::<driver::DriverDeviceTotalMem>(
            library,
            driver_symbol("cuDeviceTotalMem_v2").unwrap_or(&[]),
        )
    }
}

/// Resolve the fixed `cublasDestroy_v2` entrypoint from its retained SDK library.
pub fn resolve_cublas_destroy_v2(
    library: &Library,
) -> Result<Symbol<'_, cublas::DestroyHandle>, libloading::Error> {
    unsafe { symbol::<cublas::DestroyHandle>(library, cublas::SYM_CUBLAS_DESTROY_V2) }
}

/// Resolve the fixed `cublasDgemm_v2` entrypoint from its retained SDK library.
pub fn resolve_cublas_dgemm_v2(
    library: &Library,
) -> Result<Symbol<'_, cublas::Dgemm>, libloading::Error> {
    unsafe { symbol::<cublas::Dgemm>(library, cublas::SYM_CUBLAS_DGEMM_V2) }
}

/// Resolve the fixed `cublasCreate_v2` entrypoint from its retained SDK library.
pub fn resolve_cublas_create_v2(
    library: &Library,
) -> Result<Symbol<'_, cublas::CreateHandle>, libloading::Error> {
    unsafe { symbol::<cublas::CreateHandle>(library, cublas::SYM_CUBLAS_CREATE_V2) }
}

/// Resolve the fixed `cublasSgemm_v2` entrypoint from its retained SDK library.
pub fn resolve_cublas_sgemm_v2(
    library: &Library,
) -> Result<Symbol<'_, cublas::Sgemm>, libloading::Error> {
    unsafe { symbol::<cublas::Sgemm>(library, cublas::SYM_CUBLAS_SGEMM_V2) }
}

/// Resolve the fixed `cublasSdot_v2` entrypoint from its retained SDK library.
pub fn resolve_cublas_sdot_v2(
    library: &Library,
) -> Result<Symbol<'_, cublas::Sdot>, libloading::Error> {
    unsafe { symbol::<cublas::Sdot>(library, cublas::SYM_CUBLAS_SDOT_V2) }
}

/// Resolve the fixed `cublasSasum_v2` entrypoint from its retained SDK library.
pub fn resolve_cublas_sasum_v2(
    library: &Library,
) -> Result<Symbol<'_, cublas::Sasum>, libloading::Error> {
    unsafe { symbol::<cublas::Sasum>(library, cublas::SYM_CUBLAS_SASUM_V2) }
}

/// Resolve the fixed `cublasSscal_v2` entrypoint from its retained SDK library.
pub fn resolve_cublas_sscal_v2(
    library: &Library,
) -> Result<Symbol<'_, cublas::Sscal>, libloading::Error> {
    unsafe { symbol::<cublas::Sscal>(library, cublas::SYM_CUBLAS_SSCAL_V2) }
}

/// Resolve the fixed `cublasGetPointerMode_v2` entrypoint from its retained SDK library.
pub fn resolve_cublas_get_pointer_mode_v2(
    library: &Library,
) -> Result<Symbol<'_, cublas::GetPointerMode>, libloading::Error> {
    unsafe { symbol::<cublas::GetPointerMode>(library, cublas::SYM_CUBLAS_GET_POINTER_MODE_V2) }
}

/// Resolve the fixed `cublasSetPointerMode_v2` entrypoint from its retained SDK library.
pub fn resolve_cublas_set_pointer_mode_v2(
    library: &Library,
) -> Result<Symbol<'_, cublas::SetPointerMode>, libloading::Error> {
    unsafe { symbol::<cublas::SetPointerMode>(library, cublas::SYM_CUBLAS_SET_POINTER_MODE_V2) }
}

/// Resolve the exact cold handle-stream observation ABI.
pub fn resolve_cublas_get_stream_v2(
    library: &Library,
) -> Result<Symbol<'_, cublas::GetStream>, libloading::Error> {
    // SAFETY: fixed SDK name/type pairing; caller retains this library owner.
    unsafe { symbol::<cublas::GetStream>(library, cublas::SYM_CUBLAS_GET_STREAM_V2) }
}
/// Observe a private handle's stream while retaining its library.
///
/// # Safety
/// The handle must be live on the selected device and stream must address writable host storage.
#[allow(non_snake_case)] // Exact private SDK boundary.
pub unsafe fn call_cublas_GetStream(
    function: cublas::GetStream,
    handle: cublas::CublasHandle,
    stream: *mut *mut c_void,
) -> cublas::CublasStatus {
    #[cfg(feature = "allocation-census")]
    census::blas();
    // SAFETY: caller supplies the live handle, exact output and retained provider.
    unsafe { function(handle, stream) }
}

/// Resolve the fixed `cublasSetStream_v2` entrypoint from its retained SDK library.
pub fn resolve_cublas_set_stream_v2(
    library: &Library,
) -> Result<Symbol<'_, cublas::SetStream>, libloading::Error> {
    unsafe { symbol::<cublas::SetStream>(library, cublas::SYM_CUBLAS_SET_STREAM_V2) }
}

/// Call the declared SDK entrypoint.
///
/// # Safety
/// `function` must be the declared SDK entry point, with its originating library retained.
/// Handles must be live in the selected context. Scalar/result pointers must address their
/// declared host types. Device buffers must cover the dimensions and strides passed here;
/// all buffer owners must survive queued completion on the handle's stream.
#[allow(non_snake_case)] // ABI type identity at the private boundary.
pub unsafe fn call_runtime_GetDeviceCount(
    function: runtime::GetDeviceCount,
    arg0: *mut c_int,
) -> runtime::CudaResult {
    unsafe { function(arg0) }
}

/// Call the declared SDK entrypoint.
///
/// # Safety
/// `function` must be the declared SDK entry point, with its originating library retained.
/// Handles must be live in the selected context. Scalar/result pointers must address their
/// declared host types. Device buffers must cover the dimensions and strides passed here;
/// all buffer owners must survive queued completion on the handle's stream.
#[allow(non_snake_case)] // ABI type identity at the private boundary.
pub unsafe fn call_driver_DriverInit(function: driver::DriverInit, arg0: u32) -> CudaResult {
    unsafe { function(arg0) }
}

/// Call the declared SDK entrypoint.
///
/// # Safety
/// `function` must be the declared SDK entry point, with its originating library retained.
/// Handles must be live in the selected context. Scalar/result pointers must address their
/// declared host types. Device buffers must cover the dimensions and strides passed here;
/// all buffer owners must survive queued completion on the handle's stream.
#[allow(non_snake_case)] // ABI type identity at the private boundary.
pub unsafe fn call_cublas_CreateHandle(
    function: cublas::CreateHandle,
    arg0: *mut cublas::CublasHandle,
) -> cublas::CublasStatus {
    #[cfg(feature = "allocation-census")]
    census::blas();
    unsafe { function(arg0) }
}

/// Call the declared SDK entrypoint.
///
/// # Safety
/// `function` must be the declared SDK entry point, with its originating library retained.
/// Handles must be live in the selected context. Scalar/result pointers must address their
/// declared host types. Device buffers must cover the dimensions and strides passed here;
/// all buffer owners must survive queued completion on the handle's stream.
#[allow(non_snake_case)] // ABI type identity at the private boundary.
pub unsafe fn call_cublas_SetStream(
    function: cublas::SetStream,
    arg0: cublas::CublasHandle,
    arg1: *mut c_void,
) -> cublas::CublasStatus {
    #[cfg(feature = "allocation-census")]
    census::blas();
    unsafe { function(arg0, arg1) }
}

/// Call the declared SDK entrypoint.
///
/// # Safety
/// `function` must be the declared SDK entry point, with its originating library retained.
/// Handles must be live in the selected context. Scalar/result pointers must address their
/// declared host types. Device buffers must cover the dimensions and strides passed here;
/// all buffer owners must survive queued completion on the handle's stream.
#[allow(non_snake_case)] // ABI type identity at the private boundary.
#[allow(clippy::too_many_arguments)] // Exact SDK ABI.
pub unsafe fn call_cublas_Sgemm(
    function: cublas::Sgemm,
    handle: cublas::CublasHandle,
    transpose_left: c_int,
    transpose_right: c_int,
    rows: c_int,
    columns: c_int,
    inner: c_int,
    alpha: *const f32,
    left: *const f32,
    left_stride: c_int,
    right: *const f32,
    right_stride: c_int,
    beta: *const f32,
    output: *mut f32,
    output_stride: c_int,
) -> cublas::CublasStatus {
    #[cfg(feature = "allocation-census")]
    census::blas();
    unsafe {
        function(
            handle,
            transpose_left,
            transpose_right,
            rows,
            columns,
            inner,
            alpha,
            left,
            left_stride,
            right,
            right_stride,
            beta,
            output,
            output_stride,
        )
    }
}

/// Call the declared SDK entrypoint.
///
/// # Safety
/// `function` must be the declared SDK entry point, with its originating library retained.
/// Handles must be live in the selected context. Scalar/result pointers must address their
/// declared host types. Device buffers must cover the dimensions and strides passed here;
/// all buffer owners must survive queued completion on the handle's stream.
#[allow(non_snake_case)] // ABI type identity at the private boundary.
pub unsafe fn call_cublas_GetPointerMode(
    function: cublas::GetPointerMode,
    arg0: cublas::CublasHandle,
    arg1: *mut c_int,
) -> cublas::CublasStatus {
    #[cfg(feature = "allocation-census")]
    census::blas();
    unsafe { function(arg0, arg1) }
}

/// Call the declared SDK entrypoint.
///
/// # Safety
/// `function` must be the declared SDK entry point, with its originating library retained.
/// Handles must be live in the selected context. Scalar/result pointers must address their
/// declared host types. Device buffers must cover the dimensions and strides passed here;
/// all buffer owners must survive queued completion on the handle's stream.
#[allow(non_snake_case)] // ABI type identity at the private boundary.
pub unsafe fn call_cublas_SetPointerMode(
    function: cublas::SetPointerMode,
    arg0: cublas::CublasHandle,
    arg1: c_int,
) -> cublas::CublasStatus {
    #[cfg(feature = "allocation-census")]
    census::blas();
    unsafe { function(arg0, arg1) }
}

/// Call the declared SDK entrypoint.
///
/// # Safety
/// `function` must be the declared SDK entry point, with its originating library retained.
/// Handles must be live in the selected context. Scalar/result pointers must address their
/// declared host types. Device buffers must cover the dimensions and strides passed here;
/// all buffer owners must survive queued completion on the handle's stream.
#[allow(non_snake_case)] // ABI type identity at the private boundary.
#[allow(clippy::too_many_arguments)] // Exact SDK ABI.
pub unsafe fn call_cublas_Sdot(
    function: cublas::Sdot,
    arg0: cublas::CublasHandle,
    arg1: c_int,
    arg2: *const f32,
    arg3: c_int,
    arg4: *const f32,
    arg5: c_int,
    arg6: *mut f32,
) -> cublas::CublasStatus {
    #[cfg(feature = "allocation-census")]
    census::blas();
    unsafe { function(arg0, arg1, arg2, arg3, arg4, arg5, arg6) }
}

/// Call the declared SDK entrypoint.
///
/// # Safety
/// `function` must be the declared SDK entry point, with its originating library retained.
/// Handles must be live in the selected context. Scalar/result pointers must address their
/// declared host types. Device buffers must cover the dimensions and strides passed here;
/// all buffer owners must survive queued completion on the handle's stream.
#[allow(non_snake_case)] // ABI type identity at the private boundary.
pub unsafe fn call_cublas_Sscal(
    function: cublas::Sscal,
    arg0: cublas::CublasHandle,
    arg1: c_int,
    arg2: *const f32,
    arg3: *mut f32,
    arg4: c_int,
) -> cublas::CublasStatus {
    #[cfg(feature = "allocation-census")]
    census::blas();
    unsafe { function(arg0, arg1, arg2, arg3, arg4) }
}

/// Call the declared SDK entrypoint.
///
/// # Safety
/// `function` must be the declared SDK entry point, with its originating library retained.
/// Handles must be live in the selected context. Scalar/result pointers must address their
/// declared host types. Device buffers must cover the dimensions and strides passed here;
/// all buffer owners must survive queued completion on the handle's stream.
#[allow(non_snake_case)] // ABI type identity at the private boundary.
pub unsafe fn call_cublas_Sasum(
    function: cublas::Sasum,
    arg0: cublas::CublasHandle,
    arg1: c_int,
    arg2: *const f32,
    arg3: c_int,
    arg4: *mut f32,
) -> cublas::CublasStatus {
    #[cfg(feature = "allocation-census")]
    census::blas();
    unsafe { function(arg0, arg1, arg2, arg3, arg4) }
}

/// Call the declared SDK entrypoint.
///
/// # Safety
/// `function` must be the declared SDK entry point, with its originating library retained.
/// Handles must be live in the selected context. Scalar/result pointers must address their
/// declared host types. Device buffers must cover the dimensions and strides passed here;
/// all buffer owners must survive queued completion on the handle's stream.
#[allow(non_snake_case)] // ABI type identity at the private boundary.
#[allow(clippy::too_many_arguments)] // Exact SDK ABI.
pub unsafe fn call_cublas_Dgemm(
    function: cublas::Dgemm,
    handle: cublas::CublasHandle,
    transpose_left: c_int,
    transpose_right: c_int,
    rows: c_int,
    columns: c_int,
    inner: c_int,
    alpha: *const f64,
    left: *const f64,
    left_stride: c_int,
    right: *const f64,
    right_stride: c_int,
    beta: *const f64,
    output: *mut f64,
    output_stride: c_int,
) -> cublas::CublasStatus {
    #[cfg(feature = "allocation-census")]
    census::blas();
    unsafe {
        function(
            handle,
            transpose_left,
            transpose_right,
            rows,
            columns,
            inner,
            alpha,
            left,
            left_stride,
            right,
            right_stride,
            beta,
            output,
            output_stride,
        )
    }
}

pub fn configure_math_modes(library: &Library, handle: CublasHandle) -> Result<(), CublasError> {
    let result = configure_mode_values(
        library,
        handle,
        CUBLAS_PEDANTIC_MATH,
        CUBLAS_ATOMICS_NOT_ALLOWED,
    );
    destroy_unescaped_handle_on_error(library, handle, result)
}

fn configure_mode_values(
    library: &Library,
    handle: CublasHandle,
    math_mode: c_int,
    atomics_mode: c_int,
) -> Result<(), CublasError> {
    // NVIDIA cuBLAS 13.4 §2.4.20/22: prescribed precision and no alternate
    // atomic reductions. These settings do not establish PCU checked intermediate faults.
    for (operation, mode, symbol) in [
        ("cublasSetMathMode", math_mode, SYM_CUBLAS_SET_MATH_MODE),
        (
            "cublasSetAtomicsMode",
            atomics_mode,
            SYM_CUBLAS_SET_ATOMICS_MODE,
        ),
    ] {
        // SAFETY: both documented setters take (handle, enum represented as c_int).
        let setter = match unsafe { crate::ffi::symbol::<SetMathMode>(library, symbol) } {
            Ok(setter) => setter,
            Err(error) => {
                return Err(CublasError::MissingSymbol {
                    symbol: operation,
                    detail: error.to_string(),
                });
            }
        };
        let status = unsafe { setter(handle, mode) };
        if status != CUBLAS_SUCCESS {
            return Err(CublasError::Status {
                operation,
                code: status,
            });
        }
    }
    Ok(())
}

fn destroy_unescaped_handle_on_error<T>(
    library: &Library,
    handle: CublasHandle,
    result: Result<T, CublasError>,
) -> Result<T, CublasError> {
    if result.is_err() {
        // SAFETY: the caller passes a freshly created handle which has not escaped or queued work.
        if let Ok(destroy) = resolve_cublas_destroy_v2(library) {
            unsafe {
                destroy(handle);
            }
        }
    }
    result
}

pub fn configure_native_math_modes(
    library: &Library,
    handle: CublasHandle,
    math_mode: c_int,
) -> Result<crate::CublasConfiguredModes, CublasError> {
    let result = (|| {
        let mut version = [0; 3];
        // SAFETY: cublasGetProperty uses libraryPropertyType (C int) and a writable int.
        let property =
            unsafe { symbol::<cublas::GetProperty>(library, cublas::SYM_CUBLAS_GET_PROPERTY) }
                .map_err(|error| CublasError::MissingSymbol {
                    symbol: "cublasGetProperty",
                    detail: error.to_string(),
                })?;
        for (kind, value) in version.iter_mut().enumerate() {
            let status = unsafe {
                property(
                    c_int::try_from(kind).expect("three property enum values"),
                    value,
                )
            };
            if status != CUBLAS_SUCCESS {
                return Err(CublasError::Status {
                    operation: "cublasGetProperty",
                    code: status,
                });
            }
        }
        // The precision/environment offer is based on the stable cuBLAS 13.4 documentation.
        // Unknown library contracts must be reviewed before being admitted as Preserve.
        // Toolkit and library versions differ: the CUDA13.4 GA/Update1 release tables
        // identify cuBLAS13.7/13.8 respectively (stable public SGEMM/DGEMM APIs).
        if version[0] != 13 || !matches!(version[1], 7 | 8) {
            return Err(CublasError::UnsupportedLibraryVersion {
                major: version[0],
                minor: version[1],
                patch: version[2],
            });
        }
        configure_mode_values(library, handle, math_mode, cublas::CUBLAS_ATOMICS_ALLOWED)?;
        let mut observed = [0; 2];
        read_cublas_mode(
            library,
            handle,
            "cublasGetMathMode",
            cublas::SYM_CUBLAS_GET_MATH_MODE,
            &mut observed[0],
        )?;
        read_cublas_mode(
            library,
            handle,
            "cublasGetAtomicsMode",
            cublas::SYM_CUBLAS_GET_ATOMICS_MODE,
            &mut observed[1],
        )?;
        for (operation, requested, actual) in [
            ("math mode", math_mode, observed[0]),
            ("atomics mode", cublas::CUBLAS_ATOMICS_ALLOWED, observed[1]),
        ] {
            if requested != actual {
                return Err(CublasError::ConfigurationMismatch {
                    operation,
                    requested,
                    observed: actual,
                });
            }
        }
        Ok(crate::CublasConfiguredModes {
            library_major: version[0],
            library_minor: version[1],
            library_patch: version[2],
            math_mode: observed[0],
            atomics_mode: observed[1],
        })
    })();
    destroy_unescaped_handle_on_error(library, handle, result)
}

fn read_cublas_mode(
    library: &Library,
    handle: CublasHandle,
    operation: &'static str,
    name: &[u8],
    value: &mut c_int,
) -> Result<(), CublasError> {
    // SAFETY: both public cuBLAS mode getters have (handle, writable enum pointer) ABI.
    let getter = unsafe { symbol::<cublas::GetMode>(library, name) }.map_err(|error| {
        CublasError::MissingSymbol {
            symbol: operation,
            detail: error.to_string(),
        }
    })?;
    let status = unsafe { getter(handle, value) };
    if status == CUBLAS_SUCCESS {
        Ok(())
    } else {
        Err(CublasError::Status {
            operation,
            code: status,
        })
    }
}

#[rustfmt::skip]
use cublas::{
    CublasHandle,
    SetMathMode,
    CUBLAS_PEDANTIC_MATH,
    CUBLAS_ATOMICS_NOT_ALLOWED,
    CUBLAS_SUCCESS,
    SYM_CUBLAS_SET_MATH_MODE,
    SYM_CUBLAS_SET_ATOMICS_MODE,
};
use crate::CublasError;

#[rustfmt::skip]
use driver::{
    DriverDeviceGetUuid,
    DriverUuid,
};
fn facts_driver_call<T: Copy>(
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

pub fn facts_init(library: &Library) -> Result<(), CudaError> {
    facts_driver_call(library, "cuInit", |f: DriverInit| unsafe { f(0) })
}
pub fn facts_device(
    library: &Library,
    value: &mut CudaDevice,
    ordinal: c_int,
) -> Result<(), CudaError> {
    facts_driver_call(library, "cuDeviceGet", |f: DriverGetDevice| unsafe {
        f(value, ordinal)
    })
}
pub fn facts_uuid(
    library: &Library,
    value: &mut DriverUuid,
    device: CudaDevice,
) -> Result<(), CudaError> {
    facts_driver_call(
        library,
        "cuDeviceGetUuid_v2",
        |f: DriverDeviceGetUuid| unsafe { f(value, device) },
    )
}
pub fn facts_attribute(
    library: &Library,
    value: &mut c_int,
    attribute: c_int,
    device: CudaDevice,
) -> Result<(), CudaError> {
    facts_driver_call(
        library,
        "cuDeviceGetAttribute",
        |f: DriverDeviceGetAttribute| unsafe { f(value, attribute, device) },
    )
}

/// Destroy a cuBLAS handle after its owning adapter has proved quiescence.
///
/// # Safety
/// The handle must be live, exclusive and no longer used by queued work.
#[allow(non_snake_case)] // Exact ABI identity.
pub unsafe fn call_cublas_DestroyHandle(
    function: cublas::DestroyHandle,
    handle: cublas::CublasHandle,
) -> cublas::CublasStatus {
    #[cfg(feature = "allocation-census")]
    census::blas();
    unsafe { function(handle) }
}
/// Open an SDK library whose symbol/resource lifetime is retained by the caller.
///
/// # Safety
/// Library initializers and all resolved symbol/resource lifetimes must be valid.
pub unsafe fn open_sdk_library(candidate: &std::ffi::OsStr) -> Result<Library, libloading::Error> {
    unsafe { Library::new(candidate) }
}

/// Resolve pinned deallocation before allocation so an absent entry point cannot strand storage.
pub fn require_free_host(library: &libloading::Library) -> Result<(), crate::CudaError> {
    // SAFETY: the declared signature matches CUDA's documented runtime deallocation ABI.
    unsafe { library.get::<runtime::FreeHost>(b"cudaFreeHost\0") }
        .map(|_| ())
        .map_err(|error| crate::CudaError::MissingSymbol {
            symbol: "cudaFreeHost",
            detail: error.to_string(),
        })
}

fn runtime_symbol(name: &str) -> Option<&'static [u8]> {
    Some(match name {
        "cudaGetDeviceCount" => b"cudaGetDeviceCount\0",
        "cudaSetDevice" => b"cudaSetDevice\0",
        "cudaDeviceSynchronize" => b"cudaDeviceSynchronize\0",
        "cudaMemGetInfo" => b"cudaMemGetInfo\0",
        "cudaGetErrorString" => b"cudaGetErrorString\0",
        "cudaMalloc" => b"cudaMalloc\0",
        "cudaFree" => b"cudaFree\0",
        "cudaMallocHost" => b"cudaMallocHost\0",
        "cudaFreeHost" => b"cudaFreeHost\0",
        "cudaMemcpy" => b"cudaMemcpy\0",
        "cudaMemcpyAsync" => b"cudaMemcpyAsync\0",
        "cudaMemsetAsync" => b"cudaMemsetAsync\0",
        "cudaStreamCreate" => b"cudaStreamCreate\0",
        "cudaStreamCreateWithPriority" => b"cudaStreamCreateWithPriority\0",
        "cudaStreamGetPriority" => b"cudaStreamGetPriority\0",
        "cudaStreamGetFlags" => b"cudaStreamGetFlags\0",
        "cudaDeviceGetStreamPriorityRange" => b"cudaDeviceGetStreamPriorityRange\0",
        "cudaStreamQuery" => b"cudaStreamQuery\0",
        "cudaEventQuery" => b"cudaEventQuery\0",
        "cudaStreamDestroy" => b"cudaStreamDestroy\0",
        "cudaStreamSynchronize" => b"cudaStreamSynchronize\0",
        "cudaStreamWaitEvent" => b"cudaStreamWaitEvent\0",
        "cudaStreamBeginCapture" => b"cudaStreamBeginCapture\0",
        "cudaStreamEndCapture" => b"cudaStreamEndCapture\0",
        "cudaGraphInstantiateWithFlags" => b"cudaGraphInstantiateWithFlags\0",
        "cudaGraphDestroy" => b"cudaGraphDestroy\0",
        "cudaGraphExecDestroy" => b"cudaGraphExecDestroy\0",
        "cudaGraphLaunch" => b"cudaGraphLaunch\0",
        "cudaEventCreateWithFlags" => b"cudaEventCreateWithFlags\0",
        "cudaEventDestroy" => b"cudaEventDestroy\0",
        "cudaEventRecord" => b"cudaEventRecord\0",
        "cudaEventSynchronize" => b"cudaEventSynchronize\0",
        "cudaEventElapsedTime" => b"cudaEventElapsedTime\0",
        _ => return None,
    })
}

fn driver_symbol(name: &str) -> Option<&'static [u8]> {
    Some(match name {
        "cuInit" => b"cuInit\0",
        "cuDeviceGet" => b"cuDeviceGet\0",
        "cuDeviceGetName" => b"cuDeviceGetName\0",
        "cuDeviceTotalMem_v2" => b"cuDeviceTotalMem_v2\0",
        "cuDeviceGetPCIBusId" => b"cuDeviceGetPCIBusId\0",
        "cuDeviceGetAttribute" => b"cuDeviceGetAttribute\0",
        "cuDeviceGetUuid_v2" => b"cuDeviceGetUuid_v2\0",
        "cuModuleLoadData" => b"cuModuleLoadData\0",
        "cuModuleGetFunction" => b"cuModuleGetFunction\0",
        "cuModuleUnload" => b"cuModuleUnload\0",
        "cuLaunchKernel" => b"cuLaunchKernel\0",
        "cuGetErrorString" => b"cuGetErrorString\0",
        _ => return None,
    })
}

/// Resolve the fixed `cublasDasum_v2` entrypoint from its retained SDK library.
pub fn resolve_cublas_dasum_v2(
    library: &Library,
) -> Result<Symbol<'_, cublas::Dasum>, libloading::Error> {
    unsafe { symbol::<cublas::Dasum>(library, cublas::SYM_CUBLAS_DASUM_V2) }
}

/// Resolve the fixed `cublasDscal_v2` entrypoint from its retained SDK library.
pub fn resolve_cublas_dscal_v2(
    library: &Library,
) -> Result<Symbol<'_, cublas::Dscal>, libloading::Error> {
    unsafe { symbol::<cublas::Dscal>(library, cublas::SYM_CUBLAS_DSCAL_V2) }
}

/// Call the declared SDK entrypoint.
///
/// # Safety
/// `function` must be the declared SDK entry point, with its originating library retained.
/// Handles must be live in the selected context. Scalar/result pointers must address their
/// declared host types. Device buffers must cover the dimensions and strides passed here;
/// all buffer owners must survive queued completion on the handle's stream.
#[allow(non_snake_case)] // ABI type identity at the private boundary.
pub unsafe fn call_cublas_Dscal(
    function: cublas::Dscal,
    arg0: cublas::CublasHandle,
    arg1: c_int,
    arg2: *const f64,
    arg3: *mut f64,
    arg4: c_int,
) -> cublas::CublasStatus {
    #[cfg(feature = "allocation-census")]
    census::blas();
    unsafe { function(arg0, arg1, arg2, arg3, arg4) }
}

/// Call the declared SDK entrypoint.
///
/// # Safety
/// `function` must be the declared SDK entry point, with its originating library retained.
/// Handles must be live in the selected context. Scalar/result pointers must address their
/// declared host types. Device buffers must cover the dimensions and strides passed here;
/// all buffer owners must survive queued completion on the handle's stream.
#[allow(non_snake_case)] // ABI type identity at the private boundary.
pub unsafe fn call_cublas_Dasum(
    function: cublas::Dasum,
    arg0: cublas::CublasHandle,
    arg1: c_int,
    arg2: *const f64,
    arg3: c_int,
    arg4: *mut f64,
) -> cublas::CublasStatus {
    #[cfg(feature = "allocation-census")]
    census::blas();
    unsafe { function(arg0, arg1, arg2, arg3, arg4) }
}

#[path = "blas_vector/blas_vector.rs"]
mod blas_vector;
#[rustfmt::skip]
pub use blas_vector::{
    VectorFunctions,
    retain as retain_blas_vector_functions,
};

#[path = "blas_handle/blas_handle.rs"]
mod blas_handle;
#[rustfmt::skip]
pub use blas_handle::{
    BlasHandleFunctions,
    retain as retain_blas_handle_functions,
};

#[cfg(all(feature = "allocation-census", feature = "tensor"))]
#[rustfmt::skip]
pub use census::{
    guarded_chain,
    guarded_kernel,
};

#[cfg(test)]
mod private_scope_tests {
    use super::{CudaRuntime, SelectedRuntimeScope, SELECTED_RUNTIME_SCOPE};

    #[test]
    #[ignore = "requires CUDA GPU; nested runtime and explicit selection scope invalidation"]
    fn cuda_gpu_private_scopes_invalidate_nested_and_explicit_selections() {
        let first = CudaRuntime::new(0).unwrap();
        let second = CudaRuntime::new(0).unwrap();
        let outer = first.enter_private_scope().unwrap();
        assert!(first.private_scope_selected());
        {
            let _inner = second.enter_private_scope().unwrap();
            assert!(second.private_scope_selected());
            assert!(!first.private_scope_selected());
        }
        assert!(!first.private_scope_selected());
        assert!(!second.private_scope_selected());
        drop(outer);
        let scope = first.enter_private_scope().unwrap();
        assert!(first.private_scope_selected());
        super::invoke_cudaSetDevice(&second, 0).unwrap();
        assert!(!first.private_scope_selected());
        drop(scope);
        // Emulate a foreign Runtime client between PCU calls. Even selecting the same ordinal
        // cannot leave a cached assumption: every call boundary makes a fresh selection.
        let setter = second.0.api.cuda_set_device.as_ref().unwrap();
        // SAFETY: the retained Runtime entrypoint has its declared ABI and ordinal zero was
        // successfully initialized above. No PCU private scope is active during this call.
        assert_eq!(unsafe { setter(0) }, super::CUDA_SUCCESS);
        let _next = first.enter_private_scope().unwrap();
        assert!(first.private_scope_selected());
    }

    #[test]
    fn nested_selection_clears_outer_assumptions_and_new_calls_select_again() {
        let outer = SelectedRuntimeScope::enter(1);
        assert_eq!(SELECTED_RUNTIME_SCOPE.get(), 1);
        {
            let _inner = SelectedRuntimeScope::enter(2);
            assert_eq!(SELECTED_RUNTIME_SCOPE.get(), 2);
        }
        assert_eq!(SELECTED_RUNTIME_SCOPE.get(), 0);
        drop(outer);
        assert_eq!(SELECTED_RUNTIME_SCOPE.get(), 0);
        // A foreign selection between calls is never cached. The next private boundary
        // installs its identity only after the real CUDA selection has succeeded.
        let next = SelectedRuntimeScope::enter(1);
        assert_eq!(SELECTED_RUNTIME_SCOPE.get(), 1);
        drop(next);
        assert_eq!(SELECTED_RUNTIME_SCOPE.get(), 0);
    }

    #[test]
    fn error_and_unwind_release_private_selection() {
        let fail = || -> Result<(), ()> {
            let _scope = SelectedRuntimeScope::enter(3);
            Err(())
        };
        assert_eq!(fail(), Err(()));
        assert_eq!(SELECTED_RUNTIME_SCOPE.get(), 0);
        let unwind = std::panic::catch_unwind(|| {
            let _scope = SelectedRuntimeScope::enter(4);
            panic!("controlled private scope unwind");
        });
        assert!(unwind.is_err());
        assert_eq!(SELECTED_RUNTIME_SCOPE.get(), 0);
    }
}
