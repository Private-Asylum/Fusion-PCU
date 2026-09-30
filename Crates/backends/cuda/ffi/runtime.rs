//! CUDA Runtime API ABI declarations used by the dynamically loaded runtime layer.

#[rustfmt::skip]
use std::ffi::{
    c_char,
    c_int,
    c_void,
};

pub type CudaResult = c_int;
pub type CudaDevice = c_int;
pub type CudaStream = *mut c_void;
pub type CudaEvent = *mut c_void;
pub type CudaGraph = *mut c_void;
pub type CudaGraphExec = *mut c_void;

pub const CUDA_SUCCESS: CudaResult = 0;
pub const CUDA_ERROR_NOT_READY: CudaResult = 600;
pub const CUDA_STREAM_DEFAULT: u32 = 0;
pub const CUDA_STREAM_NON_BLOCKING: u32 = 1;
pub const CUDA_EVENT_DEFAULT: c_int = 0;
pub const CUDA_EVENT_DISABLE_TIMING: c_int = 2;
pub const CUDA_MEMCPY_HOST_TO_DEVICE: c_int = 1;
pub const CUDA_MEMCPY_DEVICE_TO_HOST: c_int = 2;
pub const CUDA_MEMCPY_DEVICE_TO_DEVICE: c_int = 3;

pub type GetDeviceCount = unsafe extern "C" fn(*mut c_int) -> CudaResult;
pub type DeviceSynchronize = unsafe extern "C" fn() -> CudaResult;
pub type SetDevice = unsafe extern "C" fn(CudaDevice) -> CudaResult;
pub type MemGetInfo = unsafe extern "C" fn(*mut usize, *mut usize) -> CudaResult;
pub type GetErrorString = unsafe extern "C" fn(CudaResult) -> *const c_char;
pub type Malloc = unsafe extern "C" fn(*mut *mut c_void, usize) -> CudaResult;
pub type Free = unsafe extern "C" fn(*mut c_void) -> CudaResult;
pub type MallocHost = unsafe extern "C" fn(*mut *mut c_void, usize) -> CudaResult;
pub type FreeHost = unsafe extern "C" fn(*mut c_void) -> CudaResult;
pub type RuntimeFree = Free;
pub type Memcpy = unsafe extern "C" fn(*mut c_void, *const c_void, usize, c_int) -> CudaResult;
pub type MemcpyAsync =
    unsafe extern "C" fn(*mut c_void, *const c_void, usize, c_int, CudaStream) -> CudaResult;
pub type MemsetAsync = unsafe extern "C" fn(*mut c_void, c_int, usize, CudaStream) -> CudaResult;
pub type StreamCreate = unsafe extern "C" fn(*mut CudaStream) -> CudaResult;
pub type StreamCreateWithPriority = unsafe extern "C" fn(*mut CudaStream, u32, c_int) -> CudaResult;
pub type StreamGetPriority = unsafe extern "C" fn(CudaStream, *mut c_int) -> CudaResult;
pub type StreamGetFlags = unsafe extern "C" fn(CudaStream, *mut u32) -> CudaResult;
pub type DeviceGetStreamPriorityRange = unsafe extern "C" fn(*mut c_int, *mut c_int) -> CudaResult;
pub type StreamQuery = unsafe extern "C" fn(CudaStream) -> CudaResult;
pub type EventQuery = unsafe extern "C" fn(CudaEvent) -> CudaResult;
pub type StreamDestroy = unsafe extern "C" fn(CudaStream) -> CudaResult;
pub type StreamSynchronize = unsafe extern "C" fn(CudaStream) -> CudaResult;
pub type StreamWaitEvent = unsafe extern "C" fn(CudaStream, CudaEvent, u32) -> CudaResult;
pub type StreamBeginCapture = unsafe extern "C" fn(CudaStream, c_int) -> CudaResult;
pub type StreamEndCapture = unsafe extern "C" fn(CudaStream, *mut CudaGraph) -> CudaResult;
pub type GraphInstantiateWithFlags =
    unsafe extern "C" fn(*mut CudaGraphExec, CudaGraph, u64) -> CudaResult;
pub type GraphDestroy = unsafe extern "C" fn(CudaGraph) -> CudaResult;
pub type GraphExecDestroy = unsafe extern "C" fn(CudaGraphExec) -> CudaResult;
pub type GraphLaunch = unsafe extern "C" fn(CudaGraphExec, CudaStream) -> CudaResult;
pub type EventCreate = unsafe extern "C" fn(*mut CudaEvent, c_int) -> CudaResult;
pub type EventDestroy = unsafe extern "C" fn(CudaEvent) -> CudaResult;
pub type EventRecord = unsafe extern "C" fn(CudaEvent, CudaStream) -> CudaResult;
pub type EventSynchronize = unsafe extern "C" fn(CudaEvent) -> CudaResult;
pub type EventElapsedTime = unsafe extern "C" fn(*mut f32, CudaEvent, CudaEvent) -> CudaResult;

pub fn runtime_symbol(name: &str) -> Option<&'static [u8]> {
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

/// Resolve pinned deallocation before allocation so an absent entry point cannot strand storage.
pub fn require_free_host(library: &libloading::Library) -> Result<(), crate::CudaError> {
    // SAFETY: the declared signature matches CUDA's documented runtime deallocation ABI.
    unsafe { library.get::<FreeHost>(b"cudaFreeHost\0") }
        .map(|_| ())
        .map_err(|error| crate::CudaError::MissingSymbol {
            symbol: "cudaFreeHost",
            detail: error.to_string(),
        })
}
