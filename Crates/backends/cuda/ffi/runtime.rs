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
