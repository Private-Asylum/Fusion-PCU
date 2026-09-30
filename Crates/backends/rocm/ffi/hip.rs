//! HIP runtime ABI declarations. Resource ownership and execution remain in the runtime layer.

#[rustfmt::skip]
use std::ffi::{
    c_char,
    c_int,
    c_void,
};

pub type HipResult = c_int;
pub type HipDevice = c_int;
pub type HipStream = *mut c_void;
pub type HipEvent = *mut c_void;
pub type HipNoArgStatus = unsafe extern "C" fn() -> HipResult;

pub const HIP_SUCCESS: HipResult = 0;
pub const HIP_EVENT_DEFAULT: c_int = 0;
pub const HIP_EVENT_DISABLE_TIMING: c_int = 2;
pub const HIP_MEMCPY_HOST_TO_DEVICE: c_int = 1;
pub const HIP_MEMCPY_DEVICE_TO_HOST: c_int = 2;
pub const HIP_MEMCPY_DEVICE_TO_DEVICE: c_int = 3;

pub type GetDeviceCount = unsafe extern "C" fn(*mut c_int) -> HipResult;
pub type GetDevice = unsafe extern "C" fn(*mut HipDevice, c_int) -> HipResult;
pub type SetDevice = unsafe extern "C" fn(HipDevice) -> HipResult;
pub type GetDeviceName = unsafe extern "C" fn(*mut c_char, c_int, HipDevice) -> HipResult;
pub type GetDevicePciBusId = unsafe extern "C" fn(*mut c_char, c_int, c_int) -> HipResult;
pub type DeviceTotalMem = unsafe extern "C" fn(*mut usize, HipDevice) -> HipResult;
pub type MemGetInfo = unsafe extern "C" fn(*mut usize, *mut usize) -> HipResult;
pub type GetErrorString = unsafe extern "C" fn(HipResult) -> *const c_char;
pub type Malloc = unsafe extern "C" fn(*mut *mut c_void, usize) -> HipResult;
pub type Free = unsafe extern "C" fn(*mut c_void) -> HipResult;
pub type Memcpy = unsafe extern "C" fn(*mut c_void, *const c_void, usize, c_int) -> HipResult;
pub type MemcpyAsync =
    unsafe extern "C" fn(*mut c_void, *const c_void, usize, c_int, HipStream) -> HipResult;
pub type StreamCreate = unsafe extern "C" fn(*mut HipStream) -> HipResult;
pub type StreamDestroy = unsafe extern "C" fn(HipStream) -> HipResult;
pub type StreamSynchronize = unsafe extern "C" fn(HipStream) -> HipResult;
pub type StreamWaitEvent = unsafe extern "C" fn(HipStream, HipEvent, u32) -> HipResult;
pub type EventCreate = unsafe extern "C" fn(*mut HipEvent, c_int) -> HipResult;
pub type EventDestroy = unsafe extern "C" fn(HipEvent) -> HipResult;
pub type EventRecord = unsafe extern "C" fn(HipEvent, HipStream) -> HipResult;
pub type EventSynchronize = unsafe extern "C" fn(HipEvent) -> HipResult;
pub type EventElapsedTime = unsafe extern "C" fn(*mut f32, HipEvent, HipEvent) -> HipResult;
pub type ModuleHandle = *mut c_void;
pub type KernelHandle = *mut c_void;
pub type ModuleLoadData = unsafe extern "C" fn(*mut ModuleHandle, *const c_void) -> HipResult;
pub type ModuleGetFunction =
    unsafe extern "C" fn(*mut KernelHandle, ModuleHandle, *const c_char) -> HipResult;
pub type ModuleUnload = unsafe extern "C" fn(ModuleHandle) -> HipResult;
pub type ModuleLaunchKernel = unsafe extern "C" fn(
    KernelHandle,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    HipStream,
    *mut *mut c_void,
    *mut *mut c_void,
) -> HipResult;

/// Stable HIP device-attribute query ABI; enum discriminants are from `hip_runtime_api.h`.
pub type DeviceGetAttribute = unsafe extern "C" fn(*mut c_int, c_int, c_int) -> HipResult;

pub const ATTRIBUTE_ASYNC_ENGINE_COUNT: c_int = 2;
pub const ATTRIBUTE_MAX_BLOCK_DIMENSIONS: [c_int; 3] = [26, 27, 28];
pub const ATTRIBUTE_MAX_GRID_DIMENSIONS: [c_int; 3] = [29, 30, 31];
pub const ATTRIBUTE_MAX_THREADS_PER_BLOCK: c_int = 56;
pub const ATTRIBUTE_MULTIPROCESSOR_COUNT: c_int = 63;
pub const ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK: c_int = 74;
pub const ATTRIBUTE_WARP_SIZE: c_int = 87;

/// Load the optional stable attribute query without extending ordinary runtime initialization.
pub fn device_get_attribute(library: &super::Library) -> Option<DeviceGetAttribute> {
    // SAFETY: HIP documents this exact C signature; the caller retains the loaded library.
    unsafe { super::symbol::<DeviceGetAttribute>(library, b"hipDeviceGetAttribute\0") }
        .ok()
        .map(|function| *function)
}
