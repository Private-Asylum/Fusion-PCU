//! CUDA Driver API ABI declarations used for device and module operations.

#[rustfmt::skip]
use std::ffi::{
    c_char,
    c_int,
    c_void,
};

#[rustfmt::skip]
use super::runtime::{
    CudaDevice,
    CudaResult,
    CudaStream,
};

pub const CUDA_ERROR_INVALID_VALUE: c_int = 1;
pub const CUDA_ERROR_NOT_SUPPORTED: c_int = 801;

pub const DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK: c_int = 1;
pub const DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X: c_int = 2;
pub const DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y: c_int = 3;
pub const DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z: c_int = 4;
pub const DEVICE_ATTRIBUTE_MAX_GRID_DIM_X: c_int = 5;
pub const DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y: c_int = 6;
pub const DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z: c_int = 7;
pub const DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK: c_int = 8;
pub const DEVICE_ATTRIBUTE_WARP_SIZE: c_int = 10;
pub const DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT: c_int = 16;
pub const DEVICE_ATTRIBUTE_INTEGRATED: c_int = 18;
pub const DEVICE_ATTRIBUTE_CONCURRENT_KERNELS: c_int = 31;
pub const DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR: c_int = 39;
pub const DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT: c_int = 40;
pub const DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING: c_int = 41;
pub const DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED: c_int = 78;

/// Documented fixed 16-octet Driver API UUID layout, independent of device property versions.
#[repr(C)]
pub struct DriverUuid {
    pub bytes: [u8; 16],
}

pub type DriverDeviceGetUuid = unsafe extern "C" fn(*mut DriverUuid, CudaDevice) -> CudaResult;

pub const DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: c_int = 75;
pub const DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: c_int = 76;

pub type ModuleHandle = *mut c_void;
pub type KernelHandle = *mut c_void;
pub type ModuleLaunchKernel = unsafe extern "C" fn(
    KernelHandle,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    CudaStream,
    *mut *mut c_void,
    *mut *mut c_void,
) -> CudaResult;

pub type DriverInit = unsafe extern "C" fn(u32) -> CudaResult;
pub type DriverGetDevice = unsafe extern "C" fn(*mut CudaDevice, c_int) -> CudaResult;
pub type DriverDeviceGetName = unsafe extern "C" fn(*mut c_char, c_int, CudaDevice) -> CudaResult;
pub type DriverDeviceTotalMem = unsafe extern "C" fn(*mut usize, CudaDevice) -> CudaResult;
pub type DriverDeviceGetPciBusId =
    unsafe extern "C" fn(*mut c_char, c_int, CudaDevice) -> CudaResult;
pub type DriverDeviceGetAttribute =
    unsafe extern "C" fn(*mut c_int, c_int, CudaDevice) -> CudaResult;
pub type DriverModuleLoadData =
    unsafe extern "C" fn(*mut ModuleHandle, *const c_void) -> CudaResult;
pub type DriverModuleGetFunction =
    unsafe extern "C" fn(*mut KernelHandle, ModuleHandle, *const c_char) -> CudaResult;
pub type DriverModuleUnload = unsafe extern "C" fn(ModuleHandle) -> CudaResult;
pub type DriverModuleLaunchKernel = ModuleLaunchKernel;
pub type DriverGetErrorString = unsafe extern "C" fn(CudaResult, *mut *const c_char) -> CudaResult;

pub fn driver_symbol(name: &str) -> Option<&'static [u8]> {
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
