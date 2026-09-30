//! NVRTC and CUDA driver ABI declarations used by the code generator.

#[rustfmt::skip]
use std::ffi::{
    c_char,
    c_int,
    c_void,
};

pub type Program = *mut c_void;
pub type ResultCode = c_int;
pub type CreateProgram = unsafe extern "C" fn(
    *mut Program,
    *const c_char,
    *const c_char,
    c_int,
    *const *const c_char,
    *const *const c_char,
) -> ResultCode;
pub type DestroyProgram = unsafe extern "C" fn(*mut Program) -> ResultCode;
pub type CompileProgram = unsafe extern "C" fn(Program, c_int, *const *const c_char) -> ResultCode;
pub type GetProgramLogSize = unsafe extern "C" fn(Program, *mut usize) -> ResultCode;
pub type GetProgramLog = unsafe extern "C" fn(Program, *mut c_char) -> ResultCode;
pub type GetPtxSize = unsafe extern "C" fn(Program, *mut usize) -> ResultCode;
pub type GetPtx = unsafe extern "C" fn(Program, *mut c_char) -> ResultCode;
pub type GetErrorString = unsafe extern "C" fn(ResultCode) -> *const c_char;

pub const NVRTC_CREATE_PROGRAM: &[u8] = b"nvrtcCreateProgram\0";
pub const NVRTC_DESTROY_PROGRAM: &[u8] = b"nvrtcDestroyProgram\0";
pub const NVRTC_COMPILE_PROGRAM: &[u8] = b"nvrtcCompileProgram\0";
pub const NVRTC_GET_PROGRAM_LOG_SIZE: &[u8] = b"nvrtcGetProgramLogSize\0";
pub const NVRTC_GET_PROGRAM_LOG: &[u8] = b"nvrtcGetProgramLog\0";
pub const NVRTC_GET_PTX_SIZE: &[u8] = b"nvrtcGetPTXSize\0";
pub const NVRTC_GET_PTX: &[u8] = b"nvrtcGetPTX\0";
pub const NVRTC_GET_ERROR_STRING: &[u8] = b"nvrtcGetErrorString\0";
pub const CU_INIT: &[u8] = b"cuInit\0";
pub const CU_DEVICE_GET: &[u8] = b"cuDeviceGet\0";
pub const CU_DEVICE_GET_ATTRIBUTE: &[u8] = b"cuDeviceGetAttribute\0";
