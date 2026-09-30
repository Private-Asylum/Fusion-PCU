//! HIPRTC ABI declarations. Compilation policy and program ownership stay in codegen.

#[rustfmt::skip]
use std::ffi::{
    c_char,
    c_int,
};

pub type Program = *mut std::ffi::c_void;
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
pub type GetCodeSize = unsafe extern "C" fn(Program, *mut usize) -> ResultCode;
pub type GetCode = unsafe extern "C" fn(Program, *mut c_char) -> ResultCode;
pub type GetErrorString = unsafe extern "C" fn(ResultCode) -> *const c_char;
