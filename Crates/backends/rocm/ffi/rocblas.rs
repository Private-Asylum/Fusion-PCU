//! rocBLAS ABI declarations. Safe adapters retain ownership and completion semantics.

#[rustfmt::skip]
use std::ffi::{
    c_int,
    c_void,
};

pub type RocblasStatus = c_int;
pub type RocblasHandle = *mut c_void;
pub type CreateHandle = unsafe extern "C" fn(*mut RocblasHandle) -> RocblasStatus;
pub type DestroyHandle = unsafe extern "C" fn(RocblasHandle) -> RocblasStatus;
pub type Sgemm = unsafe extern "C" fn(
    RocblasHandle,
    c_int,
    c_int,
    c_int,
    c_int,
    c_int,
    *const f32,
    *const f32,
    c_int,
    *const f32,
    c_int,
    *const f32,
    *mut f32,
    c_int,
) -> RocblasStatus;
pub type Dgemm = unsafe extern "C" fn(
    RocblasHandle,
    c_int,
    c_int,
    c_int,
    c_int,
    c_int,
    *const f64,
    *const f64,
    c_int,
    *const f64,
    c_int,
    *const f64,
    *mut f64,
    c_int,
) -> RocblasStatus;
pub type Sdot = unsafe extern "C" fn(
    RocblasHandle,
    c_int,
    *const f32,
    c_int,
    *const f32,
    c_int,
    *mut f32,
) -> RocblasStatus;
pub type Sasum =
    unsafe extern "C" fn(RocblasHandle, c_int, *const f32, c_int, *mut f32) -> RocblasStatus;
pub type Sscal =
    unsafe extern "C" fn(RocblasHandle, c_int, *const f32, *mut f32, c_int) -> RocblasStatus;
pub type Dasum =
    unsafe extern "C" fn(RocblasHandle, c_int, *const f64, c_int, *mut f64) -> RocblasStatus;
pub type Dscal =
    unsafe extern "C" fn(RocblasHandle, c_int, *const f64, *mut f64, c_int) -> RocblasStatus;
pub type GetPointerMode = unsafe extern "C" fn(RocblasHandle, *mut c_int) -> RocblasStatus;
pub type SetPointerMode = unsafe extern "C" fn(RocblasHandle, c_int) -> RocblasStatus;
pub type SetStream = unsafe extern "C" fn(RocblasHandle, *mut c_void) -> RocblasStatus;

pub const ROCBLAS_SUCCESS: RocblasStatus = 0;
pub const ROCBLAS_OPERATION_NONE: c_int = 111;
pub const ROCBLAS_OPERATION_TRANSPOSE: c_int = 112;
pub const ROCBLAS_POINTER_MODE_HOST: c_int = 0;
pub const ROCBLAS_POINTER_MODE_DEVICE: c_int = 1;
