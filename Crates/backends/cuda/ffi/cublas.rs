//! cuBLAS 13.4 public C ABI (`cublas_v2.h`).

#[rustfmt::skip]
use std::ffi::{
    c_int,
    c_void,
};

pub type CublasStatus = c_int;
pub type CublasHandle = *mut c_void;
pub type CreateHandle = unsafe extern "C" fn(*mut CublasHandle) -> CublasStatus;
pub type DestroyHandle = unsafe extern "C" fn(CublasHandle) -> CublasStatus;
pub type Sgemm = unsafe extern "C" fn(
    CublasHandle,
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
) -> CublasStatus;
pub type Dgemm = unsafe extern "C" fn(
    CublasHandle,
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
) -> CublasStatus;
pub type Sdot = unsafe extern "C" fn(
    CublasHandle,
    c_int,
    *const f32,
    c_int,
    *const f32,
    c_int,
    *mut f32,
) -> CublasStatus;
pub type Sasum =
    unsafe extern "C" fn(CublasHandle, c_int, *const f32, c_int, *mut f32) -> CublasStatus;
pub type Sscal =
    unsafe extern "C" fn(CublasHandle, c_int, *const f32, *mut f32, c_int) -> CublasStatus;
pub type Dasum =
    unsafe extern "C" fn(CublasHandle, c_int, *const f64, c_int, *mut f64) -> CublasStatus;
pub type Dscal =
    unsafe extern "C" fn(CublasHandle, c_int, *const f64, *mut f64, c_int) -> CublasStatus;
pub type GetPointerMode = unsafe extern "C" fn(CublasHandle, *mut c_int) -> CublasStatus;
pub type SetPointerMode = unsafe extern "C" fn(CublasHandle, c_int) -> CublasStatus;
pub type SetMathMode = unsafe extern "C" fn(CublasHandle, c_int) -> CublasStatus;
pub type GetMode = unsafe extern "C" fn(CublasHandle, *mut c_int) -> CublasStatus;
pub type GetProperty = unsafe extern "C" fn(c_int, *mut c_int) -> CublasStatus;
pub type GetStream = unsafe extern "C" fn(CublasHandle, *mut *mut c_void) -> CublasStatus;
pub type SetStream = unsafe extern "C" fn(CublasHandle, *mut c_void) -> CublasStatus;

pub const CUBLAS_SUCCESS: CublasStatus = 0;
pub const CUBLAS_OPERATION_NONE: c_int = 0;
pub const CUBLAS_OPERATION_TRANSPOSE: c_int = 1;
pub const CUBLAS_POINTER_MODE_HOST: c_int = 0;
pub const CUBLAS_POINTER_MODE_DEVICE: c_int = 1;

pub const CUBLAS_PEDANTIC_MATH: c_int = 2;
pub const CUBLAS_DEFAULT_MATH: c_int = 0;
pub const CUBLAS_TF32_TENSOR_OP_MATH: c_int = 3;
pub const CUBLAS_ATOMICS_NOT_ALLOWED: c_int = 0;
pub const CUBLAS_ATOMICS_ALLOWED: c_int = 1;
pub const SYM_CUBLAS_CREATE_V2: &[u8] = b"cublasCreate_v2\0";
pub const SYM_CUBLAS_DESTROY_V2: &[u8] = b"cublasDestroy_v2\0";
pub const SYM_CUBLAS_DGEMM_V2: &[u8] = b"cublasDgemm_v2\0";
pub const SYM_CUBLAS_GET_POINTER_MODE_V2: &[u8] = b"cublasGetPointerMode_v2\0";
pub const SYM_CUBLAS_SASUM_V2: &[u8] = b"cublasSasum_v2\0";
pub const SYM_CUBLAS_SDOT_V2: &[u8] = b"cublasSdot_v2\0";
pub const SYM_CUBLAS_SET_ATOMICS_MODE: &[u8] = b"cublasSetAtomicsMode\0";
pub const SYM_CUBLAS_SET_MATH_MODE: &[u8] = b"cublasSetMathMode\0";
pub const SYM_CUBLAS_GET_MATH_MODE: &[u8] = b"cublasGetMathMode\0";
pub const SYM_CUBLAS_GET_ATOMICS_MODE: &[u8] = b"cublasGetAtomicsMode\0";
pub const SYM_CUBLAS_GET_PROPERTY: &[u8] = b"cublasGetProperty\0";
pub const SYM_CUBLAS_SET_POINTER_MODE_V2: &[u8] = b"cublasSetPointerMode_v2\0";
pub const SYM_CUBLAS_GET_STREAM_V2: &[u8] = b"cublasGetStream_v2\0";
pub const SYM_CUBLAS_SET_STREAM_V2: &[u8] = b"cublasSetStream_v2\0";
pub const SYM_CUBLAS_SGEMM_V2: &[u8] = b"cublasSgemm_v2\0";
pub const SYM_CUBLAS_SSCAL_V2: &[u8] = b"cublasSscal_v2\0";

/// Typed double ASUM/SCAL declarations: <https://docs.nvidia.com/cuda/cublas/index.html>
pub const SYM_CUBLAS_DASUM_V2: &[u8] = b"cublasDasum_v2\0";
pub const SYM_CUBLAS_DSCAL_V2: &[u8] = b"cublasDscal_v2\0";

/// Exact-ABI failure injection; it submits no arithmetic or substitute execution.
#[cfg(all(test, feature = "allocation-census"))]
pub fn fixture_sgemm_failure() -> Sgemm {
    fixture_sgemm_execution_failed
}

#[cfg(all(test, feature = "allocation-census"))]
const unsafe extern "C" fn fixture_sgemm_execution_failed(
    _handle: CublasHandle,
    _operation_a: c_int,
    _operation_b: c_int,
    _m: c_int,
    _n: c_int,
    _k: c_int,
    _alpha: *const f32,
    _a: *const f32,
    _lda: c_int,
    _b: *const f32,
    _ldb: c_int,
    _beta: *const f32,
    _c: *mut f32,
    _ldc: c_int,
) -> CublasStatus {
    13 // CUBLAS_STATUS_EXECUTION_FAILED
}
