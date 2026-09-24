//! A small dynamically loaded rocBLAS SGEMM adapter.
//!
//! This uses rocBLAS' public `rocblas_sgemm` ABI directly and does not require hipBLASLt.

use std::{
    cell::Cell,
    ffi::{
        c_int,
        c_void,
        OsString,
    },
    fmt,
    ptr,
    rc::Rc,
    sync::Arc,
};

use libloading::Library;

use super::{
    DeviceBuffer,
    HipError,
    HipRuntime,
};

type RocblasStatus = c_int;
type RocblasHandle = *mut c_void;
type CreateHandle = unsafe extern "C" fn(*mut RocblasHandle) -> RocblasStatus;
type DestroyHandle = unsafe extern "C" fn(RocblasHandle) -> RocblasStatus;
type Sgemm = unsafe extern "C" fn(
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

const ROCBLAS_SUCCESS: RocblasStatus = 0;
const ROCBLAS_OPERATION_NONE: c_int = 111;
const ROCBLAS_OPERATION_TRANSPOSE: c_int = 112;

/// Failures returned by the rocBLAS SGEMM adapter.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RocblasError {
    LibraryUnavailable(String),
    MissingSymbol {
        symbol: &'static str,
        detail: String,
    },
    Status {
        operation: &'static str,
        code: i32,
    },
    Hip(HipError),
    DifferentRuntime,
    Busy,
    AliasedBuffers,
    CompletionUnknown,
    InvalidDimensions(&'static str),
    DimensionOverflow,
    BufferTooSmall {
        matrix: &'static str,
        allocation: usize,
        required: usize,
    },
}

impl fmt::Display for RocblasError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LibraryUnavailable(s) => write!(f, "rocBLAS library unavailable: {s}"),
            Self::MissingSymbol { symbol, detail } => {
                write!(f, "rocBLAS symbol {symbol} unavailable: {detail}")
            }
            Self::Status { operation, code } => {
                write!(f, "{operation} failed with rocBLAS status {code}")
            }
            Self::Hip(error) => error.fmt(f),
            Self::DifferentRuntime => {
                f.write_str("rocBLAS buffers belong to a different HIP runtime or device")
            }
            Self::Busy => f.write_str("rocBLAS buffer is busy with another device operation"),
            Self::AliasedBuffers => {
                f.write_str("rocBLAS SGEMM does not accept aliased A, B, or C allocations")
            }
            Self::CompletionUnknown => {
                f.write_str("a previous rocBLAS operation did not confirm device completion")
            }
            Self::InvalidDimensions(why) => write!(f, "invalid SGEMM dimensions: {why}"),
            Self::DimensionOverflow => {
                f.write_str("SGEMM dimension or matrix size overflows the supported range")
            }
            Self::BufferTooSmall {
                matrix,
                allocation,
                required,
            } => write!(
                f,
                "SGEMM {matrix} needs {required} bytes, allocation has {allocation}"
            ),
        }
    }
}

impl std::error::Error for RocblasError {}
impl From<HipError> for RocblasError {
    fn from(value: HipError) -> Self {
        Self::Hip(value)
    }
}

/// rocBLAS handle tied to the HIP runtime and device used to create it.
pub struct Rocblas {
    runtime: HipRuntime,
    _library: Arc<Library>,
    handle: RocblasHandle,
    poisoned: Cell<bool>,
}

impl Rocblas {
    /// Load `librocblas.so` (or `ROCBLAS_LIBRARY`) and create a handle for `runtime`'s device.
    pub fn new(runtime: &HipRuntime) -> Result<Self, RocblasError> {
        let candidates: Vec<OsString> = std::env::var_os("ROCBLAS_LIBRARY")
            .map(|p| vec![p])
            .unwrap_or_else(|| vec!["librocblas.so".into(), "librocblas.so.5".into()]);
        let mut last_error = None;
        for candidate in candidates {
            // SAFETY: rocBLAS exports the documented C ABI. Arc keeps it loaded for handle life.
            let library = match unsafe { Library::new(&candidate) } {
                Ok(library) => Arc::new(library),
                Err(error) => {
                    last_error = Some(error.to_string());
                    continue;
                }
            };
            runtime.hip_set_device(runtime.0.device)?;
            let mut handle = ptr::null_mut();
            // SAFETY: symbol type matches rocblas_create_handle's C declaration.
            let create = unsafe { library.get::<CreateHandle>(b"rocblas_create_handle\0") }
                .map_err(|e| RocblasError::MissingSymbol {
                    symbol: "rocblas_create_handle",
                    detail: e.to_string(),
                })?;
            let status = unsafe { create(&mut handle) };
            if status != ROCBLAS_SUCCESS {
                return Err(RocblasError::Status {
                    operation: "rocblas_create_handle",
                    code: status,
                });
            }
            return Ok(Self {
                runtime: runtime.clone(),
                _library: library,
                handle,
                poisoned: Cell::new(false),
            });
        }
        Err(RocblasError::LibraryUnavailable(
            last_error.unwrap_or_else(|| "no library candidates".into()),
        ))
    }

    /// Compute column-major `C = alpha * op(A) * op(B) + beta * C` and wait for device completion.
    /// Leading dimensions and allocation extents are validated before entering the C ABI.
    #[allow(clippy::too_many_arguments)]
    pub fn sgemm(
        &self,
        transpose_a: bool,
        transpose_b: bool,
        m: usize,
        n: usize,
        k: usize,
        alpha: f32,
        a: &DeviceBuffer,
        lda: usize,
        b: &DeviceBuffer,
        ldb: usize,
        beta: f32,
        c: &DeviceBuffer,
        ldc: usize,
    ) -> Result<(), RocblasError> {
        if self.poisoned.get() {
            return Err(RocblasError::CompletionUnknown);
        }
        let (a_rows, a_cols) = if transpose_a { (k, m) } else { (m, k) };
        let (b_rows, b_cols) = if transpose_b { (n, k) } else { (k, n) };
        matrix_bytes("A", a_rows, a_cols, lda, a.len())?;
        matrix_bytes("B", b_rows, b_cols, ldb, b.len())?;
        matrix_bytes("C", m, n, ldc, c.len())?;
        self.runtime
            .ensure_same_runtime(&a.allocation.runtime)
            .map_err(|_| RocblasError::DifferentRuntime)?;
        self.runtime
            .ensure_same_runtime(&b.allocation.runtime)
            .map_err(|_| RocblasError::DifferentRuntime)?;
        self.runtime
            .ensure_same_runtime(&c.allocation.runtime)
            .map_err(|_| RocblasError::DifferentRuntime)?;
        if Rc::ptr_eq(&a.allocation, &b.allocation)
            || Rc::ptr_eq(&a.allocation, &c.allocation)
            || Rc::ptr_eq(&b.allocation, &c.allocation)
        {
            return Err(RocblasError::AliasedBuffers);
        }
        let a_lease = a.acquire_access().map_err(|_| RocblasError::Busy)?;
        let b_lease = b.acquire_access().map_err(|_| RocblasError::Busy)?;
        let c_lease = c.acquire_access().map_err(|_| RocblasError::Busy)?;
        let (m, n, k, lda, ldb, ldc) = (
            c_int::try_from(m).map_err(|_| RocblasError::DimensionOverflow)?,
            c_int::try_from(n).map_err(|_| RocblasError::DimensionOverflow)?,
            c_int::try_from(k).map_err(|_| RocblasError::DimensionOverflow)?,
            c_int::try_from(lda).map_err(|_| RocblasError::DimensionOverflow)?,
            c_int::try_from(ldb).map_err(|_| RocblasError::DimensionOverflow)?,
            c_int::try_from(ldc).map_err(|_| RocblasError::DimensionOverflow)?,
        );
        self.runtime.hip_set_device(self.runtime.0.device)?;
        // SAFETY: signature follows rocblas_sgemm in rocblas.h; buffers are validated allocations
        // from this runtime/device and the scalar pointers remain live through the call.
        let sgemm = unsafe { self._library.get::<Sgemm>(b"rocblas_sgemm\0") }.map_err(|e| {
            RocblasError::MissingSymbol {
                symbol: "rocblas_sgemm",
                detail: e.to_string(),
            }
        })?;
        let status = unsafe {
            sgemm(
                self.handle,
                if transpose_a {
                    ROCBLAS_OPERATION_TRANSPOSE
                } else {
                    ROCBLAS_OPERATION_NONE
                },
                if transpose_b {
                    ROCBLAS_OPERATION_TRANSPOSE
                } else {
                    ROCBLAS_OPERATION_NONE
                },
                m,
                n,
                k,
                &alpha,
                a.allocation.pointer.cast(),
                lda,
                b.allocation.pointer.cast(),
                ldb,
                &beta,
                c.allocation.pointer.cast(),
                ldc,
            )
        };
        // rocBLAS enqueues on its default stream. Device synchronization makes this API explicitly
        // synchronous and ensures all borrowed buffer owners remain valid until completion. Even a
        // rocBLAS error is followed by synchronization because the call may have partially queued.
        let sync = self.runtime.call(
            "hipDeviceSynchronize",
            |f: unsafe extern "C" fn() -> c_int| unsafe { f() },
        );
        if let Err(error) = sync {
            // Completion is now unknown. Retain every allocation and the library/handle forever;
            // freeing any of them could race device work that HIP failed to confirm had stopped.
            self.poisoned.set(true);
            std::mem::forget(a_lease);
            std::mem::forget(b_lease);
            std::mem::forget(c_lease);
            return Err(error.into());
        }
        if status != ROCBLAS_SUCCESS {
            return Err(RocblasError::Status {
                operation: "rocblas_sgemm",
                code: status,
            });
        }
        Ok(())
    }
}

impl Drop for Rocblas {
    fn drop(&mut self) {
        if self.poisoned.get() {
            std::mem::forget(self._library.clone());
            return;
        }
        if self.runtime.hip_set_device(self.runtime.0.device).is_err() {
            std::mem::forget(self._library.clone());
            return;
        }
        // SAFETY: handle came from rocblas_create_handle and library is retained until this drop.
        if let Ok(destroy) = unsafe {
            self._library
                .get::<DestroyHandle>(b"rocblas_destroy_handle\0")
        } {
            let _ = unsafe { destroy(self.handle) };
        }
    }
}

fn matrix_bytes(
    matrix: &'static str,
    rows: usize,
    columns: usize,
    leading: usize,
    allocation: usize,
) -> Result<usize, RocblasError> {
    if leading < rows.max(1) {
        return Err(RocblasError::InvalidDimensions(
            "leading dimension is smaller than max(1, stored row count)",
        ));
    }
    if columns == 0 {
        return Ok(0);
    }
    let elements = leading
        .checked_mul(columns)
        .ok_or(RocblasError::DimensionOverflow)?;
    let required = elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or(RocblasError::DimensionOverflow)?;
    if required > allocation {
        return Err(RocblasError::BufferTooSmall {
            matrix,
            allocation,
            required,
        });
    }
    Ok(required)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matrix_extent_checks_leading_dimension_and_overflow() {
        assert!(matches!(
            matrix_bytes("A", 3, 2, 2, 1024),
            Err(RocblasError::InvalidDimensions(_))
        ));
        assert_eq!(matrix_bytes("B", 0, 4, 1, 16), Ok(16));
        assert_eq!(matrix_bytes("B", 2, 0, 2, 0), Ok(0));
        assert_eq!(matrix_bytes("C", 2, 3, 2, 24), Ok(24));
        assert!(matches!(
            matrix_bytes("C", 2, 3, 2, 20),
            Err(RocblasError::BufferTooSmall { .. })
        ));
        assert!(matches!(
            matrix_bytes("C", 2, usize::MAX, 2, 0),
            Err(RocblasError::DimensionOverflow)
        ));
    }
}
