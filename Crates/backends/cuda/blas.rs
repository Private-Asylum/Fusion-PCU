//! Small dynamically loaded cuBLAS GEMM and vector-operation adapters.
//!
//! This uses cuBLAS' public `cublasSgemm_v2` ABI directly and does not require cuBLASLt.

#[rustfmt::skip]
use std::{
    any::Any,
    cell::{
        Cell,
        OnceCell,
    },
    ffi::{
        c_int,
        OsString,
    },
    fmt,
    mem::size_of,
    ptr,
    rc::Rc,
    sync::Arc,
    time::Instant,
};

use crate::ffi::Library;

#[rustfmt::skip]
use super::{
    DeviceBuffer,
    CudaCompletionBatch,
    CudaError,
    CudaRuntime,
    CudaStreamHandle,
};

#[rustfmt::skip]
use crate::ffi::cublas::{
    CUBLAS_ATOMICS_NOT_ALLOWED,
    CUBLAS_OPERATION_NONE,
    CUBLAS_OPERATION_TRANSPOSE,
    CUBLAS_PEDANTIC_MATH,
    CUBLAS_POINTER_MODE_DEVICE,
    CUBLAS_POINTER_MODE_HOST,
    CUBLAS_SUCCESS,
    CreateHandle,
    CublasHandle,
    DestroyHandle,
    Dgemm,
    GetPointerMode,
    SYM_CUBLAS_CREATE_V2,
    SYM_CUBLAS_DESTROY_V2,
    SYM_CUBLAS_DGEMM_V2,
    SYM_CUBLAS_GET_POINTER_MODE_V2,
    SYM_CUBLAS_SASUM_V2,
    SYM_CUBLAS_SDOT_V2,
    SYM_CUBLAS_SET_ATOMICS_MODE,
    SYM_CUBLAS_SET_MATH_MODE,
    SYM_CUBLAS_SET_POINTER_MODE_V2,
    SYM_CUBLAS_SET_STREAM_V2,
    SYM_CUBLAS_SGEMM_V2,
    SYM_CUBLAS_SSCAL_V2,
    Sasum,
    Sdot,
    SetMathMode,
    SetPointerMode,
    SetStream,
    Sgemm,
    Sscal,
};
use crate::ffi::runtime::DeviceSynchronize;

/// Failures returned by the cuBLAS GEMM and vector adapters.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CublasError {
    LibraryUnavailable(String),
    MissingSymbol {
        symbol: &'static str,
        detail: String,
    },
    Status {
        operation: &'static str,
        code: i32,
    },
    Cuda(CudaError),
    DifferentRuntime,
    DifferentStream,
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
    InvalidVector(&'static str),
}

/// Host-side phase durations for one explicitly profiled SGEMM call.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct CublasSgemmHostTiming {
    /// Validation, leases, device selection, and symbol lookup.
    pub preflight: std::time::Duration,
    /// Time spent in the `cublasSgemm_v2` C function.
    pub cublas_call: std::time::Duration,
    /// Time spent in `cudaDeviceSynchronize`.
    pub device_synchronize: std::time::Duration,
    /// Lease release or quarantine disposition after synchronization.
    pub cleanup: std::time::Duration,
}

#[derive(Clone, Copy)]
enum SgemmPhase {
    Preflight,
    CublasCall,
    DeviceSynchronize,
    Cleanup,
}

trait SgemmTimingSink {
    type Mark;

    fn begin(&mut self, phase: SgemmPhase) -> Self::Mark;
    fn finish(&mut self, phase: SgemmPhase, mark: Self::Mark);
}

struct NoopSgemmTiming;

impl SgemmTimingSink for NoopSgemmTiming {
    type Mark = ();

    #[inline(always)]
    fn begin(&mut self, _: SgemmPhase) {}

    #[inline(always)]
    fn finish(&mut self, _: SgemmPhase, (): Self::Mark) {}
}

struct CollectSgemmTiming<'a>(&'a mut CublasSgemmHostTiming);

impl SgemmTimingSink for CollectSgemmTiming<'_> {
    type Mark = Instant;

    fn begin(&mut self, _: SgemmPhase) -> Self::Mark {
        Instant::now()
    }

    fn finish(&mut self, phase: SgemmPhase, mark: Self::Mark) {
        let elapsed = mark.elapsed();
        match phase {
            SgemmPhase::Preflight => self.0.preflight = elapsed,
            SgemmPhase::CublasCall => self.0.cublas_call = elapsed,
            SgemmPhase::DeviceSynchronize => self.0.device_synchronize = elapsed,
            SgemmPhase::Cleanup => self.0.cleanup = elapsed,
        }
    }
}

impl fmt::Display for CublasError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LibraryUnavailable(s) => write!(f, "cuBLAS library unavailable: {s}"),
            Self::MissingSymbol { symbol, detail } => {
                write!(f, "cuBLAS symbol {symbol} unavailable: {detail}")
            }
            Self::Status { operation, code } => {
                write!(f, "{operation} failed with cuBLAS status {code}")
            }
            Self::Cuda(error) => error.fmt(f),
            Self::DifferentRuntime => {
                f.write_str("cuBLAS buffers belong to a different CUDA runtime or device")
            }
            Self::DifferentStream => f.write_str("cuBLAS handle is not bound to the batch stream"),
            Self::Busy => f.write_str("cuBLAS buffer is busy with another device operation"),
            Self::AliasedBuffers => {
                f.write_str("cuBLAS operation does not accept aliased input/output allocations")
            }
            Self::CompletionUnknown => {
                f.write_str("a previous cuBLAS operation did not confirm device completion")
            }
            Self::InvalidDimensions(why) => write!(f, "invalid cuBLAS GEMM dimensions: {why}"),
            Self::DimensionOverflow => {
                f.write_str("cuBLAS dimension or matrix size overflows the supported range")
            }
            Self::BufferTooSmall {
                matrix,
                allocation,
                required,
            } => write!(
                f,
                "cuBLAS GEMM {matrix} needs {required} bytes, allocation has {allocation}"
            ),
            Self::InvalidVector(why) => write!(f, "invalid cuBLAS vector reduction: {why}"),
        }
    }
}

impl std::error::Error for CublasError {}
impl From<CudaError> for CublasError {
    fn from(value: CudaError) -> Self {
        Self::Cuda(value)
    }
}

/// cuBLAS handle tied to the CUDA runtime and device used to create it.
pub struct Cublas {
    owner: Rc<CublasHandleOwner>,
    in_flight: Rc<Cell<usize>>,
    queue_marker: Rc<dyn Any>,
    bound_stream: Option<CudaStreamHandle>,
}

struct CublasHandleOwner {
    runtime: CudaRuntime,
    library: Arc<Library>,
    handle: CublasHandle,
    poisoned: Cell<bool>,
    dgemm: OnceCell<Result<Dgemm, CublasError>>,
}

struct CublasAsyncOperationOwner {
    _handle: Rc<dyn Any>,
    in_flight: Rc<Cell<usize>>,
    scalars: [f32; 2],
}

struct CublasAsyncDoubleOperationOwner {
    _handle: Rc<dyn Any>,
    in_flight: Rc<Cell<usize>>,
    scalars: [f64; 2],
}

impl Drop for CublasAsyncDoubleOperationOwner {
    fn drop(&mut self) {
        release_async_operation(&self.in_flight);
    }
}

impl Drop for CublasAsyncOperationOwner {
    fn drop(&mut self) {
        release_async_operation(&self.in_flight);
    }
}

fn reserve_async_operation(in_flight: &Cell<usize>) -> Result<(), CublasError> {
    let next = in_flight.get().checked_add(1).ok_or(CublasError::Busy)?;
    in_flight.set(next);
    Ok(())
}

fn release_async_operation(in_flight: &Cell<usize>) {
    let active = in_flight.get();
    debug_assert!(active > 0, "cuBLAS async operation count underflow");
    in_flight.set(active.saturating_sub(1));
}

const fn queue_extension_allowed(
    poisoned: bool,
    same_stream: bool,
    has_reservation_marker: bool,
) -> bool {
    !poisoned && same_stream && has_reservation_marker
}

fn configure_math_modes(library: &Library, handle: CublasHandle) -> Result<(), CublasError> {
    // NVIDIA cuBLAS 13.4 §2.4.20/22: prescribed precision and no alternate
    // atomic reductions. These settings do not establish PCU checked intermediate faults.
    for (operation, mode, symbol) in [
        (
            "cublasSetMathMode",
            CUBLAS_PEDANTIC_MATH,
            SYM_CUBLAS_SET_MATH_MODE,
        ),
        (
            "cublasSetAtomicsMode",
            CUBLAS_ATOMICS_NOT_ALLOWED,
            SYM_CUBLAS_SET_ATOMICS_MODE,
        ),
    ] {
        // SAFETY: both documented setters take (handle, enum represented as c_int).
        let setter = match unsafe { crate::ffi::symbol::<SetMathMode>(library, symbol) } {
            Ok(setter) => setter,
            Err(error) => {
                // SAFETY: this freshly created handle has not escaped or queued work.
                if let Ok(destroy) =
                    unsafe { crate::ffi::symbol::<DestroyHandle>(library, SYM_CUBLAS_DESTROY_V2) }
                {
                    unsafe {
                        destroy(handle);
                    }
                }
                return Err(CublasError::MissingSymbol {
                    symbol: operation,
                    detail: error.to_string(),
                });
            }
        };
        let status = unsafe { setter(handle, mode) };
        if status != CUBLAS_SUCCESS {
            // SAFETY: the handle was created above and has not escaped.
            if let Ok(destroy) =
                unsafe { crate::ffi::symbol::<DestroyHandle>(library, SYM_CUBLAS_DESTROY_V2) }
            {
                unsafe {
                    destroy(handle);
                }
            }
            return Err(CublasError::Status {
                operation,
                code: status,
            });
        }
    }
    Ok(())
}

impl Cublas {
    fn ensure_idle(&self) -> Result<(), CublasError> {
        if self.owner.poisoned.get() {
            return Err(CublasError::CompletionUnknown);
        }
        if self.in_flight.get() != 0 {
            return Err(CublasError::Busy);
        }
        Ok(())
    }

    /// Validate an SGEMM operation against the exact stream captured by an owned graph node.
    /// This is crate-private so graph construction can reject every malformed operation before
    /// any node is submitted; the submission method repeats the check at its boundary.
    #[allow(
        clippy::many_single_char_names,
        clippy::similar_names,
        clippy::too_many_arguments
    )]
    pub(crate) fn validate_sgemm_for_stream(
        &self,
        stream: &CudaStreamHandle,
        transpose_a: bool,
        transpose_b: bool,
        m: usize,
        n: usize,
        k: usize,
        a: &DeviceBuffer,
        lda: usize,
        b: &DeviceBuffer,
        ldb: usize,
        c: &DeviceBuffer,
        ldc: usize,
    ) -> Result<(), CublasError> {
        self.ensure_idle()?;
        self.validate_sgemm_while_reserved(
            stream,
            transpose_a,
            transpose_b,
            m,
            n,
            k,
            a,
            lda,
            b,
            ldb,
            c,
            ldc,
        )
    }

    #[allow(
        clippy::many_single_char_names,
        clippy::similar_names,
        clippy::too_many_arguments
    )]
    fn validate_sgemm_while_reserved(
        &self,
        stream: &CudaStreamHandle,
        transpose_a: bool,
        transpose_b: bool,
        m: usize,
        n: usize,
        k: usize,
        a: &DeviceBuffer,
        lda: usize,
        b: &DeviceBuffer,
        ldb: usize,
        c: &DeviceBuffer,
        ldc: usize,
    ) -> Result<(), CublasError> {
        if self.owner.poisoned.get() {
            return Err(CublasError::CompletionUnknown);
        }
        if !stream.belongs_to_runtime(&self.owner.runtime) {
            return Err(CublasError::DifferentRuntime);
        }
        if !self
            .bound_stream
            .as_ref()
            .is_some_and(|bound| Rc::ptr_eq(&bound.inner, &stream.inner))
        {
            return Err(CublasError::DifferentStream);
        }

        let (a_rows, a_cols) = if transpose_a { (k, m) } else { (m, k) };
        let (b_rows, b_cols) = if transpose_b { (n, k) } else { (k, n) };
        matrix_bytes("A", a_rows, a_cols, lda, a.len())?;
        matrix_bytes("B", b_rows, b_cols, ldb, b.len())?;
        matrix_bytes("C", m, n, ldc, c.len())?;
        for buffer in [a, b, c] {
            self.owner
                .runtime
                .ensure_same_runtime(&buffer.allocation.runtime)
                .map_err(|_| CublasError::DifferentRuntime)?;
        }
        if Rc::ptr_eq(&a.allocation, &b.allocation)
            || Rc::ptr_eq(&a.allocation, &c.allocation)
            || Rc::ptr_eq(&b.allocation, &c.allocation)
        {
            return Err(CublasError::AliasedBuffers);
        }
        for dimension in [m, n, k, lda, ldb, ldc] {
            c_int::try_from(dimension).map_err(|_| CublasError::DimensionOverflow)?;
        }
        Ok(())
    }

    // Keep operand and leading-dimension names aligned with the public BLAS argument order.
    #[allow(
        clippy::many_single_char_names,
        clippy::similar_names,
        clippy::too_many_arguments
    )]
    fn validate_dgemm_while_reserved(
        &self,
        stream: &CudaStreamHandle,
        transpose_a: bool,
        transpose_b: bool,
        m: usize,
        n: usize,
        k: usize,
        a: &DeviceBuffer,
        lda: usize,
        b: &DeviceBuffer,
        ldb: usize,
        c: &DeviceBuffer,
        ldc: usize,
    ) -> Result<(), CublasError> {
        if self.owner.poisoned.get() {
            return Err(CublasError::CompletionUnknown);
        }
        if !stream.belongs_to_runtime(&self.owner.runtime) {
            return Err(CublasError::DifferentRuntime);
        }
        if !self
            .bound_stream
            .as_ref()
            .is_some_and(|bound| Rc::ptr_eq(&bound.inner, &stream.inner))
        {
            return Err(CublasError::DifferentStream);
        }

        let (a_rows, a_cols) = if transpose_a { (k, m) } else { (m, k) };
        let (b_rows, b_cols) = if transpose_b { (n, k) } else { (k, n) };
        matrix_bytes_f64("A", a_rows, a_cols, lda, a.len())?;
        matrix_bytes_f64("B", b_rows, b_cols, ldb, b.len())?;
        matrix_bytes_f64("C", m, n, ldc, c.len())?;
        for buffer in [a, b, c] {
            self.owner
                .runtime
                .ensure_same_runtime(&buffer.allocation.runtime)
                .map_err(|_| CublasError::DifferentRuntime)?;
        }
        if Rc::ptr_eq(&a.allocation, &b.allocation)
            || Rc::ptr_eq(&a.allocation, &c.allocation)
            || Rc::ptr_eq(&b.allocation, &c.allocation)
        {
            return Err(CublasError::AliasedBuffers);
        }
        for dimension in [m, n, k, lda, ldb, ldc] {
            c_int::try_from(dimension).map_err(|_| CublasError::DimensionOverflow)?;
        }
        Ok(())
    }

    fn dgemm_function(&self) -> Result<Dgemm, CublasError> {
        self.owner
            .dgemm
            .get_or_init(|| {
                // SAFETY: Dgemm matches the installed public cuBLAS C ABI. The owning library
                // outlives this cached function pointer and every operation that invokes it.
                unsafe { crate::ffi::symbol::<Dgemm>(&self.owner.library, SYM_CUBLAS_DGEMM_V2) }
                    .map(|symbol| *symbol)
                    .map_err(|error| CublasError::MissingSymbol {
                        symbol: "cublasDgemm_v2",
                        detail: error.to_string(),
                    })
            })
            .clone()
    }

    /// Resolve double-precision GEMM support without submitting device work.
    ///
    /// The independent lookup is cached, including failure, and never affects SGEMM setup.
    ///
    /// # Errors
    /// Returns a missing-symbol error when the loaded library cannot provide DGEMM.
    pub fn require_dgemm_support(&self) -> Result<(), CublasError> {
        self.dgemm_function().map(|_| ())
    }

    /// Whether this handle may safely admit another operation after prior completion results.
    #[must_use]
    pub fn is_usable(&self) -> bool {
        !self.owner.poisoned.get() && self.in_flight.get() == 0
    }

    /// Whether an event-chain successor may append work after this exact handle reservation.
    ///
    /// The executor must adopt this completion into its consumer batch before submitting the
    /// successor. The completion marker proves that the pending work belongs to this handle;
    /// stream identity and the wait-error state prevent extending unrelated or uncertain work.
    pub(crate) fn can_extend_from(&self, completion: &super::CudaBatchCompletion) -> bool {
        queue_extension_allowed(
            self.owner.poisoned.get(),
            self.bound_stream
                .as_ref()
                .is_some_and(|stream| completion.uses_stream(stream)),
            completion.has_queue_marker(&self.queue_marker),
        )
    }

    /// Whether this open batch inherited this handle's exact queue reservation.
    ///
    /// The executor may append another SGEMM only after adopting the predecessor completion into
    /// this batch. A marker from a different handle or stream does not grant admission.
    pub(crate) fn can_extend_in_batch(&self, batch: &CudaCompletionBatch) -> bool {
        queue_extension_allowed(
            self.owner.poisoned.get(),
            self.bound_stream
                .as_ref()
                .is_some_and(|stream| Rc::ptr_eq(&stream.inner, &batch.stream_handle().inner)),
            batch.has_queue_marker(&self.queue_marker),
        )
    }

    /// Load `libcublas.so` (or `CUBLAS_LIBRARY`) and create a handle for `runtime`'s device.
    ///
    /// # Errors
    ///
    /// Returns an error when cuBLAS cannot be loaded, a required symbol is missing, or CUDA/cuBLAS
    /// fails to create the handle.
    pub fn new(runtime: &CudaRuntime) -> Result<Self, CublasError> {
        let candidates: Vec<OsString> = std::env::var_os("CUBLAS_LIBRARY").map_or_else(
            || {
                vec![
                    "libcublas.so".into(),
                    "libcublas.so.13".into(),
                    "libcublas.so.12".into(),
                    "/usr/local/cuda/lib64/libcublas.so".into(),
                ]
            },
            |path| vec![path],
        );
        let mut last_error = None;
        for candidate in candidates {
            // SAFETY: cuBLAS exports the documented C ABI. Arc keeps it loaded for handle life.
            let library = match unsafe { crate::ffi::load_uncached_library(&candidate) } {
                Ok(library) => Arc::new(library),
                Err(error) => {
                    last_error = Some(error.to_string());
                    continue;
                }
            };
            runtime.cuda_set_device(runtime.0.ordinal)?;
            let mut handle = ptr::null_mut();
            // SAFETY: symbol type matches cublasCreate_v2's C declaration.
            let create =
                unsafe { crate::ffi::symbol::<CreateHandle>(&library, SYM_CUBLAS_CREATE_V2) }
                    .map_err(|e| CublasError::MissingSymbol {
                        symbol: "cublasCreate_v2",
                        detail: e.to_string(),
                    })?;
            // An opened handle is only useful to the tensor adapter when the required operation
            // is present. Validate the symbol during setup so assessment cannot claim Library
            // support and defer a missing-symbol failure until execution.
            unsafe { crate::ffi::symbol::<Sgemm>(&library, SYM_CUBLAS_SGEMM_V2) }.map_err(|e| {
                CublasError::MissingSymbol {
                    symbol: "cublasSgemm_v2",
                    detail: e.to_string(),
                }
            })?;
            for (symbol, result) in [
                (
                    "cublasSdot_v2",
                    unsafe { crate::ffi::symbol::<Sdot>(&library, SYM_CUBLAS_SDOT_V2) }.map(|_| ()),
                ),
                (
                    "cublasSasum_v2",
                    unsafe { crate::ffi::symbol::<Sasum>(&library, SYM_CUBLAS_SASUM_V2) }
                        .map(|_| ()),
                ),
                (
                    "cublasSscal_v2",
                    unsafe { crate::ffi::symbol::<Sscal>(&library, SYM_CUBLAS_SSCAL_V2) }
                        .map(|_| ()),
                ),
                (
                    "cublasGetPointerMode_v2",
                    unsafe {
                        crate::ffi::symbol::<GetPointerMode>(
                            &library,
                            SYM_CUBLAS_GET_POINTER_MODE_V2,
                        )
                    }
                    .map(|_| ()),
                ),
                (
                    "cublasSetPointerMode_v2",
                    unsafe {
                        crate::ffi::symbol::<SetPointerMode>(
                            &library,
                            SYM_CUBLAS_SET_POINTER_MODE_V2,
                        )
                    }
                    .map(|_| ()),
                ),
            ] {
                result.map_err(|e| CublasError::MissingSymbol {
                    symbol,
                    detail: e.to_string(),
                })?;
            }
            let status = unsafe { create(&raw mut handle) };
            if status != CUBLAS_SUCCESS {
                return Err(CublasError::Status {
                    operation: "cublasCreate_v2",
                    code: status,
                });
            }
            configure_math_modes(&library, handle)?;
            return Ok(Self {
                owner: Rc::new(CublasHandleOwner {
                    runtime: runtime.clone(),
                    library,
                    handle,
                    poisoned: Cell::new(false),
                    dgemm: OnceCell::new(),
                }),
                in_flight: Rc::new(Cell::new(0)),
                queue_marker: Rc::new(()) as Rc<dyn Any>,
                bound_stream: None,
            });
        }
        Err(CublasError::LibraryUnavailable(
            last_error.unwrap_or_else(|| "no library candidates".into()),
        ))
    }

    /// Binds this handle to a selected CUDA stream before graph operations are submitted.
    ///
    /// The handle retains the stream owner. Call only while no cuBLAS operation is in flight;
    /// the tensor adapter binds once during preparation and keeps its synchronous behavior.
    ///
    /// # Errors
    ///
    /// Returns an identity, missing-symbol, or cuBLAS status error.
    pub fn bind_stream(&mut self, stream: &CudaStreamHandle) -> Result<(), CublasError> {
        self.ensure_idle()?;
        if self.bound_stream.is_some() {
            return Err(CublasError::Busy);
        }
        if !stream.belongs_to_runtime(&self.owner.runtime) {
            return Err(CublasError::DifferentRuntime);
        }
        self.owner
            .runtime
            .cuda_set_device(self.owner.runtime.0.ordinal)?;
        // SAFETY: cublasSetStream_v2 has the documented C ABI in cublas-auxiliary.h.
        let set_stream = unsafe {
            crate::ffi::symbol::<SetStream>(&self.owner.library, SYM_CUBLAS_SET_STREAM_V2)
        }
        .map_err(|error| CublasError::MissingSymbol {
            symbol: "cublasSetStream_v2",
            detail: error.to_string(),
        })?;
        let status = unsafe { set_stream(self.owner.handle, stream.inner.raw) };
        if status != CUBLAS_SUCCESS {
            return Err(CublasError::Status {
                operation: "cublasSetStream_v2",
                code: status,
            });
        }
        self.bound_stream = Some(stream.clone());
        Ok(())
    }

    /// Queue SGEMM on this handle's bound stream and retain its buffers and handle through the
    /// batch's final event. The handle remains unavailable until that completion is released.
    /// On a cuBLAS error the operation may have been partially queued; finish or drop the batch
    /// before attempting any other operation with this handle.
    ///
    /// # Errors
    ///
    /// Returns validation, runtime/stream, busy-buffer, or cuBLAS status errors. If this returns
    /// an error after the batch has retained the operation, the batch must still be finished or
    /// dropped normally so it can establish quiescence or quarantine its resources.
    #[allow(
        clippy::many_single_char_names,
        clippy::similar_names,
        clippy::too_many_arguments,
        clippy::too_many_lines
    )]
    pub fn sgemm_into_batch(
        &self,
        batch: &mut CudaCompletionBatch,
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
    ) -> Result<(), CublasError> {
        let batch_stream = batch.stream_handle();
        let inherited_reservation = batch.has_queue_marker(&self.queue_marker);
        if inherited_reservation {
            self.validate_sgemm_while_reserved(
                batch_stream,
                transpose_a,
                transpose_b,
                m,
                n,
                k,
                a,
                lda,
                b,
                ldb,
                c,
                ldc,
            )?;
        } else {
            self.validate_sgemm_for_stream(
                batch_stream,
                transpose_a,
                transpose_b,
                m,
                n,
                k,
                a,
                lda,
                b,
                ldb,
                c,
                ldc,
            )?;
        }
        let (m, n, k, lda, ldb, ldc) = (
            c_int::try_from(m).map_err(|_| CublasError::DimensionOverflow)?,
            c_int::try_from(n).map_err(|_| CublasError::DimensionOverflow)?,
            c_int::try_from(k).map_err(|_| CublasError::DimensionOverflow)?,
            c_int::try_from(lda).map_err(|_| CublasError::DimensionOverflow)?,
            c_int::try_from(ldb).map_err(|_| CublasError::DimensionOverflow)?,
            c_int::try_from(ldc).map_err(|_| CublasError::DimensionOverflow)?,
        );
        self.owner
            .runtime
            .cuda_set_device(self.owner.runtime.0.ordinal)?;
        // SAFETY: signature matches cublasSgemm_v2; runtime, dimensions, stream, and allocations
        // have been checked before loading or invoking the C ABI.
        let sgemm =
            unsafe { crate::ffi::symbol::<Sgemm>(&self.owner.library, SYM_CUBLAS_SGEMM_V2) }
                .map_err(|error| CublasError::MissingSymbol {
                    symbol: "cublasSgemm_v2",
                    detail: error.to_string(),
                })?;

        reserve_async_operation(&self.in_flight)?;
        let owner = Rc::new(CublasAsyncOperationOwner {
            _handle: self.owner.clone(),
            in_flight: Rc::clone(&self.in_flight),
            scalars: [alpha, beta],
        });
        let external_owner: Rc<dyn Any> = owner.clone();
        batch.retain_external_operation(&[a, b, c], external_owner)?;
        // Publish permission only after the batch owns the reservation. A failed preflight or
        // lease acquisition must not leave a marker that could authorize unrelated queued work.
        if !inherited_reservation {
            batch.register_queue_marker(Rc::clone(&self.queue_marker))?;
        }

        // The host scalar pointers are backed by the retained owner and remain valid until batch
        // completion. cuBLAS may enqueue work even when it reports an error, so do not release
        // the owner or leases on that path.
        let status = unsafe {
            sgemm(
                self.owner.handle,
                if transpose_a {
                    CUBLAS_OPERATION_TRANSPOSE
                } else {
                    CUBLAS_OPERATION_NONE
                },
                if transpose_b {
                    CUBLAS_OPERATION_TRANSPOSE
                } else {
                    CUBLAS_OPERATION_NONE
                },
                m,
                n,
                k,
                std::ptr::from_ref(&owner.scalars[0]),
                a.allocation.pointer.cast(),
                lda,
                b.allocation.pointer.cast(),
                ldb,
                std::ptr::from_ref(&owner.scalars[1]),
                c.allocation.pointer.cast(),
                ldc,
            )
        };
        if status != CUBLAS_SUCCESS {
            return Err(CublasError::Status {
                operation: "cublasSgemm_v2",
                code: status,
            });
        }
        Ok(())
    }

    /// Queue DGEMM on this handle's bound stream and retain its buffers, handle, and host scalars
    /// through the batch's final event.
    ///
    /// # Errors
    ///
    /// Returns validation, runtime/stream, busy-buffer, missing-symbol, or cuBLAS status errors.
    /// If submission has retained the operation, finish or drop the batch normally so it can
    /// establish quiescence or quarantine its resources.
    // Preserve cuBLAS' operand naming and keep retention/completion steps visible at submission.
    #[allow(
        clippy::many_single_char_names,
        clippy::similar_names,
        clippy::too_many_arguments,
        clippy::too_many_lines
    )]
    pub fn dgemm_into_batch(
        &self,
        batch: &mut CudaCompletionBatch,
        transpose_a: bool,
        transpose_b: bool,
        m: usize,
        n: usize,
        k: usize,
        alpha: f64,
        a: &DeviceBuffer,
        lda: usize,
        b: &DeviceBuffer,
        ldb: usize,
        beta: f64,
        c: &DeviceBuffer,
        ldc: usize,
    ) -> Result<(), CublasError> {
        let batch_stream = batch.stream_handle();
        let inherited_reservation = batch.has_queue_marker(&self.queue_marker);
        if !inherited_reservation {
            self.ensure_idle()?;
        }
        self.validate_dgemm_while_reserved(
            batch_stream,
            transpose_a,
            transpose_b,
            m,
            n,
            k,
            a,
            lda,
            b,
            ldb,
            c,
            ldc,
        )?;
        let (m, n, k, lda, ldb, ldc) = (
            c_int::try_from(m).map_err(|_| CublasError::DimensionOverflow)?,
            c_int::try_from(n).map_err(|_| CublasError::DimensionOverflow)?,
            c_int::try_from(k).map_err(|_| CublasError::DimensionOverflow)?,
            c_int::try_from(lda).map_err(|_| CublasError::DimensionOverflow)?,
            c_int::try_from(ldb).map_err(|_| CublasError::DimensionOverflow)?,
            c_int::try_from(ldc).map_err(|_| CublasError::DimensionOverflow)?,
        );
        self.owner
            .runtime
            .cuda_set_device(self.owner.runtime.0.ordinal)?;
        let dgemm = self.dgemm_function()?;

        reserve_async_operation(&self.in_flight)?;
        let owner = Rc::new(CublasAsyncDoubleOperationOwner {
            _handle: self.owner.clone(),
            in_flight: Rc::clone(&self.in_flight),
            scalars: [alpha, beta],
        });
        let external_owner: Rc<dyn Any> = owner.clone();
        batch.retain_external_operation(&[a, b, c], external_owner)?;
        if !inherited_reservation {
            batch.register_queue_marker(Rc::clone(&self.queue_marker))?;
        }

        // SAFETY: f64 extents, leading dimensions, runtime/device, aliasing, and stream were
        // validated; the batch retains all three leases, the handle, and both scalar addresses
        // until terminal completion, including when cuBLAS reports a submission error.
        let status = unsafe {
            dgemm(
                self.owner.handle,
                if transpose_a {
                    CUBLAS_OPERATION_TRANSPOSE
                } else {
                    CUBLAS_OPERATION_NONE
                },
                if transpose_b {
                    CUBLAS_OPERATION_TRANSPOSE
                } else {
                    CUBLAS_OPERATION_NONE
                },
                m,
                n,
                k,
                std::ptr::from_ref(&owner.scalars[0]),
                a.allocation.pointer.cast(),
                lda,
                b.allocation.pointer.cast(),
                ldb,
                std::ptr::from_ref(&owner.scalars[1]),
                c.allocation.pointer.cast(),
                ldc,
            )
        };
        if status != CUBLAS_SUCCESS {
            return Err(CublasError::Status {
                operation: "cublasDgemm_v2",
                code: status,
            });
        }
        Ok(())
    }

    /// Compute column-major `C = alpha * op(A) * op(B) + beta * C` and wait for device completion.
    /// Leading dimensions and allocation extents are validated before entering the C ABI.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid dimensions, buffers from another device, concurrent buffer
    /// use, or a CUDA/cuBLAS failure.
    // The argument names follow the standardized BLAS SGEMM signature.
    #[allow(
        clippy::many_single_char_names,
        clippy::similar_names,
        clippy::too_many_arguments
    )]
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
    ) -> Result<(), CublasError> {
        self.sgemm_impl(
            transpose_a,
            transpose_b,
            m,
            n,
            k,
            alpha,
            a,
            lda,
            b,
            ldb,
            beta,
            c,
            ldc,
            &mut NoopSgemmTiming,
        )
    }

    /// Like [`Self::sgemm`], with opt-in host timings for its setup, C call, synchronization,
    /// and final lease disposition. Timings are reset before the call and remain available when
    /// the operation returns an error.
    ///
    /// # Errors
    ///
    /// Returns the same validation, runtime, concurrency, cuBLAS, or CUDA errors as [`Self::sgemm`].
    #[allow(
        clippy::many_single_char_names,
        clippy::similar_names,
        clippy::too_many_arguments
    )]
    pub fn sgemm_profiled(
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
        timing: &mut CublasSgemmHostTiming,
    ) -> Result<(), CublasError> {
        *timing = CublasSgemmHostTiming::default();
        self.sgemm_impl(
            transpose_a,
            transpose_b,
            m,
            n,
            k,
            alpha,
            a,
            lda,
            b,
            ldb,
            beta,
            c,
            ldc,
            &mut CollectSgemmTiming(timing),
        )
    }

    /// Compute column-major f64 `C = alpha * op(A) * op(B) + beta * C` and wait for device
    /// completion. Matrix extents are checked in bytes using the double-precision element size.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid dimensions, buffers from another device, concurrent buffer
    /// use, a missing DGEMM symbol, or a CUDA/cuBLAS failure.
    // Mirror the public BLAS ABI and keep leasing, synchronization, and quarantine together.
    #[allow(
        clippy::many_single_char_names,
        clippy::similar_names,
        clippy::too_many_arguments,
        clippy::too_many_lines
    )]
    pub fn dgemm(
        &self,
        transpose_a: bool,
        transpose_b: bool,
        m: usize,
        n: usize,
        k: usize,
        alpha: f64,
        a: &DeviceBuffer,
        lda: usize,
        b: &DeviceBuffer,
        ldb: usize,
        beta: f64,
        c: &DeviceBuffer,
        ldc: usize,
    ) -> Result<(), CublasError> {
        self.ensure_idle()?;
        let (a_rows, a_cols) = if transpose_a { (k, m) } else { (m, k) };
        let (b_rows, b_cols) = if transpose_b { (n, k) } else { (k, n) };
        matrix_bytes_f64("A", a_rows, a_cols, lda, a.len())?;
        matrix_bytes_f64("B", b_rows, b_cols, ldb, b.len())?;
        matrix_bytes_f64("C", m, n, ldc, c.len())?;
        for buffer in [a, b, c] {
            self.owner
                .runtime
                .ensure_same_runtime(&buffer.allocation.runtime)
                .map_err(|_| CublasError::DifferentRuntime)?;
        }
        if Rc::ptr_eq(&a.allocation, &b.allocation)
            || Rc::ptr_eq(&a.allocation, &c.allocation)
            || Rc::ptr_eq(&b.allocation, &c.allocation)
        {
            return Err(CublasError::AliasedBuffers);
        }
        let a_lease = a.acquire_access().map_err(|_| CublasError::Busy)?;
        let b_lease = b.acquire_access().map_err(|_| CublasError::Busy)?;
        let c_lease = c.acquire_access().map_err(|_| CublasError::Busy)?;
        let (m, n, k, lda, ldb, ldc) = (
            c_int::try_from(m).map_err(|_| CublasError::DimensionOverflow)?,
            c_int::try_from(n).map_err(|_| CublasError::DimensionOverflow)?,
            c_int::try_from(k).map_err(|_| CublasError::DimensionOverflow)?,
            c_int::try_from(lda).map_err(|_| CublasError::DimensionOverflow)?,
            c_int::try_from(ldb).map_err(|_| CublasError::DimensionOverflow)?,
            c_int::try_from(ldc).map_err(|_| CublasError::DimensionOverflow)?,
        );
        self.owner
            .runtime
            .cuda_set_device(self.owner.runtime.0.ordinal)?;
        let dgemm = self.dgemm_function()?;
        // SAFETY: f64 extents, leading dimensions, runtime/device, and non-aliasing were checked;
        // all buffers are leased and alpha/beta remain stable until the synchronous call and
        // terminal device wait have both returned.
        let status = unsafe {
            dgemm(
                self.owner.handle,
                if transpose_a {
                    CUBLAS_OPERATION_TRANSPOSE
                } else {
                    CUBLAS_OPERATION_NONE
                },
                if transpose_b {
                    CUBLAS_OPERATION_TRANSPOSE
                } else {
                    CUBLAS_OPERATION_NONE
                },
                m,
                n,
                k,
                std::ptr::from_ref(&alpha),
                a.allocation.pointer.cast(),
                lda,
                b.allocation.pointer.cast(),
                ldb,
                std::ptr::from_ref(&beta),
                c.allocation.pointer.cast(),
                ldc,
            )
        };
        let sync = self
            .owner
            .runtime
            .call("cudaDeviceSynchronize", |f: DeviceSynchronize| unsafe {
                f()
            });
        if let Err(error) = sync {
            self.owner.poisoned.set(true);
            std::mem::forget(a_lease);
            std::mem::forget(b_lease);
            std::mem::forget(c_lease);
            return Err(error.into());
        }
        drop(a_lease);
        drop(b_lease);
        drop(c_lease);
        if status != CUBLAS_SUCCESS {
            return Err(CublasError::Status {
                operation: "cublasDgemm_v2",
                code: status,
            });
        }
        Ok(())
    }

    #[allow(
        clippy::many_single_char_names,
        clippy::similar_names,
        clippy::too_many_arguments,
        clippy::too_many_lines // Keep lease acquisition, synchronization, and quarantine visible together.
    )]
    fn sgemm_impl<T: SgemmTimingSink>(
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
        timing: &mut T,
    ) -> Result<(), CublasError> {
        let preflight_mark = timing.begin(SgemmPhase::Preflight);
        let preflight = (|| {
            self.ensure_idle()?;
            let (a_rows, a_cols) = if transpose_a { (k, m) } else { (m, k) };
            let (b_rows, b_cols) = if transpose_b { (n, k) } else { (k, n) };
            matrix_bytes("A", a_rows, a_cols, lda, a.len())?;
            matrix_bytes("B", b_rows, b_cols, ldb, b.len())?;
            matrix_bytes("C", m, n, ldc, c.len())?;
            self.owner
                .runtime
                .ensure_same_runtime(&a.allocation.runtime)
                .map_err(|_| CublasError::DifferentRuntime)?;
            self.owner
                .runtime
                .ensure_same_runtime(&b.allocation.runtime)
                .map_err(|_| CublasError::DifferentRuntime)?;
            self.owner
                .runtime
                .ensure_same_runtime(&c.allocation.runtime)
                .map_err(|_| CublasError::DifferentRuntime)?;
            if Rc::ptr_eq(&a.allocation, &b.allocation)
                || Rc::ptr_eq(&a.allocation, &c.allocation)
                || Rc::ptr_eq(&b.allocation, &c.allocation)
            {
                return Err(CublasError::AliasedBuffers);
            }
            let a_lease = a.acquire_access().map_err(|_| CublasError::Busy)?;
            let b_lease = b.acquire_access().map_err(|_| CublasError::Busy)?;
            let c_lease = c.acquire_access().map_err(|_| CublasError::Busy)?;
            let (m, n, k, lda, ldb, ldc) = (
                c_int::try_from(m).map_err(|_| CublasError::DimensionOverflow)?,
                c_int::try_from(n).map_err(|_| CublasError::DimensionOverflow)?,
                c_int::try_from(k).map_err(|_| CublasError::DimensionOverflow)?,
                c_int::try_from(lda).map_err(|_| CublasError::DimensionOverflow)?,
                c_int::try_from(ldb).map_err(|_| CublasError::DimensionOverflow)?,
                c_int::try_from(ldc).map_err(|_| CublasError::DimensionOverflow)?,
            );
            self.owner
                .runtime
                .cuda_set_device(self.owner.runtime.0.ordinal)?;
            // SAFETY: signature follows cublasSgemm_v2 in cublas.h; buffers are validated allocations
            // from this runtime/device and the scalar pointers remain live through the call.
            let sgemm =
                unsafe { crate::ffi::symbol::<Sgemm>(&self.owner.library, SYM_CUBLAS_SGEMM_V2) }
                    .map_err(|e| CublasError::MissingSymbol {
                        symbol: "cublasSgemm_v2",
                        detail: e.to_string(),
                    })?;
            Ok((a_lease, b_lease, c_lease, m, n, k, lda, ldb, ldc, *sgemm))
        })();
        timing.finish(SgemmPhase::Preflight, preflight_mark);
        let (a_lease, b_lease, c_lease, m, n, k, lda, ldb, ldc, sgemm) = preflight?;
        let call_mark = timing.begin(SgemmPhase::CublasCall);
        let status = unsafe {
            sgemm(
                self.owner.handle,
                if transpose_a {
                    CUBLAS_OPERATION_TRANSPOSE
                } else {
                    CUBLAS_OPERATION_NONE
                },
                if transpose_b {
                    CUBLAS_OPERATION_TRANSPOSE
                } else {
                    CUBLAS_OPERATION_NONE
                },
                m,
                n,
                k,
                std::ptr::from_ref(&alpha),
                a.allocation.pointer.cast(),
                lda,
                b.allocation.pointer.cast(),
                ldb,
                std::ptr::from_ref(&beta),
                c.allocation.pointer.cast(),
                ldc,
            )
        };
        timing.finish(SgemmPhase::CublasCall, call_mark);
        // cuBLAS enqueues on its default stream. Device synchronization makes this API explicitly
        // synchronous and ensures all borrowed buffer owners remain valid until completion. Even a
        // cuBLAS error is followed by synchronization because the call may have partially queued.
        let sync_mark = timing.begin(SgemmPhase::DeviceSynchronize);
        let sync = self
            .owner
            .runtime
            .call("cudaDeviceSynchronize", |f: DeviceSynchronize| unsafe {
                f()
            });
        timing.finish(SgemmPhase::DeviceSynchronize, sync_mark);
        let cleanup_mark = timing.begin(SgemmPhase::Cleanup);
        if let Err(error) = sync {
            // Completion is now unknown. Retain every allocation and the library/handle forever;
            // freeing any of them could race device work that CUDA failed to confirm had stopped.
            self.owner.poisoned.set(true);
            std::mem::forget(a_lease);
            std::mem::forget(b_lease);
            std::mem::forget(c_lease);
            timing.finish(SgemmPhase::Cleanup, cleanup_mark);
            return Err(error.into());
        }
        drop(a_lease);
        drop(b_lease);
        drop(c_lease);
        timing.finish(SgemmPhase::Cleanup, cleanup_mark);
        if status != CUBLAS_SUCCESS {
            return Err(CublasError::Status {
                operation: "cublasSgemm_v2",
                code: status,
            });
        }
        Ok(())
    }

    /// Compute `result[0] = scale * dot(x, y)` on the device and wait for completion.
    ///
    /// The handle's pointer mode is temporarily set to device mode and restored before return.
    /// All vectors use positive element strides; `result` must hold one f32 and must not alias
    /// either input. The scalar result remains device resident.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid extents or strides, a different runtime/device, buffer use,
    /// or a CUDA/cuBLAS failure. If completion or pointer-mode restoration is uncertain, the
    /// handle is poisoned and cannot admit further work.
    #[allow(
        clippy::similar_names,
        clippy::too_many_arguments,
        clippy::too_many_lines
    )]
    pub fn sdot_scaled(
        &self,
        n: usize,
        x: &DeviceBuffer,
        incx: usize,
        y: &DeviceBuffer,
        incy: usize,
        scale: f32,
        result: &DeviceBuffer,
    ) -> Result<(), CublasError> {
        self.ensure_idle()?;
        if n == 0 || incx == 0 || incy == 0 {
            return Err(CublasError::InvalidVector(
                "length and strides must be positive",
            ));
        }
        let x_required = vector_bytes(n, incx)?;
        let y_required = vector_bytes(n, incy)?;
        for (name, required, allocation) in [("x", x_required, x.len()), ("y", y_required, y.len())]
        {
            if required > allocation {
                return Err(CublasError::BufferTooSmall {
                    matrix: name,
                    allocation,
                    required,
                });
            }
        }
        if result.len() < size_of::<f32>() {
            return Err(CublasError::BufferTooSmall {
                matrix: "result",
                allocation: result.len(),
                required: size_of::<f32>(),
            });
        }
        for buffer in [x, y, result] {
            self.owner
                .runtime
                .ensure_same_runtime(&buffer.allocation.runtime)
                .map_err(|_| CublasError::DifferentRuntime)?;
        }
        if Rc::ptr_eq(&x.allocation, &y.allocation)
            || Rc::ptr_eq(&x.allocation, &result.allocation)
            || Rc::ptr_eq(&y.allocation, &result.allocation)
        {
            return Err(CublasError::AliasedBuffers);
        }
        let x_lease = x.acquire_access().map_err(|_| CublasError::Busy)?;
        let y_lease = y.acquire_access().map_err(|_| CublasError::Busy)?;
        let result_lease = result.acquire_access().map_err(|_| CublasError::Busy)?;
        let (n, incx, incy) = (
            c_int::try_from(n).map_err(|_| CublasError::DimensionOverflow)?,
            c_int::try_from(incx).map_err(|_| CublasError::DimensionOverflow)?,
            c_int::try_from(incy).map_err(|_| CublasError::DimensionOverflow)?,
        );
        self.owner
            .runtime
            .cuda_set_device(self.owner.runtime.0.ordinal)?;
        // SAFETY: symbols match cuBLAS public declarations; live validated device allocations
        // remain leased until synchronization confirms completion.
        let get_mode = unsafe {
            crate::ffi::symbol::<GetPointerMode>(
                &self.owner.library,
                SYM_CUBLAS_GET_POINTER_MODE_V2,
            )
        }
        .map_err(|e| CublasError::MissingSymbol {
            symbol: "cublasGetPointerMode_v2",
            detail: e.to_string(),
        })?;
        let set_mode = unsafe {
            crate::ffi::symbol::<SetPointerMode>(
                &self.owner.library,
                SYM_CUBLAS_SET_POINTER_MODE_V2,
            )
        }
        .map_err(|e| CublasError::MissingSymbol {
            symbol: "cublasSetPointerMode_v2",
            detail: e.to_string(),
        })?;
        let sdot = unsafe { crate::ffi::symbol::<Sdot>(&self.owner.library, SYM_CUBLAS_SDOT_V2) }
            .map_err(|e| CublasError::MissingSymbol {
            symbol: "cublasSdot_v2",
            detail: e.to_string(),
        })?;
        let sscal =
            unsafe { crate::ffi::symbol::<Sscal>(&self.owner.library, SYM_CUBLAS_SSCAL_V2) }
                .map_err(|e| CublasError::MissingSymbol {
                    symbol: "cublasSscal_v2",
                    detail: e.to_string(),
                })?;
        let mut prior_mode = CUBLAS_POINTER_MODE_HOST;
        let get_status = unsafe { get_mode(self.owner.handle, &raw mut prior_mode) };
        if get_status != CUBLAS_SUCCESS {
            return Err(CublasError::Status {
                operation: "cublasGetPointerMode_v2",
                code: get_status,
            });
        }
        let set_status = unsafe { set_mode(self.owner.handle, CUBLAS_POINTER_MODE_DEVICE) };
        if set_status != CUBLAS_SUCCESS {
            return Err(CublasError::Status {
                operation: "cublasSetPointerMode_v2",
                code: set_status,
            });
        }
        let dot_status = unsafe {
            sdot(
                self.owner.handle,
                n,
                x.allocation.pointer.cast(),
                incx,
                y.allocation.pointer.cast(),
                incy,
                result.allocation.pointer.cast(),
            )
        };
        let scale_status = if dot_status == CUBLAS_SUCCESS {
            // cublasSscal_v2's alpha is a host scalar. Restore host mode for this call, then
            // restore the mode observed on entry after it has captured alpha.
            let host_status = unsafe { set_mode(self.owner.handle, CUBLAS_POINTER_MODE_HOST) };
            if host_status == CUBLAS_SUCCESS {
                unsafe {
                    sscal(
                        self.owner.handle,
                        1,
                        std::ptr::from_ref(&scale),
                        result.allocation.pointer.cast(),
                        1,
                    )
                }
            } else {
                host_status
            }
        } else {
            CUBLAS_SUCCESS
        };
        let restore_status = unsafe { set_mode(self.owner.handle, prior_mode) };
        let sync = self
            .owner
            .runtime
            .call("cudaDeviceSynchronize", |f: DeviceSynchronize| unsafe {
                f()
            });
        if let Err(error) = sync {
            self.owner.poisoned.set(true);
            std::mem::forget(x_lease);
            std::mem::forget(y_lease);
            std::mem::forget(result_lease);
            return Err(error.into());
        }
        if restore_status != CUBLAS_SUCCESS {
            self.owner.poisoned.set(true);
            return Err(CublasError::Status {
                operation: "cublasSetPointerMode_v2(restore)",
                code: restore_status,
            });
        }
        if dot_status != CUBLAS_SUCCESS {
            return Err(CublasError::Status {
                operation: "cublasSdot_v2",
                code: dot_status,
            });
        }
        if scale_status != CUBLAS_SUCCESS {
            return Err(CublasError::Status {
                operation: "cublasSscal_v2",
                code: scale_status,
            });
        }
        Ok(())
    }

    /// Compute `result[0] = scale * sum(abs(x))` on the device and wait for completion.
    ///
    /// cuBLAS writes the reduction result through a device pointer, then scales that scalar in
    /// place. This is suitable for reducing MSE's squared-difference buffer without allocating a
    /// same-length vector of ones. For finite nonnegative values, the mathematical reduction is
    /// equivalent to a dot product with ones; cuBLAS does not promise bitwise-identical reduction
    /// order or NaN payload behavior relative to [`Self::sdot_scaled`].
    ///
    /// # Errors
    ///
    /// Returns an error for invalid extents or strides, a different runtime/device, buffer use,
    /// or a CUDA/cuBLAS failure. If completion or pointer-mode restoration is uncertain, the
    /// handle is poisoned and cannot admit further work.
    #[allow(clippy::too_many_lines)]
    pub fn sasum_scaled(
        &self,
        n: usize,
        x: &DeviceBuffer,
        incx: usize,
        scale: f32,
        result: &DeviceBuffer,
    ) -> Result<(), CublasError> {
        self.ensure_idle()?;
        if n == 0 || incx == 0 {
            return Err(CublasError::InvalidVector(
                "length and strides must be positive",
            ));
        }
        let x_required = vector_bytes(n, incx)?;
        if x_required > x.len() {
            return Err(CublasError::BufferTooSmall {
                matrix: "x",
                allocation: x.len(),
                required: x_required,
            });
        }
        if result.len() < size_of::<f32>() {
            return Err(CublasError::BufferTooSmall {
                matrix: "result",
                allocation: result.len(),
                required: size_of::<f32>(),
            });
        }
        for buffer in [x, result] {
            self.owner
                .runtime
                .ensure_same_runtime(&buffer.allocation.runtime)
                .map_err(|_| CublasError::DifferentRuntime)?;
        }
        if Rc::ptr_eq(&x.allocation, &result.allocation) {
            return Err(CublasError::AliasedBuffers);
        }
        let x_lease = x.acquire_access().map_err(|_| CublasError::Busy)?;
        let result_lease = result.acquire_access().map_err(|_| CublasError::Busy)?;
        let (n, incx) = (
            c_int::try_from(n).map_err(|_| CublasError::DimensionOverflow)?,
            c_int::try_from(incx).map_err(|_| CublasError::DimensionOverflow)?,
        );
        self.owner
            .runtime
            .cuda_set_device(self.owner.runtime.0.ordinal)?;
        // SAFETY: symbols match cuBLAS public declarations; leased allocations remain alive
        // until synchronization confirms the operation has completed.
        let get_mode = unsafe {
            crate::ffi::symbol::<GetPointerMode>(
                &self.owner.library,
                SYM_CUBLAS_GET_POINTER_MODE_V2,
            )
        }
        .map_err(|e| CublasError::MissingSymbol {
            symbol: "cublasGetPointerMode_v2",
            detail: e.to_string(),
        })?;
        let set_mode = unsafe {
            crate::ffi::symbol::<SetPointerMode>(
                &self.owner.library,
                SYM_CUBLAS_SET_POINTER_MODE_V2,
            )
        }
        .map_err(|e| CublasError::MissingSymbol {
            symbol: "cublasSetPointerMode_v2",
            detail: e.to_string(),
        })?;
        let sasum =
            unsafe { crate::ffi::symbol::<Sasum>(&self.owner.library, SYM_CUBLAS_SASUM_V2) }
                .map_err(|e| CublasError::MissingSymbol {
                    symbol: "cublasSasum_v2",
                    detail: e.to_string(),
                })?;
        let sscal =
            unsafe { crate::ffi::symbol::<Sscal>(&self.owner.library, SYM_CUBLAS_SSCAL_V2) }
                .map_err(|e| CublasError::MissingSymbol {
                    symbol: "cublasSscal_v2",
                    detail: e.to_string(),
                })?;
        let mut prior_mode = CUBLAS_POINTER_MODE_HOST;
        let get_status = unsafe { get_mode(self.owner.handle, &raw mut prior_mode) };
        if get_status != CUBLAS_SUCCESS {
            return Err(CublasError::Status {
                operation: "cublasGetPointerMode_v2",
                code: get_status,
            });
        }
        let set_status = unsafe { set_mode(self.owner.handle, CUBLAS_POINTER_MODE_DEVICE) };
        if set_status != CUBLAS_SUCCESS {
            return Err(CublasError::Status {
                operation: "cublasSetPointerMode_v2",
                code: set_status,
            });
        }
        let reduction_status = unsafe {
            sasum(
                self.owner.handle,
                n,
                x.allocation.pointer.cast(),
                incx,
                result.allocation.pointer.cast(),
            )
        };
        let scale_status = if reduction_status == CUBLAS_SUCCESS {
            let host_status = unsafe { set_mode(self.owner.handle, CUBLAS_POINTER_MODE_HOST) };
            if host_status == CUBLAS_SUCCESS {
                unsafe {
                    sscal(
                        self.owner.handle,
                        1,
                        std::ptr::from_ref(&scale),
                        result.allocation.pointer.cast(),
                        1,
                    )
                }
            } else {
                host_status
            }
        } else {
            CUBLAS_SUCCESS
        };
        let restore_status = unsafe { set_mode(self.owner.handle, prior_mode) };
        let sync = self
            .owner
            .runtime
            .call("cudaDeviceSynchronize", |f: DeviceSynchronize| unsafe {
                f()
            });
        if let Err(error) = sync {
            self.owner.poisoned.set(true);
            std::mem::forget(x_lease);
            std::mem::forget(result_lease);
            return Err(error.into());
        }
        if restore_status != CUBLAS_SUCCESS {
            self.owner.poisoned.set(true);
            return Err(CublasError::Status {
                operation: "cublasSetPointerMode_v2(restore)",
                code: restore_status,
            });
        }
        if reduction_status != CUBLAS_SUCCESS {
            return Err(CublasError::Status {
                operation: "cublasSasum_v2",
                code: reduction_status,
            });
        }
        if scale_status != CUBLAS_SUCCESS {
            return Err(CublasError::Status {
                operation: "cublasSscal_v2",
                code: scale_status,
            });
        }
        Ok(())
    }
}

impl Drop for CublasHandleOwner {
    fn drop(&mut self) {
        if self.poisoned.get() {
            std::mem::forget(self.library.clone());
            return;
        }
        if self
            .runtime
            .cuda_set_device(self.runtime.0.ordinal)
            .is_err()
        {
            std::mem::forget(self.library.clone());
            return;
        }
        // SAFETY: handle came from cublasCreate_v2 and library is retained until this drop.
        if let Ok(destroy) =
            unsafe { crate::ffi::symbol::<DestroyHandle>(&self.library, SYM_CUBLAS_DESTROY_V2) }
        {
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
) -> Result<usize, CublasError> {
    if leading < rows.max(1) {
        return Err(CublasError::InvalidDimensions(
            "leading dimension is smaller than max(1, stored row count)",
        ));
    }
    if columns == 0 {
        return Ok(0);
    }
    let elements = leading
        .checked_mul(columns)
        .ok_or(CublasError::DimensionOverflow)?;
    let required = elements
        .checked_mul(size_of::<f32>())
        .ok_or(CublasError::DimensionOverflow)?;
    if required > allocation {
        return Err(CublasError::BufferTooSmall {
            matrix,
            allocation,
            required,
        });
    }
    Ok(required)
}

fn matrix_bytes_f64(
    matrix: &'static str,
    rows: usize,
    columns: usize,
    leading: usize,
    allocation: usize,
) -> Result<usize, CublasError> {
    if leading < rows.max(1) {
        return Err(CublasError::InvalidDimensions(
            "leading dimension is smaller than max(1, stored row count)",
        ));
    }
    if columns == 0 {
        return Ok(0);
    }
    let elements = leading
        .checked_mul(columns)
        .ok_or(CublasError::DimensionOverflow)?;
    let required = elements
        .checked_mul(size_of::<f64>())
        .ok_or(CublasError::DimensionOverflow)?;
    if required > allocation {
        return Err(CublasError::BufferTooSmall {
            matrix,
            allocation,
            required,
        });
    }
    Ok(required)
}

fn vector_bytes(n: usize, stride: usize) -> Result<usize, CublasError> {
    if n == 0 || stride == 0 {
        return Err(CublasError::InvalidVector(
            "length and strides must be positive",
        ));
    }
    let elements = (n - 1)
        .checked_mul(stride)
        .and_then(|v| v.checked_add(1))
        .ok_or(CublasError::DimensionOverflow)?;
    elements
        .checked_mul(size_of::<f32>())
        .ok_or(CublasError::DimensionOverflow)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matrix_extent_checks_leading_dimension_and_overflow() {
        assert!(matches!(
            matrix_bytes("A", 3, 2, 2, 1024),
            Err(CublasError::InvalidDimensions(_))
        ));
        assert_eq!(matrix_bytes("B", 0, 4, 1, 16), Ok(16));
        assert_eq!(matrix_bytes("B", 2, 0, 2, 0), Ok(0));
        assert_eq!(matrix_bytes("C", 2, 3, 2, 24), Ok(24));
        assert!(matches!(
            matrix_bytes("C", 2, 3, 2, 20),
            Err(CublasError::BufferTooSmall { .. })
        ));
        assert!(matches!(
            matrix_bytes("C", 2, usize::MAX, 2, 0),
            Err(CublasError::DimensionOverflow)
        ));
    }

    #[test]
    fn double_matrix_extent_uses_eight_byte_elements_and_checks_dimensions() {
        assert_eq!(matrix_bytes_f64("A", 2, 3, 2, 48), Ok(48));
        assert!(matches!(
            matrix_bytes_f64("C", 2, 3, 2, 40),
            Err(CublasError::BufferTooSmall {
                required: 48,
                allocation: 40,
                ..
            })
        ));
        assert!(matches!(
            matrix_bytes_f64("B", 3, 2, 2, 48),
            Err(CublasError::InvalidDimensions(_))
        ));
        assert!(matches!(
            matrix_bytes_f64("C", 1, usize::MAX / size_of::<f64>() + 1, 1, 0),
            Err(CublasError::DimensionOverflow)
        ));
    }

    #[test]
    #[ignore = "requires CUDA device and cuBLAS library"]
    // One fixture exercises both entry points against the same independent column-major oracle.
    #[allow(
        clippy::too_many_lines // Keep sync, batch lifetime, terminal wait, and readback together.
    )]
    fn dgemm_sync_and_batch_preserve_f64_values_and_completion_ownership() {
        let runtime = CudaRuntime::new(0).expect("CUDA runtime");
        let stream = runtime.create_stream().expect("create DGEMM stream");
        let mut cublas = Cublas::new(&runtime).expect("load cuBLAS");
        cublas.bind_stream(&stream).expect("bind DGEMM stream");

        // Column-major A contains values that f32 cannot preserve. B is identity, and C ensures
        // beta is also exercised in the independent host oracle.
        let a_values = [16_777_217.0_f64, 2.0, 1.0e-10, -3.5];
        let b_values = [1.0_f64, 0.0, 0.0, 1.0];
        let c_values = [5.0_f64, 6.0, 7.0, 8.0];
        let encode = |values: &[f64]| {
            values
                .iter()
                .flat_map(|value| value.to_ne_bytes())
                .collect::<Vec<_>>()
        };
        let expected = |alpha: f64, beta: f64| {
            let mut output = [0.0_f64; 4];
            for column in 0..2 {
                for row in 0..2 {
                    let mut sum = 0.0_f64;
                    for inner in 0..2 {
                        let product = a_values[inner * 2 + row] * b_values[column * 2 + inner];
                        sum += product;
                    }
                    let scaled_product = alpha * sum;
                    let scaled_initial = beta * c_values[column * 2 + row];
                    output[column * 2 + row] = scaled_product + scaled_initial;
                }
            }
            output
        };
        let read_output = |buffer: &DeviceBuffer| {
            let mut bytes = [0_u8; 4 * size_of::<f64>()];
            buffer.copy_to(&mut bytes).expect("read DGEMM output");
            bytes
                .as_chunks::<{ size_of::<f64>() }>()
                .0
                .iter()
                .map(|chunk| f64::from_ne_bytes(*chunk))
                .collect::<Vec<_>>()
        };
        let verify = |actual: &[f64], alpha: f64, beta: f64| {
            let expected = expected(alpha, beta);
            for (actual, expected) in actual.iter().zip(expected) {
                let tolerance = 2.0 * f64::EPSILON * expected.abs().max(1.0);
                assert!(
                    (actual - expected).abs() <= tolerance,
                    "DGEMM value {actual} differs from independent result {expected}"
                );
            }
            assert_ne!(a_values[0].to_bits(), f64::from(16_777_217.0_f32).to_bits());
            assert_ne!(a_values[2].to_bits(), f64::from(1.0e-10_f32).to_bits());
        };

        let mut a = runtime.allocate(4 * size_of::<f64>()).expect("allocate A");
        let mut b = runtime.allocate(4 * size_of::<f64>()).expect("allocate B");
        let mut sync_c = runtime
            .allocate(4 * size_of::<f64>())
            .expect("allocate sync C");
        a.copy_from(&encode(&a_values)).expect("upload A");
        b.copy_from(&encode(&b_values)).expect("upload B");
        sync_c
            .copy_from(&encode(&c_values))
            .expect("initialize sync C");

        cublas
            .dgemm(false, false, 2, 2, 2, 1.0, &a, 2, &b, 2, 0.0, &sync_c, 2)
            .expect("synchronous DGEMM");
        let sync_output = read_output(&sync_c);
        verify(&sync_output, 1.0, 0.0);
        assert_eq!(sync_output[0].to_bits(), a_values[0].to_bits());
        assert_eq!(sync_output[2].to_bits(), a_values[2].to_bits());

        let mut batch_c = runtime
            .allocate(4 * size_of::<f64>())
            .expect("allocate batch C");
        batch_c
            .copy_from(&encode(&c_values))
            .expect("initialize batch C");
        let mut batch = CudaCompletionBatch::new(&stream);
        cublas
            .dgemm_into_batch(
                &mut batch, false, false, 2, 2, 2, 1.5, &a, 2, &b, 2, -0.25, &batch_c, 2,
            )
            .expect("enqueue batched DGEMM");
        assert!(!cublas.is_usable());
        assert!(matches!(
            batch_c.copy_to(&mut [0_u8; 4 * size_of::<f64>()]),
            Err(CudaError::Busy)
        ));
        let mut completion = batch.finish().expect("finish DGEMM batch");
        drop(cublas);
        completion.wait().expect("wait for DGEMM completion");
        verify(&read_output(&batch_c), 1.5, -0.25);
    }

    #[test]
    fn vector_extent_checks_stride_and_overflow() {
        assert_eq!(vector_bytes(1, 8), Ok(4));
        assert_eq!(vector_bytes(4, 2), Ok(28));
        assert!(matches!(
            vector_bytes(0, 1),
            Err(CublasError::InvalidVector(_))
        ));
        assert!(matches!(
            vector_bytes(usize::MAX, 2),
            Err(CublasError::DimensionOverflow)
        ));
    }

    #[test]
    fn async_owner_keeps_handle_owner_and_busy_state_until_released() {
        let handle_owner = Rc::new(());
        let weak_owner = Rc::downgrade(&handle_owner);
        let in_flight = Rc::new(Cell::new(1));
        let operation_owner = Rc::new(CublasAsyncOperationOwner {
            _handle: handle_owner.clone(),
            in_flight: Rc::clone(&in_flight),
            scalars: [1.0, 0.0],
        });
        let retained_owner: Rc<dyn Any> = operation_owner.clone();
        drop(handle_owner);
        drop(operation_owner);

        assert!(weak_owner.upgrade().is_some());
        assert_eq!(in_flight.get(), 1);
        drop(retained_owner);
        assert!(weak_owner.upgrade().is_none());
        assert_eq!(in_flight.get(), 0);
    }

    #[test]
    fn double_async_owner_retains_f64_scalars_until_batch_release() {
        let handle_owner = Rc::new(());
        let weak_handle = Rc::downgrade(&handle_owner);
        let in_flight = Rc::new(Cell::new(0));
        reserve_async_operation(&in_flight).unwrap();
        let owner = Rc::new(CublasAsyncDoubleOperationOwner {
            _handle: handle_owner.clone(),
            in_flight: Rc::clone(&in_flight),
            scalars: [f64::from_bits(0x3ff0_0000_0000_0001), -0.0],
        });
        let weak_owner = Rc::downgrade(&owner);
        let retained_owner: Rc<dyn Any> = owner.clone();
        drop(handle_owner);
        drop(owner);

        assert!(weak_handle.upgrade().is_some());
        assert_eq!(in_flight.get(), 1);
        let retained = weak_owner.upgrade().expect("retained operation");
        assert_eq!(retained.scalars[0].to_bits(), 0x3ff0_0000_0000_0001);
        assert_eq!(retained.scalars[1].to_bits(), (-0.0_f64).to_bits());
        drop(retained);
        drop(retained_owner);
        assert!(weak_handle.upgrade().is_none());
        assert!(weak_owner.upgrade().is_none());
        assert_eq!(in_flight.get(), 0);
    }

    #[test]
    fn same_handle_operation_count_stays_busy_until_every_batch_owner_drops() {
        let handle = Rc::new(());
        let in_flight = Rc::new(Cell::new(0));

        reserve_async_operation(&in_flight).unwrap();
        let first = Rc::new(CublasAsyncOperationOwner {
            _handle: handle.clone(),
            in_flight: Rc::clone(&in_flight),
            scalars: [1.0, 0.0],
        });
        let retained_first: Rc<dyn Any> = first.clone();
        drop(first);

        reserve_async_operation(&in_flight).unwrap();
        let second = Rc::new(CublasAsyncOperationOwner {
            _handle: handle,
            in_flight: Rc::clone(&in_flight),
            scalars: [1.0, 0.0],
        });
        let retained_second: Rc<dyn Any> = second.clone();
        drop(second);

        assert_eq!(in_flight.get(), 2);
        drop(retained_first);
        assert_eq!(in_flight.get(), 1);
        drop(retained_second);
        assert_eq!(in_flight.get(), 0);
    }

    #[test]
    fn async_operation_count_rejects_overflow_without_changing_state() {
        let in_flight = Cell::new(usize::MAX);
        assert_eq!(reserve_async_operation(&in_flight), Err(CublasError::Busy));
        assert_eq!(in_flight.get(), usize::MAX);
    }

    #[test]
    fn queue_extension_requires_live_matching_stream_and_inherited_marker() {
        assert!(queue_extension_allowed(false, true, true));
        assert!(!queue_extension_allowed(true, true, true));
        assert!(!queue_extension_allowed(false, false, true));
        assert!(!queue_extension_allowed(false, true, false));
    }

    #[test]
    #[ignore = "requires CUDA device and cuBLAS library"]
    fn completed_batch_cannot_authorize_an_unrelated_handle_reservation() {
        let runtime = CudaRuntime::new(0).expect("CUDA runtime");
        let stream = runtime.create_stream().expect("SGEMM stream");
        let mut handle = Cublas::new(&runtime).expect("cuBLAS handle");
        handle.bind_stream(&stream).expect("bound stream");
        let identity = [1.0_f32, 0.0, 0.0, 1.0]
            .into_iter()
            .flat_map(f32::to_ne_bytes)
            .collect::<Vec<_>>();
        let mut left = runtime.allocate(identity.len()).expect("left");
        let mut right = runtime.allocate(identity.len()).expect("right");
        let output = runtime.allocate(identity.len()).expect("output");
        left.copy_from(&identity).expect("upload left");
        right.copy_from(&identity).expect("upload right");
        let queue = |batch: &mut CudaCompletionBatch| {
            handle.sgemm_into_batch(
                batch, false, false, 2, 2, 2, 1.0, &left, 2, &right, 2, 0.0, &output, 2,
            )
        };

        let mut first = CudaCompletionBatch::new(&stream);
        queue(&mut first).expect("first reservation");
        let mut completed = first.finish().expect("first completion");
        completed.wait().expect("retire first reservation");
        assert!(handle.is_usable());

        let other_stream = runtime.create_stream().expect("unrelated buffer stream");
        let buffer_reservation = output
            .acquire_stream_access(&other_stream)
            .expect("reserve output on another stream");
        let mut rejected = CudaCompletionBatch::new(&stream);
        assert!(queue(&mut rejected).is_err());
        assert!(!rejected.has_queue_marker(&handle.queue_marker));
        assert!(handle.is_usable());
        drop(buffer_reservation);

        let mut live = CudaCompletionBatch::new(&stream);
        queue(&mut live).expect("unrelated live reservation");
        assert!(matches!(queue(&mut rejected), Err(CublasError::Busy)));
        let mut stale = CudaCompletionBatch::new(&stream);
        stale
            .wait_for_batch(&mut completed)
            .expect("adopt completed batch");
        assert!(matches!(queue(&mut stale), Err(CublasError::Busy)));
        assert!(!handle.is_usable());
        live.finish()
            .expect("live completion")
            .wait()
            .expect("retire live reservation");
        assert!(handle.is_usable());
        let mut actual = vec![0; identity.len()];
        output.copy_to(&mut actual).expect("read output");
        assert_eq!(actual, identity);
    }

    #[test]
    #[ignore = "requires CUDA device and cuBLAS library"]
    fn cuda_gpu_async_sgemm_retains_handle_and_buffers_until_batch_completion() {
        let runtime = CudaRuntime::new(0).expect("CUDA GPU CUDA runtime");
        let info = runtime.device_info().expect("selected CUDA device info");
        assert!(!info.name.is_empty(), "unexpected GPU: {}", info.name);
        let stream = runtime.create_stream().expect("create SGEMM stream");
        let mut cublas = Cublas::new(&runtime).expect("load cuBLAS");
        cublas.bind_stream(&stream).expect("bind SGEMM stream");
        let a_values = [1.0_f32, 3.0, 2.0, 4.0];
        let b_values = [1.0_f32, 0.0, 0.0, 1.0];
        let zero_values = [0.0_f32; 4];
        let encode = |values: &[f32]| {
            values
                .iter()
                .flat_map(|value| value.to_ne_bytes())
                .collect::<Vec<_>>()
        };
        let mut a = runtime.allocate(16).expect("allocate A");
        let mut b = runtime.allocate(16).expect("allocate B");
        let mut c = runtime.allocate(16).expect("allocate C");
        a.copy_from(&encode(&a_values)).expect("upload A");
        b.copy_from(&encode(&b_values)).expect("upload B");
        c.copy_from(&encode(&zero_values)).expect("initialize C");

        let mut batch = CudaCompletionBatch::new(&stream);
        cublas
            .sgemm_into_batch(
                &mut batch, false, false, 2, 2, 2, 1.0, &a, 2, &b, 2, 0.0, &c, 2,
            )
            .expect("queue asynchronous SGEMM");
        assert!(!cublas.is_usable());
        assert!(matches!(c.copy_to(&mut [0_u8; 16]), Err(CudaError::Busy)));
        let mut completion = batch.finish().expect("finish SGEMM batch");
        drop(cublas);
        completion.wait().expect("wait for SGEMM completion");

        let mut actual = [0_u8; 16];
        c.copy_to(&mut actual).expect("read SGEMM output");
        let actual = actual
            .as_chunks::<{ size_of::<f32>() }>()
            .0
            .iter()
            .map(|chunk| f32::from_ne_bytes(*chunk))
            .collect::<Vec<_>>();
        assert_eq!(actual, a_values);
    }
}
