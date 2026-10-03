//! Small dynamically loaded rocBLAS GEMM and vector-operation adapters.
//!
//! This uses rocBLAS' public `rocblas_sgemm` ABI directly and does not require hipBLASLt.

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

#[rustfmt::skip]
use crate::ffi::{
    Library,
    load_uncached_library,
};

#[rustfmt::skip]
use super::{
    DeviceBuffer,
    HipCompletionBatch,
    HipError,
    HipRuntime,
    HipStreamHandle,
};

#[rustfmt::skip]
use crate::ffi::rocblas::{
    RocblasHandle,
    Dgemm,
    ROCBLAS_SUCCESS,
    ROCBLAS_OPERATION_NONE,
    ROCBLAS_OPERATION_TRANSPOSE,
    ROCBLAS_POINTER_MODE_HOST,
    ROCBLAS_POINTER_MODE_DEVICE,
};

/// Failures returned by the rocBLAS GEMM and vector adapters.
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
pub struct RocblasSgemmHostTiming {
    /// Validation, leases, device selection, and retained function-pointer access.
    pub preflight: std::time::Duration,
    /// Time spent in the `rocblas_sgemm` C function.
    pub rocblas_call: std::time::Duration,
    /// Time spent in `hipDeviceSynchronize`.
    pub device_synchronize: std::time::Duration,
    /// Lease release or quarantine disposition after synchronization.
    pub cleanup: std::time::Duration,
}

#[derive(Clone, Copy)]
enum SgemmPhase {
    Preflight,
    RocblasCall,
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

struct CollectSgemmTiming<'a>(&'a mut RocblasSgemmHostTiming);

impl SgemmTimingSink for CollectSgemmTiming<'_> {
    type Mark = Instant;

    fn begin(&mut self, _: SgemmPhase) -> Self::Mark {
        Instant::now()
    }

    fn finish(&mut self, phase: SgemmPhase, mark: Self::Mark) {
        let elapsed = mark.elapsed();
        match phase {
            SgemmPhase::Preflight => self.0.preflight = elapsed,
            SgemmPhase::RocblasCall => self.0.rocblas_call = elapsed,
            SgemmPhase::DeviceSynchronize => self.0.device_synchronize = elapsed,
            SgemmPhase::Cleanup => self.0.cleanup = elapsed,
        }
    }
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
            Self::DifferentStream => f.write_str("rocBLAS handle is not bound to the batch stream"),
            Self::Busy => f.write_str("rocBLAS buffer is busy with another device operation"),
            Self::AliasedBuffers => {
                f.write_str("rocBLAS operation does not accept aliased input/output allocations")
            }
            Self::CompletionUnknown => {
                f.write_str("a previous rocBLAS operation did not confirm device completion")
            }
            Self::InvalidDimensions(why) => write!(f, "invalid rocBLAS GEMM dimensions: {why}"),
            Self::DimensionOverflow => {
                f.write_str("rocBLAS dimension or matrix size overflows the supported range")
            }
            Self::BufferTooSmall {
                matrix,
                allocation,
                required,
            } => write!(
                f,
                "rocBLAS GEMM {matrix} needs {required} bytes, allocation has {allocation}"
            ),
            Self::InvalidVector(why) => write!(f, "invalid rocBLAS vector reduction: {why}"),
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
    owner: Rc<RocblasHandleOwner>,
    in_flight: Rc<Cell<usize>>,
    queue_marker: Rc<dyn Any>,
    bound_stream: Option<HipStreamHandle>,
}

struct RocblasHandleOwner {
    runtime: HipRuntime,
    library: Arc<Library>,
    handle: RocblasHandle,
    vectors: crate::ffi::VectorFunctions,
    functions: crate::ffi::BlasHandleFunctions,
    poisoned: Cell<bool>,
    dgemm: OnceCell<Result<Dgemm, RocblasError>>,
}

struct RocblasAsyncOperationOwner {
    _handle: Rc<dyn Any>,
    in_flight: Rc<Cell<usize>>,
    scalars: [f32; 2],
}

struct RocblasAsyncDoubleOperationOwner {
    _handle: Rc<dyn Any>,
    in_flight: Rc<Cell<usize>>,
    scalars: [f64; 2],
}

impl Drop for RocblasAsyncDoubleOperationOwner {
    fn drop(&mut self) {
        release_async_operation(&self.in_flight);
    }
}

impl Drop for RocblasAsyncOperationOwner {
    fn drop(&mut self) {
        release_async_operation(&self.in_flight);
    }
}

fn reserve_async_operation(in_flight: &Cell<usize>) -> Result<(), RocblasError> {
    let next = in_flight.get().checked_add(1).ok_or(RocblasError::Busy)?;
    in_flight.set(next);
    Ok(())
}

fn release_async_operation(in_flight: &Cell<usize>) {
    let active = in_flight.get();
    debug_assert!(active > 0, "rocBLAS async operation count underflow");
    in_flight.set(active.saturating_sub(1));
}

const fn queue_extension_allowed(
    poisoned: bool,
    same_stream: bool,
    has_reservation_marker: bool,
) -> bool {
    !poisoned && same_stream && has_reservation_marker
}

impl Rocblas {
    fn ensure_idle(&self) -> Result<(), RocblasError> {
        if self.owner.poisoned.get() {
            return Err(RocblasError::CompletionUnknown);
        }
        if self.in_flight.get() != 0 {
            return Err(RocblasError::Busy);
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
        stream: &HipStreamHandle,
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
    ) -> Result<(), RocblasError> {
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
        stream: &HipStreamHandle,
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
    ) -> Result<(), RocblasError> {
        if self.owner.poisoned.get() {
            return Err(RocblasError::CompletionUnknown);
        }
        if !stream.belongs_to_runtime(&self.owner.runtime) {
            return Err(RocblasError::DifferentRuntime);
        }
        if !self
            .bound_stream
            .as_ref()
            .is_some_and(|bound| Rc::ptr_eq(&bound.inner, &stream.inner))
        {
            return Err(RocblasError::DifferentStream);
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
                .map_err(|_| RocblasError::DifferentRuntime)?;
        }
        if Rc::ptr_eq(&a.allocation, &b.allocation)
            || Rc::ptr_eq(&a.allocation, &c.allocation)
            || Rc::ptr_eq(&b.allocation, &c.allocation)
        {
            return Err(RocblasError::AliasedBuffers);
        }
        for dimension in [m, n, k, lda, ldb, ldc] {
            c_int::try_from(dimension).map_err(|_| RocblasError::DimensionOverflow)?;
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
        stream: &HipStreamHandle,
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
    ) -> Result<(), RocblasError> {
        if self.owner.poisoned.get() {
            return Err(RocblasError::CompletionUnknown);
        }
        if !stream.belongs_to_runtime(&self.owner.runtime) {
            return Err(RocblasError::DifferentRuntime);
        }
        if !self
            .bound_stream
            .as_ref()
            .is_some_and(|bound| Rc::ptr_eq(&bound.inner, &stream.inner))
        {
            return Err(RocblasError::DifferentStream);
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
                .map_err(|_| RocblasError::DifferentRuntime)?;
        }
        if Rc::ptr_eq(&a.allocation, &b.allocation)
            || Rc::ptr_eq(&a.allocation, &c.allocation)
            || Rc::ptr_eq(&b.allocation, &c.allocation)
        {
            return Err(RocblasError::AliasedBuffers);
        }
        for dimension in [m, n, k, lda, ldb, ldc] {
            c_int::try_from(dimension).map_err(|_| RocblasError::DimensionOverflow)?;
        }
        Ok(())
    }

    fn dgemm_function(&self) -> Result<Dgemm, RocblasError> {
        self.owner
            .dgemm
            .get_or_init(|| {
                // SAFETY: Dgemm matches the installed public rocBLAS C ABI. The owning library
                // outlives this cached function pointer and every operation that invokes it.
                unsafe { crate::ffi::require_rocblas_dgemm(&self.owner.library) }
                    .map(|symbol| *symbol)
                    .map_err(|error| RocblasError::MissingSymbol {
                        symbol: "rocblas_dgemm",
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
    pub fn require_dgemm_support(&self) -> Result<(), RocblasError> {
        self.dgemm_function().map(|_| ())
    }

    // Error-only quarantine owns the exact queue as well as the vendor handle, library and
    // runtime/context. Allocation leases retain data separately. A later cache/session drop
    // cannot destroy the bound stream or code roots while completion remains unknown.
    fn retain_unknown_completion(&self) {
        self.owner.poisoned.set(true);
        std::mem::forget(Rc::clone(&self.owner));
        if let Some(stream) = &self.bound_stream {
            std::mem::forget(stream.clone());
        }
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
    pub(crate) fn can_extend_from(&self, completion: &super::HipBatchCompletion) -> bool {
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
    pub(crate) fn can_extend_in_batch(&self, batch: &HipCompletionBatch) -> bool {
        queue_extension_allowed(
            self.owner.poisoned.get(),
            self.bound_stream
                .as_ref()
                .is_some_and(|stream| Rc::ptr_eq(&stream.inner, &batch.stream_handle().inner)),
            batch.has_queue_marker(&self.queue_marker),
        )
    }

    /// Load `librocblas.so` (or `ROCBLAS_LIBRARY`) and create a handle for `runtime`'s device.
    ///
    /// # Errors
    ///
    /// Returns an error when rocBLAS cannot be loaded, a required symbol is missing, or HIP/rocBLAS
    /// fails to create the handle.
    pub fn new(runtime: &HipRuntime) -> Result<Self, RocblasError> {
        let candidates: Vec<OsString> = std::env::var_os("ROCBLAS_LIBRARY").map_or_else(
            || vec!["librocblas.so".into(), "librocblas.so.5".into()],
            |path| vec![path],
        );
        let mut last_error = None;
        for candidate in candidates {
            // SAFETY: rocBLAS exports the documented C ABI. Arc keeps it loaded for handle life.
            let library = match unsafe { load_uncached_library(&candidate) } {
                Ok(library) => Arc::new(library),
                Err(error) => {
                    last_error = Some(error.clone());
                    continue;
                }
            };
            runtime.hip_set_device(runtime.0.device)?;
            let mut handle = ptr::null_mut();
            // SAFETY: symbol type matches rocblas_create_handle's C declaration.
            let create =
                unsafe { crate::ffi::require_rocblas_create_handle(&library) }.map_err(|e| {
                    RocblasError::MissingSymbol {
                        symbol: "rocblas_create_handle",
                        detail: e.to_string(),
                    }
                })?;
            // SAFETY: retain the required ABI table before handle creation; missing entries
            // fail cold and the owner keeps this library live through every later invocation.
            let functions = unsafe { crate::ffi::retain_blas_handle_functions(&library) }.map_err(
                |(symbol, error)| RocblasError::MissingSymbol {
                    symbol,
                    detail: error.to_string(),
                },
            )?;
            // SAFETY: typed SDK declarations are paired with exact symbol names. The
            // handle owner stores this immutable table beside its retained library owner.
            let vectors = unsafe { crate::ffi::retain_blas_vector_functions(&library) }.map_err(
                |(symbol, error)| RocblasError::MissingSymbol {
                    symbol,
                    detail: error.to_string(),
                },
            )?;
            let status =
                unsafe { crate::ffi::invoke_rocblas_create_handle(&create, &raw mut handle) };
            if status != ROCBLAS_SUCCESS {
                return Err(RocblasError::Status {
                    operation: "rocblas_create_handle",
                    code: status,
                });
            }
            return Ok(Self {
                owner: Rc::new(RocblasHandleOwner {
                    runtime: runtime.clone(),
                    library,
                    handle,
                    vectors,
                    functions,
                    poisoned: Cell::new(false),
                    dgemm: OnceCell::new(),
                }),
                in_flight: Rc::new(Cell::new(0)),
                queue_marker: Rc::new(()) as Rc<dyn Any>,
                bound_stream: None,
            });
        }
        Err(RocblasError::LibraryUnavailable(
            last_error.unwrap_or_else(|| "no library candidates".into()),
        ))
    }

    /// Binds this handle to a selected HIP stream before graph operations are submitted.
    ///
    /// The handle retains the stream owner. Call only while no rocBLAS operation is in flight;
    /// the tensor adapter binds once during preparation and keeps its synchronous behavior.
    ///
    /// # Errors
    ///
    /// Returns an identity, missing-symbol, or rocBLAS status error.
    pub fn bind_stream(&mut self, stream: &HipStreamHandle) -> Result<(), RocblasError> {
        self.ensure_idle()?;
        if self.bound_stream.is_some() {
            return Err(RocblasError::Busy);
        }
        if !stream.belongs_to_runtime(&self.owner.runtime) {
            return Err(RocblasError::DifferentRuntime);
        }
        self.owner
            .runtime
            .hip_set_device(self.owner.runtime.0.device)?;
        // The cold handle table and retained library outlive this stream binding.
        let set_stream = self.owner.functions.set_stream;
        let status = unsafe {
            crate::ffi::invoke_rocblas_set_stream(&set_stream, self.owner.handle, stream.inner.raw)
        };
        if status != ROCBLAS_SUCCESS {
            return Err(RocblasError::Status {
                operation: "rocblas_set_stream",
                code: status,
            });
        }
        self.bound_stream = Some(stream.clone());
        Ok(())
    }

    /// Queue SGEMM on this handle's bound stream and retain its buffers and handle through the
    /// batch's final event. The handle remains unavailable until that completion is released.
    /// On a rocBLAS error the operation may have been partially queued; finish or drop the batch
    /// before attempting any other operation with this handle.
    ///
    /// # Errors
    ///
    /// Returns validation, runtime/stream, busy-buffer, or rocBLAS status errors. If this returns
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
        batch: &mut HipCompletionBatch,
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
            c_int::try_from(m).map_err(|_| RocblasError::DimensionOverflow)?,
            c_int::try_from(n).map_err(|_| RocblasError::DimensionOverflow)?,
            c_int::try_from(k).map_err(|_| RocblasError::DimensionOverflow)?,
            c_int::try_from(lda).map_err(|_| RocblasError::DimensionOverflow)?,
            c_int::try_from(ldb).map_err(|_| RocblasError::DimensionOverflow)?,
            c_int::try_from(ldc).map_err(|_| RocblasError::DimensionOverflow)?,
        );
        self.owner
            .runtime
            .hip_set_device(self.owner.runtime.0.device)?;
        let sgemm = self.owner.functions.sgemm;

        reserve_async_operation(&self.in_flight)?;
        let owner = Rc::new(RocblasAsyncOperationOwner {
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
        // completion. rocBLAS may enqueue work even when it reports an error, so do not release
        // the owner or leases on that path.
        let status = unsafe {
            crate::ffi::invoke_rocblas_sgemm(
                &sgemm,
                self.owner.handle,
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
        if status != ROCBLAS_SUCCESS {
            return Err(RocblasError::Status {
                operation: "rocblas_sgemm",
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
    /// Returns validation, runtime/stream, busy-buffer, missing-symbol, or rocBLAS status errors.
    /// If submission has retained the operation, finish or drop the batch normally so it can
    /// establish quiescence or quarantine its resources.
    // Preserve rocBLAS' operand naming and keep retention/completion steps visible at submission.
    #[allow(
        clippy::many_single_char_names,
        clippy::similar_names,
        clippy::too_many_arguments,
        clippy::too_many_lines
    )]
    pub fn dgemm_into_batch(
        &self,
        batch: &mut HipCompletionBatch,
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
    ) -> Result<(), RocblasError> {
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
            c_int::try_from(m).map_err(|_| RocblasError::DimensionOverflow)?,
            c_int::try_from(n).map_err(|_| RocblasError::DimensionOverflow)?,
            c_int::try_from(k).map_err(|_| RocblasError::DimensionOverflow)?,
            c_int::try_from(lda).map_err(|_| RocblasError::DimensionOverflow)?,
            c_int::try_from(ldb).map_err(|_| RocblasError::DimensionOverflow)?,
            c_int::try_from(ldc).map_err(|_| RocblasError::DimensionOverflow)?,
        );
        self.owner
            .runtime
            .hip_set_device(self.owner.runtime.0.device)?;
        let dgemm = self.dgemm_function()?;

        reserve_async_operation(&self.in_flight)?;
        let owner = Rc::new(RocblasAsyncDoubleOperationOwner {
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
        // until terminal completion, including when rocBLAS reports a submission error.
        let status = unsafe {
            crate::ffi::invoke_rocblas_dgemm(
                &dgemm,
                self.owner.handle,
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
        if status != ROCBLAS_SUCCESS {
            return Err(RocblasError::Status {
                operation: "rocblas_dgemm",
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
    /// use, or a HIP/rocBLAS failure.
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
    ) -> Result<(), RocblasError> {
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
    /// Returns the same validation, runtime, concurrency, rocBLAS, or HIP errors as [`Self::sgemm`].
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
        timing: &mut RocblasSgemmHostTiming,
    ) -> Result<(), RocblasError> {
        *timing = RocblasSgemmHostTiming::default();
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
    /// use, a missing DGEMM symbol, or a HIP/rocBLAS failure.
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
    ) -> Result<(), RocblasError> {
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
                .map_err(|_| RocblasError::DifferentRuntime)?;
        }
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
        self.owner
            .runtime
            .hip_set_device(self.owner.runtime.0.device)?;
        let dgemm = self.dgemm_function()?;
        // SAFETY: f64 extents, leading dimensions, runtime/device, and non-aliasing were checked;
        // all buffers are leased and alpha/beta remain stable until the synchronous call and
        // terminal device wait have both returned.
        let status = unsafe {
            crate::ffi::invoke_rocblas_dgemm(
                &dgemm,
                self.owner.handle,
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
        let sync = unsafe { crate::ffi::invoke_hipDeviceSynchronize(&self.owner.runtime) };
        if let Err(error) = sync {
            self.retain_unknown_completion();
            std::mem::forget(a_lease);
            std::mem::forget(b_lease);
            std::mem::forget(c_lease);
            return Err(error.into());
        }
        drop(a_lease);
        drop(b_lease);
        drop(c_lease);
        if status != ROCBLAS_SUCCESS {
            return Err(RocblasError::Status {
                operation: "rocblas_dgemm",
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
    ) -> Result<(), RocblasError> {
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
                .map_err(|_| RocblasError::DifferentRuntime)?;
            self.owner
                .runtime
                .ensure_same_runtime(&b.allocation.runtime)
                .map_err(|_| RocblasError::DifferentRuntime)?;
            self.owner
                .runtime
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
            self.owner
                .runtime
                .hip_set_device(self.owner.runtime.0.device)?;
            let sgemm = self.owner.functions.sgemm;
            Ok((a_lease, b_lease, c_lease, m, n, k, lda, ldb, ldc, sgemm))
        })();
        timing.finish(SgemmPhase::Preflight, preflight_mark);
        let (a_lease, b_lease, c_lease, m, n, k, lda, ldb, ldc, sgemm) = preflight?;
        let call_mark = timing.begin(SgemmPhase::RocblasCall);
        let status = unsafe {
            crate::ffi::invoke_rocblas_sgemm(
                &sgemm,
                self.owner.handle,
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
        timing.finish(SgemmPhase::RocblasCall, call_mark);
        // rocBLAS enqueues on its default stream. Device synchronization makes this API explicitly
        // synchronous and ensures all borrowed buffer owners remain valid until completion. Even a
        // rocBLAS error is followed by synchronization because the call may have partially queued.
        let sync_mark = timing.begin(SgemmPhase::DeviceSynchronize);
        let sync = unsafe { crate::ffi::invoke_hipDeviceSynchronize(&self.owner.runtime) };
        timing.finish(SgemmPhase::DeviceSynchronize, sync_mark);
        let cleanup_mark = timing.begin(SgemmPhase::Cleanup);
        if let Err(error) = sync {
            // Completion is now unknown. Retain every allocation and the library/handle forever;
            // freeing any of them could race device work that HIP failed to confirm had stopped.
            self.retain_unknown_completion();
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
        if status != ROCBLAS_SUCCESS {
            return Err(RocblasError::Status {
                operation: "rocblas_sgemm",
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
    /// or a HIP/rocBLAS failure. If completion or pointer-mode restoration is uncertain, the
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
    ) -> Result<(), RocblasError> {
        self.ensure_idle()?;
        if n == 0 || incx == 0 || incy == 0 {
            return Err(RocblasError::InvalidVector(
                "length and strides must be positive",
            ));
        }
        let x_required = vector_bytes(n, incx)?;
        let y_required = vector_bytes(n, incy)?;
        for (name, required, allocation) in [("x", x_required, x.len()), ("y", y_required, y.len())]
        {
            if required > allocation {
                return Err(RocblasError::BufferTooSmall {
                    matrix: name,
                    allocation,
                    required,
                });
            }
        }
        if result.len() < size_of::<f32>() {
            return Err(RocblasError::BufferTooSmall {
                matrix: "result",
                allocation: result.len(),
                required: size_of::<f32>(),
            });
        }
        for buffer in [x, y, result] {
            self.owner
                .runtime
                .ensure_same_runtime(&buffer.allocation.runtime)
                .map_err(|_| RocblasError::DifferentRuntime)?;
        }
        if Rc::ptr_eq(&x.allocation, &y.allocation)
            || Rc::ptr_eq(&x.allocation, &result.allocation)
            || Rc::ptr_eq(&y.allocation, &result.allocation)
        {
            return Err(RocblasError::AliasedBuffers);
        }
        let x_lease = x.acquire_access().map_err(|_| RocblasError::Busy)?;
        let y_lease = y.acquire_access().map_err(|_| RocblasError::Busy)?;
        let result_lease = result.acquire_access().map_err(|_| RocblasError::Busy)?;
        let (n, incx, incy) = (
            c_int::try_from(n).map_err(|_| RocblasError::DimensionOverflow)?,
            c_int::try_from(incx).map_err(|_| RocblasError::DimensionOverflow)?,
            c_int::try_from(incy).map_err(|_| RocblasError::DimensionOverflow)?,
        );
        self.owner
            .runtime
            .hip_set_device(self.owner.runtime.0.device)?;
        // All typed entries were validated cold; the handle/library and allocation leases
        // remain retained through synchronization or quarantine.
        let get_mode = self.owner.vectors.get_pointer_mode;
        let set_mode = self.owner.vectors.set_pointer_mode;
        let sdot = self.owner.functions.sdot;
        let sscal = self.owner.vectors.sscal;
        let mut prior_mode = ROCBLAS_POINTER_MODE_HOST;
        let get_status = unsafe {
            crate::ffi::invoke_rocblas_get_pointer_mode(
                &get_mode,
                self.owner.handle,
                &raw mut prior_mode,
            )
        };
        if get_status != ROCBLAS_SUCCESS {
            return Err(RocblasError::Status {
                operation: "rocblas_get_pointer_mode",
                code: get_status,
            });
        }
        let set_status = unsafe {
            crate::ffi::invoke_rocblas_set_pointer_mode(
                &set_mode,
                self.owner.handle,
                ROCBLAS_POINTER_MODE_DEVICE,
            )
        };
        if set_status != ROCBLAS_SUCCESS {
            return Err(RocblasError::Status {
                operation: "rocblas_set_pointer_mode",
                code: set_status,
            });
        }
        let dot_status = unsafe {
            crate::ffi::invoke_rocblas_sdot(
                &sdot,
                self.owner.handle,
                n,
                x.allocation.pointer.cast(),
                incx,
                y.allocation.pointer.cast(),
                incy,
                result.allocation.pointer.cast(),
            )
        };
        let scale_status = if dot_status == ROCBLAS_SUCCESS {
            // rocblas_sscal's alpha is a host scalar. Restore host mode for this call, then
            // restore the mode observed on entry after it has captured alpha.
            let host_status = unsafe {
                crate::ffi::invoke_rocblas_set_pointer_mode(
                    &set_mode,
                    self.owner.handle,
                    ROCBLAS_POINTER_MODE_HOST,
                )
            };
            if host_status == ROCBLAS_SUCCESS {
                unsafe {
                    crate::ffi::invoke_rocblas_sscal(
                        &sscal,
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
            ROCBLAS_SUCCESS
        };
        let restore_status = unsafe {
            crate::ffi::invoke_rocblas_set_pointer_mode(&set_mode, self.owner.handle, prior_mode)
        };
        let sync = unsafe { crate::ffi::invoke_hipDeviceSynchronize(&self.owner.runtime) };
        if let Err(error) = sync {
            self.retain_unknown_completion();
            std::mem::forget(x_lease);
            std::mem::forget(y_lease);
            std::mem::forget(result_lease);
            return Err(error.into());
        }
        if restore_status != ROCBLAS_SUCCESS {
            self.owner.poisoned.set(true);
            return Err(RocblasError::Status {
                operation: "rocblas_set_pointer_mode(restore)",
                code: restore_status,
            });
        }
        if dot_status != ROCBLAS_SUCCESS {
            return Err(RocblasError::Status {
                operation: "rocblas_sdot",
                code: dot_status,
            });
        }
        if scale_status != ROCBLAS_SUCCESS {
            return Err(RocblasError::Status {
                operation: "rocblas_sscal",
                code: scale_status,
            });
        }
        Ok(())
    }

    /// Compute `result[0] = scale * sum(abs(x))` on the device and wait for completion.
    ///
    /// rocBLAS writes the reduction result through a device pointer, then scales that scalar in
    /// place. This is suitable for reducing MSE's squared-difference buffer without allocating a
    /// same-length vector of ones. For finite nonnegative values, the mathematical reduction is
    /// equivalent to a dot product with ones; rocBLAS does not promise bitwise-identical reduction
    /// order or NaN payload behavior relative to [`Self::sdot_scaled`].
    ///
    /// # Errors
    ///
    /// Returns an error for invalid extents or strides, a different runtime/device, buffer use,
    /// or a HIP/rocBLAS failure. If completion or pointer-mode restoration is uncertain, the
    /// handle is poisoned and cannot admit further work.
    #[allow(clippy::too_many_lines)]
    pub fn sasum_scaled(
        &self,
        n: usize,
        x: &DeviceBuffer,
        incx: usize,
        scale: f32,
        result: &DeviceBuffer,
    ) -> Result<(), RocblasError> {
        self.ensure_idle()?;
        if n == 0 || incx == 0 {
            return Err(RocblasError::InvalidVector(
                "length and strides must be positive",
            ));
        }
        let x_required = vector_bytes(n, incx)?;
        if x_required > x.len() {
            return Err(RocblasError::BufferTooSmall {
                matrix: "x",
                allocation: x.len(),
                required: x_required,
            });
        }
        if result.len() < size_of::<f32>() {
            return Err(RocblasError::BufferTooSmall {
                matrix: "result",
                allocation: result.len(),
                required: size_of::<f32>(),
            });
        }
        for buffer in [x, result] {
            self.owner
                .runtime
                .ensure_same_runtime(&buffer.allocation.runtime)
                .map_err(|_| RocblasError::DifferentRuntime)?;
        }
        if Rc::ptr_eq(&x.allocation, &result.allocation) {
            return Err(RocblasError::AliasedBuffers);
        }
        let x_lease = x.acquire_access().map_err(|_| RocblasError::Busy)?;
        let result_lease = result.acquire_access().map_err(|_| RocblasError::Busy)?;
        let (n, incx) = (
            c_int::try_from(n).map_err(|_| RocblasError::DimensionOverflow)?,
            c_int::try_from(incx).map_err(|_| RocblasError::DimensionOverflow)?,
        );
        self.owner
            .runtime
            .hip_set_device(self.owner.runtime.0.device)?;
        // SAFETY: symbols match rocBLAS public declarations; leased allocations remain alive
        // until synchronization confirms the operation has completed.
        // No warm symbol lookup: cold handle admission owns all four exact entrypoints.
        let get_mode = self.owner.vectors.get_pointer_mode;
        let set_mode = self.owner.vectors.set_pointer_mode;
        let sasum = self.owner.vectors.sasum;
        let sscal = self.owner.vectors.sscal;
        let mut prior_mode = ROCBLAS_POINTER_MODE_HOST;
        let get_status = unsafe {
            crate::ffi::invoke_rocblas_get_pointer_mode(
                &get_mode,
                self.owner.handle,
                &raw mut prior_mode,
            )
        };
        if get_status != ROCBLAS_SUCCESS {
            return Err(RocblasError::Status {
                operation: "rocblas_get_pointer_mode",
                code: get_status,
            });
        }
        let set_status = unsafe {
            crate::ffi::invoke_rocblas_set_pointer_mode(
                &set_mode,
                self.owner.handle,
                ROCBLAS_POINTER_MODE_DEVICE,
            )
        };
        if set_status != ROCBLAS_SUCCESS {
            return Err(RocblasError::Status {
                operation: "rocblas_set_pointer_mode",
                code: set_status,
            });
        }
        let reduction_status = unsafe {
            crate::ffi::invoke_rocblas_sasum(
                &sasum,
                self.owner.handle,
                n,
                x.allocation.pointer.cast(),
                incx,
                result.allocation.pointer.cast(),
            )
        };
        let scale_status = if reduction_status == ROCBLAS_SUCCESS {
            let host_status = unsafe {
                crate::ffi::invoke_rocblas_set_pointer_mode(
                    &set_mode,
                    self.owner.handle,
                    ROCBLAS_POINTER_MODE_HOST,
                )
            };
            if host_status == ROCBLAS_SUCCESS {
                unsafe {
                    crate::ffi::invoke_rocblas_sscal(
                        &sscal,
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
            ROCBLAS_SUCCESS
        };
        let restore_status = unsafe {
            crate::ffi::invoke_rocblas_set_pointer_mode(&set_mode, self.owner.handle, prior_mode)
        };
        let sync = unsafe { crate::ffi::invoke_hipDeviceSynchronize(&self.owner.runtime) };
        if let Err(error) = sync {
            self.retain_unknown_completion();
            std::mem::forget(x_lease);
            std::mem::forget(result_lease);
            return Err(error.into());
        }
        if restore_status != ROCBLAS_SUCCESS {
            self.owner.poisoned.set(true);
            return Err(RocblasError::Status {
                operation: "rocblas_set_pointer_mode(restore)",
                code: restore_status,
            });
        }
        if reduction_status != ROCBLAS_SUCCESS {
            return Err(RocblasError::Status {
                operation: "rocblas_sasum",
                code: reduction_status,
            });
        }
        if scale_status != ROCBLAS_SUCCESS {
            return Err(RocblasError::Status {
                operation: "rocblas_sscal",
                code: scale_status,
            });
        }
        Ok(())
    }

    /// Reduce F64 magnitudes into a device scalar, then scale in F64.
    ///
    /// Typed double ASUM/SCAL preserve F64 storage and arithmetic. Reduction order and
    /// special-value propagation remain native library behavior, independently of strict
    /// checking or portable reproducibility. The selected handle configuration is retained.
    ///
    /// # Errors
    /// Rejects invalid extents, strides, aliases, conflicting access and different runtimes.
    /// Library errors restore pointer mode after terminal completion; uncertain completion
    /// quarantines leases and poisons the handle exactly as [`Self::sasum_scaled`].
    #[allow(clippy::too_many_lines)]
    pub fn dasum_scaled(
        &self,
        n: usize,
        x: &DeviceBuffer,
        incx: usize,
        scale: f64,
        result: &DeviceBuffer,
    ) -> Result<(), RocblasError> {
        self.ensure_idle()?;
        if n == 0 || incx == 0 {
            return Err(RocblasError::InvalidVector(
                "length and strides must be positive",
            ));
        }
        let x_required = vector_bytes_for_size(n, incx, size_of::<f64>())?;
        if x_required > x.len() {
            return Err(RocblasError::BufferTooSmall {
                matrix: "x",
                allocation: x.len(),
                required: x_required,
            });
        }
        if result.len() < size_of::<f64>() {
            return Err(RocblasError::BufferTooSmall {
                matrix: "result",
                allocation: result.len(),
                required: size_of::<f64>(),
            });
        }
        for buffer in [x, result] {
            self.owner
                .runtime
                .ensure_same_runtime(&buffer.allocation.runtime)
                .map_err(|_| RocblasError::DifferentRuntime)?;
        }
        if Rc::ptr_eq(&x.allocation, &result.allocation) {
            return Err(RocblasError::AliasedBuffers);
        }
        let x_lease = x.acquire_access().map_err(|_| RocblasError::Busy)?;
        let result_lease = result.acquire_access().map_err(|_| RocblasError::Busy)?;
        let (n, incx) = (
            c_int::try_from(n).map_err(|_| RocblasError::DimensionOverflow)?,
            c_int::try_from(incx).map_err(|_| RocblasError::DimensionOverflow)?,
        );
        self.owner
            .runtime
            .hip_set_device(self.owner.runtime.0.device)?;
        // SAFETY: symbols match rocBLAS public declarations; leased allocations remain alive
        // until synchronization confirms the operation has completed.
        // No warm symbol lookup: cold handle admission owns all four exact entrypoints.
        let get_mode = self.owner.vectors.get_pointer_mode;
        let set_mode = self.owner.vectors.set_pointer_mode;
        let dasum = self.owner.vectors.dasum;
        let dscal = self.owner.vectors.dscal;
        let mut prior_mode = ROCBLAS_POINTER_MODE_HOST;
        let get_status = unsafe {
            crate::ffi::invoke_rocblas_get_pointer_mode(
                &get_mode,
                self.owner.handle,
                &raw mut prior_mode,
            )
        };
        if get_status != ROCBLAS_SUCCESS {
            return Err(RocblasError::Status {
                operation: "rocblas_get_pointer_mode",
                code: get_status,
            });
        }
        let set_status = unsafe {
            crate::ffi::invoke_rocblas_set_pointer_mode(
                &set_mode,
                self.owner.handle,
                ROCBLAS_POINTER_MODE_DEVICE,
            )
        };
        if set_status != ROCBLAS_SUCCESS {
            return Err(RocblasError::Status {
                operation: "rocblas_set_pointer_mode",
                code: set_status,
            });
        }
        let reduction_status = unsafe {
            crate::ffi::invoke_rocblas_dasum(
                &dasum,
                self.owner.handle,
                n,
                x.allocation.pointer.cast(),
                incx,
                result.allocation.pointer.cast(),
            )
        };
        let scale_status = if reduction_status == ROCBLAS_SUCCESS {
            let host_status = unsafe {
                crate::ffi::invoke_rocblas_set_pointer_mode(
                    &set_mode,
                    self.owner.handle,
                    ROCBLAS_POINTER_MODE_HOST,
                )
            };
            if host_status == ROCBLAS_SUCCESS {
                unsafe {
                    crate::ffi::invoke_rocblas_dscal(
                        &dscal,
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
            ROCBLAS_SUCCESS
        };
        let restore_status = unsafe {
            crate::ffi::invoke_rocblas_set_pointer_mode(&set_mode, self.owner.handle, prior_mode)
        };
        let sync = unsafe { crate::ffi::invoke_hipDeviceSynchronize(&self.owner.runtime) };
        if let Err(error) = sync {
            self.retain_unknown_completion();
            std::mem::forget(x_lease);
            std::mem::forget(result_lease);
            return Err(error.into());
        }
        if restore_status != ROCBLAS_SUCCESS {
            self.owner.poisoned.set(true);
            return Err(RocblasError::Status {
                operation: "rocblas_set_pointer_mode(restore)",
                code: restore_status,
            });
        }
        if reduction_status != ROCBLAS_SUCCESS {
            return Err(RocblasError::Status {
                operation: "rocblas_dasum",
                code: reduction_status,
            });
        }
        if scale_status != ROCBLAS_SUCCESS {
            return Err(RocblasError::Status {
                operation: "rocblas_dscal",
                code: scale_status,
            });
        }
        Ok(())
    }
}

impl Drop for RocblasHandleOwner {
    fn drop(&mut self) {
        if self.poisoned.get() {
            std::mem::forget(self.library.clone());
            return;
        }
        if self.runtime.hip_set_device(self.runtime.0.device).is_err() {
            std::mem::forget(self.library.clone());
            return;
        }
        // SAFETY: handle came from rocblas_create_handle and library is retained until this drop.
        let _ = unsafe {
            crate::ffi::invoke_rocblas_destroy_handle(&self.functions.destroy, self.handle)
        };
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
        .checked_mul(size_of::<f32>())
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

fn matrix_bytes_f64(
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
        .checked_mul(size_of::<f64>())
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

fn vector_bytes(n: usize, stride: usize) -> Result<usize, RocblasError> {
    vector_bytes_for_size(n, stride, size_of::<f32>())
}

fn vector_bytes_for_size(
    n: usize,
    stride: usize,
    element_size: usize,
) -> Result<usize, RocblasError> {
    if n == 0 || stride == 0 {
        return Err(RocblasError::InvalidVector(
            "length and strides must be positive",
        ));
    }
    let elements = (n - 1)
        .checked_mul(stride)
        .and_then(|v| v.checked_add(1))
        .ok_or(RocblasError::DimensionOverflow)?;
    elements
        .checked_mul(element_size)
        .ok_or(RocblasError::DimensionOverflow)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn double_vector_extents_include_stride_and_detect_overflow() {
        assert_eq!(vector_bytes_for_size(1, 8, size_of::<f64>()).unwrap(), 8);
        assert_eq!(vector_bytes_for_size(4, 2, size_of::<f64>()).unwrap(), 56);
        assert!(vector_bytes_for_size(0, 1, size_of::<f64>()).is_err());
        assert!(vector_bytes_for_size(usize::MAX / 8 + 1, 1, size_of::<f64>()).is_err());
    }

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

    #[test]
    fn double_matrix_extent_uses_eight_byte_elements_and_checks_dimensions() {
        assert_eq!(matrix_bytes_f64("A", 2, 3, 2, 48), Ok(48));
        assert!(matches!(
            matrix_bytes_f64("C", 2, 3, 2, 40),
            Err(RocblasError::BufferTooSmall {
                required: 48,
                allocation: 40,
                ..
            })
        ));
        assert!(matches!(
            matrix_bytes_f64("B", 3, 2, 2, 48),
            Err(RocblasError::InvalidDimensions(_))
        ));
        assert!(matches!(
            matrix_bytes_f64("C", 1, usize::MAX / size_of::<f64>() + 1, 1, 0),
            Err(RocblasError::DimensionOverflow)
        ));
    }

    #[test]
    #[ignore = "requires ROCm device and rocBLAS library"]
    // One fixture exercises both entry points against the same independent column-major oracle.
    #[allow(
        clippy::too_many_lines // Keep sync, batch lifetime, terminal wait, and readback together.
    )]
    fn dgemm_sync_and_batch_preserve_f64_values_and_completion_ownership() {
        let runtime = HipRuntime::new(0).expect("HIP runtime");
        let stream = runtime.create_stream().expect("create DGEMM stream");
        let mut rocblas = Rocblas::new(&runtime).expect("load rocBLAS");
        rocblas.bind_stream(&stream).expect("bind DGEMM stream");

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

        rocblas
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
        let mut batch = HipCompletionBatch::new(&stream);
        rocblas
            .dgemm_into_batch(
                &mut batch, false, false, 2, 2, 2, 1.5, &a, 2, &b, 2, -0.25, &batch_c, 2,
            )
            .expect("enqueue batched DGEMM");
        assert!(!rocblas.is_usable());
        assert!(matches!(
            batch_c.copy_to(&mut [0_u8; 4 * size_of::<f64>()]),
            Err(HipError::Busy)
        ));
        let mut completion = batch.finish().expect("finish DGEMM batch");
        drop(rocblas);
        completion.wait().expect("wait for DGEMM completion");
        verify(&read_output(&batch_c), 1.5, -0.25);
    }

    #[test]
    #[cfg(feature = "allocation-census")]
    #[ignore = "requires authorized ROCm device and native BLAS library"]
    fn retained_sgemm_dot_stream_and_destroy_have_no_warm_symbol_resolution() {
        let runtime = HipRuntime::new(0).unwrap();
        let stream = runtime.create_stream().unwrap();
        let mut blas = Rocblas::new(&runtime).unwrap();
        let mut left = runtime.allocate(16).unwrap();
        let mut right = runtime.allocate(16).unwrap();
        let output = runtime.allocate(16).unwrap();
        let dot = runtime.allocate(4).unwrap();
        let encode = |values: [f32; 4]| {
            values
                .into_iter()
                .flat_map(f32::to_ne_bytes)
                .collect::<Vec<_>>()
        };
        right.copy_from(&encode([1.0, 0.0, 0.0, 1.0])).unwrap();
        crate::reset_rocm_api_census();
        blas.bind_stream(&stream).unwrap();
        for iteration in 0..8 {
            let values = if iteration % 2 == 0 {
                [1.0, 2.0, 3.0, 4.0]
            } else {
                [5.0, 6.0, 7.0, 8.0]
            };
            let bytes = encode(values);
            left.copy_from(&bytes).unwrap();
            blas.sgemm(
                false, false, 2, 2, 2, 1.0, &left, 2, &right, 2, 0.0, &output, 2,
            )
            .unwrap();
            let mut actual = [0; 16];
            output.copy_to(&mut actual).unwrap();
            assert_eq!(actual.as_slice(), bytes);
            blas.sdot_scaled(4, &left, 1, &right, 1, 0.5, &dot).unwrap();
            let mut actual_dot = [0; 4];
            dot.copy_to(&mut actual_dot).unwrap();
            assert_eq!(
                f32::from_ne_bytes(actual_dot).to_bits(),
                if iteration % 2 == 0 {
                    2.5_f32.to_bits()
                } else {
                    6.5_f32.to_bits()
                }
            );
        }
        drop(blas);
        let api = crate::rocm_api_census();
        assert_eq!(api.symbol_resolutions, 0);
        assert_eq!(api.module_loads, 0);
    }

    #[test]
    #[ignore = "requires native BLAS/device; deterministic error-only owner-retention witness"]
    fn synchronous_unknown_completion_retains_exact_handle_stream_roots() {
        for bound in [false, true] {
            let runtime = HipRuntime::new(0).unwrap();
            let stream = runtime.create_stream().unwrap();
            let mut blas = Rocblas::new(&runtime).unwrap();
            if bound {
                blas.bind_stream(&stream).unwrap();
            }
            let mut left = runtime.allocate(4).unwrap();
            let mut right = runtime.allocate(4).unwrap();
            let output = runtime.allocate(4).unwrap();
            left.copy_from(&2.0_f32.to_ne_bytes()).unwrap();
            right.copy_from(&3.0_f32.to_ne_bytes()).unwrap();
            let owner_count = Rc::strong_count(&blas.owner);
            let stream_count = Rc::strong_count(&stream.inner);
            blas.sgemm(
                false, false, 1, 1, 1, 1.0, &left, 1, &right, 1, 0.0, &output, 1,
            )
            .unwrap();
            let mut actual = [0; 4];
            output.copy_to(&mut actual).unwrap();
            assert_eq!(f32::from_ne_bytes(actual).to_bits(), 6.0_f32.to_bits());
            assert_eq!(Rc::strong_count(&blas.owner), owner_count);
            assert_eq!(Rc::strong_count(&stream.inner), stream_count);
            let owner = Rc::downgrade(&blas.owner);
            let library = Arc::downgrade(&blas.owner.library);
            let context = Arc::downgrade(&runtime.0);
            let queue = Rc::downgrade(&stream.inner);
            // Real SGEMM has already completed. Exercise the actual private unknown-result
            // retention helper deterministically; this is not an injected driver-loss test.
            blas.retain_unknown_completion();
            assert!(!blas.is_usable());
            assert!(matches!(
                blas.sgemm(
                    false, false, 1, 1, 1, 1.0, &left, 1, &right, 1, 0.0, &output, 1
                ),
                Err(RocblasError::CompletionUnknown)
            ));
            drop(blas);
            drop(left);
            drop(right);
            drop(output);
            drop(stream);
            drop(runtime);
            assert!(owner.upgrade().is_some());
            assert!(library.upgrade().is_some());
            assert!(context.upgrade().is_some());
            // NULL has no separate Rust stream owner; an unrelated created stream must drop.
            assert_eq!(queue.upgrade().is_some(), bound);
        }
    }

    #[test]
    fn vector_extent_checks_stride_and_overflow() {
        assert_eq!(vector_bytes(1, 8), Ok(4));
        assert_eq!(vector_bytes(4, 2), Ok(28));
        assert!(matches!(
            vector_bytes(0, 1),
            Err(RocblasError::InvalidVector(_))
        ));
        assert!(matches!(
            vector_bytes(usize::MAX, 2),
            Err(RocblasError::DimensionOverflow)
        ));
    }

    #[test]
    fn async_owner_keeps_handle_owner_and_busy_state_until_released() {
        let handle_owner = Rc::new(());
        let weak_owner = Rc::downgrade(&handle_owner);
        let in_flight = Rc::new(Cell::new(1));
        let operation_owner = Rc::new(RocblasAsyncOperationOwner {
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
        let owner = Rc::new(RocblasAsyncDoubleOperationOwner {
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
        let first = Rc::new(RocblasAsyncOperationOwner {
            _handle: handle.clone(),
            in_flight: Rc::clone(&in_flight),
            scalars: [1.0, 0.0],
        });
        let retained_first: Rc<dyn Any> = first.clone();
        drop(first);

        reserve_async_operation(&in_flight).unwrap();
        let second = Rc::new(RocblasAsyncOperationOwner {
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
        assert_eq!(reserve_async_operation(&in_flight), Err(RocblasError::Busy));
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
    #[ignore = "requires ROCm device and rocBLAS library"]
    fn completed_batch_cannot_authorize_an_unrelated_handle_reservation() {
        let runtime = HipRuntime::new(0).expect("HIP runtime");
        let stream = runtime.create_stream().expect("SGEMM stream");
        let mut handle = Rocblas::new(&runtime).expect("rocBLAS handle");
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
        let queue = |batch: &mut HipCompletionBatch| {
            handle.sgemm_into_batch(
                batch, false, false, 2, 2, 2, 1.0, &left, 2, &right, 2, 0.0, &output, 2,
            )
        };

        let mut first = HipCompletionBatch::new(&stream);
        queue(&mut first).expect("first reservation");
        let mut completed = first.finish().expect("first completion");
        completed.wait().expect("retire first reservation");
        assert!(handle.is_usable());

        let other_stream = runtime.create_stream().expect("unrelated buffer stream");
        let buffer_reservation = output
            .acquire_stream_access(&other_stream)
            .expect("reserve output on another stream");
        let mut rejected = HipCompletionBatch::new(&stream);
        assert!(queue(&mut rejected).is_err());
        assert!(!rejected.has_queue_marker(&handle.queue_marker));
        assert!(handle.is_usable());
        drop(buffer_reservation);

        let mut live = HipCompletionBatch::new(&stream);
        queue(&mut live).expect("unrelated live reservation");
        assert!(matches!(queue(&mut rejected), Err(RocblasError::Busy)));
        let mut stale = HipCompletionBatch::new(&stream);
        stale
            .wait_for_batch(&mut completed)
            .expect("adopt completed batch");
        assert!(matches!(queue(&mut stale), Err(RocblasError::Busy)));
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
    #[ignore = "requires ROCm device and rocBLAS library"]
    fn rx_6900_xt_async_sgemm_retains_handle_and_buffers_until_batch_completion() {
        let runtime = HipRuntime::new(0).expect("RX 6900 XT HIP runtime");
        let info = runtime.device_info().expect("selected HIP device info");
        assert!(
            info.name.contains("6900 XT"),
            "unexpected GPU: {}",
            info.name
        );
        let stream = runtime.create_stream().expect("create SGEMM stream");
        let mut rocblas = Rocblas::new(&runtime).expect("load rocBLAS");
        rocblas.bind_stream(&stream).expect("bind SGEMM stream");
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

        let mut batch = HipCompletionBatch::new(&stream);
        rocblas
            .sgemm_into_batch(
                &mut batch, false, false, 2, 2, 2, 1.0, &a, 2, &b, 2, 0.0, &c, 2,
            )
            .expect("queue asynchronous SGEMM");
        assert!(!rocblas.is_usable());
        assert!(matches!(c.copy_to(&mut [0_u8; 16]), Err(HipError::Busy)));
        let mut completion = batch.finish().expect("finish SGEMM batch");
        drop(rocblas);
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
