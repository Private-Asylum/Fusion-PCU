//! Diagnostic only: safety-patched upstream C endpoints in a private isolated library image.
//! This does not load the unmodified installed mlx-c or establish production coexistence.

#[rustfmt::skip]
use super::{
    dimensions,
    require_apple_silicon,
    status,
    text_value,
    Abi,
    ArrayNew,
    Free,
    Matmul,
    Read,
    SessionNew,
    TraceCount,
    Version,
    ERROR_BYTES,
    SDK_VERSION,
};
#[rustfmt::skip]
use crate::{
    MlxArrayResidency,
    MlxError,
};
use libloading::Library;
#[rustfmt::skip]
use std::{
    cell::Cell,
    ffi::{
        c_char,
        c_void,
    },
    path::Path,
    ptr::NonNull,
    rc::Rc,
};

type Prepare =
    unsafe extern "C" fn(*mut c_void, i32, i32, i32, *mut *mut c_void, *mut c_char) -> i32;

struct Api {
    _library: Library,
    session_new: SessionNew,
    session_free: Free,
    array_new: ArrayNew,
    array_free: Free,
    read: Read,
    direct: Matmul,
    prepare: Prepare,
    compiled: Matmul,
    prepared_free: Free,
    traces: TraceCount,
}

/// A diagnostic-only private C adapter, with literal-safe reporting and outer containment.
#[derive(Clone)]
pub struct Runtime(Rc<Api>);

impl Runtime {
    /// Loads the trusted ABI3 evaluation library built by this directory's `CMake` consumer.
    /// Its upstream C symbols are hidden; its permanent error handler belongs to that image.
    ///
    /// # Errors
    /// Returns unsupported platform, absent library, symbol, private ABI or exact SDK mismatch.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, MlxError> {
        require_apple_silicon()?;
        // SAFETY: the explicit trusted diagnostic library exposes the audited ABI3. The
        // library remains retained by every function pointer and opaque owner below.
        let library = unsafe { Library::new(path.as_ref()) }
            .map_err(|error| MlxError::Unavailable(error.to_string()))?;
        // SAFETY: signatures exactly match the eval_* exports in bridge.cpp; ABI is checked
        // before owners are opened. No mlx-c function is called directly across Rust FFI.
        let api = unsafe {
            let abi = *library.get::<Abi>(b"eval_abi\0").map_err(resolve)?;
            if abi() != 3 {
                return Err(MlxError::Abi("unsupported evaluation ABI".into()));
            }
            let version = *library.get::<Version>(b"eval_version\0").map_err(resolve)?;
            let mut text = [0; 64];
            let mut error = [0; ERROR_BYTES];
            let code = version(text.as_mut_ptr().cast(), error.as_mut_ptr().cast());
            if code == 2 {
                // A failed C string release retained a native handle before Api exists.
                std::mem::forget(library);
                return Err(MlxError::CompletionUnknown(text_value(&error)?));
            }
            status(code, &error)?;
            if text_value(&text)? != SDK_VERSION {
                return Err(MlxError::Abi("evaluation SDK mismatch".into()));
            }
            Api {
                session_new: *library
                    .get::<SessionNew>(b"eval_session_new\0")
                    .map_err(resolve)?,
                session_free: *library
                    .get::<Free>(b"eval_session_free\0")
                    .map_err(resolve)?,
                array_new: *library
                    .get::<ArrayNew>(b"eval_array_new\0")
                    .map_err(resolve)?,
                array_free: *library.get::<Free>(b"eval_array_free\0").map_err(resolve)?,
                read: *library.get::<Read>(b"eval_read\0").map_err(resolve)?,
                direct: *library.get::<Matmul>(b"eval_matmul\0").map_err(resolve)?,
                prepare: *library.get::<Prepare>(b"eval_prepare\0").map_err(resolve)?,
                compiled: *library.get::<Matmul>(b"eval_compiled\0").map_err(resolve)?,
                prepared_free: *library
                    .get::<Free>(b"eval_prepared_free\0")
                    .map_err(resolve)?,
                traces: *library
                    .get::<TraceCount>(b"eval_traces\0")
                    .map_err(resolve)?,
                _library: library,
            }
        };
        Ok(Self(Rc::new(api)))
    }

    /// Opens an explicit GPU stream; no CPU route or default-stream mutation.
    ///
    /// # Errors
    /// Returns absent GPU or contained C/SDK activation errors.
    pub fn open_gpu(&self, index: usize) -> Result<Session, MlxError> {
        let index = i32::try_from(index).map_err(|_| MlxError::InvalidExtent)?;
        let mut pointer = std::ptr::null_mut();
        let mut error = [0; ERROR_BYTES];
        // SAFETY: live ABI3, exact writable outputs; success transfers one unique session.
        let code =
            unsafe { (self.0.session_new)(index, &raw mut pointer, error.as_mut_ptr().cast()) };
        if code == 2 {
            // Constructor cleanup uncertainty also retains the library, even before Session.
            std::mem::forget(Rc::clone(&self.0));
        }
        status(code, &error)?;
        Ok(Session(Rc::new(SessionInner {
            api: Rc::clone(&self.0),
            pointer: owned(pointer)?,
            poisoned: Cell::new(false),
        })))
    }
}

struct SessionInner {
    api: Rc<Api>,
    pointer: NonNull<c_void>,
    poisoned: Cell<bool>,
}

/// Thread-confined diagnostic GPU session with private C error state.
#[derive(Clone)]
pub struct Session(Rc<SessionInner>);

struct ArrayInner {
    session: Session,
    pointer: NonNull<c_void>,
    shape: [usize; 2],
    residency: Cell<MlxArrayResidency>,
}

/// Immutable initialized C array. Successful outputs have reached terminal completion.
/// Its Rust result allocation matches the native bridge control.
#[derive(Clone)]
pub struct Array(Rc<ArrayInner>);

/// Public C compiled wrapper. Warm application still uses MLX's ambient-state cache key.
pub struct Compiled {
    session: Session,
    pointer: NonNull<c_void>,
    left: [usize; 2],
    right: [usize; 2],
}

impl Session {
    fn ready(&self) -> Result<(), MlxError> {
        if self.0.poisoned.get() {
            Err(MlxError::CompletionUnknown(
                "C evaluation session quarantined".into(),
            ))
        } else {
            Ok(())
        }
    }

    fn finish(&self, code: i32, error: &[u8; ERROR_BYTES]) -> Result<(), MlxError> {
        if code == 2 {
            self.0.poisoned.set(true);
            // The C++ pending quarantine retains accessed C handles/stream. Preserve library.
            std::mem::forget(Rc::clone(&self.0));
        }
        status(code, error)
    }

    /// Copies host F32 storage into an owned C array, without retaining caller memory.
    ///
    /// # Errors
    /// Returns invalid extent or contained allocation/SDK failures.
    pub fn upload(&self, shape: [usize; 2], data: &[f32]) -> Result<Array, MlxError> {
        self.ready()?;
        let dims = dimensions(shape)?;
        if shape[0].checked_mul(shape[1]) != Some(data.len()) {
            return Err(MlxError::InvalidExtent);
        }
        let mut pointer = std::ptr::null_mut();
        let mut error = [0; ERROR_BYTES];
        // SAFETY: initialized exact F32 extent, owned explicit session and exact outputs;
        // C array_new_data copies synchronously before the immutable slice borrow ends.
        self.finish(
            unsafe {
                (self.0.api.array_new)(
                    self.0.pointer.as_ptr(),
                    data.as_ptr(),
                    data.len(),
                    dims[0],
                    dims[1],
                    &raw mut pointer,
                    error.as_mut_ptr().cast(),
                )
            },
            &error,
        )?;
        Ok(self.array(owned(pointer)?, shape, MlxArrayResidency::HostCopied))
    }

    fn array(
        &self,
        pointer: NonNull<c_void>,
        shape: [usize; 2],
        residency: MlxArrayResidency,
    ) -> Array {
        Array(Rc::new(ArrayInner {
            session: self.clone(),
            pointer,
            shape,
            residency: Cell::new(residency),
        }))
    }

    /// Executes the explicit-stream C `MatMul` frontend and reaches terminal completion.
    ///
    /// # Errors
    /// Returns shape/session errors or contained C/SDK errors and quarantine.
    pub fn direct(&self, left: &Array, right: &Array) -> Result<Array, MlxError> {
        self.execute(self.0.api.direct, self.0.pointer, left, right)
    }

    fn execute(
        &self,
        function: Matmul,
        owner: NonNull<c_void>,
        left: &Array,
        right: &Array,
    ) -> Result<Array, MlxError> {
        self.ready()?;
        if !Rc::ptr_eq(&self.0, &left.0.session.0) || !Rc::ptr_eq(&self.0, &right.0.session.0) {
            return Err(MlxError::ForeignSession);
        }
        if left.0.shape[1] != right.0.shape[0] {
            return Err(MlxError::InvalidExtent);
        }
        let shape = [left.0.shape[0], right.0.shape[1]];
        dimensions(shape)?;
        let mut pointer = std::ptr::null_mut();
        let mut error = [0; ERROR_BYTES];
        // SAFETY: unique opaque handles, same-session exact bounded F32 shapes. Harness
        // retains both inputs/output and owner until eval/sync/wait or pending quarantine.
        self.finish(
            unsafe {
                function(
                    owner.as_ptr(),
                    left.0.pointer.as_ptr(),
                    right.0.pointer.as_ptr(),
                    &raw mut pointer,
                    error.as_mut_ptr().cast(),
                )
            },
            &error,
        )?;
        Ok(self.array(owned(pointer)?, shape, MlxArrayResidency::GpuEvaluated))
    }

    /// Cold-compiles C `MatMul` using initialized descriptor inputs outside warm sampling.
    ///
    /// # Errors
    /// Returns incompatible extents or contained compilation failures.
    pub fn prepare(&self, left: [usize; 2], right: [usize; 2]) -> Result<Compiled, MlxError> {
        self.ready()?;
        let a = dimensions(left)?;
        let b = dimensions(right)?;
        if a[1] != b[0] {
            return Err(MlxError::InvalidExtent);
        }
        dimensions([left[0], right[1]])?;
        let mut pointer = std::ptr::null_mut();
        let mut error = [0; ERROR_BYTES];
        // SAFETY: live explicit stream, checked bounded shapes, exact output; no borrowed
        // input pointer is retained by this cold compilation call.
        self.finish(
            unsafe {
                (self.0.api.prepare)(
                    self.0.pointer.as_ptr(),
                    a[0],
                    a[1],
                    b[1],
                    &raw mut pointer,
                    error.as_mut_ptr().cast(),
                )
            },
            &error,
        )?;
        Ok(Compiled {
            session: self.clone(),
            pointer: owned(pointer)?,
            left,
            right,
        })
    }
}

impl Compiled {
    /// Live callback trace count, not an inferred static API count.
    #[must_use]
    pub fn traces(&self) -> usize {
        // SAFETY: retained compiled handle and library; Rc prohibits cross-thread mutation.
        unsafe { (self.session.0.api.traces)(self.pointer.as_ptr()) }
    }

    /// Applies the public C compiler wrapper; its SDK cache lookup remains part of warm work.
    ///
    /// # Errors
    /// Returns exact shape/session or contained SDK errors.
    pub fn execute(&self, left: &Array, right: &Array) -> Result<Array, MlxError> {
        if left.0.shape != self.left || right.0.shape != self.right {
            return Err(MlxError::InvalidExtent);
        }
        self.session
            .execute(self.session.0.api.compiled, self.pointer, left, right)
    }
}

impl Array {
    /// Materializes data before publishing into the initialized exclusive host prefix.
    ///
    /// # Errors
    /// Returns short output/quarantine/materialization errors; tails stay untouched.
    pub fn read(&self, destination: &mut [f32]) -> Result<(), MlxError> {
        self.0.session.ready()?;
        let elements = self.0.shape[0]
            .checked_mul(self.0.shape[1])
            .ok_or(MlxError::InvalidExtent)?;
        let output = destination
            .get_mut(..elements)
            .ok_or(MlxError::InvalidExtent)?;
        let mut error = [0; ERROR_BYTES];
        // SAFETY: owned terminal F32 handle; exclusive initialized exact destination. The
        // harness resolves fallible data_float32 before memcpy, preserving transactionality.
        self.0.session.finish(
            unsafe {
                (self.0.session.0.api.read)(
                    self.0.pointer.as_ptr(),
                    output.as_mut_ptr(),
                    elements,
                    error.as_mut_ptr().cast(),
                )
            },
            &error,
        )?;
        self.0.residency.set(MlxArrayResidency::HostMaterialized);
        Ok(())
    }
}

impl Drop for Compiled {
    fn drop(&mut self) {
        let mut error = [0; ERROR_BYTES];
        // SAFETY: unique prepared C++ holder; pending quarantine retains independent owners.
        if unsafe {
            (self.session.0.api.prepared_free)(self.pointer.as_ptr(), error.as_mut_ptr().cast())
        } != 0
        {
            std::mem::forget(Rc::clone(&self.session.0));
        }
    }
}

impl Drop for ArrayInner {
    fn drop(&mut self) {
        let mut error = [0; ERROR_BYTES];
        // SAFETY: unique C handle; successful calls are terminal, quarantine retains backing.
        if unsafe {
            (self.session.0.api.array_free)(self.pointer.as_ptr(), error.as_mut_ptr().cast())
        } != 0
        {
            std::mem::forget(Rc::clone(&self.session.0));
        }
    }
}

impl Drop for SessionInner {
    fn drop(&mut self) {
        let mut error = [0; ERROR_BYTES];
        // SAFETY: unique session holder with all array/compiled leases released or retained.
        if unsafe { (self.api.session_free)(self.pointer.as_ptr(), error.as_mut_ptr().cast()) } != 0
        {
            std::mem::forget(Rc::clone(&self.api));
        }
    }
}

fn owned(pointer: *mut c_void) -> Result<NonNull<c_void>, MlxError> {
    NonNull::new(pointer).ok_or_else(|| MlxError::Abi("nil C evaluation owner".into()))
}

#[allow(
    clippy::needless_pass_by_value,
    reason = "map_err consumes the library error by value."
)]
fn resolve(error: libloading::Error) -> MlxError {
    MlxError::Unavailable(error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn absent_c_evaluation_library_rejects_without_fallback() {
        let result = Runtime::load("/__fusion_pcu_absent_mlx_c_evaluation__");
        if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
            assert!(matches!(result, Err(MlxError::Unavailable(_))));
        } else {
            assert!(matches!(result, Err(MlxError::UnsupportedPlatform)));
        }
    }

    #[test]
    #[ignore = "requires isolated safety-patched C evaluation library and external GPU activity check"]
    fn c_frontend_and_compiled_changing_inputs_retain_exact_owners() {
        let runtime = Runtime::load(
            std::env::var_os("PCU_MLX_C_EVALUATION").expect("set trusted ABI3 evaluation library"),
        )
        .unwrap();
        let session = runtime.open_gpu(0).unwrap();
        let foreign = runtime.open_gpu(0).unwrap();
        let compiled = session.prepare([2, 3], [3, 2]).unwrap();
        let foreign_right = foreign.upload([3, 2], &[2.0; 6]).unwrap();
        let mut retained = Vec::new();
        for phase in [0.0_f32, 1.0, 2.0] {
            let left = [1.0 + phase, 2.0, 3.0, 4.0, 5.0 + phase, 6.0];
            let right = [7.0, 8.0 + phase, 9.0, 10.0, 11.0, 12.0];
            let a = session.upload([2, 3], &left).unwrap();
            let b = session.upload([3, 2], &right).unwrap();
            assert!(matches!(
                session.direct(&a, &foreign_right),
                Err(MlxError::ForeignSession)
            ));
            for output in [
                session.direct(&a, &b).unwrap(),
                compiled.execute(&a, &b).unwrap(),
            ] {
                let mut host = [-73.0; 5];
                output.read(&mut host).unwrap();
                for row in 0..2 {
                    for column in 0..2 {
                        let expected: f32 = (0..3)
                            .map(|k| left[row * 3 + k] * right[k * 2 + column])
                            .sum();
                        assert_eq!(host[row * 2 + column].to_bits(), expected.to_bits());
                    }
                }
                assert_eq!(host[4].to_bits(), (-73.0_f32).to_bits());
                let mut short = [-13.0_f32; 3];
                assert_eq!(output.read(&mut short), Err(MlxError::InvalidExtent));
                assert_eq!(short.map(f32::to_bits), [(-13.0_f32).to_bits(); 3]);
                retained.push((output, host));
            }
        }
        assert_eq!(compiled.traces(), 1);
        drop(compiled);
        drop(session);
        drop(runtime);
        for (output, expected) in retained {
            let mut host = [-73.0; 5];
            output.read(&mut host).unwrap();
            assert_eq!(host.map(f32::to_bits), expected.map(f32::to_bits));
        }
    }
}
