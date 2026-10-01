//! All bridge loading, typed foreign calls, opaque owners and unsafe copies live here.

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
use libloading::Library;
#[rustfmt::skip]
use crate::{
    MlxDeviceFacts,
    MlxError,
    MlxGpuBackend,
};

const ERROR_BYTES: usize = 2048;
const SDK_VERSION: &str = "0.32.3";

#[cfg(feature = "c-api-evaluation")]
#[path = "c_api_evaluation/c_api_evaluation.rs"]
pub mod c_api_evaluation;
#[path = "typed/typed.rs"]
mod typed;
#[rustfmt::skip]
pub use typed::{
    as_f32,
    as_f32_mut,
};

#[repr(C)]
struct Facts {
    backend: u32,
    name: [u8; 256],
    architecture: [u8; 128],
    // Preserve the accepted ABI2 fact layout; these bytes carry no exposed device facts.
    reserved: [u8; 192],
}

type Abi = unsafe extern "C" fn() -> u32;
type Version = unsafe extern "C" fn(*mut c_char, *mut c_char) -> i32;
type Count = unsafe extern "C" fn(*mut i32, *mut c_char) -> i32;
type DeviceFacts = unsafe extern "C" fn(i32, *mut Facts, *mut c_char) -> i32;
type SessionNew = unsafe extern "C" fn(i32, *mut *mut c_void, *mut c_char) -> i32;
type Free = unsafe extern "C" fn(*mut c_void, *mut c_char) -> i32;
type TraceCount = unsafe extern "C" fn(*mut c_void) -> usize;
type ArrayNew = unsafe extern "C" fn(
    *mut c_void,
    *const f32,
    usize,
    i32,
    i32,
    *mut *mut c_void,
    *mut c_char,
) -> i32;
type Matmul = unsafe extern "C" fn(
    *mut c_void,
    *mut c_void,
    *mut c_void,
    *mut *mut c_void,
    *mut c_char,
) -> i32;
type MatmulPrepare = unsafe extern "C" fn(
    *mut c_void,
    i32,
    i32,
    i32,
    *mut *mut c_void,
    *mut usize,
    *mut c_char,
) -> i32;
type Read = unsafe extern "C" fn(*mut c_void, *mut f32, usize, *mut c_char) -> i32;

pub struct Api {
    // Library outlives every copied function pointer and opaque SDK owner.
    _library: Library,
    version: String,
    count: Count,
    facts: DeviceFacts,
    session_new: SessionNew,
    session_free: Free,
    array_new: ArrayNew,
    array_free: Free,
    #[cfg_attr(
        not(any(all(test, feature = "tensor"), feature = "c-api-evaluation")),
        allow(
            dead_code,
            reason = "Full private ABI is validated even when tensor execution is feature-disabled."
        )
    )]
    matmul: Matmul,
    #[cfg_attr(
        not(feature = "tensor"),
        allow(dead_code, reason = "Validate complete ABI without tensor execution.")
    )]
    matmul_prepare: MatmulPrepare,
    #[cfg_attr(
        not(feature = "tensor"),
        allow(dead_code, reason = "Validate complete ABI without tensor execution.")
    )]
    matmul_prepared_free: Free,
    #[cfg_attr(
        not(feature = "tensor"),
        allow(dead_code, reason = "Validate complete ABI without tensor execution.")
    )]
    matmul_replay: Matmul,
    #[cfg_attr(
        not(feature = "tensor"),
        allow(dead_code, reason = "Validate complete ABI without tensor execution.")
    )]
    matmul_trace_count: TraceCount,
    read: Read,
}

impl Api {
    pub fn load(path: &Path) -> Result<Rc<Self>, MlxError> {
        require_apple_silicon()?;
        // SAFETY: callers select the shipped, trusted bridge built against the pinned SDK.
        // All symbols below have the private ABI2 signatures defined in bridge.cpp. No owner
        // can escape until bridge ABI and SDK version succeed, and Library stays retained.
        let library = unsafe { Library::new(path) }
            .map_err(|error| MlxError::Unavailable(error.to_string()))?;
        // SAFETY: exact symbol names/signatures belong to private bridge ABI2. Copying a
        // pointer is safe while this library remains owned by Api and all dependent owners.
        let (
            abi,
            version,
            count,
            facts,
            session_new,
            session_free,
            array_new,
            array_free,
            matmul,
            matmul_prepare,
            matmul_prepared_free,
            matmul_replay,
            matmul_trace_count,
            read,
        ) = unsafe {
            (
                library
                    .get::<Abi>(b"pcu_mlx_bridge_abi\0")
                    .map(|symbol| *symbol),
                library
                    .get::<Version>(b"pcu_mlx_version\0")
                    .map(|symbol| *symbol),
                library
                    .get::<Count>(b"pcu_mlx_gpu_count\0")
                    .map(|symbol| *symbol),
                library
                    .get::<DeviceFacts>(b"pcu_mlx_gpu_facts\0")
                    .map(|symbol| *symbol),
                library
                    .get::<SessionNew>(b"pcu_mlx_session_new\0")
                    .map(|symbol| *symbol),
                library
                    .get::<Free>(b"pcu_mlx_session_free\0")
                    .map(|symbol| *symbol),
                library
                    .get::<ArrayNew>(b"pcu_mlx_array_new\0")
                    .map(|symbol| *symbol),
                library
                    .get::<Free>(b"pcu_mlx_array_free\0")
                    .map(|symbol| *symbol),
                library
                    .get::<Matmul>(b"pcu_mlx_matmul\0")
                    .map(|symbol| *symbol),
                library
                    .get::<MatmulPrepare>(b"pcu_mlx_matmul_prepare\0")
                    .map(|symbol| *symbol),
                library
                    .get::<Free>(b"pcu_mlx_matmul_prepared_free\0")
                    .map(|symbol| *symbol),
                library
                    .get::<Matmul>(b"pcu_mlx_matmul_replay\0")
                    .map(|symbol| *symbol),
                library
                    .get::<TraceCount>(b"pcu_mlx_matmul_trace_count\0")
                    .map(|symbol| *symbol),
                library
                    .get::<Read>(b"pcu_mlx_array_read\0")
                    .map(|symbol| *symbol),
            )
        };
        let resolve = |error: libloading::Error| MlxError::Abi(error.to_string());
        let abi = abi.map_err(resolve)?;
        // SAFETY: zero-argument ABI query has no SDK object or pointer precondition.
        let actual = unsafe { abi() };
        if actual != 2 {
            return Err(MlxError::Abi(format!("expected2, found{actual}")));
        }
        let version = version.map_err(resolve)?;
        let mut text = [0u8; 64];
        let mut error = [0u8; ERROR_BYTES];
        // SAFETY: ABI version call writes at most7 version bytes and ERROR_BYTES error bytes.
        status(
            unsafe { version(text.as_mut_ptr().cast(), error.as_mut_ptr().cast()) },
            &error,
        )?;
        let version = text_value(&text)?;
        if version != SDK_VERSION {
            return Err(MlxError::Abi(format!("unsupported SDK{version}")));
        }
        Ok(Rc::new(Self {
            _library: library,
            version,
            count: count.map_err(resolve)?,
            facts: facts.map_err(resolve)?,
            session_new: session_new.map_err(resolve)?,
            session_free: session_free.map_err(resolve)?,
            array_new: array_new.map_err(resolve)?,
            array_free: array_free.map_err(resolve)?,
            matmul: matmul.map_err(resolve)?,
            matmul_prepare: matmul_prepare.map_err(resolve)?,
            matmul_prepared_free: matmul_prepared_free.map_err(resolve)?,
            matmul_replay: matmul_replay.map_err(resolve)?,
            matmul_trace_count: matmul_trace_count.map_err(resolve)?,
            read: read.map_err(resolve)?,
        }))
    }

    pub fn version(&self) -> &str {
        &self.version
    }

    pub fn devices(&self) -> Result<Vec<MlxDeviceFacts>, MlxError> {
        let mut count = 0;
        let mut error = [0u8; ERROR_BYTES];
        // SAFETY: both outputs have exact ABI extents and no SDK handle is required.
        status(
            unsafe { (self.count)(&raw mut count, error.as_mut_ptr().cast()) },
            &error,
        )?;
        let count =
            usize::try_from(count).map_err(|_| MlxError::Abi("negative GPU count".into()))?;
        (0..count).map(|index| self.device(index)).collect()
    }

    pub fn device(&self, index: usize) -> Result<MlxDeviceFacts, MlxError> {
        let index = i32::try_from(index).map_err(|_| MlxError::InvalidExtent)?;
        let mut facts = Facts {
            backend: 0,
            name: [0; 256],
            architecture: [0; 128],
            reserved: [0; 192],
        };
        let mut error = [0u8; ERROR_BYTES];
        // SAFETY: live typed bridge, exact repr(C) fact/output extents; SDK validates index.
        status(
            unsafe { (self.facts)(index, &raw mut facts, error.as_mut_ptr().cast()) },
            &error,
        )?;
        Ok(MlxDeviceFacts {
            index: usize::try_from(index).map_err(|_| MlxError::InvalidExtent)?,
            backend: match facts.backend {
                1 => MlxGpuBackend::Metal,
                _ => return Err(MlxError::Abi("unknown GPU backend".into())),
            },
            name: text_value(&facts.name)?,
            architecture: text_value(&facts.architecture)?,
        })
    }

    pub fn open(self: &Rc<Self>, index: usize) -> Result<Session, MlxError> {
        let index = i32::try_from(index).map_err(|_| MlxError::InvalidExtent)?;
        let mut pointer = std::ptr::null_mut();
        let mut error = [0u8; ERROR_BYTES];
        // SAFETY: live bridge and exact outputs; successful new returns an owned SessionOwner.
        status(
            unsafe { (self.session_new)(index, &raw mut pointer, error.as_mut_ptr().cast()) },
            &error,
        )?;
        let pointer = NonNull::new(pointer).ok_or_else(|| MlxError::Abi("nil session".into()))?;
        Ok(Session(Rc::new(SessionInner {
            api: Rc::clone(self),
            pointer,
            poisoned: Cell::new(false),
        })))
    }
}

struct SessionInner {
    api: Rc<Api>,
    pointer: NonNull<c_void>,
    poisoned: Cell<bool>,
}

#[derive(Clone)]
pub struct Session(Rc<SessionInner>);
pub struct Array {
    session: Session,
    pointer: NonNull<c_void>,
    shape: [usize; 2],
}

#[cfg(feature = "tensor")]
pub struct PreparedMatmul {
    session: Session,
    pointer: NonNull<c_void>,
    left_shape: [usize; 2],
    right_shape: [usize; 2],
}

impl Session {
    #[cfg(feature = "tensor")]
    pub fn same(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.0, &other.0)
    }

    pub fn ensure_ready(&self) -> Result<(), MlxError> {
        if self.0.poisoned.get() {
            Err(MlxError::CompletionUnknown(
                "MLX session quarantined".into(),
            ))
        } else {
            Ok(())
        }
    }

    fn finish(&self, code: i32, error: &[u8; ERROR_BYTES]) -> Result<(), MlxError> {
        if code == 2 {
            self.0.poisoned.set(true);
            // Leaked C++ Pending retains actual GPU-accessed backing. Keep the loaded bridge
            // and SDK alive alongside it; no uncertain owner is merely dropped/unloaded.
            std::mem::forget(Rc::clone(&self.0));
        }
        status(code, error)
    }

    pub fn upload(&self, shape: [usize; 2], data: &[f32]) -> Result<Array, MlxError> {
        self.ensure_ready()?;
        let dimensions = dimensions(shape)?;
        if shape[0].checked_mul(shape[1]) != Some(data.len()) {
            return Err(MlxError::InvalidExtent);
        }
        let mut pointer = std::ptr::null_mut();
        let mut error = [0u8; ERROR_BYTES];
        // SAFETY: exact initialized F32 input extent validated above, owner lives throughout.
        // Iterator construction copies input before returning; no caller pointer is retained.
        self.finish(
            unsafe {
                (self.0.api.array_new)(
                    self.0.pointer.as_ptr(),
                    data.as_ptr(),
                    data.len(),
                    dimensions[0],
                    dimensions[1],
                    &raw mut pointer,
                    error.as_mut_ptr().cast(),
                )
            },
            &error,
        )?;
        Ok(Array {
            session: self.clone(),
            pointer: NonNull::new(pointer).ok_or_else(|| MlxError::Abi("nil array".into()))?,
            shape,
        })
    }

    #[cfg(any(all(test, feature = "tensor"), feature = "c-api-evaluation"))]
    pub fn matmul(&self, left: &Array, right: &Array) -> Result<Array, MlxError> {
        self.ensure_ready()?;
        if !self.same(&left.session) || !self.same(&right.session) {
            return Err(MlxError::ForeignSession);
        }
        if left.shape[1] != right.shape[0] {
            return Err(MlxError::InvalidExtent);
        }
        let shape = [left.shape[0], right.shape[1]];
        dimensions(shape)?;
        let mut pointer = std::ptr::null_mut();
        let mut error = [0u8; ERROR_BYTES];
        // SAFETY: live same-session F32 owners, exact bounded matrix shapes and outputs.
        // Bridge retains all accessed arrays before eval and terminal sync/quarantine.
        self.finish(
            unsafe {
                (self.0.api.matmul)(
                    self.0.pointer.as_ptr(),
                    left.pointer.as_ptr(),
                    right.pointer.as_ptr(),
                    &raw mut pointer,
                    error.as_mut_ptr().cast(),
                )
            },
            &error,
        )?;
        Ok(Array {
            session: self.clone(),
            pointer: NonNull::new(pointer)
                .ok_or_else(|| MlxError::Abi("nil matrix result".into()))?,
            shape,
        })
    }

    #[cfg(feature = "tensor")]
    pub fn prepare_matmul(
        &self,
        left_shape: [usize; 2],
        right_shape: [usize; 2],
    ) -> Result<PreparedMatmul, MlxError> {
        self.ensure_ready()?;
        let left = dimensions(left_shape)?;
        let right = dimensions(right_shape)?;
        if left[1] != right[0] {
            return Err(MlxError::InvalidExtent);
        }
        dimensions([left_shape[0], right_shape[1]])?;
        let mut pointer = std::ptr::null_mut();
        let mut traces = 0;
        let mut error = [0u8; ERROR_BYTES];
        // SAFETY: bounded compatible shapes, live explicit session and exact ABI2 outputs.
        // Bridge cold-traces unmaterialized logical descriptors without evaluating tensor data.
        self.finish(
            unsafe {
                (self.0.api.matmul_prepare)(
                    self.0.pointer.as_ptr(),
                    left[0],
                    left[1],
                    right[1],
                    &raw mut pointer,
                    &raw mut traces,
                    error.as_mut_ptr().cast(),
                )
            },
            &error,
        )?;
        Ok(PreparedMatmul {
            session: self.clone(),
            pointer: NonNull::new(pointer)
                .ok_or_else(|| MlxError::Abi("nil compiled MatMul".into()))?,
            left_shape,
            right_shape,
        })
    }
}

#[cfg(feature = "tensor")]
impl PreparedMatmul {
    pub fn traces(&self) -> usize {
        // SAFETY: retained CompiledOwner and loaded bridge outlive this immutable metadata read.
        // Thread confinement prevents concurrent owner destruction or callback mutation.
        unsafe { (self.session.0.api.matmul_trace_count)(self.pointer.as_ptr()) }
    }

    pub fn execute(&self, left: &Array, right: &Array) -> Result<Array, MlxError> {
        self.session.ensure_ready()?;
        if !self.session.same(&left.session) || !self.session.same(&right.session) {
            return Err(MlxError::ForeignSession);
        }
        if left.shape != self.left_shape || right.shape != self.right_shape {
            return Err(MlxError::InvalidExtent);
        }
        let mut pointer = std::ptr::null_mut();
        let mut error = [0u8; ERROR_BYTES];
        // SAFETY: frozen same-session F32 compiled primitive and exact initialized operands.
        // Pending retains compiled/input/output owners until terminal completion or quarantine.
        self.session.finish(
            unsafe {
                (self.session.0.api.matmul_replay)(
                    self.pointer.as_ptr(),
                    left.pointer.as_ptr(),
                    right.pointer.as_ptr(),
                    &raw mut pointer,
                    error.as_mut_ptr().cast(),
                )
            },
            &error,
        )?;
        Ok(Array {
            session: self.session.clone(),
            pointer: NonNull::new(pointer)
                .ok_or_else(|| MlxError::Abi("nil compiled matrix result".into()))?,
            shape: [self.left_shape[0], self.right_shape[1]],
        })
    }
}

#[cfg(feature = "tensor")]
impl Drop for PreparedMatmul {
    fn drop(&mut self) {
        let mut error = [0u8; ERROR_BYTES];
        // SAFETY: unique compiled handle. Quarantined Pending independently owns the actual
        // compiled closure/primitive, stream and accessed arrays; Api stays retained beside it.
        let code = unsafe {
            (self.session.0.api.matmul_prepared_free)(
                self.pointer.as_ptr(),
                error.as_mut_ptr().cast(),
            )
        };
        if code != 0 {
            std::mem::forget(Rc::clone(&self.session.0));
        }
    }
}

impl Array {
    pub const fn shape(&self) -> [usize; 2] {
        self.shape
    }
    #[cfg(feature = "tensor")]
    pub fn same_session(&self, session: &Session) -> bool {
        self.session.same(session)
    }
    pub fn read(&self, output: &mut [f32]) -> Result<(), MlxError> {
        self.session.ensure_ready()?;
        let count = self.shape[0]
            .checked_mul(self.shape[1])
            .ok_or(MlxError::InvalidExtent)?;
        if output.len() != count {
            return Err(MlxError::InvalidExtent);
        }
        let mut error = [0u8; ERROR_BYTES];
        // SAFETY: terminal owned F32 array and exact exclusive initialized destination.
        // Bridge resolves fallible host materialization before any memcpy publication.
        self.session.finish(
            unsafe {
                (self.session.0.api.read)(
                    self.pointer.as_ptr(),
                    output.as_mut_ptr(),
                    count,
                    error.as_mut_ptr().cast(),
                )
            },
            &error,
        )
    }
}

impl Drop for Array {
    fn drop(&mut self) {
        let mut error = [0u8; ERROR_BYTES];
        // SAFETY: unique opaque holder from successful new/matmul, no work remains in flight.
        // Quarantined accessed owners are independently retained by leaked C++ Pending.
        let code = unsafe {
            (self.session.0.api.array_free)(self.pointer.as_ptr(), error.as_mut_ptr().cast())
        };
        if code != 0 {
            std::mem::forget(Rc::clone(&self.session.0));
        }
    }
}

impl Drop for SessionInner {
    fn drop(&mut self) {
        let mut error = [0u8; ERROR_BYTES];
        // SAFETY: unique SessionOwner handle; all Array owners retained this SessionInner.
        let code =
            unsafe { (self.api.session_free)(self.pointer.as_ptr(), error.as_mut_ptr().cast()) };
        if code != 0 {
            std::mem::forget(Rc::clone(&self.api));
        }
    }
}

fn dimensions(shape: [usize; 2]) -> Result<[i32; 2], MlxError> {
    let count = shape[0]
        .checked_mul(shape[1])
        .and_then(|count| count.checked_mul(4))
        .ok_or(MlxError::InvalidExtent)?;
    if count == 0 || isize::try_from(count).is_err() {
        return Err(MlxError::InvalidExtent);
    }
    Ok([
        i32::try_from(shape[0]).map_err(|_| MlxError::InvalidExtent)?,
        i32::try_from(shape[1]).map_err(|_| MlxError::InvalidExtent)?,
    ])
}

fn text_value(bytes: &[u8]) -> Result<String, MlxError> {
    let end = bytes
        .iter()
        .position(|&byte| byte == 0)
        .ok_or_else(|| MlxError::Abi("unterminated ABI text".into()))?;
    String::from_utf8(bytes[..end].to_vec()).map_err(|_| MlxError::Abi("invalid ABI UTF8".into()))
}

const fn require_apple_silicon() -> Result<(), MlxError> {
    if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
        Ok(())
    } else {
        Err(MlxError::UnsupportedPlatform)
    }
}

fn status(code: i32, error: &[u8; ERROR_BYTES]) -> Result<(), MlxError> {
    match code {
        0 => Ok(()),
        1 => Err(MlxError::Runtime(text_value(error)?)),
        2 => Err(MlxError::CompletionUnknown(text_value(error)?)),
        _ => Err(MlxError::Abi(format!("invalid bridge status{code}"))),
    }
}

#[cfg(all(test, feature = "tensor"))]
mod tests {
    use super::*;
    #[test]
    #[ignore = "requires pinned MLX GPU bridge and an external GPU activity check"]
    fn sdk_exception_is_contained_and_valid_submission_retries() {
        let path = std::env::var_os("PCU_MLX_BRIDGE").expect("set isolated pinned bridge path");
        let api = Api::load(Path::new(&path)).unwrap();
        let session = api.open(0).unwrap();
        let a = session.upload([2, 3], &[1.0; 6]).unwrap();
        let bad = session.upload([4, 2], &[2.0; 8]).unwrap();
        let mut pointer = std::ptr::null_mut();
        let mut error = [0u8; ERROR_BYTES];
        // SAFETY: owners are valid dense F32 arrays with distinct same-session backings.
        // Deliberately incompatible inner dimensions exercise MLX's own checked exception
        // through the private bridge; no GPU kernel is launched with unchecked bounds.
        let code = unsafe {
            (api.matmul)(
                session.0.pointer.as_ptr(),
                a.pointer.as_ptr(),
                bad.pointer.as_ptr(),
                &raw mut pointer,
                error.as_mut_ptr().cast(),
            )
        };
        assert_eq!(code, 1);
        assert!(pointer.is_null());
        assert!(text_value(&error).unwrap().contains("matmul"));
        let b = session.upload([3, 2], &[2.0; 6]).unwrap();
        let result = session.matmul(&a, &b).unwrap();
        let mut host = [0.0; 4];
        result.read(&mut host).unwrap();
        assert_eq!(host.map(f32::to_bits), [6.0_f32.to_bits(); 4]);
    }
}
