//! Optional dynamically loaded ROCTX range markers for profiler correlation.

#[rustfmt::skip]
use std::{
    error::Error,
    ffi::CStr,
    os::raw::{
        c_char,
        c_int,
    },
};

use libloading::Library;

type RangePush = unsafe extern "C" fn(*const c_char) -> c_int;
type RangePop = unsafe extern "C" fn() -> c_int;

/// Runtime-loaded ROCTX adapter. Keep one instance alive while its markers are in use.
///
/// ROCTX ranges are thread-local and can be correlated with HIP activity in `ROCProfiler` output.
/// This adapter installs no callbacks and observes no application code beyond explicit calls.
pub struct TraceMarkers {
    _library: Library,
    range_push: RangePush,
    range_pop: RangePop,
}

impl TraceMarkers {
    /// Load ROCTX from the dynamic loader's normal library search path.
    pub fn open() -> Result<Self, Box<dyn Error>> {
        // SAFETY: Loading this optional ROCm marker library is an explicit caller action. The
        // `Library` owner is retained in the returned value for as long as its copied symbols.
        #[allow(unsafe_code)]
        let library = unsafe { Library::new("librocprofiler-sdk-roctx.so")? };

        // SAFETY: These signatures match `roctxRangePushA(const char*)` and `roctxRangePop()`
        // from ROCTX's public C header. Their copied pointers remain valid while `library` lives.
        #[allow(unsafe_code)]
        let (range_push, range_pop) = unsafe {
            (
                *library.get::<RangePush>(b"roctxRangePushA\0")?,
                *library.get::<RangePop>(b"roctxRangePop\0")?,
            )
        };

        Ok(Self {
            _library: library,
            range_push,
            range_pop,
        })
    }

    /// Push a thread-local named range and return ROCTX's nesting level or error code.
    pub fn push(&self, message: &CStr) -> c_int {
        // SAFETY: `CStr` guarantees a valid NUL-terminated string for this call; the ROCTX API
        // consumes the message during the call and does not retain its pointer.
        #[allow(unsafe_code)]
        unsafe {
            (self.range_push)(message.as_ptr())
        }
    }

    /// Pop the current thread-local range and return ROCTX's result code.
    pub fn pop(&self) -> c_int {
        // SAFETY: The copied function pointer has the public ROCTX signature and its library
        // owner remains alive in `self`.
        #[allow(unsafe_code)]
        unsafe {
            (self.range_pop)()
        }
    }
}
