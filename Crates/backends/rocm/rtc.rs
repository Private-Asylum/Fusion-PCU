//! HIP runtime compilation for the device selected by an existing [`HipRuntime`].

use std::{
    ffi::{
        CStr,
        CString,
        c_char,
        c_int,
    },
    fmt,
    ptr,
};

use libloading::Library;

use crate::{
    HipError,
    HipRuntime,
};

type Program = *mut std::ffi::c_void;
type ResultCode = c_int;
type CreateProgram = unsafe extern "C" fn(
    *mut Program,
    *const c_char,
    *const c_char,
    c_int,
    *const *const c_char,
    *const *const c_char,
) -> ResultCode;
type DestroyProgram = unsafe extern "C" fn(*mut Program) -> ResultCode;
type CompileProgram = unsafe extern "C" fn(Program, c_int, *const *const c_char) -> ResultCode;
type GetProgramLogSize = unsafe extern "C" fn(Program, *mut usize) -> ResultCode;
type GetProgramLog = unsafe extern "C" fn(Program, *mut c_char) -> ResultCode;
type GetCodeSize = unsafe extern "C" fn(Program, *mut usize) -> ResultCode;
type GetCode = unsafe extern "C" fn(Program, *mut c_char) -> ResultCode;
type GetErrorString = unsafe extern "C" fn(ResultCode) -> *const c_char;

/// Failures from loading HIPRTC or compiling HIP source for the selected device.
#[derive(Debug)]
pub enum HipRtcError {
    Library(String),
    MissingSymbol {
        symbol: &'static str,
        detail: String,
    },
    InvalidSource,
    DeviceSelection(HipError),
    Api {
        operation: &'static str,
        code: i32,
        detail: String,
    },
    Compilation {
        log: String,
    },
    EmptyCodeObject,
}

impl fmt::Display for HipRtcError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Library(detail) => write!(f, "HIPRTC library unavailable: {detail}"),
            Self::MissingSymbol { symbol, detail } => {
                write!(f, "HIPRTC symbol {symbol} unavailable: {detail}")
            }
            Self::InvalidSource => f.write_str("HIP source contains an interior NUL byte"),
            Self::DeviceSelection(error) => write!(f, "HIPRTC device selection failed: {error}"),
            Self::Api {
                operation,
                code,
                detail,
            } => {
                write!(f, "{operation} failed with HIPRTC status {code}: {detail}")
            }
            Self::Compilation { log } => write!(f, "HIPRTC compilation failed: {log}"),
            Self::EmptyCodeObject => f.write_str("HIPRTC returned an empty code object"),
        }
    }
}

impl std::error::Error for HipRtcError {}

/// Compile source to an HSACO code object for `runtime`'s selected device.
///
/// No architecture option is passed: HIPRTC obtains the target from the current HIP device.
/// The device is explicitly selected immediately before entering HIPRTC, since HIP's current
/// device is thread-local and may have been changed by another runtime operation.
///
/// # Errors
///
/// Returns a typed error if the runtime device cannot be selected, HIPRTC is unavailable, the
/// source cannot be compiled, or no usable code object is returned.
pub fn compile_hip_source_for_device(
    runtime: &HipRuntime,
    source: &str,
) -> Result<Vec<u8>, HipRtcError> {
    let source = CString::new(source).map_err(|_| HipRtcError::InvalidSource)?;
    let name = c"fusion-kernel.hip";
    let library = load_hiprtc()?;
    let create = symbol::<CreateProgram>(&library, "hiprtcCreateProgram")?;
    let destroy = symbol::<DestroyProgram>(&library, "hiprtcDestroyProgram")?;
    let compile = symbol::<CompileProgram>(&library, "hiprtcCompileProgram")?;
    let log_size = symbol::<GetProgramLogSize>(&library, "hiprtcGetProgramLogSize")?;
    let get_log = symbol::<GetProgramLog>(&library, "hiprtcGetProgramLog")?;
    let code_size = symbol::<GetCodeSize>(&library, "hiprtcGetCodeSize")?;
    let get_code = symbol::<GetCode>(&library, "hiprtcGetCode")?;
    let error_string = symbol::<GetErrorString>(&library, "hiprtcGetErrorString")?;

    // HIPRTC documents that it infers the architecture from the current HIP device when no
    // --gpu-architecture option is supplied. HipRuntime's own selection helper also handles
    // cloned runtimes used from different host threads.
    runtime
        .hip_set_device(runtime.0.device)
        .map_err(HipRtcError::DeviceSelection)?;

    let mut program = ptr::null_mut();
    let status = unsafe {
        create(
            &raw mut program,
            source.as_ptr(),
            name.as_ptr(),
            0,
            ptr::null(),
            ptr::null(),
        )
    };
    check("hiprtcCreateProgram", status, error_string)?;
    let result = compile_program(
        program,
        compile,
        log_size,
        get_log,
        code_size,
        get_code,
        error_string,
    );
    let destroy_status = unsafe { destroy(&raw mut program) };
    match result {
        Ok(code) => {
            check("hiprtcDestroyProgram", destroy_status, error_string)?;
            Ok(code)
        }
        Err(error) => Err(error),
    }
}

fn compile_program(
    program: Program,
    compile: CompileProgram,
    log_size: GetProgramLogSize,
    get_log: GetProgramLog,
    code_size: GetCodeSize,
    get_code: GetCode,
    error_string: GetErrorString,
) -> Result<Vec<u8>, HipRtcError> {
    let status = unsafe { compile(program, 0, ptr::null()) };
    if status != 0 {
        let mut size = 0;
        let _ = unsafe { log_size(program, &raw mut size) };
        let mut log = vec![0_u8; size.max(1)];
        let _ = unsafe { get_log(program, log.as_mut_ptr().cast()) };
        let log = CStr::from_bytes_until_nul(&log).map_or_else(
            |_| String::from_utf8_lossy(&log).into_owned(),
            |s| s.to_string_lossy().into_owned(),
        );
        return Err(HipRtcError::Compilation { log });
    }

    let mut size = 0;
    check(
        "hiprtcGetCodeSize",
        unsafe { code_size(program, &raw mut size) },
        error_string,
    )?;
    if size == 0 {
        return Err(HipRtcError::EmptyCodeObject);
    }
    let mut code = vec![0_u8; size];
    check(
        "hiprtcGetCode",
        unsafe { get_code(program, code.as_mut_ptr().cast()) },
        error_string,
    )?;
    Ok(code)
}

fn load_hiprtc() -> Result<Library, HipRtcError> {
    let candidates = std::env::var_os("HIPRTC_LIBRARY").map_or_else(
        || vec!["libhiprtc.so".into(), "libhiprtc.so.7".into()],
        |path| vec![path],
    );
    let mut details = Vec::new();
    for candidate in candidates {
        match unsafe { Library::new(&candidate) } {
            Ok(library) => return Ok(library),
            Err(error) => details.push(format!("{}: {error}", candidate.to_string_lossy())),
        }
    }
    Err(HipRtcError::Library(details.join("; ")))
}

/// Check that HIPRTC and the symbols required by compilation can be loaded.
///
/// # Errors
///
/// Returns a typed library or symbol error when the runtime compiler cannot be used.
pub fn hiprtc_available() -> Result<(), HipRtcError> {
    let library = load_hiprtc()?;
    let _ = symbol::<CreateProgram>(&library, "hiprtcCreateProgram")?;
    let _ = symbol::<DestroyProgram>(&library, "hiprtcDestroyProgram")?;
    let _ = symbol::<CompileProgram>(&library, "hiprtcCompileProgram")?;
    let _ = symbol::<GetProgramLogSize>(&library, "hiprtcGetProgramLogSize")?;
    let _ = symbol::<GetProgramLog>(&library, "hiprtcGetProgramLog")?;
    let _ = symbol::<GetCodeSize>(&library, "hiprtcGetCodeSize")?;
    let _ = symbol::<GetCode>(&library, "hiprtcGetCode")?;
    let _ = symbol::<GetErrorString>(&library, "hiprtcGetErrorString")?;
    Ok(())
}

fn symbol<T: Copy>(library: &Library, name: &'static str) -> Result<T, HipRtcError> {
    let mut bytes = name.as_bytes().to_vec();
    bytes.push(0);
    unsafe { library.get::<T>(&bytes) }
        .map(|symbol| *symbol)
        .map_err(|error| HipRtcError::MissingSymbol {
            symbol: name,
            detail: error.to_string(),
        })
}

fn check(
    operation: &'static str,
    code: ResultCode,
    error_string: GetErrorString,
) -> Result<(), HipRtcError> {
    if code == 0 {
        return Ok(());
    }
    let detail = unsafe {
        let pointer = error_string(code);
        if pointer.is_null() {
            "unknown HIPRTC error".into()
        } else {
            CStr::from_ptr(pointer).to_string_lossy().into_owned()
        }
    };
    Err(HipRtcError::Api {
        operation,
        code,
        detail,
    })
}
