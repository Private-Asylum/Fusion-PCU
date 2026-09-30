//! NVRTC compilation for the CUDA device selected by an existing [`CudaRuntime`].

#[rustfmt::skip]
use std::{
    ffi::{CStr, CString, c_char, c_int},
    fmt,
    ptr,
};

use crate::ffi::Library;

#[rustfmt::skip]
use crate::{
    CudaError,
    CudaRuntime,
    ffi::{
        driver::{
            DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
            DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
            DriverDeviceGetAttribute,
            DriverGetDevice,
            DriverInit,
        },
    },
    ffi::nvrtc::{
        CU_DEVICE_GET,
        CU_DEVICE_GET_ATTRIBUTE,
        CU_INIT,
        CompileProgram,
        CreateProgram,
        DestroyProgram,
        GetErrorString,
        GetProgramLog,
        GetProgramLogSize,
        GetPtx,
        GetPtxSize,
        Program,
        ResultCode,
        NVRTC_COMPILE_PROGRAM,
        NVRTC_CREATE_PROGRAM,
        NVRTC_DESTROY_PROGRAM,
        NVRTC_GET_ERROR_STRING,
        NVRTC_GET_PROGRAM_LOG,
        NVRTC_GET_PROGRAM_LOG_SIZE,
        NVRTC_GET_PTX,
        NVRTC_GET_PTX_SIZE,
    },
};

/// Failures from loading NVRTC or compiling CUDA source for the selected device.
#[derive(Debug)]
pub enum CudaRtcError {
    Library(String),
    MissingSymbol {
        symbol: &'static str,
        detail: String,
    },
    InvalidSource,
    DeviceSelection(CudaError),
    Driver {
        operation: &'static str,
        code: i32,
        detail: String,
    },
    InvalidComputeCapability {
        major: i32,
        minor: i32,
    },
    Api {
        operation: &'static str,
        code: i32,
        detail: String,
    },
    Compilation {
        log: String,
    },
    EmptyPtx,
}

impl fmt::Display for CudaRtcError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Library(detail) => write!(f, "NVRTC library unavailable: {detail}"),
            Self::MissingSymbol { symbol, detail } => {
                write!(f, "NVRTC symbol {symbol} unavailable: {detail}")
            }
            Self::InvalidSource => f.write_str("CUDA source contains an interior NUL byte"),
            Self::DeviceSelection(error) => write!(f, "NVRTC device selection failed: {error}"),
            Self::Driver {
                operation,
                code,
                detail,
            } => {
                write!(
                    f,
                    "{operation} failed with CUDA driver status {code}: {detail}"
                )
            }
            Self::InvalidComputeCapability { major, minor } => {
                write!(
                    f,
                    "CUDA driver returned invalid compute capability {major}.{minor}"
                )
            }
            Self::Api {
                operation,
                code,
                detail,
            } => {
                write!(f, "{operation} failed with NVRTC status {code}: {detail}")
            }
            Self::Compilation { log } => write!(f, "NVRTC compilation failed: {log}"),
            Self::EmptyPtx => f.write_str("NVRTC returned empty PTX"),
        }
    }
}

impl std::error::Error for CudaRtcError {}

/// Compile source to PTX for `runtime`'s selected device.
///
/// The compute capability is queried from the CUDA Driver API and passed as
/// `--gpu-architecture=compute_xy`. Floating-point contraction and flush-to-zero are disabled,
/// and division and square root use precise modes to preserve the checked scalar laws.
///
/// # Errors
///
/// Returns a typed error if the runtime device cannot be selected, the driver/NVRTC API is
/// unavailable, the source cannot be compiled, or no PTX is returned.
pub fn compile_cuda_source_for_device(
    runtime: &CudaRuntime,
    source: &str,
) -> Result<Vec<u8>, CudaRtcError> {
    let source = CString::new(source).map_err(|_| CudaRtcError::InvalidSource)?;
    runtime
        .cuda_set_device(runtime.0.ordinal)
        .map_err(CudaRtcError::DeviceSelection)?;
    let (major, minor) = device_compute_capability(runtime.0.device)?;
    let architecture = format!("compute_{major}{minor}");
    compile_for_architecture(&source, &architecture)
}

fn device_compute_capability(ordinal: c_int) -> Result<(i32, i32), CudaRtcError> {
    let library = load_library(&["libcuda.so.1", "libcuda.so"]).map_err(CudaRtcError::Library)?;
    let init = symbol::<DriverInit>(&library, "cuInit", CU_INIT)?;
    let get_device = symbol::<DriverGetDevice>(&library, "cuDeviceGet", CU_DEVICE_GET)?;
    let get_attribute = symbol::<DriverDeviceGetAttribute>(
        &library,
        "cuDeviceGetAttribute",
        CU_DEVICE_GET_ATTRIBUTE,
    )?;
    driver_check("cuInit", unsafe { init(0) })?;
    let mut device = 0;
    driver_check("cuDeviceGet", unsafe {
        get_device(&raw mut device, ordinal)
    })?;
    let mut major = 0;
    let mut minor = 0;
    driver_check("cuDeviceGetAttribute(compute capability major)", unsafe {
        get_attribute(
            &raw mut major,
            DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
            device,
        )
    })?;
    driver_check("cuDeviceGetAttribute(compute capability minor)", unsafe {
        get_attribute(
            &raw mut minor,
            DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
            device,
        )
    })?;
    if !(1..=12).contains(&major) || !(0..=9).contains(&minor) {
        return Err(CudaRtcError::InvalidComputeCapability { major, minor });
    }
    Ok((major, minor))
}

fn compile_for_architecture(source: &CStr, architecture: &str) -> Result<Vec<u8>, CudaRtcError> {
    let library = load_nvrtc()?;
    let create = symbol::<CreateProgram>(&library, "nvrtcCreateProgram", NVRTC_CREATE_PROGRAM)?;
    let destroy = symbol::<DestroyProgram>(&library, "nvrtcDestroyProgram", NVRTC_DESTROY_PROGRAM)?;
    let compile = symbol::<CompileProgram>(&library, "nvrtcCompileProgram", NVRTC_COMPILE_PROGRAM)?;
    let log_size = symbol::<GetProgramLogSize>(
        &library,
        "nvrtcGetProgramLogSize",
        NVRTC_GET_PROGRAM_LOG_SIZE,
    )?;
    let get_log = symbol::<GetProgramLog>(&library, "nvrtcGetProgramLog", NVRTC_GET_PROGRAM_LOG)?;
    let ptx_size = symbol::<GetPtxSize>(&library, "nvrtcGetPTXSize", NVRTC_GET_PTX_SIZE)?;
    let get_ptx = symbol::<GetPtx>(&library, "nvrtcGetPTX", NVRTC_GET_PTX)?;
    let error_string =
        symbol::<GetErrorString>(&library, "nvrtcGetErrorString", NVRTC_GET_ERROR_STRING)?;
    let name = c"fusion-kernel.cu";
    let options = [
        format!("--gpu-architecture={architecture}"),
        "--fmad=false".to_owned(),
        "--ftz=false".to_owned(),
        "--prec-div=true".to_owned(),
        "--prec-sqrt=true".to_owned(),
    ];
    let options = options
        .iter()
        .map(|option| CString::new(option.as_str()).expect("compiler option has no NUL"))
        .collect::<Vec<_>>();
    let option_ptrs = options
        .iter()
        .map(|option| option.as_ptr())
        .collect::<Vec<_>>();
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
    check("nvrtcCreateProgram", status, error_string)?;
    let result = compile_program(
        program,
        &option_ptrs,
        compile,
        log_size,
        get_log,
        ptx_size,
        get_ptx,
        error_string,
    );
    let destroy_status = unsafe { destroy(&raw mut program) };
    match result {
        Ok(ptx) => {
            check("nvrtcDestroyProgram", destroy_status, error_string)?;
            Ok(ptx)
        }
        Err(error) => Err(error),
    }
}

#[allow(clippy::too_many_arguments)]
fn compile_program(
    program: Program,
    options: &[*const c_char],
    compile: CompileProgram,
    log_size: GetProgramLogSize,
    get_log: GetProgramLog,
    ptx_size: GetPtxSize,
    get_ptx: GetPtx,
    error_string: GetErrorString,
) -> Result<Vec<u8>, CudaRtcError> {
    let status = unsafe {
        compile(
            program,
            c_int::try_from(options.len()).expect("fixed compile option count fits c_int"),
            options.as_ptr(),
        )
    };
    if status != 0 {
        let mut size = 0;
        let _ = unsafe { log_size(program, &raw mut size) };
        let mut log = vec![0_u8; size.max(1)];
        let _ = unsafe { get_log(program, log.as_mut_ptr().cast()) };
        let log = CStr::from_bytes_until_nul(&log).map_or_else(
            |_| String::from_utf8_lossy(&log).into_owned(),
            |s| s.to_string_lossy().into_owned(),
        );
        return Err(CudaRtcError::Compilation { log });
    }
    let mut size = 0;
    check(
        "nvrtcGetPTXSize",
        unsafe { ptx_size(program, &raw mut size) },
        error_string,
    )?;
    if size <= 1 {
        return Err(CudaRtcError::EmptyPtx);
    }
    let mut ptx = vec![0_u8; size];
    check(
        "nvrtcGetPTX",
        unsafe { get_ptx(program, ptx.as_mut_ptr().cast()) },
        error_string,
    )?;
    // The driver module loader consumes the terminating NUL as part of the PTX image.
    Ok(ptx)
}

fn load_nvrtc() -> Result<Library, CudaRtcError> {
    let mut candidates = Vec::new();
    if let Some(path) = std::env::var_os("NVRTC_LIBRARY") {
        candidates.push(path);
    }
    for variable in ["CUDA_HOME", "CUDA_PATH"] {
        if let Some(root) = std::env::var_os(variable) {
            let root = std::path::PathBuf::from(root);
            candidates.push(root.join("lib64/libnvrtc.so").into_os_string());
            candidates.push(
                root.join("targets/x86_64-linux/lib/libnvrtc.so")
                    .into_os_string(),
            );
        }
    }
    candidates.extend(
        [
            "libnvrtc.so",
            "libnvrtc.so.13",
            "libnvrtc.so.12",
            "/usr/local/cuda/lib64/libnvrtc.so",
        ]
        .into_iter()
        .map(Into::into),
    );
    let mut details = Vec::new();
    for candidate in candidates {
        match unsafe { crate::ffi::load_uncached_library(&candidate) } {
            Ok(library) => return Ok(library),
            Err(error) => details.push(format!("{}: {error}", candidate.to_string_lossy())),
        }
    }
    Err(CudaRtcError::Library(details.join("; ")))
}

fn load_library(candidates: &[&str]) -> Result<Library, String> {
    let mut details = Vec::new();
    for candidate in candidates {
        match unsafe { crate::ffi::load_uncached_library(candidate) } {
            Ok(library) => return Ok(library),
            Err(error) => details.push(format!("{candidate}: {error}")),
        }
    }
    Err(details.join("; "))
}

/// Check that NVRTC and all symbols needed for PTX compilation can be loaded.
///
/// # Errors
///
/// Returns a typed library or symbol error when NVRTC cannot be used.
pub fn nvrtc_available() -> Result<(), CudaRtcError> {
    let library = load_nvrtc()?;
    let _ = symbol::<CreateProgram>(&library, "nvrtcCreateProgram", NVRTC_CREATE_PROGRAM)?;
    let _ = symbol::<DestroyProgram>(&library, "nvrtcDestroyProgram", NVRTC_DESTROY_PROGRAM)?;
    let _ = symbol::<CompileProgram>(&library, "nvrtcCompileProgram", NVRTC_COMPILE_PROGRAM)?;
    let _ = symbol::<GetProgramLogSize>(
        &library,
        "nvrtcGetProgramLogSize",
        NVRTC_GET_PROGRAM_LOG_SIZE,
    )?;
    let _ = symbol::<GetProgramLog>(&library, "nvrtcGetProgramLog", NVRTC_GET_PROGRAM_LOG)?;
    let _ = symbol::<GetPtxSize>(&library, "nvrtcGetPTXSize", NVRTC_GET_PTX_SIZE)?;
    let _ = symbol::<GetPtx>(&library, "nvrtcGetPTX", NVRTC_GET_PTX)?;
    let _ = symbol::<GetErrorString>(&library, "nvrtcGetErrorString", NVRTC_GET_ERROR_STRING)?;
    Ok(())
}

fn symbol<T: Copy>(
    library: &Library,
    name: &'static str,
    symbol: &'static [u8],
) -> Result<T, CudaRtcError> {
    unsafe { crate::ffi::symbol::<T>(library, symbol) }
        .map(|symbol| *symbol)
        .map_err(|error| CudaRtcError::MissingSymbol {
            symbol: name,
            detail: error.to_string(),
        })
}

fn driver_check(operation: &'static str, code: ResultCode) -> Result<(), CudaRtcError> {
    if code == 0 {
        return Ok(());
    }
    Err(CudaRtcError::Driver {
        operation,
        code,
        detail: format!("driver error code {code}"),
    })
}

fn check(
    operation: &'static str,
    code: ResultCode,
    error_string: GetErrorString,
) -> Result<(), CudaRtcError> {
    if code == 0 {
        return Ok(());
    }
    let detail = unsafe {
        let pointer = error_string(code);
        if pointer.is_null() {
            "unknown NVRTC error".into()
        } else {
            CStr::from_ptr(pointer).to_string_lossy().into_owned()
        }
    };
    Err(CudaRtcError::Api {
        operation,
        code,
        detail,
    })
}
