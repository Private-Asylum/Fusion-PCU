//! NVRTC compilation for the CUDA device selected by an existing [`CudaRuntime`].

#[rustfmt::skip]
use std::{
    ffi::CString,
    fmt,
};

#[rustfmt::skip]
use crate::{
    CudaError,
    CudaRuntime,
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
    let (major, minor) = crate::ffi::device_compute_capability(runtime.0.device)?;
    let architecture = format!("compute_{major}{minor}");
    crate::ffi::compile_for_architecture(&source, &architecture)
}

/// Probe the exact NVRTC ABI used by compilation.
pub fn nvrtc_available() -> Result<(), CudaRtcError> {
    crate::ffi::nvrtc_available()
}
