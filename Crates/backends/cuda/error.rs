use std::fmt;

/// CUDA loading, discovery, and resource-operation failures.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CudaError {
    RuntimeUnavailable(String),
    InvalidModuleImage,
    MissingSymbol {
        symbol: &'static str,
        detail: String,
    },
    Runtime {
        operation: &'static str,
        code: i32,
        detail: Option<String>,
    },
    DeviceIndexOutOfRange {
        requested: u32,
        count: u32,
    },
    InvalidDiscoveryReference,
    MissingStableDeviceIdentity,
    MissingArchitecture,
    DeviceIdentityChanged {
        expected: String,
        actual: String,
    },
    DifferentRuntime,
    DifferentStream,
    BatchPoisoned,
    BatchNotComplete,
    InvalidReadbackId,
    ReadbackNotQueued,
    ReadbackAlreadyQueued,
    InvalidExecutionFaultWord(u64),
    Busy,
    InvalidLaunchDimensions,
    BufferTooSmall {
        allocation: usize,
        requested: usize,
    },
}

impl fmt::Display for CudaError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidModuleImage => formatter.write_str("CUDA module image is empty"),
            Self::RuntimeUnavailable(detail) => {
                write!(formatter, "CUDA runtime unavailable: {detail}")
            }
            Self::MissingSymbol { symbol, detail } => {
                write!(formatter, "CUDA symbol {symbol} unavailable: {detail}")
            }
            Self::Runtime {
                operation,
                code,
                detail,
            } => write!(
                formatter,
                "{operation} failed with CUDA status {code}{}",
                detail
                    .as_ref()
                    .map_or(String::new(), |text| format!(": {text}"))
            ),
            Self::DeviceIndexOutOfRange { requested, count } => write!(
                formatter,
                "CUDA device index {requested} is out of range (device count: {count})"
            ),
            Self::InvalidDiscoveryReference => formatter.write_str(
                "CUDA discovery reference belongs to another provider, generation, kind, or snapshot",
            ),
            Self::MissingStableDeviceIdentity => formatter.write_str(
                "cannot safely activate the CUDA device because CUDA did not provide a PCI bus ID",
            ),
            Self::MissingArchitecture => formatter.write_str(
                "cannot compile for the selected CUDA device because its exact GPU architecture could not be detected",
            ),
            Self::DeviceIdentityChanged { expected, actual } => write!(
                formatter,
                "CUDA device identity changed since discovery (expected PCI bus ID {expected}, found {actual})"
            ),
            Self::DifferentRuntime => formatter
                .write_str("CUDA resources belong to different runtime instances or devices"),
            Self::DifferentStream => {
                formatter.write_str("CUDA completion belongs to a different stream")
            }
            Self::BatchPoisoned => formatter.write_str(
                "CUDA completion batch cannot accept work after a launch failure",
            ),
            Self::BatchNotComplete => formatter
                .write_str("CUDA batch readback is unavailable before all completion dependencies succeed"),
            Self::InvalidReadbackId => formatter
                .write_str("CUDA readback identifier is foreign, invalid, or already consumed"),
            Self::ReadbackNotQueued => formatter
                .write_str("CUDA readback identifier has no queued device-to-host copy"),
            Self::ReadbackAlreadyQueued => formatter
                .write_str("CUDA readback identifier already has a queued device-to-host copy"),
            Self::InvalidExecutionFaultWord(word) => write!(
                formatter,
                "CUDA returned an invalid checked-division fault word {word:#x}"
            ),
            Self::Busy => formatter.write_str("CUDA device allocation is busy"),
            Self::InvalidLaunchDimensions => {
                formatter.write_str("CUDA grid and block dimensions must all be nonzero")
            }
            Self::BufferTooSmall {
                allocation,
                requested,
            } => write!(
                formatter,
                "copy of {requested} bytes exceeds {allocation}-byte CUDA allocation"
            ),
        }
    }
}
impl std::error::Error for CudaError {}
