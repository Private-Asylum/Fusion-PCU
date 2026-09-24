use std::fmt;

/// HIP loading, discovery, and resource-operation failures.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HipError {
    RuntimeUnavailable(String),
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
    DeviceIdentityChanged {
        expected: String,
        actual: String,
    },
    DifferentRuntime,
    Busy,
    InvalidLaunchDimensions,
    BufferTooSmall {
        allocation: usize,
        requested: usize,
    },
}

impl fmt::Display for HipError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RuntimeUnavailable(detail) => {
                write!(formatter, "HIP runtime unavailable: {detail}")
            }
            Self::MissingSymbol { symbol, detail } => {
                write!(formatter, "HIP symbol {symbol} unavailable: {detail}")
            }
            Self::Runtime {
                operation,
                code,
                detail,
            } => write!(
                formatter,
                "{operation} failed with HIP status {code}{}",
                detail
                    .as_ref()
                    .map_or(String::new(), |text| format!(": {text}"))
            ),
            Self::DeviceIndexOutOfRange { requested, count } => write!(
                formatter,
                "HIP device index {requested} is out of range (device count: {count})"
            ),
            Self::InvalidDiscoveryReference => formatter.write_str(
                "ROCm discovery reference belongs to another provider, generation, kind, or snapshot",
            ),
            Self::MissingStableDeviceIdentity => formatter.write_str(
                "cannot safely activate the ROCm device because HIP did not provide a PCI bus ID",
            ),
            Self::DeviceIdentityChanged { expected, actual } => write!(
                formatter,
                "ROCm device identity changed since discovery (expected PCI bus ID {expected}, found {actual})"
            ),
            Self::DifferentRuntime => formatter
                .write_str("HIP resources belong to different runtime instances or devices"),
            Self::Busy => formatter.write_str("HIP device allocation is busy"),
            Self::InvalidLaunchDimensions => {
                formatter.write_str("HIP grid and block dimensions must all be nonzero")
            }
            Self::BufferTooSmall {
                allocation,
                requested,
            } => write!(
                formatter,
                "copy of {requested} bytes exceeds {allocation}-byte HIP allocation"
            ),
        }
    }
}
impl std::error::Error for HipError {}
