//! Error types for generic PCU backends.

use core::fmt;

/// Kind of failure returned by a generic PCU backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PcuErrorKind {
    Unsupported,
    Invalid,
    Busy,
    ResourceExhausted,
    StateConflict,
    Platform(i32),
}

/// Error returned by a generic PCU backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PcuError {
    kind: PcuErrorKind,
}

impl PcuError {
    #[must_use]
    pub const fn unsupported() -> Self {
        Self {
            kind: PcuErrorKind::Unsupported,
        }
    }

    #[must_use]
    pub const fn invalid() -> Self {
        Self {
            kind: PcuErrorKind::Invalid,
        }
    }

    #[must_use]
    pub const fn busy() -> Self {
        Self {
            kind: PcuErrorKind::Busy,
        }
    }

    #[must_use]
    pub const fn resource_exhausted() -> Self {
        Self {
            kind: PcuErrorKind::ResourceExhausted,
        }
    }

    #[must_use]
    pub const fn state_conflict() -> Self {
        Self {
            kind: PcuErrorKind::StateConflict,
        }
    }

    #[must_use]
    pub const fn platform(code: i32) -> Self {
        Self {
            kind: PcuErrorKind::Platform(code),
        }
    }

    #[must_use]
    pub const fn kind(self) -> PcuErrorKind {
        self.kind
    }
}

impl fmt::Display for PcuErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::Unsupported => f.write_str("pcu operation unsupported"),
            Self::Invalid => f.write_str("invalid pcu request"),
            Self::Busy => f.write_str("pcu resource busy"),
            Self::ResourceExhausted => f.write_str("pcu resources exhausted"),
            Self::StateConflict => f.write_str("pcu state conflict"),
            Self::Platform(code) => write!(f, "platform pcu error {code}"),
        }
    }
}

impl fmt::Display for PcuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.kind.fmt(f)
    }
}
