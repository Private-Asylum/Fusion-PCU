//! Backend-neutral unsupported generic PCU implementation.

use super::{
    PcuBaseContract,
    PcuControlContract,
    PcuError,
    PcuExecutorClaim,
    PcuExecutorDescriptor,
    PcuExecutorId,
    PcuSupport,
};

/// Unsupported generic PCU provider placeholder.
#[derive(Debug, Clone, Copy, Default)]
pub struct UnsupportedPcu;

impl UnsupportedPcu {
    /// Creates a new unsupported generic PCU provider placeholder.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl PcuBaseContract for UnsupportedPcu {
    fn support(&self) -> PcuSupport {
        PcuSupport::unsupported()
    }

    fn executors(&self) -> &'static [PcuExecutorDescriptor] {
        &[]
    }
}

impl PcuControlContract for UnsupportedPcu {
    fn claim_executor(&self, _executor: PcuExecutorId) -> Result<PcuExecutorClaim, PcuError> {
        Err(PcuError::unsupported())
    }

    fn release_executor(&self, _claim: PcuExecutorClaim) -> Result<(), PcuError> {
        Err(PcuError::unsupported())
    }
}
