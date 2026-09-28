//! Retained execution-domain roots; tensor machinery is initialized only when requested.

use std::rc::Rc;
#[cfg(feature = "tensor")]
use std::cell::OnceCell;
use fusion_pcu_rocm::RocmOwnedDispatchBackend;
#[cfg(feature = "tensor")]
#[rustfmt::skip]
use fusion_pcu_rocm::{
    RocblasError,
    RocmOwnedTensorAssessor,
    RocmTensorAssessor,
};

/// One facade execution domain. Resident owners retain this root independently of cache entries.
/// A Dispatch-only root does not load rocBLAS or create tensor execution state.
pub(super) struct RocmSession {
    #[cfg(feature = "tensor")]
    tensor: OnceCell<RocmOwnedTensorAssessor>,
    backend: Rc<RocmOwnedDispatchBackend>,
    block_size: u32,
}

impl RocmSession {
    pub(super) fn new(backend: RocmOwnedDispatchBackend, block_size: u32) -> Self {
        Self {
            #[cfg(feature = "tensor")]
            tensor: OnceCell::new(),
            backend: Rc::new(backend),
            block_size,
        }
    }

    pub(super) fn backend(&self) -> &RocmOwnedDispatchBackend {
        &self.backend
    }

    pub(super) const fn block_size(&self) -> u32 {
        self.block_size
    }

    /// Borrow persistent tensor state without cloning a root on the warm path.
    #[cfg(feature = "tensor")]
    pub(super) fn tensor_assessor(&self) -> Result<RocmTensorAssessor<'_>, RocblasError> {
        if self.tensor.get().is_none() {
            let state = RocmOwnedTensorAssessor::new(Rc::clone(&self.backend))?;
            // Thread-owned roots cannot race. Failed initialization leaves the cell retryable.
            if self.tensor.set(state).is_err() {
                unreachable!("thread-owned tensor state initialized concurrently");
            }
        }
        Ok(self
            .tensor
            .get()
            .expect("tensor state initialized")
            .assessor())
    }
}
