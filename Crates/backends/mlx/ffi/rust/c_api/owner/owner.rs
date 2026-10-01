//! Exactly one official C-holder release and an image lease per opaque owner.

use std::rc::Rc;
use crate::MlxError;
#[rustfmt::skip]
use super::{
    abi::{Free, Opaque},
    api::Api,
};

pub(super) struct Owner<H: Opaque> {
    api: Rc<Api>,
    pub(super) raw: H,
    free: Free<H>,
    live: bool,
}
impl<H: Opaque> Owner<H> {
    pub(super) fn empty(api: Rc<Api>, free: Free<H>) -> Self {
        Self {
            api,
            raw: H::empty(),
            free,
            live: true,
        }
    }
    pub(super) fn require_live(&self) -> Result<(), MlxError> {
        if self.raw.is_empty() {
            Err(MlxError::Abi("unexpected empty C owner".into()))
        } else {
            Ok(())
        }
    }
    pub(super) fn release(mut self) -> Result<(), MlxError> {
        self.live = false;
        // SAFETY: unique official holder; null sentinel is an upstream valid free input.
        let result = self.api.status(|| unsafe { (self.free)(self.raw) });
        if result.is_err() {
            std::mem::forget(Rc::clone(&self.api));
        }
        result
    }
}
impl<H: Opaque> Drop for Owner<H> {
    fn drop(&mut self) {
        if self.live {
            // SAFETY: exactly one free for this holder; other handles only use official set.
            if self
                .api
                .status(|| unsafe { (self.free)(self.raw) })
                .is_err()
            {
                std::mem::forget(Rc::clone(&self.api));
            }
        }
    }
}
