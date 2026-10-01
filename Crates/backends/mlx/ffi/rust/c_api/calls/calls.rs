//! Calling-thread nested scoped error capture; no global handler replacement.

use crate::MlxError;
#[rustfmt::skip]
use super::{
    abi::ErrorScope,
    api::Api,
};
#[rustfmt::skip]
use super::super::{
    text_value,
    ERROR_BYTES,
};

impl Api {
    pub(super) fn guarded<T>(&self, function: impl FnOnce() -> T) -> Result<T, MlxError> {
        let mut message = [0u8; ERROR_BYTES];
        let mut scope = ErrorScope {
            previous: std::ptr::null_mut(),
            message: std::ptr::null_mut(),
            capacity: 0,
            failed: 0,
            active: 0,
        };
        // SAFETY: caller-owned scope and buffer stay at stable addresses until matching end;
        // callbacks use nested scopes on the same thread, restoring their parent LIFO.
        if unsafe { (self.begin)(&raw mut scope, message.as_mut_ptr().cast(), message.len()) } != 0
        {
            return Err(MlxError::Abi("C error scope rejected".into()));
        }
        let value = function();
        // SAFETY: exactly this active scope is topmost after the audited nonthrowing call.
        let end = unsafe { (self.end)(&raw mut scope) };
        if end != 0 {
            return Err(MlxError::Abi("C error scope order corrupted".into()));
        }
        if scope.failed != 0 {
            return Err(MlxError::Runtime(text_value(&message)?));
        }
        Ok(value)
    }

    pub(super) fn status(&self, function: impl FnOnce() -> i32) -> Result<(), MlxError> {
        if self.guarded(function)? == 0 {
            Ok(())
        } else {
            Err(MlxError::Runtime(
                "upstream C returned failure without text".into(),
            ))
        }
    }
}
