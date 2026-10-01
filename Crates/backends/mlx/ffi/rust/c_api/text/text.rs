//! Bounded retained native string reads; no foreign pointer escapes.

#[rustfmt::skip]
use std::{
    ffi::c_char,
    ptr::NonNull,
};
use crate::MlxError;

pub(super) fn bounded_text(pointer: *const c_char, capacity: usize) -> Result<String, MlxError> {
    let pointer =
        NonNull::new(pointer.cast_mut()).ok_or_else(|| MlxError::Abi("nil C text".into()))?;
    for length in 0..capacity {
        // SAFETY: audited native std::string c_str() remains retained. Inspect successive
        // accessible characters up to its first nul, never create a slice beyond that nul.
        if unsafe { *pointer.as_ptr().add(length) } == 0 {
            // SAFETY: every inspected byte through length is accessible under retained owner.
            let bytes =
                unsafe { std::slice::from_raw_parts(pointer.as_ptr().cast::<u8>(), length) };
            return String::from_utf8(bytes.to_vec())
                .map_err(|_| MlxError::Abi("invalid C UTF8".into()));
        }
    }
    Err(MlxError::Abi("C text exceeds boundary".into()))
}
