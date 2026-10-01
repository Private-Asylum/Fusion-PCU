//! Exact Rust identity proof for delegated F32 storage, without conversion or extra copies.

#[rustfmt::skip]
use std::{
    any::TypeId,
    mem::{
        align_of,
        size_of,
    },
};
use fusion_pcu::PcuScalar;
use crate::MlxError;

fn validate<T: PcuScalar>(pointer: *const T, length: usize) -> Result<(), MlxError> {
    if TypeId::of::<T>() != TypeId::of::<f32>() {
        return Err(MlxError::UnsupportedScalar(T::TYPE));
    }
    if size_of::<T>() != size_of::<f32>()
        || align_of::<T>() != align_of::<f32>()
        || !pointer.cast::<f32>().is_aligned()
        || length
            .checked_mul(size_of::<f32>())
            .is_none_or(|bytes| isize::try_from(bytes).is_err())
    {
        return Err(MlxError::InvalidExtent);
    }
    Ok(())
}

pub fn as_f32<T: PcuScalar>(data: &[T]) -> Result<&[f32], MlxError> {
    validate(data.as_ptr(), data.len())?;
    // SAFETY: TypeId proves T is exactly f32, independently of the trait's scalar tag.
    // The safe source slice owns a live initialized allocation, nonnull aligned pointer and
    // bounded extent. Identical layout/length and the returned source lifetime preserve access.
    Ok(unsafe { std::slice::from_raw_parts(data.as_ptr().cast(), data.len()) })
}

pub fn as_f32_mut<T: PcuScalar>(data: &mut [T]) -> Result<&mut [f32], MlxError> {
    validate(data.as_ptr(), data.len())?;
    // SAFETY: the exact TypeId/layout/bounds proof above applies unchanged. The exclusive
    // source borrow is transferred for the returned lifetime, with no alias or data conversion.
    Ok(unsafe { std::slice::from_raw_parts_mut(data.as_mut_ptr().cast(), data.len()) })
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
