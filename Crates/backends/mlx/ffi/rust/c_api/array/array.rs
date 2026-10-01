//! Owned dense initialized F32 metadata validation and transactional terminal readback.

#[rustfmt::skip]
use std::{
    ptr::NonNull,
};
use crate::MlxError;
#[rustfmt::skip]
use super::{
    abi::CArray,
    owner::Owner,
    session::Session,
};
use super::super::dimensions;

pub struct Array {
    pub(super) owner: Owner<CArray>,
    pub(super) session: Session,
    pub(super) shape: [usize; 2],
}

impl Array {
    pub(super) fn validate(&self) -> Result<(), MlxError> {
        self.owner.require_live()?;
        let dims = dimensions(self.shape)?;
        let api = &self.session.0.api;
        // SAFETY: live nonempty initialized C owner; metadata getters are selected catch-all
        // endpoints. No borrowed metadata is read until both rank and exact shape succeed.
        let (dtype, rank) = api.guarded(|| unsafe {
            (
                (api.array_dtype)(self.owner.raw),
                (api.array_ndim)(self.owner.raw),
            )
        })?;
        if dtype != 10 || rank != 2 {
            return Err(MlxError::Abi("unexpected C array dtype/rank".into()));
        }
        // SAFETY: exact rank2 is already proved before indexed dimension access.
        let (rows, columns, count, bytes) = api.guarded(|| unsafe {
            (
                (api.array_dim)(self.owner.raw, 0),
                (api.array_dim)(self.owner.raw, 1),
                (api.array_size)(self.owner.raw),
                (api.array_nbytes)(self.owner.raw),
            )
        })?;
        if [rows, columns] != dims
            || count != self.shape[0] * self.shape[1]
            || bytes != count * size_of::<f32>()
        {
            return Err(MlxError::Abi("unexpected C array dtype/shape/bytes".into()));
        }
        // SAFETY: validated rank2; strides pointer borrows two elements from retained owner.
        let strides = api.guarded(|| unsafe { (api.array_strides)(self.owner.raw) })?;
        let strides = NonNull::new(strides.cast_mut())
            .ok_or_else(|| MlxError::Abi("nil C strides".into()))?;
        // SAFETY: exact array rank establishes two retained native stride entries.
        let actual = unsafe { std::slice::from_raw_parts(strides.as_ptr(), 2) };
        if actual != [self.shape[1], 1] {
            return Err(MlxError::Abi("non-dense C matrix".into()));
        }
        Ok(())
    }
    pub const fn shape(&self) -> [usize; 2] {
        self.shape
    }
    #[cfg(feature = "tensor")]
    pub fn same_session(&self, session: &Session) -> bool {
        self.session.same(session)
    }
    pub fn read(&self, output: &mut [f32]) -> Result<(), MlxError> {
        self.session.ensure_ready()?;
        self.validate()?;
        let count = self.shape[0]
            .checked_mul(self.shape[1])
            .ok_or(MlxError::InvalidExtent)?;
        if output.len() != count {
            return Err(MlxError::InvalidExtent);
        }
        let api = &self.session.0.api;
        let retained = self.session.clone_array(self)?;
        let materialize = (|| {
            // SAFETY: evaluated result or copied input, retained descriptor and explicit GPU
            // stream; no unchecked pointer is read before terminal/availability validation.
            api.status(|| unsafe { (api.array_eval)(retained.raw) })?;
            api.status(|| unsafe { (api.synchronize)(self.session.0.stream.raw) })?;
            api.status(|| unsafe { (api.array_wait)(retained.raw) })?;
            let mut available = false;
            let mut contiguous = false;
            api.status(|| unsafe { (api.array_available)(&raw mut available, retained.raw) })?;
            api.status(|| unsafe { (api.array_contiguous)(&raw mut contiguous, retained.raw) })?;
            if !available || !contiguous {
                return Err(MlxError::Abi("C backing unavailable/noncontiguous".into()));
            }
            // SAFETY: terminal contiguous F32 backing; original owner retains the same data
            // after the official cloned holder is freed before transactional publication.
            api.guarded(|| unsafe { (api.array_data)(retained.raw) })
        })();
        let data = match materialize {
            Ok(data) => data,
            Err(error) => {
                self.session.0.poisoned.set(true);
                std::mem::forget(retained);
                std::mem::forget(self.session.clone());
                return Err(MlxError::CompletionUnknown(error.to_string()));
            }
        };
        let data = NonNull::new(data.cast_mut())
            .ok_or_else(|| MlxError::Abi("nil C F32 backing".into()))?;
        retained.release()?;
        // SAFETY: all fallible native work is complete; original owner retains exact initialized
        // contiguous count F32 storage, distinct from the exclusive caller output extent.
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), output.as_mut_ptr(), count) };
        Ok(())
    }
}
