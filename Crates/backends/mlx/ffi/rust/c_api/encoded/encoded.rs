//! Logical byte-aligned encodings carried by verified immutable official `UInt` arrays.
#[rustfmt::skip]
use std::{ptr::NonNull,rc::Rc,cell::Cell};
use fusion_pcu::PcuScalarType;
use crate::MlxError;
#[rustfmt::skip]
use super::{abi::{CArray,CCarrier},owner::Owner,session::Session,checked::{validate,available}};

pub struct EncodedArray {
    owner: Option<Owner<CArray>>,
    session: Session,
    scalar: PcuScalarType,
    count: usize,
    dtype: i32,
    width: usize,
    carrier_count: usize,
    carrier_width: usize,
}
#[path = "carrier/carrier.rs"]
pub(super) mod carrier;
impl EncodedArray {
    pub fn upload(
        session: &Session,
        scalar: PcuScalarType,
        count: usize,
        data: &[u8],
    ) -> Result<Self, MlxError> {
        let carrier = carrier::Carrier::assess(scalar, count)?;
        if carrier.byte_len != data.len() {
            return Err(MlxError::InvalidExtent);
        }
        session.ensure_ready()?;
        let api = &session.0.api;
        let mut owner = Owner::empty(Rc::clone(api), api.array_free);
        // SAFETY: initialized exact logical byte span. The audited endpoint copies via aligned
        // UInt storage into native ownership before returning; caller storage never escapes.
        api.status(|| unsafe {
            (api.checked_upload)(
                &raw mut owner.raw,
                data.as_ptr().cast(),
                carrier.count,
                carrier.upload_format,
            )
        })?;
        Self::from_owner(session, scalar, count, owner)
    }
    pub(super) fn from_owner(
        session: &Session,
        scalar: PcuScalarType,
        count: usize,
        owner: Owner<CArray>,
    ) -> Result<Self, MlxError> {
        let carrier = carrier::Carrier::assess(scalar, count)?;
        owner.require_live()?;
        validate(
            session,
            owner.raw,
            carrier.dtype,
            carrier.count,
            carrier.width,
        )?;
        Ok(Self {
            owner: Some(owner),
            session: session.clone(),
            scalar,
            count,
            dtype: carrier.dtype,
            width: carrier.logical_width,
            carrier_count: carrier.count,
            carrier_width: carrier.width,
        })
    }
    pub fn native_f32_view(&self, shape: [usize; 2]) -> Result<super::array::Array, MlxError> {
        if self.scalar != PcuScalarType::F32 {
            return Err(MlxError::UnsupportedScalar(self.scalar));
        }
        if shape[0].checked_mul(shape[1]) != Some(self.count) {
            return Err(MlxError::InvalidExtent);
        }
        #[cfg(feature = "view-census")]
        super::view::census::record(super::view::census::Call::Clone);
        let input = self.clone_holder()?;
        let owner = super::view::complete_view(&self.session, input, shape, true)?;
        let array = super::array::Array {
            owner,
            session: self.session.clone(),
            shape,
        };
        array.validate()?;
        Ok(array)
    }
    pub const fn scalar(&self) -> PcuScalarType {
        self.scalar
    }
    pub const fn count(&self) -> usize {
        self.count
    }
    pub const fn byte_len(&self) -> usize {
        self.count * self.width
    }
    pub fn same_session(&self, session: &Session) -> bool {
        self.session.same(session)
    }
    /// Executes the retained integer carrier primitive without warm shader/profile lowering.
    pub(super) fn copy_with(
        &self,
        primitive: CCarrier,
        count: usize,
        may_write: &Cell<bool>,
    ) -> Result<Self, MlxError> {
        may_write.set(false);
        let carrier = carrier::Carrier::assess(self.scalar, count)?;
        let input = self.clone_holder()?;
        let api = &self.session.0.api;
        let mut result = Owner::empty(Rc::clone(api), api.array_free);
        // SAFETY: real verified integer carrier, exact bounded repetition and explicit retained
        // primitive. The catch-all endpoint creates only a lazy result from that retained primitive.
        api.status(|| unsafe { (api.carrier_apply)(&raw mut result.raw, primitive, input.raw) })?;
        validate(
            &self.session,
            result.raw,
            carrier.dtype,
            carrier.count,
            carrier.width,
        )?;
        let pending = (input, result, self.session.clone());
        let terminal = (|| {
            may_write.set(true);
            // SAFETY: actual input/result/session/image remain retained across every potentially
            // asynchronous endpoint; no preexisting array can donate its original backing.
            api.status(|| unsafe { (api.array_eval)(pending.1.raw) })?;
            api.status(|| unsafe { (api.synchronize)(self.session.0.stream.raw) })?;
            api.status(|| unsafe { (api.array_wait)(pending.1.raw) })?;
            available(&self.session, pending.1.raw)
        })();
        if let Err(error) = terminal {
            self.session.0.poisoned.set(true);
            std::mem::forget(pending);
            return Err(MlxError::CompletionUnknown(error.to_string()));
        }
        let (input, result, _session) = pending;
        input.release()?;
        Self::from_owner(&self.session, self.scalar, count, result)
    }
    /// Checked completed-holder cleanup, used before the next host publication.
    pub fn release(mut self) -> Result<(), MlxError> {
        self.session.ensure_ready()?;
        self.owner
            .take()
            .ok_or_else(|| MlxError::Abi("missing encoded holder".into()))?
            .release()
    }
    /// Immutable device prefix replacement with both actual input holders retained.
    pub fn merge_prefix(&self, prefix: &Self) -> Result<Self, MlxError> {
        if prefix.scalar != self.scalar {
            return Err(MlxError::UnsupportedScalar(prefix.scalar));
        }
        if !prefix.same_session(&self.session) {
            return Err(MlxError::ForeignSession);
        }
        if prefix.count >= self.count {
            return Err(MlxError::InvalidExtent);
        }
        let old = self.clone_holder()?;
        let update = prefix.clone_holder()?;
        let api = &self.session.0.api;
        let mut result = Owner::empty(Rc::clone(api), api.array_free);
        // SAFETY: audited exception-containing endpoint receives exact dense logical UInt
        // holders and the retained GPU stream; descriptor construction is lazy.
        api.status(|| unsafe {
            (api.encoded_prefix)(
                &raw mut result.raw,
                update.raw,
                old.raw,
                self.session.0.stream.raw,
            )
        })?;
        validate(
            &self.session,
            result.raw,
            self.dtype,
            self.carrier_count,
            self.carrier_width,
        )?;
        let pending = (old, update, result, self.session.clone());
        let terminal = (|| {
            // SAFETY: real previous/prefix/result owners and stream/image remain retained
            // throughout potential device work. No original owner can donate its backing.
            api.status(|| unsafe { (api.array_eval)(pending.2.raw) })?;
            api.status(|| unsafe { (api.synchronize)(self.session.0.stream.raw) })?;
            api.status(|| unsafe { (api.array_wait)(pending.2.raw) })?;
            available(&self.session, pending.2.raw)
        })();
        if let Err(error) = terminal {
            self.session.0.poisoned.set(true);
            std::mem::forget(pending);
            return Err(MlxError::CompletionUnknown(error.to_string()));
        }
        let (old, update, result, _session) = pending;
        old.release()?;
        update.release()?;
        Self::from_owner(&self.session, self.scalar, self.count, result)
    }
    pub(super) fn clone_holder(&self) -> Result<Owner<CArray>, MlxError> {
        self.session.ensure_ready()?;
        let owner = self
            .owner
            .as_ref()
            .ok_or_else(|| MlxError::Abi("missing encoded holder".into()))?;
        validate(
            &self.session,
            owner.raw,
            self.dtype,
            self.carrier_count,
            self.carrier_width,
        )?;
        let api = &self.session.0.api;
        let mut clone = Owner::empty(Rc::clone(api), api.array_free);
        // SAFETY: official set retains the actual immutable backing in a distinct owned holder.
        // No raw pointer is imported and no host materialization or data copy occurs here.
        api.status(|| unsafe { (api.array_set)(&raw mut clone.raw, owner.raw) })?;
        clone.require_live()?;
        Ok(clone)
    }
    pub fn read(&self, output: &mut [u8]) -> Result<(), MlxError> {
        if output.len() != self.byte_len() {
            return Err(MlxError::InvalidExtent);
        }
        let retained = self.clone_holder()?;
        let api = &self.session.0.api;
        let pending = (retained, self.session.clone());
        let terminal = (|| {
            // SAFETY: pending retains the actual backing and explicit session/image across every
            // potentially asynchronous endpoint. No caller output is touched before completion.
            api.status(|| unsafe { (api.array_eval)(pending.0.raw) })?;
            api.status(|| unsafe { (api.synchronize)(self.session.0.stream.raw) })?;
            api.status(|| unsafe { (api.array_wait)(pending.0.raw) })?;
            available(&self.session, pending.0.raw)
        })();
        if let Err(error) = terminal {
            self.session.0.poisoned.set(true);
            std::mem::forget(pending);
            return Err(MlxError::CompletionUnknown(error.to_string()));
        }
        // SAFETY: verified dense, initialized terminal UInt carrier has the exact logical byte span.
        let data = api.guarded(|| unsafe {
            match self.dtype {
                1 => (api.array_data_u8)(pending.0.raw).cast::<u8>(),
                2 => (api.array_data_u16)(pending.0.raw).cast::<u8>(),
                3 => (api.array_data_u32)(pending.0.raw).cast::<u8>(),
                _ => std::ptr::null(),
            }
        })?;
        let data = NonNull::new(data.cast_mut())
            .ok_or_else(|| MlxError::Abi("nil encoded carrier backing".into()))?;
        pending.0.release()?;
        // SAFETY: original owner retains the same immutable initialized backing. The exclusive
        // caller prefix is disjoint; every fallible operation completed before publication.
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), output.as_mut_ptr(), output.len());
        }
        Ok(())
    }
}
impl Drop for EncodedArray {
    fn drop(&mut self) {
        if self.session.ensure_ready().is_err() {
            if let Some(owner) = self.owner.take() {
                std::mem::forget(owner);
            }
            std::mem::forget(self.session.clone());
        }
    }
}
