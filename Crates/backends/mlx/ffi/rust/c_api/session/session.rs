//! Thread-confined explicit GPU stream, upload/clone and terminal pending-owner protocol.

#[rustfmt::skip]
use std::{
    cell::Cell,
    rc::Rc,
};
use crate::MlxError;
#[rustfmt::skip]
use super::{
    abi::{CArray, CDevice, CStream},
    api::Api,
    array::Array,
    owner::Owner,
};
use super::super::dimensions;

pub(super) struct SessionInner {
    pub(super) api: Rc<Api>,
    _device: Owner<CDevice>,
    pub(super) stream: Owner<CStream>,
    pub(super) poisoned: Cell<bool>,
}
#[derive(Clone)]
pub struct Session(pub(super) Rc<SessionInner>);

impl Api {
    pub fn open(self: &Rc<Self>, index: usize) -> Result<Session, MlxError> {
        let device = self.device_owner(index)?;
        let mut stream = Owner::empty(Rc::clone(self), self.stream_free);
        // SAFETY: explicit live GPU device, constructor owns a fresh stream/defaults untouched.
        self.guarded(|| {
            // SAFETY: live explicit GPU owner outlives the constructed stream.
            stream.raw = unsafe { (self.stream_new)(device.raw) };
        })?;
        stream.require_live()?;
        Ok(Session(Rc::new(SessionInner {
            api: Rc::clone(self),
            _device: device,
            stream,
            poisoned: Cell::new(false),
        })))
    }
}

impl Session {
    pub fn same(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.0, &other.0)
    }
    pub fn ensure_ready(&self) -> Result<(), MlxError> {
        if self.0.poisoned.get() {
            Err(MlxError::CompletionUnknown(
                "direct C session quarantined".into(),
            ))
        } else {
            Ok(())
        }
    }
    pub fn upload(&self, shape: [usize; 2], data: &[f32]) -> Result<Array, MlxError> {
        self.ensure_ready()?;
        let dims = dimensions(shape)?;
        if shape[0].checked_mul(shape[1]) != Some(data.len()) {
            return Err(MlxError::InvalidExtent);
        }
        let api = &self.0.api;
        let mut owner = Owner::empty(Rc::clone(api), api.array_free);
        // SAFETY: dimensions/checked bytes/exact initialized host extent validated; official
        // set_data copies into native owned storage before returning, never borrows caller data.
        api.status(|| unsafe {
            (api.array_set_data)(
                &raw mut owner.raw,
                data.as_ptr().cast(),
                dims.as_ptr(),
                2,
                10,
            )
        })?;
        let array = Array {
            session: self.clone(),
            owner,
            shape,
        };
        array.validate()?;
        Ok(array)
    }
    pub(super) fn clone_array(&self, array: &Array) -> Result<Owner<CArray>, MlxError> {
        let api = &self.0.api;
        let mut owner = Owner::empty(Rc::clone(api), api.array_free);
        // SAFETY: official descriptor clone/set shares backing under a new independently
        // freed holder. Copying raw CArray into an owning wrapper is deliberately forbidden.
        api.status(|| unsafe { (api.array_set)(&raw mut owner.raw, array.owner.raw) })?;
        owner.require_live()?;
        Ok(owner)
    }
    #[cfg(feature = "tensor")]
    pub(super) fn complete(&self, pending: Pending) -> Result<Array, MlxError> {
        let api = &self.0.api;
        let result = (|| {
            // SAFETY: pending retains output, input clones, stream and library before eval.
            api.status(|| unsafe { (api.array_eval)(pending.output.owner.raw) })?;
            // SAFETY: explicit stream is retained and all graph nodes use this GPU stream.
            api.status(|| unsafe { (api.synchronize)(self.0.stream.raw) })?;
            // SAFETY: output remains valid until terminal array completion is observed.
            api.status(|| unsafe { (api.array_wait)(pending.output.owner.raw) })
        })();
        if let Err(error) = result {
            self.0.poisoned.set(true);
            // Retain real accessed holders/backings and the stream/library after any unknown
            // terminal failure; no caller storage was published or retained by this attempt.
            std::mem::forget(pending);
            return Err(MlxError::CompletionUnknown(error.to_string()));
        }
        pending.output.validate()?;
        Ok(pending.output)
    }
    #[cfg(any(all(test, feature = "tensor"), feature = "benchmark-control"))]
    pub fn matmul(&self, left: &Array, right: &Array) -> Result<Array, MlxError> {
        self.ensure_ready()?;
        let shape = self.operands(left, right)?;
        let api = &self.0.api;
        let mut pending = Pending {
            _left: self.clone_array(left)?,
            _right: self.clone_array(right)?,
            output: Array {
                session: self.clone(),
                owner: Owner::empty(Rc::clone(api), api.array_free),
                shape,
            },
        };
        // SAFETY: compatible validated same-session dense F32 arrays and explicit GPU stream;
        // this calls upstream mlx_matmul directly, constructing a lazy native graph node.
        api.status(|| unsafe {
            (api.matmul)(
                &raw mut pending.output.owner.raw,
                left.owner.raw,
                right.owner.raw,
                self.0.stream.raw,
            )
        })?;
        pending.output.validate()?;
        self.complete(pending)
    }
    #[cfg(feature = "tensor")]
    pub(super) fn operands(&self, left: &Array, right: &Array) -> Result<[usize; 2], MlxError> {
        if !self.same(&left.session) || !self.same(&right.session) {
            return Err(MlxError::ForeignSession);
        }
        if left.shape[1] != right.shape[0] {
            return Err(MlxError::InvalidExtent);
        }
        let shape = [left.shape[0], right.shape[1]];
        dimensions(shape)?;
        Ok(shape)
    }
}
#[cfg(feature = "tensor")]
pub(super) struct Pending {
    pub(super) _left: Owner<CArray>,
    pub(super) _right: Owner<CArray>,
    pub(super) output: Array,
}
