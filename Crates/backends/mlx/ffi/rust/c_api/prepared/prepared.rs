//! Cold public-C compilation and retained primitive replay with exact input/session identity.

#[rustfmt::skip]
use std::{
    cell::Cell,
    ffi::c_void,
    ptr::NonNull,
    rc::Rc,
};
use crate::MlxError;
#[rustfmt::skip]
use super::{
    abi::CClosure,
    array::Array,
    callback::{trace_callback, trace_drop, Trace},
    owner::Owner,
    session::{Pending, Session},
};
use super::super::dimensions;

pub struct PreparedMatmul {
    session: Session,
    #[cfg_attr(
        not(feature = "benchmark-control"),
        allow(
            dead_code,
            reason = "Retain the compiler closure/cache entry for the prepared primitive lifetime."
        )
    )]
    compiled: Owner<CClosure>,
    trace: Rc<Trace>,
    replay: NonNull<c_void>,
    left_shape: [usize; 2],
    right_shape: [usize; 2],
}
impl Session {
    pub fn prepare_matmul(
        &self,
        left_shape: [usize; 2],
        right_shape: [usize; 2],
    ) -> Result<PreparedMatmul, MlxError> {
        self.ensure_ready()?;
        dimensions(left_shape)?;
        dimensions(right_shape)?;
        if left_shape[1] != right_shape[0] {
            return Err(MlxError::InvalidExtent);
        }
        dimensions([left_shape[0], right_shape[1]])?;
        let api = &self.0.api;
        let trace = Rc::new(Trace {
            session: self.clone(),
            traces: Cell::new(0),
        });
        let payload = Rc::into_raw(Rc::clone(&trace)).cast_mut().cast::<c_void>();
        let mut function = Owner::empty(Rc::clone(api), api.closure_free);
        // SAFETY: stable Rc payload is consumed by official payload closure even on creation
        // failure; callback/drop never unwind and retain exact API/session/stream lifetime.
        api.guarded(|| {
            // SAFETY: native closure consumes the strong payload reference on success/failure.
            function.raw = unsafe { (api.closure_new)(trace_callback, payload, trace_drop) };
        })?;
        function.require_live()?;
        let mut compiled = Owner::empty(Rc::clone(api), api.closure_free);
        // SAFETY: official owned callback closure, fixed shapes; no global mode/cache mutation.
        api.status(|| unsafe { (api.compile)(&raw mut compiled.raw, function.raw, false) })?;
        // Public C has no unmaterialized descriptor constructor: these copied zero descriptors
        // are cold-only. They are never evaluated as arithmetic or published to the caller.
        let left = self.upload(left_shape, &vec![0.0; left_shape[0] * left_shape[1]])?;
        let right = self.upload(right_shape, &vec![0.0; right_shape[0] * right_shape[1]])?;
        let mut inputs = Owner::empty(Rc::clone(api), api.vector_free);
        let handles = [left.owner.raw, right.owner.raw];
        // SAFETY: two live official input owners, vector copies array descriptors.
        api.status(|| unsafe {
            (api.vector_set_data)(&raw mut inputs.raw, handles.as_ptr(), handles.len())
        })?;
        let mut outputs = Owner::empty(Rc::clone(api), api.vector_free);
        // SAFETY: compiled closure invokes the nonthrowing callback under this nested TLS scope.
        api.status(|| unsafe {
            (api.closure_apply)(&raw mut outputs.raw, compiled.raw, inputs.raw)
        })?;
        // SAFETY: successfully owned vector; selected getter is catch-all.
        if api.guarded(|| unsafe { (api.vector_size)(outputs.raw) })? != 1
            || trace.traces.get() != 1
        {
            return Err(MlxError::Abi(
                "C compiler did not produce one cold MatMul trace".into(),
            ));
        }
        let mut output = Owner::empty(Rc::clone(api), api.array_free);
        // SAFETY: one output; official get clones an independent holder.
        api.status(|| unsafe { (api.vector_get)(&raw mut output.raw, outputs.raw, 0) })?;
        let mut replay = std::ptr::null_mut();
        // SAFETY: safety extension validates one native Matmul primitive/inputs/shape/dtype
        // and explicit stream before retaining its primitive. It contains no operator mirror.
        api.status(|| unsafe {
            (api.replay_new)(
                output.raw,
                left.owner.raw,
                right.owner.raw,
                self.0.stream.raw,
                &raw mut replay,
            )
        })?;
        Ok(PreparedMatmul {
            session: self.clone(),
            compiled,
            trace,
            replay: NonNull::new(replay).ok_or_else(|| MlxError::Abi("nil C replay".into()))?,
            left_shape,
            right_shape,
        })
    }
}
impl PreparedMatmul {
    pub fn traces(&self) -> usize {
        self.trace.traces.get()
    }
    #[cfg(feature = "benchmark-control")]
    pub fn execute_compiled(&self, left: &Array, right: &Array) -> Result<Array, MlxError> {
        self.session.ensure_ready()?;
        let shape = self.session.operands(left, right)?;
        if left.shape != self.left_shape || right.shape != self.right_shape {
            return Err(MlxError::InvalidExtent);
        }
        let api = &self.session.0.api;
        let mut pending = Pending {
            _left: self.session.clone_array(left)?,
            _right: self.session.clone_array(right)?,
            output: Array {
                session: self.session.clone(),
                owner: Owner::empty(Rc::clone(api), api.array_free),
                shape,
            },
        };
        let pair = [left.owner.raw, right.owner.raw];
        let mut inputs = Owner::empty(Rc::clone(api), api.vector_free);
        let mut outputs = Owner::empty(Rc::clone(api), api.vector_free);
        // SAFETY: same-session frozen shapes and official vector holders; the public C
        // wrapper retains actual descriptors and calls only the nonthrowing trace callback.
        api.status(|| unsafe {
            (api.vector_set_data)(&raw mut inputs.raw, pair.as_ptr(), pair.len())
        })?;
        api.status(|| unsafe {
            (api.closure_apply)(&raw mut outputs.raw, self.compiled.raw, inputs.raw)
        })?;
        if api.guarded(|| unsafe { (api.vector_size)(outputs.raw) })? != 1 || self.traces() != 1 {
            return Err(MlxError::Abi(
                "public C compiled wrapper retraced/changed outputs".into(),
            ));
        }
        api.status(|| unsafe {
            (api.vector_get)(&raw mut pending.output.owner.raw, outputs.raw, 0)
        })?;
        pending.output.validate()?;
        self.session.complete(pending)
    }

    pub fn execute(&self, left: &Array, right: &Array) -> Result<Array, MlxError> {
        self.session.ensure_ready()?;
        let shape = self.session.operands(left, right)?;
        if left.shape != self.left_shape || right.shape != self.right_shape {
            return Err(MlxError::InvalidExtent);
        }
        let api = &self.session.0.api;
        let mut pending = Pending {
            _left: self.session.clone_array(left)?,
            _right: self.session.clone_array(right)?,
            output: Array {
                session: self.session.clone(),
                owner: Owner::empty(Rc::clone(api), api.array_free),
                shape,
            },
        };
        // SAFETY: validated frozen replay, exact new input owners and output setter sentinel;
        // warm extension only rebinds retained primitive, no compiler/default/cache/name work.
        api.status(|| unsafe {
            (api.replay_apply)(
                &raw mut pending.output.owner.raw,
                self.replay.as_ptr(),
                left.owner.raw,
                right.owner.raw,
            )
        })?;
        pending.output.validate()?;
        self.session.complete(pending)
    }
}
impl Drop for PreparedMatmul {
    fn drop(&mut self) {
        let api = &self.session.0.api;
        // SAFETY: exactly one private replay owner, selected catch-all destructor; compiled
        // closure and trace/session stay retained through its final release.
        if api
            .status(|| unsafe { (api.replay_free)(self.replay.as_ptr()) })
            .is_err()
        {
            std::mem::forget(self.session.clone());
        }
    }
}
