//! Nonthrowing public-C compiler callback and consumed Rc payload lifetime.

#[rustfmt::skip]
use std::{
    cell::Cell,
    ffi::c_void,
    rc::Rc,
};
use crate::MlxError;
#[rustfmt::skip]
use super::{
    abi::CVector,
    owner::Owner,
    session::Session,
};

pub(super) struct Trace {
    pub(super) session: Session,
    pub(super) traces: Cell<usize>,
}
pub(super) unsafe extern "C" fn trace_callback(
    result: *mut CVector,
    input: CVector,
    payload: *mut c_void,
) -> i32 {
    // SAFETY: official payload closure retains a stable Rc<Trace>; PCU never passes null.
    let trace = unsafe { &*payload.cast::<Trace>() };
    // Panic containment is strictly Rust-callback containment. Foreign catches/reporting are
    // repaired in the native image; catch_unwind is not used as C++ exception containment.
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let api = &trace.session.0.api;
        trace.traces.set(trace.traces.get() + 1);
        if trace.traces.get() != 1 {
            return 1;
        }
        let apply = (|| {
            // SAFETY: upstream callback receives a valid vector owned for callback duration.
            if api.guarded(|| unsafe { (api.vector_size)(input) })? != 2 {
                return Err(MlxError::InvalidExtent);
            }
            let mut a = Owner::empty(Rc::clone(api), api.array_free);
            let mut b = Owner::empty(Rc::clone(api), api.array_free);
            let mut output = Owner::empty(Rc::clone(api), api.array_free);
            // SAFETY: exact indices and official cloning into independent output holders;
            // explicit retained GPU stream ensures no CPU tensor fallback.
            api.status(|| unsafe { (api.vector_get)(&raw mut a.raw, input, 0) })?;
            api.status(|| unsafe { (api.vector_get)(&raw mut b.raw, input, 1) })?;
            api.status(|| unsafe {
                (api.matmul)(
                    &raw mut output.raw,
                    a.raw,
                    b.raw,
                    trace.session.0.stream.raw,
                )
            })?;
            api.status(|| unsafe { (api.vector_set_value)(result, output.raw) })
        })();
        i32::from(apply.is_err())
    }));
    match outcome {
        Ok(status) => status,
        Err(payload) => {
            std::mem::forget(payload);
            1
        }
    }
}
pub(super) unsafe extern "C" fn trace_drop(payload: *mut c_void) {
    // SAFETY: exactly one consumed strong Rc payload; callback-held clones retain it as needed.
    let trace = unsafe { Rc::from_raw(payload.cast::<Trace>()) };
    if let Err(panic) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| drop(trace))) {
        // A malicious panic payload may panic during Drop. Nothing unwinds through C.
        std::mem::forget(panic);
    }
}
