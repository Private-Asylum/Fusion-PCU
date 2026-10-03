//! Same-width official views with exact owned input retention and terminal proof.
#[rustfmt::skip]
use std::rc::Rc;
use crate::MlxError;
#[rustfmt::skip]
use super::{abi::CArray,owner::Owner,session::Session,checked::available};
#[cfg(feature = "view-census")]
#[path = "census/census.rs"]
pub mod census;
#[cfg(feature = "view-census")]
#[rustfmt::skip]
use census::{
    record,
    Call,
};
/// All callers retain the validated original owner while this cloned input is in flight.
pub(super) fn complete_view(
    session: &Session,
    input: Owner<CArray>,
    shape: [usize; 2],
    to_matrix: bool,
) -> Result<Owner<CArray>, MlxError> {
    session.ensure_ready()?;
    let dims = super::super::dimensions(shape)?;
    shape[0]
        .checked_mul(shape[1])
        .filter(|count| i32::try_from(*count).is_ok())
        .ok_or(MlxError::InvalidExtent)?;
    let api = &session.0.api;
    let mut output = Owner::empty(Rc::clone(api), api.array_free);
    // SAFETY: audited catch-all endpoint accepts the verified actual input holder, exact
    // checked dimensions and explicit retained GPU stream. It constructs lazy descriptors only.
    #[cfg(feature = "view-census")]
    record(Call::Construct);
    api.status(|| unsafe {
        (api.f32_view)(
            &raw mut output.raw,
            input.raw,
            session.0.stream.raw,
            dims[0],
            dims[1],
            i32::from(to_matrix),
        )
    })?;
    output.require_live()?;
    let pending = (input, output, session.clone());
    let terminal = (|| {
        // SAFETY: both actual owners plus stream/image remain retained until terminal proof.
        #[cfg(feature = "view-census")]
        record(Call::Eval);
        api.status(|| unsafe { (api.array_eval)(pending.1.raw) })?;
        #[cfg(feature = "view-census")]
        record(Call::Synchronize);
        api.status(|| unsafe { (api.synchronize)(session.0.stream.raw) })?;
        #[cfg(feature = "view-census")]
        record(Call::Wait);
        api.status(|| unsafe { (api.array_wait)(pending.1.raw) })?;
        #[cfg(feature = "view-census")]
        record(Call::Available);
        available(session, pending.1.raw)
    })();
    if let Err(error) = terminal {
        session.0.poisoned.set(true);
        std::mem::forget(pending);
        return Err(MlxError::CompletionUnknown(error.to_string()));
    }
    // SAFETY: both real immutable owners are terminal. This audited endpoint checks exact
    // shared native Data identity/offset/bytes without reading a host data pointer.
    #[cfg(feature = "view-census")]
    record(Call::Validate);
    api.status(|| unsafe { (api.f32_view_validate_shared)(pending.0.raw, pending.1.raw) })?;
    let (input, output, _session) = pending;
    #[cfg(feature = "view-census")]
    record(Call::Release);
    input.release()?;
    Ok(output)
}
