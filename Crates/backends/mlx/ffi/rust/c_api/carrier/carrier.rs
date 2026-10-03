//! Retained fixed integer-limb primitive and its exact logical/physical ownership contract.
#[cfg(feature = "carrier-census")]
#[path = "census/census.rs"]
pub(super) mod census;
#[rustfmt::skip]
use std::{
    cell::Cell,
    rc::Rc,
};
use fusion_pcu::PcuScalarType;
use crate::MlxError;
#[rustfmt::skip]
use super::{
    abi::CCarrier,
    owner::Owner,
    session::Session,
    encoded::{
        EncodedArray,
        carrier::Carrier,
    },
};

pub struct CarrierCopy {
    owner: Option<Owner<CCarrier>>,
    session: Session,
    scalar: PcuScalarType,
    input_count: usize,
    output_count: usize,
    may_write: Cell<bool>,
}
impl CarrierCopy {
    pub fn prepare(
        session: &Session,
        scalar: PcuScalarType,
        count: usize,
        broadcast: bool,
    ) -> Result<Self, MlxError> {
        Self::prepare_internal(session, scalar, count, broadcast, None)
    }
    pub fn prepare_with_input_extent(
        session: &Session,
        scalar: PcuScalarType,
        count: usize,
        broadcast: bool,
        input_count: usize,
    ) -> Result<Self, MlxError> {
        Self::prepare_internal(session, scalar, count, broadcast, Some(input_count))
    }
    fn prepare_internal(
        session: &Session,
        scalar: PcuScalarType,
        count: usize,
        broadcast: bool,
        input_extent: Option<usize>,
    ) -> Result<Self, MlxError> {
        session.ensure_ready()?;
        let minimum = if broadcast { 1 } else { count };
        let input_count = input_extent.unwrap_or(minimum);
        if input_count < minimum {
            return Err(MlxError::InvalidExtent);
        }
        let input = Carrier::assess(scalar, input_count)?;
        let output = Carrier::assess(scalar, count)?;
        let input_native = u32::try_from(input.count).map_err(|_| MlxError::InvalidExtent)?;
        let output_native = u32::try_from(output.count).map_err(|_| MlxError::InvalidExtent)?;
        let dtype = u32::try_from(output.dtype).map_err(|_| MlxError::InvalidExtent)?;
        let api = &session.0.api;
        let mut owner = Owner::empty(Rc::clone(api), api.carrier_free);
        // SAFETY: exact positive bounded physical counts, selected UInt dtype and actual retained
        // GPU stream. The catch-all C endpoint constructs one fixed lazy primitive, cold only.
        if input_extent.is_some() {
            let scalar_lanes = if broadcast {
                u32::try_from(Carrier::assess(scalar, 1)?.count)
                    .map_err(|_| MlxError::InvalidExtent)?
            } else {
                0
            };
            #[cfg(feature = "carrier-census")]
            census::record(census::Call::PrefixConstructor);
            // SAFETY: the full positive physical input shape is frozen separately from the
            // exact output/read span; broadcast repeats one checked complete scalar limb tile.
            api.status(|| unsafe {
                (api.carrier_prefix_new)(
                    &raw mut owner.raw,
                    session.0.stream.raw,
                    dtype,
                    input_native,
                    output_native,
                    scalar_lanes,
                    i32::from(broadcast),
                )
            })?;
        } else {
            #[cfg(feature = "carrier-census")]
            census::record(census::Call::ExactConstructor);
            // SAFETY: the legacy endpoint retains its exact input/output shape guards.
            api.status(|| unsafe {
                (api.carrier_new)(
                    &raw mut owner.raw,
                    session.0.stream.raw,
                    dtype,
                    input_native,
                    output_native,
                    i32::from(broadcast),
                )
            })?;
        }
        owner.require_live()?;
        Ok(Self {
            owner: Some(owner),
            session: session.clone(),
            scalar,
            input_count,
            output_count: count,
            may_write: Cell::new(false),
        })
    }
    pub fn reset_write_fact(&self) {
        self.may_write.set(false);
    }
    #[cfg(feature = "carrier-census")]
    pub fn record_prime_call() {
        census::record(census::Call::Prime);
    }
    pub const fn may_have_written(&self) -> bool {
        self.may_write.get()
    }
    pub fn execute(&self, input: &EncodedArray) -> Result<EncodedArray, MlxError> {
        self.reset_write_fact();
        self.session.ensure_ready()?;
        if input.scalar() != self.scalar {
            return Err(MlxError::UnsupportedScalar(input.scalar()));
        }
        if input.count() != self.input_count {
            return Err(MlxError::InvalidExtent);
        }
        if !input.same_session(&self.session) {
            return Err(MlxError::ForeignSession);
        }
        let owner = self
            .owner
            .as_ref()
            .ok_or_else(|| MlxError::Abi("missing carrier primitive".into()))?;
        owner.require_live()?;
        #[cfg(feature = "carrier-census")]
        census::record(census::Call::Apply);
        input.copy_with(owner.raw, self.output_count, &self.may_write)
    }
}
impl Drop for CarrierCopy {
    fn drop(&mut self) {
        if self.session.ensure_ready().is_err() {
            if let Some(owner) = self.owner.take() {
                std::mem::forget(owner);
            }
            // Pending GPU work can retain this actual primitive/image/stream; an unknown terminal
            // event quarantines all of them even when the previous immutable output is unchanged.
            std::mem::forget(self.session.clone());
        }
    }
}
