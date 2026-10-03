//! Contained three-sibling checked composition; all physical shapes freeze cold.
#[rustfmt::skip]
use std::{
    ffi::CString,
    rc::Rc,
};
#[rustfmt::skip]
use super::{
    abi::CComposed,
    encoded::{
        EncodedArray,
        carrier::Carrier,
    },
    owner::Owner,
    session::Session,
};
#[rustfmt::skip]
use crate::{
    MlxCheckedMapPlan,
    MlxError,
};
use fusion_pcu::PcuBindingRef;
#[path = "completion/completion.rs"]
mod completion;

pub struct Composed {
    owner: Option<Owner<CComposed>>,
    session: Session,
    plan: MlxCheckedMapPlan,
    inputs: Vec<PcuBindingRef>,
    outputs: Vec<PcuBindingRef>,
    input_extents: [usize; 4],
}
impl Composed {
    pub fn prepare(session: &Session, plan: MlxCheckedMapPlan) -> Result<Self, MlxError> {
        session.ensure_ready()?;
        let (header, body) = plan.native_source()?;
        Self::prepare_sources(session, plan, header, body)
    }
    #[cfg(feature = "benchmark-control")]
    pub fn prepare_control(
        session: &Session,
        plan: MlxCheckedMapPlan,
        header: String,
        body: String,
    ) -> Result<Self, MlxError> {
        Self::prepare_sources(session, plan, header, body)
    }
    fn prepare_sources(
        session: &Session,
        plan: MlxCheckedMapPlan,
        header: String,
        body: String,
    ) -> Result<Self, MlxError> {
        session.ensure_ready()?;
        let scalar = plan.value_type().scalar_type();
        let count = usize::try_from(plan.logical_extent()).map_err(|_| MlxError::InvalidExtent)?;
        let carrier = Carrier::assess(scalar, count)?;
        let mut inputs = Vec::new();
        let mut outputs = Vec::new();
        let mut input_extents = [0; 4];
        let mut counts = [0; 4];
        for role in plan.resources() {
            if role.minimum_initial_read_elements != 0 {
                let extent = usize::try_from(role.minimum_initial_read_elements)
                    .map_err(|_| MlxError::InvalidExtent)?;
                let input = Carrier::assess(scalar, extent)?;
                counts[inputs.len()] =
                    u32::try_from(input.count).map_err(|_| MlxError::InvalidExtent)?;
                input_extents[inputs.len()] = extent;
                inputs.push(role.binding);
            }
            if role.minimum_write_elements != 0 {
                outputs.push(role.binding);
            }
        }
        if inputs.is_empty() || outputs.is_empty() {
            return Err(MlxError::InvalidExtent);
        }
        let header = CString::new(header).map_err(|_| MlxError::InvalidExtent)?;
        let body = CString::new(body).map_err(|_| MlxError::InvalidExtent)?;
        let input_count = u32::try_from(inputs.len()).map_err(|_| MlxError::InvalidExtent)?;
        let output_count = u32::try_from(outputs.len()).map_err(|_| MlxError::InvalidExtent)?;
        let lanes = u32::try_from(carrier.count).map_err(|_| MlxError::InvalidExtent)?;
        let records =
            u32::try_from(plan.status_byte_len() / 4).map_err(|_| MlxError::InvalidExtent)?;
        let dtype = u32::try_from(carrier.dtype).map_err(|_| MlxError::InvalidExtent)?;
        let api = &session.0.api;
        let mut owner = Owner::empty(Rc::clone(api), api.composed_free);
        // SAFETY: bounded source strings/exact counts live through the contained synchronous
        // constructor. Native retains immutable descriptors; no Rust pointer escapes.
        api.status(|| unsafe {
            (api.composed_new)(
                &raw mut owner.raw,
                session.0.stream.raw,
                dtype,
                counts.as_ptr(),
                input_count,
                plan.logical_extent(),
                lanes,
                output_count,
                records,
                header.as_ptr(),
                body.as_ptr(),
            )
        })?;
        owner.require_live()?;
        Ok(Self {
            owner: Some(owner),
            session: session.clone(),
            plan,
            inputs,
            outputs,
            input_extents,
        })
    }
    pub const fn plan(&self) -> &MlxCheckedMapPlan {
        &self.plan
    }
    pub const fn input_bindings(&self) -> &[PcuBindingRef] {
        self.inputs.as_slice()
    }
    pub const fn output_bindings(&self) -> &[PcuBindingRef] {
        self.outputs.as_slice()
    }
    pub fn input_element_counts(&self) -> &[usize] {
        &self.input_extents[..self.inputs.len()]
    }
    pub fn execute(&self, inputs: &[&EncodedArray]) -> Result<completion::Completed, MlxError> {
        self.session.ensure_ready()?;
        if inputs.len() != self.inputs.len() {
            return Err(MlxError::InvalidExtent);
        }
        for (slot, input) in inputs.iter().enumerate() {
            if input.scalar() != self.plan.value_type().scalar_type() {
                return Err(MlxError::UnsupportedScalar(input.scalar()));
            }
            if input.count() != self.input_extents[slot] {
                return Err(MlxError::InvalidExtent);
            }
            if !input.same_session(&self.session) {
                return Err(MlxError::ForeignSession);
            }
        }
        let owner = self
            .owner
            .as_ref()
            .ok_or_else(|| MlxError::Abi("missing composed primitive".into()))?;
        owner.require_live()?;
        self.complete(owner.raw, inputs)
    }
}
impl Drop for Composed {
    fn drop(&mut self) {
        if self.session.ensure_ready().is_err() {
            if let Some(owner) = self.owner.take() {
                std::mem::forget(owner);
            }
            std::mem::forget(self.session.clone());
        }
    }
}
