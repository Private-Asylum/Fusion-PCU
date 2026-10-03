//! Contained actual MLX primitive replay with terminal retention of every sibling.
#[rustfmt::skip]
use std::{
    ffi::CString,
    rc::Rc,
};
#[rustfmt::skip]
use super::{
    abi::{
        CArray,
        CTransport,
        Opaque,
    },
    checked::available,
    encoded::{
        EncodedArray,
        carrier::Carrier,
    },
    owner::Owner,
    session::Session,
};
#[rustfmt::skip]
use crate::{
    MlxError,
    MlxTransportPlan,
};

pub struct Transport {
    owner: Option<Owner<CTransport>>,
    session: Session,
    scalar: fusion_pcu::PcuScalarType,
    element_count: usize,
    input_count: usize,
    output_count: usize,
    input_extents: [usize; 4],
}
impl Transport {
    pub fn prepare(session: &Session, plan: &MlxTransportPlan) -> Result<Self, MlxError> {
        let extents = plan.input_element_counts();
        Self::prepare_internal(session, plan, extents)
    }
    pub fn prepare_with_input_extents(
        session: &Session,
        plan: &MlxTransportPlan,
        extents: &[usize],
    ) -> Result<Self, MlxError> {
        let full = plan.assess_input_extents(extents)?;
        Self::prepare_internal(session, plan, full)
    }
    fn prepare_internal(
        session: &Session,
        plan: &MlxTransportPlan,
        input_extents: [usize; 4],
    ) -> Result<Self, MlxError> {
        let count = usize::try_from(plan.element_count()).map_err(|_| MlxError::InvalidExtent)?;
        Self::construct(
            session,
            plan.scalar_type(),
            count,
            plan.inputs().len(),
            plan.outputs().len(),
            plan.native_source(),
            input_extents,
        )
    }
    #[cfg(feature = "benchmark-control")]
    pub fn prepare_native_control(
        session: &Session,
        scalar: fusion_pcu::PcuScalarType,
        element_count: usize,
        workload: crate::MlxNativeTransportWorkload,
        extents: &[usize],
    ) -> Result<Self, MlxError> {
        let arity = workload.input_count();
        if extents.len() != arity {
            return Err(MlxError::InvalidExtent);
        }
        let output = Carrier::assess(scalar, element_count)?;
        let mut full = [0; 4];
        for (slot, &count) in extents.iter().enumerate() {
            let minimum = if slot == 1 { 1 } else { element_count };
            if count < minimum {
                return Err(MlxError::InvalidExtent);
            }
            Carrier::assess(scalar, count)?;
            full[slot] = count;
        }
        // Independent handwritten native workload. No PCU IR, descriptor or Plan participates.
        let body = match workload {
            crate::MlxNativeTransportWorkload::SavedInput => {
                "output0[i]=input1[limb]; output1[i]=input0[i];"
            }
            crate::MlxNativeTransportWorkload::PriorStage => {
                "output0[i]=input1[limb]; output1[i]=input2[i];"
            }
            crate::MlxNativeTransportWorkload::SwapBanks => {
                "output0[i]=input3[i]; output1[i]=input2[i];"
            }
        };
        let limbs = output.logical_width / output.width;
        let source = format!(
            "uint i=thread_position_in_grid.x; if(i>={}u) return; uint limb=i%{limbs}u; {body}\n",
            output.count
        );
        Self::construct(session, scalar, element_count, arity, 2, &source, full)
    }
    fn construct(
        session: &Session,
        scalar: fusion_pcu::PcuScalarType,
        element_count: usize,
        input_count: usize,
        output_count: usize,
        source: &str,
        input_extents: [usize; 4],
    ) -> Result<Self, MlxError> {
        session.ensure_ready()?;
        let output = Carrier::assess(scalar, element_count)?;
        let mut counts = [0_u32; 4];
        for (slot, &count) in input_extents.iter().take(input_count).enumerate() {
            counts[slot] = u32::try_from(Carrier::assess(scalar, count)?.count)
                .map_err(|_| MlxError::InvalidExtent)?;
        }
        let source = CString::new(source).map_err(|_| MlxError::InvalidExtent)?;
        let api = &session.0.api;
        let mut owner = Owner::empty(Rc::clone(api), api.transport_free);
        let inputs = u32::try_from(input_count).map_err(|_| MlxError::InvalidExtent)?;
        let outputs = u32::try_from(output_count).map_err(|_| MlxError::InvalidExtent)?;
        let lanes = u32::try_from(output.count).map_err(|_| MlxError::InvalidExtent)?;
        let dtype = u32::try_from(output.dtype).map_err(|_| MlxError::InvalidExtent)?;
        // SAFETY: bounded exact physical counts and generated integer-only source live for
        // the entire synchronous exception-containing constructor; native stores no Rust pointer.
        api.status(|| unsafe {
            (api.transport_new)(
                &raw mut owner.raw,
                session.0.stream.raw,
                dtype,
                counts.as_ptr(),
                inputs,
                lanes,
                outputs,
                source.as_ptr(),
            )
        })?;
        owner.require_live()?;
        Ok(Self {
            owner: Some(owner),
            session: session.clone(),
            scalar,
            element_count,
            input_count,
            output_count,
            input_extents,
        })
    }
    pub fn input_element_counts(&self) -> &[usize] {
        &self.input_extents[..self.input_count]
    }
    pub fn execute(&self, inputs: &[&EncodedArray]) -> Result<[Option<EncodedArray>; 2], MlxError> {
        self.session.ensure_ready()?;
        if inputs.len() != self.input_count {
            return Err(MlxError::InvalidExtent);
        }
        for (slot, input) in inputs.iter().enumerate() {
            if input.scalar() != self.scalar {
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
            .ok_or_else(|| MlxError::Abi("missing transport primitive".into()))?;
        owner.require_live()?;
        self.complete(owner.raw, inputs)
    }
    fn complete(
        &self,
        primitive: CTransport,
        inputs: &[&EncodedArray],
    ) -> Result<[Option<EncodedArray>; 2], MlxError> {
        let api = &self.session.0.api;
        let mut input_holders: [Option<Owner<CArray>>; 4] = std::array::from_fn(|_| None);
        let mut native_inputs = [CArray::empty(); 4];
        for (slot, input) in inputs.iter().enumerate() {
            let holder = input.clone_holder()?;
            native_inputs[slot] = holder.raw;
            input_holders[slot] = Some(holder);
        }
        let outputs = self.output_count;
        let mut native_outputs = [CArray::empty(); 2];
        // SAFETY: actual holders and exact arity remain live. The retained primitive creates
        // only lazy result descriptors; native contains exceptions and stores no slice pointer.
        let apply = api.status(|| unsafe {
            (api.transport_apply)(
                native_outputs.as_mut_ptr(),
                primitive,
                native_inputs.as_ptr(),
                u32::try_from(inputs.len()).unwrap(),
            )
        });
        let mut output_holders: [Option<Owner<CArray>>; 2] = std::array::from_fn(|_| None);
        for slot in 0..outputs {
            let mut holder = Owner::empty(Rc::clone(api), api.array_free);
            holder.raw = native_outputs[slot];
            output_holders[slot] = Some(holder);
        }
        apply?;
        let pending = (input_holders, output_holders, self.session.clone());
        let terminal = (|| {
            for holder in pending.1.iter().flatten() {
                holder.require_live()?;
                // SAFETY: all real input/output holders plus actual primitive/stream/image
                // are retained across every potentially asynchronous operation.
                api.status(|| unsafe { (api.array_eval)(holder.raw) })?;
            }
            // SAFETY: synchronization targets only the prepared explicit MLX stream.
            api.status(|| unsafe { (api.synchronize)(self.session.0.stream.raw) })?;
            for holder in pending.1.iter().flatten() {
                // SAFETY: checked real result holders remain retained through wait/access.
                api.status(|| unsafe { (api.array_wait)(holder.raw) })?;
                available(&self.session, holder.raw)?;
            }
            Ok::<(), MlxError>(())
        })();
        if let Err(error) = terminal {
            self.session.0.poisoned.set(true);
            std::mem::forget(pending);
            return Err(MlxError::CompletionUnknown(error.to_string()));
        }
        let (input_holders, output_holders, _session) = pending;
        for input in input_holders.into_iter().flatten() {
            input.release()?;
        }
        let mut completed = std::array::from_fn(|_| None);
        for (slot, holder) in output_holders.into_iter().enumerate() {
            if let Some(holder) = holder {
                completed[slot] = Some(EncodedArray::from_owner(
                    &self.session,
                    self.scalar,
                    self.element_count,
                    holder,
                )?);
            }
        }
        Ok(completed)
    }
}
impl Drop for Transport {
    fn drop(&mut self) {
        if self.session.ensure_ready().is_err() {
            if let Some(owner) = self.owner.take() {
                std::mem::forget(owner);
            }
            std::mem::forget(self.session.clone());
        }
    }
}
