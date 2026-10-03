//! Every actual input and all heterogeneous siblings remain retained through terminal proof.
#[rustfmt::skip]
use std::{
    ptr::NonNull,
    rc::Rc,
};
#[rustfmt::skip]
use super::super::{
    abi::{
        CArray,
        CComposed,
        Opaque,
    },
    checked::{
        available,
        validate,
    },
    encoded::{
        EncodedArray,
        carrier::Carrier,
    },
    owner::Owner,
};
use super::Composed;
use crate::MlxError;
pub type Completed = (
    [Option<EncodedArray>; 2],
    Option<fusion_pcu::PcuExecutionFault>,
);
impl Composed {
    pub(super) fn complete(
        &self,
        primitive: CComposed,
        inputs: &[&EncodedArray],
    ) -> Result<Completed, MlxError> {
        let api = &self.session.0.api;
        let mut holders: [Option<Owner<CArray>>; 4] = std::array::from_fn(|_| None);
        let mut native_inputs = [CArray::empty(); 4];
        for (slot, input) in inputs.iter().enumerate() {
            let holder = input.clone_holder()?;
            native_inputs[slot] = holder.raw;
            holders[slot] = Some(holder);
        }
        let arity = u32::try_from(inputs.len()).map_err(|_| MlxError::InvalidExtent)?;
        let mut raw_outputs = [CArray::empty(); 3];
        // SAFETY: all cloned actual holders survive the contained lazy replay. Exactly
        // one/two payload descriptors and a separate UInt32 status descriptor are returned.
        let apply = api.status(|| unsafe {
            (api.composed_apply)(
                raw_outputs.as_mut_ptr(),
                primitive,
                native_inputs.as_ptr(),
                arity,
            )
        });
        let count = self.outputs.len();
        let mut outputs: [Option<Owner<CArray>>; 3] = std::array::from_fn(|_| None);
        for slot in 0..=count {
            let mut owner = Owner::empty(Rc::clone(api), api.array_free);
            owner.raw = raw_outputs[slot];
            outputs[slot] = Some(owner);
        }
        apply?;
        let pending = (holders, outputs, self.session.clone());
        self.validate_siblings(&pending.1)?;
        let terminal = (|| {
            for holder in pending.1.iter().flatten() {
                // SAFETY: real input/payload/status/stream/image owners precede first eval.
                api.status(|| unsafe { (api.array_eval)(holder.raw) })?;
            }
            // SAFETY: only the exact prepared stream is synchronized; every sibling is retained.
            api.status(|| unsafe { (api.synchronize)(self.session.0.stream.raw) })?;
            for holder in pending.1.iter().flatten() {
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
        let records = pending.1[count]
            .as_ref()
            .ok_or_else(|| MlxError::Abi("missing composed status".into()))?;
        // SAFETY: terminal, validated dense exact UInt32 extent, retained through the scan.
        let data = api.guarded(|| unsafe { (api.array_data_u32)(records.raw) })?;
        let data = NonNull::new(data.cast_mut())
            .ok_or_else(|| MlxError::Abi("nil composed status backing".into()))?;
        let words =
            unsafe { std::slice::from_raw_parts(data.as_ptr(), self.plan.status_byte_len() / 4) };
        let fault = self.plan.validate_status_words(words)?;
        let (inputs, mut outputs, _session) = pending;
        for input in inputs.into_iter().flatten() {
            input.release()?;
        }
        outputs[count]
            .take()
            .ok_or_else(|| MlxError::Abi("missing composed status".into()))?
            .release()?;
        if let Some(fault) = fault
            && !fault.recovered
        {
            for output in outputs.into_iter().flatten() {
                output.release()?;
            }
            return Err(MlxError::Arithmetic(fault));
        }
        let scalar = self.plan.value_type().scalar_type();
        let logical =
            usize::try_from(self.plan.logical_extent()).map_err(|_| MlxError::InvalidExtent)?;
        let mut completed = std::array::from_fn(|_| None);
        for (slot, owner) in outputs.into_iter().enumerate().take(count) {
            let owner = owner.ok_or_else(|| MlxError::Abi("missing composed payload".into()))?;
            completed[slot] = Some(EncodedArray::from_owner(
                &self.session,
                scalar,
                logical,
                owner,
            )?);
        }
        Ok((completed, fault))
    }
    fn validate_siblings(&self, outputs: &[Option<Owner<CArray>>; 3]) -> Result<(), MlxError> {
        let scalar = self.plan.value_type().scalar_type();
        let count =
            usize::try_from(self.plan.logical_extent()).map_err(|_| MlxError::InvalidExtent)?;
        let carrier = Carrier::assess(scalar, count)?;
        for (slot, owner) in outputs.iter().enumerate().take(self.outputs.len() + 1) {
            let owner = owner
                .as_ref()
                .ok_or_else(|| MlxError::Abi("missing composed sibling".into()))?;
            owner.require_live()?;
            if slot == self.outputs.len() {
                validate(
                    &self.session,
                    owner.raw,
                    3,
                    self.plan.status_byte_len() / 4,
                    4,
                )?;
            } else {
                validate(
                    &self.session,
                    owner.raw,
                    carrier.dtype,
                    carrier.count,
                    carrier.width,
                )?;
            }
        }
        Ok(())
    }
}
