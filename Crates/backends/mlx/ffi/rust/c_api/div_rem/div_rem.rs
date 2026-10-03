//! Joint exact publication: joint exact quotient/remainder/status through an MLX-owned primitive.
#[cfg(feature = "division-census")]
#[path = "census/census.rs"]
pub(super) mod census;
#[cfg(test)]
#[path = "tests/tests.rs"]
mod tests;
#[rustfmt::skip]
use std::{
    cell::Cell,
    ptr::NonNull,
    rc::Rc,
};
#[rustfmt::skip]
use fusion_pcu::{
    PcuScalarType,
    PcuExecutionFault,
    PcuExecutionFaultKind,
};
use crate::MlxError;
#[rustfmt::skip]
use super::{
    abi::{
        CCheckedDivRem,
        CArray,
    },
    owner::Owner,
    session::Session,
    encoded::{
        EncodedArray,
        carrier::Carrier,
    },
    checked::{
        validate,
        available,
    },
};
pub struct CheckedDivRem {
    fault_law: fusion_pcu::PcuCheckedScalarFaultLaw,
    owner: Option<Owner<CCheckedDivRem>>,
    session: Session,
    scalar: PcuScalarType,
    carrier: Carrier,
    count: usize,
    inputs: [usize; 2],
    may_have_written: Cell<bool>,
    scratch: Vec<u8>,
}
impl CheckedDivRem {
    pub fn prepare(
        session: &Session,
        scalar: PcuScalarType,
        count: usize,
        inputs: [usize; 2],
        broadcast: [bool; 2],
    ) -> Result<Self, MlxError> {
        Self::prepare_integer(session, scalar, count, inputs, broadcast, false)
    }
    /// Cold full dense input capacities remain distinct from bounded shader read indexes.
    pub fn prepare_with_input_extents(
        session: &Session,
        scalar: PcuScalarType,
        count: usize,
        inputs: [usize; 2],
        broadcast: [bool; 2],
    ) -> Result<Self, MlxError> {
        Self::prepare_integer(session, scalar, count, inputs, broadcast, true)
    }
    #[cfg(test)]
    fn prepare_wide_proof(
        session: &Session,
        scalar: PcuScalarType,
        count: usize,
        inputs: [usize; 2],
        broadcast: [bool; 2],
    ) -> Result<Self, MlxError> {
        Self::prepare_integer(session, scalar, count, inputs, broadcast, false)
    }
    fn prepare_integer(
        session: &Session,
        scalar: PcuScalarType,
        count: usize,
        inputs: [usize; 2],
        broadcast: [bool; 2],
        full_inputs: bool,
    ) -> Result<Self, MlxError> {
        session.ensure_ready()?;
        let signed = match scalar {
            PcuScalarType::I8
            | PcuScalarType::I16
            | PcuScalarType::I32
            | PcuScalarType::I64
            | PcuScalarType::I128
            | PcuScalarType::I256
            | PcuScalarType::I512 => true,
            PcuScalarType::U8
            | PcuScalarType::U16
            | PcuScalarType::U32
            | PcuScalarType::U64
            | PcuScalarType::U128
            | PcuScalarType::U256
            | PcuScalarType::U512 => false,
            _ => return Err(MlxError::UnsupportedScalar(scalar)),
        };
        let carrier = Carrier::assess(scalar, count)?;
        let width = carrier.logical_width;
        let count_native = u32::try_from(count).map_err(|_| MlxError::InvalidExtent)?;
        let bytes = count
            .checked_mul(width)
            .and_then(|v| v.checked_mul(2))
            .filter(|&v| isize::try_from(v).is_ok())
            .ok_or(MlxError::InvalidExtent)?;
        if count == 0
            || inputs.iter().zip(broadcast).any(|(&size, scalar)| {
                if full_inputs {
                    size < if scalar { 1 } else { count }
                } else {
                    size != count && !(scalar && size == 1)
                }
            })
        {
            return Err(MlxError::InvalidExtent);
        }
        for input in inputs {
            Carrier::assess(scalar, input)?;
        }
        let api = &session.0.api;
        let constructor = if full_inputs {
            api.div_rem_prefix_new
        } else {
            api.div_rem_new
        };
        let mut owner = Owner::empty(Rc::clone(api), api.div_rem_free);
        let counts = inputs.map(|n| u32::try_from(n).map_err(|_| MlxError::InvalidExtent));
        let [left_count, right_count] = counts;
        let (left_count, right_count) = (left_count?, right_count?);
        #[cfg(feature = "division-census")]
        census::record(if full_inputs {
            census::Call::PrefixConstructor
        } else {
            census::Call::ExactConstructor
        });
        // SAFETY: admitted widths, checked physical extents, explicit retained stream and empty owner.
        api.status(|| unsafe {
            (constructor)(
                &raw mut owner.raw,
                session.0.stream.raw,
                u32::from(scalar.bit_width()),
                u32::from(signed),
                count_native,
                left_count,
                right_count,
                u32::from(broadcast[0]) | (u32::from(broadcast[1]) << 1),
            )
        })?;
        owner.require_live()?;
        let prepared = Self {
            fault_law: fusion_pcu::PcuCheckedScalarFaultLaw::integer_div_rem(scalar)
                .ok_or(MlxError::UnsupportedScalar(scalar))?,
            owner: Some(owner),
            session: session.clone(),
            scalar,
            carrier,
            count,
            inputs,
            may_have_written: Cell::new(false),
            scratch: vec![0; bytes],
        };
        let left = EncodedArray::upload(session, scalar, inputs[0], &vec![0; inputs[0] * width])?;
        let mut ones = vec![0; inputs[1] * width];
        for lane in ones.chunks_exact_mut(width) {
            lane[0] = 1;
        }
        let right = EncodedArray::upload(session, scalar, inputs[1], &ones)?;
        #[cfg(feature = "division-census")]
        census::record(census::Call::Prime);
        let [quotient, remainder] = prepared.execute_encoded([&left, &right])?;
        quotient.release()?;
        remainder.release()?;
        left.release()?;
        right.release()?;
        prepared.may_have_written.set(false);
        Ok(prepared)
    }
    pub fn execute_encoded(
        &self,
        inputs: [&EncodedArray; 2],
    ) -> Result<[EncodedArray; 2], MlxError> {
        let (outputs, records) = self.execute_recorded(inputs)?;
        super::fault::validate(
            self.status_records(&records)?,
            self.count,
            self.fault_law,
            super::fault::Encoding::DivRem,
        )?;
        let fault = select_fault(self.status_records(&records)?)?;
        records.release()?;
        if let Some(fault) = fault {
            let [quotient, remainder] = outputs;
            quotient.release()?;
            remainder.release()?;
            return Err(MlxError::Arithmetic(fault));
        }
        Ok(outputs)
    }
    pub fn execute_host(
        &mut self,
        inputs: [&[u8]; 2],
        outputs: [&mut [u8]; 2],
    ) -> Result<(), MlxError> {
        self.may_have_written.set(false);
        let width = self.carrier.logical_width;
        let bytes = self.count * width;
        if outputs.iter().any(|o| o.len() < bytes)
            || inputs
                .iter()
                .zip(self.inputs)
                .any(|(i, n)| i.len() < n * width)
        {
            return Err(MlxError::InvalidExtent);
        }
        let left = EncodedArray::upload(
            &self.session,
            self.scalar,
            self.inputs[0],
            &inputs[0][..self.inputs[0] * width],
        )?;
        let right = EncodedArray::upload(
            &self.session,
            self.scalar,
            self.inputs[1],
            &inputs[1][..self.inputs[1] * width],
        )?;
        let [quotient, remainder] = self.execute_encoded([&left, &right])?;
        let (q, r) = self.scratch.split_at_mut(bytes);
        quotient.read(q)?;
        remainder.read(r)?;
        quotient.release()?;
        remainder.release()?;
        left.release()?;
        right.release()?;
        // Both private reads and every fallible temporary cleanup completed before either caller destination changes.
        let [out_q, out_r] = outputs;
        out_q[..bytes].copy_from_slice(q);
        out_r[..bytes].copy_from_slice(r);
        Ok(())
    }
    pub fn reset_write_fact(&self) {
        self.may_have_written.set(false);
    }
    /// Stages each actual unique read once; no high-level escaped owner allocation is needed.
    pub fn execute_host_roles(
        &mut self,
        inputs: &[&[u8]],
        counts: &[usize],
        operands: [usize; 2],
        outputs: [&mut [u8]; 2],
    ) -> Result<(), MlxError> {
        self.may_have_written.set(false);
        let width = self.carrier.logical_width;
        let bytes = self.count * width;
        if !(1..=2).contains(&inputs.len())
            || counts.len() != inputs.len()
            || operands.iter().any(|&slot| slot >= inputs.len())
            || operands.map(|slot| counts[slot]) != self.inputs
            || outputs.iter().any(|output| output.len() < bytes)
            || inputs.iter().zip(counts).any(|(input, &count)| {
                !matches!(count, 1) && count != self.count
                    || count
                        .checked_mul(width)
                        .is_none_or(|bytes| input.len() < bytes)
            })
        {
            return Err(MlxError::InvalidExtent);
        }
        let mut staged = [None, None];
        for (slot, (&input, &count)) in inputs.iter().zip(counts).enumerate() {
            staged[slot] = Some(EncodedArray::upload(
                &self.session,
                self.scalar,
                count,
                &input[..count * width],
            )?);
        }
        let owner = |slot: usize| staged[slot].as_ref().ok_or(MlxError::InvalidExtent);
        let result = self.execute_encoded([owner(operands[0])?, owner(operands[1])?]);
        if self.completion_uncertain() {
            return result.map(|_| ());
        }
        let pair = match result {
            Ok(pair) => pair,
            Err(error) => {
                for input in staged.into_iter().flatten() {
                    input.release()?;
                }
                return Err(error);
            }
        };
        let [quotient, remainder] = pair;
        let (q, r) = self.scratch.split_at_mut(bytes);
        quotient.read(q)?;
        remainder.read(r)?;
        quotient.release()?;
        remainder.release()?;
        for input in staged.into_iter().flatten() {
            input.release()?;
        }
        // Both private reads and every checked temporary release precede joint caller publication.
        let [out_q, out_r] = outputs;
        out_q[..bytes].copy_from_slice(q);
        out_r[..bytes].copy_from_slice(r);
        Ok(())
    }
    pub const fn may_have_written(&self) -> bool {
        self.may_have_written.get()
    }
    pub fn completion_uncertain(&self) -> bool {
        self.session.ensure_ready().is_err()
    }
    fn status_records<'a>(&self, records: &'a Owner<CArray>) -> Result<&'a [u32], MlxError> {
        let api = &self.session.0.api;
        // SAFETY: terminal exact UInt32 status owner remains live throughout this borrowed slice.
        let pointer = api.guarded(|| unsafe { (api.array_data_u32)(records.raw) })?;
        let pointer = NonNull::new(pointer.cast_mut())
            .ok_or_else(|| MlxError::Abi("nil div/rem status".into()))?;
        // SAFETY: recorded shape/bytes verified before evaluation and retained after terminal access.
        Ok(unsafe { std::slice::from_raw_parts(pointer.as_ptr(), self.count) })
    }
    fn execute_recorded(
        &self,
        inputs: [&EncodedArray; 2],
    ) -> Result<([EncodedArray; 2], Owner<CArray>), MlxError> {
        self.may_have_written.set(false);
        self.session.ensure_ready()?;
        for (input, count) in inputs.iter().zip(self.inputs) {
            if input.scalar() != self.scalar {
                return Err(MlxError::UnsupportedScalar(input.scalar()));
            }
            if input.count() != count {
                return Err(MlxError::InvalidExtent);
            }
            if !input.same_session(&self.session) {
                return Err(MlxError::ForeignSession);
            }
        }
        let left = inputs[0].clone_holder()?;
        let right = inputs[1].clone_holder()?;
        let api = &self.session.0.api;
        let mut quotient = Owner::empty(Rc::clone(api), api.array_free);
        let mut remainder = Owner::empty(Rc::clone(api), api.array_free);
        let mut records = Owner::empty(Rc::clone(api), api.array_free);
        let primitive = self
            .owner
            .as_ref()
            .ok_or_else(|| MlxError::Abi("missing div/rem primitive".into()))?
            .raw;
        #[cfg(feature = "division-census")]
        census::record(census::Call::Apply);
        // SAFETY: exact real inputs and three separate empty sibling owners retained through call.
        api.status(|| unsafe {
            (api.div_rem_apply)(
                &raw mut quotient.raw,
                &raw mut remainder.raw,
                &raw mut records.raw,
                primitive,
                left.raw,
                right.raw,
            )
        })?;
        for output in [&quotient, &remainder] {
            validate(
                &self.session,
                output.raw,
                self.carrier.dtype,
                self.carrier.count,
                self.carrier.width,
            )?;
        }
        validate(&self.session, records.raw, 3, self.count, 4)?;
        let pending = (
            left,
            right,
            quotient,
            remainder,
            records,
            self.session.clone(),
        );
        let terminal = (|| {
            // SAFETY: inputs, all siblings, exact stream and image live before first possibly-writing eval.
            self.may_have_written.set(true);
            for owner in [&pending.2, &pending.3, &pending.4] {
                api.status(|| unsafe { (api.array_eval)(owner.raw) })?;
            }
            api.status(|| unsafe { (api.synchronize)(self.session.0.stream.raw) })?;
            for owner in [&pending.2, &pending.3, &pending.4] {
                api.status(|| unsafe { (api.array_wait)(owner.raw) })?;
                available(&self.session, owner.raw)?;
            }
            Ok::<_, MlxError>(())
        })();
        if let Err(error) = terminal {
            self.session.0.poisoned.set(true);
            std::mem::forget(pending);
            return Err(MlxError::CompletionUnknown(error.to_string()));
        }
        let (left, right, q, r, records, _session) = pending;
        left.release()?;
        right.release()?;
        Ok((
            [
                EncodedArray::from_owner(&self.session, self.scalar, self.count, q)?,
                EncodedArray::from_owner(&self.session, self.scalar, self.count, r)?,
            ],
            records,
        ))
    }
}
impl Drop for CheckedDivRem {
    fn drop(&mut self) {
        if self.session.ensure_ready().is_err() {
            if let Some(owner) = self.owner.take() {
                std::mem::forget(owner);
            }
            std::mem::forget(self.session.clone());
        }
    }
}
fn select_fault(records: &[u32]) -> Result<Option<PcuExecutionFault>, MlxError> {
    if records.iter().any(|r| !matches!(r, 0 | 1 | 4)) {
        return Err(MlxError::Abi("invalid div/rem status".into()));
    }
    Ok(records
        .iter()
        .position(|&r| r != 0)
        .map(|index| PcuExecutionFault {
            invocation_id: index as u64,
            kind: if records[index] == 4 {
                PcuExecutionFaultKind::DivideByZero
            } else {
                PcuExecutionFaultKind::SignedDivisionOverflow
            },
            recovered: false,
        }))
}
