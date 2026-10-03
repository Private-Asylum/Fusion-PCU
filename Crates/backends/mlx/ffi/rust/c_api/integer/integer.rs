//! MLX-owned exact integer primitive with retained payload/status siblings.
#[cfg(feature = "integer-census")]
#[path = "census/census.rs"]
pub mod census;
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
    PcuRangePolicy as Range,
    PcuExecutionFault,
    PcuExecutionFaultKind,
};
use crate::MlxError;
#[rustfmt::skip]
use super::{
    abi::CCheckedInteger,
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

pub struct CheckedInteger {
    fault_law: fusion_pcu::PcuCheckedScalarFaultLaw,
    owner: Option<Owner<CCheckedInteger>>,
    session: Session,
    scalar: PcuScalarType,
    dtype: i32,
    count: usize,
    carrier_count: usize,
    carrier_width: usize,
    inputs: [usize; 2],
    may_have_written: Cell<bool>,
}
impl CheckedInteger {
    #[allow(clippy::too_many_arguments)] // Independent cold logical/operation/policy/operand roles.
    pub fn prepare(
        session: &Session,
        scalar: PcuScalarType,
        operation: u32,
        range: Range,
        count: usize,
        inputs: [usize; 2],
        broadcast: [bool; 2],
    ) -> Result<Self, MlxError> {
        Self::prepare_internal(
            session, scalar, operation, range, count, inputs, broadcast, None,
        )
    }
    #[allow(clippy::too_many_arguments)] // Actual logical minima and full physical operand capacities are independent.
    pub fn prepare_with_input_extents(
        session: &Session,
        scalar: PcuScalarType,
        operation: u32,
        range: Range,
        count: usize,
        inputs: [usize; 2],
        broadcast: [bool; 2],
        full: [usize; 2],
    ) -> Result<Self, MlxError> {
        Self::prepare_internal(
            session,
            scalar,
            operation,
            range,
            count,
            inputs,
            broadcast,
            Some(full),
        )
    }
    #[allow(clippy::too_many_arguments, clippy::too_many_lines)] // Cold exact/full-shape preparation retains identical integer law and terminal priming.
    fn prepare_internal(
        session: &Session,
        scalar: PcuScalarType,
        operation: u32,
        range: Range,
        count: usize,
        inputs: [usize; 2],
        broadcast: [bool; 2],
        full: Option<[usize; 2]>,
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
        if operation > 2 {
            return Err(MlxError::InvalidExtent);
        }
        let carrier = Carrier::assess(scalar, count)?;
        let dtype = carrier.dtype;
        let width = carrier.logical_width;
        let count_native = i32::try_from(count).map_err(|_| MlxError::InvalidExtent)?;
        if count == 0
            || count
                .checked_mul(4)
                .is_none_or(|bytes| isize::try_from(bytes).is_err())
            || inputs
                .iter()
                .zip(broadcast)
                .any(|(&size, scalar)| size != count && !(scalar && size == 1))
        {
            return Err(MlxError::InvalidExtent);
        }
        let minimum = inputs;
        let inputs = full.unwrap_or(minimum);
        for (&extent, minimum) in inputs.iter().zip(minimum) {
            if extent < minimum {
                return Err(MlxError::InvalidExtent);
            }
            let _ = Carrier::assess(scalar, extent)?;
        }
        let api = &session.0.api;
        #[cfg(feature = "integer-census")]
        census::record(if full.is_some() {
            census::Call::PrefixConstructor
        } else {
            census::Call::ExactConstructor
        });
        let constructor = if full.is_some() {
            api.integer_prefix_new
        } else {
            api.integer_new
        };
        let left_count = u32::try_from(inputs[0]).map_err(|_| MlxError::InvalidExtent)?;
        let right_count = u32::try_from(inputs[1]).map_err(|_| MlxError::InvalidExtent)?;
        let mut owner = Owner::empty(Rc::clone(api), api.integer_free);
        // SAFETY: bounded admitted literals, exact explicit stream and empty unique owner;
        // the audited endpoint contains every exception and constructs lazy descriptors only.
        api.status(|| unsafe {
            constructor(
                &raw mut owner.raw,
                session.0.stream.raw,
                u32::from(scalar.bit_width()),
                u32::from(signed),
                operation,
                u32::from(range == Range::Clamp),
                count_native.cast_unsigned(),
                left_count,
                right_count,
                u32::from(broadcast[0]) | (u32::from(broadcast[1]) << 1),
            )
        })?;
        owner.require_live()?;
        let prepared = Self {
            fault_law: fusion_pcu::PcuCheckedScalarFaultLaw::integer_binary(
                scalar,
                match operation {
                    0 => fusion_pcu::PcuDispatchIntegerBinaryOp::Add,
                    1 => fusion_pcu::PcuDispatchIntegerBinaryOp::Sub,
                    _ => fusion_pcu::PcuDispatchIntegerBinaryOp::Mul,
                },
                range,
            )
            .ok_or(MlxError::UnsupportedScalar(scalar))?,
            owner: Some(owner),
            session: session.clone(),
            scalar,
            dtype,
            count,
            carrier_count: carrier.count,
            carrier_width: carrier.width,
            inputs,
            may_have_written: Cell::new(false),
        };
        // Zero operands realize the actual sibling primitive and freeze compilation cold.
        let left = EncodedArray::upload(session, scalar, inputs[0], &vec![0; inputs[0] * width])?;
        let right = EncodedArray::upload(session, scalar, inputs[1], &vec![0; inputs[1] * width])?;
        #[cfg(feature = "integer-census")]
        census::record(census::Call::Prime);
        drop(prepared.execute_encoded([&left, &right])?);
        prepared.reset_write_fact();
        Ok(prepared)
    }
    pub fn reset_write_fact(&self) {
        self.may_have_written.set(false);
    }
    pub const fn may_have_written(&self) -> bool {
        self.may_have_written.get()
    }
    pub fn completion_uncertain(&self) -> bool {
        self.session.ensure_ready().is_err()
    }
    pub fn execute_encoded(
        &self,
        inputs: [&EncodedArray; 2],
    ) -> Result<(EncodedArray, Option<PcuExecutionFault>), MlxError> {
        let (output, records) = self.execute_recorded(inputs)?;
        super::fault::validate(
            self.status_records(&records)?,
            self.count,
            self.fault_law,
            super::fault::Encoding::Scalar,
        )?;
        let fault = select_fault(self.status_records(&records)?)?;
        records.release()?;
        if let Some(fault) = fault
            && !fault.recovered
        {
            return Err(MlxError::Arithmetic(fault));
        }
        Ok((output, fault))
    }
    pub fn execute_host(&self, inputs: [&[u8]; 2], output: &mut [u8]) -> Result<(), MlxError> {
        let width = usize::from(self.scalar.bit_width()) / 8;
        self.reset_write_fact();
        if output.len() < self.count * width
            || inputs
                .iter()
                .zip(self.inputs)
                .any(|(bytes, count)| bytes.len() < count * width)
        {
            return Err(MlxError::InvalidExtent);
        }
        self.execute_host_mapped(
            &[
                &inputs[0][..self.inputs[0] * width],
                &inputs[1][..self.inputs[1] * width],
            ],
            [0, 1],
            &mut output[..self.count * width],
        )
    }
    /// Copies each unique source binding once, even when both mathematical operands reuse it.
    pub fn execute_host_mapped(
        &self,
        inputs: &[&[u8]],
        roles: [usize; 2],
        output: &mut [u8],
    ) -> Result<(), MlxError> {
        self.reset_write_fact();
        self.session.ensure_ready()?;
        let width = usize::from(self.scalar.bit_width()) / 8;
        if !(1..=2).contains(&inputs.len())
            || roles.iter().any(|&role| role >= inputs.len())
            || output.len() != self.count * width
        {
            return Err(MlxError::InvalidExtent);
        }
        let mut extents = [0; 2];
        for (slot, bytes) in inputs.iter().enumerate() {
            if bytes.is_empty() || !bytes.chunks_exact(width).remainder().is_empty() {
                return Err(MlxError::InvalidExtent);
            }
            extents[slot] = bytes.len() / width;
        }
        if roles
            .into_iter()
            .enumerate()
            .any(|(operand, role)| extents[role] != self.inputs[operand])
        {
            return Err(MlxError::InvalidExtent);
        }
        let left = EncodedArray::upload(&self.session, self.scalar, extents[0], inputs[0])?;
        let right = inputs
            .get(1)
            .map(|bytes| EncodedArray::upload(&self.session, self.scalar, extents[1], bytes))
            .transpose()?;
        let owners = [Some(&left), right.as_ref()];
        let operands = [
            owners[roles[0]].ok_or(MlxError::InvalidExtent)?,
            owners[roles[1]].ok_or(MlxError::InvalidExtent)?,
        ];
        let (completed, recovered) = self.execute_encoded(operands)?;
        completed.read(output)?;
        recovered.map_or(Ok(()), |fault| Err(MlxError::Arithmetic(fault)))
    }
    fn status_records<'a>(
        &self,
        records: &'a Owner<super::abi::CArray>,
    ) -> Result<&'a [u32], MlxError> {
        let api = &self.session.0.api;
        // SAFETY: only terminal verified dense UInt32 sibling holders reach this accessor.
        let pointer = api.guarded(|| unsafe { (api.array_data_u32)(records.raw) })?;
        let pointer = NonNull::new(pointer.cast_mut())
            .ok_or_else(|| MlxError::Abi("nil checked integer status backing".into()))?;
        // SAFETY: exact status extent is retained by records for the complete returned borrow.
        Ok(unsafe { std::slice::from_raw_parts(pointer.as_ptr(), self.count) })
    }
    fn execute_recorded(
        &self,
        inputs: [&EncodedArray; 2],
    ) -> Result<(EncodedArray, Owner<super::abi::CArray>), MlxError> {
        self.reset_write_fact();
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
        let mut payload = Owner::empty(Rc::clone(api), api.array_free);
        let mut records = Owner::empty(Rc::clone(api), api.array_free);
        let primitive = self
            .owner
            .as_ref()
            .ok_or_else(|| MlxError::Abi("missing checked integer primitive owner".into()))?
            .raw;
        #[cfg(feature = "integer-census")]
        census::record(census::Call::Apply);
        // SAFETY: retained exact-stream primitive, exact physical carrier shapes and separate
        // sibling holders. Lazy make_arrays retains both actual input backings.
        api.status(|| unsafe {
            (api.integer_apply)(
                &raw mut payload.raw,
                &raw mut records.raw,
                primitive,
                left.raw,
                right.raw,
            )
        })?;
        validate(
            &self.session,
            payload.raw,
            self.dtype,
            self.carrier_count,
            self.carrier_width,
        )?;
        validate(&self.session, records.raw, 3, self.count, 4)?;
        let pending = (left, right, payload, records, self.session.clone());
        let terminal = (|| {
            // SAFETY: pending retains both real inputs, payload/status siblings, stream/image
            // before first potentially writing evaluation, through synchronization and waits.
            self.may_have_written.set(true);
            api.status(|| unsafe { (api.array_eval)(pending.2.raw) })?;
            api.status(|| unsafe { (api.array_eval)(pending.3.raw) })?;
            api.status(|| unsafe { (api.synchronize)(self.session.0.stream.raw) })?;
            api.status(|| unsafe { (api.array_wait)(pending.2.raw) })?;
            api.status(|| unsafe { (api.array_wait)(pending.3.raw) })?;
            available(&self.session, pending.2.raw)?;
            available(&self.session, pending.3.raw)
        })();
        if let Err(error) = terminal {
            self.session.0.poisoned.set(true);
            std::mem::forget(pending);
            return Err(MlxError::CompletionUnknown(error.to_string()));
        }
        let (left, right, output, records, _session) = pending;
        left.release()?;
        right.release()?;
        Ok((
            EncodedArray::from_owner(&self.session, self.scalar, self.count, output)?,
            records,
        ))
    }
}
impl Drop for CheckedInteger {
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
    if records
        .iter()
        .any(|&record| !matches!(record, 0 | 1 | 3 | 0x101 | 0x103))
    {
        return Err(MlxError::Abi(
            "invalid checked integer status record".into(),
        ));
    }
    let index = records
        .iter()
        .position(|&record| matches!(record, 1 | 3))
        .or_else(|| records.iter().position(|&record| record != 0));
    Ok(index.map(|index| PcuExecutionFault {
        invocation_id: index as u64,
        kind: match records[index] & 0xff {
            1 => PcuExecutionFaultKind::ArithmeticOverflow,
            _ => PcuExecutionFaultKind::ArithmeticUnderflow,
        },
        recovered: records[index] & 0x100 != 0,
    }))
}
