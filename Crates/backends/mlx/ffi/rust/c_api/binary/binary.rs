//! MLX-owned exact low binary primitive with retained payload/status siblings.
#[cfg(feature = "binary-census")]
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
    PcuDispatchFloatBinaryOp as Op,
    PcuFloatUnderflowPolicy as Policy,
    PcuRangePolicy as Range,
    PcuExecutionFault,
    PcuExecutionFaultKind,
};
use crate::MlxError;
#[rustfmt::skip]
use super::{
    abi::CCheckedBinary,
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

pub struct CheckedBinary {
    fault_law: fusion_pcu::PcuCheckedScalarFaultLaw,
    owner: Option<Owner<CCheckedBinary>>,
    session: Session,
    scalar: PcuScalarType,
    dtype: i32,
    width: usize,
    count: usize,
    carrier_count: usize,
    carrier_width: usize,
    inputs: [usize; 2],
    may_have_written: Cell<bool>,
}
impl CheckedBinary {
    #[allow(clippy::too_many_arguments)] // Independent cold logical/operation/policy/operand roles.
    pub fn prepare(
        session: &Session,
        scalar: PcuScalarType,
        op: Op,
        policy: Policy,
        range: Range,
        count: usize,
        inputs: [usize; 2],
        broadcast: [bool; 2],
    ) -> Result<Self, MlxError> {
        Self::prepare_internal(
            session, scalar, op, policy, range, count, inputs, broadcast, None,
        )
    }
    #[allow(clippy::too_many_arguments)] // Separate logical operand minima and cold-frozen full capacities.
    pub fn prepare_with_input_extents(
        session: &Session,
        scalar: PcuScalarType,
        op: Op,
        policy: Policy,
        range: Range,
        count: usize,
        inputs: [usize; 2],
        broadcast: [bool; 2],
        full: [usize; 2],
    ) -> Result<Self, MlxError> {
        Self::prepare_internal(
            session,
            scalar,
            op,
            policy,
            range,
            count,
            inputs,
            broadcast,
            Some(full),
        )
    }
    #[allow(clippy::too_many_arguments, clippy::too_many_lines)] // Shared exact/full-shape cold preparation retains complete checked tuple and terminal priming.
    fn prepare_internal(
        session: &Session,
        scalar: PcuScalarType,
        op: Op,
        policy: Policy,
        range: Range,
        count: usize,
        inputs: [usize; 2],
        broadcast: [bool; 2],
        full: Option<[usize; 2]>,
    ) -> Result<Self, MlxError> {
        session.ensure_ready()?;
        let format = match scalar {
            PcuScalarType::F16 => 0,
            PcuScalarType::BF16 => 1,
            PcuScalarType::F8E4M3FN => 2,
            PcuScalarType::F8E5M2 => 3,
            PcuScalarType::F32 => 4,
            PcuScalarType::F64 => 5,
            _ => return Err(MlxError::UnsupportedScalar(scalar)),
        };
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
            // Frozen native Shape counts physical carriers, including both F64 UInt32 limbs.
            let _ = Carrier::assess(scalar, extent)?;
        }
        let operation = match op {
            Op::Add => 0,
            Op::Sub => 1,
            Op::Mul => 2,
            Op::Div => 3,
        };
        let underflow = match policy {
            Policy::IeeeAfterRounding => 0,
            Policy::RejectSubnormalResult => 1,
            Policy::AllowGradualUnderflow => 2,
        };
        let api = &session.0.api;
        #[cfg(feature = "binary-census")]
        census::record(if full.is_some() {
            census::Call::PrefixConstructor
        } else {
            census::Call::ExactConstructor
        });
        let constructor = if full.is_some() {
            api.binary_prefix_new
        } else {
            api.binary_new
        };
        let left_count = u32::try_from(inputs[0]).map_err(|_| MlxError::InvalidExtent)?;
        let right_count = u32::try_from(inputs[1]).map_err(|_| MlxError::InvalidExtent)?;
        let mut owner = Owner::empty(Rc::clone(api), api.binary_free);
        // SAFETY: bounded admitted literals, exact explicit stream and empty unique owner;
        // the audited endpoint contains every exception and constructs lazy descriptors only.
        api.status(|| unsafe {
            constructor(
                &raw mut owner.raw,
                session.0.stream.raw,
                format,
                operation,
                underflow,
                u32::from(range == Range::Clamp),
                count_native.cast_unsigned(),
                left_count,
                right_count,
                u32::from(broadcast[0]) | (u32::from(broadcast[1]) << 1),
            )
        })?;
        owner.require_live()?;
        let prepared = Self {
            fault_law: fusion_pcu::PcuCheckedScalarFaultLaw::float_binary(
                scalar, op, range, policy,
            )
            .ok_or(MlxError::UnsupportedScalar(scalar))?,
            owner: Some(owner),
            session: session.clone(),
            scalar,
            dtype,
            width,
            count,
            carrier_count: carrier.count,
            carrier_width: carrier.width,
            inputs,
            may_have_written: Cell::new(false),
        };
        // Zero / one is valid for all four operations and every admitted underflow/range
        // policy. Complete the real sibling primitive here to freeze compilation cold.
        let left = EncodedArray::upload(session, scalar, inputs[0], &vec![0; inputs[0] * width])?;
        let one = match scalar {
            PcuScalarType::F16 => 0x3c00_u64,
            PcuScalarType::BF16 => 0x3f80,
            PcuScalarType::F8E4M3FN => 0x38,
            PcuScalarType::F8E5M2 => 0x3c,
            PcuScalarType::F32 => 0x3f80_0000,
            PcuScalarType::F64 => 0x3ff0_0000_0000_0000,
            _ => return Err(MlxError::UnsupportedScalar(scalar)),
        }
        .to_le_bytes();
        let mut data = vec![0; inputs[1] * width];
        for value in data.chunks_exact_mut(width) {
            value.copy_from_slice(&one[..width]);
        }
        let right = EncodedArray::upload(session, scalar, inputs[1], &data)?;
        #[cfg(feature = "binary-census")]
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
            .ok_or_else(|| MlxError::Abi("missing checked binary primitive owner".into()))?
            .raw;
        #[cfg(feature = "binary-census")]
        census::record(census::Call::Apply);
        // SAFETY: retained exact-stream primitive, exact physical carrier shapes and separate
        // sibling holders. Lazy make_arrays retains both actual input backings.
        api.status(|| unsafe {
            (api.binary_apply)(
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
        // SAFETY: exact terminal dense initialized count UInt32 status owner is retained.
        let records = api.guarded(|| unsafe { (api.array_data_u32)(pending.3.raw) })?;
        let records = NonNull::new(records.cast_mut())
            .ok_or_else(|| MlxError::Abi("nil checked binary status backing".into()))?;
        // SAFETY: validated retained status extent remains live for the entire deterministic scan.
        let records = unsafe { std::slice::from_raw_parts(records.as_ptr(), self.count) };
        super::fault::validate(
            records,
            self.count,
            self.fault_law,
            super::fault::Encoding::Scalar,
        )?;
        let fault = select_fault(records)?;
        if let Some(fault) = fault
            && !fault.recovered
        {
            return Err(MlxError::Arithmetic(fault));
        }
        let (left, right, output, records, _session) = pending;
        left.release()?;
        right.release()?;
        records.release()?;
        Ok((
            EncodedArray::from_owner(&self.session, self.scalar, self.count, output)?,
            fault,
        ))
    }
    pub fn execute(&self, inputs: [&[u8]; 2], output: &mut [u8]) -> Result<(), MlxError> {
        self.execute_host_mapped(&inputs, [0, 1], output)
    }
    /// Copies each unique source binding once, even when both mathematical operands reuse it.
    pub fn execute_host_mapped(
        &self,
        inputs: &[&[u8]],
        roles: [usize; 2],
        output: &mut [u8],
    ) -> Result<(), MlxError> {
        self.reset_write_fact();
        if !(1..=2).contains(&inputs.len())
            || roles.iter().any(|&role| role >= inputs.len())
            || output.len() != self.count * self.width
        {
            return Err(MlxError::InvalidExtent);
        }
        let mut extents = [0; 2];
        for (slot, bytes) in inputs.iter().enumerate() {
            if bytes.is_empty() || !bytes.chunks_exact(self.width).remainder().is_empty() {
                return Err(MlxError::InvalidExtent);
            }
            extents[slot] = bytes.len() / self.width;
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
}
impl Drop for CheckedBinary {
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
        .any(|&record| !matches!(record, 0 | 1 | 2 | 3 | 4 | 0x101 | 0x103))
    {
        return Err(MlxError::Abi("invalid checked binary status record".into()));
    }
    let index = records
        .iter()
        .position(|&record| matches!(record, 1..=4))
        .or_else(|| records.iter().position(|&record| record != 0));
    Ok(index.map(|index| PcuExecutionFault {
        invocation_id: index as u64,
        kind: match records[index] & 0xff {
            1 => PcuExecutionFaultKind::ArithmeticOverflow,
            2 => PcuExecutionFaultKind::DivideByZero,
            3 => PcuExecutionFaultKind::ArithmeticUnderflow,
            _ => PcuExecutionFaultKind::InvalidFloatingOperand,
        },
        recovered: records[index] & 0x100 != 0,
    }))
}
