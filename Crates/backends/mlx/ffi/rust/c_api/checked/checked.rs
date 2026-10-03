//! MLX-owned checked carrier arrays and retained sibling payload/status primitive.
#[rustfmt::skip]
use std::{
    cell::{
        Cell,
        RefCell,
    },
    ptr::NonNull,
    rc::Rc,
};
#[rustfmt::skip]
use fusion_pcu::{
    PcuScalarType,
    PcuDispatchFloatUnaryOp,
    PcuFloatUnderflowPolicy,
    PcuRangePolicy,
    PcuExecutionFault,
    PcuExecutionFaultKind,
};
use crate::MlxError;
#[rustfmt::skip]
use super::{
    abi::{
        CArray,
        CCheckedUnary,
    },
    owner::Owner,
    session::Session,
};

pub struct CheckedUnary {
    fault_law: fusion_pcu::PcuCheckedScalarFaultLaw,
    owner: Option<Owner<CCheckedUnary>>,
    last_output: RefCell<Option<Owner<CArray>>>,
    may_have_written: Cell<bool>,
    session: Session,
    scalar: PcuScalarType,
    upload_format: u32,
    carrier_width: usize,
    carrier_count: usize,
    input_carrier_count: usize,
    dtype: i32,
    width: usize,
    count: usize,
    input_count: usize,
}
impl CheckedUnary {
    #[allow(clippy::too_many_arguments)] // Fixed dtype/op/range/underflow/shape admission stays cold.
    pub fn prepare(
        session: &Session,
        scalar: PcuScalarType,
        op: PcuDispatchFloatUnaryOp,
        policy: PcuFloatUnderflowPolicy,
        range: PcuRangePolicy,
        count: usize,
        broadcast: bool,
    ) -> Result<Self, MlxError> {
        Self::prepare_internal(session, scalar, op, policy, range, count, broadcast, None)
    }
    #[allow(clippy::too_many_arguments)] // Exact request and full resident shape are independent cold dimensions.
    pub fn prepare_with_input_extent(
        session: &Session,
        scalar: PcuScalarType,
        op: PcuDispatchFloatUnaryOp,
        policy: PcuFloatUnderflowPolicy,
        range: PcuRangePolicy,
        count: usize,
        broadcast: bool,
        input_count: usize,
    ) -> Result<Self, MlxError> {
        Self::prepare_internal(
            session,
            scalar,
            op,
            policy,
            range,
            count,
            broadcast,
            Some(input_count),
        )
    }
    #[allow(clippy::too_many_arguments, clippy::too_many_lines)] // One retained exact tuple/shape/ownership constructor guards both endpoints.
    fn prepare_internal(
        session: &Session,
        scalar: PcuScalarType,
        op: PcuDispatchFloatUnaryOp,
        policy: PcuFloatUnderflowPolicy,
        range: PcuRangePolicy,
        count: usize,
        broadcast: bool,
        full_extent: Option<usize>,
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
        let carrier = super::encoded::carrier::Carrier::assess(scalar, count)?;
        let minimum = if broadcast { 1 } else { count };
        let input_count = full_extent.unwrap_or(minimum);
        if input_count < minimum {
            return Err(MlxError::InvalidExtent);
        }
        let input_carrier = super::encoded::carrier::Carrier::assess(scalar, input_count)?;
        let count_native = i32::try_from(count).map_err(|_| MlxError::InvalidExtent)?;
        let input_native = u32::try_from(input_count).map_err(|_| MlxError::InvalidExtent)?;
        let operation = match op {
            PcuDispatchFloatUnaryOp::Neg => 0,
            PcuDispatchFloatUnaryOp::Relu => 1,
        };
        let underflow = match policy {
            PcuFloatUnderflowPolicy::IeeeAfterRounding => 0,
            PcuFloatUnderflowPolicy::RejectSubnormalResult => 1,
            PcuFloatUnderflowPolicy::AllowGradualUnderflow => 2,
        };
        let api = &session.0.api;
        let mut owner = Owner::empty(Rc::clone(api), api.checked_free);
        if full_extent.is_some() {
            // SAFETY: the full physical shape is checked independently of the output/status
            // domain; actual native apply remains an exact-shape, same-stream operation.
            api.status(|| unsafe {
                (api.checked_prefix_new)(
                    &raw mut owner.raw,
                    session.0.stream.raw,
                    format,
                    operation,
                    underflow,
                    u32::from(range == PcuRangePolicy::Clamp),
                    count_native.cast_unsigned(),
                    input_native,
                    i32::from(broadcast),
                )
            })?;
        } else {
            // SAFETY: all bounded literals, explicit live stream and empty unique owner.
            // The legacy exact endpoint contains every exception and creates lazy descriptors.
            api.status(|| unsafe {
                (api.checked_new)(
                    &raw mut owner.raw,
                    session.0.stream.raw,
                    format,
                    operation,
                    underflow,
                    u32::from(range == PcuRangePolicy::Clamp),
                    count_native.cast_unsigned(),
                    i32::from(broadcast),
                )
            })?;
        }
        owner.require_live()?;
        let prepared = Self {
            fault_law: fusion_pcu::PcuCheckedScalarFaultLaw::float_unary(scalar, op, range, policy)
                .ok_or(MlxError::UnsupportedScalar(scalar))?,
            owner: Some(owner),
            last_output: RefCell::new(None),
            may_have_written: Cell::new(false),
            session: session.clone(),
            scalar,
            upload_format: carrier.upload_format,
            carrier_width: carrier.width,
            carrier_count: carrier.count,
            input_carrier_count: input_carrier.count,
            dtype: carrier.dtype,
            width: carrier.logical_width,
            count,
            input_count,
        };
        // Compile/cache the fixed MLX library and pipeline during cold preparation, with
        // initialized zero input and a real terminal payload/status evaluation.
        prepared.execute(
            &vec![0; prepared.input_count * prepared.width],
            &mut vec![0; count * prepared.width],
        )?;
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
        input: &super::encoded::EncodedArray,
    ) -> Result<(super::encoded::EncodedArray, Option<PcuExecutionFault>), MlxError> {
        self.reset_write_fact();
        self.session.ensure_ready()?;
        let scalar = self.scalar;
        if input.scalar() != scalar {
            return Err(MlxError::UnsupportedScalar(input.scalar()));
        }
        if input.count() != self.input_count {
            return Err(MlxError::InvalidExtent);
        }
        if !input.same_session(&self.session) {
            return Err(MlxError::ForeignSession);
        }
        let input_owner = input.clone_holder()?;
        let api = &self.session.0.api;
        let mut payload = Owner::empty(Rc::clone(api), api.array_free);
        let mut records = Owner::empty(Rc::clone(api), api.array_free);
        // SAFETY: retained same-stream primitive and exact copied physical carrier/shape.
        // MLX make_arrays retains the official input backing for both sibling outputs.
        let primitive = self
            .owner
            .as_ref()
            .ok_or_else(|| MlxError::Abi("missing checked primitive owner".into()))?
            .raw;
        api.status(|| unsafe {
            (api.checked_apply)(
                &raw mut payload.raw,
                &raw mut records.raw,
                primitive,
                input_owner.raw,
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
        let pending = (input_owner, payload, records, self.session.clone());
        let terminal = (|| {
            // SAFETY: pending retains real input, both sibling outputs, explicit stream/image;
            // evaluating one sibling realizes the shared primitive, then both are awaited.
            self.may_have_written.set(true);
            api.status(|| unsafe { (api.array_eval)(pending.1.raw) })?;
            api.status(|| unsafe { (api.array_eval)(pending.2.raw) })?;
            api.status(|| unsafe { (api.synchronize)(self.session.0.stream.raw) })?;
            api.status(|| unsafe { (api.array_wait)(pending.1.raw) })?;
            api.status(|| unsafe { (api.array_wait)(pending.2.raw) })?;
            available(&self.session, pending.1.raw)?;
            available(&self.session, pending.2.raw)
        })();
        if let Err(error) = terminal {
            self.session.0.poisoned.set(true);
            std::mem::forget(pending);
            // The prepared primitive can reference pipeline/native work; quarantine its
            // session/image too. Caller output has not been inspected or published.
            std::mem::forget(self.session.clone());
            return Err(MlxError::CompletionUnknown(error.to_string()));
        }
        // SAFETY: terminal dense UInt32 status owner with verified exact byte/count extent.
        let status = api.guarded(|| unsafe { (api.array_data_u32)(pending.2.raw) })?;
        let status = NonNull::new(status.cast_mut())
            .ok_or_else(|| MlxError::Abi("nil checked status backing".into()))?;
        // SAFETY: pending retains exact dense initialized count UInt32 slots until scan ends.
        let status = unsafe { std::slice::from_raw_parts(status.as_ptr(), self.count) };
        super::fault::validate(
            status,
            self.count,
            self.fault_law,
            super::fault::Encoding::Scalar,
        )?;
        let fault = select_fault(status)?;
        if let Some(fault) = fault
            && !fault.recovered
        {
            return Err(MlxError::Arithmetic(fault));
        }

        let (input, payload, records, _session) = pending;
        input.release()?;
        records.release()?;
        let output =
            super::encoded::EncodedArray::from_owner(&self.session, scalar, self.count, payload)?;
        Ok((output, fault))
    }
    pub fn execute(&self, input: &[u8], output: &mut [u8]) -> Result<(), MlxError> {
        self.reset_write_fact();
        self.session.ensure_ready()?;
        if input.len() != self.input_count * self.width || output.len() != self.count * self.width {
            return Err(MlxError::InvalidExtent);
        }
        if let Some(prior) = self.last_output.borrow_mut().take() {
            prior.release()?;
        }
        let api = &self.session.0.api;
        let mut input_owner = Owner::empty(Rc::clone(api), api.array_free);
        // SAFETY: exact initialized span/checked count; native staging copies even unaligned
        // UInt16 bytes via aligned temporary into MLX ownership before returning.
        api.status(|| unsafe {
            (api.checked_upload)(
                &raw mut input_owner.raw,
                input.as_ptr().cast(),
                self.input_carrier_count,
                self.upload_format,
            )
        })?;
        validate(
            &self.session,
            input_owner.raw,
            self.dtype,
            self.input_carrier_count,
            self.carrier_width,
        )?;
        let mut payload = Owner::empty(Rc::clone(api), api.array_free);
        let mut records = Owner::empty(Rc::clone(api), api.array_free);
        // SAFETY: retained same-stream primitive and exact copied physical carrier/shape.
        // MLX make_arrays retains the official input backing for both sibling outputs.
        let primitive = self
            .owner
            .as_ref()
            .ok_or_else(|| MlxError::Abi("missing checked primitive owner".into()))?
            .raw;
        api.status(|| unsafe {
            (api.checked_apply)(
                &raw mut payload.raw,
                &raw mut records.raw,
                primitive,
                input_owner.raw,
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
        let pending = (input_owner, payload, records, self.session.clone());
        let terminal = (|| {
            // SAFETY: pending retains real input, both sibling outputs, explicit stream/image;
            // evaluating one sibling realizes the shared primitive, then both are awaited.
            self.may_have_written.set(true);
            api.status(|| unsafe { (api.array_eval)(pending.1.raw) })?;
            api.status(|| unsafe { (api.array_eval)(pending.2.raw) })?;
            api.status(|| unsafe { (api.synchronize)(self.session.0.stream.raw) })?;
            api.status(|| unsafe { (api.array_wait)(pending.1.raw) })?;
            api.status(|| unsafe { (api.array_wait)(pending.2.raw) })?;
            available(&self.session, pending.1.raw)?;
            available(&self.session, pending.2.raw)
        })();
        if let Err(error) = terminal {
            self.session.0.poisoned.set(true);
            std::mem::forget(pending);
            // The prepared primitive can reference pipeline/native work; quarantine its
            // session/image too. Caller output has not been inspected or published.
            std::mem::forget(self.session.clone());
            return Err(MlxError::CompletionUnknown(error.to_string()));
        }
        // SAFETY: terminal dense UInt32 status owner with verified exact byte/count extent.
        let status = api.guarded(|| unsafe { (api.array_data_u32)(pending.2.raw) })?;
        let status = NonNull::new(status.cast_mut())
            .ok_or_else(|| MlxError::Abi("nil checked status backing".into()))?;
        // SAFETY: pending retains exact dense initialized count UInt32 slots until scan ends.
        let status = unsafe { std::slice::from_raw_parts(status.as_ptr(), self.count) };
        super::fault::validate(
            status,
            self.count,
            self.fault_law,
            super::fault::Encoding::Scalar,
        )?;
        let fault = select_fault(status)?;
        if let Some(fault) = fault
            && !fault.recovered
        {
            return Err(MlxError::Arithmetic(fault));
        }
        // SAFETY: terminal validated payload dtype selects its exact official typed accessor.
        let data = api.guarded(|| unsafe {
            if self.dtype == 1 {
                (api.array_data_u8)(pending.1.raw).cast::<u8>()
            } else if self.dtype == 2 {
                (api.array_data_u16)(pending.1.raw).cast::<u8>()
            } else {
                (api.array_data_u32)(pending.1.raw).cast::<u8>()
            }
        })?;
        let data = NonNull::new(data.cast_mut())
            .ok_or_else(|| MlxError::Abi("nil checked carrier backing".into()))?;
        // Release other holders before publication, retaining the completed payload as
        // this prepared owner's latest result. Its checked release precedes the next call.
        let (input, payload, records, _session) = pending;
        input.release()?;
        records.release()?;
        *self.last_output.borrow_mut() = Some(payload);
        // SAFETY: payload retains initialized dense exact byte extent; caller output is
        // exclusive and disjoint from copied MLX backing; all fallible work is complete.
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), output.as_mut_ptr(), output.len());
        }
        fault.map_or(Ok(()), |fault| Err(MlxError::Arithmetic(fault)))
    }
}
impl Drop for CheckedUnary {
    fn drop(&mut self) {
        if self.session.ensure_ready().is_err() {
            // A failed terminal wait leaves actual primitive/backing owners and stream/image
            // quarantined, rather than releasing an object still referenced by native work.
            if let Some(owner) = self.owner.take() {
                std::mem::forget(owner);
            }
            if let Some(output) = self.last_output.get_mut().take() {
                std::mem::forget(output);
            }
            std::mem::forget(self.session.clone());
        }
    }
}
pub(super) fn validate(
    session: &Session,
    array: CArray,
    dtype: i32,
    count: usize,
    width: usize,
) -> Result<(), MlxError> {
    let api = &session.0.api;
    // SAFETY: a selected catch-all metadata endpoint on a live official holder.
    let (actual, rank) =
        api.guarded(|| unsafe { ((api.array_dtype)(array), (api.array_ndim)(array)) })?;
    if actual != dtype || rank != 1 {
        return Err(MlxError::Abi("checked carrier dtype/rank mismatch".into()));
    }
    // SAFETY: rank one precedes the indexed dimension/stride accesses.
    let (dimension, size, bytes, strides) = api.guarded(|| unsafe {
        (
            (api.array_dim)(array, 0),
            (api.array_size)(array),
            (api.array_nbytes)(array),
            (api.array_strides)(array),
        )
    })?;
    if usize::try_from(dimension).ok() != Some(count)
        || size != count
        || bytes != count * width
        || strides.is_null()
    {
        return Err(MlxError::Abi(
            "checked carrier exact byte/shape mismatch".into(),
        ));
    }
    // SAFETY: the live verified rank-one holder owns one native stride entry.
    if unsafe { *strides } != 1 {
        return Err(MlxError::Abi("checked carrier must be dense".into()));
    }
    Ok(())
}
pub(super) fn available(session: &Session, array: CArray) -> Result<(), MlxError> {
    let mut available = false;
    let mut contiguous = false;
    // SAFETY: both official getters observe a retained terminal array holder.
    session
        .0
        .api
        .status(|| unsafe { (session.0.api.array_available)(&raw mut available, array) })?;
    session
        .0
        .api
        .status(|| unsafe { (session.0.api.array_contiguous)(&raw mut contiguous, array) })?;
    if !available || !contiguous {
        return Err(MlxError::Abi(
            "terminal checked backing unavailable/noncontiguous".into(),
        ));
    }
    Ok(())
}
fn select_fault(records: &[u32]) -> Result<Option<PcuExecutionFault>, MlxError> {
    if records
        .iter()
        .any(|&record| !matches!(record, 0 | 3 | 4 | 0x103))
    {
        return Err(MlxError::Abi("invalid checked unary status record".into()));
    }
    let index = records
        .iter()
        .position(|&record| record == 3 || record == 4)
        .or_else(|| records.iter().position(|&record| record != 0));
    Ok(index.map(|index| PcuExecutionFault {
        invocation_id: index as u64,
        kind: if records[index] & 0xff == 4 {
            PcuExecutionFaultKind::InvalidFloatingOperand
        } else {
            PcuExecutionFaultKind::ArithmeticUnderflow
        },
        recovered: records[index] & 0x100 != 0,
    }))
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
