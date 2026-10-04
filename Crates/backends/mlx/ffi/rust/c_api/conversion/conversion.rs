//! Exact mixed-width conversion through a retained MLX-owned `UInt32` custom primitive.
#[rustfmt::skip]
use std::{ffi::CString,ptr::NonNull,rc::Rc};
#[rustfmt::skip]
use fusion_pcu::{PcuCheckedScalarFaultLaw,PcuDispatchCheckedFloatConversion as Direction,PcuFloatUnderflowPolicy as Policy,PcuRangePolicy as Range,PcuScalarType,PcuExecutionFault,PcuExecutionFaultKind};
#[rustfmt::skip]
use super::{abi::{CComposed,CArray,Opaque},owner::Owner,session::Session,encoded::{EncodedArray,carrier::Carrier},checked::{validate,available},fault};
use crate::MlxError;
pub struct Conversion {
    owner: Option<Owner<CComposed>>,
    session: Session,
    source: PcuScalarType,
    destination: PcuScalarType,
    input_count: usize,
    count: usize,
    law: PcuCheckedScalarFaultLaw,
}
impl Conversion {
    pub fn prepare(
        session: &Session,
        direction: Direction,
        policy: Policy,
        range: Range,
        count: usize,
        broadcast: bool,
    ) -> Result<Self, MlxError> {
        Self::prepare_with_input_extent(
            session,
            direction,
            policy,
            range,
            count,
            broadcast,
            if broadcast { 1 } else { count },
        )
    }
    pub fn prepare_with_input_extent(
        session: &Session,
        direction: Direction,
        policy: Policy,
        range: Range,
        count: usize,
        broadcast: bool,
        input_count: usize,
    ) -> Result<Self, MlxError> {
        session.ensure_ready()?;
        if count == 0 || input_count < if broadcast { 1 } else { count } {
            return Err(MlxError::InvalidExtent);
        }
        let (source, destination) = match direction {
            Direction::F32ToF64 => (PcuScalarType::F32, PcuScalarType::F64),
            Direction::F64ToF32 => (PcuScalarType::F64, PcuScalarType::F32),
        };
        let input = Carrier::assess(source, input_count)?;
        let output = Carrier::assess(destination, count)?;
        let logical = u32::try_from(count).map_err(|_| MlxError::InvalidExtent)?;
        let input_lanes = u32::try_from(input.count).map_err(|_| MlxError::InvalidExtent)?;
        let output_lanes = u32::try_from(output.count).map_err(|_| MlxError::InvalidExtent)?;
        let header = CString::new(include_str!(
            "../../../native/cpp/checked_conversion/checked_conversion.metal"
        ))
        .map_err(|_| MlxError::InvalidExtent)?;
        let body = CString::new(body(direction, policy, range, logical, broadcast))
            .map_err(|_| MlxError::InvalidExtent)?;
        let api = &session.0.api;
        let mut owner = Owner::empty(Rc::clone(api), api.composed_free);
        // SAFETY: fixed profile/header/body and exact counts remain alive through the synchronous
        // contained descriptor constructor. MLX retains its own primitive and explicit GPU stream.
        api.status(|| unsafe {
            (api.composed_new)(
                &raw mut owner.raw,
                session.0.stream.raw,
                3,
                &raw const input_lanes,
                1,
                logical,
                output_lanes,
                1,
                logical,
                header.as_ptr(),
                body.as_ptr(),
            )
        })?;
        owner.require_live()?;
        Ok(Self {
            owner: Some(owner),
            session: session.clone(),
            source,
            destination,
            input_count,
            count,
            law: PcuCheckedScalarFaultLaw::float_conversion(direction, range, policy),
        })
    }
    #[allow(clippy::too_many_lines)] // Pending inputs/payload/status survive the complete terminal and checked-release protocol.
    pub fn execute(
        &self,
        input: &EncodedArray,
    ) -> Result<(EncodedArray, Option<PcuExecutionFault>), MlxError> {
        self.session.ensure_ready()?;
        if input.scalar() != self.source || input.count() != self.input_count {
            return Err(MlxError::InvalidExtent);
        }
        if !input.same_session(&self.session) {
            return Err(MlxError::ForeignSession);
        }
        let primitive = self
            .owner
            .as_ref()
            .ok_or_else(|| MlxError::Abi("missing conversion primitive".into()))?;
        primitive.require_live()?;
        let retained = input.clone_holder()?;
        let api = &self.session.0.api;
        let mut raw = [CArray::empty(); 2];
        // SAFETY: one retained exact input and two output slots cover the contained immutable replay.
        let apply = api.status(|| unsafe {
            (api.composed_apply)(raw.as_mut_ptr(), primitive.raw, &raw const retained.raw, 1)
        });
        let mut payload = Owner::empty(Rc::clone(api), api.array_free);
        payload.raw = raw[0];
        let mut records = Owner::empty(Rc::clone(api), api.array_free);
        records.raw = raw[1];
        apply?;
        payload.require_live()?;
        records.require_live()?;
        let carrier = Carrier::assess(self.destination, self.count)?;
        validate(
            &self.session,
            payload.raw,
            carrier.dtype,
            carrier.count,
            carrier.width,
        )?;
        validate(&self.session, records.raw, 3, self.count, 4)?;
        let pending = (retained, payload, records, self.session.clone());
        let terminal = (|| {
            for array in [pending.1.raw, pending.2.raw] {
                // SAFETY: pending owns every actual input/sibling/stream/image before first eval.
                api.status(|| unsafe { (api.array_eval)(array) })?;
            }
            // SAFETY: the exact prepared GPU stream remains retained through synchronization.
            api.status(|| unsafe { (api.synchronize)(self.session.0.stream.raw) })?;
            for array in [pending.1.raw, pending.2.raw] {
                api.status(|| unsafe { (api.array_wait)(array) })?;
                available(&self.session, array)?;
            }
            Ok::<(), MlxError>(())
        })();
        if let Err(error) = terminal {
            self.session.0.poisoned.set(true);
            std::mem::forget(pending);
            return Err(MlxError::CompletionUnknown(error.to_string()));
        }
        // SAFETY: terminal dense UInt32 records with exact count, owned through validation/scan.
        let pointer = api.guarded(|| unsafe { (api.array_data_u32)(pending.2.raw) })?;
        let pointer = NonNull::new(pointer.cast_mut())
            .ok_or_else(|| MlxError::Abi("nil conversion status backing".into()))?;
        let words = unsafe { std::slice::from_raw_parts(pointer.as_ptr(), self.count) };
        fault::validate(words, self.count, self.law, fault::Encoding::Scalar)?;
        let selected = words
            .iter()
            .position(|word| *word != 0 && *word & 0x100 == 0)
            .or_else(|| words.iter().position(|word| *word != 0));
        let notice = selected.map(|index| PcuExecutionFault {
            invocation_id: index as u64,
            recovered: words[index] & 0x100 != 0,
            kind: match words[index] & 0xff {
                1 => PcuExecutionFaultKind::ArithmeticOverflow,
                3 => PcuExecutionFaultKind::ArithmeticUnderflow,
                _ => PcuExecutionFaultKind::InvalidFloatingOperand,
            },
        });
        let (input, payload, records, _session) = pending;
        input.release()?;
        records.release()?;
        if let Some(fault) = notice
            && !fault.recovered
        {
            payload.release()?;
            return Err(MlxError::Arithmetic(fault));
        }
        Ok((
            EncodedArray::from_owner(&self.session, self.destination, self.count, payload)?,
            notice,
        ))
    }
}
impl Drop for Conversion {
    fn drop(&mut self) {
        if self.session.ensure_ready().is_err() {
            if let Some(owner) = self.owner.take() {
                std::mem::forget(owner);
            }
            std::mem::forget(self.session.clone());
        }
    }
}
fn body(direction: Direction, policy: Policy, range: Range, count: u32, broadcast: bool) -> String {
    let index = if broadcast { "0u" } else { "id" };
    let policy = match policy {
        Policy::IeeeAfterRounding => 0,
        Policy::RejectSubnormalResult => 1,
        Policy::AllowGradualUnderflow => 2,
    };
    let statement = match direction {
        Direction::F32ToF64 => format!(
            "uint3 result=widen(input0[{index}]);output0[2u*id]=result.x;output0[2u*id+1u]=result.y;uint status=result.z;"
        ),
        Direction::F64ToF32 => format!(
            "Result result=narrow(uint2(input0[2u*{index}],input0[2u*{index}+1u]),{policy}u);output0[id]=result.bits;uint status=result.status;"
        ),
    };
    format!(
        "uint id=thread_position_in_grid.x;if(id>={count}u)return;{statement}if({}&&(status==1u||status==3u))status|=0x100u;records[id]=status;",
        range == Range::Clamp
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn mixed_width_emission_has_exact_input_and_payload_lanes() {
        let widen = body(
            Direction::F32ToF64,
            Policy::RejectSubnormalResult,
            Range::Clamp,
            65,
            false,
        );
        assert!(widen.contains("widen(input0[id])"));
        assert!(widen.contains("output0[2u*id+1u]=result.y"));
        let narrow = body(
            Direction::F64ToF32,
            Policy::AllowGradualUnderflow,
            Range::Reject,
            65,
            true,
        );
        assert!(narrow.contains("input0[2u*0u+1u]"));
        assert!(narrow.contains("),2u)"));
        assert!(narrow.contains("output0[id]=result.bits"));
        for source in [widen, narrow] {
            assert!(source.contains("id>=65u"));
            assert!(source.contains("records[id]=status"));
            assert!(!source.contains("float"));
        }
    }
}
