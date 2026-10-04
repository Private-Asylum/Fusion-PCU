//! Detached direct-C two-input finite backward selection with retained immutable terminal siblings.
use std::{ffi::CString, ptr::NonNull, rc::Rc};
use fusion_pcu::{
    PcuCheckedScalarFaultLaw, PcuFloatUnderflowPolicy as Policy, PcuRangePolicy as Range,
    PcuScalarType, PcuExecutionFault, PcuExecutionFaultKind,
};
use super::{
    abi::{CComposed, CArray, Opaque},
    owner::Owner,
    session::Session,
    encoded::{EncodedArray, carrier::Carrier},
    checked::{validate, available},
    fault,
};
use crate::MlxError;
pub struct ReluBackward {
    owner: Option<Owner<CComposed>>,
    session: Session,
    scalar: PcuScalarType,
    input_count: [usize; 2],
    count: usize,
    law: PcuCheckedScalarFaultLaw,
}
impl ReluBackward {
    pub fn prepare(
        session: &Session,
        scalar: PcuScalarType,
        policy: Policy,
        count: usize,
        broadcast: [bool; 2],
    ) -> Result<Self, MlxError> {
        session.ensure_ready()?;
        if !matches!(
            scalar,
            PcuScalarType::F16
                | PcuScalarType::BF16
                | PcuScalarType::F8E4M3FN
                | PcuScalarType::F8E5M2
                | PcuScalarType::F32
                | PcuScalarType::F64
        ) {
            return Err(MlxError::UnsupportedScalar(scalar));
        }
        let input_count = broadcast.map(|scalar| if scalar { 1 } else { count });
        let input = input_count.map(|count| Carrier::assess(scalar, count));
        let [left, right] = input;
        let left = left?;
        let right = right?;
        let output = Carrier::assess(scalar, count)?;
        let counts = [
            u32::try_from(left.count).map_err(|_| MlxError::InvalidExtent)?,
            u32::try_from(right.count).map_err(|_| MlxError::InvalidExtent)?,
        ];
        let logical = u32::try_from(count).map_err(|_| MlxError::InvalidExtent)?;
        let dtype = u32::try_from(output.dtype).map_err(|_| MlxError::InvalidExtent)?;
        let lanes = u32::try_from(output.count).map_err(|_| MlxError::InvalidExtent)?;
        let header = CString::new(include_str!(
            "../../../native/cpp/relu_backward/relu_backward.metal"
        ))
        .map_err(|_| MlxError::InvalidExtent)?;
        let body = CString::new(body(scalar, policy, logical, broadcast))
            .map_err(|_| MlxError::InvalidExtent)?;
        let api = &session.0.api;
        let mut owner = Owner::empty(Rc::clone(api), api.composed_free);
        // SAFETY: fixed two-input counts and bounded own source survive the contained constructor.
        api.status(|| unsafe {
            (api.composed_new)(
                &raw mut owner.raw,
                session.0.stream.raw,
                dtype,
                counts.as_ptr(),
                2,
                logical,
                lanes,
                1,
                logical,
                header.as_ptr(),
                body.as_ptr(),
            )
        })?;
        owner.require_live()?;
        let law = PcuCheckedScalarFaultLaw::float_relu_backward(scalar, Range::Reject, policy)
            .ok_or(MlxError::UnsupportedScalar(scalar))?;
        Ok(Self {
            owner: Some(owner),
            session: session.clone(),
            scalar,
            input_count,
            count,
            law,
        })
    }
    #[allow(clippy::too_many_lines)] // Pending inputs/payload/status survive the complete terminal and checked-release protocol.
    pub fn execute(
        &self,
        input: &EncodedArray,
        upstream: &EncodedArray,
    ) -> Result<(EncodedArray, Option<PcuExecutionFault>), MlxError> {
        self.session.ensure_ready()?;
        for (array, count) in [input, upstream].into_iter().zip(self.input_count) {
            if array.scalar() != self.scalar || array.count() != count {
                return Err(MlxError::InvalidExtent);
            }
            if !array.same_session(&self.session) {
                return Err(MlxError::ForeignSession);
            }
        }
        let primitive = self
            .owner
            .as_ref()
            .ok_or_else(|| MlxError::Abi("missing backward primitive".into()))?;
        primitive.require_live()?;
        let retained = [input.clone_holder()?, upstream.clone_holder()?];
        let inputs = [retained[0].raw, retained[1].raw];
        let api = &self.session.0.api;
        let mut raw = [CArray::empty(); 2];
        // SAFETY: two retained exact inputs and two output slots cover the contained immutable replay.
        let apply = api.status(|| unsafe {
            (api.composed_apply)(raw.as_mut_ptr(), primitive.raw, inputs.as_ptr(), 2)
        });
        let mut payload = Owner::empty(Rc::clone(api), api.array_free);
        payload.raw = raw[0];
        let mut records = Owner::empty(Rc::clone(api), api.array_free);
        records.raw = raw[1];
        apply?;
        payload.require_live()?;
        records.require_live()?;
        let carrier = Carrier::assess(self.scalar, self.count)?;
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
            .ok_or_else(|| MlxError::Abi("nil backward status backing".into()))?;
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
        for owner in input {
            owner.release()?;
        }
        records.release()?;
        if let Some(fault) = notice
            && !fault.recovered
        {
            payload.release()?;
            return Err(MlxError::Arithmetic(fault));
        }
        Ok((
            EncodedArray::from_owner(&self.session, self.scalar, self.count, payload)?,
            notice,
        ))
    }
}
impl Drop for ReluBackward {
    fn drop(&mut self) {
        if self.session.ensure_ready().is_err() {
            if let Some(owner) = self.owner.take() {
                std::mem::forget(owner);
            }
            std::mem::forget(self.session.clone());
        }
    }
}
fn body(scalar: PcuScalarType, policy: Policy, count: u32, broadcast: [bool; 2]) -> String {
    let index = broadcast.map(|flag| if flag { "0u" } else { "id" });
    let format = match scalar {
        PcuScalarType::F16 => 1,
        PcuScalarType::BF16 => 2,
        PcuScalarType::F8E4M3FN => 3,
        PcuScalarType::F8E5M2 => 4,
        _ => 0,
    };
    if format != 0 {
        return format!(
            "uint id=thread_position_in_grid.x;if(id>={count}u)return;uint status;uint value=backward_low(uint(input0[{}]),uint(input1[{}]),{format}u,{},status);output0[id]=value;records[id]=status;",
            index[0],
            index[1],
            policy == Policy::RejectSubnormalResult
        );
    }
    let wide = scalar == PcuScalarType::F64;
    let input = |name: &str, position: &str| {
        if wide {
            format!("uint2({name}[2u*{position}],{name}[2u*{position}+1u])")
        } else {
            format!("uint2({name}[{position}],0u)")
        }
    };
    let a = input("input0", index[0]);
    let b = input("input1", index[1]);
    let output = if wide {
        "output0[2u*id]=value.x;output0[2u*id+1u]=value.y;"
    } else {
        "output0[id]=value.x;"
    };
    format!(
        "uint id=thread_position_in_grid.x;if(id>={count}u)return;uint status;uint2 value=backward_bits({a},{b},{wide},{},status);{output}records[id]=status;",
        policy == Policy::RejectSubnormalResult
    )
}
