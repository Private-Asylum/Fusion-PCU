//! Direct-C ordered SGD update with independently retained payload and event-domain status siblings.
use fusion_pcu::dialect::tensor::TensorStrictFaultDomain;
use std::{ffi::CString, ptr::NonNull, rc::Rc};
use fusion_pcu::{
    PcuFloatUnderflowPolicy as Policy, PcuScalarType, PcuExecutionFault, PcuExecutionFaultKind,
};
use super::{
    abi::{CComposed, CArray, Opaque},
    owner::Owner,
    session::Session,
    encoded::{EncodedArray, carrier::Carrier},
    checked::{validate, available},
};
use crate::MlxError;
pub struct StrictSgd {
    owner: Option<Owner<CComposed>>,
    session: Session,
    scalar: PcuScalarType,
    input_count: [usize; 2],
    count: usize,
    domain: TensorStrictFaultDomain,
    events: usize,
}
impl StrictSgd {
    #[cfg(test)]
    pub fn validate_domain(words: &[u32], domain: TensorStrictFaultDomain) -> Result<(), MlxError> {
        validate_records(words, domain)
    }
    pub fn prepare(
        session: &Session,
        scalar: PcuScalarType,
        policy: Policy,
        count: usize,
        learning_rate: f32,
    ) -> Result<Self, MlxError> {
        session.ensure_ready()?;
        if !matches!(scalar, PcuScalarType::F32 | PcuScalarType::F64) {
            return Err(MlxError::UnsupportedScalar(scalar));
        }
        if !learning_rate.is_finite() {
            return Err(MlxError::InvalidExtent);
        }
        let input_count = [count; 2];
        let domain = TensorStrictFaultDomain::sgd(
            scalar,
            u64::try_from(count).map_err(|_| MlxError::InvalidExtent)?,
            policy,
        )
        .ok_or(MlxError::InvalidExtent)?;
        let events = usize::try_from(domain.event_extent()).map_err(|_| MlxError::InvalidExtent)?;
        Carrier::assess(PcuScalarType::U32, events)?;
        let input = input_count.map(|count| Carrier::assess(scalar, count));
        let [left, right] = input;
        let left = left?;
        let right = right?;
        let output = Carrier::assess(scalar, count)?;
        let counts = [
            u32::try_from(left.count).map_err(|_| MlxError::InvalidExtent)?,
            u32::try_from(right.count).map_err(|_| MlxError::InvalidExtent)?,
        ];
        let event_lanes = u32::try_from(events).map_err(|_| MlxError::InvalidExtent)?;
        let logical = u32::try_from(count).map_err(|_| MlxError::InvalidExtent)?;
        let lanes = u32::try_from(output.count).map_err(|_| MlxError::InvalidExtent)?;
        let header = CString::new(header(scalar)).map_err(|_| MlxError::InvalidExtent)?;
        let rate_bits = if scalar == PcuScalarType::F32 {
            u64::from(learning_rate.to_bits())
        } else {
            fusion_pcu::PcuCheckedFloatWidening::pcu_checked_to_f64(learning_rate)
                .map_err(|_| MlxError::InvalidExtent)?
                .to_bits()
        };
        let body = CString::new(body(scalar, policy, count, rate_bits))
            .map_err(|_| MlxError::InvalidExtent)?;
        let api = &session.0.api;
        let mut owner = Owner::empty(Rc::clone(api), api.composed_free);
        // SAFETY: fixed two-input counts and bounded own source survive the contained constructor.
        api.status(|| unsafe {
            (api.composed_new)(
                &raw mut owner.raw,
                session.0.stream.raw,
                3,
                counts.as_ptr(),
                2,
                logical,
                lanes,
                1,
                event_lanes,
                header.as_ptr(),
                body.as_ptr(),
            )
        })?;
        owner.require_live()?;
        Ok(Self {
            owner: Some(owner),
            session: session.clone(),
            scalar,
            input_count,
            count,
            domain,
            events,
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
            .ok_or_else(|| MlxError::Abi("missing SGD primitive".into()))?;
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
        validate(&self.session, records.raw, 3, self.events, 4)?;
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
            .ok_or_else(|| MlxError::Abi("nil SGD status backing".into()))?;
        let words = unsafe { std::slice::from_raw_parts(pointer.as_ptr(), self.events) };
        validate_records(words, self.domain)?;
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
impl Drop for StrictSgd {
    fn drop(&mut self) {
        if self.session.ensure_ready().is_err() {
            if let Some(owner) = self.owner.take() {
                std::mem::forget(owner);
            }
            std::mem::forget(self.session.clone());
        }
    }
}

pub fn validate_records(words: &[u32], domain: TensorStrictFaultDomain) -> Result<(), MlxError> {
    if u64::try_from(words.len()).ok() != Some(domain.event_extent()) {
        return Err(MlxError::Abi("strict event extent mismatch".into()));
    }
    for (ordinal, &word) in words.iter().enumerate() {
        if word == 0 {
            continue;
        }
        let kind = match word {
            1 => PcuExecutionFaultKind::ArithmeticOverflow,
            3 => PcuExecutionFaultKind::ArithmeticUnderflow,
            4 => PcuExecutionFaultKind::InvalidFloatingOperand,
            _ => return Err(MlxError::Abi("invalid strict event encoding".into())),
        };
        let fault = PcuExecutionFault {
            kind,
            invocation_id: u64::try_from(ordinal).map_err(|_| MlxError::InvalidExtent)?,
            recovered: false,
        };
        if !domain.accepts(fault) {
            return Err(MlxError::Abi(
                "strict event violates dependent arithmetic law".into(),
            ));
        }
    }
    Ok(())
}
fn header(scalar: PcuScalarType) -> &'static str {
    let wrapped = if scalar == PcuScalarType::F32 {
        include_str!("../../../native/cpp/checked_binary/shader/f32_binary.hpp")
    } else {
        include_str!("../../../native/cpp/checked_binary/shader/f64_binary.hpp")
    };
    wrapped
        .split_once("R\"PCUMLX(")
        .and_then(|(_, tail)| tail.split_once(")PCUMLX\";"))
        .map_or("", |(header, _)| header)
}
fn body(scalar: PcuScalarType, policy: Policy, count: usize, rate: u64) -> String {
    let policy = match policy {
        Policy::IeeeAfterRounding => 0,
        Policy::RejectSubnormalResult => 1,
        Policy::AllowGradualUnderflow => 2,
    };
    let wide = scalar == PcuScalarType::F64;
    let (carrier, rate, finite, multiply, subtract, arithmetic, invalid) = if wide {
        (
            "ulong",
            format!("0x{rate:x}ul"),
            "((value>>52)&2047ul)==2047ul",
            format!("multiply(gradient,rate,{policy}u)"),
            format!("add(weight,product.bits,true,{policy}u)"),
            String::new(),
            4,
        )
    } else {
        (
            "uint",
            format!("0x{rate:x}u"),
            "((value>>23)&255u)==255u",
            "arithmetic.multiply(gradient,rate)".into(),
            "arithmetic.add(weight,product.bits,true)".into(),
            format!("Binary arithmetic={{{policy}u,false}};"),
            1,
        )
    };
    let input = |name: &str| {
        if wide {
            format!("(ulong({name}[2u*id])|(ulong({name}[2u*id+1u])<<32))")
        } else {
            format!("{name}[id]")
        }
    };
    let gradient = input("input1");
    let weight = input("input0");
    let status = |name: &str| {
        if wide {
            format!("{name}.status")
        } else {
            format!("pcu_diagnostic({name}.status)")
        }
    };
    let output = if wide {
        "output0[2u*id]=uint(difference.bits);output0[2u*id+1u]=uint(difference.bits>>32);"
    } else {
        "output0[id]=difference.bits;"
    };
    format!(
        "uint id=thread_position_in_grid.x;if(id>={count}u)return;records[2u*id]=0u;records[2u*id+1u]=0u;{arithmetic}{carrier} gradient={gradient},rate={rate};Result product;{{{carrier} value=gradient;if({finite})product=fault({invalid});else product={multiply};}}records[2u*id]={product_status};if(records[2u*id]!=0u)return;{carrier} weight={weight};Result difference;{{{carrier} value=weight;if({finite})difference=fault({invalid});else difference={subtract};}}records[2u*id+1u]={difference_status};if(records[2u*id+1u]!=0u)return;{output}",
        product_status = status("product"),
        difference_status = status("difference")
    )
}
