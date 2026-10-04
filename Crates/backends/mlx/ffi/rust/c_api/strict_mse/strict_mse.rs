//! Direct-C ordered bounded squared-error reduction with independently retained payload and compact original-event/status siblings.
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
pub struct StrictMse {
    owner: Option<Owner<CComposed>>,
    session: Session,
    scalar: PcuScalarType,
    input_count: [usize; 2],
    count: usize,
    domain: TensorStrictFaultDomain,
}
impl StrictMse {
    #[cfg(test)]
    pub fn validate_domain(words: &[u32], domain: TensorStrictFaultDomain) -> Result<(), MlxError> {
        validate_records(words, domain)
    }
    pub fn prepare(
        session: &Session,
        scalar: PcuScalarType,
        policy: Policy,
        elements: usize,
    ) -> Result<Self, MlxError> {
        Self::prepare_with_receipt_fixture(session, scalar, policy, elements, None)
    }
    #[cfg(test)]
    pub fn prepare_receipt_fixture(
        session: &Session,
        scalar: PcuScalarType,
        policy: Policy,
        elements: usize,
        receipt: [u32; 2],
    ) -> Result<Self, MlxError> {
        Self::prepare_with_receipt_fixture(session, scalar, policy, elements, Some(receipt))
    }
    #[cfg(test)]
    pub fn validate_receipt(
        words: &[u32],
        domain: TensorStrictFaultDomain,
    ) -> Result<Option<PcuExecutionFault>, MlxError> {
        validate_receipt(words, domain)
    }
    fn prepare_with_receipt_fixture(
        session: &Session,
        scalar: PcuScalarType,
        policy: Policy,
        elements: usize,
        receipt: Option<[u32; 2]>,
    ) -> Result<Self, MlxError> {
        session.ensure_ready()?;
        if !matches!(scalar, PcuScalarType::F32 | PcuScalarType::F64) {
            return Err(MlxError::UnsupportedScalar(scalar));
        }
        if !(1..=65535).contains(&elements) {
            return Err(MlxError::InvalidExtent);
        }
        let count = 1;
        let input_count = [elements; 2];
        let domain = TensorStrictFaultDomain::mse(
            scalar,
            u64::try_from(elements).map_err(|_| MlxError::InvalidExtent)?,
            policy,
        )
        .ok_or(MlxError::InvalidExtent)?;
        let events = usize::try_from(domain.event_extent()).map_err(|_| MlxError::InvalidExtent)?;
        u32::try_from(events).map_err(|_| MlxError::InvalidExtent)?;
        Carrier::assess(PcuScalarType::U32, 2)?;
        let input = input_count.map(|count| Carrier::assess(scalar, count));
        let [left, right] = input;
        let left = left?;
        let right = right?;
        let output = Carrier::assess(scalar, count)?;
        let counts = [
            u32::try_from(left.count).map_err(|_| MlxError::InvalidExtent)?,
            u32::try_from(right.count).map_err(|_| MlxError::InvalidExtent)?,
        ];
        let event_lanes = 2;
        let logical = u32::try_from(count).map_err(|_| MlxError::InvalidExtent)?;
        let lanes = u32::try_from(output.count).map_err(|_| MlxError::InvalidExtent)?;
        let header = CString::new(header(scalar)).map_err(|_| MlxError::InvalidExtent)?;
        let source = receipt.map_or_else(
            || body(scalar, policy, elements),
            |[ordinal, status]| format!("uint id=thread_position_in_grid.x;if(id!=0u)return;records[0]={ordinal}u;records[1]={status}u;output0[0]=0u;{}", if scalar == PcuScalarType::F64 { "output0[1]=0u;" } else { "" }),
        );
        let body = CString::new(source).map_err(|_| MlxError::InvalidExtent)?;
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
            .ok_or_else(|| MlxError::Abi("missing MSE primitive".into()))?;
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
        validate(&self.session, records.raw, 3, 2, 4)?;
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
            .ok_or_else(|| MlxError::Abi("nil MSE status backing".into()))?;
        let words = unsafe { std::slice::from_raw_parts(pointer.as_ptr(), 2) };
        let decoded = validate_receipt(words, self.domain);
        let (input, payload, records, _session) = pending;
        for owner in input {
            owner.release()?;
        }
        records.release()?;
        let notice = match decoded {
            Ok(notice) => notice,
            Err(error) => {
                payload.release()?;
                return Err(error);
            }
        };
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
impl Drop for StrictMse {
    fn drop(&mut self) {
        if self.session.ensure_ready().is_err() {
            if let Some(owner) = self.owner.take() {
                std::mem::forget(owner);
            }
            std::mem::forget(self.session.clone());
        }
    }
}

fn validate_receipt(
    words: &[u32],
    domain: TensorStrictFaultDomain,
) -> Result<Option<PcuExecutionFault>, MlxError> {
    let &[ordinal, status] = words else {
        return Err(MlxError::Abi("strict receipt extent mismatch".into()));
    };
    if status == 0 {
        return if ordinal == 0 {
            Ok(None)
        } else {
            Err(MlxError::Abi("noncanonical success receipt".into()))
        };
    }
    let kind = match status {
        1 => PcuExecutionFaultKind::ArithmeticOverflow,
        3 => PcuExecutionFaultKind::ArithmeticUnderflow,
        4 => PcuExecutionFaultKind::InvalidFloatingOperand,
        _ => return Err(MlxError::Abi("invalid strict receipt status".into())),
    };
    let fault = PcuExecutionFault {
        invocation_id: u64::from(ordinal),
        recovered: false,
        kind,
    };
    if !domain.accepts(fault) {
        return Err(MlxError::Abi(
            "strict receipt violates dependent arithmetic law".into(),
        ));
    }
    Ok(Some(fault))
}
#[cfg(test)]
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
fn body(scalar: PcuScalarType, policy: Policy, elements: usize) -> String {
    let policy = match policy {
        Policy::IeeeAfterRounding => 0,
        Policy::RejectSubnormalResult => 1,
        Policy::AllowGradualUnderflow => 2,
    };
    let wide = scalar == PcuScalarType::F64;
    let count = u16::try_from(elements).expect("cold bounded element metadata");
    let divisor = if wide {
        format!("{}ul", f64::from(count).to_bits())
    } else {
        format!("{}u", f32::from(count).to_bits())
    };
    let (carrier, finite, subtract, multiply, add, divide, arithmetic, invalid) = if wide {
        (
            "ulong",
            "((left>>52)&2047ul)==2047ul || ((right>>52)&2047ul)==2047ul",
            format!("add(left,right,true,{policy}u)"),
            format!("multiply(difference.bits,difference.bits,{policy}u)"),
            format!("add(accumulator,product.bits,false,{policy}u)"),
            format!("divide(accumulator,{divisor},{policy}u)"),
            String::new(),
            4,
        )
    } else {
        (
            "uint",
            "((left>>23)&255u)==255u || ((right>>23)&255u)==255u",
            "arithmetic.add(left,right,true)".into(),
            "arithmetic.multiply(difference.bits,difference.bits)".into(),
            "arithmetic.add(accumulator,product.bits,false)".into(),
            format!("arithmetic.divide(accumulator,{divisor})"),
            format!("Binary arithmetic={{{policy}u,false}};"),
            1,
        )
    };
    let input = |name: &str| {
        if wide {
            format!("ulong({name}[2u*k])|(ulong({name}[2u*k+1u])<<32)")
        } else {
            format!("{name}[k]")
        }
    };
    let status = |name: &str| {
        if wide {
            format!("{name}.status")
        } else {
            format!("pcu_diagnostic({name}.status)")
        }
    };
    let output = if wide {
        "output0[0]=uint(mean.bits);output0[1]=uint(mean.bits>>32);"
    } else {
        "output0[0]=mean.bits;"
    };
    format!(
        "uint id=thread_position_in_grid.x;if(id!=0u)return;records[0]=0u;records[1]=0u;{arithmetic}{carrier} accumulator=0;for(uint k=0;k<{elements}u;k++){{{carrier} left={left},right={right};Result difference;if({finite})difference=fault({invalid});else difference={subtract};uint difference_status={difference_status};if(difference_status!=0u){{records[0]=k*3u;records[1]=difference_status;return;}}Result product={multiply};uint product_status={product_status};if(product_status!=0u){{records[0]=k*3u+1u;records[1]=product_status;return;}}Result sum={add};uint sum_status={sum_status};if(sum_status!=0u){{records[0]=k*3u+2u;records[1]=sum_status;return;}}accumulator=sum.bits;}}Result mean={divide};uint mean_status={mean_status};if(mean_status!=0u){{records[0]={mean_event}u;records[1]=mean_status;return;}}{output}",
        mean_event = elements * 3,
        left = input("input0"),
        right = input("input1"),
        difference_status = status("difference"),
        product_status = status("product"),
        sum_status = status("sum"),
        mean_status = status("mean")
    )
}
