//! Direct-C ordered dot product with independently retained payload and compact original-event/status siblings per output cell.
use fusion_pcu::dialect::tensor::TensorStrictFaultDomain;
#[rustfmt::skip]
use std::{
    ffi::CString,
    ptr::NonNull,
    rc::Rc,
};
#[rustfmt::skip]
use fusion_pcu::{
    PcuFloatUnderflowPolicy as Policy,
    PcuScalarType,
    PcuExecutionFault,
};
#[rustfmt::skip]
use super::{
    abi::CComposed,
    abi::CArray,
    abi::Opaque,
    owner::Owner,
    session::Session,
    encoded::EncodedArray,
    encoded::carrier::Carrier,
    checked::validate,
    checked::available,
};
use crate::MlxError;
pub struct StrictMatMul {
    owner: Option<Owner<CComposed>>,
    session: Session,
    scalar: PcuScalarType,
    input_count: [usize; 2],
    count: usize,
    domain: TensorStrictFaultDomain,
    events: usize,
}
impl StrictMatMul {
    #[cfg(test)]
    pub fn validate_domain(words: &[u32], domain: TensorStrictFaultDomain) -> Result<(), MlxError> {
        receipt::validate(words, domain).map(|_| ())
    }
    pub fn prepare(
        session: &Session,
        scalar: PcuScalarType,
        policy: Policy,
        shape: [usize; 3],
    ) -> Result<Self, MlxError> {
        Self::prepare_with_receipt_fixture(session, scalar, policy, shape, None)
    }
    #[cfg(test)]
    pub fn prepare_receipt_fixture(
        session: &Session,
        scalar: PcuScalarType,
        policy: Policy,
        shape: [usize; 3],
        receipt: &[u32],
    ) -> Result<Self, MlxError> {
        Self::prepare_with_receipt_fixture(session, scalar, policy, shape, Some(receipt))
    }
    fn prepare_with_receipt_fixture(
        session: &Session,
        scalar: PcuScalarType,
        policy: Policy,
        shape: [usize; 3],
        receipt: Option<&[u32]>,
    ) -> Result<Self, MlxError> {
        session.ensure_ready()?;
        if !matches!(scalar, PcuScalarType::F32 | PcuScalarType::F64) {
            return Err(MlxError::UnsupportedScalar(scalar));
        }
        let [rows, inner, columns] = shape;
        if shape.contains(&0) || !(1..=65535).contains(&inner) {
            return Err(MlxError::InvalidExtent);
        }
        let count = rows.checked_mul(columns).ok_or(MlxError::InvalidExtent)?;
        let input_count = [
            rows.checked_mul(inner).ok_or(MlxError::InvalidExtent)?,
            inner.checked_mul(columns).ok_or(MlxError::InvalidExtent)?,
        ];
        let domain = TensorStrictFaultDomain::matmul(
            scalar,
            u64::try_from(count).map_err(|_| MlxError::InvalidExtent)?,
            u64::try_from(inner).map_err(|_| MlxError::InvalidExtent)?,
            policy,
        )
        .ok_or(MlxError::InvalidExtent)?;
        u32::try_from(domain.event_extent()).map_err(|_| MlxError::InvalidExtent)?;
        let events = count.checked_mul(2).ok_or(MlxError::InvalidExtent)?;
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
        let source = receipt.map_or_else(
            || Ok(body(scalar, policy, count, inner, columns)),
            |receipt| receipt::fixture_source(receipt, count, scalar),
        )?;
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
            events,
        })
    }
    pub const fn fault_domain(&self) -> TensorStrictFaultDomain {
        self.domain
    }
    pub const fn status_word_extent(&self) -> usize {
        self.events
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
            .ok_or_else(|| MlxError::Abi("nil backward status backing".into()))?;
        let words = unsafe { std::slice::from_raw_parts(pointer.as_ptr(), self.events) };
        let decoded = receipt::validate(words, self.domain);
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
impl Drop for StrictMatMul {
    fn drop(&mut self) {
        if self.session.ensure_ready().is_err() {
            if let Some(owner) = self.owner.take() {
                std::mem::forget(owner);
            }
            std::mem::forget(self.session.clone());
        }
    }
}

#[path = "receipt/receipt.rs"]
mod receipt;
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
fn body(
    scalar: PcuScalarType,
    policy: Policy,
    cells: usize,
    inner: usize,
    columns: usize,
) -> String {
    let policy = match policy {
        Policy::IeeeAfterRounding => 0,
        Policy::RejectSubnormalResult => 1,
        Policy::AllowGradualUnderflow => 2,
    };
    let wide = scalar == PcuScalarType::F64;
    let (carrier, finite, multiply, add, arithmetic, invalid) = if wide {
        (
            "ulong",
            "((left>>52)&2047ul)==2047ul || ((right>>52)&2047ul)==2047ul",
            format!("multiply(left,right,{policy}u)"),
            format!("add(accumulator,product.bits,false,{policy}u)"),
            String::new(),
            4,
        )
    } else {
        (
            "uint",
            "((left>>23)&255u)==255u || ((right>>23)&255u)==255u",
            "arithmetic.multiply(left,right)".into(),
            "arithmetic.add(accumulator,product.bits,false)".into(),
            format!("Binary arithmetic={{{policy}u,false}};"),
            1,
        )
    };
    let input = |name: &str, index: &str| {
        if wide {
            format!("(ulong({name}[2u*({index})]) | (ulong({name}[2u*({index})+1u])<<32))")
        } else {
            format!("{name}[{index}]")
        }
    };
    let left = input("input0", &format!("(id/{columns}u)*{inner}u+k"));
    let right = input("input1", &format!("k*{columns}u+id%{columns}u"));
    let status = |name: &str| {
        if wide {
            format!("{name}.status")
        } else {
            format!("pcu_diagnostic({name}.status)")
        }
    };
    let output = if wide {
        "output0[2u*id]=uint(accumulator);output0[2u*id+1u]=uint(accumulator>>32);"
    } else {
        "output0[id]=accumulator;"
    };
    format!(
        "uint id=thread_position_in_grid.x;if(id>={cells}u)return;uint base=id*{inner}u*2u;uint record=id*2u;records[record]=0u;records[record+1u]=0u;{arithmetic}{carrier} accumulator=0;for(uint k=0;k<{inner}u;k++){{{carrier} left={left},right={right};Result product;if({finite})product=fault({invalid});else product={multiply};uint product_status={product_status};if(product_status!=0u){{records[record]=base+k*2u;records[record+1u]=product_status;return;}}Result sum={add};uint sum_status={sum_status};if(sum_status!=0u){{records[record]=base+k*2u+1u;records[record+1u]=sum_status;return;}}accumulator=sum.bits;}}{output}",
        product_status = status("product"),
        sum_status = status("sum")
    )
}
