//! Ordered nonempty SGD updates, with one status lane per arithmetic event.
use fusion_pcu::{PcuScalarType, PcuFloatUnderflowPolicy};
use fusion_pcu::dialect::tensor::TensorStrictFaultDomain;
use super::{
    MetalBuffer, MetalError, MetalSession, ffi, validate_byte_extent, validate_extent, select_fault,
};
/// Backend-local Reject-only control. Source admission and discovery are separate scopes.
pub struct MetalPreparedStrictSgd {
    session: MetalSession,
    pipeline: ffi::Pipeline,
    domain: TensorStrictFaultDomain,
    cells: usize,
    input_bytes: [usize; 2],
    output_bytes: usize,
    policy: u32,
}
impl MetalSession {
    /// Prepare ordered gradient-times-finite-rate then weight-minus-product updates.
    /// # Errors
    /// Refuses empty or unrepresentable extents, nonfinite rates, scalars or compiler failure.
    pub fn prepare_strict_sgd(
        &self,
        scalar: PcuScalarType,
        count: usize,
        learning_rate: f32,
        policy: PcuFloatUnderflowPolicy,
    ) -> Result<MetalPreparedStrictSgd, MetalError> {
        self.ensure_quiescent()?;
        if !learning_rate.is_finite() {
            return Err(MetalError::InvalidExtent);
        }
        let cells = count;
        let width = match scalar {
            PcuScalarType::F32 => 4,
            PcuScalarType::F64 => 8,
            _ => return Err(MetalError::Unsupported),
        };
        let bytes = count.checked_mul(width).ok_or(MetalError::InvalidExtent)?;
        let input_bytes = [bytes; 2];
        let output_bytes = cells.checked_mul(width).ok_or(MetalError::InvalidExtent)?;
        for bytes in [input_bytes[0], input_bytes[1], output_bytes] {
            validate_byte_extent(bytes, self.0.facts.max_buffer_bytes)?;
        }
        let domain = TensorStrictFaultDomain::sgd(
            scalar,
            u64::try_from(count).map_err(|_| MetalError::InvalidExtent)?,
            policy,
        )
        .ok_or(MetalError::InvalidExtent)?;
        validate_extent(
            usize::try_from(domain.event_extent()).map_err(|_| MetalError::InvalidExtent)?,
            self.0.facts.max_buffer_bytes,
        )?;
        let rate_bits = if scalar == PcuScalarType::F32 {
            u64::from(learning_rate.to_bits())
        } else {
            fusion_pcu::PcuCheckedFloatWidening::pcu_checked_to_f64(learning_rate)
                .map_err(|_| MetalError::InvalidExtent)?
                .to_bits()
        };
        let source = source(scalar, rate_bits);
        let pipeline = self.0.native.compile(&source, "pcu_strict_sgd")?;
        let policy = match policy {
            PcuFloatUnderflowPolicy::IeeeAfterRounding => 0,
            PcuFloatUnderflowPolicy::RejectSubnormalResult => 1,
            PcuFloatUnderflowPolicy::AllowGradualUnderflow => 2,
        };
        Ok(MetalPreparedStrictSgd {
            session: self.clone(),
            pipeline,
            domain,
            cells,
            input_bytes,
            output_bytes,
            policy,
        })
    }
}
impl MetalPreparedStrictSgd {
    /// Retained semantic event domain, independent of output allocation/launch extent.
    #[must_use]
    pub const fn fault_domain(&self) -> TensorStrictFaultDomain {
        self.domain
    }
    /// Publish one fresh private output only after all arithmetic event records validate.
    /// # Errors
    /// Refuses foreign/short operands or any native, invalid-record or arithmetic failure.
    pub fn execute_completed(
        &self,
        left: &MetalBuffer,
        right: &MetalBuffer,
    ) -> Result<MetalBuffer, MetalError> {
        self.session.ensure_quiescent()?;
        for (buffer, bytes) in [left, right].into_iter().zip(self.input_bytes) {
            if !self.session.same_session(buffer.session()) {
                return Err(MetalError::ForeignSession);
            }
            if buffer.byte_len() < bytes {
                return Err(MetalError::InvalidExtent);
            }
        }
        let output = self.session.allocate_zeroed_bytes(self.output_bytes)?;
        let events =
            usize::try_from(self.domain.event_extent()).map_err(|_| MetalError::InvalidExtent)?;
        let bytes = validate_extent(events, self.session.0.facts.max_buffer_bytes)?;
        let records = self.session.0.native.allocate(bytes)?;
        records.fill_ones();
        self.session.0.native.execute(
            &self.pipeline,
            [&left.native, &right.native, &output.native, &records],
            [
                u32::try_from(self.cells).map_err(|_| MetalError::InvalidExtent)?,
                self.policy,
            ],
            self.cells,
        )?;
        records.inspect_words(events, |records| validate(records, self.domain))?;
        Ok(output)
    }
}
fn validate(records: &[u32], domain: TensorStrictFaultDomain) -> Result<(), MetalError> {
    if u64::try_from(records.len()).ok() != Some(domain.event_extent()) {
        return Err(MetalError::Runtime("strict event extent mismatch".into()));
    }
    for (ordinal, &record) in records.iter().enumerate() {
        if record == 0 {
            continue;
        }
        let fault = match select_fault(&[record]) {
            Err(MetalError::Arithmetic(mut fault)) => {
                fault.invocation_id =
                    u64::try_from(ordinal).map_err(|_| MetalError::InvalidExtent)?;
                fault
            }
            _ => return Err(MetalError::Runtime("invalid strict event encoding".into())),
        };
        if !domain.accepts(fault) {
            return Err(MetalError::Runtime(
                "strict event violates dependent arithmetic law".into(),
            ));
        }
    }
    select_fault(records)
}
fn source(scalar: PcuScalarType, rate: u64) -> String {
    let (header, carrier, rate, finite, multiply, subtract, status, arithmetic, invalid) = if scalar
        == PcuScalarType::F32
    {
        (
            include_str!("../../ffi/native/cpp/checked_f32/checked_f32.metal"),
            "uint",
            format!("0x{rate:x}u"),
            "((value>>23)&255u)==255u",
            "arithmetic.multiply(gradient,rate)",
            "arithmetic.add(weight,product.bits,true)",
            "uint kind=result.status;uint status=kind==1?4u:kind==2?3u:kind==3?1u:kind==4?2u:0u;",
            "Binary arithmetic={parameters.y,false};",
            1,
        )
    } else {
        (
            include_str!("../../ffi/native/cpp/checked_f64/checked_f64.metal"),
            "ulong",
            format!("0x{rate:x}ul"),
            "((value>>52)&2047ul)==2047ul",
            "multiply(gradient,rate,parameters.y)",
            "add(weight,product.bits,true,parameters.y)",
            "uint status=result.status;",
            "",
            4,
        )
    };
    let header = header.split("kernel void").next().unwrap_or(header);
    let normalize = |name: &str| status.replace("result.status", &format!("{name}.status"));
    format!(
        "{header}\nkernel void pcu_strict_sgd(device const {carrier}* weights [[buffer(0)]],device const {carrier}* gradients [[buffer(1)]],device {carrier}* output [[buffer(2)]],device uint* records [[buffer(3)]],constant uint2& parameters [[buffer(4)]],uint index [[thread_position_in_grid]]){{if(index>=parameters.x)return;records[2u*index]=0u;records[2u*index+1u]=0u;{arithmetic}{carrier} gradient=gradients[index],rate={rate};Result product;{{{carrier} value=gradient;if({finite})product=fault({invalid});else product={multiply};}} {{{product_status}records[2u*index]=status;if(status!=0u)return;}}{carrier} weight=weights[index];Result difference;{{{carrier} value=weight;if({finite})difference=fault({invalid});else difference={subtract};}} {{{difference_status}records[2u*index+1u]=status;if(status!=0u)return;}}output[index]=difference.bits;}}",
        product_status = normalize("product"),
        difference_status = normalize("difference")
    )
}
#[cfg(test)]
#[path = "tests/tests.rs"]
mod tests;
