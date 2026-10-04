//! Ordered checked bounded nonempty mean squared error, with a compact original-event/status terminal receipt.
use fusion_pcu::{PcuScalarType, PcuFloatUnderflowPolicy};
use fusion_pcu::dialect::tensor::TensorStrictFaultDomain;
use super::{
    MetalBuffer, MetalError, MetalSession, ffi, validate_byte_extent, validate_extent, select_fault,
};
/// Backend-local Reject-only control. Source admission and discovery are separate scopes.
pub struct MetalPreparedStrictMse {
    session: MetalSession,
    pipeline: ffi::Pipeline,
    domain: TensorStrictFaultDomain,
    cells: usize,
    input_bytes: [usize; 2],
    output_bytes: usize,
    policy: u32,
}
impl MetalSession {
    /// Prepare a nonempty ordered squared-error reduction of at most 65535 elements.
    /// # Errors
    /// Refuses empty or physically unrepresentable shape, scalar or native compilation failure.
    pub fn prepare_strict_mse(
        &self,
        scalar: PcuScalarType,
        elements: usize,
        policy: PcuFloatUnderflowPolicy,
    ) -> Result<MetalPreparedStrictMse, MetalError> {
        self.ensure_quiescent()?;
        if !(1..=65535).contains(&elements) {
            return Err(MetalError::InvalidExtent);
        }
        let cells = 1;
        let width = match scalar {
            PcuScalarType::F32 => 4,
            PcuScalarType::F64 => 8,
            _ => return Err(MetalError::Unsupported),
        };
        let bytes = elements
            .checked_mul(width)
            .ok_or(MetalError::InvalidExtent)?;
        let input_bytes = [bytes; 2];
        let output_bytes = width;
        for bytes in [input_bytes[0], input_bytes[1], output_bytes] {
            validate_byte_extent(bytes, self.0.facts.max_buffer_bytes)?;
        }
        let domain = TensorStrictFaultDomain::mse(
            scalar,
            u64::try_from(elements).map_err(|_| MetalError::InvalidExtent)?,
            policy,
        )
        .ok_or(MetalError::InvalidExtent)?;
        u32::try_from(domain.event_extent()).map_err(|_| MetalError::InvalidExtent)?;
        validate_extent(2, self.0.facts.max_buffer_bytes)?;
        let source = source(scalar, elements);
        let pipeline = self.0.native.compile(&source, "pcu_strict_mse")?;
        let policy = match policy {
            PcuFloatUnderflowPolicy::IeeeAfterRounding => 0,
            PcuFloatUnderflowPolicy::RejectSubnormalResult => 1,
            PcuFloatUnderflowPolicy::AllowGradualUnderflow => 2,
        };
        Ok(MetalPreparedStrictMse {
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
impl MetalPreparedStrictMse {
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
        let bytes = validate_extent(2, self.session.0.facts.max_buffer_bytes)?;
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
        records.inspect_words(2, |records| validate_receipt(records, self.domain))?;
        Ok(output)
    }
}
fn validate_receipt(records: &[u32], domain: TensorStrictFaultDomain) -> Result<(), MetalError> {
    let &[ordinal, status] = records else {
        return Err(MetalError::Runtime("strict receipt extent mismatch".into()));
    };
    if status == 0 {
        return if ordinal == 0 {
            Ok(())
        } else {
            Err(MetalError::Runtime("noncanonical success receipt".into()))
        };
    }
    let mut fault = match select_fault(&[status]) {
        Err(MetalError::Arithmetic(fault)) if !fault.recovered => fault,
        _ => return Err(MetalError::Runtime("invalid strict receipt status".into())),
    };
    fault.invocation_id = u64::from(ordinal);
    if !domain.accepts(fault) {
        return Err(MetalError::Runtime(
            "strict receipt violates dependent arithmetic law".into(),
        ));
    }
    Err(MetalError::Arithmetic(fault))
}
#[cfg(test)]
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
fn source(scalar: PcuScalarType, elements: usize) -> String {
    let wide = scalar == PcuScalarType::F64;
    let header = if wide {
        include_str!("../../ffi/native/cpp/checked_f64/checked_f64.metal")
    } else {
        include_str!("../../ffi/native/cpp/checked_f32/checked_f32.metal")
    };
    let header = header.split("kernel void").next().unwrap_or(header);
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
            "add(left,right,true,parameters.y)",
            "multiply(difference.bits,difference.bits,parameters.y)",
            "add(accumulator,product.bits,false,parameters.y)",
            format!("divide(accumulator,{divisor},parameters.y)"),
            "",
            4,
        )
    } else {
        (
            "uint",
            "((left>>23)&255u)==255u || ((right>>23)&255u)==255u",
            "arithmetic.add(left,right,true)",
            "arithmetic.multiply(difference.bits,difference.bits)",
            "arithmetic.add(accumulator,product.bits,false)",
            format!("arithmetic.divide(accumulator,{divisor})"),
            "Binary arithmetic={parameters.y,false};",
            1,
        )
    };
    let status = |name: &str| {
        if wide {
            format!("{name}.status")
        } else {
            format!(
                "({name}.status==1?4u:{name}.status==2?3u:{name}.status==3?1u:{name}.status==4?2u:0u)"
            )
        }
    };
    format!(
        "{header}\nkernel void pcu_strict_mse(device const {carrier}* a [[buffer(0)]],device const {carrier}* b [[buffer(1)]],device {carrier}* output [[buffer(2)]],device uint* records [[buffer(3)]],constant uint2& parameters [[buffer(4)]],uint index [[thread_position_in_grid]]){{if(index!=0u)return;records[0]=0u;records[1]=0u;{arithmetic}{carrier} accumulator=0;for(uint k=0;k<{elements}u;k++){{{carrier} left=a[k],right=b[k];Result difference;if({finite})difference=fault({invalid});else difference={subtract};uint difference_status={difference_status};if(difference_status!=0u){{records[0]=k*3u;records[1]=difference_status;return;}}Result product={multiply};uint product_status={product_status};if(product_status!=0u){{records[0]=k*3u+1u;records[1]=product_status;return;}}Result sum={add};uint sum_status={sum_status};if(sum_status!=0u){{records[0]=k*3u+2u;records[1]=sum_status;return;}}accumulator=sum.bits;}}Result mean={divide};uint mean_status={mean_status};if(mean_status!=0u){{records[0]={mean_event}u;records[1]=mean_status;return;}}output[0]=mean.bits;}}",
        mean_event = elements * 3,
        difference_status = status("difference"),
        product_status = status("product"),
        sum_status = status("sum"),
        mean_status = status("mean")
    )
}

#[cfg(test)]
#[path = "tests/tests.rs"]
mod tests;
