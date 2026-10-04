//! Ordered checked nonempty 2D dot products, with one compact original-event/status receipt per output cell.
#[rustfmt::skip]
use fusion_pcu::{
    PcuScalarType,
    PcuFloatUnderflowPolicy,
};
use fusion_pcu::dialect::tensor::TensorStrictFaultDomain;
#[rustfmt::skip]
use super::{
    MetalBuffer,
    MetalError,
    MetalSession,
    ffi,
    validate_byte_extent,
    validate_extent,
};
/// Backend-local Reject-only control. Source admission and discovery are separate scopes.
pub struct MetalPreparedStrictMatMul {
    session: MetalSession,
    pipeline: ffi::Pipeline,
    domain: TensorStrictFaultDomain,
    cells: usize,
    input_bytes: [usize; 2],
    output_bytes: usize,
    policy: u32,
}
impl MetalSession {
    /// Prepare a nonempty row-major ordered multiply/add dot product.
    /// # Errors
    /// Refuses empty or physically unrepresentable shape, scalar or native compilation failure.
    pub fn prepare_strict_matmul(
        &self,
        scalar: PcuScalarType,
        shape: [usize; 3],
        policy: PcuFloatUnderflowPolicy,
    ) -> Result<MetalPreparedStrictMatMul, MetalError> {
        self.ensure_quiescent()?;
        let [rows, inner, columns] = shape;
        if shape.contains(&0) || !(1..=65535).contains(&inner) {
            return Err(MetalError::InvalidExtent);
        }
        let cells = rows.checked_mul(columns).ok_or(MetalError::InvalidExtent)?;
        let width = match scalar {
            PcuScalarType::F32 => 4,
            PcuScalarType::F64 => 8,
            _ => return Err(MetalError::Unsupported),
        };
        let input_bytes = [rows.checked_mul(inner), inner.checked_mul(columns)].map(|count| {
            count
                .and_then(|count| count.checked_mul(width))
                .ok_or(MetalError::InvalidExtent)
        });
        let input_bytes = [input_bytes[0].clone()?, input_bytes[1].clone()?];
        let output_bytes = cells.checked_mul(width).ok_or(MetalError::InvalidExtent)?;
        for bytes in [input_bytes[0], input_bytes[1], output_bytes] {
            validate_byte_extent(bytes, self.0.facts.max_buffer_bytes)?;
        }
        let domain = TensorStrictFaultDomain::matmul(
            scalar,
            u64::try_from(cells).map_err(|_| MetalError::InvalidExtent)?,
            u64::try_from(inner).map_err(|_| MetalError::InvalidExtent)?,
            policy,
        )
        .ok_or(MetalError::InvalidExtent)?;
        u32::try_from(domain.event_extent()).map_err(|_| MetalError::InvalidExtent)?;
        validate_extent(
            cells.checked_mul(2).ok_or(MetalError::InvalidExtent)?,
            self.0.facts.max_buffer_bytes,
        )?;
        let source = source(scalar, inner, columns);
        let pipeline = self.0.native.compile(&source, "pcu_strict_matmul")?;
        let policy = match policy {
            PcuFloatUnderflowPolicy::IeeeAfterRounding => 0,
            PcuFloatUnderflowPolicy::RejectSubnormalResult => 1,
            PcuFloatUnderflowPolicy::AllowGradualUnderflow => 2,
        };
        Ok(MetalPreparedStrictMatMul {
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
impl MetalPreparedStrictMatMul {
    /// Retained semantic event domain, independent of output allocation/launch extent.
    #[must_use]
    pub const fn fault_domain(&self) -> TensorStrictFaultDomain {
        self.domain
    }
    /// Actual compact native status word extent, independent of the semantic event domain.
    #[must_use]
    pub const fn status_word_extent(&self) -> usize {
        self.cells * 2
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
        let events = self.status_word_extent();
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
#[path = "receipt/receipt.rs"]
mod receipt;
fn validate(records: &[u32], domain: TensorStrictFaultDomain) -> Result<(), MetalError> {
    receipt::validate(records, domain)?.map_or(Ok(()), |fault| Err(MetalError::Arithmetic(fault)))
}
fn source(scalar: PcuScalarType, inner: usize, columns: usize) -> String {
    let (header, carrier, finite, multiply, add, status) = if scalar == PcuScalarType::F32 {
        (
            include_str!("../../ffi/native/cpp/checked_f32/checked_f32.metal"),
            "uint",
            "((left>>23)&255u)==255u || ((right>>23)&255u)==255u",
            "arithmetic.multiply(left,right)",
            "arithmetic.add(accumulator,product.bits,false)",
            "uint kind=result.status; uint status=kind==1?4u:kind==2?3u:kind==3?1u:kind==4?2u:0u;",
        )
    } else {
        (
            include_str!("../../ffi/native/cpp/checked_f64/checked_f64.metal"),
            "ulong",
            "((left>>52)&2047ul)==2047ul || ((right>>52)&2047ul)==2047ul",
            "multiply(left,right,parameters.y)",
            "add(accumulator,product.bits,false,parameters.y)",
            "uint status=result.status;",
        )
    };
    let header = header.split("kernel void").next().unwrap_or(header);
    let arithmetic = if scalar == PcuScalarType::F32 {
        "Binary arithmetic={parameters.y,false};"
    } else {
        ""
    };
    let normalize = |name: &str| status.replace("result.status", &format!("{name}.status"));
    format!(
        "{header}\nkernel void pcu_strict_matmul(device const {carrier}* a [[buffer(0)]], device const {carrier}* b [[buffer(1)]], device {carrier}* output [[buffer(2)]], device uint* records [[buffer(3)]], constant uint2& parameters [[buffer(4)]], uint index [[thread_position_in_grid]]) {{\n if(index>=parameters.x) return; uint base=index*{inner}u*2u; uint record=index*2u; records[record]=0u;records[record+1u]=0u; {arithmetic} {carrier} accumulator=0;\n for(uint k=0;k<{inner}u;k++) {{ {carrier} left=a[(index/{columns}u)*{inner}u+k], right=b[k*{columns}u+index%{columns}u]; Result product; if({finite}) product=fault({invalid}); else product={multiply}; {{ {product_status} if(status!=0u) {{records[record]=base+k*2u;records[record+1u]=status;return;}} }} Result sum={add}; {{ {sum_status} if(status!=0u) {{records[record]=base+k*2u+1u;records[record+1u]=status;return;}} }} accumulator=sum.bits; }} output[index]=accumulator; }}",
        invalid = if scalar == PcuScalarType::F32 { 1 } else { 4 },
        product_status = normalize("product"),
        sum_status = normalize("sum")
    )
}
#[cfg(test)]
#[path = "tests/tests.rs"]
mod tests;
