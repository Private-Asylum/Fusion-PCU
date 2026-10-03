//! Exact cold scalar fault law guards every physical lane before arbitration.
#[rustfmt::skip]
use fusion_pcu::{
    PcuCheckedScalarFaultLaw,
    PcuDispatchIntegerBinaryOp,
    PcuExecutionFault,
    PcuExecutionFaultKind,
    PcuRangePolicy,
    PcuScalarType,
};
#[rustfmt::skip]
use super::{
    MetalError,
    MetalIntegerOp,
};
#[derive(Clone, Copy)]
pub enum Encoding {
    Scalar,
    DivRem,
}
pub fn integer(
    scalar: PcuScalarType,
    op: MetalIntegerOp,
    range: PcuRangePolicy,
) -> Result<Option<PcuCheckedScalarFaultLaw>, MetalError> {
    let operation = match op {
        MetalIntegerOp::Identity => return Ok(None),
        MetalIntegerOp::Divide => {
            return PcuCheckedScalarFaultLaw::integer_div_rem(scalar)
                .map(Some)
                .ok_or(MetalError::Unsupported);
        }
        MetalIntegerOp::Add => PcuDispatchIntegerBinaryOp::Add,
        MetalIntegerOp::Subtract => PcuDispatchIntegerBinaryOp::Sub,
        MetalIntegerOp::Multiply => PcuDispatchIntegerBinaryOp::Mul,
    };
    PcuCheckedScalarFaultLaw::integer_binary(scalar, operation, range)
        .map(Some)
        .ok_or(MetalError::Unsupported)
}
pub fn validate(
    records: &[u32],
    count: usize,
    law: Option<PcuCheckedScalarFaultLaw>,
    encoding: Encoding,
) -> Result<(), MetalError> {
    if records.len() != count {
        return Err(MetalError::Runtime(
            "checked status logical extent mismatch".into(),
        ));
    }
    let extent = u64::try_from(count).map_err(|_| MetalError::InvalidExtent)?;
    for (index, &record) in records.iter().enumerate() {
        if record == 0 {
            continue;
        }
        let kind = match (encoding, record) {
            (Encoding::Scalar, 1 | 0x101) => PcuExecutionFaultKind::ArithmeticOverflow,
            (Encoding::Scalar | Encoding::DivRem, 2) => PcuExecutionFaultKind::DivideByZero,
            (Encoding::Scalar, 3 | 0x103) => PcuExecutionFaultKind::ArithmeticUnderflow,
            (Encoding::Scalar, 4) => PcuExecutionFaultKind::InvalidFloatingOperand,
            (Encoding::DivRem, 5) => PcuExecutionFaultKind::SignedDivisionOverflow,
            _ => {
                return Err(MetalError::Runtime(
                    "invalid checked physical status encoding".into(),
                ));
            }
        };
        let fault = PcuExecutionFault {
            kind,
            invocation_id: u64::try_from(index).map_err(|_| MetalError::InvalidExtent)?,
            recovered: record & 0x100 != 0,
        };
        if law.is_none_or(|law| !law.accepts(fault, extent)) {
            return Err(MetalError::Runtime(
                "checked status violates admitted operation/policy".into(),
            ));
        }
    }
    Ok(())
}
#[cfg(test)]
#[path = "tests/tests.rs"]
mod tests;
