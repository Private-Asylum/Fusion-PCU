//! Physical status decoding stays local; exact cold semantic masks come from the core contract.
#[rustfmt::skip]
use fusion_pcu::{
    PcuCheckedScalarFaultLaw,
    PcuExecutionFault,
    PcuExecutionFaultKind,
};
use crate::MlxError;
#[derive(Clone, Copy)]
pub enum Encoding {
    Scalar,
    DivRem,
}
/// Checks every record before priority arbitration or useful payload publication.
pub fn validate(
    records: &[u32],
    count: usize,
    law: PcuCheckedScalarFaultLaw,
    encoding: Encoding,
) -> Result<(), MlxError> {
    if records.len() != count {
        return Err(MlxError::Abi(
            "checked status logical extent mismatch".into(),
        ));
    }
    let extent = u64::try_from(count).map_err(|_| MlxError::InvalidExtent)?;
    for (index, &record) in records.iter().enumerate() {
        if record == 0 {
            continue;
        }
        let kind = match (encoding, record) {
            (Encoding::Scalar, 1 | 0x101) => PcuExecutionFaultKind::ArithmeticOverflow,
            (Encoding::Scalar, 2) | (Encoding::DivRem, 4) => PcuExecutionFaultKind::DivideByZero,
            (Encoding::Scalar, 3 | 0x103) => PcuExecutionFaultKind::ArithmeticUnderflow,
            (Encoding::Scalar, 4) => PcuExecutionFaultKind::InvalidFloatingOperand,
            (Encoding::DivRem, 1) => PcuExecutionFaultKind::SignedDivisionOverflow,
            _ => {
                return Err(MlxError::Abi(
                    "invalid checked physical status encoding".into(),
                ));
            }
        };
        let fault = PcuExecutionFault {
            kind,
            invocation_id: u64::try_from(index).map_err(|_| MlxError::InvalidExtent)?,
            recovered: record & 0x100 != 0,
        };
        if !law.accepts(fault, extent) {
            return Err(MlxError::Abi(
                "checked status violates admitted operation/policy".into(),
            ));
        }
    }
    Ok(())
}
#[cfg(test)]
#[path = "tests/tests.rs"]
mod tests;
