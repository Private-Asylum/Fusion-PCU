//! Canonical per-cell compact ordered `MatMul` receipts; no arithmetic-domain narrowing.
#[rustfmt::skip]
use fusion_pcu::{
    PcuExecutionFault,
    PcuExecutionFaultKind,
};
#[rustfmt::skip]
use fusion_pcu::dialect::tensor::{
    TensorArithmeticStep,
    TensorStrictFaultDomain,
};
use crate::MlxError;

pub(super) fn validate(
    words: &[u32],
    domain: TensorStrictFaultDomain,
) -> Result<Option<PcuExecutionFault>, MlxError> {
    let last = domain
        .event_extent()
        .checked_sub(1)
        .and_then(|ordinal| domain.location(ordinal))
        .filter(|location| location.step == TensorArithmeticStep::Add)
        .ok_or_else(|| MlxError::Abi("invalid compact MatMul domain".into()))?;
    let cells = last
        .element_index
        .checked_add(1)
        .and_then(|cells| usize::try_from(cells).ok())
        .ok_or(MlxError::InvalidExtent)?;
    if cells.checked_mul(2) != Some(words.len()) {
        return Err(MlxError::Abi(
            "compact MatMul receipt extent mismatch".into(),
        ));
    }
    let mut selected = None;
    for (cell, receipt) in words.as_chunks::<2>().0.iter().enumerate() {
        let [ordinal, status] = [receipt[0], receipt[1]];
        if status == 0 {
            if ordinal != 0 {
                return Err(MlxError::Abi("noncanonical compact MatMul success".into()));
            }
            continue;
        }
        let kind = match status {
            1 => PcuExecutionFaultKind::ArithmeticOverflow,
            3 => PcuExecutionFaultKind::ArithmeticUnderflow,
            4 => PcuExecutionFaultKind::InvalidFloatingOperand,
            _ => {
                return Err(MlxError::Abi(
                    "invalid compact MatMul fault encoding".into(),
                ));
            }
        };
        let fault = PcuExecutionFault {
            invocation_id: u64::from(ordinal),
            kind,
            recovered: false,
        };
        if !domain.accepts(fault)
            || domain
                .location(fault.invocation_id)
                .map(|location| location.element_index)
                != u64::try_from(cell).ok()
        {
            return Err(MlxError::Abi(
                "compact MatMul fault violates cell/domain law".into(),
            ));
        }
        selected.get_or_insert(fault);
    }
    Ok(selected)
}

// Only test fixtures inject receipts; production always supplies None.
pub(super) fn fixture_source(
    receipt: &[u32],
    cells: usize,
    scalar: fusion_pcu::PcuScalarType,
) -> Result<String, MlxError> {
    if cells.checked_mul(2) != Some(receipt.len()) {
        return Err(MlxError::InvalidExtent);
    }
    let words = receipt
        .iter()
        .map(|word| format!("{word}u"))
        .collect::<Vec<_>>()
        .join(",");
    Ok(format!(
        "uint id=thread_position_in_grid.x;if(id>={cells}u)return;const uint receipt[]={{{words}}};records[2u*id]=receipt[2u*id];records[2u*id+1u]=receipt[2u*id+1u];{}",
        if scalar == fusion_pcu::PcuScalarType::F64 {
            "output0[2u*id]=0u;output0[2u*id+1u]=0u;"
        } else {
            "output0[id]=0u;"
        }
    ))
}
