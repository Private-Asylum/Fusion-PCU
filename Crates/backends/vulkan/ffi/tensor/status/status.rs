//! Physical dense scalar/compound records retain distinct cold semantic contracts.
#[rustfmt::skip]
use fusion_pcu::{
    PcuCheckedScalarFaultLaw,
    PcuExecutionFaultKind,
    PcuFloatUnderflowPolicy,
    PcuRangePolicy,
    PcuScalarType,
    dialect::tensor::{
        TensorArithmeticStep,
        TensorStrictFaultDomain,
        TensorStrictFaultLocation,
    },
};
use fusion_pcu_spirv::PcuSpirvCompoundOperation;
#[rustfmt::skip]
use super::{
    StatusPolicy,
    VulkanCompoundFault,
    PcuVulkanError,
};

#[derive(Clone, Copy)]
pub enum TensorStatusPolicy {
    Scalar(StatusPolicy),
    Compound(CompoundStatusPolicy),
}
impl TensorStatusPolicy {
    pub(crate) fn scalar(law: PcuCheckedScalarFaultLaw) -> Result<Self, PcuVulkanError> {
        StatusPolicy::checked(Some(law), PcuRangePolicy::Reject).map(Self::Scalar)
    }
    pub(crate) fn compound(
        scalar: PcuScalarType,
        operation: PcuSpirvCompoundOperation,
        underflow: PcuFloatUnderflowPolicy,
    ) -> Result<Self, PcuVulkanError> {
        let domain = match operation {
            PcuSpirvCompoundOperation::Sgd { count, .. } => {
                TensorStrictFaultDomain::sgd(scalar, u64::from(count), underflow)
            }
            PcuSpirvCompoundOperation::MeanSquaredError { count } => {
                TensorStrictFaultDomain::mse(scalar, u64::from(count), underflow)
            }
            PcuSpirvCompoundOperation::MatMul {
                rows,
                inner,
                columns,
                ..
            } => TensorStrictFaultDomain::matmul(
                scalar,
                u64::from(rows) * u64::from(columns),
                u64::from(inner),
                underflow,
            ),
        }
        .ok_or(PcuVulkanError::UnsupportedPreparedProfile)?;
        Ok(Self::Compound(CompoundStatusPolicy { domain }))
    }
    pub(crate) const fn record_bytes(self) -> usize {
        match self {
            Self::Scalar(_) => 4,
            Self::Compound(_) => 12,
        }
    }
}

#[derive(Clone, Copy)]
pub struct CompoundStatusPolicy {
    domain: TensorStrictFaultDomain,
}
impl CompoundStatusPolicy {
    pub(crate) fn scan(
        self,
        extent: usize,
        mut read: impl FnMut(usize) -> Result<[u32; 3], PcuVulkanError>,
    ) -> Result<Option<VulkanCompoundFault>, PcuVulkanError> {
        let mut first = None;
        for element in 0..extent {
            let [code, reduction, step] = read(element)?;
            // Kind zero is the complete no-fault sentinel. Auxiliary fields have no fault
            // meaning; a nonzero kind requires every coordinate and class to be valid.
            if code == 0 {
                continue;
            }
            let element_index =
                u64::try_from(element).map_err(|_| PcuVulkanError::BufferTooLarge)?;
            let invalid = || PcuVulkanError::InvalidCompoundStatus {
                code,
                element_index,
                reduction_index: reduction,
                step,
            };
            let kind = match code {
                1 => PcuExecutionFaultKind::InvalidFloatingOperand,
                2 => PcuExecutionFaultKind::ArithmeticUnderflow,
                3 => PcuExecutionFaultKind::ArithmeticOverflow,
                4 => PcuExecutionFaultKind::DivideByZero,
                _ => return Err(invalid()),
            };
            let arithmetic_step = match step {
                0 => TensorArithmeticStep::Multiply,
                1 => TensorArithmeticStep::Add,
                2 => TensorArithmeticStep::Subtract,
                3 => TensorArithmeticStep::Divide,
                _ => return Err(invalid()),
            };
            let location = TensorStrictFaultLocation {
                element_index,
                reduction_index: u64::from(reduction),
                step: arithmetic_step,
            };
            if !self.domain.allows(location, kind, false) {
                return Err(invalid());
            }
            // Every later nonzero record is still validated before private graph output
            // can escape. Row-major records preserve the original ordered first fault.
            first.get_or_insert(VulkanCompoundFault {
                element_index: element,
                reduction_index: usize::try_from(reduction)
                    .map_err(|_| PcuVulkanError::BufferTooLarge)?,
                step: arithmetic_step,
                kind,
            });
        }
        Ok(first)
    }
}

#[cfg(test)]
#[path = "tests/tests.rs"]
mod tests;
