//! Dense physical records retain an exact cold scalar law before whole-call arbitration.
#[rustfmt::skip]
use fusion_pcu::{
    PcuCheckedScalarFaultLaw,
    PcuExecutionFault,
    PcuExecutionFaultKind,
    PcuRangePolicy,
};
use crate::PcuVulkanError;

#[derive(Clone, Copy)]
pub struct StatusPolicy {
    law: Option<PcuCheckedScalarFaultLaw>,
    recover_range: bool,
}
impl StatusPolicy {
    /// Transport has no arithmetic fault law: every physical status must be zero.
    pub(crate) const ZERO_ONLY: Self = Self {
        law: None,
        recover_range: false,
    };
    pub(crate) const fn checked(
        law: Option<PcuCheckedScalarFaultLaw>,
        range: PcuRangePolicy,
    ) -> Result<Self, PcuVulkanError> {
        let Some(law) = law else {
            return Err(PcuVulkanError::UnsupportedPreparedProfile);
        };
        Ok(Self {
            law: Some(law),
            recover_range: matches!(range, PcuRangePolicy::Clamp),
        })
    }
    pub(crate) fn scan(
        self,
        extent: usize,
        mut read: impl FnMut(usize) -> Result<u32, PcuVulkanError>,
    ) -> Result<Option<PcuExecutionFault>, PcuVulkanError> {
        let mut first_fatal = None;
        let mut first_recovered = None;
        let logical_extent = u64::try_from(extent).map_err(|_| PcuVulkanError::BufferTooLarge)?;
        for invocation in 0..extent {
            let code = read(invocation)?;
            if code == 0 {
                continue;
            }
            let invocation_id =
                u64::try_from(invocation).map_err(|_| PcuVulkanError::BufferTooLarge)?;
            let malformed = || PcuVulkanError::InvalidStatus {
                code,
                invocation_id,
            };
            let kind = match code {
                1 => PcuExecutionFaultKind::InvalidFloatingOperand,
                2 => PcuExecutionFaultKind::ArithmeticUnderflow,
                3 => PcuExecutionFaultKind::ArithmeticOverflow,
                4 => PcuExecutionFaultKind::DivideByZero,
                5 => PcuExecutionFaultKind::SignedDivisionOverflow,
                _ => return Err(malformed()),
            };
            let fault = PcuExecutionFault {
                kind,
                invocation_id,
                recovered: self.recover_range && matches!(code, 2 | 3),
            };
            if !self
                .law
                .is_some_and(|law| law.accepts(fault, logical_extent))
            {
                return Err(malformed());
            }
            // A valid early fatal cannot conceal a later impossible class or raw code.
            // Publication starts only after this entire dense logical span has been checked.
            if fault.recovered {
                first_recovered.get_or_insert(fault);
            } else {
                first_fatal.get_or_insert(fault);
            }
        }
        first_fatal.map_or(Ok(first_recovered), |fault| {
            Err(PcuVulkanError::Fault(fault))
        })
    }
}

#[cfg(test)]
#[path = "tests/tests.rs"]
mod tests;
