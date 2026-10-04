//! Physical ordered-map records validate their exact frozen instruction law.
#[rustfmt::skip]
use fusion_pcu::{
    PcuCheckedScalarFaultLaw,
    PcuExecutionFault,
    PcuExecutionFaultKind,
};
use fusion_pcu_spirv::PcuSpirvComposedProfile;
use crate::PcuVulkanError;

pub(super) fn scan(
    profile: &PcuSpirvComposedProfile,
    read: impl FnMut(usize) -> (u32, u32),
) -> Result<Option<PcuExecutionFault>, PcuVulkanError> {
    scan_records(profile.extent(), |step| profile.fault_law(step), read)
}

fn scan_records(
    extent: u32,
    law: impl Fn(usize) -> Option<PcuCheckedScalarFaultLaw>,
    mut read: impl FnMut(usize) -> (u32, u32),
) -> Result<Option<PcuExecutionFault>, PcuVulkanError> {
    let mut fatal = None;
    let mut recovered = None;
    for lane in 0..extent {
        let (code, step) = read(lane as usize);
        let invalid = || PcuVulkanError::InvalidStatus {
            code,
            invocation_id: u64::from(lane),
        };
        if code == 0 {
            if step != 0 {
                return Err(invalid());
            }
            continue;
        }
        let kind = match code & !0x100 {
            1 => PcuExecutionFaultKind::InvalidFloatingOperand,
            2 => PcuExecutionFaultKind::ArithmeticUnderflow,
            3 => PcuExecutionFaultKind::ArithmeticOverflow,
            4 => PcuExecutionFaultKind::DivideByZero,
            _ => return Err(invalid()),
        };
        let fault = PcuExecutionFault {
            invocation_id: u64::from(lane),
            kind,
            recovered: code & 0x100 != 0,
        };
        if !law(step as usize).is_some_and(|law| law.accepts(fault, u64::from(extent))) {
            return Err(invalid());
        }
        // Check every record: an earlier valid fault cannot hide later protocol damage.
        if fault.recovered {
            recovered.get_or_insert(fault);
        } else {
            fatal.get_or_insert(fault);
        }
    }
    fatal.map_or(Ok(recovered), |fault| Err(PcuVulkanError::Fault(fault)))
}

#[cfg(test)]
#[path = "tests/tests.rs"]
mod tests;
