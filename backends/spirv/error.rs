//! Error vocabulary for SPIR-V lowering.

use core::fmt;

use crate::{
    PcuDispatchOpCaps,
    PcuValueType,
};
use super::types::PcuSpirvCapability;

/// Failure returned while lowering PCU IR into SPIR-V words.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PcuSpirvError {
    UnsupportedInstruction(PcuDispatchOpCaps),
    UnsupportedValueType(PcuValueType),
    UnsupportedCapability(PcuSpirvCapability),
    InvalidKernelSignature,
    InvalidBinding,
    SinkFull,
    IdSpaceExhausted,
}

impl fmt::Display for PcuSpirvError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::UnsupportedInstruction(flags) => {
                write!(
                    f,
                    "unsupported SPIR-V lowering instruction caps 0x{:x}",
                    flags.bits()
                )
            }
            Self::UnsupportedValueType(value_type) => {
                write!(f, "unsupported SPIR-V lowering value type {value_type:?}")
            }
            Self::UnsupportedCapability(capability) => {
                write!(
                    f,
                    "SPIR-V lowering capability {capability:?} is not enabled"
                )
            }
            Self::InvalidKernelSignature => f.write_str("invalid PCU kernel signature for SPIR-V"),
            Self::InvalidBinding => f.write_str("invalid PCU binding for SPIR-V"),
            Self::SinkFull => f.write_str("SPIR-V output sink is full"),
            Self::IdSpaceExhausted => f.write_str("SPIR-V id space exhausted"),
        }
    }
}
