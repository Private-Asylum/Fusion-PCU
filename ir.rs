//! Shared PCU IR support vocabulary.
//!
//! This module owns the low-level IR nouns that are genuinely shared across model families:
//! - generic value ops
//! - ALU ops
//! - generic control ops
//! - binding/port/sync ops
//! - sample ops over binding truth

use crate::{
    PcuBinding,
    PcuBindingRef,
    PcuDispatchOpCaps,
    PcuValueType,
};
use crate::validation::validate_sample_op;
pub use crate::validation::PcuSampleValidationError;

/// Value-construction or representation-changing operation families.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PcuValueOp {
    Constant,
    Cast,
    Pack,
    Unpack,
    Swizzle,
}

impl PcuValueOp {
    #[must_use]
    pub const fn support_flag(self) -> PcuDispatchOpCaps {
        match self {
            Self::Constant => PcuDispatchOpCaps::VALUE_CONSTANT,
            Self::Cast => PcuDispatchOpCaps::VALUE_CAST,
            Self::Pack => PcuDispatchOpCaps::VALUE_PACK,
            Self::Unpack => PcuDispatchOpCaps::VALUE_UNPACK,
            Self::Swizzle => PcuDispatchOpCaps::VALUE_SWIZZLE,
        }
    }
}

/// Arithmetic / logical operation families.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PcuAluOp {
    Add,
    Sub,
    Mul,
    Div,
    Min,
    Max,
    And,
    Or,
    Xor,
    ShiftLeft,
    ShiftRight,
    Compare,
    Select,
}

impl PcuAluOp {
    #[must_use]
    pub const fn support_flag(self) -> PcuDispatchOpCaps {
        match self {
            Self::Add => PcuDispatchOpCaps::ALU_ADD,
            Self::Sub => PcuDispatchOpCaps::ALU_SUB,
            Self::Mul => PcuDispatchOpCaps::ALU_MUL,
            Self::Div => PcuDispatchOpCaps::ALU_DIV,
            Self::Min => PcuDispatchOpCaps::ALU_MIN,
            Self::Max => PcuDispatchOpCaps::ALU_MAX,
            Self::And => PcuDispatchOpCaps::ALU_AND,
            Self::Or => PcuDispatchOpCaps::ALU_OR,
            Self::Xor => PcuDispatchOpCaps::ALU_XOR,
            Self::ShiftLeft => PcuDispatchOpCaps::ALU_SHIFT_LEFT,
            Self::ShiftRight => PcuDispatchOpCaps::ALU_SHIFT_RIGHT,
            Self::Compare => PcuDispatchOpCaps::ALU_COMPARE,
            Self::Select => PcuDispatchOpCaps::ALU_SELECT,
        }
    }
}

/// Control-flow operation families.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PcuControlOp {
    Branch,
    Loop,
    Return,
}

impl PcuControlOp {
    #[must_use]
    pub const fn support_flag(self) -> PcuDispatchOpCaps {
        match self {
            Self::Branch => PcuDispatchOpCaps::CONTROL_BRANCH,
            Self::Loop => PcuDispatchOpCaps::CONTROL_LOOP,
            Self::Return => PcuDispatchOpCaps::CONTROL_RETURN,
        }
    }
}

/// Sampling level-selection model for one image sampling operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PcuSampleLevel {
    Implicit,
    ExplicitLod,
    Bias,
    Gradient,
}

/// One typed addressed image sampling operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PcuSampleOp {
    pub image: PcuBindingRef,
    pub sampler: PcuBindingRef,
    pub coordinates: PcuValueType,
    pub result_type: PcuValueType,
    pub level: PcuSampleLevel,
    pub offset_components: u8,
}

impl PcuSampleOp {
    #[must_use]
    pub const fn new(
        image: PcuBindingRef,
        sampler: PcuBindingRef,
        coordinates: PcuValueType,
        result_type: PcuValueType,
    ) -> Self {
        Self {
            image,
            sampler,
            coordinates,
            result_type,
            level: PcuSampleLevel::Implicit,
            offset_components: 0,
        }
    }

    #[must_use]
    pub const fn with_level(mut self, level: PcuSampleLevel) -> Self {
        self.level = level;
        self
    }

    #[must_use]
    pub const fn with_offset_components(mut self, offset_components: u8) -> Self {
        self.offset_components = offset_components;
        self
    }

    /// Validates that this sample op targets one readable image binding and one sampler binding.
    ///
    /// # Errors
    ///
    /// Returns the first contract mismatch that makes the operation dishonest.
    pub fn validate(self, bindings: &[PcuBinding<'_>]) -> Result<(), PcuSampleValidationError> {
        validate_sample_op(self, bindings)
    }
}

/// Binding-side memory/resource operation families.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PcuBindingOp {
    Load,
    Store,
    Atomic,
    Sample(PcuSampleOp),
}

impl PcuBindingOp {
    #[must_use]
    pub const fn support_flag(self) -> PcuDispatchOpCaps {
        match self {
            Self::Load => PcuDispatchOpCaps::BINDING_LOAD,
            Self::Store => PcuDispatchOpCaps::BINDING_STORE,
            Self::Atomic => PcuDispatchOpCaps::BINDING_ATOMIC,
            Self::Sample(_) => PcuDispatchOpCaps::BINDING_SAMPLE,
        }
    }
}

/// Port-side dataflow operation families.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PcuPortOp {
    Receive,
    Send,
    Peek,
    Discard,
}

impl PcuPortOp {
    #[must_use]
    pub const fn support_flag(self) -> PcuDispatchOpCaps {
        match self {
            Self::Receive => PcuDispatchOpCaps::PORT_RECEIVE,
            Self::Send => PcuDispatchOpCaps::PORT_SEND,
            Self::Peek => PcuDispatchOpCaps::PORT_PEEK,
            Self::Discard => PcuDispatchOpCaps::PORT_DISCARD,
        }
    }
}

/// Synchronization / ordering operation families.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PcuSyncOp {
    Barrier,
    Fence,
}

impl PcuSyncOp {
    #[must_use]
    pub const fn support_flag(self) -> PcuDispatchOpCaps {
        match self {
            Self::Barrier => PcuDispatchOpCaps::SYNC_BARRIER,
            Self::Fence => PcuDispatchOpCaps::SYNC_FENCE,
        }
    }
}
