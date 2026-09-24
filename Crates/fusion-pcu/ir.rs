//! Shared PCU IR support vocabulary.
//!
//! This module owns the low-level IR nouns that are genuinely shared across model families:
//! - generic value ops
//! - ALU ops
//! - generic control ops
//! - binding/port/sync ops
//! - sample ops over binding truth

use core::ops::{
    BitAnd,
    BitAndAssign,
    BitOr,
    BitOrAssign,
};

use crate::{
    PcuBinding,
    PcuBindingRef,
    PcuDispatchOpCaps,
    PcuValueType,
};
use crate::validation::{
    validate_sample_op,
    validate_trace_ray_op,
};
pub use crate::validation::{
    PcuSampleValidationError,
    PcuTraceRayValidationError,
};

/// Lowest-level dispatch instruction contract shared by all PCU instruction extensions.
pub trait PcuDispatchInstruction {
    fn support_flag(self) -> PcuDispatchOpCaps;
}

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

impl PcuDispatchInstruction for PcuValueOp {
    fn support_flag(self) -> PcuDispatchOpCaps {
        self.support_flag()
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

impl PcuDispatchInstruction for PcuAluOp {
    fn support_flag(self) -> PcuDispatchOpCaps {
        self.support_flag()
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

impl PcuDispatchInstruction for PcuControlOp {
    fn support_flag(self) -> PcuDispatchOpCaps {
        self.support_flag()
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

impl PcuDispatchInstruction for PcuBindingOp {
    fn support_flag(self) -> PcuDispatchOpCaps {
        self.support_flag()
    }
}

/// Coordinate-oriented raw operation families used by graphics composition layers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PcuCoordinateOp {
    LoadCoordinate,
    StorePosition,
    LoadInterpolant,
    StoreOutput,
    DerivativeX,
    DerivativeY,
    SampleMask,
}

impl PcuCoordinateOp {
    #[must_use]
    pub const fn support_flag(self) -> PcuDispatchOpCaps {
        match self {
            Self::LoadCoordinate => PcuDispatchOpCaps::COORDINATE_LOAD,
            Self::StorePosition => PcuDispatchOpCaps::POSITION_STORE,
            Self::LoadInterpolant => PcuDispatchOpCaps::INTERPOLANT_LOAD,
            Self::StoreOutput => PcuDispatchOpCaps::OUTPUT_STORE,
            Self::DerivativeX => PcuDispatchOpCaps::DERIVATIVE_X,
            Self::DerivativeY => PcuDispatchOpCaps::DERIVATIVE_Y,
            Self::SampleMask => PcuDispatchOpCaps::SAMPLE_MASK,
        }
    }
}

impl PcuDispatchInstruction for PcuCoordinateOp {
    fn support_flag(self) -> PcuDispatchOpCaps {
        self.support_flag()
    }
}

/// Ray flags attached to one ray traversal operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct PcuRayFlags(u32);

impl PcuRayFlags {
    pub const NONE: Self = Self(0);
    pub const FORCE_OPAQUE: Self = Self(1 << 0);
    pub const FORCE_NON_OPAQUE: Self = Self(1 << 1);
    pub const ACCEPT_FIRST_HIT_AND_END_SEARCH: Self = Self(1 << 2);
    pub const SKIP_CLOSEST_HIT: Self = Self(1 << 3);
    pub const CULL_BACK_FACING_TRIANGLES: Self = Self(1 << 4);
    pub const CULL_FRONT_FACING_TRIANGLES: Self = Self(1 << 5);
    pub const CULL_OPAQUE: Self = Self(1 << 6);
    pub const CULL_NON_OPAQUE: Self = Self(1 << 7);

    #[must_use]
    pub const fn empty() -> Self {
        Self(0)
    }

    #[must_use]
    pub const fn bits(self) -> u32 {
        self.0
    }

    #[must_use]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }

    #[must_use]
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }
}

impl BitOr for PcuRayFlags {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        Self(self.0 | rhs.0)
    }
}

impl BitOrAssign for PcuRayFlags {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

impl BitAnd for PcuRayFlags {
    type Output = Self;

    fn bitand(self, rhs: Self) -> Self::Output {
        Self(self.0 & rhs.0)
    }
}

impl BitAndAssign for PcuRayFlags {
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}

/// One raw ray traversal request over an acceleration-structure binding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PcuTraceRayOp {
    pub acceleration_structure: PcuBindingRef,
    pub flags: PcuRayFlags,
    pub instance_mask: u8,
    pub payload_bytes: u16,
    pub max_recursion_depth: u8,
}

impl PcuTraceRayOp {
    #[must_use]
    pub const fn new(acceleration_structure: PcuBindingRef) -> Self {
        Self {
            acceleration_structure,
            flags: PcuRayFlags::NONE,
            instance_mask: 0xff,
            payload_bytes: 0,
            max_recursion_depth: 1,
        }
    }

    #[must_use]
    pub const fn with_flags(mut self, flags: PcuRayFlags) -> Self {
        self.flags = flags;
        self
    }

    #[must_use]
    pub const fn with_instance_mask(mut self, instance_mask: u8) -> Self {
        self.instance_mask = instance_mask;
        self
    }

    #[must_use]
    pub const fn with_payload_bytes(mut self, payload_bytes: u16) -> Self {
        self.payload_bytes = payload_bytes;
        self
    }

    #[must_use]
    pub const fn with_max_recursion_depth(mut self, max_recursion_depth: u8) -> Self {
        self.max_recursion_depth = max_recursion_depth;
        self
    }

    /// Validates that this trace op targets one readable top-level acceleration structure.
    ///
    /// # Errors
    ///
    /// Returns the first contract mismatch that makes the trace request dishonest.
    pub fn validate(self, bindings: &[PcuBinding<'_>]) -> Result<(), PcuTraceRayValidationError> {
        validate_trace_ray_op(self, bindings)
    }
}

/// Ray traversal and ray-query operation families.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PcuRayTraceOp {
    TraceRay(PcuTraceRayOp),
    TraceRayInline(PcuTraceRayOp),
    RayQueryProceed,
    RayQueryCommittedStatus,
    RayQueryCommittedDistance,
    RayQueryCommittedInstance,
    RayQueryCommittedPrimitive,
    ReportHit { attribute_bytes: u16 },
    IgnoreHit,
    AcceptHitAndEndSearch,
    PayloadRead { byte_offset: u16, byte_len: u16 },
    PayloadWrite { byte_offset: u16, byte_len: u16 },
}

impl PcuRayTraceOp {
    #[must_use]
    pub const fn support_flag(self) -> PcuDispatchOpCaps {
        match self {
            Self::TraceRay(_) => PcuDispatchOpCaps::RAY_TRACE,
            Self::TraceRayInline(_) => PcuDispatchOpCaps::RAY_TRACE_INLINE,
            Self::RayQueryProceed => PcuDispatchOpCaps::RAY_QUERY_PROCEED,
            Self::RayQueryCommittedStatus => PcuDispatchOpCaps::RAY_QUERY_COMMITTED_STATUS,
            Self::RayQueryCommittedDistance => PcuDispatchOpCaps::RAY_QUERY_COMMITTED_DISTANCE,
            Self::RayQueryCommittedInstance => PcuDispatchOpCaps::RAY_QUERY_COMMITTED_INSTANCE,
            Self::RayQueryCommittedPrimitive => PcuDispatchOpCaps::RAY_QUERY_COMMITTED_PRIMITIVE,
            Self::ReportHit { .. } => PcuDispatchOpCaps::RAY_REPORT_HIT,
            Self::IgnoreHit => PcuDispatchOpCaps::RAY_IGNORE_HIT,
            Self::AcceptHitAndEndSearch => PcuDispatchOpCaps::RAY_ACCEPT_HIT_AND_END_SEARCH,
            Self::PayloadRead { .. } => PcuDispatchOpCaps::RAY_PAYLOAD_READ,
            Self::PayloadWrite { .. } => PcuDispatchOpCaps::RAY_PAYLOAD_WRITE,
        }
    }

    #[must_use]
    pub const fn trace_ray(self) -> Option<PcuTraceRayOp> {
        match self {
            Self::TraceRay(trace) | Self::TraceRayInline(trace) => Some(trace),
            Self::RayQueryProceed
            | Self::RayQueryCommittedStatus
            | Self::RayQueryCommittedDistance
            | Self::RayQueryCommittedInstance
            | Self::RayQueryCommittedPrimitive
            | Self::ReportHit { .. }
            | Self::IgnoreHit
            | Self::AcceptHitAndEndSearch
            | Self::PayloadRead { .. }
            | Self::PayloadWrite { .. } => None,
        }
    }
}

impl PcuDispatchInstruction for PcuRayTraceOp {
    fn support_flag(self) -> PcuDispatchOpCaps {
        self.support_flag()
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

impl PcuDispatchInstruction for PcuPortOp {
    fn support_flag(self) -> PcuDispatchOpCaps {
        self.support_flag()
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

impl PcuDispatchInstruction for PcuSyncOp {
    fn support_flag(self) -> PcuDispatchOpCaps {
        self.support_flag()
    }
}
