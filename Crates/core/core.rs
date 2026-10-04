//! Shared PCU core vocabulary.
//!
//! This module owns the substrate-neutral nouns shared across all PCU models:
//! - values
//! - resources
//! - parameters
//! - ports
//! - invocation semantics
//! - kernel identity and signatures

#[rustfmt::skip]
use core::ops::{
    BitAnd,
    BitAndAssign,
    BitOr,
    BitOrAssign,
};

/// Kind of deterministic arithmetic fault reported by a completed execution.
///
/// Floating range classification references IEEE Std 754-2019, clauses 7.4/7.5.
/// PCU's default error delivery and finite-input restriction are separate policies;
/// these variants do not represent every IEEE exception or status flag.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PcuExecutionFaultKind {
    /// An integer division/remainder or checked floating division used a zero divisor.
    /// Floating zero-divisor rejection is PCU policy; IEEE clauses 7.2/7.3
    /// distinguish invalid zero/zero from division of finite nonzero operands by zero.
    DivideByZero,
    /// An integer result exceeded its maximum, or floating-point rounding exceeded
    /// the finite exponent range (for either sign).
    ArithmeticOverflow,
    /// An integer result fell below its minimum, or a floating result violated the
    /// resolved underflow policy (tiny and inexact under the IEEE default).
    ArithmeticUnderflow,
    /// Signed division or remainder evaluated the minimum value with divisor `-1`.
    SignedDivisionOverflow,
    /// Checked floating arithmetic received a NaN or infinity operand.
    /// This is a PCU finite-input policy violation, not necessarily IEEE clause 7.2 invalid arithmetic.
    InvalidFloatingOperand,
}

/// A deterministic arithmetic fault within one reported execution unit.
///
/// Fatal faults take precedence over recovered range notices: a useful clamped lane must not
/// conceal another lane that failed to produce its result. Within the selected class, the
/// lowest affected logical invocation is reported. An all-recovered execution reports its
/// first range notice while retaining the prescribed continuation outputs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PcuExecutionFault {
    pub kind: PcuExecutionFaultKind,
    /// Logical fault position within the prepared execution unit's defined domain.
    /// A scalar map uses its element index; a grid-stride map uses its visited
    /// extent rather than its number of launched invocations. A compound may
    /// encode ordered arithmetic steps and must specify that mapping separately:
    /// neither its output size nor its launch count establishes the fault domain.
    /// Providers must validate decoded status against
    /// this domain and the admitted operation's fault law before lifting it.
    /// A malformed status is a backend protocol failure, not an arithmetic fault.
    pub invocation_id: u64,
    /// True when the reported execution unit completed with its defined continuation outputs.
    /// This is not merely whether the reported lane could recover: any terminal lane prevents
    /// recovered publication. A containing graph still needs its own complete success/output
    /// proof before producing or publishing a fresh owner.
    pub recovered: bool,
}

impl PcuExecutionFault {
    /// Whether this record addresses an element of the prepared logical domain.
    ///
    /// This predicate imposes no device-specific index-width limit. Providers
    /// validate their physical status encoding separately, then supply the
    /// operation-defined logical fault extent (including a grid-stride loop's
    /// visited extent or a compound's ordered-step extent). Output storage length
    /// does not generally establish this bound.
    /// It does not certify the fault kind, recovery policy or device completion.
    #[must_use]
    pub const fn is_within_logical_extent(self, extent: u64) -> bool {
        self.invocation_id < extent
    }
}

#[cfg(test)]
#[path = "fault_domain/tests/tests.rs"]
mod fault_domain_tests;

/// Named scalar representations; execution is admitted independently per operation.
///
/// F128 is IEEE binary128. F256 uses generalized binary256 interchange fields.
/// The two FP8 variants identify exact OFP8 encodings; neither is TF32, a scale
/// descriptor, or a permission to change accumulation precision. New entries
/// are appended to preserve existing capability-table indices.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PcuScalarType {
    Bool,
    I4,
    U4,
    I8,
    U8,
    I16,
    U16,
    I32,
    U32,
    I64,
    U64,
    F16,
    BF16,
    F32,
    F64,
    I128,
    U128,
    I256,
    U256,
    I512,
    U512,
    F128,
    F256,
    F8E4M3FN,
    F8E5M2,
}

impl PcuScalarType {
    /// Number of scalar variants represented by the core scalar vocabulary.
    pub const COUNT: usize = 25;

    /// All scalar variants in stable capability-table order.
    pub const ALL: [Self; Self::COUNT] = [
        Self::Bool,
        Self::I4,
        Self::U4,
        Self::I8,
        Self::U8,
        Self::I16,
        Self::U16,
        Self::I32,
        Self::U32,
        Self::I64,
        Self::U64,
        Self::F16,
        Self::BF16,
        Self::F32,
        Self::F64,
        Self::I128,
        Self::U128,
        Self::I256,
        Self::U256,
        Self::I512,
        Self::U512,
        Self::F128,
        Self::F256,
        Self::F8E4M3FN,
        Self::F8E5M2,
    ];

    /// Returns the honest bit width for this scalar type.
    #[must_use]
    pub const fn bit_width(self) -> u16 {
        match self {
            Self::Bool => 1,
            Self::I4 | Self::U4 => 4,
            Self::I8 | Self::U8 | Self::F8E4M3FN | Self::F8E5M2 => 8,
            Self::I16 | Self::U16 | Self::F16 | Self::BF16 => 16,
            Self::I32 | Self::U32 | Self::F32 => 32,
            Self::I64 | Self::U64 | Self::F64 => 64,
            Self::I128 | Self::U128 | Self::F128 => 128,
            Self::I256 | Self::U256 | Self::F256 => 256,
            Self::I512 | Self::U512 => 512,
        }
    }
}

/// Value shapes surfaced by the current PCU core.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PcuValueType {
    Scalar(PcuScalarType),
    Vector {
        scalar: PcuScalarType,
        lanes: u8,
    },
    Matrix {
        scalar: PcuScalarType,
        rows: u8,
        cols: u8,
    },
}

impl PcuValueType {
    #[must_use]
    pub const fn bool() -> Self {
        Self::Scalar(PcuScalarType::Bool)
    }

    #[must_use]
    pub const fn i4() -> Self {
        Self::Scalar(PcuScalarType::I4)
    }

    #[must_use]
    pub const fn u4() -> Self {
        Self::Scalar(PcuScalarType::U4)
    }

    #[must_use]
    pub const fn i8() -> Self {
        Self::Scalar(PcuScalarType::I8)
    }

    #[must_use]
    pub const fn u8() -> Self {
        Self::Scalar(PcuScalarType::U8)
    }

    #[must_use]
    pub const fn i16() -> Self {
        Self::Scalar(PcuScalarType::I16)
    }

    #[must_use]
    pub const fn u16() -> Self {
        Self::Scalar(PcuScalarType::U16)
    }

    #[must_use]
    pub const fn i32() -> Self {
        Self::Scalar(PcuScalarType::I32)
    }

    #[must_use]
    pub const fn u32() -> Self {
        Self::Scalar(PcuScalarType::U32)
    }

    #[must_use]
    pub const fn i64() -> Self {
        Self::Scalar(PcuScalarType::I64)
    }

    #[must_use]
    pub const fn u64() -> Self {
        Self::Scalar(PcuScalarType::U64)
    }

    #[must_use]
    pub const fn f16() -> Self {
        Self::Scalar(PcuScalarType::F16)
    }

    #[must_use]
    pub const fn bf16() -> Self {
        Self::Scalar(PcuScalarType::BF16)
    }

    #[must_use]
    pub const fn f32() -> Self {
        Self::Scalar(PcuScalarType::F32)
    }

    #[must_use]
    pub const fn f64() -> Self {
        Self::Scalar(PcuScalarType::F64)
    }

    #[must_use]
    pub const fn vector(scalar: PcuScalarType, lanes: u8) -> Self {
        Self::Vector { scalar, lanes }
    }

    #[must_use]
    pub const fn matrix(scalar: PcuScalarType, rows: u8, cols: u8) -> Self {
        Self::Matrix { scalar, rows, cols }
    }

    #[must_use]
    pub const fn scalar_type(self) -> PcuScalarType {
        match self {
            Self::Scalar(scalar) | Self::Vector { scalar, .. } | Self::Matrix { scalar, .. } => {
                scalar
            }
        }
    }

    #[must_use]
    pub const fn lanes(self) -> u16 {
        match self {
            Self::Scalar(_) => 1,
            Self::Vector { lanes, .. } => lanes as u16,
            Self::Matrix { rows, cols, .. } => (rows as u16) * (cols as u16),
        }
    }

    #[must_use]
    pub const fn linear_lanes(self) -> Option<u8> {
        match self {
            Self::Scalar(_) => Some(1),
            Self::Vector { lanes, .. } => Some(lanes),
            Self::Matrix { .. } => None,
        }
    }

    #[must_use]
    pub const fn rows(self) -> u8 {
        match self {
            Self::Scalar(_) | Self::Vector { .. } => 1,
            Self::Matrix { rows, .. } => rows,
        }
    }

    #[must_use]
    pub const fn cols(self) -> u8 {
        match self {
            Self::Scalar(_) => 1,
            Self::Vector { lanes, .. } => lanes,
            Self::Matrix { cols, .. } => cols,
        }
    }
}

/// Shared type/shape support truth for PCU value semantics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct PcuValueTypeCaps(u32);

impl PcuValueTypeCaps {
    pub const BOOL: Self = Self(1 << 0);
    pub const INT4: Self = Self(1 << 1);
    pub const UINT4: Self = Self(1 << 2);
    pub const INT8: Self = Self(1 << 3);
    pub const UINT8: Self = Self(1 << 4);
    pub const INT16: Self = Self(1 << 5);
    pub const UINT16: Self = Self(1 << 6);
    pub const INT32: Self = Self(1 << 7);
    pub const UINT32: Self = Self(1 << 8);
    pub const INT64: Self = Self(1 << 9);
    pub const UINT64: Self = Self(1 << 10);
    pub const FLOAT16: Self = Self(1 << 11);
    pub const BFLOAT16: Self = Self(1 << 12);
    pub const FLOAT32: Self = Self(1 << 13);
    pub const FLOAT64: Self = Self(1 << 14);
    pub const SCALAR_VALUES: Self = Self(1 << 15);
    pub const VECTOR_VALUES: Self = Self(1 << 16);
    pub const MATRIX_VALUES: Self = Self(1 << 17);
    pub const INT128: Self = Self(1 << 18);
    pub const UINT128: Self = Self(1 << 19);
    pub const INT256: Self = Self(1 << 20);
    pub const UINT256: Self = Self(1 << 21);
    pub const INT512: Self = Self(1 << 22);
    pub const UINT512: Self = Self(1 << 23);
    pub const FLOAT128: Self = Self(1 << 24);
    pub const FLOAT256: Self = Self(1 << 25);
    pub const FLOAT8_E4M3FN: Self = Self(1 << 26);
    pub const FLOAT8_E5M2: Self = Self(1 << 27);

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

    #[must_use]
    pub const fn for_scalar(scalar: PcuScalarType) -> Self {
        match scalar {
            PcuScalarType::Bool => Self::BOOL,
            PcuScalarType::I4 => Self::INT4,
            PcuScalarType::U4 => Self::UINT4,
            PcuScalarType::I8 => Self::INT8,
            PcuScalarType::U8 => Self::UINT8,
            PcuScalarType::I16 => Self::INT16,
            PcuScalarType::U16 => Self::UINT16,
            PcuScalarType::I32 => Self::INT32,
            PcuScalarType::U32 => Self::UINT32,
            PcuScalarType::I64 => Self::INT64,
            PcuScalarType::U64 => Self::UINT64,
            PcuScalarType::F16 => Self::FLOAT16,
            PcuScalarType::BF16 => Self::BFLOAT16,
            PcuScalarType::F32 => Self::FLOAT32,
            PcuScalarType::F64 => Self::FLOAT64,
            PcuScalarType::I128 => Self::INT128,
            PcuScalarType::U128 => Self::UINT128,
            PcuScalarType::I256 => Self::INT256,
            PcuScalarType::U256 => Self::UINT256,
            PcuScalarType::I512 => Self::INT512,
            PcuScalarType::U512 => Self::UINT512,
            PcuScalarType::F128 => Self::FLOAT128,
            PcuScalarType::F256 => Self::FLOAT256,
            PcuScalarType::F8E4M3FN => Self::FLOAT8_E4M3FN,
            PcuScalarType::F8E5M2 => Self::FLOAT8_E5M2,
        }
    }

    #[must_use]
    pub const fn for_value_type(value_type: PcuValueType) -> Self {
        let scalar = Self::for_scalar(value_type.scalar_type());
        let shape = match value_type {
            PcuValueType::Scalar(_) => Self::SCALAR_VALUES,
            PcuValueType::Vector { .. } => Self::VECTOR_VALUES,
            PcuValueType::Matrix { .. } => Self::MATRIX_VALUES,
        };
        scalar.union(shape)
    }

    #[must_use]
    pub const fn supports_value_type(self, value_type: PcuValueType) -> bool {
        self.contains(Self::for_value_type(value_type))
    }
}

impl BitOr for PcuValueTypeCaps {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        Self(self.0 | rhs.0)
    }
}

impl BitOrAssign for PcuValueTypeCaps {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

impl BitAnd for PcuValueTypeCaps {
    type Output = Self;

    fn bitand(self, rhs: Self) -> Self::Output {
        Self(self.0 & rhs.0)
    }
}

impl BitAndAssign for PcuValueTypeCaps {
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}

/// Storage-class vocabulary for one resource binding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PcuBindingStorageClass {
    Input,
    Output,
    Uniform,
    Storage,
    Shared,
    PushConstant,
    Private,
    Image,
    Sampler,
    AccelerationStructure,
    Constant,
}

/// Access mode for one resource binding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PcuBindingAccess {
    ReadOnly,
    WriteOnly,
    ReadWrite,
}

/// Canonical set/binding address for one resource attachment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PcuBindingRef {
    pub set: u32,
    pub binding: u32,
}

impl PcuBindingRef {
    #[must_use]
    pub const fn new(set: u32, binding: u32) -> Self {
        Self { set, binding }
    }
}

/// Dimensional shape for one image binding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PcuImageDimension {
    D1,
    D2,
    D3,
    Cube,
}

impl PcuImageDimension {
    #[must_use]
    pub const fn coordinate_lanes(self) -> u8 {
        match self {
            Self::D1 => 1,
            Self::D2 => 2,
            Self::D3 | Self::Cube => 3,
        }
    }
}

/// Typed image resource description for one image binding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PcuImageBindingType {
    pub dimension: PcuImageDimension,
    pub texel_type: PcuValueType,
    pub arrayed: bool,
    pub multisampled: bool,
}

impl PcuImageBindingType {
    #[must_use]
    pub const fn coordinate_type(self) -> PcuValueType {
        match self.dimension.coordinate_lanes() {
            1 => PcuValueType::Scalar(PcuScalarType::F32),
            lanes => PcuValueType::Vector {
                scalar: PcuScalarType::F32,
                lanes,
            },
        }
    }
}

/// Coordinate normalization model for one sampler binding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PcuSamplerCoordinateNormalization {
    Normalized,
    Unnormalized,
}

/// Addressing mode surfaced by one sampler binding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PcuSamplerAddressMode {
    ClampToEdge,
    ClampToBorder,
    Repeat,
    MirrorRepeat,
}

/// Filter kernel surfaced by one sampler binding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PcuSamplerFilter {
    Nearest,
    Linear,
}

/// Mipmap selection mode surfaced by one sampler binding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PcuSamplerMipmapMode {
    None,
    Nearest,
    Linear,
}

/// Typed sampler-state description for one sampler binding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PcuSamplerBindingType {
    pub coordinate_normalization: PcuSamplerCoordinateNormalization,
    pub min_filter: PcuSamplerFilter,
    pub mag_filter: PcuSamplerFilter,
    pub mipmap_mode: PcuSamplerMipmapMode,
    pub address_u: PcuSamplerAddressMode,
    pub address_v: PcuSamplerAddressMode,
    pub address_w: PcuSamplerAddressMode,
}

/// Acceleration-structure hierarchy level surfaced by one binding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PcuAccelerationStructureLevel {
    Generic,
    TopLevel,
    BottomLevel,
}

/// Typed acceleration-structure resource description for one binding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PcuAccelerationStructureBindingType {
    pub level: PcuAccelerationStructureLevel,
    pub mutable: bool,
}

/// Honest resource payload carried by one binding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PcuBindingType {
    Value(PcuValueType),
    Image(PcuImageBindingType),
    Sampler(PcuSamplerBindingType),
    AccelerationStructure(PcuAccelerationStructureBindingType),
}

impl PcuBindingType {
    #[must_use]
    pub const fn value_type(self) -> Option<PcuValueType> {
        match self {
            Self::Value(value_type) => Some(value_type),
            Self::Image(_) | Self::Sampler(_) | Self::AccelerationStructure(_) => None,
        }
    }

    #[must_use]
    pub const fn image_type(self) -> Option<PcuImageBindingType> {
        match self {
            Self::Image(image_type) => Some(image_type),
            Self::Value(_) | Self::Sampler(_) | Self::AccelerationStructure(_) => None,
        }
    }

    #[must_use]
    pub const fn sampler_type(self) -> Option<PcuSamplerBindingType> {
        match self {
            Self::Sampler(sampler_type) => Some(sampler_type),
            Self::Value(_) | Self::Image(_) | Self::AccelerationStructure(_) => None,
        }
    }

    #[must_use]
    pub const fn acceleration_structure_type(self) -> Option<PcuAccelerationStructureBindingType> {
        match self {
            Self::AccelerationStructure(acceleration_structure_type) => {
                Some(acceleration_structure_type)
            }
            Self::Value(_) | Self::Image(_) | Self::Sampler(_) => None,
        }
    }
}

/// Builtin values surfaced through the binding path.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PcuBuiltinValue<'a> {
    InvocationId,
    LaneId,
    GroupId,
    GroupCount,
    LaneIndex,
    Named(&'a str),
}

/// One typed memory/resource attachment for one program unit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PcuBinding<'a> {
    pub name: Option<&'a str>,
    pub set: u32,
    pub binding: u32,
    pub storage: PcuBindingStorageClass,
    pub access: PcuBindingAccess,
    pub binding_type: PcuBindingType,
    pub builtin: Option<PcuBuiltinValue<'a>>,
}

impl<'a> PcuBinding<'a> {
    #[must_use]
    pub const fn value(
        name: Option<&'a str>,
        set: u32,
        binding: u32,
        storage: PcuBindingStorageClass,
        access: PcuBindingAccess,
        value_type: PcuValueType,
    ) -> Self {
        Self {
            name,
            set,
            binding,
            storage,
            access,
            binding_type: PcuBindingType::Value(value_type),
            builtin: None,
        }
    }

    #[must_use]
    pub const fn image(
        name: Option<&'a str>,
        set: u32,
        binding: u32,
        access: PcuBindingAccess,
        image_type: PcuImageBindingType,
    ) -> Self {
        Self {
            name,
            set,
            binding,
            storage: PcuBindingStorageClass::Image,
            access,
            binding_type: PcuBindingType::Image(image_type),
            builtin: None,
        }
    }

    #[must_use]
    pub const fn sampler(
        name: Option<&'a str>,
        set: u32,
        binding: u32,
        sampler_type: PcuSamplerBindingType,
    ) -> Self {
        Self {
            name,
            set,
            binding,
            storage: PcuBindingStorageClass::Sampler,
            access: PcuBindingAccess::ReadOnly,
            binding_type: PcuBindingType::Sampler(sampler_type),
            builtin: None,
        }
    }

    #[must_use]
    pub const fn acceleration_structure(
        name: Option<&'a str>,
        set: u32,
        binding: u32,
        access: PcuBindingAccess,
        acceleration_structure_type: PcuAccelerationStructureBindingType,
    ) -> Self {
        Self {
            name,
            set,
            binding,
            storage: PcuBindingStorageClass::AccelerationStructure,
            access,
            binding_type: PcuBindingType::AccelerationStructure(acceleration_structure_type),
            builtin: None,
        }
    }

    #[must_use]
    pub const fn reference(self) -> PcuBindingRef {
        PcuBindingRef::new(self.set, self.binding)
    }

    #[must_use]
    pub const fn value_type(self) -> Option<PcuValueType> {
        self.binding_type.value_type()
    }

    #[must_use]
    pub const fn image_type(self) -> Option<PcuImageBindingType> {
        self.binding_type.image_type()
    }

    #[must_use]
    pub const fn sampler_type(self) -> Option<PcuSamplerBindingType> {
        self.binding_type.sampler_type()
    }

    #[must_use]
    pub const fn acceleration_structure_type(self) -> Option<PcuAccelerationStructureBindingType> {
        self.binding_type.acceleration_structure_type()
    }

    #[must_use]
    pub const fn is_well_formed(self) -> bool {
        match (self.storage, self.binding_type) {
            (PcuBindingStorageClass::Image, PcuBindingType::Image(_))
            | (
                PcuBindingStorageClass::AccelerationStructure,
                PcuBindingType::AccelerationStructure(_),
            ) => true,
            (PcuBindingStorageClass::Sampler, PcuBindingType::Sampler(_)) => {
                matches!(self.access, PcuBindingAccess::ReadOnly)
            }
            (
                PcuBindingStorageClass::Image
                | PcuBindingStorageClass::Sampler
                | PcuBindingStorageClass::AccelerationStructure,
                _,
            )
            | (
                _,
                PcuBindingType::Image(_)
                | PcuBindingType::Sampler(_)
                | PcuBindingType::AccelerationStructure(_),
            ) => false,
            (_, PcuBindingType::Value(_)) => true,
        }
    }
}

/// Stable slot naming one runtime parameter in one program-unit signature.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PcuParameterSlot(pub u8);

/// One declared runtime parameter in one program-unit signature.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PcuParameter<'a> {
    pub slot: PcuParameterSlot,
    pub name: Option<&'a str>,
    pub value_type: PcuValueType,
}

impl<'a> PcuParameter<'a> {
    #[must_use]
    pub const fn named(slot: PcuParameterSlot, name: &'a str, value_type: PcuValueType) -> Self {
        Self {
            slot,
            name: Some(name),
            value_type,
        }
    }

    #[must_use]
    pub const fn anonymous(slot: PcuParameterSlot, value_type: PcuValueType) -> Self {
        Self {
            slot,
            name: None,
            value_type,
        }
    }
}

/// One scalar runtime parameter value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PcuParameterValue {
    Bool(bool),
    I4(u8),
    U4(u8),
    I8(i8),
    U8(u8),
    I16(i16),
    U16(u16),
    I32(i32),
    U32(u32),
    I64(i64),
    U64(u64),
    F16(u16),
    BF16(u16),
    F32(u32),
    F64(u64),
    /// Exact E4M3FN payload, including signed zero and NaN encodings.
    F8E4M3FN(u8),
    /// Exact E5M2 payload, including signed zero, infinities and NaN encodings.
    F8E5M2(u8),
}

impl PcuParameterValue {
    #[must_use]
    pub const fn from_i4(value: i8) -> Option<Self> {
        if value < -8 || value > 7 {
            None
        } else {
            Some(Self::I4(u8::from_ne_bytes(value.to_ne_bytes()) & 0x0f))
        }
    }

    #[must_use]
    pub const fn from_i4_bits(bits: u8) -> Self {
        Self::I4(bits & 0x0f)
    }

    #[must_use]
    pub const fn from_u4(value: u8) -> Option<Self> {
        if value > 0x0f {
            None
        } else {
            Some(Self::U4(value))
        }
    }

    #[must_use]
    pub const fn from_u4_bits(bits: u8) -> Self {
        Self::U4(bits & 0x0f)
    }

    /// Preserves every raw E4M3FN bit pattern without numeric conversion.
    #[must_use]
    pub const fn from_f8_e4m3fn_bits(bits: u8) -> Self {
        Self::F8E4M3FN(bits)
    }

    /// Preserves every raw E5M2 bit pattern without numeric conversion.
    #[must_use]
    pub const fn from_f8_e5m2_bits(bits: u8) -> Self {
        Self::F8E5M2(bits)
    }

    #[must_use]
    pub const fn from_f16_bits(bits: u16) -> Self {
        Self::F16(bits)
    }

    #[must_use]
    pub const fn from_bf16_bits(bits: u16) -> Self {
        Self::BF16(bits)
    }

    #[must_use]
    pub const fn from_f32(value: f32) -> Self {
        Self::F32(value.to_bits())
    }

    #[must_use]
    pub const fn from_f32_bits(bits: u32) -> Self {
        Self::F32(bits)
    }

    #[must_use]
    pub const fn from_f64(value: f64) -> Self {
        Self::F64(value.to_bits())
    }

    #[must_use]
    pub const fn from_f64_bits(bits: u64) -> Self {
        Self::F64(bits)
    }

    #[must_use]
    pub const fn value_type(self) -> PcuValueType {
        match self {
            Self::Bool(_) => PcuValueType::bool(),
            Self::I4(_) => PcuValueType::i4(),
            Self::U4(_) => PcuValueType::u4(),
            Self::I8(_) => PcuValueType::i8(),
            Self::U8(_) => PcuValueType::u8(),
            Self::I16(_) => PcuValueType::i16(),
            Self::U16(_) => PcuValueType::u16(),
            Self::I32(_) => PcuValueType::i32(),
            Self::U32(_) => PcuValueType::u32(),
            Self::I64(_) => PcuValueType::i64(),
            Self::U64(_) => PcuValueType::u64(),
            Self::F16(_) => PcuValueType::f16(),
            Self::BF16(_) => PcuValueType::bf16(),
            Self::F32(_) => PcuValueType::f32(),
            Self::F64(_) => PcuValueType::f64(),
            Self::F8E4M3FN(_) => PcuValueType::Scalar(PcuScalarType::F8E4M3FN),
            Self::F8E5M2(_) => PcuValueType::Scalar(PcuScalarType::F8E5M2),
        }
    }

    #[must_use]
    pub fn matches_type(self, value_type: PcuValueType) -> bool {
        self.value_type() == value_type
    }

    #[must_use]
    pub const fn as_u8(self) -> Option<u8> {
        match self {
            Self::U8(value) => Some(value),
            _ => None,
        }
    }

    #[must_use]
    pub const fn as_i4(self) -> Option<i8> {
        match self {
            Self::I4(bits) => {
                let bits = bits & 0x0f;
                Some(if (bits & 0x08) != 0 {
                    i8::from_ne_bytes([bits | 0xf0])
                } else {
                    i8::from_ne_bytes([bits])
                })
            }
            _ => None,
        }
    }

    #[must_use]
    pub const fn as_i4_bits(self) -> Option<u8> {
        match self {
            Self::I4(bits) => Some(bits & 0x0f),
            _ => None,
        }
    }

    #[must_use]
    pub const fn as_u4(self) -> Option<u8> {
        match self {
            Self::U4(bits) => Some(bits & 0x0f),
            _ => None,
        }
    }

    #[must_use]
    pub const fn as_u4_bits(self) -> Option<u8> {
        match self {
            Self::U4(bits) => Some(bits & 0x0f),
            _ => None,
        }
    }

    #[must_use]
    pub const fn as_u16(self) -> Option<u16> {
        match self {
            Self::U16(value) => Some(value),
            _ => None,
        }
    }

    #[must_use]
    pub const fn as_u32(self) -> Option<u32> {
        match self {
            Self::U32(value) => Some(value),
            _ => None,
        }
    }

    #[must_use]
    pub const fn as_i64(self) -> Option<i64> {
        match self {
            Self::I64(value) => Some(value),
            _ => None,
        }
    }

    #[must_use]
    pub const fn as_u64(self) -> Option<u64> {
        match self {
            Self::U64(value) => Some(value),
            _ => None,
        }
    }

    /// Returns the exact E4M3FN bits only for that scalar identity.
    #[must_use]
    pub const fn as_f8_e4m3fn_bits(self) -> Option<u8> {
        match self {
            Self::F8E4M3FN(bits) => Some(bits),
            _ => None,
        }
    }

    /// Returns the exact E5M2 bits only for that scalar identity.
    #[must_use]
    pub const fn as_f8_e5m2_bits(self) -> Option<u8> {
        match self {
            Self::F8E5M2(bits) => Some(bits),
            _ => None,
        }
    }

    #[must_use]
    pub const fn as_f16_bits(self) -> Option<u16> {
        match self {
            Self::F16(bits) => Some(bits),
            _ => None,
        }
    }

    #[must_use]
    pub const fn as_bf16_bits(self) -> Option<u16> {
        match self {
            Self::BF16(bits) => Some(bits),
            _ => None,
        }
    }

    #[must_use]
    pub const fn as_f32(self) -> Option<f32> {
        match self {
            Self::F32(bits) => Some(f32::from_bits(bits)),
            _ => None,
        }
    }

    #[must_use]
    pub const fn as_f32_bits(self) -> Option<u32> {
        match self {
            Self::F32(bits) => Some(bits),
            _ => None,
        }
    }

    #[must_use]
    pub const fn as_f64(self) -> Option<f64> {
        match self {
            Self::F64(bits) => Some(f64::from_bits(bits)),
            _ => None,
        }
    }

    #[must_use]
    pub const fn as_f64_bits(self) -> Option<u64> {
        match self {
            Self::F64(bits) => Some(bits),
            _ => None,
        }
    }
}

/// One submit-time binding from a declared parameter slot to one runtime value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PcuParameterBinding {
    pub slot: PcuParameterSlot,
    pub value: PcuParameterValue,
}

impl PcuParameterBinding {
    #[must_use]
    pub const fn new(slot: PcuParameterSlot, value: PcuParameterValue) -> Self {
        Self { slot, value }
    }
}

/// Direction of one PCU port.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PcuPortDirection {
    Input,
    Output,
    InOut,
}

/// Traffic cadence for one PCU port.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PcuPortRate {
    Single,
    Stream,
    Signal,
    Latch,
}

/// Blocking behavior for one PCU port.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PcuPortBlocking {
    Blocking,
    NonBlocking,
}

/// Delivery/reliability behavior for one PCU port.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PcuPortReliability {
    Lossless,
    Lossy,
}

/// Backpressure behavior for one PCU port.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PcuPortBackpressure {
    Backpressured,
    FreeRunning,
}

/// One typed directional I/O endpoint for one program unit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PcuPort<'a> {
    pub name: Option<&'a str>,
    pub direction: PcuPortDirection,
    pub value_type: PcuValueType,
    pub rate: PcuPortRate,
    pub blocking: PcuPortBlocking,
    pub reliability: PcuPortReliability,
    pub backpressure: PcuPortBackpressure,
}

impl<'a> PcuPort<'a> {
    #[must_use]
    pub const fn new(
        name: Option<&'a str>,
        direction: PcuPortDirection,
        value_type: PcuValueType,
        rate: PcuPortRate,
        blocking: PcuPortBlocking,
        reliability: PcuPortReliability,
        backpressure: PcuPortBackpressure,
    ) -> Self {
        Self {
            name,
            direction,
            value_type,
            rate,
            blocking,
            reliability,
            backpressure,
        }
    }

    #[must_use]
    pub const fn stream_input(name: Option<&'a str>, value_type: PcuValueType) -> Self {
        Self::new(
            name,
            PcuPortDirection::Input,
            value_type,
            PcuPortRate::Stream,
            PcuPortBlocking::NonBlocking,
            PcuPortReliability::Lossless,
            PcuPortBackpressure::Backpressured,
        )
    }

    #[must_use]
    pub const fn stream_output(name: Option<&'a str>, value_type: PcuValueType) -> Self {
        Self::new(
            name,
            PcuPortDirection::Output,
            value_type,
            PcuPortRate::Stream,
            PcuPortBlocking::NonBlocking,
            PcuPortReliability::Lossless,
            PcuPortBackpressure::Backpressured,
        )
    }
}

/// Topology shape for one program unit's execution model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PcuInvocationTopology {
    Single,
    Indexed { logical_shape: [u32; 3] },
    Continuous,
    Triggered,
}

/// Parallelism relationship between simultaneously active invocations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PcuInvocationParallelism {
    Serial,
    Independent,
    Cooperative,
    Lockstep,
}

/// Progress/lifetime model for one invocation family.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PcuInvocationProgress {
    Finite,
    Persistent,
    Continuous,
}

/// Ordering contract for work issued through one invocation model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PcuInvocationOrdering {
    Unordered,
    InOrder,
    PerPort,
}

/// Full invocation model for one program-unit profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PcuInvocationModel {
    pub topology: PcuInvocationTopology,
    pub parallelism: PcuInvocationParallelism,
    pub progress: PcuInvocationProgress,
    pub ordering: PcuInvocationOrdering,
}

impl PcuInvocationModel {
    #[must_use]
    pub const fn single() -> Self {
        Self {
            topology: PcuInvocationTopology::Single,
            parallelism: PcuInvocationParallelism::Serial,
            progress: PcuInvocationProgress::Finite,
            ordering: PcuInvocationOrdering::InOrder,
        }
    }

    #[must_use]
    pub const fn indexed(logical_shape: [u32; 3]) -> Self {
        Self {
            topology: PcuInvocationTopology::Indexed { logical_shape },
            parallelism: PcuInvocationParallelism::Independent,
            progress: PcuInvocationProgress::Finite,
            ordering: PcuInvocationOrdering::Unordered,
        }
    }

    #[must_use]
    pub const fn continuous() -> Self {
        Self {
            topology: PcuInvocationTopology::Continuous,
            parallelism: PcuInvocationParallelism::Lockstep,
            progress: PcuInvocationProgress::Continuous,
            ordering: PcuInvocationOrdering::PerPort,
        }
    }

    #[must_use]
    pub const fn command() -> Self {
        Self {
            topology: PcuInvocationTopology::Single,
            parallelism: PcuInvocationParallelism::Serial,
            progress: PcuInvocationProgress::Finite,
            ordering: PcuInvocationOrdering::InOrder,
        }
    }

    #[must_use]
    pub const fn transaction() -> Self {
        Self {
            topology: PcuInvocationTopology::Single,
            parallelism: PcuInvocationParallelism::Serial,
            progress: PcuInvocationProgress::Finite,
            ordering: PcuInvocationOrdering::InOrder,
        }
    }

    #[must_use]
    pub const fn triggered() -> Self {
        Self {
            topology: PcuInvocationTopology::Triggered,
            parallelism: PcuInvocationParallelism::Serial,
            progress: PcuInvocationProgress::Persistent,
            ordering: PcuInvocationOrdering::InOrder,
        }
    }
}

/// Stable caller-supplied identifier for one generic PCU program unit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PcuKernelId(pub u32);

/// Coarse profile family carried by one generic PCU program unit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PcuIrKind {
    Dispatch,
    Stream,
    Command,
    Transaction,
    Signal,
}

/// Program-unit-facing signature over memory truth, dataflow truth, and invocation truth.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PcuKernelSignature<'a> {
    pub bindings: &'a [PcuBinding<'a>],
    pub ports: &'a [PcuPort<'a>],
    pub parameters: &'a [PcuParameter<'a>],
    pub invocation: PcuInvocationModel,
}

/// Minimal trait implemented by generic execution-profile payloads.
pub trait PcuKernelIrContract {
    fn id(&self) -> PcuKernelId;
    fn kind(&self) -> PcuIrKind;
    fn entry_point(&self) -> &str;
    fn signature(&self) -> PcuKernelSignature<'_>;
}

#[cfg(test)]
mod tests {
    #[rustfmt::skip]
    use super::{
        PcuAccelerationStructureBindingType,
        PcuAccelerationStructureLevel,
        PcuBinding,
        PcuBindingAccess,
        PcuBindingStorageClass,
        PcuParameterValue,
        PcuScalarType,
        PcuValueType,
    };

    #[test]
    fn scalar_types_cover_64_bit_widths() {
        assert_eq!(PcuScalarType::I64.bit_width(), 64);
        assert_eq!(PcuScalarType::U64.bit_width(), 64);
        assert_eq!(PcuScalarType::F64.bit_width(), 64);
        assert_eq!(PcuScalarType::I4.bit_width(), 4);
        assert_eq!(PcuScalarType::U4.bit_width(), 4);
        assert_eq!(PcuScalarType::BF16.bit_width(), 16);
        assert_eq!(PcuValueType::i64().scalar_type(), PcuScalarType::I64);
        assert_eq!(PcuValueType::u64().scalar_type(), PcuScalarType::U64);
        assert_eq!(PcuValueType::f64().scalar_type(), PcuScalarType::F64);
        assert_eq!(PcuValueType::bf16().scalar_type(), PcuScalarType::BF16);
    }

    #[test]
    fn parameter_values_round_trip_64_bit_types() {
        let signed = PcuParameterValue::I64(-9);
        let unsigned = PcuParameterValue::U64(42);
        let float = PcuParameterValue::from_f64(3.5);

        assert_eq!(signed.value_type(), PcuValueType::i64());
        assert_eq!(unsigned.value_type(), PcuValueType::u64());
        assert_eq!(float.value_type(), PcuValueType::f64());
        assert_eq!(signed.as_i64(), Some(-9));
        assert_eq!(unsigned.as_u64(), Some(42));
        assert_eq!(float.as_f64(), Some(3.5));
    }

    #[test]
    fn raw_fp8_parameters_preserve_all_bits_and_require_exact_scalar_tags() {
        #[rustfmt::skip]
        use crate::{
            PcuInvocationParameters,
            PcuParameter,
            PcuParameterBinding,
            PcuParameterSlot,
        };

        for bits in u8::MIN..=u8::MAX {
            let e4 = PcuParameterValue::from_f8_e4m3fn_bits(bits);
            let e5 = PcuParameterValue::from_f8_e5m2_bits(bits);
            assert_eq!(e4.as_f8_e4m3fn_bits(), Some(bits));
            assert_eq!(e5.as_f8_e5m2_bits(), Some(bits));
            assert_eq!(e4.as_f8_e5m2_bits(), None);
            assert_eq!(e5.as_f8_e4m3fn_bits(), None);
            for value in [e4, e5] {
                for scalar in PcuScalarType::ALL {
                    assert_eq!(
                        value.matches_type(PcuValueType::Scalar(scalar)),
                        scalar == value.value_type().scalar_type(),
                    );
                }
                let slot = PcuParameterSlot(0);
                let bindings = [PcuParameterBinding::new(slot, value)];
                let parameters = PcuInvocationParameters {
                    bindings: &bindings,
                };
                let declaration = [PcuParameter::anonymous(slot, value.value_type())];
                assert!(parameters.validate_against(&declaration));
                assert_eq!(parameters.value(slot), Some(value));
                let wrong = [PcuParameter::anonymous(slot, PcuValueType::u8())];
                assert!(!parameters.validate_against(&wrong));
                let immediate = crate::model::PcuOperand::Immediate(value);
                let crate::model::PcuOperand::Immediate(retained) = immediate else {
                    panic!("command immediate must retain its scalar payload");
                };
                assert_eq!(retained, value);
            }
        }
        assert_eq!(PcuParameterValue::U8(0).as_f8_e4m3fn_bits(), None);
        assert_eq!(PcuParameterValue::U8(0).as_f8_e5m2_bits(), None);
    }

    #[test]
    fn sub_byte_and_matrix_types_are_well_formed() {
        let i4 = PcuParameterValue::from_i4(-3).expect("i4 range should accept -3");
        let u4 = PcuParameterValue::from_u4(12).expect("u4 range should accept 12");
        let bf16 = PcuParameterValue::from_bf16_bits(0x3f80);
        let matrix = PcuValueType::matrix(PcuScalarType::BF16, 4, 4);

        assert_eq!(i4.value_type(), PcuValueType::i4());
        assert_eq!(u4.value_type(), PcuValueType::u4());
        assert_eq!(bf16.value_type(), PcuValueType::bf16());
        assert_eq!(i4.as_i4(), Some(-3));
        assert_eq!(u4.as_u4(), Some(12));
        assert_eq!(bf16.as_bf16_bits(), Some(0x3f80));
        assert_eq!(matrix.scalar_type(), PcuScalarType::BF16);
        assert_eq!(matrix.rows(), 4);
        assert_eq!(matrix.cols(), 4);
        assert_eq!(matrix.lanes(), 16);
        assert_eq!(matrix.linear_lanes(), None);
    }

    #[test]
    fn acceleration_structure_bindings_are_typed_resources() {
        let acceleration_structure = PcuBinding::acceleration_structure(
            Some("scene"),
            0,
            3,
            PcuBindingAccess::ReadOnly,
            PcuAccelerationStructureBindingType {
                level: PcuAccelerationStructureLevel::TopLevel,
                mutable: false,
            },
        );
        let malformed = PcuBinding::value(
            Some("wrong"),
            0,
            4,
            PcuBindingStorageClass::AccelerationStructure,
            PcuBindingAccess::ReadOnly,
            PcuValueType::u64(),
        );

        assert!(acceleration_structure.is_well_formed());
        assert_eq!(
            acceleration_structure
                .acceleration_structure_type()
                .expect("acceleration structure binding should expose its type")
                .level,
            PcuAccelerationStructureLevel::TopLevel
        );
        assert!(!malformed.is_well_formed());
    }
}
