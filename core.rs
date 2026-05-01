//! Shared PCU core vocabulary.
//!
//! This module owns the substrate-neutral nouns shared across all PCU models:
//! - values
//! - resources
//! - parameters
//! - ports
//! - invocation semantics
//! - kernel identity and signatures

use core::ops::{
    BitAnd,
    BitAndAssign,
    BitOr,
    BitOrAssign,
};

/// Scalar element types surfaced by the current PCU core.
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
}

impl PcuScalarType {
    /// Returns the honest bit width for this scalar type.
    #[must_use]
    pub const fn bit_width(self) -> u8 {
        match self {
            Self::Bool => 1,
            Self::I4 | Self::U4 => 4,
            Self::I8 | Self::U8 => 8,
            Self::I16 | Self::U16 | Self::F16 | Self::BF16 => 16,
            Self::I32 | Self::U32 | Self::F32 => 32,
            Self::I64 | Self::U64 | Self::F64 => 64,
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

/// Honest resource payload carried by one binding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PcuBindingType {
    Value(PcuValueType),
    Image(PcuImageBindingType),
    Sampler(PcuSamplerBindingType),
}

impl PcuBindingType {
    #[must_use]
    pub const fn value_type(self) -> Option<PcuValueType> {
        match self {
            Self::Value(value_type) => Some(value_type),
            Self::Image(_) | Self::Sampler(_) => None,
        }
    }

    #[must_use]
    pub const fn image_type(self) -> Option<PcuImageBindingType> {
        match self {
            Self::Image(image_type) => Some(image_type),
            Self::Value(_) | Self::Sampler(_) => None,
        }
    }

    #[must_use]
    pub const fn sampler_type(self) -> Option<PcuSamplerBindingType> {
        match self {
            Self::Sampler(sampler_type) => Some(sampler_type),
            Self::Value(_) | Self::Image(_) => None,
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
    pub const fn is_well_formed(self) -> bool {
        match (self.storage, self.binding_type) {
            (PcuBindingStorageClass::Image, PcuBindingType::Image(_)) => true,
            (PcuBindingStorageClass::Sampler, PcuBindingType::Sampler(_)) => {
                matches!(self.access, PcuBindingAccess::ReadOnly)
            }
            (PcuBindingStorageClass::Image, _)
            | (PcuBindingStorageClass::Sampler, _)
            | (_, PcuBindingType::Image(_))
            | (_, PcuBindingType::Sampler(_)) => false,
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
}

impl PcuParameterValue {
    #[must_use]
    pub const fn from_i4(value: i8) -> Option<Self> {
        if value < -8 || value > 7 {
            None
        } else {
            Some(Self::I4((value as u8) & 0x0f))
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

    #[must_use]
    pub const fn from_f16_bits(bits: u16) -> Self {
        Self::F16(bits)
    }

    #[must_use]
    pub const fn from_bf16_bits(bits: u16) -> Self {
        Self::BF16(bits)
    }

    #[must_use]
    pub fn from_f32(value: f32) -> Self {
        Self::F32(value.to_bits())
    }

    #[must_use]
    pub const fn from_f32_bits(bits: u32) -> Self {
        Self::F32(bits)
    }

    #[must_use]
    pub fn from_f64(value: f64) -> Self {
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
                    (bits | 0xf0) as i8
                } else {
                    bits as i8
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
    pub fn as_f32(self) -> Option<f32> {
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
    pub fn as_f64(self) -> Option<f64> {
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
    use super::{
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
}
