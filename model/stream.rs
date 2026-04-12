//! Stream-model vocabulary and backend-neutral kernel builder.

use core::ops::{
    BitAnd,
    BitAndAssign,
    BitOr,
    BitOrAssign,
};

use crate::contract::{
    PcuDispatchPolicyCaps,
    PcuError,
    PcuKernel,
    PcuKernelIrContract,
    PcuKernelId,
    PcuKernelSignature,
    PcuInvocationModel,
    PcuIrKind,
    PcuInvocationParameters,
    PcuParameter,
    PcuPort,
};
use crate::validation::{
    validate_stream_simple_transform,
};
use crate::{
    PcuParameterSlot,
    PcuScalarType,
    PcuValueType,
};

const DEFAULT_PATTERN_CAPACITY: usize = 16;

/// Stream element types surfaced by the current stream profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PcuStreamValueType {
    U8,
    U16,
    U32,
}

impl PcuStreamValueType {
    #[must_use]
    pub const fn bit_width(self) -> u8 {
        match self {
            Self::U8 => 8,
            Self::U16 => 16,
            Self::U32 => 32,
        }
    }

    #[must_use]
    pub const fn as_value_type(self) -> PcuValueType {
        match self {
            Self::U8 => PcuValueType::Scalar(PcuScalarType::U8),
            Self::U16 => PcuValueType::Scalar(PcuScalarType::U16),
            Self::U32 => PcuValueType::Scalar(PcuScalarType::U32),
        }
    }

    #[must_use]
    pub const fn from_value_type(value_type: PcuValueType) -> Option<Self> {
        match value_type {
            PcuValueType::Scalar(PcuScalarType::U8) => Some(Self::U8),
            PcuValueType::Scalar(PcuScalarType::U16) => Some(Self::U16),
            PcuValueType::Scalar(PcuScalarType::U32) => Some(Self::U32),
            _ => None,
        }
    }
}

/// One semantic parameterized stream function/pattern.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PcuStreamPattern {
    BitReverse,
    BitInvert,
    Increment,
    Decrement,
    AddParameter { parameter: PcuParameterSlot },
    XorParameter { parameter: PcuParameterSlot },
    ShiftLeft { bits: u8 },
    ShiftRight { bits: u8 },
    ExtractBits { offset: u8, width: u8 },
    MaskLower { bits: u8 },
    ByteSwap32,
}

impl PcuStreamPattern {
    #[must_use]
    pub const fn is_specialized(self) -> bool {
        matches!(
            self,
            Self::ShiftLeft { .. }
                | Self::ShiftRight { .. }
                | Self::ExtractBits { .. }
                | Self::MaskLower { .. }
        )
    }

    #[must_use]
    pub const fn supports_value_type(self, value_type: PcuStreamValueType) -> bool {
        let bit_width = value_type.bit_width();
        match self {
            Self::BitReverse
            | Self::BitInvert
            | Self::Increment
            | Self::Decrement
            | Self::AddParameter { .. }
            | Self::XorParameter { .. } => true,
            Self::ShiftLeft { bits } | Self::ShiftRight { bits } => bits >= 1 && bits <= bit_width,
            Self::ExtractBits { offset, width } => {
                width >= 1
                    && width <= bit_width
                    && offset < bit_width
                    && (offset as u16 + width as u16) <= bit_width as u16
            }
            Self::MaskLower { bits } => bits >= 1 && bits <= bit_width,
            Self::ByteSwap32 => matches!(value_type, PcuStreamValueType::U32),
        }
    }

    #[must_use]
    pub const fn support_flag(self) -> PcuStreamCapabilities {
        match self {
            Self::BitReverse => PcuStreamCapabilities::BIT_REVERSE,
            Self::BitInvert => PcuStreamCapabilities::BIT_INVERT,
            Self::Increment => PcuStreamCapabilities::INCREMENT,
            Self::Decrement => PcuStreamCapabilities::DECREMENT,
            Self::AddParameter { .. } => PcuStreamCapabilities::ADD_PARAMETER,
            Self::XorParameter { .. } => PcuStreamCapabilities::XOR_PARAMETER,
            Self::ShiftLeft { .. } => PcuStreamCapabilities::SHIFT_LEFT,
            Self::ShiftRight { .. } => PcuStreamCapabilities::SHIFT_RIGHT,
            Self::ExtractBits { .. } => PcuStreamCapabilities::EXTRACT_BITS,
            Self::MaskLower { .. } => PcuStreamCapabilities::MASK_LOWER,
            Self::ByteSwap32 => PcuStreamCapabilities::BYTE_SWAP32,
        }
    }
}

/// Coarse stream-profile capabilities required by one program unit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct PcuStreamCapabilities(u32);

impl PcuStreamCapabilities {
    pub const FIFO_INPUT: Self = Self(1 << 0);
    pub const FIFO_OUTPUT: Self = Self(1 << 1);
    pub const BIT_REVERSE: Self = Self(1 << 2);
    pub const BIT_INVERT: Self = Self(1 << 3);
    pub const INCREMENT: Self = Self(1 << 4);
    pub const DECREMENT: Self = Self(1 << 5);
    pub const ADD_PARAMETER: Self = Self(1 << 6);
    pub const XOR_PARAMETER: Self = Self(1 << 7);
    pub const SHIFT_LEFT: Self = Self(1 << 8);
    pub const SHIFT_RIGHT: Self = Self(1 << 9);
    pub const EXTRACT_BITS: Self = Self(1 << 10);
    pub const MASK_LOWER: Self = Self(1 << 11);
    pub const BYTE_SWAP32: Self = Self(1 << 12);
    pub const PRECISE_DELAY: Self = Self(1 << 13);
    pub const PIN_PARALLEL_OUTPUT: Self = Self(1 << 14);

    #[must_use]
    pub const fn empty() -> Self {
        Self(0)
    }

    #[must_use]
    pub const fn all() -> Self {
        Self::FIFO_INPUT
            .union(Self::FIFO_OUTPUT)
            .union(Self::BIT_REVERSE)
            .union(Self::BIT_INVERT)
            .union(Self::INCREMENT)
            .union(Self::DECREMENT)
            .union(Self::ADD_PARAMETER)
            .union(Self::XOR_PARAMETER)
            .union(Self::SHIFT_LEFT)
            .union(Self::SHIFT_RIGHT)
            .union(Self::EXTRACT_BITS)
            .union(Self::MASK_LOWER)
            .union(Self::BYTE_SWAP32)
            .union(Self::PRECISE_DELAY)
            .union(Self::PIN_PARALLEL_OUTPUT)
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

impl BitOr for PcuStreamCapabilities {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        Self(self.0 | rhs.0)
    }
}

impl BitOrAssign for PcuStreamCapabilities {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

impl BitAnd for PcuStreamCapabilities {
    type Output = Self;

    fn bitand(self, rhs: Self) -> Self::Output {
        Self(self.0 & rhs.0)
    }
}

impl BitAndAssign for PcuStreamCapabilities {
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}

/// Minimal semantic stream-profile IR payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PcuStreamKernelIr<'a> {
    pub id: PcuKernelId,
    pub entry_point: &'a str,
    pub bindings: &'a [crate::PcuBinding<'a>],
    pub ports: &'a [PcuPort<'a>],
    pub parameters: &'a [PcuParameter<'a>],
    pub patterns: &'a [PcuStreamPattern],
    pub capabilities: PcuStreamCapabilities,
}

impl PcuStreamKernelIr<'_> {
    /// Returns the dispatch-policy flags required to route this stream kernel honestly.
    #[must_use]
    pub const fn required_dispatch_policy(&self) -> PcuDispatchPolicyCaps {
        PcuDispatchPolicyCaps::PERSISTENT_INSTALL
    }

    /// Returns the stream-profile support flags required to execute this stream kernel.
    #[must_use]
    pub const fn required_instruction_support(&self) -> PcuStreamCapabilities {
        self.capabilities
    }

    /// Returns the simple input/output transform element type, when this kernel is one unary
    /// typed stream transform.
    #[must_use]
    pub fn simple_transform_type(&self) -> Option<PcuStreamValueType> {
        let [input, output] = self.ports else {
            return None;
        };
        if input.direction != crate::PcuPortDirection::Input
            || output.direction != crate::PcuPortDirection::Output
            || input.rate != crate::PcuPortRate::Stream
            || output.rate != crate::PcuPortRate::Stream
        {
            return None;
        }
        let input_type = PcuStreamValueType::from_value_type(input.value_type)?;
        let output_type = PcuStreamValueType::from_value_type(output.value_type)?;
        (input_type == output_type).then_some(input_type)
    }

    /// Validates that this stream kernel is one honest simple typed unary transform.
    ///
    /// # Errors
    ///
    /// Returns the first contract mismatch that makes the transform dishonest.
    pub fn validate_simple_transform(
        &self,
    ) -> Result<PcuStreamValueType, PcuStreamSimpleTransformValidationError> {
        validate_stream_simple_transform(self)
    }

    /// Returns whether the bound stream patterns are semantically valid for the simple transform
    /// shape carried by this kernel.
    #[must_use]
    pub fn simple_transform_patterns_are_valid(&self) -> bool {
        self.validate_simple_transform().is_ok()
    }

    /// Returns whether the supplied runtime-parameter table satisfies this stream kernel's
    /// declared parameter contract.
    #[must_use]
    pub fn invocation_parameters_are_valid(&self, parameters: PcuInvocationParameters<'_>) -> bool {
        parameters.validate_against(self.parameters)
    }
}

impl PcuKernelIrContract for PcuStreamKernelIr<'_> {
    fn id(&self) -> PcuKernelId {
        self.id
    }

    fn kind(&self) -> PcuIrKind {
        PcuIrKind::Stream
    }

    fn entry_point(&self) -> &str {
        self.entry_point
    }

    fn signature(&self) -> PcuKernelSignature<'_> {
        PcuKernelSignature {
            bindings: self.bindings,
            ports: self.ports,
            parameters: self.parameters,
            invocation: PcuInvocationModel::continuous(),
        }
    }
}

pub use crate::validation::PcuStreamSimpleTransformValidationError;

const BYTE_STREAM_PORTS: [PcuPort<'static>; 2] = [
    PcuPort::stream_input(Some("input"), PcuStreamValueType::U8.as_value_type()),
    PcuPort::stream_output(Some("output"), PcuStreamValueType::U8.as_value_type()),
];

const HALF_WORD_STREAM_PORTS: [PcuPort<'static>; 2] = [
    PcuPort::stream_input(Some("input"), PcuStreamValueType::U16.as_value_type()),
    PcuPort::stream_output(Some("output"), PcuStreamValueType::U16.as_value_type()),
];

const WORD_STREAM_PORTS: [PcuPort<'static>; 2] = [
    PcuPort::stream_input(Some("input"), PcuStreamValueType::U32.as_value_type()),
    PcuPort::stream_output(Some("output"), PcuStreamValueType::U32.as_value_type()),
];

/// Builder for one backend-neutral stream kernel.
#[derive(Debug, Clone, Copy)]
pub struct PcuStreamKernelBuilder<'a, const MAX_PATTERNS: usize = DEFAULT_PATTERN_CAPACITY> {
    kernel_id: PcuKernelId,
    entry_point: &'a str,
    value_type: PcuStreamValueType,
    parameters: &'a [PcuParameter<'a>],
    patterns: [PcuStreamPattern; MAX_PATTERNS],
    pattern_len: usize,
    capabilities: PcuStreamCapabilities,
}

impl<'a, const MAX_PATTERNS: usize> PcuStreamKernelBuilder<'a, MAX_PATTERNS> {
    /// Creates one `u8` stream-kernel builder.
    #[must_use]
    pub fn bytes(kernel_id: u32, entry_point: &'a str) -> Self {
        Self::new(PcuKernelId(kernel_id), entry_point, PcuStreamValueType::U8)
    }

    /// Creates one `u16` stream-kernel builder.
    #[must_use]
    pub fn half_words(kernel_id: u32, entry_point: &'a str) -> Self {
        Self::new(PcuKernelId(kernel_id), entry_point, PcuStreamValueType::U16)
    }

    /// Creates one `u32` stream-kernel builder.
    #[must_use]
    pub fn words(kernel_id: u32, entry_point: &'a str) -> Self {
        Self::new(PcuKernelId(kernel_id), entry_point, PcuStreamValueType::U32)
    }

    fn new(kernel_id: PcuKernelId, entry_point: &'a str, value_type: PcuStreamValueType) -> Self {
        Self {
            kernel_id,
            entry_point,
            value_type,
            parameters: &[],
            patterns: [PcuStreamPattern::BitReverse; MAX_PATTERNS],
            pattern_len: 0,
            capabilities: PcuStreamCapabilities::FIFO_INPUT | PcuStreamCapabilities::FIFO_OUTPUT,
        }
    }

    /// Returns the stable kernel id.
    #[must_use]
    pub const fn kernel_id(&self) -> PcuKernelId {
        self.kernel_id
    }

    /// Returns the entry-point label.
    #[must_use]
    pub const fn entry_point(&self) -> &str {
        self.entry_point
    }

    /// Returns the stream element type.
    #[must_use]
    pub const fn value_type(&self) -> PcuStreamValueType {
        self.value_type
    }

    /// Returns the declared runtime parameter slice.
    #[must_use]
    pub const fn parameters(&self) -> &'a [PcuParameter<'a>] {
        self.parameters
    }

    /// Returns the inferred capability set.
    #[must_use]
    pub const fn capabilities(&self) -> PcuStreamCapabilities {
        self.capabilities
    }

    /// Returns the current pattern count.
    #[must_use]
    pub const fn pattern_count(&self) -> usize {
        self.pattern_len
    }

    /// Returns the configured patterns.
    #[must_use]
    pub fn patterns(&self) -> &[PcuStreamPattern] {
        &self.patterns[..self.pattern_len]
    }

    /// Replaces the declared runtime parameter slice.
    #[must_use]
    pub const fn with_parameters(mut self, parameters: &'a [PcuParameter<'a>]) -> Self {
        self.parameters = parameters;
        self
    }

    /// Appends one stream pattern.
    ///
    /// # Errors
    ///
    /// Returns `ResourceExhausted` when the builder pattern capacity is exhausted.
    pub fn with_pattern(mut self, pattern: PcuStreamPattern) -> Result<Self, PcuError> {
        self.push_pattern(pattern)?;
        Ok(self)
    }

    /// Appends several stream patterns in order.
    ///
    /// # Errors
    ///
    /// Returns `ResourceExhausted` when the builder pattern capacity is exhausted.
    pub fn with_patterns(mut self, patterns: &[PcuStreamPattern]) -> Result<Self, PcuError> {
        for pattern in patterns.iter().copied() {
            self.push_pattern(pattern)?;
        }
        Ok(self)
    }

    /// Appends one `BitReverse` transform.
    ///
    /// # Errors
    ///
    /// Returns `ResourceExhausted` when the builder pattern capacity is exhausted.
    pub fn bit_reverse(self) -> Result<Self, PcuError> {
        self.with_pattern(PcuStreamPattern::BitReverse)
    }

    /// Appends one `BitInvert` transform.
    ///
    /// # Errors
    ///
    /// Returns `ResourceExhausted` when the builder pattern capacity is exhausted.
    pub fn bit_invert(self) -> Result<Self, PcuError> {
        self.with_pattern(PcuStreamPattern::BitInvert)
    }

    /// Appends one `Increment` transform.
    ///
    /// # Errors
    ///
    /// Returns `ResourceExhausted` when the builder pattern capacity is exhausted.
    pub fn increment(self) -> Result<Self, PcuError> {
        self.with_pattern(PcuStreamPattern::Increment)
    }

    /// Appends one `Decrement` transform.
    ///
    /// # Errors
    ///
    /// Returns `ResourceExhausted` when the builder pattern capacity is exhausted.
    pub fn decrement(self) -> Result<Self, PcuError> {
        self.with_pattern(PcuStreamPattern::Decrement)
    }

    /// Appends one left-shift transform.
    ///
    /// # Errors
    ///
    /// Returns `ResourceExhausted` when the builder pattern capacity is exhausted.
    pub fn shift_left(self, bits: u8) -> Result<Self, PcuError> {
        self.with_pattern(PcuStreamPattern::ShiftLeft { bits })
    }

    /// Appends one right-shift transform.
    ///
    /// # Errors
    ///
    /// Returns `ResourceExhausted` when the builder pattern capacity is exhausted.
    pub fn shift_right(self, bits: u8) -> Result<Self, PcuError> {
        self.with_pattern(PcuStreamPattern::ShiftRight { bits })
    }

    /// Appends one bit-extraction transform.
    ///
    /// # Errors
    ///
    /// Returns `ResourceExhausted` when the builder pattern capacity is exhausted.
    pub fn extract_bits(self, offset: u8, width: u8) -> Result<Self, PcuError> {
        self.with_pattern(PcuStreamPattern::ExtractBits { offset, width })
    }

    /// Appends one low-mask transform.
    ///
    /// # Errors
    ///
    /// Returns `ResourceExhausted` when the builder pattern capacity is exhausted.
    pub fn mask_lower(self, bits: u8) -> Result<Self, PcuError> {
        self.with_pattern(PcuStreamPattern::MaskLower { bits })
    }

    /// Appends one byte-swap transform.
    ///
    /// # Errors
    ///
    /// Returns `ResourceExhausted` when the builder pattern capacity is exhausted.
    pub fn byte_swap32(self) -> Result<Self, PcuError> {
        self.with_pattern(PcuStreamPattern::ByteSwap32)
    }

    /// Builds the stream-kernel IR payload.
    #[must_use]
    pub fn ir(&self) -> PcuStreamKernelIr<'_> {
        PcuStreamKernelIr {
            id: self.kernel_id,
            entry_point: self.entry_point,
            bindings: &[],
            ports: self.ports(),
            parameters: self.parameters,
            patterns: &self.patterns[..self.pattern_len],
            capabilities: self.capabilities,
        }
    }

    /// Builds the generic kernel wrapper.
    #[must_use]
    pub fn kernel(&self) -> PcuKernel<'_> {
        PcuKernel::Stream(self.ir())
    }

    fn push_pattern(&mut self, pattern: PcuStreamPattern) -> Result<(), PcuError> {
        if self.pattern_len == MAX_PATTERNS {
            return Err(PcuError::resource_exhausted());
        }
        self.patterns[self.pattern_len] = pattern;
        self.pattern_len += 1;
        self.capabilities |= required_capabilities(pattern);
        Ok(())
    }

    fn ports(&self) -> &'static [PcuPort<'static>; 2] {
        match self.value_type {
            PcuStreamValueType::U8 => &BYTE_STREAM_PORTS,
            PcuStreamValueType::U16 => &HALF_WORD_STREAM_PORTS,
            PcuStreamValueType::U32 => &WORD_STREAM_PORTS,
        }
    }
}

const fn required_capabilities(pattern: PcuStreamPattern) -> PcuStreamCapabilities {
    pattern.support_flag()
}

#[cfg(test)]
mod tests {
    use super::{
        PcuStreamCapabilities,
        PcuStreamKernelBuilder,
        PcuStreamValueType,
    };
    use crate::{
        PcuIrKind,
        PcuKernel,
        PcuKernelIrContract,
    };

    #[test]
    fn builder_synthesizes_word_stream_kernel() {
        let builder = PcuStreamKernelBuilder::<4>::words(0x33, "byte_swap")
            .byte_swap32()
            .expect("builder should accept one pattern");
        let kernel = builder.ir();

        assert_eq!(kernel.id.0, 0x33);
        assert_eq!(kernel.kind(), PcuIrKind::Stream);
        assert_eq!(builder.value_type(), PcuStreamValueType::U32);
        assert_eq!(builder.pattern_count(), 1);
        assert!(
            builder
                .capabilities()
                .contains(PcuStreamCapabilities::FIFO_INPUT | PcuStreamCapabilities::FIFO_OUTPUT)
        );
        assert!(
            builder
                .capabilities()
                .contains(PcuStreamCapabilities::BYTE_SWAP32)
        );
        assert!(kernel.simple_transform_patterns_are_valid());
    }

    #[test]
    fn builder_wraps_generic_kernel() {
        let builder = PcuStreamKernelBuilder::<2>::bytes(7, "bit_reverse")
            .bit_reverse()
            .expect("builder should accept one pattern");

        let kernel = builder.kernel();
        match kernel {
            PcuKernel::Stream(stream) => {
                assert_eq!(stream.kind(), PcuIrKind::Stream);
                assert_eq!(stream.id.0, 7);
            }
            _ => panic!("expected stream kernel"),
        }
    }
}
