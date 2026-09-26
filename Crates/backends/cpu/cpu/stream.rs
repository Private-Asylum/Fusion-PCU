use fusion_pcu::{
    PcuStreamKernelIr,
    PcuStreamPattern,
    PcuStreamValueType,
};

/// Failure to interpret the normative one-pattern U32 stream profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PcuU32StreamReferenceError {
    InvalidPortShape,
    KernelBindingsPresent,
    ParametersPresent,
    InvalidPatternCount(usize),
    UnsupportedPattern(PcuStreamPattern),
    InvalidPattern(PcuStreamPattern),
}

/// Stateless, one-word CPU reference for the common U32 Stream transform profile.
///
/// A call consumes one logical input word and returns its corresponding output word. FIFO
/// framing, buffering, and end-of-stream behavior remain the responsibility of an adapter.
#[derive(Debug, Default, Clone, Copy)]
pub struct PcuU32StreamReference;

impl PcuU32StreamReference {
    /// Validates and applies the kernel's single U32 transform to one word.
    ///
    /// # Errors
    ///
    /// Returns the first profile shape or pattern error.
    pub fn transform(
        &self,
        kernel: &PcuStreamKernelIr<'_>,
        input: u32,
    ) -> Result<u32, PcuU32StreamReferenceError> {
        if kernel.simple_transform_type() != Some(PcuStreamValueType::U32) {
            return Err(PcuU32StreamReferenceError::InvalidPortShape);
        }
        if !kernel.bindings.is_empty() {
            return Err(PcuU32StreamReferenceError::KernelBindingsPresent);
        }
        if !kernel.parameters.is_empty() {
            return Err(PcuU32StreamReferenceError::ParametersPresent);
        }
        let [pattern] = kernel.patterns else {
            return Err(PcuU32StreamReferenceError::InvalidPatternCount(
                kernel.patterns.len(),
            ));
        };
        apply_u32_stream_pattern(*pattern, input)
    }
}

fn apply_u32_stream_pattern(
    pattern: PcuStreamPattern,
    input: u32,
) -> Result<u32, PcuU32StreamReferenceError> {
    let value = match pattern {
        PcuStreamPattern::BitReverse => input.reverse_bits(),
        PcuStreamPattern::BitInvert => !input,
        PcuStreamPattern::Increment => input.wrapping_add(1),
        PcuStreamPattern::Decrement => input.wrapping_sub(1),
        PcuStreamPattern::ShiftLeft { bits } if (1..=32).contains(&bits) => {
            if bits == 32 {
                0
            } else {
                input << bits
            }
        }
        PcuStreamPattern::ShiftRight { bits } if (1..=32).contains(&bits) => {
            if bits == 32 {
                0
            } else {
                input >> bits
            }
        }
        PcuStreamPattern::ExtractBits { offset, width }
            if width >= 1 && offset < 32 && u16::from(offset) + u16::from(width) <= 32 =>
        {
            let mask = if width == 32 {
                u32::MAX
            } else {
                (1_u32 << width) - 1
            };
            (input >> offset) & mask
        }
        PcuStreamPattern::MaskLower { bits } if (1..=32).contains(&bits) => {
            if bits == 32 {
                input
            } else {
                input & ((1_u32 << bits) - 1)
            }
        }
        PcuStreamPattern::ByteSwap32 => input.swap_bytes(),
        PcuStreamPattern::AddParameter { .. } | PcuStreamPattern::XorParameter { .. } => {
            return Err(PcuU32StreamReferenceError::UnsupportedPattern(pattern));
        }
        _ => return Err(PcuU32StreamReferenceError::InvalidPattern(pattern)),
    };
    Ok(value)
}
