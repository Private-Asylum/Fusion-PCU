//! Ordered checked F32/F64 constituent arithmetic without native floating instructions.
#[path = "bytecode/bytecode.rs"]
mod bytecode;
#[rustfmt::skip]
use fusion_pcu::{
    PcuScalarType,
    PcuValueType,
    PcuFloatUnderflowPolicy,
    u64_to_f32_nearest_even,
    u64_to_f64_nearest_even,
    widen_f32_exact,
};
#[rustfmt::skip]
use crate::{
    PcuSpirvError,
    PcuSpirvSink,
    PcuSpirvLoweringOptions,
    PcuSpirvModuleInfo,
    PcuSpirvCapability,
    PcuSpirvCapabilityCaps,
    PcuSpirvVersion,
};

/// Exact ordered compound operation; the graph caller additionally proves Strict/Reject.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PcuSpirvCompoundOperation {
    /// Rounded checked rate times gradient, then rounded checked weight minus product.
    Sgd {
        /// Exact logical length of each input and output.
        count: u32,
        /// Core's finite F32 learning-rate carrier, widened exactly for F64 execution.
        learning_rate: f32,
    },
    /// Flattened subtraction, square and addition in increasing order, then division.
    MeanSquaredError {
        /// Nonzero common input length; output is one scalar.
        count: u32,
    },
    /// Each row-major output uses increasing reduction, multiply then add from positive zero.
    MatMul {
        /// Output row count.
        rows: u32,
        /// Reduction length.
        inner: u32,
        /// Output column count.
        columns: u32,
        /// Source left storage is inner by rows when true.
        transpose_left: bool,
        /// Source right storage is columns by inner when true.
        transpose_right: bool,
    },
}

/// Frozen exact spans for bindings left/right/output/status; status has three U32 per output.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PcuSpirvCompoundProfile {
    /// Logical native byte spans; zero-byte input storage still needs a valid descriptor.
    pub binding_bytes: [u32; 4],
    /// Row-major output length and complete status-record count.
    pub output_count: u32,
}

/// Emits integer-synthesized ordered constituent arithmetic for an exact native compound.
///
/// Bindings0/1/2/3 are left/right/private output/status. Each status record contains kind,
/// reduction index and step (0 multiply,1 add,2 subtract,3 divide). The caller dispatches
/// `ceil(output_count/64)`, scans row-major records after completion and never publishes a
/// fatal output. This primitive alone does not certify Boundary, graph Clamp or Portable.
///
/// # Errors
/// Rejects unsupported type, zero/overflowing output, byte extent, version/capability or sink
/// failure. Unsupported profiles emit no words; sink failure may retain a supported prefix.
pub fn lower_checked_compound_to_spirv<S: PcuSpirvSink>(
    scalar: PcuScalarType,
    operation: PcuSpirvCompoundOperation,
    underflow: PcuFloatUnderflowPolicy,
    options: PcuSpirvLoweringOptions,
    sink: &mut S,
) -> Result<(PcuSpirvModuleInfo, PcuSpirvCompoundProfile), PcuSpirvError> {
    let (words, offsets) = match scalar {
        PcuScalarType::F32 => (bytecode::F32_WORDS, bytecode::F32_OFFSETS.as_slice()),
        PcuScalarType::F64 => (bytecode::F64_WORDS, bytecode::F64_OFFSETS.as_slice()),
        _ => {
            return Err(PcuSpirvError::UnsupportedValueType(PcuValueType::Scalar(
                scalar,
            )));
        }
    };
    let (profile, values) = configure(scalar, operation, underflow)?;
    if !matches!(
        options.version,
        PcuSpirvVersion::V1_0 | PcuSpirvVersion::V1_3
    ) {
        return Err(PcuSpirvError::UnsupportedVersion(options.version));
    }
    if !options
        .capabilities
        .contains(PcuSpirvCapabilityCaps::SHADER)
    {
        return Err(PcuSpirvError::UnsupportedCapability(
            PcuSpirvCapability::Shader,
        ));
    }
    for (index, word) in words.iter().copied().enumerate() {
        let word = if index == 1 {
            options.version.0
        } else if index == 2 {
            options.generator
        } else if let Some(slot) = offsets.iter().position(|offset| *offset == index) {
            values[slot]
        } else {
            word
        };
        sink.push_word(word)?;
    }
    Ok((
        PcuSpirvModuleInfo {
            version: options.version,
            bound: words[3],
            word_count: words.len(),
            capabilities: PcuSpirvCapabilityCaps::SHADER,
        },
        profile,
    ))
}

fn configure(
    scalar: PcuScalarType,
    operation: PcuSpirvCompoundOperation,
    underflow: PcuFloatUnderflowPolicy,
) -> Result<(PcuSpirvCompoundProfile, [u32; 9]), PcuSpirvError> {
    let policy = match underflow {
        PcuFloatUnderflowPolicy::IeeeAfterRounding => 0,
        PcuFloatUnderflowPolicy::RejectSubnormalResult => 1,
        PcuFloatUnderflowPolicy::AllowGradualUnderflow => 2,
    };
    let (counts, mut values) = match operation {
        PcuSpirvCompoundOperation::Sgd {
            count,
            learning_rate,
        } => {
            if !learning_rate.is_finite() {
                return Err(PcuSpirvError::InvalidBinding);
            }
            let factor = if scalar == PcuScalarType::F32 {
                u64::from(learning_rate.to_bits())
            } else {
                widen_f32_exact(learning_rate).to_bits()
            };
            (
                [count; 3],
                [0, policy, 1, count, 1, 0, 0, low(factor), high(factor)],
            )
        }
        PcuSpirvCompoundOperation::MeanSquaredError { count } => {
            if count == 0 {
                return Err(PcuSpirvError::InvalidBinding);
            }
            let factor = denominator(scalar, count);
            (
                [count, count, 1],
                [2, policy, 1, count, 1, 0, 0, low(factor), high(factor)],
            )
        }
        PcuSpirvCompoundOperation::MatMul {
            rows,
            inner,
            columns,
            transpose_left,
            transpose_right,
        } => (
            [
                rows.checked_mul(inner)
                    .ok_or(PcuSpirvError::InvalidBinding)?,
                inner
                    .checked_mul(columns)
                    .ok_or(PcuSpirvError::InvalidBinding)?,
                rows.checked_mul(columns)
                    .ok_or(PcuSpirvError::InvalidBinding)?,
            ],
            [
                1,
                policy,
                rows,
                inner,
                columns,
                u32::from(transpose_left),
                u32::from(transpose_right),
                0,
                0,
            ],
        ),
    };
    if counts[2] == 0 {
        return Err(PcuSpirvError::InvalidBinding);
    }
    // Exact scalar and three-word status spans prove every shader product/index is U32-bounded.
    let bytes = u32::from(scalar.bit_width() / 8);
    let binding_bytes = [
        counts[0]
            .checked_mul(bytes)
            .ok_or(PcuSpirvError::InvalidBinding)?,
        counts[1]
            .checked_mul(bytes)
            .ok_or(PcuSpirvError::InvalidBinding)?,
        counts[2]
            .checked_mul(bytes)
            .ok_or(PcuSpirvError::InvalidBinding)?,
        counts[2]
            .checked_mul(12)
            .ok_or(PcuSpirvError::InvalidBinding)?,
    ];
    // F32's high factor does not exist in its module; retain a canonical unused tuple field.
    if scalar == PcuScalarType::F32 {
        values[8] = 0;
    }
    Ok((
        PcuSpirvCompoundProfile {
            binding_bytes,
            output_count: counts[2],
        },
        values,
    ))
}

const fn low(bits: u64) -> u32 {
    let bytes = bits.to_le_bytes();
    u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
}
const fn high(bits: u64) -> u32 {
    let bytes = bits.to_le_bytes();
    u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]])
}
// Named MSE metadata uses fixed RN-even conversion, independently of host FP control.
fn denominator(scalar: PcuScalarType, count: u32) -> u64 {
    if scalar == PcuScalarType::F32 {
        u64::from(u64_to_f32_nearest_even(u64::from(count)).to_bits())
    } else {
        u64_to_f64_nearest_even(u64::from(count)).to_bits()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn unequal_compound_spans_and_status_use_only_shader_u32() {
        for scalar in [PcuScalarType::F32, PcuScalarType::F64] {
            for operation in [
                PcuSpirvCompoundOperation::Sgd {
                    count: 65,
                    learning_rate: 0.5,
                },
                PcuSpirvCompoundOperation::MeanSquaredError { count: 65 },
                PcuSpirvCompoundOperation::MatMul {
                    rows: 3,
                    inner: 13,
                    columns: 5,
                    transpose_left: true,
                    transpose_right: true,
                },
            ] {
                for policy in [
                    PcuFloatUnderflowPolicy::IeeeAfterRounding,
                    PcuFloatUnderflowPolicy::AllowGradualUnderflow,
                    PcuFloatUnderflowPolicy::RejectSubnormalResult,
                ] {
                    let mut words = alloc::vec::Vec::new();
                    let (info, profile) = lower_checked_compound_to_spirv(
                        scalar,
                        operation,
                        policy,
                        PcuSpirvLoweringOptions::default(),
                        &mut words,
                    )
                    .unwrap();
                    assert_eq!(info.capabilities, PcuSpirvCapabilityCaps::SHADER);
                    assert_eq!(profile.binding_bytes[3], profile.output_count * 12);
                    let mut cursor = 5;
                    while cursor < words.len() {
                        let count = (words[cursor] >> 16) as usize;
                        match words[cursor] & 0xffff {
                            17 => assert_eq!(words[cursor + 1], 1),
                            21 => assert_eq!(words[cursor + 2], 32),
                            22 => panic!("compound declares a native floating type"),
                            _ => {}
                        }
                        cursor += count;
                    }
                }
            }
        }
    }
    #[test]
    fn unsupported_or_overflowing_profiles_emit_nothing() {
        for (scalar, operation) in [
            (
                PcuScalarType::F16,
                PcuSpirvCompoundOperation::Sgd {
                    count: 1,
                    learning_rate: 0.5,
                },
            ),
            (
                PcuScalarType::F128,
                PcuSpirvCompoundOperation::MeanSquaredError { count: 65 },
            ),
            (
                PcuScalarType::F64,
                PcuSpirvCompoundOperation::Sgd {
                    count: 0,
                    learning_rate: 0.5,
                },
            ),
            (
                PcuScalarType::F32,
                PcuSpirvCompoundOperation::Sgd {
                    count: u32::MAX / 12 + 1,
                    learning_rate: 0.5,
                },
            ),
            (
                PcuScalarType::F64,
                PcuSpirvCompoundOperation::MatMul {
                    rows: u32::MAX,
                    inner: 2,
                    columns: 1,
                    transpose_left: false,
                    transpose_right: false,
                },
            ),
        ] {
            let mut words = alloc::vec::Vec::new();
            assert!(
                lower_checked_compound_to_spirv(
                    scalar,
                    operation,
                    PcuFloatUnderflowPolicy::default(),
                    PcuSpirvLoweringOptions::default(),
                    &mut words
                )
                .is_err()
            );
            assert!(words.is_empty());
        }
    }
}
