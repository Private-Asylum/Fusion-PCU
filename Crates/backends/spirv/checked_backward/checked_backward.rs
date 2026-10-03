//! Six-format checked owned `ReLU` derivative lowering, independent of dispatch map admission.
#[path = "bytecode/bytecode.rs"]
mod bytecode;
#[rustfmt::skip]
use fusion_pcu::{
    PcuScalarType,
    PcuValueType,
    PcuFloatUnderflowPolicy,
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

/// Emits the dense two-input checked `ReLU` derivative with a logical U32 status per lane.
///
/// The caller supplies actual input/upstream/output/status storage bindings0/1/2/3 and
/// dispatches `ceil(extent/packing_lanes/64)` groups. Every nonfinite operand rejects, even
/// when the input is inactive; exact selected bits retain signed zero. Only Tight rejects
/// a selected subnormal result. No general dispatch, graph Clamp or Portable capability
/// is certified by this primitive. The owning graph compiler checks node requirements.
///
/// # Errors
/// Rejects unsupported scalar, zero/overflowing extent, version/capability or sink failure.
/// Unsupported profiles emit no words. Sink failure can retain a supported partial module.
pub fn lower_checked_relu_backward_to_spirv<S: PcuSpirvSink>(
    scalar: PcuScalarType,
    underflow: PcuFloatUnderflowPolicy,
    extent: u32,
    options: PcuSpirvLoweringOptions,
    sink: &mut S,
) -> Result<PcuSpirvModuleInfo, PcuSpirvError> {
    let format = match scalar {
        PcuScalarType::F16 => 0,
        PcuScalarType::BF16 => 1,
        PcuScalarType::F8E4M3FN => 2,
        PcuScalarType::F8E5M2 => 3,
        PcuScalarType::F32 => 4,
        PcuScalarType::F64 => 5,
        _ => {
            return Err(PcuSpirvError::UnsupportedValueType(PcuValueType::Scalar(
                scalar,
            )));
        }
    };
    let bytes = u32::from(scalar.bit_width() / 8).max(4);
    // Status consumes4 bytes per logical lane, F64 consumes8. This also proves packed
    // tail/base products and F64 doubled indices cannot wrap in the U32 shader.
    if extent == 0 || extent > u32::MAX / bytes {
        return Err(PcuSpirvError::InvalidBinding);
    }
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
    let values = [
        format,
        match underflow {
            PcuFloatUnderflowPolicy::IeeeAfterRounding => 0,
            PcuFloatUnderflowPolicy::RejectSubnormalResult => 1,
            PcuFloatUnderflowPolicy::AllowGradualUnderflow => 2,
        },
        extent,
    ];
    let words = bytecode::WORDS;
    for (index, word) in words.iter().copied().enumerate() {
        let word = if index == 1 {
            options.version.0
        } else if index == 2 {
            options.generator
        } else if let Some(slot) = bytecode::OFFSETS.iter().position(|offset| *offset == index) {
            values[slot]
        } else {
            word
        };
        sink.push_word(word)?;
    }
    Ok(PcuSpirvModuleInfo {
        version: options.version,
        bound: words[3],
        word_count: words.len(),
        capabilities: PcuSpirvCapabilityCaps::SHADER,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn six_formats_and_three_policies_have_only_shader_u32_abi() {
        for scalar in [
            PcuScalarType::F16,
            PcuScalarType::BF16,
            PcuScalarType::F8E4M3FN,
            PcuScalarType::F8E5M2,
            PcuScalarType::F32,
            PcuScalarType::F64,
        ] {
            for policy in [
                PcuFloatUnderflowPolicy::IeeeAfterRounding,
                PcuFloatUnderflowPolicy::RejectSubnormalResult,
                PcuFloatUnderflowPolicy::AllowGradualUnderflow,
            ] {
                let mut words = alloc::vec::Vec::new();
                let info = lower_checked_relu_backward_to_spirv(
                    scalar,
                    policy,
                    65,
                    PcuSpirvLoweringOptions::default(),
                    &mut words,
                )
                .unwrap();
                assert_eq!(info.capabilities, PcuSpirvCapabilityCaps::SHADER);
                assert_eq!(words[bytecode::OFFSETS[2]], 65);
                let mut cursor = 5;
                while cursor < words.len() {
                    let count = (words[cursor] >> 16) as usize;
                    if words[cursor] & 0xffff == 17 {
                        assert_eq!(words[cursor + 1], 1);
                    }
                    if words[cursor] & 0xffff == 21 {
                        assert_eq!(words[cursor + 2], 32);
                    }
                    assert_ne!(words[cursor] & 0xffff, 22); // No floating type declaration.
                    cursor += count;
                }
            }
        }
    }
    #[test]
    fn unsupported_profiles_leave_sink_empty() {
        for (scalar, extent) in [
            (PcuScalarType::I32, 65),
            (PcuScalarType::F128, 65),
            (PcuScalarType::F16, 0),
            (PcuScalarType::F64, u32::MAX / 8 + 1),
        ] {
            let mut words = alloc::vec::Vec::new();
            assert!(
                lower_checked_relu_backward_to_spirv(
                    scalar,
                    PcuFloatUnderflowPolicy::default(),
                    extent,
                    PcuSpirvLoweringOptions::default(),
                    &mut words
                )
                .is_err()
            );
            assert!(words.is_empty());
        }
    }
}
