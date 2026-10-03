//! Static call-target rewriting and removal of unused native call/parameter slots.
#[path = "../bytecode/f32.rs"]
mod f32;
#[path = "../bytecode/f64.rs"]
mod f64;
#[path = "../bytecode/low.rs"]
mod low;
#[rustfmt::skip]
use crate::{
    PcuSpirvCapability,
    PcuSpirvCapabilityCaps,
    PcuSpirvLoweringOptions,
    PcuSpirvModuleInfo,
    PcuSpirvSink,
    PcuSpirvVersion,
};
use fusion_pcu::PcuScalarType;
#[rustfmt::skip]
use super::{
    PcuSpirvComposedFloatProfile as Profile,
    PcuSpirvError as Error,
};

struct Template {
    words: &'static [u32],
    functions: &'static [u32],
    calls: &'static [usize],
    arguments: &'static [u32],
    defaults: &'static [usize],
}
const fn template(scalar: PcuScalarType) -> Template {
    match scalar {
        PcuScalarType::F32 => Template {
            words: f32::WORDS,
            functions: f32::FUNCTION_IDS,
            calls: f32::CALL_TARGET_OFFSETS,
            arguments: f32::CALL_ARGUMENT_IDS,
            defaults: f32::SPECIALIZATION_OFFSETS,
        },
        PcuScalarType::F64 => Template {
            words: f64::WORDS,
            functions: f64::FUNCTION_IDS,
            calls: f64::CALL_TARGET_OFFSETS,
            arguments: f64::CALL_ARGUMENT_IDS,
            defaults: f64::SPECIALIZATION_OFFSETS,
        },
        _ => Template {
            words: low::WORDS,
            functions: low::FUNCTION_IDS,
            calls: low::CALL_TARGET_OFFSETS,
            arguments: low::CALL_ARGUMENT_IDS,
            defaults: low::SPECIALIZATION_OFFSETS,
        },
    }
}

fn constants(profile: &Profile) -> Result<[u32; 522], Error> {
    let mut values = [0; 522];
    values[0] = profile.extent;
    values[1] = match profile.scalar {
        PcuScalarType::BF16 => 1,
        PcuScalarType::F8E4M3FN => 2,
        PcuScalarType::F8E5M2 => 3,
        _ => 0,
    };
    for (bank, resource) in profile.resources().iter().enumerate() {
        let bytes = u64::from(resource.read_elements) * u64::from(profile.scalar.bit_width() / 8);
        values[2 + bank] =
            u32::try_from(bytes.div_ceil(4)).map_err(|_| Error::InvalidKernelSignature)?;
        values[6 + bank] = u32::from(resource.write_elements != 0);
    }
    for (step, instruction) in profile.steps[..profile.step_count].iter().enumerate() {
        values[10 + step * 8..18 + step * 8].copy_from_slice(&instruction.arguments);
    }
    Ok(values)
}

pub(super) fn lower<S: PcuSpirvSink>(
    profile: &Profile,
    options: PcuSpirvLoweringOptions,
    sink: &mut S,
) -> Result<PcuSpirvModuleInfo, Error> {
    if options.version != PcuSpirvVersion::V1_0 {
        return Err(Error::UnsupportedVersion(options.version));
    }
    if !options
        .capabilities
        .contains(PcuSpirvCapabilityCaps::SHADER)
    {
        return Err(Error::UnsupportedCapability(PcuSpirvCapability::Shader));
    }
    let template = template(profile.scalar);
    let constants = constants(profile)?;
    let removed_arguments = &template.arguments[profile.step_count * 8..];
    let mut words = 0;
    for (index, word) in template.words[..5].iter().copied().enumerate() {
        sink.push_word(if index == 2 { options.generator } else { word })?;
        words += 1;
    }
    let mut cursor = 5;
    while cursor < template.words.len() {
        let instruction = &template.words[cursor..];
        let count =
            usize::try_from(instruction[0] >> 16).map_err(|_| Error::InvalidKernelSignature)?;
        let opcode = instruction[0] & 0xffff;
        let call = template
            .calls
            .iter()
            .position(|offset| *offset == cursor + 3);
        if (opcode == 71 && count == 4 && instruction[2] == 1)
            || call.is_some_and(|slot| slot >= profile.step_count)
            || ((opcode == 5 || opcode == 62) && removed_arguments.contains(&instruction[1]))
            || (opcode == 59 && removed_arguments.contains(&instruction[2]))
        {
            cursor += count;
            continue;
        }
        for (operand, word) in instruction[..count].iter().copied().enumerate() {
            let output = if opcode == 50 && operand == 0 {
                (instruction[0] & 0xffff_0000) | 0x002b // Ordinary immutable OpConstant.
            } else if opcode == 50 && operand == 3 {
                let slot = template
                    .defaults
                    .iter()
                    .position(|offset| *offset == cursor + 3)
                    .ok_or(Error::InvalidKernelSignature)?;
                constants[slot]
            } else if operand == 3
                && let Some(slot) = call
            {
                template.functions[profile.steps[slot].function]
            } else {
                word
            };
            sink.push_word(output)?;
            words += 1;
        }
        cursor += count;
    }
    Ok(PcuSpirvModuleInfo {
        version: options.version,
        bound: template.words[3],
        word_count: words,
        capabilities: PcuSpirvCapabilityCaps::SHADER,
    })
}
