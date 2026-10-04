//! Static call-target rewriting and removal of unused native call/parameter slots.
#[cfg(feature = "embedded-composed")]
#[path = "../bytecode/f32.rs"]
mod f32;
#[cfg(feature = "embedded-composed")]
#[path = "../bytecode/f64.rs"]
mod f64;
#[cfg(feature = "embedded-composed")]
#[path = "../bytecode/integer_narrow.rs"]
mod integer_narrow;
#[cfg(feature = "embedded-composed")]
#[path = "../bytecode/integer_wide.rs"]
mod integer_wide;
#[cfg(feature = "embedded-composed")]
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
    PcuSpirvComposedProfile as Profile,
    PcuSpirvError as Error,
};

#[derive(Clone, Copy)]
struct Template<'a> {
    words: &'a [u32],
    functions: &'a [u32],
    calls: &'a [usize],
    arguments: &'a [u32],
    defaults: &'a [usize],
}
#[cfg(feature = "embedded-composed")]
const fn template(scalar: PcuScalarType) -> Template<'static> {
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
        PcuScalarType::U256 | PcuScalarType::I256 | PcuScalarType::U512 | PcuScalarType::I512 => {
            Template {
                words: integer_wide::WORDS,
                functions: integer_wide::FUNCTION_IDS,
                calls: integer_wide::CALL_TARGET_OFFSETS,
                arguments: integer_wide::CALL_ARGUMENT_IDS,
                defaults: integer_wide::SPECIALIZATION_OFFSETS,
            }
        }
        PcuScalarType::U8
        | PcuScalarType::I8
        | PcuScalarType::U16
        | PcuScalarType::I16
        | PcuScalarType::U32
        | PcuScalarType::I32
        | PcuScalarType::U64
        | PcuScalarType::I64
        | PcuScalarType::U128
        | PcuScalarType::I128 => Template {
            words: integer_narrow::WORDS,
            functions: integer_narrow::FUNCTION_IDS,
            calls: integer_narrow::CALL_TARGET_OFFSETS,
            arguments: integer_narrow::CALL_ARGUMENT_IDS,
            defaults: integer_narrow::SPECIALIZATION_OFFSETS,
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
        PcuScalarType::BF16 | PcuScalarType::I256 | PcuScalarType::I8 => 1,
        PcuScalarType::F8E4M3FN | PcuScalarType::U512 | PcuScalarType::U16 => 2,
        PcuScalarType::F8E5M2 | PcuScalarType::I512 | PcuScalarType::I16 => 3,
        PcuScalarType::U32 => 4,
        PcuScalarType::I32 => 5,
        PcuScalarType::U64 => 6,
        PcuScalarType::I64 => 7,
        PcuScalarType::U128 => 8,
        PcuScalarType::I128 => 9,
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

#[cfg(feature = "embedded-composed")]
pub(super) fn lower<S: PcuSpirvSink>(
    profile: &Profile,
    options: PcuSpirvLoweringOptions,
    sink: &mut S,
) -> Result<PcuSpirvModuleInfo, Error> {
    lower_template(profile, options, sink, template(profile.scalar))
}
#[cfg(not(feature = "embedded-composed"))]
pub(super) const fn lower<S: PcuSpirvSink>(
    profile: &Profile,
    options: PcuSpirvLoweringOptions,
    sink: &mut S,
) -> Result<PcuSpirvModuleInfo, Error> {
    let _ = (profile, options, sink);
    Err(Error::InvalidKernelSignature)
}

pub(super) fn lower_external<S: PcuSpirvSink>(
    profile: &Profile,
    options: PcuSpirvLoweringOptions,
    sink: &mut S,
    validated: super::PcuSpirvComposedTemplate<'_>,
) -> Result<PcuSpirvModuleInfo, Error> {
    let data = validated.data();
    let family = match profile.scalar {
        PcuScalarType::F32 => super::PcuSpirvComposedTemplateFamily::F32,
        PcuScalarType::F64 => super::PcuSpirvComposedTemplateFamily::F64,
        PcuScalarType::U256 | PcuScalarType::I256 | PcuScalarType::U512 | PcuScalarType::I512 => {
            super::PcuSpirvComposedTemplateFamily::IntegerWide
        }
        PcuScalarType::U8
        | PcuScalarType::I8
        | PcuScalarType::U16
        | PcuScalarType::I16
        | PcuScalarType::U32
        | PcuScalarType::I32
        | PcuScalarType::U64
        | PcuScalarType::I64
        | PcuScalarType::U128
        | PcuScalarType::I128 => super::PcuSpirvComposedTemplateFamily::IntegerNarrow,
        _ => super::PcuSpirvComposedTemplateFamily::Low,
    };
    if data.family != family {
        return Err(Error::InvalidKernelSignature);
    }
    lower_template(
        profile,
        options,
        sink,
        Template {
            words: data.words,
            functions: data.function_ids,
            calls: data.call_target_offsets,
            arguments: data.call_argument_ids,
            defaults: data.specialization_offsets,
        },
    )
}

fn lower_template<S: PcuSpirvSink>(
    profile: &Profile,
    options: PcuSpirvLoweringOptions,
    sink: &mut S,
    template: Template<'_>,
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

#[cfg(feature = "embedded-composed")]
pub(super) const fn embedded_data(
    family: super::PcuSpirvComposedTemplateFamily,
) -> super::PcuSpirvComposedTemplateData<'static> {
    let scalar = match family {
        super::PcuSpirvComposedTemplateFamily::F32 => PcuScalarType::F32,
        super::PcuSpirvComposedTemplateFamily::F64 => PcuScalarType::F64,
        super::PcuSpirvComposedTemplateFamily::Low => PcuScalarType::F16,
        super::PcuSpirvComposedTemplateFamily::IntegerWide => PcuScalarType::U256,
        super::PcuSpirvComposedTemplateFamily::IntegerNarrow => PcuScalarType::U8,
    };
    let source = template(scalar);
    super::PcuSpirvComposedTemplateData {
        family,
        words: source.words,
        function_ids: source.functions,
        call_target_offsets: source.calls,
        call_argument_ids: source.arguments,
        specialization_offsets: source.defaults,
    }
}
