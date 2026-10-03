//! Freeze native calls/constants and physically remove all unused call/argument slots.
#[path = "../bytecode/bytecode.rs"]
mod bytecode;
#[rustfmt::skip]
use crate::{
    PcuSpirvCapability,
    PcuSpirvCapabilityCaps,
    PcuSpirvLoweringOptions,
    PcuSpirvModuleInfo,
    PcuSpirvSink,
    PcuSpirvVersion,
};
#[rustfmt::skip]
use super::{PcuSpirvOrderedTransportProfile as Profile,PcuSpirvError as Error};
fn constants(profile: &Profile) -> Result<[u32; 202], Error> {
    let mut values = [0; 202];
    values[0] = u32::try_from(profile.element_bytes()).map_err(|_| Error::InvalidBinding)?;
    values[1] = profile.extent;
    for (bank, resource) in profile.resources().iter().enumerate() {
        let bytes = u64::from(resource.read_elements) * u64::from(values[0]);
        values[2 + bank] =
            u32::try_from(bytes.div_ceil(4)).map_err(|_| Error::InvalidKernelSignature)?;
        values[6 + bank] = u32::from(resource.write_elements != 0);
    }
    for (step, instruction) in profile.steps[..profile.step_count].iter().enumerate() {
        values[10 + step * 3..13 + step * 3].copy_from_slice(&instruction.arguments);
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
    let values = constants(profile)?;
    let removed = &bytecode::CALL_ARGUMENT_IDS[profile.step_count * 3..];
    let mut words = 0;
    for (index, word) in bytecode::WORDS[..5].iter().copied().enumerate() {
        sink.push_word(if index == 2 { options.generator } else { word })?;
        words += 1;
    }
    let mut cursor = 5;
    while cursor < bytecode::WORDS.len() {
        let instruction = &bytecode::WORDS[cursor..];
        let count =
            usize::try_from(instruction[0] >> 16).map_err(|_| Error::InvalidKernelSignature)?;
        let opcode = instruction[0] & 0xffff;
        let call = bytecode::CALL_TARGET_OFFSETS
            .iter()
            .position(|offset| *offset == cursor + 3);
        if (opcode == 71 && count == 4 && instruction[2] == 1)
            || call.is_some_and(|slot| slot >= profile.step_count)
            || ((opcode == 5 || opcode == 62) && removed.contains(&instruction[1]))
            || (opcode == 59 && removed.contains(&instruction[2]))
        {
            cursor += count;
            continue;
        }
        for (operand, word) in instruction[..count].iter().copied().enumerate() {
            let output = if opcode == 50 && operand == 0 {
                (instruction[0] & 0xffff_0000) | 0x002b
            } else if opcode == 50 && operand == 3 {
                let slot = bytecode::SPECIALIZATION_OFFSETS
                    .iter()
                    .position(|offset| *offset == cursor + 3)
                    .ok_or(Error::InvalidKernelSignature)?;
                values[slot]
            } else if operand == 3
                && let Some(slot) = call
            {
                bytecode::FUNCTION_IDS[profile.steps[slot].function]
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
        bound: bytecode::WORDS[3],
        word_count: words,
        capabilities: PcuSpirvCapabilityCaps::SHADER,
    })
}
