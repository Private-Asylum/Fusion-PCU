//! Checked F32/F64 conversion through explicit U32 representation packing.
#[path = "bytecode/bytecode.rs"]
mod bytecode;
#[rustfmt::skip]
use fusion_pcu::{
    validate_checked_float_conversion_map_kernel,
    validate_typed_dispatch_value_flow,
    PcuBindingAccess,
    PcuBindingRef,
    PcuDispatchCheckedFloatConversion,
    PcuDispatchControlOp,
    PcuDispatchDataOp,
    PcuDispatchIndex,
    PcuDispatchKernelIr,
    PcuDispatchOp,
    PcuFloatUnderflowPolicy,
    PcuRangePolicy,
    PcuReproducibility,
    PcuScalarType,
    PcuValueTypeCaps,
};
#[rustfmt::skip]
use crate::{PcuSpirvCapability,PcuSpirvCapabilityCaps,PcuSpirvError,PcuSpirvLoweringOptions,PcuSpirvModuleInfo,PcuSpirvSink,PcuSpirvVersion};
/// Frozen mixed-width source/output roles and logical diagnostic geometry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PcuSpirvCheckedConversionProfile {
    pub input: PcuBindingRef,
    pub output: PcuBindingRef,
    pub conversion: PcuDispatchCheckedFloatConversion,
    pub range: PcuRangePolicy,
    pub underflow: PcuFloatUnderflowPolicy,
    pub extent: u32,
    pub broadcast: bool,
    pub local_size: [u32; 3],
}
impl PcuSpirvCheckedConversionProfile {
    #[must_use]
    pub const fn source_scalar(self) -> PcuScalarType {
        match self.conversion {
            PcuDispatchCheckedFloatConversion::F32ToF64 => PcuScalarType::F32,
            PcuDispatchCheckedFloatConversion::F64ToF32 => PcuScalarType::F64,
        }
    }
    #[must_use]
    pub const fn output_scalar(self) -> PcuScalarType {
        match self.conversion {
            PcuDispatchCheckedFloatConversion::F32ToF64 => PcuScalarType::F64,
            PcuDispatchCheckedFloatConversion::F64ToF32 => PcuScalarType::F32,
        }
    }
    #[must_use]
    pub const fn input_extent(self) -> u32 {
        if self.broadcast { 1 } else { self.extent }
    }
    #[must_use]
    pub const fn local_id(self) -> u32 {
        640 + match self.range {
            PcuRangePolicy::Reject => 0,
            PcuRangePolicy::Clamp => 2,
        } + match self.conversion {
            PcuDispatchCheckedFloatConversion::F32ToF64 => 0,
            PcuDispatchCheckedFloatConversion::F64ToF32 => 1,
        }
    }
    const fn specialization(self) -> [u32; 4] {
        [
            match self.conversion {
                PcuDispatchCheckedFloatConversion::F32ToF64 => 0,
                PcuDispatchCheckedFloatConversion::F64ToF32 => 1,
            },
            match self.underflow {
                PcuFloatUnderflowPolicy::IeeeAfterRounding => 0,
                PcuFloatUnderflowPolicy::RejectSubnormalResult => 1,
                PcuFloatUnderflowPolicy::AllowGradualUnderflow => 2,
            },
            self.extent,
            self.broadcast as u32,
        ]
    }
}
/// Validates exact mixed-width typed SSA and policies before emission or native preparation.
/// # Errors
/// Rejects Portable, unsupported effects, malformed SSA/header, access/index or geometry.
#[allow(clippy::too_many_lines)] // Closed three-op conversion schema retains all cold obligations beside its original roles.
pub fn validate_checked_float_conversion_map(
    kernel: &PcuDispatchKernelIr<'_>,
) -> Result<PcuSpirvCheckedConversionProfile, PcuSpirvError> {
    if kernel
        .numerical_requirements
        .numerical_options
        .reproducibility
        == PcuReproducibility::PortableV1
    {
        return Err(PcuSpirvError::UnsupportedNumericalRequirements);
    }
    validate_checked_float_conversion_map_kernel(
        kernel,
        PcuValueTypeCaps::FLOAT32 | PcuValueTypeCaps::FLOAT64,
    )
    .map_err(|_| PcuSpirvError::InvalidKernelSignature)?;
    validate_typed_dispatch_value_flow(kernel)
        .map_err(|_| PcuSpirvError::InvalidKernelSignature)?;
    let (body, extent, index) = match kernel.ops {
        [
            PcuDispatchOp::GridStrideLoop { extent, body },
            PcuDispatchOp::Control(PcuDispatchControlOp::Return),
        ] => (*body, *extent, PcuDispatchIndex::GridStrideId),
        [
            body @ ..,
            PcuDispatchOp::Control(PcuDispatchControlOp::Return),
        ] => (
            body,
            kernel.entry.logical_shape[0],
            PcuDispatchIndex::InvocationId,
        ),
        _ => return Err(PcuSpirvError::InvalidKernelSignature),
    };
    let [
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
            result: loaded,
            binding: input,
            index: source_index,
        }),
        PcuDispatchOp::Data(PcuDispatchDataOp::CheckedFloatConvert {
            result,
            value,
            conversion,
            range_policy,
            underflow_policy,
        }),
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
            binding: output,
            index: output_index,
            value: stored,
        }),
    ] = body
    else {
        return Err(PcuSpirvError::InvalidKernelSignature);
    };
    if loaded != value
        || result != stored
        || input == output
        || kernel.bindings.len() != 2
        || extent == 0
        || extent > 0x7fff_ffff
        || kernel.entry.logical_shape[1..] != [1, 1]
        || *output_index != index
        || !matches!(*source_index, PcuDispatchIndex::BindingElementZero) && *source_index != index
    {
        return Err(PcuSpirvError::InvalidKernelSignature);
    }
    if kernel.numerical_requirements.range_policy != *range_policy
        || kernel.numerical_requirements.float_underflow != *underflow_policy
    {
        return Err(PcuSpirvError::UnsupportedNumericalRequirements);
    }
    for (reference, access) in [
        (*input, PcuBindingAccess::ReadOnly),
        (*output, PcuBindingAccess::ReadWrite),
    ] {
        let binding = kernel
            .bindings
            .iter()
            .find(|binding| binding.reference() == reference)
            .ok_or(PcuSpirvError::InvalidKernelSignature)?;
        if binding.access != access {
            return Err(PcuSpirvError::InvalidKernelSignature);
        }
    }
    Ok(PcuSpirvCheckedConversionProfile {
        input: *input,
        output: *output,
        conversion: *conversion,
        range: *range_policy,
        underflow: *underflow_policy,
        extent,
        broadcast: *source_index == PcuDispatchIndex::BindingElementZero,
        local_size: [64, 1, 1],
    })
}
/// Emits a Shader-only U32 conversion template after complete cold validation.
/// # Errors
/// Returns structural/numerical admission, capability/version or sink failures.
pub fn lower_checked_float_conversion_to_spirv(
    kernel: &PcuDispatchKernelIr<'_>,
    options: PcuSpirvLoweringOptions,
    sink: &mut impl PcuSpirvSink,
) -> Result<(PcuSpirvModuleInfo, PcuSpirvCheckedConversionProfile), PcuSpirvError> {
    let profile = validate_checked_float_conversion_map(kernel)?;
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
    let words = bytecode::WORDS;
    let offsets = specialization_offsets(words);
    let values = profile.specialization();
    for (index, original) in words.iter().copied().enumerate() {
        let word = if index == 1 {
            options.version.0
        } else if index == 2 {
            options.generator
        } else if let Some(slot) = offsets.iter().position(|offset| *offset == index) {
            values[slot]
        } else {
            original
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
fn specialization_offsets(words: &[u32]) -> [usize; 4] {
    let mut ids = [0; 4];
    let mut offsets = [0; 4];
    let mut cursor = 5;
    while cursor < words.len() {
        let instruction = &words[cursor..];
        let count = (instruction[0] >> 16) as usize;
        let opcode = instruction[0] & 0xffff;
        if opcode == 71 && count == 4 && instruction[2] == 1 {
            ids[instruction[3] as usize] = instruction[1];
        } else if opcode == 50
            && count == 4
            && let Some(slot) = ids.iter().position(|id| *id == instruction[2])
        {
            offsets[slot] = cursor + 3;
        }
        cursor += count;
    }
    assert!(
        offsets.iter().all(|offset| *offset >= 5),
        "offline conversion template retains all four defaults"
    );
    offsets
}
