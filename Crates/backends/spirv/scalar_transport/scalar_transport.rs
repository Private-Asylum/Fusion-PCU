//! Exact dense and scalar-broadcast raw transport for all twenty-two sealed host carriers.
#[path = "bytecode/bytecode.rs"]
mod bytecode;
#[rustfmt::skip]
use fusion_pcu::{validate_scalar_identity_kernel,validate_scalar_broadcast_kernel,validate_typed_dispatch_value_flow,PcuBindingRef,PcuBindingAccess,PcuBindingType,PcuDispatchControlOp,PcuDispatchDataOp,PcuDispatchIndex,PcuDispatchKernelIr,PcuDispatchOp,PcuReproducibility,PcuScalarType,PcuValueType};
#[rustfmt::skip]
use crate::{PcuSpirvCapability,PcuSpirvCapabilityCaps,PcuSpirvError,PcuSpirvLoweringOptions,PcuSpirvModuleInfo,PcuSpirvSink,PcuSpirvVersion};
/// Logical binding roles are independent of the private U32 descriptor ABI at bindings0/1.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PcuSpirvScalarTransportProfile {
    pub scalar: PcuScalarType,
    pub extent: u32,
    pub broadcast: bool,
    pub input: PcuBindingRef,
    pub output: PcuBindingRef,
    pub local_size: [u32; 3],
}
impl PcuSpirvScalarTransportProfile {
    #[must_use]
    pub const fn element_bytes(self) -> usize {
        (self.scalar.bit_width() / 8) as usize
    }
    #[must_use]
    pub const fn input_extent(self) -> u32 {
        if self.broadcast { 1 } else { self.extent }
    }
    #[must_use]
    pub const fn local_id(self) -> Option<u32> {
        if carrier(self.scalar) {
            Some(if self.broadcast { 160 } else { 128 } + self.scalar as u32)
        } else {
            None
        }
    }
    #[must_use]
    pub fn logical_bytes(self) -> Option<u32> {
        u32::try_from(self.element_bytes())
            .ok()?
            .checked_mul(self.extent)
            .filter(|bytes| *bytes != 0)
    }
    #[must_use]
    pub fn dispatch_extent(self) -> Option<u32> {
        self.logical_bytes().map(|bytes| bytes.div_ceil(4))
    }
}
const fn carrier(scalar: PcuScalarType) -> bool {
    matches!(
        scalar,
        PcuScalarType::I8
            | PcuScalarType::U8
            | PcuScalarType::I16
            | PcuScalarType::U16
            | PcuScalarType::I32
            | PcuScalarType::U32
            | PcuScalarType::I64
            | PcuScalarType::U64
            | PcuScalarType::I128
            | PcuScalarType::U128
            | PcuScalarType::I256
            | PcuScalarType::U256
            | PcuScalarType::I512
            | PcuScalarType::U512
            | PcuScalarType::F16
            | PcuScalarType::BF16
            | PcuScalarType::F8E4M3FN
            | PcuScalarType::F8E5M2
            | PcuScalarType::F32
            | PcuScalarType::F64
            | PcuScalarType::F128
            | PcuScalarType::F256
    )
}
/// Separately validates dense or one-element broadcast and exact initialized carrier layout.
/// # Errors
/// Rejects unsupported carriers, Portable, malformed roles/SSA/policies and byte extents.
pub fn validate_scalar_transport_map(
    kernel: &PcuDispatchKernelIr<'_>,
) -> Result<PcuSpirvScalarTransportProfile, PcuSpirvError> {
    if kernel
        .numerical_requirements
        .numerical_options
        .reproducibility
        == PcuReproducibility::PortableV1
    {
        return Err(PcuSpirvError::UnsupportedNumericalRequirements);
    }
    let (body, extent) = match kernel.ops {
        [
            PcuDispatchOp::GridStrideLoop { extent, body },
            PcuDispatchOp::Control(PcuDispatchControlOp::Return),
        ] => (*body, *extent),
        [
            body @ ..,
            PcuDispatchOp::Control(PcuDispatchControlOp::Return),
        ] => (body, kernel.entry.logical_shape[0]),
        _ => return Err(PcuSpirvError::InvalidKernelSignature),
    };
    let [
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
            binding: input,
            index,
            ..
        }),
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
            binding: output, ..
        }),
    ] = body
    else {
        return Err(PcuSpirvError::InvalidKernelSignature);
    };
    let Some(binding) = kernel.bindings.iter().find(|b| b.reference() == *input) else {
        return Err(PcuSpirvError::InvalidBinding);
    };
    let PcuBindingType::Value(PcuValueType::Scalar(scalar)) = binding.binding_type else {
        return Err(PcuSpirvError::InvalidBinding);
    };
    if !carrier(scalar)
        || extent == 0
        || kernel.entry.logical_shape.contains(&0)
        || binding.access != PcuBindingAccess::ReadOnly
    {
        return Err(PcuSpirvError::InvalidBinding);
    }
    let broadcast = *index == PcuDispatchIndex::BindingElementZero;
    if broadcast {
        validate_scalar_broadcast_kernel(kernel, scalar)
    } else {
        validate_scalar_identity_kernel(kernel, scalar)
    }
    .map_err(|_| PcuSpirvError::InvalidKernelSignature)?;
    validate_typed_dispatch_value_flow(kernel)
        .map_err(|_| PcuSpirvError::InvalidKernelSignature)?;
    let profile = PcuSpirvScalarTransportProfile {
        scalar,
        extent,
        broadcast,
        input: *input,
        output: *output,
        local_size: [64, 1, 1],
    };
    profile
        .logical_bytes()
        .ok_or(PcuSpirvError::InvalidKernelSignature)?;
    Ok(profile)
}
/// Emits the two-descriptor Shader/U32-only transport template.
/// # Errors
/// Rejects invalid schema, versions/capabilities or sink failure before unsupported emission.
pub fn lower_scalar_transport_to_spirv<S: PcuSpirvSink>(
    kernel: &PcuDispatchKernelIr<'_>,
    options: PcuSpirvLoweringOptions,
    sink: &mut S,
) -> Result<(PcuSpirvModuleInfo, PcuSpirvScalarTransportProfile), PcuSpirvError> {
    let p = validate_scalar_transport_map(kernel)?;
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
    let defaults = [
        u32::try_from(p.element_bytes()).map_err(|_| PcuSpirvError::InvalidBinding)?,
        p.extent,
        u32::from(p.broadcast),
    ];
    for (index, original) in words.iter().copied().enumerate() {
        let word = if index == 1 {
            options.version.0
        } else if index == 2 {
            options.generator
        } else if let Some(slot) = offsets.iter().position(|offset| *offset == index) {
            defaults[slot]
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
        p,
    ))
}

fn specialization_offsets(words: &[u32]) -> [usize; 3] {
    let mut ids = [0; 3];
    let mut offsets = [0; 3];
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
        "offline scalar transport template retains all three defaults"
    );
    offsets
}
