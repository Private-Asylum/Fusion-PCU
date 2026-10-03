//! Exact six-format checked Neg/ReLU through U32 sign manipulation and selection.
#[path = "bytecode/bytecode.rs"]
mod bytecode;
#[path = "roles/roles.rs"]
mod roles;
#[rustfmt::skip]
pub use roles::{
    lower_checked_float_unary_roles_to_spirv,
    validate_checked_float_unary_roles_map,
    PcuSpirvCheckedUnaryRoleProfile,
};
#[rustfmt::skip]
use fusion_pcu::{
    validate_checked_float_map_kernel,
    validate_typed_dispatch_value_flow,
    PcuBindingAccess,
    PcuBindingRef,
    PcuDispatchControlOp,
    PcuDispatchDataOp,
    PcuDispatchFloatUnaryOp,
    PcuDispatchIndex,
    PcuDispatchKernelIr,
    PcuDispatchOp,
    PcuFloatUnderflowPolicy,
    PcuRangePolicy,
    PcuReproducibility,
    PcuScalarType,
    PcuValueType,
    PcuValueTypeCaps,
};
#[rustfmt::skip]
use crate::{
    PcuSpirvCapability,
    PcuSpirvCapabilityCaps,
    PcuSpirvError,
    PcuSpirvLoweringOptions,
    PcuSpirvModuleInfo,
    PcuSpirvSink,
    PcuSpirvVersion,
};
/// Frozen single-input checked map and logical diagnostic geometry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PcuSpirvCheckedUnaryProfile {
    pub scalar: PcuScalarType,
    pub operation: PcuDispatchFloatUnaryOp,
    pub underflow: PcuFloatUnderflowPolicy,
    pub range: PcuRangePolicy,
    pub extent: u32,
    pub broadcast: bool,
    pub local_size: [u32; 3],
    /// Original request is eligible for the separate exact-selection Portable profile.
    pub portable: bool,
}
impl PcuSpirvCheckedUnaryProfile {
    pub const INPUT: PcuBindingRef = PcuBindingRef::new(0, 0);
    pub const OUTPUT: PcuBindingRef = PcuBindingRef::new(0, 1);
    pub const STATUS: PcuBindingRef = PcuBindingRef::new(0, 2);
    #[must_use]
    pub const fn element_bytes(self) -> usize {
        (self.scalar.bit_width() / 8) as usize
    }
    #[must_use]
    pub const fn packing_lanes(self) -> u32 {
        match self.scalar {
            PcuScalarType::F16 | PcuScalarType::BF16 => 2,
            PcuScalarType::F8E4M3FN | PcuScalarType::F8E5M2 => 4,
            _ => 1,
        }
    }
    #[must_use]
    pub const fn dispatch_extent(self) -> u32 {
        self.extent.div_ceil(self.packing_lanes())
    }
    #[must_use]
    pub const fn input_extent(self) -> u32 {
        if self.broadcast { 1 } else { self.extent }
    }
    #[must_use]
    pub const fn local_id(self) -> Option<u32> {
        let Some(format) = self.format() else {
            return None;
        };
        Some(
            (if self.portable { 16640 } else { 96 })
                + format * 4
                + match self.range {
                    PcuRangePolicy::Reject => 0,
                    PcuRangePolicy::Clamp => 2,
                }
                + match self.operation {
                    PcuDispatchFloatUnaryOp::Neg => 0,
                    PcuDispatchFloatUnaryOp::Relu => 1,
                },
        )
    }
    const fn format(self) -> Option<u32> {
        match self.scalar {
            PcuScalarType::F16 => Some(0),
            PcuScalarType::BF16 => Some(1),
            PcuScalarType::F8E4M3FN => Some(2),
            PcuScalarType::F8E5M2 => Some(3),
            PcuScalarType::F32 => Some(4),
            PcuScalarType::F64 => Some(5),
            _ => None,
        }
    }
    const fn specialization(self) -> [u32; 5] {
        [
            self.format()
                .expect("cold validated exactly six checked formats"),
            match self.operation {
                PcuDispatchFloatUnaryOp::Neg => 0,
                PcuDispatchFloatUnaryOp::Relu => 1,
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
/// Validates exact scalar, policy/header, bindings, index and SSA before any emission.
/// # Errors
/// Rejects nonmember Portable requests, malformed value flow, unsupported scalars/effects or host schema.
#[allow(clippy::too_many_lines)] // The closed unary pattern stays beside all cold obligations.
pub fn validate_checked_float_unary_map(
    kernel: &PcuDispatchKernelIr<'_>,
) -> Result<PcuSpirvCheckedUnaryProfile, PcuSpirvError> {
    let portable = kernel
        .numerical_requirements
        .numerical_options
        .reproducibility
        == PcuReproducibility::PortableV1;
    if portable {
        // Eligibility alone is insufficient: the existing exact shader schema is checked below.
        fusion_pcu::describe_portable_v1_unary_map(kernel)
            .map_err(|_| PcuSpirvError::UnsupportedNumericalRequirements)?;
    }
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
            index: load,
        }),
        PcuDispatchOp::Data(PcuDispatchDataOp::CheckedFloatUnary {
            result,
            op,
            value_type:
                PcuValueType::Scalar(
                    scalar @ (PcuScalarType::F16
                    | PcuScalarType::BF16
                    | PcuScalarType::F8E4M3FN
                    | PcuScalarType::F8E5M2
                    | PcuScalarType::F32
                    | PcuScalarType::F64),
                ),
            value,
            underflow_policy,
            range_policy,
        }),
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
            binding: output,
            index: store,
            value: stored,
        }),
    ] = body
    else {
        return Err(PcuSpirvError::InvalidKernelSignature);
    };
    validate_checked_float_map_kernel(
        kernel,
        PcuValueType::Scalar(*scalar),
        PcuValueTypeCaps::for_scalar(*scalar),
    )
    .map_err(|_| PcuSpirvError::InvalidKernelSignature)?;
    validate_typed_dispatch_value_flow(kernel)
        .map_err(|_| PcuSpirvError::InvalidKernelSignature)?;
    if kernel.numerical_requirements.range_policy != *range_policy
        || kernel.numerical_requirements.float_underflow != *underflow_policy
    {
        return Err(PcuSpirvError::UnsupportedNumericalRequirements);
    }
    if extent == 0
        || kernel.entry.logical_shape.contains(&0)
        || kernel.entry.logical_shape[1..] != [1, 1]
        || kernel.bindings.len() != 2
        || !kernel.parameters.is_empty()
        || !kernel.ports.is_empty()
        || *input != PcuSpirvCheckedUnaryProfile::INPUT
        || *output != PcuSpirvCheckedUnaryProfile::OUTPUT
        || loaded != value
        || result != stored
        || (*load != index && *load != PcuDispatchIndex::BindingElementZero)
        || *store != index
        || (*scalar == PcuScalarType::F64 && extent > u32::MAX / 2)
    {
        return Err(PcuSpirvError::InvalidBinding);
    }
    for (target, access) in [
        (
            PcuSpirvCheckedUnaryProfile::INPUT,
            PcuBindingAccess::ReadOnly,
        ),
        (
            PcuSpirvCheckedUnaryProfile::OUTPUT,
            PcuBindingAccess::ReadWrite,
        ),
    ] {
        if !kernel.bindings.iter().any(|b| {
            b.reference() == target
                && (b.access == access
                    || (target == PcuSpirvCheckedUnaryProfile::OUTPUT
                        && b.access == PcuBindingAccess::WriteOnly))
        }) {
            return Err(PcuSpirvError::InvalidBinding);
        }
    }
    Ok(PcuSpirvCheckedUnaryProfile {
        scalar: *scalar,
        operation: *op,
        underflow: *underflow_policy,
        range: *range_policy,
        extent,
        broadcast: *load == PcuDispatchIndex::BindingElementZero,
        local_size: [64, 1, 1],
        portable,
    })
}
/// Emits the Shader/U32-only template with five frozen specialization defaults.
/// # Errors
/// Returns schema, version/capability or sink failure before unsupported work emits words.
pub fn lower_checked_float_unary_to_spirv<S: PcuSpirvSink>(
    kernel: &PcuDispatchKernelIr<'_>,
    options: PcuSpirvLoweringOptions,
    sink: &mut S,
) -> Result<(PcuSpirvModuleInfo, PcuSpirvCheckedUnaryProfile), PcuSpirvError> {
    let profile = validate_checked_float_unary_map(kernel)?;
    let info = emit_profile(profile, options, sink)?;
    Ok((info, profile))
}
fn emit_profile<S: PcuSpirvSink>(
    profile: PcuSpirvCheckedUnaryProfile,
    options: PcuSpirvLoweringOptions,
    sink: &mut S,
) -> Result<PcuSpirvModuleInfo, PcuSpirvError> {
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
    Ok(PcuSpirvModuleInfo {
        version: options.version,
        bound: words[3],
        word_count: words.len(),
        capabilities: PcuSpirvCapabilityCaps::SHADER,
    })
}
fn specialization_offsets(words: &[u32]) -> [usize; 5] {
    let mut ids = [0; 5];
    let mut offsets = [0; 5];
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
        "offline unary template retains all five defaults"
    );
    offsets
}
