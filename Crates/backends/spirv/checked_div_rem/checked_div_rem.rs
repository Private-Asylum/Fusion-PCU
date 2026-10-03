//! Exact fourteen-width checked quotient/remainder, synthesized from U32 limbs.
#[path = "bytecode/bytecode.rs"]
mod bytecode;
#[path = "operand_bytecode/operand_bytecode.rs"]
mod operand_bytecode;
#[path = "wide_bytecode/wide_bytecode.rs"]
mod wide_bytecode;
#[path = "wide_operand_bytecode/wide_operand_bytecode.rs"]
mod wide_operand_bytecode;
#[rustfmt::skip]
use fusion_pcu::{assess_checked_integer_div_rem_operands,validate_integer_checked_div_rem_kernel,PcuDispatchIndex,validate_typed_dispatch_value_flow,PcuBindingRef,PcuDispatchControlOp,PcuDispatchDataOp,PcuDispatchKernelIr,PcuDispatchOp,PcuRangePolicy,PcuReproducibility,PcuScalarType,PcuValueType,PcuValueTypeCaps};
#[rustfmt::skip]
use crate::{PcuSpirvCapability,PcuSpirvCapabilityCaps,PcuSpirvError,PcuSpirvLoweringOptions,PcuSpirvModuleInfo,PcuSpirvSink,PcuSpirvVersion};
/// Frozen logical binding roles and actual SSA operand routing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PcuSpirvCheckedDivRemProfile {
    pub scalar: PcuScalarType,
    pub extent: u32,
    pub local_size: [u32; 3],
    pub inputs: [PcuBindingRef; 2],
    pub outputs: [PcuBindingRef; 2],
    pub operands: [u32; 2],
    pub operand_broadcast: [bool; 2],
    pub input_count: usize,
    pub input_extents: [u32; 2],
    pub declarations: [PcuBindingRef; 4],
    pub declaration_count: usize,
    pub operand_profile: bool,
    /// Exact descriptor-qualified integer joint `PortableV1` request.
    pub portable: bool,
}
impl PcuSpirvCheckedDivRemProfile {
    #[must_use]
    pub const fn element_bytes(self) -> usize {
        (self.scalar.bit_width() / 8) as usize
    }
    #[must_use]
    pub const fn packing_lanes(self) -> u32 {
        match self.scalar {
            PcuScalarType::I8 | PcuScalarType::U8 => 4,
            PcuScalarType::I16 | PcuScalarType::U16 => 2,
            _ => 1,
        }
    }
    #[must_use]
    pub const fn dispatch_extent(self) -> u32 {
        self.extent.div_ceil(self.packing_lanes())
    }
    #[must_use]
    pub const fn signed(self) -> bool {
        matches!(
            self.scalar,
            PcuScalarType::I8
                | PcuScalarType::I16
                | PcuScalarType::I32
                | PcuScalarType::I64
                | PcuScalarType::I128
                | PcuScalarType::I256
                | PcuScalarType::I512
        )
    }
    #[must_use]
    pub const fn local_id(self) -> Option<u32> {
        let legacy = match self.scalar {
            PcuScalarType::I8 => Some(608),
            PcuScalarType::U8 => Some(609),
            PcuScalarType::I16 => Some(610),
            PcuScalarType::U16 => Some(611),
            PcuScalarType::I32 => Some(612),
            PcuScalarType::U32 => Some(613),
            PcuScalarType::I64 => Some(614),
            PcuScalarType::U64 => Some(615),
            PcuScalarType::I128 => Some(616),
            PcuScalarType::U128 => Some(617),
            PcuScalarType::I256 => Some(618),
            PcuScalarType::U256 => Some(619),
            PcuScalarType::I512 => Some(620),
            PcuScalarType::U512 => Some(621),
            _ => None,
        };
        match legacy {
            Some(id) => Some(if self.portable {
                12544 + id - 608
            } else if self.operand_profile {
                8448 + id - 608
            } else {
                id
            }),
            None => None,
        }
    }
    #[must_use]
    pub fn logical_bytes(self) -> Option<u32> {
        u32::try_from(self.element_bytes())
            .ok()?
            .checked_mul(self.extent)
            .filter(|n| *n != 0)
    }
    const fn specialization(self) -> [u32; 5] {
        [
            (self.scalar.bit_width() / 8) as u32,
            self.signed() as u32,
            self.extent,
            self.operands[0],
            self.operands[1],
        ]
    }
}
/// Validates the exact neutral checked two-output profile and executable Reject header.
/// # Errors
/// Rejects unproved widths, Clamp, nonmember Portable requests, malformed SSA and byte/index overflow.
pub fn validate_checked_div_rem_map(
    kernel: &PcuDispatchKernelIr<'_>,
) -> Result<PcuSpirvCheckedDivRemProfile, PcuSpirvError> {
    let portable = validate_numerical_requirements(kernel)?;
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
    let Some(PcuDispatchOp::Data(PcuDispatchDataOp::CheckedDivRem {
        value_type: PcuValueType::Scalar(scalar),
        ..
    })) = body.get(2)
    else {
        return Err(PcuSpirvError::InvalidKernelSignature);
    };
    let value_type = PcuValueType::Scalar(*scalar);
    let caps = PcuValueTypeCaps::for_scalar(*scalar);
    let roles = assess_checked_integer_div_rem_operands(kernel, value_type, caps)
        .map_err(|_| PcuSpirvError::InvalidKernelSignature)?;
    validate_typed_dispatch_value_flow(kernel)
        .map_err(|_| PcuSpirvError::InvalidKernelSignature)?;
    let legacy = validate_integer_checked_div_rem_kernel(kernel, value_type, caps).is_ok();
    let inputs = roles.input_bindings();
    let first = inputs
        .first()
        .copied()
        .ok_or(PcuSpirvError::InvalidBinding)?;
    let counts = roles
        .input_element_counts(usize::try_from(extent).map_err(|_| PcuSpirvError::InvalidBinding)?);
    let input_extents = [
        u32::try_from(counts[0]).map_err(|_| PcuSpirvError::InvalidBinding)?,
        u32::try_from(counts[1]).map_err(|_| PcuSpirvError::InvalidBinding)?,
    ];
    let indices = roles.operand_indices();
    let operands = roles.operand_inputs();
    let mut declarations = [first; 4];
    for (slot, declaration) in kernel.bindings.iter().enumerate() {
        declarations[slot] = declaration.reference();
    }
    let profile = PcuSpirvCheckedDivRemProfile {
        scalar: *scalar,
        extent,
        local_size: [64, 1, 1],
        inputs: [first, inputs.get(1).copied().unwrap_or(first)],
        outputs: roles.output_bindings(),
        operands: [
            u32::try_from(operands[0]).map_err(|_| PcuSpirvError::InvalidKernelSignature)?,
            u32::try_from(operands[1]).map_err(|_| PcuSpirvError::InvalidKernelSignature)?,
        ],
        operand_broadcast: [
            indices[0] == PcuDispatchIndex::BindingElementZero,
            indices[1] == PcuDispatchIndex::BindingElementZero,
        ],
        input_count: inputs.len(),
        input_extents,
        declarations,
        declaration_count: kernel.bindings.len(),
        operand_profile: !legacy,
        portable,
    };
    if profile.local_id().is_none()
        || profile.logical_bytes().is_none()
        || extent == 0
        || kernel.entry.logical_shape.contains(&0)
        || kernel.entry.logical_shape[1..] != [1, 1]
    {
        return Err(PcuSpirvError::InvalidBinding);
    }
    Ok(profile)
}

fn validate_numerical_requirements(
    kernel: &PcuDispatchKernelIr<'_>,
) -> Result<bool, PcuSpirvError> {
    if kernel.numerical_requirements.range_policy != PcuRangePolicy::Reject {
        return Err(PcuSpirvError::UnsupportedNumericalRequirements);
    }
    let portable = kernel
        .numerical_requirements
        .numerical_options
        .reproducibility
        == PcuReproducibility::PortableV1;
    if portable {
        // Eligibility is checked on the original complete request before structural
        // admission. No header rewriting, alternate policy or broader tensor claim.
        fusion_pcu::describe_portable_v1_integer_div_rem_map(kernel)
            .map_err(|_| PcuSpirvError::UnsupportedNumericalRequirements)?;
    }
    Ok(portable)
}
/// Lowers an independently auditable U32-only integer division module with frozen defaults.
/// # Errors
/// Rejects unsupported profiles/version/capability before writing any word; propagates sink exhaustion.
pub fn lower_checked_div_rem_to_spirv<S: PcuSpirvSink>(
    kernel: &PcuDispatchKernelIr<'_>,
    options: PcuSpirvLoweringOptions,
    sink: &mut S,
) -> Result<(PcuSpirvModuleInfo, PcuSpirvCheckedDivRemProfile), PcuSpirvError> {
    let profile = validate_checked_div_rem_map(kernel)?;
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
    // Preserve the independently qualified eight-width module byte-for-byte.
    // Wide arithmetic uses a separate four/eight/sixteen-U32-limb module.
    let words = match (profile.operand_profile, profile.element_bytes() > 8) {
        (false, false) => bytecode::WORDS,
        (false, true) => wide_bytecode::WORDS,
        (true, false) => operand_bytecode::WORDS,
        (true, true) => wide_operand_bytecode::WORDS,
    };
    let old_defaults = profile.specialization();
    let mut defaults = [0; 7];
    defaults[..5].copy_from_slice(&old_defaults);
    defaults[5] = u32::from(profile.operand_broadcast[0]);
    defaults[6] = u32::from(profile.operand_broadcast[1]);
    let offsets = if profile.operand_profile {
        specialization_offsets::<7>(words)
    } else {
        let original = specialization_offsets::<5>(words);
        let mut offsets = [usize::MAX; 7];
        offsets[..5].copy_from_slice(&original);
        offsets
    };
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
        profile,
    ))
}
fn specialization_offsets<const N: usize>(words: &[u32]) -> [usize; N] {
    let mut ids = [0; N];
    let mut offsets = [0; N];
    let mut cursor = 5;
    while cursor < words.len() {
        let i = &words[cursor..];
        let count = (i[0] >> 16) as usize;
        let op = i[0] & 65535;
        if op == 71 && count == 4 && i[2] == 1 {
            ids[i[3] as usize] = i[1];
        } else if op == 50
            && count == 4
            && let Some(slot) = ids.iter().position(|id| *id == i[2])
        {
            offsets[slot] = cursor + 3;
        }
        cursor += count;
    }
    assert!(
        offsets.iter().all(|offset| *offset >= 5),
        "offline DivRem template retains every declared default"
    );
    offsets
}
