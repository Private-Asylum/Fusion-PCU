//! Exact fourteen-carrier checked integer maps synthesized from U32 limbs.
#[path = "bytecode/bytecode.rs"]
mod bytecode;
#[path = "operand_bytecode/operand_bytecode.rs"]
mod operand_bytecode;
#[rustfmt::skip]
use fusion_pcu::{
    assess_checked_integer_binary_operands,
    validate_integer_checked_binary_kernel,
    validate_typed_dispatch_value_flow,
    PcuBindingAccess,
    PcuBindingRef,
    PcuDispatchControlOp,
    PcuDispatchDataOp,
    PcuDispatchIntegerBinaryOp,
    PcuDispatchIndex,
    PcuDispatchKernelIr,
    PcuDispatchOp,
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
/// Logical binding roles and input SSA routing are independent of physical descriptor slots.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PcuSpirvCheckedIntegerProfile {
    pub scalar: PcuScalarType,
    pub operation: PcuDispatchIntegerBinaryOp,
    pub range: PcuRangePolicy,
    pub extent: u32,
    pub local_size: [u32; 3],
    pub inputs: [PcuBindingRef; 2],
    pub output: PcuBindingRef,
    pub operands: [u32; 2],
    pub broadcast: [bool; 2],
    /// Independent indexing of each arithmetic operand, even when both read one binding.
    pub operand_broadcast: [bool; 2],
    pub input_count: usize,
    pub input_extents: [u32; 2],
    pub declarations: [PcuBindingRef; 3],
    pub declaration_count: usize,
    pub operand_profile: bool,
    /// Independently admitted exact integer `PortableV1` profile; not a tensor/DivRem promise.
    pub portable: bool,
}
impl PcuSpirvCheckedIntegerProfile {
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
    pub const fn input_extent(self, input: usize) -> u32 {
        self.input_extents[input]
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
        match ordinal(self.scalar) {
            Some(format) => Some(
                (if self.portable {
                    4096
                } else if self.operand_profile {
                    2048
                } else {
                    512
                }) + format * 6
                    + if matches!(self.range, PcuRangePolicy::Clamp) {
                        3
                    } else {
                        0
                    }
                    + op_code(self.operation),
            ),
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
    const fn specialization(self) -> [u32; 9] {
        [
            (self.scalar.bit_width() / 8) as u32,
            self.signed() as u32,
            op_code(self.operation),
            self.extent,
            (if self.operand_profile {
                self.operand_broadcast[0]
            } else {
                self.broadcast[0]
            }) as u32,
            (if self.operand_profile {
                self.operand_broadcast[1]
            } else {
                self.broadcast[1]
            }) as u32,
            self.operands[0],
            self.operands[1],
            matches!(self.range, PcuRangePolicy::Clamp) as u32,
        ]
    }
}
const fn op_code(operation: PcuDispatchIntegerBinaryOp) -> u32 {
    match operation {
        PcuDispatchIntegerBinaryOp::Add => 0,
        PcuDispatchIntegerBinaryOp::Sub => 1,
        PcuDispatchIntegerBinaryOp::Mul => 2,
    }
}
const fn ordinal(scalar: PcuScalarType) -> Option<u32> {
    match scalar {
        PcuScalarType::I8 => Some(0),
        PcuScalarType::U8 => Some(1),
        PcuScalarType::I16 => Some(2),
        PcuScalarType::U16 => Some(3),
        PcuScalarType::I32 => Some(4),
        PcuScalarType::U32 => Some(5),
        PcuScalarType::I64 => Some(6),
        PcuScalarType::U64 => Some(7),
        PcuScalarType::I128 => Some(8),
        PcuScalarType::U128 => Some(9),
        PcuScalarType::I256 => Some(10),
        PcuScalarType::U256 => Some(11),
        PcuScalarType::I512 => Some(12),
        PcuScalarType::U512 => Some(13),
        _ => None,
    }
}
/// Admits exact checked Add/Sub/Mul range headers, typed SSA and direct/grid/broadcast geometry.
/// # Errors
/// Rejects unsupported carriers, nonmember Portable requests, malformed bindings/SSA and overflowing byte products.
#[allow(clippy::too_many_lines)] // The frozen schema and SSA operand-to-binding derivation form one cold transaction.
pub fn validate_checked_integer_map(
    kernel: &PcuDispatchKernelIr<'_>,
) -> Result<PcuSpirvCheckedIntegerProfile, PcuSpirvError> {
    if kernel
        .numerical_requirements
        .numerical_options
        .reproducibility
        == PcuReproducibility::PortableV1
    {
        fusion_pcu::describe_portable_v1_integer_map(kernel)
            .map_err(|_| PcuSpirvError::UnsupportedNumericalRequirements)?;
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
            result: first,
            binding: first_binding,
            index: first_index,
        }),
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
            result: second,
            binding: second_binding,
            index: second_index,
        }),
        PcuDispatchOp::Data(PcuDispatchDataOp::CheckedIntegerBinary {
            value_type: PcuValueType::Scalar(scalar),
            op,
            range_policy,
            lhs,
            rhs,
            ..
        }),
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
            binding: output, ..
        }),
    ] = body
    else {
        return Err(PcuSpirvError::InvalidKernelSignature);
    };
    if ordinal(*scalar).is_none() {
        return Err(PcuSpirvError::InvalidBinding);
    }
    let schema = assess_checked_integer_binary_operands(
        kernel,
        PcuValueType::Scalar(*scalar),
        *op,
        PcuValueTypeCaps::for_scalar(*scalar),
    )
    .map_err(|_| PcuSpirvError::InvalidKernelSignature)?;
    let operand_profile = validate_integer_checked_binary_kernel(
        kernel,
        PcuValueType::Scalar(*scalar),
        *op,
        PcuValueTypeCaps::for_scalar(*scalar),
    )
    .is_err();
    validate_typed_dispatch_value_flow(kernel)
        .map_err(|_| PcuSpirvError::InvalidKernelSignature)?;
    if kernel.numerical_requirements.range_policy != *range_policy {
        return Err(PcuSpirvError::UnsupportedNumericalRequirements);
    }
    if extent == 0
        || kernel.entry.logical_shape.contains(&0)
        || kernel.entry.logical_shape[1..] != [1, 1]
    {
        return Err(PcuSpirvError::InvalidBinding);
    }
    let loaded = schema.input_bindings();
    let mut inputs = [loaded[0], *loaded.get(1).unwrap_or(&loaded[0])];
    if !operand_profile {
        for (slot, binding) in kernel
            .bindings
            .iter()
            .filter(|binding| binding.access == PcuBindingAccess::ReadOnly)
            .enumerate()
        {
            inputs[slot] = binding.reference();
        }
    }
    let extents = schema
        .input_element_counts(usize::try_from(extent).map_err(|_| PcuSpirvError::InvalidBinding)?);
    let mut extents = extents;
    if !operand_profile {
        for (slot, binding) in inputs.iter().enumerate() {
            let original = loaded
                .iter()
                .position(|input| input == binding)
                .ok_or(PcuSpirvError::InvalidBinding)?;
            extents[slot] = schema.input_element_counts(
                usize::try_from(extent).map_err(|_| PcuSpirvError::InvalidBinding)?,
            )[original];
        }
    }
    let input_extents = [
        u32::try_from(extents[0]).map_err(|_| PcuSpirvError::InvalidBinding)?,
        u32::try_from(extents[1]).map_err(|_| PcuSpirvError::InvalidBinding)?,
    ];
    let mut declarations = [schema.output_binding(); 3];
    for (slot, binding) in kernel.bindings.iter().enumerate() {
        declarations[slot] = binding.reference();
    }
    let bank = |binding| {
        inputs
            .iter()
            .position(|input| *input == binding)
            .and_then(|index| u32::try_from(index).ok())
            .ok_or(PcuSpirvError::InvalidBinding)
    };
    let first_bank = bank(*first_binding)?;
    let second_bank = bank(*second_binding)?;
    let operand = |value| {
        if value == first {
            Ok(first_bank)
        } else if value == second {
            Ok(second_bank)
        } else {
            Err(PcuSpirvError::InvalidKernelSignature)
        }
    };
    let mut broadcast = [false; 2];
    broadcast[first_bank as usize] = *first_index == PcuDispatchIndex::BindingElementZero;
    broadcast[second_bank as usize] = *second_index == PcuDispatchIndex::BindingElementZero;
    let profile = PcuSpirvCheckedIntegerProfile {
        scalar: *scalar,
        operation: *op,
        range: *range_policy,
        extent,
        local_size: [64, 1, 1],
        inputs,
        output: *output,
        operands: [operand(lhs)?, operand(rhs)?],
        broadcast,
        operand_broadcast: schema
            .operand_indices()
            .map(|index| index == PcuDispatchIndex::BindingElementZero),
        input_count: loaded.len(),
        input_extents,
        declarations,
        declaration_count: kernel.bindings.len(),
        operand_profile,
        portable: kernel
            .numerical_requirements
            .numerical_options
            .reproducibility
            == PcuReproducibility::PortableV1,
    };
    profile
        .logical_bytes()
        .ok_or(PcuSpirvError::InvalidKernelSignature)?;
    Ok(profile)
}
/// Emits the audited U32-limb template with nine frozen specialization defaults.
/// # Errors
/// Rejects unsupported schema/version/capabilities before emission, or returns sink failure.
pub fn lower_checked_integer_to_spirv<S: PcuSpirvSink>(
    kernel: &PcuDispatchKernelIr<'_>,
    options: PcuSpirvLoweringOptions,
    sink: &mut S,
) -> Result<(PcuSpirvModuleInfo, PcuSpirvCheckedIntegerProfile), PcuSpirvError> {
    let profile = validate_checked_integer_map(kernel)?;
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
    let words = if profile.operand_profile {
        operand_bytecode::WORDS
    } else {
        bytecode::WORDS
    };
    let offsets = specialization_offsets(words);
    let defaults = profile.specialization();
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
fn specialization_offsets(words: &[u32]) -> [usize; 9] {
    let mut ids = [0; 9];
    let mut offsets = [0; 9];
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
        "offline integer template retains all nine defaults"
    );
    offsets
}
