//! Exact six-format checked arithmetic synthesized from bounded U32 operations.
//!
//! The auditable GLSL source is compiled offline into a checked-in template. Preparation
//! validates the complete PCU schema and freezes six or eight specialization defaults. There is no
//! runtime compiler, native floating arithmetic, Int64 feature, or floating-controls assumption.

#[path = "bytecode/bytecode.rs"]
mod bytecode;
#[path = "bytecode/f64.rs"]
mod f64_bytecode;
#[path = "bytecode/low.rs"]
mod low_bytecode;
#[rustfmt::skip]
use fusion_pcu::{
    assess_checked_float_binary_operands,
    validate_typed_dispatch_value_flow,
    PcuBindingRef,
    PcuDispatchControlOp,
    PcuDispatchDataOp,
    PcuDispatchFloatBinaryOp,
    PcuDispatchIndex,
    PcuDispatchKernelIr,
    PcuDispatchOp,
    PcuFloatUnderflowPolicy,
    PcuRangePolicy,
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

/// Cold-frozen checked binary operation, scalar representation and original input layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PcuSpirvCheckedBinaryProfile {
    pub scalar: PcuScalarType,
    pub operation: PcuDispatchFloatBinaryOp,
    pub underflow: PcuFloatUnderflowPolicy,
    pub range: PcuRangePolicy,
    pub extent: u32,
    pub local_size: [u32; 3],
    /// Operand SSA selects original binding zero or one; repetitions are permitted.
    pub operands: [u32; 2],
    /// Each original binding independently broadcasts element zero or uses the logical index.
    pub broadcast: [bool; 2],
    /// Distinct actual input bindings in cold load order.
    pub inputs: [PcuBindingRef; 2],
    pub input_count: usize,
    pub input_extents: [u32; 2],
    pub output: PcuBindingRef,
    pub declarations: [PcuBindingRef; 3],
    pub declaration_count: usize,
}

impl PcuSpirvCheckedBinaryProfile {
    pub const INPUTS: [PcuBindingRef; 2] = [PcuBindingRef::new(0, 0), PcuBindingRef::new(0, 1)];
    pub const OUTPUT: PcuBindingRef = PcuBindingRef::new(0, 2);
    /// Private U32 diagnostic: 0 success, 1 invalid operand, 2 underflow, 3 overflow, 4 divide by zero.
    pub const STATUS: PcuBindingRef = PcuBindingRef::new(0, 3);

    /// Exact caller representation size; admission permits only the six checked formats.
    #[must_use]
    pub const fn element_bytes(self) -> usize {
        (self.scalar.bit_width() / 8) as usize
    }

    /// One narrow output word is written by exactly one invocation.
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
    pub const fn input_extent(self, input: usize) -> u32 {
        self.input_extents[input]
    }

    const fn specialization(self) -> [u32; 8] {
        [
            match self.operation {
                PcuDispatchFloatBinaryOp::Add => 0,
                PcuDispatchFloatBinaryOp::Sub => 1,
                PcuDispatchFloatBinaryOp::Mul => 2,
                PcuDispatchFloatBinaryOp::Div => 3,
            },
            match self.underflow {
                PcuFloatUnderflowPolicy::IeeeAfterRounding => 0,
                PcuFloatUnderflowPolicy::RejectSubnormalResult => 1,
                PcuFloatUnderflowPolicy::AllowGradualUnderflow => 2,
            },
            self.operands[0],
            self.operands[1],
            self.broadcast[0] as u32,
            self.broadcast[1] as u32,
            match self.scalar {
                PcuScalarType::BF16 => 1,
                PcuScalarType::F8E4M3FN => 2,
                PcuScalarType::F8E5M2 => 3,
                _ => 0,
            },
            self.extent,
        ]
    }
}

/// Admits bounded six-format checked Add/Sub/Mul/Div before emission.
///
/// # Errors
/// Rejects other representations, malformed
/// typed SSA, incompatible bindings and any operation outside the canonical binary map.
#[allow(clippy::too_many_lines)] // Canonical pattern and frozen schema stay beside their validators.
pub fn validate_checked_float_binary_map(
    kernel: &PcuDispatchKernelIr<'_>,
) -> Result<PcuSpirvCheckedBinaryProfile, PcuSpirvError> {
    // The independently qualified U32 low-format executor opts into exactly the shared
    // one-binary-map descriptor. No unary, Clamp, compound or F32/F64 promise follows.
    if kernel
        .numerical_requirements
        .numerical_options
        .reproducibility
        == fusion_pcu::PcuReproducibility::PortableV1
        && fusion_pcu::describe_portable_v1_map(kernel).is_err()
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
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad { .. }),
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad { .. }),
        PcuDispatchOp::Data(PcuDispatchDataOp::CheckedFloatBinary {
            value_type:
                PcuValueType::Scalar(
                    scalar @ (PcuScalarType::F32
                    | PcuScalarType::F64
                    | PcuScalarType::F16
                    | PcuScalarType::BF16
                    | PcuScalarType::F8E4M3FN
                    | PcuScalarType::F8E5M2),
                ),
            op,
            underflow_policy,
            range_policy,
            ..
        }),
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore { .. }),
    ] = body
    else {
        return Err(PcuSpirvError::InvalidKernelSignature);
    };
    let schema = assess_checked_float_binary_operands(
        kernel,
        PcuValueType::Scalar(*scalar),
        *op,
        *underflow_policy,
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
    {
        return Err(PcuSpirvError::InvalidBinding);
    }
    let loaded = schema.input_bindings();
    let inputs = [loaded[0], *loaded.get(1).unwrap_or(&loaded[0])];
    let input_extents = schema
        .input_element_counts(usize::try_from(extent).map_err(|_| PcuSpirvError::InvalidBinding)?);
    let indices = schema.operand_indices();
    let mut declarations = [schema.output_binding(); 3];
    for (slot, binding) in kernel.bindings.iter().enumerate() {
        declarations[slot] = binding.reference();
    }
    let slots = schema.operand_inputs();
    let operands = [
        u32::try_from(slots[0]).map_err(|_| PcuSpirvError::InvalidBinding)?,
        u32::try_from(slots[1]).map_err(|_| PcuSpirvError::InvalidBinding)?,
    ];
    let input_extents = [
        u32::try_from(input_extents[0]).map_err(|_| PcuSpirvError::InvalidBinding)?,
        u32::try_from(input_extents[1]).map_err(|_| PcuSpirvError::InvalidBinding)?,
    ];
    Ok(PcuSpirvCheckedBinaryProfile {
        scalar: *scalar,
        operation: *op,
        underflow: *underflow_policy,
        range: *range_policy,
        extent,
        local_size: [64, 1, 1],
        operands,
        broadcast: indices.map(|index| index == PcuDispatchIndex::BindingElementZero),
        inputs,
        input_count: loaded.len(),
        input_extents,
        output: schema.output_binding(),
        declarations,
        declaration_count: kernel.bindings.len(),
    })
}

/// Emits the integer-only template with frozen specialization defaults and validated schema.
///
/// # Errors
/// Returns admission, unsupported version/capability or honest sink capacity errors.
pub fn lower_checked_float_binary_to_spirv<S: PcuSpirvSink>(
    kernel: &PcuDispatchKernelIr<'_>,
    options: PcuSpirvLoweringOptions,
    sink: &mut S,
) -> Result<(PcuSpirvModuleInfo, PcuSpirvCheckedBinaryProfile), PcuSpirvError> {
    let profile = validate_checked_float_binary_map(kernel)?;
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
    let narrow = profile.packing_lanes() > 1;
    let words = if narrow {
        low_bytecode::WORDS
    } else if profile.scalar == PcuScalarType::F64 {
        f64_bytecode::WORDS
    } else {
        bytecode::WORDS
    };
    let default_count = if narrow { 8 } else { 6 };
    let defaults = specialization_offsets(words, default_count);
    let values = profile.specialization();
    for (index, original) in words.iter().copied().enumerate() {
        let word = if index == 1 {
            options.version.0
        } else if index == 2 {
            options.generator
        } else if let Some(slot) = defaults[..default_count]
            .iter()
            .position(|offset| *offset == index)
        {
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

fn specialization_offsets(words: &[u32], slots: usize) -> [usize; 8] {
    let mut ids = [0; 8];
    let mut offsets = [0; 8];
    let mut cursor = 5;
    while cursor < words.len() {
        let instruction = &words[cursor..];
        let count = (instruction[0] >> 16) as usize;
        let opcode = instruction[0] & 0xffff;
        if opcode == 71 && count == 4 && instruction[2] == 1 {
            ids[instruction[3] as usize] = instruction[1];
        } else if opcode == 50
            && count == 4
            && let Some(slot) = ids[..slots].iter().position(|id| *id == instruction[2])
        {
            offsets[slot] = cursor + 3;
        }
        cursor += count;
    }
    assert!(
        offsets[..slots].iter().all(|offset| *offset >= 5),
        "offline template must contain every unique specialization default"
    );
    offsets
}
