//! Exact F32/F64 transport and checked sign-bit Neg with an explicit status-buffer ABI.

#[rustfmt::skip]
use fusion_pcu::{
    validate_checked_float_map_kernel,
    validate_f32_map_kernel,
    validate_f64_map_kernel,
    PcuBindingAccess,
    PcuBindingRef,
    PcuDispatchControlOp,
    PcuDispatchDataOp,
    PcuDispatchFloatUnaryOp,
    PcuDispatchIndex,
    PcuDispatchKernelIr,
    PcuDispatchOp,
    PcuDispatchOpCaps,
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
    SPIRV_MAGIC,
};

/// Numeric contract of the bounded integer-bit implementation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PcuSpirvBitOperation {
    /// Every supported floating representation is copied unchanged, including nonfinite payloads.
    Copy,
    /// Finite operands only; sign XOR preserves zeros and gradual subnormal results exactly.
    CheckedNeg(PcuFloatUnderflowPolicy),
}

/// Captured schema and private diagnostic ABI, independent of temporary source IR.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PcuSpirvBitMapProfile {
    pub scalar: PcuScalarType,
    pub operation: PcuSpirvBitOperation,
    pub extent: u32,
    pub local_size: [u32; 3],
}

impl PcuSpirvBitMapProfile {
    /// Source input is set zero, binding zero; output is binding one.
    pub const INPUT: PcuBindingRef = PcuBindingRef::new(0, 0);
    pub const OUTPUT: PcuBindingRef = PcuBindingRef::new(0, 1);
    /// Private u32 diagnostic per logical element: zero success, one invalid input, two underflow.
    pub const STATUS: PcuBindingRef = PcuBindingRef::new(0, 2);
}

/// Admits the complete bounded source schema before writing SPIR-V words.
///
/// # Errors
/// Rejects other arithmetic, bindings, parameters, clamp and invalid dataflow.
#[allow(clippy::too_many_lines)] // Exact binding/dataflow patterns stay adjacent to their cold validation.
pub fn validate_float_bit_map(
    kernel: &PcuDispatchKernelIr<'_>,
) -> Result<PcuSpirvBitMapProfile, PcuSpirvError> {
    // Exact scalar arithmetic does not certify the complete PortableV1 contract.
    if kernel
        .numerical_requirements
        .numerical_options
        .reproducibility
        == fusion_pcu::PcuReproducibility::PortableV1
    {
        return Err(PcuSpirvError::UnsupportedNumericalRequirements);
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
    let Some(PcuValueType::Scalar(scalar @ (PcuScalarType::F32 | PcuScalarType::F64))) = kernel
        .bindings
        .first()
        .and_then(|binding| binding.value_type())
    else {
        return Err(PcuSpirvError::InvalidBinding);
    };
    let operation = match body {
        [
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result,
                binding,
                index: load,
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                binding: output,
                index: store,
                value,
            }),
        ] if *binding == PcuSpirvBitMapProfile::INPUT
            && *output == PcuSpirvBitMapProfile::OUTPUT
            && result == value
            && *load == index
            && *store == index =>
        {
            PcuSpirvBitOperation::Copy
        }
        [
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result: loaded,
                binding,
                index: load,
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::CheckedFloatUnary {
                value_type: PcuValueType::Scalar(unary_scalar),
                op: PcuDispatchFloatUnaryOp::Neg,
                underflow_policy,
                range_policy: PcuRangePolicy::Reject,
                result: negated,
                value,
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                binding: output,
                index: store,
                value: stored,
            }),
        ] if *binding == PcuSpirvBitMapProfile::INPUT
            && *output == PcuSpirvBitMapProfile::OUTPUT
            && *unary_scalar == scalar
            && loaded == value
            && negated == stored
            && *load == index
            && *store == index =>
        {
            PcuSpirvBitOperation::CheckedNeg(*underflow_policy)
        }
        _ => {
            return Err(PcuSpirvError::UnsupportedInstruction(
                PcuDispatchOpCaps::ALU_CHECKED_FLOAT_UNARY,
            ));
        }
    };
    if let PcuSpirvBitOperation::CheckedNeg(underflow) = operation
        && (kernel.numerical_requirements.float_underflow != underflow
            || kernel.numerical_requirements.range_policy != PcuRangePolicy::Reject)
    {
        return Err(PcuSpirvError::UnsupportedNumericalRequirements);
    }
    match operation {
        PcuSpirvBitOperation::Copy => {
            match scalar {
                PcuScalarType::F32 => validate_f32_map_kernel(kernel)
                    .map_err(|_| PcuSpirvError::InvalidKernelSignature),
                PcuScalarType::F64 => validate_f64_map_kernel(kernel)
                    .map_err(|_| PcuSpirvError::InvalidKernelSignature),
                _ => unreachable!(),
            }
        }
        PcuSpirvBitOperation::CheckedNeg(_) => validate_checked_float_map_kernel(
            kernel,
            PcuValueType::Scalar(scalar),
            if scalar == PcuScalarType::F64 {
                PcuValueTypeCaps::FLOAT64
            } else {
                PcuValueTypeCaps::FLOAT32
            },
        )
        .map_err(|_| PcuSpirvError::InvalidKernelSignature),
    }?;
    validate_bit_map_bindings(kernel, extent)?;
    Ok(PcuSpirvBitMapProfile {
        scalar,
        operation,
        extent,
        local_size: [64, 1, 1],
    })
}

fn validate_bit_map_bindings(
    kernel: &PcuDispatchKernelIr<'_>,
    extent: u32,
) -> Result<(), PcuSpirvError> {
    if extent == 0
        || kernel.entry.logical_shape[1..] != [1, 1]
        || !kernel.parameters.is_empty()
        || !kernel.ports.is_empty()
        || kernel.bindings.len() != 2
        || !kernel.bindings.iter().any(|binding| {
            binding.reference() == PcuSpirvBitMapProfile::INPUT
                && binding.access == PcuBindingAccess::ReadOnly
        })
        || !kernel.bindings.iter().any(|binding| {
            binding.reference() == PcuSpirvBitMapProfile::OUTPUT
                && matches!(
                    binding.access,
                    PcuBindingAccess::WriteOnly | PcuBindingAccess::ReadWrite
                )
        })
    {
        return Err(PcuSpirvError::InvalidBinding);
    }
    Ok(())
}

/// Emits integer-only compute code and a separate status storage buffer.
///
/// This ABI intentionally differs from legacy `lower_dispatch_to_spirv`: callers must bind
/// all three descriptors and inspect terminal status before publishing private output.
///
/// # Errors
/// Rejects inadmissible source, version/capability combinations or insufficient sink capacity.
pub fn lower_float_bit_map_to_spirv<S: PcuSpirvSink>(
    kernel: &PcuDispatchKernelIr<'_>,
    options: PcuSpirvLoweringOptions,
    sink: &mut S,
) -> Result<(PcuSpirvModuleInfo, PcuSpirvBitMapProfile), PcuSpirvError> {
    let profile = validate_float_bit_map(kernel)?;
    if !options.capabilities.supports(PcuSpirvCapability::Shader) {
        return Err(PcuSpirvError::UnsupportedCapability(
            PcuSpirvCapability::Shader,
        ));
    }
    if options.version.0 & 0xffff_00ff != 0x0001_0000 || (options.version.0 >> 8) & 0xff > 6 {
        return Err(PcuSpirvError::UnsupportedVersion(options.version));
    }
    if profile.scalar == PcuScalarType::F64
        && !options.capabilities.supports(PcuSpirvCapability::Float64)
    {
        return Err(PcuSpirvError::UnsupportedCapability(
            PcuSpirvCapability::Float64,
        ));
    }
    let mut writer = BitWriter { sink, words: 0 };
    emit_module(&mut writer, options, profile)?;
    Ok((
        PcuSpirvModuleInfo {
            version: options.version,
            bound: if profile.scalar == PcuScalarType::F64 {
                59
            } else {
                45
            },
            word_count: writer.words,
            capabilities: if profile.scalar == PcuScalarType::F64 {
                PcuSpirvCapabilityCaps::SHADER.union(PcuSpirvCapabilityCaps::FLOAT64)
            } else {
                PcuSpirvCapabilityCaps::SHADER
            },
        },
        profile,
    ))
}

struct BitWriter<'a, S> {
    sink: &'a mut S,
    words: usize,
}

impl<S: PcuSpirvSink> BitWriter<'_, S> {
    fn word(&mut self, value: u32) -> Result<(), PcuSpirvError> {
        self.sink.push_word(value)?;
        self.words += 1;
        Ok(())
    }

    fn op(&mut self, opcode: u16, operands: &[u32]) -> Result<(), PcuSpirvError> {
        self.word(
            (u32::try_from(operands.len() + 1).map_err(|_| PcuSpirvError::IdSpaceExhausted)? << 16)
                | u32::from(opcode),
        )?;
        for value in operands {
            self.word(*value)?;
        }
        Ok(())
    }
}

#[allow(clippy::too_many_lines)] // One small module's ordered declarations and CFG remain adjacent.
fn emit_module<S: PcuSpirvSink>(
    writer: &mut BitWriter<'_, S>,
    options: PcuSpirvLoweringOptions,
    profile: PcuSpirvBitMapProfile,
) -> Result<(), PcuSpirvError> {
    let f64 = profile.scalar == PcuScalarType::F64;
    let bound = if f64 { 59 } else { 45 };
    for word in [SPIRV_MAGIC, options.version.0, options.generator, bound, 0] {
        writer.word(word)?;
    }
    writer.op(17, &[1])?; // Shader
    if f64 {
        writer.op(17, &[10])?;
    } // Float64; no Int64 arithmetic is needed.
    writer.op(14, &[0, 1])?; // Logical GLSL450
    if options.version.0 >= 0x0001_0400 {
        writer.op(15, &[5, 25, 0x6e69_616d, 0, 21, 22, 23, 24])?;
    } else {
        writer.op(15, &[5, 25, 0x6e69_616d, 0, 21])?;
    }
    writer.op(16, &[25, 17, 64, 1, 1])?;
    let storage = if options.version.0 >= 0x0001_0300 {
        12
    } else {
        2
    };
    let block = if storage == 12 { 2 } else { 3 };
    writer.op(71, &[9, 6, 4])?; // ArrayStride
    if f64 {
        writer.op(71, &[47, 6, 8])?;
        writer.op(71, &[48, block])?;
        writer.op(72, &[48, 0, 35, 0])?;
    }
    writer.op(71, &[10, block])?;
    writer.op(72, &[10, 0, 35, 0])?; // Offset
    writer.op(71, &[21, 11, 28])?; // GlobalInvocationId
    for (variable, binding) in [(22, 0), (23, 1), (24, 2)] {
        writer.op(71, &[variable, 34, 0])?;
        writer.op(71, &[variable, 33, binding])?;
    }
    writer.op(71, &[22, 24])?; // NonWritable
    writer.op(71, &[23, 25])?; // NonReadable
    writer.op(71, &[24, 25])?;
    writer.op(19, &[1])?;
    writer.op(33, &[2, 1])?;
    writer.op(20, &[4])?;
    writer.op(21, &[5, 32, 0])?;
    writer.op(23, &[6, 5, 3])?;
    writer.op(32, &[7, 1, 6])?;
    writer.op(32, &[8, 1, 5])?;
    writer.op(29, &[9, 5])?;
    writer.op(30, &[10, 9])?;
    writer.op(32, &[11, storage, 10])?;
    writer.op(32, &[12, storage, 5])?;
    if f64 {
        writer.op(22, &[46, 64])?;
        writer.op(29, &[47, 46])?;
        writer.op(30, &[48, 47])?;
        writer.op(32, &[49, storage, 48])?;
        writer.op(32, &[50, storage, 46])?;
        writer.op(23, &[51, 5, 2])?;
    }
    for (id, value) in [
        (13, 0),
        (14, 1),
        (15, 0x8000_0000),
        (16, if f64 { 0x7ff0_0000 } else { 0x7f80_0000 }),
        (17, if f64 { 0x000f_ffff } else { 0x007f_ffff }),
        (19, 2),
        (20, profile.extent),
    ] {
        writer.op(43, &[5, id, value])?;
    }
    writer.op(59, &[7, 21, 1])?;
    for id in [22, 23, 24] {
        writer.op(59, &[if f64 && id != 24 { 49 } else { 11 }, id, storage])?;
    }
    writer.op(54, &[1, 25, 0, 2])?;
    writer.op(248, &[26])?;
    writer.op(65, &[8, 27, 21, 13])?;
    writer.op(61, &[5, 28, 27])?;
    writer.op(176, &[4, 29, 28, 20])?;
    writer.op(247, &[31, 0])?;
    writer.op(250, &[29, 30, 31])?;
    writer.op(248, &[30])?;
    writer.op(65, &[if f64 { 50 } else { 12 }, 32, 22, 13, 28])?;
    let (loaded, high) = if f64 {
        writer.op(61, &[46, 52, 32])?;
        writer.op(124, &[51, 53, 52])?; // Bitcast: component zero is the low word.
        writer.op(81, &[5, 54, 53, 0])?;
        writer.op(81, &[5, 55, 53, 1])?;
        (52, 55)
    } else {
        writer.op(61, &[5, 33, 32])?;
        (33, 33)
    };
    let (result, status) = match profile.operation {
        PcuSpirvBitOperation::Copy => (loaded, 13),
        PcuSpirvBitOperation::CheckedNeg(policy) => {
            writer.op(199, &[5, 34, high, 16])?; // BitwiseAnd
            writer.op(170, &[4, 36, 34, 16])?; // IEqual
            let valid_status = if policy == PcuFloatUnderflowPolicy::RejectSubnormalResult {
                writer.op(199, &[5, 35, high, 17])?;
                writer.op(170, &[4, 37, 34, 13])?;
                let fraction = if f64 {
                    writer.op(197, &[5, 56, 35, 54])?;
                    56
                } else {
                    35
                };
                writer.op(171, &[4, 38, fraction, 13])?;
                writer.op(167, &[4, 39, 37, 38])?;
                writer.op(169, &[5, 41, 39, 19, 13])?;
                41
            } else {
                13
            };
            writer.op(169, &[5, 40, 36, 14, valid_status])?;
            writer.op(198, &[5, 42, high, 15])?; // BitwiseXor
            let negated = if f64 {
                writer.op(80, &[51, 57, 54, 42])?;
                writer.op(124, &[46, 58, 57])?;
                58
            } else {
                42
            };
            (negated, 40)
        }
    };
    writer.op(65, &[if f64 { 50 } else { 12 }, 43, 23, 13, 28])?;
    writer.op(62, &[43, result])?;
    writer.op(65, &[12, 44, 24, 13, 28])?;
    writer.op(62, &[44, status])?;
    writer.op(249, &[31])?;
    writer.op(248, &[31])?;
    writer.op(253, &[])?;
    writer.op(56, &[])
}
