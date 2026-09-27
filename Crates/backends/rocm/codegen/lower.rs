//! Narrow PCU dispatch IR to HIP C++ source lowering.
//!
//! The current subset is deliberately one-dimensional: f32 indexed maps and bounded integer maps.

use std::fmt::{
    self,
    Write as _,
};

use fusion_pcu::{
    PcuBindingAccess,
    PcuBindingRef,
    PcuBindingStorageClass,
    PcuBindingType,
    PcuDispatchAluOp,
    PcuDispatchConversion as PcuConversion,
    PcuDispatchControlOp,
    PcuDispatchDataOp,
    PcuDispatchFeatureCaps,
    PcuDispatchIndex,
    PcuDispatchKernelIr,
    PcuDispatchOp,
    PcuDispatchOpCaps,
    PcuDispatchValueId,
    PcuParameterValue,
    PcuScalarType,
    PcuF32MapValidationError,
    PcuValueType,
    PcuValueTypeCaps,
    validate_u64_checked_div_rem_kernel,
    validate_i32_checked_div_rem_kernel,
    validate_i64_checked_div_rem_kernel,
    validate_f32_map_kernel,
    validate_f64_map_kernel,
    validate_i32_map_kernel,
    validate_i64_map_kernel,
    validate_i16_map_kernel,
    validate_i16_checked_div_rem_kernel,
    validate_i8_checked_div_rem_kernel,
    validate_i8_map_kernel,
    validate_u32_identity_kernel,
    validate_u32_map_kernel,
    validate_u32_checked_div_rem_kernel,
    validate_u16_map_kernel,
    validate_u16_checked_div_rem_kernel,
    validate_u8_checked_div_rem_kernel,
    validate_u8_map_kernel,
    validate_u64_identity_kernel,
    validate_u64_map_kernel,
};

/// Structured reason why a dispatch kernel is outside the HIP source-lowering subset.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RocmLowerError {
    InvalidKernelShape,
    UnsupportedKernelInterface,
    UnsupportedRequirements,
    InvalidBinding(PcuBindingRef),
    InvalidBindingAccess(PcuBindingRef),
    DuplicateValue(PcuDispatchValueId),
    UndefinedValue(PcuDispatchValueId),
    InvalidValue(PcuDispatchValueId),
    UnsupportedConstant,
    UnsupportedAlu(PcuDispatchAluOp),
    UnsupportedOperation {
        index: usize,
        support: PcuDispatchOpCaps,
    },
    UnsupportedIndex,
    MissingStoreOrReturn,
    OperationAfterReturn,
    FormattingFailure,
}

impl fmt::Display for RocmLowerError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidKernelShape => formatter.write_str(
                "HIP lowering requires a nonzero one-dimensional logical dispatch shape",
            ),
            Self::UnsupportedKernelInterface => formatter.write_str(
                "HIP lowering supports storage bindings only; ports and parameters are unsupported",
            ),
            Self::UnsupportedRequirements => formatter.write_str(
                "HIP lowering supports its scalar f32, u32, and u64 profiles with mutable/read-only resources",
            ),
            Self::InvalidBinding(binding) => write!(
                formatter,
                "binding set {} slot {} must be an f32 storage binding without a builtin",
                binding.set, binding.binding
            ),
            Self::InvalidBindingAccess(binding) => write!(
                formatter,
                "binding set {} slot {} does not permit the requested access",
                binding.set, binding.binding
            ),
            Self::DuplicateValue(value) => {
                write!(formatter, "value {} is defined more than once", value.0)
            }
            Self::UndefinedValue(value) => {
                write!(formatter, "value {} is used before definition", value.0)
            }
            Self::InvalidValue(value) => {
                write!(
                    formatter,
                    "value {} is reserved and cannot appear in the IR",
                    value.0
                )
            }
            Self::UnsupportedConstant => {
                formatter.write_str("HIP lowering supports f32 constants only")
            }
            Self::UnsupportedAlu(op) => write!(formatter, "HIP lowering does not support {op:?}"),
            Self::UnsupportedOperation { index, support } => write!(
                formatter,
                "HIP lowering does not support operation {index} (support flags {:#x})",
                support.bits()
            ),
            Self::UnsupportedIndex => {
                formatter.write_str("HIP lowering supports invocation-ID indices only")
            }
            Self::MissingStoreOrReturn => {
                formatter.write_str("HIP kernel must contain at least one store and a return")
            }
            Self::OperationAfterReturn => formatter.write_str("operation appears after return"),
            Self::FormattingFailure => {
                formatter.write_str("generated HIP source formatting failed")
            }
        }
    }
}

impl std::error::Error for RocmLowerError {}

#[derive(Clone, Copy, PartialEq, Eq)]
enum HipScalarKind {
    F32,
    F64,
    U8,
    I8,
    U16,
    I16,
    U32,
    I32,
    U64,
    I64,
    F16Bits,
    Bf16Bits,
}

impl HipScalarKind {
    fn for_kernel(kernel: &PcuDispatchKernelIr<'_>) -> Self {
        kernel
            .bindings
            .iter()
            .find_map(|binding| match binding.binding_type {
                PcuBindingType::Value(PcuValueType::Scalar(scalar)) => Some(match scalar {
                    fusion_pcu::PcuScalarType::F64 => Self::F64,
                    fusion_pcu::PcuScalarType::F16 => Self::F16Bits,
                    fusion_pcu::PcuScalarType::BF16 => Self::Bf16Bits,
                    fusion_pcu::PcuScalarType::U8 => Self::U8,
                    fusion_pcu::PcuScalarType::I8 => {
                        if kernel_uses_checked_div_rem(kernel) {
                            Self::I8
                        } else {
                            Self::U8
                        }
                    }
                    fusion_pcu::PcuScalarType::U16 => Self::U16,
                    fusion_pcu::PcuScalarType::I16 => {
                        if kernel_uses_checked_div_rem(kernel) {
                            Self::I16
                        } else {
                            Self::U16
                        }
                    }
                    fusion_pcu::PcuScalarType::U64 => Self::U64,
                    fusion_pcu::PcuScalarType::I64 => {
                        if kernel_uses_checked_div_rem(kernel) {
                            Self::I64
                        } else {
                            Self::U64
                        }
                    }
                    fusion_pcu::PcuScalarType::U32 => Self::U32,
                    fusion_pcu::PcuScalarType::I32 => Self::I32,
                    _ => Self::F32,
                }),
                _ => None,
            })
            .unwrap_or(Self::F32)
    }

    const fn cpp_type(self) -> &'static str {
        match self {
            Self::F32 => "float",
            Self::F64 => "double",
            Self::U8 => "unsigned char",
            Self::I8 => "signed char",
            Self::U16 | Self::F16Bits | Self::Bf16Bits => "unsigned short",
            Self::I16 => "short",
            Self::U32 => "unsigned int",
            Self::I32 => "int",
            Self::U64 => "unsigned long long",
            Self::I64 => "long long",
        }
    }

    const fn pointer_type(self, is_const: bool) -> &'static str {
        match (self, is_const) {
            (Self::F32, true) => "const float*",
            (Self::F32, false) => "float*",
            (Self::F64, true) => "const double*",
            (Self::F64, false) => "double*",
            (Self::U8, true) => "const unsigned char*",
            (Self::U8, false) => "unsigned char*",
            (Self::I8, true) => "const signed char*",
            (Self::I8, false) => "signed char*",
            (Self::U16 | Self::F16Bits | Self::Bf16Bits, true) => "const unsigned short*",
            (Self::U16 | Self::F16Bits | Self::Bf16Bits, false) => "unsigned short*",
            (Self::I16, true) => "const short*",
            (Self::I16, false) => "short*",
            (Self::U32, true) => "const unsigned int*",
            (Self::U32, false) => "unsigned int*",
            (Self::I32, true) => "const int*",
            (Self::I32, false) => "int*",
            (Self::U64, true) => "const unsigned long long*",
            (Self::U64, false) => "unsigned long long*",
            (Self::I64, true) => "const long long*",
            (Self::I64, false) => "long long*",
        }
    }
}

/// Lower an eligible PCU dispatch kernel to a HIP C++ kernel source string.
///
/// The generated kernel is named `fusion_kernel`. Its launch geometry is expected to cover the
/// kernel's one-dimensional `logical_shape[0]`; the caller owns compilation and launch.
///
/// # Errors
///
/// Returns a lowering error when the kernel shape, interface, types, values, or operations fall
/// outside the supported HIP subset.
pub fn lower_dispatch_to_hip_source(
    kernel: &PcuDispatchKernelIr<'_>,
) -> Result<String, RocmLowerError> {
    lower_dispatch_to_hip_source_with_preamble(kernel, true)
}

/// Lower the same kernel for HIPRTC, whose online compiler supplies HIP builtins directly.
///
/// # Errors
///
/// Returns a lowering error when the kernel is outside the supported HIP subset.
pub fn lower_dispatch_to_hip_rtc_source(
    kernel: &PcuDispatchKernelIr<'_>,
) -> Result<String, RocmLowerError> {
    lower_dispatch_to_hip_source_with_preamble(kernel, false)
}

#[allow(clippy::too_many_lines)] // Keep direct and loop emission visibly parallel for this small subset.
fn lower_dispatch_to_hip_source_with_preamble(
    kernel: &PcuDispatchKernelIr<'_>,
    include_runtime_header: bool,
) -> Result<String, RocmLowerError> {
    validate_kernel(kernel)?;
    if is_exact_f32_f64_profile(kernel) {
        return lower_exact_f32_f64_to_hip(kernel, include_runtime_header);
    }
    if let Some(profile) = half_conversion_profile(kernel) {
        return lower_half_conversion_to_hip(kernel, include_runtime_header, profile);
    }
    if let Some(profile) = mixed_widening_profile(kernel) {
        return lower_mixed_widening_to_hip(kernel, include_runtime_header, profile);
    }
    let scalar_kind = HipScalarKind::for_kernel(kernel);
    let checked_division = kernel_uses_checked_div_rem(kernel);

    let mut source = if include_runtime_header {
        String::from("#include <hip/hip_runtime.h>\n\n")
    } else {
        String::new()
    };
    source.push_str("extern \"C\" __global__ void fusion_kernel(");
    for (index, binding) in kernel.bindings.iter().enumerate() {
        if index != 0 {
            source.push_str(", ");
        }
        let binding_ref = PcuBindingRef::new(binding.set, binding.binding);
        let used_for_store = kernel.ops.iter().any(|op| {
            matches!(
                op,
                PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore { binding: target, .. })
                    if *target == binding_ref
            )
        });
        let is_const = !used_for_store && binding.access == PcuBindingAccess::ReadOnly;
        let qualifier = scalar_kind.pointer_type(is_const);
        write!(
            &mut source,
            "{qualifier} binding_{}_{}",
            binding.set, binding.binding
        )
        .map_err(|_| RocmLowerError::FormattingFailure)?;
    }
    if checked_division {
        if !source.ends_with('(') {
            source.push_str(", ");
        }
        source.push_str("unsigned long long* fusion_fault_word");
    }
    write!(
        &mut source,
        ") {{\n    const unsigned int fusion_gid = blockIdx.x * blockDim.x + threadIdx.x;\n    if (fusion_gid >= {}u) return;\n",
        kernel.entry.logical_shape[0]
    )
    .map_err(|_| RocmLowerError::FormattingFailure)?;

    for op in kernel.ops.iter().copied() {
        match op {
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result,
                binding,
                index: PcuDispatchIndex::InvocationId,
            }) => writeln!(
                &mut source,
                "    {} v{} = binding_{}_{}[fusion_gid];",
                scalar_kind.cpp_type(),
                result.0,
                binding.set,
                binding.binding
            )
            .map_err(|_| RocmLowerError::FormattingFailure)?,
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result,
                binding,
                index: PcuDispatchIndex::BindingElementZero,
            }) => writeln!(
                &mut source,
                "    {} v{} = binding_{}_{}[0];",
                scalar_kind.cpp_type(),
                result.0,
                binding.set,
                binding.binding
            )
            .map_err(|_| RocmLowerError::FormattingFailure)?,
            PcuDispatchOp::Data(PcuDispatchDataOp::Constant {
                result,
                value: PcuParameterValue::F32(bits),
            }) => writeln!(
                &mut source,
                "    float v{} = __builtin_bit_cast(float, 0x{bits:08x}u);",
                result.0
            )
            .map_err(|_| RocmLowerError::FormattingFailure)?,
            PcuDispatchOp::Data(PcuDispatchDataOp::Alu {
                result,
                op,
                lhs,
                rhs,
                ..
            }) => {
                if op == PcuDispatchAluOp::Max {
                    writeln!(
                        &mut source,
                        "    float v{} = fmaxf(v{}, v{});",
                        result.0, lhs.0, rhs.0
                    )
                    .map_err(|_| RocmLowerError::FormattingFailure)?;
                } else {
                    let operator = match op {
                        PcuDispatchAluOp::Add => "+",
                        PcuDispatchAluOp::Sub => "-",
                        PcuDispatchAluOp::Mul => "*",
                        PcuDispatchAluOp::Div => "/",
                        _ => unreachable!("validated ALU op"),
                    };
                    if scalar_kind == HipScalarKind::U8 {
                        writeln!(
                            &mut source,
                            "    unsigned char v{} = static_cast<unsigned char>((static_cast<unsigned int>(v{}) {operator} static_cast<unsigned int>(v{})) & 0xffu);",
                            result.0, lhs.0, rhs.0
                        )
                        .map_err(|_| RocmLowerError::FormattingFailure)?;
                    } else if scalar_kind == HipScalarKind::U16 {
                        writeln!(
                            &mut source,
                            "    unsigned short v{} = static_cast<unsigned short>((static_cast<unsigned int>(v{}) {operator} static_cast<unsigned int>(v{})) & 0xffffu);",
                            result.0, lhs.0, rhs.0
                        )
                        .map_err(|_| RocmLowerError::FormattingFailure)?;
                    } else {
                        writeln!(
                            &mut source,
                            "    {} v{} = v{} {operator} v{};",
                            scalar_kind.cpp_type(),
                            result.0,
                            lhs.0,
                            rhs.0
                        )
                        .map_err(|_| RocmLowerError::FormattingFailure)?;
                    }
                }
            }
            PcuDispatchOp::Data(PcuDispatchDataOp::CheckedDivRem {
                value_type: PcuValueType::Scalar(fusion_pcu::PcuScalarType::U8),
                flags,
                quotient,
                remainder,
                lhs,
                rhs,
            }) => {
                if flags.bits() != 0 {
                    return Err(RocmLowerError::UnsupportedKernelInterface);
                }
                emit_u8_checked_div_rem(
                    &mut source,
                    "    ",
                    "fusion_gid",
                    quotient,
                    remainder,
                    lhs,
                    rhs,
                )?;
            }
            PcuDispatchOp::Data(PcuDispatchDataOp::CheckedDivRem {
                value_type: PcuValueType::Scalar(fusion_pcu::PcuScalarType::I8),
                flags,
                quotient,
                remainder,
                lhs,
                rhs,
            }) => {
                if flags.bits() != 0 {
                    return Err(RocmLowerError::UnsupportedKernelInterface);
                }
                emit_i8_checked_div_rem(
                    &mut source,
                    "    ",
                    "fusion_gid",
                    quotient,
                    remainder,
                    lhs,
                    rhs,
                )?;
            }
            PcuDispatchOp::Data(PcuDispatchDataOp::CheckedDivRem {
                value_type: PcuValueType::Scalar(fusion_pcu::PcuScalarType::U16),
                flags,
                quotient,
                remainder,
                lhs,
                rhs,
            }) => {
                if flags.bits() != 0 {
                    return Err(RocmLowerError::UnsupportedKernelInterface);
                }
                emit_u16_checked_div_rem(
                    &mut source,
                    "    ",
                    "fusion_gid",
                    quotient,
                    remainder,
                    lhs,
                    rhs,
                )?;
            }
            PcuDispatchOp::Data(PcuDispatchDataOp::CheckedDivRem {
                value_type: PcuValueType::Scalar(fusion_pcu::PcuScalarType::U32),
                flags,
                quotient,
                remainder,
                lhs,
                rhs,
            }) => {
                if flags.bits() != 0 {
                    return Err(RocmLowerError::UnsupportedKernelInterface);
                }
                emit_u32_checked_div_rem(
                    &mut source,
                    "    ",
                    "fusion_gid",
                    quotient,
                    remainder,
                    lhs,
                    rhs,
                )?;
            }
            PcuDispatchOp::Data(PcuDispatchDataOp::CheckedDivRem {
                value_type: PcuValueType::Scalar(fusion_pcu::PcuScalarType::U64),
                flags,
                quotient,
                remainder,
                lhs,
                rhs,
            }) => {
                if flags.bits() != 0 {
                    return Err(RocmLowerError::UnsupportedKernelInterface);
                }
                emit_u64_checked_div_rem(
                    &mut source,
                    "    ",
                    "fusion_gid",
                    quotient,
                    remainder,
                    lhs,
                    rhs,
                )?;
            }
            PcuDispatchOp::Data(PcuDispatchDataOp::CheckedDivRem {
                value_type: PcuValueType::Scalar(fusion_pcu::PcuScalarType::I16),
                flags,
                quotient,
                remainder,
                lhs,
                rhs,
            }) => {
                if flags.bits() != 0 {
                    return Err(RocmLowerError::UnsupportedKernelInterface);
                }
                emit_i16_checked_div_rem(
                    &mut source,
                    "    ",
                    "fusion_gid",
                    quotient,
                    remainder,
                    lhs,
                    rhs,
                )?;
            }
            PcuDispatchOp::Data(PcuDispatchDataOp::CheckedDivRem {
                value_type: PcuValueType::Scalar(fusion_pcu::PcuScalarType::I32),
                flags,
                quotient,
                remainder,
                lhs,
                rhs,
            }) => {
                if flags.bits() != 0 {
                    return Err(RocmLowerError::UnsupportedKernelInterface);
                }
                emit_i32_checked_div_rem(
                    &mut source,
                    "    ",
                    "fusion_gid",
                    quotient,
                    remainder,
                    lhs,
                    rhs,
                )?;
            }
            PcuDispatchOp::Data(PcuDispatchDataOp::CheckedDivRem {
                value_type: PcuValueType::Scalar(fusion_pcu::PcuScalarType::I64),
                flags,
                quotient,
                remainder,
                lhs,
                rhs,
            }) => {
                if flags.bits() != 0 {
                    return Err(RocmLowerError::UnsupportedKernelInterface);
                }
                emit_i64_checked_div_rem(
                    &mut source,
                    "    ",
                    "fusion_gid",
                    quotient,
                    remainder,
                    lhs,
                    rhs,
                )?;
            }
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                binding,
                index: PcuDispatchIndex::InvocationId,
                value,
            }) => writeln!(
                &mut source,
                "    binding_{}_{}[fusion_gid] = v{};",
                binding.set, binding.binding, value.0
            )
            .map_err(|_| RocmLowerError::FormattingFailure)?,
            PcuDispatchOp::Control(PcuDispatchControlOp::Return) => {
                source.push_str("    return;\n");
            }
            PcuDispatchOp::GridStrideLoop { extent, body } => {
                let stride = kernel.entry.logical_shape[0];
                // The increment executes after every active iteration, including the last one.
                // Keep it in u32 only when even the conservative upper bound for that increment
                // (extent - 1 + stride) remains representable.
                let u32_induction =
                    u128::from(extent - 1) + u128::from(stride) <= u128::from(u32::MAX);
                let (index_type, suffix) = if u32_induction {
                    ("unsigned int", "u")
                } else {
                    ("unsigned long long", "ull")
                };
                writeln!(
                    &mut source,
                    "    for ({index_type} fusion_idx = fusion_gid; fusion_idx < {extent}{suffix}; fusion_idx += {stride}{suffix}) {{"
                )
                .map_err(|_| RocmLowerError::FormattingFailure)?;
                for body_op in body.iter().copied() {
                    emit_hip_data_op(
                        &mut source,
                        body_op,
                        "fusion_idx",
                        scalar_kind,
                        checked_division,
                    )?;
                }
                source.push_str("    }\n");
            }
            _ => unreachable!("validated operation"),
        }
    }
    source.push_str("}\n");
    Ok(source)
}

fn is_exact_f32_f64_profile(kernel: &PcuDispatchKernelIr<'_>) -> bool {
    matches!(
        conversion_profile_parts(kernel),
        Some((
            PcuConversion::F32ToF64Exact,
            PcuValueType::Scalar(PcuScalarType::F32),
            PcuValueType::Scalar(PcuScalarType::F64)
        ))
    )
}

fn lower_exact_f32_f64_to_hip(
    kernel: &PcuDispatchKernelIr<'_>,
    include_runtime_header: bool,
) -> Result<String, RocmLowerError> {
    let (body, grid) = match kernel.ops {
        [PcuDispatchOp::GridStrideLoop { body, .. }, _] => (*body, true),
        _ => (&kernel.ops[..3], false),
    };
    let PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad { binding: input, .. }) = body[0] else {
        unreachable!("conversion profile was validated")
    };
    let PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
        binding: output, ..
    }) = body[2]
    else {
        unreachable!("conversion profile was validated")
    };
    let mut source = if include_runtime_header {
        String::from("#include <hip/hip_runtime.h>\n\n")
    } else {
        String::new()
    };
    source.push_str(EXACT_F32_F64_HIP_HELPER);
    source.push_str("\nextern \"C\" __global__ void fusion_kernel(");
    for (index, binding) in kernel.bindings.iter().enumerate() {
        if index != 0 {
            source.push_str(", ");
        }
        let (kind, input_binding) = if binding.reference() == input {
            ("unsigned int", true)
        } else {
            ("unsigned long long", false)
        };
        write!(
            &mut source,
            "{}{kind}* binding_{}_{}",
            if input_binding && binding.access == PcuBindingAccess::ReadOnly {
                "const "
            } else {
                ""
            },
            binding.set,
            binding.binding
        )
        .map_err(|_| RocmLowerError::FormattingFailure)?;
    }
    if grid {
        let PcuDispatchOp::GridStrideLoop { extent, .. } = kernel.ops[0] else {
            unreachable!()
        };
        writeln!(
            &mut source,
            ") {{\n    const unsigned int fusion_start = blockIdx.x * blockDim.x + threadIdx.x;\n    const unsigned int fusion_invocations = {}u;\n    if (fusion_start >= fusion_invocations) return;\n    for (unsigned long long fusion_idx = fusion_start; fusion_idx < {extent}ull; fusion_idx += fusion_invocations) {{\n        binding_{}_{}[fusion_idx] = fusion_f32_to_f64_bits(binding_{}_{}[fusion_idx]);\n    }}\n}}",
            kernel.entry.logical_shape[0], output.set, output.binding, input.set, input.binding
        ).map_err(|_| RocmLowerError::FormattingFailure)?;
    } else {
        writeln!(
            &mut source,
            ") {{\n    const unsigned int fusion_gid = blockIdx.x * blockDim.x + threadIdx.x;\n    if (fusion_gid >= {}u) return;\n    binding_{}_{}[fusion_gid] = fusion_f32_to_f64_bits(binding_{}_{}[fusion_gid]);\n}}",
            kernel.entry.logical_shape[0], output.set, output.binding, input.set, input.binding
        ).map_err(|_| RocmLowerError::FormattingFailure)?;
    }
    Ok(source)
}

const EXACT_F32_F64_HIP_HELPER: &str = r"__device__ __forceinline__ unsigned long long fusion_f32_to_f64_bits(unsigned int bits) {
    const unsigned long long sign = (unsigned long long)(bits >> 31u) << 63u;
    const unsigned int exponent = (bits >> 23u) & 0xffu;
    const unsigned int fraction = bits & 0x007fffffu;
    if (exponent == 0xffu) return sign | (0x7ffull << 52u) | ((unsigned long long)fraction << 29u);
    if (exponent != 0u) return sign | ((unsigned long long)(exponent + 896u) << 52u) | ((unsigned long long)fraction << 29u);
    if (fraction == 0u) return sign;
    unsigned int leading = 0u;
    while ((fraction >> (leading + 1u)) != 0u) ++leading;
    const int unbiased = (int)leading - 149;
    const unsigned long long significand = (unsigned long long)fraction << (52u - leading);
    return sign | ((unsigned long long)(unbiased + 1023) << 52u) | (significand & 0x000fffffffffffffull);
}";

#[derive(Clone, Copy, PartialEq, Eq)]
enum MixedWideningProfile {
    I8ToI16,
    U8ToU16,
    I16ToI32,
    U16ToU32,
    I32ToI64,
    U32ToU64,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum HalfConversionProfile {
    F32ToF16Bits,
    F16BitsToF32,
    F32ToBf16Bits,
    Bf16BitsToF32,
}

fn half_conversion_profile(kernel: &PcuDispatchKernelIr<'_>) -> Option<HalfConversionProfile> {
    let (conversion, input_type, output_type) = conversion_profile_parts(kernel)?;
    match (conversion, input_type, output_type) {
        (
            PcuConversion::F32ToF16Bits,
            PcuValueType::Scalar(PcuScalarType::F32),
            PcuValueType::Scalar(PcuScalarType::F16),
        ) => Some(HalfConversionProfile::F32ToF16Bits),
        (
            PcuConversion::F16BitsToF32,
            PcuValueType::Scalar(PcuScalarType::F16),
            PcuValueType::Scalar(PcuScalarType::F32),
        ) => Some(HalfConversionProfile::F16BitsToF32),
        (
            PcuConversion::F32ToBf16Bits,
            PcuValueType::Scalar(PcuScalarType::F32),
            PcuValueType::Scalar(PcuScalarType::BF16),
        ) => Some(HalfConversionProfile::F32ToBf16Bits),
        (
            PcuConversion::Bf16BitsToF32,
            PcuValueType::Scalar(PcuScalarType::BF16),
            PcuValueType::Scalar(PcuScalarType::F32),
        ) => Some(HalfConversionProfile::Bf16BitsToF32),
        _ => None,
    }
}

fn conversion_profile_parts(
    kernel: &PcuDispatchKernelIr<'_>,
) -> Option<(PcuConversion, PcuValueType, PcuValueType)> {
    if fusion_pcu::validate_typed_dispatch_value_flow(kernel).is_err() || kernel.bindings.len() != 2
    {
        return None;
    }
    let (body, grid) = match kernel.ops {
        [
            PcuDispatchOp::Data(_),
            PcuDispatchOp::Data(_),
            PcuDispatchOp::Data(_),
            PcuDispatchOp::Control(PcuDispatchControlOp::Return),
        ] => (&kernel.ops[..3], false),
        [
            PcuDispatchOp::GridStrideLoop { body, .. },
            PcuDispatchOp::Control(PcuDispatchControlOp::Return),
        ] => (*body, true),
        _ => return None,
    };
    let [
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
            result: loaded,
            binding: input,
            index: load_index,
        }),
        PcuDispatchOp::Data(PcuDispatchDataOp::Convert {
            result: converted,
            value,
            conversion,
        }),
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
            binding: output,
            index: store_index,
            value: stored,
        }),
    ] = body
    else {
        return None;
    };
    let expected_index = if grid {
        PcuDispatchIndex::GridStrideId
    } else {
        PcuDispatchIndex::InvocationId
    };
    if *load_index != expected_index
        || *store_index != expected_index
        || value != loaded
        || stored != converted
        || input == output
    {
        return None;
    }
    let input_binding = kernel
        .bindings
        .iter()
        .find(|binding| binding.reference() == *input)?;
    let output_binding = kernel
        .bindings
        .iter()
        .find(|binding| binding.reference() == *output)?;
    if input_binding.storage != PcuBindingStorageClass::Storage
        || output_binding.storage != PcuBindingStorageClass::Storage
        || input_binding.builtin.is_some()
        || output_binding.builtin.is_some()
        || !matches!(
            input_binding.access,
            PcuBindingAccess::ReadOnly | PcuBindingAccess::ReadWrite
        )
        || !matches!(
            output_binding.access,
            PcuBindingAccess::WriteOnly | PcuBindingAccess::ReadWrite
        )
    {
        return None;
    }
    Some((
        *conversion,
        input_binding.value_type()?,
        output_binding.value_type()?,
    ))
}

fn lower_half_conversion_to_hip(
    kernel: &PcuDispatchKernelIr<'_>,
    include_runtime_header: bool,
    profile: HalfConversionProfile,
) -> Result<String, RocmLowerError> {
    let (input_type, output_type, function) = match profile {
        HalfConversionProfile::F32ToF16Bits => {
            ("float", "unsigned short", "fusion_f32_to_f16_bits")
        }
        HalfConversionProfile::F16BitsToF32 => {
            ("unsigned short", "float", "fusion_f16_bits_to_f32")
        }
        HalfConversionProfile::F32ToBf16Bits => {
            ("float", "unsigned short", "fusion_f32_to_bf16_bits")
        }
        HalfConversionProfile::Bf16BitsToF32 => {
            ("unsigned short", "float", "fusion_bf16_bits_to_f32")
        }
    };
    let (body, _) = match kernel.ops {
        [PcuDispatchOp::GridStrideLoop { body, .. }, _] => (*body, true),
        _ => (&kernel.ops[..3], false),
    };
    let PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad { binding: input, .. }) = body[0] else {
        unreachable!("profile was validated")
    };
    let mut source = if include_runtime_header {
        String::from("#include <hip/hip_runtime.h>\n\n")
    } else {
        String::new()
    };
    source.push_str(half_conversion_helpers(profile));
    source.push_str("\nextern \"C\" __global__ void fusion_kernel(");
    for (index, binding) in kernel.bindings.iter().enumerate() {
        if index != 0 {
            source.push_str(", ");
        }
        let (kind, is_input) = if binding.reference() == input {
            (input_type, true)
        } else {
            (output_type, false)
        };
        let qualifier = if is_input && binding.access == PcuBindingAccess::ReadOnly {
            "const "
        } else {
            ""
        };
        write!(
            &mut source,
            "{qualifier}{kind}* binding_{}_{}",
            binding.set, binding.binding
        )
        .map_err(|_| RocmLowerError::FormattingFailure)?;
    }
    source.push_str(") {\n");
    emit_half_conversion_body(&mut source, kernel, input_type, output_type, function)?;
    source.push_str("}\n");
    Ok(source)
}

fn emit_half_conversion_body(
    source: &mut String,
    kernel: &PcuDispatchKernelIr<'_>,
    input_type: &str,
    output_type: &str,
    function: &str,
) -> Result<(), RocmLowerError> {
    let (body, grid) = match kernel.ops {
        [PcuDispatchOp::GridStrideLoop { body, .. }, _] => (*body, true),
        _ => (&kernel.ops[..3], false),
    };
    let PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
        result: loaded,
        binding: input,
        ..
    }) = body[0]
    else {
        unreachable!("profile was validated")
    };
    let PcuDispatchOp::Data(PcuDispatchDataOp::Convert {
        result: converted,
        value,
        ..
    }) = body[1]
    else {
        unreachable!("profile was validated")
    };
    let PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
        binding: output, ..
    }) = body[2]
    else {
        unreachable!("profile was validated")
    };
    if grid {
        let PcuDispatchOp::GridStrideLoop { extent, .. } = kernel.ops[0] else {
            unreachable!()
        };
        writeln!(
            source,
            "    const unsigned int fusion_start = blockIdx.x * blockDim.x + threadIdx.x;\n    const unsigned int fusion_invocations = {}u;\n    if (fusion_start >= fusion_invocations) return;\n    for (unsigned long long fusion_idx = fusion_start; fusion_idx < {extent}ull; fusion_idx += fusion_invocations) {{\n        {input_type} v{} = binding_{}_{}[fusion_idx];\n        {output_type} v{} = {function}(v{});\n        binding_{}_{}[fusion_idx] = v{};\n    }}",
            kernel.entry.logical_shape[0],
            loaded.0,
            input.set,
            input.binding,
            converted.0,
            value.0,
            output.set,
            output.binding,
            converted.0
        )
        .map_err(|_| RocmLowerError::FormattingFailure)?;
    } else {
        writeln!(
            source,
            "    const unsigned int fusion_gid = blockIdx.x * blockDim.x + threadIdx.x;\n    if (fusion_gid >= {}u) return;\n    {input_type} v{} = binding_{}_{}[fusion_gid];\n    {output_type} v{} = {function}(v{});\n    binding_{}_{}[fusion_gid] = v{};",
            kernel.entry.logical_shape[0],
            loaded.0,
            input.set,
            input.binding,
            converted.0,
            value.0,
            output.set,
            output.binding,
            converted.0
        )
        .map_err(|_| RocmLowerError::FormattingFailure)?;
    }
    Ok(())
}

const fn half_conversion_helpers(profile: HalfConversionProfile) -> &'static str {
    match profile {
        HalfConversionProfile::F32ToF16Bits => F32_TO_F16_HELPER,
        HalfConversionProfile::F16BitsToF32 => F16_TO_F32_HELPER,
        HalfConversionProfile::F32ToBf16Bits => F32_TO_BF16_HELPER,
        HalfConversionProfile::Bf16BitsToF32 => BF16_TO_F32_HELPER,
    }
}

const F32_TO_F16_HELPER: &str = r"__device__ __forceinline__ unsigned int fusion_round_shift_rne(unsigned int value, unsigned int shift) {
    if (shift == 0u) return value;
    if (shift >= 32u) return 0u;
    const unsigned int truncated = value >> shift;
    const unsigned int mask = (1u << shift) - 1u;
    const unsigned int discarded = value & mask;
    const unsigned int halfway = 1u << (shift - 1u);
    return truncated + ((discarded > halfway || (discarded == halfway && (truncated & 1u) != 0u)) ? 1u : 0u);
}

__device__ __forceinline__ unsigned short fusion_f32_to_f16_bits(float value) {
    const unsigned int bits = __builtin_bit_cast(unsigned int, value);
    const unsigned short sign = (unsigned short)((bits >> 16u) & 0x8000u);
    const int exponent = (int)((bits >> 23u) & 0xffu);
    const unsigned int fraction = bits & 0x007fffffu;
    if (exponent == 0xff) {
        if (fraction == 0u) return (unsigned short)(sign | 0x7c00u);
        const unsigned short payload = (unsigned short)(fraction >> 13u);
        return (unsigned short)(sign | 0x7c00u | payload | 0x0200u);
    }
    if (exponent == 0) return sign;
    const int unbiased = exponent - 127;
    const unsigned int significand = 0x00800000u | fraction;
    if (unbiased > 15) return (unsigned short)(sign | 0x7c00u);
    if (unbiased >= -14) {
        const unsigned int rounded = fusion_round_shift_rne(significand, 13u);
        const unsigned int half_exponent = rounded == 0x0800u
            ? (unsigned int)(unbiased + 16)
            : (unsigned int)(unbiased + 15);
        const unsigned int half_fraction = rounded == 0x0800u ? 0u : rounded & 0x03ffu;
        if (half_exponent >= 31u) return (unsigned short)(sign | 0x7c00u);
        return (unsigned short)(sign | (half_exponent << 10u) | half_fraction);
    }
    if (unbiased < -25) return sign;
    const unsigned int rounded = fusion_round_shift_rne(significand, (unsigned int)(-unbiased - 1));
    return (unsigned short)(sign | rounded);
}";

const F16_TO_F32_HELPER: &str = r"__device__ __forceinline__ float fusion_f16_bits_to_f32(unsigned short value) {
    const unsigned int bits = (unsigned int)value;
    const unsigned int sign = (bits & 0x8000u) << 16u;
    const unsigned int exponent = (bits >> 10u) & 0x1fu;
    const unsigned int fraction = bits & 0x03ffu;
    unsigned int output;
    if (exponent == 0u) {
        if (fraction == 0u) output = sign;
        else {
            unsigned int normalized = fraction;
            int unbiased = -14;
            while ((normalized & 0x0400u) == 0u) { normalized <<= 1u; unbiased -= 1; }
            normalized &= 0x03ffu;
            output = sign | ((unsigned int)(unbiased + 127) << 23u) | (normalized << 13u);
        }
    } else if (exponent == 0x1fu) {
        output = sign | 0x7f800000u | (fraction << 13u);
    } else {
        output = sign | ((exponent + 112u) << 23u) | (fraction << 13u);
    }
    return __builtin_bit_cast(float, output);
}";

const F32_TO_BF16_HELPER: &str = r"__device__ __forceinline__ unsigned short fusion_f32_to_bf16_bits(float value) {
    const unsigned int bits = __builtin_bit_cast(unsigned int, value);
    const unsigned int exponent = bits & 0x7f800000u;
    const unsigned int fraction = bits & 0x007fffffu;
    if (exponent == 0x7f800000u && fraction != 0u) {
        const unsigned short upper = (unsigned short)(bits >> 16u);
        return (unsigned short)((upper & 0x8000u) | (upper & 0x7fffu) | 0x0040u);
    }
    const unsigned int upper = bits >> 16u;
    const unsigned int lower = bits & 0xffffu;
    const unsigned int increment = (lower > 0x8000u || (lower == 0x8000u && (upper & 1u) != 0u)) ? 1u : 0u;
    return (unsigned short)(upper + increment);
}";

const BF16_TO_F32_HELPER: &str = r"__device__ __forceinline__ float fusion_bf16_bits_to_f32(unsigned short value) {
    const unsigned int bits = (unsigned int)value << 16u;
    return __builtin_bit_cast(float, bits);
}";

fn mixed_widening_profile(kernel: &PcuDispatchKernelIr<'_>) -> Option<MixedWideningProfile> {
    if fusion_pcu::validate_typed_dispatch_value_flow(kernel).is_err() || kernel.bindings.len() != 2
    {
        return None;
    }
    let (body, grid) = match kernel.ops {
        [
            PcuDispatchOp::Data(_),
            PcuDispatchOp::Data(_),
            PcuDispatchOp::Data(_),
            PcuDispatchOp::Control(PcuDispatchControlOp::Return),
        ] => (&kernel.ops[..3], false),
        [
            PcuDispatchOp::GridStrideLoop { body, .. },
            PcuDispatchOp::Control(PcuDispatchControlOp::Return),
        ] => (*body, true),
        _ => return None,
    };
    let [
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
            result: loaded,
            binding: input,
            index: load_index,
        }),
        PcuDispatchOp::Data(PcuDispatchDataOp::Convert {
            result: converted,
            value,
            conversion,
        }),
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
            binding: output,
            index: store_index,
            value: stored,
        }),
    ] = body
    else {
        return None;
    };
    let expected_index = if grid {
        PcuDispatchIndex::GridStrideId
    } else {
        PcuDispatchIndex::InvocationId
    };
    if *load_index != expected_index
        || *store_index != expected_index
        || value != loaded
        || stored != converted
        || input == output
    {
        return None;
    }
    let input_binding = kernel
        .bindings
        .iter()
        .find(|binding| binding.reference() == *input)?;
    let output_binding = kernel
        .bindings
        .iter()
        .find(|binding| binding.reference() == *output)?;
    if input_binding.storage != PcuBindingStorageClass::Storage
        || output_binding.storage != PcuBindingStorageClass::Storage
        || input_binding.builtin.is_some()
        || output_binding.builtin.is_some()
        || !matches!(
            input_binding.access,
            PcuBindingAccess::ReadOnly | PcuBindingAccess::ReadWrite
        )
        || !matches!(
            output_binding.access,
            PcuBindingAccess::WriteOnly | PcuBindingAccess::ReadWrite
        )
    {
        return None;
    }
    let input_type = input_binding.value_type()?;
    let output_type = output_binding.value_type()?;
    exact_integer_widening_profile(*conversion, input_type, output_type)
}

fn exact_integer_widening_profile(
    conversion: fusion_pcu::PcuDispatchConversion,
    input: PcuValueType,
    output: PcuValueType,
) -> Option<MixedWideningProfile> {
    use fusion_pcu::{
        PcuDispatchConversion as C,
        PcuScalarType as S,
    };
    let profile = match conversion {
        C::I8ToI16
            if input == PcuValueType::Scalar(S::I8) && output == PcuValueType::Scalar(S::I16) =>
        {
            MixedWideningProfile::I8ToI16
        }
        C::U8ToU16
            if input == PcuValueType::Scalar(S::U8) && output == PcuValueType::Scalar(S::U16) =>
        {
            MixedWideningProfile::U8ToU16
        }
        C::I16ToI32
            if input == PcuValueType::Scalar(S::I16) && output == PcuValueType::Scalar(S::I32) =>
        {
            MixedWideningProfile::I16ToI32
        }
        C::U16ToU32
            if input == PcuValueType::Scalar(S::U16) && output == PcuValueType::Scalar(S::U32) =>
        {
            MixedWideningProfile::U16ToU32
        }
        C::I32ToI64
            if input == PcuValueType::Scalar(S::I32) && output == PcuValueType::Scalar(S::I64) =>
        {
            MixedWideningProfile::I32ToI64
        }
        C::U32ToU64
            if input == PcuValueType::Scalar(S::U32) && output == PcuValueType::Scalar(S::U64) =>
        {
            MixedWideningProfile::U32ToU64
        }
        _ => return None,
    };
    Some(profile)
}

fn lower_mixed_widening_to_hip(
    kernel: &PcuDispatchKernelIr<'_>,
    include_runtime_header: bool,
    profile: MixedWideningProfile,
) -> Result<String, RocmLowerError> {
    let (source_kind, target_kind, cast) = match profile {
        MixedWideningProfile::I8ToI16 => (HipScalarKind::I8, HipScalarKind::I16, "short"),
        MixedWideningProfile::U8ToU16 => (HipScalarKind::U8, HipScalarKind::U16, "unsigned short"),
        MixedWideningProfile::I16ToI32 => (HipScalarKind::I16, HipScalarKind::I32, "int"),
        MixedWideningProfile::U16ToU32 => (HipScalarKind::U16, HipScalarKind::U32, "unsigned int"),
        MixedWideningProfile::I32ToI64 => (HipScalarKind::I32, HipScalarKind::I64, "long long"),
        MixedWideningProfile::U32ToU64 => {
            (HipScalarKind::U32, HipScalarKind::U64, "unsigned long long")
        }
    };
    let (body, grid) = match kernel.ops {
        [PcuDispatchOp::GridStrideLoop { body, .. }, _] => (*body, true),
        _ => (&kernel.ops[..3], false),
    };
    let PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
        result: loaded,
        binding: input,
        ..
    }) = body[0]
    else {
        unreachable!("profile was validated")
    };
    let PcuDispatchOp::Data(PcuDispatchDataOp::Convert {
        result: converted,
        value,
        ..
    }) = body[1]
    else {
        unreachable!("profile was validated")
    };
    let PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
        binding: output, ..
    }) = body[2]
    else {
        unreachable!("profile was validated")
    };
    let mut source = if include_runtime_header {
        String::from("#include <hip/hip_runtime.h>\n\n")
    } else {
        String::new()
    };
    source.push_str("extern \"C\" __global__ void fusion_kernel(");
    for (index, binding) in kernel.bindings.iter().enumerate() {
        if index != 0 {
            source.push_str(", ");
        }
        let kind = if binding.reference() == input {
            source_kind
        } else {
            target_kind
        };
        let qualifier =
            if binding.reference() == input && binding.access == PcuBindingAccess::ReadOnly {
                "const "
            } else {
                ""
            };
        write!(
            &mut source,
            "{qualifier}{}* binding_{}_{}",
            kind.cpp_type(),
            binding.set,
            binding.binding
        )
        .map_err(|_| RocmLowerError::FormattingFailure)?;
    }
    source.push_str(") {\n");
    if grid {
        let PcuDispatchOp::GridStrideLoop { extent, .. } = kernel.ops[0] else {
            unreachable!()
        };
        writeln!(&mut source, "    const unsigned int fusion_start = blockIdx.x * blockDim.x + threadIdx.x;\n    const unsigned int fusion_invocations = {}u;\n    if (fusion_start >= fusion_invocations) return;\n    for (unsigned long long fusion_idx = fusion_start; fusion_idx < {extent}ull; fusion_idx += fusion_invocations) {{\n        {} v{} = binding_{}_{}[fusion_idx];\n        {cast} v{} = static_cast<{cast}>(v{});\n        binding_{}_{}[fusion_idx] = v{};\n    }}", kernel.entry.logical_shape[0], source_kind.cpp_type(), loaded.0, input.set, input.binding, converted.0, value.0, output.set, output.binding, converted.0).map_err(|_| RocmLowerError::FormattingFailure)?;
    } else {
        writeln!(&mut source, "    const unsigned int fusion_gid = blockIdx.x * blockDim.x + threadIdx.x;\n    if (fusion_gid >= {}u) return;\n    {} v{} = binding_{}_{}[fusion_gid];\n    {cast} v{} = static_cast<{cast}>(v{});\n    binding_{}_{}[fusion_gid] = v{};", kernel.entry.logical_shape[0], source_kind.cpp_type(), loaded.0, input.set, input.binding, converted.0, value.0, output.set, output.binding, converted.0).map_err(|_| RocmLowerError::FormattingFailure)?;
    }
    source.push_str("}\n");
    Ok(source)
}

#[allow(clippy::too_many_lines)]
fn emit_hip_data_op(
    source: &mut String,
    op: PcuDispatchOp<'_>,
    index_name: &str,
    scalar_kind: HipScalarKind,
    checked_division: bool,
) -> Result<(), RocmLowerError> {
    match op {
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
            result,
            binding,
            index: PcuDispatchIndex::GridStrideId,
        }) => writeln!(
            source,
            "        {} v{} = binding_{}_{}[{index_name}];",
            scalar_kind.cpp_type(),
            result.0,
            binding.set,
            binding.binding
        )
        .map_err(|_| RocmLowerError::FormattingFailure),
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
            result,
            binding,
            index: PcuDispatchIndex::BindingElementZero,
        }) => writeln!(
            source,
            "        {} v{} = binding_{}_{}[0];",
            scalar_kind.cpp_type(),
            result.0,
            binding.set,
            binding.binding
        )
        .map_err(|_| RocmLowerError::FormattingFailure),
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
            binding,
            index: PcuDispatchIndex::GridStrideId,
            value,
        }) => writeln!(
            source,
            "        binding_{}_{}[{index_name}] = v{};",
            binding.set, binding.binding, value.0
        )
        .map_err(|_| RocmLowerError::FormattingFailure),
        PcuDispatchOp::Data(PcuDispatchDataOp::Constant {
            result,
            value: PcuParameterValue::F32(bits),
        }) => writeln!(
            source,
            "        float v{} = __builtin_bit_cast(float, 0x{bits:08x}u);",
            result.0
        )
        .map_err(|_| RocmLowerError::FormattingFailure),
        PcuDispatchOp::Data(PcuDispatchDataOp::Alu {
            result,
            op,
            lhs,
            rhs,
            ..
        }) => {
            if op == PcuDispatchAluOp::Max {
                writeln!(
                    source,
                    "        float v{} = fmaxf(v{}, v{});",
                    result.0, lhs.0, rhs.0
                )
                .map_err(|_| RocmLowerError::FormattingFailure)
            } else {
                let operator = match op {
                    PcuDispatchAluOp::Add => "+",
                    PcuDispatchAluOp::Sub => "-",
                    PcuDispatchAluOp::Mul => "*",
                    PcuDispatchAluOp::Div => "/",
                    _ => unreachable!("validated ALU op"),
                };
                if scalar_kind == HipScalarKind::U8 {
                    writeln!(
                        source,
                        "        unsigned char v{} = static_cast<unsigned char>((static_cast<unsigned int>(v{}) {operator} static_cast<unsigned int>(v{})) & 0xffu);",
                        result.0, lhs.0, rhs.0
                    )
                    .map_err(|_| RocmLowerError::FormattingFailure)
                } else if scalar_kind == HipScalarKind::U16 {
                    writeln!(
                        source,
                        "        unsigned short v{} = static_cast<unsigned short>((static_cast<unsigned int>(v{}) {operator} static_cast<unsigned int>(v{})) & 0xffffu);",
                        result.0, lhs.0, rhs.0
                    )
                    .map_err(|_| RocmLowerError::FormattingFailure)
                } else {
                    writeln!(
                        source,
                        "        {} v{} = v{} {operator} v{};",
                        scalar_kind.cpp_type(),
                        result.0,
                        lhs.0,
                        rhs.0
                    )
                    .map_err(|_| RocmLowerError::FormattingFailure)
                }
            }
        }
        PcuDispatchOp::Data(PcuDispatchDataOp::CheckedDivRem {
            value_type: PcuValueType::Scalar(fusion_pcu::PcuScalarType::U8),
            flags,
            quotient,
            remainder,
            lhs,
            rhs,
        }) => {
            if !checked_division || flags.bits() != 0 {
                return Err(RocmLowerError::UnsupportedKernelInterface);
            }
            emit_u8_checked_div_rem(
                source, "        ", index_name, quotient, remainder, lhs, rhs,
            )
        }
        PcuDispatchOp::Data(PcuDispatchDataOp::CheckedDivRem {
            value_type: PcuValueType::Scalar(fusion_pcu::PcuScalarType::I8),
            flags,
            quotient,
            remainder,
            lhs,
            rhs,
        }) => {
            if !checked_division || flags.bits() != 0 {
                return Err(RocmLowerError::UnsupportedKernelInterface);
            }
            emit_i8_checked_div_rem(
                source, "        ", index_name, quotient, remainder, lhs, rhs,
            )
        }
        PcuDispatchOp::Data(PcuDispatchDataOp::CheckedDivRem {
            value_type: PcuValueType::Scalar(fusion_pcu::PcuScalarType::U32),
            flags,
            quotient,
            remainder,
            lhs,
            rhs,
        }) => {
            if !checked_division || flags.bits() != 0 {
                return Err(RocmLowerError::UnsupportedKernelInterface);
            }
            emit_u32_checked_div_rem(
                source, "        ", index_name, quotient, remainder, lhs, rhs,
            )
        }
        PcuDispatchOp::Data(PcuDispatchDataOp::CheckedDivRem {
            value_type: PcuValueType::Scalar(fusion_pcu::PcuScalarType::U16),
            flags,
            quotient,
            remainder,
            lhs,
            rhs,
        }) => {
            if !checked_division || flags.bits() != 0 {
                return Err(RocmLowerError::UnsupportedKernelInterface);
            }
            emit_u16_checked_div_rem(
                source, "        ", index_name, quotient, remainder, lhs, rhs,
            )
        }
        PcuDispatchOp::Data(PcuDispatchDataOp::CheckedDivRem {
            value_type: PcuValueType::Scalar(fusion_pcu::PcuScalarType::U64),
            flags,
            quotient,
            remainder,
            lhs,
            rhs,
        }) => {
            if !checked_division || flags.bits() != 0 {
                return Err(RocmLowerError::UnsupportedKernelInterface);
            }
            emit_u64_checked_div_rem(
                source, "        ", index_name, quotient, remainder, lhs, rhs,
            )
        }
        PcuDispatchOp::Data(PcuDispatchDataOp::CheckedDivRem {
            value_type: PcuValueType::Scalar(fusion_pcu::PcuScalarType::I16),
            flags,
            quotient,
            remainder,
            lhs,
            rhs,
        }) => {
            if !checked_division || flags.bits() != 0 {
                return Err(RocmLowerError::UnsupportedKernelInterface);
            }
            emit_i16_checked_div_rem(
                source, "        ", index_name, quotient, remainder, lhs, rhs,
            )
        }
        PcuDispatchOp::Data(PcuDispatchDataOp::CheckedDivRem {
            value_type: PcuValueType::Scalar(fusion_pcu::PcuScalarType::I32),
            flags,
            quotient,
            remainder,
            lhs,
            rhs,
        }) => {
            if !checked_division || flags.bits() != 0 {
                return Err(RocmLowerError::UnsupportedKernelInterface);
            }
            emit_i32_checked_div_rem(
                source, "        ", index_name, quotient, remainder, lhs, rhs,
            )
        }
        PcuDispatchOp::Data(PcuDispatchDataOp::CheckedDivRem {
            value_type: PcuValueType::Scalar(fusion_pcu::PcuScalarType::I64),
            flags,
            quotient,
            remainder,
            lhs,
            rhs,
        }) => {
            if !checked_division || flags.bits() != 0 {
                return Err(RocmLowerError::UnsupportedKernelInterface);
            }
            emit_i64_checked_div_rem(
                source, "        ", index_name, quotient, remainder, lhs, rhs,
            )
        }
        _ => unreachable!("validated grid-stride body operation"),
    }
}

fn emit_u8_checked_div_rem(
    source: &mut String,
    indent: &str,
    logical_index: &str,
    quotient: PcuDispatchValueId,
    remainder: PcuDispatchValueId,
    lhs: PcuDispatchValueId,
    rhs: PcuDispatchValueId,
) -> Result<(), RocmLowerError> {
    writeln!(
        source,
        "{indent}unsigned char v{} = 0u; unsigned char v{} = 0u;\n{indent}if (v{} == 0u) {{ atomicMin(fusion_fault_word, (static_cast<unsigned long long>({logical_index}) << 2u) | 1ull); }} else {{ v{} = static_cast<unsigned char>(static_cast<unsigned int>(v{}) / static_cast<unsigned int>(v{})); v{} = static_cast<unsigned char>(static_cast<unsigned int>(v{}) % static_cast<unsigned int>(v{})); }}",
        quotient.0, remainder.0, rhs.0, quotient.0, lhs.0, rhs.0, remainder.0, lhs.0, rhs.0
    )
    .map_err(|_| RocmLowerError::FormattingFailure)
}

fn emit_i8_checked_div_rem(
    source: &mut String,
    indent: &str,
    logical_index: &str,
    quotient: PcuDispatchValueId,
    remainder: PcuDispatchValueId,
    lhs: PcuDispatchValueId,
    rhs: PcuDispatchValueId,
) -> Result<(), RocmLowerError> {
    writeln!(
        source,
        "{indent}signed char v{} = 0; signed char v{} = v{};\n{indent}if (v{} == 0) {{ atomicMin(fusion_fault_word, (static_cast<unsigned long long>({logical_index}) << 2u) | 1ull); }} else if (v{} == (-127 - 1) && v{} == -1) {{ atomicMin(fusion_fault_word, (static_cast<unsigned long long>({logical_index}) << 2u) | 2ull); }} else {{ v{} = static_cast<signed char>(static_cast<int>(v{}) / static_cast<int>(v{})); v{} = static_cast<signed char>(static_cast<int>(v{}) % static_cast<int>(v{})); }}",
        quotient.0, remainder.0, lhs.0, rhs.0, lhs.0, rhs.0,
        quotient.0, lhs.0, rhs.0, remainder.0, lhs.0, rhs.0
    )
    .map_err(|_| RocmLowerError::FormattingFailure)
}

fn emit_u16_checked_div_rem(
    source: &mut String,
    indent: &str,
    logical_index: &str,
    quotient: PcuDispatchValueId,
    remainder: PcuDispatchValueId,
    lhs: PcuDispatchValueId,
    rhs: PcuDispatchValueId,
) -> Result<(), RocmLowerError> {
    writeln!(
        source,
        "{indent}unsigned short v{} = 0u; unsigned short v{} = 0u;\n{indent}if (v{} == 0u) {{ atomicMin(fusion_fault_word, (static_cast<unsigned long long>({logical_index}) << 2u) | 1ull); }} else {{ v{} = static_cast<unsigned short>(static_cast<unsigned int>(v{}) / static_cast<unsigned int>(v{})); v{} = static_cast<unsigned short>(static_cast<unsigned int>(v{}) % static_cast<unsigned int>(v{})); }}",
        quotient.0, remainder.0, rhs.0, quotient.0, lhs.0, rhs.0, remainder.0, lhs.0, rhs.0
    )
    .map_err(|_| RocmLowerError::FormattingFailure)
}

fn emit_u32_checked_div_rem(
    source: &mut String,
    indent: &str,
    logical_index: &str,
    quotient: PcuDispatchValueId,
    remainder: PcuDispatchValueId,
    lhs: PcuDispatchValueId,
    rhs: PcuDispatchValueId,
) -> Result<(), RocmLowerError> {
    writeln!(
        source,
        "{indent}unsigned int v{} = 0u; unsigned int v{} = 0u;\n{indent}if (v{} == 0u) {{ atomicMin(fusion_fault_word, (static_cast<unsigned long long>({logical_index}) << 2u) | 1ull); }} else {{ v{} = v{} / v{}; v{} = v{} % v{}; }}",
        quotient.0,
        remainder.0,
        rhs.0,
        quotient.0,
        lhs.0,
        rhs.0,
        remainder.0,
        lhs.0,
        rhs.0
    )
    .map_err(|_| RocmLowerError::FormattingFailure)
}

fn emit_u64_checked_div_rem(
    source: &mut String,
    indent: &str,
    logical_index: &str,
    quotient: PcuDispatchValueId,
    remainder: PcuDispatchValueId,
    lhs: PcuDispatchValueId,
    rhs: PcuDispatchValueId,
) -> Result<(), RocmLowerError> {
    writeln!(
        source,
        "{indent}unsigned long long v{} = 0ull; unsigned long long v{} = 0ull;\n{indent}if (v{} == 0ull) {{ atomicMin(fusion_fault_word, (static_cast<unsigned long long>({logical_index}) << 2u) | 1ull); }} else {{ v{} = v{} / v{}; v{} = v{} % v{}; }}",
        quotient.0, remainder.0, rhs.0, quotient.0, lhs.0, rhs.0, remainder.0, lhs.0, rhs.0
    )
    .map_err(|_| RocmLowerError::FormattingFailure)
}

fn emit_i16_checked_div_rem(
    source: &mut String,
    indent: &str,
    logical_index: &str,
    quotient: PcuDispatchValueId,
    remainder: PcuDispatchValueId,
    lhs: PcuDispatchValueId,
    rhs: PcuDispatchValueId,
) -> Result<(), RocmLowerError> {
    writeln!(
        source,
        "{indent}short v{} = 0; short v{} = v{};\n{indent}if (v{} == 0) {{ atomicMin(fusion_fault_word, (static_cast<unsigned long long>({logical_index}) << 2u) | 1ull); }} else if (v{} == (-32767 - 1) && v{} == -1) {{ atomicMin(fusion_fault_word, (static_cast<unsigned long long>({logical_index}) << 2u) | 2ull); }} else {{ v{} = static_cast<short>(static_cast<int>(v{}) / static_cast<int>(v{})); v{} = static_cast<short>(static_cast<int>(v{}) % static_cast<int>(v{})); }}",
        quotient.0, remainder.0, lhs.0, rhs.0, lhs.0, rhs.0,
        quotient.0, lhs.0, rhs.0, remainder.0, lhs.0, rhs.0
    )
    .map_err(|_| RocmLowerError::FormattingFailure)
}

fn emit_i32_checked_div_rem(
    source: &mut String,
    indent: &str,
    logical_index: &str,
    quotient: PcuDispatchValueId,
    remainder: PcuDispatchValueId,
    lhs: PcuDispatchValueId,
    rhs: PcuDispatchValueId,
) -> Result<(), RocmLowerError> {
    writeln!(
        source,
        "{indent}int v{} = 0; int v{} = v{};\n{indent}if (v{} == 0) {{ atomicMin(fusion_fault_word, (static_cast<unsigned long long>({logical_index}) << 2u) | 1ull); }} else if (v{} == (-2147483647 - 1) && v{} == -1) {{ atomicMin(fusion_fault_word, (static_cast<unsigned long long>({logical_index}) << 2u) | 2ull); }} else {{ v{} = v{} / v{}; v{} = v{} % v{}; }}",
        quotient.0, remainder.0, lhs.0, rhs.0, lhs.0, rhs.0,
        quotient.0, lhs.0, rhs.0, remainder.0, lhs.0, rhs.0
    )
    .map_err(|_| RocmLowerError::FormattingFailure)
}

fn emit_i64_checked_div_rem(
    source: &mut String,
    indent: &str,
    logical_index: &str,
    quotient: PcuDispatchValueId,
    remainder: PcuDispatchValueId,
    lhs: PcuDispatchValueId,
    rhs: PcuDispatchValueId,
) -> Result<(), RocmLowerError> {
    writeln!(
        source,
        "{indent}long long v{} = 0ll; long long v{} = v{};\n{indent}if (v{} == 0ll) {{ atomicMin(fusion_fault_word, (static_cast<unsigned long long>({logical_index}) << 2u) | 1ull); }} else if (v{} == (-9223372036854775807ll - 1ll) && v{} == -1ll) {{ atomicMin(fusion_fault_word, (static_cast<unsigned long long>({logical_index}) << 2u) | 2ull); }} else {{ v{} = v{} / v{}; v{} = v{} % v{}; }}",
        quotient.0, remainder.0, lhs.0, rhs.0, lhs.0, rhs.0,
        quotient.0, lhs.0, rhs.0, remainder.0, lhs.0, rhs.0
    )
    .map_err(|_| RocmLowerError::FormattingFailure)
}

mod validation;
use validation::{
    kernel_uses_checked_div_rem,
    validate_kernel,
};

#[cfg(test)]
mod tests {
    use super::{
        RocmLowerError,
        lower_dispatch_to_hip_rtc_source,
        lower_dispatch_to_hip_source,
    };
    use fusion_pcu::{
        PcuBinding,
        PcuBindingAccess,
        PcuBindingRef,
        PcuBindingStorageClass,
        PcuDispatchAluOp,
        PcuDispatchConversion,
        PcuDispatchControlOp,
        PcuDispatchDataOp,
        PcuDispatchEntryPoint,
        PcuDispatchIndex,
        PcuDispatchKernelIr,
        PcuDispatchOp,
        PcuDispatchValueId,
        PcuDispatchFeatureCaps,
        PcuKernelId,
        PcuParameterValue,
        PcuPort,
        PcuPortBlocking,
        PcuPortDirection,
        PcuPortBackpressure,
        PcuPortRate,
        PcuPortReliability,
        PcuValueType,
        PcuValueTypeCaps,
    };
    use fusion_pcu::model::PcuIntegerDivFlags;

    fn kernel<'a>(
        ops: &'a [PcuDispatchOp<'a>],
        bindings: &'a [PcuBinding<'a>],
    ) -> PcuDispatchKernelIr<'a> {
        PcuDispatchKernelIr {
            id: PcuKernelId(1),
            entry: PcuDispatchEntryPoint {
                name: "test",
                logical_shape: [64, 1, 1],
            },
            bindings,
            ports: &[],
            parameters: &[],
            ops,
            type_caps: PcuValueTypeCaps::empty(),
            feature_caps: PcuDispatchFeatureCaps::empty(),
        }
    }

    #[test]
    fn lowers_exact_f32_f64_widening_through_integer_bits_for_direct_and_grid() {
        let bindings = [
            PcuBinding::value(
                Some("in"),
                0,
                0,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::ReadOnly,
                PcuValueType::f32(),
            ),
            PcuBinding::value(
                Some("out"),
                0,
                1,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::WriteOnly,
                PcuValueType::f64(),
            ),
        ];
        let direct = [
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result: PcuDispatchValueId(1),
                binding: PcuBindingRef::new(0, 0),
                index: PcuDispatchIndex::InvocationId,
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::Convert {
                result: PcuDispatchValueId(2),
                value: PcuDispatchValueId(1),
                conversion: PcuDispatchConversion::F32ToF64Exact,
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                binding: PcuBindingRef::new(0, 1),
                index: PcuDispatchIndex::InvocationId,
                value: PcuDispatchValueId(2),
            }),
            PcuDispatchOp::Control(PcuDispatchControlOp::Return),
        ];
        let mut direct_kernel = kernel(&direct, &bindings);
        direct_kernel.type_caps = PcuValueTypeCaps::for_scalar(fusion_pcu::PcuScalarType::F32)
            .union(PcuValueTypeCaps::for_scalar(fusion_pcu::PcuScalarType::F64));
        let source = lower_dispatch_to_hip_rtc_source(&direct_kernel).unwrap();
        assert!(source.contains("const unsigned int* binding_0_0"));
        assert!(source.contains("unsigned long long* binding_0_1"));
        assert!(source.contains("fusion_f32_to_f64_bits"));
        assert!(source.contains("fraction << 29u"));
        assert!(!source.contains("double v2"));

        let reversed = [bindings[1], bindings[0]];
        let mut reversed_kernel = kernel(&direct, &reversed);
        reversed_kernel.type_caps = direct_kernel.type_caps;
        let source = lower_dispatch_to_hip_rtc_source(&reversed_kernel).unwrap();
        assert!(source.contains(
            "fusion_kernel(unsigned long long* binding_0_1, const unsigned int* binding_0_0)"
        ));

        let body = [
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result: PcuDispatchValueId(1),
                binding: PcuBindingRef::new(0, 0),
                index: PcuDispatchIndex::GridStrideId,
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::Convert {
                result: PcuDispatchValueId(2),
                value: PcuDispatchValueId(1),
                conversion: PcuDispatchConversion::F32ToF64Exact,
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                binding: PcuBindingRef::new(0, 1),
                index: PcuDispatchIndex::GridStrideId,
                value: PcuDispatchValueId(2),
            }),
        ];
        let grid = [
            PcuDispatchOp::GridStrideLoop {
                extent: u32::MAX,
                body: &body,
            },
            PcuDispatchOp::Control(PcuDispatchControlOp::Return),
        ];
        let mut grid_kernel = kernel(&grid, &bindings);
        grid_kernel.type_caps = PcuValueTypeCaps::for_scalar(fusion_pcu::PcuScalarType::F32)
            .union(PcuValueTypeCaps::for_scalar(fusion_pcu::PcuScalarType::F64));
        let source = lower_dispatch_to_hip_rtc_source(&grid_kernel).unwrap();
        assert!(source.contains("fusion_idx < 4294967295ull"));
        assert!(source.contains("unsigned long long fusion_idx"));
        assert!(source.contains("fusion_f32_to_f64_bits(binding_0_0[fusion_idx])"));
    }

    fn mixed_widening_bindings(
        source: fusion_pcu::PcuScalarType,
        target: fusion_pcu::PcuScalarType,
    ) -> [PcuBinding<'static>; 2] {
        [
            PcuBinding::value(
                Some("input"),
                0,
                0,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::ReadOnly,
                PcuValueType::Scalar(source),
            ),
            PcuBinding::value(
                Some("output"),
                0,
                1,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::WriteOnly,
                PcuValueType::Scalar(target),
            ),
        ]
    }

    type WideningCase = (
        fusion_pcu::PcuScalarType,
        fusion_pcu::PcuScalarType,
        fusion_pcu::PcuDispatchConversion,
        &'static str,
        &'static str,
    );

    const fn widening_cases() -> [WideningCase; 6] {
        use fusion_pcu::{
            PcuDispatchConversion as C,
            PcuScalarType as S,
        };
        [
            (S::I8, S::I16, C::I8ToI16, "signed char", "short"),
            (S::U8, S::U16, C::U8ToU16, "unsigned char", "unsigned short"),
            (S::I16, S::I32, C::I16ToI32, "short", "int"),
            (
                S::U16,
                S::U32,
                C::U16ToU32,
                "unsigned short",
                "unsigned int",
            ),
            (S::I32, S::I64, C::I32ToI64, "int", "long long"),
            (
                S::U32,
                S::U64,
                C::U32ToU64,
                "unsigned int",
                "unsigned long long",
            ),
        ]
    }

    #[test]
    fn lowers_exact_mixed_integer_widening_for_direct_and_grid_stride() {
        for (source_scalar, target_scalar, conversion, source_type, target_type) in widening_cases()
        {
            let bindings = mixed_widening_bindings(source_scalar, target_scalar);
            let direct = [
                PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                    result: PcuDispatchValueId(1),
                    binding: PcuBindingRef::new(0, 0),
                    index: PcuDispatchIndex::InvocationId,
                }),
                PcuDispatchOp::Data(PcuDispatchDataOp::Convert {
                    result: PcuDispatchValueId(2),
                    value: PcuDispatchValueId(1),
                    conversion,
                }),
                PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                    binding: PcuBindingRef::new(0, 1),
                    index: PcuDispatchIndex::InvocationId,
                    value: PcuDispatchValueId(2),
                }),
                PcuDispatchOp::Control(PcuDispatchControlOp::Return),
            ];
            let source = lower_dispatch_to_hip_rtc_source(&kernel(&direct, &bindings)).unwrap();
            assert!(source.contains(&format!("const {source_type}* binding_0_0")));
            assert!(source.contains(&format!("{target_type}* binding_0_1")));
            assert!(source.contains(&format!("{source_type} v1 = binding_0_0[fusion_gid];")));
            assert!(source.contains(&format!(
                "{target_type} v2 = static_cast<{target_type}>(v1);"
            )));

            let body = [
                PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                    result: PcuDispatchValueId(1),
                    binding: PcuBindingRef::new(0, 0),
                    index: PcuDispatchIndex::GridStrideId,
                }),
                PcuDispatchOp::Data(PcuDispatchDataOp::Convert {
                    result: PcuDispatchValueId(2),
                    value: PcuDispatchValueId(1),
                    conversion,
                }),
                PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                    binding: PcuBindingRef::new(0, 1),
                    index: PcuDispatchIndex::GridStrideId,
                    value: PcuDispatchValueId(2),
                }),
            ];
            let grid = [
                PcuDispatchOp::GridStrideLoop {
                    extent: 2048,
                    body: &body,
                },
                PcuDispatchOp::Control(PcuDispatchControlOp::Return),
            ];
            let mut grid_kernel = kernel(&grid, &bindings);
            grid_kernel.entry.logical_shape[0] = 250;
            let source = lower_dispatch_to_hip_rtc_source(&grid_kernel).unwrap();
            assert!(source.contains(&format!("{source_type} v1 = binding_0_0[fusion_idx];")));
            assert!(source.contains(&format!(
                "{target_type} v2 = static_cast<{target_type}>(v1);"
            )));
            assert!(source.contains("const unsigned int fusion_invocations = 250u;"));
            assert!(source.contains("if (fusion_start >= fusion_invocations) return;"));
            assert!(source.contains("fusion_idx += fusion_invocations"));
            assert!(!source.contains("gridDim.x * blockDim.x"));
        }
    }

    #[test]
    fn lowers_u32_identity_copy_without_admitting_arithmetic() {
        let bindings = [
            PcuBinding::value(
                Some("input"),
                0,
                0,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::ReadOnly,
                PcuValueType::u32(),
            ),
            PcuBinding::value(
                Some("output"),
                0,
                1,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::WriteOnly,
                PcuValueType::u32(),
            ),
        ];
        let copy = [
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result: PcuDispatchValueId(1),
                binding: PcuBindingRef::new(0, 0),
                index: PcuDispatchIndex::InvocationId,
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                binding: PcuBindingRef::new(0, 1),
                index: PcuDispatchIndex::InvocationId,
                value: PcuDispatchValueId(1),
            }),
            PcuDispatchOp::Control(PcuDispatchControlOp::Return),
        ];
        let source = lower_dispatch_to_hip_rtc_source(&kernel(&copy, &bindings)).unwrap();
        assert!(source.contains("const unsigned int* binding_0_0"));
        assert!(source.contains("unsigned int* binding_0_1"));
        assert!(source.contains("unsigned int v1 = binding_0_0[fusion_gid];"));
        let grid_body = [
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result: PcuDispatchValueId(1),
                binding: PcuBindingRef::new(0, 0),
                index: PcuDispatchIndex::GridStrideId,
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                binding: PcuBindingRef::new(0, 1),
                index: PcuDispatchIndex::GridStrideId,
                value: PcuDispatchValueId(1),
            }),
        ];
        let grid_ops = [
            PcuDispatchOp::GridStrideLoop {
                extent: 256,
                body: &grid_body,
            },
            PcuDispatchOp::Control(PcuDispatchControlOp::Return),
        ];
        let grid_source = lower_dispatch_to_hip_rtc_source(&kernel(&grid_ops, &bindings)).unwrap();
        assert!(grid_source.contains("unsigned int v1 = binding_0_0[fusion_idx];"));
        let arithmetic = [
            copy[0],
            PcuDispatchOp::Data(PcuDispatchDataOp::Alu {
                value_type: fusion_pcu::PcuValueType::f32(),
                result: PcuDispatchValueId(2),
                op: PcuDispatchAluOp::Add,
                lhs: PcuDispatchValueId(1),
                rhs: PcuDispatchValueId(1),
            }),
            copy[1],
            copy[2],
        ];
        assert!(lower_dispatch_to_hip_rtc_source(&kernel(&arithmetic, &bindings)).is_err());
    }

    #[test]
    fn lowers_f64_add_map_with_double_hip_operations() {
        let bindings = [
            PcuBinding::value(
                Some("a"),
                0,
                0,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::ReadOnly,
                PcuValueType::f64(),
            ),
            PcuBinding::value(
                Some("b"),
                0,
                1,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::ReadOnly,
                PcuValueType::f64(),
            ),
            PcuBinding::value(
                Some("out"),
                0,
                2,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::WriteOnly,
                PcuValueType::f64(),
            ),
        ];
        let ops = [
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result: PcuDispatchValueId(1),
                binding: PcuBindingRef::new(0, 0),
                index: PcuDispatchIndex::InvocationId,
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result: PcuDispatchValueId(2),
                binding: PcuBindingRef::new(0, 1),
                index: PcuDispatchIndex::InvocationId,
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::Alu {
                value_type: PcuValueType::f64(),
                result: PcuDispatchValueId(3),
                op: PcuDispatchAluOp::Add,
                lhs: PcuDispatchValueId(1),
                rhs: PcuDispatchValueId(2),
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                binding: PcuBindingRef::new(0, 2),
                index: PcuDispatchIndex::InvocationId,
                value: PcuDispatchValueId(3),
            }),
            PcuDispatchOp::Control(PcuDispatchControlOp::Return),
        ];
        let source = lower_dispatch_to_hip_rtc_source(&kernel(&ops, &bindings)).unwrap();
        assert!(source.contains("const double* binding_0_0"));
        assert!(source.contains("double* binding_0_2"));
        assert!(source.contains("double v1 = binding_0_0[fusion_gid];"));
        assert!(source.contains("double v3 = v1 + v2;"));

        let body = [
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result: PcuDispatchValueId(1),
                binding: PcuBindingRef::new(0, 0),
                index: PcuDispatchIndex::GridStrideId,
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result: PcuDispatchValueId(2),
                binding: PcuBindingRef::new(0, 1),
                index: PcuDispatchIndex::GridStrideId,
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::Alu {
                value_type: PcuValueType::f64(),
                result: PcuDispatchValueId(3),
                op: PcuDispatchAluOp::Mul,
                lhs: PcuDispatchValueId(1),
                rhs: PcuDispatchValueId(2),
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                binding: PcuBindingRef::new(0, 2),
                index: PcuDispatchIndex::GridStrideId,
                value: PcuDispatchValueId(3),
            }),
        ];
        let loop_ops = [
            PcuDispatchOp::GridStrideLoop {
                extent: 256,
                body: &body,
            },
            PcuDispatchOp::Control(PcuDispatchControlOp::Return),
        ];
        let loop_source = lower_dispatch_to_hip_rtc_source(&kernel(&loop_ops, &bindings)).unwrap();
        assert!(loop_source.contains("double v1 = binding_0_0[fusion_idx];"));
        assert!(loop_source.contains("double v3 = v1 * v2;"));
    }

    #[test]
    fn lowers_u64_add_map_with_native_hip_u64_operations() {
        let bindings = [
            PcuBinding::value(
                Some("a"),
                0,
                0,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::ReadOnly,
                PcuValueType::u64(),
            ),
            PcuBinding::value(
                Some("b"),
                0,
                1,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::ReadOnly,
                PcuValueType::u64(),
            ),
            PcuBinding::value(
                Some("out"),
                0,
                2,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::WriteOnly,
                PcuValueType::u64(),
            ),
        ];
        let ops = [
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result: PcuDispatchValueId(1),
                binding: PcuBindingRef::new(0, 0),
                index: PcuDispatchIndex::InvocationId,
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result: PcuDispatchValueId(2),
                binding: PcuBindingRef::new(0, 1),
                index: PcuDispatchIndex::InvocationId,
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::Alu {
                value_type: PcuValueType::u64(),
                result: PcuDispatchValueId(3),
                op: PcuDispatchAluOp::Add,
                lhs: PcuDispatchValueId(1),
                rhs: PcuDispatchValueId(2),
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                binding: PcuBindingRef::new(0, 2),
                index: PcuDispatchIndex::InvocationId,
                value: PcuDispatchValueId(3),
            }),
            PcuDispatchOp::Control(PcuDispatchControlOp::Return),
        ];
        let source = lower_dispatch_to_hip_rtc_source(&kernel(&ops, &bindings)).unwrap();
        assert!(source.contains("const unsigned long long* binding_0_0"));
        assert!(source.contains("unsigned long long* binding_0_2"));
        assert!(source.contains("unsigned long long v1 = binding_0_0[fusion_gid];"));
        assert!(source.contains("unsigned long long v3 = v1 + v2;"));

        let body = [
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result: PcuDispatchValueId(1),
                binding: PcuBindingRef::new(0, 0),
                index: PcuDispatchIndex::GridStrideId,
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result: PcuDispatchValueId(2),
                binding: PcuBindingRef::new(0, 1),
                index: PcuDispatchIndex::GridStrideId,
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::Alu {
                value_type: PcuValueType::u64(),
                result: PcuDispatchValueId(3),
                op: PcuDispatchAluOp::Mul,
                lhs: PcuDispatchValueId(1),
                rhs: PcuDispatchValueId(2),
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                binding: PcuBindingRef::new(0, 2),
                index: PcuDispatchIndex::GridStrideId,
                value: PcuDispatchValueId(3),
            }),
        ];
        let loop_ops = [
            PcuDispatchOp::GridStrideLoop {
                extent: 256,
                body: &body,
            },
            PcuDispatchOp::Control(PcuDispatchControlOp::Return),
        ];
        let loop_source = lower_dispatch_to_hip_rtc_source(&kernel(&loop_ops, &bindings)).unwrap();
        assert!(loop_source.contains("unsigned long long v1 = binding_0_0[fusion_idx];"));
        assert!(loop_source.contains("unsigned long long v3 = v1 * v2;"));
    }

    #[test]
    #[allow(clippy::too_many_lines)] // Keep direct and loop u32 lowering coverage together.
    fn lowers_u32_wrapping_map_with_unsigned_alu_in_direct_and_loop_kernels() {
        let bindings = [
            PcuBinding::value(
                Some("a"),
                0,
                0,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::ReadOnly,
                PcuValueType::u32(),
            ),
            PcuBinding::value(
                Some("b"),
                0,
                1,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::ReadOnly,
                PcuValueType::u32(),
            ),
            PcuBinding::value(
                Some("out"),
                0,
                2,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::WriteOnly,
                PcuValueType::u32(),
            ),
        ];
        let direct = [
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result: PcuDispatchValueId(1),
                binding: PcuBindingRef::new(0, 0),
                index: PcuDispatchIndex::InvocationId,
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result: PcuDispatchValueId(2),
                binding: PcuBindingRef::new(0, 1),
                index: PcuDispatchIndex::InvocationId,
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::Alu {
                value_type: PcuValueType::u32(),
                result: PcuDispatchValueId(3),
                op: PcuDispatchAluOp::Add,
                lhs: PcuDispatchValueId(1),
                rhs: PcuDispatchValueId(2),
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::Alu {
                value_type: PcuValueType::u32(),
                result: PcuDispatchValueId(4),
                op: PcuDispatchAluOp::Mul,
                lhs: PcuDispatchValueId(3),
                rhs: PcuDispatchValueId(2),
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                binding: PcuBindingRef::new(0, 2),
                index: PcuDispatchIndex::InvocationId,
                value: PcuDispatchValueId(4),
            }),
            PcuDispatchOp::Control(PcuDispatchControlOp::Return),
        ];
        let source = lower_dispatch_to_hip_rtc_source(&kernel(&direct, &bindings)).unwrap();
        assert!(source.contains("unsigned int v3 = v1 + v2;"));
        assert!(source.contains("unsigned int v4 = v3 * v2;"));

        let body = [
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result: PcuDispatchValueId(1),
                binding: PcuBindingRef::new(0, 0),
                index: PcuDispatchIndex::GridStrideId,
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result: PcuDispatchValueId(2),
                binding: PcuBindingRef::new(0, 1),
                index: PcuDispatchIndex::GridStrideId,
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::Alu {
                value_type: PcuValueType::u32(),
                result: PcuDispatchValueId(3),
                op: PcuDispatchAluOp::Mul,
                lhs: PcuDispatchValueId(1),
                rhs: PcuDispatchValueId(2),
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                binding: PcuBindingRef::new(0, 2),
                index: PcuDispatchIndex::GridStrideId,
                value: PcuDispatchValueId(3),
            }),
        ];
        let loop_ops = [
            PcuDispatchOp::GridStrideLoop {
                extent: 256,
                body: &body,
            },
            PcuDispatchOp::Control(PcuDispatchControlOp::Return),
        ];
        let loop_source = lower_dispatch_to_hip_rtc_source(&kernel(&loop_ops, &bindings)).unwrap();
        assert!(loop_source.contains("unsigned int v3 = v1 * v2;"));

        let division = [
            direct[0],
            direct[1],
            PcuDispatchOp::Data(PcuDispatchDataOp::Alu {
                value_type: PcuValueType::u32(),
                result: PcuDispatchValueId(3),
                op: PcuDispatchAluOp::Div,
                lhs: PcuDispatchValueId(1),
                rhs: PcuDispatchValueId(2),
            }),
            direct[3],
            direct[4],
        ];
        assert!(lower_dispatch_to_hip_rtc_source(&kernel(&division, &bindings)).is_err());
    }

    #[test]
    fn lowers_u16_maps_with_explicit_modulo_results_direct_and_grid_stride() {
        let u16_type = PcuValueType::Scalar(fusion_pcu::PcuScalarType::U16);
        let bindings = [
            PcuBinding::value(
                Some("a"),
                0,
                0,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::ReadOnly,
                u16_type,
            ),
            PcuBinding::value(
                Some("b"),
                0,
                1,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::ReadOnly,
                u16_type,
            ),
            PcuBinding::value(
                Some("out"),
                0,
                2,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::WriteOnly,
                u16_type,
            ),
        ];
        let direct = [
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result: PcuDispatchValueId(1),
                binding: PcuBindingRef::new(0, 0),
                index: PcuDispatchIndex::InvocationId,
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result: PcuDispatchValueId(2),
                binding: PcuBindingRef::new(0, 1),
                index: PcuDispatchIndex::InvocationId,
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::Alu {
                value_type: u16_type,
                result: PcuDispatchValueId(3),
                op: PcuDispatchAluOp::Add,
                lhs: PcuDispatchValueId(1),
                rhs: PcuDispatchValueId(2),
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                binding: PcuBindingRef::new(0, 2),
                index: PcuDispatchIndex::InvocationId,
                value: PcuDispatchValueId(3),
            }),
            PcuDispatchOp::Control(PcuDispatchControlOp::Return),
        ];
        let direct_source = lower_dispatch_to_hip_rtc_source(&kernel(&direct, &bindings)).unwrap();
        assert!(direct_source.contains("const unsigned short* binding_0_0"));
        assert!(direct_source.contains("unsigned short* binding_0_2"));
        assert!(direct_source.contains("unsigned short v1 = binding_0_0[fusion_gid];"));
        assert!(direct_source.contains("unsigned short v3 = static_cast<unsigned short>((static_cast<unsigned int>(v1) + static_cast<unsigned int>(v2)) & 0xffffu);"));

        let body = [
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result: PcuDispatchValueId(1),
                binding: PcuBindingRef::new(0, 0),
                index: PcuDispatchIndex::GridStrideId,
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result: PcuDispatchValueId(2),
                binding: PcuBindingRef::new(0, 1),
                index: PcuDispatchIndex::GridStrideId,
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::Alu {
                value_type: u16_type,
                result: PcuDispatchValueId(3),
                op: PcuDispatchAluOp::Sub,
                lhs: PcuDispatchValueId(1),
                rhs: PcuDispatchValueId(2),
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::Alu {
                value_type: u16_type,
                result: PcuDispatchValueId(4),
                op: PcuDispatchAluOp::Mul,
                lhs: PcuDispatchValueId(3),
                rhs: PcuDispatchValueId(2),
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                binding: PcuBindingRef::new(0, 2),
                index: PcuDispatchIndex::GridStrideId,
                value: PcuDispatchValueId(4),
            }),
        ];
        let loop_ops = [
            PcuDispatchOp::GridStrideLoop {
                extent: 256,
                body: &body,
            },
            PcuDispatchOp::Control(PcuDispatchControlOp::Return),
        ];
        let loop_source = lower_dispatch_to_hip_rtc_source(&kernel(&loop_ops, &bindings)).unwrap();
        assert!(loop_source.contains("unsigned short v3 = static_cast<unsigned short>((static_cast<unsigned int>(v1) - static_cast<unsigned int>(v2)) & 0xffffu);"));
        assert!(loop_source.contains("unsigned short v4 = static_cast<unsigned short>((static_cast<unsigned int>(v3) * static_cast<unsigned int>(v2)) & 0xffffu);"));
    }

    #[test]
    fn lowers_u8_maps_with_explicit_modulo_results_direct_and_grid_stride() {
        let u8_type = PcuValueType::Scalar(fusion_pcu::PcuScalarType::U8);
        let bindings = [
            PcuBinding::value(
                Some("a"),
                0,
                0,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::ReadOnly,
                u8_type,
            ),
            PcuBinding::value(
                Some("b"),
                0,
                1,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::ReadOnly,
                u8_type,
            ),
            PcuBinding::value(
                Some("out"),
                0,
                2,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::WriteOnly,
                u8_type,
            ),
        ];
        let direct = [
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result: PcuDispatchValueId(1),
                binding: PcuBindingRef::new(0, 0),
                index: PcuDispatchIndex::InvocationId,
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result: PcuDispatchValueId(2),
                binding: PcuBindingRef::new(0, 1),
                index: PcuDispatchIndex::InvocationId,
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::Alu {
                value_type: u8_type,
                result: PcuDispatchValueId(3),
                op: PcuDispatchAluOp::Add,
                lhs: PcuDispatchValueId(1),
                rhs: PcuDispatchValueId(2),
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                binding: PcuBindingRef::new(0, 2),
                index: PcuDispatchIndex::InvocationId,
                value: PcuDispatchValueId(3),
            }),
            PcuDispatchOp::Control(PcuDispatchControlOp::Return),
        ];
        let direct_source = lower_dispatch_to_hip_rtc_source(&kernel(&direct, &bindings)).unwrap();
        assert!(direct_source.contains("const unsigned char* binding_0_0"));
        assert!(direct_source.contains("unsigned char* binding_0_2"));
        assert!(direct_source.contains("unsigned char v1 = binding_0_0[fusion_gid];"));
        assert!(direct_source.contains("unsigned char v3 = static_cast<unsigned char>((static_cast<unsigned int>(v1) + static_cast<unsigned int>(v2)) & 0xffu);"));

        let body = [
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result: PcuDispatchValueId(1),
                binding: PcuBindingRef::new(0, 0),
                index: PcuDispatchIndex::GridStrideId,
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result: PcuDispatchValueId(2),
                binding: PcuBindingRef::new(0, 1),
                index: PcuDispatchIndex::GridStrideId,
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::Alu {
                value_type: u8_type,
                result: PcuDispatchValueId(3),
                op: PcuDispatchAluOp::Sub,
                lhs: PcuDispatchValueId(1),
                rhs: PcuDispatchValueId(2),
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::Alu {
                value_type: u8_type,
                result: PcuDispatchValueId(4),
                op: PcuDispatchAluOp::Mul,
                lhs: PcuDispatchValueId(3),
                rhs: PcuDispatchValueId(2),
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                binding: PcuBindingRef::new(0, 2),
                index: PcuDispatchIndex::GridStrideId,
                value: PcuDispatchValueId(4),
            }),
        ];
        let loop_ops = [
            PcuDispatchOp::GridStrideLoop {
                extent: 256,
                body: &body,
            },
            PcuDispatchOp::Control(PcuDispatchControlOp::Return),
        ];
        let loop_source = lower_dispatch_to_hip_rtc_source(&kernel(&loop_ops, &bindings)).unwrap();
        assert!(loop_source.contains("unsigned char v3 = static_cast<unsigned char>((static_cast<unsigned int>(v1) - static_cast<unsigned int>(v2)) & 0xffu);"));
        assert!(loop_source.contains("unsigned char v4 = static_cast<unsigned char>((static_cast<unsigned int>(v3) * static_cast<unsigned int>(v2)) & 0xffu);"));
    }

    #[test]
    fn lowers_i16_maps_as_unsigned_bits_with_explicit_modulo_results() {
        let i16_type = PcuValueType::Scalar(fusion_pcu::PcuScalarType::I16);
        let bindings = [
            PcuBinding::value(
                Some("a"),
                0,
                0,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::ReadOnly,
                i16_type,
            ),
            PcuBinding::value(
                Some("b"),
                0,
                1,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::ReadOnly,
                i16_type,
            ),
            PcuBinding::value(
                Some("out"),
                0,
                2,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::WriteOnly,
                i16_type,
            ),
        ];
        let direct = [
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result: PcuDispatchValueId(1),
                binding: PcuBindingRef::new(0, 0),
                index: PcuDispatchIndex::InvocationId,
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result: PcuDispatchValueId(2),
                binding: PcuBindingRef::new(0, 1),
                index: PcuDispatchIndex::InvocationId,
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::Alu {
                value_type: i16_type,
                result: PcuDispatchValueId(3),
                op: PcuDispatchAluOp::Add,
                lhs: PcuDispatchValueId(1),
                rhs: PcuDispatchValueId(2),
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                binding: PcuBindingRef::new(0, 2),
                index: PcuDispatchIndex::InvocationId,
                value: PcuDispatchValueId(3),
            }),
            PcuDispatchOp::Control(PcuDispatchControlOp::Return),
        ];
        let direct_source = lower_dispatch_to_hip_rtc_source(&kernel(&direct, &bindings)).unwrap();
        assert!(direct_source.contains("const unsigned short* binding_0_0"));
        assert!(direct_source.contains("unsigned short* binding_0_2"));
        assert!(direct_source.contains("unsigned short v1 = binding_0_0[fusion_gid];"));
        assert!(direct_source.contains("unsigned short v3 = static_cast<unsigned short>((static_cast<unsigned int>(v1) + static_cast<unsigned int>(v2)) & 0xffffu);"));

        let body = [
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result: PcuDispatchValueId(1),
                binding: PcuBindingRef::new(0, 0),
                index: PcuDispatchIndex::GridStrideId,
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result: PcuDispatchValueId(2),
                binding: PcuBindingRef::new(0, 1),
                index: PcuDispatchIndex::GridStrideId,
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::Alu {
                value_type: i16_type,
                result: PcuDispatchValueId(3),
                op: PcuDispatchAluOp::Sub,
                lhs: PcuDispatchValueId(1),
                rhs: PcuDispatchValueId(2),
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::Alu {
                value_type: i16_type,
                result: PcuDispatchValueId(4),
                op: PcuDispatchAluOp::Mul,
                lhs: PcuDispatchValueId(3),
                rhs: PcuDispatchValueId(2),
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                binding: PcuBindingRef::new(0, 2),
                index: PcuDispatchIndex::GridStrideId,
                value: PcuDispatchValueId(4),
            }),
        ];
        let loop_ops = [
            PcuDispatchOp::GridStrideLoop {
                extent: 256,
                body: &body,
            },
            PcuDispatchOp::Control(PcuDispatchControlOp::Return),
        ];
        let loop_source = lower_dispatch_to_hip_rtc_source(&kernel(&loop_ops, &bindings)).unwrap();
        assert!(loop_source.contains("unsigned short v3 = static_cast<unsigned short>((static_cast<unsigned int>(v1) - static_cast<unsigned int>(v2)) & 0xffffu);"));
        assert!(loop_source.contains("unsigned short v4 = static_cast<unsigned short>((static_cast<unsigned int>(v3) * static_cast<unsigned int>(v2)) & 0xffffu);"));
    }

    #[test]
    fn lowers_checked_i8_div_rem_with_signed_loads_and_overflow_guard() {
        let value_type = PcuValueType::Scalar(fusion_pcu::PcuScalarType::I8);
        let bindings = [
            PcuBinding::value(
                None,
                0,
                0,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::ReadOnly,
                value_type,
            ),
            PcuBinding::value(
                None,
                0,
                1,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::ReadOnly,
                value_type,
            ),
            PcuBinding::value(
                None,
                0,
                2,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::WriteOnly,
                value_type,
            ),
            PcuBinding::value(
                None,
                0,
                3,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::WriteOnly,
                value_type,
            ),
        ];
        let ops = [
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result: PcuDispatchValueId(1),
                binding: PcuBindingRef::new(0, 0),
                index: PcuDispatchIndex::InvocationId,
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result: PcuDispatchValueId(2),
                binding: PcuBindingRef::new(0, 1),
                index: PcuDispatchIndex::InvocationId,
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::CheckedDivRem {
                value_type,
                flags: PcuIntegerDivFlags::CHECKED,
                quotient: PcuDispatchValueId(3),
                remainder: PcuDispatchValueId(4),
                lhs: PcuDispatchValueId(1),
                rhs: PcuDispatchValueId(2),
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                binding: PcuBindingRef::new(0, 2),
                index: PcuDispatchIndex::InvocationId,
                value: PcuDispatchValueId(3),
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                binding: PcuBindingRef::new(0, 3),
                index: PcuDispatchIndex::InvocationId,
                value: PcuDispatchValueId(4),
            }),
            PcuDispatchOp::Control(PcuDispatchControlOp::Return),
        ];
        let source = lower_dispatch_to_hip_source(&kernel(&ops, &bindings)).unwrap();
        assert!(source.contains("const signed char* binding_0_0, const signed char* binding_0_1, signed char* binding_0_2, signed char* binding_0_3"));
        assert!(source.contains("signed char v1 = binding_0_0[fusion_gid];"));
        assert!(source.contains("v1 == (-127 - 1) && v2 == -1"));
        assert!(source.contains("static_cast<int>(v1) / static_cast<int>(v2)"));
    }

    #[test]
    #[allow(clippy::too_many_lines)]
    fn lowers_checked_i16_div_rem_with_fault_guards_direct_and_grid_stride() {
        let value_type = PcuValueType::Scalar(fusion_pcu::PcuScalarType::I16);
        let bindings = [
            PcuBinding::value(
                None,
                0,
                0,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::ReadOnly,
                value_type,
            ),
            PcuBinding::value(
                None,
                0,
                1,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::ReadOnly,
                value_type,
            ),
            PcuBinding::value(
                None,
                0,
                2,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::WriteOnly,
                value_type,
            ),
            PcuBinding::value(
                None,
                0,
                3,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::WriteOnly,
                value_type,
            ),
        ];
        let checked = PcuDispatchOp::Data(PcuDispatchDataOp::CheckedDivRem {
            value_type,
            flags: PcuIntegerDivFlags::CHECKED,
            quotient: PcuDispatchValueId(3),
            remainder: PcuDispatchValueId(4),
            lhs: PcuDispatchValueId(1),
            rhs: PcuDispatchValueId(2),
        });
        let direct = [
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result: PcuDispatchValueId(1),
                binding: PcuBindingRef::new(0, 0),
                index: PcuDispatchIndex::InvocationId,
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result: PcuDispatchValueId(2),
                binding: PcuBindingRef::new(0, 1),
                index: PcuDispatchIndex::InvocationId,
            }),
            checked,
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                binding: PcuBindingRef::new(0, 2),
                index: PcuDispatchIndex::InvocationId,
                value: PcuDispatchValueId(3),
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                binding: PcuBindingRef::new(0, 3),
                index: PcuDispatchIndex::InvocationId,
                value: PcuDispatchValueId(4),
            }),
            PcuDispatchOp::Control(PcuDispatchControlOp::Return),
        ];
        let source = lower_dispatch_to_hip_source(&kernel(&direct, &bindings)).unwrap();
        assert!(source.contains("const short* binding_0_0, const short* binding_0_1, short* binding_0_2, short* binding_0_3"));
        assert!(source.contains("short v1 = binding_0_0[fusion_gid];"));
        assert!(source.contains("if (v2 == 0) { atomicMin(fusion_fault_word, (static_cast<unsigned long long>(fusion_gid) << 2u) | 1ull); }"));
        assert!(source.contains("v1 == (-32767 - 1) && v2 == -1"));
        assert!(source.contains("static_cast<int>(v1) / static_cast<int>(v2)"));
        assert!(
            source.find("v1 == (-32767 - 1)").unwrap()
                < source.find("static_cast<int>(v1) /").unwrap()
        );

        let body = [
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result: PcuDispatchValueId(1),
                binding: PcuBindingRef::new(0, 0),
                index: PcuDispatchIndex::GridStrideId,
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result: PcuDispatchValueId(2),
                binding: PcuBindingRef::new(0, 1),
                index: PcuDispatchIndex::GridStrideId,
            }),
            checked,
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                binding: PcuBindingRef::new(0, 2),
                index: PcuDispatchIndex::GridStrideId,
                value: PcuDispatchValueId(3),
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                binding: PcuBindingRef::new(0, 3),
                index: PcuDispatchIndex::GridStrideId,
                value: PcuDispatchValueId(4),
            }),
        ];
        let grid = [
            PcuDispatchOp::GridStrideLoop {
                extent: 257,
                body: &body,
            },
            PcuDispatchOp::Control(PcuDispatchControlOp::Return),
        ];
        let source = lower_dispatch_to_hip_source(&kernel(&grid, &bindings)).unwrap();
        assert!(source.contains("static_cast<unsigned long long>(fusion_idx) << 2u) | 1ull"));
        assert!(source.contains("static_cast<unsigned long long>(fusion_idx) << 2u) | 2ull"));
        assert!(!source.contains("static_cast<unsigned long long>(fusion_gid) << 2u"));
    }

    #[test]
    fn lowers_i8_maps_as_unsigned_bits_with_explicit_modulo_results() {
        let i8_type = PcuValueType::Scalar(fusion_pcu::PcuScalarType::I8);
        let bindings = [
            PcuBinding::value(
                Some("a"),
                0,
                0,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::ReadOnly,
                i8_type,
            ),
            PcuBinding::value(
                Some("b"),
                0,
                1,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::ReadOnly,
                i8_type,
            ),
            PcuBinding::value(
                Some("out"),
                0,
                2,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::WriteOnly,
                i8_type,
            ),
        ];
        let direct = [
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result: PcuDispatchValueId(1),
                binding: PcuBindingRef::new(0, 0),
                index: PcuDispatchIndex::InvocationId,
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result: PcuDispatchValueId(2),
                binding: PcuBindingRef::new(0, 1),
                index: PcuDispatchIndex::InvocationId,
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::Alu {
                value_type: i8_type,
                result: PcuDispatchValueId(3),
                op: PcuDispatchAluOp::Add,
                lhs: PcuDispatchValueId(1),
                rhs: PcuDispatchValueId(2),
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                binding: PcuBindingRef::new(0, 2),
                index: PcuDispatchIndex::InvocationId,
                value: PcuDispatchValueId(3),
            }),
            PcuDispatchOp::Control(PcuDispatchControlOp::Return),
        ];
        let direct_source = lower_dispatch_to_hip_rtc_source(&kernel(&direct, &bindings)).unwrap();
        assert!(direct_source.contains("const unsigned char* binding_0_0"));
        assert!(direct_source.contains("unsigned char* binding_0_2"));
        assert!(direct_source.contains("unsigned char v1 = binding_0_0[fusion_gid];"));
        assert!(direct_source.contains("unsigned char v3 = static_cast<unsigned char>((static_cast<unsigned int>(v1) + static_cast<unsigned int>(v2)) & 0xffu);"));

        let body = [
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result: PcuDispatchValueId(1),
                binding: PcuBindingRef::new(0, 0),
                index: PcuDispatchIndex::GridStrideId,
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result: PcuDispatchValueId(2),
                binding: PcuBindingRef::new(0, 1),
                index: PcuDispatchIndex::GridStrideId,
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::Alu {
                value_type: i8_type,
                result: PcuDispatchValueId(3),
                op: PcuDispatchAluOp::Sub,
                lhs: PcuDispatchValueId(1),
                rhs: PcuDispatchValueId(2),
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::Alu {
                value_type: i8_type,
                result: PcuDispatchValueId(4),
                op: PcuDispatchAluOp::Mul,
                lhs: PcuDispatchValueId(3),
                rhs: PcuDispatchValueId(2),
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                binding: PcuBindingRef::new(0, 2),
                index: PcuDispatchIndex::GridStrideId,
                value: PcuDispatchValueId(4),
            }),
        ];
        let loop_ops = [
            PcuDispatchOp::GridStrideLoop {
                extent: 256,
                body: &body,
            },
            PcuDispatchOp::Control(PcuDispatchControlOp::Return),
        ];
        let loop_source = lower_dispatch_to_hip_rtc_source(&kernel(&loop_ops, &bindings)).unwrap();
        assert!(loop_source.contains("unsigned char v3 = static_cast<unsigned char>((static_cast<unsigned int>(v1) - static_cast<unsigned int>(v2)) & 0xffu);"));
        assert!(loop_source.contains("unsigned char v4 = static_cast<unsigned char>((static_cast<unsigned int>(v3) * static_cast<unsigned int>(v2)) & 0xffu);"));
    }

    #[test]
    fn lowers_f32_load_alu_store_kernel() {
        let bindings = [
            PcuBinding::value(
                Some("input"),
                0,
                0,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::ReadOnly,
                PcuValueType::f32(),
            ),
            PcuBinding::value(
                Some("output"),
                0,
                1,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::WriteOnly,
                PcuValueType::f32(),
            ),
        ];
        let ops = [
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result: PcuDispatchValueId(4),
                binding: PcuBindingRef::new(0, 0),
                index: PcuDispatchIndex::InvocationId,
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::Constant {
                result: PcuDispatchValueId(7),
                value: PcuParameterValue::F32(1.0_f32.to_bits()),
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::Alu {
                value_type: fusion_pcu::PcuValueType::f32(),
                result: PcuDispatchValueId(9),
                op: PcuDispatchAluOp::Add,
                lhs: PcuDispatchValueId(4),
                rhs: PcuDispatchValueId(7),
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                binding: PcuBindingRef::new(0, 1),
                index: PcuDispatchIndex::InvocationId,
                value: PcuDispatchValueId(9),
            }),
            PcuDispatchOp::Control(PcuDispatchControlOp::Return),
        ];

        let source = lower_dispatch_to_hip_source(&kernel(&ops, &bindings)).unwrap();
        let rtc_source = lower_dispatch_to_hip_rtc_source(&kernel(&ops, &bindings)).unwrap();
        assert_eq!(
            source.strip_prefix("#include <hip/hip_runtime.h>\n\n"),
            Some(rtc_source.as_str())
        );
        assert!(source.contains("const float* binding_0_0"));
        assert!(source.contains("float* binding_0_1"));
        assert!(source.contains("float v7 = __builtin_bit_cast(float, 0x3f800000u);"));
        assert!(source.contains("float v9 = v4 + v7;"));
        assert!(source.contains("binding_0_1[fusion_gid] = v9;"));
        if std::env::var_os("FUSION_ROCM_COMPILE_TEST").is_some() {
            let image = crate::compile_hip_source(&source, "gfx1030").unwrap();
            assert!(!image.is_empty());
        }
    }

    #[test]
    fn lowers_binding_element_zero_as_a_compact_broadcast_read() {
        let bindings = [
            PcuBinding::value(
                Some("scalar"),
                0,
                0,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::ReadOnly,
                PcuValueType::f32(),
            ),
            PcuBinding::value(
                Some("output"),
                0,
                1,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::WriteOnly,
                PcuValueType::f32(),
            ),
        ];
        let ops = [
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result: PcuDispatchValueId(1),
                binding: PcuBindingRef::new(0, 0),
                index: PcuDispatchIndex::BindingElementZero,
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                binding: PcuBindingRef::new(0, 1),
                index: PcuDispatchIndex::InvocationId,
                value: PcuDispatchValueId(1),
            }),
            PcuDispatchOp::Control(PcuDispatchControlOp::Return),
        ];

        let source = lower_dispatch_to_hip_source(&kernel(&ops, &bindings)).expect("HIP lower");

        assert!(source.contains("float v1 = binding_0_0[0];"));
        assert!(source.contains("binding_0_1[fusion_gid] = v1;"));
        if std::env::var_os("FUSION_ROCM_COMPILE_TEST").is_some() {
            let image = crate::compile_hip_source(&source, "gfx1030").expect("HIP compile");
            assert!(!image.is_empty());
        }
    }

    #[test]
    fn lowers_grid_stride_map_to_real_hip_loop() {
        let bindings = [
            PcuBinding::value(
                Some("input"),
                0,
                0,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::ReadOnly,
                PcuValueType::f32(),
            ),
            PcuBinding::value(
                Some("output"),
                0,
                1,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::WriteOnly,
                PcuValueType::f32(),
            ),
        ];
        let body = [
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result: PcuDispatchValueId(1),
                binding: PcuBindingRef::new(0, 0),
                index: PcuDispatchIndex::GridStrideId,
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::Constant {
                result: PcuDispatchValueId(2),
                value: PcuParameterValue::F32(1.0_f32.to_bits()),
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::Alu {
                value_type: fusion_pcu::PcuValueType::f32(),
                result: PcuDispatchValueId(3),
                op: PcuDispatchAluOp::Add,
                lhs: PcuDispatchValueId(1),
                rhs: PcuDispatchValueId(2),
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                binding: PcuBindingRef::new(0, 1),
                index: PcuDispatchIndex::GridStrideId,
                value: PcuDispatchValueId(3),
            }),
        ];
        let ops = [
            PcuDispatchOp::GridStrideLoop {
                extent: 2048,
                body: &body,
            },
            PcuDispatchOp::Control(PcuDispatchControlOp::Return),
        ];
        let mut loop_kernel = kernel(&ops, &bindings);
        loop_kernel.entry.logical_shape = [250, 1, 1];

        let source = lower_dispatch_to_hip_source(&loop_kernel).expect("grid-stride lower");
        assert!(source.contains("fusion_gid >= 250u"));
        assert!(source.contains(
            "for (unsigned int fusion_idx = fusion_gid; fusion_idx < 2048u; fusion_idx += 250u)"
        ));
        assert!(source.contains("binding_0_0[fusion_idx]"));
        assert!(source.contains("binding_0_1[fusion_idx] = v3;"));

        let mut near_limit_kernel = loop_kernel;
        let near_limit_ops = [
            PcuDispatchOp::GridStrideLoop {
                extent: u32::MAX - 100,
                body: &body,
            },
            PcuDispatchOp::Control(PcuDispatchControlOp::Return),
        ];
        near_limit_kernel.ops = &near_limit_ops;
        let near_limit_source =
            lower_dispatch_to_hip_source(&near_limit_kernel).expect("near-limit grid-stride lower");
        assert!(near_limit_source.contains("for (unsigned long long fusion_idx = fusion_gid; fusion_idx < 4294967195ull; fusion_idx += 250ull)"));
    }

    #[test]
    fn rejects_undefined_values_with_structured_error() {
        let bindings = [PcuBinding::value(
            Some("output"),
            0,
            0,
            PcuBindingStorageClass::Storage,
            PcuBindingAccess::WriteOnly,
            PcuValueType::f32(),
        )];
        let ops = [
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                binding: PcuBindingRef::new(0, 0),
                index: PcuDispatchIndex::InvocationId,
                value: PcuDispatchValueId(99),
            }),
            PcuDispatchOp::Control(PcuDispatchControlOp::Return),
        ];

        assert_eq!(
            lower_dispatch_to_hip_source(&kernel(&ops, &bindings)),
            Err(RocmLowerError::UndefinedValue(PcuDispatchValueId(99)))
        );
    }

    #[test]
    fn rejects_reserved_zero_value_id_from_common_map_validation() {
        let bindings = [PcuBinding::value(
            Some("output"),
            0,
            0,
            PcuBindingStorageClass::Storage,
            PcuBindingAccess::WriteOnly,
            PcuValueType::f32(),
        )];
        let ops = [
            PcuDispatchOp::Data(PcuDispatchDataOp::Constant {
                result: PcuDispatchValueId(0),
                value: PcuParameterValue::F32(1.0_f32.to_bits()),
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                binding: PcuBindingRef::new(0, 0),
                index: PcuDispatchIndex::InvocationId,
                value: PcuDispatchValueId(0),
            }),
            PcuDispatchOp::Control(PcuDispatchControlOp::Return),
        ];

        assert_eq!(
            lower_dispatch_to_hip_source(&kernel(&ops, &bindings)),
            Err(RocmLowerError::InvalidValue(PcuDispatchValueId(0)))
        );
    }

    #[test]
    fn rejects_ports_and_parameters() {
        let port = PcuPort::new(
            Some("port"),
            PcuPortDirection::Input,
            PcuValueType::f32(),
            PcuPortRate::Stream,
            PcuPortBlocking::Blocking,
            PcuPortReliability::Lossless,
            PcuPortBackpressure::Backpressured,
        );
        let mut candidate = kernel(&[], &[]);
        candidate.ports = std::slice::from_ref(&port);
        assert_eq!(
            lower_dispatch_to_hip_source(&candidate),
            Err(RocmLowerError::UnsupportedKernelInterface)
        );
    }

    #[test]
    fn rejects_declared_requirements_outside_f32_subset() {
        let mut candidate = kernel(&[], &[]);
        candidate.type_caps = PcuValueTypeCaps::FLOAT64;
        assert_eq!(
            lower_dispatch_to_hip_source(&candidate),
            Err(RocmLowerError::UnsupportedRequirements)
        );
    }

    #[test]
    fn guards_rounded_launch_size() {
        let bindings = [PcuBinding::value(
            Some("output"),
            0,
            0,
            PcuBindingStorageClass::Storage,
            PcuBindingAccess::WriteOnly,
            PcuValueType::f32(),
        )];
        let ops = [
            PcuDispatchOp::Data(PcuDispatchDataOp::Constant {
                result: PcuDispatchValueId(1),
                value: PcuParameterValue::F32(0),
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                binding: PcuBindingRef::new(0, 0),
                index: PcuDispatchIndex::InvocationId,
                value: PcuDispatchValueId(1),
            }),
            PcuDispatchOp::Control(PcuDispatchControlOp::Return),
        ];
        let mut candidate = kernel(&ops, &bindings);
        candidate.entry.logical_shape = [65, 1, 1];

        let source = lower_dispatch_to_hip_source(&candidate).unwrap();
        assert!(source.contains("if (fusion_gid >= 65u) return;"));
    }

    #[test]
    fn rejects_duplicate_binding_coordinates() {
        let bindings = [
            PcuBinding::value(
                Some("first"),
                0,
                0,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::ReadOnly,
                PcuValueType::f32(),
            ),
            PcuBinding::value(
                Some("second"),
                0,
                0,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::WriteOnly,
                PcuValueType::f32(),
            ),
        ];
        assert_eq!(
            lower_dispatch_to_hip_source(&kernel(&[], &bindings)),
            Err(RocmLowerError::InvalidBinding(PcuBindingRef::new(0, 0)))
        );
    }

    #[test]
    fn rejects_unsupported_min_with_structured_error() {
        let ops = [PcuDispatchOp::Data(PcuDispatchDataOp::Alu {
            value_type: fusion_pcu::PcuValueType::f32(),
            result: PcuDispatchValueId(3),
            op: PcuDispatchAluOp::Min,
            lhs: PcuDispatchValueId(1),
            rhs: PcuDispatchValueId(2),
        })];
        assert_eq!(
            lower_dispatch_to_hip_source(&kernel(&ops, &[])),
            Err(RocmLowerError::UnsupportedAlu(PcuDispatchAluOp::Min))
        );
    }

    #[test]
    #[allow(clippy::too_many_lines)]
    fn lowers_checked_u32_div_rem_with_first_fault_record_direct_and_grid_stride() {
        let bindings = [
            PcuBinding::value(
                Some("n"),
                0,
                0,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::ReadOnly,
                PcuValueType::u32(),
            ),
            PcuBinding::value(
                Some("d"),
                0,
                1,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::ReadOnly,
                PcuValueType::u32(),
            ),
            PcuBinding::value(
                Some("q"),
                0,
                2,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::WriteOnly,
                PcuValueType::u32(),
            ),
            PcuBinding::value(
                Some("r"),
                0,
                3,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::WriteOnly,
                PcuValueType::u32(),
            ),
        ];
        let direct = [
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result: PcuDispatchValueId(1),
                binding: PcuBindingRef::new(0, 0),
                index: PcuDispatchIndex::InvocationId,
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result: PcuDispatchValueId(2),
                binding: PcuBindingRef::new(0, 1),
                index: PcuDispatchIndex::InvocationId,
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::CheckedDivRem {
                value_type: PcuValueType::u32(),
                flags: PcuIntegerDivFlags::CHECKED,
                quotient: PcuDispatchValueId(3),
                remainder: PcuDispatchValueId(4),
                lhs: PcuDispatchValueId(1),
                rhs: PcuDispatchValueId(2),
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                binding: PcuBindingRef::new(0, 2),
                index: PcuDispatchIndex::InvocationId,
                value: PcuDispatchValueId(3),
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                binding: PcuBindingRef::new(0, 3),
                index: PcuDispatchIndex::InvocationId,
                value: PcuDispatchValueId(4),
            }),
            PcuDispatchOp::Control(PcuDispatchControlOp::Return),
        ];
        let direct_source = lower_dispatch_to_hip_source(&kernel(&direct, &bindings)).unwrap();
        assert!(direct_source.contains("unsigned long long* fusion_fault_word"));
        assert!(direct_source.contains("atomicMin(fusion_fault_word, (static_cast<unsigned long long>(fusion_gid) << 2u) | 1ull)"));
        assert!(direct_source.contains("if (v2 == 0u)"));
        assert!(direct_source.contains("v3 = v1 / v2; v4 = v1 % v2;"));

        let loop_body = [
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result: PcuDispatchValueId(1),
                binding: PcuBindingRef::new(0, 0),
                index: PcuDispatchIndex::GridStrideId,
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result: PcuDispatchValueId(2),
                binding: PcuBindingRef::new(0, 1),
                index: PcuDispatchIndex::GridStrideId,
            }),
            direct[2],
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                binding: PcuBindingRef::new(0, 2),
                index: PcuDispatchIndex::GridStrideId,
                value: PcuDispatchValueId(3),
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                binding: PcuBindingRef::new(0, 3),
                index: PcuDispatchIndex::GridStrideId,
                value: PcuDispatchValueId(4),
            }),
        ];
        let grid_ops = [
            PcuDispatchOp::GridStrideLoop {
                extent: 257,
                body: &loop_body,
            },
            PcuDispatchOp::Control(PcuDispatchControlOp::Return),
        ];
        let grid_source = lower_dispatch_to_hip_source(&kernel(&grid_ops, &bindings)).unwrap();
        assert!(grid_source.contains("atomicMin(fusion_fault_word, (static_cast<unsigned long long>(fusion_idx) << 2u) | 1ull)"));
        assert!(!grid_source.contains("static_cast<unsigned long long>(fusion_gid) << 2u"));
    }

    #[test]
    fn lowers_checked_u8_div_rem_direct_and_grid_stride_with_zero_guard() {
        let bindings = [
            PcuBinding::value(
                Some("n"),
                0,
                0,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::ReadOnly,
                PcuValueType::u8(),
            ),
            PcuBinding::value(
                Some("d"),
                0,
                1,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::ReadOnly,
                PcuValueType::u8(),
            ),
            PcuBinding::value(
                Some("q"),
                0,
                2,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::WriteOnly,
                PcuValueType::u8(),
            ),
            PcuBinding::value(
                Some("r"),
                0,
                3,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::WriteOnly,
                PcuValueType::u8(),
            ),
        ];
        let make_ops = |index| {
            [
                PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                    result: PcuDispatchValueId(1),
                    binding: PcuBindingRef::new(0, 0),
                    index,
                }),
                PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                    result: PcuDispatchValueId(2),
                    binding: PcuBindingRef::new(0, 1),
                    index,
                }),
                PcuDispatchOp::Data(PcuDispatchDataOp::CheckedDivRem {
                    value_type: PcuValueType::u8(),
                    flags: PcuIntegerDivFlags::CHECKED,
                    quotient: PcuDispatchValueId(3),
                    remainder: PcuDispatchValueId(4),
                    lhs: PcuDispatchValueId(1),
                    rhs: PcuDispatchValueId(2),
                }),
                PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                    binding: PcuBindingRef::new(0, 2),
                    index,
                    value: PcuDispatchValueId(3),
                }),
                PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                    binding: PcuBindingRef::new(0, 3),
                    index,
                    value: PcuDispatchValueId(4),
                }),
            ]
        };
        let direct_body = make_ops(PcuDispatchIndex::InvocationId);
        let direct_ops = [
            direct_body[0],
            direct_body[1],
            direct_body[2],
            direct_body[3],
            direct_body[4],
            PcuDispatchOp::Control(PcuDispatchControlOp::Return),
        ];
        let direct = lower_dispatch_to_hip_source(&kernel(&direct_ops, &bindings)).unwrap();
        assert!(direct.contains("unsigned long long* fusion_fault_word"));
        assert!(direct.contains("unsigned char v3 = 0u; unsigned char v4 = 0u;"));
        assert!(direct.contains("if (v2 == 0u) { atomicMin(fusion_fault_word, (static_cast<unsigned long long>(fusion_gid) << 2u) | 1ull); }"));
        assert!(direct.contains("static_cast<unsigned int>(v1) / static_cast<unsigned int>(v2)"));

        let loop_body = make_ops(PcuDispatchIndex::GridStrideId);
        let grid_ops = [
            PcuDispatchOp::GridStrideLoop {
                extent: 257,
                body: &loop_body,
            },
            PcuDispatchOp::Control(PcuDispatchControlOp::Return),
        ];
        let grid = lower_dispatch_to_hip_source(&kernel(&grid_ops, &bindings)).unwrap();
        assert!(grid.contains("static_cast<unsigned long long>(fusion_idx) << 2u"));
        assert!(grid.contains("if (v2 == 0u)"));
    }

    #[test]
    #[allow(clippy::too_many_lines)]
    fn lowers_checked_u16_div_rem_as_unsigned_short_direct_and_grid_stride() {
        let bindings = [
            PcuBinding::value(
                Some("n"),
                0,
                0,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::ReadOnly,
                PcuValueType::u16(),
            ),
            PcuBinding::value(
                Some("d"),
                0,
                1,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::ReadOnly,
                PcuValueType::u16(),
            ),
            PcuBinding::value(
                Some("q"),
                0,
                2,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::WriteOnly,
                PcuValueType::u16(),
            ),
            PcuBinding::value(
                Some("r"),
                0,
                3,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::WriteOnly,
                PcuValueType::u16(),
            ),
        ];
        let body = [
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result: PcuDispatchValueId(1),
                binding: PcuBindingRef::new(0, 0),
                index: PcuDispatchIndex::InvocationId,
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result: PcuDispatchValueId(2),
                binding: PcuBindingRef::new(0, 1),
                index: PcuDispatchIndex::InvocationId,
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::CheckedDivRem {
                value_type: PcuValueType::u16(),
                flags: PcuIntegerDivFlags::empty(),
                quotient: PcuDispatchValueId(3),
                remainder: PcuDispatchValueId(4),
                lhs: PcuDispatchValueId(1),
                rhs: PcuDispatchValueId(2),
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                binding: PcuBindingRef::new(0, 2),
                index: PcuDispatchIndex::InvocationId,
                value: PcuDispatchValueId(3),
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                binding: PcuBindingRef::new(0, 3),
                index: PcuDispatchIndex::InvocationId,
                value: PcuDispatchValueId(4),
            }),
        ];
        let direct_ops = [
            body[0],
            body[1],
            body[2],
            body[3],
            body[4],
            PcuDispatchOp::Control(PcuDispatchControlOp::Return),
        ];
        let direct_source = lower_dispatch_to_hip_source(&kernel(&direct_ops, &bindings)).unwrap();
        assert!(direct_source.contains("unsigned short"));
        assert!(direct_source.contains("unsigned short v3 = 0u; unsigned short v4 = 0u;"));
        assert!(direct_source.contains("if (v2 == 0u) { atomicMin(fusion_fault_word, (static_cast<unsigned long long>(fusion_gid) << 2u) | 1ull); }"));
        assert!(
            direct_source.contains("static_cast<unsigned int>(v1) / static_cast<unsigned int>(v2)")
        );
        assert!(
            direct_source.contains("static_cast<unsigned int>(v1) % static_cast<unsigned int>(v2)")
        );

        let loop_body = [
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result: PcuDispatchValueId(1),
                binding: PcuBindingRef::new(0, 0),
                index: PcuDispatchIndex::GridStrideId,
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result: PcuDispatchValueId(2),
                binding: PcuBindingRef::new(0, 1),
                index: PcuDispatchIndex::GridStrideId,
            }),
            body[2],
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                binding: PcuBindingRef::new(0, 2),
                index: PcuDispatchIndex::GridStrideId,
                value: PcuDispatchValueId(3),
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                binding: PcuBindingRef::new(0, 3),
                index: PcuDispatchIndex::GridStrideId,
                value: PcuDispatchValueId(4),
            }),
        ];
        let grid_ops = [
            PcuDispatchOp::GridStrideLoop {
                extent: 257,
                body: &loop_body,
            },
            PcuDispatchOp::Control(PcuDispatchControlOp::Return),
        ];
        let grid_source = lower_dispatch_to_hip_source(&kernel(&grid_ops, &bindings)).unwrap();
        assert!(grid_source.contains("unsigned short"));
        assert!(grid_source.contains("static_cast<unsigned long long>(fusion_idx) << 2u"));
        assert!(
            grid_source.contains("static_cast<unsigned int>(v1) / static_cast<unsigned int>(v2)")
        );
    }

    #[test]
    #[allow(clippy::too_many_lines)]
    fn lowers_checked_u64_div_rem_with_separate_fault_word_direct_and_grid_stride() {
        let bindings = [
            PcuBinding::value(
                Some("n"),
                0,
                0,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::ReadOnly,
                PcuValueType::u64(),
            ),
            PcuBinding::value(
                Some("d"),
                0,
                1,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::ReadOnly,
                PcuValueType::u64(),
            ),
            PcuBinding::value(
                Some("q"),
                0,
                2,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::WriteOnly,
                PcuValueType::u64(),
            ),
            PcuBinding::value(
                Some("r"),
                0,
                3,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::WriteOnly,
                PcuValueType::u64(),
            ),
        ];
        let direct = [
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result: PcuDispatchValueId(1),
                binding: PcuBindingRef::new(0, 0),
                index: PcuDispatchIndex::InvocationId,
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result: PcuDispatchValueId(2),
                binding: PcuBindingRef::new(0, 1),
                index: PcuDispatchIndex::InvocationId,
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::CheckedDivRem {
                value_type: PcuValueType::u64(),
                flags: PcuIntegerDivFlags::CHECKED,
                quotient: PcuDispatchValueId(3),
                remainder: PcuDispatchValueId(4),
                lhs: PcuDispatchValueId(1),
                rhs: PcuDispatchValueId(2),
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                binding: PcuBindingRef::new(0, 2),
                index: PcuDispatchIndex::InvocationId,
                value: PcuDispatchValueId(3),
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                binding: PcuBindingRef::new(0, 3),
                index: PcuDispatchIndex::InvocationId,
                value: PcuDispatchValueId(4),
            }),
            PcuDispatchOp::Control(PcuDispatchControlOp::Return),
        ];
        let direct_source = lower_dispatch_to_hip_source(&kernel(&direct, &bindings)).unwrap();
        assert!(direct_source.contains("const unsigned long long* binding_0_0, const unsigned long long* binding_0_1, unsigned long long* binding_0_2, unsigned long long* binding_0_3, unsigned long long* fusion_fault_word"));
        assert!(
            direct_source.contains("unsigned long long v3 = 0ull; unsigned long long v4 = 0ull;")
        );
        assert!(direct_source.contains("if (v2 == 0ull)"));
        assert!(direct_source.contains("v3 = v1 / v2; v4 = v1 % v2;"));
        assert!(direct_source.contains("atomicMin(fusion_fault_word, (static_cast<unsigned long long>(fusion_gid) << 2u) | 1ull)"));

        let body = [
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result: PcuDispatchValueId(1),
                binding: PcuBindingRef::new(0, 0),
                index: PcuDispatchIndex::GridStrideId,
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result: PcuDispatchValueId(2),
                binding: PcuBindingRef::new(0, 1),
                index: PcuDispatchIndex::GridStrideId,
            }),
            direct[2],
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                binding: PcuBindingRef::new(0, 2),
                index: PcuDispatchIndex::GridStrideId,
                value: PcuDispatchValueId(3),
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                binding: PcuBindingRef::new(0, 3),
                index: PcuDispatchIndex::GridStrideId,
                value: PcuDispatchValueId(4),
            }),
        ];
        let grid_ops = [
            PcuDispatchOp::GridStrideLoop {
                extent: 257,
                body: &body,
            },
            PcuDispatchOp::Control(PcuDispatchControlOp::Return),
        ];
        let grid_source = lower_dispatch_to_hip_source(&kernel(&grid_ops, &bindings)).unwrap();
        assert!(grid_source.contains("atomicMin(fusion_fault_word, (static_cast<unsigned long long>(fusion_idx) << 2u) | 1ull)"));
        assert!(!grid_source.contains("static_cast<unsigned long long>(fusion_gid) << 2u"));
    }

    #[test]
    #[allow(clippy::too_many_lines)]
    fn lowers_checked_i32_div_rem_with_both_fault_guards_direct_and_grid_stride() {
        let bindings = [
            PcuBinding::value(
                Some("n"),
                0,
                0,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::ReadOnly,
                PcuValueType::i32(),
            ),
            PcuBinding::value(
                Some("d"),
                0,
                1,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::ReadOnly,
                PcuValueType::i32(),
            ),
            PcuBinding::value(
                Some("q"),
                0,
                2,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::WriteOnly,
                PcuValueType::i32(),
            ),
            PcuBinding::value(
                Some("r"),
                0,
                3,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::WriteOnly,
                PcuValueType::i32(),
            ),
        ];
        let direct = [
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result: PcuDispatchValueId(1),
                binding: PcuBindingRef::new(0, 0),
                index: PcuDispatchIndex::InvocationId,
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result: PcuDispatchValueId(2),
                binding: PcuBindingRef::new(0, 1),
                index: PcuDispatchIndex::InvocationId,
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::CheckedDivRem {
                value_type: PcuValueType::i32(),
                flags: PcuIntegerDivFlags::CHECKED,
                quotient: PcuDispatchValueId(3),
                remainder: PcuDispatchValueId(4),
                lhs: PcuDispatchValueId(1),
                rhs: PcuDispatchValueId(2),
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                binding: PcuBindingRef::new(0, 2),
                index: PcuDispatchIndex::InvocationId,
                value: PcuDispatchValueId(3),
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                binding: PcuBindingRef::new(0, 3),
                index: PcuDispatchIndex::InvocationId,
                value: PcuDispatchValueId(4),
            }),
            PcuDispatchOp::Control(PcuDispatchControlOp::Return),
        ];
        let direct_source = lower_dispatch_to_hip_source(&kernel(&direct, &bindings)).unwrap();
        assert!(direct_source.contains(
            "const int* binding_0_0, const int* binding_0_1, int* binding_0_2, int* binding_0_3"
        ));
        assert!(direct_source.contains("int v1 = binding_0_0[fusion_gid];"));
        assert!(direct_source.contains("int v2 = binding_0_1[fusion_gid];"));
        assert!(direct_source.contains("if (v2 == 0) { atomicMin(fusion_fault_word, (static_cast<unsigned long long>(fusion_gid) << 2u) | 1ull); }"));
        assert!(direct_source.contains("v1 == (-2147483647 - 1) && v2 == -1"));
        assert!(
            direct_source.contains("static_cast<unsigned long long>(fusion_gid) << 2u) | 2ull")
        );
        assert!(direct_source.contains("v3 = v1 / v2; v4 = v1 % v2;"));
        assert!(
            direct_source.find("v1 == (-2147483647 - 1)").unwrap()
                < direct_source.find("v3 = v1 / v2").unwrap()
        );

        let body = [
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result: PcuDispatchValueId(1),
                binding: PcuBindingRef::new(0, 0),
                index: PcuDispatchIndex::GridStrideId,
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result: PcuDispatchValueId(2),
                binding: PcuBindingRef::new(0, 1),
                index: PcuDispatchIndex::GridStrideId,
            }),
            direct[2],
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                binding: PcuBindingRef::new(0, 2),
                index: PcuDispatchIndex::GridStrideId,
                value: PcuDispatchValueId(3),
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                binding: PcuBindingRef::new(0, 3),
                index: PcuDispatchIndex::GridStrideId,
                value: PcuDispatchValueId(4),
            }),
        ];
        let grid_ops = [
            PcuDispatchOp::GridStrideLoop {
                extent: 257,
                body: &body,
            },
            PcuDispatchOp::Control(PcuDispatchControlOp::Return),
        ];
        let grid_source = lower_dispatch_to_hip_source(&kernel(&grid_ops, &bindings)).unwrap();
        assert!(grid_source.contains("int v1 = binding_0_0[fusion_idx];"));
        assert!(grid_source.contains("int v2 = binding_0_1[fusion_idx];"));
        assert!(grid_source.contains("static_cast<unsigned long long>(fusion_idx) << 2u) | 1ull"));
        assert!(grid_source.contains("static_cast<unsigned long long>(fusion_idx) << 2u) | 2ull"));
        assert!(!grid_source.contains("static_cast<unsigned long long>(fusion_gid) << 2u"));
    }

    #[test]
    #[allow(clippy::too_many_lines)]
    fn lowers_checked_i64_div_rem_with_signed_loads_and_both_fault_guards() {
        let bindings = [
            PcuBinding::value(
                None,
                0,
                0,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::ReadOnly,
                PcuValueType::i64(),
            ),
            PcuBinding::value(
                None,
                0,
                1,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::ReadOnly,
                PcuValueType::i64(),
            ),
            PcuBinding::value(
                None,
                0,
                2,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::WriteOnly,
                PcuValueType::i64(),
            ),
            PcuBinding::value(
                None,
                0,
                3,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::WriteOnly,
                PcuValueType::i64(),
            ),
        ];
        let direct = [
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result: PcuDispatchValueId(1),
                binding: PcuBindingRef::new(0, 0),
                index: PcuDispatchIndex::InvocationId,
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result: PcuDispatchValueId(2),
                binding: PcuBindingRef::new(0, 1),
                index: PcuDispatchIndex::InvocationId,
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::CheckedDivRem {
                value_type: PcuValueType::i64(),
                flags: PcuIntegerDivFlags::CHECKED,
                quotient: PcuDispatchValueId(3),
                remainder: PcuDispatchValueId(4),
                lhs: PcuDispatchValueId(1),
                rhs: PcuDispatchValueId(2),
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                binding: PcuBindingRef::new(0, 2),
                index: PcuDispatchIndex::InvocationId,
                value: PcuDispatchValueId(3),
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                binding: PcuBindingRef::new(0, 3),
                index: PcuDispatchIndex::InvocationId,
                value: PcuDispatchValueId(4),
            }),
            PcuDispatchOp::Control(PcuDispatchControlOp::Return),
        ];
        let source = lower_dispatch_to_hip_source(&kernel(&direct, &bindings)).unwrap();
        assert!(source.contains("const long long* binding_0_0, const long long* binding_0_1, long long* binding_0_2, long long* binding_0_3"));
        assert!(source.contains("long long v1 = binding_0_0[fusion_gid];"));
        assert!(source.contains("long long v2 = binding_0_1[fusion_gid];"));
        assert!(source.contains("if (v2 == 0ll)"));
        assert!(source.contains("v1 == (-9223372036854775807ll - 1ll) && v2 == -1ll"));
        assert!(source.contains("v3 = v1 / v2; v4 = v1 % v2;"));
        assert!(
            source.find("v1 == (-9223372036854775807ll - 1ll)").unwrap()
                < source.find("v3 = v1 / v2").unwrap()
        );

        let body = [
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result: PcuDispatchValueId(1),
                binding: PcuBindingRef::new(0, 0),
                index: PcuDispatchIndex::GridStrideId,
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result: PcuDispatchValueId(2),
                binding: PcuBindingRef::new(0, 1),
                index: PcuDispatchIndex::GridStrideId,
            }),
            direct[2],
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                binding: PcuBindingRef::new(0, 2),
                index: PcuDispatchIndex::GridStrideId,
                value: PcuDispatchValueId(3),
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                binding: PcuBindingRef::new(0, 3),
                index: PcuDispatchIndex::GridStrideId,
                value: PcuDispatchValueId(4),
            }),
        ];
        let grid = [
            PcuDispatchOp::GridStrideLoop {
                extent: 257,
                body: &body,
            },
            PcuDispatchOp::Control(PcuDispatchControlOp::Return),
        ];
        let grid_source = lower_dispatch_to_hip_source(&kernel(&grid, &bindings)).unwrap();
        assert!(grid_source.contains("long long v1 = binding_0_0[fusion_idx];"));
        assert!(grid_source.contains("long long v2 = binding_0_1[fusion_idx];"));
        assert!(grid_source.contains("static_cast<unsigned long long>(fusion_idx) << 2u) | 1ull"));
        assert!(grid_source.contains("static_cast<unsigned long long>(fusion_idx) << 2u) | 2ull"));
    }

    #[test]
    fn lowers_f16_and_bf16_as_opaque_identity_payloads_only() {
        for scalar in [
            fusion_pcu::PcuScalarType::F16,
            fusion_pcu::PcuScalarType::BF16,
        ] {
            let value_type = PcuValueType::Scalar(scalar);
            let bindings = [
                PcuBinding::value(
                    Some("input"),
                    0,
                    0,
                    PcuBindingStorageClass::Storage,
                    PcuBindingAccess::ReadOnly,
                    value_type,
                ),
                PcuBinding::value(
                    Some("output"),
                    0,
                    1,
                    PcuBindingStorageClass::Storage,
                    PcuBindingAccess::WriteOnly,
                    value_type,
                ),
            ];
            let direct = [
                PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                    result: PcuDispatchValueId(1),
                    binding: PcuBindingRef::new(0, 0),
                    index: PcuDispatchIndex::InvocationId,
                }),
                PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                    binding: PcuBindingRef::new(0, 1),
                    index: PcuDispatchIndex::InvocationId,
                    value: PcuDispatchValueId(1),
                }),
                PcuDispatchOp::Control(PcuDispatchControlOp::Return),
            ];
            let source = lower_dispatch_to_hip_rtc_source(&kernel(&direct, &bindings)).unwrap();
            assert!(source.contains("const unsigned short* binding_0_0"));
            assert!(source.contains("unsigned short* binding_0_1"));
            assert!(source.contains("unsigned short v1 = binding_0_0[fusion_gid];"));

            let body = [
                PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                    result: PcuDispatchValueId(1),
                    binding: PcuBindingRef::new(0, 0),
                    index: PcuDispatchIndex::GridStrideId,
                }),
                PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                    binding: PcuBindingRef::new(0, 1),
                    index: PcuDispatchIndex::GridStrideId,
                    value: PcuDispatchValueId(1),
                }),
            ];
            let grid = [
                PcuDispatchOp::GridStrideLoop {
                    extent: 257,
                    body: &body,
                },
                PcuDispatchOp::Control(PcuDispatchControlOp::Return),
            ];
            let source = lower_dispatch_to_hip_rtc_source(&kernel(&grid, &bindings)).unwrap();
            assert!(source.contains("unsigned short v1 = binding_0_0[fusion_idx];"));

            let arithmetic = [
                direct[0],
                PcuDispatchOp::Data(PcuDispatchDataOp::Alu {
                    value_type,
                    result: PcuDispatchValueId(2),
                    op: PcuDispatchAluOp::Add,
                    lhs: PcuDispatchValueId(1),
                    rhs: PcuDispatchValueId(1),
                }),
                PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                    binding: PcuBindingRef::new(0, 1),
                    index: PcuDispatchIndex::InvocationId,
                    value: PcuDispatchValueId(2),
                }),
                direct[2],
            ];
            assert!(lower_dispatch_to_hip_rtc_source(&kernel(&arithmetic, &bindings)).is_err());
        }
    }

    type HalfConversionTestCase = (
        PcuDispatchConversion,
        fusion_pcu::PcuScalarType,
        fusion_pcu::PcuScalarType,
        &'static str,
        &'static str,
        &'static str,
    );

    fn half_conversion_test_cases() -> [HalfConversionTestCase; 4] {
        [
            (
                PcuDispatchConversion::F32ToF16Bits,
                fusion_pcu::PcuScalarType::F32,
                fusion_pcu::PcuScalarType::F16,
                "fusion_f32_to_f16_bits",
                "float",
                "unsigned short",
            ),
            (
                PcuDispatchConversion::F16BitsToF32,
                fusion_pcu::PcuScalarType::F16,
                fusion_pcu::PcuScalarType::F32,
                "fusion_f16_bits_to_f32",
                "unsigned short",
                "float",
            ),
            (
                PcuDispatchConversion::F32ToBf16Bits,
                fusion_pcu::PcuScalarType::F32,
                fusion_pcu::PcuScalarType::BF16,
                "fusion_f32_to_bf16_bits",
                "float",
                "unsigned short",
            ),
            (
                PcuDispatchConversion::Bf16BitsToF32,
                fusion_pcu::PcuScalarType::BF16,
                fusion_pcu::PcuScalarType::F32,
                "fusion_bf16_bits_to_f32",
                "unsigned short",
                "float",
            ),
        ]
    }

    fn half_conversion_test_bindings(
        input: fusion_pcu::PcuScalarType,
        output: fusion_pcu::PcuScalarType,
    ) -> [PcuBinding<'static>; 2] {
        [
            PcuBinding::value(
                Some("input"),
                0,
                0,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::ReadOnly,
                PcuValueType::Scalar(input),
            ),
            PcuBinding::value(
                Some("output"),
                0,
                1,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::WriteOnly,
                PcuValueType::Scalar(output),
            ),
        ]
    }

    #[test]
    fn lowers_each_half_conversion_directly_with_bit_level_helpers() {
        for (conversion, input_scalar, output_scalar, helper, input_type, output_type) in
            half_conversion_test_cases()
        {
            let bindings = half_conversion_test_bindings(input_scalar, output_scalar);
            let direct = [
                PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                    result: PcuDispatchValueId(1),
                    binding: PcuBindingRef::new(0, 0),
                    index: PcuDispatchIndex::InvocationId,
                }),
                PcuDispatchOp::Data(PcuDispatchDataOp::Convert {
                    result: PcuDispatchValueId(2),
                    value: PcuDispatchValueId(1),
                    conversion,
                }),
                PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                    binding: PcuBindingRef::new(0, 1),
                    index: PcuDispatchIndex::InvocationId,
                    value: PcuDispatchValueId(2),
                }),
                PcuDispatchOp::Control(PcuDispatchControlOp::Return),
            ];
            let direct_kernel = kernel(&direct, &bindings);
            let direct_source = lower_dispatch_to_hip_rtc_source(&direct_kernel).unwrap();
            assert!(direct_source.contains(&format!("{helper}(v1)")));
            assert!(direct_source.contains(&format!("const {input_type}* binding_0_0")));
            assert!(direct_source.contains(&format!("{output_type}* binding_0_1")));
            assert!(direct_source.contains("if (fusion_gid >= 64u) return;"));
        }
    }

    #[test]
    fn lowers_each_half_conversion_over_250_invocations_and_2048_grid_extent() {
        for (conversion, input_scalar, output_scalar, helper, _, _) in half_conversion_test_cases()
        {
            let bindings = half_conversion_test_bindings(input_scalar, output_scalar);
            let grid_body = [
                PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                    result: PcuDispatchValueId(1),
                    binding: PcuBindingRef::new(0, 0),
                    index: PcuDispatchIndex::GridStrideId,
                }),
                PcuDispatchOp::Data(PcuDispatchDataOp::Convert {
                    result: PcuDispatchValueId(2),
                    value: PcuDispatchValueId(1),
                    conversion,
                }),
                PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                    binding: PcuBindingRef::new(0, 1),
                    index: PcuDispatchIndex::GridStrideId,
                    value: PcuDispatchValueId(2),
                }),
            ];
            let grid = [
                PcuDispatchOp::GridStrideLoop {
                    extent: 2048,
                    body: &grid_body,
                },
                PcuDispatchOp::Control(PcuDispatchControlOp::Return),
            ];
            let mut grid_kernel = kernel(&grid, &bindings);
            grid_kernel.entry.logical_shape[0] = 250;
            let grid_source = lower_dispatch_to_hip_rtc_source(&grid_kernel).unwrap();
            assert!(grid_source.contains(&format!("{helper}(v1)")));
            assert!(grid_source.contains("const unsigned int fusion_invocations = 250u;"));
            assert!(grid_source.contains("fusion_idx < 2048ull; fusion_idx += fusion_invocations"));
            let near_limit = [
                PcuDispatchOp::GridStrideLoop {
                    extent: u32::MAX,
                    body: &grid_body,
                },
                PcuDispatchOp::Control(PcuDispatchControlOp::Return),
            ];
            grid_kernel.ops = &near_limit;
            let near_limit_source = lower_dispatch_to_hip_rtc_source(&grid_kernel).unwrap();
            assert!(near_limit_source.contains("fusion_idx < 4294967295ull"));
            assert!(near_limit_source.contains("unsigned long long fusion_idx"));
        }
    }

    #[test]
    fn rejects_half_conversion_type_lies_and_unimplemented_following_operations() {
        let bindings = half_conversion_test_bindings(
            fusion_pcu::PcuScalarType::F32,
            fusion_pcu::PcuScalarType::BF16,
        );
        let mismatch = [
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result: PcuDispatchValueId(1),
                binding: PcuBindingRef::new(0, 0),
                index: PcuDispatchIndex::InvocationId,
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::Convert {
                result: PcuDispatchValueId(2),
                value: PcuDispatchValueId(1),
                conversion: PcuDispatchConversion::F32ToF16Bits,
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                binding: PcuBindingRef::new(0, 1),
                index: PcuDispatchIndex::InvocationId,
                value: PcuDispatchValueId(2),
            }),
            PcuDispatchOp::Control(PcuDispatchControlOp::Return),
        ];
        assert!(lower_dispatch_to_hip_rtc_source(&kernel(&mismatch, &bindings)).is_err());

        let f16_bindings = half_conversion_test_bindings(
            fusion_pcu::PcuScalarType::F32,
            fusion_pcu::PcuScalarType::F16,
        );
        let unsupported = [
            mismatch[0],
            PcuDispatchOp::Data(PcuDispatchDataOp::Convert {
                result: PcuDispatchValueId(2),
                value: PcuDispatchValueId(1),
                conversion: PcuDispatchConversion::F32ToF16Bits,
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::Alu {
                value_type: PcuValueType::Scalar(fusion_pcu::PcuScalarType::F16),
                result: PcuDispatchValueId(3),
                op: PcuDispatchAluOp::Add,
                lhs: PcuDispatchValueId(2),
                rhs: PcuDispatchValueId(2),
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                binding: PcuBindingRef::new(0, 1),
                index: PcuDispatchIndex::InvocationId,
                value: PcuDispatchValueId(3),
            }),
            PcuDispatchOp::Control(PcuDispatchControlOp::Return),
        ];
        assert!(lower_dispatch_to_hip_rtc_source(&kernel(&unsupported, &f16_bindings)).is_err());
    }
}
