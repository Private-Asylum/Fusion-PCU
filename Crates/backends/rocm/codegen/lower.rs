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
    PcuDispatchControlOp,
    PcuDispatchDataOp,
    PcuDispatchFeatureCaps,
    PcuDispatchIndex,
    PcuDispatchKernelIr,
    PcuDispatchOp,
    PcuDispatchOpCaps,
    PcuDispatchValueId,
    PcuParameterValue,
    PcuF32MapValidationError,
    PcuValueType,
    PcuValueTypeCaps,
    validate_u64_checked_div_rem_kernel,
    validate_i32_checked_div_rem_kernel,
    validate_f32_map_kernel,
    validate_f64_map_kernel,
    validate_i32_map_kernel,
    validate_i64_map_kernel,
    validate_i16_map_kernel,
    validate_i8_map_kernel,
    validate_u32_identity_kernel,
    validate_u32_map_kernel,
    validate_u32_checked_div_rem_kernel,
    validate_u16_map_kernel,
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
    U16,
    U32,
    I32,
    U64,
}

impl HipScalarKind {
    fn for_kernel(kernel: &PcuDispatchKernelIr<'_>) -> Self {
        kernel
            .bindings
            .iter()
            .find_map(|binding| match binding.binding_type {
                PcuBindingType::Value(PcuValueType::Scalar(scalar)) => Some(match scalar {
                    fusion_pcu::PcuScalarType::F64 => Self::F64,
                    fusion_pcu::PcuScalarType::U8 | fusion_pcu::PcuScalarType::I8 => Self::U8,
                    fusion_pcu::PcuScalarType::U16 | fusion_pcu::PcuScalarType::I16 => Self::U16,
                    fusion_pcu::PcuScalarType::U64 | fusion_pcu::PcuScalarType::I64 => Self::U64,
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
            Self::U16 => "unsigned short",
            Self::U32 => "unsigned int",
            Self::I32 => "int",
            Self::U64 => "unsigned long long",
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
            (Self::U16, true) => "const unsigned short*",
            (Self::U16, false) => "unsigned short*",
            (Self::U32, true) => "const unsigned int*",
            (Self::U32, false) => "unsigned int*",
            (Self::I32, true) => "const int*",
            (Self::I32, false) => "int*",
            (Self::U64, true) => "const unsigned long long*",
            (Self::U64, false) => "unsigned long long*",
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
                "    float v{} = binding_{}_{}[0];",
                result.0, binding.set, binding.binding
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
            "        float v{} = binding_{}_{}[0];",
            result.0, binding.set, binding.binding
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
        _ => unreachable!("validated grid-stride body operation"),
    }
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
}
