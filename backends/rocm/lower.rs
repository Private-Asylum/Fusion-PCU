//! Narrow PCU dispatch IR to HIP C++ source lowering.
//!
//! The current subset is deliberately one-dimensional: f32 storage bindings, invocation-ID
//! indexed loads/stores, f32 constants, add/subtract/multiply/divide, and a terminal return.

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
    PcuValueType,
    PcuValueTypeCaps,
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
                "HIP lowering supports only scalar f32 values and mutable/read-only resources",
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
    validate_kernel(kernel)?;

    let mut source =
        String::from("#include <hip/hip_runtime.h>\n\nextern \"C\" __global__ void fusion_kernel(");
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
        let qualifier = if used_for_store || binding.access != PcuBindingAccess::ReadOnly {
            "float*"
        } else {
            "const float*"
        };
        write!(
            &mut source,
            "{qualifier} binding_{}_{}",
            binding.set, binding.binding
        )
        .map_err(|_| RocmLowerError::FormattingFailure)?;
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
                "    float v{} = binding_{}_{}[fusion_gid];",
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
            }) => {
                let operator = match op {
                    PcuDispatchAluOp::Add => "+",
                    PcuDispatchAluOp::Sub => "-",
                    PcuDispatchAluOp::Mul => "*",
                    PcuDispatchAluOp::Div => "/",
                    _ => unreachable!("validated ALU op"),
                };
                writeln!(
                    &mut source,
                    "    float v{} = v{} {operator} v{};",
                    result.0, lhs.0, rhs.0
                )
                .map_err(|_| RocmLowerError::FormattingFailure)?;
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
            _ => unreachable!("validated operation"),
        }
    }
    source.push_str("}\n");
    Ok(source)
}

#[allow(clippy::too_many_lines)] // One pass keeps the supported IR subset's validation rules together.
fn validate_kernel(kernel: &PcuDispatchKernelIr<'_>) -> Result<(), RocmLowerError> {
    if kernel.entry.logical_shape[0] == 0
        || kernel.entry.logical_shape[1] != 1
        || kernel.entry.logical_shape[2] != 1
    {
        return Err(RocmLowerError::InvalidKernelShape);
    }
    if !kernel.ports.is_empty() || !kernel.parameters.is_empty() {
        return Err(RocmLowerError::UnsupportedKernelInterface);
    }
    let allowed_types = PcuValueTypeCaps::FLOAT32 | PcuValueTypeCaps::SCALAR_VALUES;
    let allowed_features =
        PcuDispatchFeatureCaps::MUTABLE_RESOURCES | PcuDispatchFeatureCaps::READ_ONLY_RESOURCES;
    if kernel.required_type_support().bits() & !allowed_types.bits() != 0
        || kernel.required_feature_support().bits() & !allowed_features.bits() != 0
    {
        return Err(RocmLowerError::UnsupportedRequirements);
    }
    for (index, binding) in kernel.bindings.iter().copied().enumerate() {
        if binding.storage != PcuBindingStorageClass::Storage
            || binding.binding_type != PcuBindingType::Value(PcuValueType::f32())
            || binding.builtin.is_some()
        {
            return Err(RocmLowerError::InvalidBinding(PcuBindingRef::new(
                binding.set,
                binding.binding,
            )));
        }
        if kernel.bindings[..index]
            .iter()
            .any(|other| other.set == binding.set && other.binding == binding.binding)
        {
            return Err(RocmLowerError::InvalidBinding(PcuBindingRef::new(
                binding.set,
                binding.binding,
            )));
        }
    }

    let mut definitions = Vec::new();
    let mut has_store = false;
    let mut saw_return = false;
    for (index, op) in kernel.ops.iter().copied().enumerate() {
        if saw_return {
            return Err(RocmLowerError::OperationAfterReturn);
        }
        match op {
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result,
                binding,
                index,
            }) => {
                require_index(index)?;
                require_binding(kernel, binding, false)?;
                define(&mut definitions, result)?;
            }
            PcuDispatchOp::Data(PcuDispatchDataOp::Constant { result, value }) => {
                if !matches!(value, PcuParameterValue::F32(_)) {
                    return Err(RocmLowerError::UnsupportedConstant);
                }
                define(&mut definitions, result)?;
            }
            PcuDispatchOp::Data(PcuDispatchDataOp::Alu {
                result,
                op,
                lhs,
                rhs,
            }) => {
                if !matches!(
                    op,
                    PcuDispatchAluOp::Add
                        | PcuDispatchAluOp::Sub
                        | PcuDispatchAluOp::Mul
                        | PcuDispatchAluOp::Div
                ) {
                    return Err(RocmLowerError::UnsupportedAlu(op));
                }
                require_defined(&definitions, lhs)?;
                require_defined(&definitions, rhs)?;
                define(&mut definitions, result)?;
            }
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                binding,
                index,
                value,
            }) => {
                require_index(index)?;
                require_binding(kernel, binding, true)?;
                require_defined(&definitions, value)?;
                has_store = true;
            }
            PcuDispatchOp::Control(PcuDispatchControlOp::Return) => saw_return = true,
            _ => {
                return Err(RocmLowerError::UnsupportedOperation {
                    index,
                    support: op.support_flag(),
                });
            }
        }
    }
    if has_store && saw_return {
        Ok(())
    } else {
        Err(RocmLowerError::MissingStoreOrReturn)
    }
}

fn define(
    definitions: &mut Vec<PcuDispatchValueId>,
    result: PcuDispatchValueId,
) -> Result<(), RocmLowerError> {
    if definitions.contains(&result) {
        return Err(RocmLowerError::DuplicateValue(result));
    }
    definitions.push(result);
    Ok(())
}

fn require_defined(
    definitions: &[PcuDispatchValueId],
    value: PcuDispatchValueId,
) -> Result<(), RocmLowerError> {
    if definitions.contains(&value) {
        Ok(())
    } else {
        Err(RocmLowerError::UndefinedValue(value))
    }
}

fn require_index(index: PcuDispatchIndex) -> Result<(), RocmLowerError> {
    if index == PcuDispatchIndex::InvocationId {
        Ok(())
    } else {
        Err(RocmLowerError::UnsupportedIndex)
    }
}

fn require_binding(
    kernel: &PcuDispatchKernelIr<'_>,
    reference: PcuBindingRef,
    write: bool,
) -> Result<(), RocmLowerError> {
    let Some(binding) = kernel
        .bindings
        .iter()
        .find(|binding| binding.set == reference.set && binding.binding == reference.binding)
    else {
        return Err(RocmLowerError::InvalidBinding(reference));
    };
    let allowed = if write {
        matches!(
            binding.access,
            PcuBindingAccess::WriteOnly | PcuBindingAccess::ReadWrite
        )
    } else {
        matches!(
            binding.access,
            PcuBindingAccess::ReadOnly | PcuBindingAccess::ReadWrite
        )
    };
    if allowed {
        Ok(())
    } else {
        Err(RocmLowerError::InvalidBindingAccess(reference))
    }
}

#[cfg(test)]
mod tests {
    use super::{
        RocmLowerError,
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
                result: PcuDispatchValueId(0),
                value: PcuParameterValue::F32(0),
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                binding: PcuBindingRef::new(0, 0),
                index: PcuDispatchIndex::InvocationId,
                value: PcuDispatchValueId(0),
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
    fn rejects_unsupported_alu_with_structured_error() {
        let ops = [PcuDispatchOp::Data(PcuDispatchDataOp::Alu {
            result: PcuDispatchValueId(3),
            op: PcuDispatchAluOp::Max,
            lhs: PcuDispatchValueId(1),
            rhs: PcuDispatchValueId(2),
        })];
        assert_eq!(
            lower_dispatch_to_hip_source(&kernel(&ops, &[])),
            Err(RocmLowerError::UnsupportedAlu(PcuDispatchAluOp::Max))
        );
    }
}
