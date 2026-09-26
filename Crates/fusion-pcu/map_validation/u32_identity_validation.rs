//! Structural admission for the narrow scalar `u32` invocation-index identity copy profile.

use crate::{
    PcuBindingAccess,
    PcuBindingRef,
    PcuBindingStorageClass,
    PcuBindingType,
    PcuDispatchControlOp,
    PcuDispatchDataOp,
    PcuDispatchIndex,
    PcuDispatchKernelIr,
    PcuDispatchOp,
    PcuDispatchFeatureCaps,
    PcuDispatchValueId,
    PcuValueType,
    PcuValueTypeCaps,
};

/// Structural failure within the scalar `u32` invocation-index identity profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PcuU32IdentityValidationError {
    UnsupportedInterface,
    UnsupportedRequirements,
    InvalidBinding(PcuBindingRef),
    DuplicateBinding(PcuBindingRef),
    UnsupportedOperation(usize),
    InvalidIndex(usize),
    InvalidValue(PcuDispatchValueId),
    MissingLoad,
    MissingStore,
    MissingReturn,
}

/// Validates one `u32` storage copy per invocation, directly or through a grid-stride loop.
///
/// Binding coverage and byte extents remain the core ABI's responsibility.
///
/// # Errors
///
/// Returns the first structural violation of the identity-copy profile.
pub fn validate_u32_identity_kernel(
    kernel: &PcuDispatchKernelIr<'_>,
) -> Result<(), PcuU32IdentityValidationError> {
    if !kernel.ports.is_empty() || !kernel.parameters.is_empty() || kernel.bindings.len() != 2 {
        return Err(PcuU32IdentityValidationError::UnsupportedInterface);
    }
    let types = PcuValueTypeCaps::UINT32 | PcuValueTypeCaps::SCALAR_VALUES;
    let features =
        PcuDispatchFeatureCaps::MUTABLE_RESOURCES | PcuDispatchFeatureCaps::READ_ONLY_RESOURCES;
    if kernel.required_type_support().bits() & !types.bits() != 0
        || kernel.required_feature_support().bits() & !features.bits() != 0
    {
        return Err(PcuU32IdentityValidationError::UnsupportedRequirements);
    }
    for (index, binding) in kernel.bindings.iter().enumerate() {
        let reference = binding.reference();
        if kernel.bindings[..index]
            .iter()
            .any(|previous| previous.reference() == reference)
        {
            return Err(PcuU32IdentityValidationError::DuplicateBinding(reference));
        }
        if binding.storage != PcuBindingStorageClass::Storage
            || binding.builtin.is_some()
            || binding.binding_type != PcuBindingType::Value(PcuValueType::u32())
        {
            return Err(PcuU32IdentityValidationError::InvalidBinding(reference));
        }
    }

    if let [
        PcuDispatchOp::GridStrideLoop { extent, body },
        PcuDispatchOp::Control(PcuDispatchControlOp::Return),
    ] = kernel.ops
    {
        if *extent == 0 {
            return Err(PcuU32IdentityValidationError::UnsupportedOperation(0));
        }
        return validate_grid_stride_copy(kernel, body);
    }

    if kernel.ops.len() < 3 {
        return Err(match kernel.ops.len() {
            0 => PcuU32IdentityValidationError::MissingLoad,
            1 => PcuU32IdentityValidationError::MissingStore,
            _ => PcuU32IdentityValidationError::MissingReturn,
        });
    }
    if kernel.ops.len() > 3 {
        return Err(PcuU32IdentityValidationError::UnsupportedOperation(3));
    }
    let (loaded_value, input) = match kernel.ops[0] {
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
            result,
            binding,
            index: PcuDispatchIndex::InvocationId,
        }) => (result, binding),
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad { .. }) => {
            return Err(PcuU32IdentityValidationError::InvalidIndex(0));
        }
        _ => return Err(PcuU32IdentityValidationError::MissingLoad),
    };
    if loaded_value.0 == 0 {
        return Err(PcuU32IdentityValidationError::InvalidValue(loaded_value));
    }
    validate_binding(kernel, input, false)?;

    match kernel.ops[1] {
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
            binding,
            index: PcuDispatchIndex::InvocationId,
            value,
        }) => {
            if value != loaded_value {
                return Err(PcuU32IdentityValidationError::InvalidValue(value));
            }
            if binding == input {
                return Err(PcuU32IdentityValidationError::InvalidBinding(binding));
            }
            validate_binding(kernel, binding, true)?;
        }
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore { .. }) => {
            return Err(PcuU32IdentityValidationError::InvalidIndex(1));
        }
        _ => return Err(PcuU32IdentityValidationError::MissingStore),
    }
    if !matches!(
        kernel.ops[2],
        PcuDispatchOp::Control(PcuDispatchControlOp::Return)
    ) {
        return Err(PcuU32IdentityValidationError::MissingReturn);
    }
    Ok(())
}

fn validate_grid_stride_copy(
    kernel: &PcuDispatchKernelIr<'_>,
    body: &[PcuDispatchOp<'_>],
) -> Result<(), PcuU32IdentityValidationError> {
    let [
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
            result,
            binding: input,
            index: PcuDispatchIndex::GridStrideId,
        }),
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
            binding: output,
            index: PcuDispatchIndex::GridStrideId,
            value,
        }),
    ] = body
    else {
        return Err(PcuU32IdentityValidationError::UnsupportedOperation(0));
    };
    if result.0 == 0 || value != result {
        return Err(PcuU32IdentityValidationError::InvalidValue(*value));
    }
    if input == output {
        return Err(PcuU32IdentityValidationError::InvalidBinding(*output));
    }
    validate_binding(kernel, *input, false)?;
    validate_binding(kernel, *output, true)
}

fn validate_binding(
    kernel: &PcuDispatchKernelIr<'_>,
    target: PcuBindingRef,
    write: bool,
) -> Result<(), PcuU32IdentityValidationError> {
    let Some(binding) = kernel
        .bindings
        .iter()
        .find(|binding| binding.reference() == target)
    else {
        return Err(PcuU32IdentityValidationError::InvalidBinding(target));
    };
    if (write && binding.access == PcuBindingAccess::ReadOnly)
        || (!write && binding.access == PcuBindingAccess::WriteOnly)
    {
        Err(PcuU32IdentityValidationError::InvalidBinding(target))
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{
        PcuU32IdentityValidationError as Error,
        validate_u32_identity_kernel,
    };
    use crate::{
        PcuBinding,
        PcuBindingAccess,
        PcuBindingRef,
        PcuBindingStorageClass,
        PcuDispatchControlOp,
        PcuDispatchDataOp,
        PcuDispatchIndex,
        PcuDispatchKernelIr,
        PcuDispatchOp,
        PcuDispatchValueId,
        PcuValueType,
    };

    fn bindings() -> [PcuBinding<'static>; 2] {
        [
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
        ]
    }

    #[test]
    fn accepts_only_typed_u32_invocation_identity_copy() {
        let bindings = bindings();
        let ops = [
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
        let kernel = PcuDispatchKernelIr {
            id: crate::PcuKernelId(1),
            entry: crate::PcuDispatchEntryPoint {
                name: "identity",
                logical_shape: [1, 1, 1],
            },
            bindings: &bindings,
            ports: &[],
            parameters: &[],
            ops: &ops,
            type_caps: crate::PcuValueTypeCaps::empty(),
            feature_caps: crate::PcuDispatchFeatureCaps::empty(),
        };
        assert_eq!(validate_u32_identity_kernel(&kernel), Ok(()));

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
                extent: 16,
                body: &grid_body,
            },
            ops[2],
        ];
        assert_eq!(
            validate_u32_identity_kernel(&PcuDispatchKernelIr {
                ops: &grid_ops,
                ..kernel
            }),
            Ok(())
        );

        let wrong_index = [
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result: PcuDispatchValueId(1),
                binding: PcuBindingRef::new(0, 0),
                index: PcuDispatchIndex::BindingElementZero,
            }),
            ops[1],
            ops[2],
        ];
        assert_eq!(
            validate_u32_identity_kernel(&PcuDispatchKernelIr {
                ops: &wrong_index,
                ..kernel
            }),
            Err(Error::InvalidIndex(0))
        );

        let extra_alu = [
            ops[0],
            ops[1],
            PcuDispatchOp::Arithmetic(crate::PcuDispatchAluOp::Add),
        ];
        assert_eq!(
            validate_u32_identity_kernel(&PcuDispatchKernelIr {
                ops: &extra_alu,
                ..kernel
            }),
            Err(Error::MissingReturn)
        );
    }
}
