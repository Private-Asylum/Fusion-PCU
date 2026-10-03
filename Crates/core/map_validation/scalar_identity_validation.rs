//! Structural admission for typed scalar identity transport.

#[rustfmt::skip]
use crate::{
    PcuBindingAccess,
    PcuBindingRef,
    PcuBindingStorageClass,
    PcuBindingType,
    PcuDispatchControlOp,
    PcuDispatchDataOp,
    PcuDispatchFeatureCaps,
    PcuDispatchIndex,
    PcuDispatchKernelIr,
    PcuDispatchOp,
    PcuDispatchValueId,
    PcuScalarType,
    PcuValueType,
    PcuValueTypeCaps,
};

/// Structural failure in the bounded scalar identity-transport profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PcuScalarIdentityValidationError {
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

/// Validates one typed scalar copy per invocation or grid-stride iteration.
///
/// The profile admits exactly one read-only storage binding, one writable storage binding of the
/// same scalar type, one matching load/store pair, and a terminal return. It performs no numeric
/// operation and covers every scalar type encoding in [`PcuScalarType`]. Backend support for a
/// particular encoding is a separate admission decision. Dispatch launch/geometry, binding extent,
/// and byte-size checks remain the caller's responsibility.
///
/// # Errors
///
/// Returns the first structural failure; the kernel is left untouched.
pub fn validate_scalar_identity_kernel(
    kernel: &PcuDispatchKernelIr<'_>,
    scalar: PcuScalarType,
) -> Result<(), PcuScalarIdentityValidationError> {
    validate_scalar_transport(kernel, scalar, false)
}

/// Validates a representation-preserving broadcast from one read-only scalar.
///
/// The source always loads element zero; direct/grid output remains indexed.
/// This separate admission entry deliberately leaves dense identity validation
/// unchanged: an executor that only implements prefix copies must reject broadcast.
/// Backend binding spans, ownership and launch validation remain separate obligations.
///
/// # Errors
/// Returns the first invalid schema, SSA, index or access role.
pub fn validate_scalar_broadcast_kernel(
    kernel: &PcuDispatchKernelIr<'_>,
    scalar: PcuScalarType,
) -> Result<(), PcuScalarIdentityValidationError> {
    validate_scalar_transport(kernel, scalar, true)
}

#[allow(clippy::too_many_lines)] // Keep exact load/store/SSA admission in a single cold pass.
fn validate_scalar_transport(
    kernel: &PcuDispatchKernelIr<'_>,
    scalar: PcuScalarType,
    broadcast: bool,
) -> Result<(), PcuScalarIdentityValidationError> {
    use PcuScalarIdentityValidationError as Error;

    if !kernel.ports.is_empty() || !kernel.parameters.is_empty() || kernel.bindings.len() != 2 {
        return Err(Error::UnsupportedInterface);
    }
    let supported_types = PcuValueTypeCaps::for_value_type(PcuValueType::Scalar(scalar));
    let supported_features = PcuDispatchFeatureCaps::READ_ONLY_RESOURCES
        .union(PcuDispatchFeatureCaps::MUTABLE_RESOURCES);
    for (index, binding) in kernel.bindings.iter().enumerate() {
        let target = binding.reference();
        if kernel.bindings[..index]
            .iter()
            .any(|previous| previous.reference() == target)
        {
            return Err(Error::DuplicateBinding(target));
        }
        if binding.storage != PcuBindingStorageClass::Storage
            || binding.builtin.is_some()
            || binding.binding_type != PcuBindingType::Value(PcuValueType::Scalar(scalar))
        {
            return Err(Error::InvalidBinding(target));
        }
    }
    if kernel.required_type_support().bits() & !supported_types.bits() != 0
        || kernel.required_feature_support().bits() & !supported_features.bits() != 0
    {
        return Err(Error::UnsupportedRequirements);
    }

    if let [
        PcuDispatchOp::GridStrideLoop { extent, body },
        PcuDispatchOp::Control(PcuDispatchControlOp::Return),
    ] = kernel.ops
    {
        if *extent == 0 {
            return Err(Error::UnsupportedOperation(0));
        }
        return validate_grid_body(kernel, body, broadcast);
    }

    if kernel.ops.len() != 3 {
        return Err(match kernel.ops.len() {
            0 => Error::MissingLoad,
            1 => Error::MissingStore,
            2 => Error::MissingReturn,
            position => Error::UnsupportedOperation(position.min(3)),
        });
    }
    let (value, input) = match kernel.ops[0] {
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
            result,
            binding,
            index,
        }) if index
            == if broadcast {
                PcuDispatchIndex::BindingElementZero
            } else {
                PcuDispatchIndex::InvocationId
            } =>
        {
            (result, binding)
        }
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad { .. }) => {
            return Err(Error::InvalidIndex(0));
        }
        _ => return Err(Error::MissingLoad),
    };
    if value.0 == 0 {
        return Err(Error::InvalidValue(value));
    }
    validate_input(kernel, input)?;
    if broadcast {
        validate_broadcast_input(kernel, input)?;
    }
    match kernel.ops[1] {
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
            binding,
            index: PcuDispatchIndex::InvocationId,
            value: stored,
        }) => {
            if stored != value {
                return Err(Error::InvalidValue(stored));
            }
            if input == binding {
                return Err(Error::InvalidBinding(binding));
            }
            validate_output(kernel, binding)?;
        }
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore { .. }) => {
            return Err(Error::InvalidIndex(1));
        }
        _ => return Err(Error::MissingStore),
    }
    if !matches!(
        kernel.ops[2],
        PcuDispatchOp::Control(PcuDispatchControlOp::Return)
    ) {
        return Err(Error::MissingReturn);
    }
    Ok(())
}

fn validate_grid_body(
    kernel: &PcuDispatchKernelIr<'_>,
    body: &[PcuDispatchOp<'_>],
    broadcast: bool,
) -> Result<(), PcuScalarIdentityValidationError> {
    use PcuScalarIdentityValidationError as Error;

    let [
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
            result,
            binding: input,
            index,
        }),
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
            binding: output,
            index: PcuDispatchIndex::GridStrideId,
            value,
        }),
    ] = body
    else {
        return Err(Error::UnsupportedOperation(0));
    };
    let expected = if broadcast {
        PcuDispatchIndex::BindingElementZero
    } else {
        PcuDispatchIndex::GridStrideId
    };
    if *index != expected {
        return Err(Error::UnsupportedOperation(0));
    }
    if result.0 == 0 || value != result {
        return Err(Error::InvalidValue(*value));
    }
    if input == output {
        return Err(Error::InvalidBinding(*output));
    }
    validate_input(kernel, *input)?;
    if broadcast {
        validate_broadcast_input(kernel, *input)?;
    }
    validate_output(kernel, *output)
}

fn validate_broadcast_input(
    kernel: &PcuDispatchKernelIr<'_>,
    target: PcuBindingRef,
) -> Result<(), PcuScalarIdentityValidationError> {
    if kernel.bindings.iter().any(|binding| {
        binding.reference() == target && binding.access == PcuBindingAccess::ReadOnly
    }) {
        Ok(())
    } else {
        Err(PcuScalarIdentityValidationError::InvalidBinding(target))
    }
}

fn validate_input(
    kernel: &PcuDispatchKernelIr<'_>,
    target: PcuBindingRef,
) -> Result<(), PcuScalarIdentityValidationError> {
    let Some(binding) = kernel
        .bindings
        .iter()
        .find(|binding| binding.reference() == target)
    else {
        return Err(PcuScalarIdentityValidationError::InvalidBinding(target));
    };
    if !matches!(
        binding.access,
        PcuBindingAccess::ReadOnly | PcuBindingAccess::ReadWrite
    ) {
        return Err(PcuScalarIdentityValidationError::InvalidBinding(target));
    }
    Ok(())
}

fn validate_output(
    kernel: &PcuDispatchKernelIr<'_>,
    target: PcuBindingRef,
) -> Result<(), PcuScalarIdentityValidationError> {
    let Some(binding) = kernel
        .bindings
        .iter()
        .find(|binding| binding.reference() == target)
    else {
        return Err(PcuScalarIdentityValidationError::InvalidBinding(target));
    };
    if !matches!(
        binding.access,
        PcuBindingAccess::WriteOnly | PcuBindingAccess::ReadWrite
    ) {
        return Err(PcuScalarIdentityValidationError::InvalidBinding(target));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    #[rustfmt::skip]
    use super::{
        PcuScalarIdentityValidationError as Error,
        validate_scalar_broadcast_kernel,
        validate_scalar_identity_kernel,
    };
    #[rustfmt::skip]
    use crate::{
        PcuBinding,
        PcuBindingAccess,
        PcuBindingRef,
        PcuBindingStorageClass,
        PcuDispatchControlOp,
        PcuDispatchDataOp,
        PcuDispatchFeatureCaps,
        PcuDispatchIndex,
        PcuDispatchKernelIr,
        PcuDispatchOp,
        PcuDispatchValueId,
        PcuKernelId,
        PcuScalarType,
        PcuValueType,
        PcuValueTypeCaps,
    };

    const SCALAR_TYPES: [PcuScalarType; PcuScalarType::COUNT] = [
        PcuScalarType::Bool,
        PcuScalarType::I4,
        PcuScalarType::U4,
        PcuScalarType::F16,
        PcuScalarType::BF16,
        PcuScalarType::F32,
        PcuScalarType::F64,
        PcuScalarType::U8,
        PcuScalarType::U16,
        PcuScalarType::U32,
        PcuScalarType::U64,
        PcuScalarType::I8,
        PcuScalarType::I16,
        PcuScalarType::I32,
        PcuScalarType::I64,
        PcuScalarType::I128,
        PcuScalarType::U128,
        PcuScalarType::I256,
        PcuScalarType::U256,
        PcuScalarType::I512,
        PcuScalarType::U512,
        PcuScalarType::F128,
        PcuScalarType::F256,
        PcuScalarType::F8E4M3FN,
        PcuScalarType::F8E5M2,
    ];

    fn bindings(scalar: PcuScalarType) -> [PcuBinding<'static>; 2] {
        [
            PcuBinding::value(
                Some("input"),
                0,
                0,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::ReadOnly,
                PcuValueType::Scalar(scalar),
            ),
            PcuBinding::value(
                Some("output"),
                0,
                1,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::WriteOnly,
                PcuValueType::Scalar(scalar),
            ),
        ]
    }

    fn kernel<'a>(
        bindings: &'a [PcuBinding<'static>],
        ops: &'a [PcuDispatchOp<'a>],
    ) -> PcuDispatchKernelIr<'a> {
        PcuDispatchKernelIr {
            numerical_requirements: PcuDispatchKernelIr::DEFAULT_REQUIREMENTS,
            id: PcuKernelId(1),
            entry: crate::PcuDispatchEntryPoint {
                name: "identity",
                logical_shape: [4, 1, 1],
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
    fn scalar_broadcast_is_separate_from_dense_copy_and_keeps_readonly_access() {
        for scalar in SCALAR_TYPES {
            let bindings = bindings(scalar);
            for grid in [false, true] {
                let output_index = if grid {
                    PcuDispatchIndex::GridStrideId
                } else {
                    PcuDispatchIndex::InvocationId
                };
                let body = [
                    PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                        result: PcuDispatchValueId(1),
                        binding: PcuBindingRef::new(0, 0),
                        index: PcuDispatchIndex::BindingElementZero,
                    }),
                    PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                        binding: PcuBindingRef::new(0, 1),
                        index: output_index,
                        value: PcuDispatchValueId(1),
                    }),
                ];
                let direct = [
                    body[0],
                    body[1],
                    PcuDispatchOp::Control(PcuDispatchControlOp::Return),
                ];
                let looped = [
                    PcuDispatchOp::GridStrideLoop {
                        extent: 17,
                        body: &body,
                    },
                    PcuDispatchOp::Control(PcuDispatchControlOp::Return),
                ];
                let ir = kernel(&bindings, if grid { &looped } else { &direct });
                validate_scalar_broadcast_kernel(&ir, scalar).unwrap();
                assert!(validate_scalar_identity_kernel(&ir, scalar).is_err());
                let mut wrong_access = bindings;
                wrong_access[0].access = PcuBindingAccess::ReadWrite;
                let wrong = PcuDispatchKernelIr {
                    bindings: &wrong_access,
                    ..ir
                };
                assert_eq!(
                    validate_scalar_broadcast_kernel(&wrong, scalar),
                    Err(Error::InvalidBinding(PcuBindingRef::new(0, 0)))
                );
            }
        }
    }

    #[test]
    fn accepts_every_sealed_scalar_type_for_direct_and_grid_identity() {
        for scalar in SCALAR_TYPES {
            let bindings = bindings(scalar);
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
            assert_eq!(
                validate_scalar_identity_kernel(&kernel(&bindings, &direct), scalar),
                Ok(())
            );

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
                    extent: 16,
                    body: &body,
                },
                direct[2],
            ];
            assert_eq!(
                validate_scalar_identity_kernel(&kernel(&bindings, &grid), scalar),
                Ok(())
            );
        }
    }

    #[test]
    fn rejects_wrong_scalar_type_index_access_and_extra_operations() {
        let scalar = PcuScalarType::I64;
        let bindings = bindings(scalar);
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
        assert_eq!(
            validate_scalar_identity_kernel(&kernel(&bindings, &direct), PcuScalarType::U64),
            Err(Error::InvalidBinding(PcuBindingRef::new(0, 0)))
        );
        let wrong_index = [
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result: PcuDispatchValueId(1),
                binding: PcuBindingRef::new(0, 0),
                index: PcuDispatchIndex::BindingElementZero,
            }),
            direct[1],
            direct[2],
        ];
        assert_eq!(
            validate_scalar_identity_kernel(&kernel(&bindings, &wrong_index), scalar),
            Err(Error::InvalidIndex(0))
        );
        let extra = [direct[0], direct[1], direct[2], direct[2]];
        assert_eq!(
            validate_scalar_identity_kernel(&kernel(&bindings, &extra), scalar),
            Err(Error::UnsupportedOperation(3))
        );
        let unsupported_types = PcuDispatchKernelIr {
            type_caps: PcuValueTypeCaps::VECTOR_VALUES,
            ..kernel(&bindings, &direct)
        };
        assert_eq!(
            validate_scalar_identity_kernel(&unsupported_types, scalar),
            Err(Error::UnsupportedRequirements)
        );
        let unsupported_features = PcuDispatchKernelIr {
            feature_caps: PcuDispatchFeatureCaps::INLINE_PARAMETERS,
            ..kernel(&bindings, &direct)
        };
        assert_eq!(
            validate_scalar_identity_kernel(&unsupported_features, scalar),
            Err(Error::UnsupportedRequirements)
        );
        let read_write_input = [
            PcuBinding::value(
                Some("input"),
                0,
                0,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::ReadWrite,
                PcuValueType::Scalar(scalar),
            ),
            bindings[1],
        ];
        assert_eq!(
            validate_scalar_identity_kernel(&kernel(&read_write_input, &direct), scalar),
            Ok(())
        );
        let write_only_input = [
            PcuBinding::value(
                Some("input"),
                0,
                0,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::WriteOnly,
                PcuValueType::Scalar(scalar),
            ),
            bindings[1],
        ];
        assert_eq!(
            validate_scalar_identity_kernel(&kernel(&write_only_input, &direct), scalar),
            Err(Error::InvalidBinding(PcuBindingRef::new(0, 0)))
        );
    }
}
