//! Validation for ordered command kernels.

use super::find_binding;
use crate::{
    PcuBindingAccess,
    PcuBindingRef,
    PcuPortDirection,
    PcuValueType,
};
use crate::model::{
    PcuCommandEffectKind,
    PcuCommandKernelIr,
    PcuCommandOp,
    PcuCommandPredicate,
    PcuCommandResultId,
    PcuOperand,
    PcuTarget,
};

/// Contract failures surfaced when a typed command result flow is malformed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PcuCommandValidationError {
    DuplicateResultId(PcuCommandResultId),
    ResultUsedBeforeDefinition(PcuCommandResultId),
    LegacyPreviousResultUnverified,
    NonScalarReadResult(PcuValueType),
    ReadWidthMismatch {
        step: usize,
        expected: u8,
        found: u8,
    },
    ReadTargetTypeMismatch {
        expected: PcuValueType,
        found: PcuValueType,
    },
    ReadTargetNotReadable(PcuBindingRef),
    ReadTargetHasNoValueType(PcuBindingRef),
    MissingReadTargetBinding(PcuBindingRef),
    UndeclaredReadPort,
    OpaqueReadTarget,
    ReadEffectMismatch {
        step: usize,
        expected: PcuCommandEffectKind,
        found: PcuCommandEffectKind,
    },
}

/// Validates typed results produced by ordered command reads and their subsequent uses.
///
/// Result IDs are local to one command kernel. A result is visible only after its defining
/// `ReadResult` step, and typed reads must state a scalar type whose bit width matches the
/// declared access width. Targets with declared binding or port types are checked against that
/// type; named and intrinsic targets remain intentionally opaque to this backend-neutral layer.
///
/// # Errors
///
/// Returns the first invalid typed read, duplicate result, or use-before-definition error.
#[allow(clippy::too_many_lines)] // Ordered read/effect checks are one verifier pass.
pub fn validate_command_kernel(
    kernel: &PcuCommandKernelIr<'_>,
) -> Result<(), PcuCommandValidationError> {
    for (index, step) in kernel.steps.iter().enumerate() {
        if let PcuCommandOp::ReadResult {
            target,
            result,
            width_bits,
            effect,
        } = step.op
        {
            let expected_effect = PcuCommandEffectKind::Read;
            if effect != expected_effect {
                return Err(PcuCommandValidationError::ReadEffectMismatch {
                    step: index,
                    expected: expected_effect,
                    found: effect,
                });
            }
            let PcuValueType::Scalar(scalar) = result.value_type else {
                return Err(PcuCommandValidationError::NonScalarReadResult(
                    result.value_type,
                ));
            };
            let expected_width = scalar.bit_width();
            if width_bits != expected_width {
                return Err(PcuCommandValidationError::ReadWidthMismatch {
                    step: index,
                    expected: expected_width,
                    found: width_bits,
                });
            }
            match target {
                PcuTarget::Binding(reference) => {
                    let Some(binding) = find_binding(kernel.bindings, reference) else {
                        return Err(PcuCommandValidationError::MissingReadTargetBinding(
                            reference,
                        ));
                    };
                    if matches!(binding.access, PcuBindingAccess::WriteOnly) {
                        return Err(PcuCommandValidationError::ReadTargetNotReadable(reference));
                    }
                    let Some(target_type) = binding.value_type() else {
                        return Err(PcuCommandValidationError::ReadTargetHasNoValueType(
                            reference,
                        ));
                    };
                    if target_type != result.value_type {
                        return Err(PcuCommandValidationError::ReadTargetTypeMismatch {
                            expected: target_type,
                            found: result.value_type,
                        });
                    }
                }
                PcuTarget::Port(name) => {
                    let Some(port) = kernel.ports.iter().find(|port| port.name == Some(name))
                    else {
                        return Err(PcuCommandValidationError::UndeclaredReadPort);
                    };
                    if port.direction != PcuPortDirection::Input {
                        return Err(PcuCommandValidationError::UndeclaredReadPort);
                    }
                    if port.value_type != result.value_type {
                        return Err(PcuCommandValidationError::ReadTargetTypeMismatch {
                            expected: port.value_type,
                            found: result.value_type,
                        });
                    }
                }
                PcuTarget::Named(_) | PcuTarget::Intrinsic(_) => {
                    return Err(PcuCommandValidationError::OpaqueReadTarget);
                }
            }
            if kernel.steps[..index].iter().any(|prior| {
                matches!(prior.op, PcuCommandOp::ReadResult { result: prior_result, .. }
                    if prior_result.id == result.id)
            }) {
                return Err(PcuCommandValidationError::DuplicateResultId(result.id));
            }
        }

        match step.op {
            PcuCommandOp::Write { value, .. }
            | PcuCommandOp::Modify { value, .. }
            | PcuCommandOp::Return { value: Some(value) } => {
                validate_command_result_use(kernel, index, value)?;
            }
            PcuCommandOp::Invoke { args, .. } => {
                for argument in args.iter().copied() {
                    validate_command_result_use(kernel, index, argument)?;
                }
            }
            PcuCommandOp::Await { predicate } => match predicate {
                PcuCommandPredicate::Equals { left, right } => {
                    validate_command_result_use(kernel, index, left)?;
                    validate_command_result_use(kernel, index, right)?;
                }
                PcuCommandPredicate::NonZero(value) => {
                    validate_command_result_use(kernel, index, value)?;
                }
                PcuCommandPredicate::Ready(_) | PcuCommandPredicate::Named(_) => {}
            },
            PcuCommandOp::Read { .. }
            | PcuCommandOp::ReadResult { .. }
            | PcuCommandOp::Copy { .. }
            | PcuCommandOp::Stall { .. }
            | PcuCommandOp::Sleep { .. }
            | PcuCommandOp::Barrier
            | PcuCommandOp::Return { value: None } => {}
        }
    }
    Ok(())
}

fn validate_command_result_use(
    kernel: &PcuCommandKernelIr<'_>,
    index: usize,
    operand: PcuOperand<'_>,
) -> Result<(), PcuCommandValidationError> {
    match operand {
        PcuOperand::Result(id)
            if !kernel.steps[..index].iter().any(|prior| {
                matches!(prior.op, PcuCommandOp::ReadResult { result, .. } if result.id == id)
            }) => return Err(PcuCommandValidationError::ResultUsedBeforeDefinition(id)),
        PcuOperand::PreviousResult => {
            return Err(PcuCommandValidationError::LegacyPreviousResultUnverified);
        }
        _ => {}
    }
    Ok(())
}

#[cfg(test)]
mod command_result_tests {
    use super::*;
    #[cfg(test)]
    use crate::PcuBinding;
    use crate::model::{
        PcuCommandResult,
        PcuCommandStep,
    };

    fn typed_read(id: u16, value_type: PcuValueType, width_bits: u8) -> PcuCommandOp<'static> {
        PcuCommandOp::ReadResult {
            target: PcuTarget::Port("status"),
            result: PcuCommandResult {
                id: PcuCommandResultId(id),
                value_type,
            },
            width_bits,
            effect: PcuCommandEffectKind::Read,
        }
    }

    fn validate_steps(steps: &[PcuCommandStep<'_>]) -> Result<(), PcuCommandValidationError> {
        let ports = &[crate::PcuPort::stream_input(
            Some("status"),
            PcuValueType::u32(),
        )];
        let kernel = PcuCommandKernelIr {
            id: crate::PcuKernelId(1),
            entry_point: "command",
            bindings: &[],
            ports,
            parameters: &[],
            steps,
        };
        validate_command_kernel(&kernel)
    }

    fn validate_steps_with_bindings(
        steps: &[PcuCommandStep<'_>],
        bindings: &[PcuBinding<'_>],
    ) -> Result<(), PcuCommandValidationError> {
        let kernel = PcuCommandKernelIr {
            id: crate::PcuKernelId(1),
            entry_point: "command",
            bindings,
            ports: &[],
            parameters: &[],
            steps,
        };
        validate_command_kernel(&kernel)
    }

    #[test]
    fn validates_ordered_typed_read_return_flow() {
        let steps = [
            PcuCommandStep {
                name: Some("read"),
                op: typed_read(3, PcuValueType::u32(), 32),
            },
            PcuCommandStep {
                name: Some("return"),
                op: PcuCommandOp::Return {
                    value: Some(PcuOperand::Result(PcuCommandResultId(3))),
                },
            },
        ];
        assert_eq!(validate_steps(&steps), Ok(()));
    }

    #[test]
    fn rejects_undefined_and_duplicate_result_ids() {
        let undefined = [PcuCommandStep {
            name: Some("return"),
            op: PcuCommandOp::Return {
                value: Some(PcuOperand::Result(PcuCommandResultId(3))),
            },
        }];
        assert_eq!(
            validate_steps(&undefined),
            Err(PcuCommandValidationError::ResultUsedBeforeDefinition(
                PcuCommandResultId(3)
            ))
        );

        let duplicate = [
            PcuCommandStep {
                name: None,
                op: typed_read(3, PcuValueType::u32(), 32),
            },
            PcuCommandStep {
                name: None,
                op: typed_read(3, PcuValueType::u32(), 32),
            },
        ];
        assert_eq!(
            validate_steps(&duplicate),
            Err(PcuCommandValidationError::DuplicateResultId(
                PcuCommandResultId(3)
            ))
        );
    }

    #[test]
    fn rejects_wrong_width_type_access_and_opaque_target() {
        let wrong_width = [PcuCommandStep {
            name: Some("read"),
            op: typed_read(3, PcuValueType::u32(), 16),
        }];
        assert_eq!(
            validate_steps(&wrong_width),
            Err(PcuCommandValidationError::ReadWidthMismatch {
                step: 0,
                expected: 32,
                found: 16
            })
        );

        let wrong_type = [PcuCommandStep {
            name: Some("read"),
            op: typed_read(3, PcuValueType::u16(), 16),
        }];
        assert_eq!(
            validate_steps(&wrong_type),
            Err(PcuCommandValidationError::ReadTargetTypeMismatch {
                expected: PcuValueType::u32(),
                found: PcuValueType::u16()
            })
        );

        let opaque = [PcuCommandStep {
            name: Some("read"),
            op: PcuCommandOp::ReadResult {
                target: PcuTarget::Named("opaque-register"),
                result: PcuCommandResult {
                    id: PcuCommandResultId(3),
                    value_type: PcuValueType::u32(),
                },
                width_bits: 32,
                effect: PcuCommandEffectKind::Read,
            },
        }];
        assert_eq!(
            validate_steps(&opaque),
            Err(PcuCommandValidationError::OpaqueReadTarget)
        );

        let write_only = [crate::PcuBinding::value(
            Some("status"),
            0,
            0,
            crate::PcuBindingStorageClass::Storage,
            PcuBindingAccess::WriteOnly,
            PcuValueType::u32(),
        )];
        let binding_read = [PcuCommandStep {
            name: Some("read"),
            op: PcuCommandOp::ReadResult {
                target: PcuTarget::Binding(PcuBindingRef::new(0, 0)),
                result: PcuCommandResult {
                    id: PcuCommandResultId(3),
                    value_type: PcuValueType::u32(),
                },
                width_bits: 32,
                effect: PcuCommandEffectKind::Read,
            },
        }];
        assert_eq!(
            validate_steps_with_bindings(&binding_read, &write_only),
            Err(PcuCommandValidationError::ReadTargetNotReadable(
                PcuBindingRef::new(0, 0)
            ))
        );

        let wrong_effect = [PcuCommandStep {
            name: Some("read"),
            op: PcuCommandOp::ReadResult {
                target: PcuTarget::Port("status"),
                result: PcuCommandResult {
                    id: PcuCommandResultId(3),
                    value_type: PcuValueType::u32(),
                },
                width_bits: 32,
                effect: PcuCommandEffectKind::Write,
            },
        }];
        assert_eq!(
            validate_steps(&wrong_effect),
            Err(PcuCommandValidationError::ReadEffectMismatch {
                step: 0,
                expected: PcuCommandEffectKind::Read,
                found: PcuCommandEffectKind::Write,
            })
        );

        let previous_result = [PcuCommandStep {
            name: Some("return"),
            op: PcuCommandOp::Return {
                value: Some(PcuOperand::PreviousResult),
            },
        }];
        assert_eq!(
            validate_steps(&previous_result),
            Err(PcuCommandValidationError::LegacyPreviousResultUnverified)
        );
    }
}
