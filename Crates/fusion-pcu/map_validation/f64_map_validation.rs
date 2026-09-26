//! Common structural admission for the bounded scalar `f64` indexed-map profile.

use crate::{
    PcuBindingAccess,
    PcuBindingRef,
    PcuBindingStorageClass,
    PcuBindingType,
    PcuDispatchAluOp,
    PcuDispatchControlOp,
    PcuDispatchDataOp,
    PcuDispatchIndex,
    PcuDispatchKernelIr,
    PcuDispatchOp,
    PcuDispatchValueId,
    PcuValueType,
};

/// Structural failure within the portable scalar `f64` map profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PcuF64MapValidationError {
    UnsupportedInterface,
    InvalidBinding(PcuBindingRef),
    DuplicateBinding(PcuBindingRef),
    UnsupportedOperation(usize),
    InvalidIndex(usize),
    InvalidValue(PcuDispatchValueId),
    DuplicateValue(PcuDispatchValueId),
    MissingStore,
    MissingReturn,
}

/// Validates the shared f64 indexed-map IR law before any backend-specific assessment.
///
/// This profile admits only storage buffers of scalar f64, invocation-ID indexed loads, element-
/// zero broadcast loads, invocation-ID indexed stores,
/// f64 add/subtract/multiply/divide, and one terminal return. It
/// does not assert
/// backend numeric equivalence or resource availability. The current cross-backend numeric
/// conformance domain for arithmetic remains deliberately narrower: finite normal operands and
/// exact results that do not depend on subnormal handling, overflow/underflow, or
/// contraction/reassociation. Min/Max have a shared edge contract: one NaN returns the numeric
/// operand, two NaNs return NaN, Max(-0.0, +0.0) returns +0.0, and Min(-0.0, +0.0) returns -0.0.
///
/// # Errors
///
/// Returns the first structural failure and leaves the kernel untouched.
#[allow(clippy::too_many_lines)]
pub fn validate_f64_map_kernel(
    kernel: &PcuDispatchKernelIr<'_>,
) -> Result<(), PcuF64MapValidationError> {
    if !kernel.ports.is_empty() || !kernel.parameters.is_empty() {
        return Err(PcuF64MapValidationError::UnsupportedInterface);
    }
    for (index, binding) in kernel.bindings.iter().enumerate() {
        let target = binding.reference();
        if kernel.bindings[..index]
            .iter()
            .any(|previous| previous.reference() == target)
        {
            return Err(PcuF64MapValidationError::DuplicateBinding(target));
        }
        if binding.storage != PcuBindingStorageClass::Storage
            || binding.builtin.is_some()
            || binding.binding_type != PcuBindingType::Value(PcuValueType::f64())
        {
            return Err(PcuF64MapValidationError::InvalidBinding(target));
        }
    }

    if let Some((position, extent, body)) =
        kernel
            .ops
            .iter()
            .enumerate()
            .find_map(|(position, op)| match op {
                PcuDispatchOp::GridStrideLoop { extent, body } => Some((position, extent, body)),
                _ => None,
            })
    {
        if position != 0
            || kernel.ops.len() != 2
            || *extent == 0
            || !matches!(
                kernel.ops[1],
                PcuDispatchOp::Control(PcuDispatchControlOp::Return)
            )
        {
            return Err(PcuF64MapValidationError::UnsupportedOperation(position));
        }
        return validate_grid_stride_body(kernel, body);
    }

    let mut saw_store = false;
    let mut saw_return = false;
    for (position, op) in kernel.ops.iter().copied().enumerate() {
        if saw_return {
            return Err(PcuF64MapValidationError::UnsupportedOperation(position));
        }
        match op {
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result,
                binding,
                index,
            }) => {
                check_load_index(index, position)?;
                check_binding(kernel, binding, false)?;
                define(kernel, position, result)?;
            }
            PcuDispatchOp::Data(PcuDispatchDataOp::Alu {
                value_type,
                result,
                op,
                lhs,
                rhs,
            }) => {
                if value_type != PcuValueType::f64() {
                    return Err(PcuF64MapValidationError::UnsupportedOperation(position));
                }
                if !matches!(
                    op,
                    PcuDispatchAluOp::Add
                        | PcuDispatchAluOp::Sub
                        | PcuDispatchAluOp::Mul
                        | PcuDispatchAluOp::Div
                ) {
                    return Err(PcuF64MapValidationError::UnsupportedOperation(position));
                }
                require(kernel, position, lhs)?;
                require(kernel, position, rhs)?;
                define(kernel, position, result)?;
            }
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                binding,
                index,
                value,
            }) => {
                check_store_index(index, position)?;
                check_binding(kernel, binding, true)?;
                require(kernel, position, value)?;
                saw_store = true;
            }
            PcuDispatchOp::Control(PcuDispatchControlOp::Return) => saw_return = true,
            _ => return Err(PcuF64MapValidationError::UnsupportedOperation(position)),
        }
    }
    if !saw_store {
        return Err(PcuF64MapValidationError::MissingStore);
    }
    if !saw_return {
        return Err(PcuF64MapValidationError::MissingReturn);
    }
    Ok(())
}

fn validate_grid_stride_body(
    kernel: &PcuDispatchKernelIr<'_>,
    body: &[PcuDispatchOp<'_>],
) -> Result<(), PcuF64MapValidationError> {
    let mut saw_store = false;
    for (body_position, op) in body.iter().copied().enumerate() {
        match op {
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result,
                binding,
                index,
            }) => {
                check_grid_load_index(index, body_position)?;
                check_binding(kernel, binding, false)?;
                define_in(body, body_position, result)?;
            }
            PcuDispatchOp::Data(PcuDispatchDataOp::Alu {
                value_type,
                result,
                op,
                lhs,
                rhs,
            }) => {
                if value_type != PcuValueType::f64() {
                    return Err(PcuF64MapValidationError::UnsupportedOperation(
                        body_position,
                    ));
                }
                if !matches!(
                    op,
                    PcuDispatchAluOp::Add
                        | PcuDispatchAluOp::Sub
                        | PcuDispatchAluOp::Mul
                        | PcuDispatchAluOp::Div
                ) {
                    return Err(PcuF64MapValidationError::UnsupportedOperation(
                        body_position,
                    ));
                }
                require_in(body, body_position, lhs)?;
                require_in(body, body_position, rhs)?;
                define_in(body, body_position, result)?;
            }
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                binding,
                index,
                value,
            }) => {
                check_grid_store_index(index, body_position)?;
                check_binding(kernel, binding, true)?;
                require_in(body, body_position, value)?;
                saw_store = true;
            }
            _ => {
                return Err(PcuF64MapValidationError::UnsupportedOperation(
                    body_position,
                ));
            }
        }
    }
    if saw_store {
        Ok(())
    } else {
        Err(PcuF64MapValidationError::MissingStore)
    }
}

const fn check_load_index(
    index: PcuDispatchIndex,
    position: usize,
) -> Result<(), PcuF64MapValidationError> {
    if matches!(
        index,
        PcuDispatchIndex::InvocationId | PcuDispatchIndex::BindingElementZero
    ) {
        Ok(())
    } else {
        Err(PcuF64MapValidationError::InvalidIndex(position))
    }
}

const fn check_store_index(
    index: PcuDispatchIndex,
    position: usize,
) -> Result<(), PcuF64MapValidationError> {
    if matches!(index, PcuDispatchIndex::InvocationId) {
        Ok(())
    } else {
        Err(PcuF64MapValidationError::InvalidIndex(position))
    }
}

const fn check_grid_load_index(
    index: PcuDispatchIndex,
    position: usize,
) -> Result<(), PcuF64MapValidationError> {
    if matches!(
        index,
        PcuDispatchIndex::GridStrideId | PcuDispatchIndex::BindingElementZero
    ) {
        Ok(())
    } else {
        Err(PcuF64MapValidationError::InvalidIndex(position))
    }
}

const fn check_grid_store_index(
    index: PcuDispatchIndex,
    position: usize,
) -> Result<(), PcuF64MapValidationError> {
    if matches!(index, PcuDispatchIndex::GridStrideId) {
        Ok(())
    } else {
        Err(PcuF64MapValidationError::InvalidIndex(position))
    }
}

fn define_in(
    ops: &[PcuDispatchOp<'_>],
    position: usize,
    value: PcuDispatchValueId,
) -> Result<(), PcuF64MapValidationError> {
    if value.0 == 0 {
        return Err(PcuF64MapValidationError::InvalidValue(value));
    }
    if ops[..position]
        .iter()
        .copied()
        .any(|op| result_id(op) == Some(value))
    {
        Err(PcuF64MapValidationError::DuplicateValue(value))
    } else {
        Ok(())
    }
}

fn require_in(
    ops: &[PcuDispatchOp<'_>],
    position: usize,
    value: PcuDispatchValueId,
) -> Result<(), PcuF64MapValidationError> {
    if value.0 != 0
        && ops[..position]
            .iter()
            .copied()
            .any(|op| result_id(op) == Some(value))
    {
        Ok(())
    } else {
        Err(PcuF64MapValidationError::InvalidValue(value))
    }
}

fn check_binding(
    kernel: &PcuDispatchKernelIr<'_>,
    target: PcuBindingRef,
    write: bool,
) -> Result<(), PcuF64MapValidationError> {
    let Some(binding) = kernel
        .bindings
        .iter()
        .find(|binding| binding.reference() == target)
    else {
        return Err(PcuF64MapValidationError::InvalidBinding(target));
    };
    if (write && binding.access == PcuBindingAccess::ReadOnly)
        || (!write && binding.access == PcuBindingAccess::WriteOnly)
    {
        return Err(PcuF64MapValidationError::InvalidBinding(target));
    }
    Ok(())
}

fn define(
    kernel: &PcuDispatchKernelIr<'_>,
    position: usize,
    value: PcuDispatchValueId,
) -> Result<(), PcuF64MapValidationError> {
    if value.0 == 0 {
        return Err(PcuF64MapValidationError::InvalidValue(value));
    }
    if kernel.ops[..position]
        .iter()
        .copied()
        .any(|op| result_id(op) == Some(value))
    {
        return Err(PcuF64MapValidationError::DuplicateValue(value));
    }
    Ok(())
}

fn require(
    kernel: &PcuDispatchKernelIr<'_>,
    position: usize,
    value: PcuDispatchValueId,
) -> Result<(), PcuF64MapValidationError> {
    if value.0 != 0
        && kernel.ops[..position]
            .iter()
            .copied()
            .any(|op| result_id(op) == Some(value))
    {
        Ok(())
    } else {
        Err(PcuF64MapValidationError::InvalidValue(value))
    }
}

const fn result_id(op: PcuDispatchOp<'_>) -> Option<PcuDispatchValueId> {
    match op {
        PcuDispatchOp::Data(
            PcuDispatchDataOp::BindingLoad { result, .. }
            | PcuDispatchDataOp::Constant { result, .. }
            | PcuDispatchDataOp::Alu { result, .. },
        ) => Some(result),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        PcuF64MapValidationError,
        validate_f64_map_kernel,
    };
    use crate::{
        PcuBinding,
        PcuBindingAccess,
        PcuBindingRef,
        PcuBindingStorageClass,
        PcuDispatchAluOp,
        PcuDispatchDataOp,
        PcuDispatchEntryPoint,
        PcuDispatchIndex,
        PcuDispatchKernelIr,
        PcuDispatchOp,
        PcuDispatchValueId,
        PcuKernelId,
        PcuValueTypeCaps,
        PcuDispatchFeatureCaps,
    };

    #[test]
    fn accepts_f64_arithmetic_map_and_rejects_minimum() {
        let bindings = [
            PcuBinding::scalar::<f64>(
                Some("input"),
                0,
                0,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::ReadOnly,
            ),
            PcuBinding::scalar::<f64>(
                Some("output"),
                0,
                1,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::ReadWrite,
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
                binding: PcuBindingRef::new(0, 0),
                index: PcuDispatchIndex::InvocationId,
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::Alu {
                value_type: crate::PcuValueType::f64(),
                result: PcuDispatchValueId(3),
                op: PcuDispatchAluOp::Add,
                lhs: PcuDispatchValueId(1),
                rhs: PcuDispatchValueId(2),
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                binding: PcuBindingRef::new(0, 1),
                index: PcuDispatchIndex::InvocationId,
                value: PcuDispatchValueId(3),
            }),
            PcuDispatchOp::Control(crate::PcuDispatchControlOp::Return),
        ];
        let kernel = PcuDispatchKernelIr {
            id: PcuKernelId(1),
            entry: PcuDispatchEntryPoint {
                name: "f64_map",
                logical_shape: [4, 1, 1],
            },
            bindings: &bindings,
            ports: &[],
            parameters: &[],
            ops: &ops,
            type_caps: PcuValueTypeCaps::empty(),
            feature_caps: PcuDispatchFeatureCaps::empty(),
        };
        assert_eq!(validate_f64_map_kernel(&kernel), Ok(()));
        let invalid_ops = [
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result: PcuDispatchValueId(1),
                binding: PcuBindingRef::new(0, 0),
                index: PcuDispatchIndex::InvocationId,
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result: PcuDispatchValueId(2),
                binding: PcuBindingRef::new(0, 0),
                index: PcuDispatchIndex::InvocationId,
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::Alu {
                value_type: crate::PcuValueType::f64(),
                result: PcuDispatchValueId(3),
                op: PcuDispatchAluOp::Min,
                lhs: PcuDispatchValueId(1),
                rhs: PcuDispatchValueId(2),
            }),
        ];
        assert_eq!(
            validate_f64_map_kernel(&PcuDispatchKernelIr {
                ops: &invalid_ops,
                ..kernel
            }),
            Err(PcuF64MapValidationError::UnsupportedOperation(2))
        );
    }
}
