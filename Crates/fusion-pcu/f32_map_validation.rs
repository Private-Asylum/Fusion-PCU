//! Common structural admission for the bounded scalar `f32` indexed-map profile.

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
    PcuParameterValue,
    PcuValueType,
};

/// Structural failure within the portable scalar `f32` map profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PcuF32MapValidationError {
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

/// Validates the shared f32 indexed-map IR law before any backend-specific assessment.
///
/// This profile admits only storage buffers of scalar f32, invocation-ID indexed loads/stores,
/// f32 constants, f32 add/subtract/multiply/divide/minimum/maximum, and one terminal return. It
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
pub fn validate_f32_map_kernel(
    kernel: &PcuDispatchKernelIr<'_>,
) -> Result<(), PcuF32MapValidationError> {
    if !kernel.ports.is_empty() || !kernel.parameters.is_empty() {
        return Err(PcuF32MapValidationError::UnsupportedInterface);
    }
    for (index, binding) in kernel.bindings.iter().enumerate() {
        let target = binding.reference();
        if kernel.bindings[..index]
            .iter()
            .any(|previous| previous.reference() == target)
        {
            return Err(PcuF32MapValidationError::DuplicateBinding(target));
        }
        if binding.storage != PcuBindingStorageClass::Storage
            || binding.builtin.is_some()
            || binding.binding_type != PcuBindingType::Value(PcuValueType::f32())
        {
            return Err(PcuF32MapValidationError::InvalidBinding(target));
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
            return Err(PcuF32MapValidationError::UnsupportedOperation(position));
        }
        return validate_grid_stride_body(kernel, body);
    }

    let mut saw_store = false;
    let mut saw_return = false;
    for (position, op) in kernel.ops.iter().copied().enumerate() {
        if saw_return {
            return Err(PcuF32MapValidationError::UnsupportedOperation(position));
        }
        match op {
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result,
                binding,
                index,
            }) => {
                check_index(index, position)?;
                check_binding(kernel, binding, false)?;
                define(kernel, position, result)?;
            }
            PcuDispatchOp::Data(PcuDispatchDataOp::Constant { result, value }) => {
                if !matches!(value, PcuParameterValue::F32(_)) {
                    return Err(PcuF32MapValidationError::UnsupportedOperation(position));
                }
                define(kernel, position, result)?;
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
                        | PcuDispatchAluOp::Min
                        | PcuDispatchAluOp::Max
                ) {
                    return Err(PcuF32MapValidationError::UnsupportedOperation(position));
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
                check_index(index, position)?;
                check_binding(kernel, binding, true)?;
                require(kernel, position, value)?;
                saw_store = true;
            }
            PcuDispatchOp::Control(PcuDispatchControlOp::Return) => saw_return = true,
            _ => return Err(PcuF32MapValidationError::UnsupportedOperation(position)),
        }
    }
    if !saw_store {
        return Err(PcuF32MapValidationError::MissingStore);
    }
    if !saw_return {
        return Err(PcuF32MapValidationError::MissingReturn);
    }
    Ok(())
}

fn validate_grid_stride_body(
    kernel: &PcuDispatchKernelIr<'_>,
    body: &[PcuDispatchOp<'_>],
) -> Result<(), PcuF32MapValidationError> {
    let mut saw_store = false;
    for (body_position, op) in body.iter().copied().enumerate() {
        match op {
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result,
                binding,
                index,
            }) => {
                check_grid_index(index, body_position)?;
                check_binding(kernel, binding, false)?;
                define_in(body, body_position, result)?;
            }
            PcuDispatchOp::Data(PcuDispatchDataOp::Constant { result, value }) => {
                if !matches!(value, PcuParameterValue::F32(_)) {
                    return Err(PcuF32MapValidationError::UnsupportedOperation(
                        body_position,
                    ));
                }
                define_in(body, body_position, result)?;
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
                        | PcuDispatchAluOp::Min
                        | PcuDispatchAluOp::Max
                ) {
                    return Err(PcuF32MapValidationError::UnsupportedOperation(
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
                check_grid_index(index, body_position)?;
                check_binding(kernel, binding, true)?;
                require_in(body, body_position, value)?;
                saw_store = true;
            }
            _ => {
                return Err(PcuF32MapValidationError::UnsupportedOperation(
                    body_position,
                ));
            }
        }
    }
    if saw_store {
        Ok(())
    } else {
        Err(PcuF32MapValidationError::MissingStore)
    }
}

const fn check_index(
    index: PcuDispatchIndex,
    position: usize,
) -> Result<(), PcuF32MapValidationError> {
    if matches!(index, PcuDispatchIndex::InvocationId) {
        Ok(())
    } else {
        Err(PcuF32MapValidationError::InvalidIndex(position))
    }
}

const fn check_grid_index(
    index: PcuDispatchIndex,
    position: usize,
) -> Result<(), PcuF32MapValidationError> {
    if matches!(index, PcuDispatchIndex::GridStrideId) {
        Ok(())
    } else {
        Err(PcuF32MapValidationError::InvalidIndex(position))
    }
}

fn define_in(
    ops: &[PcuDispatchOp<'_>],
    position: usize,
    value: PcuDispatchValueId,
) -> Result<(), PcuF32MapValidationError> {
    if value.0 == 0 {
        return Err(PcuF32MapValidationError::InvalidValue(value));
    }
    if ops[..position]
        .iter()
        .copied()
        .any(|op| result_id(op) == Some(value))
    {
        Err(PcuF32MapValidationError::DuplicateValue(value))
    } else {
        Ok(())
    }
}

fn require_in(
    ops: &[PcuDispatchOp<'_>],
    position: usize,
    value: PcuDispatchValueId,
) -> Result<(), PcuF32MapValidationError> {
    if value.0 != 0
        && ops[..position]
            .iter()
            .copied()
            .any(|op| result_id(op) == Some(value))
    {
        Ok(())
    } else {
        Err(PcuF32MapValidationError::InvalidValue(value))
    }
}

fn check_binding(
    kernel: &PcuDispatchKernelIr<'_>,
    target: PcuBindingRef,
    write: bool,
) -> Result<(), PcuF32MapValidationError> {
    let Some(binding) = kernel
        .bindings
        .iter()
        .find(|binding| binding.reference() == target)
    else {
        return Err(PcuF32MapValidationError::InvalidBinding(target));
    };
    if (write && binding.access == PcuBindingAccess::ReadOnly)
        || (!write && binding.access == PcuBindingAccess::WriteOnly)
    {
        return Err(PcuF32MapValidationError::InvalidBinding(target));
    }
    Ok(())
}

fn define(
    kernel: &PcuDispatchKernelIr<'_>,
    position: usize,
    value: PcuDispatchValueId,
) -> Result<(), PcuF32MapValidationError> {
    if value.0 == 0 {
        return Err(PcuF32MapValidationError::InvalidValue(value));
    }
    if kernel.ops[..position]
        .iter()
        .copied()
        .any(|op| result_id(op) == Some(value))
    {
        return Err(PcuF32MapValidationError::DuplicateValue(value));
    }
    Ok(())
}

fn require(
    kernel: &PcuDispatchKernelIr<'_>,
    position: usize,
    value: PcuDispatchValueId,
) -> Result<(), PcuF32MapValidationError> {
    if value.0 != 0
        && kernel.ops[..position]
            .iter()
            .copied()
            .any(|op| result_id(op) == Some(value))
    {
        Ok(())
    } else {
        Err(PcuF32MapValidationError::InvalidValue(value))
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
        PcuF32MapValidationError,
        validate_f32_map_kernel,
    };
    use crate::{
        F32MapBuilder,
        PcuBinding,
        PcuBindingAccess,
        PcuBindingRef,
        PcuBindingStorageClass,
        PcuDispatchAluOp,
        PcuDispatchDataOp,
        PcuDispatchOp,
        PcuDispatchValueId,
    };

    #[test]
    fn accepts_builder_ir_and_rejects_duplicate_or_unbound_values() {
        let bindings = [
            PcuBinding::scalar::<f32>(
                Some("input"),
                0,
                0,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::ReadOnly,
            ),
            PcuBinding::scalar::<f32>(
                Some("output"),
                0,
                1,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::ReadWrite,
            ),
        ];
        let (builder, input) = F32MapBuilder::<8>::new(3, "map", [4, 1, 1], &bindings)
            .load_f32(PcuBindingRef::new(0, 0))
            .expect("load");
        let (builder, two) = builder.constant(2.0).expect("constant");
        let (builder, product) = builder.mul(input, two).expect("multiply");
        let builder = builder
            .store_f32(PcuBindingRef::new(0, 1), product)
            .expect("store");
        let kernel = builder.ir();
        assert_eq!(validate_f32_map_kernel(&kernel), Ok(()));

        let minimum_and_maximum = [
            PcuDispatchOp::Data(PcuDispatchDataOp::Constant {
                result: PcuDispatchValueId(1),
                value: crate::PcuParameterValue::F32(1.0_f32.to_bits()),
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::Constant {
                result: PcuDispatchValueId(2),
                value: crate::PcuParameterValue::F32(2.0_f32.to_bits()),
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::Alu {
                result: PcuDispatchValueId(3),
                op: PcuDispatchAluOp::Min,
                lhs: PcuDispatchValueId(1),
                rhs: PcuDispatchValueId(2),
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::Alu {
                result: PcuDispatchValueId(4),
                op: PcuDispatchAluOp::Max,
                lhs: PcuDispatchValueId(1),
                rhs: PcuDispatchValueId(2),
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                binding: PcuBindingRef::new(0, 1),
                index: crate::PcuDispatchIndex::InvocationId,
                value: PcuDispatchValueId(3),
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                binding: PcuBindingRef::new(0, 1),
                index: crate::PcuDispatchIndex::InvocationId,
                value: PcuDispatchValueId(4),
            }),
            PcuDispatchOp::Control(crate::PcuDispatchControlOp::Return),
        ];
        assert_eq!(
            validate_f32_map_kernel(&crate::PcuDispatchKernelIr {
                ops: &minimum_and_maximum,
                ..kernel
            }),
            Ok(())
        );

        let duplicate = [
            PcuDispatchOp::Data(PcuDispatchDataOp::Constant {
                result: PcuDispatchValueId(1),
                value: crate::PcuParameterValue::F32(1.0_f32.to_bits()),
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::Constant {
                result: PcuDispatchValueId(1),
                value: crate::PcuParameterValue::F32(2.0_f32.to_bits()),
            }),
        ];
        assert_eq!(
            validate_f32_map_kernel(&crate::PcuDispatchKernelIr {
                ops: &duplicate,
                ..kernel
            }),
            Err(PcuF32MapValidationError::DuplicateValue(
                PcuDispatchValueId(1)
            ))
        );
        let unbound = [PcuDispatchOp::Data(PcuDispatchDataOp::Alu {
            result: PcuDispatchValueId(2),
            op: PcuDispatchAluOp::Add,
            lhs: PcuDispatchValueId(1),
            rhs: PcuDispatchValueId(1),
        })];
        assert_eq!(
            validate_f32_map_kernel(&crate::PcuDispatchKernelIr {
                ops: &unbound,
                ..kernel
            }),
            Err(PcuF32MapValidationError::InvalidValue(PcuDispatchValueId(
                1
            )))
        );
    }
}
