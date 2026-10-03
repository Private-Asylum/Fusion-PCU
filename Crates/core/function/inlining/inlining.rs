//! Allocation-free scalar helper expansion into caller-owned SSA storage.

#[rustfmt::skip]
use super::{
    PcuFunctionCall,
    PcuFunctionId,
    PcuFunctionSignature,
    PcuFunctionValidationError,
    validate_function_call,
};
#[rustfmt::skip]
use crate::{
    PcuDispatchDataOp,
    PcuDispatchValueId,
    PcuValueType,
};

/// A pure, single-block dispatch function body in the first composable IR profile.
///
/// `parameters` and `results` name function-local SSA values. The initial inliner accepts
/// constants, ALU operations, specified conversions and checked scalar arithmetic;
/// instruction-local numerical policies are retained unchanged. Resource accesses and control flow remain explicit
/// future extensions rather than silently acquiring semantics here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PcuDispatchFunctionBody<'a> {
    pub function: PcuFunctionId,
    pub parameters: &'a [PcuDispatchValueId],
    pub operations: &'a [PcuDispatchDataOp],
    pub results: &'a [PcuDispatchValueId],
}

/// Operand values for one typed dispatch-function call.
#[derive(Debug, PartialEq, Eq)]
pub struct PcuDispatchFunctionOperands<'a> {
    pub signature: PcuFunctionCall<'a>,
    pub arguments: &'a [PcuDispatchValueId],
    pub results: &'a mut [PcuDispatchValueId],
}

/// Bounded failure cases for the pure dispatch-function inliner.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PcuFunctionInlineError {
    InvalidSignature(PcuFunctionValidationError),
    BodyFunctionMismatch,
    ParameterCount,
    ResultCount,
    ValueMapTooSmall,
    OutputTooSmall,
    UnsupportedOperation,
    UndefinedValue(PcuDispatchValueId),
    InvalidArgumentValue(PcuDispatchValueId),
    TypeMismatch(PcuDispatchValueId),
    DuplicateDefinition(PcuDispatchValueId),
    ValueIdOverflow,
}

/// Inline one pure dispatch function into caller-owned operation and value-map storage.
///
/// The body uses local value ids; this function remaps parameters to the caller's argument
/// ids and assigns fresh caller ids to body definitions. `value_map` is indexed by a local id,
/// and its zero slot is reserved. `type_map` follows the same local-id layout. `output`
/// receives the expanded operations. The function is allocation-free and deliberately limited
/// to pure scalar instructions. It preserves each checked instruction's policies and both
/// results of joint division; it does not evaluate arithmetic or erase possible faults.
/// On error, scratch and already-written output entries may
/// have changed and must be discarded. The caller must choose `first_fresh_value` outside the
/// caller's live SSA ids, then run the complete dispatch verifier on the assembled kernel; this
/// inliner checks helper-body value typing but does not replace backend-profile validation.
///
/// # Errors
/// Returns an error for a mismatched signature/body, malformed SSA values, insufficient
/// caller-provided storage, unsupported body instructions, or id exhaustion.
#[allow(clippy::too_many_lines)] // Keeps bounded SSA remapping and validation in one auditable pass.
pub fn inline_pcu_dispatch_function(
    functions: &[PcuFunctionSignature<'_>],
    body: &PcuDispatchFunctionBody<'_>,
    operands: &mut PcuDispatchFunctionOperands<'_>,
    first_fresh_value: u16,
    value_map: &mut [PcuDispatchValueId],
    type_map: &mut [Option<PcuValueType>],
    output: &mut [PcuDispatchDataOp],
) -> Result<usize, PcuFunctionInlineError> {
    validate_function_call(functions, &operands.signature)
        .map_err(PcuFunctionInlineError::InvalidSignature)?;
    if body.function != operands.signature.callee {
        return Err(PcuFunctionInlineError::BodyFunctionMismatch);
    }
    if body.parameters.len() != operands.arguments.len() {
        return Err(PcuFunctionInlineError::ParameterCount);
    }
    let callee = functions
        .iter()
        .find(|function| function.id == body.function)
        .ok_or(PcuFunctionInlineError::BodyFunctionMismatch)?;
    if body.parameters.len() != callee.parameters.len() {
        return Err(PcuFunctionInlineError::ParameterCount);
    }
    if body.results.len() != operands.results.len() {
        return Err(PcuFunctionInlineError::ResultCount);
    }
    if body.results.len() != callee.results.len() {
        return Err(PcuFunctionInlineError::ResultCount);
    }
    if output.len() < body.operations.len() {
        return Err(PcuFunctionInlineError::OutputTooSmall);
    }
    let max_local = max_body_local_id(body);
    if value_map.len() <= max_local || type_map.len() <= max_local {
        return Err(PcuFunctionInlineError::ValueMapTooSmall);
    }
    value_map.fill(PcuDispatchValueId(0));
    type_map.fill(None);
    for (index, (local, actual)) in body
        .parameters
        .iter()
        .zip(operands.arguments.iter())
        .enumerate()
    {
        if local.0 == 0 {
            return Err(PcuFunctionInlineError::DuplicateDefinition(*local));
        }
        if actual.0 == 0 {
            return Err(PcuFunctionInlineError::InvalidArgumentValue(*actual));
        }
        let slot = &mut value_map[local.0 as usize];
        if slot.0 != 0 {
            return Err(PcuFunctionInlineError::DuplicateDefinition(*local));
        }
        *slot = *actual;
        type_map[local.0 as usize] = Some(callee.parameters[index]);
    }

    let mut next = first_fresh_value;
    for (index, operation) in body.operations.iter().copied().enumerate() {
        let remapped = match operation {
            PcuDispatchDataOp::Constant { result, value } => {
                let mapped = define_inline_value(result, &mut next, value_map)?;
                type_map[result.0 as usize] = Some(value.value_type());
                PcuDispatchDataOp::Constant {
                    result: mapped,
                    value,
                }
            }
            PcuDispatchDataOp::Alu {
                value_type,
                result: local_result,
                op,
                lhs,
                rhs,
            } => {
                if type_map.get(lhs.0 as usize).copied().flatten() != Some(value_type) {
                    return Err(PcuFunctionInlineError::TypeMismatch(lhs));
                }
                if type_map.get(rhs.0 as usize).copied().flatten() != Some(value_type) {
                    return Err(PcuFunctionInlineError::TypeMismatch(rhs));
                }
                let lhs = remap_inline_value(lhs, value_map)?;
                let rhs = remap_inline_value(rhs, value_map)?;
                let result = define_inline_value(local_result, &mut next, value_map)?;
                type_map[local_result.0 as usize] = Some(value_type);
                PcuDispatchDataOp::Alu {
                    value_type,
                    result,
                    op,
                    lhs,
                    rhs,
                }
            }
            PcuDispatchDataOp::CheckedFloatUnary {
                value_type,
                op,
                underflow_policy,
                range_policy,
                result: local_result,
                value,
            } => {
                if type_map.get(value.0 as usize).copied().flatten() != Some(value_type) {
                    return Err(PcuFunctionInlineError::TypeMismatch(value));
                }
                let value = remap_inline_value(value, value_map)?;
                let result = define_inline_value(local_result, &mut next, value_map)?;
                type_map[local_result.0 as usize] = Some(value_type);
                PcuDispatchDataOp::CheckedFloatUnary {
                    value_type,
                    op,
                    underflow_policy,
                    range_policy,
                    result,
                    value,
                }
            }
            PcuDispatchDataOp::Convert {
                result: local_result,
                value,
                conversion,
            } => {
                let value =
                    remap_typed_inline_value(value, conversion.source_type(), value_map, type_map)?;
                let result = define_typed_inline_value(
                    local_result,
                    conversion.target_type(),
                    &mut next,
                    value_map,
                    type_map,
                )?;
                PcuDispatchDataOp::Convert {
                    result,
                    value,
                    conversion,
                }
            }
            PcuDispatchDataOp::CheckedFloatConvert {
                result: local_result,
                value,
                conversion,
                underflow_policy,
                range_policy,
            } => {
                let value =
                    remap_typed_inline_value(value, conversion.source_type(), value_map, type_map)?;
                let result = define_typed_inline_value(
                    local_result,
                    conversion.target_type(),
                    &mut next,
                    value_map,
                    type_map,
                )?;
                PcuDispatchDataOp::CheckedFloatConvert {
                    result,
                    value,
                    conversion,
                    underflow_policy,
                    range_policy,
                }
            }
            PcuDispatchDataOp::CheckedFloatBinary {
                value_type,
                op,
                underflow_policy,
                range_policy,
                result: local_result,
                lhs,
                rhs,
            } => {
                let lhs = remap_typed_inline_value(lhs, value_type, value_map, type_map)?;
                let rhs = remap_typed_inline_value(rhs, value_type, value_map, type_map)?;
                let result = define_typed_inline_value(
                    local_result,
                    value_type,
                    &mut next,
                    value_map,
                    type_map,
                )?;
                PcuDispatchDataOp::CheckedFloatBinary {
                    value_type,
                    op,
                    underflow_policy,
                    range_policy,
                    result,
                    lhs,
                    rhs,
                }
            }
            PcuDispatchDataOp::CheckedIntegerBinary {
                value_type,
                op,
                range_policy,
                result: local_result,
                lhs,
                rhs,
            } => {
                let lhs = remap_typed_inline_value(lhs, value_type, value_map, type_map)?;
                let rhs = remap_typed_inline_value(rhs, value_type, value_map, type_map)?;
                let result = define_typed_inline_value(
                    local_result,
                    value_type,
                    &mut next,
                    value_map,
                    type_map,
                )?;
                PcuDispatchDataOp::CheckedIntegerBinary {
                    value_type,
                    op,
                    range_policy,
                    result,
                    lhs,
                    rhs,
                }
            }
            PcuDispatchDataOp::CheckedDivRem {
                value_type,
                flags,
                quotient: local_quotient,
                remainder: local_remainder,
                lhs,
                rhs,
            } => {
                // The reserved quotient-only override has no joint-division semantics.
                if flags != crate::model::PcuIntegerDivFlags::CHECKED {
                    return Err(PcuFunctionInlineError::UnsupportedOperation);
                }
                let lhs = remap_typed_inline_value(lhs, value_type, value_map, type_map)?;
                let rhs = remap_typed_inline_value(rhs, value_type, value_map, type_map)?;
                let quotient = define_typed_inline_value(
                    local_quotient,
                    value_type,
                    &mut next,
                    value_map,
                    type_map,
                )?;
                let remainder = define_typed_inline_value(
                    local_remainder,
                    value_type,
                    &mut next,
                    value_map,
                    type_map,
                )?;
                PcuDispatchDataOp::CheckedDivRem {
                    value_type,
                    flags,
                    quotient,
                    remainder,
                    lhs,
                    rhs,
                }
            }
            PcuDispatchDataOp::BindingLoad { .. } | PcuDispatchDataOp::BindingStore { .. } => {
                return Err(PcuFunctionInlineError::UnsupportedOperation);
            }
        };
        output[index] = remapped;
    }
    for (index, local) in body.results.iter().copied().enumerate() {
        if type_map.get(local.0 as usize).copied().flatten() != Some(callee.results[index]) {
            return Err(PcuFunctionInlineError::TypeMismatch(local));
        }
        operands.results[index] = remap_inline_value(local, value_map)?;
    }
    Ok(body.operations.len())
}

fn max_body_local_id(body: &PcuDispatchFunctionBody<'_>) -> usize {
    body.parameters
        .iter()
        .chain(body.results)
        .map(|id| id.0 as usize)
        .chain(
            body.operations
                .iter()
                .flat_map(dispatch_result_ids)
                .flatten()
                .map(|id| id.0 as usize),
        )
        .max()
        .unwrap_or(0)
}

const fn dispatch_result_ids(operation: &PcuDispatchDataOp) -> [Option<PcuDispatchValueId>; 2] {
    match operation {
        PcuDispatchDataOp::Constant { result, .. }
        | PcuDispatchDataOp::Alu { result, .. }
        | PcuDispatchDataOp::Convert { result, .. }
        | PcuDispatchDataOp::CheckedIntegerBinary { result, .. }
        | PcuDispatchDataOp::CheckedFloatBinary { result, .. }
        | PcuDispatchDataOp::CheckedFloatUnary { result, .. }
        | PcuDispatchDataOp::CheckedFloatConvert { result, .. } => [Some(*result), None],
        PcuDispatchDataOp::CheckedDivRem {
            quotient,
            remainder,
            ..
        } => [Some(*quotient), Some(*remainder)],
        PcuDispatchDataOp::BindingLoad { .. } | PcuDispatchDataOp::BindingStore { .. } => {
            [None, None]
        }
    }
}

fn remap_typed_inline_value(
    local: PcuDispatchValueId,
    value_type: PcuValueType,
    value_map: &[PcuDispatchValueId],
    type_map: &[Option<PcuValueType>],
) -> Result<PcuDispatchValueId, PcuFunctionInlineError> {
    if type_map.get(local.0 as usize).copied().flatten() != Some(value_type) {
        return Err(PcuFunctionInlineError::TypeMismatch(local));
    }
    remap_inline_value(local, value_map)
}

fn define_typed_inline_value(
    local: PcuDispatchValueId,
    value_type: PcuValueType,
    next: &mut u16,
    value_map: &mut [PcuDispatchValueId],
    type_map: &mut [Option<PcuValueType>],
) -> Result<PcuDispatchValueId, PcuFunctionInlineError> {
    let mapped = define_inline_value(local, next, value_map)?;
    type_map[local.0 as usize] = Some(value_type);
    Ok(mapped)
}

fn remap_inline_value(
    local: PcuDispatchValueId,
    map: &[PcuDispatchValueId],
) -> Result<PcuDispatchValueId, PcuFunctionInlineError> {
    map.get(local.0 as usize)
        .copied()
        .filter(|value| value.0 != 0)
        .ok_or(PcuFunctionInlineError::UndefinedValue(local))
}

fn define_inline_value(
    local: PcuDispatchValueId,
    next: &mut u16,
    map: &mut [PcuDispatchValueId],
) -> Result<PcuDispatchValueId, PcuFunctionInlineError> {
    let slot = map
        .get_mut(local.0 as usize)
        .ok_or(PcuFunctionInlineError::ValueMapTooSmall)?;
    if local.0 == 0 || slot.0 != 0 {
        return Err(PcuFunctionInlineError::DuplicateDefinition(local));
    }
    if *next == 0 {
        return Err(PcuFunctionInlineError::ValueIdOverflow);
    }
    let result = PcuDispatchValueId(*next);
    *next = next
        .checked_add(1)
        .ok_or(PcuFunctionInlineError::ValueIdOverflow)?;
    *slot = result;
    Ok(result)
}

#[cfg(test)]
#[path = "tests/tests.rs"]
mod tests;
