//! Typed, backend-neutral declarations and call-site checks for reusable PCU functions.
//!
//! Function signatures and call-graph checks are backend-neutral. Dispatch functions have an
//! allocation-free, pure constant/ALU inliner; this does not lower Rust source or promise that
//! any backend can execute an out-of-line function call.

use crate::{
    PcuDispatchDataOp,
    PcuDispatchValueId,
    PcuValueType,
};

/// Stable identifier for a function within one PCU module.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PcuFunctionId(pub u32);

/// Effect bits visible in a function signature.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct PcuFunctionEffects(u8);

impl PcuFunctionEffects {
    pub const PURE: Self = Self(0);
    pub const READ_MEMORY: Self = Self(1 << 0);
    pub const WRITE_MEMORY: Self = Self(1 << 1);
    pub const SYNCHRONIZE: Self = Self(1 << 2);
    pub const EXTERNAL: Self = Self(1 << 3);

    #[must_use]
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    #[must_use]
    pub const fn contains(self, effects: Self) -> bool {
        self.0 & effects.0 == effects.0
    }
}

/// A reusable function signature. The name and port types are borrowed for `no_std` use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PcuFunctionSignature<'a> {
    pub id: PcuFunctionId,
    pub name: &'a str,
    pub parameters: &'a [PcuValueType],
    pub results: &'a [PcuValueType],
    pub effects: PcuFunctionEffects,
}

/// A typed call reference retained in IR until a lowering pass resolves it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PcuFunctionCall<'a> {
    /// Function containing this call.
    pub caller: PcuFunctionId,
    pub callee: PcuFunctionId,
    pub arguments: &'a [PcuValueType],
    pub results: &'a [PcuValueType],
}

/// A pure, single-block dispatch function body in the first composable IR profile.
///
/// `parameters` and `results` name function-local SSA values. The initial inliner accepts
/// constants and ALU operations only; resource accesses and control flow remain explicit
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
/// to constants and ALU operations. On error, scratch and already-written output entries may
/// have changed and must be discarded. The caller must choose `first_fresh_value` outside the
/// caller's live SSA ids, then run the complete dispatch verifier on the assembled kernel; this
/// inliner checks helper-body value typing but does not replace backend-profile validation.
///
/// # Errors
/// Returns an error for a mismatched signature/body, malformed SSA values, insufficient
/// caller-provided storage, unsupported body instructions, or id exhaustion.
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
            _ => return Err(PcuFunctionInlineError::UnsupportedOperation),
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
                .filter_map(dispatch_result_id)
                .map(|id| id.0 as usize),
        )
        .max()
        .unwrap_or(0)
}

const fn dispatch_result_id(operation: &PcuDispatchDataOp) -> Option<PcuDispatchValueId> {
    match operation {
        PcuDispatchDataOp::Constant { result, .. } | PcuDispatchDataOp::Alu { result, .. } => {
            Some(*result)
        }
        _ => None,
    }
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PcuFunctionValidationError {
    DuplicateFunction(PcuFunctionId),
    DuplicateName,
    EmptyName(PcuFunctionId),
    MissingCaller(PcuFunctionId),
    WorkspaceTooSmall,
    RecursiveCall(PcuFunctionId),
    MissingCallee(PcuFunctionId),
    ArgumentCount {
        function: PcuFunctionId,
        expected: usize,
        actual: usize,
    },
    ArgumentType {
        function: PcuFunctionId,
        index: usize,
    },
    ResultCount {
        function: PcuFunctionId,
        expected: usize,
        actual: usize,
    },
    ResultType {
        function: PcuFunctionId,
        index: usize,
    },
    EffectNotAllowed {
        function: PcuFunctionId,
        required: PcuFunctionEffects,
        allowed: PcuFunctionEffects,
    },
}

/// Validate declaration uniqueness, call signatures, effects, and recursion policy.
///
/// `state` is caller-owned scratch with at least one byte per function. Recursion is
/// rejected because the execution profile does not yet define stack behavior.
///
/// # Errors
/// Returns an error for malformed declarations, incompatible calls, unsupported effects,
/// insufficient scratch space, or recursive call graphs.
pub fn validate_function_module(
    functions: &[PcuFunctionSignature<'_>],
    calls: &[PcuFunctionCall<'_>],
    state: &mut [u8],
) -> Result<(), PcuFunctionValidationError> {
    if state.len() < functions.len() {
        return Err(PcuFunctionValidationError::WorkspaceTooSmall);
    }
    for (index, function) in functions.iter().enumerate() {
        if function.name.is_empty() {
            return Err(PcuFunctionValidationError::EmptyName(function.id));
        }
        if functions[..index]
            .iter()
            .any(|prior| prior.id == function.id)
        {
            return Err(PcuFunctionValidationError::DuplicateFunction(function.id));
        }
        if functions[..index]
            .iter()
            .any(|prior| prior.name == function.name)
        {
            return Err(PcuFunctionValidationError::DuplicateName);
        }
    }
    for call in calls {
        validate_function_call(functions, call)?;
    }

    // Mark pending and visited nodes in the supplied scratch, rescanning calls per node.
    // This bounds storage to O(functions) and detects indirect cycles without allocation.
    for (origin_index, origin) in functions.iter().enumerate() {
        state[..functions.len()].fill(0);
        state[origin_index] = 1;
        while let Some(current) = state[..functions.len()].iter().position(|&mark| mark == 1) {
            state[current] = 2;
            let current_id = functions[current].id;
            for call in calls.iter().filter(|call| call.caller == current_id) {
                if call.callee == origin.id {
                    return Err(PcuFunctionValidationError::RecursiveCall(origin.id));
                }
                if let Some(next) = functions.iter().position(|item| item.id == call.callee)
                    && state[next] == 0
                {
                    state[next] = 1;
                }
            }
        }
    }
    Ok(())
}

/// Check uniqueness and resolve one call against a module's declarations.
///
/// # Errors
/// Returns an error when declarations are duplicated, the callee is missing, types or arity
/// differ, or the caller's declared effect allowance is insufficient.
pub fn validate_function_call(
    functions: &[PcuFunctionSignature<'_>],
    call: &PcuFunctionCall<'_>,
) -> Result<(), PcuFunctionValidationError> {
    for (index, function) in functions.iter().enumerate() {
        if functions[..index]
            .iter()
            .any(|prior| prior.id == function.id)
        {
            return Err(PcuFunctionValidationError::DuplicateFunction(function.id));
        }
        if functions[..index]
            .iter()
            .any(|prior| prior.name == function.name)
        {
            return Err(PcuFunctionValidationError::DuplicateName);
        }
    }

    let function = functions
        .iter()
        .find(|function| function.id == call.callee)
        .ok_or(PcuFunctionValidationError::MissingCallee(call.callee))?;
    if call.arguments.len() != function.parameters.len() {
        return Err(PcuFunctionValidationError::ArgumentCount {
            function: call.callee,
            expected: function.parameters.len(),
            actual: call.arguments.len(),
        });
    }
    for (index, (actual, expected)) in call.arguments.iter().zip(function.parameters).enumerate() {
        if actual != expected {
            return Err(PcuFunctionValidationError::ArgumentType {
                function: call.callee,
                index,
            });
        }
    }
    if call.results.len() != function.results.len() {
        return Err(PcuFunctionValidationError::ResultCount {
            function: call.callee,
            expected: function.results.len(),
            actual: call.results.len(),
        });
    }
    for (index, (actual, expected)) in call.results.iter().zip(function.results).enumerate() {
        if actual != expected {
            return Err(PcuFunctionValidationError::ResultType {
                function: call.callee,
                index,
            });
        }
    }
    let caller = functions
        .iter()
        .find(|item| item.id == call.caller)
        .ok_or(PcuFunctionValidationError::MissingCaller(call.caller))?;
    if !caller.effects.contains(function.effects) {
        return Err(PcuFunctionValidationError::EffectNotAllowed {
            function: call.callee,
            required: function.effects,
            allowed: caller.effects,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        PcuDispatchAluOp,
        PcuValueType,
    };

    #[test]
    fn call_checks_signature_and_effects_without_lowering() {
        let params = [PcuValueType::u32()];
        let results = [PcuValueType::u32()];
        let functions = [
            PcuFunctionSignature {
                id: PcuFunctionId(7),
                name: "increment",
                parameters: &params,
                results: &results,
                effects: PcuFunctionEffects::READ_MEMORY,
            },
            PcuFunctionSignature {
                id: PcuFunctionId(8),
                name: "caller",
                parameters: &params,
                results: &results,
                effects: PcuFunctionEffects::READ_MEMORY.union(PcuFunctionEffects::WRITE_MEMORY),
            },
        ];
        let args = [PcuValueType::u32()];
        let out = [PcuValueType::u32()];
        let call = PcuFunctionCall {
            caller: PcuFunctionId(8),
            callee: PcuFunctionId(7),
            arguments: &args,
            results: &out,
        };
        assert_eq!(validate_function_call(&functions, &call), Ok(()));

        let mut restricted_functions = functions;
        restricted_functions[1].effects = PcuFunctionEffects::PURE;
        assert_eq!(
            validate_function_call(&restricted_functions, &call),
            Err(PcuFunctionValidationError::EffectNotAllowed {
                function: PcuFunctionId(7),
                required: PcuFunctionEffects::READ_MEMORY,
                allowed: PcuFunctionEffects::PURE,
            })
        );
        assert_eq!(
            validate_function_call(
                &functions,
                &PcuFunctionCall {
                    caller: PcuFunctionId(99),
                    ..call
                }
            ),
            Err(PcuFunctionValidationError::MissingCaller(PcuFunctionId(99)))
        );
        let wrong_args = [PcuValueType::i32()];
        let mistyped = PcuFunctionCall {
            arguments: &wrong_args,
            ..call
        };
        assert_eq!(
            validate_function_call(&functions, &mistyped),
            Err(PcuFunctionValidationError::ArgumentType {
                function: PcuFunctionId(7),
                index: 0
            })
        );
    }

    #[test]
    fn module_verifier_rejects_indirect_recursion() {
        let empty: [PcuValueType; 0] = [];
        let functions = [
            PcuFunctionSignature {
                id: PcuFunctionId(1),
                name: "a",
                parameters: &empty,
                results: &empty,
                effects: PcuFunctionEffects::PURE,
            },
            PcuFunctionSignature {
                id: PcuFunctionId(2),
                name: "b",
                parameters: &empty,
                results: &empty,
                effects: PcuFunctionEffects::PURE,
            },
        ];
        let calls = [
            PcuFunctionCall {
                caller: PcuFunctionId(1),
                callee: PcuFunctionId(2),
                arguments: &empty,
                results: &empty,
            },
            PcuFunctionCall {
                caller: PcuFunctionId(2),
                callee: PcuFunctionId(1),
                arguments: &empty,
                results: &empty,
            },
        ];
        let mut scratch = [0; 2];
        assert_eq!(
            validate_function_module(&functions, &calls, &mut scratch),
            Err(PcuFunctionValidationError::RecursiveCall(PcuFunctionId(1)))
        );
    }

    #[test]
    fn pure_dispatch_function_inlines_and_remaps_local_ssa_values() {
        let parameters = [PcuDispatchValueId(1), PcuDispatchValueId(2)];
        let results = [PcuDispatchValueId(3)];
        let body_ops = [PcuDispatchDataOp::Alu {
            value_type: PcuValueType::f32(),
            result: PcuDispatchValueId(3),
            op: PcuDispatchAluOp::Add,
            lhs: PcuDispatchValueId(1),
            rhs: PcuDispatchValueId(2),
        }];
        let body = PcuDispatchFunctionBody {
            function: PcuFunctionId(7),
            parameters: &parameters,
            operations: &body_ops,
            results: &results,
        };
        let types = [PcuValueType::f32(), PcuValueType::f32()];
        let result_types = [PcuValueType::f32()];
        let functions = [
            PcuFunctionSignature {
                id: PcuFunctionId(7),
                name: "sum",
                parameters: &types,
                results: &result_types,
                effects: PcuFunctionEffects::PURE,
            },
            PcuFunctionSignature {
                id: PcuFunctionId(8),
                name: "caller",
                parameters: &[],
                results: &[],
                effects: PcuFunctionEffects::PURE,
            },
        ];
        let argument_types = [PcuValueType::f32(), PcuValueType::f32()];
        let call_result_types = [PcuValueType::f32()];
        let mut call_results = [PcuDispatchValueId(0)];
        let mut operands = PcuDispatchFunctionOperands {
            signature: PcuFunctionCall {
                caller: PcuFunctionId(8),
                callee: PcuFunctionId(7),
                arguments: &argument_types,
                results: &call_result_types,
            },
            arguments: &[PcuDispatchValueId(10), PcuDispatchValueId(11)],
            results: &mut call_results,
        };
        let mut map = [PcuDispatchValueId(0); 4];
        let mut type_map = [None; 4];
        let mut output = [body_ops[0]];
        let len = inline_pcu_dispatch_function(
            &functions,
            &body,
            &mut operands,
            100,
            &mut map,
            &mut type_map,
            &mut output,
        )
        .expect("well-typed pure helper inlines");
        assert_eq!(len, 1);
        assert_eq!(
            output[0],
            PcuDispatchDataOp::Alu {
                value_type: PcuValueType::f32(),
                result: PcuDispatchValueId(100),
                op: PcuDispatchAluOp::Add,
                lhs: PcuDispatchValueId(10),
                rhs: PcuDispatchValueId(11),
            }
        );
        assert_eq!(call_results, [PcuDispatchValueId(100)]);
    }

    #[test]
    fn pure_dispatch_inliner_rejects_resource_effects() {
        let params = [PcuDispatchValueId(1)];
        let results = [PcuDispatchValueId(2)];
        let ops = [PcuDispatchDataOp::BindingLoad {
            result: PcuDispatchValueId(2),
            binding: crate::PcuBindingRef::new(0, 0),
            index: crate::PcuDispatchIndex::BindingElementZero,
        }];
        let body = PcuDispatchFunctionBody {
            function: PcuFunctionId(1),
            parameters: &params,
            operations: &ops,
            results: &results,
        };
        let types = [PcuValueType::f32()];
        let functions = [
            PcuFunctionSignature {
                id: PcuFunctionId(1),
                name: "bad",
                parameters: &types,
                results: &types,
                effects: PcuFunctionEffects::PURE,
            },
            PcuFunctionSignature {
                id: PcuFunctionId(2),
                name: "caller",
                parameters: &[],
                results: &[],
                effects: PcuFunctionEffects::PURE,
            },
        ];
        let mut result_ids = [PcuDispatchValueId(0)];
        let mut operands = PcuDispatchFunctionOperands {
            signature: PcuFunctionCall {
                caller: PcuFunctionId(2),
                callee: PcuFunctionId(1),
                arguments: &types,
                results: &types,
            },
            arguments: &[PcuDispatchValueId(9)],
            results: &mut result_ids,
        };
        let mut map = [PcuDispatchValueId(0); 3];
        let mut type_map = [None; 3];
        let mut output = [ops[0]];
        assert_eq!(
            inline_pcu_dispatch_function(
                &functions,
                &body,
                &mut operands,
                20,
                &mut map,
                &mut type_map,
                &mut output
            ),
            Err(PcuFunctionInlineError::UnsupportedOperation),
        );
    }

    #[test]
    fn pure_dispatch_inliner_rejects_body_type_mismatch() {
        let parameters = [PcuDispatchValueId(1), PcuDispatchValueId(2)];
        let results = [PcuDispatchValueId(3)];
        let operations = [PcuDispatchDataOp::Alu {
            value_type: PcuValueType::i32(),
            result: PcuDispatchValueId(3),
            op: PcuDispatchAluOp::Add,
            lhs: PcuDispatchValueId(1),
            rhs: PcuDispatchValueId(2),
        }];
        let body = PcuDispatchFunctionBody {
            function: PcuFunctionId(3),
            parameters: &parameters,
            operations: &operations,
            results: &results,
        };
        let parameter_types = [PcuValueType::f32(), PcuValueType::f32()];
        let result_types = [PcuValueType::f32()];
        let functions = [
            PcuFunctionSignature {
                id: PcuFunctionId(3),
                name: "bad_add",
                parameters: &parameter_types,
                results: &result_types,
                effects: PcuFunctionEffects::PURE,
            },
            PcuFunctionSignature {
                id: PcuFunctionId(4),
                name: "caller",
                parameters: &[],
                results: &[],
                effects: PcuFunctionEffects::PURE,
            },
        ];
        let mut call_results = [PcuDispatchValueId(0)];
        let mut operands = PcuDispatchFunctionOperands {
            signature: PcuFunctionCall {
                caller: PcuFunctionId(4),
                callee: PcuFunctionId(3),
                arguments: &parameter_types,
                results: &result_types,
            },
            arguments: &[PcuDispatchValueId(10), PcuDispatchValueId(11)],
            results: &mut call_results,
        };
        let mut map = [PcuDispatchValueId(0); 4];
        let mut type_map = [None; 4];
        let mut output = [operations[0]];
        assert_eq!(
            inline_pcu_dispatch_function(
                &functions,
                &body,
                &mut operands,
                100,
                &mut map,
                &mut type_map,
                &mut output,
            ),
            Err(PcuFunctionInlineError::TypeMismatch(PcuDispatchValueId(1))),
        );
    }

    #[test]
    fn pure_dispatch_inliner_rejects_zero_and_overflowing_fresh_ids() {
        let parameters = [PcuDispatchValueId(1), PcuDispatchValueId(2)];
        let results = [PcuDispatchValueId(3)];
        let operations = [PcuDispatchDataOp::Alu {
            value_type: PcuValueType::f32(),
            result: PcuDispatchValueId(3),
            op: PcuDispatchAluOp::Add,
            lhs: PcuDispatchValueId(1),
            rhs: PcuDispatchValueId(2),
        }];
        let body = PcuDispatchFunctionBody {
            function: PcuFunctionId(3),
            parameters: &parameters,
            operations: &operations,
            results: &results,
        };
        let parameter_types = [PcuValueType::f32(), PcuValueType::f32()];
        let result_types = [PcuValueType::f32()];
        let functions = [
            PcuFunctionSignature {
                id: PcuFunctionId(3),
                name: "sum",
                parameters: &parameter_types,
                results: &result_types,
                effects: PcuFunctionEffects::PURE,
            },
            PcuFunctionSignature {
                id: PcuFunctionId(4),
                name: "caller",
                parameters: &[],
                results: &[],
                effects: PcuFunctionEffects::PURE,
            },
        ];
        let mut call_results = [PcuDispatchValueId(0)];
        let mut operands = PcuDispatchFunctionOperands {
            signature: PcuFunctionCall {
                caller: PcuFunctionId(4),
                callee: PcuFunctionId(3),
                arguments: &parameter_types,
                results: &result_types,
            },
            arguments: &[PcuDispatchValueId(10), PcuDispatchValueId(11)],
            results: &mut call_results,
        };
        let mut map = [PcuDispatchValueId(0); 4];
        let mut type_map = [None; 4];
        let mut output = [operations[0]];
        for (first, expected) in [
            (0, PcuFunctionInlineError::ValueIdOverflow),
            (u16::MAX, PcuFunctionInlineError::ValueIdOverflow),
        ] {
            assert_eq!(
                inline_pcu_dispatch_function(
                    &functions,
                    &body,
                    &mut operands,
                    first,
                    &mut map,
                    &mut type_map,
                    &mut output,
                ),
                Err(expected),
            );
        }
    }
}
