//! Handwritten fixed workload body with explicitly shared checked arithmetic primitives.
#[rustfmt::skip]
use fusion_pcu::{
    PcuBindingRef,
    PcuDispatchDataOp as Data,
    PcuDispatchFloatBinaryOp as FloatOp,
    PcuDispatchIntegerBinaryOp as IntegerOp,
    PcuDispatchIndex,
    PcuDispatchValueId,
    PcuFloatUnderflowPolicy,
    PcuRangePolicy,
    PcuScalarType,
};
#[rustfmt::skip]
use crate::{
    MetalCheckedMapPlan,
    MetalError,
};
#[path = "body/body.rs"]
mod body;
#[derive(Clone, Copy, PartialEq, Eq)]
enum Value {
    Original,
    Seed,
    First,
    Discarded,
    Product,
    Final,
}
#[derive(Clone, Copy, Default)]
struct Policy {
    clamp: bool,
    underflow: u32,
}
/// The control recognizes only the documented saved-stage workload, not arbitrary IR.
pub fn sources(plan: &MetalCheckedMapPlan) -> Result<String, MetalError> {
    let scalar = plan.value_type().scalar_type();
    let [input, seed, stage, output] = bindings(plan)?;
    let count = plan.logical_extent();
    let mut values = Vec::new();
    let mut stored = false;
    let mut finished = false;
    let mut policies = [Policy::default(); 4];
    let mut site = 0;
    for instruction in plan.instructions() {
        let (result, value) = match *instruction {
            Data::BindingLoad {
                result,
                binding,
                index,
            } => {
                let value = if binding == input && indexed(index) {
                    Value::Original
                } else if binding == seed && index == PcuDispatchIndex::BindingElementZero {
                    Value::Seed
                } else if binding == stage && indexed(index) && stored {
                    Value::Original
                } else {
                    return Err(refused());
                };
                (result, value)
            }
            Data::BindingStore {
                binding,
                value,
                index,
            } if indexed(index) => {
                if binding == stage
                    && site == 0
                    && lookup(&values, value)? == Value::Original
                    && !stored
                {
                    stored = true;
                } else if binding == output
                    && site == 4
                    && lookup(&values, value)? == Value::Final
                    && !finished
                {
                    finished = true;
                } else {
                    return Err(refused());
                }
                continue;
            }
            Data::CheckedIntegerBinary { .. } | Data::CheckedFloatBinary { .. } => {
                let (result, lhs, rhs, rule) = arithmetic(instruction, scalar, site)?;
                policies[site] = rule;
                (result, operands(&values, lhs, rhs, site)?)
            }
            _ => return Err(refused()),
        };
        if matches!(
            value,
            Value::First | Value::Discarded | Value::Product | Value::Final
        ) {
            site += 1;
        }
        values.push((result, value));
    }
    if !stored || !finished || site != 4 {
        return Err(refused());
    }
    // Shared scalar primitives only; no PCU composed body lowering is called here.
    Ok(format!(
        "{}\n{}",
        super::shader::header(scalar)?,
        body::source(scalar, count, policies)
    ))
}
fn lookup(
    values: &[(PcuDispatchValueId, Value)],
    id: PcuDispatchValueId,
) -> Result<Value, MetalError> {
    values
        .iter()
        .find(|(candidate, _)| *candidate == id)
        .map(|(_, value)| *value)
        .ok_or_else(refused)
}
fn operands(
    values: &[(PcuDispatchValueId, Value)],
    lhs: PcuDispatchValueId,
    rhs: PcuDispatchValueId,
    site: usize,
) -> Result<Value, MetalError> {
    let expected = [
        (Value::Original, Value::Seed, Value::First),
        (Value::First, Value::Seed, Value::Discarded),
        (Value::First, Value::Original, Value::Product),
        (Value::Product, Value::Original, Value::Final),
    ];
    let Some(&(left, right, result)) = expected.get(site) else {
        return Err(refused());
    };
    if lookup(values, lhs)? != left || lookup(values, rhs)? != right {
        return Err(refused());
    }
    Ok(result)
}
const fn policy(range: PcuRangePolicy, underflow: PcuFloatUnderflowPolicy) -> Policy {
    Policy {
        clamp: matches!(range, PcuRangePolicy::Clamp),
        underflow: match underflow {
            PcuFloatUnderflowPolicy::IeeeAfterRounding => 0,
            PcuFloatUnderflowPolicy::RejectSubnormalResult => 1,
            PcuFloatUnderflowPolicy::AllowGradualUnderflow => 2,
        },
    }
}
const fn refused() -> MetalError {
    MetalError::Unsupported
}

const fn indexed(index: PcuDispatchIndex) -> bool {
    matches!(
        index,
        PcuDispatchIndex::InvocationId | PcuDispatchIndex::GridStrideId
    )
}

fn bindings(plan: &MetalCheckedMapPlan) -> Result<[PcuBindingRef; 4], MetalError> {
    let roles = plan.resources();
    let inputs: Vec<_> = roles
        .iter()
        .filter(|role| role.minimum_initial_read_elements != 0)
        .collect();
    let outputs: Vec<_> = roles
        .iter()
        .filter(|role| role.minimum_write_elements != 0)
        .collect();
    let [input, seed] = inputs.as_slice() else {
        return Err(refused());
    };
    let [stage, output] = outputs.as_slice() else {
        return Err(refused());
    };
    let count = plan.logical_extent();
    if input.minimum_initial_read_elements != count
        || seed.minimum_initial_read_elements != 1
        || stage.minimum_initial_read_elements != 0
        || output.minimum_initial_read_elements != 0
        || stage.minimum_write_elements != count
        || output.minimum_write_elements != count
        || plan.checked_effects().len() != 4
    {
        return Err(refused());
    }
    let bindings = [input.binding, seed.binding, stage.binding, output.binding];
    if roles.iter().map(|role| role.binding).ne([
        bindings[0],
        bindings[2],
        bindings[1],
        bindings[3],
    ]) {
        return Err(refused());
    }
    Ok(bindings)
}

fn arithmetic(
    instruction: &Data,
    scalar: PcuScalarType,
    site: usize,
) -> Result<
    (
        PcuDispatchValueId,
        PcuDispatchValueId,
        PcuDispatchValueId,
        Policy,
    ),
    MetalError,
> {
    let floating = matches!(scalar, PcuScalarType::F32 | PcuScalarType::F64);
    match *instruction {
        Data::CheckedIntegerBinary {
            result,
            lhs,
            rhs,
            op,
            range_policy,
            ..
        } if !floating => {
            let expected = [
                IntegerOp::Add,
                IntegerOp::Add,
                IntegerOp::Mul,
                IntegerOp::Sub,
            ];
            if expected.get(site) != Some(&op) {
                return Err(refused());
            }
            Ok((
                result,
                lhs,
                rhs,
                policy(range_policy, PcuFloatUnderflowPolicy::IeeeAfterRounding),
            ))
        }
        Data::CheckedFloatBinary {
            result,
            lhs,
            rhs,
            op,
            range_policy,
            underflow_policy,
            ..
        } if floating => {
            let expected = [FloatOp::Add, FloatOp::Div, FloatOp::Mul, FloatOp::Add];
            if expected.get(site) != Some(&op) {
                return Err(refused());
            }
            Ok((result, lhs, rhs, policy(range_policy, underflow_policy)))
        }
        _ => Err(refused()),
    }
}
