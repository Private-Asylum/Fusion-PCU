//! Error-only translation from provider arithmetic ordinals to Rust tensor coordinates.
//!
//! Successful replay never enters this module. Raw provider diagnostics keep
//! their ordered status ABI; ordinary Rust callers receive the same rich tensor
//! error as the reference evaluator, including discarded-but-required effects.
#[rustfmt::skip]
use crate::{
    PcuExecutionFault,
    PcuImplementationRequirements,
    PcuNumericalMode,
    PcuCheckedScalarFaultLaw,
    PcuDispatchFloatBinaryOp,
    PcuDispatchFloatUnaryOp,
    PcuDispatchIntegerBinaryOp,
    dialect::tensor::{
        NodeDescriptor,
        OpDescriptor,
        TensorError,
        TensorOwnedSelectedProgram,
        TensorStrictFaultDomain,
        ValueId,
    },
    global::PcuExecutionError,
};

pub(super) fn numerical(
    program: &TensorOwnedSelectedProgram,
    effect: ValueId,
    requirements: PcuImplementationRequirements,
    fault: PcuExecutionFault,
) -> PcuExecutionError {
    translate(program, effect, requirements, fault)
        .unwrap_or(PcuExecutionError::InvalidTensorSourcePlan)
}

fn translate(
    program: &TensorOwnedSelectedProgram,
    effect: ValueId,
    requirements: PcuImplementationRequirements,
    fault: PcuExecutionFault,
) -> Option<PcuExecutionError> {
    let graph = program.graph();
    let node = graph.node(effect).ok()?;
    // The selected output can be transport under an outer Boundary envelope while a
    // mandatory discarded helper is Strict. Fault domains belong to the actual effect,
    // exactly as in cold leaf admission; producer/caller flags cannot replace them.
    // This normalization runs only after an error, never on successful replay.
    let requirements = PcuImplementationRequirements {
        numerical_mode: node.numerical_mode.unwrap_or(requirements.numerical_mode),
        numerical_options: node.numerical_options,
        float_underflow: node
            .float_underflow_policy
            .unwrap_or(requirements.float_underflow),
        ..requirements
    };
    let elements = |shape: &[usize]| {
        shape.iter().try_fold(1_u64, |count, &dimension| {
            count.checked_mul(u64::try_from(dimension).ok()?)
        })
    };
    if let Some(law) = pointwise_law(node, requirements) {
        if fault.recovered || !law.accepts(fault, elements(node.shape)?) {
            return None;
        }
        return Some(PcuExecutionError::TensorBuild(
            TensorError::ArithmeticFault {
                value: effect,
                element_index: usize::try_from(fault.invocation_id).ok()?,
                kind: fault.kind,
            },
        ));
    }
    // IEEE 754 defines scalar exception/rounding rules. PCU separately specifies
    // the ordered multiply/add, multiply/subtract and loss-reduction sequences.
    // Never reinterpret a Boundary/library result using a Strict event domain.
    if requirements.numerical_mode != PcuNumericalMode::Strict {
        return None;
    }
    let domain = match node.op {
        OpDescriptor::MatMul {
            left,
            transpose_left,
            ..
        } => {
            let shape = graph.node(left).ok()?.shape;
            let inner = *shape.get(usize::from(!transpose_left))?;
            TensorStrictFaultDomain::matmul(
                node.scalar_type,
                elements(node.shape)?,
                u64::try_from(inner).ok()?,
                requirements.float_underflow,
            )?
        }
        OpDescriptor::SgdUpdate { .. } => TensorStrictFaultDomain::sgd(
            node.scalar_type,
            elements(node.shape)?,
            requirements.float_underflow,
        )?,
        OpDescriptor::MeanSquaredError { prediction, .. } => TensorStrictFaultDomain::mse(
            node.scalar_type,
            elements(graph.node(prediction).ok()?.shape)?,
            requirements.float_underflow,
        )?,
        _ => return None,
    };
    if !domain.accepts(fault) {
        return None;
    }
    let location = domain.location(fault.invocation_id)?;
    Some(PcuExecutionError::TensorBuild(
        TensorError::CompoundArithmeticFault {
            value: effect,
            element_index: usize::try_from(location.element_index).ok()?,
            reduction_index: usize::try_from(location.reduction_index).ok()?,
            step: location.step,
            kind: fault.kind,
        },
    ))
}

/// A graph stage still uses its operation-specific scalar law, not a family-wide whitelist.
/// In particular exact ReLU/backward cannot overflow, and IEEE Add/Sub cannot underflow.
fn pointwise_law(
    node: NodeDescriptor<'_>,
    requirements: PcuImplementationRequirements,
) -> Option<PcuCheckedScalarFaultLaw> {
    let scalar = node.scalar_type;
    let underflow = node
        .float_underflow_policy
        .unwrap_or(requirements.float_underflow);
    let range = requirements.range_policy;
    let binary = |float, integer| {
        PcuCheckedScalarFaultLaw::float_binary(scalar, float, range, underflow)
            .or_else(|| PcuCheckedScalarFaultLaw::integer_binary(scalar, integer, range))
    };
    match node.op {
        OpDescriptor::Add { .. } => binary(
            PcuDispatchFloatBinaryOp::Add,
            PcuDispatchIntegerBinaryOp::Add,
        ),
        OpDescriptor::Sub { .. } => binary(
            PcuDispatchFloatBinaryOp::Sub,
            PcuDispatchIntegerBinaryOp::Sub,
        ),
        OpDescriptor::Mul { .. } => binary(
            PcuDispatchFloatBinaryOp::Mul,
            PcuDispatchIntegerBinaryOp::Mul,
        ),
        OpDescriptor::Div { .. } => PcuCheckedScalarFaultLaw::float_binary(
            scalar,
            PcuDispatchFloatBinaryOp::Div,
            range,
            underflow,
        ),
        OpDescriptor::Relu { .. } => PcuCheckedScalarFaultLaw::float_unary(
            scalar,
            PcuDispatchFloatUnaryOp::Relu,
            range,
            underflow,
        ),
        OpDescriptor::ReluBackward { .. } => {
            PcuCheckedScalarFaultLaw::float_relu_backward(scalar, range, underflow)
        }
        _ => None,
    }
}

#[cfg(test)]
#[path = "tests/tests.rs"]
mod tests;
