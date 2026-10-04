//! Cold representative graph headers; every computed leaf still has its own contract.
#[rustfmt::skip]
use crate::{
    PcuExecutionError,
    PcuImplementationRequirements,
    dialect::tensor::OpDescriptor,
    global::{
        PcuExecutionPolicy,
        tensor::capture::PcuCapturedTensorProgram,
    },
};

pub(super) fn requirements(
    built: &PcuCapturedTensorProgram,
    options: PcuExecutionPolicy,
) -> Result<PcuImplementationRequirements, PcuExecutionError> {
    let mut request = PcuImplementationRequirements {
        numerical_mode: options.numerical_mode,
        numerical_options: options.numerical_options,
        float_underflow: options.float_underflow,
        range_policy: options.range_policy,
    };
    // Validate the whole selected closure, including mandatory discarded effects.
    // Storage transport has no arithmetic tininess contract, even for IEEE carriers.
    for &value in built.program.node_order() {
        let node = built
            .program
            .graph()
            .node(value)
            .map_err(crate::global::tensor_build_error)?;
        if !matches!(
            node.op,
            OpDescriptor::Input | OpDescriptor::Constant(_) | OpDescriptor::Uniform { .. }
        ) && node.scalar_type.binary_float_format().is_some()
            && node.float_underflow_policy.is_none()
        {
            return Err(PcuExecutionError::InvalidTensorSourcePlan);
        }
    }
    let [output] = built.program.output_values() else {
        return Err(PcuExecutionError::InvalidTensorSourcePlan);
    };
    let node = built
        .program
        .graph()
        .node(*output)
        .map_err(crate::global::tensor_build_error)?;
    // An escaped input is storage, not a new arithmetic evaluation of its producer.
    // Immutable producers retain their captured options but have no mode or tininess.
    if !matches!(node.op, OpDescriptor::Input) {
        request.numerical_mode = node.numerical_mode.unwrap_or(request.numerical_mode);
        request.numerical_options = node.numerical_options;
        request.float_underflow = node
            .float_underflow_policy
            .unwrap_or(request.float_underflow);
    }
    // This representative tuple identifies the enclosing offer/receipt. Earlier and
    // later discarded leaves require separate exact admission; it is not a blanket
    // policy for the graph. In particular, append order cannot change this header.
    Ok(request)
}

#[cfg(test)]
#[path = "tests/tests.rs"]
mod tests;
