//! Cold concrete admission and retained static execution; no warm graph traversal.
#[rustfmt::skip]
use fusion_pcu_metal::{
    MetalPreparedTensorBinaryProgram,
    MetalPreparedTensorProgram,
    MetalSession,
    MetalTensorBinaryPlan,
    MetalTensorInput,
    MetalTensorOwner,
    MetalTensorPlan,
};
#[rustfmt::skip]
use crate::{
    PcuExecutionError,
    PcuImplementationRequirements,
    PcuMemoryPoolId,
    PcuScalarType,
    dialect::tensor::OpDescriptor,
    global::{
        PcuExecutionPolicy,
        tensor::capture::PcuCapturedTensorProgram,
    },
};

pub(super) enum Plan {
    Leaf(MetalTensorPlan),
    Binary(MetalTensorBinaryPlan),
}

impl Plan {
    pub(super) const fn scalar_type(&self) -> PcuScalarType {
        match self {
            Self::Leaf(plan) => plan.scalar_type(),
            Self::Binary(plan) => plan.scalar_type(),
        }
    }

    pub(super) fn prepare(self, session: &MetalSession) -> Result<Program, PcuExecutionError> {
        match self {
            Self::Leaf(plan) => session
                .prepare_tensor_program(plan, PcuMemoryPoolId(0))
                .map(Program::Leaf),
            Self::Binary(plan) => session
                .prepare_tensor_binary_program(plan, PcuMemoryPoolId(0))
                .map(Program::Binary),
        }
        .map_err(super::map_error)
    }
}

pub(super) enum Program {
    Leaf(MetalPreparedTensorProgram),
    Binary(MetalPreparedTensorBinaryProgram),
}

impl Program {
    pub(super) fn execute(
        &self,
        inputs: &[MetalTensorInput<'_>],
    ) -> Result<MetalTensorOwner, PcuExecutionError> {
        match self {
            Self::Leaf(program) => {
                let [input] = inputs else {
                    return Err(PcuExecutionError::InvalidTensorSourcePlan);
                };
                program.execute(*input)
            }
            Self::Binary(program) => program.execute(inputs),
        }
        .map_err(super::map_error)
    }
}

pub(super) fn assess(
    built: &PcuCapturedTensorProgram,
    options: PcuExecutionPolicy,
) -> Result<Plan, PcuExecutionError> {
    let mut requirements = PcuImplementationRequirements {
        numerical_mode: options.numerical_mode,
        numerical_options: options.numerical_options,
        float_underflow: options.float_underflow,
        range_policy: options.range_policy,
    };
    if let Ok(plan) = MetalTensorPlan::assess_program(&built.program, requirements) {
        return Ok(Plan::Leaf(plan));
    }
    // A detached single effect owns its local tuple. Input transport metadata
    // cannot authorize arithmetic; each backend assessor validates the whole closure.
    let effect = built
        .program
        .node_order()
        .last()
        .copied()
        .ok_or(PcuExecutionError::InvalidTensorSourcePlan)?;
    let node = built
        .program
        .graph()
        .node(effect)
        .map_err(crate::global::tensor_build_error)?;
    requirements.numerical_mode = node.numerical_mode.unwrap_or(options.numerical_mode);
    requirements.numerical_options = node.numerical_options;
    requirements.float_underflow = node
        .float_underflow_policy
        .unwrap_or(options.float_underflow);
    match node.op {
        OpDescriptor::Relu { .. } => {
            MetalTensorPlan::assess_relu_program(&built.program, requirements).map(Plan::Leaf)
        }
        OpDescriptor::Add { .. }
        | OpDescriptor::Sub { .. }
        | OpDescriptor::Mul { .. }
        | OpDescriptor::Div { .. } => {
            MetalTensorBinaryPlan::assess_program(&built.program, requirements).map(Plan::Binary)
        }
        _ => return Err(PcuExecutionError::InvalidTensorSourcePlan),
    }
    .map_err(|_| PcuExecutionError::InvalidTensorSourcePlan)
}
