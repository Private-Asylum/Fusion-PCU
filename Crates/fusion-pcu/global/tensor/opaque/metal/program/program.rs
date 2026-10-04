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
    MetalSelectedNumericalTensorPlan,
    MetalPreparedSelectedNumericalTensorProgram,
    MetalSelectedTensorGraphPlan,
    MetalPreparedSelectedTensorGraph,
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
    Numerical(MetalSelectedNumericalTensorPlan),
    Graph(MetalSelectedTensorGraphPlan),
}

impl Plan {
    pub(super) const fn scalar_type(&self) -> PcuScalarType {
        match self {
            Self::Leaf(plan) => plan.scalar_type(),
            Self::Binary(plan) => plan.scalar_type(),
            Self::Numerical(plan) => plan.scalar_type(),
            Self::Graph(plan) => plan.scalar_type(),
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
            Self::Numerical(plan) => session
                .prepare_selected_numerical_tensor_program(plan, PcuMemoryPoolId(0))
                .map(Program::Numerical),
            Self::Graph(plan) => session
                .prepare_selected_tensor_graph(plan, PcuMemoryPoolId(0))
                .map(Program::Graph),
        }
        .map_err(super::map_error)
    }
}

#[allow(clippy::large_enum_variant)]
// Retained cold cache entries are borrowed on warm calls. Boxing would add a
// separate allocation and pointer indirection to the numerical execution path.
pub(super) enum Program {
    Leaf(MetalPreparedTensorProgram),
    Binary(MetalPreparedTensorBinaryProgram),
    Numerical(MetalPreparedSelectedNumericalTensorProgram),
    Graph(MetalPreparedSelectedTensorGraph),
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
            Self::Graph(program) => {
                let plan = program.plan();
                if inputs.len() != plan.input_values().len() {
                    return Err(PcuExecutionError::InvalidTensorSourcePlan);
                }
                // Parent IDs are frozen cold; the common four-input training
                // envelope stays inline. Larger arities remain unrestricted.
                let bindings: smallvec::SmallVec<[_; 4]> = plan
                    .input_values()
                    .iter()
                    .copied()
                    .zip(inputs.iter().copied())
                    .collect();
                return program.execute_mixed(&bindings).map_err(|error| {
                    match (error.effect, error.cause) {
                        (Some(effect), fusion_pcu_metal::MetalError::Arithmetic(fault)) => {
                            super::super::fault::numerical(
                                plan.program(),
                                effect,
                                plan.requirements(),
                                fault,
                            )
                        }
                        (_, other) => super::map_error(other),
                    }
                });
            }
            Self::Numerical(program) => {
                return program.execute(inputs).map_err(|error| match error {
                    fusion_pcu_metal::MetalError::Arithmetic(fault) => {
                        let plan = program.plan();
                        super::super::fault::numerical(
                            plan.program(),
                            plan.effect(),
                            plan.requirements(),
                            fault,
                        )
                    }
                    other => super::map_error(other),
                });
            }
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
    // Use the selected output's representative tuple. A later discarded effect
    // has its own request and cannot replace the enclosing offer's identity.
    let numerical = super::super::numerical::requirements(built, options)?;
    if let Ok(plan) = MetalSelectedNumericalTensorPlan::assess_program(
        std::sync::Arc::clone(&built.program),
        numerical,
    ) {
        return Ok(Plan::Numerical(plan));
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
    let leaf = match node.op {
        OpDescriptor::Relu { .. } => {
            MetalTensorPlan::assess_relu_program(&built.program, requirements).map(Plan::Leaf)
        }
        OpDescriptor::Add { .. }
        | OpDescriptor::Sub { .. }
        | OpDescriptor::Mul { .. }
        | OpDescriptor::Div { .. } => {
            MetalTensorBinaryPlan::assess_program(&built.program, requirements).map(Plan::Binary)
        }
        _ => Err(crate::dialect::tensor::TensorUnsupportedReason::Operation),
    };
    if let Ok(plan) = leaf {
        return Ok(plan);
    }
    // Preserve qualified leaf fast paths. Only a closure they cannot represent
    // proceeds to the prepared multi-stage envelope.
    MetalSelectedTensorGraphPlan::assess_program(std::sync::Arc::clone(&built.program), numerical)
        .map(Plan::Graph)
        .map_err(|_| PcuExecutionError::InvalidTensorSourcePlan)
}
