//! Sealed cold selected numerical envelope and statically dispatched prepared execution.
use std::sync::Arc;
use fusion_pcu::{PcuScalarType, PcuImplementationRequirements};
use fusion_pcu::dialect::tensor::{TensorOwnedSelectedProgram, TensorUnsupportedReason, ValueId};
use crate::{
    MlxCheckedTensorMsePlan, MlxPreparedTensorMseProgram, MlxSession, MlxError,
    MlxCheckedTensorBackwardPlan, MlxCheckedTensorMatMulPlan, MlxCheckedTensorSgdPlan,
    MlxPreparedTensorBackwardProgram, MlxPreparedTensorMatMulProgram, MlxPreparedTensorSgdProgram,
    MlxCheckedProgramInput, MlxEncodedArray,
};
/// One individually bounded operation family; holding this enum does not widen its policies.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MlxSelectedNumericalTensorOperation {
    /// Exact finite-mask and bit-select backward effect.
    ReluBackward,
    /// Ordered checked nontransposed matrix product.
    StrictMatMul,
    /// Ordered checked update with an immutable finite F32 rate.
    StrictSgd,
    /// Ordered checked bounded loss reduction with scalar effect output.
    StrictMse,
}
#[derive(Clone)]
enum Envelope {
    Backward(MlxCheckedTensorBackwardPlan),
    MatMul(MlxCheckedTensorMatMulPlan),
    Sgd(MlxCheckedTensorSgdPlan),
    Mse(MlxCheckedTensorMsePlan),
}
/// Structural eligibility, original source ownership and exact immutable metadata.
/// This value neither compiles nor promises a successful native preparation.
#[derive(Clone)]
pub struct MlxSelectedNumericalTensorPlan {
    program: Arc<TensorOwnedSelectedProgram>,
    envelope: Envelope,
}
impl MlxSelectedNumericalTensorPlan {
    /// Assess only existing qualified individual envelopes, retaining the exact passed Arc.
    /// # Errors
    /// Refuses unsupported operations/types/shapes/policies or extra effects/declarations.
    pub fn assess_program(
        program: Arc<TensorOwnedSelectedProgram>,
        requirements: PcuImplementationRequirements,
    ) -> Result<Self, TensorUnsupportedReason> {
        if let Ok(plan) = MlxCheckedTensorBackwardPlan::assess_program(&program, requirements) {
            return Ok(Self {
                program,
                envelope: Envelope::Backward(plan),
            });
        }
        if let Ok(plan) = MlxCheckedTensorMatMulPlan::assess_program(&program, requirements) {
            return Ok(Self {
                program,
                envelope: Envelope::MatMul(plan),
            });
        }
        if let Ok(plan) = MlxCheckedTensorSgdPlan::assess_program(&program, requirements) {
            return Ok(Self {
                program,
                envelope: Envelope::Sgd(plan),
            });
        }
        if let Ok(plan) = MlxCheckedTensorMsePlan::assess_program(&program, requirements) {
            return Ok(Self {
                program,
                envelope: Envelope::Mse(plan),
            });
        }
        Err(TensorUnsupportedReason::Operation)
    }
    /// Classification within the sealed individual numerical envelopes.
    #[must_use]
    pub const fn operation(&self) -> MlxSelectedNumericalTensorOperation {
        match self.envelope {
            Envelope::Backward(_) => MlxSelectedNumericalTensorOperation::ReluBackward,
            Envelope::MatMul(_) => MlxSelectedNumericalTensorOperation::StrictMatMul,
            Envelope::Sgd(_) => MlxSelectedNumericalTensorOperation::StrictSgd,
            Envelope::Mse(_) => MlxSelectedNumericalTensorOperation::StrictMse,
        }
    }
    #[must_use]
    pub fn program(&self) -> &TensorOwnedSelectedProgram {
        &self.program
    }
    #[must_use]
    pub fn program_owner(&self) -> Arc<TensorOwnedSelectedProgram> {
        Arc::clone(&self.program)
    }
    #[must_use]
    pub fn input_values(&self) -> &[ValueId] {
        match &self.envelope {
            Envelope::Backward(plan) => plan.input_values(),
            Envelope::MatMul(plan) => plan.input_values(),
            Envelope::Sgd(plan) => plan.input_values(),
            Envelope::Mse(plan) => plan.input_values(),
        }
    }
    #[must_use]
    pub const fn operand_inputs(&self) -> [usize; 2] {
        match &self.envelope {
            Envelope::Backward(plan) => plan.operand_inputs(),
            Envelope::MatMul(plan) => plan.operand_inputs(),
            Envelope::Sgd(plan) => plan.operand_inputs(),
            Envelope::Mse(plan) => plan.operand_inputs(),
        }
    }
    #[must_use]
    pub const fn effect(&self) -> ValueId {
        match &self.envelope {
            Envelope::Backward(plan) => plan.effect(),
            Envelope::MatMul(plan) => plan.effect(),
            Envelope::Sgd(plan) => plan.effect(),
            Envelope::Mse(plan) => plan.effect(),
        }
    }
    #[must_use]
    pub const fn output(&self) -> ValueId {
        match &self.envelope {
            Envelope::Backward(plan) => plan.output(),
            Envelope::MatMul(plan) => plan.output(),
            Envelope::Sgd(plan) => plan.output(),
            Envelope::Mse(plan) => plan.output(),
        }
    }
    #[must_use]
    pub const fn scalar_type(&self) -> PcuScalarType {
        match &self.envelope {
            Envelope::Backward(plan) => plan.scalar_type(),
            Envelope::MatMul(plan) => plan.scalar_type(),
            Envelope::Sgd(plan) => plan.scalar_type(),
            Envelope::Mse(plan) => plan.scalar_type(),
        }
    }
    #[must_use]
    pub fn shape(&self) -> &[usize] {
        match &self.envelope {
            Envelope::Backward(plan) => plan.shape(),
            Envelope::MatMul(plan) => plan.shape(),
            Envelope::Sgd(plan) => plan.shape(),
            Envelope::Mse(plan) => plan.shape(),
        }
    }
    #[must_use]
    pub const fn element_count(&self) -> usize {
        match &self.envelope {
            Envelope::Backward(plan) => plan.element_count(),
            Envelope::MatMul(plan) => plan.element_count(),
            Envelope::Sgd(plan) => plan.element_count(),
            Envelope::Mse(plan) => plan.element_count(),
        }
    }
    #[must_use]
    pub const fn byte_len(&self) -> usize {
        match &self.envelope {
            Envelope::Backward(plan) => plan.byte_len(),
            Envelope::MatMul(plan) => plan.byte_len(),
            Envelope::Sgd(plan) => plan.byte_len(),
            Envelope::Mse(plan) => plan.byte_len(),
        }
    }
    #[must_use]
    pub const fn requirements(&self) -> PcuImplementationRequirements {
        match &self.envelope {
            Envelope::Backward(plan) => plan.requirements(),
            Envelope::MatMul(plan) => plan.requirements(),
            Envelope::Sgd(plan) => plan.requirements(),
            Envelope::Mse(plan) => plan.requirements(),
        }
    }
    /// Operation-family implementation receipt; source, shapes, rate and full policy
    /// remain separate immutable cache identity and are not encoded into this ID.
    #[must_use]
    pub const fn implementation_id(
        &self,
        device: fusion_pcu::PcuDeviceIdentity,
    ) -> fusion_pcu::PcuImplementationId {
        let operation = match self.operation() {
            MlxSelectedNumericalTensorOperation::ReluBackward => 0,
            MlxSelectedNumericalTensorOperation::StrictMatMul => 1,
            MlxSelectedNumericalTensorOperation::StrictSgd => 2,
            MlxSelectedNumericalTensorOperation::StrictMse => 3,
        };
        fusion_pcu::PcuImplementationId {
            device,
            executor: fusion_pcu::PcuExecutorId(0),
            local_id: 0x7000 + operation,
            revision: if operation == 0 {
                0x0000_0008_0000_0302
            } else if operation == 3 {
                0x0000_0008_0000_0301
            } else {
                0x0000_0008_0000_0300
            },
        }
    }
    /// Full logical shape of this actual input; matrix inputs may differ from output.
    #[must_use]
    pub fn input_shape(&self, slot: usize) -> Option<&[usize]> {
        let id = *self.input_values().get(slot)?;
        self.program.graph().node(id).ok().map(|node| node.shape)
    }
    #[must_use]
    pub fn input_element_count(&self, slot: usize) -> Option<usize> {
        match &self.envelope {
            Envelope::Backward(plan) => {
                (slot < plan.input_values().len()).then_some(plan.element_count())
            }
            Envelope::MatMul(plan) => plan.input_element_count(slot),
            Envelope::Sgd(plan) => plan.input_element_count(slot),
            Envelope::Mse(plan) => plan.input_element_count(slot),
        }
    }
    #[must_use]
    pub fn input_byte_len(&self, slot: usize) -> Option<usize> {
        self.input_element_count(slot)?
            .checked_mul(usize::from(self.scalar_type().bit_width()) / 8)
    }
    /// Exact selected input slot, after its mandatory checked effect; otherwise effect output.
    #[must_use]
    pub const fn selected_input(&self) -> Option<usize> {
        match &self.envelope {
            Envelope::Backward(plan) => plan.identity_output(),
            Envelope::MatMul(plan) => plan.identity_output(),
            Envelope::Sgd(plan) => plan.identity_output(),
            Envelope::Mse(plan) => plan.identity_output(),
        }
    }
    /// Rate identity remains exact, including negative zero and subnormal encodings.
    #[must_use]
    pub const fn learning_rate_bits(&self) -> Option<u32> {
        match &self.envelope {
            Envelope::Sgd(plan) => Some(plan.learning_rate().to_bits()),
            _ => None,
        }
    }
    #[must_use]
    pub const fn matmul_dimensions(&self) -> Option<[usize; 3]> {
        match &self.envelope {
            Envelope::MatMul(plan) => Some(plan.dimensions()),
            _ => None,
        }
    }
}
enum Execution {
    Backward(MlxPreparedTensorBackwardProgram),
    MatMul(MlxPreparedTensorMatMulProgram),
    Sgd(MlxPreparedTensorSgdProgram),
    Mse(MlxPreparedTensorMseProgram),
}
/// Native prepared execution and its retained original cold envelope.
pub struct MlxPreparedSelectedNumericalTensorProgram {
    plan: MlxSelectedNumericalTensorPlan,
    execution: Execution,
}
impl MlxSession {
    /// Compile one sealed cold envelope on this actual retained session.
    /// # Errors
    /// Refuses device/compiler/extent/session failure; structural eligibility is separate.
    pub fn prepare_selected_numerical_tensor_program(
        &self,
        plan: MlxSelectedNumericalTensorPlan,
    ) -> Result<MlxPreparedSelectedNumericalTensorProgram, MlxError> {
        let source = plan.program_owner();
        let requirements = plan.requirements();
        let execution = match &plan.envelope {
            Envelope::Backward(_) => {
                Execution::Backward(self.prepare_tensor_backward_program(source, requirements)?)
            }
            Envelope::MatMul(_) => {
                Execution::MatMul(self.prepare_tensor_matmul_program(source, requirements)?)
            }
            Envelope::Sgd(_) => {
                Execution::Sgd(self.prepare_tensor_sgd_program(source, requirements)?)
            }
            Envelope::Mse(_) => {
                Execution::Mse(self.prepare_tensor_mse_program(source, requirements)?)
            }
        };
        Ok(MlxPreparedSelectedNumericalTensorProgram { plan, execution })
    }
}
impl MlxPreparedSelectedNumericalTensorProgram {
    #[must_use]
    pub const fn plan(&self) -> &MlxSelectedNumericalTensorPlan {
        &self.plan
    }
    #[must_use]
    pub fn program(&self) -> &TensorOwnedSelectedProgram {
        self.plan.program()
    }
    /// Authentic operation-family receipt for this actual retained session.
    #[must_use]
    pub fn implementation_id(&self) -> Option<fusion_pcu::PcuImplementationId> {
        self.session()
            .identity()
            .map(|device| self.plan.implementation_id(device))
    }
    /// Actual retained native session of the prepared leaf.
    #[must_use]
    pub const fn session(&self) -> &MlxSession {
        match &self.execution {
            Execution::Backward(kernel) => kernel.session(),
            Execution::MatMul(kernel) => kernel.session(),
            Execution::Sgd(kernel) => kernel.session(),
            Execution::Mse(kernel) => kernel.session(),
        }
    }
    /// Execute the exact actual inputs after complete native preflight.
    /// # Errors
    /// Preserves each individual leaf completion and publication law.
    pub fn execute_mixed(
        &self,
        inputs: &[(ValueId, MlxCheckedProgramInput<'_>)],
    ) -> Result<MlxEncodedArray, MlxError> {
        match &self.execution {
            Execution::Backward(kernel) => kernel.execute_mixed(inputs),
            Execution::MatMul(kernel) => kernel.execute_mixed(inputs),
            Execution::Sgd(kernel) => kernel.execute_mixed(inputs),
            Execution::Mse(kernel) => kernel.execute_mixed(inputs),
        }
    }
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
