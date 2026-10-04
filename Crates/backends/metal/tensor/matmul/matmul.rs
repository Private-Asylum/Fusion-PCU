//! Independently frozen matmul ownership execution over authentic session resources.
#[path = "plan/plan.rs"]
mod plan;
pub use plan::MetalTensorMatMulPlan;
#[rustfmt::skip]
use std::{
    cell::{
        RefCell,
        Ref,
    },
    rc::Rc,
};
#[rustfmt::skip]
use fusion_pcu::{
    PcuMemoryPoolId,
    PcuMemoryAccess,
    PcuMemoryResource,
};
#[rustfmt::skip]
use crate::{
    MetalSession,
    MetalError,
    MetalBuffer,
    MetalPreparedStrictMatMul,
    MetalPreparedCarrierControl,
};
#[rustfmt::skip]
use super::{
    MetalTensorInput,
    MetalTensorOwner,
};
/// Retained exact single-binary program, immutable input schema and private terminal output.
pub struct MetalPreparedTensorMatMulProgram {
    session: MetalSession,
    plan: MetalTensorMatMulPlan,
    kernel: MetalPreparedStrictMatMul,
    program: std::sync::Arc<fusion_pcu::dialect::tensor::TensorOwnedSelectedProgram>,
    carrier: Option<MetalPreparedCarrierControl>,
    pool: PcuMemoryPoolId,
}
impl MetalSession {
    /// Freezes exact integer-synthesized arithmetic and an optional checked-unused identity copy.
    /// # Errors
    /// Returns unsupported byte order, native extent/compiler or session quarantine failure.
    pub fn prepare_tensor_matmul_program(
        &self,
        program: std::sync::Arc<fusion_pcu::dialect::tensor::TensorOwnedSelectedProgram>,
        requirements: fusion_pcu::PcuImplementationRequirements,
        pool: PcuMemoryPoolId,
    ) -> Result<MetalPreparedTensorMatMulProgram, MetalError> {
        if cfg!(target_endian = "big") {
            return Err(MetalError::Unsupported);
        }
        let plan = MetalTensorMatMulPlan::assess_program(&program, requirements)
            .map_err(|_| MetalError::Unsupported)?;
        let kernel = self.prepare_strict_matmul(
            plan.scalar_type(),
            plan.dimensions(),
            requirements.float_underflow,
        )?;
        let carrier = if plan.identity_output().is_some() {
            Some(self.prepare_carrier_control(plan.scalar_type(), plan.element_count(), false)?)
        } else {
            None
        };
        Ok(MetalPreparedTensorMatMulProgram {
            session: self.clone(),
            program,
            plan,
            kernel,
            carrier,
            pool,
        })
    }
}
impl MetalPreparedTensorMatMulProgram {
    #[must_use]
    pub fn program(&self) -> &fusion_pcu::dialect::tensor::TensorOwnedSelectedProgram {
        &self.program
    }

    #[must_use]
    pub const fn plan(&self) -> &MetalTensorMatMulPlan {
        &self.plan
    }
    #[must_use]
    pub const fn session(&self) -> &MetalSession {
        &self.session
    }
    /// Executes unique host/resident inputs in frozen `input_values` order and publishes only success.
    /// # Errors
    /// Rejects every input tag/span/access/session before staging. Fatal private work leaves inputs
    /// unchanged; unknown completion retains/quarantines the actual session and resources.
    pub fn execute(&self, inputs: &[MetalTensorInput<'_>]) -> Result<MetalTensorOwner, MetalError> {
        self.session.ensure_quiescent()?;
        if inputs.len() != self.plan.input_values().len() {
            return Err(MetalError::InvalidExtent);
        }
        for (slot, &input) in inputs.iter().enumerate() {
            self.validate(input, slot)?;
        }
        let mut uploads: [Option<MetalBuffer>; 2] = [None, None];
        let mut leases: [Option<Rc<RefCell<MetalBuffer>>>; 2] = [None, None];
        for (slot, &input) in inputs.iter().enumerate() {
            match input {
                MetalTensorInput::HostBytes { bytes, .. } => {
                    uploads[slot] = Some(
                        self.session.upload_bytes(
                            &bytes[..self
                                .plan
                                .input_byte_len(slot)
                                .ok_or(MetalError::InvalidExtent)?],
                        )?,
                    );
                }
                MetalTensorInput::Resident { resource, .. } => {
                    leases[slot] = Some(resource.lease());
                }
            }
        }
        let borrowed = [borrow(leases[0].as_ref())?, borrow(leases[1].as_ref())?];
        let buffer = |slot: usize| {
            uploads[slot]
                .as_ref()
                .or(borrowed[slot].as_deref())
                .ok_or(MetalError::InvalidExtent)
        };
        let slots = self.plan.operand_inputs();
        let operands = [buffer(slots[0])?, buffer(slots[1])?];
        let output = self.kernel.execute_completed(operands[0], operands[1])?;
        let output = if let Some(slot) = self.plan.identity_output() {
            drop(output);
            self.carrier
                .as_ref()
                .ok_or(MetalError::Unsupported)?
                .execute(buffer(slot)?)?
        } else {
            output
        };
        Ok(MetalTensorOwner {
            session: self.session.clone(),
            scalar: self.plan.scalar_type(),
            shape: self.plan.shape_owner(),
            count: self.plan.element_count(),
            resource: self
                .session
                .initialized_tensor_resource(self.pool, output)?,
        })
    }
    fn validate(&self, input: MetalTensorInput<'_>, slot: usize) -> Result<(), MetalError> {
        let (scalar, elements) = match input {
            MetalTensorInput::HostBytes {
                scalar, elements, ..
            }
            | MetalTensorInput::Resident {
                scalar, elements, ..
            } => (scalar, elements),
        };
        if scalar != self.plan.scalar_type() {
            return Err(MetalError::Unsupported);
        }
        if Some(elements) != self.plan.input_element_count(slot) {
            return Err(MetalError::InvalidExtent);
        }
        match input {
            MetalTensorInput::HostBytes { bytes, .. } => {
                if bytes.len()
                    < self
                        .plan
                        .input_byte_len(slot)
                        .ok_or(MetalError::InvalidExtent)?
                {
                    return Err(MetalError::InvalidExtent);
                }
            }
            MetalTensorInput::Resident { resource, .. } => {
                resource.validate_access_available()?;
                if resource.access() == PcuMemoryAccess::WriteOnly {
                    return Err(MetalError::Unsupported);
                }
                if resource.size_bytes()
                    < u64::try_from(
                        self.plan
                            .input_byte_len(slot)
                            .ok_or(MetalError::InvalidExtent)?,
                    )
                    .map_err(|_| MetalError::InvalidExtent)?
                {
                    return Err(MetalError::InvalidExtent);
                }
                let lease = resource.lease();
                let buffer = lease.try_borrow().map_err(|_| MetalError::Unsupported)?;
                if !self.session.same_session(buffer.session()) {
                    return Err(MetalError::ForeignSession);
                }
            }
        }
        Ok(())
    }
}
fn borrow(
    lease: Option<&Rc<RefCell<MetalBuffer>>>,
) -> Result<Option<Ref<'_, MetalBuffer>>, MetalError> {
    lease
        .map(|lease| lease.try_borrow().map_err(|_| MetalError::Unsupported))
        .transpose()
}

#[cfg(test)]
#[path = "tests/tests.rs"]
mod tests;
