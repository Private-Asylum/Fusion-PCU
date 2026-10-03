//! Independently frozen binary ownership execution over authentic session resources.
#[path = "plan/plan.rs"]
mod plan;
pub use plan::MetalTensorBinaryPlan;
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
    PcuDispatchFloatBinaryOp,
};
#[rustfmt::skip]
use crate::{
    MetalSession,
    MetalError,
    MetalBuffer,
    MetalIntegerOp,
    MetalPreparedFloatBinary,
    MetalPreparedCarrierControl,
};
#[rustfmt::skip]
use super::{
    MetalTensorInput,
    MetalTensorOwner,
};
enum Kernel {
    Integer(crate::runtime::integer::IntegerMap),
    Float(MetalPreparedFloatBinary),
}
impl Kernel {
    fn execute(&self, inputs: [&MetalBuffer; 2], bytes: usize) -> Result<MetalBuffer, MetalError> {
        match self {
            Self::Integer(kernel) => kernel.execute(inputs),
            Self::Float(kernel) => kernel.execute_prefix(inputs, bytes),
        }
    }
}
/// Retained exact single-binary program, immutable input schema and private terminal output.
pub struct MetalPreparedTensorBinaryProgram {
    session: MetalSession,
    plan: MetalTensorBinaryPlan,
    kernel: Kernel,
    carrier: Option<MetalPreparedCarrierControl>,
    pool: PcuMemoryPoolId,
}
impl MetalSession {
    /// Freezes exact integer-synthesized arithmetic and an optional checked-unused identity copy.
    /// # Errors
    /// Returns unsupported byte order, native extent/compiler or session quarantine failure.
    pub fn prepare_tensor_binary_program(
        &self,
        plan: MetalTensorBinaryPlan,
        pool: PcuMemoryPoolId,
    ) -> Result<MetalPreparedTensorBinaryProgram, MetalError> {
        if cfg!(target_endian = "big") {
            return Err(MetalError::Unsupported);
        }
        let kernel = if plan.is_integer() {
            let op = match plan.operation() {
                PcuDispatchFloatBinaryOp::Add => MetalIntegerOp::Add,
                PcuDispatchFloatBinaryOp::Sub => MetalIntegerOp::Subtract,
                PcuDispatchFloatBinaryOp::Mul => MetalIntegerOp::Multiply,
                PcuDispatchFloatBinaryOp::Div => return Err(MetalError::Unsupported),
            };
            Kernel::Integer(self.prepare_checked_integer_map(
                plan.scalar_type(),
                op,
                plan.requirements().range_policy,
                plan.element_count(),
                [false; 2],
            )?)
        } else {
            Kernel::Float(self.prepare_checked_float_binary_with_range(
                plan.scalar_type(),
                plan.operation(),
                plan.requirements().float_underflow,
                plan.requirements().range_policy,
            )?)
        };
        let carrier = if plan.identity_output().is_some() {
            Some(self.prepare_carrier_control(plan.scalar_type(), plan.element_count(), false)?)
        } else {
            None
        };
        Ok(MetalPreparedTensorBinaryProgram {
            session: self.clone(),
            plan,
            kernel,
            carrier,
            pool,
        })
    }
}
impl MetalPreparedTensorBinaryProgram {
    #[must_use]
    pub const fn plan(&self) -> &MetalTensorBinaryPlan {
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
        for &input in inputs {
            self.validate(input)?;
        }
        let mut uploads: [Option<MetalBuffer>; 2] = [None, None];
        let mut leases: [Option<Rc<RefCell<MetalBuffer>>>; 2] = [None, None];
        for (slot, &input) in inputs.iter().enumerate() {
            match input {
                MetalTensorInput::HostBytes { bytes, .. } => {
                    uploads[slot] =
                        Some(self.session.upload_bytes(&bytes[..self.plan.byte_len()])?);
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
        let output = self.kernel.execute(operands, self.plan.byte_len())?;
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
    fn validate(&self, input: MetalTensorInput<'_>) -> Result<(), MetalError> {
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
        if elements != self.plan.element_count() {
            return Err(MetalError::InvalidExtent);
        }
        match input {
            MetalTensorInput::HostBytes { bytes, .. } => {
                if bytes.len() < self.plan.byte_len() {
                    return Err(MetalError::InvalidExtent);
                }
            }
            MetalTensorInput::Resident { resource, .. } => {
                resource.validate_access_available()?;
                if resource.access() == PcuMemoryAccess::WriteOnly {
                    return Err(MetalError::Unsupported);
                }
                if resource.size_bytes()
                    < u64::try_from(self.plan.byte_len()).map_err(|_| MetalError::InvalidExtent)?
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
#[cfg(feature = "benchmark-control")]
#[path = "control/control.rs"]
mod control;
#[cfg(test)]
#[path = "tests/tests.rs"]
mod tests;
#[cfg(feature = "benchmark-control")]
pub use control::MetalNativeTensorBinaryControl;
