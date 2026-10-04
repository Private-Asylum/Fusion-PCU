//! Authentic bounded graph ownership over retained exact-session initialized Metal storage.
#[path = "plan/plan.rs"]
mod plan;
pub use plan::MetalTensorPlan;
#[path = "binary/binary.rs"]
mod binary;
#[rustfmt::skip]
pub use binary::{
    MetalTensorBinaryPlan,
    MetalPreparedTensorBinaryProgram,
};
#[rustfmt::skip]
use std::rc::Rc;
#[rustfmt::skip]
use fusion_pcu::{PcuScalar,PcuScalarType,PcuMemoryPoolId,PcuMemoryResource,PcuMemoryAccess,PcuHostArgument,PcuBindingRef};
#[rustfmt::skip]
use crate::{MetalSession,MetalPreparedCarrierControl,MetalPreparedFloatUnary,MetalBuffer,MetalMemoryResource,MetalError};
/// One borrowed host byte span or real typed same-session resident allocation.
#[derive(Clone, Copy)]
pub enum MetalTensorInput<'a> {
    HostBytes {
        scalar: PcuScalarType,
        elements: usize,
        bytes: &'a [u8],
    },
    Resident {
        scalar: PcuScalarType,
        elements: usize,
        resource: &'a MetalMemoryResource,
    },
}
/// Fresh initialized resident owner retaining logical type, immutable shape and actual session.
pub struct MetalTensorOwner {
    session: MetalSession,
    scalar: PcuScalarType,
    shape: Rc<[usize]>,
    count: usize,
    resource: MetalMemoryResource,
}
/// Frozen Input or checked `ReLU` executable publishing a fresh terminal owner on success.
pub struct MetalPreparedTensorProgram {
    session: MetalSession,
    plan: MetalTensorPlan,
    carrier: MetalPreparedCarrierControl,
    relu: Option<MetalPreparedFloatUnary>,
    pool: PcuMemoryPoolId,
}
impl MetalSession {
    /// Freezes a previously assessed bounded graph against this actual retained device session.
    /// # Errors
    /// Rejects unsupported byte order, actual buffer extent or native compiler/quarantine error.
    pub fn prepare_tensor_program(
        &self,
        plan: MetalTensorPlan,
        pool: PcuMemoryPoolId,
    ) -> Result<MetalPreparedTensorProgram, MetalError> {
        if cfg!(target_endian = "big") {
            return Err(MetalError::Unsupported);
        }
        let carrier =
            self.prepare_carrier_control(plan.scalar_type(), plan.element_count(), false)?;
        let relu = if plan.relu_effect().is_some() {
            Some(self.prepare_checked_float_unary_with_range(
                plan.scalar_type(),
                fusion_pcu::PcuDispatchFloatUnaryOp::Relu,
                plan.requirements().float_underflow,
                plan.requirements().range_policy,
            )?)
        } else {
            None
        };
        Ok(MetalPreparedTensorProgram {
            session: self.clone(),
            plan,
            carrier,
            relu,
            pool,
        })
    }
}
impl MetalPreparedTensorProgram {
    #[must_use]
    pub const fn plan(&self) -> &MetalTensorPlan {
        &self.plan
    }
    #[must_use]
    pub const fn session(&self) -> &MetalSession {
        &self.session
    }
    /// Executes the frozen logical profile; the initialized owner escapes only on success.
    /// # Errors
    /// Rejects mismatched logical type/extent/access/session before work and returns native errors.
    /// No result owner escapes on failure; unknown completion quarantines the actual session.
    pub fn execute(&self, input: MetalTensorInput<'_>) -> Result<MetalTensorOwner, MetalError> {
        self.session.ensure_quiescent()?;
        let (scalar, elements) = match &input {
            MetalTensorInput::HostBytes {
                scalar, elements, ..
            }
            | MetalTensorInput::Resident {
                scalar, elements, ..
            } => (*scalar, *elements),
        };
        if scalar != self.plan.scalar_type() {
            return Err(MetalError::Unsupported);
        }
        if elements != self.plan.element_count() {
            return Err(MetalError::InvalidExtent);
        }
        let output = match input {
            MetalTensorInput::HostBytes { bytes, .. } => {
                if bytes.len() < self.plan.byte_len() {
                    return Err(MetalError::InvalidExtent);
                }
                let input = self.session.upload_bytes(&bytes[..self.plan.byte_len()])?;
                self.execute_buffer(&input)?
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
                if !self.session.same_session(lease.borrow().session()) {
                    return Err(MetalError::ForeignSession);
                }
                self.execute_buffer(&lease.borrow())?
            }
        };
        let resource = self
            .session
            .initialized_tensor_resource(self.pool, output)?;
        Ok(MetalTensorOwner {
            session: self.session.clone(),
            scalar: self.plan.scalar_type(),
            shape: self.plan.shape_owner(),
            count: self.plan.element_count(),
            resource,
        })
    }
    fn execute_buffer(&self, input: &MetalBuffer) -> Result<MetalBuffer, MetalError> {
        if let Some(relu) = &self.relu {
            let output = relu.execute_prefix(input, self.plan.byte_len())?;
            if self.plan.output() == self.plan.relu_effect().ok_or(MetalError::Unsupported)? {
                return Ok(output);
            }
            // A checked unused effect must complete successfully before publishing identity.
            drop(output);
        }
        self.carrier.execute(input)
    }
}
impl MetalTensorOwner {
    #[must_use]
    pub const fn scalar_type(&self) -> PcuScalarType {
        self.scalar
    }
    #[must_use]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
    #[must_use]
    pub fn shape_owner(&self) -> Rc<[usize]> {
        Rc::clone(&self.shape)
    }
    #[must_use]
    pub const fn element_count(&self) -> usize {
        self.count
    }
    #[must_use]
    pub const fn session(&self) -> &MetalSession {
        &self.session
    }
    #[must_use]
    pub const fn resource(&self) -> &MetalMemoryResource {
        &self.resource
    }
    /// Transfers the real initialized resource and its original session together.
    #[must_use]
    pub fn into_resource(self) -> (MetalSession, MetalMemoryResource) {
        (self.session, self.resource)
    }
    /// Reads exact logical bytes into an initialized prefix, preserving caller tails.
    /// # Errors
    /// Rejects short output or quarantined native completion before copying anything.
    pub fn read_bytes_into(&self, bytes: &mut [u8]) -> Result<(), MetalError> {
        let lease = self.resource.lease();
        let buffer = lease.borrow();
        if bytes.len() < buffer.byte_len() {
            return Err(MetalError::InvalidExtent);
        }
        buffer.read_into_bytes(&mut bytes[..buffer.byte_len()])
    }
    /// Reads the exact sealed logical type with no numerical conversion.
    /// # Errors
    /// Rejects type, byte order, short destination or quarantined completion before publication.
    pub fn read_into<T: PcuScalar>(&self, values: &mut [T]) -> Result<(), MetalError> {
        if cfg!(target_endian = "big") || T::TYPE != self.scalar {
            return Err(MetalError::Unsupported);
        }
        if values.len() < self.count {
            return Err(MetalError::InvalidExtent);
        }
        let mut argument = PcuHostArgument::read_write(PcuBindingRef::new(0, 0), values);
        self.read_bytes_into(argument.bytes_mut().ok_or(MetalError::Unsupported)?)
    }
}

#[cfg(feature = "benchmark-control")]
#[path = "control/control.rs"]
mod control;
#[cfg(feature = "benchmark-control")]
#[rustfmt::skip]
pub use control::{MetalNativeTensorReluControl,MetalNativeTensorCarrierControl};
#[cfg(feature = "benchmark-control")]
pub use binary::MetalNativeTensorBinaryControl;

#[cfg(test)]
#[path = "tests/tests.rs"]
mod tests;

#[path = "backward/backward.rs"]
mod backward;
pub use backward::{MetalTensorBackwardPlan, MetalPreparedTensorBackwardProgram};

#[path = "matmul/matmul.rs"]
mod matmul;
pub use matmul::{MetalTensorMatMulPlan, MetalPreparedTensorMatMulProgram};

#[path = "sgd/sgd.rs"]
mod sgd;
pub use sgd::{MetalTensorSgdPlan, MetalPreparedTensorSgdProgram};

#[path = "selected_numerical/selected_numerical.rs"]
mod selected_numerical;
pub use selected_numerical::{MetalSelectedNumericalTensorPlan,MetalPreparedSelectedNumericalTensorProgram,MetalSelectedNumericalTensorOperation};

#[path="mse/mse.rs"]
mod mse;
pub use mse::{MetalTensorMsePlan,MetalPreparedTensorMseProgram};

#[path="selected_graph/selected_graph.rs"]
mod selected_graph;
pub use selected_graph::{MetalSelectedTensorGraphPlan, MetalPreparedSelectedTensorGraph, MetalTensorGraphError};
