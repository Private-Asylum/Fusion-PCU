//! Byte-value transport over the existing terminal four-buffer Metal protocol.

#[path = "plan/plan.rs"]
mod plan;
pub use plan::MetalTransportPlan;
#[path = "host/host.rs"]
mod host;
pub use host::{MetalTransportHostBackend, MetalPreparedTransportHostKernel};
#[rustfmt::skip]
use super::{
    ffi,
    validate_byte_extent,
    MetalBuffer,
    MetalError,
    MetalSession,
};

/// Retained transport pipeline. Every call checks real resources before submission.
pub struct MetalPreparedTransportKernel {
    session: MetalSession,
    plan: MetalTransportPlan,
    pipeline: ffi::Pipeline,
}
impl MetalSession {
    /// Compiles an independently assessed byte transport plan in this exact session.
    /// # Errors
    /// Returns allocation-limit, quarantine or native compilation errors.
    pub fn prepare_transport_plan(
        &self,
        plan: MetalTransportPlan,
    ) -> Result<MetalPreparedTransportKernel, MetalError> {
        for resource in plan.resources() {
            let bytes = usize::try_from(resource.minimum_elements())
                .map_err(|_| MetalError::InvalidExtent)?
                .checked_mul(plan.width)
                .ok_or(MetalError::InvalidExtent)?;
            validate_byte_extent(bytes, self.0.facts.max_buffer_bytes)?;
        }
        let pipeline = self
            .0
            .native
            .compile(&plan.shader, "pcu_scalar_transport")?;
        Ok(MetalPreparedTransportKernel {
            session: self.clone(),
            plan,
            pipeline,
        })
    }
}
impl MetalPreparedTransportKernel {
    #[must_use]
    pub const fn plan(&self) -> &MetalTransportPlan {
        &self.plan
    }

    /// Executes in actual first-access resource order. Writable physical aliases reject.
    ///
    /// Larger read/write owners are valid views; only the described prefix is touched.
    /// All typed owners and native leases survive terminal completion. A possible-write
    /// fatal/uncertain outcome retains the existing resident discard/quarantine law.
    /// # Errors
    /// Returns arity, extent, alias, foreign-session, quarantine or operational errors.
    pub fn execute_into(&self, resources: &[&MetalBuffer]) -> Result<(), MetalError> {
        self.validate_resources(resources)?;
        let first = resources[0];
        let buffers: [&ffi::Buffer; 4] =
            core::array::from_fn(|slot| &resources.get(slot).copied().unwrap_or(first).native);
        self.session.0.native.execute(
            &self.pipeline,
            buffers,
            [self.plan.element_count(), 0],
            usize::try_from(self.plan.element_count()).map_err(|_| MetalError::InvalidExtent)?,
        )
    }

    pub(super) fn validate_resources(&self, resources: &[&MetalBuffer]) -> Result<(), MetalError> {
        self.session.ensure_quiescent()?;
        if resources.len() != self.plan.resources().len() {
            return Err(MetalError::InvalidExtent);
        }
        for (slot, (buffer, role)) in resources.iter().zip(self.plan.resources()).enumerate() {
            if !self.session.same_session(&buffer.session) {
                return Err(MetalError::ForeignSession);
            }
            let bytes = usize::try_from(role.minimum_elements())
                .map_err(|_| MetalError::InvalidExtent)?
                .checked_mul(self.plan.width)
                .ok_or(MetalError::InvalidExtent)?;
            if buffer.byte_len() < bytes {
                return Err(MetalError::InvalidExtent);
            }
            if resources[..slot]
                .iter()
                .enumerate()
                .any(|(prior, candidate)| {
                    core::ptr::eq(*candidate, *buffer)
                        && (role.minimum_write_elements != 0
                            || self.plan.resources()[prior].minimum_write_elements != 0)
                })
            {
                return Err(MetalError::Unsupported);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "tests/tests.rs"]
mod tests;
