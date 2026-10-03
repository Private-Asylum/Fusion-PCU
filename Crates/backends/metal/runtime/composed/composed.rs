//! Retained original-header composition candidate; no discovery capability grant.
#[rustfmt::skip]
use super::{
    ffi,
    validate_byte_extent,
    MetalBuffer,
    MetalError,
    MetalSession,
};
use crate::MetalCheckedMapPlan;

/// One retained native pipeline and private status bank with scoped mapped inspection.
///
/// Resources are supplied in the plan's actual first-access order. The caller
/// owns public host staging/commit; resident possible-write failures follow the
/// established discard/quarantine law, rather than a rollback guarantee.
pub struct MetalPreparedCheckedMapKernel {
    session: MetalSession,
    plan: MetalCheckedMapPlan,
    pipeline: ffi::Pipeline,
    status: MetalBuffer,
    may_write: bool,
}

impl MetalSession {
    /// Compile this exact plan once and retain its status storage.
    /// # Errors
    /// Refuses unsupported emission, device allocation bounds or native failures.
    pub fn prepare_checked_map_plan(
        &self,
        plan: MetalCheckedMapPlan,
    ) -> Result<MetalPreparedCheckedMapKernel, MetalError> {
        self.ensure_quiescent()?;
        let source = plan.native_source()?;
        self.prepare_checked_map_source(plan, &source)
    }
    /// Benchmark-only handwritten workload body with shared checked scalar primitives.
    /// The original plan retains metadata and terminal law; it does not lower this body.
    /// This is an orchestration control, not an independent arithmetic oracle or optimal codegen.
    /// # Errors
    /// Refuses a different workload, unsupported native source or resource bounds.
    #[cfg(feature = "benchmark-control")]
    pub fn prepare_native_saved_stage_control(
        &self,
        plan: MetalCheckedMapPlan,
    ) -> Result<MetalPreparedCheckedMapKernel, MetalError> {
        self.ensure_quiescent()?;
        let source = crate::composed::control::sources(&plan)?;
        self.prepare_checked_map_source(plan, &source)
    }
    fn prepare_checked_map_source(
        &self,
        plan: MetalCheckedMapPlan,
        source: &str,
    ) -> Result<MetalPreparedCheckedMapKernel, MetalError> {
        self.ensure_quiescent()?;
        let width = usize::from(plan.value_type().scalar_type().bit_width()) / 8;
        for role in plan.resources() {
            let count = role.minimum_read_elements.max(role.minimum_write_elements);
            let bytes = usize::try_from(count)
                .map_err(|_| MetalError::InvalidExtent)?
                .checked_mul(width)
                .ok_or(MetalError::InvalidExtent)?;
            validate_byte_extent(bytes, self.0.facts.max_buffer_bytes)?;
        }
        validate_byte_extent(plan.status_byte_len(), self.0.facts.max_buffer_bytes)?;
        let pipeline = self.0.native.compile(source, "pcu_checked_composed")?;
        let status = self.allocate_zeroed_bytes(plan.status_byte_len())?;
        Ok(MetalPreparedCheckedMapKernel {
            session: self.clone(),
            plan,
            pipeline,
            status,
            may_write: false,
        })
    }
}

impl MetalPreparedCheckedMapKernel {
    #[must_use]
    pub const fn plan(&self) -> &MetalCheckedMapPlan {
        &self.plan
    }

    /// True once this call may have written an existing supplied resource.
    /// Preflight failures leave this false; uncertain work leaves it true.
    #[must_use]
    pub const fn last_call_may_have_written(&self) -> bool {
        self.may_write
    }

    /// Execute exact actual resources and validate every per-effect record before arbitration.
    ///
    /// Read/write capacities may exceed the logical span; untouched suffixes remain native.
    /// The launch uses one worker per logical lane to preserve ordered local operations,
    /// even when the retained original source request described a grid-stride loop.
    /// # Errors
    /// Returns preflight, native terminal/status errors or exact fatal/recovered arithmetic.
    pub fn execute_into(&mut self, resources: &[&MetalBuffer]) -> Result<(), MetalError> {
        self.may_write = false;
        self.validate_resources(resources)?;
        self.status.native.fill_ones();
        let first = resources[0];
        let buffers: [&ffi::Buffer; 5] = core::array::from_fn(|slot| {
            if slot == 4 {
                &self.status.native
            } else {
                &resources.get(slot).copied().unwrap_or(first).native
            }
        });
        self.may_write = true;
        let count = self.plan.logical_extent();
        self.session.0.native.execute(
            &self.pipeline,
            buffers,
            [count, 0],
            usize::try_from(count).map_err(|_| MetalError::InvalidExtent)?,
        )?;
        self.status
            .native
            .inspect_words(self.plan.status_byte_len() / 4, |words| {
                self.plan
                    .validate_status_words(words)?
                    .map_or(Ok(()), |fault| Err(MetalError::Arithmetic(fault)))
            })
    }

    fn validate_resources(&self, resources: &[&MetalBuffer]) -> Result<(), MetalError> {
        self.session.ensure_quiescent()?;
        if resources.len() != self.plan.resources().len() || resources.is_empty() {
            return Err(MetalError::InvalidExtent);
        }
        let width = usize::from(self.plan.value_type().scalar_type().bit_width()) / 8;
        for (slot, (buffer, role)) in resources.iter().zip(self.plan.resources()).enumerate() {
            if !self.session.same_session(&buffer.session) {
                return Err(MetalError::ForeignSession);
            }
            let count = role.minimum_read_elements.max(role.minimum_write_elements);
            let bytes = usize::try_from(count)
                .map_err(|_| MetalError::InvalidExtent)?
                .checked_mul(width)
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

#[path = "host/host.rs"]
mod host;
#[rustfmt::skip]
pub use host::{
    MetalComposedHostBackend,
    MetalPreparedCheckedMapHostKernel,
};
