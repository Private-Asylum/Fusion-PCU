//! Private retained storage for an owning selected program; no leases escape as results.
#[rustfmt::skip]
use std::cell::{
    RefCell,
    RefMut,
};
use fusion_pcu::PcuScalarType;
use crate::CudaRuntime;
#[rustfmt::skip]
use super::{
    OpDescriptor,
    PcuMemoryAccess,
    PcuMemoryPoolId,
    PcuMemoryProvider,
    PcuMemoryResource,
    PcuMemoryResourceCapability,
    CudaMemoryResource,
    CudaPreparedGraphView,
    CudaTensorExecutionError,
    TensorScratchStoragePlan,
};

type Error = CudaTensorExecutionError;
type Resource = CudaMemoryResource;

pub(super) struct State {
    plan: Plan,
    banks: RefCell<Vec<Bank>>,
    #[cfg(feature = "insights")]
    guarded_report_epoch: std::cell::Cell<u64>,
}

struct Plan {
    slots: TensorScratchStoragePlan,
    assignments: Vec<(usize, usize)>,
    literals: Vec<usize>,
    node_count: usize,
    mse_count: usize,
    status_nodes: Vec<usize>,
    element_bytes: usize,
    alignment: usize,
}

impl State {
    pub(super) fn new(view: &CudaPreparedGraphView<'_, '_>) -> Result<Self, Error> {
        let mut eligible = Vec::new();
        let mut literals = Vec::new();
        let mut mse_count = 0;
        let mut status_nodes = Vec::new();
        for index in 0..view.node_values.len() {
            let node = view.node(index)?;
            if view.operation_index_of(node.value).is_some() && requires_status(view, index, node) {
                status_nodes.push(index);
            }
            if !view.suppressed_adds.contains(&node.value)
                && super::scratch_stores_node(node.op, node.value, &view.outputs)
            {
                if super::is_scratch_computed_op(node.op)
                    && view.operation_index_of(node.value).is_some()
                {
                    eligible.push(node.value);
                } else if matches!(
                    node.op,
                    OpDescriptor::Constant(_) | OpDescriptor::Uniform { .. }
                ) {
                    literals.push(index);
                }
            }
            if let OpDescriptor::MeanSquaredError { prediction, .. } = node.op
                && node.numerical_mode != Some(fusion_pcu::PcuNumericalMode::Strict)
            {
                let count = view
                    .graph
                    .shape(prediction)?
                    .iter()
                    .try_fold(1usize, |n, &d| n.checked_mul(d).ok_or(Error::SizeOverflow))?;
                mse_count = mse_count.max(count);
            }
        }
        let slots = view.scratch_storage_plan(&eligible)?;
        let assignments = slots
            .assignments()
            .iter()
            .map(|assignment| {
                Ok((
                    view.index_of(assignment.value)
                        .ok_or(Error::InvalidPlan(assignment.value))?,
                    assignment.slot,
                ))
            })
            .collect::<Result<Vec<_>, Error>>()?;
        let scalar = view.scalar_type.unwrap_or(PcuScalarType::F32);
        let (element_bytes, alignment) = super::scalar_layout(scalar)?;
        Ok(Self {
            plan: Plan {
                slots,
                assignments,
                literals,
                node_count: view.node_values.len(),
                mse_count,
                status_nodes,
                element_bytes,
                alignment: usize::try_from(alignment).map_err(|_| Error::SizeOverflow)?,
            },
            banks: RefCell::new(Vec::new()),
            #[cfg(feature = "insights")]
            guarded_report_epoch: std::cell::Cell::new(0),
        })
    }

    #[cfg(feature = "insights")]
    pub(super) fn begin_guarded_attempt(&self) {
        self.guarded_report_epoch
            .set(self.guarded_report_epoch.get().wrapping_add(1));
    }
    #[cfg(feature = "insights")]
    pub(super) const fn guarded_epoch(&self) -> u64 {
        self.guarded_report_epoch.get()
    }

    #[cfg(feature = "insights")]
    pub(super) fn guarded_report(
        &self,
        stream: &crate::CudaStreamHandle,
        pool: PcuMemoryPoolId,
    ) -> Option<super::guarded::CudaGuardedExecutionReport> {
        let banks = self.banks.try_borrow().ok()?;
        let bank = banks
            .iter()
            .find(|bank| bank.pool == pool && bank.runtime.same_instance(&stream.inner.runtime))?;
        if bank.poisoned {
            return None;
        }
        let storage = bank.guarded.as_ref()?;
        if !storage.uses_stream(stream) {
            return None;
        }
        (storage.report_epoch == self.guarded_epoch())
            .then_some(storage.report)
            .flatten()
    }

    /// Validate an existing bank without allocation or device work before pending uploads.
    pub(super) fn preflight_initialized(
        &self,
        runtime: &CudaRuntime,
        pool: PcuMemoryPoolId,
        inputs: &[&Resource],
    ) -> Result<bool, Error> {
        let banks = self.banks.try_borrow().map_err(|_| Error::ScratchBusy)?;
        let Some(bank) = banks
            .iter()
            .find(|bank| bank.pool == pool && bank.runtime.same_instance(runtime))
        else {
            return Ok(false);
        };
        if bank.poisoned {
            return Err(Error::ScratchMismatch);
        }
        bank.validate_available()?;
        if bank
            .physical
            .iter()
            .any(|scratch| inputs.iter().any(|input| scratch.may_overlap(input)))
        {
            return Err(Error::ScratchMismatch);
        }
        Ok(true)
    }

    /// Cold binding initializes a bank once. The mapped mutable borrow excludes every alias of
    /// this exact preparation until terminal completion and fault handling have finished.
    pub(super) fn bind<'a, P>(
        &'a self,
        runtime: &CudaRuntime,
        view: &CudaPreparedGraphView<'_, '_>,
        pool: PcuMemoryPoolId,
        memory: &mut P,
    ) -> Result<RefMut<'a, Bank>, Error>
    where
        P: PcuMemoryProvider<Resource = Resource>,
    {
        let mut banks = self
            .banks
            .try_borrow_mut()
            .map_err(|_| Error::ScratchBusy)?;
        let index = if let Some(index) = banks
            .iter()
            .position(|bank| bank.pool == pool && bank.runtime.same_instance(runtime))
        {
            index
        } else {
            let bank = self.plan.allocate(runtime, view, pool, memory)?;
            banks.push(bank);
            banks.len() - 1
        };
        if banks[index].poisoned {
            return Err(Error::ScratchMismatch);
        }
        banks[index].validate_available()?;
        Ok(RefMut::map(banks, |banks| &mut banks[index]))
    }
}

impl Plan {
    fn allocate<P>(
        &self,
        runtime: &CudaRuntime,
        view: &CudaPreparedGraphView<'_, '_>,
        pool: PcuMemoryPoolId,
        memory: &mut P,
    ) -> Result<Bank, Error>
    where
        P: PcuMemoryProvider<Resource = Resource>,
    {
        let mut slots = Vec::with_capacity(self.slots.slots().len());
        for slot in self.slots.slots() {
            let resource = super::allocate_tensor_for_size(
                memory,
                pool,
                &[slot.capacity_bytes],
                1,
                slot.alignment_bytes,
            )?;
            validate_resource(&resource, runtime, pool)?;
            if slots.iter().any(|other| resource.may_overlap(other)) {
                return Err(Error::ScratchMismatch);
            }
            slots.push(resource);
        }
        let mut resources = (0..self.node_count).map(|_| None).collect::<Vec<_>>();
        for &(index, slot) in &self.assignments {
            let resource = slots.get(slot).ok_or(Error::ScratchMismatch)?;
            let node = view.node(index)?;
            if resource.size_bytes() < view.physical_layout(node.value)?.physical_bytes {
                return Err(Error::ScratchMismatch);
            }
            resources[index] = Some(resource.clone_for_tensor_input());
        }
        for &index in &self.literals {
            let node = view.node(index)?;
            let layout = view.physical_layout(node.value)?;
            let resource = if super::is_checked_integer_scalar(node.scalar_type) {
                super::literal::upload_integer(node, layout, None, pool, memory)?
            } else if super::is_low_float_type(node.scalar_type) {
                super::literal::upload_low_float(node, layout, None, pool, memory)?
            } else if super::is_raw_float_type(node.scalar_type) {
                super::literal::upload_raw_float(node, layout, None, pool, memory)?
            } else if node.scalar_type == PcuScalarType::F64 {
                super::literal::upload_f64(node, layout, None, pool, memory)?
            } else {
                match node.op {
                    OpDescriptor::Constant(value) => super::upload_tensor(
                        memory,
                        pool,
                        value
                            .as_typed::<f32>()
                            .map_err(|_| Error::InvalidPlan(node.value))?,
                    )?,
                    OpDescriptor::Uniform { value } => {
                        let tensor = layout.uniform_tensor(
                            node.shape,
                            value
                                .as_typed::<f32>()
                                .map_err(|_| Error::InvalidPlan(node.value))?,
                        )?;
                        super::upload_tensor(memory, pool, &tensor)?
                    }
                    _ => return Err(Error::InvalidPlan(node.value)),
                }
            };
            validate_resource(&resource, runtime, pool)?;
            if resources
                .iter()
                .flatten()
                .any(|other| resource.may_overlap(other))
            {
                return Err(Error::ScratchMismatch);
            }
            slots.push(resource.clone_for_tensor_input());
            resources[index] = Some(resource);
        }
        let mse_squared = if self.mse_count == 0 {
            None
        } else {
            let resource = super::allocate_tensor_for_size(
                memory,
                pool,
                &[self.mse_count],
                self.element_bytes,
                self.alignment,
            )?;
            validate_resource(&resource, runtime, pool)?;
            if resources
                .iter()
                .flatten()
                .any(|other| resource.may_overlap(other))
            {
                return Err(Error::ScratchMismatch);
            }
            slots.push(resource.clone_for_tensor_input());
            Some(resource)
        };
        let statuses = self.allocate_statuses(runtime, pool, memory, &mut slots)?;
        Ok(Bank {
            runtime: runtime.clone(),
            pool,
            resources,
            mse_squared,
            statuses,
            guarded: None,
            physical: slots,
            poisoned: false,
        })
    }

    fn allocate_statuses<P>(
        &self,
        runtime: &CudaRuntime,
        pool: PcuMemoryPoolId,
        memory: &mut P,
        physical: &mut Vec<Resource>,
    ) -> Result<Vec<Option<Status>>, Error>
    where
        P: PcuMemoryProvider<Resource = Resource>,
    {
        let mut statuses = (0..self.node_count).map(|_| None).collect::<Vec<_>>();
        for &index in &self.status_nodes {
            let resource = super::allocate_tensor_for_size(memory, pool, &[1], 8, 8)?;
            validate_resource(&resource, runtime, pool)?;
            if physical.iter().any(|other| resource.may_overlap(other)) {
                return Err(Error::ScratchMismatch);
            }
            let mut buffer = resource.device_buffer().clone();
            buffer
                .copy_from(&u64::MAX.to_le_bytes())
                .map_err(crate::CudaOwnedDispatchError::from)
                .map_err(Error::Backend)?;
            physical.push(resource);
            statuses[index] = Some(Status {
                buffer,
                state: crate::owned_dispatch::FaultWordState::Sentinel,
            });
        }
        Ok(statuses)
    }
}

fn validate_resource(
    resource: &Resource,
    runtime: &CudaRuntime,
    pool: PcuMemoryPoolId,
) -> Result<(), Error> {
    if resource.pool() != pool
        || !resource.belongs_to_runtime(runtime)
        || resource.access() != PcuMemoryAccess::ReadWrite
        || !resource.supports(PcuMemoryResourceCapability::ReusableStorage)
    {
        Err(Error::ScratchMismatch)
    } else {
        Ok(())
    }
}

pub(super) struct Bank {
    pub(super) guarded: Option<Box<super::guarded::Storage>>,
    runtime: CudaRuntime,
    pool: PcuMemoryPoolId,
    pub(super) resources: Vec<Option<Resource>>,
    pub(super) mse_squared: Option<Resource>,
    pub(super) statuses: Vec<Option<Status>>,
    pub(super) physical: Vec<Resource>,
    poisoned: bool,
}

impl Bank {
    fn validate_available(&self) -> Result<(), Error> {
        for resource in &self.physical {
            resource
                .validate_access_available()
                .map_err(|_| Error::ScratchMismatch)?;
        }
        Ok(())
    }

    /// Physical access gates are authoritative: successful waits release them, and unknown
    /// completion retains/quarantines the affected leases. Prelaunch failures and terminal
    /// operational/arithmetic errors may retry only when every private allocation is idle.
    pub(super) fn finish(&mut self, error: Option<&Error>) -> Result<(), Error> {
        if self.validate_available().is_ok() {
            return Ok(());
        }
        self.quarantine();
        if error.is_none() {
            Err(Error::ScratchMismatch)
        } else {
            Ok(())
        }
    }

    fn quarantine(&mut self) {
        self.poisoned = true;
        // Unknown completion must not free any possibly affected allocation. These private
        // owners never escape; forgetting every shared slot/literal/workspace keeps it alive.
        for resource in self.physical.drain(..) {
            std::mem::forget(resource);
        }
        // One leaked physical owner retains each allocation; mapped aliases may then drop.
        self.resources.clear();
        self.mse_squared = None;
        self.statuses.clear();
    }
}

/// Cold factories decide status presence; native and transport-only nodes stay status-free.
fn requires_status(
    view: &CudaPreparedGraphView<'_, '_>,
    index: usize,
    node: fusion_pcu::dialect::tensor::NodeDescriptor<'_>,
) -> bool {
    view.fixed_dispatches[index]
        .as_ref()
        .is_some_and(|fixed| crate::owned_dispatch::kernel_uses_checked_arithmetic(&fixed.kernel))
        || view.strict_sgd_profiles[index].is_some()
        || super::strict_mse::Profile::from_node(view.graph, node).is_ok()
        || super::strict_matmul::Profile::from_node(view.graph, node).is_ok()
        || super::relu_backward::Profile::from_node(node)
            .is_ok_and(super::relu_backward::Profile::checked)
}

/// A private word is reusable only after terminal success observed its unchanged sentinel.
pub(super) struct Status {
    buffer: crate::DeviceBuffer,
    state: crate::owned_dispatch::FaultWordState,
}

impl Status {
    pub(super) fn submit(
        &mut self,
        prepared: &crate::CudaPreparedDispatch,
        bindings: &[fusion_pcu::PcuOwnedBinding<crate::DeviceBuffer>],
    ) -> Result<crate::CudaOwnedCompletion, Error> {
        // The same state proof as the sequential scalar wrapper handles fatal/recovered faults
        // and rejected submissions. Only a terminal decoded success restores the sentinel.
        let reset = self.state.begin_submission();
        prepared
            .submit_with_fault_word_state(bindings, &mut self.buffer, reset)
            .map_err(Error::Backend)
    }

    pub(super) const fn observe(&mut self, outcome: fusion_pcu::PcuCompletionOutcome) {
        self.state = crate::owned_dispatch::FaultWordState::after_terminal(outcome);
    }
}

/// Public/non-owning routes keep independent status allocation; private owners opt in explicitly.
pub(super) fn submit(
    prepared: &crate::CudaPreparedDispatch,
    bindings: &[fusion_pcu::PcuOwnedBinding<crate::DeviceBuffer>],
    status: Option<&mut Status>,
) -> Result<crate::CudaOwnedCompletion, Error> {
    status.map_or_else(
        || prepared.submit(bindings).map_err(Error::Backend),
        |status| status.submit(prepared, bindings),
    )
}

#[cfg(test)]
#[path = "tests/tests.rs"]
mod tests;
