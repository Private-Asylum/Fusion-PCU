//! Bounded, owning, fixed-kernel ordered execution with one terminal observation.
#[rustfmt::skip]
use super::{
    RocmTensorAssessor,
    RocmOwnedPreparedTensorGraph,
    RocmTensorExecutionError,
    RocmTensorInputRef,
    RocmTensorInputDescriptor,
    PcuMemoryResource,
    PcuOwnedDispatchMemorySession,
    RocmTensorOwnedOutput,
    RocmMemoryResource,
    RocmPreparedDispatch,
    FreshTensorOutput,
    OpDescriptor,
    PcuMemoryPoolId,
    PcuMemoryProvider,
    PcuBindingAccess,
    PcuBindingType,
    ValueId,
    SmallVec,
    Rc,
    owned_scratch,
};

type Error = RocmTensorExecutionError;
const MAX_STAGES: usize = 32;

pub(super) struct Plan {
    nodes: Vec<usize>,
    stream: crate::HipStreamHandle,
    initial: Rc<Vec<u8>>,
}

impl Plan {
    pub(super) fn uses_stream(&self, stream: &crate::HipStreamHandle) -> bool {
        Rc::ptr_eq(&self.stream.inner, &stream.inner)
    }
}

/// Requires the `insights` feature.
/// Validated terminal diagnostics, published after host observation of the device fault.
#[cfg(feature = "insights")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RocmGuardedExecutionReport {
    pub stage_count: usize,
    pub values: [Option<ValueId>; MAX_STAGES],
    pub stages: [Option<fusion_pcu::PcuGuardedExecutionStageOutcome>; MAX_STAGES],
    pub outcome: fusion_pcu::PcuGuardedExecutionOutcome,
}

impl RocmOwnedPreparedTensorGraph {
    #[cfg(feature = "insights")]
    pub(super) fn begin_guarded_attempt(&self) {
        if self.guarded.is_some() {
            self.scratch.begin_guarded_attempt();
        }
    }

    #[cfg(feature = "insights")]
    /// Requires the `insights` feature.
    /// Copies the last validated terminal report for the current guarded stream and pool.
    /// Returns `None` before execution, after protocol failure/rebind, or during a live borrow.
    #[must_use]
    pub fn last_guarded_execution_report(
        &self,
        pool: PcuMemoryPoolId,
    ) -> Option<RocmGuardedExecutionReport> {
        self.scratch
            .guarded_report(&self.guarded.as_ref()?.stream, pool)
    }
}

pub(super) struct Storage {
    resource: RocmMemoryResource,
    batch: crate::guarded_batch::GuardedBatch,
    #[cfg(feature = "insights")]
    pub(super) report: Option<RocmGuardedExecutionReport>,
    #[cfg(feature = "insights")]
    pub(super) report_epoch: u64,
}

impl Storage {
    #[cfg(feature = "insights")]
    pub(super) fn uses_stream(&self, stream: &crate::HipStreamHandle) -> bool {
        self.batch.uses_stream(stream)
    }
}

impl<'session> RocmTensorAssessor<'session> {
    /// Cold opt-in for a bounded complete fixed-kernel scope; decline precedes execution.
    ///
    /// # Errors
    /// Returns guarded variant compilation errors. Original public batch admission is unchanged.
    pub fn prepare_owned_program_guarded_execution(
        &self,
        prepared: &mut RocmOwnedPreparedTensorGraph,
    ) -> Result<bool, Error> {
        if prepared
            .guarded
            .as_ref()
            .is_some_and(|plan| plan.uses_stream(&self.state().stream))
        {
            return Ok(true);
        }
        #[cfg(feature = "insights")]
        prepared.begin_guarded_attempt();
        prepared.guarded = None;
        let view = prepared.view();
        if view.node_values.len() > 64
            || view.outputs.len() != 1
            || !view.suppressed_adds.is_empty()
            || !view.fused_add_by_relu.is_empty()
            || !view.bounded_pointwise_by_output.is_empty()
            || !view.bounded_mul_by_output.is_empty()
            || view.consuming_action.is_some()
        {
            return Ok(false);
        }
        let mut nodes = Vec::new();
        for index in 0..view.node_values.len() {
            let node = view.node(index)?;
            match node.op {
                OpDescriptor::Input | OpDescriptor::Constant(_) | OpDescriptor::Uniform { .. } => {
                    // Direct literal/input outputs need separate transfer semantics.
                    if view.outputs.contains(&node.value) {
                        return Ok(false);
                    }
                }
                OpDescriptor::Add { .. }
                | OpDescriptor::Sub { .. }
                | OpDescriptor::Mul { .. }
                | OpDescriptor::Div { .. }
                | OpDescriptor::Relu { .. } => {
                    let Ok(fixed) = view.fixed_dispatch(index) else {
                        return Ok(false);
                    };
                    if !fixed
                        .retained
                        .is_some_and(|retained| retained.uses_stream(&self.state().stream))
                    {
                        return Ok(false);
                    }
                    nodes.push(index);
                    if nodes.len() > MAX_STAGES {
                        return Ok(false);
                    }
                }
                _ => return Ok(false),
            }
        }
        if nodes
            .iter()
            .filter(|&&index| {
                view.fixed_dispatch(index).is_ok_and(|fixed| {
                    crate::owned_dispatch::kernel_uses_checked_arithmetic(&fixed.kernel)
                })
            })
            .count()
            < 2
        {
            return Ok(false);
        }
        // Variant compilation begins only after the entire scope is admitted.
        for &index in &nodes {
            let fixed = view.fixed_dispatch(index)?;
            let retained = fixed
                .retained
                .ok_or(Error::InvalidPlan(view.node(index)?.value))?;
            self.session
                .retain_guarded_dispatch(retained, &fixed.kernel)
                .map_err(Error::Backend)?;
        }
        let mut initial = Vec::with_capacity(nodes.len() * 16);
        for _ in &nodes {
            initial.extend_from_slice(&u64::MAX.to_le_bytes());
            initial.extend_from_slice(&0u64.to_le_bytes());
        }
        prepared.guarded = Some(Plan {
            nodes,
            stream: self.state().stream.clone(),
            initial: Rc::new(initial),
        });
        Ok(true)
    }

    #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
    pub(super) fn execute_guarded_owned_program<'input, T, P>(
        &'input self,
        prepared: &RocmOwnedPreparedTensorGraph,
        plan: &Plan,
        inputs: &[(ValueId, &RocmTensorInputRef<'input>)],
        outputs: &[(ValueId, FreshTensorOutput<'_>)],
        bank: &mut owned_scratch::Bank,
        pool: PcuMemoryPoolId,
        memory: &mut P,
    ) -> Result<SmallVec<[RocmTensorOwnedOutput<T>; 1]>, Error>
    where
        T: fusion_pcu::PcuScalar,
        P: PcuMemoryProvider<Resource = RocmMemoryResource>,
        'session: 'input,
    {
        let view = prepared.view();
        if bank.guarded.is_none() {
            let resource =
                super::allocate_tensor_for_size(memory, pool, &[plan.nodes.len()], 16, 8)?;
            if !resource.belongs_to_runtime(self.session.tensor_runtime())
                || resource.pool() != pool
                || resource.access() != fusion_pcu::PcuMemoryAccess::ReadWrite
                || bank
                    .physical
                    .iter()
                    .any(|other| resource.may_overlap(other))
                || inputs
                    .iter()
                    .any(|(_, input)| resource.may_overlap(input.resource()))
                || outputs
                    .iter()
                    .any(|(_, output)| resource.may_overlap(&output.resource))
            {
                return Err(Error::ScratchMismatch);
            }
            let batch = crate::guarded_batch::GuardedBatch::new(
                &self.state().stream,
                plan.nodes.len(),
                plan.initial.len(),
            )
            .map_err(crate::RocmOwnedDispatchError::from)
            .map_err(Error::Backend)?;
            bank.physical.push(resource.clone_for_tensor_input());
            bank.guarded = Some(Box::new(Storage {
                resource,
                batch,
                #[cfg(feature = "insights")]
                report: None,
                #[cfg(feature = "insights")]
                report_epoch: 0,
            }));
        }
        let storage = bank.guarded.as_mut().ok_or(Error::ScratchMismatch)?;
        if !storage.batch.uses_stream(&self.state().stream) {
            storage.batch = crate::guarded_batch::GuardedBatch::new(
                &self.state().stream,
                plan.nodes.len(),
                plan.initial.len(),
            )
            .map_err(crate::RocmOwnedDispatchError::from)
            .map_err(Error::Backend)?;
        }
        storage
            .resource
            .validate_access_available()
            .map_err(|_| Error::ScratchMismatch)?;
        #[cfg(feature = "insights")]
        {
            storage.report = None;
        }
        let resource = |value: ValueId| -> Result<&RocmMemoryResource, Error> {
            outputs
                .iter()
                .find(|(candidate, _)| *candidate == value)
                .map(|(_, output)| &output.resource)
                .or_else(|| {
                    inputs
                        .iter()
                        .find(|(candidate, _)| *candidate == value)
                        .map(|(_, input)| input.resource())
                })
                .or_else(|| {
                    view.index_of(value)
                        .and_then(|index| bank.resources.get(index)?.as_ref())
                })
                .ok_or(Error::MissingResource(value))
        };
        // Construct and validate every binding before the first initialization or launch.
        let mut stages = SmallVec::<[(&RocmPreparedDispatch, SmallVec<[_; 3]>); MAX_STAGES]>::new();
        for &index in &plan.nodes {
            let node = view.node(index)?;
            let fixed = view.fixed_dispatch(index)?;
            let executable = fixed
                .retained
                .filter(|retained| retained.uses_stream(&self.state().stream))
                .filter(|retained| retained.has_guarded_dispatch())
                .ok_or(Error::InvalidPlan(node.value))?;
            let (left, right) = match node.op {
                OpDescriptor::Add { left, right }
                | OpDescriptor::Sub { left, right }
                | OpDescriptor::Mul { left, right }
                | OpDescriptor::Div { left, right } => (left, Some(right)),
                OpDescriptor::Relu { input } => (input, None),
                _ => return Err(Error::InvalidPlan(node.value)),
            };
            let mut bindings = SmallVec::<[_; 3]>::new();
            bindings.push(
                self.session
                    .bind(
                        fixed.left_binding,
                        PcuBindingAccess::ReadOnly,
                        PcuBindingType::Value(fixed.value_type),
                        resource(left)?,
                    )
                    .map_err(Error::Backend)?,
            );
            if let Some(right) = right {
                bindings.push(
                    self.session
                        .bind(
                            fixed.right_binding.ok_or(Error::InvalidPlan(node.value))?,
                            PcuBindingAccess::ReadOnly,
                            PcuBindingType::Value(fixed.value_type),
                            resource(right)?,
                        )
                        .map_err(Error::Backend)?,
                );
            }
            bindings.push(
                self.session
                    .bind(
                        fixed.output_binding,
                        PcuBindingAccess::WriteOnly,
                        PcuBindingType::Value(fixed.value_type),
                        resource(node.value)?,
                    )
                    .map_err(Error::Backend)?,
            );
            executable
                .validate_guarded_bindings(&bindings)
                .map_err(Error::Backend)?;
            stages.push((executable, bindings));
        }
        #[cfg(feature = "insights")]
        let mut report = RocmGuardedExecutionReport {
            stage_count: stages.len(),
            values: [None; MAX_STAGES],
            stages: [None; MAX_STAGES],
            outcome: fusion_pcu::PcuGuardedExecutionOutcome::Succeeded,
        };
        // All caller-supplied provider operations and binding admission are complete. The
        // following synchronous submission/cleanup path uses only concrete backend owners;
        // no caller or foreign callback can alter HIP selection inside this private scope.
        // Failed native calls still invalidate the scope before cleanup reselects the device.
        let runtime_scope = self
            .session
            .tensor_runtime()
            .enter_private_scope()
            .map_err(crate::RocmOwnedDispatchError::from)
            .map_err(Error::Backend)?;
        let submission: Result<_, Error> = (|| {
            #[cfg(feature = "allocation-census")]
            crate::ffi::guarded_chain();
            storage
                .batch
                .initialize(storage.resource.device_buffer(), Rc::clone(&plan.initial))
                .map_err(crate::RocmOwnedDispatchError::from)
                .map_err(Error::Backend)?;
            for (stage, (executable, bindings)) in stages.iter().enumerate() {
                executable
                    .submit_guarded_into_batch(
                        bindings,
                        storage.batch.batch(),
                        storage.resource.device_buffer(),
                        u32::try_from(stage).map_err(|_| Error::SizeOverflow)?,
                    )
                    .map_err(Error::Backend)?;
            }
            storage
                .batch
                .queue_readback(storage.resource.device_buffer(), plan.initial.len())
                .map_err(crate::RocmOwnedDispatchError::from)
                .map_err(Error::Backend)?;
            let bytes = storage
                .batch
                .complete()
                .map_err(crate::RocmOwnedDispatchError::from)
                .map_err(Error::Backend)?;
            let outcome = fusion_pcu::validate_guarded_execution(stages.len(), |stage| {
                let record = &bytes[stage * 16..(stage + 1) * 16];
                let word = u64::from_le_bytes(
                    record[..8]
                        .try_into()
                        .map_err(|_| crate::HipError::InvalidExecutionFaultWord(0))?,
                );
                let disposition = u64::from_le_bytes(
                    record[8..]
                        .try_into()
                        .map_err(|_| crate::HipError::InvalidExecutionFaultWord(0))?,
                );
                let decoded = stages[stage].0.decode_guarded_record(word, disposition)?;
                #[cfg(feature = "insights")]
                {
                    report.values[stage] = Some(view.node_values[plan.nodes[stage]]);
                    report.stages[stage] = Some(decoded);
                }
                Ok::<_, crate::HipError>(decoded)
            })
            .map_err(|_| Error::FailedCompletion)?;
            Ok(outcome)
        })();
        storage
            .batch
            .reset_after_terminal()
            .map_err(crate::RocmOwnedDispatchError::from)
            .map_err(Error::Backend)?;
        drop(runtime_scope);
        let outcome = submission?;
        #[cfg(feature = "insights")]
        {
            report.outcome = outcome;
            storage.report = Some(report);
            storage.report_epoch = prepared.scratch.guarded_epoch();
        }
        if let fusion_pcu::PcuGuardedExecutionOutcome::Fault { fault, .. } = outcome {
            return Err(Error::ExecutionFault(fault));
        }

        outputs
            .iter()
            .map(|(value, output)| {
                let input = super::RocmTensorInput {
                    session: self.session,
                    shape: super::PcuOwnedShape::from_slice(output.shape),
                    resource: output.resource.clone_for_tensor_input(),
                    scalar_type: T::TYPE,
                    updateable: true,
                };
                Ok((*value, input.into_device_tensor::<T>()?))
            })
            .collect()
    }
}

#[cfg(test)]
#[path = "tests/tests.rs"]
mod tests;
