//! `ROCm` storage adapter for the tensor dialect's backend-neutral feedback schedule.

use std::num::NonZeroUsize;

use fusion_pcu::{
    allocate_with_resource_policy,
    PcuMemoryAccess,
    PcuMemoryAllocateWithPolicyError,
    PcuMemoryAllocationRequest,
    PcuMemoryHostAccess,
    PcuMemoryPoolId,
    PcuMemoryProvider,
    PcuMemoryReleaseError,
    PcuMemoryReservationLedger,
    PcuMemoryReservationSet,
    PcuMemoryResourcePolicy,
};
use fusion_pcu_tensor::{
    TensorError,
    TensorFeedbackInput,
    TensorFeedbackPlan,
    ValueId,
};
use smallvec::SmallVec;

use super::{
    RocmMemoryResource,
    RocmPreparedTensorGraph,
    RocmTensorAssessor,
    RocmTensorExecutionError,
    RocmTensorInput,
    RocmTensorOutputBank,
    RocmTensorScratch,
};

/// Scratch and two output banks for an explicitly selected, synchronous feedback run.
///
/// The tensor dialect defines the feedback relationships and bank rotation. This `ROCm` adapter
/// owns the actual device resources and keeps its existing failure-poisoning rules.
pub struct RocmTensorFeedbackResources<'plan, 'graph, 'session> {
    prepared: &'plan RocmPreparedTensorGraph<'graph>,
    scratch: RocmTensorScratch<'plan, 'graph, 'session>,
    banks: [RocmTensorOutputBank<'plan, 'graph, 'session>; 2],
    completed_steps: usize,
    reservations: Vec<PcuMemoryReservationSet>,
}

/// Feedback resources whose policy reservations are reconciled when they are dropped.
///
/// The mutable ledger borrow prevents reservations from being invalidated while the resources
/// are alive. Execution through [`DerefMut`](std::ops::DerefMut) remains synchronous, so dropping
/// this guard first drops device resources and then releases their reservations.
#[must_use = "dropping this guard releases its resources and admission reservations"]
pub struct RocmAdmittedTensorFeedbackResources<'ledger, 'plan, 'graph, 'session, const N: usize> {
    resources: Option<RocmTensorFeedbackResources<'plan, 'graph, 'session>>,
    ledger: Option<&'ledger mut PcuMemoryReservationLedger<N>>,
}

impl<'plan, 'graph, 'session, const N: usize> std::ops::Deref
    for RocmAdmittedTensorFeedbackResources<'_, 'plan, 'graph, 'session, N>
{
    type Target = RocmTensorFeedbackResources<'plan, 'graph, 'session>;

    fn deref(&self) -> &Self::Target {
        self.resources
            .as_ref()
            .expect("feedback resources exist until explicit release")
    }
}

impl<const N: usize> std::ops::DerefMut for RocmAdmittedTensorFeedbackResources<'_, '_, '_, '_, N> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.resources
            .as_mut()
            .expect("feedback resources exist until explicit release")
    }
}

impl<const N: usize> RocmAdmittedTensorFeedbackResources<'_, '_, '_, '_, N> {
    /// Explicitly drops device resources and releases their admission reservations.
    ///
    /// # Errors
    ///
    /// Returns a retryable error retaining the remaining reservation handles. Retry it against
    /// the same ledger with [`RocmTensorFeedbackReleaseError::retry`].
    ///
    /// # Panics
    ///
    /// Panics only if the guard's internal resource or ledger state was already taken, which is
    /// prevented by safe callers and indicates an internal lifecycle bug.
    pub fn release(mut self) -> Result<(), RocmTensorFeedbackReleaseError> {
        let resources = self
            .resources
            .take()
            .expect("feedback resources exist until explicit release");
        let RocmTensorFeedbackResources {
            scratch,
            banks,
            mut reservations,
            ..
        } = resources;
        drop((scratch, banks));
        let ledger = self
            .ledger
            .as_deref_mut()
            .expect("the admission ledger exists until explicit release");
        match release_reservations(&mut reservations, ledger) {
            Ok(()) => Ok(()),
            Err(error) => Err(RocmTensorFeedbackReleaseError {
                error,
                reservations,
            }),
        }
    }
}

impl<const N: usize> Drop for RocmAdmittedTensorFeedbackResources<'_, '_, '_, '_, N> {
    fn drop(&mut self) {
        let Some(resources) = self.resources.take() else {
            return;
        };
        let RocmTensorFeedbackResources {
            scratch,
            banks,
            mut reservations,
            ..
        } = resources;
        drop((scratch, banks));
        if let Some(ledger) = self.ledger.as_deref_mut() {
            // Handles are private and the ledger is exclusively borrowed for the full lifetime
            // of this guard, so an invalid handle here indicates an internal accounting bug.
            release_reservations(&mut reservations, ledger)
                .expect("valid feedback reservation must release from its borrowed ledger");
        }
    }
}

/// Preparation failed and admission rollback also failed; reservations remain retryable.
#[derive(Debug)]
pub enum RocmTensorFeedbackPrepareError {
    Execution(RocmTensorExecutionError),
    Rollback {
        execution: RocmTensorExecutionError,
        rollback: PcuMemoryReleaseError,
        reservations: Vec<PcuMemoryReservationSet>,
    },
}

impl RocmTensorFeedbackPrepareError {
    /// Retries releasing reservations left by a failed preparation rollback.
    ///
    /// # Errors
    ///
    /// Returns a ledger release failure and retains every un-released handle in this error.
    pub fn retry_rollback<const N: usize>(
        &mut self,
        ledger: &mut PcuMemoryReservationLedger<N>,
    ) -> Result<(), PcuMemoryReleaseError> {
        match self {
            Self::Execution(_) => Ok(()),
            Self::Rollback { reservations, .. } => release_reservations(reservations, ledger),
        }
    }
}

impl std::fmt::Display for RocmTensorFeedbackPrepareError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for RocmTensorFeedbackPrepareError {}

fn rollback_prepare<const N: usize>(
    execution: RocmTensorExecutionError,
    mut reservations: Vec<PcuMemoryReservationSet>,
    ledger: &mut PcuMemoryReservationLedger<N>,
) -> RocmTensorFeedbackPrepareError {
    match release_reservations(&mut reservations, ledger) {
        Ok(()) => RocmTensorFeedbackPrepareError::Execution(execution),
        Err(rollback) => RocmTensorFeedbackPrepareError::Rollback {
            execution,
            rollback,
            reservations,
        },
    }
}

/// Reservations remain after resources have been dropped and can be released by retrying.
#[derive(Debug)]
pub struct RocmTensorFeedbackReleaseError {
    pub error: PcuMemoryReleaseError,
    pub reservations: Vec<PcuMemoryReservationSet>,
}

impl std::fmt::Display for RocmTensorFeedbackReleaseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for RocmTensorFeedbackReleaseError {}

impl RocmTensorFeedbackReleaseError {
    /// Retries releasing the remaining ledger reservations.
    ///
    /// # Errors
    ///
    /// Returns the next reservation release failure while retaining all un-released handles.
    pub fn retry<const N: usize>(
        &mut self,
        ledger: &mut PcuMemoryReservationLedger<N>,
    ) -> Result<(), PcuMemoryReleaseError> {
        release_reservations(&mut self.reservations, ledger)
    }
}

impl RocmTensorFeedbackResources<'_, '_, '_> {
    /// Drops scratch and output resources, then releases their admission reservations.
    ///
    /// Call only after all executions and borrowed output uses have completed. A release failure
    /// returns the remaining reservation handles for [`RocmTensorFeedbackReleaseError::retry`].
    ///
    /// # Errors
    ///
    /// Returns any reservation that the supplied ledger could not release.
    pub fn release<const N: usize>(
        self,
        ledger: &mut PcuMemoryReservationLedger<N>,
    ) -> Result<(), RocmTensorFeedbackReleaseError> {
        let Self {
            scratch,
            banks,
            mut reservations,
            ..
        } = self;
        drop((scratch, banks));
        release_reservations(&mut reservations, ledger).map_err(|error| {
            RocmTensorFeedbackReleaseError {
                error,
                reservations,
            }
        })
    }
}

fn release_reservations<const N: usize>(
    reservations: &mut Vec<PcuMemoryReservationSet>,
    ledger: &mut PcuMemoryReservationLedger<N>,
) -> Result<(), PcuMemoryReleaseError> {
    while !reservations.is_empty() {
        reservations[0].release_all(ledger)?;
        reservations.remove(0);
    }
    Ok(())
}

impl<'session> RocmTensorAssessor<'session> {
    /// Allocates reusable scratch and both output banks for a prepared feedback graph.
    ///
    /// # Errors
    ///
    /// Returns a provider, session, shape, or allocation error before any iteration begins.
    pub fn prepare_feedback_resources<'plan, 'graph, P>(
        &'session self,
        prepared: &'plan RocmPreparedTensorGraph<'graph>,
        pool: PcuMemoryPoolId,
        memory: &mut P,
    ) -> Result<RocmTensorFeedbackResources<'plan, 'graph, 'session>, RocmTensorExecutionError>
    where
        P: PcuMemoryProvider<Resource = RocmMemoryResource>,
    {
        let scratch = self.prepare_scratch(prepared, pool, memory)?;
        let first = self.prepare_output_bank(prepared, pool, memory)?;
        let second = self.prepare_output_bank(prepared, pool, memory)?;
        Ok(RocmTensorFeedbackResources {
            prepared,
            scratch,
            banks: [first, second],
            completed_steps: 0,
            reservations: Vec::new(),
        })
    }

    /// Prepares feedback scratch and both output banks under explicit placement/admission policy.
    ///
    /// Reservations remain owned by the returned feedback resource and must be explicitly
    /// released after execution is quiescent. On preparation failure, allocations are dropped
    /// before reservations are rolled back; failed rollback returns retry handles.
    ///
    /// # Errors
    ///
    /// Returns preparation/admission failure. If rollback also fails, the error retains the
    /// remaining reservation handles for explicit retry.
    fn prepare_feedback_resources_with_policy<'plan, 'graph, P, const N: usize>(
        &'session self,
        prepared: &'plan RocmPreparedTensorGraph<'graph>,
        pool: PcuMemoryPoolId,
        memory: &mut P,
        ledger: &mut PcuMemoryReservationLedger<N>,
        policy: PcuMemoryResourcePolicy,
    ) -> Result<RocmTensorFeedbackResources<'plan, 'graph, 'session>, RocmTensorFeedbackPrepareError>
    where
        P: PcuMemoryProvider<Resource = RocmMemoryResource>,
    {
        let mut reservations = Vec::new();
        let mut allocate = |memory: &mut P,
                            pool,
                            shape: &[usize],
                            upload: Option<&fusion_pcu_tensor::Tensor>| {
            let size_bytes = u64::try_from(super::byte_len(shape)?)
                .map_err(|_| RocmTensorExecutionError::SizeOverflow)?;
            let request = policy.apply(PcuMemoryAllocationRequest {
                pool,
                size_bytes,
                alignment_bytes: 4,
                access: PcuMemoryAccess::ReadWrite,
                host_access: PcuMemoryHostAccess::TransferOnly,
                require_device_local: false,
            });
            let admitted = match allocate_with_resource_policy(memory, ledger, request, policy) {
                Ok(admitted) => admitted,
                Err(error) => {
                    let retained = match &error {
                        PcuMemoryAllocateWithPolicyError::AdmissionRollback {
                            reservations,
                            ..
                        }
                        | PcuMemoryAllocateWithPolicyError::ProviderRollback {
                            reservations, ..
                        } => Some(*reservations),
                        _ => None,
                    };
                    if let Some(set) = retained.filter(|set| !set.is_empty()) {
                        reservations.push(set);
                    }
                    return Err(RocmTensorExecutionError::MemoryAdmission(error));
                }
            };
            let (mut resource, set) = admitted.into_parts();
            if !set.is_empty() {
                reservations.push(set);
            }
            if let Some(tensor) = upload {
                super::transfer_tensor(memory, &mut resource, tensor)?;
            }
            Ok(resource)
        };
        let scratch_result =
            self.prepare_scratch_with_allocator(prepared, pool, memory, &mut allocate);
        let scratch = match scratch_result {
            Ok(value) => value,
            Err(execution) => return Err(rollback_prepare(execution, reservations, ledger)),
        };
        let first = match self.prepare_output_bank_with_allocator(
            prepared,
            pool,
            memory,
            &mut |memory, pool, shape| allocate(memory, pool, shape, None),
        ) {
            Ok(value) => value,
            Err(execution) => {
                drop(scratch);
                return Err(rollback_prepare(execution, reservations, ledger));
            }
        };
        let second = match self.prepare_output_bank_with_allocator(
            prepared,
            pool,
            memory,
            &mut |memory, pool, shape| allocate(memory, pool, shape, None),
        ) {
            Ok(value) => value,
            Err(execution) => {
                drop(first);
                drop(scratch);
                return Err(rollback_prepare(execution, reservations, ledger));
            }
        };
        Ok(RocmTensorFeedbackResources {
            prepared,
            scratch,
            banks: [first, second],
            completed_steps: 0,
            reservations,
        })
    }

    /// Prepares policy-admitted feedback resources with automatic ledger reconciliation on drop.
    ///
    /// The returned guard exclusively borrows `ledger` until it is explicitly released or
    /// dropped. This prevents another caller from invalidating its reservation handles. Keep the
    /// guard in a narrow scope when more allocations need to use the same ledger.
    ///
    /// # Errors
    ///
    /// Returns preparation/admission failure. If preparation rollback fails, the error retains
    /// the remaining reservation handles for explicit retry.
    pub fn prepare_feedback_resources_with_policy_guard<'ledger, 'plan, 'graph, P, const N: usize>(
        &'session self,
        prepared: &'plan RocmPreparedTensorGraph<'graph>,
        pool: PcuMemoryPoolId,
        memory: &mut P,
        ledger: &'ledger mut PcuMemoryReservationLedger<N>,
        policy: PcuMemoryResourcePolicy,
    ) -> Result<
        RocmAdmittedTensorFeedbackResources<'ledger, 'plan, 'graph, 'session, N>,
        RocmTensorFeedbackPrepareError,
    >
    where
        P: PcuMemoryProvider<Resource = RocmMemoryResource>,
    {
        let resources =
            self.prepare_feedback_resources_with_policy(prepared, pool, memory, ledger, policy)?;
        Ok(RocmAdmittedTensorFeedbackResources {
            resources: Some(resources),
            ledger: Some(ledger),
        })
    }

    /// Runs a selected tensor graph repeatedly with dialect-defined feedback and bank rotation.
    ///
    /// All graph inputs must be supplied for the initial step. Unmapped inputs retain their
    /// initial resource in later steps; mapped inputs read the previous step's selected output.
    /// Completion is synchronous for every step. The prepared plan, scratch, banks, and inputs
    /// remain tied to this session; failed execution poisons the affected reusable resources.
    ///
    /// # Errors
    ///
    /// Returns a feedback/plan mismatch, missing input or output, or the underlying `ROCm` error.
    pub fn execute_feedback_steps<P>(
        &self,
        feedback: &TensorFeedbackPlan<'_, '_>,
        initial_inputs: &[(ValueId, &RocmTensorInput<'_>)],
        resources: &mut RocmTensorFeedbackResources<'_, '_, 'session>,
        steps: NonZeroUsize,
        memory: &mut P,
    ) -> Result<(), RocmTensorExecutionError>
    where
        P: PcuMemoryProvider<Resource = RocmMemoryResource>,
    {
        resources.completed_steps = 0;
        let result = self.execute_feedback_steps_with_banks(
            resources.prepared,
            feedback,
            initial_inputs,
            &mut resources.scratch,
            &mut resources.banks,
            steps,
            memory,
        );
        if result.is_ok() {
            resources.completed_steps = steps.get();
        }
        result
    }

    /// Execute feedback steps with HIP dispatch batching enabled inside each step.
    ///
    /// Each step reaches final-event quiescence before the next step reads or rotates an output
    /// bank. This keeps reusable bank and scratch state safe while allowing consecutive HIP nodes
    /// within a step to share a stream batch. The synchronous method remains the default.
    ///
    /// # Errors
    ///
    /// Returns the same feedback, plan, input, or execution errors as [`Self::execute_feedback_steps`].
    pub fn execute_feedback_steps_batched<P>(
        &self,
        feedback: &TensorFeedbackPlan<'_, '_>,
        initial_inputs: &[(ValueId, &RocmTensorInput<'_>)],
        resources: &mut RocmTensorFeedbackResources<'_, '_, 'session>,
        steps: NonZeroUsize,
        memory: &mut P,
    ) -> Result<(), RocmTensorExecutionError>
    where
        P: PcuMemoryProvider<Resource = RocmMemoryResource>,
    {
        resources.completed_steps = 0;
        let result = self.execute_feedback_steps_with_banks_batched(
            resources.prepared,
            feedback,
            initial_inputs,
            &mut resources.scratch,
            &mut resources.banks,
            steps,
            memory,
        );
        if result.is_ok() {
            resources.completed_steps = steps.get();
        }
        result
    }

    /// Executes a feedback schedule using already prepared scratch and two output banks.
    ///
    /// This variant lets an application share prepared scratch between fresh-output and banked
    /// benchmark routes without reallocating it. The dialect still controls input feedback and
    /// bank rotation; the caller supplies only the initially bound resources.
    ///
    /// # Errors
    ///
    /// Returns a plan mismatch, missing binding, or the underlying synchronous execution error.
    #[allow(clippy::too_many_arguments)]
    pub fn execute_feedback_steps_with_banks<P>(
        &self,
        prepared: &RocmPreparedTensorGraph<'_>,
        feedback: &TensorFeedbackPlan<'_, '_>,
        initial_inputs: &[(ValueId, &RocmTensorInput<'_>)],
        scratch: &mut RocmTensorScratch<'_, '_, 'session>,
        banks: &mut [RocmTensorOutputBank<'_, '_, 'session>; 2],
        steps: NonZeroUsize,
        memory: &mut P,
    ) -> Result<(), RocmTensorExecutionError>
    where
        P: PcuMemoryProvider<Resource = RocmMemoryResource>,
    {
        self.execute_feedback_steps_with_banks_mode(
            prepared,
            feedback,
            initial_inputs,
            scratch,
            banks,
            steps,
            memory,
            false,
        )
    }

    /// Executes feedback steps with HIP batching inside each graph step.
    ///
    /// Every step waits for its final batch event before the next bank is read or overwritten.
    ///
    /// # Errors
    ///
    /// Returns the same plan mismatch, missing binding, or execution errors as the synchronous
    /// banked feedback path.
    #[allow(clippy::too_many_arguments)]
    pub fn execute_feedback_steps_with_banks_batched<P>(
        &self,
        prepared: &RocmPreparedTensorGraph<'_>,
        feedback: &TensorFeedbackPlan<'_, '_>,
        initial_inputs: &[(ValueId, &RocmTensorInput<'_>)],
        scratch: &mut RocmTensorScratch<'_, '_, 'session>,
        banks: &mut [RocmTensorOutputBank<'_, '_, 'session>; 2],
        steps: NonZeroUsize,
        memory: &mut P,
    ) -> Result<(), RocmTensorExecutionError>
    where
        P: PcuMemoryProvider<Resource = RocmMemoryResource>,
    {
        self.execute_feedback_steps_with_banks_mode(
            prepared,
            feedback,
            initial_inputs,
            scratch,
            banks,
            steps,
            memory,
            true,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn execute_feedback_steps_with_banks_mode<P>(
        &self,
        prepared: &RocmPreparedTensorGraph<'_>,
        feedback: &TensorFeedbackPlan<'_, '_>,
        initial_inputs: &[(ValueId, &RocmTensorInput<'_>)],
        scratch: &mut RocmTensorScratch<'_, '_, 'session>,
        banks: &mut [RocmTensorOutputBank<'_, '_, 'session>; 2],
        steps: NonZeroUsize,
        memory: &mut P,
        batched: bool,
    ) -> Result<(), RocmTensorExecutionError>
    where
        P: PcuMemoryProvider<Resource = RocmMemoryResource>,
    {
        if !std::ptr::eq(feedback.execution_plan(), prepared.tensor_plan())
            || feedback.bank_count() != 2
        {
            return Err(RocmTensorExecutionError::FeedbackPlanMismatch);
        }
        validate_initial_bindings(feedback, initial_inputs)?;
        for first in banks[0].outputs() {
            for second in banks[1].outputs() {
                if first.resource.may_overlap(&second.resource) {
                    return Err(RocmTensorExecutionError::OutputResourceMismatch);
                }
            }
        }
        for step in 0..steps.get() {
            if let Some(previous_index) = feedback.previous_output_bank(step) {
                let (left, right) = banks.split_at_mut(1);
                let (previous, next) = if previous_index == 0 {
                    (&left[0], &mut right[0])
                } else {
                    (&right[0], &mut left[0])
                };
                let inputs = feedback_inputs(feedback, step, initial_inputs, Some(previous))?;
                if batched {
                    self.execute_prepared_outputs_into_bank_batched(
                        prepared, &inputs, scratch, next, memory,
                    )?;
                } else {
                    self.execute_prepared_outputs_into_bank(
                        prepared, &inputs, scratch, next, memory,
                    )?;
                }
            } else {
                let inputs = feedback_inputs(feedback, step, initial_inputs, None)?;
                if batched {
                    self.execute_prepared_outputs_into_bank_batched(
                        prepared,
                        &inputs,
                        scratch,
                        &mut banks[feedback.output_bank(step)],
                        memory,
                    )?;
                } else {
                    self.execute_prepared_outputs_into_bank(
                        prepared,
                        &inputs,
                        scratch,
                        &mut banks[feedback.output_bank(step)],
                        memory,
                    )?;
                }
            }
        }
        Ok(())
    }
}

fn validate_initial_bindings(
    feedback: &TensorFeedbackPlan<'_, '_>,
    initial_inputs: &[(ValueId, &RocmTensorInput<'_>)],
) -> Result<(), RocmTensorExecutionError> {
    let required = feedback.execution_plan().input_values();
    for (index, (value, _)) in initial_inputs.iter().enumerate() {
        if !required.contains(value) {
            return Err(TensorError::ExtraInput(*value).into());
        }
        if initial_inputs[..index]
            .iter()
            .any(|(prior, _)| prior == value)
        {
            return Err(TensorError::DuplicateInput(*value).into());
        }
    }
    for &value in required {
        if !initial_inputs
            .iter()
            .any(|(provided, _)| *provided == value)
        {
            return Err(TensorError::MissingInput(value).into());
        }
    }
    Ok(())
}

fn feedback_inputs<'a, 'session>(
    feedback: &TensorFeedbackPlan<'_, '_>,
    step: usize,
    initial_inputs: &'a [(ValueId, &'a RocmTensorInput<'session>)],
    previous: Option<&'a RocmTensorOutputBank<'_, '_, 'session>>,
) -> Result<SmallVec<[(ValueId, &'a RocmTensorInput<'session>); 8]>, RocmTensorExecutionError> {
    let mut inputs = SmallVec::new();
    for &input in feedback.execution_plan().input_values() {
        let resource = match feedback
            .input_source(input, step)
            .ok_or(RocmTensorExecutionError::MissingResource(input))?
        {
            TensorFeedbackInput::Initial { .. } => initial_inputs
                .iter()
                .find(|(value, _)| *value == input)
                .map(|(_, resource)| *resource)
                .ok_or(RocmTensorExecutionError::MissingResource(input))?,
            TensorFeedbackInput::PreviousOutput { output, .. } => previous
                .and_then(|bank| bank.output(output))
                .ok_or(RocmTensorExecutionError::MissingResource(output))?,
        };
        inputs.push((input, resource));
    }
    Ok(inputs)
}

impl<'session> RocmTensorFeedbackResources<'_, '_, 'session> {
    /// Returns outputs from one successfully completed feedback run, in selected plan output order.
    ///
    /// Returns an error if the requested step was not completed by the most recent successful
    /// [`RocmTensorAssessor::execute_feedback_steps`] call. Later execution can overwrite this
    /// bank; callers must not retain its references across a mutable borrow of this resource set.
    ///
    /// # Errors
    ///
    /// Returns [`RocmTensorExecutionError::FeedbackStepUnavailable`] if the requested step was
    /// not completed by the latest run or its bank has since been overwritten.
    pub fn outputs_for_step(
        &self,
        step: usize,
    ) -> Result<&[RocmTensorInput<'session>], RocmTensorExecutionError> {
        validate_completed_step(step, self.completed_steps)?;
        Ok(self.banks[step % 2].outputs())
    }
}

fn validate_completed_step(
    requested: usize,
    completed: usize,
) -> Result<(), RocmTensorExecutionError> {
    let first_available = completed.saturating_sub(2);
    if (first_available..completed).contains(&requested) {
        Ok(())
    } else {
        Err(RocmTensorExecutionError::FeedbackStepUnavailable {
            requested,
            first_available,
            completed,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn feedback_outputs_require_a_completed_step() {
        assert!(validate_completed_step(0, 2).is_ok());
        assert!(validate_completed_step(1, 2).is_ok());
        assert!(matches!(
            validate_completed_step(2, 2),
            Err(RocmTensorExecutionError::FeedbackStepUnavailable {
                requested: 2,
                first_available: 0,
                completed: 2
            })
        ));
        assert!(matches!(
            validate_completed_step(0, 0),
            Err(RocmTensorExecutionError::FeedbackStepUnavailable {
                requested: 0,
                first_available: 0,
                completed: 0
            })
        ));
        assert!(validate_completed_step(1, 3).is_ok());
        assert!(validate_completed_step(2, 3).is_ok());
        assert!(matches!(
            validate_completed_step(0, 3),
            Err(RocmTensorExecutionError::FeedbackStepUnavailable {
                requested: 0,
                first_available: 1,
                completed: 3
            })
        ));
    }
}
