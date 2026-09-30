//! Owned automatic execution binding for prepared `CUDA` tensor graphs.

use std::num::NonZeroUsize;

#[rustfmt::skip]
use fusion_pcu::{
    PcuMemoryPoolId,
    PcuMemoryProvider,
};
#[rustfmt::skip]
use fusion_pcu::dialect::tensor::{
    Tensor,
    TensorFeedbackPlan,
    ValueId,
};
use smallvec::SmallVec;

use super::validate_initial_bindings;
#[rustfmt::skip]
use super::super::{
    CudaMemoryResource,
    CudaPreparedTensorGraph,
    CudaTensorAssessor,
    CudaTensorExecutionError,
    CudaTensorInput,
    CudaTensorOwnedOutput,
    CudaTensorOutputBank,
    CudaTensorScratch,
};

/// Unforgeable proof for one exact prepared plan, session, and pool.
pub(in crate::tensor) struct ResidentFeedbackProof<'prepared, 'session, 'graph> {
    prepared: &'prepared CudaPreparedTensorGraph<'graph>,
    session: &'session super::super::CudaOwnedDispatchBackend,
    pool: PcuMemoryPoolId,
}

impl<'prepared, 'session, 'graph> ResidentFeedbackProof<'prepared, 'session, 'graph> {
    const fn new(
        prepared: &'prepared CudaPreparedTensorGraph<'graph>,
        session: &'session super::super::CudaOwnedDispatchBackend,
        pool: PcuMemoryPoolId,
    ) -> Self {
        Self {
            prepared,
            session,
            pool,
        }
    }

    pub(in crate::tensor) fn matches(
        &self,
        prepared: &CudaPreparedTensorGraph<'_>,
        session: &super::super::CudaOwnedDispatchBackend,
        pool: PcuMemoryPoolId,
    ) -> bool {
        std::ptr::eq(self.prepared, prepared)
            && std::ptr::eq(self.session, session)
            && self.pool == pool
    }

    pub(in crate::tensor) const fn pool(&self) -> PcuMemoryPoolId {
        self.pool
    }
}

struct ExecutionResources<'plan, 'graph, 'session> {
    scratch: CudaTensorScratch<'plan, 'graph, 'session>,
    banks: SmallVec<[CudaTensorOutputBank<'plan, 'graph, 'session>; 2]>,
}

/// An owned, automatically validated execution binding for a prepared tensor graph.
pub struct CudaTensorExecution<'prepared, 'graph, 'session, P> {
    assessor: &'session CudaTensorAssessor<'session>,
    prepared: &'prepared CudaPreparedTensorGraph<'graph>,
    feedback: Option<TensorFeedbackPlan<'prepared, 'graph>>,
    inputs: Vec<(ValueId, CudaTensorInput<'session>)>,
    resources: ExecutionResources<'prepared, 'graph, 'session>,
    memory: P,
    pool: PcuMemoryPoolId,
    routes: Vec<Route>,
    proof: ResidentFeedbackProof<'prepared, 'session, 'graph>,
    proof_dirty: bool,
    completed_steps: usize,
}

impl<'session> CudaTensorAssessor<'session> {
    /// Binds owned inputs, reusable resources, and a memory provider for automatic execution.
    ///
    /// # Errors
    ///
    /// Returns invalid input, plan, pool, alias, or allocation errors.
    pub fn bind<'prepared, 'graph, P>(
        &'session self,
        prepared: &'prepared CudaPreparedTensorGraph<'graph>,
        inputs: Vec<(ValueId, CudaTensorInput<'session>)>,
        pool: PcuMemoryPoolId,
        memory: P,
    ) -> Result<CudaTensorExecution<'prepared, 'graph, 'session, P>, CudaTensorExecutionError>
    where
        P: PcuMemoryProvider<Resource = CudaMemoryResource>,
    {
        self.bind_inner(prepared, None, inputs, pool, memory)
    }

    /// Uploads host inputs and binds them with automatic resource preparation.
    ///
    /// # Errors
    ///
    /// Returns an upload, input validation, plan, pool, alias, or allocation error.
    pub fn bind_host<'prepared, 'graph, P>(
        &'session self,
        prepared: &'prepared CudaPreparedTensorGraph<'graph>,
        inputs: &[(ValueId, Tensor)],
        pool: PcuMemoryPoolId,
        mut memory: P,
    ) -> Result<CudaTensorExecution<'prepared, 'graph, 'session, P>, CudaTensorExecutionError>
    where
        P: PcuMemoryProvider<Resource = CudaMemoryResource>,
    {
        let mut owned = Vec::with_capacity(inputs.len());
        for (value, tensor) in inputs {
            owned.push((*value, self.upload_input(tensor, pool, &mut memory)?));
        }
        self.bind(prepared, owned, pool, memory)
    }

    /// Binds an automatic two-bank feedback schedule and all resources it needs.
    ///
    /// # Errors
    ///
    /// Returns an invalid feedback mapping, input, pool, alias, or allocation error.
    pub fn bind_feedback<'prepared, 'graph, P>(
        &'session self,
        prepared: &'prepared CudaPreparedTensorGraph<'graph>,
        mappings: &[(ValueId, ValueId)],
        inputs: Vec<(ValueId, CudaTensorInput<'session>)>,
        pool: PcuMemoryPoolId,
        memory: P,
    ) -> Result<CudaTensorExecution<'prepared, 'graph, 'session, P>, CudaTensorExecutionError>
    where
        P: PcuMemoryProvider<Resource = CudaMemoryResource>,
    {
        if mappings.is_empty() {
            return self.bind(prepared, inputs, pool, memory);
        }
        let feedback = TensorFeedbackPlan::new(prepared.tensor_plan(), mappings)?;
        self.bind_inner(prepared, Some(feedback), inputs, pool, memory)
    }

    fn bind_inner<'prepared, 'graph, P>(
        &'session self,
        prepared: &'prepared CudaPreparedTensorGraph<'graph>,
        feedback: Option<TensorFeedbackPlan<'prepared, 'graph>>,
        inputs: Vec<(ValueId, CudaTensorInput<'session>)>,
        pool: PcuMemoryPoolId,
        mut memory: P,
    ) -> Result<CudaTensorExecution<'prepared, 'graph, 'session, P>, CudaTensorExecutionError>
    where
        P: PcuMemoryProvider<Resource = CudaMemoryResource>,
    {
        let feedback_view = feedback.as_ref();
        let empty_plan = TensorFeedbackPlan::new(prepared.tensor_plan(), &[])?;
        let check_plan = feedback_view.unwrap_or(&empty_plan);
        let refs = input_refs(&inputs);
        validate_initial_bindings(check_plan, &refs)?;
        drop(refs);
        let scratch = self.prepare_scratch(prepared, pool, &mut memory)?;
        let bank_count = if feedback.is_some() { 2 } else { 1 };
        let mut banks: SmallVec<[CudaTensorOutputBank<'prepared, 'graph, 'session>; 2]> =
            SmallVec::new();
        for _ in 0..bank_count {
            banks.push(self.prepare_output_bank(prepared, pool, &mut memory)?);
        }
        if banks.len() == 2 {
            for first in banks[0].outputs() {
                for second in banks[1].outputs() {
                    if first.resource.may_overlap(&second.resource) {
                        return Err(CudaTensorExecutionError::OutputResourceMismatch);
                    }
                }
            }
        }
        let resources = ExecutionResources { scratch, banks };
        let routes = feedback_routes(feedback_view, &inputs, prepared)?;
        for step in 0..3 {
            if feedback.is_none() && step > 0 {
                break;
            }
            let previous = feedback_view
                .and_then(|plan| plan.previous_output_bank(step))
                .map(|index| &resources.banks[index]);
            let supplied = routed_inputs(&routes, &inputs, previous)?;
            let destination = feedback_view.map_or(0, |plan| plan.output_bank(step));
            self.validate_execution_sources(
                prepared,
                &[],
                &supplied,
                pool,
                Some(&resources.scratch),
                Some(&resources.banks[destination]),
            )?;
        }
        let proof = ResidentFeedbackProof::new(prepared, self.session, pool);
        Ok(CudaTensorExecution {
            assessor: self,
            prepared,
            feedback,
            inputs,
            resources,
            memory,
            pool,
            routes,
            proof,
            proof_dirty: false,
            completed_steps: 0,
        })
    }
}

impl<'session, P> CudaTensorExecution<'_, '_, 'session, P>
where
    P: PcuMemoryProvider<Resource = CudaMemoryResource>,
{
    /// Updates one bound input. A changed provider binding triggers full revalidation next run.
    ///
    /// # Errors
    ///
    /// Returns a missing input, shape error, or provider transfer error.
    pub fn update_input(
        &mut self,
        value: ValueId,
        tensor: &Tensor,
    ) -> Result<(), CudaTensorExecutionError> {
        self.completed_steps = 0;
        let (_, input) = self
            .inputs
            .iter_mut()
            .find(|(id, _)| *id == value)
            .ok_or(CudaTensorExecutionError::MissingResource(value))?;
        let prior = input.resource.clone_for_tensor_input();
        let result = self.assessor.update_input(input, tensor, &mut self.memory);
        if !input.resource.same_binding(&prior) {
            self.proof_dirty = true;
        }
        result
    }

    /// Replaces an owned input resource and revalidates it before its next use.
    ///
    /// # Errors
    ///
    /// Returns an error when `value` is not a graph input.
    pub fn rebind_input(
        &mut self,
        value: ValueId,
        input: CudaTensorInput<'session>,
    ) -> Result<(), CudaTensorExecutionError> {
        let slot = self
            .inputs
            .iter_mut()
            .find(|(id, _)| *id == value)
            .ok_or(CudaTensorExecutionError::MissingResource(value))?;
        slot.1 = input;
        self.completed_steps = 0;
        self.proof_dirty = true;
        Ok(())
    }

    /// Executes one synchronous step.
    ///
    /// # Errors
    ///
    /// Returns a binding, validation, provider, or execution error.
    pub fn execute(&mut self) -> Result<(), CudaTensorExecutionError> {
        self.execute_steps(NonZeroUsize::MIN)
    }

    /// Executes a positive number of synchronous feedback steps.
    ///
    /// # Errors
    ///
    /// Returns a binding, validation, provider, or execution error.
    pub fn execute_steps(&mut self, steps: NonZeroUsize) -> Result<(), CudaTensorExecutionError> {
        self.completed_steps = 0;
        if self.proof_dirty {
            for step in 0..if self.feedback.is_some() { 3 } else { 1 } {
                let previous = self
                    .feedback
                    .as_ref()
                    .and_then(|plan| plan.previous_output_bank(step))
                    .map(|index| &self.resources.banks[index]);
                let supplied = routed_inputs(&self.routes, &self.inputs, previous)?;
                let destination = self
                    .feedback
                    .as_ref()
                    .map_or(0, |plan| plan.output_bank(step));
                self.assessor.validate_execution_sources(
                    self.prepared,
                    &[],
                    &supplied,
                    self.pool,
                    Some(&self.resources.scratch),
                    Some(&self.resources.banks[destination]),
                )?;
            }
            if self.resources.banks.len() == 2 {
                for first in self.resources.banks[0].outputs() {
                    for second in self.resources.banks[1].outputs() {
                        if first.resource.may_overlap(&second.resource) {
                            return Err(CudaTensorExecutionError::OutputResourceMismatch);
                        }
                    }
                }
            }
            self.proof_dirty = false;
        }
        let mut completed = 0;
        for step in 0..steps.get() {
            let previous = self
                .feedback
                .as_ref()
                .and_then(|plan| plan.previous_output_bank(step));
            let destination_index = self
                .feedback
                .as_ref()
                .map_or(0, |plan| plan.output_bank(step));
            if let Some(previous_index) = previous {
                let (left, right) = self.resources.banks.split_at_mut(1);
                let (source, destination) = if previous_index == 0 {
                    (&left[0], &mut right[0])
                } else {
                    (&right[0], &mut left[0])
                };
                let supplied = routed_inputs(&self.routes, &self.inputs, Some(source))?;
                self.assessor
                    .execute_prepared_outputs_into_bank_batched_proven(
                        self.prepared,
                        &supplied,
                        &mut self.resources.scratch,
                        destination,
                        &mut self.memory,
                        &self.proof,
                    )?;
            } else {
                let supplied = routed_inputs(&self.routes, &self.inputs, None)?;
                self.assessor
                    .execute_prepared_outputs_into_bank_batched_proven(
                        self.prepared,
                        &supplied,
                        &mut self.resources.scratch,
                        &mut self.resources.banks[destination_index],
                        &mut self.memory,
                        &self.proof,
                    )?;
            }
            completed += 1;
        }
        self.completed_steps = completed;
        Ok(())
    }

    /// Returns selected outputs from the latest successful run.
    ///
    /// # Errors
    ///
    /// Returns an error before a successful run or when the output bank is poisoned.
    pub fn outputs(&self) -> Result<&[CudaTensorInput<'_>], CudaTensorExecutionError> {
        let bank = completed_output_bank_index(
            self.completed_steps,
            self.feedback.is_some(),
            self.resources.banks.len(),
        )?;
        if self.resources.banks[bank].poisoned {
            return Err(CudaTensorExecutionError::OutputResourceMismatch);
        }
        Ok(self.resources.banks[bank].outputs())
    }

    /// Consumes this execution and transfers its latest completed outputs into owned typed
    /// tensors. This performs no host readback; the returned buffers retain the exact output
    /// allocations and can be passed directly to `PcuDeviceArgument`.
    ///
    /// The execution must have completed successfully. Pending, failed, or poisoned output
    /// storage is never exposed. The output vector preserves the prepared graph's output order
    /// and includes each `ValueId` beside its tensor.
    ///
    /// # Errors
    ///
    /// Returns an unavailable-output error before a successful step or a storage error if the
    /// completed output bank is poisoned or inconsistent with its shape metadata.
    pub fn into_outputs(self) -> Result<Vec<CudaTensorOwnedOutput>, CudaTensorExecutionError> {
        let Self {
            prepared,
            completed_steps,
            feedback,
            resources,
            ..
        } = self;
        let bank_index = completed_output_bank_index(
            completed_steps,
            feedback.is_some(),
            resources.banks.len(),
        )?;
        if resources.banks[bank_index].poisoned {
            return Err(CudaTensorExecutionError::OutputResourceMismatch);
        }
        if prepared.outputs.len() != resources.banks[bank_index].outputs.len() {
            return Err(CudaTensorExecutionError::OutputResourceMismatch);
        }
        let bank = resources
            .banks
            .into_iter()
            .nth(bank_index)
            .ok_or(CudaTensorExecutionError::OutputResourceMismatch)?;
        prepared
            .outputs
            .iter()
            .copied()
            .zip(bank.outputs)
            .map(|(value, output)| Ok((value, output.into_device_tensor()?)))
            .collect()
    }

    /// Downloads one selected output from the latest successful run.
    ///
    /// # Errors
    ///
    /// Returns an unknown or unavailable output, or a provider readback error.
    pub fn read_output(&mut self, value: ValueId) -> Result<Tensor, CudaTensorExecutionError> {
        let bank = completed_output_bank_index(
            self.completed_steps,
            self.feedback.is_some(),
            self.resources.banks.len(),
        )?;
        let index = self
            .prepared
            .outputs
            .iter()
            .position(|output| *output == value)
            .ok_or(CudaTensorExecutionError::MissingResource(value))?;
        if self.resources.banks[bank].poisoned {
            return Err(CudaTensorExecutionError::OutputResourceMismatch);
        }
        let output = self.resources.banks[bank]
            .outputs()
            .get(index)
            .ok_or(CudaTensorExecutionError::MissingResource(value))?;
        self.assessor
            .download_output(output, self.pool, &mut self.memory)
    }
}

const fn completed_output_bank_index(
    completed_steps: usize,
    feedback: bool,
    bank_count: usize,
) -> Result<usize, CudaTensorExecutionError> {
    if completed_steps == 0 {
        return Err(CudaTensorExecutionError::FeedbackStepUnavailable {
            requested: 0,
            first_available: 0,
            completed: 0,
        });
    }
    let bank = if feedback {
        (completed_steps - 1) % 2
    } else {
        0
    };
    if bank >= bank_count {
        return Err(CudaTensorExecutionError::OutputResourceMismatch);
    }
    Ok(bank)
}

impl<P> fusion_pcu::dialect::tensor::TensorExecution for CudaTensorExecution<'_, '_, '_, P>
where
    P: PcuMemoryProvider<Resource = CudaMemoryResource>,
{
    type Error = CudaTensorExecutionError;
    fn update_input(&mut self, input: ValueId, value: &Tensor) -> Result<(), Self::Error> {
        CudaTensorExecution::update_input(self, input, value)
    }
    fn execute_steps(&mut self, steps: NonZeroUsize) -> Result<(), Self::Error> {
        CudaTensorExecution::execute_steps(self, steps)
    }
    fn read_output(&mut self, output: ValueId) -> Result<Tensor, Self::Error> {
        CudaTensorExecution::read_output(self, output)
    }
}

fn input_refs<'a, 'session>(
    inputs: &'a [(ValueId, CudaTensorInput<'session>)],
) -> SmallVec<[(ValueId, &'a CudaTensorInput<'session>); 16]> {
    inputs.iter().map(|(id, input)| (*id, input)).collect()
}

#[derive(Clone, Copy)]
struct Route {
    value: ValueId,
    input_index: usize,
    output_index: Option<usize>,
}

fn feedback_routes(
    feedback: Option<&TensorFeedbackPlan<'_, '_>>,
    inputs: &[(ValueId, CudaTensorInput<'_>)],
    prepared: &CudaPreparedTensorGraph<'_>,
) -> Result<Vec<Route>, CudaTensorExecutionError> {
    let required = feedback.map_or_else(
        || prepared.tensor_plan().input_values(),
        |feedback| feedback.execution_plan().input_values(),
    );
    required
        .iter()
        .map(|value| {
            let input_index = inputs
                .iter()
                .position(|(id, _)| id == value)
                .ok_or(CudaTensorExecutionError::MissingResource(*value))?;
            let output_index = feedback
                .and_then(|plan| plan.bindings().iter().find(|b| b.input == *value))
                .map(|b| {
                    prepared
                        .outputs
                        .iter()
                        .position(|id| *id == b.output)
                        .ok_or(CudaTensorExecutionError::MissingResource(b.output))
                })
                .transpose()?;
            Ok(Route {
                value: *value,
                input_index,
                output_index,
            })
        })
        .collect()
}

fn routed_inputs<'a, 'session>(
    routes: &[Route],
    inputs: &'a [(ValueId, CudaTensorInput<'session>)],
    previous: Option<&'a super::super::CudaTensorOutputBank<'_, '_, 'session>>,
) -> Result<SmallVec<[(ValueId, &'a CudaTensorInput<'session>); 16]>, CudaTensorExecutionError> {
    routes
        .iter()
        .map(|route| {
            let input = match (previous, route.output_index) {
                (Some(bank), Some(index)) => bank
                    .outputs()
                    .get(index)
                    .ok_or(CudaTensorExecutionError::MissingResource(route.value))?,
                _ => &inputs[route.input_index].1,
            };
            Ok((route.value, input))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{CudaTensorExecutionError, completed_output_bank_index};

    #[test]
    fn output_bank_is_unavailable_until_a_successful_step() {
        assert!(matches!(
            completed_output_bank_index(0, false, 1),
            Err(CudaTensorExecutionError::FeedbackStepUnavailable { .. })
        ));
    }

    #[test]
    fn feedback_output_bank_tracks_the_last_completed_step() {
        assert_eq!(completed_output_bank_index(1, true, 2).unwrap(), 0);
        assert_eq!(completed_output_bank_index(2, true, 2).unwrap(), 1);
        assert_eq!(completed_output_bank_index(3, true, 2).unwrap(), 0);
        assert!(matches!(
            completed_output_bank_index(2, true, 1),
            Err(CudaTensorExecutionError::OutputResourceMismatch)
        ));
    }
}
