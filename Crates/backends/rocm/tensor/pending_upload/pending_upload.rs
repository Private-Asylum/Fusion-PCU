//! Owned host endpoints covered by the unchanged terminal fixed-dispatch schedule.
#[rustfmt::skip]
use crate::{
    HipError,
    HipStreamHandle,
    RetainedHostUploads,
    RocmMemoryResource,
};
#[rustfmt::skip]
use super::{
    RocmTensorAssessor,
    RocmTensorExecutionError,
    RocmTensorInputRef,
    RocmOwnedPreparedTensorGraph,
    byte_len_for_size,
    scalar_layout,
};
#[rustfmt::skip]
use fusion_pcu::{
    PcuMemoryPoolId,
    PcuDeviceTensor,
    PcuMemoryAccess,
    PcuMemoryProvider,
    PcuScalar,
};
#[rustfmt::skip]
use fusion_pcu::dialect::tensor::{
    OpDescriptor,
    ValueId,
};
use std::rc::Rc;

pub(super) struct StagingEligibility {
    terminal_dispatch: bool,
    stream: Option<HipStreamHandle>,
}

impl StagingEligibility {
    pub(super) const fn supported(&self) -> bool {
        self.terminal_dispatch
    }

    pub(super) const fn disabled() -> Self {
        Self {
            terminal_dispatch: false,
            stream: None,
        }
    }

    pub(super) fn anchor_stream(&mut self, stream: &HipStreamHandle) {
        self.stream = Some(stream.clone());
    }

    pub(super) fn new(prepared: &RocmOwnedPreparedTensorGraph) -> Self {
        Self {
            terminal_dispatch: eligible(prepared),
            stream: None,
        }
    }
}

fn eligible(prepared: &RocmOwnedPreparedTensorGraph) -> bool {
    if prepared.data.input_values.len() > RetainedHostUploads::INLINE_CAPACITY
        || prepared.data.outputs.len() != 1
        || prepared.data.requires_blas
        || prepared.data.fixed_dispatches.len() != prepared.data.node_values.len()
    {
        return false;
    }
    // Prepared fixed dispatches run on the assessor stream and, with batching disabled,
    // execute_elementwise waits their completion before returning success. Their numerical
    // operation/policy is irrelevant to upload lifetime: the terminal queue event orders all
    // earlier copies. Non-source nodes without this concrete implementation proof are refused.
    // In particular, identity/source-only schedules never establish a consumer terminal event.
    if prepared.data.batch_native_matmuls {
        return false;
    }
    let mut terminal_dispatch = false;
    for (&value, dispatch) in prepared
        .data
        .node_values
        .iter()
        .zip(&prepared.data.fixed_dispatches)
    {
        let Ok(node) = prepared.program.graph().node(value) else {
            return false;
        };
        if matches!(
            node.op,
            OpDescriptor::Input | OpDescriptor::Constant(_) | OpDescriptor::Uniform { .. }
        ) {
            continue;
        }
        let Some(dispatch) = dispatch else {
            return false;
        };
        if dispatch.value != value {
            return false;
        }
        terminal_dispatch = true;
    }
    terminal_dispatch
}

/// One fully described host payload for an already initialized tensor input allocation.
/// The aggregate validates every binding before ordering any upload.
#[derive(Clone, Copy)]
pub struct RocmHostedTensorInput<'input> {
    pub value: ValueId,
    pub resource: &'input RocmMemoryResource,
    pub shape: &'input [usize],
    pub source: &'input [u8],
}

impl<'session> RocmTensorAssessor<'session> {
    /// Orders owned host endpoints and executes a cold-proved terminal-dispatch schedule.
    ///
    /// `None` means an unsupported schedule or cold endpoint/scratch/cache bank, before any work.
    /// All input metadata, graph roles, spans, access and aliases are checked before upload.
    /// Public transfers and ordinary borrowed input availability rules remain unchanged.
    ///
    /// # Errors
    /// Returns input/preflight, allocation, arithmetic, backend or completion errors. An error
    /// after enqueue is terminal for this attempt and must never trigger a fallback execution.
    #[expect(
        clippy::too_many_lines,
        reason = "Keep all preflight gates before uploads and the terminal ownership proof together."
    )]
    pub fn execute_owned_program_output_from_host_staging<'input, T, P, const N: usize>(
        &'input self,
        prepared: &RocmOwnedPreparedTensorGraph,
        inputs: &[RocmHostedTensorInput<'input>],
        pool: PcuMemoryPoolId,
        memory: &mut P,
    ) -> Result<Option<PcuDeviceTensor<T, RocmMemoryResource>>, RocmTensorExecutionError>
    where
        T: PcuScalar,
        P: PcuMemoryProvider<Resource = RocmMemoryResource>,
        'session: 'input,
    {
        #[cfg(feature = "insights")]
        prepared.begin_guarded_attempt();
        if !prepared.staging.terminal_dispatch
            || inputs.is_empty()
            || !prepared
                .staging
                .stream
                .as_ref()
                .is_some_and(|stream| Rc::ptr_eq(&stream.inner, &self.state().stream.inner))
        {
            return Ok(None);
        }
        if inputs.len() > N || inputs.len() != prepared.data.input_values.len() {
            return Err(RocmTensorExecutionError::InputResourceMismatch);
        }
        super::validate_owned_scalar_profile::<T>(prepared.data.scalar_type)?;
        let (size, _) = scalar_layout(T::TYPE)?;
        let mut descriptors: [Option<RocmTensorInputRef<'input>>; N] =
            std::array::from_fn(|_| None);
        let mut endpoints_ready = true;
        for (index, input) in inputs.iter().enumerate() {
            let required = byte_len_for_size(input.shape, size)?;
            if required == 0 || required != input.source.len() {
                return Err(RocmTensorExecutionError::InputResourceMismatch);
            }
            if input.resource.access() != PcuMemoryAccess::ReadWrite {
                return Err(RocmTensorExecutionError::InputAccessMismatch);
            }
            descriptors[index] =
                Some(self.borrow_resource_input_ref(input.resource, input.shape, T::TYPE, pool)?);
            input
                .resource
                .validate_access_available()
                .map_err(RocmTensorExecutionError::Completion)?;
            if inputs[..index]
                .iter()
                .any(|other| input.resource.may_overlap(other.resource))
            {
                return Err(RocmTensorExecutionError::InputResourceMismatch);
            }
            let endpoint = input
                .resource
                .device_buffer()
                .allocation
                .host_transfer
                .try_borrow()
                .map_err(|_| RocmTensorExecutionError::Completion(HipError::Busy))?;
            endpoints_ready &= endpoint.capacity() >= required;
        }
        let bindings: smallvec::SmallVec<[(ValueId, &RocmTensorInputRef<'input>); N]> = inputs
            .iter()
            .enumerate()
            .map(|(index, input)| {
                Ok((
                    input.value,
                    descriptors[index]
                        .as_ref()
                        .ok_or(RocmTensorExecutionError::InputResourceMismatch)?,
                ))
            })
            .collect::<Result<_, RocmTensorExecutionError>>()?;
        self.validate_execution_sources_view(&prepared.view(), &[], &bindings, pool)?;
        let resources: smallvec::SmallVec<[&RocmMemoryResource; N]> =
            inputs.iter().map(|input| input.resource).collect();
        if !prepared.scratch.preflight_initialized(
            self.session.tensor_runtime(),
            pool,
            &resources,
        )? || !endpoints_ready
        {
            return Ok(None);
        }
        let retained = prepared
            .data
            .fixed_dispatches
            .iter()
            .enumerate()
            .filter(|(_, dispatch)| dispatch.is_some())
            .all(|(index, _)| {
                prepared
                    .retained_fixed
                    .get(index)
                    .and_then(Option::as_ref)
                    .is_some_and(|retained| retained.uses_stream(&self.state().stream))
            });
        if !retained {
            // Metadata-only programs preserve their conservative cache snapshot route.
            // Release before provider callbacks; exact-stream retained programs need no snapshot.
            let Ok(cache) = self.state().add_dispatches.try_borrow() else {
                return Ok(None);
            };
            if prepared
                .data
                .fixed_dispatches
                .iter()
                .flatten()
                .any(|dispatch| !cache.iter().any(|(key, _)| *key == dispatch.cache_key))
            {
                return Ok(None);
            }
        }
        let mut uploads = RetainedHostUploads::new(self.state().stream.clone());
        for input in inputs {
            uploads
                .enqueue(input.resource.device_buffer(), input.source)
                .map_err(RocmTensorExecutionError::Completion)?;
        }
        let terminal = self
            .execute_owned_program_output_from_inputs::<T, P>(prepared, &bindings, pool, memory);
        if terminal.is_ok() {
            // Every upload precedes the existing waited terminal dispatch on the anchored
            // stream. All arithmetic/status gates and output publication are unchanged.
            uploads.release_after_terminal_launch();
        }
        // A failed attempt must fence or quarantine endpoints before returning its error.
        drop(uploads);
        terminal.map(Some)
    }
}

#[cfg(test)]
#[path = "tests/tests.rs"]
mod tests;
