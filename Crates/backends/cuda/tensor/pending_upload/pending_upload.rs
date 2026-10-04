//! Private owned upload endpoints ordered on the exact tensor execution stream.
#[rustfmt::skip]
use crate::{
    AllocationAccessKind,
    AllocationAccessState,
    CudaError,
    CudaMemoryResource,
    CudaStreamHandle,
    DeviceAccessLease,
    DeviceBuffer,
    ffi::runtime::CUDA_MEMCPY_HOST_TO_DEVICE,
};
#[rustfmt::skip]
use super::{
    CudaTensorAssessor,
    CudaTensorExecutionError,
    CudaTensorInputRef,
    CudaOwnedPreparedTensorGraph,
    TensorDispatchCacheKey,
    TensorDispatchKind,
    byte_len_for_size,
    scalar_layout,
};
#[rustfmt::skip]
use fusion_pcu::{
    PcuMemoryPoolId,
    PcuScalarType,
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
#[rustfmt::skip]
use std::{
    mem,
    rc::Rc,
};

struct PendingUpload {
    lease: Option<DeviceAccessLease>,
    stream: CudaStreamHandle,
}

pub(super) struct StagingEligibility {
    checked_terminal: bool,
    stream: Option<CudaStreamHandle>,
}

impl StagingEligibility {
    pub(super) const fn supported(&self) -> bool {
        self.checked_terminal
    }

    pub(super) const fn disabled() -> Self {
        Self {
            checked_terminal: false,
            stream: None,
        }
    }

    pub(super) fn anchor_stream(&mut self, stream: &CudaStreamHandle) {
        self.stream = Some(stream.clone());
    }

    pub(super) fn new(prepared: &CudaOwnedPreparedTensorGraph) -> Self {
        Self {
            checked_terminal: eligible(prepared),
            stream: None,
        }
    }
}

fn eligible(prepared: &CudaOwnedPreparedTensorGraph) -> bool {
    if prepared.data.outputs.len() != 1
        || prepared.data.requires_blas
        || prepared.data.fixed_dispatches.len() != prepared.data.node_values.len()
    {
        return false;
    }
    let mut checked = false;
    for (&value, dispatch) in prepared
        .data
        .node_values
        .iter()
        .zip(&prepared.data.fixed_dispatches)
    {
        let Ok(node) = prepared.program.graph().node(value) else {
            return false;
        };
        match node.op {
            OpDescriptor::Input | OpDescriptor::Constant(_) | OpDescriptor::Uniform { .. } => (),
            OpDescriptor::Add { .. } | OpDescriptor::Mul { .. } => {
                if !dispatch.as_ref().is_some_and(|dispatch| {
                    matches!(
                        dispatch.cache_key,
                        TensorDispatchCacheKey::Fixed(
                            TensorDispatchKind::CheckedIntegerAdd
                                | TensorDispatchKind::CheckedIntegerMul
                                | TensorDispatchKind::CheckedFloatAdd
                                | TensorDispatchKind::CheckedFloatMul,
                            ..
                        )
                    )
                }) {
                    return false;
                }
                checked = true;
            }
            _ => return false,
        }
    }
    checked
}

impl PendingUpload {
    fn queue(
        assessor: &CudaTensorAssessor<'_>,
        resource: &CudaMemoryResource,
        shape: &[usize],
        scalar_type: PcuScalarType,
        pool: PcuMemoryPoolId,
        source: &[u8],
    ) -> Result<Self, CudaTensorExecutionError> {
        let (size, _) = scalar_layout(scalar_type)?;
        let required = byte_len_for_size(shape, size)?;
        if required == 0 || required != source.len() {
            return Err(CudaTensorExecutionError::InputResourceMismatch);
        }
        assessor.borrow_resource_input_ref(resource, shape, scalar_type, pool)?;
        let destination = resource.device_buffer();
        let stream = &assessor.state().stream;
        stream
            .inner
            .runtime
            .ensure_same_runtime(&destination.allocation.runtime)
            .map_err(CudaTensorExecutionError::Completion)?;
        destination
            .check_range(0, source.len())
            .map_err(CudaTensorExecutionError::Completion)?;
        // Rc/Cell gates cannot race on this host thread. Unlike a same-queue kernel lease,
        // another upload must never overwrite an endpoint still being read by queued DMA.
        destination
            .validate_access_available()
            .map_err(CudaTensorExecutionError::Completion)?;
        let lease = destination
            .acquire_stream_access(stream)
            .map_err(CudaTensorExecutionError::Completion)?;
        let mut scratch = destination
            .allocation
            .host_transfer
            .try_borrow_mut()
            .map_err(|_| CudaTensorExecutionError::Completion(CudaError::Busy))?;
        if scratch.len() < source.len() {
            scratch.resize(source.len(), 0);
        }
        scratch[..source.len()].copy_from_slice(source);
        let pointer = scratch.as_ptr().cast();
        drop(scratch);
        let pending = Self {
            lease: Some(lease),
            stream: stream.clone(),
        };
        // SAFETY: the exact allocation and private Vec endpoint remain retained and protected
        // from host overwrite until known queue completion. Same-stream consumer kernels may
        // acquire ordered leases; foreign queues and ordinary host transfers remain Busy.
        unsafe {
            crate::ffi::invoke_cudaMemcpyAsync(
                &stream.inner.runtime,
                destination.allocation.pointer,
                pointer,
                source.len(),
                CUDA_MEMCPY_HOST_TO_DEVICE,
                stream.inner.raw,
            )
        }
        .map_err(CudaTensorExecutionError::Completion)?;
        Ok(pending)
    }

    fn authorizes(&self, destination: &DeviceBuffer, stream: &CudaStreamHandle) -> bool {
        Rc::ptr_eq(&self.stream.inner, &stream.inner)
            && self.lease.as_ref().is_some_and(|lease| {
                Rc::ptr_eq(&lease.allocation, &destination.allocation)
                    && lease.guard.kind
                        == AllocationAccessKind::Stream(Rc::as_ptr(&stream.inner) as usize)
                    && matches!(lease.guard.access.state.get(),
                        AllocationAccessState::Stream { identity, .. }
                            if identity == Rc::as_ptr(&stream.inner) as usize)
            })
    }

    #[cfg(test)]
    fn borrow_input<'input, 'session: 'input>(
        &'input self,
        assessor: &'input CudaTensorAssessor<'session>,
        resource: &'input CudaMemoryResource,
        shape: &'input [usize],
        scalar_type: PcuScalarType,
        pool: PcuMemoryPoolId,
    ) -> Result<CudaTensorInputRef<'input>, CudaTensorExecutionError> {
        if !self.authorizes(resource.device_buffer(), &assessor.state().stream) {
            return Err(CudaTensorExecutionError::InputResourceMismatch);
        }
        // Normal layout/type/shape/pool/runtime/access validation remains unchanged. This
        // private token is the additional authority for this exact pending resource/queue.
        assessor.borrow_resource_input_ref(resource, shape, scalar_type, pool)
    }

    fn complete_after_schedule<T>(
        mut self,
        eligibility: &StagingEligibility,
        terminal: &Result<T, CudaTensorExecutionError>,
    ) {
        // Called only inside the proposed aggregate after synchronous checked execution on
        // this exact assessor queue. Success is its documented terminal publication proof.
        // Identity/no-dispatch and error routes cannot use this shortcut: Drop fences them.
        if eligibility.checked_terminal && terminal.is_ok() {
            self.lease.take();
        }
    }

    fn release_after_completion(&mut self, completion: &Result<(), CudaError>) {
        if let Some(lease) = self.lease.take()
            && completion.is_err()
        {
            lease.retain_after_unknown_completion();
            mem::forget(self.stream.clone());
        }
    }
}

impl Drop for PendingUpload {
    fn drop(&mut self) {
        if self.lease.is_some() {
            let completion = self.stream.synchronize();
            self.release_after_completion(&completion);
        }
    }
}

#[cfg(test)]
#[path = "tests/tests.rs"]
mod tests;

/// One fully described host payload for an already initialized tensor input allocation.
/// The aggregate validates every binding before ordering any upload.
#[derive(Clone, Copy)]
pub struct CudaHostedTensorInput<'input> {
    pub value: ValueId,
    pub resource: &'input CudaMemoryResource,
    pub shape: &'input [usize],
    pub source: &'input [u8],
}

impl<'session> CudaTensorAssessor<'session> {
    /// Orders owned host endpoints and executes one eligible checked Add/Mul schedule.
    ///
    /// `None` means an unsupported schedule or cold endpoint/scratch bank, before any work.
    /// All input metadata, graph roles, spans, access and aliases are checked before upload.
    /// Public transfers and ordinary borrowed input availability rules remain unchanged.
    ///
    /// # Errors
    /// Returns input/preflight, allocation, arithmetic, backend or completion errors. An error
    /// after enqueue is terminal for this attempt and must never trigger a fallback execution.
    pub fn execute_owned_program_output_from_host_staging<'input, T, P, const N: usize>(
        &'input self,
        prepared: &CudaOwnedPreparedTensorGraph,
        inputs: &[CudaHostedTensorInput<'input>],
        pool: PcuMemoryPoolId,
        memory: &mut P,
    ) -> Result<Option<PcuDeviceTensor<T, CudaMemoryResource>>, CudaTensorExecutionError>
    where
        T: PcuScalar,
        P: PcuMemoryProvider<Resource = CudaMemoryResource>,
        'session: 'input,
    {
        if !prepared.staging.checked_terminal
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
            return Err(CudaTensorExecutionError::InputResourceMismatch);
        }
        super::validate_owned_scalar_profile::<T>(prepared.data.scalar_type)?;
        let (size, _) = scalar_layout(T::TYPE)?;
        let mut descriptors: [Option<CudaTensorInputRef<'input>>; N] =
            std::array::from_fn(|_| None);
        let mut endpoints_ready = true;
        for (index, input) in inputs.iter().enumerate() {
            let required = byte_len_for_size(input.shape, size)?;
            if required == 0 || required != input.source.len() {
                return Err(CudaTensorExecutionError::InputResourceMismatch);
            }
            if input.resource.access() != PcuMemoryAccess::ReadWrite {
                return Err(CudaTensorExecutionError::InputAccessMismatch);
            }
            descriptors[index] =
                Some(self.borrow_resource_input_ref(input.resource, input.shape, T::TYPE, pool)?);
            input
                .resource
                .validate_access_available()
                .map_err(CudaTensorExecutionError::Completion)?;
            if inputs[..index]
                .iter()
                .any(|other| input.resource.may_overlap(other.resource))
            {
                return Err(CudaTensorExecutionError::InputResourceMismatch);
            }
            let endpoint = input
                .resource
                .device_buffer()
                .allocation
                .host_transfer
                .try_borrow()
                .map_err(|_| CudaTensorExecutionError::Completion(CudaError::Busy))?;
            endpoints_ready &= endpoint.capacity() >= required;
        }
        let bindings: smallvec::SmallVec<[(ValueId, &CudaTensorInputRef<'input>); N]> = inputs
            .iter()
            .enumerate()
            .map(|(index, input)| {
                Ok((
                    input.value,
                    descriptors[index]
                        .as_ref()
                        .ok_or(CudaTensorExecutionError::InputResourceMismatch)?,
                ))
            })
            .collect::<Result<_, CudaTensorExecutionError>>()?;
        self.validate_execution_sources_view(&prepared.view(), &[], &bindings, pool)?;
        let resources: smallvec::SmallVec<[&CudaMemoryResource; N]> =
            inputs.iter().map(|input| input.resource).collect();
        if !prepared.scratch.preflight_initialized(
            self.session.tensor_runtime(),
            pool,
            &resources,
        )? || !endpoints_ready
        {
            return Ok(None);
        }
        let mut pending: [Option<PendingUpload>; N] = std::array::from_fn(|_| None);
        for (index, input) in inputs.iter().enumerate() {
            pending[index] = Some(PendingUpload::queue(
                self,
                input.resource,
                input.shape,
                T::TYPE,
                pool,
                input.source,
            )?);
        }
        for (guard, input) in pending[..inputs.len()].iter().zip(inputs) {
            if !guard
                .as_ref()
                .ok_or(CudaTensorExecutionError::InputResourceMismatch)?
                .authorizes(input.resource.device_buffer(), &self.state().stream)
            {
                return Err(CudaTensorExecutionError::InputResourceMismatch);
            }
        }
        let terminal = self
            .execute_owned_program_output_from_inputs::<T, P>(prepared, &bindings, pool, memory);
        for guard in &mut pending[..inputs.len()] {
            guard
                .take()
                .ok_or(CudaTensorExecutionError::InputResourceMismatch)?
                .complete_after_schedule(&prepared.staging, &terminal);
        }
        terminal.map(Some)
    }
}
