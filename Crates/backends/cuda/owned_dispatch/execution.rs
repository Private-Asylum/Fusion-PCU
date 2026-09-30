//! Bounded execution of validated, fault-gated graphs of owned `CUDA` operations.

#[rustfmt::skip]
use std::{
    error::Error,
    fmt,
    rc::Rc,
    sync::Arc,
};

#[rustfmt::skip]
use fusion_pcu::{
    PcuCompletionOutcome,
    PcuBindingAccess,
    PcuExecutionAdmission,
    PcuExecutionFaultGateState,
    PcuExecutionGraphError,
    PcuExecutionNode,
    PcuExecutionNodeState,
    PcuExecutionResourceUse,
    PcuExecutionResourceId,
    PcuExecutionSuccessGate,
    PcuOwnedBinding,
    PcuOwnedCompletion,
    PcuMemoryAccess,
    validate_execution_submission,
};

use crate::DeviceBuffer;
use crate::CudaBatchCompletion;
use crate::CudaCompletionBatch;
use crate::CudaError;
use crate::CudaReadbackId;
use crate::CudaRuntime;
use crate::CudaStreamHandle;
use crate::Cublas;

#[rustfmt::skip]
use super::{
    CudaOwnedCompletion,
    CudaOwnedDispatchError,
    CudaPreparedDispatch,
};

/// One operation scheduled by an owned execution graph.
pub enum CudaOwnedExecutionOperation<'a> {
    /// A prepared GPU dispatch and its owned buffer bindings.
    Dispatch {
        prepared: &'a CudaPreparedDispatch,
        bindings: &'a [PcuOwnedBinding<DeviceBuffer>],
    },
    /// An asynchronous device-to-device copy on a caller-selected stream.
    DeviceCopy {
        stream: &'a CudaStreamHandle,
        source: &'a DeviceBuffer,
        destination: &'a DeviceBuffer,
        bytes: usize,
    },
    /// Copy a device range into host-owned bytes available after this node completes.
    DeviceReadback {
        stream: &'a CudaStreamHandle,
        source: &'a DeviceBuffer,
        source_offset: usize,
        bytes: usize,
    },
    /// An owned host payload copied into a device allocation on a caller-selected stream.
    HostUpload {
        stream: &'a CudaStreamHandle,
        source: Arc<[u8]>,
        destination: &'a DeviceBuffer,
        offset: usize,
    },
    /// An asynchronous cuBLAS SGEMM on its explicitly captured stream.
    Sgemm {
        handle: &'a Cublas,
        stream: &'a CudaStreamHandle,
        transpose_a: bool,
        transpose_b: bool,
        m: usize,
        n: usize,
        k: usize,
        alpha: f32,
        a: &'a DeviceBuffer,
        lda: usize,
        b: &'a DeviceBuffer,
        ldb: usize,
        beta: f32,
        c: &'a DeviceBuffer,
        ldc: usize,
    },
}

impl CudaOwnedExecutionOperation<'_> {
    fn runtime(&self) -> &CudaRuntime {
        match self {
            Self::Dispatch { prepared, .. } => &prepared.runtime,
            Self::DeviceCopy { stream, .. }
            | Self::DeviceReadback { stream, .. }
            | Self::HostUpload { stream, .. }
            | Self::Sgemm { stream, .. } => &stream.inner.runtime,
        }
    }

    const fn stream(&self) -> &CudaStreamHandle {
        match self {
            Self::Dispatch { prepared, .. } => &prepared.stream,
            Self::DeviceCopy { stream, .. }
            | Self::DeviceReadback { stream, .. }
            | Self::HostUpload { stream, .. }
            | Self::Sgemm { stream, .. } => stream,
        }
    }

    const fn checked(&self) -> bool {
        matches!(self, Self::Dispatch { prepared, .. } if prepared.checked_arithmetic)
    }

    fn can_submit_into_event_chain(&self, predecessor: &ExecutionCompletion) -> bool {
        match self {
            Self::Sgemm { handle, .. } => {
                handle.is_usable()
                    || matches!(predecessor, ExecutionCompletion::Batch { completion, .. }
                        if handle.can_extend_from(completion))
            }
            Self::Dispatch { .. }
            | Self::DeviceCopy { .. }
            | Self::DeviceReadback { .. }
            | Self::HostUpload { .. } => true,
        }
    }

    fn can_submit_into_open_batch(&self, batch: &CudaCompletionBatch) -> bool {
        if !std::rc::Rc::ptr_eq(&self.stream().inner, &batch.stream_handle().inner) {
            return false;
        }
        match self {
            Self::Sgemm { handle, .. } => handle.can_extend_in_batch(batch),
            Self::Dispatch { .. }
            | Self::DeviceCopy { .. }
            | Self::DeviceReadback { .. }
            | Self::HostUpload { .. } => true,
        }
    }

    fn can_submit_directly(&self) -> bool {
        match self {
            Self::Sgemm { handle, .. } => handle.is_usable(),
            Self::Dispatch { .. }
            | Self::DeviceCopy { .. }
            | Self::DeviceReadback { .. }
            | Self::HostUpload { .. } => true,
        }
    }

    #[allow(clippy::too_many_lines)] // Keep each operation's preflight and resource contract together.
    fn derive_resources(
        &self,
        allocation_ids: &mut AllocationIds<super::super::DeviceAllocation>,
    ) -> Result<Vec<PcuExecutionResourceUse>, CudaOwnedExecutionError> {
        match self {
            Self::Dispatch { prepared, bindings } => {
                derive_dispatch_resources(prepared, bindings, allocation_ids)
            }
            Self::DeviceCopy {
                stream,
                source,
                destination,
                bytes,
            } => {
                validate_device_copy_operation(stream, source, destination, *bytes)?;
                let mut resources = Vec::with_capacity(2);
                record_resource(
                    &mut resources,
                    allocation_ids.id(&source.allocation)?,
                    PcuMemoryAccess::ReadOnly,
                );
                record_resource(
                    &mut resources,
                    allocation_ids.id(&destination.allocation)?,
                    PcuMemoryAccess::WriteOnly,
                );
                Ok(resources)
            }
            Self::HostUpload {
                stream,
                source,
                destination,
                offset,
            } => {
                validate_host_upload_operation(stream, source, destination, *offset)?;
                Ok(vec![PcuExecutionResourceUse {
                    resource: allocation_ids.id(&destination.allocation)?,
                    access: PcuMemoryAccess::WriteOnly,
                }])
            }
            Self::DeviceReadback {
                stream,
                source,
                source_offset,
                bytes,
            } => {
                validate_device_readback_operation(stream, source, *source_offset, *bytes)?;
                Ok(vec![device_readback_resource(
                    allocation_ids.id(&source.allocation)?,
                )])
            }
            Self::Sgemm {
                handle,
                stream,
                transpose_a,
                transpose_b,
                m,
                n,
                k,
                a,
                lda,
                b,
                ldb,
                c,
                ldc,
                ..
            } => {
                handle
                    .validate_sgemm_for_stream(
                        stream,
                        *transpose_a,
                        *transpose_b,
                        *m,
                        *n,
                        *k,
                        a,
                        *lda,
                        b,
                        *ldb,
                        c,
                        *ldc,
                    )
                    .map_err(|error| {
                        CudaOwnedExecutionError::Dispatch(CudaOwnedDispatchError::Cublas(error))
                    })?;
                let mut resources = Vec::with_capacity(3);
                record_resource(
                    &mut resources,
                    allocation_ids.id(&a.allocation)?,
                    PcuMemoryAccess::ReadOnly,
                );
                record_resource(
                    &mut resources,
                    allocation_ids.id(&b.allocation)?,
                    PcuMemoryAccess::ReadOnly,
                );
                record_resource(
                    &mut resources,
                    allocation_ids.id(&c.allocation)?,
                    PcuMemoryAccess::ReadWrite,
                );
                Ok(resources)
            }
        }
    }

    fn submit(&self) -> Result<ExecutionCompletion, CudaOwnedDispatchError> {
        match self {
            Self::Dispatch { prepared, bindings } => prepared
                .submit(bindings)
                .map(|completion| ExecutionCompletion::Dispatch(Box::new(completion))),
            Self::DeviceCopy {
                stream,
                source,
                destination,
                bytes,
            } => {
                let mut batch = CudaCompletionBatch::new(stream);
                if let Err(error) = batch.copy_device_to_device(destination, source, *bytes) {
                    drop(batch);
                    return Err(error.into());
                }
                batch
                    .finish()
                    .map(|completion| ExecutionCompletion::Batch {
                        completion: Box::new(completion),
                        readback: None,
                    })
                    .map_err(Into::into)
            }
            Self::DeviceReadback {
                stream,
                source,
                source_offset,
                bytes,
            } => {
                let mut batch = CudaCompletionBatch::new(stream);
                let readback = batch.allocate_readback(*bytes)?;
                batch.copy_device_to_host_at(source, *source_offset, &readback, 0, *bytes)?;
                batch
                    .finish()
                    .map(|completion| ExecutionCompletion::Batch {
                        completion: Box::new(completion),
                        readback: Some(readback),
                    })
                    .map_err(Into::into)
            }
            Self::HostUpload {
                stream,
                source,
                destination,
                offset,
            } => {
                let mut batch = CudaCompletionBatch::new(stream);
                batch.copy_host_to_device_at(destination, *offset, Arc::clone(source))?;
                batch
                    .finish()
                    .map(|completion| ExecutionCompletion::Batch {
                        completion: Box::new(completion),
                        readback: None,
                    })
                    .map_err(Into::into)
            }
            Self::Sgemm {
                stream,
                handle,
                transpose_a,
                transpose_b,
                m,
                n,
                k,
                alpha,
                a,
                lda,
                b,
                ldb,
                beta,
                c,
                ldc,
            } => {
                let mut batch = CudaCompletionBatch::new(stream);
                handle.sgemm_into_batch(
                    &mut batch,
                    *transpose_a,
                    *transpose_b,
                    *m,
                    *n,
                    *k,
                    *alpha,
                    a,
                    *lda,
                    b,
                    *ldb,
                    *beta,
                    c,
                    *ldc,
                )?;
                batch
                    .finish()
                    .map(|completion| ExecutionCompletion::Batch {
                        completion: Box::new(completion),
                        readback: None,
                    })
                    .map_err(Into::into)
            }
        }
    }

    fn submit_into_batch(
        &self,
        batch: &mut CudaCompletionBatch,
    ) -> Result<Option<CudaReadbackId>, CudaOwnedExecutionError> {
        match self {
            Self::Dispatch { prepared, bindings } => {
                prepared
                    .submit_into_batch(bindings, batch)
                    .map_err(CudaOwnedExecutionError::Dispatch)?;
                Ok(None)
            }
            Self::DeviceCopy {
                stream: _,
                source,
                destination,
                bytes,
            } => {
                batch
                    .copy_device_to_device(destination, source, *bytes)
                    .map_err(CudaOwnedDispatchError::from)
                    .map_err(CudaOwnedExecutionError::Dispatch)?;
                Ok(None)
            }
            Self::DeviceReadback {
                source,
                source_offset,
                bytes,
                ..
            } => {
                let readback = batch
                    .allocate_readback(*bytes)
                    .map_err(CudaOwnedDispatchError::from)
                    .map_err(CudaOwnedExecutionError::Dispatch)?;
                batch
                    .copy_device_to_host_at(source, *source_offset, &readback, 0, *bytes)
                    .map_err(CudaOwnedDispatchError::from)
                    .map_err(CudaOwnedExecutionError::Dispatch)?;
                Ok(Some(readback))
            }
            Self::HostUpload {
                stream: _,
                source,
                destination,
                offset,
            } => {
                batch
                    .copy_host_to_device_at(destination, *offset, Arc::clone(source))
                    .map_err(CudaOwnedDispatchError::from)
                    .map_err(CudaOwnedExecutionError::Dispatch)?;
                Ok(None)
            }
            Self::Sgemm {
                stream: _,
                handle,
                transpose_a,
                transpose_b,
                m,
                n,
                k,
                alpha,
                a,
                lda,
                b,
                ldb,
                beta,
                c,
                ldc,
            } => {
                handle
                    .sgemm_into_batch(
                        batch,
                        *transpose_a,
                        *transpose_b,
                        *m,
                        *n,
                        *k,
                        *alpha,
                        a,
                        *lda,
                        b,
                        *ldb,
                        *beta,
                        c,
                        *ldc,
                    )
                    .map_err(CudaOwnedDispatchError::from)
                    .map_err(CudaOwnedExecutionError::Dispatch)?;
                Ok(None)
            }
        }
    }
}

fn validate_device_copy_operation(
    stream: &CudaStreamHandle,
    source: &DeviceBuffer,
    destination: &DeviceBuffer,
    bytes: usize,
) -> Result<(), CudaOwnedExecutionError> {
    stream
        .inner
        .runtime
        .ensure_same_runtime(&source.allocation.runtime)
        .and_then(|()| {
            stream
                .inner
                .runtime
                .ensure_same_runtime(&destination.allocation.runtime)
        })
        .map_err(|error| CudaOwnedExecutionError::Dispatch(CudaOwnedDispatchError::Cuda(error)))?;
    for allocation in [source, destination] {
        if bytes > allocation.len() {
            return Err(CudaOwnedExecutionError::Dispatch(
                CudaOwnedDispatchError::Cuda(crate::CudaError::BufferTooSmall {
                    allocation: allocation.len(),
                    requested: bytes,
                }),
            ));
        }
    }
    if bytes != 0 && Rc::ptr_eq(&source.allocation, &destination.allocation) {
        return Err(CudaOwnedExecutionError::Dispatch(
            CudaOwnedDispatchError::Cuda(crate::CudaError::Busy),
        ));
    }
    Ok(())
}

fn validate_device_readback_operation(
    stream: &CudaStreamHandle,
    source: &DeviceBuffer,
    source_offset: usize,
    bytes: usize,
) -> Result<(), CudaOwnedExecutionError> {
    stream
        .inner
        .runtime
        .ensure_same_runtime(&source.allocation.runtime)
        .map_err(|error| CudaOwnedExecutionError::Dispatch(CudaOwnedDispatchError::Cuda(error)))?;
    source
        .check_range(source_offset, bytes)
        .map_err(|error| CudaOwnedExecutionError::Dispatch(CudaOwnedDispatchError::Cuda(error)))
}

fn validate_host_upload_operation(
    stream: &CudaStreamHandle,
    source: &[u8],
    destination: &DeviceBuffer,
    offset: usize,
) -> Result<(), CudaOwnedExecutionError> {
    stream
        .inner
        .runtime
        .ensure_same_runtime(&destination.allocation.runtime)
        .map_err(|error| CudaOwnedExecutionError::Dispatch(CudaOwnedDispatchError::Cuda(error)))?;
    if offset
        .checked_add(source.len())
        .is_none_or(|end| end > destination.len())
    {
        return Err(CudaOwnedExecutionError::Dispatch(
            CudaOwnedDispatchError::Cuda(crate::CudaError::BufferTooSmall {
                allocation: destination.len().saturating_sub(offset),
                requested: source.len(),
            }),
        ));
    }
    Ok(())
}

/// An operation and its predecessor indices in an owned execution graph.
pub struct CudaOwnedExecutionNode<'a> {
    pub dependencies: &'a [usize],
    pub operation: CudaOwnedExecutionOperation<'a>,
}

/// Result of advancing one node in a serial owned execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CudaExecutionStep {
    /// A node reached successful device completion.
    Succeeded { node: usize },
    /// A node reached terminal failure or reported a checked arithmetic fault.
    Failed {
        node: usize,
        outcome: PcuCompletionOutcome,
    },
    /// The current node has not been submitted because a success gate is unresolved.
    Waiting { node: usize },
    /// A success-gated node was not submitted because a predecessor failed or was cancelled.
    Blocked { node: usize },
    /// A remaining node was cancelled after an unstructured dispatch or submission failure.
    Cancelled { node: usize },
    /// Every node has reached a terminal outcome or was blocked or cancelled.
    Complete,
}

/// Validation or dispatch submission failure for a `CUDA` owned execution.
#[derive(Debug)]
pub enum CudaOwnedExecutionError {
    Graph(PcuExecutionGraphError),
    StateCountMismatch { required: usize, provided: usize },
    DeviceReadbackHasSuccessor { node: usize, successor: usize },
    ReadbackNotReady { node: usize },
    ReadbackNotPresent { node: usize },
    ReadbackAlreadyTaken { node: usize },
    MixedRuntimeOrDevice,
    TooManyResources,
    InvalidState,
    Dispatch(CudaOwnedDispatchError),
}

impl fmt::Display for CudaOwnedExecutionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Graph(error) => write!(f, "invalid CUDA execution graph: {error:?}"),
            Self::StateCountMismatch { required, provided } => write!(
                f,
                "CUDA execution needs {required} node states but received {provided}"
            ),
            Self::DeviceReadbackHasSuccessor { node, successor } => write!(
                f,
                "CUDA device readback node {node} cannot have graph successor {successor}"
            ),
            Self::ReadbackNotReady { node } => {
                write!(f, "CUDA device readback at node {node} is not ready")
            }
            Self::ReadbackNotPresent { node } => {
                write!(f, "CUDA execution node {node} has no device readback")
            }
            Self::ReadbackAlreadyTaken { node } => {
                write!(f, "CUDA device readback at node {node} was already taken")
            }
            Self::MixedRuntimeOrDevice => {
                f.write_str("two-slot CUDA execution requires one selected runtime and device")
            }
            Self::TooManyResources => f.write_str("CUDA execution has too many physical resources"),
            Self::InvalidState => f.write_str("CUDA execution gate state is inconsistent"),
            Self::Dispatch(error) => error.fmt(f),
        }
    }
}

impl Error for CudaOwnedExecutionError {}

struct AllocationIds<T> {
    allocations: Vec<Rc<T>>,
}

impl<T> AllocationIds<T> {
    const fn new() -> Self {
        Self {
            allocations: Vec::new(),
        }
    }

    fn id(
        &mut self,
        allocation: &Rc<T>,
    ) -> Result<PcuExecutionResourceId, CudaOwnedExecutionError> {
        if let Some(index) = self
            .allocations
            .iter()
            .position(|known| Rc::ptr_eq(known, allocation))
        {
            return u32::try_from(index)
                .map(PcuExecutionResourceId)
                .map_err(|_| CudaOwnedExecutionError::TooManyResources);
        }
        let index = u32::try_from(self.allocations.len())
            .map_err(|_| CudaOwnedExecutionError::TooManyResources)?;
        self.allocations.push(Rc::clone(allocation));
        Ok(PcuExecutionResourceId(index))
    }
}

const fn memory_access(access: PcuBindingAccess) -> PcuMemoryAccess {
    match access {
        PcuBindingAccess::ReadOnly => PcuMemoryAccess::ReadOnly,
        PcuBindingAccess::WriteOnly => PcuMemoryAccess::WriteOnly,
        PcuBindingAccess::ReadWrite => PcuMemoryAccess::ReadWrite,
    }
}

fn merge_access(left: PcuMemoryAccess, right: PcuMemoryAccess) -> PcuMemoryAccess {
    if left == right {
        left
    } else {
        PcuMemoryAccess::ReadWrite
    }
}

fn record_resource(
    resources: &mut Vec<PcuExecutionResourceUse>,
    resource: PcuExecutionResourceId,
    access: PcuMemoryAccess,
) {
    if let Some(existing) = resources.iter_mut().find(|use_| use_.resource == resource) {
        existing.access = merge_access(existing.access, access);
    } else {
        resources.push(PcuExecutionResourceUse { resource, access });
    }
}

const fn device_readback_resource(resource: PcuExecutionResourceId) -> PcuExecutionResourceUse {
    PcuExecutionResourceUse {
        resource,
        access: PcuMemoryAccess::ReadOnly,
    }
}

fn derive_node_resources(
    operation: &CudaOwnedExecutionOperation<'_>,
    allocation_ids: &mut AllocationIds<super::super::DeviceAllocation>,
) -> Result<Vec<PcuExecutionResourceUse>, CudaOwnedExecutionError> {
    operation.derive_resources(allocation_ids)
}

fn derive_dispatch_resources(
    dispatch: &CudaPreparedDispatch,
    bindings: &[PcuOwnedBinding<DeviceBuffer>],
    allocation_ids: &mut AllocationIds<super::super::DeviceAllocation>,
) -> Result<Vec<PcuExecutionResourceUse>, CudaOwnedExecutionError> {
    super::validate_owned_binding_requirements(
        &dispatch.binding_requirements,
        dispatch.device,
        bindings,
    )
    .map_err(|error| CudaOwnedExecutionError::Dispatch(CudaOwnedDispatchError::Binding(error)))?;
    for binding in bindings {
        let actual = binding.resource.len();
        if binding.byte_len != actual as u64 {
            return Err(CudaOwnedExecutionError::Dispatch(
                CudaOwnedDispatchError::BufferSizeMismatch {
                    binding: binding.target,
                    metadata: binding.byte_len,
                    actual,
                },
            ));
        }
        dispatch
            .runtime
            .ensure_same_runtime(&binding.resource.allocation.runtime)
            .map_err(|_| {
                CudaOwnedExecutionError::Dispatch(CudaOwnedDispatchError::DifferentRuntime(
                    binding.target,
                ))
            })?;
    }

    let mut resources = Vec::<PcuExecutionResourceUse>::new();
    for binding in bindings {
        let resource = allocation_ids.id(&binding.resource.allocation)?;
        let access = memory_access(binding.access);
        record_resource(&mut resources, resource, access);
    }
    Ok(resources)
}

#[cfg(test)]
enum CompletionUpdateError<E> {
    Wait(E),
    InvalidState,
}

#[cfg(test)]
fn wait_and_finish<E>(
    gate_state: &mut PcuExecutionFaultGateState<'_>,
    node: usize,
    wait: impl FnOnce() -> Result<PcuCompletionOutcome, E>,
) -> Result<PcuCompletionOutcome, CompletionUpdateError<E>> {
    let outcome = wait().map_err(CompletionUpdateError::Wait)?;
    let state = match outcome {
        PcuCompletionOutcome::Succeeded => PcuExecutionNodeState::Succeeded,
        PcuCompletionOutcome::Failed | PcuCompletionOutcome::Fault(_) => {
            PcuExecutionNodeState::Failed
        }
    };
    if !gate_state.finish(node, state) {
        return Err(CompletionUpdateError::InvalidState);
    }
    Ok(outcome)
}

fn validate_owned_execution_graph(
    nodes: &[CudaOwnedExecutionNode<'_>],
    success_gates: &[PcuExecutionSuccessGate],
    scratch: &mut [bool],
) -> Result<(), CudaOwnedExecutionError> {
    let readback_nodes = nodes
        .iter()
        .map(|node| {
            matches!(
                &node.operation,
                CudaOwnedExecutionOperation::DeviceReadback { .. }
            )
        })
        .collect::<Vec<_>>();
    let dependencies = nodes
        .iter()
        .map(|node| node.dependencies)
        .collect::<Vec<_>>();
    if let Some((node, successor)) =
        first_readback_successor(&readback_nodes, &dependencies, success_gates)
    {
        return Err(CudaOwnedExecutionError::DeviceReadbackHasSuccessor { node, successor });
    }
    if let Some(first) = nodes.first()
        && nodes.iter().skip(1).any(|node| {
            first
                .operation
                .runtime()
                .ensure_same_runtime(node.operation.runtime())
                .is_err()
        })
    {
        return Err(CudaOwnedExecutionError::MixedRuntimeOrDevice);
    }
    let mut allocation_ids = AllocationIds::new();
    let mut resources = Vec::with_capacity(nodes.len());
    for node in nodes {
        resources.push(derive_node_resources(&node.operation, &mut allocation_ids)?);
    }
    let resource_count = u32::try_from(allocation_ids.allocations.len())
        .map_err(|_| CudaOwnedExecutionError::TooManyResources)?;
    let descriptors: Vec<_> = nodes
        .iter()
        .zip(&resources)
        .map(|(node, uses)| PcuExecutionNode {
            dependencies: node.dependencies,
            resources: uses,
        })
        .collect();
    let may_fault: Vec<_> = nodes.iter().map(|node| node.operation.checked()).collect();
    validate_execution_submission(
        &descriptors,
        resource_count,
        &may_fault,
        success_gates,
        scratch,
    )
    .map_err(CudaOwnedExecutionError::Graph)
}

fn first_readback_successor(
    readback_nodes: &[bool],
    dependencies: &[&[usize]],
    success_gates: &[PcuExecutionSuccessGate],
) -> Option<(usize, usize)> {
    readback_nodes
        .iter()
        .enumerate()
        .filter(|(_, is_readback)| **is_readback)
        .find_map(|(node, _)| {
            dependencies
                .iter()
                .enumerate()
                .find(|(_, predecessors)| predecessors.contains(&node))
                .map(|(successor, _)| (node, successor))
                .or_else(|| {
                    success_gates
                        .iter()
                        .find(|gate| gate.predecessor == node)
                        .map(|gate| (node, gate.successor))
                })
        })
}

/// Caller-owned, serial executor for validated `CUDA` operation nodes.
///
/// It retains graph bindings and any in-flight completion. `step` waits for each submitted node
/// before advancing, so ordinary dependencies are satisfied and checked arithmetic faults are
/// known before a gated consumer can be submitted. Device copies, uploads, and readbacks are
/// queued on their supplied streams and retained by a final batch event. Readback bytes remain
/// private until successful completion and are then available once through [`Self::take_readback`].
/// If waiting returns an error, the completion remains owned by this value and the caller may
/// retry `step`; resources remain live while the completion is indeterminate. Other users of a
/// prepared dispatch's stream are outside this graph's gate law and must not enqueue work dependent
/// on faulting outputs before observing this execution's terminal result.
pub struct CudaOwnedExecution<'a, 'state> {
    nodes: &'a [CudaOwnedExecutionNode<'a>],
    gate_state: PcuExecutionFaultGateState<'state>,
    pending: Option<(usize, ExecutionCompletion)>,
    next: usize,
    halted: bool,
    readbacks: Vec<ExecutionReadback>,
}

impl<'a, 'state> CudaOwnedExecution<'a, 'state> {
    /// Validates and creates a serial `CUDA` graph executor.
    ///
    /// Resource identities and access modes are derived from the actual owned buffer allocations
    /// and binding metadata. The state slice is reset to `Pending`; graph validation and binding
    /// preflight happen before any dispatch is submitted. Dependencies and success gates remain
    /// caller supplied.
    ///
    /// # Errors
    ///
    /// Returns an error if state count, binding metadata, graph validation, or scratch capacity
    /// does not match the supplied graph.
    pub fn new(
        nodes: &'a [CudaOwnedExecutionNode<'a>],
        success_gates: &'state [PcuExecutionSuccessGate],
        scratch: &mut [bool],
        states: &'state mut [PcuExecutionNodeState],
    ) -> Result<Self, CudaOwnedExecutionError> {
        if states.len() != nodes.len() {
            return Err(CudaOwnedExecutionError::StateCountMismatch {
                required: nodes.len(),
                provided: states.len(),
            });
        }
        validate_owned_execution_graph(nodes, success_gates, scratch)?;
        let readbacks = nodes
            .iter()
            .map(|node| match node.operation {
                CudaOwnedExecutionOperation::DeviceReadback { .. } => ExecutionReadback::Pending,
                _ => ExecutionReadback::NotPresent,
            })
            .collect();
        Ok(Self {
            nodes,
            gate_state: PcuExecutionFaultGateState::new(success_gates, states),
            pending: None,
            next: 0,
            halted: false,
            readbacks,
        })
    }

    /// Takes one host readback after its node has succeeded. Each readback is single-use.
    ///
    /// # Errors
    ///
    /// Returns `ReadbackNotReady` while the node is pending or failed, `ReadbackNotPresent` for
    /// an unknown node or a node without a readback, and `ReadbackAlreadyTaken` after consumption.
    pub fn take_readback(&mut self, node: usize) -> Result<Box<[u8]>, CudaOwnedExecutionError> {
        take_execution_readback(&mut self.readbacks, node)
    }

    /// Advances at most one graph node. Completion errors leave the current completion retained
    /// so a later call can retry without releasing device resources or opening its gates.
    ///
    /// # Errors
    ///
    /// Returns an operation submission or completion error. Completion errors leave the graph
    /// resumable; submission errors mark the node failed and stop later submissions.
    pub fn step(&mut self) -> Result<CudaExecutionStep, CudaOwnedExecutionError> {
        if let Some((node_index, completion)) = self.pending.as_mut() {
            let node = *node_index;
            let outcome = completion
                .wait()
                .map_err(CudaOwnedExecutionError::Dispatch)?;
            let mut readback_bytes = None;
            if outcome == PcuCompletionOutcome::Succeeded {
                match completion.take_readback() {
                    Ok(bytes) => readback_bytes = bytes,
                    Err(error) => {
                        if !self.gate_state.finish(node, PcuExecutionNodeState::Failed) {
                            return Err(CudaOwnedExecutionError::InvalidState);
                        }
                        self.halted = true;
                        self.pending.take();
                        self.next += 1;
                        return Err(CudaOwnedExecutionError::Dispatch(error));
                    }
                }
            }
            let state = match outcome {
                PcuCompletionOutcome::Succeeded => PcuExecutionNodeState::Succeeded,
                PcuCompletionOutcome::Failed | PcuCompletionOutcome::Fault(_) => {
                    PcuExecutionNodeState::Failed
                }
            };
            if !self.gate_state.finish(node, state) {
                return Err(CudaOwnedExecutionError::InvalidState);
            }
            self.halted |= outcome == PcuCompletionOutcome::Failed;
            self.pending.take();
            if let Some(bytes) = readback_bytes {
                let Some(readback) = self.readbacks.get_mut(node) else {
                    return Err(CudaOwnedExecutionError::InvalidState);
                };
                if !matches!(readback, ExecutionReadback::Pending) {
                    return Err(CudaOwnedExecutionError::InvalidState);
                }
                *readback = ExecutionReadback::Ready(bytes);
            }
            self.next += 1;
            return Ok(match outcome {
                PcuCompletionOutcome::Succeeded => CudaExecutionStep::Succeeded { node },
                other => CudaExecutionStep::Failed {
                    node,
                    outcome: other,
                },
            });
        }

        if self.next >= self.nodes.len() {
            return Ok(CudaExecutionStep::Complete);
        }
        if self.halted {
            let node = self.next;
            if !self.gate_state.cancel_pending(node) {
                return Err(CudaOwnedExecutionError::InvalidState);
            }
            self.next += 1;
            return Ok(CudaExecutionStep::Cancelled { node });
        }
        let node_index = self.next;
        match self.gate_state.try_start(node_index) {
            Some(PcuExecutionAdmission::Started) => {}
            Some(PcuExecutionAdmission::Waiting) => {
                return Ok(CudaExecutionStep::Waiting { node: node_index });
            }
            Some(PcuExecutionAdmission::Blocked) => {
                if !self.gate_state.cancel_pending(node_index) {
                    return Err(CudaOwnedExecutionError::InvalidState);
                }
                // Later nodes may have plain ordering edges from this cancelled node. Those
                // edges do not carry a success gate, so stopping the serial graph is necessary
                // to avoid consuming output that was never produced.
                self.halted = true;
                self.next += 1;
                return Ok(CudaExecutionStep::Blocked { node: node_index });
            }
            Some(PcuExecutionAdmission::AlreadyStarted) | None => {
                return Err(CudaOwnedExecutionError::InvalidState);
            }
        }

        let node = &self.nodes[node_index];
        match node.operation.submit() {
            Ok(completion) => {
                self.pending = Some((node_index, completion));
                self.step()
            }
            Err(error) => {
                // CudaKernel::launch synchronizes on an enqueue error, or quarantines retained
                // launch resources if quiescence cannot be confirmed. No completion token exists
                // to retry, so this node is terminally failed and its gated consumers stay shut.
                self.halted = true;
                if !self
                    .gate_state
                    .finish(node_index, PcuExecutionNodeState::Failed)
                {
                    return Err(CudaOwnedExecutionError::InvalidState);
                }
                self.next += 1;
                Err(CudaOwnedExecutionError::Dispatch(error))
            }
        }
    }
}

/// A bounded-overlap executor for owned `CUDA` operations.
///
/// It may keep at most two validated lineages in flight. By default, dependencies wait for
/// predecessor completion. The opt-in event-chain constructor can attach a linear chain of
/// unchecked, non-gated single-consumer nodes through CUDA events, and coalesces eligible
/// same-stream nodes into one batch and final event. Prepared dispatches retain their captured
/// streams or explicit copy streams, so independent nodes can overlap when those streams differ.
/// SGEMM nodes use the batch-owned asynchronous cuBLAS path. One handle may queue dependent
/// SGEMMs only within the same adopted lineage and bound stream; independent overlap still needs
/// distinct handles. This is not a general multi-queue scheduler and does not schedule arbitrary
/// library calls. Device readbacks are supported as terminal graph nodes; they cannot have graph
/// successors because host output is only available after completion. All nodes must use the same
/// CUDA runtime and device.
pub struct CudaOwnedExecutionTwoSlot<'a, 'state> {
    nodes: &'a [CudaOwnedExecutionNode<'a>],
    gate_state: PcuExecutionFaultGateState<'state>,
    outcomes: Vec<PcuExecutionNodeState>,
    readbacks: Vec<ExecutionReadback>,
    pending: [Option<PendingExecutionCompletion>; 2],
    consumer_counts: Vec<usize>,
    success_gates: &'state [PcuExecutionSuccessGate],
    event_chaining: bool,
    next_wait_slot: usize,
}

struct PendingExecutionCompletion {
    completion: ExecutionCompletion,
    lineage: Vec<usize>,
}

enum ExecutionCompletion {
    Dispatch(Box<CudaOwnedCompletion>),
    OpenBatch {
        batch: Box<CudaCompletionBatch>,
        readback: Option<CudaReadbackId>,
    },
    Batch {
        completion: Box<CudaBatchCompletion>,
        readback: Option<CudaReadbackId>,
    },
}

impl ExecutionCompletion {
    const fn has_readback(&self) -> bool {
        matches!(
            self,
            Self::Batch {
                readback: Some(_),
                ..
            } | Self::OpenBatch {
                readback: Some(_),
                ..
            }
        )
    }

    fn wait(&mut self) -> Result<PcuCompletionOutcome, CudaOwnedDispatchError> {
        match self {
            Self::Dispatch(completion) => completion.wait().map_err(CudaOwnedDispatchError::Cuda),
            Self::Batch { completion, .. } => completion
                .wait()
                .map(|()| PcuCompletionOutcome::Succeeded)
                .map_err(CudaOwnedDispatchError::Cuda),
            Self::OpenBatch { .. } => {
                Err(CudaOwnedDispatchError::Cuda(CudaError::BatchNotComplete))
            }
        }
    }

    fn take_readback(&mut self) -> Result<Option<Box<[u8]>>, CudaOwnedDispatchError> {
        match self {
            Self::Batch {
                completion,
                readback: Some(id),
            } => completion
                .take_readback(id)
                .map(Some)
                .map_err(CudaOwnedDispatchError::Cuda),
            Self::OpenBatch { .. } => {
                Err(CudaOwnedDispatchError::Cuda(CudaError::BatchNotComplete))
            }
            Self::Dispatch(_) | Self::Batch { readback: None, .. } => Ok(None),
        }
    }
}

enum ExecutionReadback {
    NotPresent,
    Pending,
    Ready(Box<[u8]>),
    Taken,
}

fn take_execution_readback(
    readbacks: &mut [ExecutionReadback],
    node: usize,
) -> Result<Box<[u8]>, CudaOwnedExecutionError> {
    let Some(readback) = readbacks.get_mut(node) else {
        return Err(CudaOwnedExecutionError::ReadbackNotPresent { node });
    };
    match std::mem::replace(readback, ExecutionReadback::Taken) {
        ExecutionReadback::Ready(bytes) => Ok(bytes),
        ExecutionReadback::Pending => {
            *readback = ExecutionReadback::Pending;
            Err(CudaOwnedExecutionError::ReadbackNotReady { node })
        }
        ExecutionReadback::NotPresent => {
            *readback = ExecutionReadback::NotPresent;
            Err(CudaOwnedExecutionError::ReadbackNotPresent { node })
        }
        ExecutionReadback::Taken => Err(CudaOwnedExecutionError::ReadbackAlreadyTaken { node }),
    }
}

/// Progress reported by [`CudaOwnedExecutionTwoSlot`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CudaTwoSlotExecutionStep {
    /// One ready operation was submitted and its completion is retained in a slot.
    Submitted { node: usize },
    /// One submitted node reached successful device completion.
    Succeeded { node: usize },
    /// One submitted node reached terminal failure or reported a checked arithmetic fault.
    Failed {
        node: usize,
        outcome: PcuCompletionOutcome,
    },
    /// No node can currently advance while one or more submitted nodes remain in flight.
    Waiting { node: usize },
    /// A node was not submitted because a predecessor failed or was cancelled.
    Blocked { node: usize },
    /// Every node reached a terminal state.
    Complete,
}

impl<'a, 'state> CudaOwnedExecutionTwoSlot<'a, 'state> {
    /// Validates and creates a two-slot executor. `states` is reset to `Pending`.
    ///
    /// # Errors
    ///
    /// Returns an error if state count, binding metadata, graph validation, or scratch capacity
    /// does not match the supplied graph.
    pub fn new(
        nodes: &'a [CudaOwnedExecutionNode<'a>],
        success_gates: &'state [PcuExecutionSuccessGate],
        scratch: &mut [bool],
        states: &'state mut [PcuExecutionNodeState],
    ) -> Result<Self, CudaOwnedExecutionError> {
        Self::new_with_event_chaining(nodes, success_gates, scratch, states, false)
    }

    /// Creates a two-slot executor that may event-chain eligible unchecked ordinary edges,
    /// including longer linear chains whose prior completion is already a CUDA batch. A terminal
    /// device readback may consume a chain; it cannot have graph successors.
    ///
    /// Checked dispatches, success-gated edges, fan-out producers, and handoffs whose access
    /// leases cannot move remain on the host-wait path.
    ///
    /// # Errors
    ///
    /// Returns an error if state count, binding metadata, graph validation, or scratch capacity
    /// does not match the supplied graph, or if a device readback has a graph successor.
    pub fn new_event_chained(
        nodes: &'a [CudaOwnedExecutionNode<'a>],
        success_gates: &'state [PcuExecutionSuccessGate],
        scratch: &mut [bool],
        states: &'state mut [PcuExecutionNodeState],
    ) -> Result<Self, CudaOwnedExecutionError> {
        Self::new_with_event_chaining(nodes, success_gates, scratch, states, true)
    }

    fn new_with_event_chaining(
        nodes: &'a [CudaOwnedExecutionNode<'a>],
        success_gates: &'state [PcuExecutionSuccessGate],
        scratch: &mut [bool],
        states: &'state mut [PcuExecutionNodeState],
        event_chaining: bool,
    ) -> Result<Self, CudaOwnedExecutionError> {
        if states.len() != nodes.len() {
            return Err(CudaOwnedExecutionError::StateCountMismatch {
                required: nodes.len(),
                provided: states.len(),
            });
        }
        validate_owned_execution_graph(nodes, success_gates, scratch)?;
        let readbacks = nodes
            .iter()
            .map(|node| match node.operation {
                CudaOwnedExecutionOperation::DeviceReadback { .. } => ExecutionReadback::Pending,
                _ => ExecutionReadback::NotPresent,
            })
            .collect();
        let mut consumer_counts = vec![0; nodes.len()];
        for node in nodes {
            for &dependency in node.dependencies {
                consumer_counts[dependency] += 1;
            }
        }
        Ok(Self {
            nodes,
            gate_state: PcuExecutionFaultGateState::new(success_gates, states),
            outcomes: vec![PcuExecutionNodeState::Pending; nodes.len()],
            readbacks,
            pending: [None, None],
            consumer_counts,
            success_gates,
            event_chaining,
            next_wait_slot: 0,
        })
    }

    /// Takes one host readback after its node has succeeded. Each readback is single-use.
    ///
    /// # Errors
    ///
    /// Returns `ReadbackNotReady` while the node is pending or failed, `ReadbackNotPresent` for
    /// an unknown node or a node without a readback, and `ReadbackAlreadyTaken` after consumption.
    pub fn take_readback(&mut self, node: usize) -> Result<Box<[u8]>, CudaOwnedExecutionError> {
        take_execution_readback(&mut self.readbacks, node)
    }

    /// Advances the graph by submitting one ready node or observing one completion.
    ///
    /// A wait error leaves its completion in its slot so a later call can retry while retaining
    /// all device resources. The default constructor waits for predecessor success; the opt-in
    /// event-chain mode orders eligible ordinary edges and retains their full lineage.
    /// After an error, callers may keep stepping to drain independent in-flight work; dropping
    /// this executor instead delegates quiescence or quarantine to each retained completion.
    /// Dropping or unwinding while work is pending can leave caller-provided states `Running`;
    /// only `step` transitions them after confirmed completion. Catching a panic inside
    /// `step` does not make this executor resumable, although retained device resources remain
    /// protected by batch synchronization or quarantine.
    ///
    /// # Errors
    ///
    /// Returns an operation submission or completion error. Completion errors retain the
    /// completion for a later retry; submission errors mark the node failed.
    #[allow(clippy::too_many_lines)] // Keep token transfer and its failure cleanup in one visible transition.
    pub fn step(&mut self) -> Result<CudaTwoSlotExecutionStep, CudaOwnedExecutionError> {
        if let Some(slot) = self.pending.iter().position(Option::is_none)
            && let Some(node) = self.next_ready_node()
            && self.nodes[node].operation.can_submit_directly()
        {
            match self.gate_state.try_start(node) {
                Some(PcuExecutionAdmission::Started) => {
                    let batch_mode = self.event_chaining
                        && !self.nodes[node].operation.checked()
                        && self.has_structural_chain_successor(node, &[node]);
                    let submitted = if batch_mode {
                        let mut batch =
                            CudaCompletionBatch::new(self.nodes[node].operation.stream());
                        match self.nodes[node].operation.submit_into_batch(&mut batch) {
                            Ok(readback) => {
                                self.outcomes[node] = PcuExecutionNodeState::Running;
                                if self.has_open_batch_successor(node, &batch, &[node]) {
                                    Ok(ExecutionCompletion::OpenBatch {
                                        batch: Box::new(batch),
                                        readback,
                                    })
                                } else {
                                    batch
                                        .finish()
                                        .map(|completion| ExecutionCompletion::Batch {
                                            completion: Box::new(completion),
                                            readback,
                                        })
                                        .map_err(|error| {
                                            CudaOwnedExecutionError::Dispatch(
                                                CudaOwnedDispatchError::Cuda(error),
                                            )
                                        })
                                }
                            }
                            Err(error) => Err(error),
                        }
                    } else {
                        self.nodes[node]
                            .operation
                            .submit()
                            .map_err(CudaOwnedExecutionError::Dispatch)
                    };
                    match submitted {
                        Ok(completion) => {
                            self.pending[slot] = Some(PendingExecutionCompletion {
                                completion,
                                lineage: vec![node],
                            });
                            self.outcomes[node] = PcuExecutionNodeState::Running;
                            return Ok(CudaTwoSlotExecutionStep::Submitted { node });
                        }
                        Err(error) => {
                            if !self.gate_state.finish(node, PcuExecutionNodeState::Failed) {
                                return Err(CudaOwnedExecutionError::InvalidState);
                            }
                            self.outcomes[node] = PcuExecutionNodeState::Failed;
                            return Err(error);
                        }
                    }
                }
                Some(PcuExecutionAdmission::Blocked) => {
                    if !self.gate_state.cancel_pending(node) {
                        return Err(CudaOwnedExecutionError::InvalidState);
                    }
                    self.outcomes[node] = PcuExecutionNodeState::Cancelled;
                    return Ok(CudaTwoSlotExecutionStep::Blocked { node });
                }
                Some(PcuExecutionAdmission::Waiting) => {}
                Some(PcuExecutionAdmission::AlreadyStarted) | None => {
                    return Err(CudaOwnedExecutionError::InvalidState);
                }
            }
        }

        if self.event_chaining
            && let Some((node, _predecessor, predecessor_slot)) = self.next_chain_candidate()
        {
            if matches!(
                self.pending[predecessor_slot]
                    .as_ref()
                    .map(|pending| &pending.completion),
                Some(ExecutionCompletion::OpenBatch { .. })
            ) {
                let PendingExecutionCompletion {
                    completion,
                    mut lineage,
                } = self.pending[predecessor_slot]
                    .take()
                    .ok_or(CudaOwnedExecutionError::InvalidState)?;
                let ExecutionCompletion::OpenBatch {
                    mut batch,
                    readback: predecessor_readback,
                } = completion
                else {
                    return Err(CudaOwnedExecutionError::InvalidState);
                };
                if predecessor_readback.is_some()
                    || self.gate_state.try_start(node) != Some(PcuExecutionAdmission::Started)
                {
                    lineage.push(node);
                    self.finish_failed_lineage(lineage)?;
                    return Err(CudaOwnedExecutionError::InvalidState);
                }
                match self.nodes[node].operation.submit_into_batch(&mut batch) {
                    Ok(readback) => {
                        lineage.push(node);
                        self.outcomes[node] = PcuExecutionNodeState::Running;
                        let completion = if self.has_open_batch_successor(node, &batch, &lineage) {
                            ExecutionCompletion::OpenBatch { batch, readback }
                        } else {
                            match batch.finish() {
                                Ok(completion) => ExecutionCompletion::Batch {
                                    completion: Box::new(completion),
                                    readback,
                                },
                                Err(error) => {
                                    self.finish_failed_lineage(lineage)?;
                                    return Err(CudaOwnedExecutionError::Dispatch(
                                        CudaOwnedDispatchError::Cuda(error),
                                    ));
                                }
                            }
                        };
                        self.pending[predecessor_slot] = Some(PendingExecutionCompletion {
                            completion,
                            lineage,
                        });
                        return Ok(CudaTwoSlotExecutionStep::Submitted { node });
                    }
                    Err(error) => {
                        lineage.push(node);
                        self.finish_failed_lineage(lineage)?;
                        return Err(error);
                    }
                }
            }

            let mut batch = CudaCompletionBatch::new(self.nodes[node].operation.stream());
            let can_handoff = match self.pending[predecessor_slot]
                .as_ref()
                .map(|pending| &pending.completion)
            {
                Some(ExecutionCompletion::Dispatch(completion)) => {
                    completion.can_handoff_to_batch(&batch)
                }
                Some(ExecutionCompletion::Batch { completion, .. }) => {
                    batch.can_wait_for_batch(completion)
                }
                Some(ExecutionCompletion::OpenBatch { .. }) | None => Ok(false),
            }
            .map_err(|error| {
                CudaOwnedExecutionError::Dispatch(CudaOwnedDispatchError::Cuda(error))
            })?;
            if can_handoff {
                let PendingExecutionCompletion {
                    completion: predecessor_completion,
                    lineage,
                } = self.pending[predecessor_slot]
                    .take()
                    .ok_or(CudaOwnedExecutionError::InvalidState)?;
                let handoff = match predecessor_completion {
                    ExecutionCompletion::Dispatch(mut completion) => completion
                        .take_cuda_for_handoff()
                        .ok_or(CudaOwnedExecutionError::InvalidState)
                        .and_then(|completion| {
                            batch.wait_for(completion).map_err(|error| {
                                CudaOwnedExecutionError::Dispatch(CudaOwnedDispatchError::Cuda(
                                    error,
                                ))
                            })
                        }),
                    ExecutionCompletion::Batch {
                        mut completion,
                        readback,
                    } => {
                        debug_assert!(readback.is_none());
                        batch.wait_for_batch(completion.as_mut()).map_err(|error| {
                            CudaOwnedExecutionError::Dispatch(CudaOwnedDispatchError::Cuda(error))
                        })
                    }
                    ExecutionCompletion::OpenBatch { .. } => {
                        Err(CudaOwnedExecutionError::InvalidState)
                    }
                };
                if let Err(error) = handoff {
                    self.finish_failed_lineage(lineage)?;
                    drop(batch);
                    return Err(error);
                }
                if self.gate_state.try_start(node) != Some(PcuExecutionAdmission::Started) {
                    self.finish_failed_lineage(lineage)?;
                    drop(batch);
                    return Err(CudaOwnedExecutionError::InvalidState);
                }
                match self.nodes[node].operation.submit_into_batch(&mut batch) {
                    Ok(readback) => match batch.finish() {
                        Ok(completion) => {
                            let slot = predecessor_slot;
                            let mut lineage = lineage;
                            lineage.push(node);
                            self.pending[slot] = Some(PendingExecutionCompletion {
                                completion: ExecutionCompletion::Batch {
                                    completion: Box::new(completion),
                                    readback,
                                },
                                lineage,
                            });
                            self.outcomes[node] = PcuExecutionNodeState::Running;
                            return Ok(CudaTwoSlotExecutionStep::Submitted { node });
                        }
                        Err(error) => {
                            drop(batch);
                            let mut lineage = lineage;
                            lineage.push(node);
                            self.finish_failed_lineage(lineage)?;
                            return Err(CudaOwnedExecutionError::Dispatch(
                                CudaOwnedDispatchError::Cuda(error),
                            ));
                        }
                    },
                    Err(error) => {
                        drop(batch);
                        let mut lineage = lineage;
                        lineage.push(node);
                        self.finish_failed_lineage(lineage)?;
                        return Err(error);
                    }
                }
            }
        }

        if let Some(slot) = self.pending.iter().position(|pending| {
            pending.as_ref().is_some_and(|pending| {
                matches!(&pending.completion, ExecutionCompletion::OpenBatch { .. })
            })
        }) {
            let PendingExecutionCompletion {
                completion,
                lineage,
            } = self.pending[slot]
                .take()
                .ok_or(CudaOwnedExecutionError::InvalidState)?;
            let ExecutionCompletion::OpenBatch {
                mut batch,
                readback,
            } = completion
            else {
                return Err(CudaOwnedExecutionError::InvalidState);
            };
            match batch.finish() {
                Ok(completion) => {
                    self.pending[slot] = Some(PendingExecutionCompletion {
                        completion: ExecutionCompletion::Batch {
                            completion: Box::new(completion),
                            readback,
                        },
                        lineage,
                    });
                }
                Err(error) => {
                    self.finish_failed_lineage(lineage)?;
                    return Err(CudaOwnedExecutionError::Dispatch(
                        CudaOwnedDispatchError::Cuda(error),
                    ));
                }
            }
        }

        if let Some(slot) = next_pending_slot(
            [self.pending[0].is_some(), self.pending[1].is_some()],
            self.next_wait_slot,
        ) {
            let pending = self.pending[slot]
                .as_mut()
                .ok_or(CudaOwnedExecutionError::InvalidState)?;
            let outcome = match pending.completion.wait() {
                Ok(outcome) => outcome,
                Err(error) => {
                    self.next_wait_slot = (slot + 1) % 2;
                    return Err(CudaOwnedExecutionError::Dispatch(error));
                }
            };
            let mut pending = self.pending[slot]
                .take()
                .ok_or(CudaOwnedExecutionError::InvalidState)?;
            self.next_wait_slot = (slot + 1) % 2;
            let completed_node = pending.lineage.last().copied().unwrap_or(0);
            let mut readback_bytes = None;
            if outcome == PcuCompletionOutcome::Succeeded {
                match pending.completion.take_readback() {
                    Ok(bytes) => readback_bytes = bytes,
                    Err(error) => {
                        self.finish_failed_lineage(pending.lineage.iter().copied())?;
                        return Err(CudaOwnedExecutionError::Dispatch(error));
                    }
                }
            }
            let terminal_readback = pending.lineage.last().is_some_and(|&node| {
                matches!(
                    &self.nodes[node].operation,
                    CudaOwnedExecutionOperation::DeviceReadback { .. }
                )
            });
            if outcome == PcuCompletionOutcome::Succeeded
                && terminal_readback != readback_bytes.is_some()
            {
                self.finish_failed_lineage(pending.lineage.iter().copied())?;
                return Err(CudaOwnedExecutionError::InvalidState);
            }
            let outcome_state = match outcome {
                PcuCompletionOutcome::Succeeded => PcuExecutionNodeState::Succeeded,
                PcuCompletionOutcome::Failed | PcuCompletionOutcome::Fault(_) => {
                    PcuExecutionNodeState::Failed
                }
            };
            for node in pending.lineage {
                if !self.gate_state.finish(node, outcome_state) {
                    return Err(CudaOwnedExecutionError::InvalidState);
                }
                self.outcomes[node] = outcome_state;
            }
            if let Some(bytes) = readback_bytes {
                let Some(readback) = self.readbacks.get_mut(completed_node) else {
                    return Err(CudaOwnedExecutionError::InvalidState);
                };
                if !matches!(readback, ExecutionReadback::Pending) {
                    return Err(CudaOwnedExecutionError::InvalidState);
                }
                *readback = ExecutionReadback::Ready(bytes);
            }
            return Ok(match outcome {
                PcuCompletionOutcome::Succeeded => CudaTwoSlotExecutionStep::Succeeded {
                    node: completed_node,
                },
                other => CudaTwoSlotExecutionStep::Failed {
                    node: completed_node,
                    outcome: other,
                },
            });
        }

        if self.outcomes.iter().all(|state| {
            matches!(
                state,
                PcuExecutionNodeState::Succeeded
                    | PcuExecutionNodeState::Failed
                    | PcuExecutionNodeState::Cancelled
            )
        }) {
            return Ok(CudaTwoSlotExecutionStep::Complete);
        }

        if let Some(node) = self.next_blocked_node() {
            if !self.gate_state.cancel_pending(node) {
                return Err(CudaOwnedExecutionError::InvalidState);
            }
            self.outcomes[node] = PcuExecutionNodeState::Cancelled;
            return Ok(CudaTwoSlotExecutionStep::Blocked { node });
        }
        Err(CudaOwnedExecutionError::InvalidState)
    }

    fn next_ready_node(&self) -> Option<usize> {
        first_ready_node(
            self.nodes.iter().map(|node| node.dependencies),
            &self.outcomes,
        )
    }

    fn next_blocked_node(&self) -> Option<usize> {
        first_blocked_node(
            self.nodes.iter().map(|node| node.dependencies),
            &self.outcomes,
        )
    }

    fn has_structural_chain_successor(&self, predecessor: usize, lineage: &[usize]) -> bool {
        self.nodes.iter().enumerate().any(|(successor, node)| {
            self.outcomes.get(successor) == Some(&PcuExecutionNodeState::Pending)
                && node.dependencies == [predecessor]
                && !node.operation.checked()
                && std::rc::Rc::ptr_eq(
                    &node.operation.stream().inner,
                    &self.nodes[predecessor].operation.stream().inner,
                )
                && open_batch_edge_eligible(
                    lineage,
                    predecessor,
                    successor,
                    true,
                    true,
                    self.consumer_counts[predecessor],
                    self.success_gates,
                )
        })
    }

    fn has_open_batch_successor(
        &self,
        predecessor: usize,
        batch: &CudaCompletionBatch,
        lineage: &[usize],
    ) -> bool {
        self.nodes.iter().enumerate().any(|(successor, node)| {
            self.outcomes.get(successor) == Some(&PcuExecutionNodeState::Pending)
                && node.dependencies == [predecessor]
                && !node.operation.checked()
                && node.operation.can_submit_into_open_batch(batch)
                && open_batch_edge_eligible(
                    lineage,
                    predecessor,
                    successor,
                    true,
                    std::rc::Rc::ptr_eq(
                        &self.nodes[predecessor].operation.stream().inner,
                        &node.operation.stream().inner,
                    ),
                    self.consumer_counts[predecessor],
                    self.success_gates,
                )
        })
    }

    fn next_chain_candidate(&self) -> Option<(usize, usize, usize)> {
        self.nodes
            .iter()
            .enumerate()
            .find_map(|(node_index, node)| {
                if self.outcomes[node_index] != PcuExecutionNodeState::Pending
                    || node.operation.checked()
                    || node.dependencies.len() != 1
                {
                    return None;
                }
                let predecessor = node.dependencies[0];
                let slot = self.pending.iter().position(|pending| {
                    pending.as_ref().is_some_and(|pending| {
                        event_chain_lineage_eligible(
                            &pending.lineage,
                            predecessor,
                            matches!(
                                &pending.completion,
                                ExecutionCompletion::OpenBatch { .. }
                                    | ExecutionCompletion::Batch { .. }
                            ),
                        ) && !pending.completion.has_readback()
                            && !event_chain_lineage_has_success_gate(
                                &pending.lineage,
                                node_index,
                                self.success_gates,
                            )
                    })
                })?;
                let predecessor_completion = &self.pending[slot].as_ref()?.completion;
                let submission_ready = match predecessor_completion {
                    ExecutionCompletion::OpenBatch { batch, .. } => {
                        node.operation.can_submit_into_open_batch(batch)
                    }
                    completion => node.operation.can_submit_into_event_chain(completion),
                };
                if !event_chain_edge_eligible(
                    predecessor,
                    node_index,
                    self.outcomes.get(predecessor).copied()?,
                    self.nodes[predecessor].operation.checked(),
                    node.operation.checked(),
                    submission_ready,
                    self.consumer_counts.get(predecessor).copied()?,
                    self.success_gates,
                ) {
                    return None;
                }
                Some((node_index, predecessor, slot))
            })
    }

    fn finish_failed_lineage(
        &mut self,
        lineage: impl IntoIterator<Item = usize>,
    ) -> Result<(), CudaOwnedExecutionError> {
        // A failure while attaching or submitting the consumer makes the entire moved lineage
        // unusable to successors, even if an upstream dispatch may have completed successfully.
        finish_lineage_state(
            &mut self.gate_state,
            &mut self.outcomes,
            lineage,
            PcuExecutionNodeState::Failed,
        )
    }
}

fn finish_lineage_state(
    gate_state: &mut PcuExecutionFaultGateState<'_>,
    outcomes: &mut [PcuExecutionNodeState],
    lineage: impl IntoIterator<Item = usize>,
    state: PcuExecutionNodeState,
) -> Result<(), CudaOwnedExecutionError> {
    for node in lineage {
        if !gate_state.finish(node, state) {
            return Err(CudaOwnedExecutionError::InvalidState);
        }
        let Some(outcome) = outcomes.get_mut(node) else {
            return Err(CudaOwnedExecutionError::InvalidState);
        };
        *outcome = state;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn event_chain_edge_eligible(
    predecessor: usize,
    successor: usize,
    predecessor_state: PcuExecutionNodeState,
    predecessor_checked: bool,
    successor_checked: bool,
    successor_submission_ready: bool,
    consumer_count: usize,
    success_gates: &[PcuExecutionSuccessGate],
) -> bool {
    predecessor_state == PcuExecutionNodeState::Running
        && !predecessor_checked
        && !successor_checked
        && successor_submission_ready
        && consumer_count == 1
        && !success_gates.contains(&PcuExecutionSuccessGate {
            predecessor,
            successor,
        })
}

#[allow(clippy::too_many_arguments)]
fn open_batch_edge_eligible(
    lineage: &[usize],
    predecessor: usize,
    successor: usize,
    submission_ready: bool,
    same_stream: bool,
    consumer_count: usize,
    success_gates: &[PcuExecutionSuccessGate],
) -> bool {
    same_stream
        && !event_chain_lineage_has_success_gate(lineage, successor, success_gates)
        && event_chain_edge_eligible(
            predecessor,
            successor,
            PcuExecutionNodeState::Running,
            false,
            false,
            submission_ready,
            consumer_count,
            success_gates,
        )
}

fn event_chain_lineage_eligible(
    lineage: &[usize],
    predecessor: usize,
    is_batch_completion: bool,
) -> bool {
    lineage.last() == Some(&predecessor) && (is_batch_completion || lineage.len() == 1)
}

fn event_chain_lineage_has_success_gate(
    lineage: &[usize],
    successor: usize,
    success_gates: &[PcuExecutionSuccessGate],
) -> bool {
    lineage.iter().any(|&predecessor| {
        success_gates.contains(&PcuExecutionSuccessGate {
            predecessor,
            successor,
        })
    })
}

fn first_ready_node<'a>(
    dependencies: impl Iterator<Item = &'a [usize]>,
    outcomes: &[PcuExecutionNodeState],
) -> Option<usize> {
    dependencies.enumerate().find_map(|(index, dependencies)| {
        (outcomes.get(index) == Some(&PcuExecutionNodeState::Pending)
            && dependencies.iter().all(|&dependency| {
                outcomes.get(dependency) == Some(&PcuExecutionNodeState::Succeeded)
            }))
        .then_some(index)
    })
}

fn first_blocked_node<'a>(
    dependencies: impl Iterator<Item = &'a [usize]>,
    outcomes: &[PcuExecutionNodeState],
) -> Option<usize> {
    dependencies.enumerate().find_map(|(index, dependencies)| {
        (outcomes.get(index) == Some(&PcuExecutionNodeState::Pending)
            && dependencies.iter().any(|&dependency| {
                matches!(
                    outcomes.get(dependency),
                    Some(PcuExecutionNodeState::Failed | PcuExecutionNodeState::Cancelled)
                )
            }))
        .then_some(index)
    })
}

fn next_pending_slot(pending: [bool; 2], start: usize) -> Option<usize> {
    (0..2)
        .map(|offset| (start + offset) % 2)
        .find(|&slot| pending[slot])
}

#[cfg(test)]
mod tests {
    use super::*;
    #[rustfmt::skip]
    use fusion_pcu::{
        PcuExecutionFault,
        PcuExecutionFaultKind,
        PcuExecutionResourceId,
        PcuMemoryAccess,
    };

    const OUTPUT: PcuExecutionResourceUse = PcuExecutionResourceUse {
        resource: PcuExecutionResourceId(0),
        access: PcuMemoryAccess::WriteOnly,
    };
    const INPUT: PcuExecutionResourceUse = PcuExecutionResourceUse {
        resource: PcuExecutionResourceId(0),
        access: PcuMemoryAccess::ReadOnly,
    };

    #[test]
    fn validation_rejects_fault_output_consumer_without_gate() {
        let nodes = [
            PcuExecutionNode {
                dependencies: &[],
                resources: &[OUTPUT],
            },
            PcuExecutionNode {
                dependencies: &[0],
                resources: &[INPUT],
            },
        ];
        assert_eq!(
            validate_execution_submission(&nodes, 1, &[true, false], &[], &mut [false; 2]),
            Err(PcuExecutionGraphError::FaultOutputNotSuccessGated {
                producer: 0,
                consumer: 1,
                resource: PcuExecutionResourceId(0),
            })
        );
    }

    #[test]
    fn retryable_wait_error_keeps_node_running_and_fault_blocks_consumer() {
        let gates = [PcuExecutionSuccessGate {
            predecessor: 0,
            successor: 1,
        }];
        let mut states = [PcuExecutionNodeState::Pending; 2];
        let mut tracker = PcuExecutionFaultGateState::new(&gates, &mut states);
        assert_eq!(tracker.try_start(0), Some(PcuExecutionAdmission::Started));

        let first = wait_and_finish(&mut tracker, 0, || Err("transient wait error"));
        assert!(matches!(first, Err(CompletionUpdateError::Wait(_))));
        assert_eq!(
            tracker.try_start(0),
            Some(PcuExecutionAdmission::AlreadyStarted)
        );

        let fault = PcuCompletionOutcome::Fault(PcuExecutionFault {
            kind: PcuExecutionFaultKind::DivideByZero,
            invocation_id: 9,
            recovered: false,
        });
        assert!(matches!(
            wait_and_finish(&mut tracker, 0, || Ok::<_, ()>(fault)),
            Ok(outcome) if outcome == fault
        ));
        assert_eq!(tracker.try_start(1), Some(PcuExecutionAdmission::Blocked));
    }

    #[test]
    fn device_readback_is_unavailable_until_ready_and_then_single_use() {
        let mut readbacks = [ExecutionReadback::Pending];
        assert!(matches!(
            take_execution_readback(&mut readbacks, 0),
            Err(CudaOwnedExecutionError::ReadbackNotReady { node: 0 })
        ));
        assert!(matches!(readbacks[0], ExecutionReadback::Pending));

        readbacks[0] = ExecutionReadback::Ready(Box::from([1_u8, 2, 3]));
        let bytes = take_execution_readback(&mut readbacks, 0).unwrap();
        assert_eq!(bytes.as_ref(), &[1, 2, 3]);
        assert!(matches!(readbacks[0], ExecutionReadback::Taken));
        assert!(matches!(
            take_execution_readback(&mut readbacks, 0),
            Err(CudaOwnedExecutionError::ReadbackAlreadyTaken { node: 0 })
        ));
    }

    #[test]
    fn device_readback_absence_is_reported_for_non_readback_and_unknown_nodes() {
        let mut readbacks = [ExecutionReadback::NotPresent];
        assert!(matches!(
            take_execution_readback(&mut readbacks, 0),
            Err(CudaOwnedExecutionError::ReadbackNotPresent { node: 0 })
        ));
        assert!(matches!(
            take_execution_readback(&mut readbacks, 1),
            Err(CudaOwnedExecutionError::ReadbackNotPresent { node: 1 })
        ));
    }

    #[test]
    fn device_readback_resources_are_read_only() {
        assert_eq!(
            device_readback_resource(PcuExecutionResourceId(7)),
            PcuExecutionResourceUse {
                resource: PcuExecutionResourceId(7),
                access: PcuMemoryAccess::ReadOnly,
            }
        );
    }

    #[test]
    fn readback_planner_requires_terminal_nodes_for_direct_and_gated_edges() {
        let no_gates = [];
        assert_eq!(
            first_readback_successor(&[true, false], &[&[], &[0]], &no_gates),
            Some((0, 1))
        );
        assert_eq!(
            first_readback_successor(&[false, true], &[&[], &[0]], &no_gates),
            None
        );

        let gates = [PcuExecutionSuccessGate {
            predecessor: 0,
            successor: 1,
        }];
        assert_eq!(
            first_readback_successor(&[true, false], &[&[], &[]], &gates),
            Some((0, 1))
        );
    }

    #[test]
    fn two_slot_readiness_admits_independent_nodes_before_waiting() {
        let dependencies: [&[usize]; 3] = [&[], &[], &[0]];
        let mut states = [PcuExecutionNodeState::Pending; 3];

        let first = first_ready_node(dependencies.iter().copied(), &states).unwrap();
        assert_eq!(first, 0);
        states[first] = PcuExecutionNodeState::Running;

        let second = first_ready_node(dependencies.iter().copied(), &states).unwrap();
        assert_eq!(second, 1);
        states[second] = PcuExecutionNodeState::Running;

        // The dependent node cannot be submitted until node 0 has terminal success.
        assert_eq!(
            first_ready_node(dependencies.iter().copied(), &states),
            None
        );
        states[0] = PcuExecutionNodeState::Succeeded;
        assert_eq!(
            first_ready_node(dependencies.iter().copied(), &states),
            Some(2)
        );
    }

    #[test]
    fn failed_predecessor_blocks_consumer_and_wait_retry_does_not_starve_other_slot() {
        let dependencies: [&[usize]; 2] = [&[], &[0]];
        let states = [
            PcuExecutionNodeState::Failed,
            PcuExecutionNodeState::Pending,
        ];
        assert_eq!(
            first_blocked_node(dependencies.iter().copied(), &states),
            Some(1)
        );
        assert_eq!(
            first_ready_node(dependencies.iter().copied(), &states),
            None
        );

        assert_eq!(next_pending_slot([true, true], 0), Some(0));
        // A transient error in slot 0 advances the retry cursor, allowing slot 1 to reap.
        assert_eq!(next_pending_slot([true, true], 1), Some(1));
        assert_eq!(next_pending_slot([true, false], 1), Some(0));
    }

    #[test]
    fn event_chain_admits_only_unchecked_ordinary_single_consumer_edges() {
        let ordinary = [];
        let gate = [PcuExecutionSuccessGate {
            predecessor: 0,
            successor: 1,
        }];
        assert!(event_chain_edge_eligible(
            0,
            1,
            PcuExecutionNodeState::Running,
            false,
            false,
            true,
            1,
            &ordinary,
        ));
        assert!(!event_chain_edge_eligible(
            0,
            1,
            PcuExecutionNodeState::Running,
            false,
            false,
            false,
            1,
            &ordinary,
        ));
        assert!(!event_chain_edge_eligible(
            0,
            1,
            PcuExecutionNodeState::Running,
            true,
            false,
            true,
            1,
            &ordinary,
        ));
        assert!(!event_chain_edge_eligible(
            0,
            1,
            PcuExecutionNodeState::Running,
            false,
            true,
            true,
            1,
            &ordinary,
        ));
        assert!(!event_chain_edge_eligible(
            0,
            1,
            PcuExecutionNodeState::Running,
            false,
            false,
            true,
            2,
            &ordinary,
        ));
        assert!(!event_chain_edge_eligible(
            0,
            1,
            PcuExecutionNodeState::Running,
            false,
            false,
            true,
            1,
            &gate,
        ));
        assert!(!event_chain_edge_eligible(
            0,
            1,
            PcuExecutionNodeState::Pending,
            false,
            false,
            true,
            1,
            &ordinary,
        ));
    }

    #[test]
    fn event_chain_falls_back_to_wait_when_sgemm_handle_is_busy() {
        assert!(!event_chain_edge_eligible(
            0,
            1,
            PcuExecutionNodeState::Running,
            false,
            false,
            false,
            1,
            &[],
        ));
    }

    #[test]
    fn open_batch_requires_ready_same_stream_linear_ungated_edge() {
        let gates = [PcuExecutionSuccessGate {
            predecessor: 0,
            successor: 1,
        }];
        assert!(open_batch_edge_eligible(&[0], 0, 1, true, true, 1, &[]));
        assert!(!open_batch_edge_eligible(&[0], 0, 1, false, true, 1, &[]));
        assert!(!open_batch_edge_eligible(&[0], 0, 1, true, false, 1, &[]));
        assert!(!open_batch_edge_eligible(&[0], 0, 1, true, true, 2, &[]));
        assert!(!open_batch_edge_eligible(&[0], 0, 1, true, true, 1, &gates));

        let gated_ancestor = [PcuExecutionSuccessGate {
            predecessor: 0,
            successor: 2,
        }];
        assert!(!open_batch_edge_eligible(
            &[0, 1],
            1,
            2,
            true,
            true,
            1,
            &gated_ancestor
        ));
    }

    #[test]
    fn event_chain_extends_only_the_tail_of_a_retained_linear_lineage() {
        assert!(event_chain_lineage_eligible(&[0], 0, false));
        assert!(event_chain_lineage_eligible(&[0, 1], 1, true));
        assert!(event_chain_lineage_eligible(&[0, 1, 2], 2, true));
        assert!(!event_chain_lineage_eligible(&[0, 1], 0, true));
        assert!(!event_chain_lineage_eligible(&[0, 1], 1, false));
    }

    #[test]
    fn event_chain_does_not_cross_a_success_gate_from_any_lineage_ancestor() {
        let gated_ancestor = [PcuExecutionSuccessGate {
            predecessor: 0,
            successor: 2,
        }];
        let gated_tail = [PcuExecutionSuccessGate {
            predecessor: 1,
            successor: 2,
        }];
        let unrelated_gate = [PcuExecutionSuccessGate {
            predecessor: 0,
            successor: 3,
        }];
        assert!(event_chain_lineage_has_success_gate(
            &[0, 1],
            2,
            &gated_ancestor
        ));
        assert!(event_chain_lineage_has_success_gate(
            &[0, 1],
            2,
            &gated_tail
        ));
        assert!(!event_chain_lineage_has_success_gate(
            &[0, 1],
            2,
            &unrelated_gate
        ));
    }

    #[test]
    fn event_chained_lineage_opens_gates_only_after_batch_terminal_success() {
        let gates = [PcuExecutionSuccessGate {
            predecessor: 0,
            successor: 2,
        }];
        let mut states = [PcuExecutionNodeState::Pending; 3];
        let mut tracker = PcuExecutionFaultGateState::new(&gates, &mut states);
        assert_eq!(tracker.try_start(0), Some(PcuExecutionAdmission::Started));
        assert_eq!(tracker.try_start(1), Some(PcuExecutionAdmission::Started));
        assert_eq!(tracker.try_start(2), Some(PcuExecutionAdmission::Waiting));

        assert!(tracker.finish(0, PcuExecutionNodeState::Succeeded));
        assert!(tracker.finish(1, PcuExecutionNodeState::Succeeded));
        assert_eq!(tracker.try_start(2), Some(PcuExecutionAdmission::Started));
    }

    #[test]
    fn failed_lineage_marks_every_node_failed_and_keeps_success_gates_closed() {
        let gates = [
            PcuExecutionSuccessGate {
                predecessor: 0,
                successor: 2,
            },
            PcuExecutionSuccessGate {
                predecessor: 1,
                successor: 3,
            },
        ];
        let mut states = [PcuExecutionNodeState::Pending; 4];
        let mut outcomes = [PcuExecutionNodeState::Pending; 4];
        {
            let mut tracker = PcuExecutionFaultGateState::new(&gates, &mut states);
            assert_eq!(tracker.try_start(0), Some(PcuExecutionAdmission::Started));
            assert_eq!(tracker.try_start(1), Some(PcuExecutionAdmission::Started));
            outcomes[0] = PcuExecutionNodeState::Running;
            outcomes[1] = PcuExecutionNodeState::Running;

            finish_lineage_state(
                &mut tracker,
                &mut outcomes,
                [0, 1],
                PcuExecutionNodeState::Failed,
            )
            .unwrap();

            assert_eq!(tracker.try_start(2), Some(PcuExecutionAdmission::Blocked));
            assert_eq!(tracker.try_start(3), Some(PcuExecutionAdmission::Blocked));
            assert!(tracker.cancel_pending(2));
            assert!(tracker.cancel_pending(3));
        }

        assert_eq!(
            outcomes,
            [
                PcuExecutionNodeState::Failed,
                PcuExecutionNodeState::Failed,
                PcuExecutionNodeState::Pending,
                PcuExecutionNodeState::Pending,
            ]
        );
        assert_eq!(
            states,
            [
                PcuExecutionNodeState::Failed,
                PcuExecutionNodeState::Failed,
                PcuExecutionNodeState::Cancelled,
                PcuExecutionNodeState::Cancelled,
            ]
        );
    }

    #[test]
    fn cloned_allocation_handles_share_resource_identity_and_cannot_claim_disjointness() {
        let allocation = Rc::new(());
        let cloned_handle = Rc::clone(&allocation);
        let mut ids = AllocationIds::new();
        let first_id = ids.id(&allocation).unwrap();
        let clone_id = ids.id(&cloned_handle).unwrap();
        assert_eq!(first_id, clone_id);

        let mut writer = Vec::new();
        record_resource(&mut writer, first_id, PcuMemoryAccess::WriteOnly);
        let mut reader = Vec::new();
        record_resource(&mut reader, clone_id, PcuMemoryAccess::ReadOnly);
        let nodes = [
            PcuExecutionNode {
                dependencies: &[],
                resources: &writer,
            },
            PcuExecutionNode {
                dependencies: &[],
                resources: &reader,
            },
        ];
        assert_eq!(
            fusion_pcu::validate_execution_graph(&nodes, 1, &mut [false; 2]),
            Err(PcuExecutionGraphError::UnorderedConflict {
                earlier: 0,
                later: 1,
                resource: first_id,
            })
        );
    }
}
