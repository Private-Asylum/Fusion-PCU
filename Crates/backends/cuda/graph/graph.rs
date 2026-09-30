//! Retained native CUDA graph replay for validated PCU dispatches.
//!
//! A private capture stream prevents unrelated submissions entering the graph. Allocation leases
//! remain exclusive for the graph's entire lifetime. Quiescent transfers use the graph owner.
//! Checked capture permits exactly one kernel: every replay resets a private sentinel, waits,
//! then reads its terminal status. No dependent kernel can consume faulted output.

#[rustfmt::skip]
use std::{
    error::Error,
    fmt,
    ptr,
};
#[rustfmt::skip]
use crate::{
    CudaCompletionBatch,
    CudaError,
    CudaOwnedDispatchError,
    CudaPreparedDispatch,
    CudaRuntime,
    DeviceBuffer,
};
#[rustfmt::skip]
use crate::ffi::runtime::{
    CudaGraph,
    CudaGraphExec,
};
#[rustfmt::skip]
use fusion_pcu::{
    PcuCompletionOutcome,
    PcuOwnedBinding,
};
#[rustfmt::skip]
use super::{
    CUDA_MEMCPY_DEVICE_TO_HOST,
    CUDA_MEMCPY_HOST_TO_DEVICE,
    LaunchAccessLeases,
    LaunchResources,
    ensure_batch_open,
    owned_dispatch::decode_fault_word,
};

/// One validated prepared dispatch and its owned device bindings to capture.
pub struct CudaGraphDispatch<'a> {
    pub dispatch: &'a CudaPreparedDispatch,
    pub bindings: &'a [PcuOwnedBinding<DeviceBuffer>],
}

/// Native graph setup or replay failure.
#[derive(Debug)]
pub enum CudaNativeGraphError {
    Cuda(CudaError),
    Dispatch(CudaOwnedDispatchError),
    Empty,
    CheckedArithmeticUnsupported,
    CheckedOutcomeRequired,
    Poisoned,
}

impl fmt::Display for CudaNativeGraphError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cuda(error) => error.fmt(f),
            Self::Dispatch(error) => error.fmt(f),
            Self::Empty => f.write_str("CUDA native graph requires at least one dispatch"),
            Self::CheckedArithmeticUnsupported => {
                f.write_str("checked CUDA native graph requires exactly one dispatch")
            }
            Self::CheckedOutcomeRequired => {
                f.write_str("checked CUDA graph replay requires terminal outcome observation")
            }
            Self::Poisoned => {
                f.write_str("CUDA native graph is poisoned after uncertain replay completion")
            }
        }
    }
}
impl Error for CudaNativeGraphError {}
impl From<CudaError> for CudaNativeGraphError {
    fn from(error: CudaError) -> Self {
        Self::Cuda(error)
    }
}

/// An instantiated native CUDA graph retaining every captured module and device allocation.
///
/// Replay waits for completion before returning. This owner is neither `Send` nor `Sync`, and
/// mutable replay prevents concurrent access to the executable. Drop releases its allocation
/// leases; an uncertain completion instead quarantines them permanently.
pub struct CudaNativeGraph {
    runtime: CudaRuntime,
    executable: CudaGraphExec,
    retained: Option<CudaCompletionBatch>,
    poisoned: bool,
    fault_word: Option<DeviceBuffer>,
}

impl CudaRuntime {
    /// Capture validated dispatches; a checked graph must contain exactly one dispatch.
    ///
    /// The bindings are retained without borrowing the input slice. Allocation contents are
    /// captured by address, so successive replays consume the preceding replay's device results.
    ///
    /// # Errors
    /// Returns an empty sequence, a checked multi-dispatch sequence, binding validation, capture,
    /// or instantiation error. Checked multi-dispatch sequences are rejected before capture.
    pub fn capture_dispatch_graph(
        &self,
        dispatches: &[CudaGraphDispatch<'_>],
    ) -> Result<CudaNativeGraph, CudaNativeGraphError> {
        validate_dispatches(dispatches)?;
        let stream = self.create_stream()?;
        let mut retained = CudaCompletionBatch::new(&stream);
        let fault_word = if dispatches[0].dispatch.requires_checked_fault_word() {
            Some(self.allocate(size_of::<u64>())?)
        } else {
            None
        };
        // Thread-local capture is sufficient: the private stream never escapes this owner.
        unsafe { crate::ffi::invoke_cudaStreamBeginCapture(self, stream.raw_stream(), 1) }?;
        let submission = if let Some(fault_word) = fault_word.as_ref() {
            retained
                .graph_reset_fault_word(fault_word)
                .map_err(CudaOwnedDispatchError::Cuda)
                .and_then(|()| {
                    dispatches[0].dispatch.submit_checked_into_batch(
                        dispatches[0].bindings,
                        &mut retained,
                        fault_word,
                    )
                })
        } else {
            dispatches.iter().try_for_each(|node| {
                node.dispatch
                    .submit_into_batch(node.bindings, &mut retained)
            })
        };
        // Always end capture before dropping the retained batch; synchronization during capture
        // is forbidden. A failed submission may invalidate capture and leave a null graph.
        let mut graph: CudaGraph = ptr::null_mut();
        let ended = unsafe {
            crate::ffi::invoke_cudaStreamEndCapture(self, stream.raw_stream(), &raw mut graph)
        };
        if let Err(error) = submission {
            destroy_graph(self, graph);
            return Err(CudaNativeGraphError::Dispatch(error));
        }
        if let Err(error) = ended {
            destroy_graph(self, graph);
            return Err(error.into());
        }
        let mut executable: CudaGraphExec = ptr::null_mut();
        let instantiated = unsafe {
            crate::ffi::invoke_cudaGraphInstantiateWithFlags(self, &raw mut executable, graph, 0)
        };
        destroy_graph(self, graph);
        instantiated?;
        Ok(CudaNativeGraph {
            runtime: self.clone(),
            executable,
            retained: Some(retained),
            poisoned: false,
            fault_word,
        })
    }
}

fn validate_dispatches(dispatches: &[CudaGraphDispatch<'_>]) -> Result<(), CudaNativeGraphError> {
    if dispatches.is_empty() {
        return Err(CudaNativeGraphError::Empty);
    }
    validate_checked_capture(
        dispatches.len(),
        dispatches
            .iter()
            .any(|node| node.dispatch.requires_checked_fault_word()),
    )
}

const fn validate_checked_capture(nodes: usize, checked: bool) -> Result<(), CudaNativeGraphError> {
    if checked && nodes != 1 {
        return Err(CudaNativeGraphError::CheckedArithmeticUnsupported);
    }
    Ok(())
}

const fn validate_replay(
    poisoned: bool,
    checked: bool,
    observes_outcome: bool,
) -> Result<(), CudaNativeGraphError> {
    if poisoned {
        return Err(CudaNativeGraphError::Poisoned);
    }
    if checked && !observes_outcome {
        return Err(CudaNativeGraphError::CheckedOutcomeRequired);
    }
    Ok(())
}

fn destroy_graph(runtime: &CudaRuntime, graph: CudaGraph) {
    if !graph.is_null() {
        let _ = unsafe { crate::ffi::invoke_cudaGraphDestroy(runtime, graph) };
    }
}

impl CudaNativeGraph {
    /// Capture using the runtime owned by the first prepared dispatch.
    ///
    /// # Errors
    /// Returns the same validation and CUDA errors as [`CudaRuntime::capture_dispatch_graph`].
    pub fn capture(dispatches: &[CudaGraphDispatch<'_>]) -> Result<Self, CudaNativeGraphError> {
        validate_dispatches(dispatches)?;
        let runtime = dispatches[0].dispatch.stream_handle().inner.runtime.clone();
        runtime.capture_dispatch_graph(dispatches)
    }

    /// Launch this retained executable and wait for all graph work to finish.
    ///
    /// # Errors
    /// Returns the CUDA launch or wait error. Any such error poisons the executable, even if a
    /// subsequent synchronization proves the resources can safely be released.
    pub fn replay_and_wait(&mut self) -> Result<(), CudaNativeGraphError> {
        validate_replay(self.poisoned, self.fault_word.is_some(), false)?;
        self.launch_and_wait()
    }

    /// Replay and observe the terminal arithmetic outcome. Every checked replay resets status.
    /// A reported arithmetic fault is terminal and retryable; CUDA uncertainty poisons the owner.
    /// Output from a fatal fault must not be consumed. Recovered clamp output remains readable.
    ///
    /// # Errors
    /// Returns launch, synchronization, transfer, invalid-status, or poisoned-owner errors.
    pub fn replay_checked_and_wait(
        &mut self,
    ) -> Result<PcuCompletionOutcome, CudaNativeGraphError> {
        self.launch_and_wait()?;
        let Some(fault_word) = self.fault_word.clone() else {
            return Ok(PcuCompletionOutcome::Succeeded);
        };
        let mut bytes = [0; size_of::<u64>()];
        let outcome = self.readback(&fault_word, 0, &mut bytes).and_then(|()| {
            decode_fault_word(u64::from_le_bytes(bytes))
                .map(|fault| {
                    fault.map_or(PcuCompletionOutcome::Succeeded, PcuCompletionOutcome::Fault)
                })
                .map_err(Into::into)
        });
        if outcome.is_err() {
            self.poisoned = true;
        }
        outcome
    }

    /// Refresh a retained allocation after proving graph-stream quiescence.
    ///
    /// # Errors
    /// Returns an unretained allocation, range, CUDA transfer, or poisoned-owner error.
    pub fn refresh_input(
        &mut self,
        buffer: &DeviceBuffer,
        offset: usize,
        bytes: &[u8],
    ) -> Result<(), CudaNativeGraphError> {
        self.transfer(|batch| batch.graph_copy_from(buffer, offset, bytes))
    }

    /// Read a retained allocation after proving graph-stream quiescence.
    ///
    /// # Errors
    /// Returns an unretained allocation, range, CUDA transfer, or poisoned-owner error.
    pub fn readback(
        &mut self,
        buffer: &DeviceBuffer,
        offset: usize,
        bytes: &mut [u8],
    ) -> Result<(), CudaNativeGraphError> {
        self.transfer(|batch| batch.graph_copy_to(buffer, offset, bytes))
    }

    fn transfer(
        &mut self,
        copy: impl FnOnce(&CudaCompletionBatch) -> Result<(), CudaError>,
    ) -> Result<(), CudaNativeGraphError> {
        if self.poisoned {
            return Err(CudaNativeGraphError::Poisoned);
        }
        let batch = self
            .retained
            .as_ref()
            .ok_or(CudaNativeGraphError::Poisoned)?;
        let result = copy(batch);
        // Transfers retain leases and synchronize on CUDA errors. Preserve the graph owner until
        // a final stream wait proves release safe; uncertainty must quarantine every capture owner.
        if result.is_err() && batch.stream.synchronize().is_err() {
            self.poisoned = true;
            if let Some(mut batch) = self.retained.take() {
                batch.quarantine_and_forget();
                std::mem::forget(batch);
            }
        }
        result.map_err(Into::into)
    }

    fn launch_and_wait(&mut self) -> Result<(), CudaNativeGraphError> {
        if self.poisoned {
            return Err(CudaNativeGraphError::Poisoned);
        }
        let retained = self
            .retained
            .as_ref()
            .ok_or(CudaNativeGraphError::Poisoned)?;
        let launch = unsafe {
            crate::ffi::invoke_cudaGraphLaunch(
                &self.runtime,
                self.executable,
                retained.stream.raw_stream(),
            )
        };
        let completion = retained.stream.synchronize();
        if launch.is_err() || completion.is_err() {
            self.poisoned = true;
        }
        if completion.is_err()
            && let Some(mut retained) = self.retained.take()
        {
            retained.quarantine_and_forget();
            std::mem::forget(retained);
        }
        launch?;
        completion?;
        Ok(())
    }
}

// These helpers belong to retained graph ownership: the private stream never escapes, and
// access is serialized by the graph's exclusive borrow. Ordinary DeviceBuffer clones stay busy.
impl CudaCompletionBatch {
    fn graph_reset_fault_word(&mut self, buffer: &DeviceBuffer) -> Result<(), CudaError> {
        ensure_batch_open(self.failed)?;
        self.stream
            .inner
            .runtime
            .ensure_same_runtime(&buffer.allocation.runtime)?;
        let mut access_leases = LaunchAccessLeases::new();
        access_leases.push(buffer.acquire_stream_access(&self.stream)?);
        self.resources.push(LaunchResources {
            _module: None,
            _external_owner: None,
            access_leases,
            stream: self.stream.clone(),
        });
        let result = unsafe {
            crate::ffi::invoke_cudaMemsetAsync(
                &self.stream.inner.runtime,
                buffer.allocation.pointer,
                0xff,
                size_of::<u64>(),
                self.stream.raw_stream(),
            )
        };
        // During capture, do not synchronize on error; caller must end capture first.
        if result.is_err() {
            self.failed = true;
        }
        result
    }

    fn graph_validate_transfer(
        &self,
        buffer: &DeviceBuffer,
        offset: usize,
        bytes: usize,
    ) -> Result<(), CudaError> {
        ensure_batch_open(self.failed)?;
        self.stream
            .inner
            .runtime
            .ensure_same_runtime(&buffer.allocation.runtime)?;
        buffer.check_range(offset, bytes)?;
        if !self
            .resources
            .iter()
            .any(|resource| resource.access_leases.contains(&buffer.allocation))
        {
            return Err(CudaError::Busy);
        }
        self.stream.synchronize()
    }

    fn graph_copy_from(
        &self,
        buffer: &DeviceBuffer,
        offset: usize,
        bytes: &[u8],
    ) -> Result<(), CudaError> {
        self.graph_validate_transfer(buffer, offset, bytes.len())?;
        let lease = buffer.acquire_stream_access(&self.stream)?;
        let result = unsafe {
            crate::ffi::invoke_cudaMemcpy(
                &self.stream.inner.runtime,
                buffer
                    .allocation
                    .pointer
                    .cast::<u8>()
                    .wrapping_add(offset)
                    .cast(),
                bytes.as_ptr().cast(),
                bytes.len(),
                CUDA_MEMCPY_HOST_TO_DEVICE,
            )
        };
        lease.finish_synchronous(result)
    }

    fn graph_copy_to(
        &self,
        buffer: &DeviceBuffer,
        offset: usize,
        bytes: &mut [u8],
    ) -> Result<(), CudaError> {
        self.graph_validate_transfer(buffer, offset, bytes.len())?;
        let lease = buffer.acquire_stream_access(&self.stream)?;
        let result = unsafe {
            crate::ffi::invoke_cudaMemcpy(
                &self.stream.inner.runtime,
                bytes.as_mut_ptr().cast(),
                buffer
                    .allocation
                    .pointer
                    .cast::<u8>()
                    .wrapping_add(offset)
                    .cast(),
                bytes.len(),
                CUDA_MEMCPY_DEVICE_TO_HOST,
            )
        };
        lease.finish_synchronous(result)
    }
}

impl Drop for CudaNativeGraph {
    fn drop(&mut self) {
        // Synchronize while the executable and every resource are still owned.
        if let Some(retained) = self.retained.as_mut() {
            if retained.stream.synchronize().is_err() {
                retained.quarantine_and_forget();
                if let Some(retained) = self.retained.take() {
                    std::mem::forget(retained);
                }
                // CUDA may still access graph metadata; quarantine the executable too.
                return;
            }
        } else if self.poisoned {
            return;
        }
        let _ = unsafe { crate::ffi::invoke_cudaGraphExecDestroy(&self.runtime, self.executable) };
    }
}

#[cfg(test)]
mod tests {
    #[rustfmt::skip]
    use super::{
        *,
    };
    #[test]
    fn empty_graph_is_rejected_without_loading_cuda() {
        assert!(matches!(
            validate_dispatches(&[]),
            Err(CudaNativeGraphError::Empty)
        ));
    }

    #[test]
    fn checked_capture_is_rejected_before_reusing_a_fault_sentinel() {
        assert!(matches!(
            validate_checked_capture(2, true),
            Err(CudaNativeGraphError::CheckedArithmeticUnsupported)
        ));
        assert!(validate_checked_capture(2, false).is_ok());
        assert!(validate_checked_capture(1, true).is_ok());
    }
    #[test]
    fn checked_capture_never_admits_work_behind_a_fault() {
        for nodes in [0, 2, 3, 8] {
            assert!(validate_checked_capture(nodes, true).is_err());
        }
        assert!(validate_checked_capture(1, true).is_ok());
    }

    #[test]
    fn poisoned_replays_are_rejected_and_checked_outcomes_cannot_be_discarded() {
        for checked in [false, true] {
            for observes_outcome in [false, true] {
                assert!(matches!(
                    validate_replay(true, checked, observes_outcome),
                    Err(CudaNativeGraphError::Poisoned)
                ));
            }
        }
        assert!(matches!(
            validate_replay(false, true, false),
            Err(CudaNativeGraphError::CheckedOutcomeRequired)
        ));
        assert!(validate_replay(false, true, true).is_ok());
        assert!(validate_replay(false, false, false).is_ok());
    }

    #[test]
    fn terminal_status_preserves_fatal_and_recovered_faults() {
        assert_eq!(decode_fault_word(u64::MAX).unwrap(), None);
        let fatal = decode_fault_word((4 << 3) | 3).unwrap().unwrap();
        assert_eq!(fatal.invocation_id, 4);
        assert!(!fatal.recovered);
        let clamp = decode_fault_word((1 << 63) | (4 << 3) | 3)
            .unwrap()
            .unwrap();
        assert_eq!(clamp.invocation_id, 4);
        assert!(clamp.recovered);
        assert!(decode_fault_word(0).is_err());
    }
}

#[cfg(test)]
mod hardware_tests {
    #[rustfmt::skip]
    use super::{
        *,
    };
    #[rustfmt::skip]
    use crate::{
        CudaDiscovery,
        CudaKernelArgument,
        CudaOwnedDispatchBackend,
    };
    #[rustfmt::skip]
    use fusion_pcu::{
        PcuBinding,
        PcuBindingAccess,
        PcuBindingRef,
        PcuBindingStorageClass,
        PcuBindingType,
        PcuDeviceClass,
        PcuDeviceDescriptor,
        PcuDispatchControlOp,
        PcuDispatchDataOp,
        PcuDispatchEntryPoint,
        PcuDispatchFeatureCaps,
        PcuDispatchFloatBinaryOp,
        PcuExecutionFault,
        PcuExecutionFaultKind,
        PcuFloatUnderflowPolicy,
        PcuRangePolicy,
        PcuDispatchIndex,
        PcuDispatchKernelIr,
        PcuDispatchOp,
        PcuDispatchSubmission,
        PcuDispatchValueId,
        PcuInvocationShape,
        PcuKernelId,
        PcuObjectKind,
        PcuObjectRef,
        PcuProviderDescriptor,
        PcuProviderId,
        PcuProviderReadiness,
        PcuProviderStatus,
        PcuRuntimeDiscovery,
        PcuTargetDescriptor,
        PcuValueType,
        PcuValueTypeCaps,
    };

    #[test]
    #[ignore = "requires CUDA GPU and device access"]
    fn native_graph_replays_retained_kernel_and_releases_readback_after_drop() {
        let runtime = CudaRuntime::new(0).expect("CUDA runtime");
        let image = crate::compile_cuda_source_for_device(
            &runtime,
            r#"
extern "C" __global__ void increment(unsigned int* value) {
    if (blockIdx.x == 0 && threadIdx.x == 0) value[0] += 1u;
}
"#,
        )
        .expect("compile kernel");
        let module = runtime.load_module(&image).expect("load module");
        let kernel = module.function(c"increment").expect("resolve kernel");
        let mut buffer = runtime.allocate(size_of::<u32>()).expect("allocation");
        buffer.copy_from(&0_u32.to_ne_bytes()).expect("initialize");
        let stream = runtime.create_stream().expect("stream");
        let mut retained = CudaCompletionBatch::new(&stream);
        unsafe { crate::ffi::invoke_cudaStreamBeginCapture(&runtime, stream.raw_stream(), 1) }
            .expect("begin capture");
        // SAFETY: the compiled kernel takes exactly one pointer to a live u32 allocation.
        unsafe {
            kernel
                .launch_into_batch(
                    &mut retained,
                    [1; 3],
                    [1; 3],
                    0,
                    &[CudaKernelArgument::Buffer(&buffer)],
                )
                .expect("capture launch");
        }
        let mut graph = ptr::null_mut();
        unsafe {
            crate::ffi::invoke_cudaStreamEndCapture(&runtime, stream.raw_stream(), &raw mut graph)
        }
        .expect("end capture");
        let mut executable = ptr::null_mut();
        unsafe {
            crate::ffi::invoke_cudaGraphInstantiateWithFlags(
                &runtime,
                &raw mut executable,
                graph,
                0,
            )
        }
        .expect("instantiate");
        destroy_graph(&runtime, graph);
        let mut executable = CudaNativeGraph {
            runtime,
            executable,
            retained: Some(retained),
            poisoned: false,
            fault_word: None,
        };
        drop(kernel);
        drop(module);
        for _ in 0..8 {
            executable.replay_and_wait().expect("replay retained graph");
        }
        let mut actual = [0; size_of::<u32>()];
        assert!(matches!(buffer.copy_to(&mut actual), Err(CudaError::Busy)));
        drop(executable);
        buffer
            .copy_to(&mut actual)
            .expect("read after graph release");
        assert_eq!(u32::from_ne_bytes(actual), 8);
    }
    fn selected_device() -> (CudaDiscovery, CudaOwnedDispatchBackend) {
        let discovery = CudaDiscovery::new();
        let invalid = PcuObjectRef {
            provider: PcuProviderId(0),
            generation: 0,
            kind: PcuObjectKind::Device,
            id: 0,
        };
        let mut providers = [PcuProviderDescriptor {
            id: PcuProviderId(0),
            generation: 0,
            backend: "",
            readiness: PcuProviderReadiness {
                status: PcuProviderStatus::Unavailable,
                reason: None,
            },
        }];
        assert_eq!(discovery.providers(&mut providers).unwrap(), 1);
        let mut targets = [PcuTargetDescriptor {
            reference: invalid,
            name: "",
            readiness: PcuProviderReadiness {
                status: PcuProviderStatus::Unavailable,
                reason: None,
            },
        }];
        assert_eq!(
            discovery
                .targets(providers[0].id, providers[0].generation, &mut targets)
                .unwrap(),
            1
        );
        let count = discovery.devices(targets[0].reference, &mut []).unwrap();
        assert!(count > 0, "test requires a visible CUDA device");
        let mut devices = vec![
            PcuDeviceDescriptor {
                reference: invalid,
                target: invalid,
                name: "",
                class: PcuDeviceClass::Other,
                vendor: None,
                architecture: None,
                generation: None,
                location: None,
            };
            count
        ];
        discovery
            .devices(targets[0].reference, &mut devices)
            .unwrap();
        let session = CudaOwnedDispatchBackend::open(&discovery, devices[0].reference, 2)
            .expect("open selected CUDA device");
        (discovery, session)
    }

    fn prepare_identity(backend: &CudaOwnedDispatchBackend) -> CudaPreparedDispatch {
        let declarations = [
            PcuBinding::value(
                None,
                0,
                0,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::ReadOnly,
                PcuValueType::u32(),
            ),
            PcuBinding::value(
                None,
                0,
                1,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::WriteOnly,
                PcuValueType::u32(),
            ),
        ];
        let ops = [
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result: PcuDispatchValueId(1),
                binding: PcuBindingRef::new(0, 0),
                index: PcuDispatchIndex::InvocationId,
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                binding: PcuBindingRef::new(0, 1),
                index: PcuDispatchIndex::InvocationId,
                value: PcuDispatchValueId(1),
            }),
            PcuDispatchOp::Control(PcuDispatchControlOp::Return),
        ];
        let kernel = PcuDispatchKernelIr {
            id: PcuKernelId(0x670),
            entry: PcuDispatchEntryPoint {
                name: "graph_identity",
                logical_shape: [4, 1, 1],
            },
            bindings: &declarations,
            ports: &[],
            parameters: &[],
            ops: &ops,
            type_caps: PcuValueTypeCaps::UINT32,
            feature_caps: PcuDispatchFeatureCaps::READ_ONLY_RESOURCES
                .union(PcuDispatchFeatureCaps::MUTABLE_RESOURCES),
        };
        backend
            .prepare_dispatch(PcuDispatchSubmission {
                kernel: &kernel,
                shape: PcuInvocationShape::invocations(std::num::NonZeroU32::new(4).unwrap()),
            })
            .expect("prepare identity dispatch")
    }

    #[test]
    #[ignore = "requires CUDA GPU and device access"]
    fn public_dispatch_graph_retains_dependent_identity_nodes_for_replay() {
        let (_discovery, backend) = selected_device();
        let prepared = prepare_identity(&backend);
        let mut source = backend.allocate(16).expect("source");
        let middle = backend.allocate(16).expect("middle");
        let output = backend.allocate(16).expect("output");
        let expected: Vec<u8> = [0_u32, 1, u32::MAX, 0x1234_5678]
            .into_iter()
            .flat_map(u32::to_ne_bytes)
            .collect();
        source.copy_from(&expected).expect("initialize input");
        let bind = |input: DeviceBuffer, output: DeviceBuffer| {
            [
                backend
                    .binding(
                        PcuBindingRef::new(0, 0),
                        PcuBindingAccess::ReadOnly,
                        PcuBindingType::Value(PcuValueType::u32()),
                        input,
                    )
                    .expect("input binding"),
                backend
                    .binding(
                        PcuBindingRef::new(0, 1),
                        PcuBindingAccess::WriteOnly,
                        PcuBindingType::Value(PcuValueType::u32()),
                        output,
                    )
                    .expect("output binding"),
            ]
        };
        let first = bind(source.clone(), middle.clone());
        let second = bind(middle, output.clone());
        let mut graph = CudaNativeGraph::capture(&[
            CudaGraphDispatch {
                dispatch: &prepared,
                bindings: &first,
            },
            CudaGraphDispatch {
                dispatch: &prepared,
                bindings: &second,
            },
        ])
        .expect("capture validated dependent identity nodes");
        drop(first);
        drop(second);
        drop(prepared);
        drop(source);
        drop(backend);
        for _ in 0..8 {
            graph.replay_and_wait().expect("replay public graph");
        }
        let mut actual = [0_u8; 16];
        assert!(matches!(output.copy_to(&mut actual), Err(CudaError::Busy)));
        drop(graph);
        output
            .copy_to(&mut actual)
            .expect("read graph output after release");
        assert_eq!(actual.as_slice(), expected);
    }

    fn prepare_checked_add(
        backend: &CudaOwnedDispatchBackend,
        policy: PcuRangePolicy,
    ) -> CudaPreparedDispatch {
        let declarations = std::array::from_fn::<_, 3, _>(|slot| {
            PcuBinding::value(
                None,
                0,
                u32::try_from(slot).unwrap(),
                PcuBindingStorageClass::Storage,
                if slot == 2 {
                    PcuBindingAccess::WriteOnly
                } else {
                    PcuBindingAccess::ReadOnly
                },
                PcuValueType::f32(),
            )
        });
        let ops = [
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result: PcuDispatchValueId(1),
                binding: PcuBindingRef::new(0, 0),
                index: PcuDispatchIndex::InvocationId,
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result: PcuDispatchValueId(2),
                binding: PcuBindingRef::new(0, 1),
                index: PcuDispatchIndex::InvocationId,
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::CheckedFloatBinary {
                value_type: PcuValueType::f32(),
                op: PcuDispatchFloatBinaryOp::Add,
                underflow_policy: PcuFloatUnderflowPolicy::IeeeAfterRounding,
                range_policy: policy,
                result: PcuDispatchValueId(3),
                lhs: PcuDispatchValueId(1),
                rhs: PcuDispatchValueId(2),
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                binding: PcuBindingRef::new(0, 2),
                index: PcuDispatchIndex::InvocationId,
                value: PcuDispatchValueId(3),
            }),
            PcuDispatchOp::Control(PcuDispatchControlOp::Return),
        ];
        let kernel = PcuDispatchKernelIr {
            id: PcuKernelId(0x672),
            entry: PcuDispatchEntryPoint {
                name: "checked_graph_add",
                logical_shape: [2, 1, 1],
            },
            bindings: &declarations,
            ports: &[],
            parameters: &[],
            ops: &ops,
            type_caps: PcuValueTypeCaps::FLOAT32,
            feature_caps: PcuDispatchFeatureCaps::READ_ONLY_RESOURCES
                .union(PcuDispatchFeatureCaps::MUTABLE_RESOURCES)
                .union(if policy == PcuRangePolicy::Clamp {
                    PcuDispatchFeatureCaps::RANGE_CLAMP
                } else {
                    PcuDispatchFeatureCaps::empty()
                }),
        };
        backend
            .prepare_dispatch(PcuDispatchSubmission {
                kernel: &kernel,
                shape: PcuInvocationShape::invocations(std::num::NonZeroU32::new(2).unwrap()),
            })
            .unwrap()
    }

    #[test]
    #[ignore = "requires CUDA GPU and device access"]
    fn checked_single_kernel_graph_refreshes_inputs_reports_faults_and_recovers() {
        let (_discovery, backend) = selected_device();
        for policy in [PcuRangePolicy::Reject, PcuRangePolicy::Clamp] {
            let prepared = prepare_checked_add(&backend, policy);
            let buffers = std::array::from_fn::<_, 3, _>(|_| backend.allocate(8).unwrap());
            let bindings = std::array::from_fn::<_, 3, _>(|slot| {
                backend
                    .binding(
                        PcuBindingRef::new(0, u32::try_from(slot).unwrap()),
                        if slot == 2 {
                            PcuBindingAccess::WriteOnly
                        } else {
                            PcuBindingAccess::ReadOnly
                        },
                        PcuBindingType::Value(PcuValueType::f32()),
                        buffers[slot].clone(),
                    )
                    .unwrap()
            });
            let node = CudaGraphDispatch {
                dispatch: &prepared,
                bindings: &bindings,
            };
            assert!(matches!(
                CudaNativeGraph::capture(&[
                    CudaGraphDispatch {
                        dispatch: &prepared,
                        bindings: &bindings
                    },
                    CudaGraphDispatch {
                        dispatch: &prepared,
                        bindings: &bindings
                    },
                ]),
                Err(CudaNativeGraphError::CheckedArithmeticUnsupported)
            ));
            let mut graph = CudaNativeGraph::capture(&[node]).unwrap();
            assert!(matches!(
                graph.replay_and_wait(),
                Err(CudaNativeGraphError::CheckedOutcomeRequired)
            ));
            let bytes = |a: f32, b: f32| [a.to_le_bytes(), b.to_le_bytes()].concat();
            graph
                .refresh_input(&buffers[0], 0, &bytes(2.0, 3.0))
                .unwrap();
            graph
                .refresh_input(&buffers[1], 0, &bytes(4.0, 5.0))
                .unwrap();
            assert_eq!(
                graph.replay_checked_and_wait().unwrap(),
                PcuCompletionOutcome::Succeeded
            );
            let mut actual = [0; 8];
            graph.readback(&buffers[2], 0, &mut actual).unwrap();
            assert_eq!(actual.as_slice(), bytes(6.0, 8.0));
            graph
                .refresh_input(&buffers[0], 0, &bytes(2.0, f32::MAX))
                .unwrap();
            graph
                .refresh_input(&buffers[1], 0, &bytes(4.0, f32::MAX))
                .unwrap();
            assert_eq!(
                graph.replay_checked_and_wait().unwrap(),
                PcuCompletionOutcome::Fault(PcuExecutionFault {
                    kind: PcuExecutionFaultKind::ArithmeticOverflow,
                    invocation_id: 1,
                    recovered: policy == PcuRangePolicy::Clamp,
                })
            );
            if policy == PcuRangePolicy::Clamp {
                graph.readback(&buffers[2], 0, &mut actual).unwrap();
                assert_eq!(actual.as_slice(), bytes(6.0, f32::MAX));
            }
            graph
                .refresh_input(&buffers[0], 0, &bytes(7.0, 8.0))
                .unwrap();
            graph
                .refresh_input(&buffers[1], 0, &bytes(1.0, 2.0))
                .unwrap();
            assert_eq!(
                graph.replay_checked_and_wait().unwrap(),
                PcuCompletionOutcome::Succeeded
            );
            graph.readback(&buffers[2], 0, &mut actual).unwrap();
            assert_eq!(actual.as_slice(), bytes(8.0, 10.0));
            assert!(matches!(
                buffers[2].copy_to(&mut actual),
                Err(CudaError::Busy)
            ));
            let foreign = backend.allocate(8).unwrap();
            assert!(matches!(
                graph.readback(&foreign, 0, &mut actual),
                Err(CudaNativeGraphError::Cuda(CudaError::Busy))
            ));
            drop(graph);
            buffers[2].copy_to(&mut actual).unwrap();
            assert_eq!(actual.as_slice(), bytes(8.0, 10.0));
        }
    }
}
