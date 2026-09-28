//! Typed owned tensor source execution. Graph preparation is cold; escaped results own storage.

use crate::PcuScalar;
#[rustfmt::skip]
use super::{
    PcuArgumentError,
    PcuExecutionError,
    PcuHostCallSite,
    PcuTensor,
};
#[rustfmt::skip]
use core::{
    any::TypeId,
    marker::PhantomData,
};
#[cfg(all(feature = "rocm", feature = "tensor"))]
use core::sync::atomic::{AtomicUsize, Ordering};
#[cfg(feature = "tensor")]
use crate::dialect::tensor::{Graph, ValueId};
#[cfg(feature = "tensor")]
use alloc::vec::Vec;

#[cfg(all(feature = "rocm", feature = "tensor"))]
static NEXT_CAPTURE_ID: AtomicUsize = AtomicUsize::new(1);

/// Opaque value token tied to one cold source graph capture.
#[doc(hidden)]
#[derive(Clone, Copy)]
pub struct PcuTensorGraphValue {
    #[cfg(feature = "tensor")]
    value: ValueId,
    #[cfg(feature = "tensor")]
    capture_id: usize,
    marker: PhantomData<fn() -> ()>,
}

/// Cold graph builder passed to generated owned-source companions.
#[doc(hidden)]
pub struct PcuTensorGraphCapture {
    #[cfg(feature = "tensor")]
    graph: Graph,
    #[cfg(feature = "tensor")]
    capture_id: usize,
    #[cfg(feature = "tensor")]
    active_markers: Vec<TypeId>,
    marker: PhantomData<fn() -> ()>,
}

#[cfg_attr(not(feature = "tensor"), allow(clippy::missing_const_for_fn))]
// Executable configurations mutate the graph and recursion stack; disabled stubs are constant.
impl PcuTensorGraphCapture {
    #[cfg(all(feature = "rocm", feature = "tensor"))]
    fn new<const N: usize>(
        lengths: [usize; N],
    ) -> Result<(Self, [PcuTensorGraphValue; N]), PcuExecutionError> {
        let capture_id = NEXT_CAPTURE_ID
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                current.checked_add(1)
            })
            .map_err(|_| PcuExecutionError::InvalidTensorSourcePlan)?;
        let mut graph = Graph::default();
        let mut values = Vec::with_capacity(N);
        for length in lengths {
            let value = graph.input([length]).map_err(super::build_error)?;
            values.push(PcuTensorGraphValue {
                value,
                capture_id,
                marker: PhantomData,
            });
        }
        let values = values
            .try_into()
            .map_err(|_| PcuExecutionError::InvalidTensorSourcePlan)?;
        Ok((
            Self {
                graph,
                capture_id,
                active_markers: Vec::new(),
                marker: PhantomData,
            },
            values,
        ))
    }

    /// Enters a generated source/helper companion, rejecting recursion and excessive nesting.
    #[doc(hidden)]
    pub fn enter(&mut self, marker: TypeId) -> Result<(), PcuExecutionError> {
        #[cfg(feature = "tensor")]
        {
            if self.active_markers.contains(&marker) {
                return Err(PcuExecutionError::RecursiveTensorSource);
            }
            if self.active_markers.len() >= 64 {
                return Err(PcuExecutionError::TensorSourceNestingLimit);
            }
            self.active_markers.push(marker);
            Ok(())
        }
        #[cfg(not(feature = "tensor"))]
        {
            let _ = marker;
            Err(PcuExecutionError::TensorExecutionUnavailable)
        }
    }

    /// Leaves the most recently entered generated companion.
    #[doc(hidden)]
    pub fn leave(&mut self) {
        #[cfg(feature = "tensor")]
        {
            let _ = self.active_markers.pop();
        }
    }

    /// Captures an identity edge without adding a graph operation.
    #[doc(hidden)]
    pub fn identity(
        &mut self,
        value: PcuTensorGraphValue,
    ) -> Result<PcuTensorGraphValue, PcuExecutionError> {
        #[cfg(feature = "tensor")]
        {
            self.validate_value(value)?;
            Ok(value)
        }
        #[cfg(not(feature = "tensor"))]
        {
            let _ = value;
            Err(PcuExecutionError::TensorExecutionUnavailable)
        }
    }

    /// Captures one elementwise ReLU node from an existing value in this graph.
    #[doc(hidden)]
    pub fn relu(
        &mut self,
        value: PcuTensorGraphValue,
    ) -> Result<PcuTensorGraphValue, PcuExecutionError> {
        #[cfg(feature = "tensor")]
        {
            self.validate_value(value)?;
            let value = self.graph.relu(value.value).map_err(super::build_error)?;
            Ok(PcuTensorGraphValue {
                value,
                capture_id: self.capture_id,
                marker: PhantomData,
            })
        }
        #[cfg(not(feature = "tensor"))]
        {
            let _ = value;
            Err(PcuExecutionError::TensorExecutionUnavailable)
        }
    }

    /// Captures one elementwise addition after validating both operands belong to this graph.
    #[doc(hidden)]
    pub fn add(
        &mut self,
        lhs: PcuTensorGraphValue,
        rhs: PcuTensorGraphValue,
    ) -> Result<PcuTensorGraphValue, PcuExecutionError> {
        #[cfg(feature = "tensor")]
        {
            self.validate_value(lhs)?;
            self.validate_value(rhs)?;
            let value = self
                .graph
                .add(lhs.value, rhs.value)
                .map_err(super::build_error)?;
            Ok(PcuTensorGraphValue {
                value,
                capture_id: self.capture_id,
                marker: PhantomData,
            })
        }
        #[cfg(not(feature = "tensor"))]
        {
            let _ = (lhs, rhs);
            Err(PcuExecutionError::TensorExecutionUnavailable)
        }
    }

    /// Captures one elementwise subtraction after validating both operands belong to this graph.
    #[doc(hidden)]
    pub fn sub(
        &mut self,
        lhs: PcuTensorGraphValue,
        rhs: PcuTensorGraphValue,
    ) -> Result<PcuTensorGraphValue, PcuExecutionError> {
        #[cfg(feature = "tensor")]
        {
            self.validate_value(lhs)?;
            self.validate_value(rhs)?;
            let value = self
                .graph
                .sub(lhs.value, rhs.value)
                .map_err(super::build_error)?;
            Ok(PcuTensorGraphValue {
                value,
                capture_id: self.capture_id,
                marker: PhantomData,
            })
        }
        #[cfg(not(feature = "tensor"))]
        {
            let _ = (lhs, rhs);
            Err(PcuExecutionError::TensorExecutionUnavailable)
        }
    }

    /// Captures one elementwise multiplication after validating both operands belong to this graph.
    #[doc(hidden)]
    pub fn mul(
        &mut self,
        lhs: PcuTensorGraphValue,
        rhs: PcuTensorGraphValue,
    ) -> Result<PcuTensorGraphValue, PcuExecutionError> {
        #[cfg(feature = "tensor")]
        {
            self.validate_value(lhs)?;
            self.validate_value(rhs)?;
            let value = self
                .graph
                .mul(lhs.value, rhs.value)
                .map_err(super::build_error)?;
            Ok(PcuTensorGraphValue {
                value,
                capture_id: self.capture_id,
                marker: PhantomData,
            })
        }
        #[cfg(not(feature = "tensor"))]
        {
            let _ = (lhs, rhs);
            Err(PcuExecutionError::TensorExecutionUnavailable)
        }
    }

    #[cfg(feature = "tensor")]
    fn validate_value(&self, value: PcuTensorGraphValue) -> Result<(), PcuExecutionError> {
        if value.capture_id != self.capture_id {
            return Err(PcuExecutionError::InvalidTensorSourcePlan);
        }
        self.graph.shape(value.value).map_err(super::build_error)?;
        Ok(())
    }

    #[cfg(all(feature = "rocm", feature = "tensor"))]
    fn finish(self, value: PcuTensorGraphValue) -> Result<(Graph, ValueId), PcuExecutionError> {
        self.validate_value(value)?;
        if !self.active_markers.is_empty() {
            return Err(PcuExecutionError::InvalidTensorSourcePlan);
        }
        Ok((self.graph, value.value))
    }
}

#[doc(hidden)]
pub struct PcuTensorInput<'a, T: PcuScalar> {
    #[cfg_attr(not(all(feature = "rocm", feature = "tensor")), allow(dead_code))]
    kind: TensorInputKind<'a, T>,
}

#[cfg_attr(not(all(feature = "rocm", feature = "tensor")), allow(dead_code))]
enum TensorInputKind<'a, T: PcuScalar> {
    Host(&'a [T]),
    Resident(&'a PcuTensor<T>),
}

impl<'a, T: PcuScalar> PcuTensorInput<'a, T> {
    const fn host(values: &'a [T]) -> Self {
        Self {
            kind: TensorInputKind::Host(values),
        }
    }
    #[cfg(feature = "rocm")]
    const fn resident(owner: &'a PcuTensor<T>) -> Self {
        Self {
            kind: TensorInputKind::Resident(owner),
        }
    }
}

mod sealed {
    pub trait TensorSource<T> {}
}

/// Storage borrowed by generated owned-result entries; references reborrow as ordinary Rust does.
#[doc(hidden)]
pub trait PcuTensorSource<T: PcuScalar>: sealed::TensorSource<T> {
    /// # Errors
    /// Rejects resident values whose shape, initialization or completion does not permit reads.
    fn as_tensor_source(&self) -> Result<PcuTensorInput<'_, T>, PcuArgumentError>;
}

impl<T: PcuScalar> sealed::TensorSource<T> for [T] {}
impl<T: PcuScalar> PcuTensorSource<T> for [T] {
    fn as_tensor_source(&self) -> Result<PcuTensorInput<'_, T>, PcuArgumentError> {
        Ok(PcuTensorInput::host(self))
    }
}
impl<T: PcuScalar, const N: usize> sealed::TensorSource<T> for [T; N] {}
impl<T: PcuScalar, const N: usize> PcuTensorSource<T> for [T; N] {
    fn as_tensor_source(&self) -> Result<PcuTensorInput<'_, T>, PcuArgumentError> {
        Ok(PcuTensorInput::host(self))
    }
}
impl<T: PcuScalar> sealed::TensorSource<T> for alloc::vec::Vec<T> {}
impl<T: PcuScalar> PcuTensorSource<T> for alloc::vec::Vec<T> {
    fn as_tensor_source(&self) -> Result<PcuTensorInput<'_, T>, PcuArgumentError> {
        Ok(PcuTensorInput::host(self.as_slice()))
    }
}

// Closed wrapper implementations preserve native borrows without a blanket Deref bound.
macro_rules! host_container_source {
    ($($container:ident)::+) => {
        impl<T: PcuScalar> sealed::TensorSource<T> for $($container)::+<[T]> {}
        impl<T: PcuScalar> PcuTensorSource<T> for $($container)::+<[T]> {
            fn as_tensor_source(&self) -> Result<PcuTensorInput<'_, T>, PcuArgumentError> {
                Ok(PcuTensorInput::host(self))
            }
        }
        impl<T: PcuScalar, const N: usize> sealed::TensorSource<T> for $($container)::+<[T; N]> {}
        impl<T: PcuScalar, const N: usize> PcuTensorSource<T> for $($container)::+<[T; N]> {
            fn as_tensor_source(&self) -> Result<PcuTensorInput<'_, T>, PcuArgumentError> {
                Ok(PcuTensorInput::host(self.as_ref().as_slice()))
            }
        }
    };
}
host_container_source!(alloc::boxed::Box);
host_container_source!(alloc::rc::Rc);
host_container_source!(alloc::sync::Arc);

#[cfg(feature = "rocm")]
impl<T: PcuScalar> sealed::TensorSource<T> for PcuTensor<T> {}
#[cfg(feature = "rocm")]
impl<T: PcuScalar> PcuTensorSource<T> for PcuTensor<T> {
    fn as_tensor_source(&self) -> Result<PcuTensorInput<'_, T>, PcuArgumentError> {
        let length = self.device_tensor().buffer().len();
        self.validate_read(super::PcuSourceShape::Slice { length })?;
        Ok(PcuTensorInput::resident(self))
    }
}

impl<T: PcuScalar> core::fmt::Debug for PcuTensor<T> {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("PcuTensor")
            .field("scalar", &T::TYPE)
            .field("shape", &self.shape())
            .finish_non_exhaustive()
    }
}

// Provider-disabled builds retain the same instance API, although no owner can be minted.
#[cfg_attr(not(feature = "rocm"), allow(clippy::unused_self))]
impl<T: PcuScalar> PcuTensor<T> {
    /// Dense logical dimensions; no device transfer occurs.
    #[must_use]
    #[allow(clippy::missing_const_for_fn)] // Provider configurations borrow dynamic shape metadata.
    pub fn shape(&self) -> &[usize] {
        #[cfg(feature = "rocm")]
        {
            self.device_tensor().shape()
        }
        #[cfg(not(feature = "rocm"))]
        {
            &[]
        } // No safe constructor exists when no execution provider is compiled.
    }

    /// Logical scalar element count; no device transfer occurs.
    #[must_use]
    #[allow(clippy::missing_const_for_fn)] // Provider metadata comes through its typed storage view.
    pub fn len(&self) -> usize {
        #[cfg(feature = "rocm")]
        {
            self.device_tensor().buffer().len()
        }
        #[cfg(not(feature = "rocm"))]
        {
            0
        } // The disabled-provider owner is not constructible by consumers.
    }

    /// Whether the logical tensor contains no scalar elements.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Copies this initialized logical tensor into caller-owned RAM without consuming its owner.
    ///
    /// The destination must contain exactly the tensor's flattened element count. Device storage
    /// stays owned by this value until ordinary Drop or consumption; readback does not free it.
    ///
    /// # Errors
    /// Rejects unavailable backends, invalid initialization, uncertain completion, length mismatch,
    /// or backend transfer failure. No implicit CPU computation or backend migration occurs.
    #[allow(clippy::missing_const_for_fn)] // Executable configurations perform validated device IO.
    pub fn read_into(&self, destination: &mut [T]) -> Result<(), PcuExecutionError> {
        #[cfg(feature = "rocm")]
        {
            self.validate_initialized().map_err(super::argument_error)?;
            let pool = crate::PcuMemoryResource::pool(self.device_tensor().buffer().resource());
            self.session()
                .backend()
                .download_buffer(pool, self.device_tensor().buffer(), destination)
                .map_err(PcuExecutionError::DeviceExecution)
        }
        #[cfg(not(feature = "rocm"))]
        {
            let _ = destination;
            Err(PcuExecutionError::NoBackendEnabled)
        }
    }
}

/// Generated typed owned-result entry; its graph factory runs only on a cold cache miss.
///
/// # Errors
/// Returns unsupported feature, selection, preparation, allocation or device execution errors.
#[doc(hidden)]
#[allow(clippy::needless_pass_by_value)] // Generated calls transfer the validated source carrier into execution.
#[cfg_attr(
    not(all(feature = "rocm", feature = "tensor")),
    allow(clippy::missing_const_for_fn)
)] // Enabled providers execute runtime work through the same API.
pub fn call_owned_tensor_capture<const N: usize, F>(
    site: &PcuHostCallSite,
    specialization: TypeId,
    inputs: [PcuTensorInput<'_, f32>; N],
    build: F,
) -> Result<PcuTensor<f32>, PcuExecutionError>
where
    F: FnOnce(
            &mut PcuTensorGraphCapture,
            [PcuTensorGraphValue; N],
        ) -> Result<PcuTensorGraphValue, PcuExecutionError>
        + 'static,
{
    #[cfg(all(feature = "rocm", feature = "tensor"))]
    {
        execution::call(site, specialization, &inputs, build)
    }
    #[cfg(not(all(feature = "rocm", feature = "tensor")))]
    {
        let _ = (site, specialization, inputs, build);
        Err(PcuExecutionError::TensorExecutionUnavailable)
    }
}

#[cfg(all(feature = "rocm", feature = "tensor"))]
mod execution {
    #[rustfmt::skip]
    use super::{
        TypeId,
        PcuArgumentError,
        PcuHostCallSite,
        PcuTensorInput,
        PcuTensor,
        PcuExecutionError,
        TensorInputKind,
        PcuTensorGraphCapture,
        PcuTensorGraphValue,
    };
    #[rustfmt::skip]
    use crate::{
        PcuSourceShape,
        PcuDeviceTensor,
        PcuHostArgument,
        PcuBindingRef,
        PcuMemoryPoolId,
        PcuMemoryProvider,
        PcuOwnedDispatchBackend,
    };
    #[rustfmt::skip]
    use crate::dialect::tensor::{
        TensorArithmeticCapability,
        TensorArithmeticRewritePolicy,
        TensorPointwiseGroupingPolicy,
        ValueId,
    };
    #[rustfmt::skip]
    use fusion_pcu_rocm::{
        RocmMemoryProvider,
        RocmMemoryResource,
        RocmOwnedPreparedTensorGraph,
    };
    #[rustfmt::skip]
    use std::{
        cell::RefCell,
        rc::Rc,
        sync::Arc,
        sync::atomic::Ordering,
    };
    use crate::global::session::RocmSession;

    struct Entry {
        specialization: TypeId,
        factory: TypeId,
        lengths: Vec<usize>,
        resident_affinity: bool,
        session: Rc<RocmSession>,
        prepared: RocmOwnedPreparedTensorGraph,
        input_ids: Vec<ValueId>,
        pool: PcuMemoryPoolId,
        memory: RocmMemoryProvider,
        host_inputs: Vec<Option<PcuDeviceTensor<f32, RocmMemoryResource>>>,
    }
    #[derive(Default)]
    struct State {
        generation: u64,
        entries: Vec<Entry>,
    }
    std::thread_local! { static STATE: RefCell<State> = RefCell::new(State::default()); }

    fn prepare_entry<const N: usize, F>(
        affinity: Option<&Rc<RocmSession>>,
        specialization: TypeId,
        factory: TypeId,
        lengths: [usize; N],
        build: F,
    ) -> Result<(Entry, u64, usize), PcuExecutionError>
    where
        F: FnOnce(
                &mut PcuTensorGraphCapture,
                [PcuTensorGraphValue; N],
            ) -> Result<PcuTensorGraphValue, PcuExecutionError>
            + 'static,
    {
        let (mut capture, input_values) = PcuTensorGraphCapture::new(lengths)?;
        let input_ids = input_values.map(|value| value.value).to_vec();
        let output_value = build(&mut capture, input_values)?;
        let (graph, output_id) = capture.finish(output_value)?;
        let program = Arc::new(
            graph
                .into_selected_program(
                    &[output_id],
                    TensorArithmeticRewritePolicy::Disabled,
                    TensorArithmeticCapability::Strict,
                    TensorPointwiseGroupingPolicy::Disabled,
                )
                .map_err(crate::global::build_error)?,
        );
        let (session, prepared, generation, capacity) =
            crate::global::hosted::prepare_tensor(affinity, |session| {
                let prepared = session
                    .tensor_assessor()
                    .map_err(PcuExecutionError::TensorInitialization)?
                    .prepare_shared_owned_program(Arc::clone(&program))
                    .map_err(PcuExecutionError::TensorExecution)?;
                Ok(prepared)
            })?;
        let pool = PcuMemoryPoolId(session.backend().device_identity().device_id());
        let memory = session.backend().memory_provider(pool);
        let entry = Entry {
            specialization,
            factory,
            lengths: lengths.to_vec(),
            resident_affinity: affinity.is_some(),
            session,
            prepared,
            input_ids,
            pool,
            memory,
            host_inputs: (0..N).map(|_| None).collect(),
        };
        Ok((entry, generation, capacity))
    }

    fn stage_host_input(
        entry: &mut Entry,
        index: usize,
        values: &[f32],
    ) -> Result<(), PcuExecutionError> {
        if let Some(staging) = entry.host_inputs[index].as_mut() {
            let argument = PcuHostArgument::read(PcuBindingRef::new(0, 0), values);
            if let Err(error) =
                entry
                    .memory
                    .transfer_to(staging.resource_mut(), 0, argument.bytes())
            {
                // A quarantined allocation is retained by its physical access lease. Do not reuse
                // that failed staging handle on the next call or pin it through this cache entry.
                entry.host_inputs[index] = None;
                return Err(PcuExecutionError::Memory(error));
            }
        } else {
            let buffer = entry
                .session
                .backend()
                .upload_buffer(entry.pool, values)
                .map_err(PcuExecutionError::DeviceExecution)?;
            entry.host_inputs[index] = Some(
                PcuDeviceTensor::new([values.len()], buffer)
                    .map_err(PcuExecutionError::TensorStorage)?,
            );
        }
        Ok(())
    }

    fn preflight_inputs<'a, const N: usize>(
        inputs: &'a [PcuTensorInput<'_, f32>; N],
    ) -> Result<([usize; N], Option<&'a Rc<RocmSession>>), PcuExecutionError> {
        if N == 0 {
            return Err(PcuExecutionError::EmptyTensorInput);
        }
        let lengths = core::array::from_fn(|index| match &inputs[index].kind {
            TensorInputKind::Host(values) => values.len(),
            TensorInputKind::Resident(owner) => owner.device_tensor().buffer().len(),
        });
        if lengths.contains(&0) {
            return Err(PcuExecutionError::EmptyTensorInput);
        }

        let mut affinity = None;
        for input in inputs {
            if let TensorInputKind::Resident(owner) = &input.kind {
                let owner_session = owner.session();
                if affinity.is_some_and(|existing| !Rc::ptr_eq(existing, owner_session)) {
                    return Err(PcuExecutionError::Argument(
                        PcuArgumentError::SessionMismatch,
                    ));
                }
                let length = owner.device_tensor().buffer().len();
                owner
                    .validate_read(PcuSourceShape::Slice { length })
                    .map_err(PcuExecutionError::Argument)?;
                owner
                    .device_tensor()
                    .buffer()
                    .resource()
                    .validate_access_available()
                    .map_err(|_| {
                        PcuExecutionError::Argument(PcuArgumentError::ResidentCompletionUncertain)
                    })?;
                affinity = Some(owner_session);
            }
        }
        Ok((lengths, affinity))
    }

    pub(super) fn call<const N: usize, F>(
        site: &PcuHostCallSite,
        specialization: TypeId,
        inputs: &[PcuTensorInput<'_, f32>; N],
        build: F,
    ) -> Result<PcuTensor<f32>, PcuExecutionError>
    where
        F: FnOnce(
                &mut PcuTensorGraphCapture,
                [PcuTensorGraphValue; N],
            ) -> Result<PcuTensorGraphValue, PcuExecutionError>
            + 'static,
    {
        let factory = TypeId::of::<F>();
        let (lengths, affinity) = preflight_inputs(inputs)?;
        STATE
            .try_with(|state| {
                let mut state = state
                    .try_borrow_mut()
                    .map_err(|_| PcuExecutionError::ReentrantCall)?;
                let generation = crate::global::hosted::current_generation();
                if state.generation != generation {
                    state.entries.clear();
                    state.generation = generation;
                }
                let matches = |entry: &Entry| {
                    entry.specialization == specialization
                        && entry.factory == factory
                        && entry.lengths.as_slice() == &lengths[..]
                        && affinity.map_or(!entry.resident_affinity, |root| {
                            entry.resident_affinity && Rc::ptr_eq(root, &entry.session)
                        })
                };
                let hint = site.slot.load(Ordering::Relaxed);
                let cached = if state.entries.get(hint).is_some_and(matches) {
                    Some(hint)
                } else {
                    state.entries.iter().position(matches)
                };
                let slot = if let Some(slot) = cached {
                    slot
                } else {
                    let (entry, generation, capacity) =
                        prepare_entry(affinity, specialization, factory, lengths, build)?;
                    if state.generation != generation {
                        state.entries.clear();
                        state.generation = generation;
                    }
                    if state.entries.len() == capacity {
                        let victim = hint.min(state.entries.len() - 1);
                        state.entries[victim] = entry;
                        victim
                    } else {
                        state.entries.push(entry);
                        state.entries.len() - 1
                    }
                };
                site.slot.store(slot, Ordering::Relaxed);
                let entry = &mut state.entries[slot];
                // Preflight above examines every resident affinity before this loop uploads any
                // host input. Then stage all host inputs before constructing the stack bindings.
                for (index, input) in inputs.iter().enumerate() {
                    if let TensorInputKind::Host(values) = &input.kind {
                        stage_host_input(entry, index, values)?;
                    }
                }
                let sources: [&PcuDeviceTensor<f32, RocmMemoryResource>; N] =
                    core::array::from_fn(|index| match &inputs[index].kind {
                        TensorInputKind::Host(_) => entry.host_inputs[index]
                            .as_ref()
                            .expect("host input staged before bindings"),
                        TensorInputKind::Resident(owner) => owner.device_tensor(),
                    });
                let bindings: [(ValueId, &PcuDeviceTensor<f32, RocmMemoryResource>); N] =
                    core::array::from_fn(|index| (entry.input_ids[index], sources[index]));
                let outputs = entry
                    .session
                    .tensor_assessor()
                    .map_err(PcuExecutionError::TensorInitialization)?
                    .execute_owned_program_outputs(
                        &entry.prepared,
                        &bindings,
                        entry.pool,
                        &mut entry.memory,
                    )
                    .map_err(PcuExecutionError::TensorExecution)?;
                let mut outputs = outputs.into_iter();
                let (_, tensor) = outputs
                    .next()
                    .ok_or(PcuExecutionError::PreparationDidNotProduceKernel)?;
                let extra_output = outputs.next();
                debug_assert!(extra_output.is_none(), "source capture has one output");
                Ok(PcuTensor::from_successful_output(
                    tensor,
                    Rc::clone(&entry.session),
                ))
            })
            .map_err(|_| PcuExecutionError::ThreadUnavailable)?
    }

    pub(super) fn clear_cache() -> Result<(), PcuExecutionError> {
        STATE
            .try_with(|state| {
                state
                    .try_borrow_mut()
                    .map_err(|_| PcuExecutionError::ReentrantCall)?
                    .entries
                    .clear();
                Ok(())
            })
            .map_err(|_| PcuExecutionError::ThreadUnavailable)?
    }
}

#[cfg(all(test, feature = "rocm", feature = "tensor"))]
mod capture_tests {
    use super::{PcuExecutionError, PcuTensorGraphCapture};
    use core::any::TypeId;

    #[test]
    fn graph_values_cannot_cross_capture_boundaries() {
        let (mut first, [value]) = PcuTensorGraphCapture::new([4]).unwrap();
        let (second, _) = PcuTensorGraphCapture::new([4]).unwrap();
        let foreign = first.identity(value).unwrap();
        assert!(second.validate_value(foreign).is_err());
        first.finish(value).unwrap();
    }

    #[test]
    fn recursion_and_nesting_guards_recover_after_leave() {
        let (mut capture, [value]) = PcuTensorGraphCapture::new([1]).unwrap();
        let marker = TypeId::of::<u8>();
        capture.enter(marker).unwrap();
        assert!(matches!(
            capture.enter(marker),
            Err(PcuExecutionError::RecursiveTensorSource)
        ));
        capture.leave();
        capture.enter(marker).unwrap();
        capture.leave();
        capture.active_markers.resize(64, marker);
        assert!(matches!(
            capture.enter(TypeId::of::<u16>()),
            Err(PcuExecutionError::TensorSourceNestingLimit)
        ));
        capture.active_markers.clear();
        capture.finish(value).unwrap();
    }

    #[test]
    fn graph_finish_rejects_unbalanced_companion_scope() {
        let (mut capture, [value]) = PcuTensorGraphCapture::new([2]).unwrap();
        capture.enter(TypeId::of::<u8>()).unwrap();
        assert!(matches!(
            capture.finish(value),
            Err(PcuExecutionError::InvalidTensorSourcePlan)
        ));
    }

    #[test]
    fn addition_checks_capture_provenance_and_shape() {
        let (mut capture, [lhs, rhs]) = PcuTensorGraphCapture::new([4, 4]).unwrap();
        let output = capture.add(lhs, rhs).unwrap();
        let (graph, output_id) = capture.finish(output).unwrap();
        assert_eq!(graph.shape(output_id).unwrap(), [4]);

        let (mut mismatched, [lhs, rhs]) = PcuTensorGraphCapture::new([4, 5]).unwrap();
        assert!(mismatched.add(lhs, rhs).is_err());
        let (_other, [foreign]) = PcuTensorGraphCapture::new([4]).unwrap();
        assert!(mismatched.add(lhs, foreign).is_err());
    }

    #[test]
    fn subtraction_and_multiplication_preserve_tensor_shape() {
        let (mut capture, [lhs, rhs]) = PcuTensorGraphCapture::new([3, 3]).unwrap();
        let difference = capture.sub(lhs, rhs).unwrap();
        let product = capture.mul(difference, rhs).unwrap();
        let (graph, output) = capture.finish(product).unwrap();
        assert_eq!(graph.shape(output).unwrap(), [3]);
    }
}

#[cfg(all(feature = "rocm", feature = "tensor"))]
pub(super) fn clear_cache() -> Result<(), PcuExecutionError> {
    execution::clear_cache()
}
