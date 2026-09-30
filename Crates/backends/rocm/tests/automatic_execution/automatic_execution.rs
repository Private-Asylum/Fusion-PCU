//! Hardware integration regressions for automatic prepared tensor execution.
//!
//! Run explicitly on a `ROCm` host with:
//! `cargo test -p fusion-pcu-rocm --features tensor --test automatic_execution -- --ignored --nocapture`

extern crate pcu_facade as fusion_pcu;

#[rustfmt::skip]
use std::{
    cell::Cell,
    num::NonZeroUsize,
    rc::Rc,
    sync::Arc,
};

#[rustfmt::skip]
use fusion_pcu::{
    PcuDeviceBufferAllocator,
    PcuDeviceTensor,
    PcuMemoryAccess,
    PcuMemoryAllocationRequest,
    PcuMemoryDisposition,
    PcuMemoryHostAccess,
    PcuMemoryPoolId,
    PcuMemoryPoolSnapshot,
    PcuMemoryProvider,
    PcuMemoryProviderError,
    PcuMemoryProviderFailure,
    PcuMemoryProviderOperation,
    PcuMemoryRange,
    PcuMemoryResource,
    PcuOwnedDispatchMemorySession,
};
use fusion_pcu_macros::pcu;
#[rustfmt::skip]
use fusion_pcu_rocm::{
    RocmDiscovery,
    RocmMemoryResource,
    RocmOwnedDispatchBackend,
    RocmTensorAssessor,
    RocmOwnedTensorAssessor,
    RocmTensorExecutionError,
};
#[rustfmt::skip]
use fusion_pcu::dialect::tensor::{
    Graph,
    TensorArithmeticCapability,
    TensorArithmeticRewritePolicy,
    TensorPointwiseGroupingPolicy,
    Tensor,
    TensorStorageValidationError,
    TensorExecution,
    ValueId,
};

#[path = "../../examples/support/selection/selection.rs"]
mod selection;

#[derive(Clone, Copy)]
enum Mutation {
    TransferFailureOnly,
    TransferOk,
    TransferErr,
    TransferWrongSize,
    TransferErrSecond,
    CopyOk,
    CopyErr,
}

struct ReplacingMemory<P> {
    inner: P,
    mutation: Mutation,
    armed: Rc<Cell<bool>>,
    calls: Rc<Cell<usize>>,
    allocation_calls: Option<Rc<Cell<usize>>>,
}

impl<P> ReplacingMemory<P> {
    fn new(inner: P, mutation: Mutation) -> (Self, Rc<Cell<bool>>, Rc<Cell<usize>>) {
        let armed = Rc::new(Cell::new(false));
        let calls = Rc::new(Cell::new(0));
        (
            Self {
                inner,
                mutation,
                armed: Rc::clone(&armed),
                calls: Rc::clone(&calls),
                allocation_calls: None,
            },
            armed,
            calls,
        )
    }

    fn with_allocation_count(inner: P) -> (Self, Rc<Cell<usize>>) {
        let (mut memory, _, _) = Self::new(inner, Mutation::TransferFailureOnly);
        let allocation_calls = Rc::new(Cell::new(0));
        memory.allocation_calls = Some(Rc::clone(&allocation_calls));
        (memory, allocation_calls)
    }

    fn replacement_request(
        resource: &RocmMemoryResource,
        size_bytes: u64,
    ) -> PcuMemoryAllocationRequest {
        PcuMemoryAllocationRequest {
            pool: resource.pool(),
            size_bytes,
            alignment_bytes: resource.alignment_bytes(),
            access: resource.access(),
            host_access: PcuMemoryHostAccess::TransferOnly,
            require_device_local: false,
        }
    }

    const fn error(
        pool: PcuMemoryPoolId,
        operation: PcuMemoryProviderOperation,
    ) -> PcuMemoryProviderError {
        PcuMemoryProviderError {
            pool,
            operation,
            disposition: PcuMemoryDisposition::Reject,
            failure: PcuMemoryProviderFailure::BackendFailure,
        }
    }
}

impl<P: PcuMemoryProvider<Resource = RocmMemoryResource>> PcuMemoryProvider for ReplacingMemory<P> {
    type Resource = RocmMemoryResource;
    type ImportDescriptor = P::ImportDescriptor;
    type Mapping<'a>
        = P::Mapping<'a>
    where
        Self: 'a;

    fn snapshot(
        &self,
        pool: PcuMemoryPoolId,
    ) -> Result<PcuMemoryPoolSnapshot, PcuMemoryProviderError> {
        self.inner.snapshot(pool)
    }
    fn allocate(
        &mut self,
        request: PcuMemoryAllocationRequest,
    ) -> Result<Self::Resource, PcuMemoryProviderError> {
        if let Some(calls) = &self.allocation_calls {
            calls.set(calls.get() + 1);
        }
        self.inner.allocate(request)
    }
    fn import(
        &mut self,
        descriptor: Self::ImportDescriptor,
    ) -> Result<Self::Resource, PcuMemoryProviderError> {
        self.inner.import(descriptor)
    }
    fn map<'a>(
        &'a mut self,
        resource: &'a mut Self::Resource,
        range: PcuMemoryRange,
    ) -> Result<Self::Mapping<'a>, PcuMemoryProviderError> {
        self.inner.map(resource, range)
    }
    fn transfer_to(
        &mut self,
        resource: &mut Self::Resource,
        offset: u64,
        bytes: &[u8],
    ) -> Result<(), PcuMemoryProviderError> {
        if !self.armed.get() {
            return self.inner.transfer_to(resource, offset, bytes);
        }
        match self.mutation {
            Mutation::TransferFailureOnly => {
                self.armed.set(false);
                Err(Self::error(
                    resource.pool(),
                    PcuMemoryProviderOperation::TransferTo,
                ))
            }
            Mutation::TransferErrSecond => {
                let calls = self.calls.get() + 1;
                self.calls.set(calls);
                if calls == 2 {
                    self.armed.set(false);
                    Err(Self::error(
                        resource.pool(),
                        PcuMemoryProviderOperation::TransferTo,
                    ))
                } else {
                    self.inner.transfer_to(resource, offset, bytes)
                }
            }
            Mutation::TransferOk | Mutation::TransferErr | Mutation::TransferWrongSize => {
                self.armed.set(false);
                let pool = resource.pool();
                let requested_size = if matches!(self.mutation, Mutation::TransferWrongSize) {
                    resource
                        .size_bytes()
                        .checked_sub(1)
                        .ok_or_else(|| Self::error(pool, PcuMemoryProviderOperation::TransferTo))?
                } else {
                    resource.size_bytes()
                };
                let replacement = self
                    .inner
                    .allocate(Self::replacement_request(resource, requested_size))?;
                *resource = replacement;
                if matches!(self.mutation, Mutation::TransferWrongSize) {
                    // Simulate a faulty provider that reports success after replacing the
                    // allocation with storage too small for the tensor.
                    return Ok(());
                }
                if matches!(self.mutation, Mutation::TransferErr) {
                    Err(Self::error(pool, PcuMemoryProviderOperation::TransferTo))
                } else {
                    self.inner.transfer_to(resource, offset, bytes)
                }
            }
            Mutation::CopyOk | Mutation::CopyErr => self.inner.transfer_to(resource, offset, bytes),
        }
    }
    fn transfer_from(
        &mut self,
        resource: &Self::Resource,
        offset: u64,
        bytes: &mut [u8],
    ) -> Result<(), PcuMemoryProviderError> {
        self.inner.transfer_from(resource, offset, bytes)
    }
    fn copy_resource(
        &mut self,
        destination: &mut Self::Resource,
        source: &Self::Resource,
        size: u64,
    ) -> Result<(), PcuMemoryProviderError> {
        if !self.armed.get() || !matches!(self.mutation, Mutation::CopyOk | Mutation::CopyErr) {
            return self.inner.copy_resource(destination, source, size);
        }
        self.armed.set(false);
        let pool = destination.pool();
        let replacement = self.inner.allocate(Self::replacement_request(
            destination,
            destination.size_bytes(),
        ))?;
        *destination = replacement;
        if matches!(self.mutation, Mutation::CopyErr) {
            Err(Self::error(pool, PcuMemoryProviderOperation::CopyResource))
        } else {
            self.inner.copy_resource(destination, source, size)
        }
    }
}

fn open_device() -> (RocmDiscovery, RocmOwnedDispatchBackend, PcuMemoryPoolId) {
    let discovery = RocmDiscovery::new();
    let candidates = selection::ranked_devices(
        &discovery,
        selection::preferred_device().expect("valid FUSION_ROCM_DEVICE"),
        true,
    )
    .expect("ROCm discovery must succeed; this integration test requires ROCm hardware");
    let candidate = candidates
        .into_iter()
        .next()
        .expect("a capable ROCm device must be discovered");
    let session = RocmOwnedDispatchBackend::open(&discovery, candidate.device, 64)
        .expect("selected ROCm device must open");
    (discovery, session, candidate.pool)
}

fn expect_values(actual: &Tensor, expected: &[f32]) {
    assert_eq!(actual.shape(), &[1]);
    for (actual, expected) in actual.data().iter().zip(expected) {
        assert!(
            (actual - expected).abs() <= 1.0e-5,
            "{actual} != {expected}"
        );
    }
}

fn execute_and_read<E: TensorExecution>(
    execution: &mut E,
    output: ValueId,
) -> Result<Tensor, E::Error> {
    execution.execute()?;
    execution.read_output(output)
}

#[pcu(invocations = 65)]
fn resident_copy<const N: usize>(input: &[f32], output: &mut [f32]) {
    let mut id = pcu::context::global_invocation_id();
    let stride = pcu::context::invocation_count();
    while id < N {
        output[id] = input[id];
        id += stride;
    }
}

#[test]
#[ignore = "requires an explicitly selected working ROCm device"]
fn fresh_owned_outputs_remain_independent_across_prepared_calls() {
    let (_discovery, session, pool) = open_device();
    let mut call = resident_copy_prepare_device::<4, _>(&session).expect("prepare resident copy");
    let first_input = session
        .upload_buffer(pool, &[1.0_f32, 2.0, 3.0, 4.0])
        .expect("first input");
    let second_input = session
        .upload_buffer(pool, &[5.0_f32, 6.0, 7.0, 8.0])
        .expect("second input");
    let mut allocator = session.memory_provider(pool);
    let request = PcuMemoryAllocationRequest {
        pool,
        size_bytes: 16,
        alignment_bytes: 16,
        access: PcuMemoryAccess::ReadWrite,
        host_access: PcuMemoryHostAccess::TransferOnly,
        require_device_local: false,
    };
    // This bounded copy writes every output lane before the uninitialized allocation can escape.
    let mut first = allocator
        .allocate_device_buffer::<f32>(request, 4)
        .expect("first allocation");
    call(&first_input, &mut first).expect("first complete writer");
    let first = PcuDeviceTensor::new([4], first).expect("first owned shape");
    let mut second = allocator
        .allocate_device_buffer::<f32>(request, 4)
        .expect("second allocation");
    call(&second_input, &mut second).expect("second complete writer");
    let second = PcuDeviceTensor::new([4], second).expect("second owned shape");
    let mut host = [0.0_f32; 4];
    session
        .download_buffer(pool, first.buffer(), &mut host)
        .expect("first readback");
    assert_eq!(
        host.map(f32::to_bits),
        [1.0_f32, 2.0, 3.0, 4.0].map(f32::to_bits)
    );
    session
        .download_buffer(pool, second.buffer(), &mut host)
        .expect("second readback");
    assert_eq!(
        host.map(f32::to_bits),
        [5.0_f32, 6.0, 7.0, 8.0].map(f32::to_bits)
    );
    drop(second);
    session
        .download_buffer(pool, first.buffer(), &mut host)
        .expect("first remains live");
    assert_eq!(
        host.map(f32::to_bits),
        [1.0_f32, 2.0, 3.0, 4.0].map(f32::to_bits)
    );
}

#[test]
#[ignore = "requires an explicitly selected working ROCm device"]
fn bound_feedback_matches_cpu_for_one_through_four_steps() {
    let (_discovery, session, pool) = open_device();
    let assessor = RocmTensorAssessor::new(&session).expect("tensor assessor");
    let mut graph = Graph::default();
    let state = graph
        .input([1], fusion_pcu::PcuScalarType::F32)
        .expect("input");
    let increment = graph.constant_value(fusion_pcu::dialect::tensor::TensorValue::F32(
        Tensor::new([1], vec![2.0]).expect("constant"),
    ));
    let output = graph.add(state, increment).expect("add");
    let prepared = assessor
        .prepare_graph(&graph, output)
        .expect("prepare graph");
    let initial = Tensor::new([1], vec![1.0]).expect("initial tensor");
    let mut memory = PcuOwnedDispatchMemorySession::memory_provider(&session, pool);
    let input = assessor
        .upload_input(&initial, pool, &mut memory)
        .expect("upload");
    let mut execution = assessor
        .bind_feedback(
            &prepared,
            &[(output, state)],
            vec![(state, input)],
            pool,
            memory,
        )
        .expect("bind");
    assert!(matches!(
        execution.outputs(),
        Err(RocmTensorExecutionError::FeedbackStepUnavailable { .. })
    ));
    for steps in 1..=4 {
        execution
            .execute_steps(NonZeroUsize::new(steps).expect("positive steps"))
            .expect("execute");
        let expected = [3.0, 5.0, 7.0, 9.0][steps - 1];
        expect_values(
            &execution.read_output(output).expect("read output"),
            &[expected],
        );
    }
    execution
        .update_input(state, &initial)
        .expect("refresh input");
    assert!(execution.outputs().is_err());
    execution.execute().expect("single-step rerun");
    expect_values(&execution.read_output(output).expect("read output"), &[3.0]);
}

#[test]
#[ignore = "requires an explicitly selected working ROCm device"]
fn plain_bind_and_generic_trait_default_execute_work() {
    let (_discovery, session, pool) = open_device();
    let assessor = RocmTensorAssessor::new(&session).expect("tensor assessor");
    let mut graph = Graph::default();
    let input_value = graph
        .input([1], fusion_pcu::PcuScalarType::F32)
        .expect("input");
    let output = graph.relu(input_value).expect("relu");
    let prepared = assessor
        .prepare_graph(&graph, output)
        .expect("prepare graph");
    let initial = Tensor::new([1], vec![2.0]).expect("initial tensor");
    let mut memory = PcuOwnedDispatchMemorySession::memory_provider(&session, pool);
    let input = assessor
        .upload_input(&initial, pool, &mut memory)
        .expect("upload");
    let mut execution = assessor
        .bind(&prepared, vec![(input_value, input)], pool, memory)
        .expect("bind without feedback");
    expect_values(
        &execute_and_read(&mut execution, output).expect("generic execute and read"),
        &[2.0],
    );
}

#[test]
#[ignore = "requires an explicitly selected working ROCm device"]
fn tensor_output_moves_into_typed_device_call_without_intermediate_readback() {
    const N: usize = 65;
    let (_discovery, session, pool) = open_device();
    let assessor = RocmTensorAssessor::new(&session).expect("tensor assessor");
    let mut graph = Graph::default();
    let input_value = graph
        .input([N], fusion_pcu::PcuScalarType::F32)
        .expect("input");
    let output_value = graph.relu(input_value).expect("relu");
    let prepared = assessor
        .prepare_graph(&graph, output_value)
        .expect("prepare graph");
    let initial = Tensor::new(
        [N],
        (0..N)
            .map(|index| f32::from(u8::try_from(index).expect("small index fits")) - 32.0)
            .collect(),
    )
    .expect("initial tensor");
    let mut memory = PcuOwnedDispatchMemorySession::memory_provider(&session, pool);
    let input = assessor
        .upload_input(&initial, pool, &mut memory)
        .expect("upload");
    let mut execution = assessor
        .bind(&prepared, vec![(input_value, input)], pool, memory)
        .expect("bind");
    execution.execute().expect("execute tensor graph");

    let outputs = execution.into_outputs().expect("take resident outputs");
    let (value, tensor) = outputs.into_iter().next().expect("one selected output");
    assert_eq!(value, output_value);
    assert_eq!(tensor.shape(), &[N]);

    let mut downstream =
        resident_copy_prepare_device::<N, _>(&session).expect("prepare typed device kernel");
    let mut copied = session
        .upload_buffer(pool, &[0.0_f32; N])
        .expect("allocate downstream output");
    downstream(tensor.buffer(), &mut copied).expect("consume tensor output directly on device");

    let mut observed = [0.0_f32; N];
    session
        .download_buffer(pool, &copied, &mut observed)
        .expect("download final validation output");
    for (index, actual) in observed.iter().enumerate() {
        let expected = (f32::from(u8::try_from(index).expect("small index fits")) - 32.0).max(0.0);
        assert_eq!(actual.to_bits(), expected.to_bits());
    }
}

#[test]
#[ignore = "requires an explicitly selected working ROCm device"]
fn scratch_reuses_slots_after_fanout_lifetimes_end() {
    let (_discovery, session, pool) = open_device();
    let assessor = RocmTensorAssessor::new(&session).expect("tensor assessor");
    let mut graph = Graph::default();
    let left = graph
        .input([16], fusion_pcu::PcuScalarType::F32)
        .expect("left input");
    let right = graph
        .input([16], fusion_pcu::PcuScalarType::F32)
        .expect("right input");
    let fanout = graph.add(left, right).expect("fanout producer");
    let first_output = graph.relu(fanout).expect("first branch");
    let later = graph.mul(fanout, right).expect("second fanout consumer");
    let second_output = graph.relu(later).expect("second branch");
    let last_temporary = graph.sub(left, right).expect("later temporary");
    let third_output = graph.relu(last_temporary).expect("third branch");
    let prepared = assessor
        .prepare_graph_outputs(&graph, &[first_output, second_output, third_output])
        .expect("prepare multi-output fanout graph");

    let provider = PcuOwnedDispatchMemorySession::memory_provider(&session, pool);
    let (mut memory, allocation_calls) = ReplacingMemory::with_allocation_count(provider);
    let scratch = assessor
        .prepare_scratch(&prepared, pool, &mut memory)
        .expect("allocate planned scratch slots");
    assert_eq!(
        allocation_calls.get(),
        2,
        "three transient values use two slots"
    );
    drop(scratch);
}

#[test]
#[ignore = "requires an explicitly selected working ROCm device"]
fn valid_replacement_is_revalidated_and_provider_errors_are_preserved() {
    for mutation in [Mutation::TransferOk, Mutation::TransferErr] {
        let (_discovery, session, pool) = open_device();
        let assessor = RocmTensorAssessor::new(&session).expect("tensor assessor");
        let mut graph = Graph::default();
        let state = graph
            .input([1], fusion_pcu::PcuScalarType::F32)
            .expect("input");
        let output = graph.relu(state).expect("relu");
        let prepared = assessor
            .prepare_graph(&graph, output)
            .expect("prepare graph");
        let initial = Tensor::new([1], vec![1.0]).expect("initial tensor");
        let provider = PcuOwnedDispatchMemorySession::memory_provider(&session, pool);
        let (mut memory, armed, _) = ReplacingMemory::new(provider, mutation);
        let input = assessor
            .upload_input(&initial, pool, &mut memory)
            .expect("upload");
        let mut execution = assessor
            .bind_feedback(
                &prepared,
                &[(output, state)],
                vec![(state, input)],
                pool,
                memory,
            )
            .expect("bind");
        armed.set(true);
        let update = execution.update_input(state, &initial);
        if matches!(mutation, Mutation::TransferErr) {
            assert!(matches!(update, Err(RocmTensorExecutionError::Memory(_))));
            assert!(execution.outputs().is_err(), "failed update clears outputs");
            execution
                .update_input(state, &initial)
                .expect("retry update");
        } else {
            update.expect("provider replacement update");
        }
        execution
            .execute()
            .expect("replacement revalidated before execute");
        expect_values(&execution.read_output(output).expect("read output"), &[1.0]);
    }
}

#[test]
#[ignore = "requires an explicitly selected working ROCm device"]
fn invalid_replacement_is_rejected_and_rebind_recovers() {
    let (_discovery, session, pool) = open_device();
    let assessor = RocmTensorAssessor::new(&session).expect("tensor assessor");
    let mut graph = Graph::default();
    let state = graph
        .input([1], fusion_pcu::PcuScalarType::F32)
        .expect("input");
    let output = graph.relu(state).expect("relu");
    let prepared = assessor
        .prepare_graph(&graph, output)
        .expect("prepare graph");
    let initial = Tensor::new([1], vec![1.0]).expect("initial tensor");
    let provider = PcuOwnedDispatchMemorySession::memory_provider(&session, pool);
    let (mut memory, armed, _) = ReplacingMemory::new(provider, Mutation::TransferWrongSize);
    let input = assessor
        .upload_input(&initial, pool, &mut memory)
        .expect("upload");
    let mut recovery_memory = PcuOwnedDispatchMemorySession::memory_provider(&session, pool);
    let recovery_input = assessor
        .upload_input(&initial, pool, &mut recovery_memory)
        .expect("recovery upload");
    let mut execution = assessor
        .bind_feedback(
            &prepared,
            &[(output, state)],
            vec![(state, input)],
            pool,
            memory,
        )
        .expect("bind");
    armed.set(true);
    execution
        .update_input(state, &initial)
        .expect("replacement transfer itself succeeds");
    assert!(
        matches!(
            execution.execute(),
            Err(RocmTensorExecutionError::StorageConstraint(
                TensorStorageValidationError::ResourceTooSmall {
                    value,
                    required_bytes: 4,
                    available_bytes: 3,
                }
            )) if value == state
        ),
        "undersized replacement must fail automatic revalidation"
    );
    assert!(execution.outputs().is_err());
    execution
        .rebind_input(state, recovery_input)
        .expect("rebind valid input");
    execution.execute().expect("rebound input validates");
    expect_values(&execution.read_output(output).expect("read output"), &[1.0]);
}

#[test]
#[ignore = "requires an explicitly selected working ROCm device"]
fn failed_update_without_replacement_clears_outputs_and_can_retry() {
    let (_discovery, session, pool) = open_device();
    let assessor = RocmTensorAssessor::new(&session).expect("tensor assessor");
    let mut graph = Graph::default();
    let state = graph
        .input([1], fusion_pcu::PcuScalarType::F32)
        .expect("input");
    let output = graph.relu(state).expect("relu");
    let prepared = assessor
        .prepare_graph(&graph, output)
        .expect("prepare graph");
    let initial = Tensor::new([1], vec![2.0]).expect("input tensor");
    let provider = PcuOwnedDispatchMemorySession::memory_provider(&session, pool);
    let (mut memory, armed, _) = ReplacingMemory::new(provider, Mutation::TransferFailureOnly);
    let input = assessor
        .upload_input(&initial, pool, &mut memory)
        .expect("upload");
    let mut execution = assessor
        .bind_feedback(
            &prepared,
            &[(output, state)],
            vec![(state, input)],
            pool,
            memory,
        )
        .expect("bind");
    execution.execute().expect("initial execution");
    armed.set(true);
    assert!(matches!(
        execution.update_input(state, &initial),
        Err(RocmTensorExecutionError::Memory(_))
    ));
    assert!(execution.outputs().is_err());
    execution
        .execute()
        .expect("execute revalidates retained binding");
    expect_values(&execution.read_output(output).expect("read output"), &[2.0]);
    execution
        .update_input(state, &initial)
        .expect("retry upload");
    execution.execute().expect("retry execution");
    expect_values(&execution.read_output(output).expect("read output"), &[2.0]);
}

#[test]
#[ignore = "requires an explicitly selected working ROCm device"]
fn second_step_failure_publishes_no_partial_output() {
    let (_discovery, session, pool) = open_device();
    let assessor = RocmTensorAssessor::new(&session).expect("tensor assessor");
    let mut graph = Graph::default();
    let state = graph
        .input([1], fusion_pcu::PcuScalarType::F32)
        .expect("input");
    let leaf = graph.constant_value(fusion_pcu::dialect::tensor::TensorValue::F32(
        Tensor::new([1], vec![3.0]).expect("constant"),
    ));
    let output = graph.add(state, leaf).expect("add");
    let prepared = assessor
        .prepare_graph_outputs(&graph, &[output, leaf])
        .expect("prepare graph");
    let initial = Tensor::new([1], vec![1.0]).expect("initial tensor");
    let provider = PcuOwnedDispatchMemorySession::memory_provider(&session, pool);
    let (mut memory, armed, calls) = ReplacingMemory::new(provider, Mutation::TransferErrSecond);
    let input = assessor
        .upload_input(&initial, pool, &mut memory)
        .expect("upload");
    let mut execution = assessor
        .bind_feedback(
            &prepared,
            &[(output, state)],
            vec![(state, input)],
            pool,
            memory,
        )
        .expect("bind");
    calls.set(0);
    armed.set(true);
    assert!(matches!(
        execution.execute_steps(NonZeroUsize::new(2).expect("two steps")),
        Err(RocmTensorExecutionError::Memory(_))
    ));
    assert!(
        execution.outputs().is_err(),
        "a successful first step must not become visible after step two fails"
    );
    assert!(execution.read_output(output).is_err());
    assert!(
        execution.into_outputs().is_err(),
        "a failed run must not escape partially written output storage"
    );
}

#[test]
#[ignore = "requires an explicitly selected working ROCm device"]
fn selected_constant_or_uniform_output_replacement_is_rejected() {
    for uniform in [false, true] {
        for mutation in [Mutation::TransferOk, Mutation::TransferErr] {
            let (_discovery, session, pool) = open_device();
            let assessor = RocmTensorAssessor::new(&session).expect("tensor assessor");
            let mut graph = Graph::default();
            let state = graph
                .input([1], fusion_pcu::PcuScalarType::F32)
                .expect("input");
            let leaf = if uniform {
                graph
                    .uniform_value(
                        [1],
                        fusion_pcu::dialect::tensor::TensorScalarValue::F32(3.0),
                    )
                    .expect("uniform")
            } else {
                graph.constant_value(fusion_pcu::dialect::tensor::TensorValue::F32(
                    Tensor::new([1], vec![3.0]).expect("constant"),
                ))
            };
            let output = graph.add(state, leaf).expect("add");
            let prepared = assessor
                .prepare_graph_outputs(&graph, &[output, leaf])
                .expect("prepare graph");
            let initial = Tensor::new([1], vec![1.0]).expect("initial tensor");
            let provider = PcuOwnedDispatchMemorySession::memory_provider(&session, pool);
            let (mut memory, armed, _) = ReplacingMemory::new(provider, mutation);
            let input = assessor
                .upload_input(&initial, pool, &mut memory)
                .expect("upload");
            let mut execution = assessor
                .bind_feedback(
                    &prepared,
                    &[(output, state)],
                    vec![(state, input)],
                    pool,
                    memory,
                )
                .expect("bind");
            armed.set(true);
            assert!(matches!(
                execution.execute(),
                Err(RocmTensorExecutionError::OutputResourceMismatch)
            ));
            assert!(execution.outputs().is_err());
            assert!(
                execution.execute().is_err(),
                "failed output bank must remain unusable"
            );
        }
    }
}

#[test]
#[ignore = "requires an explicitly selected working ROCm device"]
fn selected_input_copy_replacement_is_rejected_and_poisoned() {
    for mutation in [Mutation::CopyOk, Mutation::CopyErr] {
        let (_discovery, session, pool) = open_device();
        let assessor = RocmTensorAssessor::new(&session).expect("tensor assessor");
        let mut graph = Graph::default();
        let state = graph
            .input([1], fusion_pcu::PcuScalarType::F32)
            .expect("input");
        let output = graph.relu(state).expect("relu");
        let prepared = assessor
            .prepare_graph_outputs(&graph, &[state, output])
            .expect("prepare graph");
        let initial = Tensor::new([1], vec![1.0]).expect("initial tensor");
        let provider = PcuOwnedDispatchMemorySession::memory_provider(&session, pool);
        let (mut memory, armed, _) = ReplacingMemory::new(provider, mutation);
        let input = assessor
            .upload_input(&initial, pool, &mut memory)
            .expect("upload");
        let mut execution = assessor
            .bind_feedback(
                &prepared,
                &[(output, state)],
                vec![(state, input)],
                pool,
                memory,
            )
            .expect("bind");
        armed.set(true);
        assert!(matches!(
            execution.execute(),
            Err(RocmTensorExecutionError::OutputResourceMismatch)
        ));
        assert!(execution.outputs().is_err());
        assert!(
            matches!(
                execution.execute(),
                Err(RocmTensorExecutionError::OutputResourceMismatch)
            ),
            "a replaced output destination must remain poisoned"
        );
    }
}

#[test]
#[ignore = "requires a working ROCm device"]
fn typed_device_owner_is_borrowed_by_graph_without_transfer_or_mutation() {
    let (_discovery, session, pool) = open_device();
    let assessor = RocmTensorAssessor::new(&session).expect("tensor assessor");
    let input = session
        .upload_buffer(pool, &[-2.0_f32, 3.0, -4.0, 5.0])
        .expect("initial upload");
    let owner = PcuDeviceTensor::new([4], input).expect("dense owner");
    assert!(
        assessor
            .borrow_device_input(&owner, PcuMemoryPoolId(pool.0.wrapping_add(1)))
            .is_err()
    );
    let mut borrowed = assessor
        .borrow_device_input(&owner, pool)
        .expect("borrow resident input");
    let mut memory = session.memory_provider(pool);
    let replacement = Tensor::splat([4], 99.0).expect("replacement values");
    // Borrowing an immutable owner cannot grant mutation through the graph-input adapter.
    assert!(
        assessor
            .update_input(&mut borrowed, &replacement, &mut memory)
            .is_err()
    );
    let mut graph = Graph::default();
    let input_value = graph
        .input([4], fusion_pcu::PcuScalarType::F32)
        .expect("input shape");
    let output_value = graph.relu(input_value).expect("relu");
    let prepared = assessor
        .prepare_graph(&graph, output_value)
        .expect("prepare graph");
    let mut execution = assessor
        .bind(&prepared, vec![(input_value, borrowed)], pool, memory)
        .expect("bind resident borrow");
    execution.execute().expect("execute graph");
    let outputs = execution.into_outputs().expect("publish completed output");
    let (_, output) = outputs.into_iter().next().expect("one output");
    let mut observed = [0.0_f32; 4];
    session
        .download_buffer(pool, output.buffer(), &mut observed)
        .expect("final output validation");
    assert_eq!(
        observed.map(f32::to_bits),
        [0.0_f32, 3.0, 0.0, 5.0].map(f32::to_bits)
    );
    drop(output);
    session
        .download_buffer(pool, owner.buffer(), &mut observed)
        .expect("original owner remains intact");
    assert_eq!(
        observed.map(f32::to_bits),
        [-2.0_f32, 3.0, -4.0, 5.0].map(f32::to_bits)
    );
}

#[test]
#[ignore = "requires a working ROCm device"]
fn escaping_identity_output_is_independent_of_its_borrowed_source() {
    let (_discovery, session, pool) = open_device();
    let assessor = RocmTensorAssessor::new(&session).expect("tensor assessor");
    let initial = [-2.0_f32, 3.0, -4.0, 5.0];
    let buffer = session
        .upload_buffer(pool, &initial)
        .expect("initial upload");
    let owner = PcuDeviceTensor::new([4], buffer).expect("dense owner");
    let borrowed = assessor
        .borrow_device_input(&owner, pool)
        .expect("resident borrow");
    let mut graph = Graph::default();
    let value = graph
        .input([4], fusion_pcu::PcuScalarType::F32)
        .expect("input shape");
    // Selecting the input itself as an escaping result cannot export a hidden alias of owner.
    let prepared = assessor
        .prepare_graph(&graph, value)
        .expect("identity graph");
    let mut execution = assessor
        .bind(
            &prepared,
            vec![(value, borrowed)],
            pool,
            session.memory_provider(pool),
        )
        .expect("bind identity graph");
    execution.execute().expect("identity copy");
    let (_, output) = execution
        .into_outputs()
        .expect("publish independent output")
        .into_iter()
        .next()
        .expect("one selected output");
    // Consuming the source is legal after execution releases its borrow. Changing its actual
    // backing must not change the previously escaped logical output.
    let mut source = owner.into_buffer();
    session
        .refresh_buffer(pool, &mut source, &[99.0_f32; 4])
        .expect("change source");
    let mut observed = [0.0_f32; 4];
    session
        .download_buffer(pool, output.buffer(), &mut observed)
        .expect("output snapshot");
    assert_eq!(observed.map(f32::to_bits), initial.map(f32::to_bits));
    session
        .download_buffer(pool, &source, &mut observed)
        .expect("changed source");
    assert_eq!(observed.map(f32::to_bits), [99.0_f32; 4].map(f32::to_bits));
}

#[test]
#[ignore = "requires a working ROCm device"]
fn owned_program_reuses_schedule_without_overwriting_escaped_results() {
    let (_discovery, session, pool) = open_device();
    let assessor = RocmTensorAssessor::new(&session).expect("tensor assessor");
    let (input, positive, prepared) = {
        let mut graph = Graph::default();
        let input = graph
            .input([4], fusion_pcu::PcuScalarType::F32)
            .expect("input shape");
        let positive = graph.relu(input).expect("relu");
        let program = graph
            .into_selected_program(
                &[input, positive],
                TensorArithmeticRewritePolicy::Disabled,
                TensorArithmeticCapability::Strict,
                TensorPointwiseGroupingPolicy::Disabled,
            )
            .expect("owned selected program");
        (
            input,
            positive,
            assessor
                .prepare_owned_program(program)
                .expect("owned backend preparation"),
        )
    };
    let initial = [-2.0_f32, 3.0, -4.0, 5.0];
    let buffer = session
        .upload_buffer(pool, &initial)
        .expect("initial upload");
    let owner = PcuDeviceTensor::new([4], buffer).expect("input owner");
    let mut memory = session.memory_provider(pool);
    let first = assessor
        .execute_owned_program_outputs(&prepared, &[(input, &owner)], pool, &mut memory)
        .expect("first owned execution");
    let mut source = owner.into_buffer();
    let changed = [9.0_f32, -10.0, 11.0, -12.0];
    session
        .refresh_buffer(pool, &mut source, &changed)
        .expect("change source");
    let owner = PcuDeviceTensor::new([4], source).expect("same input backing");
    let second = assessor
        .execute_owned_program_outputs(&prepared, &[(input, &owner)], pool, &mut memory)
        .expect("second owned execution");
    for (outputs, expected_input) in [(&first, initial), (&second, changed)] {
        for (value, expected) in [
            (input, expected_input),
            (positive, expected_input.map(|x| x.max(0.0))),
        ] {
            let (_, output) = outputs
                .iter()
                .find(|(id, _)| *id == value)
                .expect("selected owned output");
            assert_eq!(output.shape(), [4]);
            let mut observed = [0.0_f32; 4];
            session
                .download_buffer(pool, output.buffer(), &mut observed)
                .expect("owned output readback");
            assert_eq!(observed.map(f32::to_bits), expected.map(f32::to_bits));
        }
    }
}

#[test]
#[ignore = "requires a working ROCm device"]
fn retained_session_reuses_warm_state_after_outer_handles_drop() {
    let (_discovery, backend, pool) = open_device();
    let backend = Rc::new(backend);
    let weak_backend = Rc::downgrade(&backend);
    let root = Rc::new(RocmOwnedTensorAssessor::new(Rc::clone(&backend)).expect("owned session"));
    let buffer = root
        .backend()
        .upload_buffer(pool, &[-2.0_f32, 3.0, -4.0, 5.0])
        .expect("source upload");
    let source = PcuDeviceTensor::new([4], buffer).expect("source shape");
    drop(backend);
    assert!(weak_backend.upgrade().is_some());
    let mut graph = Graph::default();
    let input = graph
        .input([4], fusion_pcu::PcuScalarType::F32)
        .expect("input");
    let positive = graph.relu(input).expect("relu");
    {
        let view = root.assessor();
        let prepared = view.prepare_graph(&graph, positive).expect("prepare graph");
        let cold = view
            .prewarm_prepared_graph(&prepared)
            .expect("cold prewarm");
        assert!(cold.compiled_keys > 0);
        drop(view);
        let view = root.assessor();
        let warm = view
            .prewarm_prepared_graph(&prepared)
            .expect("second view prewarm");
        assert_eq!(warm.compiled_keys, 0);
        assert_eq!(warm.cache_hits, warm.requested_keys);
    }
    let program = graph
        .into_selected_program(
            &[positive],
            TensorArithmeticRewritePolicy::Disabled,
            TensorArithmeticCapability::Strict,
            TensorPointwiseGroupingPolicy::Disabled,
        )
        .expect("owned graph");
    let prepared = root
        .assessor()
        .prepare_owned_program(program)
        .expect("owned preparation");
    let mut memory = root.backend().memory_provider(pool);
    let outputs = root
        .assessor()
        .execute_owned_program_outputs(&prepared, &[(input, &source)], pool, &mut memory)
        .expect("first execution");
    let (_, output) = outputs.into_iter().next().expect("one output");
    // The future facade owner carries this pair privately. The Rc retains both the selected
    // backend and warm assessor state even after the outer execution-environment handle drops.
    let retained = (Rc::clone(&root), output);
    drop(root);
    let mut observed = [0.0_f32; 4];
    retained
        .0
        .backend()
        .download_buffer(pool, retained.1.buffer(), &mut observed)
        .expect("held result");
    assert_eq!(
        observed.map(f32::to_bits),
        [0.0_f32, 3.0, 0.0, 5.0].map(f32::to_bits)
    );
    let outputs = retained
        .0
        .assessor()
        .execute_owned_program_outputs(&prepared, &[(input, &source)], pool, &mut memory)
        .expect("execute after outer handle drop");
    retained
        .0
        .backend()
        .download_buffer(pool, outputs[0].1.buffer(), &mut observed)
        .expect("second result");
    assert_eq!(
        observed.map(f32::to_bits),
        [0.0_f32, 3.0, 0.0, 5.0].map(f32::to_bits)
    );
    drop(outputs);
    drop(retained);
    assert!(
        weak_backend.upgrade().is_none(),
        "session root must not leak the backend"
    );
}

#[test]
#[ignore = "requires a working ROCm device"]
fn shared_captured_program_retains_identity_across_candidate_preparations() {
    let (_discovery, session, pool) = open_device();
    let assessor = RocmTensorAssessor::new(&session).expect("assessor");
    let mut graph = Graph::default();
    let input = graph
        .input([4], fusion_pcu::PcuScalarType::F32)
        .expect("input");
    let output = graph.relu(input).expect("relu");
    let program = Arc::new(
        graph
            .into_selected_program(
                &[output],
                TensorArithmeticRewritePolicy::Disabled,
                TensorArithmeticCapability::Strict,
                TensorPointwiseGroupingPolicy::Disabled,
            )
            .expect("captured program"),
    );
    let first = assessor
        .prepare_shared_owned_program(Arc::clone(&program))
        .expect("first admission");
    let second = assessor
        .prepare_shared_owned_program(Arc::clone(&program))
        .expect("second admission");
    assert!(core::ptr::eq(
        first.tensor_program(),
        second.tensor_program()
    ));
    assert_eq!(Arc::strong_count(&program), 3);
    let weak = Arc::downgrade(&program);
    drop(program);
    let values = [-2.0_f32, 3.0, -4.0, 5.0];
    let source = PcuDeviceTensor::new([4], session.upload_buffer(pool, &values).expect("upload"))
        .expect("shape");
    let mut memory = session.memory_provider(pool);
    for prepared in [&first, &second] {
        let outputs = assessor
            .execute_owned_program_outputs(prepared, &[(input, &source)], pool, &mut memory)
            .expect("execute captured program");
        let mut observed = [0.0_f32; 4];
        session
            .download_buffer(pool, outputs[0].1.buffer(), &mut observed)
            .expect("readback");
        assert_eq!(
            observed.map(f32::to_bits),
            values.map(|x| x.max(0.0).to_bits())
        );
    }
    drop(first);
    assert!(weak.upgrade().is_some());
    drop(second);
    assert!(weak.upgrade().is_none());
}
