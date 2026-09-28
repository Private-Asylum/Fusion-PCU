//! Hardware integration regressions for automatic prepared tensor execution.
//!
//! Run explicitly on a `ROCm` host with:
//! `cargo test -p fusion-pcu-example-rocm --test automatic_execution -- --ignored --nocapture`

#[rustfmt::skip]
use std::{
    cell::Cell,
    num::NonZeroUsize,
    rc::Rc,
};

#[rustfmt::skip]
use fusion_pcu::{
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
#[rustfmt::skip]
use fusion_pcu_rocm::{
    RocmDiscovery,
    RocmMemoryResource,
    RocmOwnedDispatchBackend,
    RocmTensorAssessor,
    RocmTensorExecutionError,
};
#[rustfmt::skip]
use fusion_pcu_tensor::{
    Graph,
    Tensor,
    TensorStorageValidationError,
    TensorExecution,
    ValueId,
};

#[path = "../selection.rs"]
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
            },
            armed,
            calls,
        )
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

#[test]
#[ignore = "requires an explicitly selected working ROCm device"]
fn bound_feedback_matches_cpu_for_one_through_four_steps() {
    let (_discovery, session, pool) = open_device();
    let assessor = RocmTensorAssessor::new(&session).expect("tensor assessor");
    let mut graph = Graph::default();
    let state = graph.input([1]).expect("input");
    let increment = graph.constant(Tensor::new([1], vec![2.0]).expect("constant"));
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
    let input_value = graph.input([1]).expect("input");
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
fn valid_replacement_is_revalidated_and_provider_errors_are_preserved() {
    for mutation in [Mutation::TransferOk, Mutation::TransferErr] {
        let (_discovery, session, pool) = open_device();
        let assessor = RocmTensorAssessor::new(&session).expect("tensor assessor");
        let mut graph = Graph::default();
        let state = graph.input([1]).expect("input");
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
    let state = graph.input([1]).expect("input");
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
    let state = graph.input([1]).expect("input");
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
    let state = graph.input([1]).expect("input");
    let leaf = graph.constant(Tensor::new([1], vec![3.0]).expect("constant"));
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
}

#[test]
#[ignore = "requires an explicitly selected working ROCm device"]
fn selected_constant_or_uniform_output_replacement_is_rejected() {
    for uniform in [false, true] {
        for mutation in [Mutation::TransferOk, Mutation::TransferErr] {
            let (_discovery, session, pool) = open_device();
            let assessor = RocmTensorAssessor::new(&session).expect("tensor assessor");
            let mut graph = Graph::default();
            let state = graph.input([1]).expect("input");
            let leaf = if uniform {
                graph.uniform([1], 3.0).expect("uniform")
            } else {
                graph.constant(Tensor::new([1], vec![3.0]).expect("constant"))
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
        let state = graph.input([1]).expect("input");
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
