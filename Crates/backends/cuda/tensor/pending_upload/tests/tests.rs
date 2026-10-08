#[rustfmt::skip]
use super::{
    PendingUpload,
    StagingEligibility,
    eligible,
};
#[rustfmt::skip]
use super::super::{
    CudaOwnedPreparedTensorGraph,
    CudaTensorAssessor,
    CudaTensorExecutionError,
    Graph,
    Tensor,
    TensorArithmeticCapability,
    TensorArithmeticRewritePolicy,
    TensorPointwiseGroupingPolicy,
    tests::cuda_test_session,
};
#[rustfmt::skip]
use fusion_pcu::{
    PcuDeviceTensor,
    PcuMemoryPoolId,
    PcuScalarType,
};
use fusion_pcu::dialect::tensor::ValueId;
#[rustfmt::skip]
use std::{
    rc::Rc,
    sync::Arc,
};

fn prepare(
    assessor: &CudaTensorAssessor<'_>,
    n: usize,
    math: bool,
) -> (ValueId, CudaOwnedPreparedTensorGraph) {
    let mut graph = Graph::default();
    let input = graph.input_typed::<u32>([n]).unwrap();
    let output = if math {
        let literal = graph.constant_typed(Tensor::new([n], vec![1_u32; n]).unwrap());
        let factor = graph.uniform_typed([n], 2_u32).unwrap();
        let sum = graph.add_typed(input, literal).unwrap();
        graph.mul_typed(sum, factor).unwrap()
    } else {
        input
    };
    let program = graph
        .into_selected_program(
            &[output.erase()],
            TensorArithmeticRewritePolicy::Disabled,
            TensorArithmeticCapability::Strict,
            TensorPointwiseGroupingPolicy::Disabled,
        )
        .unwrap();
    (
        input.erase(),
        assessor.prepare_owned_program(program).unwrap(),
    )
}

#[test]
#[ignore = "requires native CUDA; private same-queue input staging checked Add/Mul prototype"]
#[expect(
    clippy::too_many_lines,
    reason = "Keep endpoint authority and repeated terminal census in one resource-lifetime witness."
)]
fn pending_upload_preserves_changed_inputs_exact_queue_authority_and_terminal_replay() {
    let (_discovery, session) = cuda_test_session();
    let pool = PcuMemoryPoolId(0x4352_0190);
    let assessor = CudaTensorAssessor::new(&session).unwrap();
    let mut memory = session.memory_provider(pool);
    for n in [65, 4096] {
        let shape = [n];
        let (input, prepared) = prepare(&assessor, n, true);
        assert!(eligible(&prepared));
        let eligibility = StagingEligibility::new(&prepared);
        let owner =
            PcuDeviceTensor::new([n], session.upload_buffer(pool, &vec![0_u32; n]).unwrap())
                .unwrap();
        // Complete cold scratch/status/kernel preparation before changing-input scopes.
        let initial = assessor
            .execute_owned_program_outputs(&prepared, &[(input, &owner)], pool, &mut memory)
            .unwrap();
        drop(initial);
        let foreign_queue = session.create_stream().unwrap();
        let foreign_assessor = CudaTensorAssessor::new(&session).unwrap();
        let other =
            PcuDeviceTensor::new([n], session.upload_buffer(pool, &vec![0_u32; n]).unwrap())
                .unwrap();
        let resource = owner.buffer().resource();
        let mut bytes = vec![0_u8; n * 4];
        for generation in 1_u32..=64 {
            for (index, word) in bytes.as_chunks_mut::<4>().0.iter_mut().enumerate() {
                word.copy_from_slice(
                    &(generation + u32::try_from(index % 7).unwrap()).to_le_bytes(),
                );
            }
            #[cfg(feature = "allocation-census")]
            let before = crate::cuda_api_census();
            let pending =
                PendingUpload::queue(&assessor, resource, &[n], PcuScalarType::U32, pool, &bytes)
                    .unwrap();
            assert!(pending.authorizes(resource.device_buffer(), &assessor.state().stream));
            assert!(!pending.authorizes(resource.device_buffer(), &foreign_queue));
            assert!(!pending.authorizes(
                other.buffer().resource().device_buffer(),
                &assessor.state().stream
            ));
            assert!(
                pending
                    .borrow_input(&foreign_assessor, resource, &[n], PcuScalarType::U32, pool)
                    .is_err()
            );
            assert!(resource.validate_access_available().is_err());
            let mut host_alias = resource.device_buffer().clone();
            assert!(host_alias.copy_from(&bytes).is_err());
            assert!(
                PendingUpload::queue(&assessor, resource, &[n], PcuScalarType::U32, pool, &bytes)
                    .is_err()
            );
            // Caller payload is no DMA endpoint; mutation after enqueue cannot change results.
            bytes.fill(0);
            let terminal = {
                let borrowed = pending
                    .borrow_input(&assessor, resource, &shape, PcuScalarType::U32, pool)
                    .unwrap();
                assessor.execute_owned_program_output_from_inputs::<u32, _>(
                    &prepared,
                    &[(input, &borrowed)],
                    pool,
                    &mut memory,
                )
            };
            pending.complete_after_schedule(&eligibility, &terminal);
            #[cfg(feature = "allocation-census")]
            {
                let after = crate::cuda_api_census();
                assert_eq!(
                    after.host_to_device_copies - before.host_to_device_copies,
                    1
                );
                assert_eq!(after.kernel_launches - before.kernel_launches, 2);
                // Prepared status owners retain their terminal events across warm calls.
                assert_eq!(after.event_creates, before.event_creates);
                assert_eq!(after.event_destroys, before.event_destroys);
                assert_eq!(after.event_records - before.event_records, 2);
                assert_eq!(after.event_waits - before.event_waits, 2);
                assert_eq!(
                    after.stream_synchronizations,
                    before.stream_synchronizations
                );
                assert_eq!(
                    after.device_synchronizations,
                    before.device_synchronizations
                );
                assert_eq!(after.symbol_resolutions, before.symbol_resolutions);
                assert_eq!(after.module_loads, before.module_loads);
            }
            let output = terminal.unwrap();
            assert!(resource.validate_access_available().is_ok());
            let mut actual = vec![999_u32; n + 1];
            session
                .download_buffer(pool, output.buffer(), &mut actual[..n])
                .unwrap();
            for (index, &value) in actual[..n].iter().enumerate() {
                assert_eq!(
                    value,
                    2 * (generation + u32::try_from(index % 7).unwrap() + 1)
                );
            }
            assert_eq!(actual[n], 999);
        }
    }
}

#[test]
#[ignore = "requires native CUDA; pending-upload no-dispatch and uncertain completion ownership"]
fn pending_upload_identity_drop_fences_and_unknown_completion_retains_endpoint_roots() {
    let (_discovery, session) = cuda_test_session();
    let pool = PcuMemoryPoolId(0x4352_0191);
    let assessor = CudaTensorAssessor::new(&session).unwrap();
    let (_, identity) = prepare(&assessor, 65, false);
    assert!(!eligible(&identity));
    let eligibility = StagingEligibility::new(&identity);
    let owner =
        PcuDeviceTensor::new([65], session.upload_buffer(pool, &[0_u32; 65]).unwrap()).unwrap();
    let resource = owner.buffer().resource();
    let bytes = [0x11_u8; 260];
    let pending =
        PendingUpload::queue(&assessor, resource, &[65], PcuScalarType::U32, pool, &bytes).unwrap();
    // An identity success has no consumer terminal proof, so it must take the Drop fence.
    pending.complete_after_schedule(&eligibility, &Ok::<_, CudaTensorExecutionError>(()));
    let mut actual = [0_u32; 65];
    session
        .download_buffer(pool, owner.buffer(), &mut actual)
        .unwrap();
    assert_eq!(actual, [0x1111_1111; 65]);
    let mut stale =
        PendingUpload::queue(&assessor, resource, &[65], PcuScalarType::U32, pool, &bytes).unwrap();
    stale.stream.synchronize().unwrap();
    stale.release_after_completion(&Ok(()));
    assert!(!stale.authorizes(resource.device_buffer(), &assessor.state().stream));
    assert!(
        stale
            .borrow_input(&assessor, resource, &[65], PcuScalarType::U32, pool)
            .is_err()
    );
    drop(stale);
    let mut pending =
        PendingUpload::queue(&assessor, resource, &[65], PcuScalarType::U32, pool, &bytes).unwrap();
    let allocation = Rc::downgrade(&resource.device_buffer().allocation);
    let queue = Rc::downgrade(&pending.stream.inner);
    let runtime = Arc::downgrade(&pending.stream.inner.runtime.0);
    // Establish actual quiescence before injecting only an owned result. No invalid handle,
    // raced GPU work or device-loss claim; the production quarantine branch is exercised.
    pending.stream.synchronize().unwrap();
    pending.release_after_completion(&Err(crate::CudaError::Busy));
    assert!(resource.validate_access_available().is_err());
    drop(pending);
    drop(owner);
    drop(assessor);
    drop(session);
    assert!(allocation.upgrade().is_some());
    assert!(queue.upgrade().is_some());
    assert!(runtime.upgrade().is_some());
}

#[test]
#[ignore = "requires native CUDA; pending staging preflight, ordered fault and terminal retry"]
#[expect(
    clippy::too_many_lines,
    reason = "One ordered preflight/fault/retry witness retains the escaped predecessor output throughout."
)]
fn pending_upload_preflights_without_work_and_preserves_add_phase_fault_before_mul() {
    let (_discovery, session) = cuda_test_session();
    let pool = PcuMemoryPoolId(0x4352_0192);
    let assessor = CudaTensorAssessor::new(&session).unwrap();
    let mut memory = session.memory_provider(pool);
    let (input, prepared) = prepare(&assessor, 65, true);
    let eligibility = StagingEligibility::new(&prepared);
    let owner =
        PcuDeviceTensor::new([65], session.upload_buffer(pool, &[3_u32; 65]).unwrap()).unwrap();
    let initial = assessor
        .execute_owned_program_outputs(&prepared, &[(input, &owner)], pool, &mut memory)
        .unwrap();
    let resource = owner.buffer().resource();
    let healthy = [3_u32; 65]
        .into_iter()
        .flat_map(u32::to_le_bytes)
        .collect::<Vec<_>>();
    #[cfg(feature = "allocation-census")]
    let before = crate::cuda_api_census();
    assert!(
        PendingUpload::queue(
            &assessor,
            resource,
            &[66],
            PcuScalarType::U32,
            pool,
            &healthy
        )
        .is_err()
    );
    assert!(
        PendingUpload::queue(
            &assessor,
            resource,
            &[65],
            PcuScalarType::U32,
            pool,
            &healthy[..256]
        )
        .is_err()
    );
    assert!(
        PendingUpload::queue(&assessor, resource, &[0], PcuScalarType::U32, pool, &[]).is_err()
    );
    assert!(
        PendingUpload::queue(
            &assessor,
            resource,
            &[65],
            PcuScalarType::U32,
            PcuMemoryPoolId(pool.0 + 1),
            &healthy
        )
        .is_err()
    );
    #[cfg(feature = "allocation-census")]
    assert_eq!(before, crate::cuda_api_census());
    let mut bad = [3_u32; 65];
    bad[3] = u32::MAX / 2 + 1; // Add healthy, earlier-invocation Mul overflow.
    bad[7] = u32::MAX; // Later-invocation Add must still win the earlier global phase.
    let bad = bad
        .into_iter()
        .flat_map(u32::to_le_bytes)
        .collect::<Vec<_>>();
    for (bytes, fault) in [(&bad, true), (&healthy, false)] {
        let pending =
            PendingUpload::queue(&assessor, resource, &[65], PcuScalarType::U32, pool, bytes)
                .unwrap();
        let terminal = {
            let borrowed = pending
                .borrow_input(&assessor, resource, &[65], PcuScalarType::U32, pool)
                .unwrap();
            assessor.execute_owned_program_output_from_inputs::<u32, _>(
                &prepared,
                &[(input, &borrowed)],
                pool,
                &mut memory,
            )
        };
        pending.complete_after_schedule(&eligibility, &terminal);
        assert!(resource.validate_access_available().is_ok());
        if fault {
            let error = terminal
                .err()
                .expect("the Add phase must fault before publication");
            assert!(
                matches!(error, CudaTensorExecutionError::ExecutionFault(fault) if fault.invocation_id == 7)
            );
        } else {
            let output = terminal.unwrap();
            let mut actual = [99_u32; 66];
            session
                .download_buffer(pool, output.buffer(), &mut actual[..65])
                .unwrap();
            assert_eq!(actual[..65], [8_u32; 65]);
            assert_eq!(actual[65], 99);
        }
    }
    // The earlier escaped result remains immutable across failed publication and retry.
    let mut old = [99_u32; 66];
    session
        .download_buffer(pool, initial[0].1.buffer(), &mut old[..65])
        .unwrap();
    assert_eq!(old[..65], [8_u32; 65]);
    assert_eq!(old[65], 99);
}

#[path = "aggregate/aggregate.rs"]
mod aggregate;

fn prepare_pure(graph: Graph, outputs: &[ValueId]) -> CudaOwnedPreparedTensorGraph {
    let data = super::super::tests::prepared_for_request_test(
        &graph,
        outputs,
        TensorPointwiseGroupingPolicy::Disabled,
    )
    .data;
    let program = graph
        .into_selected_program(
            outputs,
            TensorArithmeticRewritePolicy::Disabled,
            TensorArithmeticCapability::Strict,
            TensorPointwiseGroupingPolicy::Disabled,
        )
        .unwrap();
    CudaOwnedPreparedTensorGraph::from_parts(Arc::new(program), data).unwrap()
}

#[test]
fn staging_terminal_proof_uses_concrete_dispatches_without_arithmetic_whitelist() {
    let mut graph = Graph::default();
    let left = graph.input([65], PcuScalarType::F32).unwrap();
    let right = graph.input([65], PcuScalarType::F32).unwrap();
    let sub = graph.sub(left, right).unwrap();
    let div = graph.div(sub, right).unwrap();
    let output = graph.relu(div).unwrap();
    let mut prepared = prepare_pure(graph, &[output]);
    assert!(eligible(&prepared));
    // Same graph and numerical policy, but losing a concrete implementation fact must refuse
    // staging before any upload. This is an execution proof, not source-op recognition.
    let index = prepared.data.index_by_value[&sub];
    let dispatch = prepared.data.fixed_dispatches[index].take();
    assert!(!eligible(&prepared));
    prepared.data.fixed_dispatches[index] = dispatch;
    assert!(eligible(&prepared));
    prepared.data.fixed_dispatches[index]
        .as_mut()
        .unwrap()
        .value = output;
    assert!(!eligible(&prepared));
}

#[test]
fn staging_terminal_proof_refuses_source_only_multiple_outputs_and_library_work() {
    let mut identity = Graph::default();
    let input = identity.input([65], PcuScalarType::F32).unwrap();
    assert!(!eligible(&prepare_pure(identity, &[input])));

    let mut multiple = Graph::default();
    let input = multiple.input([65], PcuScalarType::F32).unwrap();
    let output = multiple.relu(input).unwrap();
    assert!(!eligible(&prepare_pure(multiple, &[input, output])));

    let mut library = Graph::default();
    let left = library.input([2, 2], PcuScalarType::F32).unwrap();
    let right = library.input([2, 2], PcuScalarType::F32).unwrap();
    let output = library.matmul(left, right).unwrap();
    assert!(!eligible(&prepare_pure(library, &[output])));
}

#[test]
#[ignore = "requires native CUDA; generic fixed Sub/Div/Relu upload terminal and fault replay"]
#[expect(
    clippy::too_many_lines,
    reason = "Keep changed-input success, fault retry, early failure and retained escaped output in one ownership witness."
)]
fn staging_fixed_dispatch_chain_preserves_changed_inputs_and_division_fault_retry() {
    let (_discovery, session) = cuda_test_session();
    let pool = PcuMemoryPoolId(0x4352_0195);
    let assessor = CudaTensorAssessor::new(&session).unwrap();
    let mut memory = session.memory_provider(pool);
    let shape = [65];
    let mut graph = Graph::default();
    let input = graph.input_typed::<f32>(shape).unwrap();
    let denominator = graph.input_typed::<f32>(shape).unwrap();
    let difference = graph.sub_typed(input, denominator).unwrap();
    let quotient = graph.div_typed(difference, denominator).unwrap();
    let output = graph.relu_typed(quotient).unwrap();
    let program = graph
        .into_selected_program(
            &[output.erase()],
            TensorArithmeticRewritePolicy::Disabled,
            TensorArithmeticCapability::Strict,
            TensorPointwiseGroupingPolicy::Disabled,
        )
        .unwrap();
    let prepared = assessor.prepare_owned_program(program).unwrap();
    assert!(eligible(&prepared));
    let source =
        PcuDeviceTensor::new(shape, session.upload_buffer(pool, &[8.0_f32; 65]).unwrap()).unwrap();
    let divisor =
        PcuDeviceTensor::new(shape, session.upload_buffer(pool, &[2.0_f32; 65]).unwrap()).unwrap();
    let cold = assessor
        .execute_owned_program_outputs(
            &prepared,
            &[(input.erase(), &source), (denominator.erase(), &divisor)],
            pool,
            &mut memory,
        )
        .unwrap();
    for generation in 0_u32..=8 {
        let source_bytes = [8.0 + f32::from(u16::try_from(generation).unwrap()); 65]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>();
        let bad = generation == 4;
        let mut divisors = [2.0_f32; 65];
        if bad {
            divisors[17] = 0.0;
        }
        let divisor_bytes = divisors
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>();
        let bindings = [
            super::CudaHostedTensorInput {
                value: input.erase(),
                resource: source.buffer().resource(),
                shape: &shape,
                source: &source_bytes,
            },
            super::CudaHostedTensorInput {
                value: denominator.erase(),
                resource: divisor.buffer().resource(),
                shape: &shape,
                source: &divisor_bytes,
            },
        ];
        let terminal = assessor.execute_owned_program_output_from_host_staging::<f32, _, 2>(
            &prepared,
            &bindings,
            pool,
            &mut memory,
        );
        assert!(
            source
                .buffer()
                .resource()
                .validate_access_available()
                .is_ok()
        );
        assert!(
            divisor
                .buffer()
                .resource()
                .validate_access_available()
                .is_ok()
        );
        if bad {
            assert!(matches!(terminal,
                Err(CudaTensorExecutionError::ExecutionFault(fault))
                    if fault.invocation_id == 17));
        } else {
            let result = terminal.unwrap().expect("warm generic dispatch staging");
            let mut host = [99.0_f32; 66];
            session
                .download_buffer(pool, result.buffer(), &mut host[..65])
                .unwrap();
            let numerator = 6_u16
                .checked_add(u16::try_from(generation).unwrap())
                .unwrap();
            let expected = f32::from(numerator) / 2.0;
            assert!(
                host[..65]
                    .iter()
                    .all(|value| value.to_bits() == expected.to_bits())
            );
            assert_eq!(host[65].to_bits(), 99.0_f32.to_bits());
        }
    }
    // A provider-domain rejection occurs after uploads but before the first dispatch. Drop
    // must fence both pending endpoints rather than leaving input leases permanently busy.
    let mut foreign_memory = session.memory_provider(PcuMemoryPoolId(pool.0 + 1));
    let bytes = [2.0_f32; 65]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect::<Vec<_>>();
    let bindings = [
        super::CudaHostedTensorInput {
            value: input.erase(),
            resource: source.buffer().resource(),
            shape: &shape,
            source: &bytes,
        },
        super::CudaHostedTensorInput {
            value: denominator.erase(),
            resource: divisor.buffer().resource(),
            shape: &shape,
            source: &bytes,
        },
    ];
    #[cfg(feature = "allocation-census")]
    let before = crate::cuda_api_census();
    assert!(
        assessor
            .execute_owned_program_output_from_host_staging::<f32, _, 2>(
                &prepared,
                &bindings,
                pool,
                &mut foreign_memory,
            )
            .is_err()
    );
    #[cfg(feature = "allocation-census")]
    {
        let after = crate::cuda_api_census();
        assert_eq!(
            after.host_to_device_copies - before.host_to_device_copies,
            2
        );
        assert_eq!(after.kernel_launches, before.kernel_launches);
        assert!(after.stream_synchronizations > before.stream_synchronizations);
    }
    assert!(
        source
            .buffer()
            .resource()
            .validate_access_available()
            .is_ok()
    );
    assert!(
        divisor
            .buffer()
            .resource()
            .validate_access_available()
            .is_ok()
    );
    let mut original = [0.0_f32; 65];
    session
        .download_buffer(pool, cold[0].1.buffer(), &mut original)
        .unwrap();
    assert_eq!(original.map(f32::to_bits), [3.0_f32.to_bits(); 65]);
}
