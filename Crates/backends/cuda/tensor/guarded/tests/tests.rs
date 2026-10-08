//! Original-kernel native guarded scope, fault, retry, records and escaped ownership.
use super::super::*;
use crate::tensor::tests as tensor_tests;

fn program() -> (
    ValueId,
    fusion_pcu::dialect::tensor::TensorOwnedSelectedProgram,
) {
    let mut graph = Graph::default();
    let input = graph.input_typed::<i32>([65]).unwrap();
    let first = graph.add_typed(input, input).unwrap();
    let output = graph.add_typed(first, input).unwrap();
    let program = graph
        .into_selected_program(
            &[output.erase()],
            TensorArithmeticRewritePolicy::Disabled,
            TensorArithmeticCapability::Strict,
            TensorPointwiseGroupingPolicy::Disabled,
        )
        .unwrap();
    (input.erase(), program)
}

#[test]
#[ignore = "requires native CUDA device; one terminal event, exact guarded status and retry"]
#[allow(clippy::too_many_lines)] // Keep actual ownership, fault, retry and stream assertions together.
fn guarded_chain_success_fault_retry_and_stream_rebind() {
    let (_discovery, session) = tensor_tests::cuda_test_session();
    let a = CudaTensorAssessor::new(&session).unwrap();
    let b = CudaTensorAssessor::new(&session).unwrap();
    let pool = PcuMemoryPoolId(0x4755_0101);
    let (input, program) = program();
    let mut prepared = a.prepare_owned_program(program).unwrap();
    assert!(
        !a.prepare_owned_program_guarded_execution(&mut prepared)
            .unwrap()
    );
    a.retain_owned_program_fixed_dispatches(&mut prepared)
        .unwrap();
    assert!(
        a.prepare_owned_program_guarded_execution(&mut prepared)
            .unwrap()
    );
    assert_eq!(prepared.guarded.as_ref().unwrap().nodes.len(), 2);
    let mut memory = session.memory_provider(pool);
    let owner =
        fusion_pcu::PcuDeviceTensor::new([65], session.upload_buffer(pool, &[3_i32; 65]).unwrap())
            .unwrap();
    let escaped = a
        .execute_owned_program_outputs(&prepared, &[(input, &owner)], pool, &mut memory)
        .unwrap()
        .pop()
        .unwrap()
        .1;
    let mut observed = [0_i32; 65];
    session
        .download_buffer(pool, escaped.buffer(), &mut observed)
        .unwrap();
    assert_eq!(observed, [9; 65]);
    #[cfg(feature = "insights")]
    {
        let report = prepared.last_guarded_execution_report(pool).unwrap();
        assert_eq!(report.stage_count, 2);
        assert_eq!(
            report.outcome,
            fusion_pcu::PcuGuardedExecutionOutcome::Succeeded
        );
        assert_eq!(
            &report.stages[..2],
            &[Some(fusion_pcu::PcuGuardedExecutionStageOutcome::Succeeded); 2]
        );
    }
    #[cfg(feature = "allocation-census")]
    let before = crate::cuda_api_census();
    let cache = a.state().add_dispatches.borrow_mut();
    let second = a
        .execute_owned_program_outputs(&prepared, &[(input, &owner)], pool, &mut memory)
        .unwrap()
        .pop()
        .unwrap()
        .1;
    drop(cache);
    #[cfg(feature = "allocation-census")]
    {
        let after = crate::cuda_api_census();
        assert_eq!(
            after.guarded_chain_submissions - before.guarded_chain_submissions,
            1
        );
        assert_eq!(
            after.guarded_kernel_launches - before.guarded_kernel_launches,
            2
        );
        assert_eq!(after.kernel_launches - before.kernel_launches, 2);
        assert_eq!(after.event_waits - before.event_waits, 1);
        assert_eq!(after.event_records - before.event_records, 1);
        assert_eq!(after.event_creates - before.event_creates, 0);
        // The escaping output allocation selects independently; the entire private guarded
        // submission selects once. This bounds host API work without hiding allocation work.
        assert_eq!(after.allocations - before.allocations, 1);
        assert_eq!(after.device_selections - before.device_selections, 2);
        assert_eq!(
            after.host_to_device_copies - before.host_to_device_copies,
            1
        );
        assert_eq!(
            after.device_to_host_copies - before.device_to_host_copies,
            1
        );
        assert_eq!(after.module_loads, before.module_loads);
        assert_eq!(
            after.stream_synchronizations,
            before.stream_synchronizations
        );
    }
    assert!(
        !escaped
            .buffer()
            .resource()
            .may_overlap(second.buffer().resource())
    );
    for (value, expected_stage) in [(i32::MAX, 0_usize), (i32::MAX / 2, 1)] {
        let bad = fusion_pcu::PcuDeviceTensor::new(
            [65],
            session.upload_buffer(pool, &[value; 65]).unwrap(),
        )
        .unwrap();
        assert!(matches!(
            a.execute_owned_program_outputs(&prepared, &[(input, &bad)], pool, &mut memory),
            Err(CudaTensorExecutionError::ExecutionFault(_))
        ));
        #[cfg(feature = "insights")]
        {
            let report = prepared.last_guarded_execution_report(pool).unwrap();
            assert!(
                matches!(report.outcome, fusion_pcu::PcuGuardedExecutionOutcome::Fault { stage, .. } if stage == expected_stage)
            );
            assert!(report.values[..2].iter().all(Option::is_some));
            if expected_stage == 0 {
                assert_eq!(
                    report.stages[1],
                    Some(fusion_pcu::PcuGuardedExecutionStageOutcome::Skipped)
                );
            }
        }
        let view = prepared.view();
        let bank = prepared
            .scratch
            .bind(session.tensor_runtime(), &view, pool, &mut memory)
            .unwrap();
        let slab = bank.guarded.as_ref().unwrap();
        let mut bytes = [0_u8; 32];
        slab.resource.device_buffer().copy_to(&mut bytes).unwrap();
        let records = bytes
            .as_chunks::<16>()
            .0
            .iter()
            .map(|record| {
                (
                    u64::from_le_bytes(record[..8].try_into().unwrap()),
                    u64::from_le_bytes(record[8..].try_into().unwrap()),
                )
            })
            .collect::<Vec<_>>();
        assert_ne!(records[expected_stage].0, u64::MAX);
        assert_eq!(records[expected_stage].1, 1);
        if expected_stage == 0 {
            assert_eq!(records[1], (u64::MAX, 2));
        } else {
            assert_eq!(records[0], (u64::MAX, 1));
        }
        drop(bank);
        let retry = a
            .execute_owned_program_outputs(&prepared, &[(input, &owner)], pool, &mut memory)
            .unwrap()
            .pop()
            .unwrap()
            .1;
        session
            .download_buffer(pool, retry.buffer(), &mut observed)
            .unwrap();
        assert_eq!(observed, [9; 65]);
    }
    session
        .download_buffer(pool, escaped.buffer(), &mut observed)
        .unwrap();
    assert_eq!(observed, [9; 65]);
    let mut ordered_values = [1_i32; 65];
    ordered_values[0] = i32::MAX / 2;
    ordered_values[64] = i32::MAX;
    let ordered = fusion_pcu::PcuDeviceTensor::new(
        [65],
        session.upload_buffer(pool, &ordered_values).unwrap(),
    )
    .unwrap();
    assert!(
        matches!(a.execute_owned_program_outputs(&prepared, &[(input, &ordered)], pool, &mut memory), Err(CudaTensorExecutionError::ExecutionFault(fault)) if fault.invocation_id == 64)
    );
    let short =
        fusion_pcu::PcuDeviceTensor::new([64], session.upload_buffer(pool, &[1_i32; 64]).unwrap())
            .unwrap();
    #[cfg(feature = "allocation-census")]
    let before = crate::cuda_api_census();
    assert!(
        a.execute_owned_program_outputs(&prepared, &[(input, &short)], pool, &mut memory)
            .is_err()
    );
    #[cfg(feature = "insights")]
    assert!(prepared.last_guarded_execution_report(pool).is_none());
    #[cfg(feature = "allocation-census")]
    assert_eq!(crate::cuda_api_census(), before);
    b.retain_owned_program_fixed_dispatches(&mut prepared)
        .unwrap();
    assert!(
        b.prepare_owned_program_guarded_execution(&mut prepared)
            .unwrap()
    );
    #[cfg(feature = "insights")]
    assert!(prepared.last_guarded_execution_report(pool).is_none());
    let rebound = b
        .execute_owned_program_outputs(&prepared, &[(input, &owner)], pool, &mut memory)
        .unwrap()
        .pop()
        .unwrap()
        .1;
    session
        .download_buffer(pool, rebound.buffer(), &mut observed)
        .unwrap();
    assert_eq!(observed, [9; 65]);
    // Foreign exact-stream sidecars decline before work and retain ordinary host gates.
    let replay = a
        .execute_owned_program_outputs(&prepared, &[(input, &owner)], pool, &mut memory)
        .unwrap()
        .pop()
        .unwrap()
        .1;
    #[cfg(feature = "insights")]
    assert!(prepared.last_guarded_execution_report(pool).is_none());
    session
        .download_buffer(pool, replay.buffer(), &mut observed)
        .unwrap();
    assert_eq!(observed, [9; 65]);
    // Real work is terminal above. This is deterministic unknown-proof injection, not device loss.
    let mut bank = prepared
        .scratch
        .bind(
            session.tensor_runtime(),
            &prepared.view(),
            pool,
            &mut memory,
        )
        .unwrap();
    let slab = bank
        .guarded
        .as_ref()
        .unwrap()
        .resource
        .clone_for_tensor_input();
    slab.device_buffer()
        .with_access_lease_for_test(|| {
            bank.finish(Some(&CudaTensorExecutionError::FailedCompletion))
                .unwrap();
        })
        .unwrap();
    drop(bank);
    assert!(matches!(
        prepared.scratch.bind(
            session.tensor_runtime(),
            &prepared.view(),
            pool,
            &mut memory
        ),
        Err(CudaTensorExecutionError::ScratchMismatch)
    ));
}

#[test]
#[ignore = "requires native CUDA device; recovered clamp blocks successor user writes"]
#[allow(clippy::too_many_lines)] // Preserve original checked and unchecked native canaries together.
fn guarded_recovered_clamp_skips_successor_without_changing_original_body() {
    use fusion_pcu::PcuOwnedDispatchBackend;
    let (_discovery, session) = tensor_tests::cuda_test_session();
    let runtime = session.tensor_runtime();
    let mut kernel = checked_float::kernel(
        fusion_pcu::PcuDispatchFloatBinaryOp::Add,
        fusion_pcu::PcuScalarType::F32,
        fusion_pcu::PcuFloatUnderflowPolicy::IeeeAfterRounding,
        1,
    )
    .unwrap();
    let mut ops = kernel.ops.to_vec();
    for op in &mut ops {
        if let fusion_pcu::PcuDispatchOp::Data(
            fusion_pcu::PcuDispatchDataOp::CheckedFloatBinary { range_policy, .. },
        ) = op
        {
            *range_policy = fusion_pcu::PcuRangePolicy::Clamp;
        }
    }
    kernel.ops = &ops;
    kernel.numerical_requirements.range_policy = fusion_pcu::PcuRangePolicy::Clamp;
    kernel.feature_caps = kernel
        .feature_caps
        .union(fusion_pcu::PcuDispatchFeatureCaps::RANGE_CLAMP);
    let prepared = session
        .prepare_dispatch(fusion_pcu::PcuDispatchSubmission {
            kernel: &kernel,
            shape: fusion_pcu::PcuInvocationShape::invocations(
                core::num::NonZeroU32::new(1).unwrap(),
            ),
        })
        .unwrap();
    session.retain_guarded_dispatch(&prepared, &kernel).unwrap();
    let mut left = runtime.allocate(4).unwrap();
    let mut right = runtime.allocate(4).unwrap();
    let mut first = runtime.allocate(4).unwrap();
    let mut second = runtime.allocate(4).unwrap();
    left.copy_from(&f32::MAX.to_le_bytes()).unwrap();
    right.copy_from(&f32::MAX.to_le_bytes()).unwrap();
    first.copy_from(&17_f32.to_le_bytes()).unwrap();
    second.copy_from(&19_f32.to_le_bytes()).unwrap();
    let bindings = |output: &crate::DeviceBuffer| {
        kernel
            .bindings
            .iter()
            .zip([left.clone(), right.clone(), output.clone()])
            .map(|(binding, resource)| {
                fusion_pcu::PcuOwnedBinding::new(
                    binding.reference(),
                    session.device_identity(),
                    4,
                    binding.access,
                    binding.binding_type,
                    resource,
                )
            })
            .collect::<Vec<_>>()
    };
    let a = bindings(&first);
    let b = bindings(&second);
    prepared.validate_guarded_bindings(&a).unwrap();
    prepared.validate_guarded_bindings(&b).unwrap();
    let transport_ops = [
        RELU_OPS[1],
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
            binding: RELU_OUTPUT_REF,
            index: PcuDispatchIndex::InvocationId,
            value: RELU_INPUT,
        }),
        PcuDispatchOp::Control(PcuDispatchControlOp::Return),
    ];
    let mut unchecked_ir = relu_kernel(1);
    unchecked_ir.ops = &transport_ops;
    unchecked_ir.feature_caps = PcuDispatchFeatureCaps::READ_ONLY_RESOURCES
        .union(PcuDispatchFeatureCaps::MUTABLE_RESOURCES);
    let unchecked = session
        .prepare_dispatch_on_stream(
            unchecked_ir,
            fusion_pcu::PcuInvocationShape::invocations(core::num::NonZeroU32::new(1).unwrap()),
            &prepared.stream_handle(),
        )
        .unwrap();
    assert!(!unchecked.requires_checked_fault_word());
    session
        .retain_guarded_dispatch(&unchecked, &unchecked_ir)
        .unwrap();
    let mut third = runtime.allocate(4).unwrap();
    third.copy_from(&23_f32.to_le_bytes()).unwrap();
    let c = unchecked_ir
        .bindings
        .iter()
        .zip([first.clone(), third.clone()])
        .map(|(binding, resource)| {
            fusion_pcu::PcuOwnedBinding::new(
                binding.reference(),
                session.device_identity(),
                4,
                binding.access,
                binding.binding_type,
                resource,
            )
        })
        .collect::<Vec<_>>();
    unchecked.validate_guarded_bindings(&c).unwrap();
    let slab = runtime.allocate(48).unwrap();
    let mut initial = Vec::new();
    for _ in 0..3 {
        initial.extend_from_slice(&u64::MAX.to_le_bytes());
        initial.extend_from_slice(&0u64.to_le_bytes());
    }
    let mut batch =
        crate::guarded_batch::GuardedBatch::new(&prepared.stream_handle(), 3, 48).unwrap();
    batch.initialize(&slab, Rc::new(initial)).unwrap();
    prepared
        .submit_guarded_into_batch(&a, batch.batch(), &slab, 0)
        .unwrap();
    prepared
        .submit_guarded_into_batch(&b, batch.batch(), &slab, 1)
        .unwrap();
    unchecked
        .submit_guarded_into_batch(&c, batch.batch(), &slab, 2)
        .unwrap();
    batch.queue_readback(&slab, 48).unwrap();
    let bytes = batch.complete().unwrap();
    let records = bytes
        .as_chunks::<16>()
        .0
        .iter()
        .map(|record| {
            (
                u64::from_le_bytes(record[..8].try_into().unwrap()),
                u64::from_le_bytes(record[8..].try_into().unwrap()),
            )
        })
        .collect::<Vec<_>>();
    assert!(matches!(
        prepared.decode_guarded_record(records[0].0, records[0].1),
        Ok(fusion_pcu::PcuGuardedExecutionStageOutcome::Fault(
            fusion_pcu::PcuExecutionFault {
                recovered: true,
                ..
            }
        ))
    ));
    assert_eq!(records[1], (u64::MAX, 2));
    assert_eq!(records[2], (u64::MAX, 2));
    batch.reset_after_terminal().unwrap();
    let mut bytes = [0; 4];
    first.copy_to(&mut bytes).unwrap();
    assert_eq!(u32::from_le_bytes(bytes), f32::MAX.to_bits());
    second.copy_to(&mut bytes).unwrap();
    assert_eq!(u32::from_le_bytes(bytes), 19_f32.to_bits());
    third.copy_to(&mut bytes).unwrap();
    assert_eq!(u32::from_le_bytes(bytes), 23_f32.to_bits());
}

#[test]
#[ignore = "requires native device; checked F32/F64 guarded success, overflow ordering and retry"]
#[allow(clippy::too_many_lines)] // Exercise both native float formats through the same bounded contract.
fn guarded_float_formats_preserve_success_fault_order_and_retry() {
    let (_discovery, session) = tensor_tests::cuda_test_session();
    let assessor = CudaTensorAssessor::new(&session).unwrap();
    let pool = PcuMemoryPoolId(0x4755_0103);
    macro_rules! check_format {
        ($scalar:ty, $first_overflow:expr, $second_overflow:expr) => {{
            let mut graph = Graph::default();
            let input = graph.input_typed::<$scalar>([65]).unwrap();
            let first = graph.add_typed(input, input).unwrap();
            let output = graph.mul_typed(first, input).unwrap();
            let program = graph
                .into_selected_program(
                    &[output.erase()],
                    TensorArithmeticRewritePolicy::Disabled,
                    TensorArithmeticCapability::Strict,
                    TensorPointwiseGroupingPolicy::Disabled,
                )
                .unwrap();
            let mut prepared = assessor.prepare_owned_program(program).unwrap();
            assessor.retain_owned_program_fixed_dispatches(&mut prepared).unwrap();
            assert!(assessor.prepare_owned_program_guarded_execution(&mut prepared).unwrap());
            let mut memory = session.memory_provider(pool);
            let valid = fusion_pcu::PcuDeviceTensor::new(
                [65], session.upload_buffer(pool, &[<$scalar>::from(2_u8); 65]).unwrap(),
            ).unwrap();
            let mut escaped: Option<fusion_pcu::PcuDeviceTensor<$scalar, crate::CudaMemoryResource>> = None;
            for expected_stage in [None, Some(0_usize), None, Some(1), None] {
                let bad = expected_stage.map(|stage| {
                    let value: $scalar = if stage == 0 { $first_overflow } else { $second_overflow };
                    assert!(value.is_finite());
                    fusion_pcu::PcuDeviceTensor::new(
                        [65], session.upload_buffer(pool, &[value; 65]).unwrap(),
                    ).unwrap()
                });
                let owner = bad.as_ref().unwrap_or(&valid);
                #[cfg(feature = "allocation-census")]
                let before = crate::cuda_api_census();
                let result = assessor.execute_owned_program_outputs(
                    &prepared, &[(input.erase(), owner)], pool, &mut memory,
                );
                #[cfg(feature = "allocation-census")]
                {
                    let after = crate::cuda_api_census();
                    assert_eq!(after.guarded_chain_submissions - before.guarded_chain_submissions, 1);
                    assert_eq!(after.guarded_kernel_launches - before.guarded_kernel_launches, 2);
                }
                if expected_stage.is_some() {
                    // The error variant carries no owner, including when the final kernel faults.
                    assert!(matches!(result, Err(CudaTensorExecutionError::ExecutionFault(fault)) if fault.invocation_id == 0 && !fault.recovered));
                } else {
                    let output = result.unwrap().pop().unwrap().1;
                    let mut observed = [<$scalar>::default(); 65];
                    session.download_buffer(pool, output.buffer(), &mut observed).unwrap();
                    assert_eq!(observed.map(<$scalar>::to_bits), [ <$scalar>::from(8_u8).to_bits(); 65]);
                    if let Some(previous) = escaped.as_ref() {
                        assert!(!output.buffer().resource().may_overlap(previous.buffer().resource()));
                    }
                    escaped = Some(output);
                }
                let view = prepared.view();
                let bank = prepared.scratch.bind(session.tensor_runtime(), &view, pool, &mut memory).unwrap();
                let mut bytes = [0_u8; 32];
                bank.guarded.as_ref().unwrap().resource.device_buffer().copy_to(&mut bytes).unwrap();
                let records: [(u64, u64); 2] = std::array::from_fn(|stage| {
                    let record = &bytes[stage * 16..(stage + 1) * 16];
                    (u64::from_le_bytes(record[..8].try_into().unwrap()), u64::from_le_bytes(record[8..].try_into().unwrap()))
                });
                match expected_stage {
                    None => assert_eq!(records, [(u64::MAX, 1); 2]),
                    Some(0) => { assert_ne!(records[0].0, u64::MAX); assert_eq!(records[0].1, 1); assert_eq!(records[1], (u64::MAX, 2)); }
                    Some(1) => { assert_eq!(records[0], (u64::MAX, 1)); assert_ne!(records[1].0, u64::MAX); assert_eq!(records[1].1, 1); }
                    Some(_) => unreachable!(),
                }
                drop(bank);
                #[cfg(feature = "insights")]
                {
                    let report = prepared.last_guarded_execution_report(pool).unwrap();
                    assert_eq!(report.stage_count, 2);
                    assert_eq!(&report.values[..2], &[Some(first.erase()), Some(output.erase())]);
                    match (expected_stage, report.outcome) {
                        (None, fusion_pcu::PcuGuardedExecutionOutcome::Succeeded) => {}
                        (Some(expected), fusion_pcu::PcuGuardedExecutionOutcome::Fault { stage, .. }) => assert_eq!(stage, expected),
                        _ => panic!("float report differs from physical execution"),
                    }
                    if expected_stage == Some(0) {
                        assert_eq!(report.stages[1], Some(fusion_pcu::PcuGuardedExecutionStageOutcome::Skipped));
                    }
                }
            }
        }};
    }
    check_format!(f32, f32::from_bits(254 << 23), f32::from_bits(191 << 23));
    check_format!(f64, f64::from_bits(2046 << 52), f64::from_bits(1535 << 52));
}
