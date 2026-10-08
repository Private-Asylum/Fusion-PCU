use crate::tensor::*;
use fusion_pcu::PcuDeviceTensor;

#[test]
#[ignore = "requires native device; guarded fault ordering, skipped effects, retry and stream fallback"]
#[allow(clippy::too_many_lines)]
fn guarded_chain_preserves_fault_order_skips_effects_and_retries() {
    let (_discovery, session) = tests::rocm_test_session();
    let a = RocmTensorAssessor::new(&session).unwrap();
    let b = RocmTensorAssessor::new(&session).unwrap();
    let pool = PcuMemoryPoolId(0x4755_0101);
    let mut graph = Graph::default();
    let input = graph.input_typed::<i32>([65]).unwrap();
    let first = graph.add_typed(input, input).unwrap();
    let second = graph.mul_typed(first, input).unwrap();
    let output = graph.add_typed(second, input).unwrap();
    let program = graph
        .into_selected_program(
            &[output.erase()],
            TensorArithmeticRewritePolicy::Disabled,
            TensorArithmeticCapability::Strict,
            TensorPointwiseGroupingPolicy::Disabled,
        )
        .unwrap();
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
    let mut memory = session.memory_provider(pool);
    // Stage 0 faults at lane 7. Stage 1 would fault at lane 0 if it ran.
    let mut invalid = [2_i32; 65];
    invalid[0] = 50_000;
    invalid[7] = i32::MAX;
    let invalid =
        PcuDeviceTensor::new([65], session.upload_buffer(pool, &invalid).unwrap()).unwrap();
    let valid =
        PcuDeviceTensor::new([65], session.upload_buffer(pool, &[3_i32; 65]).unwrap()).unwrap();
    let canary = 0x1234_5678_i32;
    let canary_bytes = bytemuck::cast_slice(&[canary; 65]).to_vec();
    let second_resource = {
        let view = prepared.view();
        let bank = prepared
            .scratch
            .bind(session.tensor_runtime(), &view, pool, &mut memory)
            .unwrap();
        let resource = bank.resources[view.index_of(second.erase()).unwrap()]
            .as_ref()
            .unwrap();
        let mut buffer = resource.device_buffer().clone();
        buffer.copy_from(&canary_bytes).unwrap();
        let alias = RocmTensorInput {
            session: &session,
            shape: PcuOwnedShape::from_slice(&[65]),
            resource: resource.clone_for_tensor_input(),
            scalar_type: fusion_pcu::PcuScalarType::I32,
            updateable: true,
        }
        .into_device_tensor::<i32>()
        .unwrap();
        (buffer, alias)
    };
    let (second_resource, alias) = second_resource;
    #[cfg(feature = "allocation-census")]
    let preflight = crate::rocm_api_census();
    let short =
        PcuDeviceTensor::new([64], session.upload_buffer(pool, &[1_i32; 64]).unwrap()).unwrap();
    assert!(
        a.execute_owned_program_outputs(&prepared, &[(input.erase(), &short)], pool, &mut memory)
            .is_err()
    );
    assert!(matches!(
        a.execute_owned_program_outputs(&prepared, &[(input.erase(), &alias)], pool, &mut memory),
        Err(RocmTensorExecutionError::ScratchMismatch)
    ));
    #[cfg(feature = "allocation-census")]
    {
        let after = crate::rocm_api_census();
        assert_eq!(
            after.guarded_chain_submissions,
            preflight.guarded_chain_submissions
        );
        assert_eq!(after.kernel_launches, preflight.kernel_launches);
        assert_eq!(after.event_waits, preflight.event_waits);
    }
    a.state().add_dispatches.borrow_mut().clear();
    for generation in 0..3 {
        #[cfg(feature = "allocation-census")]
        let before = crate::rocm_api_census();
        let cache = a.state().add_dispatches.borrow_mut();
        let result = a.execute_owned_program_outputs(
            &prepared,
            &[(input.erase(), &invalid)],
            pool,
            &mut memory,
        );
        drop(cache);
        assert!(
            matches!(result, Err(RocmTensorExecutionError::ExecutionFault(fault)) if fault.invocation_id == 7)
        );
        {
            let view = prepared.view();
            let bank = prepared
                .scratch
                .bind(session.tensor_runtime(), &view, pool, &mut memory)
                .unwrap();
            let bytes = bank
                .guarded
                .as_ref()
                .unwrap()
                .batch
                .retained_readback_bytes();
            let words = bytes
                .as_chunks::<8>()
                .0
                .iter()
                .map(|word| u64::from_le_bytes(*word))
                .collect::<Vec<_>>();
            assert_eq!(words, [(7 << 3) | 3, 1, u64::MAX, 2, u64::MAX, 2]);
        }
        #[cfg(feature = "insights")]
        {
            let report = prepared.last_guarded_execution_report(pool).unwrap();
            assert_eq!(report.stage_count, 3);
            assert_eq!(
                &report.values[..3],
                &[
                    Some(first.erase()),
                    Some(second.erase()),
                    Some(output.erase())
                ]
            );
            assert!(
                matches!(report.outcome, fusion_pcu::PcuGuardedExecutionOutcome::Fault { stage: 0, fault } if fault.invocation_id == 7)
            );
            assert!(matches!(
                report.stages[0],
                Some(fusion_pcu::PcuGuardedExecutionStageOutcome::Fault(_))
            ));
            assert_eq!(
                report.stages[1],
                Some(fusion_pcu::PcuGuardedExecutionStageOutcome::Skipped)
            );
            assert_eq!(
                report.stages[2],
                Some(fusion_pcu::PcuGuardedExecutionStageOutcome::Skipped)
            );
        }
        let mut bytes = vec![0; canary_bytes.len()];
        second_resource.copy_to(&mut bytes).unwrap();
        assert_eq!(
            bytes, canary_bytes,
            "blocked successor must not store user output"
        );
        #[cfg(feature = "allocation-census")]
        {
            let after = crate::rocm_api_census();
            assert_eq!(
                after.guarded_chain_submissions - before.guarded_chain_submissions,
                1
            );
            assert_eq!(
                after.guarded_kernel_launches - before.guarded_kernel_launches,
                3
            );
            assert_eq!(after.event_waits - before.event_waits, 1);
            assert_eq!(after.module_loads, before.module_loads);
        }
        #[cfg(feature = "allocation-census")]
        let success_before = crate::rocm_api_census();
        let output = a
            .execute_owned_program_outputs(&prepared, &[(input.erase(), &valid)], pool, &mut memory)
            .unwrap()
            .pop()
            .unwrap()
            .1;
        #[cfg(feature = "allocation-census")]
        {
            let after = crate::rocm_api_census();
            // Fresh output allocation and the callback-free private submission each select once.
            assert_eq!(after.allocations - success_before.allocations, 1);
            assert_eq!(
                after.device_selections - success_before.device_selections,
                2
            );
        }
        let mut values = [0_i32; 65];
        session
            .download_buffer(pool, output.buffer(), &mut values)
            .unwrap();
        assert_eq!(values, [21; 65]);
        #[cfg(feature = "insights")]
        {
            let report = prepared.last_guarded_execution_report(pool).unwrap();
            assert_eq!(
                report.outcome,
                fusion_pcu::PcuGuardedExecutionOutcome::Succeeded
            );
            assert!(report.stages[..3].iter().all(
                |stage| *stage == Some(fusion_pcu::PcuGuardedExecutionStageOutcome::Succeeded)
            ));
        }
        // Re-arm canary after successful work to prove every failed retry skips stores again.
        if generation < 2 {
            second_resource.clone().copy_from(&canary_bytes).unwrap();
        }
    }
    #[cfg(feature = "insights")]
    assert!(prepared.last_guarded_execution_report(pool).is_some());
    assert!(
        a.execute_owned_program_outputs(&prepared, &[(input.erase(), &short)], pool, &mut memory)
            .is_err()
    );
    #[cfg(feature = "insights")]
    assert!(prepared.last_guarded_execution_report(pool).is_none());
    #[cfg(feature = "allocation-census")]
    let before = crate::rocm_api_census();
    b.execute_owned_program_outputs(&prepared, &[(input.erase(), &valid)], pool, &mut memory)
        .unwrap();
    #[cfg(feature = "allocation-census")]
    assert_eq!(
        crate::rocm_api_census().guarded_chain_submissions,
        before.guarded_chain_submissions
    );
    b.retain_owned_program_fixed_dispatches(&mut prepared)
        .unwrap();
    assert!(prepared.guarded.is_none());
    a.execute_owned_program_outputs(&prepared, &[(input.erase(), &valid)], pool, &mut memory)
        .unwrap();
    assert!(
        b.prepare_owned_program_guarded_execution(&mut prepared)
            .unwrap()
    );
    b.execute_owned_program_outputs(&prepared, &[(input.erase(), &valid)], pool, &mut memory)
        .unwrap();
}

#[test]
#[ignore = "requires native device; recovered status blocks checked and unchecked successor effects"]
#[allow(clippy::too_many_lines)]
fn recovered_guarded_fault_blocks_checked_and_unchecked_successors() {
    use fusion_pcu::{PcuDispatchDataOp, PcuDispatchOp, PcuRangePolicy};
    let (_discovery, session) = tests::rocm_test_session();
    let assessor = RocmTensorAssessor::new(&session).unwrap();
    let mut graph = Graph::default();
    let input = graph.input_typed::<i32>([65]).unwrap();
    let add = graph.add_typed(input, input).unwrap();
    let program = graph
        .into_selected_program(
            &[add.erase()],
            TensorArithmeticRewritePolicy::Disabled,
            TensorArithmeticCapability::Strict,
            TensorPointwiseGroupingPolicy::Disabled,
        )
        .unwrap();
    let prepared = assessor.prepare_owned_program(program).unwrap();
    let view = prepared.view();
    let fixed = view
        .fixed_dispatch(view.index_of(add.erase()).unwrap())
        .unwrap();
    let mut body = match fixed.kernel.ops {
        [PcuDispatchOp::GridStrideLoop { body, .. }, _] => body.to_vec(),
        ops => ops.to_vec(),
    };
    let mut changed = false;
    for operation in &mut body {
        if let PcuDispatchOp::Data(PcuDispatchDataOp::CheckedIntegerBinary {
            range_policy, ..
        }) = operation
        {
            *range_policy = PcuRangePolicy::Clamp;
            changed = true;
        }
    }
    assert!(changed);
    let ops = match fixed.kernel.ops {
        [PcuDispatchOp::GridStrideLoop { extent, .. }, terminal] => vec![
            PcuDispatchOp::GridStrideLoop {
                extent: *extent,
                body: &body,
            },
            *terminal,
        ],
        _ => body.clone(),
    };
    let mut kernel = PcuDispatchKernelIr {
        ops: &ops,
        ..fixed.kernel
    };
    kernel.numerical_requirements.range_policy = PcuRangePolicy::Clamp;
    kernel.feature_caps = kernel
        .feature_caps
        .union(fusion_pcu::PcuDispatchFeatureCaps::RANGE_CLAMP);
    let stream = &assessor.state().stream;
    let checked = session
        .prepare_dispatch_on_stream(kernel, fixed.invocation_shape, stream)
        .unwrap();
    session.retain_guarded_dispatch(&checked, &kernel).unwrap();
    let unchecked_bindings = [
        *fixed
            .kernel
            .bindings
            .iter()
            .find(|binding| binding.reference() == fixed.left_binding)
            .unwrap(),
        *fixed
            .kernel
            .bindings
            .iter()
            .find(|binding| binding.reference() == fixed.output_binding)
            .unwrap(),
    ];
    let unchecked_ops = [
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
            result: fusion_pcu::PcuDispatchValueId(0),
            binding: fixed.left_binding,
            index: fusion_pcu::PcuDispatchIndex::InvocationId,
        }),
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
            binding: fixed.output_binding,
            index: fusion_pcu::PcuDispatchIndex::InvocationId,
            value: fusion_pcu::PcuDispatchValueId(0),
        }),
        PcuDispatchOp::Control(fusion_pcu::PcuDispatchControlOp::Return),
    ];
    let unchecked_kernel = PcuDispatchKernelIr {
        ops: &unchecked_ops,
        bindings: &unchecked_bindings,
        feature_caps: fusion_pcu::PcuDispatchFeatureCaps::READ_ONLY_RESOURCES
            .union(fusion_pcu::PcuDispatchFeatureCaps::MUTABLE_RESOURCES),
        numerical_requirements: PcuDispatchKernelIr::DEFAULT_REQUIREMENTS,
        ..fixed.kernel
    };
    assert!(!crate::owned_dispatch::kernel_uses_checked_arithmetic(
        &unchecked_kernel
    ));
    let unchecked = session
        .prepare_dispatch_on_stream(unchecked_kernel, fixed.invocation_shape, stream)
        .unwrap();
    session
        .retain_guarded_dispatch(&unchecked, &unchecked_kernel)
        .unwrap();
    let runtime = session.tensor_runtime();
    let mut input = runtime.allocate(65 * 4).unwrap();
    input
        .copy_from(bytemuck::cast_slice(&[i32::MAX; 65]))
        .unwrap();
    let mut first = runtime.allocate(65 * 4).unwrap();
    let mut second = runtime.allocate(65 * 4).unwrap();
    let mut third = runtime.allocate(65 * 4).unwrap();
    let canary = bytemuck::cast_slice(&[0x1234_5678_i32; 65]).to_vec();
    first.copy_from(&canary).unwrap();
    second.copy_from(&canary).unwrap();
    third.copy_from(&canary).unwrap();
    let binding = |target, access, buffer: &crate::DeviceBuffer| {
        session
            .binding(
                target,
                access,
                PcuBindingType::Value(fixed.value_type),
                buffer.clone(),
            )
            .unwrap()
    };
    let checked_bindings = |output: &crate::DeviceBuffer| {
        vec![
            binding(fixed.left_binding, PcuBindingAccess::ReadOnly, &input),
            binding(
                fixed.right_binding.unwrap(),
                PcuBindingAccess::ReadOnly,
                &input,
            ),
            binding(fixed.output_binding, PcuBindingAccess::WriteOnly, output),
        ]
    };
    let a = checked_bindings(&first);
    let b = checked_bindings(&second);
    let c = vec![
        binding(fixed.left_binding, PcuBindingAccess::ReadOnly, &input),
        binding(fixed.output_binding, PcuBindingAccess::WriteOnly, &third),
    ];
    checked.validate_guarded_bindings(&a).unwrap();
    checked.validate_guarded_bindings(&b).unwrap();
    unchecked.validate_guarded_bindings(&c).unwrap();
    let slab = runtime.allocate(48).unwrap();
    let initial = Rc::new(
        [u64::MAX, 0, u64::MAX, 0, u64::MAX, 0]
            .into_iter()
            .flat_map(u64::to_le_bytes)
            .collect::<Vec<_>>(),
    );
    let mut batch = crate::guarded_batch::GuardedBatch::new(stream, 3, 48).unwrap();
    batch.initialize(&slab, Rc::clone(&initial)).unwrap();
    checked
        .submit_guarded_into_batch(&a, batch.batch(), &slab, 0)
        .unwrap();
    checked
        .submit_guarded_into_batch(&b, batch.batch(), &slab, 1)
        .unwrap();
    unchecked
        .submit_guarded_into_batch(&c, batch.batch(), &slab, 2)
        .unwrap();
    batch.queue_readback(&slab, 48).unwrap();
    let bytes = batch.complete().unwrap();
    let words = bytes
        .as_chunks::<8>()
        .0
        .iter()
        .map(|bytes| u64::from_le_bytes(*bytes))
        .collect::<Vec<_>>();
    assert_eq!(words, [(1 << 63) | 3, 1, u64::MAX, 2, u64::MAX, 2]);
    assert!(
        matches!(checked.decode_guarded_record(words[0], words[1]).unwrap(),
        fusion_pcu::PcuGuardedExecutionStageOutcome::Fault(fault) if fault.recovered)
    );
    batch.reset_after_terminal().unwrap();
    let mut observed = vec![0; canary.len()];
    first.copy_to(&mut observed).unwrap();
    assert_ne!(
        observed, canary,
        "recovered first stage executed its clamped stores"
    );
    second.copy_to(&mut observed).unwrap();
    assert_eq!(observed, canary);
    third.copy_to(&mut observed).unwrap();
    assert_eq!(observed, canary);
    // A post-enqueue operational refusal still holds leases until stream quiescence.
    let foreign_stream = runtime.create_stream().unwrap();
    let foreign = session
        .prepare_dispatch_on_stream(unchecked_kernel, fixed.invocation_shape, &foreign_stream)
        .unwrap();
    session
        .retain_guarded_dispatch(&foreign, &unchecked_kernel)
        .unwrap();
    batch.initialize(&slab, Rc::clone(&initial)).unwrap();
    checked
        .submit_guarded_into_batch(&a, batch.batch(), &slab, 0)
        .unwrap();
    assert!(
        foreign
            .submit_guarded_into_batch(&c, batch.batch(), &slab, 1)
            .is_err()
    );
    assert!(first.copy_to(&mut observed).is_err());
    batch.reset_after_terminal().unwrap();
    first.copy_to(&mut observed).unwrap();
    batch.initialize(&slab, initial).unwrap();
    unchecked
        .submit_guarded_into_batch(&c, batch.batch(), &slab, 0)
        .unwrap();
    batch.queue_readback(&slab, 48).unwrap();
    let bytes = batch.complete().unwrap();
    let word = u64::from_le_bytes(bytes[..8].try_into().unwrap());
    let disposition = u64::from_le_bytes(bytes[8..16].try_into().unwrap());
    assert_eq!(
        unchecked.decode_guarded_record(word, disposition).unwrap(),
        fusion_pcu::PcuGuardedExecutionStageOutcome::Succeeded
    );
    batch.reset_after_terminal().unwrap();
}

#[test]
#[ignore = "requires native device; checked F32/F64 guarded success, overflow ordering and retry"]
#[allow(clippy::too_many_lines)] // Exercise both native float formats through the same bounded contract.
fn guarded_float_formats_preserve_success_fault_order_and_retry() {
    let (_discovery, session) = tests::rocm_test_session();
    let assessor = RocmTensorAssessor::new(&session).unwrap();
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
            let mut escaped: Option<fusion_pcu::PcuDeviceTensor<$scalar, crate::RocmMemoryResource>> = None;
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
                let before = crate::rocm_api_census();
                let result = assessor.execute_owned_program_outputs(
                    &prepared, &[(input.erase(), owner)], pool, &mut memory,
                );
                #[cfg(feature = "allocation-census")]
                {
                    let after = crate::rocm_api_census();
                    assert_eq!(after.guarded_chain_submissions - before.guarded_chain_submissions, 1);
                    assert_eq!(after.guarded_kernel_launches - before.guarded_kernel_launches, 2);
                }
                if expected_stage.is_some() {
                    // The error variant carries no owner, including when the final kernel faults.
                    assert!(matches!(result, Err(RocmTensorExecutionError::ExecutionFault(fault)) if fault.invocation_id == 0 && !fault.recovered));
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
