#[rustfmt::skip]
use super::{
    CudaTensorAssessor,
    CudaTensorExecutionError,
    Graph,
    PcuDeviceTensor,
    PcuMemoryPoolId,
    TensorArithmeticCapability,
    TensorArithmeticRewritePolicy,
    TensorPointwiseGroupingPolicy,
    cuda_test_session,
    prepare,
};
use super::super::CudaHostedTensorInput;

#[test]
#[ignore = "requires native CUDA; production aggregate private upload changing-input ownership"]
#[expect(
    clippy::too_many_lines,
    reason = "Retain the same prepared schedule and endpoint across cold refusal, warm generations and identity checks."
)]
fn aggregate_staging_changed_inputs_and_cold_foreign_identity_refusals() {
    let (_discovery, session) = cuda_test_session();
    let pool = PcuMemoryPoolId(0x4352_0193);
    let assessor = CudaTensorAssessor::new(&session).unwrap();
    let foreign = CudaTensorAssessor::new(&session).unwrap();
    let mut memory = session.memory_provider(pool);
    for n in [65, 4096] {
        let shape = [n];
        let (input, mut prepared) = prepare(&assessor, n, true);
        let owner =
            PcuDeviceTensor::new(shape, session.upload_buffer(pool, &vec![0_u32; n]).unwrap())
                .unwrap();
        let mut bytes = vec![0_u8; n * 4];
        let resource = owner.buffer().resource();
        let binding = [CudaHostedTensorInput {
            value: input,
            resource,
            shape: &shape,
            source: &bytes,
        }];
        #[cfg(feature = "allocation-census")]
        let before = crate::cuda_api_census();
        assert!(
            assessor
                .execute_owned_program_output_from_host_staging::<u32, _, 2>(
                    &prepared,
                    &binding,
                    pool,
                    &mut memory,
                )
                .unwrap()
                .is_none()
        );
        #[cfg(feature = "allocation-census")]
        assert_eq!(before, crate::cuda_api_census());
        drop(
            assessor
                .execute_owned_program_outputs(&prepared, &[(input, &owner)], pool, &mut memory)
                .unwrap(),
        );
        #[cfg(feature = "allocation-census")]
        let before = crate::cuda_api_census();
        assert!(
            foreign
                .execute_owned_program_output_from_host_staging::<u32, _, 2>(
                    &prepared,
                    &binding,
                    pool,
                    &mut memory,
                )
                .unwrap()
                .is_none()
        );
        #[cfg(feature = "allocation-census")]
        assert_eq!(before, crate::cuda_api_census());
        assessor
            .retain_owned_program_fixed_dispatches(&mut prepared)
            .unwrap();
        assessor.state().add_dispatches.borrow_mut().clear();
        #[cfg(feature = "allocation-census")]
        let before = crate::cuda_api_census();
        assert!(
            foreign
                .execute_owned_program_output_from_host_staging::<u32, _, 2>(
                    &prepared,
                    &binding,
                    pool,
                    &mut memory,
                )
                .unwrap()
                .is_none()
        );
        #[cfg(feature = "allocation-census")]
        assert_eq!(before, crate::cuda_api_census());
        let held_cache = assessor.state().add_dispatches.borrow_mut();
        for generation in 1_u32..=64 {
            for (index, word) in bytes.as_chunks_mut::<4>().0.iter_mut().enumerate() {
                word.copy_from_slice(
                    &(generation + u32::try_from(index % 7).unwrap()).to_le_bytes(),
                );
            }
            let binding = [CudaHostedTensorInput {
                value: input,
                resource,
                shape: &shape,
                source: &bytes,
            }];
            #[cfg(feature = "allocation-census")]
            let before = crate::cuda_api_census();
            let output = assessor
                .execute_owned_program_output_from_host_staging::<u32, _, 2>(
                    &prepared,
                    &binding,
                    pool,
                    &mut memory,
                )
                .unwrap()
                .unwrap();
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
        drop(held_cache);
        // Explicit foreign rebinding preserves the original staging anchor: refuse before enqueue.
        foreign
            .retain_owned_program_fixed_dispatches(&mut prepared)
            .unwrap();
        let binding = [CudaHostedTensorInput {
            value: input,
            resource,
            shape: &shape,
            source: &bytes,
        }];
        #[cfg(feature = "allocation-census")]
        let before = crate::cuda_api_census();
        assert!(
            foreign
                .execute_owned_program_output_from_host_staging::<u32, _, 2>(
                    &prepared,
                    &binding,
                    pool,
                    &mut memory,
                )
                .unwrap()
                .is_none()
        );
        #[cfg(feature = "allocation-census")]
        assert_eq!(before, crate::cuda_api_census());
        let (_, identity) = prepare(&assessor, n, false);
        let binding = [CudaHostedTensorInput {
            value: input,
            resource,
            shape: &shape,
            source: &bytes,
        }];
        #[cfg(feature = "allocation-census")]
        let before = crate::cuda_api_census();
        assert!(
            assessor
                .execute_owned_program_output_from_host_staging::<u32, _, 2>(
                    &identity,
                    &binding,
                    pool,
                    &mut memory,
                )
                .unwrap()
                .is_none()
        );
        #[cfg(feature = "allocation-census")]
        assert_eq!(before, crate::cuda_api_census());
    }
}

#[test]
#[ignore = "requires native CUDA; all-input aggregate preflight, duplicate backing and phase fault"]
#[expect(
    clippy::too_many_lines,
    reason = "Keep later-input refusal and fault/retry ownership in one initialized two-input schedule."
)]
fn aggregate_preflights_every_input_before_enqueue_and_preserves_phase_fault_retry() {
    let (_discovery, session) = cuda_test_session();
    let pool = PcuMemoryPoolId(0x4352_0194);
    let assessor = CudaTensorAssessor::new(&session).unwrap();
    let mut memory = session.memory_provider(pool);
    let mut graph = Graph::default();
    let left = graph.input_typed::<u32>([65]).unwrap();
    let right = graph.input_typed::<u32>([65]).unwrap();
    let factor = graph.uniform_typed([65], 2_u32).unwrap();
    let sum = graph.add_typed(left, right).unwrap();
    let output = graph.mul_typed(sum, factor).unwrap();
    let program = graph
        .into_selected_program(
            &[output.erase()],
            TensorArithmeticRewritePolicy::Disabled,
            TensorArithmeticCapability::Strict,
            TensorPointwiseGroupingPolicy::Disabled,
        )
        .unwrap();
    let mut prepared = assessor.prepare_owned_program(program).unwrap();
    assessor
        .retain_owned_program_fixed_dispatches(&mut prepared)
        .unwrap();
    let left_owner =
        PcuDeviceTensor::new([65], session.upload_buffer(pool, &[3_u32; 65]).unwrap()).unwrap();
    let right_owner =
        PcuDeviceTensor::new([65], session.upload_buffer(pool, &[1_u32; 65]).unwrap()).unwrap();
    let initial = assessor
        .execute_owned_program_outputs(
            &prepared,
            &[(left.erase(), &left_owner), (right.erase(), &right_owner)],
            pool,
            &mut memory,
        )
        .unwrap();
    let left_resource = left_owner.buffer().resource();
    let right_resource = right_owner.buffer().resource();
    let bytes = [3_u32; 65]
        .into_iter()
        .flat_map(u32::to_le_bytes)
        .collect::<Vec<_>>();
    let right_bytes = [1_u32; 65]
        .into_iter()
        .flat_map(u32::to_le_bytes)
        .collect::<Vec<_>>();
    let binding = |value, resource, source| CudaHostedTensorInput {
        value,
        resource,
        shape: &[65],
        source,
    };
    #[cfg(feature = "allocation-census")]
    let before = crate::cuda_api_census();
    let later_bad = [
        binding(left.erase(), left_resource, &bytes),
        binding(right.erase(), right_resource, &right_bytes[..256]),
    ];
    assert!(
        assessor
            .execute_owned_program_output_from_host_staging::<u32, _, 3>(
                &prepared,
                &later_bad,
                pool,
                &mut memory,
            )
            .is_err()
    );
    let aliases = [
        binding(left.erase(), left_resource, &bytes),
        binding(right.erase(), left_resource, &right_bytes),
    ];
    assert!(
        assessor
            .execute_owned_program_output_from_host_staging::<u32, _, 3>(
                &prepared,
                &aliases,
                pool,
                &mut memory,
            )
            .is_err()
    );
    let duplicate_value = [
        binding(left.erase(), left_resource, &bytes),
        binding(left.erase(), right_resource, &right_bytes),
    ];
    assert!(
        assessor
            .execute_owned_program_output_from_host_staging::<u32, _, 3>(
                &prepared,
                &duplicate_value,
                pool,
                &mut memory,
            )
            .is_err()
    );
    assert!(
        assessor
            .execute_owned_program_output_from_host_staging::<u32, _, 0>(
                &prepared,
                &aliases,
                pool,
                &mut memory,
            )
            .is_err()
    );
    assert!(
        assessor
            .execute_owned_program_output_from_host_staging::<u32, _, 3>(
                &prepared,
                &later_bad[..1],
                pool,
                &mut memory,
            )
            .is_err()
    );
    #[cfg(feature = "allocation-census")]
    assert_eq!(before, crate::cuda_api_census());
    let mut untouched = [0_u32; 65];
    session
        .download_buffer(pool, left_owner.buffer(), &mut untouched)
        .unwrap();
    assert_eq!(untouched, [3_u32; 65]);
    let mut bad = [3_u32; 65];
    bad[3] = u32::MAX / 2 + 1;
    bad[7] = u32::MAX;
    let bad = bad
        .into_iter()
        .flat_map(u32::to_le_bytes)
        .collect::<Vec<_>>();
    for (source, fault) in [(&bad, true), (&bytes, false)] {
        let bindings = [
            binding(left.erase(), left_resource, source),
            binding(right.erase(), right_resource, &right_bytes),
        ];
        let terminal = assessor.execute_owned_program_output_from_host_staging::<u32, _, 3>(
            &prepared,
            &bindings,
            pool,
            &mut memory,
        );
        assert!(left_resource.validate_access_available().is_ok());
        assert!(right_resource.validate_access_available().is_ok());
        if fault {
            assert!(
                matches!(terminal.err().unwrap(), CudaTensorExecutionError::ExecutionFault(fault) if fault.invocation_id == 7)
            );
        } else {
            let output = terminal.unwrap().unwrap();
            let mut actual = [99_u32; 66];
            session
                .download_buffer(pool, output.buffer(), &mut actual[..65])
                .unwrap();
            assert_eq!(actual[..65], [8_u32; 65]);
            assert_eq!(actual[65], 99);
        }
    }
    let mut old = [99_u32; 66];
    session
        .download_buffer(pool, initial[0].1.buffer(), &mut old[..65])
        .unwrap();
    assert_eq!(old[..65], [8_u32; 65]);
    assert_eq!(old[65], 99);
}
