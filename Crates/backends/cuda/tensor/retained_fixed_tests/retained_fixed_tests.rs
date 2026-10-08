//! Retained fixed kernels are immutable owners, independent of FIFO residency and status banks.
use super::*;
use fusion_pcu::PcuDeviceTensor;

fn program() -> (
    ValueId,
    fusion_pcu::dialect::tensor::TensorOwnedSelectedProgram,
) {
    let mut graph = Graph::default();
    let input = graph.input_typed::<i32>([65]).unwrap();
    let output = graph.add_typed(input, input).unwrap();
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
#[ignore = "requires native device; retain more fixed kernels than the FIFO can own"]
fn retained_fixed_exceeds_fifo_capacity_and_releases_module() {
    let (_discovery, session) = tests::cuda_test_session();
    let assessor = CudaTensorAssessor::new(&session).unwrap();
    let pool = PcuMemoryPoolId(0x5254_0101);
    let mut graph = Graph::default();
    let mut input_values = Vec::new();
    let mut outputs = Vec::new();
    for n in 1..=ADD_DISPATCH_CACHE_CAPACITY + 1 {
        let input = graph.input_typed::<i32>([n]).unwrap();
        outputs.push(graph.add_typed(input, input).unwrap().erase());
        input_values.push(input.erase());
    }
    let program = graph
        .into_selected_program(
            &outputs,
            TensorArithmeticRewritePolicy::Disabled,
            TensorArithmeticCapability::Strict,
            TensorPointwiseGroupingPolicy::Disabled,
        )
        .unwrap();
    let mut prepared = assessor.prepare_owned_program(program).unwrap();
    assert!(prepared.retained_fixed.is_empty());
    assessor
        .retain_owned_program_fixed_dispatches(&mut prepared)
        .unwrap();
    let fixed = prepared
        .data
        .fixed_dispatches
        .iter()
        .flatten()
        .collect::<Vec<_>>();
    assert_eq!(fixed.len(), ADD_DISPATCH_CACHE_CAPACITY + 1);
    assert!(
        prepared
            .retained_fixed
            .iter()
            .flatten()
            .all(|retained| retained.uses_stream(&assessor.state().stream))
    );
    let first = prepared.retained_fixed.iter().flatten().next().unwrap();
    let executable = Rc::downgrade(first);
    let kernel = first.cuda_kernel();
    let module = Rc::downgrade(&kernel.module);
    drop(kernel);
    assert!(
        !assessor
            .state()
            .add_dispatches
            .borrow()
            .iter()
            .any(|(key, _)| *key == fixed[0].cache_key)
    );
    assert_eq!(
        assessor.state().add_dispatches.borrow().len(),
        ADD_DISPATCH_CACHE_CAPACITY
    );
    drop(fixed);
    let mut memory = session.memory_provider(pool);
    for generation in [3_i32, 7] {
        let owners = (1..=ADD_DISPATCH_CACHE_CAPACITY + 1)
            .map(|n| {
                PcuDeviceTensor::new(
                    [n],
                    session.upload_buffer(pool, &vec![generation; n]).unwrap(),
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
        let inputs = input_values
            .iter()
            .copied()
            .zip(&owners)
            .collect::<Vec<_>>();
        #[cfg(feature = "allocation-census")]
        let before = crate::cuda_api_census();
        // A held mutable cache borrow proves retained execution does not even borrow the cache.
        let cache = assessor.state().add_dispatches.borrow_mut();
        let observed = assessor
            .execute_owned_program_outputs(&prepared, &inputs, pool, &mut memory)
            .unwrap();
        drop(cache);
        #[cfg(feature = "allocation-census")]
        assert_eq!(crate::cuda_api_census().module_loads, before.module_loads);
        for (n, (_, output)) in observed.iter().enumerate() {
            let mut words = vec![0_i32; n + 1];
            session
                .download_buffer(pool, output.buffer(), &mut words)
                .unwrap();
            assert_eq!(words, vec![generation * 2; n + 1]);
        }
    }
    assert!(executable.upgrade().is_some());
    assert!(module.upgrade().is_some());
    drop(prepared);
    assert!(executable.upgrade().is_none());
    assert!(
        module.upgrade().is_none(),
        "evicted executable's module releases with its final prepared owner"
    );
}

#[test]
#[ignore = "requires native device; exact stream fast route, foreign stream/session fallback and fault retry"]
fn retained_fixed_replay_preserves_stream_session_and_fault_gates() {
    let (_discovery, session) = tests::cuda_test_session();
    let a = CudaTensorAssessor::new(&session).unwrap();
    let b = CudaTensorAssessor::new(&session).unwrap();
    let pool = PcuMemoryPoolId(0x5254_0102);
    let (input, program) = program();
    let mut prepared = a.prepare_owned_program(program).unwrap();
    a.retain_owned_program_fixed_dispatches(&mut prepared)
        .unwrap();
    let retained = prepared.retained_fixed.iter().flatten().next().unwrap();
    assert!(retained.uses_stream(&a.state().stream));
    assert!(!retained.uses_stream(&b.state().stream));
    let identity = Rc::as_ptr(retained);
    a.state().add_dispatches.borrow_mut().clear();
    let owner =
        PcuDeviceTensor::new([65], session.upload_buffer(pool, &[4_i32; 65]).unwrap()).unwrap();
    let invalid =
        PcuDeviceTensor::new([65], session.upload_buffer(pool, &[i32::MAX; 65]).unwrap()).unwrap();
    let short =
        PcuDeviceTensor::new([64], session.upload_buffer(pool, &[1_i32; 64]).unwrap()).unwrap();
    let mut memory = session.memory_provider(pool);
    assert!(
        a.execute_owned_program_outputs(&prepared, &[(input, &short)], pool, &mut memory)
            .is_err()
    );
    assert!(matches!(
        a.execute_owned_program_outputs(&prepared, &[(input, &invalid)], pool, &mut memory),
        Err(CudaTensorExecutionError::ExecutionFault(_))
    ));
    let cache = a.state().add_dispatches.borrow_mut();
    let output = a
        .execute_owned_program_outputs(&prepared, &[(input, &owner)], pool, &mut memory)
        .unwrap()
        .pop()
        .unwrap()
        .1;
    drop(cache);
    let mut values = [0_i32; 65];
    session
        .download_buffer(pool, output.buffer(), &mut values)
        .unwrap();
    assert_eq!(values, [8; 65]);
    drop(output);
    let other_output = b
        .execute_owned_program_outputs(&prepared, &[(input, &owner)], pool, &mut memory)
        .unwrap()
        .pop()
        .unwrap()
        .1;
    session
        .download_buffer(pool, other_output.buffer(), &mut values)
        .unwrap();
    assert_eq!(values, [8; 65]);
    assert_eq!(b.state().add_dispatches.borrow().len(), 1);
    assert_eq!(Rc::as_ptr(retained), identity);
    let (_other_discovery, other_session) = tests::cuda_test_session();
    let other = CudaTensorAssessor::new(&other_session).unwrap();
    let other_owner = PcuDeviceTensor::new(
        [65],
        other_session.upload_buffer(pool, &[5_i32; 65]).unwrap(),
    )
    .unwrap();
    let mut other_memory = other_session.memory_provider(pool);
    let output = other
        .execute_owned_program_outputs(&prepared, &[(input, &other_owner)], pool, &mut other_memory)
        .unwrap()
        .pop()
        .unwrap()
        .1;
    other_session
        .download_buffer(pool, output.buffer(), &mut values)
        .unwrap();
    assert_eq!(values, [10; 65]);
    let descriptor = other.borrow_device_input_ref(&other_owner, pool).unwrap();
    assert!(
        a.execute_owned_program_output_from_inputs::<i32, _>(
            &prepared,
            &[(input, &descriptor)],
            pool,
            &mut memory
        )
        .is_err()
    );
    b.retain_owned_program_fixed_dispatches(&mut prepared)
        .unwrap();
    assert!(
        prepared
            .retained_fixed
            .iter()
            .flatten()
            .all(|retained| retained.uses_stream(&b.state().stream))
    );
}

#[test]
#[ignore = "requires native device; prepared executable owns stream and module after assessor drop"]
fn retained_fixed_outlives_assessor_without_a_session_cycle() {
    let (_discovery, session) = tests::cuda_test_session();
    let (_, program) = program();
    let (prepared, weak) = {
        let assessor = CudaTensorAssessor::new(&session).unwrap();
        let mut prepared = assessor.prepare_owned_program(program).unwrap();
        assessor
            .retain_owned_program_fixed_dispatches(&mut prepared)
            .unwrap();
        let retained = prepared.retained_fixed.iter().flatten().next().unwrap();
        let weak = Rc::downgrade(retained);
        (prepared, weak)
    };
    assert_eq!(weak.strong_count(), 1);
    drop(prepared);
    assert!(weak.upgrade().is_none());
}
