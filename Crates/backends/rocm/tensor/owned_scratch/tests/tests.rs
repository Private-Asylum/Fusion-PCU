//! Actual private status admission, terminal transitions and physical-owner uniqueness.
#[path = "memory/memory.rs"]
mod memory;
#[rustfmt::skip]
use fusion_pcu::{
    PcuCompletionOutcome,
    PcuDispatchDataOp,
    PcuDispatchFloatBinaryOp,
    PcuDispatchOp,
    PcuDispatchSubmission,
    PcuFloatUnderflowPolicy,
    PcuInvocationShape,
    PcuMemoryPoolId,
    PcuMemoryResource,
    PcuOwnedBinding,
    PcuOwnedCompletion,
    PcuOwnedDispatchBackend,
    PcuRangePolicy,
    PcuScalarType,
};
use super::Status;

#[test]
#[ignore = "requires native device; recovered/fatal/rejected status reset and sentinel reuse"]
#[allow(clippy::too_many_lines)] // Keep the actual recovered/rejected/reset/success lifecycle in one fixture.
fn private_status_recovered_fault_and_preflight_require_reset() {
    let (_discovery, session) = super::super::tests::rocm_test_session();
    let pool = PcuMemoryPoolId(0x5354_0182);
    let mut memory = memory::Memory {
        provider: session.memory_provider(pool),
        requests: Vec::new(),
    };
    let resource = super::super::allocate_tensor_for_size(&mut memory, pool, &[1], 8, 8).unwrap();
    assert_eq!(memory.requests.len(), 1);
    assert_eq!(memory.requests[0].size_bytes, 8);
    let runtime = session.tensor_runtime();
    let mut buffer = resource.device_buffer().clone();
    buffer.copy_from(&u64::MAX.to_le_bytes()).unwrap();
    let mut status = Status {
        buffer,
        state: crate::owned_dispatch::FaultWordState::Sentinel,
    };
    let mut ir = super::super::checked_float::kernel(
        PcuDispatchFloatBinaryOp::Add,
        PcuScalarType::F32,
        PcuFloatUnderflowPolicy::IeeeAfterRounding,
        1,
    )
    .unwrap();
    let mut ops = ir.ops.to_vec();
    for op in &mut ops {
        if let PcuDispatchOp::Data(PcuDispatchDataOp::CheckedFloatBinary { range_policy, .. }) = op
        {
            *range_policy = PcuRangePolicy::Clamp;
        }
    }
    ir.ops = &ops;
    ir.numerical_requirements.range_policy = PcuRangePolicy::Clamp;
    ir.feature_caps = ir
        .feature_caps
        .union(fusion_pcu::PcuDispatchFeatureCaps::RANGE_CLAMP);
    let prepared = session
        .prepare_dispatch(PcuDispatchSubmission {
            kernel: &ir,
            shape: PcuInvocationShape::invocations(core::num::NonZeroU32::new(1).unwrap()),
        })
        .unwrap();
    let mut left = runtime.allocate(4).unwrap();
    let mut right = runtime.allocate(4).unwrap();
    let output = runtime.allocate(4).unwrap();
    left.copy_from(&f32::MAX.to_le_bytes()).unwrap();
    right.copy_from(&f32::MAX.to_le_bytes()).unwrap();
    let buffers = [left.clone(), right.clone(), output.clone()];
    let bindings = ir
        .bindings
        .iter()
        .zip(buffers)
        .map(|(binding, buffer)| {
            PcuOwnedBinding::new(
                binding.reference(),
                session.device_identity(),
                4,
                binding.access,
                binding.binding_type,
                buffer,
            )
        })
        .collect::<Vec<_>>();
    let mut completion = status.submit(&prepared, &bindings).unwrap();
    let outcome = completion.wait().unwrap();
    status.observe(outcome);
    assert!(matches!(outcome, PcuCompletionOutcome::Fault(f) if f.recovered));
    drop(completion);
    let mut clamped = [0; 4];
    output.copy_to(&mut clamped).unwrap();
    assert_eq!(u32::from_le_bytes(clamped), f32::MAX.to_bits());
    left.copy_from(&1.0_f32.to_le_bytes()).unwrap();
    right.copy_from(&1.0_f32.to_le_bytes()).unwrap();
    assert!(status.submit(&prepared, &bindings[..2]).is_err());
    assert_eq!(
        status.state,
        crate::owned_dispatch::FaultWordState::NeedsReset
    );
    #[cfg(feature = "allocation-census")]
    crate::reset_rocm_api_census();
    let mut completion = status.submit(&prepared, &bindings).unwrap();
    let outcome = completion.wait().unwrap();
    status.observe(outcome);
    assert_eq!(outcome, PcuCompletionOutcome::Succeeded);
    drop(completion);
    #[cfg(feature = "allocation-census")]
    assert_eq!(crate::rocm_api_census().host_to_device_copies, 1);
    #[cfg(feature = "allocation-census")]
    crate::reset_rocm_api_census();
    let mut completion = status.submit(&prepared, &bindings).unwrap();
    let outcome = completion.wait().unwrap();
    status.observe(outcome);
    assert_eq!(outcome, PcuCompletionOutcome::Succeeded);
    #[cfg(feature = "allocation-census")]
    assert_eq!(crate::rocm_api_census().host_to_device_copies, 0);
    drop(completion);
    // Even an untouched successful word loses its proof on a rejected submit attempt.
    assert!(status.submit(&prepared, &bindings[..2]).is_err());
    #[cfg(feature = "allocation-census")]
    crate::reset_rocm_api_census();
    let mut completion = status.submit(&prepared, &bindings).unwrap();
    let outcome = completion.wait().unwrap();
    status.observe(outcome);
    assert_eq!(outcome, PcuCompletionOutcome::Succeeded);
    #[cfg(feature = "allocation-census")]
    assert_eq!(crate::rocm_api_census().host_to_device_copies, 1);
}

#[test]
#[ignore = "requires native device; cold accounting, reused slots and session/pool identity"]
#[allow(clippy::too_many_lines)] // Admission accounting, reused ownership and preflight recovery share this bank.
fn private_bank_accounts_status_words_and_unique_physical_owners() {
    #[rustfmt::skip]
    use fusion_pcu::dialect::tensor::{
        Graph,
        TensorArithmeticRewritePolicy,
        TensorArithmeticCapability,
        TensorPointwiseGroupingPolicy,
    };
    let (_discovery, session) = super::super::tests::rocm_test_session();
    let pool = PcuMemoryPoolId(0x5354_0183);
    let assessor = super::super::RocmTensorAssessor::new(&session).unwrap();
    let mut graph = Graph::default();
    let input = graph.input([65], PcuScalarType::F32).unwrap();
    let mut output = input;
    for _ in 0..6 {
        output = graph.add(output, input).unwrap();
    }
    let prepared = assessor
        .prepare_owned_program(
            graph
                .into_selected_program(
                    &[output],
                    TensorArithmeticRewritePolicy::Disabled,
                    TensorArithmeticCapability::Strict,
                    TensorPointwiseGroupingPolicy::Disabled,
                )
                .unwrap(),
        )
        .unwrap();
    let mut memory = memory::Memory {
        provider: session.memory_provider(pool),
        requests: Vec::new(),
    };
    let bank = prepared
        .scratch
        .bind(
            session.tensor_runtime(),
            &prepared.view(),
            pool,
            &mut memory,
        )
        .unwrap();
    assert_eq!(bank.statuses.iter().flatten().count(), 6);
    assert_eq!(
        memory.requests.iter().filter(|r| r.size_bytes == 8).count(),
        6
    );
    assert_eq!(
        memory
            .requests
            .iter()
            .filter(|r| r.size_bytes == 8)
            .map(|r| r.size_bytes)
            .sum::<u64>(),
        48
    );
    let mapped = bank.resources.iter().flatten().count();
    let physical_values = bank.physical.iter().filter(|r| r.size_bytes() != 8).count();
    assert!(
        mapped > physical_values,
        "liveness must reuse physical slots: {mapped}/{physical_values}"
    );
    for (i, owner) in bank.physical.iter().enumerate() {
        assert!(
            bank.physical[..i]
                .iter()
                .all(|other| !owner.may_overlap(other))
        );
    }
    eprintln!(
        "cold-bank: status_bytes=48 status_allocations=6 mapped_values={mapped} unique_values={physical_values} physical_allocations={}",
        bank.physical.len()
    );
    drop(bank);
    let owner = fusion_pcu::PcuDeviceTensor::new(
        [65],
        session.upload_buffer(pool, &[1.0_f32; 65]).unwrap(),
    )
    .unwrap();
    let (_other_discovery, other_session) = super::super::tests::rocm_test_session();
    let foreign = fusion_pcu::PcuDeviceTensor::new(
        [65],
        other_session.upload_buffer(pool, &[1.0_f32; 65]).unwrap(),
    )
    .unwrap();
    // Raw allocations may be imported across compatible loaded runtime/device owners.
    // A borrowed descriptor instead carries its exact originating assessor session.
    let imported = assessor
        .execute_owned_program_outputs(&prepared, &[(input, &foreign)], pool, &mut memory)
        .unwrap();
    drop(imported);
    let other_assessor = super::super::RocmTensorAssessor::new(&other_session).unwrap();
    let foreign_ref = other_assessor
        .borrow_device_input_ref(&foreign, pool)
        .unwrap();
    assert!(
        assessor
            .execute_owned_program_output_from_inputs::<f32, _>(
                &prepared,
                &[(input, &foreign_ref)],
                pool,
                &mut memory
            )
            .is_err()
    );
    let other_pool = PcuMemoryPoolId(pool.0 + 1);
    assert!(
        assessor
            .execute_owned_program_outputs(&prepared, &[(input, &owner)], other_pool, &mut memory)
            .is_err()
    );
    let before = memory.requests.len();
    let escaped = assessor
        .execute_owned_program_outputs(&prepared, &[(input, &owner)], pool, &mut memory)
        .unwrap()
        .pop()
        .unwrap()
        .1;
    assert_eq!(
        memory.requests.len() - before,
        1,
        "only escaped output allocates after preflight errors"
    );
    let mut values = [0.0_f32; 65];
    session
        .download_buffer(pool, escaped.buffer(), &mut values)
        .unwrap();
    assert_eq!(values.map(f32::to_bits), [7.0_f32.to_bits(); 65]);
}
