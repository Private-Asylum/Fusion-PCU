//! Actual sixty-four-declaration mixed projection with one live input/output.
#[rustfmt::skip]
use super::{
    device,
    graph,
    oracle,
};
use oracle::Native;
#[path = "../../../../cpu/tests/scalar_tensor/source/source.rs"]
#[allow(dead_code)] // Consumption is a separate owned-leaf contract.
mod owned;
static EXPECTED_REQUEST: std::sync::Mutex<pcu_facade::PcuImplementationRequirements> =
    std::sync::Mutex::new(pcu_facade::PcuImplementationRequirements::DEFAULT);

fn score(candidate: &pcu_facade::global::PcuInvocationCandidate<'_>) -> i128 {
    assert_eq!(
        candidate.kernel.numerical_requirements,
        *EXPECTED_REQUEST.lock().unwrap()
    );
    assert!(pcu_facade::describe_checked_float_unary_map(candidate.kernel).is_ok());
    0
}
#[path = "source/source.rs"]
mod source;
#[rustfmt::skip]
use fusion_pcu_vulkan::{
    PcuVulkanArgument,
    PcuVulkanBackend,
    PcuVulkanError,
    PcuVulkanOwnedBuffer,
    PcuVulkanPreparedMixed,
};
#[rustfmt::skip]
use pcu_facade::{
    PcuBinding,
    PcuBindingAccess,
    PcuBindingRef,
    PcuBindingStorageClass,
    PcuDispatchFloatUnaryOp,
    PcuDispatchKernelIr,
    PcuExecutionFault,
    PcuFloatUnderflowPolicy,
    PcuHostArgument,
    PcuRangePolicy,
    PcuValueType,
};
fn same<T: Native>(actual: &[T], expected: &[T]) {
    assert_eq!(actual.len(), expected.len());
    for (a, b) in actual.iter().zip(expected) {
        assert_eq!(a.bits(), b.bits());
    }
}
enum Destination<'a, T: Native> {
    Host(&'a mut [T]),
    Owned(&'a mut PcuVulkanOwnedBuffer<T>),
}
fn invoke<T: Native>(
    plan: &mut PcuVulkanPreparedMixed,
    input: &PcuVulkanOwnedBuffer<T>,
    ghost: &PcuVulkanOwnedBuffer<T>,
    output: Destination<'_, T>,
) -> Result<(), PcuVulkanError> {
    let empty: [T; 0] = [];
    let mut arguments = core::array::from_fn::<_, 64, _>(|index| {
        PcuVulkanArgument::host(PcuHostArgument::read(
            PcuBindingRef::new(0, u32::try_from(index).unwrap()),
            &empty,
        ))
    });
    arguments[0] = ghost.read_argument(PcuBindingRef::new(0, 0));
    arguments[1] = match output {
        Destination::Host(bytes) => {
            PcuVulkanArgument::host(PcuHostArgument::read_write(PcuBindingRef::new(0, 1), bytes))
        }
        Destination::Owned(owner) => owner.write_argument(PcuBindingRef::new(0, 1)),
    };
    arguments[2] = input.read_argument(PcuBindingRef::new(0, 2));
    plan.call(&mut arguments)
}
fn fault(error: PcuVulkanError) -> PcuExecutionFault {
    match error {
        PcuVulkanError::Fault(fault) => fault,
        other => panic!("unexpected native error {other:?}"),
    }
}
fn transaction<T: Native>(
    backend: &PcuVulkanBackend,
    foreign: &PcuVulkanBackend,
    base: &PcuDispatchKernelIr<'_>,
    description: graph::Graph,
) {
    let mut bindings = base.bindings.to_vec();
    for index in 4..64 {
        bindings.push(PcuBinding::value(
            None,
            0,
            index,
            PcuBindingStorageClass::Storage,
            PcuBindingAccess::ReadOnly,
            PcuValueType::Scalar(T::TYPE),
        ));
    }
    let mut kernel = *base;
    kernel.bindings = &bindings;
    let mut plan = backend.prepare_mixed_kernel(&kernel).unwrap();
    assert_eq!(plan.argument_count(), 64);
    let bank = [T::from_bits(T::MAX + 1); 11];
    let mut output = backend.upload_owned(&bank).unwrap();
    let ghost = foreign.upload_owned(&bank).unwrap();
    for phase in 0..3 {
        let mut input = [T::from_bits(0); 11];
        for (lane, value) in input[..7].iter_mut().enumerate() {
            *value = T::from_bits(match (lane + phase) % 4 {
                0 => 0,
                1 => T::SIGN,
                2 => 1,
                _ => T::MAX,
            });
        }
        if phase == 2 {
            input[5] = T::from_bits(T::MAX + 1);
        }
        let owner = backend.upload_owned(&input).unwrap();
        let mut prior = bank;
        output.read_into(&mut prior).unwrap();
        let mut expected = prior;
        let oracle = if description.broadcast {
            oracle::native_broadcast::<T, 7>(
                &input,
                &mut expected,
                description.op,
                description.policy,
                description.range,
            )
        } else {
            oracle::native::<T, 7>(
                &input,
                &mut expected,
                description.op,
                description.policy,
                description.range,
            )
        };
        let actual =
            invoke(&mut plan, &owner, &ghost, Destination::Owned(&mut output)).map_err(fault);
        assert_eq!(actual, oracle);
        assert_eq!(
            plan.last_call_may_have_written(),
            oracle.is_ok() || oracle.is_err_and(|fault| fault.recovered)
        );
        assert!(!plan.last_call_completion_uncertain());
        let mut observed = bank;
        output.read_into(&mut observed).unwrap();
        same(&observed, &expected);
        let mut host = prior;
        assert_eq!(
            invoke(&mut plan, &owner, &ghost, Destination::Host(&mut host)).map_err(fault),
            oracle
        );
        same(&host, &expected);
        assert!(invoke(&mut plan, &ghost, &owner, Destination::Owned(&mut output)).is_err());
        assert!(!plan.last_call_may_have_written());
        assert!(!plan.last_call_completion_uncertain());
        output.read_into(&mut observed).unwrap();
        same(&observed, &expected);
        owner.read_into(&mut observed).unwrap();
        same(&observed, &input);
    }
}
fn width<T: Native>(backend: &PcuVulkanBackend, foreign: &PcuVulkanBackend) {
    for operation in [PcuDispatchFloatUnaryOp::Neg, PcuDispatchFloatUnaryOp::Relu] {
        for policy in [
            PcuFloatUnderflowPolicy::IeeeAfterRounding,
            PcuFloatUnderflowPolicy::AllowGradualUnderflow,
            PcuFloatUnderflowPolicy::RejectSubnormalResult,
        ] {
            for range in [PcuRangePolicy::Reject, PcuRangePolicy::Clamp] {
                for grid in [false, true] {
                    for broadcast in [false, true] {
                        let mut description =
                            graph::Graph::new(T::TYPE, 7, operation, policy, range);
                        description.grid = grid;
                        description.broadcast = broadcast;
                        description
                            .with(|base| transaction::<T>(backend, foreign, base, description));
                    }
                }
            }
        }
    }
}

fn ordinary<T: Native>(mut policy: pcu_facade::global::PcuExecutionPolicy) {
    use pcu_facade::global;
    *EXPECTED_REQUEST.lock().unwrap() = pcu_facade::PcuImplementationRequirements {
        numerical_mode: policy.numerical_mode,
        numerical_options: policy.numerical_options,
        float_underflow: policy.float_underflow,
        range_policy: policy.range_policy,
    };
    source::verify::<T>(*EXPECTED_REQUEST.lock().unwrap());
    policy.score_invocation = Some(score);
    // Owned identity is a separate Reject-only leaf contract. Form test owners
    // under that exact policy, then restore the unmodified arithmetic request.
    let mut setup = policy;
    setup.range_policy = PcuRangePolicy::Reject;
    setup.score_invocation = None;
    global::configure(setup).unwrap();
    let bank = [T::from_bits(T::MAX + 1); 11];
    let ghost = owned::identity(&bank).unwrap();
    global::clear_thread_cache().unwrap();
    let mut output = owned::identity(&bank).unwrap();
    let banks = core::array::from_fn::<_, 18, _>(|index| phase_input::<T>(index / 6, index % 6));
    let owners = banks
        .each_ref()
        .map(|input| owned::identity(input).unwrap());
    global::configure(policy).unwrap();
    let nan = T::from_bits(T::MAX + 1);
    for phase in 0..3 {
        for role in 0..6 {
            let input = banks[phase * 6 + role];
            let input_owner = &owners[phase * 6 + role];
            let operation = if role.is_multiple_of(2) {
                PcuDispatchFloatUnaryOp::Neg
            } else {
                PcuDispatchFloatUnaryOp::Relu
            };
            let mut prior = bank;
            output.read_into(&mut prior).unwrap();
            let mut expected = prior;
            let oracle = if role >= 4 {
                oracle::native_broadcast::<T, 7>(
                    &input,
                    &mut expected,
                    operation,
                    policy.float_underflow,
                    policy.range_policy,
                )
            } else {
                oracle::native::<T, 7>(
                    &input,
                    &mut expected,
                    operation,
                    policy.float_underflow,
                    policy.range_policy,
                )
            };
            let result = match role {
                0 => source::direct_neg(&ghost, &mut output, input_owner, &nan),
                1 => source::direct_relu(&ghost, &mut output, input_owner, &nan),
                2 => source::grid_neg(&ghost, &mut output, input_owner, &nan),
                3 => source::grid_relu(&ghost, &mut output, input_owner, &nan),
                4 => source::broadcast_neg(&ghost, &mut output, input_owner, &nan),
                5 => source::broadcast_relu(&ghost, &mut output, input_owner, &nan),
                _ => unreachable!("six frozen annotated roles"),
            }
            .map_err(|error| error.arithmetic_fault().expect("actual arithmetic fault"));
            assert_eq!(result, oracle);
            let mut observed = bank;
            output.read_into(&mut observed).unwrap();
            same(&observed, &expected);
            input_owner.read_into(&mut observed).unwrap();
            same(&observed, &input);
        }
    }
    global::clear_thread_cache().unwrap();
    let mut observed = bank;
    ghost.read_into(&mut observed).unwrap();
    same(&observed, &bank);
}

fn phase_input<T: Native>(phase: usize, role: usize) -> [T; 11] {
    let mut input = [T::from_bits(if phase == 1 { 1 << T::FRACTION } else { 1 }); 11];
    if phase == 2 {
        input[if role >= 4 { 0 } else { 5 }] = T::from_bits(T::MAX + 1);
    }
    input
}

fn policies() -> [pcu_facade::global::PcuExecutionPolicy; 48] {
    #[rustfmt::skip]
    use pcu_facade::{
        global,
        PcuCompoundArithmeticPolicy,
        PcuNumericalMode,
        PcuPrecisionPolicy,
    };
    core::array::from_fn(|index| {
        let mut policy = global::PcuExecutionPolicy {
            backend: global::PcuBackendChoice::Vulkan,
            numerical_mode: [PcuNumericalMode::Boundary, PcuNumericalMode::Strict][index / 24],
            float_underflow: [
                PcuFloatUnderflowPolicy::IeeeAfterRounding,
                PcuFloatUnderflowPolicy::AllowGradualUnderflow,
                PcuFloatUnderflowPolicy::RejectSubnormalResult,
            ][index / 2 % 3],
            range_policy: [PcuRangePolicy::Reject, PcuRangePolicy::Clamp][index % 2],
            ..Default::default()
        };
        policy.numerical_options.compound_arithmetic = [
            PcuCompoundArithmeticPolicy::Checked,
            PcuCompoundArithmeticPolicy::BackendDefined,
        ][index / 12 % 2];
        policy.numerical_options.precision = [
            PcuPrecisionPolicy::Preserve,
            PcuPrecisionPolicy::BackendOptimized,
        ][index / 6 % 2];
        policy
    })
}
#[test]
fn genuine_source_freezes_all_complete_requests_before_resident_affinity_bypasses_ranking() {
    for policy in policies() {
        let request = pcu_facade::PcuImplementationRequirements {
            numerical_mode: policy.numerical_mode,
            numerical_options: policy.numerical_options,
            float_underflow: policy.float_underflow,
            range_policy: policy.range_policy,
        };
        source::verify::<pcu_facade::PcuF16Bits>(request);
        source::verify::<pcu_facade::PcuBf16Bits>(request);
        source::verify::<pcu_facade::PcuF8E4M3FnBits>(request);
        source::verify::<pcu_facade::PcuF8E5M2Bits>(request);
        source::verify::<f32>(request);
        source::verify::<f64>(request);
    }
}
#[test]
#[ignore = "requires physical Vulkan GPU and exclusive untimed correctness window"]
fn six_format_genuine_mixed_unary_roles_complete_requests_and_cache_lifetime() {
    for policy in policies() {
        ordinary::<pcu_facade::PcuF16Bits>(policy);
        ordinary::<pcu_facade::PcuBf16Bits>(policy);
        ordinary::<pcu_facade::PcuF8E4M3FnBits>(policy);
        ordinary::<pcu_facade::PcuF8E5M2Bits>(policy);
        ordinary::<f32>(policy);
        ordinary::<f64>(policy);
    }
    pcu_facade::global::use_defaults().unwrap();
}
#[test]
#[ignore = "requires physical Vulkan GPU and exclusive untimed correctness window"]
fn six_format_mixed_actual_roles_foreign_unused_prefix_and_transaction() {
    let (backend, _) = device::selected();
    let (foreign, _) = device::selected();
    width::<pcu_facade::PcuF16Bits>(&backend, &foreign);
    width::<pcu_facade::PcuBf16Bits>(&backend, &foreign);
    width::<pcu_facade::PcuF8E4M3FnBits>(&backend, &foreign);
    width::<pcu_facade::PcuF8E5M2Bits>(&backend, &foreign);
    width::<f32>(&backend, &foreign);
    width::<f64>(&backend, &foreign);
}
