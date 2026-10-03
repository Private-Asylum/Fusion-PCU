//! Actual native role projection, cold identity, readonly metadata and private publication.
#[path = "../scalar_transport/device/device.rs"]
mod device;
#[path = "../../../cpu/tests/unary_roles/graph/graph.rs"]
mod graph;
#[path = "mixed/mixed.rs"]
mod mixed;
#[path = "../../../cpu/tests/portable_unary/oracle/oracle.rs"]
mod oracle;
use oracle::Native;
#[rustfmt::skip]
use pcu_facade::{
    PcuBinding,
    PcuBindingAccess,
    PcuBindingRef,
    PcuBindingStorageClass,
    PcuCostBoundary,
    PcuDispatchFloatUnaryOp,
    PcuDispatchKernelIr,
    PcuExecutionFault,
    PcuExecutionFaultKind,
    PcuExecutorId,
    PcuFloatUnderflowPolicy,
    PcuHostArgument,
    PcuHostKernelBackend,
    PcuImplementationOffers,
    PcuImplementationRequest,
    PcuPreparedHostKernel,
    PcuRangePolicy,
    PcuReproducibility,
    PcuValueType,
};
#[rustfmt::skip]
use fusion_pcu_vulkan::{
    PcuVulkanBackend,
    PcuVulkanError,
    PcuVulkanPreparedHost,
};
fn fault(error: PcuVulkanError) -> PcuExecutionFault {
    match error {
        PcuVulkanError::Fault(fault) => fault,
        other => panic!("unexpected {other:?}"),
    }
}
fn width<T: Native>(backend: &PcuVulkanBackend, ordinal: u32) {
    for operation in [PcuDispatchFloatUnaryOp::Neg, PcuDispatchFloatUnaryOp::Relu] {
        for policy in [
            PcuFloatUnderflowPolicy::IeeeAfterRounding,
            PcuFloatUnderflowPolicy::AllowGradualUnderflow,
            PcuFloatUnderflowPolicy::RejectSubnormalResult,
        ] {
            for range in [PcuRangePolicy::Reject, PcuRangePolicy::Clamp] {
                for grid in [false, true] {
                    for broadcast in [false, true] {
                        let mut graph = graph::Graph::new(T::TYPE, 7, operation, policy, range);
                        graph.grid = grid;
                        graph.broadcast = broadcast;
                        graph.with(|base| {
                            profile::<T>(
                                backend, base, ordinal, operation, policy, range, broadcast,
                            );
                        });
                    }
                }
            }
        }
    }
}
fn profile<T: Native>(
    backend: &PcuVulkanBackend,
    base: &PcuDispatchKernelIr<'_>,
    ordinal: u32,
    operation: PcuDispatchFloatUnaryOp,
    policy: PcuFloatUnderflowPolicy,
    range: PcuRangePolicy,
    broadcast: bool,
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
    let id = 18176
        + ordinal * 4
        + u32::from(operation == PcuDispatchFloatUnaryOp::Relu)
        + u32::from(range == PcuRangePolicy::Clamp) * 2;
    let mut plan = backend.prepare_host_kernel(&kernel).unwrap();
    let PcuVulkanPreparedHost::UnaryRoles(role) = &plan else {
        panic!("actual role realization required");
    };
    assert_eq!(role.argument_count(), 64);
    assert_eq!(role.profile().local_id(), Some(id));
    assert_eq!(role.memory_realizations().unwrap().len(), 3);
    offer(backend, &kernel, id);
    transaction::<T>(&mut plan, operation, policy, range, broadcast);
    let mut wrong = kernel;
    wrong
        .numerical_requirements
        .numerical_options
        .reproducibility = PcuReproducibility::PortableV1;
    assert!(backend.prepare_host_kernel(&wrong).is_err());
    let mut invalid_bindings = bindings.clone();
    invalid_bindings[63].access = PcuBindingAccess::ReadWrite;
    let mut invalid_kernel = *base;
    invalid_kernel.bindings = &invalid_bindings;
    assert!(backend.prepare_host_kernel(&invalid_kernel).is_err());
}
fn offer(backend: &PcuVulkanBackend, kernel: &PcuDispatchKernelIr<'_>, id: u32) {
    let request = PcuImplementationRequest {
        device: backend.device_identity().unwrap(),
        executor: PcuExecutorId(0),
        operation: kernel,
        requirements: kernel.numerical_requirements,
        boundary: PcuCostBoundary::Host,
    };
    let mut slots = [None];
    assert_eq!(
        backend.implementation_offers(&request, &mut slots).unwrap(),
        1
    );
    let offer = slots[0].unwrap();
    offer.validate_request(&request).unwrap();
    assert_eq!(
        (offer.implementation.local_id, offer.implementation.revision),
        (id, 1)
    );
    let mut mismatch = request;
    mismatch.requirements.range_policy =
        if kernel.numerical_requirements.range_policy == PcuRangePolicy::Reject {
            PcuRangePolicy::Clamp
        } else {
            PcuRangePolicy::Reject
        };
    assert_eq!(
        backend.implementation_offers(&mismatch, &mut []).unwrap(),
        0
    );
}
fn call<T: Native>(
    plan: &mut PcuVulkanPreparedHost,
    input: &[T],
    output: &mut [T],
) -> Result<(), PcuExecutionFault> {
    // Dynamic caller schema construction is outside the separate stack-based caller census.
    let mut arguments: Vec<_> = (0..64)
        .map(|index| PcuHostArgument::read(PcuBindingRef::new(0, index), &[] as &[T]))
        .collect();
    arguments[2] = PcuHostArgument::read(PcuBindingRef::new(0, 2), input);
    arguments[1] = PcuHostArgument::read_write(PcuBindingRef::new(0, 1), output);
    plan.call(&mut arguments).map_err(fault)
}
fn same<T: Native>(left: &[T], right: &[T]) {
    assert_eq!(oracle::bits(left), oracle::bits(right));
}
fn expected<T: Native>(
    input: &[T],
    output: &mut [T],
    operation: PcuDispatchFloatUnaryOp,
    policy: PcuFloatUnderflowPolicy,
    range: PcuRangePolicy,
    broadcast: bool,
) -> Result<(), PcuExecutionFault> {
    if broadcast {
        oracle::native_broadcast::<T, 7>(input, output, operation, policy, range)
    } else {
        oracle::native::<T, 7>(input, output, operation, policy, range)
    }
}
fn transaction<T: Native>(
    plan: &mut PcuVulkanPreparedHost,
    operation: PcuDispatchFloatUnaryOp,
    policy: PcuFloatUnderflowPolicy,
    range: PcuRangePolicy,
    broadcast: bool,
) {
    let mut input = [T::from_bits(1 << T::FRACTION); 7];
    let mut output = [T::from_bits(17); 10];
    let mut golden = output;
    for phase in 0..3 {
        input.fill(T::from_bits((1 << T::FRACTION) + phase));
        assert_eq!(
            call(plan, &input, &mut output),
            expected(&input, &mut golden, operation, policy, range, broadcast)
        );
        same(&output, &golden);
    }
    input.fill(T::from_bits(1));
    assert_eq!(
        call(plan, &input, &mut output),
        expected(&input, &mut golden, operation, policy, range, broadcast)
    );
    same(&output, &golden);
    let before = output;
    input[if broadcast { 0 } else { 5 }] = T::from_bits(T::SIGN - 1);
    let actual = call(plan, &input, &mut output);
    assert_eq!(
        actual,
        expected(&input, &mut golden, operation, policy, range, broadcast)
    );
    same(&output, &before);
    if policy != PcuFloatUnderflowPolicy::RejectSubnormalResult
        || range == PcuRangePolicy::Clamp
        || broadcast
    {
        assert_eq!(
            actual.unwrap_err().kind,
            PcuExecutionFaultKind::InvalidFloatingOperand
        );
    }
    input.fill(T::from_bits(1 << T::FRACTION));
    call(plan, &input, &mut output).unwrap();
    assert_eq!(
        call(plan, &input, &mut output),
        expected(&input, &mut golden, operation, policy, range, broadcast)
    );
    same(&output, &golden);
}
#[test]
#[ignore = "requires actual Vulkan compute, 64-declaration cold schemas and native transactional role proof"]
fn six_format_actual_roles_keep_only_two_native_resources_and_exact_fault_law() {
    let (backend, _) = device::selected();
    width::<pcu_facade::PcuF16Bits>(&backend, 0);
    width::<pcu_facade::PcuBf16Bits>(&backend, 1);
    width::<pcu_facade::PcuF8E4M3FnBits>(&backend, 2);
    width::<pcu_facade::PcuF8E5M2Bits>(&backend, 3);
    width::<f32>(&backend, 4);
    width::<f64>(&backend, 5);
}
