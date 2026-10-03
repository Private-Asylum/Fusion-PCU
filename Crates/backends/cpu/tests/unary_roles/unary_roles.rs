//! Arbitrarily many unread declarations retain typed cold metadata but only two resources execute.
#[path = "../low_unary/graph/graph.rs"]
mod graph;
#[path = "../portable_unary/oracle/oracle.rs"]
mod oracle;
#[rustfmt::skip]
use pcu_facade::{
    PcuBinding,
    PcuBindingAccess,
    PcuBindingRef,
    PcuBindingStorageClass,
    PcuCostBoundary,
    PcuDeviceIdentity,
    PcuDispatchFloatUnaryOp,
    PcuExecutorId,
    PcuFloatUnderflowPolicy,
    PcuHostArgument,
    PcuHostKernelBackend,
    PcuImplementationOffers,
    PcuImplementationRequest,
    PcuObjectKind,
    PcuObjectRef,
    PcuPreparedHostKernel,
    PcuProviderId,
    PcuRangePolicy,
    PcuReproducibility,
    PcuValueType,
};
#[rustfmt::skip]
use fusion_pcu_cpu::{
    PcuCpuHostBackend,
    PcuCpuHostOffers,
    PcuCpuPreparedHost,
};
use oracle::Native;
const fn device() -> PcuDeviceIdentity {
    PcuDeviceIdentity::from_device_ref(PcuObjectRef {
        provider: PcuProviderId(3),
        generation: 7,
        kind: PcuObjectKind::Device,
        id: 0,
    })
    .unwrap()
}
fn verify<T: Native>(format: u32) {
    for op in [PcuDispatchFloatUnaryOp::Neg, PcuDispatchFloatUnaryOp::Relu] {
        for uf in [
            PcuFloatUnderflowPolicy::IeeeAfterRounding,
            PcuFloatUnderflowPolicy::RejectSubnormalResult,
            PcuFloatUnderflowPolicy::AllowGradualUnderflow,
        ] {
            for range in [PcuRangePolicy::Reject, PcuRangePolicy::Clamp] {
                for grid in [false, true] {
                    for broadcast in [false, true] {
                        let mut graph = graph::Graph::new(T::TYPE, 7, op, uf, range);
                        graph.grid = grid;
                        graph.broadcast = broadcast;
                        graph.with(|kernel| {
                            let mut bindings = kernel.bindings.to_vec();
                            bindings.extend((2..64).map(|index| {
                                PcuBinding::value(
                                    None,
                                    0,
                                    index,
                                    PcuBindingStorageClass::Storage,
                                    PcuBindingAccess::ReadOnly,
                                    PcuValueType::Scalar(T::TYPE),
                                )
                            }));
                            let mut kernel = *kernel;
                            kernel.bindings = &bindings;
                            let backend = PcuCpuHostBackend::scalar();
                            let mut prepared = backend.prepare_host_kernel(&kernel).unwrap();
                            let PcuCpuPreparedHost::UnaryRoles(plan) = &prepared else {
                                panic!("owned role plan");
                            };
                            let expected_id = 17920
                                + format * 4
                                + u32::from(op == PcuDispatchFloatUnaryOp::Relu)
                                + u32::from(range == PcuRangePolicy::Clamp) * 2;
                            assert_eq!(plan.local_id(), expected_id);
                            assert_eq!(prepared.argument_count(), 64);
                            check_offer(backend, &kernel, expected_id);
                            for invalid in [false, true, false] {
                                let mut input = [T::from_bits(1 << T::FRACTION); 7];
                                if invalid {
                                    input[if broadcast { 0 } else { 5 }] =
                                        T::from_bits(T::SIGN - 1);
                                }
                                let mut output = [T::from_bits(17); 9];
                                let mut expected = output;
                                let reference = if broadcast {
                                    oracle::native_broadcast::<T, 7>(
                                        &input,
                                        &mut expected,
                                        op,
                                        uf,
                                        range,
                                    )
                                } else {
                                    oracle::native::<T, 7>(&input, &mut expected, op, uf, range)
                                };
                                let mut arguments = vec![
                                    PcuHostArgument::read(PcuBindingRef::new(0, 0), &input),
                                    PcuHostArgument::read_write(
                                        PcuBindingRef::new(0, 1),
                                        &mut output,
                                    ),
                                ];
                                arguments.extend((2..64).map(|index| {
                                    PcuHostArgument::read(PcuBindingRef::new(0, index), &[] as &[T])
                                }));
                                assert_eq!(
                                    prepared
                                        .call(&mut arguments)
                                        .map_err(|e| e.fault().unwrap()),
                                    reference
                                );
                                drop(arguments);
                                assert_eq!(oracle::bits(&output), oracle::bits(&expected));
                            }
                            kernel
                                .numerical_requirements
                                .numerical_options
                                .reproducibility = PcuReproducibility::PortableV1;
                            assert!(backend.prepare_host_kernel(&kernel).is_err());
                            kernel
                                .numerical_requirements
                                .numerical_options
                                .reproducibility = PcuReproducibility::Unspecified;
                            let mut invalid_bindings = bindings.clone();
                            invalid_bindings[63].access = PcuBindingAccess::ReadWrite;
                            kernel.bindings = &invalid_bindings;
                            assert!(backend.prepare_host_kernel(&kernel).is_err());
                        });
                    }
                }
            }
        }
    }
}
fn check_offer(
    backend: PcuCpuHostBackend,
    kernel: &pcu_facade::PcuDispatchKernelIr<'_>,
    expected: u32,
) {
    let offers = PcuCpuHostOffers::new(backend, device(), PcuExecutorId(0));
    let request = PcuImplementationRequest {
        device: device(),
        executor: PcuExecutorId(0),
        boundary: PcuCostBoundary::Host,
        operation: kernel,
        requirements: kernel.numerical_requirements,
    };
    let mut slots = [None];
    assert_eq!(offers.implementation_offers(&request, &mut slots), Ok(1));
    let offer = slots[0].unwrap();
    assert_eq!(offer.implementation.local_id, expected);
    assert_eq!(offer.implementation.revision, 1);
    assert_eq!(offer.requirements, request.requirements);
    offer.validate_request(&request).unwrap();
    let mut mismatch = request.requirements;
    mismatch.float_underflow = match mismatch.float_underflow {
        PcuFloatUnderflowPolicy::IeeeAfterRounding => {
            PcuFloatUnderflowPolicy::AllowGradualUnderflow
        }
        _ => PcuFloatUnderflowPolicy::IeeeAfterRounding,
    };
    assert_eq!(
        offers.implementation_offers(
            &PcuImplementationRequest {
                requirements: mismatch,
                ..request
            },
            &mut []
        ),
        Ok(0)
    );
}
#[test]
fn six_formats_unread_metadata_is_unbounded_and_never_an_operand() {
    verify::<pcu_facade::PcuF16Bits>(0);
    verify::<pcu_facade::PcuBf16Bits>(1);
    verify::<pcu_facade::PcuF8E4M3FnBits>(2);
    verify::<pcu_facade::PcuF8E5M2Bits>(3);
    verify::<f32>(4);
    verify::<f64>(5);
}
