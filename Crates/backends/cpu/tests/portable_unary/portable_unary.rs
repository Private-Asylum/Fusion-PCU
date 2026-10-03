//! Descriptor-first six-format exact selection, independent bit oracle and full cold tuple.
#[path = "../low_unary/graph/graph.rs"]
mod graph;
#[path = "oracle/oracle.rs"]
mod oracle;
#[rustfmt::skip]
use pcu_facade::{
    PcuBf16Bits,
    PcuBindingRef,
    PcuCompoundArithmeticPolicy,
    PcuCostBoundary,
    PcuDeviceIdentity,
    PcuDispatchFloatUnaryOp,
    PcuExecutorId,
    PcuF16Bits,
    PcuF8E4M3FnBits,
    PcuF8E5M2Bits,
    PcuFloatUnderflowPolicy,
    PcuHostArgument,
    PcuHostKernelBackend,
    PcuImplementationOffers,
    PcuImplementationRequest,
    PcuNumericalMode,
    PcuObjectKind,
    PcuObjectRef,
    PcuPrecisionPolicy,
    PcuPreparedHostKernel,
    PcuProviderId,
    PcuRangePolicy,
    PcuReproducibility,
};
#[rustfmt::skip]
use fusion_pcu_cpu::{
    PcuCpuCheckedUnary,
    PcuCpuHostBackend,
    PcuCpuHostOffers,
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
                            for mode in [PcuNumericalMode::Boundary, PcuNumericalMode::Strict] {
                                for compound in [
                                    PcuCompoundArithmeticPolicy::Checked,
                                    PcuCompoundArithmeticPolicy::BackendDefined,
                                ] {
                                    for precision in [
                                        PcuPrecisionPolicy::Preserve,
                                        PcuPrecisionPolicy::BackendOptimized,
                                    ] {
                                        let mut kernel = *kernel;
                                        let requirements = &mut kernel.numerical_requirements;
                                        requirements.numerical_mode = mode;
                                        requirements.numerical_options.compound_arithmetic =
                                            compound;
                                        requirements.numerical_options.precision = precision;
                                        requirements.numerical_options.reproducibility =
                                            PcuReproducibility::PortableV1;
                                        verify_profile::<T>(&kernel, format, broadcast);
                                    }
                                }
                            }
                        });
                    }
                }
            }
        }
    }
}
fn verify_profile<T: Native>(
    kernel: &pcu_facade::PcuDispatchKernelIr<'_>,
    format: u32,
    broadcast: bool,
) {
    let backend = PcuCpuHostBackend::scalar();
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
    let mut typed = PcuCpuCheckedUnary::<T>::new()
        .prepare_host_kernel(kernel)
        .unwrap();
    assert_eq!(
        offer.implementation.local_id,
        16384
            + format * 4
            + u32::from(typed.range_policy() == PcuRangePolicy::Clamp) * 2
            + u32::from(typed.operation() == PcuDispatchFloatUnaryOp::Relu)
    );
    assert_eq!(offer.implementation.local_id, typed.local_id());
    assert_eq!(offer.implementation.revision, 1);
    assert_eq!(offer.requirements, request.requirements);
    assert_eq!(offer.workspace_bytes, Some(0));
    offer.validate_request(&request).unwrap();
    let mut normal = request.requirements;
    normal.numerical_options.reproducibility = PcuReproducibility::Unspecified;
    assert_eq!(
        offers.implementation_offers(
            &PcuImplementationRequest {
                requirements: normal,
                ..request
            },
            &mut []
        ),
        Ok(0)
    );
    let values = [
        1,
        0,
        T::SIGN,
        T::SIGN | 1,
        T::MAX,
        T::SIGN | T::MAX,
        1 << T::FRACTION,
    ];
    for bad_lane in [None, Some(0), Some(6), None] {
        let mut input = values.map(T::from_bits);
        if let Some(lane) = bad_lane {
            input[lane] = T::from_bits(T::SIGN - 1);
        }
        let mut output = [T::from_bits(17); 10];
        let mut expected = output;
        let native = if broadcast {
            oracle::native_broadcast::<T, 7>(
                &input,
                &mut expected,
                typed.operation(),
                typed.underflow_policy(),
                typed.range_policy(),
            )
        } else {
            oracle::native::<T, 7>(
                &input,
                &mut expected,
                typed.operation(),
                typed.underflow_policy(),
                typed.range_policy(),
            )
        };
        let actual = typed
            .call(&mut [
                PcuHostArgument::read(
                    PcuBindingRef::new(0, 0),
                    &input[..if broadcast { 1 } else { 7 }],
                ),
                PcuHostArgument::read_write(PcuBindingRef::new(0, 1), &mut output),
            ])
            .map_err(|error| error.fault().unwrap());
        assert_eq!(actual, native);
        assert_eq!(oracle::bits(&output), oracle::bits(&expected));
    }
    let mut bad = *kernel;
    bad.numerical_requirements.float_underflow =
        if typed.underflow_policy() == PcuFloatUnderflowPolicy::AllowGradualUnderflow {
            PcuFloatUnderflowPolicy::RejectSubnormalResult
        } else {
            PcuFloatUnderflowPolicy::AllowGradualUnderflow
        };
    assert!(backend.prepare_host_kernel(&bad).is_err());
}
#[test]
fn six_formats_exact_offers_and_typed_transactions() {
    verify::<PcuF16Bits>(0);
    verify::<PcuBf16Bits>(1);
    verify::<PcuF8E4M3FnBits>(2);
    verify::<PcuF8E5M2Bits>(3);
    verify::<f32>(4);
    verify::<f64>(5);
}

fn complete<T: Native>() {
    for op in [PcuDispatchFloatUnaryOp::Neg, PcuDispatchFloatUnaryOp::Relu] {
        for uf in [
            PcuFloatUnderflowPolicy::IeeeAfterRounding,
            PcuFloatUnderflowPolicy::RejectSubnormalResult,
            PcuFloatUnderflowPolicy::AllowGradualUnderflow,
        ] {
            for range in [PcuRangePolicy::Reject, PcuRangePolicy::Clamp] {
                let mut prepared = graph::Graph::new(T::TYPE, 1, op, uf, range)
                    .with(|kernel| {
                        let mut kernel = *kernel;
                        kernel
                            .numerical_requirements
                            .numerical_options
                            .reproducibility = PcuReproducibility::PortableV1;
                        PcuCpuCheckedUnary::<T>::new().prepare_host_kernel(&kernel)
                    })
                    .unwrap();
                for raw in 0..T::SIGN * 2 {
                    let input = [T::from_bits(raw)];
                    let mut output = [T::from_bits(17); 3];
                    let mut expected = output;
                    let native = oracle::native::<T, 1>(&input, &mut expected, op, uf, range);
                    let actual = prepared
                        .call(&mut [
                            PcuHostArgument::read(PcuBindingRef::new(0, 0), &input),
                            PcuHostArgument::read_write(PcuBindingRef::new(0, 1), &mut output),
                        ])
                        .map_err(|error| error.fault().unwrap());
                    assert_eq!(actual, native, "raw={raw:x}/{op:?}/{uf:?}/{range:?}");
                    assert_eq!(oracle::bits(&output), oracle::bits(&expected));
                }
            }
        }
    }
}
#[test]
fn four_low_formats_complete_encoding_portable_contract() {
    complete::<PcuF16Bits>();
    complete::<PcuBf16Bits>();
    complete::<PcuF8E4M3FnBits>();
    complete::<PcuF8E5M2Bits>();
}
