//! Exact opt-in offers are distinct from legacy maps and retain original argument extents.
#[path = "../../benches/integer_operands/source/source.rs"]
#[allow(dead_code)] // All ten genuine source functions are measured in the paired benchmark target.
mod source;
#[rustfmt::skip]
use fusion_pcu_cpu::{
    PcuCpuCheckedInteger,
    PcuCpuCheckedIntegerError,
    PcuCpuIntegerOffers,
};
#[rustfmt::skip]
use pcu_facade::{
    PcuCheckedInteger,
    PcuHostKernelBackend,
    PcuPreparedHostKernel,
    PcuHostArgument,
    PcuBindingRef,
    PcuDeviceIdentity,
    PcuObjectRef,
    PcuProviderId,
    PcuObjectKind,
    PcuExecutorId,
    PcuImplementationRequest,
    PcuImplementationOffers,
    PcuCostBoundary,
    PcuNumericalMode,
    PcuCompoundArithmeticPolicy,
    PcuPrecisionPolicy,
    PcuNumericalOptions,
    PcuFloatUnderflowPolicy,
    PcuReproducibility,
    PcuI256,
    PcuU256,
    PcuI512,
    PcuU512,
};
const fn device() -> PcuDeviceIdentity {
    PcuDeviceIdentity::from_device_ref(PcuObjectRef {
        provider: PcuProviderId(3),
        generation: 1,
        kind: PcuObjectKind::Device,
        id: 0,
    })
    .unwrap()
}
fn offers<T: PcuCheckedInteger>(ordinal: u32) {
    let bindings = source::doubled_bindings::<T>();
    let reject = source::doubled_ir::<T, 65>(&bindings).unwrap();
    let bindings_clamp = source::doubled_clamp_bindings::<T>();
    let clamp = source::doubled_clamp_ir::<T, 65>(&bindings_clamp).unwrap();
    let assessor = PcuCpuIntegerOffers::<T>::new(device(), PcuExecutorId(0));
    for (offset, builder) in [(0, &reject), (3, &clamp)] {
        for mode in [PcuNumericalMode::Boundary, PcuNumericalMode::Strict] {
            for compound in [
                PcuCompoundArithmeticPolicy::Checked,
                PcuCompoundArithmeticPolicy::BackendDefined,
            ] {
                for precision in [
                    PcuPrecisionPolicy::Preserve,
                    PcuPrecisionPolicy::BackendOptimized,
                ] {
                    for underflow in [
                        PcuFloatUnderflowPolicy::IeeeAfterRounding,
                        PcuFloatUnderflowPolicy::RejectSubnormalResult,
                        PcuFloatUnderflowPolicy::AllowGradualUnderflow,
                    ] {
                        let mut kernel = builder.ir();
                        kernel.numerical_requirements.numerical_mode = mode;
                        kernel.numerical_requirements.float_underflow = underflow;
                        kernel.numerical_requirements.numerical_options = PcuNumericalOptions {
                            compound_arithmetic: compound,
                            precision,
                            ..Default::default()
                        };
                        let mut output = [None; 2];
                        let mut request = PcuImplementationRequest {
                            device: device(),
                            executor: PcuExecutorId(0),
                            requirements: kernel.numerical_requirements,
                            boundary: PcuCostBoundary::Host,
                            operation: &kernel,
                        };
                        assert_eq!(assessor.implementation_offers(&request, &mut output), Ok(1));
                        let offer = output[0].unwrap();
                        assert_eq!(offer.implementation.local_id, 2048 + ordinal * 6 + offset);
                        assert_eq!(offer.implementation.revision, 1);
                        assert_eq!(offer.requirements, request.requirements);
                        assert_eq!(offer.workspace_bytes, Some(0));
                        assert!(output[1].is_none());
                        request.boundary = PcuCostBoundary::Resident;
                        assert_eq!(assessor.implementation_offers(&request, &mut []), Ok(0));
                        let mut portable = kernel;
                        portable
                            .numerical_requirements
                            .numerical_options
                            .reproducibility = PcuReproducibility::PortableV1;
                        request.operation = &portable;
                        request.requirements = portable.numerical_requirements;
                        request.boundary = PcuCostBoundary::Host;
                        let mut portable_offer = [None];
                        assert_eq!(
                            assessor.implementation_offers(&request, &mut portable_offer),
                            Ok(1)
                        );
                        assert_eq!(
                            portable_offer[0].unwrap().implementation.local_id,
                            4096 + ordinal * 6 + offset
                        );
                        assert_eq!(
                            portable_offer[0].unwrap().requirements,
                            request.requirements
                        );
                        request.requirements.numerical_options.reproducibility =
                            PcuReproducibility::Unspecified;
                        assert_eq!(assessor.implementation_offers(&request, &mut []), Ok(0));
                    }
                }
            }
        }
    }
}
#[test]
fn all_fourteen_exact_offers_all_permissions_underflow_and_portable() {
    offers::<i8>(0);
    offers::<u8>(1);
    offers::<i16>(2);
    offers::<u16>(3);
    offers::<i32>(4);
    offers::<u32>(5);
    offers::<i64>(6);
    offers::<u64>(7);
    offers::<i128>(8);
    offers::<u128>(9);
    offers::<PcuI256>(10);
    offers::<PcuU256>(11);
    offers::<PcuI512>(12);
    offers::<PcuU512>(13);
}
#[test]
fn repeated_extent_and_unused_declaration_are_cold_frozen() {
    let backend = PcuCpuCheckedInteger::<u32>::new();
    let bindings = source::independent_bindings::<u32>();
    let kernel = source::independent_ir::<u32, 65>(&bindings).unwrap();
    let mut plan = backend.prepare_host_kernel(&kernel.ir()).unwrap();
    assert_eq!(plan.argument_count(), 2);
    assert_eq!(plan.local_id(), 2079);
    assert_eq!(plan.implementation_revision(), 1);
    let mut output = [77; 68];
    assert_eq!(
        plan.call(&mut [
            PcuHostArgument::read(PcuBindingRef::new(0, 0), &[2_u32]),
            PcuHostArgument::read_write(PcuBindingRef::new(0, 1), &mut output)
        ]),
        Err(PcuCpuCheckedIntegerError::InvalidArguments)
    );
    assert_eq!(output, [77; 68]);
    let input = std::array::from_fn::<_, 65, _>(|lane| u32::try_from(lane + 2).unwrap());
    plan.call(&mut [
        PcuHostArgument::read(PcuBindingRef::new(0, 0), &input),
        PcuHostArgument::read_write(PcuBindingRef::new(0, 1), &mut output),
    ])
    .unwrap();
    assert_eq!(
        output[..65],
        std::array::from_fn::<_, 65, _>(|lane| u32::try_from(lane).unwrap())
    );
    assert_eq!(output[65..], [77; 3]);
    let bindings = source::squared_bindings::<u32>();
    let kernel = source::squared_ir::<u32, 65>(&bindings).unwrap();
    let mut square = backend.prepare_host_kernel(&kernel.ir()).unwrap();
    assert_eq!(square.argument_count(), 3);
    assert_eq!(square.local_id(), 2080);
    square
        .call(&mut [
            PcuHostArgument::read(PcuBindingRef::new(0, 0), &[] as &[u32]),
            PcuHostArgument::read_write(PcuBindingRef::new(0, 1), &mut output),
            PcuHostArgument::read(PcuBindingRef::new(0, 2), &[3_u32; 65]),
        ])
        .unwrap();
    assert_eq!(output[..65], [9; 65]);
    assert_eq!(output[65..], [77; 3]);
    let bindings = source::reordered_bindings::<u32>();
    let kernel = source::reordered_ir::<u32, 65>(&bindings).unwrap();
    let legacy = backend.prepare_host_kernel(&kernel.ir()).unwrap();
    assert_eq!(legacy.local_id(), 20);
    assert_eq!(legacy.implementation_revision(), 2);
}
