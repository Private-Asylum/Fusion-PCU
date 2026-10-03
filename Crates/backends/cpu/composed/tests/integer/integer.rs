//! Genuine scalar SSA integer maps, independent primitive endpoint and small-value oracles.
use super::*;
use fusion_pcu::PcuCheckedInteger;
#[rustfmt::skip]
use fusion_pcu::{
    PcuCompoundArithmeticPolicy,
    PcuCostBoundary,
    PcuDeviceIdentity,
    PcuExecutorId,
    PcuImplementationOffers,
    PcuImplementationRequest,
    PcuNumericalMode,
    PcuObjectKind,
    PcuObjectRef,
    PcuPrecisionPolicy,
    PcuProviderId,
};

#[pcu(crate_path=::pcu_facade,invocations=7)]
fn integer_composed<T: PcuCheckedInteger>(input: &[T], seed: &T, output: &mut [T]) {
    let id = context.global_invocation_id;
    output[id] = ((input[id] + *seed) * input[id]) - input[id];
}

fn offers<T: PcuCheckedInteger>(id: u32, input: &[T; 7], expected: &[T; 7], seed: T) {
    let device = PcuDeviceIdentity::from_device_ref(PcuObjectRef {
        provider: PcuProviderId(3),
        generation: 1,
        kind: PcuObjectKind::Device,
        id: 0,
    })
    .unwrap();
    let executor = PcuExecutorId(0);
    let backend = crate::PcuCpuHostBackend::scalar();
    let offers = crate::PcuCpuHostOffers::new(backend, device, executor);
    let bindings = integer_composed_bindings::<T>();
    integer_composed_ir::<T>(&bindings)
        .unwrap()
        .with_ir(|base| {
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
                            PcuFloatUnderflowPolicy::AllowGradualUnderflow,
                            PcuFloatUnderflowPolicy::RejectSubnormalResult,
                        ] {
                            for range in [PcuRangePolicy::Reject, PcuRangePolicy::Clamp] {
                                let mut kernel = *base;
                                kernel.numerical_requirements.numerical_mode = mode;
                                kernel
                                    .numerical_requirements
                                    .numerical_options
                                    .compound_arithmetic = compound;
                                kernel.numerical_requirements.numerical_options.precision =
                                    precision;
                                kernel.numerical_requirements.float_underflow = underflow;
                                kernel.numerical_requirements.range_policy = range;
                                // Header changes do not rewrite lexical instruction policy.
                                let request = PcuImplementationRequest {
                                    device,
                                    executor,
                                    boundary: PcuCostBoundary::Host,
                                    operation: &kernel,
                                    requirements: kernel.numerical_requirements,
                                };
                                let mut slots = [None];
                                assert_eq!(
                                    offers.implementation_offers(&request, &mut slots),
                                    Ok(1)
                                );
                                let offer = slots[0].unwrap();
                                offer.validate_request(&request).unwrap();
                                assert_eq!(offer.implementation.local_id, id);
                                assert_eq!(offer.implementation.revision, 1);
                                publication(&kernel, input, expected, seed);
                                let mut wrong = request.requirements;
                                wrong.range_policy = if range == PcuRangePolicy::Reject {
                                    PcuRangePolicy::Clamp
                                } else {
                                    PcuRangePolicy::Reject
                                };
                                assert_eq!(
                                    offers.implementation_offers(
                                        &PcuImplementationRequest {
                                            requirements: wrong,
                                            ..request
                                        },
                                        &mut []
                                    ),
                                    Ok(0)
                                );
                                assert_eq!(
                                    offers.implementation_offers(
                                        &PcuImplementationRequest {
                                            boundary: PcuCostBoundary::Resident,
                                            ..request
                                        },
                                        &mut []
                                    ),
                                    Ok(0)
                                );
                            }
                        }
                    }
                }
            }
        });
}

#[pcu(crate_path=::pcu_facade,invocations=7)]
fn unused_sub<T: PcuCheckedInteger>(input: &[T], seed: &T, output: &mut [T]) {
    let id = context.global_invocation_id;
    let saved = input[id];
    let _must_check = saved - *seed;
    output[id] = saved;
}

#[test]
fn ten_primitive_integer_composition_and_unused_underflow() {
    let backend = crate::PcuCpuHostBackend::scalar();
    macro_rules! width {
        ($ty:ty, $id:expr) => {{
            let mut prepared = integer_composed_prepare::<$ty, _>(&backend).unwrap();
            let seed = <$ty>::try_from(1_u8).unwrap();
            let mut output = [<$ty>::try_from(17_u8).unwrap(); 10];
            let input = [1_u8, 2, 3, 4, 5, 6, 7].map(|value| <$ty>::try_from(value).unwrap());
            prepared(&input, &seed, &mut output).unwrap();
            assert_eq!(output[..7], [1, 4, 9, 16, 25, 36, 49]);
            assert_eq!(output[7..], [17; 3]);
            let expected =
                [1_u8, 4, 9, 16, 25, 36, 49].map(|value| <$ty>::try_from(value).unwrap());
            offers::<$ty>($id, &input, &expected, seed);
            let bindings = integer_composed_bindings::<$ty>();
            integer_composed_ir::<$ty>(&bindings)
                .unwrap()
                .with_ir(|kernel| {
                    let plan = prepare_erased(kernel).unwrap();
                    assert_eq!(plan.local_id(), $id);
                    assert_eq!(plan.requirements(), kernel.numerical_requirements);
                    let mut portable = *kernel;
                    portable
                        .numerical_requirements
                        .numerical_options
                        .reproducibility = PcuReproducibility::PortableV1;
                    assert_eq!(
                        prepare_erased(&portable).unwrap_err(),
                        PcuCpuComposedMapError::UnsupportedProfile
                    );
                });
            let mut unused = unused_sub_prepare::<$ty, _>(&backend).unwrap();
            let old = output;
            assert_eq!(
                unused(&[<$ty>::MIN; 7], &seed, &mut output)
                    .unwrap_err()
                    .fault(),
                Some(PcuExecutionFault {
                    invocation_id: 0,
                    kind: PcuExecutionFaultKind::ArithmeticUnderflow,
                    recovered: false,
                })
            );
            assert_eq!(output, old);
            unused(&input, &seed, &mut output).unwrap();
            assert_eq!(output[..7], input);
            assert_eq!(output[7..], [17; 3]);
        }};
    }
    width!(i8, 19712);
    width!(u8, 19713);
    width!(i16, 19714);
    width!(u16, 19715);
    width!(i32, 19716);
    width!(u32, 19717);
    width!(i64, 19718);
    width!(u64, 19719);
    width!(i128, 19720);
    width!(u128, 19721);
}

fn publication<T: PcuCheckedInteger>(
    kernel: &PcuDispatchKernelIr<'_>,
    input: &[T; 7],
    expected: &[T; 7],
    seed: T,
) {
    let mut prepared = crate::PcuCpuHostBackend::scalar()
        .prepare_host_kernel(kernel)
        .unwrap();
    let mut output = [seed; 10];
    prepared
        .call(&mut [
            PcuHostArgument::read(kernel.bindings[0].reference(), input),
            PcuHostArgument::read(kernel.bindings[1].reference(), &[seed]),
            PcuHostArgument::read_write(kernel.bindings[2].reference(), &mut output),
        ])
        .unwrap();
    for (actual, expected) in output[..7].iter().zip(expected) {
        assert_eq!(actual.encode_le().as_ref(), expected.encode_le().as_ref());
    }
    for tail in &output[7..] {
        assert_eq!(tail.encode_le().as_ref(), seed.encode_le().as_ref());
    }
}
