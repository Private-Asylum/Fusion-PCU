//! Typed low-precision execution is qualified before cold host offers are admitted.
#[path = "oracle/oracle.rs"]
mod oracle;
#[path = "source/source.rs"]
#[allow(dead_code)] // Ordinary entry points are exercised after exact host admission.
mod source;
use oracle::Low;
#[rustfmt::skip]
use fusion_pcu_cpu::{
    PcuCpuCheckedBinary,
    PcuCpuPreparedBinaryError,
};
#[rustfmt::skip]
use pcu_facade::{
    PcuBindingRef,
    PcuDispatchDataOp,
    PcuDispatchFloatBinaryOp,
    PcuDispatchOp,
    PcuF16Bits,
    PcuBf16Bits,
    PcuF8E4M3FnBits,
    PcuF8E5M2Bits,
    PcuFloatUnderflowPolicy,
    PcuHostArgument,
    PcuHostKernelBackend,
    PcuPreparedHostKernel,
};
const OPS: [PcuDispatchFloatBinaryOp; 4] = [
    PcuDispatchFloatBinaryOp::Add,
    PcuDispatchFloatBinaryOp::Sub,
    PcuDispatchFloatBinaryOp::Mul,
    PcuDispatchFloatBinaryOp::Div,
];
const POLICIES: [PcuFloatUnderflowPolicy; 3] = [
    PcuFloatUnderflowPolicy::IeeeAfterRounding,
    PcuFloatUnderflowPolicy::AllowGradualUnderflow,
    PcuFloatUnderflowPolicy::RejectSubnormalResult,
];
#[path = "proof/proof.rs"]
mod proof;
fn check<T: Low>(exhaustive: bool) {
    let bindings = source::add_bindings::<T>();
    let builder = source::add_ir::<T, 1>(&bindings).unwrap();
    proof::check::<T>(exhaustive, builder.ir());
}

#[test]
fn exhaustive_e4m3fn() {
    check::<PcuF8E4M3FnBits>(true);
}
#[test]
fn exhaustive_e5m2() {
    check::<PcuF8E5M2Bits>(true);
}
#[test]
fn all_binary16_encodings_with_edges() {
    check::<PcuF16Bits>(false);
}
#[test]
fn all_bf16_encodings_with_edges() {
    check::<PcuBf16Bits>(false);
}
fn layouts<T: Low>() {
    let backend = PcuCpuCheckedBinary::<T>::new();
    let one = T::from_bits(u16::try_from(T::FORMAT.bias).unwrap() << T::FORMAT.fraction);
    let two = T::from_bits(one.bits() + (1 << T::FORMAT.fraction));
    let sentinel = T::from_bits(T::FORMAT.max);
    let negative_zero = T::from_bits(T::FORMAT.sign);
    let lhs = [one, two, negative_zero, one, two, one, two];
    let rhs = [one; 7];
    let mut output = [sentinel; 9];
    let mut grid = source::grid_prepare::<T, 7, _>(&backend).unwrap();
    grid(&lhs, &rhs, &mut output).unwrap();
    assert_eq!(output[..7], lhs);
    assert_eq!(output[7..], [sentinel; 2]);
    let before = output;
    let mut bad = rhs;
    bad[4] = T::from_bits(0);
    let error = grid(&lhs, &bad, &mut output).unwrap_err();
    assert!(
        matches!(error, PcuCpuPreparedBinaryError::Fault(fault) if fault.invocation_id == 4 && fault.kind == pcu_facade::PcuExecutionFaultKind::DivideByZero && !fault.recovered)
    );
    assert_eq!(output, before);
    grid(&lhs, &rhs, &mut output).unwrap();
    let mut scale = source::scale_prepare::<T, 7, _>(&backend).unwrap();
    scale(&lhs, &one, &mut output).unwrap();
    assert_eq!(output[..7], lhs);
    let mut strict = source::strict_prepare::<T, 7, _>(&backend).unwrap();
    strict(&[negative_zero; 7], &[negative_zero; 7], &mut output).unwrap();
    assert_eq!(output[..7], [negative_zero; 7]);
    let before = output;
    let mut tiny = [one; 7];
    tiny[3] = T::from_bits(1);
    assert!(strict(&tiny, &[T::from_bits(0); 7], &mut output).is_err());
    assert_eq!(output, before);
    let bindings = source::div_bindings::<T>();
    let builder = source::div_ir::<T, 7>(&bindings).unwrap();
    let mut direct = backend.prepare_host_kernel(&builder.ir()).unwrap();
    for permutation in [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ] {
        let mut arguments = [
            Some(PcuHostArgument::read(PcuBindingRef::new(0, 0), &lhs)),
            Some(PcuHostArgument::read(PcuBindingRef::new(0, 1), &rhs)),
            Some(PcuHostArgument::read_write(
                PcuBindingRef::new(0, 2),
                &mut output,
            )),
        ];
        direct
            .call(&mut permutation.map(|index| arguments[index].take().unwrap()))
            .unwrap();
        assert_eq!(output[..7], lhs);
        assert_eq!(output[7..], [sentinel; 2]);
    }
    let before = output;
    assert!(
        direct
            .call(&mut [
                PcuHostArgument::read(PcuBindingRef::new(0, 0), &lhs[..6]),
                PcuHostArgument::read(PcuBindingRef::new(0, 1), &rhs),
                PcuHostArgument::read_write(PcuBindingRef::new(0, 2), &mut output)
            ])
            .is_err()
    );
    assert_eq!(output, before);
    assert!(
        direct
            .call(&mut [
                PcuHostArgument::read(PcuBindingRef::new(0, 0), &[0_u32; 7]),
                PcuHostArgument::read(PcuBindingRef::new(0, 1), &rhs),
                PcuHostArgument::read_write(PcuBindingRef::new(0, 2), &mut output)
            ])
            .is_err()
    );
    assert_eq!(output, before);
}
#[test]
fn binary16_layouts_transaction_schema() {
    layouts::<PcuF16Bits>();
}
#[test]
fn bf16_layouts_transaction_schema() {
    layouts::<PcuBf16Bits>();
}
#[test]
fn e4m3fn_layouts_transaction_schema() {
    layouts::<PcuF8E4M3FnBits>();
}
#[test]
fn e5m2_layouts_transaction_schema() {
    layouts::<PcuF8E5M2Bits>();
}
#[allow(clippy::too_many_lines)] // One cold matrix keeps format/operation/policy and permission negatives adjacent.
fn offers<T: Low>(base: u32) {
    use fusion_pcu_cpu::{PcuCpuHostBackend, PcuCpuHostOffers};
    #[rustfmt::skip]
    use pcu_facade::{
        PcuCostBoundary,
        PcuDeviceIdentity,
        PcuExecutorId,
        PcuImplementationOffers,
        PcuImplementationRequest,
        PcuImplementationRequirements,
        PcuNumericalOptions,
        PcuNumericalMode,
        PcuObjectKind,
        PcuObjectRef,
        PcuProviderId,
        PcuRangePolicy,
        PcuReproducibility,
        PcuPrecisionPolicy,
        PcuCompoundArithmeticPolicy,
    };
    let device = PcuDeviceIdentity::from_device_ref(PcuObjectRef {
        provider: PcuProviderId(3),
        generation: 7,
        kind: PcuObjectKind::Device,
        id: 0,
    })
    .unwrap();
    let provider = PcuCpuHostOffers::new(PcuCpuHostBackend::scalar(), device, PcuExecutorId(0));
    let bindings = source::add_bindings::<T>();
    let builder = source::add_ir::<T, 3>(&bindings).unwrap();
    let kernel = builder.ir();
    let mut operations = kernel.ops.to_vec();
    for (offset, op) in OPS.into_iter().enumerate() {
        for policy in POLICIES {
            for operation in &mut operations {
                if let PcuDispatchOp::Data(PcuDispatchDataOp::CheckedFloatBinary {
                    op: selected,
                    underflow_policy,
                    ..
                }) = operation
                {
                    *selected = op;
                    *underflow_policy = policy;
                }
            }
            let kernel = pcu_facade::PcuDispatchKernelIr {
                ops: &operations,
                ..kernel
            };
            for mode in [PcuNumericalMode::Boundary, PcuNumericalMode::Strict] {
                let kernel = pcu_facade::PcuDispatchKernelIr {
                    numerical_requirements: PcuImplementationRequirements {
                        float_underflow: policy,
                        numerical_mode: mode,
                        ..Default::default()
                    },
                    ..kernel
                };
                let mut request = PcuImplementationRequest {
                    device,
                    executor: PcuExecutorId(0),
                    operation: &kernel,
                    requirements: PcuImplementationRequirements {
                        float_underflow: policy,
                        numerical_mode: mode,
                        ..Default::default()
                    },
                    boundary: PcuCostBoundary::Host,
                };
                let mut output = [None];
                assert_eq!(provider.implementation_offers(&request, &mut output), Ok(1));
                let offer = output[0].unwrap();
                assert_eq!(
                    (offer.implementation.local_id, offer.implementation.revision),
                    (base + u32::try_from(offset).unwrap(), 2)
                );
                assert_eq!(offer.requirements, request.requirements);
                assert_eq!(offer.workspace_bytes, Some(0));
                for numerical_options in [
                    PcuNumericalOptions {
                        reproducibility: PcuReproducibility::PortableV1,
                        ..Default::default()
                    },
                    PcuNumericalOptions {
                        precision: PcuPrecisionPolicy::BackendOptimized,
                        ..Default::default()
                    },
                    PcuNumericalOptions {
                        compound_arithmetic: PcuCompoundArithmeticPolicy::BackendDefined,
                        ..Default::default()
                    },
                ] {
                    request.requirements.numerical_options = numerical_options;
                    assert_eq!(provider.implementation_offers(&request, &mut output), Ok(0));
                }
                request.requirements.numerical_options = PcuNumericalOptions::default();
                request.boundary = PcuCostBoundary::Resident;
                assert_eq!(provider.implementation_offers(&request, &mut output), Ok(0));
                request.boundary = PcuCostBoundary::Host;
                request.requirements.range_policy = PcuRangePolicy::Clamp;
                assert_eq!(provider.implementation_offers(&request, &mut output), Ok(0));
                request.requirements.range_policy = PcuRangePolicy::Reject;
                request.requirements.float_underflow =
                    if policy == PcuFloatUnderflowPolicy::IeeeAfterRounding {
                        PcuFloatUnderflowPolicy::AllowGradualUnderflow
                    } else {
                        PcuFloatUnderflowPolicy::IeeeAfterRounding
                    };
                assert_eq!(provider.implementation_offers(&request, &mut output), Ok(0));
            }
        }
    }
}
#[test]
fn exact_low_format_offers() {
    offers::<PcuF16Bits>(160);
    offers::<PcuBf16Bits>(164);
    offers::<PcuF8E4M3FnBits>(168);
    offers::<PcuF8E5M2Bits>(172);
}
#[cfg(feature = "source-low-precision")]
fn ordinary<T: Low>() {
    let one = T::from_bits(u16::try_from(T::FORMAT.bias).unwrap() << T::FORMAT.fraction);
    let two = T::from_bits(one.bits() + (1 << T::FORMAT.fraction));
    let sentinel = T::from_bits(T::FORMAT.max);
    let mut output = [sentinel; 5];
    for (op, entry) in [
        (
            PcuDispatchFloatBinaryOp::Add,
            source::add::<T, 3> as fn(&[T], &[T], &mut [T]) -> _,
        ),
        (PcuDispatchFloatBinaryOp::Sub, source::sub::<T, 3>),
        (PcuDispatchFloatBinaryOp::Mul, source::mul::<T, 3>),
        (PcuDispatchFloatBinaryOp::Div, source::div::<T, 3>),
    ] {
        entry(&[two; 3], &[one; 3], &mut output).unwrap();
        let expected = T::FORMAT
            .evaluate(
                two.bits(),
                one.bits(),
                op,
                PcuFloatUnderflowPolicy::IeeeAfterRounding,
            )
            .unwrap();
        assert_eq!(output[..3], [T::from_bits(expected); 3]);
        assert_eq!(output[3..], [sentinel; 2]);
        let before = output;
        assert!(
            entry(
                &[two, T::from_bits(T::FORMAT.sign - 1), two],
                &[one; 3],
                &mut output
            )
            .is_err()
        );
        assert_eq!(output, before);
        entry(&[two; 3], &[one; 3], &mut output).unwrap();
    }
    source::grid::<T, 3>(&[two; 3], &[one; 3], &mut output).unwrap();
    source::scale::<T, 3>(&[two; 3], &one, &mut output).unwrap();
    source::strict::<T, 3>(&[one; 3], &[one; 3], &mut output).unwrap();
}
#[cfg(feature = "source-low-precision")]
#[test]
fn ordinary_four_formats_keep_transaction_and_exact_source_policies() {
    pcu_facade::global::configure(pcu_facade::global::PcuExecutionPolicy {
        backend: pcu_facade::global::PcuBackendChoice::Cpu,
        ..Default::default()
    })
    .unwrap();
    ordinary::<PcuF16Bits>();
    ordinary::<PcuBf16Bits>();
    ordinary::<PcuF8E4M3FnBits>();
    ordinary::<PcuF8E5M2Bits>();
}
#[allow(clippy::too_many_lines)] // Complete four-axis cold matrix and successful checked execution are one proof.
fn tuple_permissions<T: Low>() {
    #[rustfmt::skip]
    use pcu_facade::{
        PcuCompoundArithmeticPolicy,
        PcuPrecisionPolicy,
        PcuReproducibility,
        PcuNumericalMode,
        PcuCostBoundary,
        PcuDeviceIdentity,
        PcuExecutorId,
        PcuObjectRef,
        PcuObjectKind,
        PcuProviderId,
        PcuImplementationOffers,
        PcuImplementationRequest,
    };
    let bindings = source::mul_bindings::<T>();
    let builder = source::mul_ir::<T, 1>(&bindings).unwrap();
    let kernel = builder.ir();
    let device = PcuDeviceIdentity::from_device_ref(PcuObjectRef {
        provider: PcuProviderId(3),
        generation: 7,
        kind: PcuObjectKind::Device,
        id: 0,
    })
    .unwrap();
    let backend = fusion_pcu_cpu::PcuCpuHostBackend::scalar();
    let offers = fusion_pcu_cpu::PcuCpuHostOffers::new(backend, device, PcuExecutorId(0));
    for compound in [
        PcuCompoundArithmeticPolicy::Checked,
        PcuCompoundArithmeticPolicy::BackendDefined,
    ] {
        for precision in [
            PcuPrecisionPolicy::Preserve,
            PcuPrecisionPolicy::BackendOptimized,
        ] {
            for mode in [PcuNumericalMode::Boundary, PcuNumericalMode::Strict] {
                for reproducibility in [
                    PcuReproducibility::Unspecified,
                    PcuReproducibility::PortableV1,
                ] {
                    let mut kernel = kernel;
                    kernel.numerical_requirements.numerical_mode = mode;
                    kernel
                        .numerical_requirements
                        .numerical_options
                        .compound_arithmetic = compound;
                    kernel.numerical_requirements.numerical_options.precision = precision;
                    kernel
                        .numerical_requirements
                        .numerical_options
                        .reproducibility = reproducibility;
                    let mut request = PcuImplementationRequest {
                        device,
                        executor: PcuExecutorId(0),
                        operation: &kernel,
                        requirements: kernel.numerical_requirements,
                        boundary: PcuCostBoundary::Host,
                    };
                    let mut output = [None];

                    let mut prepared = PcuCpuCheckedBinary::<T>::new()
                        .prepare_host_kernel(&kernel)
                        .unwrap();
                    assert_eq!(
                        backend
                            .prepare_host_kernel(&kernel)
                            .unwrap()
                            .argument_count(),
                        3
                    );
                    assert_eq!(offers.implementation_offers(&request, &mut output), Ok(1));
                    assert_eq!(
                        output[0].unwrap().requirements,
                        kernel.numerical_requirements
                    );
                    let one =
                        T::from_bits(u16::try_from(T::FORMAT.bias).unwrap() << T::FORMAT.fraction);
                    let mut value = [T::from_bits(0); 2];
                    prepared
                        .call(&mut [
                            PcuHostArgument::read(PcuBindingRef::new(0, 0), &[one]),
                            PcuHostArgument::read(PcuBindingRef::new(0, 1), &[one]),
                            PcuHostArgument::read_write(PcuBindingRef::new(0, 2), &mut value),
                        ])
                        .unwrap();
                    assert_eq!(value, [one, T::from_bits(0)]);
                    request.requirements.numerical_options.reproducibility =
                        if reproducibility == PcuReproducibility::PortableV1 {
                            PcuReproducibility::Unspecified
                        } else {
                            PcuReproducibility::PortableV1
                        };
                    assert_eq!(offers.implementation_offers(&request, &mut output), Ok(0));
                }
            }
        }
    }
}
#[test]
fn complete_numerical_tuple_admits_only_qualified_portable_maps() {
    tuple_permissions::<PcuF16Bits>();
    tuple_permissions::<PcuBf16Bits>();
    tuple_permissions::<PcuF8E4M3FnBits>();
    tuple_permissions::<PcuF8E5M2Bits>();
}
