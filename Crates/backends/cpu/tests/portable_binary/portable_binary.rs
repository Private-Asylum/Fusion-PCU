//! Actual requested `PortableV1` conformance, not relabeled normal execution.
#[path = "../low_precision/oracle/oracle.rs"]
#[allow(dead_code)] // This slice uses the independent Reject oracle.
mod oracle;
#[path = "../low_precision/proof/proof.rs"]
mod proof;
#[path = "source/source.rs"]
#[allow(dead_code)] // All source ops run in benchmark; policies/layouts are proved here.
mod source;
use oracle::Low;
#[rustfmt::skip]
use fusion_pcu_cpu::{
    PcuCpuCheckedBinary,
    PcuCpuHostBackend,
    PcuCpuHostOffers,
};
#[rustfmt::skip]
use pcu_facade::{
    global,
    PcuBf16Bits,
    PcuBindingRef,
    PcuCompoundArithmeticPolicy,
    PcuCostBoundary,
    PcuDeviceIdentity,
    PcuDispatchFloatBinaryOp,
    PcuExecutorId,
    PcuExecutionFaultKind,
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
    PcuRangePolicy,
    PcuReproducibility,
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
fn exhaustive<T: Low>(all_pairs: bool) {
    let bindings = source::add_bindings::<T>();
    let builder = source::add_ir::<T, 1>(&bindings).unwrap();
    assert_eq!(
        builder
            .ir()
            .numerical_requirements
            .numerical_options
            .reproducibility,
        PcuReproducibility::PortableV1
    );
    proof::check::<T>(all_pairs, builder.ir());
}
#[test]
fn portable_exhaustive_e4m3fn() {
    exhaustive::<PcuF8E4M3FnBits>(true);
}
#[test]
fn portable_exhaustive_e5m2() {
    exhaustive::<PcuF8E5M2Bits>(true);
}
#[test]
fn portable_all_f16_encodings() {
    exhaustive::<PcuF16Bits>(false);
}
#[test]
fn portable_all_bf16_encodings() {
    exhaustive::<PcuBf16Bits>(false);
}
#[allow(clippy::too_many_lines)] // Closed operation/policy/permission matrix with exact IDs and executable exclusions.
fn offers<T: Low>(base: u32) {
    let device = PcuDeviceIdentity::from_device_ref(PcuObjectRef {
        provider: pcu_facade::PcuProviderId(3),
        generation: 7,
        kind: PcuObjectKind::Device,
        id: 0,
    })
    .unwrap();
    let backend = PcuCpuHostBackend::scalar();
    let offers = PcuCpuHostOffers::new(backend, device, PcuExecutorId(0));
    let bindings = source::add_bindings::<T>();
    let builder = source::add_ir::<T, 3>(&bindings).unwrap();
    let template = builder.ir();
    let mut operations = template.ops.to_vec();
    for (offset, op) in OPS.into_iter().enumerate() {
        for policy in POLICIES {
            for operation in &mut operations {
                if let pcu_facade::PcuDispatchOp::Data(
                    pcu_facade::PcuDispatchDataOp::CheckedFloatBinary {
                        op: selected,
                        underflow_policy,
                        ..
                    },
                ) = operation
                {
                    *selected = op;
                    *underflow_policy = policy;
                }
            }
            for mode in [PcuNumericalMode::Boundary, PcuNumericalMode::Strict] {
                for compound in [
                    PcuCompoundArithmeticPolicy::Checked,
                    PcuCompoundArithmeticPolicy::BackendDefined,
                ] {
                    for precision in [
                        PcuPrecisionPolicy::Preserve,
                        PcuPrecisionPolicy::BackendOptimized,
                    ] {
                        let mut kernel = pcu_facade::PcuDispatchKernelIr {
                            ops: &operations,
                            ..template
                        };
                        kernel.numerical_requirements.float_underflow = policy;
                        kernel.numerical_requirements.numerical_mode = mode;
                        kernel
                            .numerical_requirements
                            .numerical_options
                            .compound_arithmetic = compound;
                        kernel.numerical_requirements.numerical_options.precision = precision;
                        let prepared = PcuCpuCheckedBinary::<T>::new()
                            .prepare_host_kernel(&kernel)
                            .unwrap();
                        assert_eq!(prepared.reproducibility(), PcuReproducibility::PortableV1);
                        assert_eq!(
                            backend
                                .prepare_host_kernel(&kernel)
                                .unwrap()
                                .argument_count(),
                            3
                        );
                        let mut request = PcuImplementationRequest {
                            device,
                            executor: PcuExecutorId(0),
                            operation: &kernel,
                            requirements: kernel.numerical_requirements,
                            boundary: PcuCostBoundary::Host,
                        };
                        let mut output = [None];
                        assert_eq!(offers.implementation_offers(&request, &mut output), Ok(1));
                        let offer = output[0].unwrap();
                        assert_eq!(
                            (offer.implementation.local_id, offer.implementation.revision),
                            (base + u32::try_from(offset).unwrap(), 2)
                        );
                        assert_eq!(offer.requirements, kernel.numerical_requirements);
                        assert_eq!(offer.workspace_bytes, Some(0));
                        request.requirements.numerical_options.reproducibility =
                            PcuReproducibility::Unspecified;
                        assert_eq!(offers.implementation_offers(&request, &mut output), Ok(0));
                        request.requirements = kernel.numerical_requirements;
                        request.boundary = PcuCostBoundary::Resident;
                        assert_eq!(offers.implementation_offers(&request, &mut output), Ok(0));
                        kernel.numerical_requirements.range_policy = PcuRangePolicy::Clamp;
                        assert!(backend.prepare_host_kernel(&kernel).is_err());
                        assert!(
                            PcuCpuCheckedBinary::<T>::new()
                                .prepare_host_kernel(&kernel)
                                .is_err()
                        );
                    }
                }
            }
        }
    }
}
#[test]
fn exact_portable_offers_and_closed_negative_profiles() {
    offers::<PcuF16Bits>(224);
    offers::<PcuBf16Bits>(228);
    offers::<PcuF8E4M3FnBits>(232);
    offers::<PcuF8E5M2Bits>(236);
}
fn source_routes<T: Low>() {
    let one = T::from_bits(u16::try_from(T::FORMAT.bias).unwrap() << T::FORMAT.fraction);
    let two = T::from_bits(one.bits() + (1 << T::FORMAT.fraction));
    let sentinel = T::from_bits(T::FORMAT.max);
    let left = [one, two, T::from_bits(T::FORMAT.sign), one, two, one, two];
    let right = [one; 7];
    let mut output = [sentinel; 9];
    source::grid::<T, 7>(&left, &right, &mut output).unwrap();
    assert_eq!(output[..7], left);
    assert_eq!(output[7..], [sentinel; 2]);
    let before = output;
    let mut bad = right;
    bad[4] = T::from_bits(0);
    assert!(
        matches!(source::grid::<T, 7>(&left, &bad, &mut output), Err(global::PcuExecutionError::ArithmeticFault(fault)) if !fault.recovered && fault.invocation_id == 4 && fault.kind == PcuExecutionFaultKind::DivideByZero)
    );
    assert_eq!(output, before);
    source::grid::<T, 7>(&left, &right, &mut output).unwrap();
    source::scale::<T, 7>(&left, &one, &mut output).unwrap();
    assert_eq!(output[..7], left);
    source::strict::<T, 7>(&[one; 7], &[one; 7], &mut output).unwrap();
    assert_eq!(output[..7], [two; 7]);
    let bindings = source::div_bindings::<T>();
    let builder = source::div_ir::<T, 7>(&bindings).unwrap();
    let mut prepared = PcuCpuCheckedBinary::<T>::new()
        .prepare_host_kernel(&builder.ir())
        .unwrap();
    for permutation in [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ] {
        let mut args = [
            Some(PcuHostArgument::read(PcuBindingRef::new(0, 0), &left)),
            Some(PcuHostArgument::read(PcuBindingRef::new(0, 1), &right)),
            Some(PcuHostArgument::read_write(
                PcuBindingRef::new(0, 2),
                &mut output,
            )),
        ];
        prepared
            .call(&mut permutation.map(|index| args[index].take().unwrap()))
            .unwrap();
        assert_eq!(output[..7], left);
    }
    for repeated in [false, true] {
        let kernel = builder.ir();
        let mut ops = kernel.ops.to_vec();
        if let pcu_facade::PcuDispatchOp::Data(
            pcu_facade::PcuDispatchDataOp::CheckedFloatBinary { lhs, rhs, .. },
        ) = &mut ops[2]
        {
            if repeated {
                *rhs = *lhs;
            } else {
                core::mem::swap(lhs, rhs);
            }
        }
        let changed = pcu_facade::PcuDispatchKernelIr {
            ops: &ops,
            ..kernel
        };
        let mut mapped = PcuCpuCheckedBinary::<T>::new()
            .prepare_host_kernel(&changed)
            .unwrap();
        mapped
            .call(&mut [
                PcuHostArgument::read(PcuBindingRef::new(0, 0), &[two; 7]),
                PcuHostArgument::read(PcuBindingRef::new(0, 1), &[one; 7]),
                PcuHostArgument::read_write(PcuBindingRef::new(0, 2), &mut output),
            ])
            .unwrap();
        let expected = if repeated {
            one
        } else {
            T::from_bits(one.bits() - (1 << T::FORMAT.fraction))
        };
        assert_eq!(output[..7], [expected; 7]);
        assert_eq!(output[7..], [sentinel; 2]);
    }
    let before = output;
    assert!(
        prepared
            .call(&mut [
                PcuHostArgument::read(PcuBindingRef::new(0, 0), &left[..6]),
                PcuHostArgument::read(PcuBindingRef::new(0, 1), &right),
                PcuHostArgument::read_write(PcuBindingRef::new(0, 2), &mut output)
            ])
            .is_err()
    );
    assert_eq!(output, before);
}
#[test]
fn genuine_portable_ordinary_direct_grid_broadcast_strict_transaction() {
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Cpu,
        ..Default::default()
    })
    .unwrap();
    source_routes::<PcuF16Bits>();
    source_routes::<PcuBf16Bits>();
    source_routes::<PcuF8E4M3FnBits>();
    source_routes::<PcuF8E5M2Bits>();
    let mut output = [77.0_f32; 7];
    assert!(matches!(
        source::add::<f32, 7>(&[1.0; 7], &[1.0; 7], &mut output),
        Err(global::PcuExecutionError::UnsupportedNumericalOptions(_))
    ));
    assert_eq!(output.map(f32::to_bits), [77.0_f32; 7].map(f32::to_bits));
}
#[test]
fn legacy_reference_api_does_not_inherit_prepared_portable_admission() {
    use pcu_facade::PcuSynchronousHostDispatchBackend;
    let bindings = source::add_bindings::<PcuF16Bits>();
    let builder = source::add_ir::<PcuF16Bits, 1>(&bindings).unwrap();
    let kernel = builder.ir();
    let one = [PcuF16Bits::from_bits(0x3c00)];
    let mut output = [PcuF16Bits::from_bits(0x1234)];
    let before = output;
    let submission = pcu_facade::PcuDispatchSubmission {
        kernel: &kernel,
        shape: pcu_facade::PcuInvocationShape::invocations(core::num::NonZeroU32::new(1).unwrap()),
    };
    let result = fusion_pcu_cpu::PcuCheckedBinaryReference::<PcuF16Bits>::new().run_host_direct(
        submission,
        &mut [
            pcu_facade::PcuHostScalarBinding {
                target: PcuBindingRef::new(0, 0),
                slice: pcu_facade::PcuHostScalarSlice::Read(&one),
            },
            pcu_facade::PcuHostScalarBinding {
                target: PcuBindingRef::new(0, 1),
                slice: pcu_facade::PcuHostScalarSlice::Read(&one),
            },
            pcu_facade::PcuHostScalarBinding {
                target: PcuBindingRef::new(0, 2),
                slice: pcu_facade::PcuHostScalarSlice::ReadWrite(&mut output),
            },
        ],
        pcu_facade::PcuInvocationParameters::empty(),
    );
    assert_eq!(
        result,
        Err(fusion_pcu_cpu::PcuCpuPreparedBinaryError::UnsupportedProfile)
    );
    assert_eq!(output, before);
}
