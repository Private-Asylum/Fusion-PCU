//! Prepared/source F64 Neg laws, across every runtime-admitted instruction family.

#[rustfmt::skip]
use fusion_pcu_cpu::{
    PcuCpuCheckedNeg,
    PcuCpuHostArgumentError,
    PcuCpuHostBackend,
    PcuCpuHostError,
    PcuCpuHostOfferError,
    PcuCpuHostOffers,
    PcuCpuImplementation,
    PcuCpuNegOfferError,
    PcuCpuNegOffers,
    PcuCpuPreparedNeg,
    PcuCpuPreparedNegError,
    PcuCpuProcessor,
};
#[rustfmt::skip]
use pcu_facade::{
    PcuBindingRef,
    PcuCheckedFloat,
    PcuCompoundArithmeticPolicy,
    PcuCostBoundary,
    PcuDeviceIdentity,
    PcuDispatchControlOp,
    PcuDispatchDataOp,
    PcuDispatchFloatUnaryOp,
    PcuDispatchIndex,
    PcuDispatchKernelIr,
    PcuDispatchOp,
    PcuDispatchValueId,
    PcuExecutionFault,
    PcuExecutionFaultKind,
    PcuExecutorId,
    PcuFloatUnderflowPolicy,
    PcuHostArgument,
    PcuHostKernelBackend,
    PcuImplementationCost,
    PcuImplementationOffers,
    PcuImplementationRequest,
    PcuImplementationRequirements,
    PcuNumericalMode,
    PcuObjectKind,
    PcuObjectRef,
    PcuPrecisionPolicy,
    PcuPreparedHostKernel,
    PcuProviderId,
    PcuRangePolicy,
    PcuReproducibility,
    PcuScalarType,
    PcuValueType,
};

#[path = "source/source.rs"]
mod source;

const COUNT: usize = 17;
const INPUT: PcuBindingRef = PcuBindingRef::new(0, 0);
const OUTPUT: PcuBindingRef = PcuBindingRef::new(0, 1);

fn backends() -> std::vec::Vec<PcuCpuCheckedNeg> {
    let mut backends = std::vec![PcuCpuCheckedNeg::scalar()];
    #[cfg(feature = "std")]
    {
        let processor = PcuCpuProcessor::detect();
        for implementation in [
            PcuCpuImplementation::Sse2,
            PcuCpuImplementation::Avx2,
            PcuCpuImplementation::Avx,
            PcuCpuImplementation::Avx512,
            PcuCpuImplementation::Neon,
        ] {
            if let Ok(backend) = PcuCpuCheckedNeg::new(processor, implementation) {
                backends.push(backend);
            }
        }
    }
    #[cfg(not(feature = "std"))]
    let _ = &mut backends;
    backends
}

fn fault(kind: PcuExecutionFaultKind, invocation: usize) -> PcuCpuPreparedNegError {
    PcuCpuPreparedNegError::Fault(PcuExecutionFault {
        recovered: false,
        kind,
        invocation_id: u64::try_from(invocation).unwrap(),
    })
}

fn invoke(
    prepared: &mut PcuCpuPreparedNeg,
    input: &[f64],
    output: &mut [f64],
) -> Result<(), PcuCpuPreparedNegError> {
    prepared.call(&mut [
        PcuHostArgument::read(INPUT, input),
        PcuHostArgument::read_write(OUTPUT, output),
    ])
}

fn with_kernel(
    grid: bool,
    policy: PcuFloatUnderflowPolicy,
    range: PcuRangePolicy,
    inspect: impl FnOnce(&PcuDispatchKernelIr<'_>),
) {
    let bindings = source::negate_bindings();
    let builder = source::negate_ir::<COUNT>(&bindings).unwrap();
    let original = builder.ir();
    let mut body = original.ops[..3].to_vec();
    for op in &mut body {
        match op {
            PcuDispatchOp::Data(
                PcuDispatchDataOp::BindingLoad { index, .. }
                | PcuDispatchDataOp::BindingStore { index, .. },
            ) => {
                *index = if grid {
                    PcuDispatchIndex::GridStrideId
                } else {
                    PcuDispatchIndex::InvocationId
                };
            }
            PcuDispatchOp::Data(PcuDispatchDataOp::CheckedFloatUnary {
                underflow_policy,
                range_policy,
                ..
            }) => {
                *underflow_policy = policy;
                *range_policy = range;
            }
            _ => unreachable!("canonical source body"),
        }
    }
    let grid_ops = [
        PcuDispatchOp::GridStrideLoop {
            extent: u32::try_from(COUNT).unwrap(),
            body: &body,
        },
        PcuDispatchOp::Control(PcuDispatchControlOp::Return),
    ];
    let direct = [
        body[0],
        body[1],
        body[2],
        PcuDispatchOp::Control(PcuDispatchControlOp::Return),
    ];
    let mut kernel = PcuDispatchKernelIr {
        numerical_requirements: PcuImplementationRequirements {
            float_underflow: policy,
            range_policy: range,
            ..original.numerical_requirements
        },
        ops: if grid { &grid_ops } else { &direct },
        ..original
    };
    if grid {
        kernel.entry.logical_shape[0] = 2;
    }
    inspect(&kernel);
}

fn prepare_policy(
    backend: PcuCpuCheckedNeg,
    grid: bool,
    policy: PcuFloatUnderflowPolicy,
) -> PcuCpuPreparedNeg {
    let mut prepared = None;
    with_kernel(grid, policy, PcuRangePolicy::Reject, |kernel| {
        prepared = Some(backend.prepare_host_kernel(kernel).unwrap());
    });
    prepared.unwrap()
}

fn check_source<const N: usize>(backend: PcuCpuCheckedNeg) {
    let edges = [
        0,
        0x8000_0000_0000_0000,
        1,
        0x8000_0000_0000_0001,
        0x000f_ffff_ffff_ffff,
        0x800f_ffff_ffff_ffff,
        0x0010_0000_0000_0000,
        0x8010_0000_0000_0000,
        0x7fef_ffff_ffff_ffff,
        0xffef_ffff_ffff_ffff,
        0x3ff0_0000_0000_0001,
        0xbff0_0000_0000_0001,
    ];
    let input: std::vec::Vec<_> = (0..N)
        .map(|index| f64::from_bits(edges[index % edges.len()]))
        .collect();
    let mut output = std::vec![31.0; N + 3];
    let mut call = source::negate_prepare::<N, _>(&backend).unwrap();
    call(&input, &mut output).unwrap();
    for (value, actual) in input.iter().zip(&output) {
        assert_eq!(actual.to_bits(), value.pcu_checked_neg().unwrap().to_bits());
    }
    assert_eq!(&output[N..], &[31.0; 3]);
    call(&std::vec![2.0; N], &mut output).unwrap();
    assert_eq!(&output[..N], &std::vec![-2.0; N]);
}

#[test]
fn source_edges_vector_widths_tails_and_changed_inputs_match_core() {
    for backend in backends() {
        check_source::<1>(backend);
        check_source::<2>(backend);
        check_source::<3>(backend);
        check_source::<4>(backend);
        check_source::<5>(backend);
        check_source::<7>(backend);
        check_source::<8>(backend);
        check_source::<9>(backend);
        check_source::<17>(backend);
        check_source::<257>(backend);
    }
}

#[test]
fn source_random_full_width_finite_bits_and_unaligned_subslices_match_core() {
    let mut state = 0x243f_6a88_85a3_08d3_u64;
    let mut input = std::vec![0.0; 4098];
    for value in &mut input {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let bits = if state & 0x7ff0_0000_0000_0000 == 0x7ff0_0000_0000_0000 {
            state ^ 0x0010_0000_0000_0000
        } else {
            state
        };
        *value = f64::from_bits(bits);
    }
    for backend in backends() {
        let mut output = std::vec![43.0; 4101];
        let mut call = source::negate_prepare::<4097, _>(&backend).unwrap();
        call(&input[1..], &mut output[1..]).unwrap();
        for (value, actual) in input[1..].iter().zip(&output[1..]) {
            assert_eq!(actual.to_bits(), value.pcu_checked_neg().unwrap().to_bits());
        }
        assert_eq!(output[0].to_bits(), 43.0_f64.to_bits());
        assert_eq!(&output[4098..], &[43.0; 3]);
    }
}

#[test]
fn every_nonfinite_lane_faults_first_preserves_all_output_and_retries() {
    for backend in backends() {
        let mut call = source::negate_prepare::<COUNT, _>(&backend).unwrap();
        for lane in 0..COUNT {
            for bits in [
                f64::INFINITY.to_bits(),
                f64::NEG_INFINITY.to_bits(),
                0x7ff8_0000_0000_0001,
                0xfff0_0000_0000_0001,
            ] {
                let mut input = [1.0; COUNT];
                input[COUNT - 1] = f64::NAN;
                input[lane] = f64::from_bits(bits);
                let mut output = [19.0_f64; COUNT + 3];
                assert_eq!(
                    call(&input, &mut output),
                    Err(fault(PcuExecutionFaultKind::InvalidFloatingOperand, lane))
                );
                assert_eq!(
                    output.map(f64::to_bits),
                    [19.0_f64; COUNT + 3].map(f64::to_bits)
                );
                call(&[2.0; COUNT], &mut output).unwrap();
                assert_eq!(&output[..COUNT], &[-2.0; COUNT]);
                assert_eq!(&output[COUNT..], &[19.0; 3]);
            }
        }
    }
}

#[test]
fn detached_direct_and_grid_obey_every_policy_and_logical_fault_order() {
    for backend in backends() {
        for grid in [false, true] {
            for policy in [
                PcuFloatUnderflowPolicy::IeeeAfterRounding,
                PcuFloatUnderflowPolicy::AllowGradualUnderflow,
                PcuFloatUnderflowPolicy::RejectSubnormalResult,
            ] {
                let mut prepared = prepare_policy(backend, grid, policy);
                assert_eq!(prepared.scalar_type(), PcuScalarType::F64);
                assert_eq!(prepared.underflow_policy(), policy);
                let mut input = [1.0; COUNT];
                input[3] = f64::from_bits(1);
                input[9] = f64::from_bits(0x8000_0000_0000_0001);
                let mut output = [23.0_f64; COUNT + 2];
                let result = invoke(&mut prepared, &input, &mut output);
                if policy == PcuFloatUnderflowPolicy::RejectSubnormalResult {
                    assert_eq!(
                        result,
                        Err(fault(PcuExecutionFaultKind::ArithmeticUnderflow, 3))
                    );
                    assert_eq!(
                        output.map(f64::to_bits),
                        [23.0_f64; COUNT + 2].map(f64::to_bits)
                    );
                } else {
                    result.unwrap();
                    for (value, actual) in input.iter().zip(&output) {
                        assert_eq!(
                            actual.to_bits(),
                            value.pcu_checked_neg_with_policy(policy).unwrap().to_bits()
                        );
                    }
                    assert_eq!(&output[COUNT..], &[23.0; 2]);
                }
                output.fill(23.0);
                input[7] = f64::NAN;
                let first = if policy == PcuFloatUnderflowPolicy::RejectSubnormalResult {
                    fault(PcuExecutionFaultKind::ArithmeticUnderflow, 3)
                } else {
                    fault(PcuExecutionFaultKind::InvalidFloatingOperand, 7)
                };
                assert_eq!(invoke(&mut prepared, &input, &mut output), Err(first));
                assert_eq!(
                    output.map(f64::to_bits),
                    [23.0_f64; COUNT + 2].map(f64::to_bits)
                );
                invoke(&mut prepared, &[0.0; COUNT], &mut output).unwrap();
                assert_eq!(
                    output[..COUNT]
                        .iter()
                        .map(|v| v.to_bits())
                        .collect::<std::vec::Vec<_>>(),
                    std::vec![1 << 63; COUNT]
                );
                assert_eq!(&output[COUNT..], &[23.0; 2]);
            }
        }
    }
}

#[test]
fn typed_host_schema_rejects_types_duplicates_access_and_short_buffers_before_write() {
    let mut prepared = prepare_policy(
        PcuCpuCheckedNeg::scalar(),
        false,
        PcuFloatUnderflowPolicy::default(),
    );
    let input = [1.0_f64; COUNT];
    let mut output = [37.0_f64; COUNT + 1];
    assert_eq!(
        prepared.call(&mut []),
        Err(PcuCpuPreparedNegError::InvalidArguments)
    );
    assert_eq!(
        prepared.call(&mut [PcuHostArgument::read(INPUT, &input)]),
        Err(PcuCpuPreparedNegError::InvalidArguments)
    );
    assert_eq!(
        prepared.call(&mut [
            PcuHostArgument::read(INPUT, &input),
            PcuHostArgument::read_write(INPUT, &mut output)
        ]),
        Err(PcuCpuPreparedNegError::InvalidArguments)
    );
    assert_eq!(
        prepared.call(&mut [
            PcuHostArgument::read(INPUT, &[1.0_f32; COUNT * 2]),
            PcuHostArgument::read_write(OUTPUT, &mut output)
        ]),
        Err(PcuCpuPreparedNegError::InvalidArguments)
    );
    assert_eq!(
        prepared.call(&mut [
            PcuHostArgument::read(INPUT, &input),
            PcuHostArgument::read(OUTPUT, &input)
        ]),
        Err(PcuCpuPreparedNegError::InvalidArguments)
    );
    assert_eq!(
        invoke(&mut prepared, &input[..COUNT - 1], &mut output),
        Err(PcuCpuPreparedNegError::InvalidArguments)
    );
    assert_eq!(
        invoke(&mut prepared, &input, &mut output[..COUNT - 1]),
        Err(PcuCpuPreparedNegError::InvalidArguments)
    );
    assert_eq!(
        output.map(f64::to_bits),
        [37.0_f64; COUNT + 1].map(f64::to_bits)
    );
    prepared
        .call(&mut [
            PcuHostArgument::read_write(OUTPUT, &mut output),
            PcuHostArgument::read(INPUT, &input),
        ])
        .unwrap();
    assert_eq!(&output[..COUNT], &[-1.0; COUNT]);
    assert_eq!(output[COUNT].to_bits(), 37.0_f64.to_bits());
}

#[test]
fn unified_source_and_schema_retain_f64_details_faults_and_retry() {
    let backend = PcuCpuHostBackend::scalar();
    let mut call = source::negate_prepare::<COUNT, _>(&backend).unwrap();
    let mut output = [29.0_f64; COUNT + 2];
    let error = call(&[1.0; COUNT - 1], &mut output).unwrap_err();
    assert_eq!(
        error,
        PcuCpuHostError::Arguments(PcuCpuHostArgumentError::BufferTooSmall {
            binding: INPUT,
            required_bytes: COUNT * 8,
            actual_bytes: (COUNT - 1) * 8,
        })
    );
    assert_eq!(
        output.map(f64::to_bits),
        [29.0_f64; COUNT + 2].map(f64::to_bits)
    );
    let mut input = [2.0; COUNT];
    input[5] = f64::NAN;
    let error = call(&input, &mut output).unwrap_err();
    assert_eq!(error.fault().unwrap().invocation_id, 5);
    assert_eq!(
        output.map(f64::to_bits),
        [29.0_f64; COUNT + 2].map(f64::to_bits)
    );
    call(&[3.0; COUNT], &mut output).unwrap();
    assert_eq!(&output[..COUNT], &[-3.0; COUNT]);
    assert_eq!(&output[COUNT..], &[29.0; 2]);
    let bindings = source::negate_bindings();
    let builder = source::negate_ir::<COUNT>(&bindings).unwrap();
    let mut prepared = backend.prepare_host_kernel(&builder.ir()).unwrap();
    assert_eq!(
        prepared.call(&mut [
            PcuHostArgument::read(INPUT, &[1_u64; COUNT]),
            PcuHostArgument::read_write(OUTPUT, &mut output)
        ]),
        Err(PcuCpuHostError::Arguments(
            PcuCpuHostArgumentError::TypeMismatch {
                binding: INPUT,
                expected: PcuScalarType::F64,
                actual: PcuScalarType::U64,
            }
        ))
    );
    assert_eq!(&output[..COUNT], &[-3.0; COUNT]);
}

#[test]
fn malformed_ssa_bindings_shapes_and_grid_retain_cold_errors() {
    let backend = PcuCpuCheckedNeg::scalar();
    with_kernel(
        false,
        PcuFloatUnderflowPolicy::default(),
        PcuRangePolicy::Reject,
        |kernel| {
            let mut body = kernel.ops.to_vec();
            if let PcuDispatchOp::Data(PcuDispatchDataOp::CheckedFloatUnary { value, .. }) =
                &mut body[1]
            {
                *value = PcuDispatchValueId(99);
            }
            assert!(matches!(
                backend.prepare_host_kernel(&PcuDispatchKernelIr {
                    ops: &body,
                    ..*kernel
                }),
                Err(PcuCpuPreparedNegError::InvalidValueFlow(_))
            ));
            let bindings = [kernel.bindings[0], kernel.bindings[0]];
            assert!(matches!(
                backend.prepare_host_kernel(&PcuDispatchKernelIr {
                    bindings: &bindings,
                    ..*kernel
                }),
                Err(PcuCpuPreparedNegError::InvalidKernel(_))
            ));
            let mut zero = *kernel;
            zero.entry.logical_shape = [0, 1, 1];
            assert!(matches!(
                backend.prepare_host_kernel(&zero),
                Err(PcuCpuPreparedNegError::InvalidKernel(_))
            ));
            let mut mismatch = kernel.ops.to_vec();
            if let PcuDispatchOp::Data(PcuDispatchDataOp::CheckedFloatUnary {
                value_type, ..
            }) = &mut mismatch[1]
            {
                *value_type = PcuValueType::f32();
            }
            assert!(matches!(
                backend.prepare_host_kernel(&PcuDispatchKernelIr {
                    ops: &mismatch,
                    ..*kernel
                }),
                Err(PcuCpuPreparedNegError::InvalidValueFlow(_))
            ));
        },
    );
    with_kernel(
        true,
        PcuFloatUnderflowPolicy::default(),
        PcuRangePolicy::Reject,
        |kernel| {
            let mut ops = kernel.ops.to_vec();
            if let PcuDispatchOp::GridStrideLoop { extent, .. } = &mut ops[0] {
                *extent = 0;
            }
            assert!(matches!(
                backend.prepare_host_kernel(&PcuDispatchKernelIr {
                    ops: &ops,
                    ..*kernel
                }),
                Err(PcuCpuPreparedNegError::InvalidGridExtent)
            ));
        },
    );
}

#[test]
fn broader_valid_relu_clamp_broadcast_dimensions_and_in_place_reject_cold() {
    let backend = PcuCpuCheckedNeg::scalar();
    for grid in [false, true] {
        with_kernel(
            grid,
            PcuFloatUnderflowPolicy::default(),
            PcuRangePolicy::Clamp,
            |kernel| {
                assert!(matches!(
                    backend.prepare_host_kernel(kernel),
                    Err(PcuCpuPreparedNegError::UnsupportedProfile)
                ));
            },
        );
    }
    with_kernel(
        false,
        PcuFloatUnderflowPolicy::default(),
        PcuRangePolicy::Reject,
        |kernel| {
            let mut relu = kernel.ops.to_vec();
            if let PcuDispatchOp::Data(PcuDispatchDataOp::CheckedFloatUnary { op, .. }) =
                &mut relu[1]
            {
                *op = PcuDispatchFloatUnaryOp::Relu;
            }
            assert!(matches!(
                backend.prepare_host_kernel(&PcuDispatchKernelIr {
                    ops: &relu,
                    ..*kernel
                }),
                Err(PcuCpuPreparedNegError::UnsupportedProfile)
            ));
            let mut broadcast = kernel.ops.to_vec();
            if let PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad { index, .. }) =
                &mut broadcast[0]
            {
                *index = PcuDispatchIndex::BindingElementZero;
            }
            assert!(matches!(
                backend.prepare_host_kernel(&PcuDispatchKernelIr {
                    ops: &broadcast,
                    ..*kernel
                }),
                Err(PcuCpuPreparedNegError::UnsupportedProfile)
            ));
            let mut dimensions = *kernel;
            dimensions.entry.logical_shape = [u32::try_from(COUNT).unwrap(), 2, 1];
            assert!(matches!(
                backend.prepare_host_kernel(&dimensions),
                Err(PcuCpuPreparedNegError::UnsupportedProfile)
            ));
            let mut inplace = kernel.ops.to_vec();
            if let PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore { binding, .. }) =
                &mut inplace[2]
            {
                *binding = INPUT;
            }
            assert!(
                backend
                    .prepare_host_kernel(&PcuDispatchKernelIr {
                        ops: &inplace,
                        ..*kernel
                    })
                    .is_err()
            );
        },
    );
}

const fn identity(generation: u64) -> PcuDeviceIdentity {
    PcuDeviceIdentity::from_device_ref(PcuObjectRef {
        provider: PcuProviderId(3),
        generation,
        kind: PcuObjectKind::Device,
        id: 0,
    })
    .unwrap()
}

#[test]
fn f64_instruction_ids_are_distinct_and_typed_unified_offers_agree() {
    for backend in backends() {
        let processor = PcuCpuProcessor::scalar();
        #[cfg(feature = "std")]
        let processor = {
            let _ = processor;
            PcuCpuProcessor::detect()
        };
        let implementation = {
            let prepared = prepare_policy(backend, false, PcuFloatUnderflowPolicy::default());
            prepared.implementation()
        };
        let typed = PcuCpuNegOffers::new(backend, identity(1), PcuExecutorId(0));
        let unified = PcuCpuHostOffers::new(
            PcuCpuHostBackend::new(processor, implementation).unwrap(),
            identity(1),
            PcuExecutorId(0),
        );
        let local_id = match implementation {
            PcuCpuImplementation::Scalar => 28,
            PcuCpuImplementation::Sse2 => 29,
            PcuCpuImplementation::Avx2 => 30,
            PcuCpuImplementation::Neon => 31,
            PcuCpuImplementation::Avx => 40962,
            PcuCpuImplementation::Avx512 => 40963,
        };
        for grid in [false, true] {
            for underflow in [
                PcuFloatUnderflowPolicy::IeeeAfterRounding,
                PcuFloatUnderflowPolicy::AllowGradualUnderflow,
                PcuFloatUnderflowPolicy::RejectSubnormalResult,
            ] {
                with_kernel(grid, underflow, PcuRangePolicy::Reject, |kernel| {
                    for mode in [PcuNumericalMode::Boundary, PcuNumericalMode::Strict] {
                        let mut kernel = *kernel;
                        kernel.numerical_requirements.numerical_mode = mode;
                        kernel.numerical_requirements.float_underflow = underflow;
                        let kernel = &kernel;
                        let request = PcuImplementationRequest {
                            device: identity(1),
                            executor: PcuExecutorId(0),
                            operation: kernel,
                            requirements: PcuImplementationRequirements {
                                numerical_mode: mode,
                                float_underflow: underflow,
                                ..PcuImplementationRequirements::default()
                            },
                            boundary: PcuCostBoundary::Host,
                        };
                        let mut result = [None; 2];
                        let mut host_result = [None; 2];
                        assert_eq!(typed.implementation_offers(&request, &mut result), Ok(1));
                        assert_eq!(
                            unified.implementation_offers(&request, &mut host_result),
                            Ok(1)
                        );
                        assert_eq!(host_result, result);
                        let offer = result[0].unwrap();
                        assert_eq!(offer.implementation.local_id, local_id);
                        assert_eq!(offer.implementation.revision, 1);
                        assert_eq!(offer.requirements, request.requirements);
                        assert_eq!(offer.workspace_bytes, Some(0));
                        assert_eq!(
                            offer.cost,
                            PcuImplementationCost::unknown(PcuCostBoundary::Host)
                        );
                        assert_eq!(offer.validate_request(&request), Ok(()));
                        assert_eq!(result[1], None);
                        assert_eq!(typed.implementation_offers(&request, &mut []), Ok(1));
                    }
                });
            }
        }
    }
}

#[test]
fn f64_legacy_and_unified_offers_keep_distinct_portable_profiles() {
    let typed = PcuCpuNegOffers::new(PcuCpuCheckedNeg::scalar(), identity(1), PcuExecutorId(0));
    let unified = PcuCpuHostOffers::new(PcuCpuHostBackend::scalar(), identity(1), PcuExecutorId(0));
    with_kernel(
        false,
        PcuFloatUnderflowPolicy::default(),
        PcuRangePolicy::Reject,
        |kernel| {
            for mode in [PcuNumericalMode::Boundary, PcuNumericalMode::Strict] {
                for determinism in [
                    PcuReproducibility::Unspecified,
                    PcuReproducibility::PortableV1,
                ] {
                    for arithmetic in [
                        PcuCompoundArithmeticPolicy::Checked,
                        PcuCompoundArithmeticPolicy::BackendDefined,
                    ] {
                        for precision in [
                            PcuPrecisionPolicy::Preserve,
                            PcuPrecisionPolicy::BackendOptimized,
                        ] {
                            let mut request = PcuImplementationRequest {
                                device: identity(1),
                                executor: PcuExecutorId(0),
                                operation: kernel,
                                requirements: PcuImplementationRequirements {
                                    numerical_mode: mode,
                                    ..PcuImplementationRequirements::default()
                                },
                                boundary: PcuCostBoundary::Host,
                            };
                            request.requirements.numerical_options.reproducibility = determinism;
                            request.requirements.numerical_options.compound_arithmetic = arithmetic;
                            request.requirements.numerical_options.precision = precision;
                            let kernel = PcuDispatchKernelIr {
                                numerical_requirements: request.requirements,
                                ..*kernel
                            };
                            let request = PcuImplementationRequest {
                                operation: &kernel,
                                ..request
                            };
                            let count = usize::from(determinism == PcuReproducibility::Unspecified);
                            assert_eq!(typed.implementation_offers(&request, &mut []), Ok(count));
                            let mut slots = [None];
                            assert_eq!(unified.implementation_offers(&request, &mut slots), Ok(1));
                            let offer = slots[0].unwrap();
                            offer.validate_request(&request).unwrap();
                            assert_eq!(offer.requirements, request.requirements);
                            if determinism == PcuReproducibility::PortableV1 {
                                assert_eq!(
                                    (offer.implementation.local_id, offer.implementation.revision),
                                    (16404, 1)
                                );
                                assert_eq!(offer.workspace_bytes, Some(0));
                                let mut mismatched = request.requirements;
                                mismatched.numerical_options.reproducibility =
                                    PcuReproducibility::Unspecified;
                                assert_eq!(
                                    unified.implementation_offers(
                                        &PcuImplementationRequest {
                                            requirements: mismatched,
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
            let mut request = PcuImplementationRequest {
                device: identity(1),
                executor: PcuExecutorId(0),
                operation: kernel,
                requirements: PcuImplementationRequirements::default(),
                boundary: PcuCostBoundary::Resident,
            };
            assert_eq!(typed.implementation_offers(&request, &mut []), Ok(0));
            assert_eq!(unified.implementation_offers(&request, &mut []), Ok(0));
            request.boundary = PcuCostBoundary::Host;
            request.requirements.range_policy = PcuRangePolicy::Clamp;
            assert_eq!(typed.implementation_offers(&request, &mut []), Ok(0));
            assert_eq!(unified.implementation_offers(&request, &mut []), Ok(0));
        },
    );
}

#[test]
fn f64_offer_snapshot_and_underflow_mismatches_are_structured_errors() {
    let typed = PcuCpuNegOffers::new(PcuCpuCheckedNeg::scalar(), identity(1), PcuExecutorId(0));
    let unified = PcuCpuHostOffers::new(PcuCpuHostBackend::scalar(), identity(1), PcuExecutorId(0));
    with_kernel(
        false,
        PcuFloatUnderflowPolicy::default(),
        PcuRangePolicy::Reject,
        |kernel| {
            let mut request = PcuImplementationRequest {
                device: identity(2),
                executor: PcuExecutorId(0),
                operation: kernel,
                requirements: PcuImplementationRequirements::default(),
                boundary: PcuCostBoundary::Host,
            };
            assert_eq!(
                typed.implementation_offers(&request, &mut []),
                Err(PcuCpuNegOfferError::DeviceMismatch)
            );
            assert_eq!(
                unified.implementation_offers(&request, &mut []),
                Err(PcuCpuHostOfferError::DeviceMismatch)
            );
            request.device = identity(1);
            request.executor = PcuExecutorId(1);
            assert_eq!(
                typed.implementation_offers(&request, &mut []),
                Err(PcuCpuNegOfferError::ExecutorMismatch)
            );
            assert_eq!(
                unified.implementation_offers(&request, &mut []),
                Err(PcuCpuHostOfferError::ExecutorMismatch)
            );
            request.executor = PcuExecutorId(0);
            request.requirements.float_underflow = PcuFloatUnderflowPolicy::RejectSubnormalResult;
            let kernel = PcuDispatchKernelIr {
                numerical_requirements: request.requirements,
                ..*request.operation
            };
            let request = PcuImplementationRequest {
                operation: &kernel,
                ..request
            };
            assert_eq!(
                typed.implementation_offers(&request, &mut []),
                Err(PcuCpuNegOfferError::UnderflowMismatch)
            );
            assert_eq!(
                unified.implementation_offers(&request, &mut []),
                Err(PcuCpuHostOfferError::UnderflowMismatch)
            );
        },
    );
}

#[test]
fn f64_malformed_clamp_offer_keeps_ssa_error_instead_of_zero_admission() {
    let typed = PcuCpuNegOffers::new(PcuCpuCheckedNeg::scalar(), identity(1), PcuExecutorId(0));
    let unified = PcuCpuHostOffers::new(PcuCpuHostBackend::scalar(), identity(1), PcuExecutorId(0));
    for grid in [false, true] {
        with_kernel(
            grid,
            PcuFloatUnderflowPolicy::default(),
            PcuRangePolicy::Clamp,
            |kernel| {
                let request = PcuImplementationRequest {
                    device: identity(1),
                    executor: PcuExecutorId(0),
                    operation: kernel,
                    requirements: PcuImplementationRequirements::default(),
                    boundary: PcuCostBoundary::Host,
                };
                assert_eq!(typed.implementation_offers(&request, &mut []), Ok(0));
                assert_eq!(unified.implementation_offers(&request, &mut []), Ok(0));
            },
        );
    }
    with_kernel(
        false,
        PcuFloatUnderflowPolicy::default(),
        PcuRangePolicy::Clamp,
        |kernel| {
            let mut ops = kernel.ops.to_vec();
            if let PcuDispatchOp::Data(PcuDispatchDataOp::CheckedFloatUnary { value, .. }) =
                &mut ops[1]
            {
                *value = PcuDispatchValueId(99);
            }
            let malformed = PcuDispatchKernelIr {
                ops: &ops,
                ..*kernel
            };
            let request = PcuImplementationRequest {
                device: identity(1),
                executor: PcuExecutorId(0),
                operation: &malformed,
                requirements: PcuImplementationRequirements::default(),
                boundary: PcuCostBoundary::Host,
            };
            assert!(matches!(
                typed.implementation_offers(&request, &mut []),
                Err(PcuCpuNegOfferError::Provider(
                    PcuCpuPreparedNegError::InvalidValueFlow(_)
                ))
            ));
            assert!(matches!(
                unified.implementation_offers(&request, &mut []),
                Err(PcuCpuHostOfferError::Provider(
                    PcuCpuHostError::InvalidValueFlow(_)
                ))
            ));
        },
    );
}

#[cfg(all(feature = "std", target_arch = "aarch64"))]
#[test]
fn actual_neon_selection_executes_annotated_f64_source() {
    let backend = PcuCpuCheckedNeg::new(PcuCpuProcessor::detect(), PcuCpuImplementation::Neon)
        .expect("actual AArch64 host must admit NEON");
    let bindings = source::negate_bindings();
    let builder = source::negate_ir::<257>(&bindings).unwrap();
    let prepared = backend.prepare_host_kernel(&builder.ir()).unwrap();
    assert_eq!(prepared.implementation(), PcuCpuImplementation::Neon);
    check_source::<257>(backend);
}
