//! Actual #[pcu] source through the existing prepared host-call seam.

#[rustfmt::skip]
use fusion_pcu_cpu::{
    PcuCpuCheckedNeg,
    PcuCpuImplementation,
    PcuCpuPreparedNegError,
    PcuCpuPreparedNeg,
    PcuCpuProcessor,
};
#[rustfmt::skip]
use pcu_facade::{
    pcu,
    PcuCheckedFloat,
    PcuExecutionFault,
    PcuExecutionFaultKind,
    PcuBinding,
    PcuBindingAccess,
    PcuBindingRef,
    PcuBindingStorageClass,
    PcuDispatchControlOp,
    PcuDispatchDataOp,
    PcuDispatchEntryPoint,
    PcuDispatchFeatureCaps,
    PcuDispatchFloatUnaryOp,
    PcuDispatchIndex,
    PcuDispatchKernelIr,
    PcuDispatchOp,
    PcuDispatchValueId,
    PcuFloatUnderflowPolicy,
    PcuHostArgument,
    PcuHostKernelBackend,
    PcuKernelId,
    PcuPreparedHostKernel,
    PcuRangePolicy,
    PcuValueType,
    PcuValueTypeCaps,
};

#[pcu(invocations = N, crate_path = ::pcu_facade)]
fn negate<const N: usize>(input: &[f32], output: &mut [f32]) {
    let id = context.global_invocation_id;
    output[id] = -input[id];
}

fn implementations() -> std::vec::Vec<PcuCpuCheckedNeg> {
    let mut backends = std::vec![PcuCpuCheckedNeg::scalar()];
    #[cfg(feature = "std")]
    {
        let processor = PcuCpuProcessor::detect();
        for implementation in [
            PcuCpuImplementation::Sse2,
            PcuCpuImplementation::Avx2,
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

fn compare<const N: usize>(backend: PcuCpuCheckedNeg) {
    let input: std::vec::Vec<f32> = (0..N)
        .map(|index| {
            f32::from_bits(
                [
                    0,
                    0x8000_0000,
                    1,
                    0x8000_0001,
                    0x7f7f_ffff,
                    0xff7f_ffff,
                    0x3f80_0000,
                    0xbf80_0000,
                ][index % 8],
            )
        })
        .collect();
    let mut output = std::vec![17.0; N + 3];
    let mut call = negate_prepare::<N, _>(&backend).expect("source prepares");
    call(&input, &mut output).expect("finite sign-bit negation");
    for (value, actual) in input.iter().zip(&output) {
        assert_eq!(actual.to_bits(), value.pcu_checked_neg().unwrap().to_bits());
    }
    assert_eq!(&output[N..], &[17.0; 3]);
    let changed = std::vec![2.0; N];
    call(&changed, &mut output).expect("fresh inputs on prepared reuse");
    assert_eq!(&output[..N], &std::vec![-2.0; N]);
}

#[test]
fn source_matches_scalar_bits_across_vectors_and_tails() {
    for backend in implementations() {
        compare::<1>(backend);
        compare::<3>(backend);
        compare::<4>(backend);
        compare::<7>(backend);
        compare::<8>(backend);
        compare::<9>(backend);
        compare::<17>(backend);
        compare::<257>(backend);
    }
}

#[test]
fn source_preserves_finite_random_bits_with_unaligned_vector_subslices() {
    let mut state = 0x243f_6a88_u32;
    let mut input = std::vec![0.0_f32; 1026];
    for value in &mut input {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        let bits = if state & 0x7f80_0000 == 0x7f80_0000 {
            state ^ 0x0080_0000
        } else {
            state
        };
        *value = f32::from_bits(bits);
    }
    for backend in implementations() {
        let mut output = std::vec![41.0_f32; 1029];
        let mut call = negate_prepare::<1025, _>(&backend).unwrap();
        call(&input[1..], &mut output[1..]).unwrap();
        for (value, actual) in input[1..].iter().zip(&output[1..]) {
            assert_eq!(actual.to_bits(), value.pcu_checked_neg().unwrap().to_bits());
        }
        assert_eq!(output[0].to_bits(), 41.0_f32.to_bits());
        assert!(
            output[1026..]
                .iter()
                .all(|value| value.to_bits() == 41.0_f32.to_bits())
        );
    }
}

#[test]
fn source_fault_is_lowest_invocation_transactional_and_reusable() {
    for backend in implementations() {
        let mut call = negate_prepare::<17, _>(&backend).unwrap();
        let mut input = [1.0; 17];
        let mut output = [19.0; 20];
        for bits in [
            f32::INFINITY.to_bits(),
            f32::NEG_INFINITY.to_bits(),
            0x7fc0_0001,
            0x7f80_0001,
        ] {
            input[7] = f32::from_bits(bits);
            input[16] = f32::NAN;
            assert_eq!(
                call(&input, &mut output),
                Err(PcuCpuPreparedNegError::Fault(PcuExecutionFault {
                    recovered: false,
                    kind: PcuExecutionFaultKind::InvalidFloatingOperand,
                    invocation_id: 7,
                }))
            );
            assert_eq!(output.map(f32::to_bits), [19.0_f32; 20].map(f32::to_bits));
        }
        input.fill(1.0);
        call(&input, &mut output).unwrap();
        assert_eq!(&output[..17], &[-1.0; 17]);
        assert_eq!(&output[17..], &[19.0; 3]);
    }
}

#[test]
fn source_rejects_short_buffers_before_writing() {
    let backend = PcuCpuCheckedNeg::scalar();
    let mut call = negate_prepare::<8, _>(&backend).unwrap();
    let mut output = [31.0; 8];
    assert_eq!(
        call(&[1.0; 7], &mut output),
        Err(PcuCpuPreparedNegError::InvalidArguments)
    );
    assert_eq!(output.map(f32::to_bits), [31.0_f32; 8].map(f32::to_bits));
    assert_eq!(
        call(&[1.0; 8], &mut output[..7]),
        Err(PcuCpuPreparedNegError::InvalidArguments)
    );
    assert_eq!(output.map(f32::to_bits), [31.0_f32; 8].map(f32::to_bits));
}

#[test]
fn explicit_unavailable_instruction_is_rejected() {
    for implementation in [
        PcuCpuImplementation::Sse2,
        PcuCpuImplementation::Avx2,
        PcuCpuImplementation::Neon,
    ] {
        assert!(PcuCpuCheckedNeg::new(PcuCpuProcessor::scalar(), implementation).is_err());
    }
}

fn prepare_policy(
    backend: PcuCpuCheckedNeg,
    underflow_policy: PcuFloatUnderflowPolicy,
    range_policy: PcuRangePolicy,
    grid: bool,
) -> Result<PcuCpuPreparedNeg, PcuCpuPreparedNegError> {
    let bindings = [
        PcuBinding::scalar::<f32>(
            Some("input"),
            0,
            0,
            PcuBindingStorageClass::Storage,
            PcuBindingAccess::ReadOnly,
        ),
        PcuBinding::scalar::<f32>(
            Some("output"),
            0,
            1,
            PcuBindingStorageClass::Storage,
            PcuBindingAccess::WriteOnly,
        ),
    ];
    let index = if grid {
        PcuDispatchIndex::GridStrideId
    } else {
        PcuDispatchIndex::InvocationId
    };
    let body = [
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
            result: PcuDispatchValueId(1),
            binding: PcuBindingRef::new(0, 0),
            index,
        }),
        PcuDispatchOp::Data(PcuDispatchDataOp::CheckedFloatUnary {
            value_type: PcuValueType::f32(),
            op: PcuDispatchFloatUnaryOp::Neg,
            underflow_policy,
            range_policy,
            result: PcuDispatchValueId(2),
            value: PcuDispatchValueId(1),
        }),
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
            binding: PcuBindingRef::new(0, 1),
            index,
            value: PcuDispatchValueId(2),
        }),
    ];
    let direct = [
        body[0],
        body[1],
        body[2],
        PcuDispatchOp::Control(PcuDispatchControlOp::Return),
    ];
    let grid_ops = [
        PcuDispatchOp::GridStrideLoop {
            extent: 17,
            body: &body,
        },
        PcuDispatchOp::Control(PcuDispatchControlOp::Return),
    ];
    backend.prepare_host_kernel(&PcuDispatchKernelIr {
        numerical_requirements: pcu_facade::PcuImplementationRequirements {
            float_underflow: underflow_policy,
            range_policy,
            ..PcuDispatchKernelIr::DEFAULT_REQUIREMENTS
        },
        id: PcuKernelId(70),
        entry: PcuDispatchEntryPoint {
            name: "policy_neg",
            logical_shape: [if grid { 2 } else { 17 }, 1, 1],
        },
        bindings: &bindings,
        ports: &[],
        parameters: &[],
        ops: if grid { &grid_ops } else { &direct },
        type_caps: PcuValueTypeCaps::FLOAT32,
        feature_caps: PcuDispatchFeatureCaps::default(),
    })
}

#[test]
fn detached_direct_and_grid_obey_every_underflow_policy_and_reject_clamp() {
    for backend in implementations() {
        for grid in [false, true] {
            for policy in [
                PcuFloatUnderflowPolicy::IeeeAfterRounding,
                PcuFloatUnderflowPolicy::AllowGradualUnderflow,
                PcuFloatUnderflowPolicy::RejectSubnormalResult,
            ] {
                let mut prepared =
                    prepare_policy(backend, policy, PcuRangePolicy::Reject, grid).unwrap();
                let mut input = [1.0; 17];
                input[9] = f32::from_bits(1);
                let mut output = [23.0; 19];
                let result = prepared.call(&mut [
                    PcuHostArgument::read(PcuBindingRef::new(0, 0), &input),
                    PcuHostArgument::read_write(PcuBindingRef::new(0, 1), &mut output),
                ]);
                if policy == PcuFloatUnderflowPolicy::RejectSubnormalResult {
                    assert_eq!(
                        result,
                        Err(PcuCpuPreparedNegError::Fault(PcuExecutionFault {
                            recovered: false,
                            kind: PcuExecutionFaultKind::ArithmeticUnderflow,
                            invocation_id: 9,
                        }))
                    );
                    assert_eq!(output.map(f32::to_bits), [23.0_f32; 19].map(f32::to_bits));
                } else {
                    result.unwrap();
                    assert_eq!(output[9].to_bits(), 0x8000_0001);
                    assert_eq!(
                        output[17..]
                            .iter()
                            .map(|value| value.to_bits())
                            .collect::<std::vec::Vec<_>>(),
                        std::vec![23.0_f32.to_bits(); 2]
                    );
                }
                assert!(matches!(
                    prepare_policy(backend, policy, PcuRangePolicy::Clamp, grid),
                    Err(PcuCpuPreparedNegError::UnsupportedProfile)
                ));
            }
        }
    }
}

#[test]
fn schema_rejects_duplicates_types_access_and_handles_reordered_arguments() {
    let backend = PcuCpuCheckedNeg::scalar();
    let mut prepared = prepare_policy(
        backend,
        PcuFloatUnderflowPolicy::default(),
        PcuRangePolicy::Reject,
        false,
    )
    .unwrap();
    let input = [1.0_f32; 17];
    let mut output = [37.0_f32; 17];
    assert_eq!(
        prepared.call(&mut [
            PcuHostArgument::read(PcuBindingRef::new(0, 0), &input),
            PcuHostArgument::read_write(PcuBindingRef::new(0, 0), &mut output),
        ]),
        Err(PcuCpuPreparedNegError::InvalidArguments)
    );
    assert_eq!(
        prepared.call(&mut [
            PcuHostArgument::read(PcuBindingRef::new(0, 0), &[1_u32; 17]),
            PcuHostArgument::read_write(PcuBindingRef::new(0, 1), &mut output),
        ]),
        Err(PcuCpuPreparedNegError::InvalidArguments)
    );
    assert_eq!(
        prepared.call(&mut [
            PcuHostArgument::read(PcuBindingRef::new(0, 0), &input),
            PcuHostArgument::read(PcuBindingRef::new(0, 1), &input),
        ]),
        Err(PcuCpuPreparedNegError::InvalidArguments)
    );
    assert_eq!(output.map(f32::to_bits), [37.0_f32; 17].map(f32::to_bits));
    prepared
        .call(&mut [
            PcuHostArgument::read_write(PcuBindingRef::new(0, 1), &mut output),
            PcuHostArgument::read(PcuBindingRef::new(0, 0), &input),
        ])
        .unwrap();
    assert_eq!(output.map(f32::to_bits), [-1.0_f32; 17].map(f32::to_bits));
}

#[cfg(all(feature = "std", target_arch = "aarch64"))]
#[test]
fn actual_neon_selection_executes_annotated_f32_source() {
    let backend = PcuCpuCheckedNeg::new(PcuCpuProcessor::detect(), PcuCpuImplementation::Neon)
        .expect("actual AArch64 host must admit NEON");
    let bindings = negate_bindings();
    let builder = negate_ir::<257>(&bindings).unwrap();
    let prepared = backend.prepare_host_kernel(&builder.ir()).unwrap();
    assert_eq!(prepared.implementation(), PcuCpuImplementation::Neon);
    compare::<257>(backend);
}
