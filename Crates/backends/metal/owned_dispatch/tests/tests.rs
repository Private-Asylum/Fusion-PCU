//! Owned session capability and native lifecycle acceptance.
use super::*;
#[rustfmt::skip]
use fusion_pcu::{
    PcuDispatchOpCaps,
    PcuScalarType,
    PcuValueTypeCaps,
};

#[test]
fn caps_admit_checked_integer_and_exact_float_alu_floors() {
    let support = caps::support();
    assert!(
        support
            .value_type_support
            .direct
            .contains(PcuValueTypeCaps::UINT32)
    );
    assert!(
        support
            .value_type_support
            .direct
            .contains(PcuValueTypeCaps::FLOAT32)
    );
    assert!(
        support
            .value_type_support
            .direct
            .contains(PcuValueTypeCaps::INT32)
    );
    assert!(
        support
            .dispatch_support
            .scalar_alu
            .direct
            .for_scalar(PcuScalarType::I32)
            .contains(PcuDispatchOpCaps::ALU_CHECKED_INTEGER_BINARY)
    );
    assert!(
        support
            .dispatch_support
            .scalar_alu
            .direct
            .for_scalar(PcuScalarType::I128)
            .contains(PcuDispatchOpCaps::ALU_CHECKED_INTEGER_BINARY)
    );
    assert!(
        support
            .value_type_support
            .direct
            .contains(PcuValueTypeCaps::FLOAT64)
    );
    assert!(
        support
            .dispatch_support
            .scalar_alu
            .direct
            .for_scalar(PcuScalarType::F64)
            .contains(PcuDispatchOpCaps::ALU_CHECKED_FLOAT_UNARY)
    );
    assert!(
        support
            .dispatch_support
            .scalar_alu
            .direct
            .for_scalar(PcuScalarType::F64)
            .contains(PcuDispatchOpCaps::ALU_CHECKED_FLOAT_BINARY)
    );
    assert!(
        !support
            .dispatch_support
            .instructions
            .direct
            .contains(PcuDispatchOpCaps::ALU_ADD)
    );
    assert!(
        support
            .dispatch_support
            .scalar_alu
            .direct
            .for_scalar(PcuScalarType::U32)
            .contains(PcuDispatchOpCaps::ALU_CHECKED_INTEGER_BINARY)
    );
    assert!(
        support
            .dispatch_support
            .scalar_alu
            .direct
            .for_scalar(PcuScalarType::F32)
            .contains(PcuDispatchOpCaps::ALU_CHECKED_FLOAT_BINARY)
    );
}

#[test]
fn low_float_caps_admit_proved_binary_and_unary_without_raw_alu() {
    let support = caps::support();
    for scalar in [
        PcuScalarType::F16,
        PcuScalarType::BF16,
        PcuScalarType::F8E4M3FN,
        PcuScalarType::F8E5M2,
    ] {
        assert!(
            support
                .value_type_support
                .direct
                .contains(PcuValueTypeCaps::for_scalar(scalar))
        );
        let operations = support
            .dispatch_support
            .scalar_alu
            .direct
            .for_scalar(scalar);
        assert!(operations.contains(PcuDispatchOpCaps::ALU_CHECKED_FLOAT_BINARY));
        assert!(operations.contains(PcuDispatchOpCaps::ALU_CHECKED_FLOAT_UNARY));
        assert!(!operations.contains(PcuDispatchOpCaps::ALU_MUL));
    }
    for scalar in PcuScalarType::ALL {
        assert_eq!(
            support
                .value_type_support
                .direct
                .contains(PcuValueTypeCaps::for_scalar(scalar)),
            scalar.bit_width() >= 8
        );
        if matches!(scalar, PcuScalarType::F128 | PcuScalarType::F256) {
            assert!(
                !support
                    .dispatch_support
                    .scalar_alu
                    .direct
                    .for_scalar(scalar)
                    .contains(PcuDispatchOpCaps::ALU_CHECKED_INTEGER_BINARY)
            );
            assert!(
                !support
                    .dispatch_support
                    .scalar_alu
                    .direct
                    .for_scalar(scalar)
                    .contains(PcuDispatchOpCaps::ALU_CHECKED_FLOAT_BINARY)
            );
        }
    }
}

#[cfg(target_os = "macos")]
mod native {
    use super::*;
    #[rustfmt::skip]
    use fusion_pcu::{
        PcuBinding,
        PcuBindingStorageClass,
        PcuDispatchEntryPoint,
        PcuDispatchFloatUnaryOp,
        PcuDispatchIndex,
        PcuDispatchValueId,
        PcuFloatUnderflowPolicy,
        PcuKernelId,
        PcuRangePolicy,
        PcuMemoryAllocationRequest,
        PcuMemoryBackingOwnership,
        PcuMemoryHostAccess,
        PcuMemoryProvider,
        PcuObjectKind,
        PcuOwnedDispatchError,
        PcuDispatchKernelIr,
        PcuDispatchIntegerBinaryOp,
        PcuDispatchOp,
        PcuDispatchDataOp,
        PcuDispatchControlOp,
        PcuExecutionFaultKind,
    };
    fn open_backend() -> MetalOwnedDispatchBackend {
        let inventory = crate::MetalDiscovery::discover().unwrap();
        inventory
            .open_owned_device(inventory.reference(PcuObjectKind::Device, 0))
            .unwrap()
    }
    fn allocation(provider: &mut MetalMemoryProvider, words: &[u32]) -> MetalMemoryResource {
        let mut resource = provider
            .allocate(PcuMemoryAllocationRequest {
                pool: PcuMemoryPoolId(9),
                size_bytes: words.len() as u64 * 4,
                alignment_bytes: 4,
                access: PcuMemoryAccess::ReadWrite,
                host_access: PcuMemoryHostAccess::TransferOnly,
                require_device_local: false,
            })
            .unwrap();
        let bytes: Vec<u8> = words.iter().flat_map(|word| word.to_ne_bytes()).collect();
        provider.transfer_to(&mut resource, 0, &bytes).unwrap();
        resource
    }
    fn read(provider: &mut MetalMemoryProvider, resource: &MetalMemoryResource) -> Vec<u32> {
        let mut bytes = vec![0; usize::try_from(resource.size_bytes()).unwrap()];
        provider.transfer_from(resource, 0, &mut bytes).unwrap();
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|bytes| u32::from_ne_bytes(*bytes))
            .collect()
    }
    fn bind(
        backend: &MetalOwnedDispatchBackend,
        kernel: &PcuDispatchKernelIr<'_>,
        resources: [&MetalMemoryResource; 3],
    ) -> Vec<PcuOwnedBinding<MetalOwnedResource>> {
        kernel
            .bindings
            .iter()
            .zip(resources)
            .map(|(binding, resource)| {
                backend
                    .bind(
                        binding.reference(),
                        binding.access,
                        binding.binding_type,
                        resource,
                    )
                    .unwrap()
            })
            .collect()
    }

    #[allow(
        clippy::too_many_lines,
        reason = "One native fixture compares exact-byte publication across normal/Portable, direct/grid and actual SSA operand roles."
    )]
    fn low_owned<T: fusion_pcu::PcuCheckedFloat>(value: impl Fn(f32) -> T) {
        #[rustfmt::skip]
        use fusion_pcu::{
            PcuHostArgument,
            PcuDispatchFloatBinaryOp,
        };
        use crate::admission::binary::tests::Variant;
        for (portable, grid, variant) in [false, true].into_iter().flat_map(|portable| {
            [false, true].into_iter().flat_map(move |grid| {
                [
                    Variant::Plain,
                    Variant::Swap,
                    Variant::Repeat,
                    Variant::PortableBroadcast,
                ]
                .into_iter()
                .filter(move |variant| portable || !matches!(variant, Variant::PortableBroadcast))
                .map(move |variant| (portable, grid, variant))
            })
        }) {
            crate::admission::binary::tests::fixture::<T>(
                grid,
                PcuDispatchFloatBinaryOp::Div,
                PcuFloatUnderflowPolicy::IeeeAfterRounding,
                variant,
                |kernel| {
                    let mut kernel = *kernel;
                    if portable {
                        kernel.numerical_requirements.float_underflow =
                            PcuFloatUnderflowPolicy::IeeeAfterRounding;
                        kernel
                            .numerical_requirements
                            .numerical_options
                            .reproducibility = fusion_pcu::PcuReproducibility::PortableV1;
                    }
                    let backend = open_backend();
                    let mut provider = backend.memory_provider(PcuMemoryPoolId(9));
                    let mut allocate = |data: &[T]| {
                        let bytes = PcuHostArgument::read(PcuBindingRef::new(0, 0), data);
                        let mut resource = provider
                            .allocate(PcuMemoryAllocationRequest {
                                pool: PcuMemoryPoolId(9),
                                size_bytes: bytes.bytes().len() as u64,
                                alignment_bytes: 4,
                                access: PcuMemoryAccess::ReadWrite,
                                host_access: PcuMemoryHostAccess::TransferOnly,
                                require_device_local: false,
                            })
                            .unwrap();
                        provider
                            .transfer_to(&mut resource, 0, bytes.bytes())
                            .unwrap();
                        resource
                    };
                    let left = allocate(&[value(1.0), value(2.0), value(3.0)]);
                    let right = allocate(&[value(2.0), value(4.0), value(6.0)]);
                    let output = allocate(&[value(91.0); 5]);
                    let shape =
                        PcuInvocationShape::invocations(core::num::NonZeroU32::new(3).unwrap());
                    let prepared = backend
                        .prepare_dispatch_owned(
                            PcuDispatchSubmission {
                                kernel: &kernel,
                                shape,
                            },
                            PcuInvocationParameters::empty(),
                        )
                        .unwrap_or_else(|error| panic!("owned portable={portable} grid={grid} variant={variant:?}: {error:?}"));
                    let mut completion = prepared
                        .submit_owned(bind(&backend, &kernel, [&left, &right, &output]))
                        .unwrap();
                    assert_eq!(completion.wait().unwrap(), PcuCompletionOutcome::Succeeded);
                    let expected = match variant {
                        Variant::Swap => [value(2.0); 3],
                        Variant::Repeat => [value(1.0); 3],
                        Variant::PortableBroadcast => [
                            value(0.5),
                            value(0.25),
                            value(1.0).pcu_checked_div(value(6.0)).unwrap(),
                        ],
                        _ => [value(0.5); 3],
                    };
                    let expected = [
                        expected[0],
                        expected[1],
                        expected[2],
                        value(91.0),
                        value(91.0),
                    ];
                    let expected = PcuHostArgument::read(PcuBindingRef::new(0, 0), &expected);
                    let mut bytes = vec![0; expected.bytes().len()];
                    provider.transfer_from(&output, 0, &mut bytes).unwrap();
                    assert_eq!(bytes, expected.bytes());
                },
            );
        }
    }
    #[test]
    #[ignore = "Requires actual Metal advanced owned publication for exact odd low payloads."]
    fn low_formats_owned_dispatch_exact_byte_publication_and_tails() {
        low_owned::<fusion_pcu::PcuF16Bits>(|value| {
            fusion_pcu::PcuF16Bits::pcu_checked_from_f32(value).unwrap()
        });
        low_owned::<fusion_pcu::PcuBf16Bits>(|value| {
            fusion_pcu::PcuBf16Bits::pcu_checked_from_f32(value).unwrap()
        });
        low_owned::<fusion_pcu::PcuF8E4M3FnBits>(|value| {
            fusion_pcu::PcuF8E4M3FnBits::pcu_checked_from_f32(value).unwrap()
        });
        low_owned::<fusion_pcu::PcuF8E5M2Bits>(|value| {
            fusion_pcu::PcuF8E5M2Bits::pcu_checked_from_f32(value).unwrap()
        });
    }

    fn integer_fixture(
        operation: PcuDispatchIntegerBinaryOp,
        grid: bool,
        visit: impl FnOnce(&PcuDispatchKernelIr<'_>),
    ) {
        crate::admission::tests::fixture(grid, false, |template| {
            let mut body = if let PcuDispatchOp::GridStrideLoop { body, .. } = template.ops[0] {
                body.to_vec()
            } else {
                template.ops[..template.ops.len() - 1].to_vec()
            };
            for node in &mut body {
                if let PcuDispatchOp::Data(PcuDispatchDataOp::CheckedIntegerBinary { op, .. }) =
                    node
                {
                    *op = operation;
                }
            }
            let grid_ops = [
                PcuDispatchOp::GridStrideLoop {
                    extent: 3,
                    body: &body,
                },
                PcuDispatchOp::Control(PcuDispatchControlOp::Return),
            ];
            let mut direct = body.clone();
            direct.push(PcuDispatchOp::Control(PcuDispatchControlOp::Return));
            let mut kernel = *template;
            kernel.ops = if grid { &grid_ops } else { &direct };
            if grid {
                kernel.entry.logical_shape[0] = 1;
            }
            visit(&kernel);
        });
    }

    #[test]
    #[ignore = "Requires actual Metal native owned dispatch and memory APIs."]
    #[allow(
        clippy::too_many_lines,
        reason = "One owned resource lifecycle proves success, fault preservation, metadata rejection and escaped owner retention."
    )]
    fn owned_u32_direct_grid_fault_lease_metadata_and_tail() {
        for operation in [
            PcuDispatchIntegerBinaryOp::Add,
            PcuDispatchIntegerBinaryOp::Sub,
            PcuDispatchIntegerBinaryOp::Mul,
        ] {
            for grid in [false, true] {
                integer_fixture(operation, grid, |kernel| {
                    let backend = open_backend();
                    let mut provider = backend.memory_provider(PcuMemoryPoolId(9));
                    let left = allocation(&mut provider, &[2, 2, 3, 100]);
                    let mut right = allocation(&mut provider, &[4, 5, 6, 101]);
                    let output = allocation(&mut provider, &[91; 5]);
                    let shape = PcuInvocationShape::invocations(
                        core::num::NonZeroU32::new(kernel.entry.logical_shape[0]).unwrap(),
                    );
                    let expected = match operation {
                        PcuDispatchIntegerBinaryOp::Add => [6, 7, 9, 91, 91],
                        PcuDispatchIntegerBinaryOp::Sub => [2, 3, 3, 91, 91],
                        PcuDispatchIntegerBinaryOp::Mul => [8, 10, 18, 91, 91],
                    };
                    let fault_kind = if operation == PcuDispatchIntegerBinaryOp::Sub {
                        PcuExecutionFaultKind::ArithmeticUnderflow
                    } else {
                        PcuExecutionFaultKind::ArithmeticOverflow
                    };
                    let prepared = backend
                        .prepare_dispatch_owned(
                            PcuDispatchSubmission { kernel, shape },
                            PcuInvocationParameters::empty(),
                        )
                        .unwrap();
                    let bindings = bind(&backend, kernel, [&left, &right, &output]);
                    assert_eq!(
                        output.backing_ownership(),
                        PcuMemoryBackingOwnership::Shared
                    );
                    let mut completion = prepared.submit_owned(bindings).unwrap();
                    assert_eq!(completion.wait().unwrap(), PcuCompletionOutcome::Succeeded);
                    assert_eq!(read(&mut provider, &output), expected);
                    assert_eq!(completion.wait().unwrap(), PcuCompletionOutcome::Succeeded);
                    drop(completion);
                    assert_eq!(
                        output.backing_ownership(),
                        PcuMemoryBackingOwnership::Exclusive
                    );
                    let words = if operation == PcuDispatchIntegerBinaryOp::Sub {
                        [0_u32, 1, 2, 101]
                    } else {
                        [u32::MAX, 5, 6, 101]
                    };
                    let bytes: Vec<u8> = words.iter().flat_map(|word| word.to_ne_bytes()).collect();
                    provider.transfer_to(&mut right, 0, &bytes).unwrap();
                    let mut completion = prepared
                        .submit_owned(bind(&backend, kernel, [&left, &right, &output]))
                        .unwrap();
                    assert!(
                        matches!(completion.wait().unwrap(), PcuCompletionOutcome::Fault(fault) if fault.invocation_id == 0 && fault.kind == fault_kind)
                    );
                    assert_eq!(read(&mut provider, &output), expected);
                    drop(completion);
                    let mut forged = bind(&backend, kernel, [&left, &right, &output]);
                    forged[0].access = PcuBindingAccess::ReadWrite;
                    assert!(matches!(
                        prepared.submit_owned(forged),
                        Err(PcuOwnedDispatchError::Backend(
                            MetalOwnedDispatchError::Binding(
                                PcuOwnedDispatchBindingError::AccessMismatch(_)
                            )
                        ))
                    ));
                    let other = open_backend();
                    assert!(matches!(
                        other.bind(
                            kernel.bindings[0].reference(),
                            PcuBindingAccess::ReadOnly,
                            PcuBindingType::Value(PcuValueType::u32()),
                            &left
                        ),
                        Err(MetalOwnedDispatchError::Binding(
                            PcuOwnedDispatchBindingError::WrongDevice(_)
                        ))
                    ));
                    let bindings = bind(&backend, kernel, [&left, &right, &output]);
                    drop(backend);
                    drop(left);
                    drop(right);
                    let mut completion = prepared.submit_owned(bindings).unwrap();
                    assert!(matches!(
                        completion.wait().unwrap(),
                        PcuCompletionOutcome::Fault(_)
                    ));
                    assert_eq!(read(&mut provider, &output), expected);
                });
            }
        }
    }

    fn float_fixture(
        operation: PcuDispatchFloatUnaryOp,
        grid: bool,
        underflow_policy: PcuFloatUnderflowPolicy,
        range_policy: PcuRangePolicy,
        visit: impl FnOnce(&PcuDispatchKernelIr<'_>),
    ) {
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
                binding: bindings[0].reference(),
                index,
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::CheckedFloatUnary {
                value_type: PcuValueType::f32(),
                op: operation,
                underflow_policy,
                range_policy,
                result: PcuDispatchValueId(2),
                value: PcuDispatchValueId(1),
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                binding: bindings[1].reference(),
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
                extent: 3,
                body: &body,
            },
            PcuDispatchOp::Control(PcuDispatchControlOp::Return),
        ];
        visit(&PcuDispatchKernelIr {
            numerical_requirements: fusion_pcu::PcuImplementationRequirements {
                float_underflow: underflow_policy,
                range_policy,
                ..PcuDispatchKernelIr::DEFAULT_REQUIREMENTS
            },
            id: PcuKernelId(14),
            entry: PcuDispatchEntryPoint {
                name: "owned-checked-f32",
                logical_shape: [if grid { 1 } else { 3 }, 1, 1],
            },
            bindings: &bindings,
            ports: &[],
            parameters: &[],
            ops: if grid { &grid_ops } else { &direct },
            type_caps: PcuValueTypeCaps::FLOAT32,
            feature_caps: fusion_pcu::PcuDispatchFeatureCaps::empty(),
        });
    }

    #[test]
    #[ignore = "Requires actual Metal owned F32 bit-map execution."]
    #[allow(
        clippy::too_many_lines,
        clippy::cognitive_complexity,
        reason = "One allocation set proves bits, first fault, transactional rollback, retry and recovered Clamp across profiles."
    )]
    fn owned_f32_direct_grid_exact_bits_fault_retry_and_recovered_clamp() {
        for operation in [PcuDispatchFloatUnaryOp::Neg, PcuDispatchFloatUnaryOp::Relu] {
            for grid in [false, true] {
                let backend = open_backend();
                let mut provider = backend.memory_provider(PcuMemoryPoolId(9));
                let mut input = allocation(&mut provider, &[0, 0x8000_0000, 1, 99]);
                let output = allocation(&mut provider, &[91; 5]);
                let mut run = |policy, range, input_bits: [u32; 4]| {
                    let bytes: Vec<u8> = input_bits
                        .iter()
                        .flat_map(|word| word.to_ne_bytes())
                        .collect();
                    provider.transfer_to(&mut input, 0, &bytes).unwrap();
                    let mut outcome = None;
                    float_fixture(operation, grid, policy, range, |kernel| {
                        let shape = PcuInvocationShape::invocations(
                            core::num::NonZeroU32::new(kernel.entry.logical_shape[0]).unwrap(),
                        );
                        let prepared = backend.prepare_dispatch_owned(
                            PcuDispatchSubmission { kernel, shape },
                            PcuInvocationParameters::empty(),
                        );
                        let prepared = prepared.unwrap();
                        let bindings = kernel
                            .bindings
                            .iter()
                            .zip([&input, &output])
                            .map(|(binding, resource)| {
                                backend
                                    .bind(
                                        binding.reference(),
                                        binding.access,
                                        binding.binding_type,
                                        resource,
                                    )
                                    .unwrap()
                            })
                            .collect();
                        let mut completion = prepared.submit_owned(bindings).unwrap();
                        outcome = Some(completion.wait().unwrap());
                    });
                    (outcome, read(&mut provider, &output))
                };
                let (outcome, baseline) = run(
                    PcuFloatUnderflowPolicy::IeeeAfterRounding,
                    PcuRangePolicy::Reject,
                    [0, 0x8000_0000, 1, 99],
                );
                assert_eq!(outcome, Some(PcuCompletionOutcome::Succeeded));
                let expected = if operation == PcuDispatchFloatUnaryOp::Neg {
                    [0x8000_0000, 0, 0x8000_0001, 91, 91]
                } else {
                    [0, 0, 1, 91, 91]
                };
                assert_eq!(baseline, expected);
                let (outcome, after) = run(
                    PcuFloatUnderflowPolicy::IeeeAfterRounding,
                    PcuRangePolicy::Reject,
                    [0, 0x7f80_0001, 0x7f80_0000, 99],
                );
                assert!(
                    matches!(outcome, Some(PcuCompletionOutcome::Fault(fault)) if fault.kind == PcuExecutionFaultKind::InvalidFloatingOperand && fault.invocation_id == 1)
                );
                assert_eq!(after, baseline);
                let (outcome, after) = run(
                    PcuFloatUnderflowPolicy::RejectSubnormalResult,
                    PcuRangePolicy::Reject,
                    [0, 1, 0x7f80_0000, 99],
                );
                assert!(
                    matches!(outcome, Some(PcuCompletionOutcome::Fault(fault)) if fault.kind == PcuExecutionFaultKind::ArithmeticUnderflow && fault.invocation_id == 1)
                );
                assert_eq!(after, baseline);
                let (outcome, after) = run(
                    PcuFloatUnderflowPolicy::IeeeAfterRounding,
                    PcuRangePolicy::Reject,
                    [0, 0x3f80_0000, 0xbf80_0000, 99],
                );
                assert_eq!(outcome, Some(PcuCompletionOutcome::Succeeded));
                let retry = if operation == PcuDispatchFloatUnaryOp::Neg {
                    [0x8000_0000, 0xbf80_0000, 0x3f80_0000, 91, 91]
                } else {
                    [0, 0x3f80_0000, 0, 91, 91]
                };
                assert_eq!(after, retry);
                let (outcome, after) = run(
                    PcuFloatUnderflowPolicy::RejectSubnormalResult,
                    PcuRangePolicy::Clamp,
                    [0, 1, 0x3f80_0000, 99],
                );
                assert!(
                    matches!(outcome,Some(PcuCompletionOutcome::Fault(fault)) if fault.recovered
                    && fault.invocation_id==1 && fault.kind==PcuExecutionFaultKind::ArithmeticUnderflow)
                );
                let recovered = if operation == PcuDispatchFloatUnaryOp::Neg {
                    [0x8000_0000, 0x8000_0001, 0xbf80_0000, 91, 91]
                } else {
                    [0, 1, 0x3f80_0000, 91, 91]
                };
                assert_eq!(after, recovered);
                let (outcome, after) = run(
                    PcuFloatUnderflowPolicy::RejectSubnormalResult,
                    PcuRangePolicy::Clamp,
                    [0, 1, 0x7f80_0000, 99],
                );
                assert!(
                    matches!(outcome,Some(PcuCompletionOutcome::Fault(fault)) if !fault.recovered
                    && fault.invocation_id==2 && fault.kind==PcuExecutionFaultKind::InvalidFloatingOperand)
                );
                assert_eq!(after, recovered);
            }
        }
    }
    #[test]
    #[ignore = "Requires actual Metal resident checked integer direct/grid execution."]
    fn typed_resident_u32_direct_grid_validates_access_type_and_lifetime() {
        use fusion_pcu::{
            PcuDeviceBuffer, PcuDeviceArgument, PcuDeviceKernelBackend, PcuPreparedDeviceKernel,
        };
        for operation in [
            PcuDispatchIntegerBinaryOp::Add,
            PcuDispatchIntegerBinaryOp::Sub,
            PcuDispatchIntegerBinaryOp::Mul,
        ] {
            for grid in [false, true] {
                integer_fixture(operation, grid, |kernel| {
                    let backend = open_backend();
                    let mut provider = backend.memory_provider(PcuMemoryPoolId(9));
                    let left = PcuDeviceBuffer::<u32, _>::new(
                        allocation(&mut provider, &[2, 2, 3, 100]),
                        4,
                    );
                    let right = PcuDeviceBuffer::<u32, _>::new(
                        allocation(&mut provider, &[4, 5, 6, 101]),
                        4,
                    );
                    let mut output =
                        PcuDeviceBuffer::<u32, _>::new(allocation(&mut provider, &[91; 5]), 5);
                    let mut prepared = backend.prepare_device_kernel(kernel).unwrap();
                    let refs: Vec<_> = kernel
                        .bindings
                        .iter()
                        .map(|binding| binding.reference())
                        .collect();
                    drop(backend);
                    let mut arguments = [
                        PcuDeviceArgument::read(refs[0], &left),
                        PcuDeviceArgument::read(refs[1], &right),
                        PcuDeviceArgument::read_write(refs[2], &mut output),
                    ];
                    prepared.call(&mut arguments).unwrap();
                    let expected = match operation {
                        PcuDispatchIntegerBinaryOp::Add => [6, 7, 9, 91, 91],
                        PcuDispatchIntegerBinaryOp::Sub => [2, 3, 3, 91, 91],
                        PcuDispatchIntegerBinaryOp::Mul => [8, 10, 18, 91, 91],
                    };
                    assert_eq!(read(&mut provider, output.resource()), expected);
                    let mut arguments = [
                        PcuDeviceArgument::read(refs[0], &left),
                        PcuDeviceArgument::read(refs[1], &right),
                        PcuDeviceArgument::read(refs[2], &output),
                    ];
                    assert!(matches!(
                        prepared.call(&mut arguments),
                        Err(MetalOwnedDispatchError::Binding(
                            PcuOwnedDispatchBindingError::AccessMismatch(_)
                        ))
                    ));
                    let mut arguments = [
                        PcuDeviceArgument::read(refs[0], &left),
                        PcuDeviceArgument::read(refs[0], &right),
                        PcuDeviceArgument::read_write(refs[2], &mut output),
                    ];
                    assert!(matches!(
                        prepared.call(&mut arguments),
                        Err(MetalOwnedDispatchError::Binding(
                            PcuOwnedDispatchBindingError::Duplicate(_)
                        ))
                    ));
                    let wrong = PcuDeviceBuffer::<f32, _>::new(right.into_resource(), 4);
                    let mut arguments = [
                        PcuDeviceArgument::read(refs[0], &left),
                        PcuDeviceArgument::read(refs[1], &wrong),
                        PcuDeviceArgument::read_write(refs[2], &mut output),
                    ];
                    assert!(matches!(
                        prepared.call(&mut arguments),
                        Err(MetalOwnedDispatchError::Binding(
                            PcuOwnedDispatchBindingError::TypeMismatch(_)
                        ))
                    ));
                    assert_eq!(read(&mut provider, output.resource()), expected);
                });
            }
        }
    }

    #[test]
    #[ignore = "Requires actual Metal staging, resident writes and checked fault completion."]
    fn mixed_call_write_fact_resets_before_validation_and_tracks_terminal_fault() {
        #[rustfmt::skip]
        use fusion_pcu::{
            PcuDeviceBuffer,
            PcuDeviceArgument,
            PcuHostArgument,
            PcuHostDispatchError,
            PcuHostKernelBackend,
            PcuPreparedHostKernel,
        };
        use crate::MetalMixedHostArgument as Argument;
        integer_fixture(PcuDispatchIntegerBinaryOp::Add, false, |kernel| {
            let backend = open_backend();
            let mut provider = backend.memory_provider(PcuMemoryPoolId(9));
            let left =
                PcuDeviceBuffer::<u32, _>::new(allocation(&mut provider, &[1, u32::MAX, 3]), 3);
            let mut output = PcuDeviceBuffer::<u32, _>::new(allocation(&mut provider, &[91; 5]), 5);
            let refs: Vec<_> = kernel
                .bindings
                .iter()
                .map(|binding| binding.reference())
                .collect();
            let mut prepared = backend.prepare_host_kernel(kernel).unwrap();
            assert!(!prepared.last_call_may_have_written());
            let mut arguments = [
                Argument::Resident(PcuDeviceArgument::read(refs[0], &left)),
                Argument::Host(PcuHostArgument::read(refs[1], &[2_u32, 1, 4])),
                Argument::Resident(PcuDeviceArgument::read_write(refs[2], &mut output)),
            ];
            assert!(matches!(prepared.call_mixed(&mut arguments),
                Err(PcuHostDispatchError::Backend(MetalError::Arithmetic(fault)))
                if fault.invocation_id == 1 && !fault.recovered));
            assert!(prepared.last_call_may_have_written());
            assert!(!prepared.last_call_completion_uncertain());
            let after = read(&mut provider, output.resource());
            assert_eq!(after[0], 3);
            assert_eq!(&after[3..], &[91; 2]);
            let mut arguments = [
                Argument::Resident(PcuDeviceArgument::read(refs[0], &left)),
                Argument::Host(PcuHostArgument::read(refs[1], &[2_u32, 1, 4])),
                Argument::Resident(PcuDeviceArgument::read(refs[2], &output)),
            ];
            assert!(matches!(prepared.call_mixed(&mut arguments),
                Err(PcuHostDispatchError::AccessMismatch(target)) if target == refs[2]));
            assert!(!prepared.last_call_may_have_written());
            assert_eq!(read(&mut provider, output.resource()), after);
            let mut host_output = [91_u32; 5];
            let mut arguments = [
                PcuHostArgument::read(refs[0], &[1_u32, 2, 3]),
                PcuHostArgument::read(refs[1], &[2_u32, 3, 4]),
                PcuHostArgument::read_write(refs[2], &mut host_output),
            ];
            prepared.call(&mut arguments).unwrap();
            assert!(prepared.last_call_may_have_written());
            let mut arguments = [PcuHostArgument::read(refs[0], &[1_u32])];
            assert!(matches!(
                prepared.call(&mut arguments),
                Err(PcuHostDispatchError::BufferTooSmall(_))
            ));
            assert!(!prepared.last_call_may_have_written());
            assert_eq!(host_output, [3, 5, 7, 91, 91]);
        });
    }
}

#[test]
fn div_rem_capability_is_exact_fourteen_integer_widths() {
    let support = caps::support();
    for scalar in PcuScalarType::ALL {
        let expected = matches!(
            scalar,
            PcuScalarType::I8
                | PcuScalarType::U8
                | PcuScalarType::I16
                | PcuScalarType::U16
                | PcuScalarType::I32
                | PcuScalarType::U32
                | PcuScalarType::I64
                | PcuScalarType::U64
                | PcuScalarType::U128
                | PcuScalarType::I128
                | PcuScalarType::U256
                | PcuScalarType::I256
                | PcuScalarType::U512
                | PcuScalarType::I512
        );
        assert_eq!(
            support
                .dispatch_support
                .scalar_alu
                .direct
                .for_scalar(scalar)
                .contains(PcuDispatchOpCaps::ALU_CHECKED_DIV_REM),
            expected
        );
    }
}
