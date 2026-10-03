//! Authentic six-format checked graph owners and exact unique-input execution.
#[path = "support/support.rs"]
mod support;
#[rustfmt::skip]
use fusion_pcu_mlx::{
    MlxCheckedTensorBinaryPlan,
    MlxCheckedProgramInput,
    MlxError,
    MlxRuntime,
    MlxDiscovery,
    MlxTensorBinaryRequest,
    MLX_EXECUTOR,
    MlxEncodedArray,
};
#[rustfmt::skip]
use pcu_facade::{
    PcuCheckedFloat,
    PcuScalar,
    PcuImplementationRequirements,
    PcuNumericalMode,
    PcuCompoundArithmeticPolicy,
    PcuPrecisionPolicy,
    PcuFloatUnderflowPolicy,
    PcuRangePolicy,
    PcuReproducibility,
    PcuHostArgument,
    PcuBindingRef,
    PcuDeviceIdentity,
    PcuImplementationOffers,
    PcuImplementationRequest,
    PcuImplementationCost,
    PcuCostBoundary,
    PcuExecutionFaultKind,
    PcuF16Bits,
    PcuBf16Bits,
    PcuF8E4M3FnBits,
    PcuF8E5M2Bits,
};
fn verify<T: PcuScalar>(owner: &MlxEncodedArray, expected: &[T; 5], sentinel: T) {
    let mut actual = [sentinel; 7];
    owner.read_into(&mut actual).unwrap();
    let wanted: Vec<T> = expected.iter().copied().chain([sentinel; 2]).collect();
    assert_eq!(
        PcuHostArgument::read(PcuBindingRef::new(0, 0), &actual).bytes(),
        PcuHostArgument::read(PcuBindingRef::new(0, 0), &wanted).bytes(),
    );
    let mut short = [sentinel; 4];
    assert!(owner.read_into(&mut short).is_err());
    assert_eq!(
        PcuHostArgument::read(PcuBindingRef::new(0, 0), &short).bytes(),
        PcuHostArgument::read(PcuBindingRef::new(0, 0), &[sentinel; 4]).bytes(),
    );
}
#[allow(clippy::too_many_lines)] // Kept together: frozen tuple, physical host/resident boundaries and immutable failure/lifetime law.
fn native<T: PcuCheckedFloat>(value: impl Fn(f32) -> T + Copy, invalid: T) {
    let runtime = MlxRuntime::load_default().unwrap();
    let session = runtime.open_gpu(0).unwrap();
    let foreign = runtime.open_gpu(0).unwrap();
    let a = [value(2.0); 5];
    let b = [value(4.0); 5];
    let left = session.upload_encoded(&a).unwrap();
    let right = session.upload_encoded(&b).unwrap();
    let other = foreign.upload_encoded(&b).unwrap();
    let bytes = PcuHostArgument::read(PcuBindingRef::new(0, 0), &a);
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
                    let mut request = PcuImplementationRequirements {
                        numerical_mode: mode,
                        float_underflow: underflow,
                        ..Default::default()
                    };
                    request.numerical_options.compound_arithmetic = compound;
                    request.numerical_options.precision = precision;
                    for profile in 0..6 {
                        for program in [
                            support::capture::<T>(profile, 5, request),
                            support::graph::<T>(profile, 5, request),
                        ] {
                            let prepared = session
                                .prepare_tensor_binary_program(program, request)
                                .unwrap();
                            let ids = prepared.plan().input_values();
                            let expected =
                                [value([6.0, 2.0, 8.0, 2.0, 4.0, 16.0][usize::from(profile)]); 5];
                            let output = if ids.len() == 1 {
                                prepared
                                    .execute_mixed(&[(
                                        ids[0],
                                        MlxCheckedProgramInput::Resident(&right),
                                    )])
                                    .unwrap()
                            } else {
                                let result = prepared
                                    .execute_mixed(&[
                                        (ids[1], MlxCheckedProgramInput::Resident(&right)),
                                        (
                                            ids[0],
                                            MlxCheckedProgramInput::Host {
                                                scalar: T::TYPE,
                                                bytes: bytes.bytes(),
                                            },
                                        ),
                                    ])
                                    .unwrap();
                                assert!(matches!(
                                    prepared.execute_mixed(&[
                                        (ids[0], MlxCheckedProgramInput::Resident(&left)),
                                        (ids[1], MlxCheckedProgramInput::Resident(&other)),
                                    ]),
                                    Err(MlxError::ForeignSession)
                                ));
                                result
                            };
                            verify(&output, &expected, value(6.0));
                            let bad = session.upload_encoded(&[invalid; 5]).unwrap();
                            let fatal = if ids.len() == 1 {
                                prepared.execute_mixed(&[(
                                    ids[0],
                                    MlxCheckedProgramInput::Resident(&bad),
                                )])
                            } else {
                                prepared.execute_mixed(&[
                                    (ids[0], MlxCheckedProgramInput::Resident(&bad)),
                                    (ids[1], MlxCheckedProgramInput::Resident(&right)),
                                ])
                            };
                            assert!(
                                matches!(fatal, Err(MlxError::Arithmetic(fault)) if !fault.recovered)
                            );
                            verify(&output, &expected, value(6.0));
                            if profile == 3 || profile == 4 {
                                let zero = session.upload_encoded(&[value(0.0); 5]).unwrap();
                                let failed = prepared.execute_mixed(&[
                                    (ids[0], MlxCheckedProgramInput::Resident(&zero)),
                                    (ids[1], MlxCheckedProgramInput::Resident(&right)),
                                ]);
                                assert!(
                                    matches!(failed, Err(MlxError::Arithmetic(fault)) if !fault.recovered && fault.invocation_id == 0 && fault.kind == PcuExecutionFaultKind::DivideByZero)
                                );
                                verify(&output, &expected, value(6.0));
                            }
                            assert!(prepared.execute_mixed(&[]).is_err());
                            verify(&left, &a, value(6.0));
                            verify(&right, &b, value(6.0));
                            drop(prepared);
                            verify(&output, &expected, value(6.0));
                        }
                    }
                }
            }
        }
    }
}
#[test]
fn detached_source_and_graph_binary_schema_and_policy() {
    for profile in 0..6 {
        for program in [
            support::capture::<f64>(profile, 5, PcuImplementationRequirements::default()),
            support::graph::<f64>(profile, 5, PcuImplementationRequirements::default()),
        ] {
            let plan = MlxCheckedTensorBinaryPlan::assess_program(
                &program,
                PcuImplementationRequirements::default(),
            )
            .unwrap();
            assert_eq!(plan.element_count(), 5);
            assert_eq!(plan.byte_len(), 40);
            assert_eq!(plan.input_values().len(), if profile == 5 { 1 } else { 2 });
            let mut request = PcuImplementationRequirements {
                range_policy: PcuRangePolicy::Clamp,
                ..PcuImplementationRequirements::default()
            };
            assert!(MlxCheckedTensorBinaryPlan::assess_program(&program, request).is_err());
            request.range_policy = PcuRangePolicy::Reject;
            request.numerical_options.reproducibility = PcuReproducibility::PortableV1;
            assert!(MlxCheckedTensorBinaryPlan::assess_program(&program, request).is_err());
        }
    }
}
#[test]
#[ignore = "Requires actual pinned MLX six-format checked graph owners and exact retained native sessions."]
fn six_format_binary_graph_private_outputs_and_checked_unused_effects() {
    native(
        |v| PcuF16Bits::pcu_checked_from_f32(v).unwrap(),
        PcuF16Bits::from_bits(0x7c00),
    );
    native(
        |v| PcuBf16Bits::pcu_checked_from_f32(v).unwrap(),
        PcuBf16Bits::from_bits(0x7f80),
    );
    native(
        |v| PcuF8E4M3FnBits::pcu_checked_from_f32(v).unwrap(),
        PcuF8E4M3FnBits::from_bits(0x7f),
    );
    native(
        |v| PcuF8E5M2Bits::pcu_checked_from_f32(v).unwrap(),
        PcuF8E5M2Bits::from_bits(0x7c),
    );
    native::<f32>(|v| v, f32::NAN);
    native::<f64>(f64::from, f64::NAN);
}

#[test]
#[ignore = "Requires actual MLX discovery and exact escaped binary owner boundary offers."]
fn native_binary_graph_offers_preserve_exact_owner_boundary() {
    let discovery = MlxDiscovery::discover_default().unwrap();
    let device =
        PcuDeviceIdentity::from_device_ref(discovery.device_reference(0).unwrap()).unwrap();
    for profile in 0..6 {
        let program = support::capture::<f64>(profile, 5, PcuImplementationRequirements::default());
        let operation = MlxTensorBinaryRequest { program: &program };
        for boundary in [
            PcuCostBoundary::Host,
            PcuCostBoundary::Resident,
            PcuCostBoundary::HostInputsResidentOutput,
            PcuCostBoundary::MixedInputsResidentOutput,
        ] {
            let request = PcuImplementationRequest {
                device,
                executor: MLX_EXECUTOR,
                operation: &operation,
                requirements: PcuImplementationRequirements::default(),
                boundary,
            };
            let mut offers = [None];
            let count = discovery
                .implementation_offers(&request, &mut offers)
                .unwrap();
            let admitted = boundary != PcuCostBoundary::Host
                && !(profile == 5 && boundary == PcuCostBoundary::MixedInputsResidentOutput);
            assert_eq!(count, usize::from(admitted));
            if admitted {
                let offer = offers[0].unwrap();
                offer.validate_request(&request).unwrap();
                assert_eq!(offer.cost, PcuImplementationCost::unknown(boundary));
                assert_eq!(offer.workspace_bytes, None);
                assert_eq!(offer.implementation.revision, 0x0003_0020_0003_0800);
            }
        }
    }
}
