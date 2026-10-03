//! Actual authentic low-format source closure, immutable owners and unused checked effect replay.
#[rustfmt::skip]
use std::sync::Arc;
#[rustfmt::skip]
use fusion_pcu_mlx::{MlxDiscovery,MlxError,MlxCheckedTensorRequest,MlxCheckedProgramInput};
#[rustfmt::skip]
use pcu_facade::{PcuDeviceActivation,PcuImplementationOffers,PcuImplementationRequest,PcuImplementationRequirements,PcuCostBoundary,PcuDeviceIdentity,PcuHostArgument,PcuBindingRef,PcuNumericalMode,PcuNumericalOptions,PcuCompoundArithmeticPolicy,PcuPrecisionPolicy,PcuReproducibility,PcuRangePolicy,PcuExecutionFaultKind as Kind,PcuFloatUnderflowPolicy as Policy,PcuDispatchFloatUnaryOp as Op,PcuScalarType,PcuF16Bits,PcuBf16Bits,PcuF8E4M3FnBits,PcuF8E5M2Bits};
#[rustfmt::skip]
use pcu_facade::dialect::tensor::TensorOwnedSelectedProgram;
#[path = "../checked_unary/oracle/oracle.rs"]
mod oracle;
use oracle::Low;
#[path = "source/source.rs"]
mod annotated;
fn source<T: Low>(
    profile: u8,
    requirements: PcuImplementationRequirements,
) -> Arc<TensorOwnedSelectedProgram> {
    let capture = match profile {
        0 => pcu_facade::global::__pcu_capture_tensor_program(
            [pcu_facade::global::PcuSourceShape::Slice { length: 5 }],
            requirements.float_underflow,
            requirements.numerical_mode,
            requirements.numerical_options,
            annotated::identity::__pcu_capture_entry::<T>,
        ),
        1 => pcu_facade::global::__pcu_capture_tensor_program(
            [pcu_facade::global::PcuSourceShape::Slice { length: 5 }],
            requirements.float_underflow,
            requirements.numerical_mode,
            requirements.numerical_options,
            annotated::activate::__pcu_capture_entry::<T>,
        ),
        _ => pcu_facade::global::__pcu_capture_tensor_program(
            [pcu_facade::global::PcuSourceShape::Slice { length: 5 }],
            requirements.float_underflow,
            requirements.numerical_mode,
            requirements.numerical_options,
            annotated::effect_then_identity::__pcu_capture_entry::<T>,
        ),
    }
    .unwrap();
    Arc::clone(capture.program())
}
#[allow(clippy::too_many_lines, clippy::cognitive_complexity)] // Full frozen graph tuple/ownership/fault matrix is one independent actual runtime oracle.
fn qualify<T: Low>() {
    let discovery = MlxDiscovery::discover_default().unwrap();
    let reference = discovery.device_reference(0).unwrap();
    let device = PcuDeviceIdentity::from_device_ref(reference).unwrap();
    let session = discovery.open_device(reference).unwrap();
    let foreign = discovery.open_device(reference).unwrap();
    let normal = T::from_bits(1 << T::FRACTION);
    let input = [
        T::from_bits(1),
        T::from_bits(T::SIGN | 1),
        T::from_bits(T::SIGN),
        normal,
        T::from_bits(T::SIGN | normal.bits()),
    ];
    let sentinel = T::from_bits(T::MAX);
    let resident = session.upload_encoded(&input).unwrap();
    let other = foreign.upload_encoded(&input).unwrap();
    for mode in [PcuNumericalMode::Boundary, PcuNumericalMode::Strict] {
        for compound in [
            PcuCompoundArithmeticPolicy::Checked,
            PcuCompoundArithmeticPolicy::BackendDefined,
        ] {
            for precision in [
                PcuPrecisionPolicy::Preserve,
                PcuPrecisionPolicy::BackendOptimized,
            ] {
                for policy in [
                    Policy::IeeeAfterRounding,
                    Policy::RejectSubnormalResult,
                    Policy::AllowGradualUnderflow,
                ] {
                    let requirements = PcuImplementationRequirements {
                        numerical_mode: mode,
                        numerical_options: PcuNumericalOptions {
                            compound_arithmetic: compound,
                            precision,
                            reproducibility: PcuReproducibility::Unspecified,
                        },
                        float_underflow: policy,
                        range_policy: PcuRangePolicy::Reject,
                    };
                    for profile in 0..3 {
                        let program = source::<T>(profile, requirements);
                        let request = MlxCheckedTensorRequest { program: &program };
                        for boundary in [
                            PcuCostBoundary::Host,
                            PcuCostBoundary::Resident,
                            PcuCostBoundary::HostInputsResidentOutput,
                        ] {
                            let request = PcuImplementationRequest {
                                device,
                                executor: fusion_pcu_mlx::MLX_EXECUTOR,
                                requirements,
                                boundary,
                                operation: &request,
                            };
                            let mut offers = [None];
                            assert_eq!(
                                discovery
                                    .implementation_offers(&request, &mut offers)
                                    .unwrap(),
                                1
                            );
                            let offer = offers[0].unwrap();
                            offer.validate_request(&request).unwrap();
                            assert_eq!(offer.cost.boundary, boundary);
                            assert_eq!(
                                offer.cost,
                                pcu_facade::PcuImplementationCost::unknown(boundary)
                            );
                        }
                        let request = PcuImplementationRequest {
                            device,
                            executor: fusion_pcu_mlx::MLX_EXECUTOR,
                            requirements,
                            boundary: PcuCostBoundary::MixedInputsResidentOutput,
                            operation: &request,
                        };
                        assert_eq!(
                            discovery
                                .implementation_offers(&request, &mut [None])
                                .unwrap(),
                            0
                        );
                        let prepared = session
                            .prepare_checked_program(Arc::clone(&program), requirements)
                            .unwrap();
                        assert_eq!(prepared.program().node_order(), program.node_order());
                        assert!(prepared.implementation_id().is_some());
                        let mut expected = [sentinel; 8];
                        let fault = if profile == 0 {
                            expected[..5].copy_from_slice(&input);
                            None
                        } else {
                            oracle::native::<T, 5>(
                                &input,
                                &mut expected,
                                Op::Relu,
                                policy,
                                PcuRangePolicy::Reject,
                            )
                            .err()
                        };
                        if profile == 2 && fault.is_none() {
                            expected[..5].copy_from_slice(&input);
                        }
                        let execute = || {
                            prepared.execute_mixed(&[(
                                prepared.plan().input(),
                                MlxCheckedProgramInput::Resident(&resident),
                            )])
                        };
                        let result = execute();
                        if let Some(fault) = fault {
                            assert!(
                                matches!(result,Err(MlxError::Arithmetic(actual)) if actual==fault)
                            );
                        } else {
                            let old = result.unwrap();
                            let mut actual = [sentinel; 8];
                            old.read_into(&mut actual).unwrap();
                            assert_eq!(actual, expected);
                            let changed = [
                                T::from_bits(T::SIGN),
                                normal,
                                T::from_bits(T::SIGN),
                                normal,
                                T::from_bits(T::SIGN),
                            ];
                            let next = prepared.execute_host(&changed).unwrap();
                            next.read_into(&mut actual).unwrap();
                            old.read_into(&mut actual).unwrap();
                            assert_eq!(actual, expected);
                            let bytes = PcuHostArgument::read(PcuBindingRef::new(0, 0), &input);
                            let copied = prepared
                                .execute_mixed(&[(
                                    prepared.plan().input(),
                                    MlxCheckedProgramInput::Host {
                                        scalar: T::TYPE,
                                        bytes: bytes.bytes(),
                                    },
                                )])
                                .unwrap();
                            copied.read_into(&mut actual).unwrap();
                            assert_eq!(actual, expected);
                        }
                        assert!(matches!(
                            prepared.execute_mixed(&[(
                                prepared.plan().input(),
                                MlxCheckedProgramInput::Resident(&other)
                            )]),
                            Err(MlxError::ForeignSession)
                        ));
                        assert!(prepared.execute_host(&input[..1]).is_err());
                        let bytes = PcuHostArgument::read(PcuBindingRef::new(0, 0), &input);
                        assert!(matches!(
                            prepared.execute_mixed(&[(
                                prepared.plan().input(),
                                MlxCheckedProgramInput::Host {
                                    scalar: PcuScalarType::U32,
                                    bytes: bytes.bytes()
                                }
                            )]),
                            Err(MlxError::UnsupportedScalar(PcuScalarType::U32))
                        ));
                        if profile != 0 {
                            let bad = [normal, T::from_bits(T::SIGN - 1), normal, normal, normal];
                            assert!(
                                matches!(prepared.execute_host(&bad),Err(MlxError::Arithmetic(fault)) if fault.kind==Kind::InvalidFloatingOperand&&fault.invocation_id==1)
                            );
                        }
                        let mut preserved = [sentinel; 5];
                        resident.read_into(&mut preserved).unwrap();
                        assert_eq!(preserved, input);
                    }
                }
            }
        }
    }
}
macro_rules! formats{($name:ident,$ty:ty)=>{#[test]#[ignore="Requires actual MLX retained checked graph Identity/ReLU and unused-effect proof."]fn $name(){qualify::<$ty>();}};}
formats!(half_graph, PcuF16Bits);
formats!(bfloat_graph, PcuBf16Bits);
formats!(e4_graph, PcuF8E4M3FnBits);
formats!(e5_graph, PcuF8E5M2Bits);

#[path = "native_float/native_float.rs"]
mod native_float;
