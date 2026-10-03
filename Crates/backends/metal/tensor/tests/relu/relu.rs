//! Six-format selected checked effects and immutable fatal-publication proof.
use super::*;
use fusion_pcu::PcuExecutionFaultKind;
const FORMATS: [PcuScalarType; 6] = [
    PcuScalarType::F16,
    PcuScalarType::BF16,
    PcuScalarType::F8E4M3FN,
    PcuScalarType::F8E5M2,
    PcuScalarType::F32,
    PcuScalarType::F64,
];
fn requirements() -> Vec<PcuImplementationRequirements> {
    let mut requests = Vec::new();
    for numerical_mode in [PcuNumericalMode::Boundary, PcuNumericalMode::Strict] {
        for compound_arithmetic in [
            PcuCompoundArithmeticPolicy::Checked,
            PcuCompoundArithmeticPolicy::BackendDefined,
        ] {
            for precision in [
                PcuPrecisionPolicy::Preserve,
                PcuPrecisionPolicy::BackendOptimized,
            ] {
                for float_underflow in [
                    PcuFloatUnderflowPolicy::IeeeAfterRounding,
                    PcuFloatUnderflowPolicy::RejectSubnormalResult,
                    PcuFloatUnderflowPolicy::AllowGradualUnderflow,
                ] {
                    requests.push(PcuImplementationRequirements {
                        numerical_mode,
                        numerical_options: PcuNumericalOptions {
                            compound_arithmetic,
                            precision,
                            reproducibility: PcuReproducibility::Unspecified,
                        },
                        float_underflow,
                        range_policy: PcuRangePolicy::Reject,
                    });
                }
            }
        }
    }
    requests
}
fn program(
    scalar: PcuScalarType,
    request: PcuImplementationRequirements,
    unused: bool,
) -> TensorOwnedSelectedProgram {
    let mut graph = Graph::try_new().unwrap();
    // Ordinary capture creates transport leaves before installing arithmetic options.
    let input = graph.input([5, 13], scalar).unwrap();
    graph.set_numerical_options(request.numerical_options);
    let effect = graph.relu(input).unwrap();
    graph
        .set_value_float_underflow_policy(effect, request.float_underflow)
        .unwrap();
    selected(graph, if unused { input } else { effect })
}
#[test]
fn exact_six_format_relu_assessment_preserves_effects_and_explicit_policy() {
    for scalar in FORMATS {
        for request in requirements() {
            for unused in [false, true] {
                let program = program(scalar, request, unused);
                assert!(MetalTensorPlan::assess_program(&program, request).is_err());
                let plan = MetalTensorPlan::assess_relu_program(&program, request).unwrap();
                assert_eq!(plan.requirements(), request);
                assert!(plan.relu_effect().is_some());
                assert_eq!(plan.output() == plan.input(), unused);
                let mut wrong = request;
                wrong.float_underflow =
                    if request.float_underflow == PcuFloatUnderflowPolicy::AllowGradualUnderflow {
                        PcuFloatUnderflowPolicy::IeeeAfterRounding
                    } else {
                        PcuFloatUnderflowPolicy::AllowGradualUnderflow
                    };
                assert!(MetalTensorPlan::assess_relu_program(&program, wrong).is_err());
                wrong = request;
                wrong.numerical_options.precision =
                    if request.numerical_options.precision == PcuPrecisionPolicy::Preserve {
                        PcuPrecisionPolicy::BackendOptimized
                    } else {
                        PcuPrecisionPolicy::Preserve
                    };
                assert!(MetalTensorPlan::assess_relu_program(&program, wrong).is_err());
                wrong = request;
                wrong.range_policy = PcuRangePolicy::Clamp;
                assert!(MetalTensorPlan::assess_relu_program(&program, wrong).is_err());
                wrong = request;
                wrong.numerical_options.reproducibility = PcuReproducibility::PortableV1;
                assert!(MetalTensorPlan::assess_relu_program(&program, wrong).is_err());
            }
        }
    }
}
// Independent bit encodings: sign, exactly one, smallest normal, a NaN.
fn encoding(scalar: PcuScalarType) -> (u64, u64, u64, u64) {
    match scalar {
        PcuScalarType::F16 => (0x8000, 0x3c00, 0x0400, 0x7e01),
        PcuScalarType::BF16 => (0x8000, 0x3f80, 0x0080, 0x7fc1),
        PcuScalarType::F8E4M3FN => (0x80, 0x38, 0x08, 0x7f),
        PcuScalarType::F8E5M2 => (0x80, 0x3c, 0x04, 0x7f),
        PcuScalarType::F32 => (0x8000_0000, 0x3f80_0000, 0x0080_0000, 0x7fc1_2345),
        PcuScalarType::F64 => (
            0x8000_0000_0000_0000,
            0x3ff0_0000_0000_0000,
            0x0010_0000_0000_0000,
            0x7ff8_1234_5678_9abc,
        ),
        _ => unreachable!("six-format fixture"),
    }
}
fn bytes(scalar: PcuScalarType, bits: &[u64]) -> Vec<u8> {
    let width = usize::from(scalar.bit_width()) / 8;
    bits.iter()
        .flat_map(|bits| bits.to_le_bytes()[..width].to_vec())
        .collect()
}
#[test]
#[ignore = "Requires actual Metal six-format owned checked ReLU and unused-effect publication proof."]
fn native_six_format_relu_effects_and_escaped_owners() {
    let session = MetalSession::open(0).unwrap();
    for scalar in FORMATS {
        let (sign, one, normal, nan) = encoding(scalar);
        let bits = (0..65)
            .map(|i| [sign | one, sign, 0, one, normal, sign | 1][i % 6])
            .collect::<Vec<_>>();
        let input = bytes(scalar, &bits);
        let output = bytes(
            scalar,
            &bits
                .iter()
                .map(|&bits| if bits & sign != 0 { 0 } else { bits })
                .collect::<Vec<_>>(),
        );
        for request in requirements() {
            for unused in [false, true] {
                native_case(&session, scalar, request, unused, &input, &output, nan);
            }
        }
    }
}
#[allow(clippy::too_many_arguments)] // Each independent encoding/policy fixture supplies its own literal bit oracle.
fn native_case(
    session: &MetalSession,
    scalar: PcuScalarType,
    request: PcuImplementationRequirements,
    unused: bool,
    input: &[u8],
    output: &[u8],
    nan: u64,
) {
    let plan =
        MetalTensorPlan::assess_relu_program(&program(scalar, request, unused), request).unwrap();
    let prepared = session
        .prepare_tensor_program(plan, PcuMemoryPoolId(133))
        .unwrap();
    let old = prepared
        .execute(MetalTensorInput::HostBytes {
            scalar,
            elements: 65,
            bytes: input,
        })
        .unwrap();
    let retained_input = session
        .prepare_tensor_program(super::plan(scalar, request), PcuMemoryPoolId(133))
        .unwrap()
        .execute(MetalTensorInput::HostBytes {
            scalar,
            elements: 65,
            bytes: input,
        })
        .unwrap();
    let resident = prepared
        .execute(MetalTensorInput::Resident {
            scalar,
            elements: 65,
            resource: retained_input.resource(),
        })
        .unwrap();
    let mut actual = vec![91; input.len() + 3];
    old.read_bytes_into(&mut actual).unwrap();
    assert_eq!(&actual[..input.len()], if unused { input } else { output });
    assert_eq!(&actual[input.len()..], [91; 3]);
    resident.read_bytes_into(&mut actual).unwrap();
    assert_eq!(&actual[..input.len()], if unused { input } else { output });
    let mut unchanged_input = vec![91; input.len()];
    retained_input
        .read_bytes_into(&mut unchanged_input)
        .unwrap();
    assert_eq!(unchanged_input, input);
    drop(resident);
    drop(retained_input);
    let width = usize::from(scalar.bit_width()) / 8;
    let mut bad = input.to_vec();
    bad[2 * width..3 * width].copy_from_slice(&nan.to_le_bytes()[..width]);
    assert!(
        matches!(prepared.execute(MetalTensorInput::HostBytes{scalar,elements:65,bytes:&bad}),Err(MetalError::Arithmetic(fault)) if fault.kind==PcuExecutionFaultKind::InvalidFloatingOperand&&fault.invocation_id==2)
    );
    bad[2 * width..3 * width].copy_from_slice(&1_u64.to_le_bytes()[..width]);
    let tiny = prepared.execute(MetalTensorInput::HostBytes {
        scalar,
        elements: 65,
        bytes: &bad,
    });
    if request.float_underflow == PcuFloatUnderflowPolicy::RejectSubnormalResult {
        assert!(
            matches!(tiny,Err(MetalError::Arithmetic(fault)) if fault.kind==PcuExecutionFaultKind::ArithmeticUnderflow&&fault.invocation_id==2)
        );
    } else {
        tiny.unwrap();
    }
    let retry = prepared
        .execute(MetalTensorInput::HostBytes {
            scalar,
            elements: 65,
            bytes: input,
        })
        .unwrap();
    drop(prepared);
    drop(retry);
    old.read_bytes_into(&mut actual).unwrap();
    assert_eq!(&actual[..input.len()], if unused { input } else { output });
}

#[cfg(feature = "source-tensor")]
#[path = "../../../benches/tensor_relu/source/source.rs"]
mod source;
#[cfg(feature = "source-tensor")]
fn captured<T: fusion_pcu::PcuScalar>() {
    for request in requirements() {
        let capture = pcu_facade::global::__pcu_capture_tensor_program::<T, 1, _>(
            [pcu_facade::global::PcuSourceShape::Slice { length: 65 }],
            request.float_underflow,
            request.numerical_mode,
            request.numerical_options,
            source::relu::__pcu_capture_entry::<T>,
        )
        .unwrap();
        let plan = MetalTensorPlan::assess_relu_program(capture.program(), request).unwrap();
        assert_eq!(plan.scalar_type(), T::TYPE);
        assert_eq!(plan.requirements(), request);
    }
}
#[cfg(feature = "source-tensor")]
#[test]
fn genuine_six_format_source_capture_retains_full_policy_tuple() {
    captured::<fusion_pcu::PcuF16Bits>();
    captured::<fusion_pcu::PcuBf16Bits>();
    captured::<fusion_pcu::PcuF8E4M3FnBits>();
    captured::<fusion_pcu::PcuF8E5M2Bits>();
    captured::<f32>();
    captured::<f64>();
}
