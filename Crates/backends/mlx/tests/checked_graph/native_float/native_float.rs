//! Separate F32/F64 owned `ReLU` profile; sampled scalar math proof remains independently frozen.
#[rustfmt::skip]
use std::sync::Arc;
#[rustfmt::skip]
use pcu_facade::{
    global,
    PcuCheckedFloat,
    PcuExecutionFaultKind as Kind,
    PcuImplementationRequirements,
    PcuNumericalMode,
    PcuNumericalOptions,
    PcuRangePolicy,
    PcuCompoundArithmeticPolicy,
    PcuPrecisionPolicy,
    PcuReproducibility,
    PcuFloatUnderflowPolicy,
    PcuDeviceActivation,
};
#[rustfmt::skip]
use pcu_facade::dialect::tensor::{
    Graph,
    TensorOwnedSelectedProgram,
    TensorArithmeticRewritePolicy,
    TensorArithmeticCapability,
    TensorPointwiseGroupingPolicy,
};
#[rustfmt::skip]
use fusion_pcu_mlx::{
    MlxDiscovery,
    MlxCheckedProgramInput,
    MlxError,
};
#[path = "ordinary/ordinary.rs"]
mod ordinary;
trait Encoding: PcuCheckedFloat {
    const SIGN: u64;
    const NORMAL: u64;
    const NAN: u64;
    fn raw(bits: u64) -> Self;
}
impl Encoding for f32 {
    const SIGN: u64 = 0x8000_0000;
    const NORMAL: u64 = 0x80_0000;
    const NAN: u64 = 0x7fc1_2345;
    fn raw(bits: u64) -> Self {
        Self::from_bits(u32::try_from(bits).unwrap())
    }
}
impl Encoding for f64 {
    const SIGN: u64 = 0x8000_0000_0000_0000;
    const NORMAL: u64 = 0x10_0000_0000_0000;
    const NAN: u64 = 0x7ff8_1234_5678_9abc;
    fn raw(bits: u64) -> Self {
        Self::from_bits(bits)
    }
}
macro_rules! low_encoding {
    ($ty:ty,$bits:ty,$sign:expr,$normal:expr,$nan:expr) => {
        impl Encoding for $ty {
            const SIGN: u64 = $sign;
            const NORMAL: u64 = $normal;
            const NAN: u64 = $nan;
            fn raw(bits: u64) -> Self {
                Self::from_bits(<$bits>::try_from(bits).unwrap())
            }
        }
    };
}
low_encoding!(super::PcuF16Bits, u16, 0x8000, 0x0400, 0x7e19);
low_encoding!(super::PcuBf16Bits, u16, 0x8000, 0x0080, 0x7fd9);
low_encoding!(super::PcuF8E4M3FnBits, u8, 0x80, 0x08, 0x7f);
low_encoding!(super::PcuF8E5M2Bits, u8, 0x80, 0x04, 0x7e);
fn captured<T: Encoding>(
    profile: u8,
    request: PcuImplementationRequirements,
) -> Arc<TensorOwnedSelectedProgram> {
    let capture = global::__pcu_capture_tensor_program::<T, 1, _>(
        [global::PcuSourceShape::Slice { length: 5 }],
        request.float_underflow,
        request.numerical_mode,
        request.numerical_options,
        |capture, inputs| match profile {
            0 => super::annotated::identity::__pcu_capture_entry::<T>(capture, inputs),
            1 => super::annotated::activate::__pcu_capture_entry::<T>(capture, inputs),
            _ => super::annotated::effect_then_identity::__pcu_capture_entry::<T>(capture, inputs),
        },
    )
    .unwrap();
    Arc::clone(capture.program())
}
fn explicit<T: Encoding>(
    profile: u8,
    request: PcuImplementationRequirements,
) -> Arc<TensorOwnedSelectedProgram> {
    let mut graph = Graph::try_new().unwrap();
    let input = graph.input([5], T::TYPE).unwrap();
    graph.set_numerical_options(request.numerical_options);
    let effect = if profile == 0 {
        None
    } else {
        let effect = graph.relu(input).unwrap();
        graph
            .set_value_float_underflow_policy(effect, request.float_underflow)
            .unwrap();
        Some(effect)
    };
    Arc::new(
        graph
            .into_selected_program(
                &[if profile == 1 { effect.unwrap() } else { input }],
                TensorArithmeticRewritePolicy::Disabled,
                TensorArithmeticCapability::Strict,
                TensorPointwiseGroupingPolicy::Disabled,
            )
            .unwrap(),
    )
}
fn same<T: Encoding>(actual: &[T], expected: &[T]) {
    for (a, b) in actual.iter().zip(expected) {
        assert_eq!(a.encode_le().as_ref(), b.encode_le().as_ref());
    }
}
#[allow(clippy::too_many_lines)] // One independent full permission/underflow/source/owner fault matrix.
fn qualify<T: Encoding>() {
    let discovery = MlxDiscovery::discover_default().unwrap();
    let reference = discovery.device_reference(0).unwrap();
    let session = discovery.open_device(reference).unwrap();
    let foreign = discovery.open_device(reference).unwrap();
    let input = [
        T::raw(1),
        T::raw(T::SIGN | 1),
        T::raw(T::SIGN),
        T::raw(T::NORMAL),
        T::raw(T::SIGN | T::NORMAL),
    ];
    let sentinel = T::raw(T::NORMAL + 19);
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
                    PcuFloatUnderflowPolicy::IeeeAfterRounding,
                    PcuFloatUnderflowPolicy::RejectSubnormalResult,
                    PcuFloatUnderflowPolicy::AllowGradualUnderflow,
                ] {
                    let request = PcuImplementationRequirements {
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
                        for program in [
                            captured::<T>(profile, request),
                            explicit::<T>(profile, request),
                        ] {
                            let prepared = session
                                .prepare_checked_program(Arc::clone(&program), request)
                                .unwrap();
                            assert_eq!(prepared.plan().requirements(), request);
                            if profile != 0 {
                                assert_eq!(
                                    prepared.implementation_id().unwrap().revision,
                                    0x0003_0020_0003_0900
                                );
                            }
                            let result = prepared.execute_host(&input);
                            if profile != 0
                                && policy == PcuFloatUnderflowPolicy::RejectSubnormalResult
                            {
                                assert!(
                                    matches!(result,Err(MlxError::Arithmetic(fault)) if fault.kind==Kind::ArithmeticUnderflow && fault.invocation_id==0 && !fault.recovered)
                                );
                            } else {
                                let output = result.unwrap();
                                let mut expected = [sentinel; 7];
                                if profile == 1 {
                                    for (slot, &value) in expected[..5].iter_mut().zip(&input) {
                                        *slot = value.pcu_checked_relu_with_policy(policy).unwrap();
                                    }
                                } else {
                                    expected[..5].copy_from_slice(&input);
                                }
                                let mut actual = [sentinel; 7];
                                output.read_into(&mut actual).unwrap();
                                same(&actual, &expected);
                                let from_resident = prepared
                                    .execute_mixed(&[(
                                        prepared.plan().input(),
                                        MlxCheckedProgramInput::Resident(&resident),
                                    )])
                                    .unwrap();
                                from_resident.read_into(&mut actual).unwrap();
                                same(&actual, &expected);
                                let changed = [T::raw(T::NORMAL); 5];
                                let next = prepared.execute_host(&changed).unwrap();
                                next.read_into(&mut actual).unwrap();
                                output.read_into(&mut actual).unwrap();
                                same(&actual, &expected);
                                let mut short = [sentinel; 4];
                                assert!(output.read_into(&mut short).is_err());
                                same(&short, &[sentinel; 4]);
                            }
                            assert!(matches!(
                                prepared.execute_mixed(&[(
                                    prepared.plan().input(),
                                    MlxCheckedProgramInput::Resident(&other)
                                )]),
                                Err(MlxError::ForeignSession)
                            ));
                            assert!(prepared.execute_host(&input[..4]).is_err());
                            if profile != 0 {
                                let bad = [
                                    T::raw(T::NORMAL),
                                    T::raw(T::NAN),
                                    T::raw(T::NORMAL),
                                    T::raw(T::NORMAL),
                                    T::raw(T::NORMAL),
                                ];
                                assert!(
                                    matches!(prepared.execute_host(&bad),Err(MlxError::Arithmetic(fault)) if fault.kind==Kind::InvalidFloatingOperand && fault.invocation_id==1)
                                );
                            }
                            let mut preserved = [sentinel; 7];
                            resident.read_into(&mut preserved).unwrap();
                            same(&preserved[..5], &input);
                            same(&preserved[5..], &[sentinel; 2]);
                            let mut portable = request;
                            portable.numerical_options.reproducibility =
                                PcuReproducibility::PortableV1;
                            assert!(session.prepare_checked_program(program, portable).is_err());
                        }
                    }
                }
            }
        }
    }
}
#[test]
#[ignore = "Requires actual MLX F32/F64 owned ReLU, unused checked effects and immutable retained source owners."]
fn f32_f64_owned_relu_source_and_fault_lifetimes() {
    qualify::<f32>();
    qualify::<f64>();
}

fn ordinary_width<T: Encoding>() {
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
                    ordinary::verify(
                        PcuImplementationRequirements {
                            numerical_mode,
                            numerical_options: PcuNumericalOptions {
                                compound_arithmetic,
                                precision,
                                reproducibility: PcuReproducibility::Unspecified,
                            },
                            float_underflow,
                            range_policy: PcuRangePolicy::Reject,
                        },
                        [
                            T::raw(1),
                            T::raw(T::SIGN | 1),
                            T::raw(T::SIGN),
                            T::raw(T::NORMAL),
                            T::raw(T::SIGN | T::NORMAL),
                        ],
                        T::raw(T::NORMAL + 1),
                    );
                }
            }
        }
    }
}
#[test]
#[ignore = "Requires actual ordinary six-format owned MLX ReLU and immutable escaped source lifetime."]
fn six_format_ordinary_owned_relu_and_unused_faults() {
    ordinary_width::<super::PcuF16Bits>();
    ordinary_width::<super::PcuBf16Bits>();
    ordinary_width::<super::PcuF8E4M3FnBits>();
    ordinary_width::<super::PcuF8E5M2Bits>();
    ordinary_width::<f32>();
    ordinary_width::<f64>();
    global::clear_thread_cache().unwrap();
    global::use_defaults().unwrap();
}
