//! Reciprocal cold capture law; native arithmetic and SDK control have separate gates.
mod oracle {
    pub use crate::low_oracle::Format;
    pub fn bits<T: pcu_facade::PcuScalar>(actual: &[T], expected: &[T]) {
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected) {
            assert_eq!(actual.encode_le().as_ref(), expected.encode_le().as_ref());
        }
    }
    pub fn payload<T: Format>(lane: usize) -> T {
        T::from(match lane % 3 {
            0 => T::ONE,
            1 => 1,
            _ => T::SIGN,
        })
    }
    pub fn factor<T: Format>(half: bool) -> T {
        T::from(if half {
            T::ONE - (1 << T::FRACTION)
        } else {
            T::ONE + (1 << T::FRACTION)
        })
    }
}
#[path = "../../../../rocm/benches/low_tensor_producers/source/source.rs"]
#[allow(dead_code)] // Reciprocal authoring retains the original source fixture provenance.
mod source;
use source::ProducerSource;
#[rustfmt::skip]
use pcu_facade::{
    PcuNumericalMode,
    PcuNumericalOptions,
    PcuCompoundArithmeticPolicy,
    PcuPrecisionPolicy,
    PcuReproducibility,
    PcuFloatUnderflowPolicy,
    PcuF16Bits,
    PcuBf16Bits,
    PcuF8E4M3FnBits,
    PcuF8E5M2Bits,
};
use pcu_facade::dialect::tensor::{OpDescriptor, TensorElement};
use fusion_pcu_vulkan::PcuVulkanPreparedTensorGraph;
fn capture<T: ProducerSource + TensorElement, const N: usize, const HALF: bool>() {
    for mode in [PcuNumericalMode::Boundary, PcuNumericalMode::Strict] {
        for compound_arithmetic in [
            PcuCompoundArithmeticPolicy::Checked,
            PcuCompoundArithmeticPolicy::BackendDefined,
        ] {
            for precision in [
                PcuPrecisionPolicy::Preserve,
                PcuPrecisionPolicy::BackendOptimized,
            ] {
                for policy in [
                    PcuFloatUnderflowPolicy::IeeeAfterRounding,
                    PcuFloatUnderflowPolicy::AllowGradualUnderflow,
                    PcuFloatUnderflowPolicy::RejectSubnormalResult,
                ] {
                    let options = PcuNumericalOptions {
                        compound_arithmetic,
                        precision,
                        ..Default::default()
                    };
                    let captured = T::capture::<N, HALF>(mode, options, policy).unwrap();
                    let program = captured.program();
                    assert_eq!(captured.argument_indices(), [0]);
                    assert_eq!(program.input_values().len(), 1);
                    PcuVulkanPreparedTensorGraph::<T>::assess(
                        program.graph(),
                        program.output_values(),
                    )
                    .unwrap();
                    let mut producers = 0;
                    let mut arithmetic = 0;
                    for node in program.graph().nodes() {
                        match node.op {
                            OpDescriptor::Constant(value) => {
                                assert_eq!(node.shape, [N]);
                                assert_eq!(node.numerical_mode, None);
                                assert_eq!(node.float_underflow_policy, None);
                                oracle::bits(
                                    value.as_typed::<T>().unwrap().data(),
                                    &(0..N).map(oracle::payload).collect::<Vec<T>>(),
                                );
                                producers += 1;
                            }
                            OpDescriptor::Uniform { value } => {
                                assert_eq!(node.numerical_mode, None);
                                assert_eq!(node.float_underflow_policy, None);
                                oracle::bits(
                                    &[value.as_typed::<T>().unwrap()],
                                    &[oracle::factor::<T>(HALF)],
                                );
                                producers += 1;
                            }
                            OpDescriptor::Add { .. } | OpDescriptor::Mul { .. } => {
                                assert_eq!(node.numerical_mode, None);
                                assert_eq!(node.float_underflow_policy, Some(policy));
                                arithmetic += 1;
                            }
                            OpDescriptor::Input => continue,
                            _ => panic!("unexpected captured operation"),
                        }
                        assert_eq!(node.numerical_options, options);
                    }
                    assert_eq!((producers, arithmetic), (2, 2));
                    let portable = T::capture::<N, HALF>(
                        mode,
                        PcuNumericalOptions {
                            reproducibility: PcuReproducibility::PortableV1,
                            ..options
                        },
                        policy,
                    )
                    .unwrap();
                    assert!(
                        PcuVulkanPreparedTensorGraph::<T>::assess(
                            portable.program().graph(),
                            portable.program().output_values()
                        )
                        .is_err()
                    );
                }
            }
        }
    }
}
#[test]
fn low_producer_source_capture_matches_existing_vulkan_admission() {
    macro_rules! width {
        ($ty:ty) => {{
            capture::<$ty, 65, false>();
            capture::<$ty, 65, true>();
            capture::<$ty, 4096, false>();
            capture::<$ty, 4096, true>();
        }};
    }
    width!(PcuF16Bits);
    width!(PcuBf16Bits);
    width!(PcuF8E4M3FnBits);
    width!(PcuF8E5M2Bits);
}
