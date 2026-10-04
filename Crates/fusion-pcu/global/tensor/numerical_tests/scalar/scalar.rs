//! Scalar checks do not have a constituent-versus-primitive boundary distinction.
#[rustfmt::skip]
use super::{
    PcuExecutionError,
    PcuSourceShape,
    PcuTensorGraphCapture,
    PcuNumericalMode,
    PcuNumericalOptions,
    PcuReproducibility,
};
#[rustfmt::skip]
use crate::{
    PcuCheckedFloat,
    PcuFloatUnderflowPolicy,
    PcuPrecisionPolicy,
    PcuCompoundArithmeticPolicy,
};

macro_rules! binary {
    ($name:ident, $op:ident) => {
        #[crate::pcu(crate_path = crate)]
        fn $name<T: PcuCheckedFloat>(
            left: &[T],
            right: &[T],
        ) -> Result<crate::PcuTensor<T>, PcuExecutionError> {
            pcu::$op(left, right)
        }
    };
}
binary!(add, add);
binary!(sub, sub);
binary!(mul, mul);
binary!(div, div);

#[crate::pcu(crate_path = crate)]
fn relu<T: PcuCheckedFloat>(input: &[T]) -> Result<crate::PcuTensor<T>, PcuExecutionError> {
    pcu::relu(input)
}

#[crate::pcu(crate_path = crate, flag(strict), flag(reject_subnormal_result), flag(native_compound), flag(backend_precision))]
fn scoped<T: PcuCheckedFloat>(
    left: &[T],
    right: &[T],
) -> Result<crate::PcuTensor<T>, PcuExecutionError> {
    pcu::mul(left, right)
}

#[crate::pcu(crate_path = crate, flag(strict), flag(reject_subnormal_result), flag(native_compound), flag(backend_precision))]
fn scoped_helper<T: PcuCheckedFloat>(
    left: &[T],
    right: &[T],
) -> Result<crate::PcuTensor<T>, PcuExecutionError> {
    mul(left, right)
}

fn check<T: PcuCheckedFloat>() {
    let shape = PcuSourceShape::Slice { length: 3 };
    for mode in [PcuNumericalMode::Boundary, PcuNumericalMode::Strict] {
        for underflow in [
            PcuFloatUnderflowPolicy::IeeeAfterRounding,
            PcuFloatUnderflowPolicy::AllowGradualUnderflow,
            PcuFloatUnderflowPolicy::RejectSubnormalResult,
        ] {
            let (mut capture, inputs) = PcuTensorGraphCapture::new::<T, 2>([shape; 2]).unwrap();
            let options = PcuNumericalOptions {
                reproducibility: PcuReproducibility::PortableV1,
                ..Default::default()
            };
            capture.numerical_mode.set(mode);
            capture.numerical_options.set(options);
            capture.float_underflow_policy.set(underflow);
            // Unannotated helpers use the capture's base UF policy, whereas
            // compound mode and numerical options inherit the active scope.
            capture.base_float_underflow_policy = underflow;
            let outputs = [
                add::__pcu_capture_entry::<T>(&mut capture, inputs).unwrap(),
                sub::__pcu_capture_entry::<T>(&mut capture, inputs).unwrap(),
                mul::__pcu_capture_entry::<T>(&mut capture, inputs).unwrap(),
                div::__pcu_capture_entry::<T>(&mut capture, inputs).unwrap(),
                relu::__pcu_capture_entry::<T>(&mut capture, [inputs[0]]).unwrap(),
            ];
            for output in outputs {
                let node = capture.graph.node(output.value.erase()).unwrap();
                assert_eq!(node.scalar_type, T::TYPE);
                assert_eq!(node.numerical_mode, None);
                assert_eq!(node.float_underflow_policy, Some(underflow));
                assert_eq!(node.numerical_options, options);
                // Core deliberately refuses compound-mode metadata on a scalar node.
                assert!(matches!(
                    capture
                        .graph
                        .set_value_numerical_mode(output.value.erase(), mode),
                    Err(crate::dialect::tensor::TensorError::UnsupportedNumericalMode { .. })
                ));
            }
            let output = scoped::__pcu_capture_entry::<T>(&mut capture, inputs).unwrap();
            let node = capture.graph.node(output.value.erase()).unwrap();
            assert_eq!(node.numerical_mode, None);
            assert_eq!(
                node.float_underflow_policy,
                Some(PcuFloatUnderflowPolicy::RejectSubnormalResult)
            );
            assert_eq!(
                node.numerical_options,
                PcuNumericalOptions {
                    compound_arithmetic: PcuCompoundArithmeticPolicy::BackendDefined,
                    precision: PcuPrecisionPolicy::BackendOptimized,
                    reproducibility: PcuReproducibility::PortableV1,
                }
            );
            assert_eq!(capture.numerical_mode.get(), mode);
            assert_eq!(capture.float_underflow_policy.get(), underflow);
            assert_eq!(capture.numerical_options.get(), options);
            let scoped_options = node.numerical_options;
            let nested = scoped_helper::__pcu_capture_entry::<T>(&mut capture, inputs).unwrap();
            let nested_node = capture.graph.node(nested.value.erase()).unwrap();
            assert_eq!(nested_node.numerical_mode, None);
            assert_eq!(nested_node.float_underflow_policy, Some(underflow));
            assert_eq!(nested_node.numerical_options, scoped_options);
            assert_eq!(capture.numerical_mode.get(), mode);
            assert_eq!(capture.float_underflow_policy.get(), underflow);
            assert_eq!(capture.numerical_options.get(), options);
        }
    }
}

#[test]
fn six_float_scalar_sources_normalize_mode_without_losing_underflow_or_helper_options() {
    check::<crate::PcuF16Bits>();
    check::<crate::PcuBf16Bits>();
    check::<crate::PcuF8E4M3FnBits>();
    check::<crate::PcuF8E5M2Bits>();
    check::<f32>();
    check::<f64>();
}
