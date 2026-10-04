//! Real annotated helpers capture independent numerical options without executing a backend.

extern crate std;

#[path = "numerical_tests/identity/identity.rs"]
mod identity;
#[path = "numerical_tests/scalar/scalar.rs"]
mod scalar;

macro_rules! low_binary_helper {
    ($name:ident, $operation:ident) => {
        #[crate::pcu(crate_path = crate, flag(reject_subnormal_result))]
        fn $name<T: crate::PcuCheckedFloat>(
            left: &[T],
            right: &[T],
        ) -> Result<crate::PcuTensor<T>, crate::PcuExecutionError> {
            pcu::$operation(left, right)
        }
    };
}
low_binary_helper!(low_add, add);
low_binary_helper!(low_sub, sub);
low_binary_helper!(low_mul, mul);
low_binary_helper!(low_div, div);

fn low_policy<T: crate::PcuCheckedFloat>() {
    let shape = PcuSourceShape::Slice { length: 5 };
    let (mut capture, values) = PcuTensorGraphCapture::new::<T, 2>([shape; 2]).unwrap();
    for operation in [
        low_add::__pcu_capture_entry::<T>,
        low_sub::__pcu_capture_entry::<T>,
        low_mul::__pcu_capture_entry::<T>,
        low_div::__pcu_capture_entry::<T>,
    ] {
        let output = operation(&mut capture, values).unwrap();
        let node = capture.graph.node(output.value.erase()).unwrap();
        assert_eq!(node.scalar_type, T::TYPE);
        assert_eq!(
            node.float_underflow_policy,
            Some(crate::PcuFloatUnderflowPolicy::RejectSubnormalResult)
        );
        assert_eq!(
            capture.float_underflow_policy.get(),
            crate::PcuFloatUnderflowPolicy::IeeeAfterRounding
        );
    }
}

#[test]
fn low_float_owned_binary_helpers_capture_local_underflow_policy() {
    low_policy::<crate::PcuF16Bits>();
    low_policy::<crate::PcuBf16Bits>();
    low_policy::<crate::PcuF8E4M3FnBits>();
    low_policy::<crate::PcuF8E5M2Bits>();
}
#[rustfmt::skip]
use super::{
    PcuExecutionError,
    PcuSourceShape,
    PcuTensorGraphCapture,
};
#[rustfmt::skip]
use crate::{
    PcuCompoundArithmeticPolicy,
    PcuNumericalMode,
    PcuNumericalOptions,
    PcuNumericalOverrides,
    PcuPrecisionPolicy,
    PcuReproducibility,
};

#[crate::pcu(crate_path=crate)]
fn inherited(
    lhs: &[[f32; 1]; 1],
    rhs: &[[f32; 1]; 1],
) -> Result<crate::PcuTensor<f32>, PcuExecutionError> {
    Ok(pcu::matmul(lhs, rhs)?)
}

#[crate::pcu(crate_path=crate, flag(native_compound), flag(backend_precision))]
fn native(
    lhs: &[[f32; 1]; 1],
    rhs: &[[f32; 1]; 1],
) -> Result<crate::PcuTensor<f32>, PcuExecutionError> {
    inherited(lhs, rhs)
}

#[crate::pcu(crate_path=crate, flag(checked_compound), flag(preserve_precision), flag(non_deterministic))]
fn checked(
    lhs: &[[f32; 1]; 1],
    rhs: &[[f32; 1]; 1],
) -> Result<crate::PcuTensor<f32>, PcuExecutionError> {
    inherited(lhs, rhs)
}

#[crate::pcu(crate_path=crate, flag(strict), flag(deterministic))]
fn strict_portable(
    lhs: &[[f32; 1]; 1],
    rhs: &[[f32; 1]; 1],
) -> Result<crate::PcuTensor<f32>, PcuExecutionError> {
    inherited(lhs, rhs)
}

const fn shape() -> PcuSourceShape {
    PcuSourceShape::FixedMatrix {
        rows: 1,
        columns: 1,
    }
}

#[crate::pcu(crate_path=crate, flag(native_compound), flag(backend_precision))]
fn native_loss(
    prediction: &[[f32; 1]; 1],
    target: &[[f32; 1]; 1],
) -> Result<crate::PcuTensor<f32>, PcuExecutionError> {
    pcu::mean_squared_error(prediction, target)
}

#[crate::pcu(crate_path=crate, flag(non_strict), flag(native_compound), flag(backend_precision))]
fn native_update(
    weights: &[[f32; 1]; 1],
    gradient: &[[f32; 1]; 1],
) -> Result<crate::PcuTensor<f32>, PcuExecutionError> {
    pcu::sgd_update(weights, gradient, -0.25_f32)
}

#[test]
fn annotated_sgd_freezes_literal_rate_and_scoped_policies_without_weakening_caller() {
    let (mut capture, values) = PcuTensorGraphCapture::new::<f32, 2>([shape(), shape()]).unwrap();
    let outer = PcuNumericalOptions {
        reproducibility: PcuReproducibility::PortableV1,
        ..PcuNumericalOptions::default()
    };
    capture.numerical_mode.set(PcuNumericalMode::Strict);
    capture.numerical_options.set(outer);
    let updated = native_update::__pcu_capture_entry(&mut capture, values).unwrap();
    let node = capture.graph.node(updated.value.erase()).unwrap();
    assert_eq!(node.shape, [1, 1]);
    assert_eq!(node.numerical_mode, Some(PcuNumericalMode::Boundary));
    assert_eq!(
        node.numerical_options,
        PcuNumericalOptions {
            compound_arithmetic: PcuCompoundArithmeticPolicy::BackendDefined,
            precision: PcuPrecisionPolicy::BackendOptimized,
            reproducibility: PcuReproducibility::PortableV1,
        }
    );
    let crate::dialect::tensor::OpDescriptor::SgdUpdate { learning_rate, .. } = node.op else {
        panic!("source must capture an SGD primitive");
    };
    assert_eq!(learning_rate.to_bits(), (-0.25_f32).to_bits());
    assert_eq!(capture.numerical_mode.get(), PcuNumericalMode::Strict);
    assert_eq!(capture.numerical_options.get(), outer);
}

#[test]
fn annotated_loss_freezes_all_axes_and_restores_its_callers_policy() {
    let (mut capture, values) = PcuTensorGraphCapture::new::<f32, 2>([shape(), shape()]).unwrap();
    let outer = PcuNumericalOptions {
        reproducibility: PcuReproducibility::PortableV1,
        ..PcuNumericalOptions::default()
    };
    capture.numerical_options.set(outer);
    let loss = native_loss::__pcu_capture_entry(&mut capture, values).unwrap();
    let node = capture.graph.node(loss.value.erase()).unwrap();
    assert!(node.shape.is_empty());
    assert_eq!(
        node.numerical_options,
        PcuNumericalOptions {
            compound_arithmetic: PcuCompoundArithmeticPolicy::BackendDefined,
            precision: PcuPrecisionPolicy::BackendOptimized,
            reproducibility: PcuReproducibility::PortableV1,
        }
    );
    assert_eq!(capture.numerical_options.get(), outer);
}

#[test]
fn annotated_native_helper_inherits_other_axes_and_restores_caller() {
    let (mut capture, values) = PcuTensorGraphCapture::new::<f32, 2>([shape(), shape()]).unwrap();
    let outer = PcuNumericalOptions {
        reproducibility: PcuReproducibility::PortableV1,
        ..PcuNumericalOptions::default()
    };
    capture.numerical_options.set(outer);
    let result = native::__pcu_capture_entry(&mut capture, values).unwrap();
    let node = capture.graph.node(result.value.erase()).unwrap();
    assert_eq!(
        node.numerical_options,
        PcuNumericalOptions {
            compound_arithmetic: PcuCompoundArithmeticPolicy::BackendDefined,
            precision: PcuPrecisionPolicy::BackendOptimized,
            reproducibility: PcuReproducibility::PortableV1,
        }
    );
    assert_eq!(node.numerical_mode, Some(PcuNumericalMode::Boundary));
    assert_eq!(capture.numerical_options.get(), outer);
    let result = checked::__pcu_capture_entry(&mut capture, values).unwrap();
    assert_eq!(
        capture
            .graph
            .node(result.value.erase())
            .unwrap()
            .numerical_options,
        PcuNumericalOptions::default()
    );
    assert_eq!(capture.numerical_options.get(), outer);
}

#[test]
fn annotated_strict_portable_helper_preserves_compound_and_precision_permissions() {
    let (mut capture, values) = PcuTensorGraphCapture::new::<f32, 2>([shape(), shape()]).unwrap();
    let outer = PcuNumericalOptions {
        compound_arithmetic: PcuCompoundArithmeticPolicy::BackendDefined,
        precision: PcuPrecisionPolicy::BackendOptimized,
        ..PcuNumericalOptions::default()
    };
    capture.numerical_options.set(outer);
    let result = strict_portable::__pcu_capture_entry(&mut capture, values).unwrap();
    let node = capture.graph.node(result.value.erase()).unwrap();
    assert_eq!(node.numerical_mode, Some(PcuNumericalMode::Strict));
    assert_eq!(
        node.numerical_options,
        PcuNumericalOptions {
            reproducibility: PcuReproducibility::PortableV1,
            ..outer
        }
    );
    assert_eq!(capture.numerical_mode.get(), PcuNumericalMode::Boundary);
    assert_eq!(capture.numerical_options.get(), outer);
}

#[test]
fn nested_option_scopes_restore_on_result_error_and_unwind() {
    let (mut capture, _) = PcuTensorGraphCapture::new::<f32, 0>([]).unwrap();
    let native_options = PcuNumericalOptions {
        compound_arithmetic: PcuCompoundArithmeticPolicy::BackendDefined,
        ..PcuNumericalOptions::default()
    };
    capture.with_numerical_options(
        PcuNumericalOverrides {
            compound_arithmetic: Some(PcuCompoundArithmeticPolicy::BackendDefined),
            ..PcuNumericalOverrides::default()
        },
        |capture| {
            assert_eq!(capture.numerical_options.get(), native_options);
            let failed: Result<(), ()> = capture.with_numerical_options(
                PcuNumericalOverrides {
                    reproducibility: Some(PcuReproducibility::PortableV1),
                    ..PcuNumericalOverrides::default()
                },
                |_| Err(()),
            );
            assert_eq!(failed, Err(()));
            assert_eq!(capture.numerical_options.get(), native_options);
            let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                capture.with_numerical_options(
                    PcuNumericalOverrides {
                        precision: Some(PcuPrecisionPolicy::BackendOptimized),
                        ..PcuNumericalOverrides::default()
                    },
                    |_| panic!("scoped numerical override"),
                );
            }));
            assert!(panic.is_err());
            assert_eq!(capture.numerical_options.get(), native_options);
        },
    );
    assert_eq!(
        capture.numerical_options.get(),
        PcuNumericalOptions::default()
    );
}
