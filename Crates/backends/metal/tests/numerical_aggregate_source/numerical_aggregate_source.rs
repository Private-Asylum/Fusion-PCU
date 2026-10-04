//! Authentic annotated source capture; ordinary facade routing is a separate qualification.
use std::sync::Arc;
use pcu_facade::{
    pcu, PcuTensor, PcuExecutionError, global, PcuNumericalMode, PcuNumericalOptions,
    PcuFloatUnderflowPolicy,
};
use fusion_pcu_metal::{MetalSelectedNumericalTensorPlan, MetalSelectedNumericalTensorOperation};
#[pcu(crate_path=pcu_facade,flag(strict))]
fn backward(left: &[f32], right: &[f32]) -> Result<PcuTensor<f32>, PcuExecutionError> {
    pcu::relu_backward(left, right)
}
#[pcu(crate_path=pcu_facade,flag(strict))]
fn update(left: &[f32], right: &[f32]) -> Result<PcuTensor<f32>, PcuExecutionError> {
    pcu::sgd_update(left, right, 0.5_f32)
}
#[pcu(crate_path=pcu_facade,flag(strict))]
fn matrix(
    left: &[[f32; 3]; 2],
    right: &[[f32; 4]; 3],
) -> Result<PcuTensor<f32>, PcuExecutionError> {
    pcu::matmul(left, right)
}
#[pcu(crate_path=pcu_facade,flag(strict))]
fn loss(left: &[f32], right: &[f32]) -> Result<PcuTensor<f32>, PcuExecutionError> {
    pcu::mean_squared_error(left, right)
}
#[test]
fn annotated_numerical_aggregate_capture_retains_actual_source_and_tuple() {
    for precision in [
        pcu_facade::PcuPrecisionPolicy::Preserve,
        pcu_facade::PcuPrecisionPolicy::BackendOptimized,
    ] {
        for compound_arithmetic in [
            pcu_facade::PcuCompoundArithmeticPolicy::Checked,
            pcu_facade::PcuCompoundArithmeticPolicy::BackendDefined,
        ] {
            for float_underflow in [
                PcuFloatUnderflowPolicy::IeeeAfterRounding,
                PcuFloatUnderflowPolicy::RejectSubnormalResult,
                PcuFloatUnderflowPolicy::AllowGradualUnderflow,
            ] {
                let numerical_options = PcuNumericalOptions {
                    precision,
                    compound_arithmetic,
                    ..Default::default()
                };
                let slices = [global::PcuSourceShape::Slice { length: 6 }; 2];
                let matrices = [
                    global::PcuSourceShape::FixedMatrix {
                        rows: 2,
                        columns: 3,
                    },
                    global::PcuSourceShape::FixedMatrix {
                        rows: 3,
                        columns: 4,
                    },
                ];
                let captures = [
                    (
                        global::__pcu_capture_tensor_program(
                            slices,
                            float_underflow,
                            PcuNumericalMode::Strict,
                            numerical_options,
                            backward::__pcu_capture_entry,
                        )
                        .unwrap(),
                        MetalSelectedNumericalTensorOperation::ReluBackward,
                    ),
                    (
                        global::__pcu_capture_tensor_program(
                            slices,
                            float_underflow,
                            PcuNumericalMode::Strict,
                            numerical_options,
                            update::__pcu_capture_entry,
                        )
                        .unwrap(),
                        MetalSelectedNumericalTensorOperation::StrictSgd,
                    ),
                    (
                        global::__pcu_capture_tensor_program(
                            matrices,
                            float_underflow,
                            PcuNumericalMode::Strict,
                            numerical_options,
                            matrix::__pcu_capture_entry,
                        )
                        .unwrap(),
                        MetalSelectedNumericalTensorOperation::StrictMatMul,
                    ),
                    (
                        global::__pcu_capture_tensor_program(
                            slices,
                            float_underflow,
                            PcuNumericalMode::Strict,
                            numerical_options,
                            loss::__pcu_capture_entry,
                        )
                        .unwrap(),
                        MetalSelectedNumericalTensorOperation::StrictMse,
                    ),
                ];
                let request = pcu_facade::PcuImplementationRequirements {
                    numerical_mode: PcuNumericalMode::Strict,
                    numerical_options,
                    float_underflow,
                    ..Default::default()
                };
                for (capture, operation) in captures {
                    let plan = MetalSelectedNumericalTensorPlan::assess_program(
                        Arc::clone(capture.program()),
                        request,
                    )
                    .unwrap();
                    assert!(Arc::ptr_eq(&plan.program_owner(), capture.program()));
                    assert_eq!(plan.operation(), operation);
                    assert_eq!(plan.requirements(), request);
                    assert_eq!(plan.input_values(), capture.input_values());
                    assert_eq!(capture.argument_indices(), [0, 1]);
                    verify_layout(&plan, operation);
                }
            }
        }
    }
}

fn verify_layout(
    plan: &MetalSelectedNumericalTensorPlan,
    operation: MetalSelectedNumericalTensorOperation,
) {
    if operation == MetalSelectedNumericalTensorOperation::StrictSgd {
        assert_eq!(plan.learning_rate_bits(), Some(0.5f32.to_bits()));
    }
    if operation == MetalSelectedNumericalTensorOperation::StrictMse {
        assert_eq!(plan.shape(), &[]);
        assert_eq!(plan.element_count(), 1);
        assert_eq!(plan.input_element_count(0), Some(6));
        assert_eq!(plan.input_byte_len(1), Some(24));
    }
}
