use super::*;
use pcu_facade::{pcu, PcuTensor, PcuExecutionError, global};
#[pcu(crate_path=pcu_facade,flag(strict))]
fn training_f32(
    x: &[[f32; 2]; 2],
    transpose: &[[f32; 2]; 2],
    weights: &[[f32; 1]; 2],
    target: &[[f32; 1]; 2],
) -> Result<PcuTensor<f32>, PcuExecutionError> {
    let pre = pcu::matmul(x, weights)?;
    let prediction = pcu::relu(&pre)?;
    let _loss = pcu::mean_squared_error(&prediction, target)?;
    let difference = pcu::sub(&prediction, target)?;
    let derivative = pcu::relu_backward(&pre, &difference)?;
    let gradient = pcu::matmul(transpose, &derivative)?;
    pcu::sgd_update(weights, &gradient, 0.5)
}
#[pcu(crate_path=pcu_facade,flag(strict))]
fn training_f64(
    x: &[[f64; 2]; 2],
    transpose: &[[f64; 2]; 2],
    weights: &[[f64; 1]; 2],
    target: &[[f64; 1]; 2],
) -> Result<PcuTensor<f64>, PcuExecutionError> {
    let pre = pcu::matmul(x, weights)?;
    let prediction = pcu::relu(&pre)?;
    let _loss = pcu::mean_squared_error(&prediction, target)?;
    let difference = pcu::sub(&prediction, target)?;
    let derivative = pcu::relu_backward(&pre, &difference)?;
    let gradient = pcu::matmul(transpose, &derivative)?;
    pcu::sgd_update(weights, &gradient, 0.5)
}
#[test]
fn annotated_training_capture_preserves_actual_inputs_and_complete_effects() {
    let shapes = [
        global::PcuSourceShape::FixedMatrix {
            rows: 2,
            columns: 2,
        },
        global::PcuSourceShape::FixedMatrix {
            rows: 2,
            columns: 2,
        },
        global::PcuSourceShape::FixedMatrix {
            rows: 2,
            columns: 1,
        },
        global::PcuSourceShape::FixedMatrix {
            rows: 2,
            columns: 1,
        },
    ];
    for request in super::tests::requests() {
        let captures = [
            global::__pcu_capture_tensor_program(
                shapes,
                request.float_underflow,
                request.numerical_mode,
                request.numerical_options,
                training_f32::__pcu_capture_entry,
            )
            .unwrap(),
            global::__pcu_capture_tensor_program(
                shapes,
                request.float_underflow,
                request.numerical_mode,
                request.numerical_options,
                training_f64::__pcu_capture_entry,
            )
            .unwrap(),
        ];
        for capture in captures {
            let plan = MetalSelectedTensorGraphPlan::assess_program(
                Arc::clone(capture.program()),
                request,
            )
            .unwrap();
            assert!(Arc::ptr_eq(&plan.program_owner(), capture.program()));
            assert_eq!(plan.stage_count(), 7);
            assert_eq!(plan.input_values(), capture.input_values());
            assert_eq!(capture.argument_indices().len(), 4);
            let mut arguments = capture.argument_indices().to_vec();
            arguments.sort_unstable();
            assert_eq!(arguments, [0, 1, 2, 3]);
            for stage in &*plan.stages {
                let node = plan.program.graph().node(stage.effect).unwrap();
                assert_eq!(node.numerical_options, request.numerical_options);
                assert_eq!(node.float_underflow_policy, Some(request.float_underflow));
            }
        }
    }
}

use fusion_pcu::{PcuF16Bits, PcuBf16Bits, PcuF8E4M3FnBits, PcuF8E5M2Bits};
#[pcu(crate_path=pcu_facade,flag(strict))]
fn low_backward_f16(
    left: &[[PcuF16Bits; 2]; 2],
    right: &[[PcuF16Bits; 2]; 2],
    target: &[[PcuF16Bits; 2]; 2],
) -> Result<PcuTensor<PcuF16Bits>, PcuExecutionError> {
    let sum = pcu::add(left, right)?;
    let prediction = pcu::relu(&sum)?;
    let difference = pcu::sub(&prediction, target)?;
    pcu::relu_backward(&sum, &difference)
}
#[pcu(crate_path=pcu_facade,flag(strict))]
fn low_backward_bf16(
    left: &[[PcuBf16Bits; 2]; 2],
    right: &[[PcuBf16Bits; 2]; 2],
    target: &[[PcuBf16Bits; 2]; 2],
) -> Result<PcuTensor<PcuBf16Bits>, PcuExecutionError> {
    let sum = pcu::add(left, right)?;
    let prediction = pcu::relu(&sum)?;
    let difference = pcu::sub(&prediction, target)?;
    pcu::relu_backward(&sum, &difference)
}
#[pcu(crate_path=pcu_facade,flag(strict))]
fn low_backward_e4m3fn(
    left: &[[PcuF8E4M3FnBits; 2]; 2],
    right: &[[PcuF8E4M3FnBits; 2]; 2],
    target: &[[PcuF8E4M3FnBits; 2]; 2],
) -> Result<PcuTensor<PcuF8E4M3FnBits>, PcuExecutionError> {
    let sum = pcu::add(left, right)?;
    let prediction = pcu::relu(&sum)?;
    let difference = pcu::sub(&prediction, target)?;
    pcu::relu_backward(&sum, &difference)
}
#[pcu(crate_path=pcu_facade,flag(strict))]
fn low_backward_e5m2(
    left: &[[PcuF8E5M2Bits; 2]; 2],
    right: &[[PcuF8E5M2Bits; 2]; 2],
    target: &[[PcuF8E5M2Bits; 2]; 2],
) -> Result<PcuTensor<PcuF8E5M2Bits>, PcuExecutionError> {
    let sum = pcu::add(left, right)?;
    let prediction = pcu::relu(&sum)?;
    let difference = pcu::sub(&prediction, target)?;
    pcu::relu_backward(&sum, &difference)
}

#[test]
fn annotated_low_pointwise_backward_capture_preserves_all_roles_and_effects() {
    let shapes = [global::PcuSourceShape::FixedMatrix {
        rows: 2,
        columns: 2,
    }; 3];
    for request in super::tests::requests() {
        for capture in [
            global::__pcu_capture_tensor_program(
                shapes,
                request.float_underflow,
                request.numerical_mode,
                request.numerical_options,
                low_backward_f16::__pcu_capture_entry,
            )
            .unwrap(),
            global::__pcu_capture_tensor_program(
                shapes,
                request.float_underflow,
                request.numerical_mode,
                request.numerical_options,
                low_backward_bf16::__pcu_capture_entry,
            )
            .unwrap(),
            global::__pcu_capture_tensor_program(
                shapes,
                request.float_underflow,
                request.numerical_mode,
                request.numerical_options,
                low_backward_e4m3fn::__pcu_capture_entry,
            )
            .unwrap(),
            global::__pcu_capture_tensor_program(
                shapes,
                request.float_underflow,
                request.numerical_mode,
                request.numerical_options,
                low_backward_e5m2::__pcu_capture_entry,
            )
            .unwrap(),
        ] {
            let plan = MetalSelectedTensorGraphPlan::assess_program(
                Arc::clone(capture.program()),
                request,
            )
            .unwrap();
            assert!(Arc::ptr_eq(&plan.program_owner(), capture.program()));
            assert_eq!(plan.stage_count(), 4);
            assert_eq!(plan.input_values(), capture.input_values());
            assert_eq!(capture.argument_indices().len(), 3);
            let mut arguments = capture.argument_indices().to_vec();
            arguments.sort_unstable();
            assert_eq!(arguments, [0, 1, 2]);
            for stage in &*plan.stages {
                let node = plan.program.graph().node(stage.effect).unwrap();
                assert_eq!(node.numerical_options, request.numerical_options);
                assert_eq!(node.float_underflow_policy, Some(request.float_underflow));
            }
        }
    }
}
