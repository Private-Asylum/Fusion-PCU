//! Real annotated companions share the provider-independent cold builder.

#[path = "tests/unread/unread.rs"]
mod unread;

#[rustfmt::skip]
use super::{
    __pcu_capture_tensor_program,
    __pcu_capture_tensor_program_outputs,
    PcuExecutionError,
    PcuFloatUnderflowPolicy,
    PcuNumericalMode,
    PcuNumericalOptions,
    PcuSourceShape,
};
#[rustfmt::skip]
use crate::{
    PcuCompoundArithmeticPolicy,
    PcuPrecisionPolicy,
    PcuReproducibility,
};
use crate::dialect::tensor::OpDescriptor;

#[crate::pcu(crate_path = crate, flag(non_strict), flag(non_deterministic), flag(native_compound), flag(backend_precision))]
fn product<const M: usize, const K: usize, const N: usize>(
    left: &[[f32; K]; M],
    right: &[[f32; N]; K],
) -> Result<crate::PcuTensor<f32>, PcuExecutionError> {
    pcu::matmul(left, right)
}

#[crate::pcu(crate_path = crate)]
fn select_last(
    _unused: &[[f32; 2]; 2],
    selected: &[[f32; 2]; 2],
) -> Result<crate::PcuTensor<f32>, PcuExecutionError> {
    pcu::identity(selected)
}

#[test]
fn authentic_generic_source_retains_policies_operand_order_and_detached_ir() {
    let captured = __pcu_capture_tensor_program::<f32, 2, _>(
        [
            PcuSourceShape::FixedMatrix {
                rows: 3,
                columns: 2,
            },
            PcuSourceShape::FixedMatrix {
                rows: 2,
                columns: 4,
            },
        ],
        PcuFloatUnderflowPolicy::IeeeAfterRounding,
        PcuNumericalMode::Strict,
        PcuNumericalOptions {
            reproducibility: PcuReproducibility::PortableV1,
            ..Default::default()
        },
        product::__pcu_capture_entry::<3, 2, 4>,
    )
    .unwrap();
    assert_eq!(captured.argument_indices(), [0, 1]);
    assert_eq!(captured.input_values(), captured.program().input_values());
    let program = alloc::sync::Arc::clone(captured.program());
    drop(captured);
    let output = program.graph().node(program.output_values()[0]).unwrap();
    assert_eq!(output.shape, [3, 4]);
    assert_eq!(output.numerical_mode, Some(PcuNumericalMode::Boundary));
    assert_eq!(
        output.float_underflow_policy,
        Some(PcuFloatUnderflowPolicy::IeeeAfterRounding)
    );
    assert_eq!(
        output.numerical_options,
        PcuNumericalOptions {
            compound_arithmetic: PcuCompoundArithmeticPolicy::BackendDefined,
            precision: PcuPrecisionPolicy::BackendOptimized,
            ..Default::default()
        }
    );
    let OpDescriptor::MatMul { left, right, .. } = output.op else {
        panic!("annotated product must select MatMul")
    };
    assert_eq!(program.graph().node(left).unwrap().shape, [3, 2]);
    assert_eq!(program.graph().node(right).unwrap().shape, [2, 4]);
}

#[test]
fn unused_source_inputs_are_pruned_without_losing_argument_identity() {
    let captured = __pcu_capture_tensor_program::<f32, 2, _>(
        [PcuSourceShape::FixedMatrix {
            rows: 2,
            columns: 2,
        }; 2],
        PcuFloatUnderflowPolicy::IeeeAfterRounding,
        PcuNumericalMode::Boundary,
        PcuNumericalOptions::default(),
        select_last::__pcu_capture_entry,
    )
    .unwrap();
    assert_eq!(captured.argument_indices(), [1]);
    assert_eq!(captured.input_values().len(), 1);
    assert_eq!(captured.input_values(), captured.program().input_values());
}

#[test]
fn malformed_host_shapes_reject_before_calling_the_companion() {
    let shape = PcuSourceShape::FixedMatrix {
        rows: usize::MAX,
        columns: 2,
    };
    let result = __pcu_capture_tensor_program::<f32, 1, _>(
        [shape],
        PcuFloatUnderflowPolicy::IeeeAfterRounding,
        PcuNumericalMode::Boundary,
        PcuNumericalOptions::default(),
        |_, _| panic!("invalid input shapes must precede capture"),
    );
    assert!(matches!(
        result,
        Err(PcuExecutionError::InvalidTensorSourcePlan)
    ));
}

#[test]
fn tuple_selection_retains_output_order_shapes_and_only_real_source_bindings() {
    let captured = __pcu_capture_tensor_program_outputs::<f32, 3, 2, _>(
        [
            PcuSourceShape::FixedArray { length: 0 },
            PcuSourceShape::FixedArray { length: 2 },
            PcuSourceShape::FixedArray { length: 3 },
        ],
        PcuFloatUnderflowPolicy::IeeeAfterRounding,
        PcuNumericalMode::Boundary,
        PcuNumericalOptions::default(),
        |capture, [_, left, right]| Ok([capture.relu(right)?, capture.relu(left)?]),
    )
    .unwrap();
    assert_eq!(captured.argument_indices(), [1, 2]);
    let outputs = captured.program().output_values();
    assert_eq!(outputs.len(), 2);
    assert_eq!(captured.program().graph().shape(outputs[0]).unwrap(), [3]);
    assert_eq!(captured.program().graph().shape(outputs[1]).unwrap(), [2]);
}

#[test]
fn duplicate_logical_terminals_are_rejected_before_device_preparation() {
    let result = __pcu_capture_tensor_program_outputs::<f32, 1, 2, _>(
        [PcuSourceShape::FixedArray { length: 2 }],
        PcuFloatUnderflowPolicy::IeeeAfterRounding,
        PcuNumericalMode::Boundary,
        PcuNumericalOptions::default(),
        |capture, [input]| {
            let alias = capture.identity(input)?;
            Ok([input, alias])
        },
    );
    assert!(matches!(
        result,
        Err(PcuExecutionError::TensorBuild(
            crate::dialect::tensor::TensorError::DuplicateOutput(_)
        ))
    ));
}
