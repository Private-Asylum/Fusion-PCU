//! Real annotated companions share the provider-independent cold builder.

#[rustfmt::skip]
use super::{
    __pcu_capture_tensor_program,
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
    for (shape, empty) in [
        (PcuSourceShape::FixedArray { length: 0 }, true),
        (
            PcuSourceShape::FixedMatrix {
                rows: usize::MAX,
                columns: 2,
            },
            false,
        ),
    ] {
        let result = __pcu_capture_tensor_program::<f32, 1, _>(
            [shape],
            PcuFloatUnderflowPolicy::IeeeAfterRounding,
            PcuNumericalMode::Boundary,
            PcuNumericalOptions::default(),
            |_, _| panic!("invalid input shapes must precede capture"),
        );
        assert!(matches!(
            (empty, result),
            (true, Err(PcuExecutionError::EmptyTensorInput))
                | (false, Err(PcuExecutionError::InvalidTensorSourcePlan))
        ));
    }
}
