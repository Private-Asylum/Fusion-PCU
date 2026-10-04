use super::*;
#[rustfmt::skip]
use crate::{
    PcuCompoundArithmeticPolicy,
    PcuFloatUnderflowPolicy,
    PcuNumericalMode,
    PcuPrecisionPolicy,
    PcuTensor,
    global::{
        PcuSourceShape,
        tensor::{
            capture,
            PcuTensorShapeWitness,
        },
    },
};

#[crate::pcu(crate_path = crate, flag(allow_gradual_underflow), flag(backend_precision))]
fn relaxed(left: &[f32; 3], right: &[f32; 3]) -> Result<PcuTensor<f32>, PcuExecutionError> {
    pcu::mul(left, right)
}

#[crate::pcu(crate_path = crate, flag(reject_subnormal_result))]
fn tightened(left: &[f32; 3], right: &[f32; 3]) -> Result<PcuTensor<f32>, PcuExecutionError> {
    pcu::mul(left, right)
}

#[crate::pcu(crate_path = crate)]
fn escape_before_discard(
    left: &[f32; 3],
    right: &[f32; 3],
) -> Result<PcuTensor<f32>, PcuExecutionError> {
    let output = relaxed(left, right)?;
    let _discarded = tightened(left, right)?;
    Ok(output)
}

#[crate::pcu(crate_path = crate)]
fn input_only(input: &[f32; 3]) -> Result<PcuTensor<f32>, PcuExecutionError> {
    pcu::identity(input)
}

#[test]
fn later_discarded_effect_cannot_replace_selected_output_header() {
    let options = PcuExecutionPolicy::default();
    let built = capture::build(
        [PcuTensorShapeWitness::Static(PcuSourceShape::FixedArray { length: 3 }); 2],
        options.float_underflow,
        options.numerical_mode,
        options.numerical_options,
        escape_before_discard::__pcu_capture_entry,
    )
    .unwrap();
    let request = requirements(&built, options).unwrap();
    assert_eq!(
        request.float_underflow,
        PcuFloatUnderflowPolicy::AllowGradualUnderflow
    );
    assert_eq!(
        request.numerical_options.precision,
        PcuPrecisionPolicy::BackendOptimized
    );
    let final_effect = built
        .program
        .graph()
        .node(*built.program.node_order().last().unwrap())
        .unwrap();
    assert_ne!(final_effect.value, built.program.output_values()[0]);
    assert_eq!(
        final_effect.float_underflow_policy,
        Some(PcuFloatUnderflowPolicy::RejectSubnormalResult)
    );
    assert_eq!(
        final_effect.numerical_options.precision,
        PcuPrecisionPolicy::Preserve
    );
}

#[test]
fn escaped_input_is_transport_under_every_outer_header() {
    for mode in [PcuNumericalMode::Boundary, PcuNumericalMode::Strict] {
        for underflow in [
            PcuFloatUnderflowPolicy::IeeeAfterRounding,
            PcuFloatUnderflowPolicy::AllowGradualUnderflow,
            PcuFloatUnderflowPolicy::RejectSubnormalResult,
        ] {
            for precision in [
                PcuPrecisionPolicy::Preserve,
                PcuPrecisionPolicy::BackendOptimized,
            ] {
                for compound in [
                    PcuCompoundArithmeticPolicy::Checked,
                    PcuCompoundArithmeticPolicy::BackendDefined,
                ] {
                    let mut options = PcuExecutionPolicy {
                        numerical_mode: mode,
                        float_underflow: underflow,
                        ..Default::default()
                    };
                    options.numerical_options.precision = precision;
                    options.numerical_options.compound_arithmetic = compound;
                    let built = capture::build(
                        [PcuTensorShapeWitness::Static(PcuSourceShape::FixedArray {
                            length: 3,
                        })],
                        underflow,
                        mode,
                        options.numerical_options,
                        input_only::__pcu_capture_entry,
                    )
                    .unwrap();
                    let request = requirements(&built, options).unwrap();
                    assert_eq!(request.numerical_mode, mode);
                    assert_eq!(request.float_underflow, underflow);
                    assert_eq!(request.numerical_options, options.numerical_options);
                }
            }
        }
    }
}
