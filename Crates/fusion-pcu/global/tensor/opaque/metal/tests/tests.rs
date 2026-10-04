//! Exact source effects retain qualified leaf paths and cold graph eligibility.
#[rustfmt::skip]
use crate::{
    PcuExecutionError,
    PcuTensor,
    PcuRangePolicy,
    PcuReproducibility,
    PcuFloatUnderflowPolicy,
    global::{
        PcuBackendChoice,
        PcuExecutionPolicy,
        PcuSourceShape,
        tensor::{
            PcuTensorInput,
            PcuTensorShapeWitness,
            capture,
        },
    },
};

#[crate::pcu(crate_path = crate)]
fn checked_unused(input: &[f32]) -> Result<PcuTensor<f32>, PcuExecutionError> {
    let first = pcu::div(input, input);
    let _checked = pcu::mul(&first, input);
    pcu::identity(input)
}

#[crate::pcu(crate_path = crate)]
fn identity(input: &[f32]) -> Result<PcuTensor<f32>, PcuExecutionError> {
    pcu::identity(input)
}

#[crate::pcu(crate_path = crate, flag(allow_gradual_underflow))]
fn local_relu(input: &[f32]) -> Result<PcuTensor<f32>, PcuExecutionError> {
    pcu::relu(input)
}

#[crate::pcu(crate_path = crate, flag(allow_gradual_underflow))]
fn unused_relu(input: &[f32]) -> Result<PcuTensor<f32>, PcuExecutionError> {
    let _checked = pcu::relu(input);
    pcu::identity(input)
}

#[test]
fn multistage_checked_effects_are_retained_and_unsupported_policy_rejects_cold() {
    let options = PcuExecutionPolicy {
        backend: PcuBackendChoice::Metal,
        ..Default::default()
    };
    let shape = PcuSourceShape::Slice { length: 2 };
    let built = capture::build(
        [PcuTensorShapeWitness::Static(shape)],
        options.float_underflow,
        options.numerical_mode,
        options.numerical_options,
        checked_unused::__pcu_capture_entry,
    )
    .unwrap();
    assert_eq!(built.program.node_order().len(), 3);
    let super::program::Plan::Graph(plan) = super::assess(&built, options).unwrap() else {
        panic!("expected retained two-stage graph");
    };
    assert_eq!(plan.stage_count(), 2);
    assert_eq!(plan.output(), built.program.input_values()[0]);
    assert_eq!(
        plan.stage_identity(0).unwrap().0,
        built.program.node_order()[1]
    );
    assert_eq!(
        plan.stage_identity(1).unwrap().0,
        built.program.node_order()[2]
    );
    let inputs = [PcuTensorInput::host(&[f32::NAN, 1.0], shape)];
    let options = PcuExecutionPolicy {
        range_policy: PcuRangePolicy::Clamp,
        ..options
    };
    assert!(matches!(
        super::Prepared::prepare(&built, &inputs, options),
        Err(PcuExecutionError::InvalidTensorSourcePlan)
    ));
}

#[test]
fn unsupported_leaf_policy_rejects_before_discovery_and_native_allocation() {
    let mut options = PcuExecutionPolicy {
        backend: PcuBackendChoice::Metal,
        ..Default::default()
    };
    let shape = PcuSourceShape::Slice { length: 2 };
    let built = capture::build(
        [PcuTensorShapeWitness::Static(shape)],
        options.float_underflow,
        options.numerical_mode,
        options.numerical_options,
        identity::__pcu_capture_entry,
    )
    .unwrap();
    let inputs = [PcuTensorInput::host(&[f32::NAN, -0.0], shape)];
    options.range_policy = PcuRangePolicy::Clamp;
    assert!(matches!(
        super::Prepared::prepare(&built, &inputs, options),
        Err(PcuExecutionError::InvalidTensorSourcePlan)
    ));
    options.range_policy = PcuRangePolicy::Reject;
    options.numerical_options.reproducibility = PcuReproducibility::PortableV1;
    assert!(matches!(
        super::Prepared::prepare(&built, &inputs, options),
        Err(PcuExecutionError::InvalidTensorSourcePlan)
    ));
}

#[test]
fn admitted_relu_keeps_local_underflow_and_unused_checked_effect() {
    let options = PcuExecutionPolicy {
        backend: PcuBackendChoice::Metal,
        float_underflow: PcuFloatUnderflowPolicy::RejectSubnormalResult,
        ..Default::default()
    };
    let shape = PcuSourceShape::Slice { length: 2 };
    for unused in [false, true] {
        let built = capture::build(
            [PcuTensorShapeWitness::Static(shape)],
            options.float_underflow,
            options.numerical_mode,
            options.numerical_options,
            |capture, inputs| {
                if unused {
                    unused_relu::__pcu_capture_entry(capture, inputs)
                } else {
                    local_relu::__pcu_capture_entry(capture, inputs)
                }
            },
        )
        .unwrap();
        let super::program::Plan::Leaf(plan) = super::assess(&built, options).unwrap() else {
            panic!("expected unary plan")
        };
        let effect = built.program.node_order()[1];
        assert_eq!(plan.relu_effect(), Some(effect));
        assert_eq!(
            plan.requirements().float_underflow,
            PcuFloatUnderflowPolicy::AllowGradualUnderflow
        );
        assert_eq!(plan.output(), if unused { plan.input() } else { effect });
    }
}

#[crate::pcu(crate_path = crate, flag(allow_gradual_underflow))]
fn reversed_binary(left: &[f32], right: &[f32]) -> Result<PcuTensor<f32>, PcuExecutionError> {
    pcu::sub(right, left)
}

#[crate::pcu(crate_path = crate)]
fn repeated_with_unused<T: crate::PcuScalar>(
    input: &[T],
    _unused: &[T],
) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::add(input, input)
}

#[crate::pcu(crate_path = crate)]
fn checked_unused_binary(left: &[f32], right: &[f32]) -> Result<PcuTensor<f32>, PcuExecutionError> {
    let _checked = pcu::div(left, right);
    pcu::identity(left)
}

#[test]
fn binary_assessment_retains_real_operand_order_and_local_policy() {
    let options = PcuExecutionPolicy {
        backend: PcuBackendChoice::Metal,
        float_underflow: PcuFloatUnderflowPolicy::RejectSubnormalResult,
        ..Default::default()
    };
    let shape = PcuTensorShapeWitness::Static(PcuSourceShape::Slice { length: 2 });
    let built = capture::build(
        [shape; 2],
        options.float_underflow,
        options.numerical_mode,
        options.numerical_options,
        reversed_binary::__pcu_capture_entry,
    )
    .unwrap();
    let super::program::Plan::Binary(plan) = super::assess(&built, options).unwrap() else {
        panic!("expected binary plan");
    };
    assert_eq!(built.input_indices, [0, 1]);
    assert_eq!(plan.operand_inputs(), [1, 0]);
    assert_eq!(plan.output(), plan.effect());
    assert_eq!(
        plan.requirements().float_underflow,
        PcuFloatUnderflowPolicy::AllowGradualUnderflow
    );
}

#[test]
fn repeated_wide_source_does_not_create_an_unused_native_input() {
    let options = PcuExecutionPolicy {
        backend: PcuBackendChoice::Metal,
        ..Default::default()
    };
    let built = capture::build(
        [
            PcuTensorShapeWitness::Static(PcuSourceShape::Slice { length: 2 }),
            PcuTensorShapeWitness::Static(PcuSourceShape::Slice { length: 99 }),
        ],
        options.float_underflow,
        options.numerical_mode,
        options.numerical_options,
        repeated_with_unused::__pcu_capture_entry::<crate::PcuI512>,
    )
    .unwrap();
    let super::program::Plan::Binary(plan) = super::assess(&built, options).unwrap() else {
        panic!("expected binary plan");
    };
    assert_eq!(built.input_indices, [0]);
    assert_eq!(plan.input_values().len(), 1);
    assert_eq!(plan.operand_inputs(), [0, 0]);
    assert_eq!(plan.element_count(), 2);
    assert_eq!(plan.byte_len(), 128);
}

#[test]
fn checked_unused_binary_stays_an_effect_and_refuses_unsupported_policy_cold() {
    let mut options = PcuExecutionPolicy {
        backend: PcuBackendChoice::Metal,
        ..Default::default()
    };
    let shape = PcuSourceShape::Slice { length: 2 };
    let built = capture::build(
        [PcuTensorShapeWitness::Static(shape); 2],
        options.float_underflow,
        options.numerical_mode,
        options.numerical_options,
        checked_unused_binary::__pcu_capture_entry,
    )
    .unwrap();
    let super::program::Plan::Binary(plan) = super::assess(&built, options).unwrap() else {
        panic!("expected binary plan");
    };
    assert_eq!(plan.output(), plan.input_values()[0]);
    assert_ne!(plan.output(), plan.effect());
    let inputs = [
        PcuTensorInput::host(&[1.0_f32, 2.0], shape),
        PcuTensorInput::host(&[0.0_f32, 0.0], shape),
    ];
    options.range_policy = PcuRangePolicy::Clamp;
    assert!(matches!(
        super::Prepared::prepare(&built, &inputs, options),
        Err(PcuExecutionError::InvalidTensorSourcePlan)
    ));
}
