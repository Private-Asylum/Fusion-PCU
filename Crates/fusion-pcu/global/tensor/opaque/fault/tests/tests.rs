use super::*;
#[rustfmt::skip]
use crate::{
    PcuScalarType, PcuExecutionFaultKind,
    dialect::tensor::{Graph, TensorArithmeticRewritePolicy, TensorArithmeticCapability,
        TensorPointwiseGroupingPolicy, TensorArithmeticStep},
};

#[test]
fn ordered_faults_keep_effect_coordinates_and_reject_impossible_status() {
    let requirements = PcuImplementationRequirements {
        numerical_mode: PcuNumericalMode::Strict,
        ..Default::default()
    };
    for operation in 0..3 {
        let mut graph = Graph::default();
        graph.set_numerical_mode(PcuNumericalMode::Strict);
        let left = graph.input([2, 2], PcuScalarType::F64).unwrap();
        let right = graph.input([2, 2], PcuScalarType::F64).unwrap();
        let effect = match operation {
            0 => graph.sgd_update(left, right, 0.5).unwrap(),
            1 => graph.matmul(left, right).unwrap(),
            _ => graph.mean_squared_error(left, right).unwrap(),
        };
        // Selected output is an input; the arithmetic effect remains mandatory.
        let program = graph
            .into_selected_program(
                &[right],
                TensorArithmeticRewritePolicy::Disabled,
                TensorArithmeticCapability::Strict,
                TensorPointwiseGroupingPolicy::Disabled,
            )
            .unwrap();
        let (ordinal, element, reduction, step) = match operation {
            0 => (5, 2, 0, TensorArithmeticStep::Subtract),
            1 => (5, 1, 0, TensorArithmeticStep::Add),
            _ => (11, 0, 3, TensorArithmeticStep::Add),
        };
        let fault = PcuExecutionFault {
            invocation_id: ordinal,
            kind: PcuExecutionFaultKind::ArithmeticOverflow,
            recovered: false,
        };
        // First reduction addition cannot overflow, even if its encoding is in range.
        if operation == 1 {
            assert!(matches!(
                numerical(&program, effect, requirements, fault),
                PcuExecutionError::InvalidTensorSourcePlan
            ));
            continue;
        }
        let error = numerical(&program, effect, requirements, fault);
        assert_eq!(
            error.arithmetic_fault().unwrap().invocation_id,
            element as u64
        );
        assert!(matches!(error, PcuExecutionError::TensorBuild(
            TensorError::CompoundArithmeticFault { value,element_index,reduction_index,
                step: actual,kind: PcuExecutionFaultKind::ArithmeticOverflow }
        ) if value==effect && element_index==element && reduction_index==reduction && actual==step));
        for invalid in [
            PcuExecutionFault {
                invocation_id: u64::MAX,
                ..fault
            },
            PcuExecutionFault {
                recovered: true,
                ..fault
            },
        ] {
            assert!(matches!(
                numerical(&program, effect, requirements, invalid),
                PcuExecutionError::InvalidTensorSourcePlan
            ));
        }
        let boundary = PcuImplementationRequirements {
            numerical_mode: PcuNumericalMode::Boundary,
            ..requirements
        };
        // A discarded Strict effect keeps its own domain even when the escaped Input
        // retains the global Boundary receipt. This is not Boundary-as-Strict admission.
        assert_eq!(
            numerical(&program, effect, boundary, fault).arithmetic_fault(),
            error.arithmetic_fault(),
        );
    }
}

#[test]
fn compound_effect_contract_overrides_outer_mode_and_tininess_without_widening() {
    let mut graph = Graph::try_new().unwrap();
    graph.set_numerical_mode(PcuNumericalMode::Strict);
    let left = graph.input([2, 2], PcuScalarType::F64).unwrap();
    let right = graph.input([2, 2], PcuScalarType::F64).unwrap();
    let effect = graph.matmul(left, right).unwrap();
    graph
        .set_value_float_underflow_policy(
            effect,
            crate::PcuFloatUnderflowPolicy::RejectSubnormalResult,
        )
        .unwrap();
    let program = graph
        .into_selected_program(
            &[right],
            TensorArithmeticRewritePolicy::Disabled,
            TensorArithmeticCapability::Strict,
            TensorPointwiseGroupingPolicy::Disabled,
        )
        .unwrap();
    let outer = PcuImplementationRequirements {
        numerical_mode: PcuNumericalMode::Boundary,
        float_underflow: crate::PcuFloatUnderflowPolicy::AllowGradualUnderflow,
        ..Default::default()
    };
    let fault = PcuExecutionFault {
        // A later exact cancellation can yield a subnormal after normal products.
        // Only reject-subnormal permits this Add status; the first Add cannot fault
        // because its already-checked product and +0 were admitted previously.
        invocation_id: 3,
        kind: PcuExecutionFaultKind::ArithmeticUnderflow,
        recovered: false,
    };
    assert!(matches!(numerical(&program, effect, outer, fault),
        PcuExecutionError::TensorBuild(TensorError::CompoundArithmeticFault {
            value, element_index: 0, reduction_index: 1,
            step: TensorArithmeticStep::Add,
            kind: PcuExecutionFaultKind::ArithmeticUnderflow,
        }) if value == effect));
    for policy in [
        crate::PcuFloatUnderflowPolicy::IeeeAfterRounding,
        crate::PcuFloatUnderflowPolicy::AllowGradualUnderflow,
    ] {
        let mut graph = Graph::try_new().unwrap();
        graph.set_numerical_mode(PcuNumericalMode::Strict);
        let left = graph.input([2, 2], PcuScalarType::F64).unwrap();
        let right = graph.input([2, 2], PcuScalarType::F64).unwrap();
        let effect = graph.matmul(left, right).unwrap();
        graph
            .set_value_float_underflow_policy(effect, policy)
            .unwrap();
        let program = graph
            .into_selected_program(
                &[right],
                TensorArithmeticRewritePolicy::Disabled,
                TensorArithmeticCapability::Strict,
                TensorPointwiseGroupingPolicy::Disabled,
            )
            .unwrap();
        let tightening_outer = PcuImplementationRequirements {
            numerical_mode: PcuNumericalMode::Strict,
            float_underflow: crate::PcuFloatUnderflowPolicy::RejectSubnormalResult,
            ..Default::default()
        };
        assert!(matches!(
            numerical(&program, effect, tightening_outer, fault),
            PcuExecutionError::InvalidTensorSourcePlan
        ));
    }
    // Conversely an actual Boundary effect is never decoded using Strict ordinals,
    // even if a global caller happens to request Strict.
    let mut graph = Graph::try_new().unwrap();
    graph.set_numerical_mode(PcuNumericalMode::Boundary);
    let left = graph.input([2, 2], PcuScalarType::F64).unwrap();
    let right = graph.input([2, 2], PcuScalarType::F64).unwrap();
    let effect = graph.matmul(left, right).unwrap();
    let program = graph
        .into_selected_program(
            &[right],
            TensorArithmeticRewritePolicy::Disabled,
            TensorArithmeticCapability::Strict,
            TensorPointwiseGroupingPolicy::Disabled,
        )
        .unwrap();
    assert!(matches!(
        numerical(
            &program,
            effect,
            PcuImplementationRequirements {
                numerical_mode: PcuNumericalMode::Strict,
                ..Default::default()
            },
            PcuExecutionFault {
                invocation_id: 0,
                kind: PcuExecutionFaultKind::ArithmeticOverflow,
                recovered: false
            }
        ),
        PcuExecutionError::InvalidTensorSourcePlan
    ));
}

#[test]
fn pointwise_stage_faults_keep_parent_identity_and_reject_impossible_classes() {
    let requirements = PcuImplementationRequirements::default();
    for (operation, scalar, accepted, rejected) in [
        (
            0,
            PcuScalarType::F64,
            PcuExecutionFaultKind::InvalidFloatingOperand,
            PcuExecutionFaultKind::ArithmeticOverflow,
        ),
        (
            1,
            PcuScalarType::F32,
            PcuExecutionFaultKind::ArithmeticOverflow,
            PcuExecutionFaultKind::ArithmeticUnderflow,
        ),
        (
            2,
            PcuScalarType::F64,
            PcuExecutionFaultKind::DivideByZero,
            PcuExecutionFaultKind::SignedDivisionOverflow,
        ),
        (
            3,
            PcuScalarType::U32,
            PcuExecutionFaultKind::ArithmeticUnderflow,
            PcuExecutionFaultKind::ArithmeticOverflow,
        ),
    ] {
        let mut graph = Graph::try_new().unwrap();
        let left = graph.input([4], scalar).unwrap();
        let right = graph.input([4], scalar).unwrap();
        let effect = match operation {
            0 => graph.relu_backward(left, right).unwrap(),
            1 => graph.add(left, right).unwrap(),
            2 => graph.div(left, right).unwrap(),
            _ => graph.sub(left, right).unwrap(),
        };
        let program = graph
            .into_selected_program(
                &[left],
                TensorArithmeticRewritePolicy::Disabled,
                TensorArithmeticCapability::Strict,
                TensorPointwiseGroupingPolicy::Disabled,
            )
            .unwrap();
        let fault = PcuExecutionFault {
            invocation_id: 2,
            kind: accepted,
            recovered: false,
        };
        assert!(
            matches!(numerical(&program,effect,requirements,fault),PcuExecutionError::TensorBuild(TensorError::ArithmeticFault{value,element_index:2,kind}) if value==effect && kind==accepted)
        );
        for bad in [
            PcuExecutionFault {
                kind: rejected,
                ..fault
            },
            PcuExecutionFault {
                invocation_id: 4,
                ..fault
            },
            PcuExecutionFault {
                recovered: true,
                ..fault
            },
        ] {
            assert!(matches!(
                numerical(&program, effect, requirements, bad),
                PcuExecutionError::InvalidTensorSourcePlan
            ));
        }
    }
}

#[test]
fn backward_exact_tininess_uses_original_node_policy() {
    let requirements = PcuImplementationRequirements::default();
    let mut graph = Graph::try_new().unwrap();
    let left = graph.input([1], PcuScalarType::F64).unwrap();
    let right = graph.input([1], PcuScalarType::F64).unwrap();
    let effect = graph.relu_backward(left, right).unwrap();
    let fault = PcuExecutionFault {
        invocation_id: 0,
        kind: PcuExecutionFaultKind::ArithmeticUnderflow,
        recovered: false,
    };
    graph
        .set_value_float_underflow_policy(
            effect,
            crate::PcuFloatUnderflowPolicy::RejectSubnormalResult,
        )
        .unwrap();
    let program = graph
        .into_selected_program(
            &[effect],
            TensorArithmeticRewritePolicy::Disabled,
            TensorArithmeticCapability::Strict,
            TensorPointwiseGroupingPolicy::Disabled,
        )
        .unwrap();
    assert!(
        matches!(numerical(&program,effect,requirements,fault),PcuExecutionError::TensorBuild(TensorError::ArithmeticFault{value,..}) if value==effect)
    );
    // Exact selected subnormals cannot report IEEE or explicitly gradual underflow.
    for policy in [
        crate::PcuFloatUnderflowPolicy::IeeeAfterRounding,
        crate::PcuFloatUnderflowPolicy::AllowGradualUnderflow,
    ] {
        let mut graph = Graph::try_new().unwrap();
        let left = graph.input([1], PcuScalarType::F64).unwrap();
        let right = graph.input([1], PcuScalarType::F64).unwrap();
        let effect = graph.relu_backward(left, right).unwrap();
        graph
            .set_value_float_underflow_policy(effect, policy)
            .unwrap();
        let program = graph
            .into_selected_program(
                &[effect],
                TensorArithmeticRewritePolicy::Disabled,
                TensorArithmeticCapability::Strict,
                TensorPointwiseGroupingPolicy::Disabled,
            )
            .unwrap();
        assert!(matches!(
            numerical(&program, effect, requirements, fault),
            PcuExecutionError::InvalidTensorSourcePlan
        ));
    }
}
