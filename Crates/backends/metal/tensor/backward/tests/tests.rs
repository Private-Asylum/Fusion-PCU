//! Selected graph policy/actual role preservation and immutable checked effect ownership.
use super::*;
use std::sync::Arc;
use fusion_pcu::dialect::tensor::TensorOwnedSelectedProgram;
use fusion_pcu::PcuImplementationRequirements;
use fusion_pcu::{
    PcuScalar, PcuScalarType, PcuFloatUnderflowPolicy as Policy, PcuRangePolicy,
    PcuReproducibility, PcuNumericalMode, PcuCompoundArithmeticPolicy, PcuPrecisionPolicy,
};
use fusion_pcu::dialect::tensor::{
    Graph, TensorArithmeticRewritePolicy, TensorArithmeticCapability, TensorPointwiseGroupingPolicy,
};
fn program(
    scalar: PcuScalarType,
    request: PcuImplementationRequirements,
    profile: u8,
) -> Arc<TensorOwnedSelectedProgram> {
    let mut graph = Graph::default();
    graph.set_numerical_mode(request.numerical_mode);
    graph.set_numerical_options(request.numerical_options);
    let left = graph.input([5], scalar).unwrap();
    let right = if profile == 0 {
        left
    } else {
        graph.input([5], scalar).unwrap()
    };
    let effect = graph.relu_backward(left, right).unwrap();
    graph
        .set_value_float_underflow_policy(effect, request.float_underflow)
        .unwrap();
    let output = match profile {
        2 => left,
        3 => right,
        _ => effect,
    };
    Arc::new(
        graph
            .into_selected_program(
                &[output],
                TensorArithmeticRewritePolicy::Disabled,
                TensorArithmeticCapability::Strict,
                TensorPointwiseGroupingPolicy::Disabled,
            )
            .unwrap(),
    )
}
fn tuples() -> Vec<PcuImplementationRequirements> {
    let mut values = Vec::new();
    for mode in [PcuNumericalMode::Boundary, PcuNumericalMode::Strict] {
        for compound in [
            PcuCompoundArithmeticPolicy::Checked,
            PcuCompoundArithmeticPolicy::BackendDefined,
        ] {
            for precision in [
                PcuPrecisionPolicy::Preserve,
                PcuPrecisionPolicy::BackendOptimized,
            ] {
                for policy in [
                    Policy::IeeeAfterRounding,
                    Policy::RejectSubnormalResult,
                    Policy::AllowGradualUnderflow,
                ] {
                    let mut value = PcuImplementationRequirements {
                        numerical_mode: mode,
                        float_underflow: policy,
                        ..Default::default()
                    };
                    value.numerical_options.compound_arithmetic = compound;
                    value.numerical_options.precision = precision;
                    values.push(value);
                }
            }
        }
    }
    values
}
#[test]
fn backward_selected_schema_freezes_full_tuple_roles_and_discarded_effect() {
    for scalar in [
        PcuScalarType::F16,
        PcuScalarType::BF16,
        PcuScalarType::F8E4M3FN,
        PcuScalarType::F8E5M2,
        PcuScalarType::F32,
        PcuScalarType::F64,
    ] {
        for request in tuples() {
            for profile in 0..4 {
                let source = program(scalar, request, profile);
                let plan = MetalTensorBackwardPlan::assess_program(&source, request).unwrap();
                assert_eq!(plan.requirements(), request);
                assert_eq!(plan.scalar_type(), scalar);
                assert_eq!(plan.shape(), &[5]);
                assert_eq!(plan.input_values().len(), if profile == 0 { 1 } else { 2 });
                let mut changed_mode = request;
                changed_mode.numerical_mode = if request.numerical_mode == PcuNumericalMode::Strict
                {
                    PcuNumericalMode::Boundary
                } else {
                    PcuNumericalMode::Strict
                };
                assert!(MetalTensorBackwardPlan::assess_program(&source, changed_mode).is_err());
                let mut other = request;
                other.range_policy = PcuRangePolicy::Clamp;
                assert!(MetalTensorBackwardPlan::assess_program(&source, other).is_err());
                other = request;
                other.numerical_options.reproducibility = PcuReproducibility::PortableV1;
                assert!(MetalTensorBackwardPlan::assess_program(&source, other).is_err());
                other = request;
                other.float_underflow = if request.float_underflow == Policy::RejectSubnormalResult
                {
                    Policy::AllowGradualUnderflow
                } else {
                    Policy::RejectSubnormalResult
                };
                assert!(MetalTensorBackwardPlan::assess_program(&source, other).is_err());
            }
        }
    }
}
fn bytes<T: PcuScalar>(values: &[T]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.encode_le().as_ref().to_vec())
        .collect()
}

fn owner<T: PcuScalar>(session: &MetalSession, values: &[T]) -> MetalTensorOwner {
    let mut graph = Graph::default();
    let input = graph.input([values.len()], T::TYPE).unwrap();
    let selected = graph
        .into_selected_program(
            &[input],
            TensorArithmeticRewritePolicy::Disabled,
            TensorArithmeticCapability::Strict,
            TensorPointwiseGroupingPolicy::Disabled,
        )
        .unwrap();
    let plan = super::super::MetalTensorPlan::assess_program(
        &selected,
        PcuImplementationRequirements::default(),
    )
    .unwrap();
    let map = session
        .prepare_tensor_program(plan, PcuMemoryPoolId(122))
        .unwrap();
    map.execute(MetalTensorInput::HostBytes {
        scalar: T::TYPE,
        elements: values.len(),
        bytes: &bytes(values),
    })
    .unwrap()
}
fn resident<T: PcuScalar>(owner: &MetalTensorOwner) -> MetalTensorInput<'_> {
    MetalTensorInput::Resident {
        scalar: T::TYPE,
        elements: 5,
        resource: owner.resource(),
    }
}
fn host<T: PcuScalar>(bytes: &[u8]) -> MetalTensorInput<'_> {
    MetalTensorInput::HostBytes {
        scalar: T::TYPE,
        elements: 5,
        bytes,
    }
}
fn native<T: fusion_pcu::PcuCheckedFloat>(
    session: &MetalSession,
    foreign: &MetalSession,
    values: [T; 5],
    gradient: [T; 5],
    invalid: T,
    tiny: T,
) {
    let input = owner(session, &values);
    let upstream = owner(session, &gradient);
    let other = owner(foreign, &gradient);
    let input_bytes = bytes(&values);
    let gradient_bytes = bytes(&gradient);
    for request in tuples() {
        for profile in 0..4 {
            let source = program(T::TYPE, request, profile);
            let prepared = session
                .prepare_tensor_backward_program(Arc::clone(&source), request, PcuMemoryPoolId(122))
                .unwrap();
            assert!(std::ptr::eq(prepared.program(), source.as_ref()));
            let ids = prepared.plan().input_values();
            let expected = if profile == 2 {
                values.to_vec()
            } else if profile == 3 {
                gradient.to_vec()
            } else {
                values
                    .iter()
                    .zip(if profile == 0 { &values } else { &gradient })
                    .map(|(&a, &b)| {
                        a.pcu_checked_relu_backward_with_policy(b, request.float_underflow)
                            .unwrap()
                    })
                    .collect()
            };
            let mut published = Vec::new();
            for mode in 0..4 {
                let left = if mode & 1 == 0 {
                    host::<T>(&input_bytes)
                } else {
                    resident::<T>(&input)
                };
                let result = if ids.len() == 1 {
                    prepared.execute(&[left])
                } else {
                    let right = if mode & 2 == 0 {
                        host::<T>(&gradient_bytes)
                    } else {
                        resident::<T>(&upstream)
                    };
                    prepared.execute(&[left, right])
                };
                let completed = result.unwrap();
                let mut output = values;
                completed.read_into(&mut output).unwrap();
                assert_eq!(bytes(&output), bytes(&expected));
                published.push((completed, bytes(&expected)));
            }
            let bad = owner(session, &[invalid; 5]);
            let fatal = if ids.len() == 1 {
                prepared.execute(&[resident::<T>(&bad)])
            } else {
                prepared.execute(&[resident::<T>(&input), resident::<T>(&bad)])
            };
            assert!(matches!(fatal,Err(MetalError::Arithmetic(fault)) if !fault.recovered));
            if ids.len() == 2 {
                assert!(matches!(
                    prepared.execute(&[resident::<T>(&input), resident::<T>(&other)]),
                    Err(MetalError::ForeignSession)
                ));
            }
            verify_faults(
                &prepared,
                session,
                ids,
                [values, gradient],
                [invalid, tiny],
                profile,
            );
            for (completed, expected_bytes) in &published {
                let mut output = values;
                completed.read_into(&mut output).unwrap();
                assert_eq!(bytes(&output), *expected_bytes);
            }
            let mut original = values;
            input.read_into(&mut original).unwrap();
            assert_eq!(bytes(&original), input_bytes);
        }
    }
}
#[test]
#[ignore = "Requires actual Metal selected backward graph ownership."]
fn backward_selected_graph_all_tuples_host_resident_discarded_fatal_and_old_owners() {
    let session = MetalSession::open(0).unwrap();
    let foreign = MetalSession::open(0).unwrap();
    native(
        &session,
        &foreign,
        [1.0_f32, -1.0, 0.0, -0.0, 2.0],
        [-0.0_f32, 3.0, 4.0, 5.0, -2.0],
        f32::NAN,
        f32::from_bits(1),
    );
    native(
        &session,
        &foreign,
        [1.0_f64, -1.0, 0.0, -0.0, 2.0],
        [-0.0_f64, 3.0, 4.0, 5.0, -2.0],
        f64::NAN,
        f64::from_bits(1),
    );
}

fn verify_faults<T: fusion_pcu::PcuCheckedFloat>(
    prepared: &MetalPreparedTensorBackwardProgram,
    session: &MetalSession,
    ids: &[fusion_pcu::dialect::tensor::ValueId],
    banks: [[T; 5]; 2],
    faults: [T; 2],
    profile: u8,
) {
    let [mut left, mut right] = banks;
    left[4] = faults[1];
    right[4] = faults[1];
    for masked_invalid in [false, true] {
        if masked_invalid {
            if ids.len() == 1 {
                left[1] = faults[0];
            }
            right[1] = faults[0];
        }
        let a = owner(session, &left);
        let b = owner(session, &right);
        let result = if ids.len() == 1 {
            prepared.execute(&[resident::<T>(&a)])
        } else {
            prepared.execute(&[resident::<T>(&a), resident::<T>(&b)])
        };
        if masked_invalid {
            assert!(
                matches!(result,Err(MetalError::Arithmetic(fault)) if fault.kind==fusion_pcu::PcuExecutionFaultKind::InvalidFloatingOperand&&fault.invocation_id==1)
            );
        } else if prepared.plan().requirements().float_underflow == Policy::RejectSubnormalResult {
            assert!(
                matches!(result,Err(MetalError::Arithmetic(fault)) if fault.kind==fusion_pcu::PcuExecutionFaultKind::ArithmeticUnderflow&&fault.invocation_id==4)
            );
        } else {
            let expected = if profile == 2 {
                left.to_vec()
            } else if profile == 3 {
                right.to_vec()
            } else {
                left.iter()
                    .zip(if ids.len() == 1 { &left } else { &right })
                    .map(|(&a, &b)| {
                        a.pcu_checked_relu_backward_with_policy(
                            b,
                            prepared.plan().requirements().float_underflow,
                        )
                        .unwrap()
                    })
                    .collect()
            };
            let completed = result.unwrap();
            let mut output = left;
            completed.read_into(&mut output).unwrap();
            assert_eq!(bytes(&output), bytes(&expected));
        }
    }
}

#[test]
#[ignore = "Requires actual Apple selected low-format backward graph ownership and ordered faults."]
fn backward_selected_four_low_formats_all_tuples_roles_and_faults() {
    let session = MetalSession::open(0).unwrap();
    let foreign = MetalSession::open(0).unwrap();
    macro_rules! low {
        ($ty:ty, $word:ty, $one:expr, $minus_one:expr, $negative_zero:expr, $two:expr, $minus_two:expr, $nan:expr) => {
            native(
                &session,
                &foreign,
                [$one, $minus_one, 0, $negative_zero, $two]
                    .map(|bits| <$ty>::from_bits(<$word>::try_from(bits).unwrap())),
                [$negative_zero, $one, $two, $one, $minus_two]
                    .map(|bits| <$ty>::from_bits(<$word>::try_from(bits).unwrap())),
                <$ty>::from_bits($nan),
                <$ty>::from_bits(1),
            );
        };
    }
    low!(
        fusion_pcu::PcuF16Bits,
        u16,
        0x3c00,
        0xbc00,
        0x8000,
        0x4000,
        0xc000,
        0x7e01
    );
    low!(
        fusion_pcu::PcuBf16Bits,
        u16,
        0x3f80,
        0xbf80,
        0x8000,
        0x4000,
        0xc000,
        0x7fc1
    );
    low!(
        fusion_pcu::PcuF8E4M3FnBits,
        u8,
        0x38,
        0xb8,
        0x80,
        0x40,
        0xc0,
        0x7f
    );
    low!(
        fusion_pcu::PcuF8E5M2Bits,
        u8,
        0x3c,
        0xbc,
        0x80,
        0x40,
        0xc0,
        0x7f
    );
}
