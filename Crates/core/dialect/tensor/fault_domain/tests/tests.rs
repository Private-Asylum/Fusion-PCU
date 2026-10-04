#[rustfmt::skip]
use crate::{
    PcuExecutionFault,
    PcuExecutionFaultKind as Kind,
    PcuCheckedFloat,
    PcuFloatUnderflowPolicy as Underflow,
    PcuNumericalMode,
    PcuScalarType as Scalar,
    dialect::tensor::{
        Graph,
        Tensor,
        TensorError,
        TensorElement,
        TensorArithmeticStep as Step,
        TensorStrictFaultDomain as Domain,
        TensorStrictFaultLocation as Location,
    },
};

const IEEE: Underflow = Underflow::IeeeAfterRounding;

#[test]
fn single_element_mean_cannot_introduce_a_new_fault() {
    for scalar in [Scalar::F32, Scalar::F64] {
        for policy in [
            IEEE,
            Underflow::AllowGradualUnderflow,
            Underflow::RejectSubnormalResult,
        ] {
            let domain = Domain::mse(scalar, 1, policy).unwrap();
            let mean = domain.location(3).unwrap();
            assert_eq!(mean.step, Step::Divide);
            for kind in [
                Kind::InvalidFloatingOperand,
                Kind::DivideByZero,
                Kind::ArithmeticOverflow,
                Kind::ArithmeticUnderflow,
            ] {
                for recovered in [false, true] {
                    assert!(!domain.allows(mean, kind, recovered));
                    assert!(!domain.accepts(PcuExecutionFault {
                        invocation_id: 3,
                        kind,
                        recovered,
                    }));
                }
            }
            // Larger means can still round an admitted sum to a tiny result.
            let larger = Domain::mse(scalar, 2, policy).unwrap();
            assert_eq!(
                larger.allows(
                    larger.location(6).unwrap(),
                    Kind::ArithmeticUnderflow,
                    false
                ),
                !matches!(policy, Underflow::AllowGradualUnderflow)
            );
        }
    }
}

#[test]
fn ordered_domains_are_not_output_or_launch_sizes() {
    let matmul = Domain::matmul(Scalar::F32, 6, 4, IEEE).unwrap();
    assert_eq!(matmul.event_extent(), 48);
    assert_eq!(
        matmul.location(47),
        Some(Location {
            element_index: 5,
            reduction_index: 3,
            step: Step::Add,
        })
    );
    let sgd = Domain::sgd(Scalar::F64, 19, IEEE).unwrap();
    assert_eq!(sgd.event_extent(), 38);
    assert_eq!(
        sgd.location(37),
        Some(Location {
            element_index: 18,
            reduction_index: 0,
            step: Step::Subtract,
        })
    );
    let mse = Domain::mse(Scalar::F32, 19, IEEE).unwrap();
    assert_eq!(mse.event_extent(), 58);
    assert_eq!(
        mse.location(56),
        Some(Location {
            element_index: 0,
            reduction_index: 18,
            step: Step::Add,
        })
    );
    assert_eq!(
        mse.location(57),
        Some(Location {
            element_index: 0,
            reduction_index: 19,
            step: Step::Divide,
        })
    );
    for domain in [matmul, sgd, mse] {
        for ordinal in 0..domain.event_extent() {
            assert_eq!(
                domain.ordinal(domain.location(ordinal).unwrap()),
                Some(ordinal)
            );
        }
        assert_eq!(domain.location(domain.event_extent()), None);
        assert_eq!(domain.location(u64::MAX), None);
    }
}

#[test]
fn coordinates_reject_wrong_step_reduction_and_scalar_output_domains() {
    let mse = Domain::mse(Scalar::F32, 4, IEEE).unwrap();
    for location in [
        Location {
            element_index: 1,
            reduction_index: 0,
            step: Step::Subtract,
        },
        Location {
            element_index: 0,
            reduction_index: 4,
            step: Step::Add,
        },
        Location {
            element_index: 0,
            reduction_index: 3,
            step: Step::Divide,
        },
    ] {
        assert_eq!(mse.ordinal(location), None);
    }
    let sgd = Domain::sgd(Scalar::F32, 4, IEEE).unwrap();
    assert_eq!(
        sgd.ordinal(Location {
            element_index: 0,
            reduction_index: 1,
            step: Step::Multiply
        }),
        None
    );
    assert_eq!(
        sgd.ordinal(Location {
            element_index: 0,
            reduction_index: 0,
            step: Step::Add
        }),
        None
    );
    let mm = Domain::matmul(Scalar::F64, 4, 3, IEEE).unwrap();
    assert_eq!(
        mm.ordinal(Location {
            element_index: 4,
            reduction_index: 0,
            step: Step::Multiply
        }),
        None
    );
    assert_eq!(
        mm.ordinal(Location {
            element_index: 0,
            reduction_index: 3,
            step: Step::Multiply
        }),
        None
    );
}

#[test]
fn genuine_high_u32_event_ordinals_remain_legal_in_neutral_domain() {
    let domain = Domain::sgd(Scalar::F32, (1_u64 << 32) + 1, IEEE).unwrap();
    let fault = PcuExecutionFault {
        invocation_id: 1_u64 << 32,
        kind: Kind::ArithmeticOverflow,
        recovered: false,
    };
    assert!(domain.accepts(fault));
    assert_eq!(
        domain.location(fault.invocation_id),
        Some(Location {
            element_index: 1_u64 << 31,
            reduction_index: 0,
            step: Step::Multiply,
        })
    );
    assert!(!domain.accepts(PcuExecutionFault {
        invocation_id: domain.event_extent(),
        ..fault
    }));
    assert!(!domain.accepts(PcuExecutionFault {
        recovered: true,
        ..fault
    }));
}

#[test]
fn checked_domain_construction_rejects_overflow_empty_mean_and_unimplemented_formats() {
    for scalar in [Scalar::F32, Scalar::F64] {
        assert!(Domain::matmul(scalar, u64::MAX, 2, IEEE).is_none());
        assert!(Domain::sgd(scalar, u64::MAX, IEEE).is_none());
        assert!(Domain::mse(scalar, u64::MAX / 3 + 1, IEEE).is_none());
        assert!(Domain::mse(scalar, 0, IEEE).is_none());
    }
    for scalar in Scalar::ALL {
        assert_eq!(
            Domain::mse(scalar, 1, IEEE).is_some(),
            matches!(scalar, Scalar::F32 | Scalar::F64)
        );
    }
    let max_pairs = Domain::sgd(Scalar::F64, u64::MAX / 2, IEEE).unwrap();
    let last = max_pairs.event_extent() - 1;
    assert_eq!(
        max_pairs.ordinal(max_pairs.location(last).unwrap()),
        Some(last)
    );
    let max_triples = Domain::mse(Scalar::F64, (u64::MAX - 1) / 3, IEEE).unwrap();
    let last = max_triples.event_extent() - 1;
    assert_eq!(
        max_triples.ordinal(max_triples.location(last).unwrap()),
        Some(last)
    );
}

#[test]
fn empty_arithmetic_domains_have_no_reportable_fault_positions() {
    for scalar in [Scalar::F32, Scalar::F64] {
        for domain in [
            Domain::matmul(scalar, 4, 0, IEEE).unwrap(),
            Domain::matmul(scalar, 0, 4, IEEE).unwrap(),
            Domain::sgd(scalar, 0, IEEE).unwrap(),
        ] {
            assert_eq!(domain.event_extent(), 0);
            assert_eq!(domain.location(0), None);
            assert_eq!(
                domain.ordinal(Location {
                    element_index: 0,
                    reduction_index: 0,
                    step: Step::Multiply
                }),
                None
            );
            assert!(!domain.accepts(PcuExecutionFault {
                kind: Kind::ArithmeticOverflow,
                invocation_id: 0,
                recovered: false
            }));
        }
        assert!(Domain::mse(scalar, 0, IEEE).is_none());
    }
    let mut graph = Graph::default();
    graph.set_numerical_mode(PcuNumericalMode::Strict);
    let a = graph.constant_typed(Tensor::<f32>::new([2, 0], [].to_vec()).unwrap());
    let b = graph.constant_typed(Tensor::<f32>::new([0, 3], [].to_vec()).unwrap());
    let output = graph.matmul_typed(a, b).unwrap();
    let execution = graph.evaluate_checked(&[]).unwrap();
    assert!(execution
        .value_typed::<f32>(output.erase())
        .unwrap()
        .data()
        .iter()
        .all(|value| value.to_bits() == 0));
}

#[test]
fn step_policies_and_known_nonzero_mean_denominator_reject_impossible_status() {
    let mse = Domain::mse(Scalar::F32, 2, IEEE).unwrap();
    let mean = mse.location(6).unwrap();
    assert!(mse.allows(mean, Kind::ArithmeticUnderflow, false));
    for kind in [
        Kind::DivideByZero,
        Kind::ArithmeticOverflow,
        Kind::SignedDivisionOverflow,
        Kind::InvalidFloatingOperand,
    ] {
        assert!(!mse.allows(mean, kind, false));
    }
    assert!(!mse.allows(mean, Kind::ArithmeticUnderflow, true));
    let relaxed = Domain::mse(Scalar::F32, 2, Underflow::AllowGradualUnderflow).unwrap();
    assert!(!relaxed.allows(mean, Kind::ArithmeticUnderflow, false));
    let add = mse.location(2).unwrap();
    assert!(!mse.allows(add, Kind::ArithmeticUnderflow, false));
    assert!(!mse.allows(add, Kind::DivideByZero, false));
}

fn assert_reference_location(error: &TensorError, domain: Domain, expected_ordinal: u64) {
    let TensorError::CompoundArithmeticFault {
        element_index,
        reduction_index,
        step,
        kind,
        ..
    } = error
    else {
        panic!("expected checked reference arithmetic fault")
    };
    let location = Location {
        element_index: (*element_index).try_into().unwrap(),
        reduction_index: (*reduction_index).try_into().unwrap(),
        step: *step,
    };
    assert_eq!(domain.ordinal(location), Some(expected_ordinal));
    assert!(domain.allows(location, *kind, false));
}

#[test]
fn independent_reference_faults_at_later_steps_match_the_neutral_order() {
    let mut graph = Graph::default();
    graph.set_numerical_mode(PcuNumericalMode::Strict);
    let a = graph.constant_typed(Tensor::new([1, 3], [1.0_f32, 1.0, f32::MAX].to_vec()).unwrap());
    let b = graph.constant_typed(Tensor::new([3, 1], [1.0_f32, 1.0, 2.0].to_vec()).unwrap());
    graph.matmul_typed(a, b).unwrap();
    assert_reference_location(
        &graph.evaluate_checked(&[]).unwrap_err(),
        Domain::matmul(Scalar::F32, 1, 3, IEEE).unwrap(),
        4,
    );

    let mut graph = Graph::default();
    graph.set_numerical_mode(PcuNumericalMode::Strict);
    let weights =
        graph.constant_typed(Tensor::new([3], [0.0_f32, 0.0, f32::NAN].to_vec()).unwrap());
    let gradient = graph.constant_typed(Tensor::new([3], [1.0_f32; 3].to_vec()).unwrap());
    graph.sgd_update_typed(weights, gradient, 1.0).unwrap();
    assert_reference_location(
        &graph.evaluate_checked(&[]).unwrap_err(),
        Domain::sgd(Scalar::F32, 3, IEEE).unwrap(),
        5,
    );

    let mut graph = Graph::default();
    graph.set_numerical_mode(PcuNumericalMode::Strict);
    let prediction =
        graph.constant_typed(Tensor::new([3], [1.0_f32, 1.0, f32::NAN].to_vec()).unwrap());
    let target = graph.constant_typed(Tensor::new([3], [0.0_f32; 3].to_vec()).unwrap());
    graph.mean_squared_error_typed(prediction, target).unwrap();
    assert_reference_location(
        &graph.evaluate_checked(&[]).unwrap_err(),
        Domain::mse(Scalar::F32, 3, IEEE).unwrap(),
        6,
    );
}

#[test]
fn final_mean_fault_is_legal_after_every_reduction_step_succeeds() {
    // (2^-74)^2 = 2^-148 is an exact tiny product. Dividing its sum by
    // three is tiny and inexact, so the final mean is the first IEEE fault.
    let mut graph = Graph::default();
    graph.set_numerical_mode(PcuNumericalMode::Strict);
    let prediction = graph
        .constant_typed(Tensor::new([3], [f32::from_bits(53 << 23), 0.0, 0.0].to_vec()).unwrap());
    let target = graph.constant_typed(Tensor::new([3], [0.0_f32; 3].to_vec()).unwrap());
    graph.mean_squared_error_typed(prediction, target).unwrap();
    let error = graph.evaluate_checked(&[]).unwrap_err();
    assert_reference_location(&error, Domain::mse(Scalar::F32, 3, IEEE).unwrap(), 9);
    assert!(matches!(
        error,
        TensorError::CompoundArithmeticFault {
            step: Step::Divide,
            kind: Kind::ArithmeticUnderflow,
            ..
        }
    ));
}

#[test]
fn dependent_steps_reject_faults_excluded_by_successful_predecessors() {
    let kinds = [
        Kind::ArithmeticUnderflow,
        Kind::ArithmeticOverflow,
        Kind::InvalidFloatingOperand,
        Kind::DivideByZero,
        Kind::SignedDivisionOverflow,
    ];
    for scalar in [Scalar::F32, Scalar::F64] {
        for policy in [
            IEEE,
            Underflow::RejectSubnormalResult,
            Underflow::AllowGradualUnderflow,
        ] {
            let dot = Domain::matmul(scalar, 2, 3, policy).unwrap();
            for cell in 0..2 {
                let first = Location {
                    element_index: cell,
                    reduction_index: 0,
                    step: Step::Add,
                };
                for kind in kinds {
                    assert!(!dot.allows(first, kind, false));
                }
                let later = Location {
                    reduction_index: 1,
                    ..first
                };
                assert!(dot.allows(later, Kind::ArithmeticOverflow, false));
                assert!(!dot.allows(later, Kind::InvalidFloatingOperand, false));
                assert_eq!(
                    dot.allows(later, Kind::ArithmeticUnderflow, false),
                    policy == Underflow::RejectSubnormalResult,
                );
            }
            let mse = Domain::mse(scalar, 3, policy).unwrap();
            for index in 0..3 {
                let square = Location {
                    element_index: 0,
                    reduction_index: index,
                    step: Step::Multiply,
                };
                assert!(!mse.allows(square, Kind::InvalidFloatingOperand, false));
                let sum = Location {
                    step: Step::Add,
                    ..square
                };
                for kind in kinds {
                    assert_eq!(
                        mse.allows(sum, kind, false),
                        index != 0 && kind == Kind::ArithmeticOverflow,
                    );
                    assert!(!mse.allows(sum, kind, true));
                }
            }
            // These steps still read external data: predecessor success cannot
            // prove the weight or the next prediction/target is finite.
            let sgd = Domain::sgd(scalar, 3, policy).unwrap();
            assert!(sgd.allows(
                sgd.location(1).unwrap(),
                Kind::InvalidFloatingOperand,
                false
            ));
            assert!(mse.allows(
                mse.location(3).unwrap(),
                Kind::InvalidFloatingOperand,
                false
            ));
        }
    }
}

fn check_finite_dependent_steps<T: PcuCheckedFloat + TensorElement>(values: &[T], zero: T, one: T) {
    for policy in [
        IEEE,
        Underflow::RejectSubnormalResult,
        Underflow::AllowGradualUnderflow,
    ] {
        for &value in values {
            let mut graph = Graph::default();
            graph.set_numerical_mode(PcuNumericalMode::Strict);
            let a = graph.constant_typed(Tensor::new([1, 1], [value].to_vec()).unwrap());
            let b = graph.constant_typed(Tensor::new([1, 1], [one].to_vec()).unwrap());
            let dot = graph.matmul_typed(a, b).unwrap();
            graph
                .set_value_float_underflow_policy(dot.erase(), policy)
                .unwrap();
            if let Err(error) = graph.evaluate_checked(&[]) {
                // The first +0 addition must never manufacture a range fault,
                // even when the product is exact tiny or at the format boundary.
                assert_reference_location(
                    &error,
                    Domain::matmul(T::TYPE, 1, 1, policy).unwrap(),
                    0,
                );
            }

            let mut graph = Graph::default();
            graph.set_numerical_mode(PcuNumericalMode::Strict);
            let prediction = graph.constant_typed(Tensor::new([2], [value; 2].to_vec()).unwrap());
            let target = graph.constant_typed(Tensor::new([2], [zero; 2].to_vec()).unwrap());
            let mse = graph.mean_squared_error_typed(prediction, target).unwrap();
            graph
                .set_value_float_underflow_policy(mse.erase(), policy)
                .unwrap();
            if let Err(error) = graph.evaluate_checked(&[]) {
                let TensorError::CompoundArithmeticFault {
                    element_index,
                    reduction_index,
                    step,
                    kind,
                    ..
                } = error
                else {
                    panic!("expected strict reference arithmetic fault");
                };
                assert!(Domain::mse(T::TYPE, 2, policy).unwrap().allows(
                    Location {
                        element_index: element_index.try_into().unwrap(),
                        reduction_index: reduction_index.try_into().unwrap(),
                        step,
                    },
                    kind,
                    false,
                ));
            }
        }
    }
}

#[test]
fn f32_reference_preserves_dependent_step_law_at_tiny_and_range_boundaries() {
    check_finite_dependent_steps(
        &[
            0.0_f32,
            -0.0,
            f32::from_bits(1),
            f32::from_bits(0x007f_ffff),
            f32::MIN_POSITIVE,
            f32::from_bits(53 << 23),
            0.5,
            -1.0,
            f32::from_bits((190 << 23) | (1 << 22)),
            f32::MAX,
            -f32::MAX,
        ],
        0.0,
        1.0,
    );
}

#[test]
fn f64_reference_preserves_dependent_step_law_at_tiny_and_range_boundaries() {
    check_finite_dependent_steps(
        &[
            0.0_f64,
            -0.0,
            f64::from_bits(1),
            f64::from_bits(0x000f_ffff_ffff_ffff),
            f64::MIN_POSITIVE,
            f64::from_bits(486 << 52),
            0.5,
            -1.0,
            f64::from_bits((1534 << 52) | (1 << 51)),
            f64::MAX,
            -f64::MAX,
        ],
        0.0,
        1.0,
    );
}
