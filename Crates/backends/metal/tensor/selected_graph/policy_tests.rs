//! Separate heterogeneous stage qualification; source16 remains unchanged.
use super::*;
#[rustfmt::skip]
use super::super::{
    Envelope,
    MetalSelectedTensorGraphPlan,
};
#[rustfmt::skip]
use fusion_pcu::{
    PcuNumericalOptions,
    PcuFloatUnderflowPolicy,
    PcuCompoundArithmeticPolicy,
    PcuPrecisionPolicy,
};
fn other_options(request: PcuImplementationRequirements) -> PcuNumericalOptions {
    let mut options = request.numerical_options;
    options.precision = match options.precision {
        PcuPrecisionPolicy::Preserve => PcuPrecisionPolicy::BackendOptimized,
        PcuPrecisionPolicy::BackendOptimized => PcuPrecisionPolicy::Preserve,
    };
    options.compound_arithmetic = match options.compound_arithmetic {
        PcuCompoundArithmeticPolicy::Checked => PcuCompoundArithmeticPolicy::BackendDefined,
        PcuCompoundArithmeticPolicy::BackendDefined => PcuCompoundArithmeticPolicy::Checked,
    };
    options
}
fn other_underflow(request: PcuImplementationRequirements) -> PcuFloatUnderflowPolicy {
    if request.float_underflow == PcuFloatUnderflowPolicy::RejectSubnormalResult {
        PcuFloatUnderflowPolicy::AllowGradualUnderflow
    } else {
        PcuFloatUnderflowPolicy::RejectSubnormalResult
    }
}
fn heterogeneous_source(
    scalar: PcuScalarType,
    request: PcuImplementationRequirements,
    raw_output: bool,
) -> Arc<TensorOwnedSelectedProgram> {
    let mut graph = Graph::default();
    // Input storage carries deliberately different options. They are not
    // arithmetic permissions and must not constrain later checked operations.
    graph.set_numerical_options(other_options(request));
    let weights = graph.input([2, 2], scalar).unwrap();
    let target = graph.input([2, 2], scalar).unwrap();
    let earlier = graph.relu(target).unwrap();
    graph
        .set_value_float_underflow_policy(earlier, other_underflow(request))
        .unwrap();
    graph.set_numerical_options(request.numerical_options);
    graph.set_numerical_mode(PcuNumericalMode::Strict);
    let updated = graph.sgd_update(weights, earlier, 0.5).unwrap();
    graph
        .set_value_float_underflow_policy(updated, request.float_underflow)
        .unwrap();
    graph.set_numerical_options(other_options(request));
    let discarded = graph.relu(weights).unwrap();
    graph
        .set_value_float_underflow_policy(discarded, other_underflow(request))
        .unwrap();
    freeze(graph, if raw_output { weights } else { updated })
}
fn stage_request(stage: &super::super::Stage) -> PcuImplementationRequirements {
    match &stage.envelope {
        Envelope::Unary(leaf) => leaf.requirements(),
        Envelope::Binary(leaf) => leaf.requirements(),
        Envelope::Numerical(leaf) => leaf.requirements(),
    }
}
#[test]
fn heterogeneous_graph_freezes_each_stage_and_keeps_selected_output_envelope() {
    for request in requests() {
        for scalar in [PcuScalarType::F32, PcuScalarType::F64] {
            let source = heterogeneous_source(scalar, request, false);
            let plan =
                MetalSelectedTensorGraphPlan::assess_program(Arc::clone(&source), request).unwrap();
            assert!(Arc::ptr_eq(&source, &plan.program_owner()));
            assert_eq!(plan.requirements(), request);
            assert_eq!(plan.stage_count(), 3);
            let local = PcuImplementationRequirements {
                numerical_options: other_options(request),
                float_underflow: other_underflow(request),
                ..request
            };
            assert_eq!(stage_request(&plan.stages[0]), local);
            assert_eq!(stage_request(&plan.stages[1]), request);
            assert_eq!(stage_request(&plan.stages[2]), local);
            assert_ne!(plan.output(), plan.stages[2].effect);
            for wrong in [
                PcuImplementationRequirements {
                    numerical_options: other_options(request),
                    ..request
                },
                PcuImplementationRequirements {
                    float_underflow: other_underflow(request),
                    ..request
                },
                PcuImplementationRequirements {
                    numerical_mode: PcuNumericalMode::Boundary,
                    ..request
                },
            ] {
                assert!(
                    MetalSelectedTensorGraphPlan::assess_program(Arc::clone(&source), wrong)
                        .is_err()
                );
            }
        }
    }
}
#[test]
fn heterogeneous_raw_input_output_preserves_all_twenty_four_enclosing_headers() {
    for mut request in requests() {
        for mode in [PcuNumericalMode::Boundary, PcuNumericalMode::Strict] {
            request.numerical_mode = mode;
            for scalar in [PcuScalarType::F32, PcuScalarType::F64] {
                let source = heterogeneous_source(scalar, request, true);
                let plan =
                    MetalSelectedTensorGraphPlan::assess_program(Arc::clone(&source), request)
                        .unwrap();
                assert_eq!(plan.requirements(), request);
                assert!(source.input_values().contains(&plan.output()));
                assert_eq!(plan.stage_count(), 3);
                assert_eq!(
                    stage_request(&plan.stages[1]).numerical_mode,
                    PcuNumericalMode::Strict
                );
                assert_eq!(
                    stage_request(&plan.stages[1]).numerical_options,
                    request.numerical_options
                );
                assert_eq!(
                    stage_request(&plan.stages[0]).numerical_options,
                    other_options(request)
                );
            }
        }
    }
}

#[test]
fn portable_transport_header_requires_qualified_captured_computed_stages() {
    for request in requests() {
        let mut portable = request;
        portable.numerical_options.reproducibility = fusion_pcu::PcuReproducibility::PortableV1;
        let mut graph = Graph::default();
        graph.set_numerical_options(request.numerical_options);
        let input = graph.input([2, 2], PcuScalarType::F32).unwrap();
        assert!(
            MetalSelectedTensorGraphPlan::assess_program(freeze(graph, input), portable).is_err()
        );
        let program = heterogeneous_source(PcuScalarType::F32, request, true);
        let plan = MetalSelectedTensorGraphPlan::assess_program(program, portable).unwrap();
        assert_eq!(plan.requirements(), portable);
        assert_eq!(
            stage_request(&plan.stages[0])
                .numerical_options
                .reproducibility,
            fusion_pcu::PcuReproducibility::Unspecified
        );
        // Recapturing a local arithmetic effect as Portable still requires a
        // concrete Portable leaf, which this bounded backend has not qualified.
        let program = heterogeneous_source(PcuScalarType::F32, portable, true);
        assert!(MetalSelectedTensorGraphPlan::assess_program(program, portable).is_err());
    }
}

fn native_source(
    profile: usize,
    scalar: PcuScalarType,
    mut request: PcuImplementationRequirements,
) -> Arc<TensorOwnedSelectedProgram> {
    // Only the representative escaped-input header is Portable here. The
    // captured arithmetic effects retain their independently admitted options.
    request.numerical_options.reproducibility = fusion_pcu::PcuReproducibility::Unspecified;
    heterogeneous_source(scalar, request, profile != 0)
}
fn native_matrix<T: TensorElement + PcuCheckedFloat>(
    session: &MetalSession,
    foreign: &MetalSession,
    values: &[T],
    invalid: T,
    sentinel: T,
) {
    let strict = requests();
    super::native_with_requests(
        session,
        foreign,
        values,
        invalid,
        sentinel,
        native_source,
        (&strict, &[0, 2]),
    );
    let boundary: Vec<_> = strict
        .iter()
        .map(|&request| PcuImplementationRequirements {
            numerical_mode: PcuNumericalMode::Boundary,
            ..request
        })
        .collect();
    super::native_with_requests(
        session,
        foreign,
        values,
        invalid,
        sentinel,
        native_source,
        (&boundary, &[2]),
    );
    let portable: Vec<_> = strict
        .iter()
        .map(|&request| {
            let mut request = request;
            request.numerical_options.reproducibility = fusion_pcu::PcuReproducibility::PortableV1;
            request
        })
        .collect();
    super::native_with_requests(
        session,
        foreign,
        values,
        invalid,
        sentinel,
        native_source,
        (&portable, &[2]),
    );
}
#[test]
#[ignore = "Required actual independent stage requests, four native role layouts and retained fault owners."]
fn heterogeneous_graph_native_local_requests_roles_faults_and_retained_owners() {
    let session = MetalSession::open(0).unwrap();
    let foreign = MetalSession::open(0).unwrap();
    native_matrix(
        &session,
        &foreign,
        &[1.0_f32, 2.0, 3.0, 4.0],
        f32::NAN,
        -19.0,
    );
    native_matrix(
        &session,
        &foreign,
        &[1.0_f64, 2.0, 3.0, 4.0],
        f64::NAN,
        -19.0,
    );
    native_local_faults(&session, f32::from_bits(1), f32::NAN);
    native_local_faults(&session, f64::from_bits(1), f64::NAN);
}

fn fault_oracle<T: TensorElement + PcuCheckedFloat>(
    source: &TensorOwnedSelectedProgram,
    banks: &[[T; 4]; 2],
) -> (ValueId, u64, fusion_pcu::PcuExecutionFaultKind) {
    use fusion_pcu::dialect::tensor::{TensorStrictFaultDomain, TensorStrictFaultLocation};
    let data: Vec<_> = source
        .input_values()
        .iter()
        .zip(banks)
        .map(|(&id, bank)| {
            (
                id,
                TensorValue::from_tensor(Tensor::new([2, 2], bank.to_vec()).unwrap()),
            )
        })
        .collect();
    match source.graph().evaluate_checked(&data).unwrap_err() {
        TensorError::ArithmeticFault {
            value,
            element_index,
            kind,
        } => (value, u64::try_from(element_index).unwrap(), kind),
        TensorError::CompoundArithmeticFault {
            value,
            element_index,
            reduction_index,
            step,
            kind,
        } => {
            let node = source.graph().node(value).unwrap();
            let domain =
                TensorStrictFaultDomain::sgd(T::TYPE, 4, node.float_underflow_policy.unwrap())
                    .unwrap();
            let ordinal = domain
                .ordinal(TensorStrictFaultLocation {
                    element_index: u64::try_from(element_index).unwrap(),
                    reduction_index: u64::try_from(reduction_index).unwrap(),
                    step,
                })
                .unwrap();
            (value, ordinal, kind)
        }
        other => panic!("Unexpected heterogeneous fault oracle: {other:?}"),
    }
}
fn fault_banks<T: TensorElement + PcuCheckedFloat>(tiny: T, invalid: T) -> [[[T; 4]; 2]; 3] {
    let zero = tiny.pcu_checked_sub(tiny).unwrap();
    let mut last = [zero; 4];
    last[3] = tiny;
    let mut first_invalid = [zero; 4];
    first_invalid[0] = invalid;
    [[[zero; 4], last], [last, [zero; 4]], [first_invalid, last]]
}
#[test]
fn heterogeneous_oracle_distinguishes_local_underflow_and_stage_order() {
    fn check<T: TensorElement + PcuCheckedFloat>(tiny: T, invalid: T) {
        for request in requests() {
            for raw_output in [false, true] {
                let source = heterogeneous_source(T::TYPE, request, raw_output);
                let banks = fault_banks(tiny, invalid);
                let effects: Vec<_> = source
                    .operation_fragments()
                    .unwrap()
                    .iter()
                    .map(fusion_pcu::dialect::tensor::TensorOperationFragment::parent_value)
                    .collect();
                let first = fault_oracle(&source, &banks[0]);
                let later = fault_oracle(&source, &banks[1]);
                let simultaneous = fault_oracle(&source, &banks[2]);
                if other_underflow(request) == PcuFloatUnderflowPolicy::RejectSubnormalResult {
                    assert_eq!(first.0, effects[0]);
                    assert_eq!(first.1, 3);
                    assert_eq!(later.0, effects[2]);
                    assert_eq!(later.1, 3);
                    assert_eq!(simultaneous, first);
                } else {
                    assert_eq!(first.0, effects[1]);
                    assert_eq!(first.1, 6);
                    assert_eq!(later.0, effects[1]);
                    assert_eq!(later.1, 7);
                    assert_eq!(simultaneous.0, effects[1]);
                    assert_eq!(simultaneous.1, 1);
                }
            }
        }
    }
    check(f32::from_bits(1), f32::NAN);
    check(f64::from_bits(1), f64::NAN);
}
#[allow(clippy::too_many_lines)] // Exact local-fault coordinates, escaped old owner and retry stay in one independent native witness.
fn native_local_faults<T: TensorElement + PcuCheckedFloat>(
    session: &MetalSession,
    tiny: T,
    invalid: T,
) {
    let zero = tiny.pcu_checked_sub(tiny).unwrap();
    let good_raw = bytes(&[zero; 4]);
    for request in requests() {
        for raw_output in [false, true] {
            let source = heterogeneous_source(T::TYPE, request, raw_output);
            let plan =
                MetalSelectedTensorGraphPlan::assess_program(Arc::clone(&source), request).unwrap();
            let kernel = session
                .prepare_selected_tensor_graph(plan, PcuMemoryPoolId(193))
                .unwrap();
            let good: Vec<_> = source
                .input_values()
                .iter()
                .map(|&id| {
                    (
                        id,
                        MetalTensorInput::HostBytes {
                            scalar: T::TYPE,
                            elements: 4,
                            bytes: &good_raw,
                        },
                    )
                })
                .collect();
            let old = kernel.execute_mixed(&good).unwrap();
            for banks in fault_banks(tiny, invalid) {
                let (effect, ordinal, kind) = fault_oracle(&source, &banks);
                let raw: Vec<_> = banks.iter().map(|bank| bytes(bank)).collect();
                let residents: Vec<_> = kernel
                    .plan
                    .inputs
                    .iter()
                    .zip(&raw)
                    .map(|(spec, raw)| {
                        kernel
                            .owner(
                                session.upload_bytes(raw).unwrap(),
                                spec.scalar,
                                Rc::clone(&spec.shape),
                                spec.count,
                            )
                            .unwrap()
                    })
                    .collect();
                for pattern in 0..4 {
                    let mut bad: Vec<_> = source
                        .input_values()
                        .iter()
                        .zip(&raw)
                        .enumerate()
                        .map(|(index, (&id, raw))| {
                            let resident = pattern == 1
                                || pattern == 2 && index == 0
                                || pattern == 3 && index == 1;
                            (
                                id,
                                if resident {
                                    MetalTensorInput::Resident {
                                        scalar: T::TYPE,
                                        elements: 4,
                                        resource: residents[index].resource(),
                                    }
                                } else {
                                    MetalTensorInput::HostBytes {
                                        scalar: T::TYPE,
                                        elements: 4,
                                        bytes: raw,
                                    }
                                },
                            )
                        })
                        .collect();
                    // Binding order is independent of actual role identity.
                    if pattern == 3 {
                        bad.reverse();
                    }
                    let failure = kernel.execute_mixed(&bad).err().unwrap();
                    assert_eq!(failure.effect, Some(effect));
                    assert!(
                        matches!(failure.cause, MetalError::Arithmetic(fault) if !fault.recovered && fault.kind == kind && fault.invocation_id == ordinal)
                    );
                    let mut actual = [tiny; 6];
                    old.read_into(&mut actual).unwrap();
                    assert_eq!(bytes(&actual[..4]), good_raw);
                    assert_eq!(bytes(&actual[4..]), bytes(&[tiny; 2]));
                    let retry = kernel.execute_mixed(&good).unwrap();
                    let mut actual = [tiny; 6];
                    retry.read_into(&mut actual).unwrap();
                    assert_eq!(bytes(&actual[..4]), good_raw);
                    assert_eq!(bytes(&actual[4..]), bytes(&[tiny; 2]));
                }
            }
            drop(kernel);
            let mut actual = [tiny; 6];
            old.read_into(&mut actual).unwrap();
            assert_eq!(bytes(&actual[..4]), good_raw);
            assert_eq!(bytes(&actual[4..]), bytes(&[tiny; 2]));
        }
    }
}
