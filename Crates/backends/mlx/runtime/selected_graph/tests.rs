use super::*;
use fusion_pcu::{
    PcuNumericalMode, PcuCompoundArithmeticPolicy, PcuPrecisionPolicy, PcuFloatUnderflowPolicy,
    PcuHostArgument, PcuBindingRef, PcuCheckedFloat,
};
use fusion_pcu::dialect::tensor::{
    Graph, TensorArithmeticRewritePolicy, TensorArithmeticCapability,
    TensorPointwiseGroupingPolicy, Tensor, TensorElement, TensorValue, TensorError,
};
pub(super) fn requests() -> Vec<PcuImplementationRequirements> {
    let mut requests = Vec::new();
    for precision in [
        PcuPrecisionPolicy::Preserve,
        PcuPrecisionPolicy::BackendOptimized,
    ] {
        for compound in [
            PcuCompoundArithmeticPolicy::Checked,
            PcuCompoundArithmeticPolicy::BackendDefined,
        ] {
            for float_underflow in [
                PcuFloatUnderflowPolicy::IeeeAfterRounding,
                PcuFloatUnderflowPolicy::RejectSubnormalResult,
                PcuFloatUnderflowPolicy::AllowGradualUnderflow,
            ] {
                let mut request = PcuImplementationRequirements {
                    numerical_mode: PcuNumericalMode::Strict,
                    float_underflow,
                    ..Default::default()
                };
                request.numerical_options.precision = precision;
                request.numerical_options.compound_arithmetic = compound;
                requests.push(request);
            }
        }
    }
    requests
}
pub(super) fn freeze(graph: Graph, output: ValueId) -> Arc<TensorOwnedSelectedProgram> {
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
pub(super) fn source(
    profile: usize,
    scalar: PcuScalarType,
    request: PcuImplementationRequirements,
) -> Arc<TensorOwnedSelectedProgram> {
    let mut graph = Graph::default();
    graph.set_numerical_mode(request.numerical_mode);
    graph.set_numerical_options(request.numerical_options);
    let x = graph.input([2, 2], scalar).unwrap();
    let output = if profile < 3 {
        let transpose = graph.input([2, 2], scalar).unwrap();
        let weights = graph.input([2, 1], scalar).unwrap();
        let target = graph.input([2, 1], scalar).unwrap();
        let pre = graph.matmul(x, weights).unwrap();
        let prediction = graph.relu(pre).unwrap();
        let loss = graph.mean_squared_error(prediction, target).unwrap();
        let difference = graph.sub(prediction, target).unwrap();
        let derivative = graph.relu_backward(pre, difference).unwrap();
        let gradient = graph.matmul(transpose, derivative).unwrap();
        let updated = graph.sgd_update(weights, gradient, 0.5).unwrap();
        match profile {
            0 => updated,
            1 => loss,
            _ => weights,
        }
    } else {
        let target = graph.input([2, 2], scalar).unwrap();
        let prediction = graph.relu(x).unwrap();
        let difference = graph.sub(prediction, target).unwrap();
        graph.mean_squared_error(difference, difference).unwrap();
        graph.sgd_update(x, difference, 0.25).unwrap()
    };
    let effects: Vec<_> = graph
        .nodes()
        .filter(|node| !matches!(node.op, OpDescriptor::Input))
        .map(|node| node.value)
        .collect();
    for effect in effects {
        graph
            .set_value_float_underflow_policy(effect, request.float_underflow)
            .unwrap();
    }
    freeze(graph, output)
}
#[test]
fn graph_plan_preserves_full_parent_schedule_and_last_use() {
    for request in requests() {
        for scalar in [PcuScalarType::F32, PcuScalarType::F64] {
            for profile in 0..4 {
                let source = source(profile, scalar, request);
                let plan = MlxSelectedTensorGraphPlan::assess_program(Arc::clone(&source), request)
                    .unwrap();
                assert!(Arc::ptr_eq(&source, &plan.program_owner()));
                assert_eq!(plan.requirements(), request);
                assert_eq!(plan.stage_count(), if profile < 3 { 7 } else { 4 });
                assert_eq!(plan.output(), source.output_values()[0]);
                assert_eq!(plan.slots, source.operations().len());
                for stage in &*plan.stages {
                    assert_eq!(
                        source.operation_index_of(stage.effect),
                        Some(stage.operation_index)
                    );
                    for &release in &*stage.releases {
                        let life = &source.operation_liveness()[release];
                        assert_eq!(life.last_live_node, stage.operation_index);
                        assert_ne!(life.value, plan.output);
                    }
                    for &(child, parent_slot) in &*stage.bindings {
                        assert!(stage.program.input_values().contains(&child));
                        assert!(parent_slot < stage.operation_index);
                    }
                }
            }
        }
    }
}
#[test]
fn graph_plan_refuses_mismatch_constant_and_multiple_outputs() {
    let request = requests()[0];
    let source = source(0, PcuScalarType::F32, request);
    let mut wrong = request;
    wrong.numerical_options.compound_arithmetic = PcuCompoundArithmeticPolicy::BackendDefined;
    assert!(MlxSelectedTensorGraphPlan::assess_program(source, wrong).is_err());
    let mut graph = Graph::default();
    let input = graph.input([2], PcuScalarType::F32).unwrap();
    let constant = graph.constant_typed(Tensor::new([2], vec![1.0_f32; 2]).unwrap());
    let result = graph.add(input, constant.erase()).unwrap();
    assert!(MlxSelectedTensorGraphPlan::assess_program(freeze(graph, result), request).is_err());
    let mut graph = Graph::default();
    let input = graph.input([2], PcuScalarType::F32).unwrap();
    let result = graph.relu(input).unwrap();
    let source = Arc::new(
        graph
            .into_selected_program(
                &[input, result],
                TensorArithmeticRewritePolicy::Disabled,
                TensorArithmeticCapability::Strict,
                TensorPointwiseGroupingPolicy::Disabled,
            )
            .unwrap(),
    );
    assert!(MlxSelectedTensorGraphPlan::assess_program(source, request).is_err());
}
pub(super) fn bytes<T: fusion_pcu::PcuScalar>(values: &[T]) -> Vec<u8> {
    PcuHostArgument::read(PcuBindingRef::new(0, 0), values)
        .bytes()
        .to_vec()
}
fn input_values<T: TensorElement + PcuCheckedFloat>(
    source: &TensorOwnedSelectedProgram,
    values: &[T],
) -> Vec<(ValueId, TensorValue)> {
    source
        .input_values()
        .iter()
        .enumerate()
        .map(|(index, &id)| {
            let shape = source.graph().node(id).unwrap().shape;
            let count = shape.iter().product::<usize>();
            let data = if index == 1 && source.input_values().len() == 4 {
                vec![values[0], values[2], values[1], values[3]]
            } else {
                values[..count].to_vec()
            };
            (
                id,
                TensorValue::from_tensor(Tensor::new(shape, data).unwrap()),
            )
        })
        .collect()
}
fn native<T: TensorElement + PcuCheckedFloat>(
    session: &MlxSession,
    foreign: &MlxSession,
    values: &[T],
    invalid: T,
    sentinel: T,
    build: fn(
        usize,
        PcuScalarType,
        PcuImplementationRequirements,
    ) -> Arc<TensorOwnedSelectedProgram>,
) {
    native_with_requests(
        session,
        foreign,
        values,
        invalid,
        sentinel,
        build,
        (&requests(), &[0, 1, 2, 3]),
    );
}

#[allow(clippy::too_many_lines)] // One native whole-graph matrix retains independent oracle and old-owner failure witnesses.
fn native_with_requests<T: TensorElement + PcuCheckedFloat>(
    session: &MlxSession,
    foreign: &MlxSession,
    values: &[T],
    invalid: T,
    sentinel: T,
    build: fn(
        usize,
        PcuScalarType,
        PcuImplementationRequirements,
    ) -> Arc<TensorOwnedSelectedProgram>,
    matrix: (&[PcuImplementationRequirements], &[usize]),
) {
    let (matrix_requests, profiles) = matrix;
    for &request in matrix_requests {
        if request.numerical_options.reproducibility == fusion_pcu::PcuReproducibility::Unspecified
        {
            phase_order(session, request, values, invalid);
        }
        for &profile in profiles {
            let source = build(profile, T::TYPE, request);
            let plan =
                MlxSelectedTensorGraphPlan::assess_program(Arc::clone(&source), request).unwrap();
            let kernel = session.prepare_selected_tensor_graph(plan).unwrap();
            let data = input_values(&source, values);
            let reference = source.graph().evaluate_checked(&data).unwrap();
            let expected = reference
                .value_typed::<T>(source.output_values()[0])
                .unwrap()
                .data();
            let raw: Vec<_> = data
                .iter()
                .map(|(_, value)| bytes(value.as_typed::<T>().unwrap().data()))
                .collect();
            let residents: Vec<_> = kernel
                .plan
                .inputs
                .iter()
                .zip(&raw)
                .map(|(spec, raw)| {
                    session
                        .upload_encoded_bytes(spec.scalar, spec.count, raw)
                        .unwrap()
                })
                .collect();
            let mut retained = None;
            for pattern in 0..4 {
                let mut bindings: Vec<_> = data
                    .iter()
                    .enumerate()
                    .map(|(index, (id, _))| {
                        (
                            *id,
                            if pattern == 1 || pattern == 2 && index.is_multiple_of(2) {
                                MlxCheckedProgramInput::Resident(&residents[index])
                            } else {
                                MlxCheckedProgramInput::Host {
                                    scalar: T::TYPE,
                                    bytes: &raw[index],
                                }
                            },
                        )
                    })
                    .collect();
                if pattern == 3 {
                    bindings.reverse();
                }
                let output = kernel.execute_mixed(&bindings).unwrap();
                let mut actual = vec![sentinel; expected.len() + 2];
                output.read_into(&mut actual).unwrap();
                assert_eq!(bytes(&actual[..expected.len()]), bytes(expected));
                assert_eq!(bytes(&actual[expected.len()..]), bytes(&[sentinel; 2]));
                retained = Some(output);
            }
            for index in 0..data.len() {
                let mut changed = data.clone();
                let node = source.graph().node(data[index].0).unwrap();
                let mut mutated = data[index].1.as_typed::<T>().unwrap().data().to_vec();
                *mutated.last_mut().unwrap() = invalid;
                changed[index].1 =
                    TensorValue::from_tensor(Tensor::new(node.shape, mutated).unwrap());
                let error = source.graph().evaluate_checked(&changed).unwrap_err();
                let expected_effect = match error {
                    TensorError::ArithmeticFault { value, .. }
                    | TensorError::CompoundArithmeticFault { value, .. } => value,
                    other => panic!("unexpected oracle refusal: {other:?}"),
                };
                let changed_raw: Vec<_> = changed
                    .iter()
                    .map(|(_, value)| bytes(value.as_typed::<T>().unwrap().data()))
                    .collect();
                let bindings: Vec<_> = changed
                    .iter()
                    .enumerate()
                    .map(|(index, (id, _))| {
                        (
                            *id,
                            MlxCheckedProgramInput::Host {
                                scalar: T::TYPE,
                                bytes: &changed_raw[index],
                            },
                        )
                    })
                    .collect();
                let failure = kernel.execute_mixed(&bindings).err().unwrap();
                assert_eq!(failure.effect, Some(expected_effect));
                assert!(matches!(failure.cause, MlxError::Arithmetic(_)));
            }
            let spec = &kernel.plan.inputs[0];
            let wrong = foreign
                .upload_encoded_bytes(spec.scalar, spec.count, &raw[0])
                .unwrap();
            let mut bindings: Vec<_> = data
                .iter()
                .enumerate()
                .map(|(index, (id, _))| {
                    (
                        *id,
                        MlxCheckedProgramInput::Host {
                            scalar: T::TYPE,
                            bytes: &raw[index],
                        },
                    )
                })
                .collect();
            bindings[0].1 = MlxCheckedProgramInput::Resident(&wrong);
            let failure = kernel.execute_mixed(&bindings).err().unwrap();
            assert!(matches!(failure.cause, MlxError::ForeignSession));
            assert_eq!(failure.effect, None);
            let mut actual = vec![sentinel; expected.len()];
            retained.as_ref().unwrap().read_into(&mut actual).unwrap();
            assert_eq!(bytes(&actual), bytes(expected));
            retained.unwrap().release().unwrap();
            for resident in residents {
                resident.release().unwrap();
            }
            wrong.release().unwrap();
        }
    }
}
#[test]
#[ignore = "Requires actual native resident multi-effect graph execution and terminal ownership."]
fn selected_graph_native_training_and_generic_branches() {
    let runtime = crate::MlxRuntime::load_default().unwrap();
    let session = runtime.open_gpu(0).unwrap();
    let foreign = runtime.open_gpu(0).unwrap();
    native(
        &session,
        &foreign,
        &[1.0_f32, 2.0, 3.0, 4.0],
        f32::NAN,
        -19.0,
        source,
    );
    native(
        &session,
        &foreign,
        &[1.0_f64, 2.0, 3.0, 4.0],
        f64::NAN,
        -19.0,
        source,
    );
}

fn phase_order<T: TensorElement + PcuCheckedFloat>(
    session: &MlxSession,
    request: PcuImplementationRequirements,
    values: &[T],
    invalid: T,
) {
    let mut graph = Graph::default();
    graph.set_numerical_options(request.numerical_options);
    let left = graph.input([2, 2], T::TYPE).unwrap();
    let right = graph.input([2, 2], T::TYPE).unwrap();
    let first = graph.relu(left).unwrap();
    let second = graph.relu(right).unwrap();
    for value in [first, second] {
        graph
            .set_value_float_underflow_policy(value, request.float_underflow)
            .unwrap();
    }
    let source = freeze(graph, left);
    let plan = MlxSelectedTensorGraphPlan::assess_program(source, request).unwrap();
    assert_eq!(plan.stage_count(), 2);
    let kernel = session.prepare_selected_tensor_graph(plan).unwrap();
    let mut a = values[..4].to_vec();
    a[3] = invalid;
    let mut b = values[..4].to_vec();
    b[0] = invalid;
    let a_raw = bytes(&a);
    let b_raw = bytes(&b);
    let valid_raw = bytes(&values[..4]);
    let bindings = [
        (
            left,
            MlxCheckedProgramInput::Host {
                scalar: T::TYPE,
                bytes: &a_raw,
            },
        ),
        (
            right,
            MlxCheckedProgramInput::Host {
                scalar: T::TYPE,
                bytes: &b_raw,
            },
        ),
    ];
    let failure = kernel.execute_mixed(&bindings).err().unwrap();
    assert_eq!(failure.effect, Some(first));
    assert!(matches!(failure.cause,MlxError::Arithmetic(fault) if fault.invocation_id == 3));
    let bindings = [
        (
            left,
            MlxCheckedProgramInput::Host {
                scalar: T::TYPE,
                bytes: &valid_raw,
            },
        ),
        (
            right,
            MlxCheckedProgramInput::Host {
                scalar: T::TYPE,
                bytes: &b_raw,
            },
        ),
    ];
    let failure = kernel.execute_mixed(&bindings).err().unwrap();
    assert_eq!(failure.effect, Some(second));
    assert!(matches!(failure.cause,MlxError::Arithmetic(fault) if fault.invocation_id == 0));
}

pub(super) fn low_source(
    profile: usize,
    scalar: PcuScalarType,
    request: PcuImplementationRequirements,
) -> Arc<TensorOwnedSelectedProgram> {
    let mut graph = Graph::default();
    graph.set_numerical_mode(request.numerical_mode);
    graph.set_numerical_options(request.numerical_options);
    let left = graph.input([2, 2], scalar).unwrap();
    let right = graph.input([2, 2], scalar).unwrap();
    let target = graph.input([2, 2], scalar).unwrap();
    let sum = graph.add(left, right).unwrap();
    let prediction = graph.relu(sum).unwrap();
    let difference = graph.sub(prediction, target).unwrap();
    let backward = graph.relu_backward(sum, difference).unwrap();
    for effect in [sum, prediction, difference, backward] {
        graph
            .set_value_float_underflow_policy(effect, request.float_underflow)
            .unwrap();
    }
    freeze(
        graph,
        match profile {
            0 => backward,
            1 => prediction,
            2 => left,
            _ => target,
        },
    )
}
#[test]
fn selected_graph_low_formats_freeze_exact_stages_roles_and_discarded_effects() {
    for request in requests() {
        for scalar in [
            PcuScalarType::F16,
            PcuScalarType::BF16,
            PcuScalarType::F8E4M3FN,
            PcuScalarType::F8E5M2,
        ] {
            for profile in 0..4 {
                let source = low_source(profile, scalar, request);
                let plan = MlxSelectedTensorGraphPlan::assess_program(Arc::clone(&source), request)
                    .unwrap();
                assert!(Arc::ptr_eq(&source, &plan.program_owner()));
                assert_eq!(plan.requirements(), request);
                assert_eq!(plan.stage_count(), 4);
                assert_eq!(plan.input_values().len(), 3);
                assert_eq!(plan.output(), source.output_values()[0]);
                for stage in &*plan.stages {
                    assert_eq!(
                        source.operation_index_of(stage.effect),
                        Some(stage.operation_index)
                    );
                }
            }
        }
    }
}
#[test]
#[ignore = "Requires actual low-format multistage Apple graph, exact original faults and terminal ownership."]
fn selected_graph_native_four_low_formats_pointwise_backward() {
    let runtime = crate::MlxRuntime::load_default().unwrap();
    let session = runtime.open_gpu(0).unwrap();
    let foreign = runtime.open_gpu(0).unwrap();
    macro_rules! low {
        ($ty:ty, $word:ty, $one:expr, $two:expr, $three:expr, $four:expr, $negative_two:expr, $nan:expr) => {
            native(
                &session,
                &foreign,
                &[$one, $two, $three, $four]
                    .map(|bits| <$ty>::from_bits(<$word>::try_from(bits).unwrap())),
                <$ty>::from_bits($nan),
                <$ty>::from_bits($negative_two),
                low_source,
            );
            low_stage_underflow(&session, <$ty>::from_bits(1));
        };
    }
    low!(
        fusion_pcu::PcuF16Bits,
        u16,
        0x3c00,
        0x4000,
        0x4200,
        0x4400,
        0xc000,
        0x7e01
    );
    low!(
        fusion_pcu::PcuBf16Bits,
        u16,
        0x3f80,
        0x4000,
        0x4040,
        0x4080,
        0xc000,
        0x7fc1
    );
    low!(
        fusion_pcu::PcuF8E4M3FnBits,
        u8,
        0x38,
        0x40,
        0x44,
        0x48,
        0xc0,
        0x7f
    );
    low!(
        fusion_pcu::PcuF8E5M2Bits,
        u8,
        0x3c,
        0x40,
        0x42,
        0x44,
        0xc0,
        0x7f
    );
}

fn low_stage_underflow<T: TensorElement + PcuCheckedFloat>(session: &MlxSession, tiny: T) {
    let zero = tiny.pcu_checked_sub(tiny).unwrap();
    for request in requests() {
        for profile in [0, 2] {
            let source = low_source(profile, T::TYPE, request);
            let plan =
                MlxSelectedTensorGraphPlan::assess_program(Arc::clone(&source), request).unwrap();
            let kernel = session.prepare_selected_tensor_graph(plan).unwrap();
            let mut left = [zero; 4];
            left[3] = tiny;
            let banks = [left, [zero; 4], [zero; 4]];
            let data: Vec<_> = source
                .input_values()
                .iter()
                .zip(&banks)
                .map(|(&id, bank)| {
                    (
                        id,
                        TensorValue::from_tensor(Tensor::new([2, 2], bank.to_vec()).unwrap()),
                    )
                })
                .collect();
            let raw: Vec<_> = banks.iter().map(|bank| bytes(bank)).collect();
            let bindings: Vec<_> = source
                .input_values()
                .iter()
                .zip(&raw)
                .map(|(&id, raw)| {
                    (
                        id,
                        MlxCheckedProgramInput::Host {
                            scalar: T::TYPE,
                            bytes: raw,
                        },
                    )
                })
                .collect();
            let actual = kernel.execute_mixed(&bindings);
            match source.graph().evaluate_checked(&data) {
                Ok(reference) => {
                    let expected = reference
                        .value_typed::<T>(source.output_values()[0])
                        .unwrap()
                        .data();
                    let owner = actual.unwrap();
                    let mut output = [zero; 6];
                    owner.read_into(&mut output).unwrap();
                    assert_eq!(bytes(&output[..4]), bytes(expected));
                    assert_eq!(bytes(&output[4..]), bytes(&[zero; 2]));
                    owner.release().unwrap();
                }
                Err(TensorError::ArithmeticFault {
                    value,
                    element_index,
                    kind,
                }) => {
                    let actual = actual.err().unwrap();
                    assert_eq!(actual.effect, Some(value));
                    assert!(
                        matches!(actual.cause, MlxError::Arithmetic(fault) if !fault.recovered && fault.kind == kind && fault.invocation_id == u64::try_from(element_index).unwrap())
                    );
                }
                Err(other) => panic!("Unexpected pointwise underflow oracle error: {other:?}"),
            }
        }
    }
}

#[path = "producer_tests.rs"]
mod producer_tests;

#[path = "policy_tests.rs"]
mod policy_tests;
