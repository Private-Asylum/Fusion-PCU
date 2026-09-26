//! Explicit `ROCm` selection and a two-layer PCU MLP forward graph.

use std::process::ExitCode;

#[path = "selection.rs"]
mod selection;

use fusion_pcu::{
    PcuMemoryAllocateWithPolicyError,
    PcuMemoryProviderFailure,
    PcuMemoryReservationLedger,
    PcuMemoryPoolId,
    PcuMemoryProvider,
    PcuOwnedDispatchMemorySession,
    PcuMemoryResourcePolicy,
};
use fusion_pcu_rocm::{
    RocmDiscovery,
    RocmMemoryResource,
    RocmPreparedTensorGraph,
    RocmTensorAssessor,
};
use fusion_pcu_tensor::{
    Graph,
    Tensor,
    TensorArithmeticRewritePolicy,
    TensorExecutionRoute,
    TensorGraphAssessment,
    TensorOperationSupport,
    TensorPointwiseGroupingPolicy,
    ValueId,
};

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("ERROR fusion-rocm-tensor: {error}");
            ExitCode::FAILURE
        }
    }
}

#[allow(clippy::too_many_lines)] // Keeps candidate selection and per-device failure reporting in one place.
fn run() -> Result<(), Box<dyn std::error::Error>> {
    let discovery = RocmDiscovery::new();
    let candidates = selection::ranked_devices(&discovery, selection::preferred_device()?, false)?;
    let mut graph = Graph::default();
    let samples = graph.input([2, 3])?;
    let hidden_weights = graph.input([3, 2])?;
    let bias = graph.input([2, 2])?;
    let output_weights = graph.input([2, 2])?;
    let hidden = graph.matmul(samples, hidden_weights)?;
    let biased = graph.add(hidden, bias)?;
    let activated = graph.relu(biased)?;
    let output = graph.matmul(activated, output_weights)?;
    let inputs = [
        (
            samples,
            Tensor::new([2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])?,
        ),
        (
            hidden_weights,
            Tensor::new([3, 2], vec![1.0, -2.0, 3.0, -4.0, 5.0, -6.0])?,
        ),
        (bias, Tensor::new([2, 2], vec![0.0, -1.0, 0.0, 1.0])?),
        (
            output_weights,
            Tensor::new([2, 2], vec![1.0, 2.0, -1.0, 3.0])?,
        ),
    ];
    let expected = graph.evaluate(&inputs)?.value(output)?.clone();
    let mut failures = Vec::new();
    for candidate in candidates {
        let session =
            match fusion_pcu_rocm::RocmOwnedDispatchBackend::open(&discovery, candidate.device, 64)
            {
                Ok(session) => session,
                Err(error) => {
                    failures.push(format!("device {} open: {error}", candidate.device.id));
                    continue;
                }
            };
        let assessor = match RocmTensorAssessor::new(&session) {
            Ok(assessor) => assessor,
            Err(error) => {
                failures.push(format!(
                    "device {} tensor assessor: {error}",
                    candidate.device.id
                ));
                continue;
            }
        };
        let assessment = graph.assess_with(&assessor);
        if !supports_mixed_routes(&assessment, hidden, biased, activated, output) {
            failures.push(format!(
                "device {} cannot run the rocBLAS and synthesized graph: {assessment:?}",
                candidate.device.id
            ));
            continue;
        }
        let mut memory = PcuOwnedDispatchMemorySession::memory_provider(&session, candidate.pool);
        match assessor.execute_graph(&graph, &inputs, output, candidate.pool, &mut memory) {
            Ok(actual) if actual == expected => {
                let prepared = assessor.prepare_graph(&graph, output)?;
                let batched = assessor.execute_prepared_batched(
                    &prepared,
                    &inputs,
                    candidate.pool,
                    &mut memory,
                )?;
                if batched != expected {
                    return Err(format!(
                        "batched tensor output {batched:?}, expected {expected:?}"
                    )
                    .into());
                }
                verify_fused_forward(
                    &assessor,
                    &graph,
                    &inputs,
                    biased,
                    output,
                    &expected,
                    candidate.pool,
                    &mut memory,
                )?;
                verify_secondary_paths(&assessor, candidate.pool, &mut memory)?;
                println!(
                    "PCU two-layer MLP forward graph passed on {}",
                    candidate.name
                );
                return Ok(());
            }
            Ok(actual) => failures.push(format!(
                "device {} output {actual:?}, expected {expected:?}",
                candidate.device.id
            )),
            Err(error) => failures.push(format!(
                "device {} graph execution: {error}",
                candidate.device.id
            )),
        }
    }
    Err(format!(
        "no ROCm device completed the graph: {}",
        failures.join("; ")
    )
    .into())
}

#[allow(clippy::too_many_arguments)] // A graph, selected values, and the same memory route as the caller.
fn verify_fused_forward(
    assessor: &RocmTensorAssessor<'_>,
    graph: &Graph,
    inputs: &[(ValueId, Tensor)],
    biased: ValueId,
    output: ValueId,
    expected: &Tensor,
    pool: PcuMemoryPoolId,
    memory: &mut impl PcuMemoryProvider<Resource = RocmMemoryResource>,
) -> Result<(), Box<dyn std::error::Error>> {
    let fused = assessor.prepare_graph_outputs_with_policies(
        graph,
        &[output],
        TensorArithmeticRewritePolicy::Disabled,
        TensorPointwiseGroupingPolicy::SingleUseAddRelu,
    )?;
    if fused.lowering_plan().operation_index_of(biased).is_some()
        || fused.lowering_plan().pointwise_fusion_groups().len() != 1
    {
        return Err("Add->ReLU grouping did not suppress its intermediate".into());
    }
    let actual = assessor.execute_prepared(&fused, inputs, pool, memory)?;
    if &actual == expected {
        Ok(())
    } else {
        Err(format!("fused tensor output {actual:?}, expected {expected:?}").into())
    }
}

fn verify_secondary_paths(
    assessor: &RocmTensorAssessor<'_>,
    pool: PcuMemoryPoolId,
    memory: &mut impl PcuMemoryProvider<Resource = RocmMemoryResource>,
) -> Result<(), Box<dyn std::error::Error>> {
    verify_relu_edges(assessor, pool, memory)?;
    verify_fused_add_relu_edges(assessor, pool, memory)?;
    verify_bounded_add_sub_relu_edges(assessor, pool, memory)?;
    verify_bounded_uniform_right_sub_edges(assessor, pool, memory)?;
    verify_bounded_identity_edges(assessor, pool, memory)?;
    verify_bounded_mul_edges(assessor, pool, memory)?;
    verify_relu_backward(assessor, pool, memory)?;
    verify_relu_autograd(assessor, pool, memory)?;
    verify_mse(assessor, pool, memory)?;
    verify_sgd_update(assessor, pool, memory)?;
    verify_backward_algebra(assessor, pool, memory)?;
    verify_uniform_storage(assessor, pool, memory)?;
    verify_input_output_bank(assessor, pool, memory)?;
    verify_policy_feedback(assessor, pool, memory)
}

fn verify_uniform_storage(
    assessor: &RocmTensorAssessor<'_>,
    pool: PcuMemoryPoolId,
    memory: &mut impl PcuMemoryProvider<Resource = RocmMemoryResource>,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut graph = Graph::default();
    let input = graph.input([8])?;
    let uniform = graph.uniform([8], 0.5)?;
    let add_left = graph.add(uniform, input)?;
    let subtract_right = graph.sub(add_left, uniform)?;
    let add_both_uniform = graph.add(uniform, uniform)?;
    let output = graph.mul(subtract_right, add_both_uniform)?;
    let inputs = [(
        input,
        Tensor::new([8], vec![1.0, -2.0, 3.5, 0.0, 8.0, -1.0, 4.0, 2.0])?,
    )];
    let evaluated = graph.evaluate(&inputs)?;
    let expected_subtract = evaluated.value(subtract_right)?.clone();
    let expected_output = evaluated.value(output)?.clone();
    let selected_outputs = [subtract_right, output];
    let prepared = assessor.prepare_graph_outputs(&graph, &selected_outputs)?;
    let device_input = assessor.upload_input(&inputs[0].1, pool, memory)?;
    let mut scratch = assessor.prepare_scratch(&prepared, pool, memory)?;
    let mut output_bank = assessor.prepare_output_bank(&prepared, pool, memory)?;
    assessor.execute_prepared_outputs_into_bank(
        &prepared,
        &[(input, &device_input)],
        &mut scratch,
        &mut output_bank,
        memory,
    )?;
    let actual_subtract = assessor.download_output(&output_bank.outputs()[0], pool, memory)?;
    let actual_output = assessor.download_output(&output_bank.outputs()[1], pool, memory)?;
    if actual_subtract != expected_subtract || actual_output != expected_output {
        return Err(format!(
            "compact uniform fan-out outputs were {actual_subtract:?} and {actual_output:?}, expected {expected_subtract:?} and {expected_output:?}"
        )
        .into());
    }

    // An exported Uniform value has to expose its full logical dense shape even though internal
    // operands are allowed to occupy one scalar element on the device.
    let dense_expected = Tensor::splat([8], 0.5)?;
    let dense_actual = assessor.execute_graph(&graph, &[], uniform, pool, memory)?;
    if dense_actual != dense_expected {
        return Err(format!(
            "selected Uniform output {dense_actual:?}, expected {dense_expected:?}"
        )
        .into());
    }
    verify_uniform_relu_dense_fallback(assessor, pool, memory)?;
    verify_uniform_matmul_dense_fallback(assessor, pool, memory)?;
    Ok(())
}

fn verify_uniform_relu_dense_fallback(
    assessor: &RocmTensorAssessor<'_>,
    pool: PcuMemoryPoolId,
    memory: &mut impl PcuMemoryProvider<Resource = RocmMemoryResource>,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut graph = Graph::default();
    let uniform = graph.uniform([2, 2], -0.5)?;
    let output = graph.relu(uniform)?;
    let expected = graph.evaluate(&[])?.value(output)?.clone();
    let actual = assessor.execute_graph(&graph, &[], output, pool, memory)?;
    if actual != expected {
        return Err(format!("Uniform→ReLU produced {actual:?}, expected {expected:?}").into());
    }
    Ok(())
}

fn verify_uniform_matmul_dense_fallback(
    assessor: &RocmTensorAssessor<'_>,
    pool: PcuMemoryPoolId,
    memory: &mut impl PcuMemoryProvider<Resource = RocmMemoryResource>,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut graph = Graph::default();
    let uniform = graph.uniform([1, 2], 0.5)?;
    let input = graph.input([2, 1])?;
    let output = graph.matmul(uniform, input)?;
    let inputs = [(input, Tensor::new([2, 1], vec![2.0, 3.0])?)];
    let expected = graph.evaluate(&inputs)?.value(output)?.clone();
    let actual = assessor.execute_graph(&graph, &inputs, output, pool, memory)?;
    if actual != expected {
        return Err(format!("Uniform→MatMul produced {actual:?}, expected {expected:?}").into());
    }
    Ok(())
}

fn verify_policy_feedback(
    assessor: &RocmTensorAssessor<'_>,
    pool: PcuMemoryPoolId,
    memory: &mut impl PcuMemoryProvider<Resource = RocmMemoryResource>,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut graph = Graph::default();
    let state = graph.input([4])?;
    let next = graph.relu(state)?;
    let prepared = assessor.prepare_graph_outputs(&graph, &[next])?;
    let feedback = prepared.tensor_plan().feedback_plan(&[(next, state)])?;
    let initial = Tensor::new([4], vec![1.0, -2.0, 3.0, -4.0])?;
    let device_initial = assessor.upload_input(&initial, pool, memory)?;
    let mut ledger = PcuMemoryReservationLedger::<8>::new();
    verify_rejected_device_local_admission(assessor, &prepared, pool, memory, &mut ledger)?;
    // Exercise implicit RAII reconciliation as well as explicit release.
    {
        let dropped = assessor.prepare_feedback_resources_with_policy_guard(
            &prepared,
            pool,
            memory,
            &mut ledger,
            PcuMemoryResourcePolicy {
                require_device_local: false,
                admission: fusion_pcu::PcuMemoryAdmissionPolicy::default(),
            },
        )?;
        drop(dropped);
    }
    if ledger.reserved_bytes(pool) != Some(0) {
        return Err("dropping feedback resources did not release ledger reservations".into());
    }

    let mut resources = assessor.prepare_feedback_resources_with_policy_guard(
        &prepared,
        pool,
        memory,
        &mut ledger,
        PcuMemoryResourcePolicy {
            require_device_local: false,
            admission: fusion_pcu::PcuMemoryAdmissionPolicy::default(),
        },
    )?;
    assessor.execute_feedback_steps(
        &feedback,
        &[(state, &device_initial)],
        &mut resources,
        std::num::NonZeroUsize::new(3).expect("three feedback steps"),
        memory,
    )?;
    let actual = assessor.download_output(&resources.outputs_for_step(2)?[0], pool, memory)?;
    resources.release()?;

    let mut batched_resources = assessor.prepare_feedback_resources_with_policy_guard(
        &prepared,
        pool,
        memory,
        &mut ledger,
        PcuMemoryResourcePolicy {
            require_device_local: false,
            admission: fusion_pcu::PcuMemoryAdmissionPolicy::default(),
        },
    )?;
    assessor.execute_feedback_steps_batched(
        &feedback,
        &[(state, &device_initial)],
        &mut batched_resources,
        std::num::NonZeroUsize::new(3).expect("three feedback steps"),
        memory,
    )?;
    let actual_batched =
        assessor.download_output(&batched_resources.outputs_for_step(2)?[0], pool, memory)?;
    batched_resources.release()?;
    let expected = Tensor::new([4], vec![1.0, 0.0, 3.0, 0.0])?;
    if actual == expected && actual_batched == expected {
        Ok(())
    } else {
        Err(format!(
            "policy feedback outputs sync={actual:?}, batched={actual_batched:?}, expected {expected:?}"
        )
        .into())
    }
}

fn verify_rejected_device_local_admission(
    assessor: &RocmTensorAssessor<'_>,
    prepared: &RocmPreparedTensorGraph<'_>,
    pool: PcuMemoryPoolId,
    memory: &mut impl PcuMemoryProvider<Resource = RocmMemoryResource>,
    ledger: &mut PcuMemoryReservationLedger<8>,
) -> Result<(), Box<dyn std::error::Error>> {
    let rejected_as_required = {
        let rejected = assessor.prepare_feedback_resources_with_policy_guard(
            prepared,
            pool,
            memory,
            ledger,
            PcuMemoryResourcePolicy {
                require_device_local: true,
                admission: fusion_pcu::PcuMemoryAdmissionPolicy::default(),
            },
        );
        matches!(
            &rejected,
            Err(fusion_pcu_rocm::RocmTensorFeedbackPrepareError::Execution(
                fusion_pcu_rocm::RocmTensorExecutionError::MemoryAdmission(
                    PcuMemoryAllocateWithPolicyError::Provider(error)
                )
            )) if error.failure == PcuMemoryProviderFailure::DeviceLocalRequired
        )
    };
    if !rejected_as_required {
        return Err("device-local policy was not rejected with DeviceLocalRequired".into());
    }
    if ledger.reserved_bytes(pool) != Some(0) {
        return Err(format!(
            "failed device-local admission leaked {} reserved bytes",
            ledger.reserved_bytes(pool).unwrap_or_default()
        )
        .into());
    }
    Ok(())
}

fn verify_input_output_bank(
    assessor: &RocmTensorAssessor<'_>,
    pool: PcuMemoryPoolId,
    memory: &mut impl PcuMemoryProvider<Resource = RocmMemoryResource>,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut graph = Graph::default();
    let input = graph.input([4])?;
    let expected = Tensor::new([4], vec![1.0, -2.0, 3.0, -4.0])?;
    let prepared = assessor.prepare_graph(&graph, input)?;
    let source = assessor.upload_input(&expected, pool, memory)?;
    let mut scratch = assessor.prepare_scratch(&prepared, pool, memory)?;
    let mut bank = assessor.prepare_output_bank(&prepared, pool, memory)?;
    assessor.execute_prepared_outputs_into_bank(
        &prepared,
        &[(input, &source)],
        &mut scratch,
        &mut bank,
        memory,
    )?;
    let actual = assessor.download_output(&bank.outputs()[0], pool, memory)?;
    if actual == expected {
        Ok(())
    } else {
        Err(format!("ROCm output-bank copy {actual:?}, expected {expected:?}").into())
    }
}

fn verify_backward_algebra(
    assessor: &RocmTensorAssessor<'_>,
    pool: PcuMemoryPoolId,
    memory: &mut impl PcuMemoryProvider<Resource = RocmMemoryResource>,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut graph = Graph::default();
    let left = graph.input([2, 3])?;
    let right = graph.input([2, 3])?;
    let difference = graph.sub(left, right)?;
    let squared = graph.mul(difference, difference)?;
    let inputs = [
        (
            left,
            Tensor::new([2, 3], vec![1.0, 2.0, -3.0, 4.0, -5.0, 6.0])?,
        ),
        (
            right,
            Tensor::new([2, 3], vec![0.5, -2.0, 1.0, 3.0, 2.0, -1.0])?,
        ),
    ];
    verify_against_reference(assessor, pool, memory, &graph, &inputs, squared)?;

    for (transpose_left, transpose_right) in
        [(false, false), (true, false), (false, true), (true, true)]
    {
        let left_shape = if transpose_left { [3, 2] } else { [2, 3] };
        let right_shape = if transpose_right { [4, 3] } else { [3, 4] };
        let mut graph = Graph::default();
        let left = graph.input(left_shape)?;
        let right = graph.input(right_shape)?;
        let output = graph.matmul_transposed(left, right, transpose_left, transpose_right)?;
        let inputs = [
            (
                left,
                Tensor::new(left_shape, (1_u8..=6).map(f32::from).collect())?,
            ),
            (
                right,
                Tensor::new(
                    right_shape,
                    (1_u8..=12).map(|value| f32::from(value) * 0.25).collect(),
                )?,
            ),
        ];
        verify_against_reference(assessor, pool, memory, &graph, &inputs, output)?;
    }
    Ok(())
}

fn verify_against_reference(
    assessor: &RocmTensorAssessor<'_>,
    pool: PcuMemoryPoolId,
    memory: &mut impl PcuMemoryProvider<Resource = RocmMemoryResource>,
    graph: &Graph,
    inputs: &[(ValueId, Tensor)],
    output: ValueId,
) -> Result<(), Box<dyn std::error::Error>> {
    let expected = graph.evaluate(inputs)?.value(output)?.clone();
    let actual = assessor.execute_graph(graph, inputs, output, pool, memory)?;
    if actual.shape() == expected.shape()
        && actual.data().len() == expected.data().len()
        && actual
            .data()
            .iter()
            .zip(expected.data())
            .all(|(a, b)| (a - b).abs() <= 1.0e-4)
    {
        Ok(())
    } else {
        Err(format!("ROCm backward-algebra output {actual:?}, expected {expected:?}").into())
    }
}

fn verify_mse(
    assessor: &RocmTensorAssessor<'_>,
    pool: PcuMemoryPoolId,
    memory: &mut impl PcuMemoryProvider<Resource = RocmMemoryResource>,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut graph = Graph::default();
    let prediction = graph.input([2, 3])?;
    let target = graph.input([2, 3])?;
    let loss = graph.mean_squared_error(prediction, target)?;
    let inputs = [
        (
            prediction,
            Tensor::new([2, 3], vec![1.0, -2.0, 4.0, 0.5, 3.0, -1.0])?,
        ),
        (
            target,
            Tensor::new([2, 3], vec![0.0, 2.0, 1.0, -0.5, 1.0, 3.0])?,
        ),
    ];
    let expected = graph.evaluate(&inputs)?.value(loss)?.clone();
    let actual = assessor.execute_graph(&graph, &inputs, loss, pool, memory)?;
    if (actual.data()[0] - expected.data()[0]).abs() <= 1.0e-5 {
        verify_mse_special_values(assessor, pool, memory, &graph, prediction, target, loss)
    } else {
        Err(format!("ROCm MSE output {actual:?}, expected {expected:?}").into())
    }
}

fn verify_mse_special_values(
    assessor: &RocmTensorAssessor<'_>,
    pool: PcuMemoryPoolId,
    memory: &mut impl PcuMemoryProvider<Resource = RocmMemoryResource>,
    graph: &Graph,
    prediction: ValueId,
    target: ValueId,
    loss: ValueId,
) -> Result<(), Box<dyn std::error::Error>> {
    let fixtures = [
        (
            "NaN propagation",
            vec![f32::NAN, 0.0, 1.0, 2.0, 3.0, 4.0],
            vec![0.0; 6],
            0_u8,
        ),
        (
            "infinity propagation",
            vec![f32::INFINITY, 0.0, f32::NEG_INFINITY, 2.0, 3.0, 4.0],
            vec![0.0, f32::INFINITY, 0.0, 2.0, 3.0, 4.0],
            1_u8,
        ),
        (
            "signed-zero reduction",
            vec![-0.0, 0.0, -0.0, 0.0, -0.0, 0.0],
            vec![0.0, -0.0, 0.0, -0.0, 0.0, -0.0],
            2_u8,
        ),
    ];
    for (name, prediction_values, target_values, expected_kind) in fixtures {
        let inputs = [
            (prediction, Tensor::new([2, 3], prediction_values)?),
            (target, Tensor::new([2, 3], target_values)?),
        ];
        let expected = graph.evaluate(&inputs)?.value(loss)?.data()[0];
        let actual = assessor
            .execute_graph(graph, &inputs, loss, pool, memory)?
            .data()[0];
        let matches_contract = match expected_kind {
            0 => expected.is_nan() && actual.is_nan(),
            1 => expected.is_infinite() && actual.is_infinite(),
            2 => expected.to_bits() == 0 && actual.to_bits() == 0,
            _ => unreachable!(),
        };
        if !matches_contract {
            return Err(format!(
                "ROCm MSE {name} contract failed: expected {expected:?}, got {actual:?}"
            )
            .into());
        }
    }
    Ok(())
}

fn verify_sgd_update(
    assessor: &RocmTensorAssessor<'_>,
    pool: PcuMemoryPoolId,
    memory: &mut impl PcuMemoryProvider<Resource = RocmMemoryResource>,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut graph = Graph::default();
    let weights = graph.input([1])?;
    let gradient = graph.input([1])?;
    let rate = f32::from_bits(1.0_f32.to_bits() + 1);
    let update = graph.sgd_update(weights, gradient, rate)?;
    let inputs = [
        (weights, Tensor::new([1], vec![1.0])?),
        (
            gradient,
            Tensor::new([1], vec![f32::from_bits(1.0_f32.to_bits() - 2)])?,
        ),
    ];
    let actual = assessor.execute_graph(&graph, &inputs, update, pool, memory)?;
    let actual_bits = actual.data()[0].to_bits();
    if actual_bits != 0x0000_0000 {
        return Err(format!(
            "strict ROCm SgdUpdate returned {actual_bits:#010x}; expected separately rounded +0.0 (0x00000000)"
        )
        .into());
    }
    println!("ROCm strict SgdUpdate contraction witness passed (0x00000000)");

    // Exercise the opt-in graph rewrite separately: it is permitted to contract Mul/Sub only
    // because this preparation names both the policy and the backend arithmetic capability.
    let mut graph = Graph::default();
    let weights = graph.input([1])?;
    let gradient = graph.input([1])?;
    let rate_value = f32::from_bits(1.0_f32.to_bits() + 1);
    let rate = graph.constant(Tensor::new([1], vec![rate_value])?);
    let scaled = graph.mul(rate, gradient)?;
    let updated = graph.sub(weights, scaled)?;
    let inputs = [
        (weights, Tensor::new([1], vec![1.0])?),
        (
            gradient,
            Tensor::new([1], vec![f32::from_bits(1.0_f32.to_bits() - 2)])?,
        ),
    ];
    let strict = assessor.prepare_graph_outputs(&graph, &[updated])?;
    if !strict.lowering_plan().rewrites().is_empty() {
        return Err(
            "default ROCm tensor preparation unexpectedly selected contracted arithmetic".into(),
        );
    }
    let strict_actual = assessor.execute_prepared(&strict, &inputs, pool, memory)?;
    let strict_bits = strict_actual.data()[0].to_bits();
    if strict_bits != 0x0000_0000 {
        return Err(format!("strict Mul/Sub returned {strict_bits:#010x}; expected +0.0").into());
    }
    let contracted = assessor.prepare_graph_outputs_with_policy(
        &graph,
        &[updated],
        TensorArithmeticRewritePolicy::AllowContractedArithmetic,
    )?;
    if contracted.lowering_plan().rewrites().len() != 1 {
        return Err(
            "explicit contracted ROCm preparation did not select the eligible SGD rewrite".into(),
        );
    }
    let contracted_actual = assessor.execute_prepared(&contracted, &inputs, pool, memory)?;
    let contracted_bits = contracted_actual.data()[0].to_bits();
    if contracted_bits != 0x2880_0000 {
        return Err(format!(
            "contracted Mul/Sub returned {contracted_bits:#010x}; expected +2^-46 (0x28800000)"
        )
        .into());
    }
    println!("ROCm explicit contracted SGD witness passed (0x{contracted_bits:08x})");
    Ok(())
}

fn supports_mixed_routes(
    assessment: &TensorGraphAssessment<'_>,
    hidden: ValueId,
    biased: ValueId,
    activated: ValueId,
    output: ValueId,
) -> bool {
    if !assessment.is_supported() {
        return false;
    }
    [
        (hidden, TensorExecutionRoute::Library),
        (biased, TensorExecutionRoute::Synthesized),
        (activated, TensorExecutionRoute::Synthesized),
        (output, TensorExecutionRoute::Library),
    ]
    .into_iter()
    .all(|(value, route)| {
        assessment
            .nodes
            .iter()
            .find(|node| node.node.value == value)
            .is_some_and(|node| {
                matches!(
                    node.support,
                    TensorOperationSupport::Supported { route: actual, .. } if actual == route
                )
            })
    })
}

fn verify_relu_edges(
    assessor: &RocmTensorAssessor<'_>,
    pool: PcuMemoryPoolId,
    memory: &mut impl PcuMemoryProvider<Resource = RocmMemoryResource>,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut graph = Graph::default();
    let input = graph.input([6])?;
    let output = graph.relu(input)?;
    let inputs = [(
        input,
        Tensor::new([6], vec![f32::NAN, -0.0, 0.0, -1.0, 1.0, f32::INFINITY])?,
    )];
    let actual = assessor.execute_graph(&graph, &inputs, output, pool, memory)?;
    let expected = [0.0_f32, 0.0, 0.0, 0.0, 1.0, f32::INFINITY];
    if actual
        .data()
        .iter()
        .zip(expected)
        .all(|(value, expected)| value.to_bits() == expected.to_bits())
    {
        Ok(())
    } else {
        Err(format!("ROCm ReLU edge output {actual:?}").into())
    }
}

fn verify_fused_add_relu_edges(
    assessor: &RocmTensorAssessor<'_>,
    pool: PcuMemoryPoolId,
    memory: &mut impl PcuMemoryProvider<Resource = RocmMemoryResource>,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut graph = Graph::default();
    let left = graph.input([6])?;
    let right = graph.input([6])?;
    let added = graph.add(left, right)?;
    let output = graph.relu(added)?;
    let inputs = [
        (
            left,
            Tensor::new(
                [6],
                vec![f32::NAN, -0.0, f32::INFINITY, f32::MAX, -2.0, 0.25],
            )?,
        ),
        (
            right,
            Tensor::new([6], vec![1.0, -0.0, f32::NEG_INFINITY, f32::MAX, 1.0, 0.5])?,
        ),
    ];
    let expected = graph.evaluate(&inputs)?.value(output)?.clone();
    let fused = assessor.prepare_graph_outputs_with_policies(
        &graph,
        &[output],
        TensorArithmeticRewritePolicy::Disabled,
        TensorPointwiseGroupingPolicy::SingleUseAddRelu,
    )?;
    let actual = assessor.execute_prepared(&fused, &inputs, pool, memory)?;
    if actual
        .data()
        .iter()
        .map(|value| value.to_bits())
        .eq(expected.data().iter().map(|value| value.to_bits()))
    {
        Ok(())
    } else {
        Err(format!("fused Add->ReLU edge output {actual:?}, expected {expected:?}").into())
    }
}

fn verify_bounded_add_sub_relu_edges(
    assessor: &RocmTensorAssessor<'_>,
    pool: PcuMemoryPoolId,
    memory: &mut impl PcuMemoryProvider<Resource = RocmMemoryResource>,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut graph = Graph::default();
    let a = graph.input([8])?;
    let b = graph.input([8])?;
    let c = graph.input([8])?;
    let first = graph.add(a, b)?;
    let second = graph.sub(first, c)?;
    let third = graph.add(second, a)?;
    let output = graph.relu(third)?;
    let inputs = [
        (
            a,
            Tensor::new(
                [8],
                vec![
                    f32::NAN,
                    -0.0,
                    f32::INFINITY,
                    f32::MAX,
                    -2.0,
                    0.25,
                    f32::MIN_POSITIVE,
                    1.0,
                ],
            )?,
        ),
        (
            b,
            Tensor::new(
                [8],
                vec![
                    1.0,
                    -0.0,
                    f32::NEG_INFINITY,
                    f32::MAX,
                    1.0,
                    0.5,
                    -f32::MIN_POSITIVE,
                    -1.0,
                ],
            )?,
        ),
        (
            c,
            Tensor::new([8], vec![0.0, 0.0, 0.0, f32::MAX, -1.0, 0.25, 0.0, -1.0])?,
        ),
    ];
    let expected = graph.evaluate(&inputs)?.value(output)?.clone();
    let grouped = assessor.prepare_graph_outputs_with_policies(
        &graph,
        &[output],
        TensorArithmeticRewritePolicy::Disabled,
        TensorPointwiseGroupingPolicy::BoundedAddSubRelu,
    )?;
    if [first, second, third]
        .into_iter()
        .any(|value| grouped.lowering_plan().operation_index_of(value).is_some())
    {
        return Err("bounded Add/Sub->ReLU group retained an intermediate".into());
    }
    let actual = assessor.execute_prepared(&grouped, &inputs, pool, memory)?;
    if !actual
        .data()
        .iter()
        .map(|value| value.to_bits())
        .eq(expected.data().iter().map(|value| value.to_bits()))
    {
        return Err(
            format!("bounded Add/Sub->ReLU output {actual:?}, expected {expected:?}").into(),
        );
    }
    Ok(())
}

// Exercise a compact Uniform leaf and a chain result on the right of Sub.
fn verify_bounded_uniform_right_sub_edges(
    assessor: &RocmTensorAssessor<'_>,
    pool: PcuMemoryPoolId,
    memory: &mut impl PcuMemoryProvider<Resource = RocmMemoryResource>,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut graph = Graph::default();
    let left = graph.input([8])?;
    let right = graph.input([8])?;
    let uniform = graph.uniform([8], 0.25)?;
    let first = graph.add(left, uniform)?;
    let second = graph.sub(right, first)?;
    let third = graph.add(second, uniform)?;
    let output = graph.relu(third)?;
    let inputs = [
        (
            left,
            Tensor::new(
                [8],
                vec![1.0, -1.0, 0.0, -0.0, 4.0, f32::MAX, f32::INFINITY, f32::NAN],
            )?,
        ),
        (
            right,
            Tensor::new(
                [8],
                vec![3.0, -3.0, -0.0, 0.0, -4.0, f32::MAX, f32::NEG_INFINITY, 1.0],
            )?,
        ),
    ];
    let expected = graph.evaluate(&inputs)?.value(output)?.clone();
    let grouped = assessor.prepare_graph_outputs_with_policies(
        &graph,
        &[output],
        TensorArithmeticRewritePolicy::Disabled,
        TensorPointwiseGroupingPolicy::BoundedAddSubRelu,
    )?;
    if [first, second, third]
        .into_iter()
        .any(|value| grouped.lowering_plan().operation_index_of(value).is_some())
    {
        return Err("bounded right-hand Sub group retained an intermediate".into());
    }
    let actual = assessor.execute_prepared(&grouped, &inputs, pool, memory)?;
    if actual
        .data()
        .iter()
        .map(|value| value.to_bits())
        .eq(expected.data().iter().map(|value| value.to_bits()))
    {
        Ok(())
    } else {
        Err(format!("bounded right-hand Sub output {actual:?}, expected {expected:?}").into())
    }
}

fn verify_bounded_identity_edges(
    assessor: &RocmTensorAssessor<'_>,
    pool: PcuMemoryPoolId,
    memory: &mut impl PcuMemoryProvider<Resource = RocmMemoryResource>,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut graph = Graph::default();
    let left = graph.input([8])?;
    let right = graph.input([8])?;
    let uniform = graph.uniform([8], 0.25)?;
    let first = graph.add(left, uniform)?;
    let second = graph.sub(right, first)?;
    let output = graph.add(second, uniform)?;
    let inputs = [
        (
            left,
            Tensor::new(
                [8],
                vec![1.0, -1.0, 0.0, -0.0, 4.0, f32::MAX, f32::INFINITY, f32::NAN],
            )?,
        ),
        (
            right,
            Tensor::new(
                [8],
                vec![3.0, -3.0, -0.0, 0.0, -4.0, f32::MAX, f32::NEG_INFINITY, 1.0],
            )?,
        ),
    ];
    let expected = graph.evaluate(&inputs)?.value(output)?.clone();
    let grouped = assessor.prepare_graph_outputs_with_policies(
        &graph,
        &[output],
        TensorArithmeticRewritePolicy::Disabled,
        TensorPointwiseGroupingPolicy::BoundedAddSubIdentity,
    )?;
    if grouped.lowering_plan().operation_index_of(first).is_some()
        || grouped.lowering_plan().operation_index_of(second).is_some()
        || grouped.lowering_plan().operation_index_of(output).is_none()
    {
        return Err("bounded Identity group did not preserve its terminal output".into());
    }
    let first_warm = assessor.prewarm_prepared_graph(&grouped)?;
    let second_warm = assessor.prewarm_prepared_graph(&grouped)?;
    if first_warm.retained_keys != first_warm.requested_keys || second_warm.compiled_keys != 0 {
        return Err("bounded Identity executable prewarm was not retained".into());
    }
    let actual = assessor.execute_prepared(&grouped, &inputs, pool, memory)?;
    if actual
        .data()
        .iter()
        .zip(expected.data())
        .all(|(actual, expected)| {
            (actual.is_nan() && expected.is_nan()) || actual.to_bits() == expected.to_bits()
        })
    {
        Ok(())
    } else {
        Err(format!(
            "bounded Identity bits actual={:?}, expected={:?}",
            actual
                .data()
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            expected
                .data()
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        )
        .into())
    }
}

fn verify_bounded_mul_edges(
    assessor: &RocmTensorAssessor<'_>,
    pool: PcuMemoryPoolId,
    memory: &mut impl PcuMemoryProvider<Resource = RocmMemoryResource>,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut graph = Graph::default();
    let left = graph.input([8])?;
    let right = graph.input([8])?;
    let uniform = graph.uniform([8], 0.25)?;
    let first = graph.mul(left, uniform)?;
    let second = graph.mul(right, first)?;
    let output = graph.mul(second, uniform)?;
    let inputs = [
        (
            left,
            Tensor::new(
                [8],
                vec![1.0, -1.0, 0.0, -0.0, 4.0, f32::MAX, f32::INFINITY, f32::NAN],
            )?,
        ),
        (
            right,
            Tensor::new(
                [8],
                vec![3.0, -3.0, -0.0, 0.0, -4.0, f32::MAX, f32::NEG_INFINITY, 1.0],
            )?,
        ),
    ];
    let expected = graph.evaluate(&inputs)?.value(output)?.clone();
    let grouped = assessor.prepare_graph_outputs_with_policies(
        &graph,
        &[output],
        TensorArithmeticRewritePolicy::Disabled,
        TensorPointwiseGroupingPolicy::BoundedMulIdentity,
    )?;
    if grouped.lowering_plan().operation_index_of(first).is_some()
        || grouped.lowering_plan().operation_index_of(second).is_some()
        || grouped.lowering_plan().operation_index_of(output).is_none()
    {
        return Err("bounded Mul group did not preserve its terminal output".into());
    }
    let first_warm = assessor.prewarm_prepared_graph(&grouped)?;
    let second_warm = assessor.prewarm_prepared_graph(&grouped)?;
    if first_warm.retained_keys != first_warm.requested_keys || second_warm.compiled_keys != 0 {
        return Err("bounded Mul executable prewarm was not retained".into());
    }
    let actual = assessor.execute_prepared(&grouped, &inputs, pool, memory)?;
    if actual
        .data()
        .iter()
        .zip(expected.data())
        .all(|(actual, expected)| {
            (actual.is_nan() && expected.is_nan()) || actual.to_bits() == expected.to_bits()
        })
    {
        Ok(())
    } else {
        Err(format!(
            "bounded Mul bits actual={:?}, expected={:?}",
            actual
                .data()
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            expected
                .data()
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        )
        .into())
    }
}

fn verify_relu_backward(
    assessor: &RocmTensorAssessor<'_>,
    pool: PcuMemoryPoolId,
    memory: &mut impl PcuMemoryProvider<Resource = RocmMemoryResource>,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut graph = Graph::default();
    let input = graph.input([6])?;
    let upstream = graph.input([6])?;
    let output = graph.relu_backward(input, upstream)?;
    let inputs = [
        (
            input,
            Tensor::new([6], vec![f32::NAN, -0.0, 0.0, -1.0, 1.0, f32::INFINITY])?,
        ),
        (
            upstream,
            Tensor::new([6], vec![2.0, 3.0, 4.0, 5.0, -6.0, -7.0])?,
        ),
    ];
    let expected = graph.evaluate(&inputs)?.value(output)?.clone();
    let actual = assessor.execute_graph(&graph, &inputs, output, pool, memory)?;
    if actual
        .data()
        .iter()
        .map(|v| v.to_bits())
        .eq(expected.data().iter().map(|v| v.to_bits()))
    {
        Ok(())
    } else {
        Err(format!("ROCm ReLU backward output {actual:?}, expected {expected:?}").into())
    }
}

fn verify_relu_autograd(
    assessor: &RocmTensorAssessor<'_>,
    pool: PcuMemoryPoolId,
    memory: &mut impl PcuMemoryProvider<Resource = RocmMemoryResource>,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut graph = Graph::default();
    let samples = graph.input([2, 2])?;
    let w1 = graph.input([2, 2])?;
    let w2 = graph.input([2, 2])?;
    let w3 = graph.input([2, 1])?;
    let target = graph.input([2, 1])?;
    let hidden1 = graph.matmul(samples, w1)?;
    let activated1 = graph.relu(hidden1)?;
    let hidden2 = graph.matmul(activated1, w2)?;
    let activated2 = graph.relu(hidden2)?;
    let prediction = graph.matmul(activated2, w3)?;
    let loss = graph.mean_squared_error(prediction, target)?;
    let gradients = graph.backward_mse(loss)?;
    let w1_index = graph
        .nodes()
        .position(|node| node.value == w1)
        .ok_or("first weight missing from MLP graph")?;
    let w1_gradient = gradients[w1_index].ok_or("first weight gradient missing")?;
    let inputs = [
        (samples, Tensor::new([2, 2], vec![1.0, -2.0, 2.0, 1.0])?),
        (w1, Tensor::new([2, 2], vec![0.5, -0.25, 0.75, 1.0])?),
        (w2, Tensor::new([2, 2], vec![1.0, -0.5, 0.25, 0.75])?),
        (w3, Tensor::new([2, 1], vec![0.5, -1.0])?),
        (target, Tensor::new([2, 1], vec![0.25, -0.75])?),
    ];
    let execution = graph.evaluate(&inputs)?;
    let expected = execution.gradients(&graph, loss)?[w1_index]
        .as_ref()
        .ok_or("CPU weight gradient missing")?
        .clone();
    let actual = assessor.execute_graph(&graph, &inputs, w1_gradient, pool, memory)?;
    if actual.shape() == expected.shape()
        && actual
            .data()
            .iter()
            .zip(expected.data())
            .all(|(a, b)| (a - b).abs() <= 1.0e-4)
    {
        Ok(())
    } else {
        Err(format!("ROCm ReLU autograd output {actual:?}, expected {expected:?}").into())
    }
}
