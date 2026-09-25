//! Explicit `ROCm` selection and a two-layer PCU MLP forward graph.

use std::process::ExitCode;

#[path = "selection.rs"]
mod selection;

use fusion_pcu::{
    PcuMemoryPoolId,
    PcuMemoryProvider,
    PcuOwnedDispatchMemorySession,
};
use fusion_pcu_rocm::{
    RocmDiscovery,
    RocmMemoryResource,
    RocmTensorAssessor,
};
use fusion_pcu_tensor::{
    Graph,
    Tensor,
    TensorExecutionRoute,
    TensorGraphAssessment,
    TensorOperationSupport,
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
                verify_relu_edges(&assessor, candidate.pool, &mut memory)?;
                verify_relu_backward(&assessor, candidate.pool, &mut memory)?;
                verify_relu_autograd(&assessor, candidate.pool, &mut memory)?;
                verify_mse(&assessor, candidate.pool, &mut memory)?;
                verify_backward_algebra(&assessor, candidate.pool, &mut memory)?;
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
        Ok(())
    } else {
        Err(format!("ROCm MSE output {actual:?}, expected {expected:?}").into())
    }
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
