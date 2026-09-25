//! One explicitly selected `ROCm` linear-regression gradient and SGD update graph.

use std::{
    error::Error,
    process::ExitCode,
};

#[path = "selection.rs"]
mod selection;

use fusion_pcu::PcuOwnedDispatchMemorySession;
use fusion_pcu_rocm::{
    RocmDiscovery,
    RocmOwnedDispatchBackend,
    RocmTensorAssessor,
};
use fusion_pcu_tensor::{
    Graph,
    Tensor,
    ValueId,
};

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("ERROR fusion-rocm-train-step: {error}");
            ExitCode::FAILURE
        }
    }
}

struct Program {
    graph: Graph,
    samples: ValueId,
    weights: ValueId,
    target: ValueId,
    learning_rate: ValueId,
    loss: ValueId,
    updated_weights: ValueId,
}

fn program() -> Result<Program, Box<dyn Error>> {
    let mut graph = Graph::default();
    let samples = graph.input([4, 2])?;
    let weights = graph.input([2, 1])?;
    let target = graph.input([4, 1])?;
    let learning_rate = graph.input([2, 1])?;
    let prediction = graph.matmul(samples, weights)?;
    let loss = graph.mean_squared_error(prediction, target)?;
    let gradients = graph.backward_mse(loss)?;
    let weight_index = graph
        .nodes()
        .position(|node| node.value == weights)
        .ok_or("weight input absent from graph")?;
    let weight_gradient = gradients[weight_index].ok_or("weight gradient absent")?;
    let scaled_gradient = graph.mul(weight_gradient, learning_rate)?;
    let updated_weights = graph.sub(weights, scaled_gradient)?;
    Ok(Program {
        graph,
        samples,
        weights,
        target,
        learning_rate,
        loss,
        updated_weights,
    })
}

fn expected_step(
    program: &Program,
    inputs: &[(ValueId, Tensor)],
) -> Result<Tensor, Box<dyn Error>> {
    let execution = program.graph.evaluate(inputs)?;
    let gradients = execution.gradients(&program.graph, program.loss)?;
    let weight_index = program
        .graph
        .nodes()
        .position(|node| node.value == program.weights)
        .ok_or("weight input absent from graph")?;
    let gradient = gradients[weight_index]
        .as_ref()
        .ok_or("weight gradient absent")?;
    let weights = inputs
        .iter()
        .find(|(id, _)| *id == program.weights)
        .ok_or("weights absent")?
        .1
        .data();
    let rate = inputs
        .iter()
        .find(|(id, _)| *id == program.learning_rate)
        .ok_or("learning rate absent")?
        .1
        .data();
    Ok(Tensor::new(
        [2, 1],
        weights
            .iter()
            .zip(gradient.data())
            .zip(rate)
            .map(|((w, g), rate)| w - rate * g)
            .collect(),
    )?)
}

fn run() -> Result<(), Box<dyn Error>> {
    let discovery = RocmDiscovery::new();
    let candidates = selection::ranked_devices(&discovery, selection::preferred_device()?, false)?;
    let program = program()?;
    let samples = Tensor::new([4, 2], vec![1.0, 2.0, 3.0, 4.0, -1.0, 2.0, 2.0, -3.0])?;
    let initial_weights = Tensor::new([2, 1], vec![0.25, -0.5])?;
    let target = Tensor::new([4, 1], vec![1.0, -1.0, 0.5, 2.0])?;
    let rate = Tensor::new([2, 1], vec![0.05; 2])?;
    let mut failures = Vec::new();
    for candidate in candidates {
        match run_on(
            &discovery,
            &candidate,
            &program,
            &samples,
            &initial_weights,
            &target,
            &rate,
        ) {
            Ok(()) => {
                println!(
                    "PCU linear-regression gradient and two SGD steps passed on {}",
                    candidate.name
                );
                return Ok(());
            }
            Err(error) => failures.push(format!("{}: {error}", candidate.name)),
        }
    }
    Err(format!(
        "no ROCm device completed train-step graph: {}",
        failures.join("; ")
    )
    .into())
}

fn run_on(
    discovery: &RocmDiscovery,
    candidate: &selection::Candidate,
    program: &Program,
    samples: &Tensor,
    initial_weights: &Tensor,
    target: &Tensor,
    rate: &Tensor,
) -> Result<(), Box<dyn Error>> {
    let session = RocmOwnedDispatchBackend::open(discovery, candidate.device, 64)?;
    let assessor = RocmTensorAssessor::new(&session)?;
    let prepared = assessor.prepare_graph(&program.graph, program.updated_weights)?;
    let mut memory = PcuOwnedDispatchMemorySession::memory_provider(&session, candidate.pool);
    let samples_device = assessor.upload_input(samples, candidate.pool, &mut memory)?;
    let mut weights_device = assessor.upload_input(initial_weights, candidate.pool, &mut memory)?;
    let target_device = assessor.upload_input(target, candidate.pool, &mut memory)?;
    let rate_device = assessor.upload_input(rate, candidate.pool, &mut memory)?;
    let mut weights = initial_weights.clone();
    for step in 0..2 {
        let host_inputs = [
            (program.samples, samples.clone()),
            (program.weights, weights.clone()),
            (program.target, target.clone()),
            (program.learning_rate, rate.clone()),
        ];
        let loss_before = program
            .graph
            .evaluate(&host_inputs)?
            .value(program.loss)?
            .data()[0];
        let expected = expected_step(program, &host_inputs)?;
        let device_inputs = [
            (program.samples, &samples_device),
            (program.weights, &weights_device),
            (program.target, &target_device),
            (program.learning_rate, &rate_device),
        ];
        let next_weights_device = assessor.execute_prepared_with_resources_resident(
            &prepared,
            &device_inputs,
            candidate.pool,
            &mut memory,
        )?;
        let actual = assessor.download_output(&next_weights_device, candidate.pool, &mut memory)?;
        if actual.shape() != expected.shape()
            || actual
                .data()
                .iter()
                .zip(expected.data())
                .any(|(a, b)| (a - b).abs() > 1.0e-5)
        {
            return Err(format!("step {step} produced {actual:?}, expected {expected:?}").into());
        }
        let mut next_inputs = host_inputs;
        next_inputs[1].1 = actual.clone();
        let loss_after = program
            .graph
            .evaluate(&next_inputs)?
            .value(program.loss)?
            .data()[0];
        if !matches!(
            loss_after.partial_cmp(&loss_before),
            Some(std::cmp::Ordering::Less)
        ) {
            return Err(
                format!("step {step} did not reduce loss: {loss_before} -> {loss_after}").into(),
            );
        }
        weights = actual;
        weights_device = next_weights_device;
    }
    Ok(())
}
