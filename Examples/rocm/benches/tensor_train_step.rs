//! Compare a prepared, resident-input PCU training graph with direct HIP and rocBLAS.

mod support;

use std::{
    error::Error,
    hint::black_box,
};

use criterion::{
    criterion_group,
    criterion_main,
    BenchmarkId,
    Criterion,
    Throughput,
};
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
#[path = "support/memory_profile.rs"]
mod memory_profile;
use memory_profile::ProfiledMemory;

#[path = "support/train_step.rs"]
mod native;

struct Program {
    graph: Graph,
    samples: ValueId,
    weights: ValueId,
    target: ValueId,
    rate: ValueId,
    output: ValueId,
}

fn program(rows: usize, features: usize) -> Result<Program, Box<dyn Error>> {
    let mut graph = Graph::default();
    let samples = graph.input([rows, features])?;
    let weights = graph.input([features, 1])?;
    let target = graph.input([rows, 1])?;
    let rate = graph.input([features, 1])?;
    let prediction = graph.matmul(samples, weights)?;
    let loss = graph.mean_squared_error(prediction, target)?;
    let gradients = graph.backward_mse(loss)?;
    let weight_index = graph
        .nodes()
        .position(|node| node.value == weights)
        .ok_or("weight input missing")?;
    let gradient = gradients[weight_index].ok_or("weight gradient missing")?;
    let scaled = graph.mul(gradient, rate)?;
    let output = graph.sub(weights, scaled)?;
    Ok(Program {
        graph,
        samples,
        weights,
        target,
        rate,
        output,
    })
}

fn bench(criterion: &mut Criterion) {
    run(criterion).expect("paired ROCm training-step benchmark failed");
}

fn run(criterion: &mut Criterion) -> Result<(), Box<dyn Error>> {
    let discovery = RocmDiscovery::new();
    let mut failures = Vec::new();
    for candidate in support::selected_candidates(&discovery)? {
        match run_on(criterion, &discovery, &candidate) {
            Ok(()) => return Ok(()),
            Err(error) => failures.push(format!("{}: {error}", candidate.name)),
        }
    }
    Err(format!(
        "no ROCm device completed training-step benchmark: {}",
        failures.join("; ")
    )
    .into())
}

#[allow(clippy::too_many_lines)] // Keep paired setup, identical inputs, and verification together.
fn run_on(
    criterion: &mut Criterion,
    discovery: &RocmDiscovery,
    selected: &support::selection::Candidate,
) -> Result<(), Box<dyn Error>> {
    let session = RocmOwnedDispatchBackend::open(discovery, selected.device, 64)?;
    let assessor = RocmTensorAssessor::new(&session)?;
    println!(
        "Device: {}; two-step linear-regression training",
        selected.name
    );
    for (rows, features) in [(4_usize, 2_usize), (1024, 64), (8192, 1024)] {
        let program = program(rows, features)?;
        let sample_values = (0..rows * features)
            .map(|index| (small_integer(index % 17) - 8.0) / 8.0)
            .collect::<Vec<_>>();
        let target_values = (0..rows)
            .map(|index| small_integer(index % 7) / 4.0 - 0.5)
            .collect::<Vec<_>>();
        let factor = 2.0 / f32::from(u16::try_from(rows)?);
        let factor_values = vec![factor; rows];
        let initial_weights = (0..features)
            .map(|index| (small_integer(index % 9) - 4.0) / 16.0)
            .collect::<Vec<_>>();
        let rate_values = vec![0.001_f32; features];
        let samples = Tensor::new([rows, features], sample_values.clone())?;
        let target = Tensor::new([rows, 1], target_values.clone())?;
        let rate = Tensor::new([features, 1], rate_values.clone())?;
        let initial = Tensor::new([features, 1], initial_weights.clone())?;
        let prepared = assessor.prepare_graph(&program.graph, program.output)?;
        let mut memory = PcuOwnedDispatchMemorySession::memory_provider(&session, selected.pool);
        let device_samples = assessor.upload_input(&samples, selected.pool, &mut memory)?;
        let device_weights = assessor.upload_input(&initial, selected.pool, &mut memory)?;
        let device_target = assessor.upload_input(&target, selected.pool, &mut memory)?;
        let device_rate = assessor.upload_input(&rate, selected.pool, &mut memory)?;
        let mut scratch = assessor.prepare_scratch(&prepared, selected.pool, &mut memory)?;
        let mut diagnostic_scratch =
            assessor.prepare_scratch(&prepared, selected.pool, &mut memory)?;
        let native = support::cold_once("native HIP training setup", || {
            native::NativeTrainStep::prepare(
                discovery,
                selected.device,
                &native::TrainInputs {
                    rows,
                    features,
                    samples: &sample_values,
                    target: &target_values,
                    factor: &factor_values,
                    initial_weights: &initial_weights,
                    rate: &rate_values,
                },
            )
        })?;
        let expected = cpu_two_steps(&program, &samples, &target, &rate, &initial)?;
        let mut execute_pcu = || -> Result<Vec<f32>, Box<dyn Error>> {
            let first_inputs = [
                (program.samples, &device_samples),
                (program.weights, &device_weights),
                (program.target, &device_target),
                (program.rate, &device_rate),
            ];
            let first = assessor.execute_prepared_with_resources_and_scratch_resident(
                &prepared,
                &first_inputs,
                &mut scratch,
                &mut memory,
            )?;
            let second_inputs = [
                (program.samples, &device_samples),
                (program.weights, &first),
                (program.target, &device_target),
                (program.rate, &device_rate),
            ];
            let final_output = assessor.execute_prepared_with_resources_and_scratch_resident(
                &prepared,
                &second_inputs,
                &mut scratch,
                &mut memory,
            )?;
            let final_output =
                assessor.download_output(&final_output, selected.pool, &mut memory)?;
            Ok(final_output.data().to_vec())
        };
        let pcu_cold = execute_pcu()?;
        let native_cold = native.execute_two()?;
        verify(&expected, &pcu_cold).map_err(|error| format!("PCU cold: {error}"))?;
        verify(&expected, &native_cold).map_err(|error| format!("native cold: {error}"))?;
        {
            let mut group = criterion.benchmark_group("tensor_train_step");
            group.throughput(Throughput::Elements(u64::try_from(rows * features * 2)?));
            group.bench_function(
                BenchmarkId::new("pcu_prepared_resident", format!("{rows}x{features}")),
                |b| {
                    b.iter(|| black_box(execute_pcu().expect("PCU two-step training failed")));
                },
            );
            group.bench_function(
                BenchmarkId::new("native_hip_rocblas", format!("{rows}x{features}")),
                |b| {
                    b.iter(|| {
                        black_box(
                            native
                                .execute_two()
                                .expect("native two-step training failed"),
                        )
                    });
                },
            );
            group.finish();
        }
        verify(&expected, &execute_pcu()?).map_err(|error| format!("PCU final: {error}"))?;
        verify(&expected, &native.execute_two()?)
            .map_err(|error| format!("native final: {error}"))?;
        let mut diagnostic = ProfiledMemory::new(PcuOwnedDispatchMemorySession::memory_provider(
            &session,
            selected.pool,
        ));
        let first_inputs = [
            (program.samples, &device_samples),
            (program.weights, &device_weights),
            (program.target, &device_target),
            (program.rate, &device_rate),
        ];
        let first = assessor.execute_prepared_with_resources_and_scratch_resident(
            &prepared,
            &first_inputs,
            &mut diagnostic_scratch,
            &mut diagnostic,
        )?;
        let second_inputs = [
            (program.samples, &device_samples),
            (program.weights, &first),
            (program.target, &device_target),
            (program.rate, &device_rate),
        ];
        let last = assessor.execute_prepared_with_resources_and_scratch_resident(
            &prepared,
            &second_inputs,
            &mut diagnostic_scratch,
            &mut diagnostic,
        )?;
        let diagnostic_result = assessor.download_output(&last, selected.pool, &mut diagnostic)?;
        verify(&expected, diagnostic_result.data())?;
        let profile = diagnostic.profile();
        println!(
            "PCU {rows}x{features} provider: {} allocations ({} bytes, {:?}), {} uploads ({} bytes, {:?}), {} downloads ({} bytes, {:?}) per two steps",
            profile.allocations,
            profile.allocated_bytes,
            profile.allocation_time,
            profile.uploads,
            profile.uploaded_bytes,
            profile.upload_time,
            profile.downloads,
            profile.downloaded_bytes,
            profile.download_time,
        );
    }
    Ok(())
}

fn cpu_two_steps(
    program: &Program,
    samples: &Tensor,
    target: &Tensor,
    rate: &Tensor,
    initial: &Tensor,
) -> Result<Vec<f32>, Box<dyn Error>> {
    let mut weights = initial.clone();
    for _ in 0..2 {
        let inputs = [
            (program.samples, samples.clone()),
            (program.weights, weights.clone()),
            (program.target, target.clone()),
            (program.rate, rate.clone()),
        ];
        let execution = program.graph.evaluate(&inputs)?;
        weights = execution.value(program.output)?.clone();
    }
    Ok(weights.data().to_vec())
}

fn verify(expected: &[f32], actual: &[f32]) -> Result<(), Box<dyn Error>> {
    if expected.len() != actual.len() {
        return Err(format!(
            "training output length mismatch: expected {}, got {}",
            expected.len(),
            actual.len()
        )
        .into());
    }
    let (index, error) = expected
        .iter()
        .zip(actual)
        .enumerate()
        .map(|(index, (expected, actual))| (index, (expected - actual).abs()))
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .ok_or("empty training output")?;
    if error <= 2.0e-4 {
        Ok(())
    } else {
        Err(format!(
            "training output mismatch: max abs error {error} at {index}: expected {}, got {}",
            expected[index], actual[index]
        )
        .into())
    }
}

fn small_integer(value: usize) -> f32 {
    f32::from(u8::try_from(value).expect("bounded benchmark pattern fits in u8"))
}

criterion_group! { name = benches; config = support::criterion_config(); targets = bench }
criterion_main!(benches);
