//! Compare fresh-output and banked prepared PCU training with direct HIP and rocBLAS.

extern crate pcu_facade as fusion_pcu;

#[path = "../support/alloc.rs"]
pub mod alloc;
#[path = "../../examples/support/reference/reference.rs"]
mod reference;
#[path = "../support/support.rs"]
mod support;

#[rustfmt::skip]
use std::{
    error::Error,
    hint::black_box,
    time::Instant,
};

#[rustfmt::skip]
use criterion::{
    criterion_group,
    criterion_main,
    BenchmarkId,
    Criterion,
    Throughput,
};
#[rustfmt::skip]
use alloc::{
    AllocationCapture,
    AllocationCounts,
};
use fusion_pcu::PcuOwnedDispatchMemorySession;
#[rustfmt::skip]
use fusion_pcu_rocm::{
    RocmDiscovery,
    RocmOwnedDispatchBackend,
    RocmTensorAssessor,
};
#[rustfmt::skip]
use fusion_pcu::dialect::tensor::{
    Graph,
    Tensor,
    ValueId,
};
#[path = "../support/memory_profile.rs"]
mod memory_profile;
use memory_profile::ProfiledMemory;
#[path = "../support/train_step_reference.rs"]
mod train_step_reference;
#[rustfmt::skip]
use train_step_reference::{
    cpu_strict_two_steps,
    cpu_two_steps,
    verify,
};

#[path = "../support/train_step.rs"]
#[allow(dead_code)] // Shared native reference also supplies opt-in volume diagnostics.
mod native;

struct Program {
    graph: Graph,
    samples: ValueId,
    weights: ValueId,
    target: ValueId,
    rate: ValueId,
    output: ValueId,
}

struct StrictProgram {
    graph: Graph,
    samples: ValueId,
    weights: ValueId,
    target: ValueId,
    output: ValueId,
}

#[derive(Clone, Copy)]
struct PcuPhaseTimings {
    step_one: std::time::Duration,
    step_two: std::time::Duration,
    output_readback: std::time::Duration,
    total: std::time::Duration,
}

macro_rules! record_node_timings {
    ($timings:expr, $node_keys:expr, $node_samples:expr, $sgemm_samples:expr,
     $elementwise_samples:expr, $step_wall_samples:expr,
     $node_sum_samples:expr, $non_node_samples:expr, $step_wall:expr $(,)?) => {{
        let timings = $timings;
        let node_keys = $node_keys;
        let node_samples = $node_samples;
        let sgemm_samples = $sgemm_samples;
        let elementwise_samples = $elementwise_samples;
        if node_keys.is_empty() {
            node_keys.extend(
                timings
                    .iter()
                    .map(|timing| (timing.operation, timing.value)),
            );
            node_samples.resize_with(timings.len(), Vec::new);
            sgemm_samples.resize_with(timings.len(), Vec::new);
            elementwise_samples.resize_with(timings.len(), Vec::new);
        } else if node_keys.len() != timings.len()
            || node_keys
                .iter()
                .zip(timings)
                .any(|(key, timing)| *key != (timing.operation, timing.value))
        {
            return Err("strict profiled node sequence changed between executions".into());
        }
        let node_sum: std::time::Duration = timings.iter().map(|timing| timing.elapsed).sum();
        for (((samples, sgemm), elementwise), timing) in node_samples
            .iter_mut()
            .zip(sgemm_samples.iter_mut())
            .zip(elementwise_samples.iter_mut())
            .zip(timings)
        {
            samples.push(timing.elapsed.as_secs_f64());
            if let Some(host) = timing.sgemm_host {
                sgemm.push(host);
            }
            if let Some(host) = timing.elementwise_host {
                elementwise.push(host);
            }
        }
        let step_wall = $step_wall;
        $step_wall_samples.push(step_wall);
        $node_sum_samples.push(node_sum);
        $non_node_samples.push(step_wall.saturating_sub(node_sum));
    }};
}

fn program(rows: usize, features: usize) -> Result<Program, Box<dyn Error>> {
    let mut graph = Graph::default();
    let samples = graph.input([rows, features], fusion_pcu::PcuScalarType::F32)?;
    let weights = graph.input([features, 1], fusion_pcu::PcuScalarType::F32)?;
    let target = graph.input([rows, 1], fusion_pcu::PcuScalarType::F32)?;
    let rate = graph.input([features, 1], fusion_pcu::PcuScalarType::F32)?;
    let prediction = graph.matmul(samples, weights)?;
    let loss = graph.mean_squared_error(prediction, target)?;
    let gradients = graph.backward_mse(loss)?;
    let weight_index = graph
        .nodes()
        .position(|node| node.value == weights)
        .ok_or("weight input missing")?;
    let gradient = gradients[weight_index].ok_or("weight gradient missing")?;
    // Keep the established PCU graph unchanged: Mul and Sub are separate operations, while the
    // native training_update kernel evaluates weights - rate * gradient in one kernel. The
    // banked-versus-fresh PCU comparison isolates output storage; PCU/native still includes this
    // kernel-count and floating-point contraction difference.
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

fn strict_program(rows: usize, features: usize) -> Result<StrictProgram, Box<dyn Error>> {
    let mut graph = Graph::default();
    let samples = graph.input([rows, features], fusion_pcu::PcuScalarType::F32)?;
    let weights = graph.input([features, 1], fusion_pcu::PcuScalarType::F32)?;
    let target = graph.input([rows, 1], fusion_pcu::PcuScalarType::F32)?;
    let prediction = graph.matmul(samples, weights)?;
    let loss = graph.mean_squared_error(prediction, target)?;
    let gradients = graph.backward_mse(loss)?;
    let weight_index = graph
        .nodes()
        .position(|node| node.value == weights)
        .ok_or("weight input missing")?;
    let gradient = gradients[weight_index].ok_or("weight gradient missing")?;
    // SgdUpdate preserves the multiply-then-subtract rounding boundary in the strict ROCm path.
    let output = graph.sgd_update(weights, gradient, 0.001)?;
    Ok(StrictProgram {
        graph,
        samples,
        weights,
        target,
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

#[allow(clippy::too_many_lines, clippy::cognitive_complexity)]
// Keep paired setup and verified two-step profiling beside the benchmark's shared inputs.
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
        let mut bank_memory =
            PcuOwnedDispatchMemorySession::memory_provider(&session, selected.pool);
        let (mut bank_scratch, mut first_bank, mut second_bank) =
            support::cold_once("PCU two-step output-bank and scratch preparation", || {
                Ok::<_, Box<dyn Error>>((
                    assessor.prepare_scratch(&prepared, selected.pool, &mut bank_memory)?,
                    assessor.prepare_output_bank(&prepared, selected.pool, &mut bank_memory)?,
                    assessor.prepare_output_bank(&prepared, selected.pool, &mut bank_memory)?,
                ))
            })?;
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
            Ok(final_output.into_data())
        };
        let mut execute_pcu_banked = || -> Result<Vec<f32>, Box<dyn Error>> {
            let first_inputs = [
                (program.samples, &device_samples),
                (program.weights, &device_weights),
                (program.target, &device_target),
                (program.rate, &device_rate),
            ];
            assessor.execute_prepared_outputs_into_bank(
                &prepared,
                &first_inputs,
                &mut bank_scratch,
                &mut first_bank,
                &mut bank_memory,
            )?;
            let first_weights = first_bank
                .outputs()
                .first()
                .ok_or("first output bank is empty")?;
            let second_inputs = [
                (program.samples, &device_samples),
                (program.weights, first_weights),
                (program.target, &device_target),
                (program.rate, &device_rate),
            ];
            assessor.execute_prepared_outputs_into_bank(
                &prepared,
                &second_inputs,
                &mut bank_scratch,
                &mut second_bank,
                &mut bank_memory,
            )?;
            let final_weights = second_bank
                .outputs()
                .first()
                .ok_or("second output bank is empty")?;
            let final_output =
                assessor.download_output(final_weights, selected.pool, &mut bank_memory)?;
            Ok(final_output.into_data())
        };
        let pcu_cold = execute_pcu()?;
        let pcu_banked_cold = execute_pcu_banked()?;
        let native_cold = native.execute_two()?;
        verify(&expected, &pcu_cold).map_err(|error| format!("PCU cold: {error}"))?;
        verify(&expected, &pcu_banked_cold).map_err(|error| format!("PCU banked cold: {error}"))?;
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
                BenchmarkId::new(
                    "pcu_prepared_resident_output_bank",
                    format!("{rows}x{features}"),
                ),
                |b| {
                    b.iter(|| {
                        black_box(
                            execute_pcu_banked().expect("PCU banked two-step training failed"),
                        )
                    });
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
        verify(&expected, &execute_pcu_banked()?)
            .map_err(|error| format!("PCU banked final: {error}"))?;
        verify(&expected, &native.execute_two()?)
            .map_err(|error| format!("native final: {error}"))?;
        let mut paired_fresh = Vec::with_capacity(16);
        let mut paired_fresh_native = Vec::with_capacity(16);
        let mut paired_fresh_ratio = Vec::with_capacity(16);
        let mut paired_bank = Vec::with_capacity(16);
        let mut paired_bank_native = Vec::with_capacity(16);
        let mut paired_bank_ratio = Vec::with_capacity(16);
        for pair in 0_usize..16 {
            let (fresh_time, native_time, bank_time, bank_native_time) = if pair.is_multiple_of(2) {
                let fresh_time = time_execution(&mut execute_pcu)?;
                let native_time = time_execution(|| native.execute_two())?;
                let bank_time = time_execution(&mut execute_pcu_banked)?;
                let bank_native_time = time_execution(|| native.execute_two())?;
                (fresh_time, native_time, bank_time, bank_native_time)
            } else {
                let bank_native_time = time_execution(|| native.execute_two())?;
                let bank_time = time_execution(&mut execute_pcu_banked)?;
                let native_time = time_execution(|| native.execute_two())?;
                let fresh_time = time_execution(&mut execute_pcu)?;
                (fresh_time, native_time, bank_time, bank_native_time)
            };
            paired_fresh.push(fresh_time);
            paired_fresh_native.push(native_time);
            paired_fresh_ratio.push(fresh_time / native_time);
            paired_bank.push(bank_time);
            paired_bank_native.push(bank_native_time);
            paired_bank_ratio.push(bank_time / bank_native_time);
        }
        println!(
            "{rows}x{features} order-alternating paired host-wall diagnostics (16 pairs): fresh PCU median {:.3} us, native median {:.3} us, paired ratio median {:.3}x, range {:.3}–{:.3}x; banked PCU median {:.3} us, native median {:.3} us, banked/native median {:.3}x, range {:.3}–{:.3}x",
            median(&mut paired_fresh) * 1_000_000.0,
            median(&mut paired_fresh_native) * 1_000_000.0,
            median(&mut paired_fresh_ratio),
            min(&paired_fresh_ratio),
            max(&paired_fresh_ratio),
            median(&mut paired_bank) * 1_000_000.0,
            median(&mut paired_bank_native) * 1_000_000.0,
            median(&mut paired_bank_ratio),
            min(&paired_bank_ratio),
            max(&paired_bank_ratio),
        );
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
        let mut banked_scratch = assessor.prepare_scratch(&prepared, selected.pool, &mut memory)?;
        let mut banked_first =
            assessor.prepare_output_bank(&prepared, selected.pool, &mut memory)?;
        let mut banked_second =
            assessor.prepare_output_bank(&prepared, selected.pool, &mut memory)?;
        let mut banked_diagnostic = ProfiledMemory::new(
            PcuOwnedDispatchMemorySession::memory_provider(&session, selected.pool),
        );
        let first_inputs = [
            (program.samples, &device_samples),
            (program.weights, &device_weights),
            (program.target, &device_target),
            (program.rate, &device_rate),
        ];
        assessor.execute_prepared_outputs_into_bank(
            &prepared,
            &first_inputs,
            &mut banked_scratch,
            &mut banked_first,
            &mut banked_diagnostic,
        )?;
        let first_weights = banked_first
            .outputs()
            .first()
            .ok_or("first output bank is empty")?;
        let second_inputs = [
            (program.samples, &device_samples),
            (program.weights, first_weights),
            (program.target, &device_target),
            (program.rate, &device_rate),
        ];
        assessor.execute_prepared_outputs_into_bank(
            &prepared,
            &second_inputs,
            &mut banked_scratch,
            &mut banked_second,
            &mut banked_diagnostic,
        )?;
        let last = banked_second
            .outputs()
            .first()
            .ok_or("second output bank is empty")?;
        let diagnostic_result =
            assessor.download_output(last, selected.pool, &mut banked_diagnostic)?;
        verify(&expected, diagnostic_result.data())?;
        let profile = banked_diagnostic.profile();
        println!(
            "PCU {rows}x{features} warm output-bank provider: {} allocations ({} bytes, {:?}), {} uploads ({} bytes, {:?}), {} downloads ({} bytes, {:?}) per two steps",
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

        // A route-matched strict profile uses the dedicated SGD update operation, which keeps
        // the multiply/subtract rounding boundary explicit and reduces the PCU pointwise work
        // to the same three kernels per step as the strict native peer.
        let strict = strict_program(rows, features)?;
        let strict_prepared = assessor.prepare_graph(&strict.graph, strict.output)?;
        let mut strict_bank_memory =
            PcuOwnedDispatchMemorySession::memory_provider(&session, selected.pool);
        let (mut strict_bank_scratch, mut strict_first_bank, mut strict_second_bank) =
            support::cold_once(
                "strict PCU two-step output-bank and scratch preparation",
                || {
                    Ok::<_, Box<dyn Error>>((
                        assessor.prepare_scratch(
                            &strict_prepared,
                            selected.pool,
                            &mut strict_bank_memory,
                        )?,
                        assessor.prepare_output_bank(
                            &strict_prepared,
                            selected.pool,
                            &mut strict_bank_memory,
                        )?,
                        assessor.prepare_output_bank(
                            &strict_prepared,
                            selected.pool,
                            &mut strict_bank_memory,
                        )?,
                    ))
                },
            )?;
        let strict_native = support::cold_once("strict native HIP training setup", || {
            native::NativeTrainStep::prepare_strict(
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
                0.001,
            )
        })?;
        let strict_expected = cpu_strict_two_steps(&strict, &samples, &target, &initial)?;
        {
            let mut execute_strict_pcu_banked = |batched: bool,
                                                 mut device_times: Option<&mut Vec<f32>>|
             -> Result<Vec<f32>, Box<dyn Error>> {
                let first_inputs = [
                    (strict.samples, &device_samples),
                    (strict.weights, &device_weights),
                    (strict.target, &device_target),
                ];
                if batched {
                    if let Some(times) = device_times.as_deref_mut() {
                        times.extend(
                            assessor.execute_prepared_outputs_into_bank_batched_device_timed(
                                &strict_prepared,
                                &first_inputs,
                                &mut strict_bank_scratch,
                                &mut strict_first_bank,
                                &mut strict_bank_memory,
                            )?,
                        );
                    } else {
                        assessor.execute_prepared_outputs_into_bank_batched(
                            &strict_prepared,
                            &first_inputs,
                            &mut strict_bank_scratch,
                            &mut strict_first_bank,
                            &mut strict_bank_memory,
                        )?;
                    }
                } else {
                    assessor.execute_prepared_outputs_into_bank(
                        &strict_prepared,
                        &first_inputs,
                        &mut strict_bank_scratch,
                        &mut strict_first_bank,
                        &mut strict_bank_memory,
                    )?;
                }
                let first_weights = strict_first_bank
                    .outputs()
                    .first()
                    .ok_or("strict first output bank is empty")?;
                let second_inputs = [
                    (strict.samples, &device_samples),
                    (strict.weights, first_weights),
                    (strict.target, &device_target),
                ];
                if batched {
                    if let Some(times) = device_times.as_deref_mut() {
                        times.extend(
                            assessor.execute_prepared_outputs_into_bank_batched_device_timed(
                                &strict_prepared,
                                &second_inputs,
                                &mut strict_bank_scratch,
                                &mut strict_second_bank,
                                &mut strict_bank_memory,
                            )?,
                        );
                    } else {
                        assessor.execute_prepared_outputs_into_bank_batched(
                            &strict_prepared,
                            &second_inputs,
                            &mut strict_bank_scratch,
                            &mut strict_second_bank,
                            &mut strict_bank_memory,
                        )?;
                    }
                } else {
                    assessor.execute_prepared_outputs_into_bank(
                        &strict_prepared,
                        &second_inputs,
                        &mut strict_bank_scratch,
                        &mut strict_second_bank,
                        &mut strict_bank_memory,
                    )?;
                }
                let final_weights = strict_second_bank
                    .outputs()
                    .first()
                    .ok_or("strict second output bank is empty")?;
                let final_output = assessor.download_output(
                    final_weights,
                    selected.pool,
                    &mut strict_bank_memory,
                )?;
                Ok(final_output.into_data())
            };
            verify(&strict_expected, &execute_strict_pcu_banked(false, None)?)
                .map_err(|error| format!("strict PCU banked: {error}"))?;
            verify(&strict_expected, &execute_strict_pcu_banked(true, None)?)
                .map_err(|error| format!("strict PCU batched banked: {error}"))?;
            verify(&strict_expected, &strict_native.execute_two()?)
                .map_err(|error| format!("strict native: {error}"))?;
            verify(&strict_expected, &strict_native.execute_two_batched()?)
                .map_err(|error| format!("strict native batched: {error}"))?;
            verify(
                &strict_expected,
                &strict_native.execute_two_fully_batched()?,
            )
            .map_err(|error| format!("strict native fully batched: {error}"))?;
            let mut pcu_allocations = [AllocationCounts::default(); 16];
            let mut native_allocations = [AllocationCounts::default(); 16];
            for pair in 0_usize..16 {
                if pair.is_multiple_of(2) {
                    let _capture = AllocationCapture::start();
                    let output = execute_strict_pcu_banked(false, None)?;
                    let measured = AllocationCapture::finish();
                    pcu_allocations[pair] = measured;
                    verify(&strict_expected, &output)
                        .map_err(|error| format!("strict PCU allocation diagnostic: {error}"))?;

                    let _capture = AllocationCapture::start();
                    let output = strict_native.execute_two()?;
                    let measured = AllocationCapture::finish();
                    native_allocations[pair] = measured;
                    verify(&strict_expected, &output)
                        .map_err(|error| format!("strict native allocation diagnostic: {error}"))?;
                } else {
                    let _capture = AllocationCapture::start();
                    let output = strict_native.execute_two()?;
                    let measured = AllocationCapture::finish();
                    native_allocations[pair] = measured;
                    verify(&strict_expected, &output)
                        .map_err(|error| format!("strict native allocation diagnostic: {error}"))?;

                    let _capture = AllocationCapture::start();
                    let output = execute_strict_pcu_banked(false, None)?;
                    let measured = AllocationCapture::finish();
                    pcu_allocations[pair] = measured;
                    verify(&strict_expected, &output)
                        .map_err(|error| format!("strict PCU allocation diagnostic: {error}"))?;
                }
            }
            println!(
                "{rows}x{features} strict warm alternating allocation diagnostics (16 executions each; per-execution median): PCU {} allocs, {} reallocs, {} deallocs, {} requested bytes; native {} allocs, {} reallocs, {} deallocs, {} requested bytes",
                median_allocation_field(&pcu_allocations, |counts| counts.alloc_calls),
                median_allocation_field(&pcu_allocations, |counts| counts.realloc_calls),
                median_allocation_field(&pcu_allocations, |counts| counts.dealloc_calls),
                median_allocation_field(&pcu_allocations, |counts| counts.requested_bytes),
                median_allocation_field(&native_allocations, |counts| counts.alloc_calls),
                median_allocation_field(&native_allocations, |counts| counts.realloc_calls),
                median_allocation_field(&native_allocations, |counts| counts.dealloc_calls),
                median_allocation_field(&native_allocations, |counts| counts.requested_bytes),
            );
            let mut pcu_batched_allocations = [AllocationCounts::default(); 16];
            let mut native_batched_allocations = [AllocationCounts::default(); 16];
            for pair in 0_usize..16 {
                if pair.is_multiple_of(2) {
                    let _capture = AllocationCapture::start();
                    let output = execute_strict_pcu_banked(true, None)?;
                    pcu_batched_allocations[pair] = AllocationCapture::finish();
                    verify(&strict_expected, &output).map_err(|error| {
                        format!("strict PCU batched allocation diagnostic: {error}")
                    })?;

                    let _capture = AllocationCapture::start();
                    let output = strict_native.execute_two_batched()?;
                    native_batched_allocations[pair] = AllocationCapture::finish();
                    verify(&strict_expected, &output).map_err(|error| {
                        format!("strict native batched allocation diagnostic: {error}")
                    })?;
                } else {
                    let _capture = AllocationCapture::start();
                    let output = strict_native.execute_two_batched()?;
                    native_batched_allocations[pair] = AllocationCapture::finish();
                    verify(&strict_expected, &output).map_err(|error| {
                        format!("strict native batched allocation diagnostic: {error}")
                    })?;

                    let _capture = AllocationCapture::start();
                    let output = execute_strict_pcu_banked(true, None)?;
                    pcu_batched_allocations[pair] = AllocationCapture::finish();
                    verify(&strict_expected, &output).map_err(|error| {
                        format!("strict PCU batched allocation diagnostic: {error}")
                    })?;
                }
            }
            println!(
                "{rows}x{features} strict warm pointwise-batched alternating allocation diagnostics (16 executions each; per-execution median): PCU {} allocs, {} reallocs, {} deallocs, {} requested bytes; native {} allocs, {} reallocs, {} deallocs, {} requested bytes",
                median_allocation_field(&pcu_batched_allocations, |counts| counts.alloc_calls),
                median_allocation_field(&pcu_batched_allocations, |counts| counts.realloc_calls),
                median_allocation_field(&pcu_batched_allocations, |counts| counts.dealloc_calls),
                median_allocation_field(&pcu_batched_allocations, |counts| counts.requested_bytes),
                median_allocation_field(&native_batched_allocations, |counts| counts.alloc_calls),
                median_allocation_field(&native_batched_allocations, |counts| counts.realloc_calls),
                median_allocation_field(&native_batched_allocations, |counts| counts.dealloc_calls),
                median_allocation_field(&native_batched_allocations, |counts| counts
                    .requested_bytes),
            );
            let mut pcu_fully_batched_allocations = [AllocationCounts::default(); 16];
            let mut native_fully_batched_allocations = [AllocationCounts::default(); 16];
            for pair in 0_usize..16 {
                if pair.is_multiple_of(2) {
                    let _capture = AllocationCapture::start();
                    let output = execute_strict_pcu_banked(true, None)?;
                    pcu_fully_batched_allocations[pair] = AllocationCapture::finish();
                    verify(&strict_expected, &output).map_err(|error| {
                        format!("strict PCU fully batched allocation diagnostic: {error}")
                    })?;

                    let _capture = AllocationCapture::start();
                    let output = strict_native.execute_two_fully_batched()?;
                    native_fully_batched_allocations[pair] = AllocationCapture::finish();
                    verify(&strict_expected, &output).map_err(|error| {
                        format!("strict native fully batched allocation diagnostic: {error}")
                    })?;
                } else {
                    let _capture = AllocationCapture::start();
                    let output = strict_native.execute_two_fully_batched()?;
                    native_fully_batched_allocations[pair] = AllocationCapture::finish();
                    verify(&strict_expected, &output).map_err(|error| {
                        format!("strict native fully batched allocation diagnostic: {error}")
                    })?;

                    let _capture = AllocationCapture::start();
                    let output = execute_strict_pcu_banked(true, None)?;
                    pcu_fully_batched_allocations[pair] = AllocationCapture::finish();
                    verify(&strict_expected, &output).map_err(|error| {
                        format!("strict PCU fully batched allocation diagnostic: {error}")
                    })?;
                }
            }
            println!(
                "{rows}x{features} strict warm fully-batched alternating allocation diagnostics (16 executions each; per-execution median): PCU {} allocs, {} reallocs, {} deallocs, {} requested bytes; native {} allocs, {} reallocs, {} deallocs, {} requested bytes",
                median_allocation_field(&pcu_fully_batched_allocations, |counts| counts
                    .alloc_calls),
                median_allocation_field(&pcu_fully_batched_allocations, |counts| counts
                    .realloc_calls),
                median_allocation_field(&pcu_fully_batched_allocations, |counts| counts
                    .dealloc_calls),
                median_allocation_field(&pcu_fully_batched_allocations, |counts| counts
                    .requested_bytes),
                median_allocation_field(&native_fully_batched_allocations, |counts| counts
                    .alloc_calls),
                median_allocation_field(&native_fully_batched_allocations, |counts| counts
                    .realloc_calls),
                median_allocation_field(&native_fully_batched_allocations, |counts| counts
                    .dealloc_calls),
                median_allocation_field(&native_fully_batched_allocations, |counts| counts
                    .requested_bytes),
            );
            {
                let mut group = criterion.benchmark_group("tensor_train_step_strict");
                group.throughput(Throughput::Elements(u64::try_from(rows * features * 2)?));
                group.bench_function(
                    BenchmarkId::new("pcu_sgd_update_output_bank", format!("{rows}x{features}")),
                    |b| {
                        b.iter(|| {
                            black_box(
                                execute_strict_pcu_banked(false, None)
                                    .expect("strict PCU banked training failed"),
                            )
                        });
                    },
                );
                group.bench_function(
                    BenchmarkId::new(
                        "pcu_sgd_update_output_bank_batched",
                        format!("{rows}x{features}"),
                    ),
                    |b| {
                        b.iter(|| {
                            black_box(
                                execute_strict_pcu_banked(true, None)
                                    .expect("strict PCU batched banked training failed"),
                            )
                        });
                    },
                );
                group.bench_function(
                    BenchmarkId::new("native_hip_rocblas_strict", format!("{rows}x{features}")),
                    |b| {
                        b.iter(|| {
                            black_box(
                                strict_native
                                    .execute_two()
                                    .expect("strict native training failed"),
                            )
                        });
                    },
                );
                group.bench_function(
                    BenchmarkId::new(
                        "native_hip_rocblas_strict_batched",
                        format!("{rows}x{features}"),
                    ),
                    |b| {
                        b.iter(|| {
                            black_box(
                                strict_native
                                    .execute_two_batched()
                                    .expect("strict native batched training failed"),
                            )
                        });
                    },
                );
                group.bench_function(
                    BenchmarkId::new(
                        "native_hip_rocblas_strict_fully_batched",
                        format!("{rows}x{features}"),
                    ),
                    |b| {
                        b.iter(|| {
                            black_box(
                                strict_native
                                    .execute_two_fully_batched()
                                    .expect("strict native fully batched training failed"),
                            )
                        });
                    },
                );
                group.finish();
            }
            verify(&strict_expected, &execute_strict_pcu_banked(false, None)?)
                .map_err(|error| format!("strict PCU final: {error}"))?;
            verify(&strict_expected, &execute_strict_pcu_banked(true, None)?)
                .map_err(|error| format!("strict PCU batched final: {error}"))?;
            verify(&strict_expected, &strict_native.execute_two()?)
                .map_err(|error| format!("strict native final: {error}"))?;
            verify(&strict_expected, &strict_native.execute_two_batched()?)
                .map_err(|error| format!("strict native batched final: {error}"))?;
            verify(
                &strict_expected,
                &strict_native.execute_two_fully_batched()?,
            )
            .map_err(|error| format!("strict native fully batched final: {error}"))?;
            let mut strict_pair_pcu = Vec::with_capacity(16);
            let mut strict_pair_native = Vec::with_capacity(16);
            let mut strict_pair_ratio = Vec::with_capacity(16);
            for pair in 0_usize..16 {
                let (pcu_time, native_time) = if pair.is_multiple_of(2) {
                    let pcu_time = time_execution(|| execute_strict_pcu_banked(false, None))?;
                    let native_time = time_execution(|| strict_native.execute_two())?;
                    (pcu_time, native_time)
                } else {
                    let native_time = time_execution(|| strict_native.execute_two())?;
                    let pcu_time = time_execution(|| execute_strict_pcu_banked(false, None))?;
                    (pcu_time, native_time)
                };
                strict_pair_pcu.push(pcu_time);
                strict_pair_native.push(native_time);
                strict_pair_ratio.push(pcu_time / native_time);
            }
            println!(
                "{rows}x{features} strict order-alternating paired host-wall diagnostics (16 pairs): PCU median {:.3} us, native median {:.3} us, PCU/native median {:.3}x, range {:.3}–{:.3}x",
                median(&mut strict_pair_pcu) * 1_000_000.0,
                median(&mut strict_pair_native) * 1_000_000.0,
                median(&mut strict_pair_ratio),
                min(&strict_pair_ratio),
                max(&strict_pair_ratio),
            );
            let mut strict_sync = Vec::with_capacity(16);
            let mut strict_batched = Vec::with_capacity(16);
            let mut strict_batch_ratio = Vec::with_capacity(16);
            for pair in 0_usize..16 {
                let (sync_time, batch_time) = if pair.is_multiple_of(2) {
                    (
                        time_execution(|| execute_strict_pcu_banked(false, None))?,
                        time_execution(|| execute_strict_pcu_banked(true, None))?,
                    )
                } else {
                    let batch_time = time_execution(|| execute_strict_pcu_banked(true, None))?;
                    let sync_time = time_execution(|| execute_strict_pcu_banked(false, None))?;
                    (sync_time, batch_time)
                };
                strict_sync.push(sync_time);
                strict_batched.push(batch_time);
                strict_batch_ratio.push(batch_time / sync_time);
            }
            println!(
                "{rows}x{features} strict PCU output-bank batched/sync paired host-wall diagnostics (16 pairs): sync median {:.3} us, batched median {:.3} us, batched/sync median {:.3}x, range {:.3}–{:.3}x",
                median(&mut strict_sync) * 1_000_000.0,
                median(&mut strict_batched) * 1_000_000.0,
                median(&mut strict_batch_ratio),
                min(&strict_batch_ratio),
                max(&strict_batch_ratio),
            );
            let mut native_sync = Vec::with_capacity(16);
            let mut native_batched = Vec::with_capacity(16);
            let mut native_batch_ratio = Vec::with_capacity(16);
            let mut batched_pcu = Vec::with_capacity(16);
            let mut batched_native = Vec::with_capacity(16);
            let mut batched_pcu_native_ratio = Vec::with_capacity(16);
            for pair in 0_usize..16 {
                let (native_sync_time, native_batch_time, pcu_batch_time) = if pair
                    .is_multiple_of(2)
                {
                    (
                        time_execution(|| strict_native.execute_two())?,
                        time_execution(|| strict_native.execute_two_batched())?,
                        time_execution(|| execute_strict_pcu_banked(true, None))?,
                    )
                } else {
                    let pcu_batch_time = time_execution(|| execute_strict_pcu_banked(true, None))?;
                    let native_batch_time = time_execution(|| strict_native.execute_two_batched())?;
                    let native_sync_time = time_execution(|| strict_native.execute_two())?;
                    (native_sync_time, native_batch_time, pcu_batch_time)
                };
                native_sync.push(native_sync_time);
                native_batched.push(native_batch_time);
                native_batch_ratio.push(native_batch_time / native_sync_time);
                batched_pcu.push(pcu_batch_time);
                batched_native.push(native_batch_time);
                batched_pcu_native_ratio.push(pcu_batch_time / native_batch_time);
            }
            println!(
                "{rows}x{features} strict native batched/sync paired host-wall diagnostics (16 pairs): sync median {:.3} us, batched median {:.3} us, batched/sync median {:.3}x, range {:.3}–{:.3}x",
                median(&mut native_sync) * 1_000_000.0,
                median(&mut native_batched) * 1_000_000.0,
                median(&mut native_batch_ratio),
                min(&native_batch_ratio),
                max(&native_batch_ratio),
            );
            println!(
                "{rows}x{features} strict matched batched PCU/native paired host-wall diagnostics (16 pairs): PCU median {:.3} us, native median {:.3} us, PCU/native median {:.3}x, range {:.3}–{:.3}x",
                median(&mut batched_pcu) * 1_000_000.0,
                median(&mut batched_native) * 1_000_000.0,
                median(&mut batched_pcu_native_ratio),
                min(&batched_pcu_native_ratio),
                max(&batched_pcu_native_ratio),
            );
            if std::env::var_os("FUSION_ROCM_STRICT_DEVICE_PROFILE").is_some() {
                let mut pcu_segments: [Vec<f64>; 4] =
                    std::array::from_fn(|_| Vec::with_capacity(16));
                let mut native_segments: [Vec<f64>; 4] =
                    std::array::from_fn(|_| Vec::with_capacity(16));
                for pair in 0_usize..16 {
                    let mut pcu_times = Vec::with_capacity(4);
                    let (pcu_output, native_output, native_times) = if pair.is_multiple_of(2) {
                        let pcu_output = execute_strict_pcu_banked(true, Some(&mut pcu_times))?;
                        let (native_output, native_times) =
                            strict_native.execute_two_batched_device_timed()?;
                        (pcu_output, native_output, native_times)
                    } else {
                        let (native_output, native_times) =
                            strict_native.execute_two_batched_device_timed()?;
                        let pcu_output = execute_strict_pcu_banked(true, Some(&mut pcu_times))?;
                        (pcu_output, native_output, native_times)
                    };
                    verify(&strict_expected, &pcu_output)
                        .map_err(|error| format!("device-timed PCU output: {error}"))?;
                    verify(&strict_expected, &native_output)
                        .map_err(|error| format!("device-timed native output: {error}"))?;
                    if pcu_times.len() != 4 || native_times.len() != 4 {
                        return Err(format!(
                            "expected four device pointwise segments, got PCU {} and native {}",
                            pcu_times.len(),
                            native_times.len()
                        )
                        .into());
                    }
                    for segment in 0..4 {
                        pcu_segments[segment].push(f64::from(pcu_times[segment]) * 1000.0);
                        native_segments[segment].push(f64::from(native_times[segment]) * 1000.0);
                    }
                }
                for (segment, label) in [
                    "step1 delta+scale",
                    "step1 update",
                    "step2 delta+scale",
                    "step2 update",
                ]
                .into_iter()
                .enumerate()
                {
                    println!(
                        "{rows}x{features} strict batched device {label} (16 alternating pairs): PCU median {:.3} us, native median {:.3} us",
                        median(&mut pcu_segments[segment]),
                        median(&mut native_segments[segment]),
                    );
                }
            }
        }
        if std::env::var_os("FUSION_ROCM_STRICT_BATCH_HOST_PROFILE").is_some() {
            let mut profile_batched_pcu =
                || -> Result<(Vec<f32>, PcuPhaseTimings), Box<dyn Error>> {
                    let total_started = Instant::now();
                    let first_inputs = [
                        (strict.samples, &device_samples),
                        (strict.weights, &device_weights),
                        (strict.target, &device_target),
                    ];
                    let started = Instant::now();
                    assessor.execute_prepared_outputs_into_bank_batched(
                        &strict_prepared,
                        &first_inputs,
                        &mut strict_bank_scratch,
                        &mut strict_first_bank,
                        &mut strict_bank_memory,
                    )?;
                    let step_one = started.elapsed();
                    let first_weights = strict_first_bank
                        .outputs()
                        .first()
                        .ok_or("strict first output bank is empty")?;
                    let second_inputs = [
                        (strict.samples, &device_samples),
                        (strict.weights, first_weights),
                        (strict.target, &device_target),
                    ];
                    let started = Instant::now();
                    assessor.execute_prepared_outputs_into_bank_batched(
                        &strict_prepared,
                        &second_inputs,
                        &mut strict_bank_scratch,
                        &mut strict_second_bank,
                        &mut strict_bank_memory,
                    )?;
                    let step_two = started.elapsed();
                    let output = strict_second_bank
                        .outputs()
                        .first()
                        .ok_or("strict second output bank is empty")?;
                    let started = Instant::now();
                    let result =
                        assessor.download_output(output, selected.pool, &mut strict_bank_memory)?;
                    let output_readback = started.elapsed();
                    Ok((
                        result.into_data(),
                        PcuPhaseTimings {
                            step_one,
                            step_two,
                            output_readback,
                            total: total_started.elapsed(),
                        },
                    ))
                };
            let mut pcu_phases = Vec::with_capacity(16);
            let mut native_phases = Vec::with_capacity(16);
            for pair in 0_usize..16 {
                if pair.is_multiple_of(2) {
                    let (output, timings) = profile_batched_pcu()?;
                    verify(&strict_expected, &output)?;
                    pcu_phases.push(timings);
                    let (output, timings) = strict_native.execute_two_batched_host_profiled()?;
                    verify(&strict_expected, &output)?;
                    native_phases.push(timings);
                } else {
                    let (output, timings) = strict_native.execute_two_batched_host_profiled()?;
                    verify(&strict_expected, &output)?;
                    native_phases.push(timings);
                    let (output, timings) = profile_batched_pcu()?;
                    verify(&strict_expected, &output)?;
                    pcu_phases.push(timings);
                }
            }
            print_duration_summary(
                &format!("Strict batched PCU {rows}x{features} step one"),
                pcu_phases.iter().map(|timings| timings.step_one),
            );
            print_duration_summary(
                &format!("Strict batched PCU {rows}x{features} step two"),
                pcu_phases.iter().map(|timings| timings.step_two),
            );
            print_duration_summary(
                &format!("Strict batched PCU {rows}x{features} readback"),
                pcu_phases.iter().map(|timings| timings.output_readback),
            );
            print_duration_summary(
                &format!("Strict batched PCU {rows}x{features} total"),
                pcu_phases.iter().map(|timings| timings.total),
            );
            print_duration_summary(
                &format!("Strict batched native {rows}x{features} forward SGEMM"),
                native_phases.iter().map(|timings| timings.forward_sgemm),
            );
            print_duration_summary(
                &format!("Strict batched native {rows}x{features} gradient SGEMM"),
                native_phases.iter().map(|timings| timings.gradient_sgemm),
            );
            print_duration_summary(
                &format!("Strict batched native {rows}x{features} delta+scale"),
                native_phases
                    .iter()
                    .map(|timings| timings.batched_delta_scale),
            );
            print_duration_summary(
                &format!("Strict batched native {rows}x{features} delta+scale launch"),
                native_phases
                    .iter()
                    .map(|timings| timings.batched_delta_scale_launch),
            );
            print_duration_summary(
                &format!("Strict batched native {rows}x{features} delta+scale wait"),
                native_phases
                    .iter()
                    .map(|timings| timings.batched_delta_scale_wait),
            );
            print_duration_summary(
                &format!("Strict batched native {rows}x{features} update"),
                native_phases.iter().map(|timings| timings.batched_update),
            );
            print_duration_summary(
                &format!("Strict batched native {rows}x{features} update launch"),
                native_phases
                    .iter()
                    .map(|timings| timings.batched_update_launch),
            );
            print_duration_summary(
                &format!("Strict batched native {rows}x{features} update wait"),
                native_phases
                    .iter()
                    .map(|timings| timings.batched_update_wait),
            );
            print_duration_summary(
                &format!("Strict batched native {rows}x{features} readback"),
                native_phases.iter().map(|timings| timings.output_readback),
            );
            print_duration_summary(
                &format!("Strict batched native {rows}x{features} total"),
                native_phases.iter().map(|timings| timings.total),
            );
        }
        if std::env::var_os("FUSION_ROCM_STRICT_BATCH_NODE_PROFILE").is_some() {
            let mut node_keys = Vec::new();
            let mut node_samples: Vec<Vec<f64>> = Vec::new();
            let mut sgemm_samples: Vec<Vec<f64>> = Vec::new();
            let mut step_wall = Vec::with_capacity(32);
            let mut step_node_sum = Vec::with_capacity(32);
            let mut pcu_forward_host = Vec::with_capacity(16);
            let mut pcu_gradient_host = Vec::with_capacity(16);
            let mut native_batched_phases = Vec::with_capacity(16);
            for pair in 0_usize..16 {
                if !pair.is_multiple_of(2) {
                    let (output, timing) = strict_native.execute_two_batched_host_profiled()?;
                    verify(&strict_expected, &output)?;
                    native_batched_phases.push(timing);
                }
                let first_inputs = [
                    (strict.samples, &device_samples),
                    (strict.weights, &device_weights),
                    (strict.target, &device_target),
                ];
                let started = Instant::now();
                let first_timings = assessor.execute_prepared_outputs_into_bank_batched_profiled(
                    &strict_prepared,
                    &first_inputs,
                    &mut strict_bank_scratch,
                    &mut strict_first_bank,
                    &mut strict_bank_memory,
                )?;
                let first_wall = started.elapsed();
                let first_weights = strict_first_bank
                    .outputs()
                    .first()
                    .ok_or("strict first output bank is empty")?;
                let second_inputs = [
                    (strict.samples, &device_samples),
                    (strict.weights, first_weights),
                    (strict.target, &device_target),
                ];
                let started = Instant::now();
                let second_timings = assessor.execute_prepared_outputs_into_bank_batched_profiled(
                    &strict_prepared,
                    &second_inputs,
                    &mut strict_bank_scratch,
                    &mut strict_second_bank,
                    &mut strict_bank_memory,
                )?;
                let second_wall = started.elapsed();
                for (timings, wall) in
                    [(&first_timings, first_wall), (&second_timings, second_wall)]
                {
                    if node_keys.is_empty() {
                        node_keys.extend(
                            timings
                                .iter()
                                .map(|timing| (timing.operation, timing.value)),
                        );
                        node_samples.resize_with(timings.len(), Vec::new);
                        sgemm_samples.resize_with(timings.len(), Vec::new);
                    } else if node_keys.len() != timings.len()
                        || node_keys
                            .iter()
                            .zip(timings)
                            .any(|(key, timing)| *key != (timing.operation, timing.value))
                    {
                        return Err("strict batched profiled node sequence changed".into());
                    }
                    step_wall.push(wall.as_secs_f64());
                    step_node_sum.push(
                        timings
                            .iter()
                            .map(|timing| timing.elapsed.as_secs_f64())
                            .sum(),
                    );
                    for ((node, sgemm), timing) in node_samples
                        .iter_mut()
                        .zip(sgemm_samples.iter_mut())
                        .zip(timings)
                    {
                        node.push(timing.elapsed.as_secs_f64());
                        if let Some(host) = timing.sgemm_host {
                            sgemm.push(
                                (host.preflight
                                    + host.rocblas_call
                                    + host.device_synchronize
                                    + host.cleanup)
                                    .as_secs_f64(),
                            );
                        }
                    }
                }
                let final_weights = strict_second_bank
                    .outputs()
                    .first()
                    .ok_or("strict second output bank is empty")?;
                let output = assessor.download_output(
                    final_weights,
                    selected.pool,
                    &mut strict_bank_memory,
                )?;
                verify(&strict_expected, output.data())?;
                let mut forward_host = fusion_pcu_rocm::RocblasSgemmHostTiming::default();
                let mut gradient_host = fusion_pcu_rocm::RocblasSgemmHostTiming::default();
                let mut matmul_index = 0_usize;
                for timing in first_timings.iter().chain(&second_timings) {
                    if timing.operation == "MatMul" {
                        let host = timing
                            .sgemm_host
                            .ok_or("batched MatMul has no rocBLAS phase timing")?;
                        add_sgemm_host_timing(
                            if matmul_index.is_multiple_of(2) {
                                &mut forward_host
                            } else {
                                &mut gradient_host
                            },
                            host,
                        );
                        matmul_index += 1;
                    }
                }
                if matmul_index != 4 {
                    return Err(
                        format!("expected four batched MatMul calls, got {matmul_index}").into(),
                    );
                }
                pcu_forward_host.push(forward_host);
                pcu_gradient_host.push(gradient_host);
                if pair.is_multiple_of(2) {
                    let (output, timing) = strict_native.execute_two_batched_host_profiled()?;
                    verify(&strict_expected, &output)?;
                    native_batched_phases.push(timing);
                }
            }
            println!("Strict batched PCU {rows}x{features} node host wall (32 steps):");
            for (ordinal, (((operation, value), samples), sgemm)) in node_keys
                .iter()
                .zip(&mut node_samples)
                .zip(&mut sgemm_samples)
                .enumerate()
            {
                println!(
                    "  node {ordinal} {operation} value {value:?}: median {:.3} us",
                    median(samples) * 1_000_000.0,
                );
                if !sgemm.is_empty() {
                    println!(
                        "    rocBLAS measured host phases median {:.3} us",
                        median(sgemm) * 1_000_000.0,
                    );
                }
            }
            println!(
                "Strict batched PCU {rows}x{features} step/node-sum host-wall medians: {:.3}/{:.3} us",
                median(&mut step_wall) * 1_000_000.0,
                median(&mut step_node_sum) * 1_000_000.0,
            );
            print_pcu_sgemm_host_pair_phases(
                &format!("Strict batched PCU {rows}x{features} forward SGEMM"),
                &pcu_forward_host,
            );
            print_sgemm_host_phases(
                &format!("Strict batched native {rows}x{features} forward SGEMM"),
                &native_batched_phases,
                |timing| timing.forward_sgemm_host,
            );
            print_pcu_sgemm_host_pair_phases(
                &format!("Strict batched PCU {rows}x{features} gradient SGEMM"),
                &pcu_gradient_host,
            );
            print_sgemm_host_phases(
                &format!("Strict batched native {rows}x{features} gradient SGEMM"),
                &native_batched_phases,
                |timing| timing.gradient_sgemm_host,
            );
        }
        let mut strict_profile = ProfiledMemory::new(
            PcuOwnedDispatchMemorySession::memory_provider(&session, selected.pool),
        );
        let mut profile_pcu = || -> Result<(Vec<f32>, PcuPhaseTimings), Box<dyn Error>> {
            let total_started = Instant::now();
            let first_inputs = [
                (strict.samples, &device_samples),
                (strict.weights, &device_weights),
                (strict.target, &device_target),
            ];
            let started = Instant::now();
            assessor.execute_prepared_outputs_into_bank(
                &strict_prepared,
                &first_inputs,
                &mut strict_bank_scratch,
                &mut strict_first_bank,
                &mut strict_profile,
            )?;
            let step_one = started.elapsed();
            let first_weights = strict_first_bank
                .outputs()
                .first()
                .ok_or("strict first output bank is empty")?;
            let second_inputs = [
                (strict.samples, &device_samples),
                (strict.weights, first_weights),
                (strict.target, &device_target),
            ];
            let started = Instant::now();
            assessor.execute_prepared_outputs_into_bank(
                &strict_prepared,
                &second_inputs,
                &mut strict_bank_scratch,
                &mut strict_second_bank,
                &mut strict_profile,
            )?;
            let step_two = started.elapsed();
            let output = strict_second_bank
                .outputs()
                .first()
                .ok_or("strict second output bank is empty")?;
            let started = Instant::now();
            let result = assessor.download_output(output, selected.pool, &mut strict_profile)?;
            let output_readback = started.elapsed();
            Ok((
                result.into_data(),
                PcuPhaseTimings {
                    step_one,
                    step_two,
                    output_readback,
                    total: total_started.elapsed(),
                },
            ))
        };
        let mut pcu_phases = Vec::with_capacity(16);
        let mut native_phases = Vec::with_capacity(16);
        for pair in 0_usize..16 {
            if pair.is_multiple_of(2) {
                let (output, timings) = profile_pcu()?;
                verify(&strict_expected, &output)?;
                pcu_phases.push(timings);
                let (output, timings) = strict_native.execute_two_profiled()?;
                verify(&strict_expected, &output)?;
                native_phases.push(timings);
            } else {
                let (output, timings) = strict_native.execute_two_profiled()?;
                verify(&strict_expected, &output)?;
                native_phases.push(timings);
                let (output, timings) = profile_pcu()?;
                verify(&strict_expected, &output)?;
                pcu_phases.push(timings);
            }
        }
        print_duration_summary(
            &format!("Strict PCU {rows}x{features} step one"),
            pcu_phases.iter().map(|timings| timings.step_one),
        );
        print_duration_summary(
            &format!("Strict PCU {rows}x{features} step two"),
            pcu_phases.iter().map(|timings| timings.step_two),
        );
        print_duration_summary(
            &format!("Strict PCU {rows}x{features} output readback"),
            pcu_phases.iter().map(|timings| timings.output_readback),
        );
        print_duration_summary(
            &format!("Strict PCU {rows}x{features} total"),
            pcu_phases.iter().map(|timings| timings.total),
        );
        print_duration_summary(
            &format!("Strict native {rows}x{features} rocBLAS"),
            native_phases.iter().map(|timings| timings.rocblas),
        );
        print_duration_summary(
            &format!("Strict native {rows}x{features} forward SGEMM"),
            native_phases.iter().map(|timings| timings.forward_sgemm),
        );
        print_duration_summary(
            &format!("Strict native {rows}x{features} gradient SGEMM"),
            native_phases.iter().map(|timings| timings.gradient_sgemm),
        );
        print_sgemm_host_phases(
            &format!("Strict native {rows}x{features} forward SGEMM"),
            &native_phases,
            |timings| timings.forward_sgemm_host,
        );
        print_sgemm_host_phases(
            &format!("Strict native {rows}x{features} gradient SGEMM"),
            &native_phases,
            |timings| timings.gradient_sgemm_host,
        );
        print_duration_summary(
            &format!("Strict native {rows}x{features} delta HIP"),
            native_phases.iter().map(|timings| timings.delta_hip),
        );
        print_duration_summary(
            &format!("Strict native {rows}x{features} scale HIP"),
            native_phases.iter().map(|timings| timings.scale_hip),
        );
        print_duration_summary(
            &format!("Strict native {rows}x{features} update HIP"),
            native_phases.iter().map(|timings| timings.update_hip),
        );
        print_duration_summary(
            &format!("Strict native {rows}x{features} HIP launch+wait"),
            native_phases.iter().map(|timings| timings.hip_kernels),
        );
        print_duration_summary(
            &format!("Strict native {rows}x{features} output readback"),
            native_phases.iter().map(|timings| timings.output_readback),
        );
        print_duration_summary(
            &format!("Strict native {rows}x{features} total"),
            native_phases.iter().map(|timings| timings.total),
        );
        let profile = strict_profile.profile();
        println!(
            "Strict PCU {rows}x{features} warm output-bank provider across 16 profiled pairs: {} allocations ({} bytes, {:?}), {} uploads ({} bytes, {:?}), {} downloads ({} bytes, {:?})",
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
        if std::env::var_os("FUSION_ROCM_STRICT_NODE_PROFILE").is_some() {
            let mut node_samples: Vec<Vec<f64>> = Vec::new();
            let mut sgemm_samples = Vec::new();
            let mut elementwise_samples = Vec::new();
            let mut pair_node_samples: Vec<Vec<std::time::Duration>> = Vec::new();
            let mut pcu_forward_sgemm_host = Vec::with_capacity(16);
            let mut pcu_gradient_sgemm_host = Vec::with_capacity(16);
            let mut native_phase_samples = Vec::with_capacity(16);
            let mut step_wall_samples = Vec::with_capacity(32);
            let mut node_sum_samples = Vec::with_capacity(32);
            let mut non_node_samples = Vec::with_capacity(32);
            let mut node_keys = Vec::new();
            let profile_native = || -> Result<native::NativeTrainTimings, Box<dyn Error>> {
                let (output, timings) = strict_native.execute_two_profiled()?;
                verify(&strict_expected, &output)
                    .map_err(|error| format!("strict profiled native pass: {error}"))?;
                Ok(timings)
            };
            for pass in 0_usize..16 {
                if !pass.is_multiple_of(2) {
                    native_phase_samples.push(profile_native()?);
                }
                let mut pair_node_totals = Vec::new();
                let mut matmul_index = 0;
                let mut pair_forward_host = fusion_pcu_rocm::RocblasSgemmHostTiming::default();
                let mut pair_gradient_host = fusion_pcu_rocm::RocblasSgemmHostTiming::default();
                let first_inputs = [
                    (strict.samples, &device_samples),
                    (strict.weights, &device_weights),
                    (strict.target, &device_target),
                ];
                let step_started = Instant::now();
                let first_timings = assessor.execute_prepared_outputs_into_bank_profiled(
                    &strict_prepared,
                    &first_inputs,
                    &mut strict_bank_scratch,
                    &mut strict_first_bank,
                    &mut strict_bank_memory,
                )?;
                let step_wall = step_started.elapsed();
                record_node_timings!(
                    &first_timings,
                    &mut node_keys,
                    &mut node_samples,
                    &mut sgemm_samples,
                    &mut elementwise_samples,
                    &mut step_wall_samples,
                    &mut node_sum_samples,
                    &mut non_node_samples,
                    step_wall,
                );
                pair_node_totals.resize(node_keys.len(), std::time::Duration::ZERO);
                accumulate_pair_node_timings(
                    &first_timings,
                    &node_keys,
                    &mut pair_node_totals,
                    &mut matmul_index,
                    &mut pair_forward_host,
                    &mut pair_gradient_host,
                );

                // Keep step two's output in the other bank so its input remains live throughout
                // execution; writing into the bank that supplies weights would alias the graph.
                let first_weights = strict_first_bank
                    .outputs()
                    .first()
                    .ok_or("strict first output bank is empty")?;
                let second_inputs = [
                    (strict.samples, &device_samples),
                    (strict.weights, first_weights),
                    (strict.target, &device_target),
                ];
                let step_started = Instant::now();
                let second_timings = assessor.execute_prepared_outputs_into_bank_profiled(
                    &strict_prepared,
                    &second_inputs,
                    &mut strict_bank_scratch,
                    &mut strict_second_bank,
                    &mut strict_bank_memory,
                )?;
                let step_wall = step_started.elapsed();
                record_node_timings!(
                    &second_timings,
                    &mut node_keys,
                    &mut node_samples,
                    &mut sgemm_samples,
                    &mut elementwise_samples,
                    &mut step_wall_samples,
                    &mut node_sum_samples,
                    &mut non_node_samples,
                    step_wall,
                );
                accumulate_pair_node_timings(
                    &second_timings,
                    &node_keys,
                    &mut pair_node_totals,
                    &mut matmul_index,
                    &mut pair_forward_host,
                    &mut pair_gradient_host,
                );
                if pair_node_samples.is_empty() {
                    pair_node_samples.resize_with(node_keys.len(), Vec::new);
                }
                for (samples, duration) in pair_node_samples.iter_mut().zip(pair_node_totals) {
                    samples.push(duration);
                }
                pcu_forward_sgemm_host.push(pair_forward_host);
                pcu_gradient_sgemm_host.push(pair_gradient_host);

                let final_weights = strict_second_bank
                    .outputs()
                    .first()
                    .ok_or("strict second output bank is empty")?;
                let result = assessor.download_output(
                    final_weights,
                    selected.pool,
                    &mut strict_bank_memory,
                )?;
                verify(&strict_expected, result.data())
                    .map_err(|error| format!("strict profiled PCU pass: {error}"))?;
                if pass.is_multiple_of(2) {
                    native_phase_samples.push(profile_native()?);
                }
            }
            println!(
                "Strict PCU {rows}x{features} per-node diagnostic (16 two-step passes; host wall time, profiler enabled; timings include synchronous completion and profiling perturbs short kernels):"
            );
            for (ordinal, ((operation, value), samples)) in
                node_keys.iter().zip(&mut node_samples).enumerate()
            {
                println!(
                    "  node {ordinal} {operation} value {:?}: median {:.3} us, range {:.3}–{:.3} us",
                    value,
                    median(samples) * 1_000_000.0,
                    min(samples) * 1_000_000.0,
                    max(samples) * 1_000_000.0,
                );
                if !sgemm_samples[ordinal].is_empty() {
                    print_pcu_sgemm_host_phases(
                        &format!("Strict PCU {rows}x{features} node {ordinal} MatMul"),
                        &sgemm_samples[ordinal],
                    );
                }
                if !elementwise_samples[ordinal].is_empty() {
                    print_elementwise_host_phases(
                        &format!("Strict PCU {rows}x{features} node {ordinal} {operation}"),
                        &elementwise_samples[ordinal],
                    );
                }
            }
            for (ordinal, ((operation, value), samples)) in
                node_keys.iter().zip(&mut pair_node_samples).enumerate()
            {
                if matches!(*operation, "Sub" | "Mul" | "SgdUpdate") {
                    print_duration_summary_with_context(
                        &format!(
                            "Strict PCU {rows}x{features} two-step node {ordinal} {operation} value {value:?}"
                        ),
                        samples.iter().copied(),
                        "16 alternating pairs",
                    );
                }
            }
            print_pcu_sgemm_host_pair_phases(
                &format!("Strict PCU {rows}x{features} two-step forward SGEMM"),
                &pcu_forward_sgemm_host,
            );
            print_pcu_sgemm_host_pair_phases(
                &format!("Strict PCU {rows}x{features} two-step gradient SGEMM"),
                &pcu_gradient_sgemm_host,
            );
            print_duration_summary(
                &format!("Strict native {rows}x{features} two-step delta HIP"),
                native_phase_samples.iter().map(|timings| timings.delta_hip),
            );
            print_duration_summary(
                &format!("Strict native {rows}x{features} two-step scale HIP"),
                native_phase_samples.iter().map(|timings| timings.scale_hip),
            );
            print_duration_summary(
                &format!("Strict native {rows}x{features} two-step update HIP"),
                native_phase_samples
                    .iter()
                    .map(|timings| timings.update_hip),
            );
            print_native_hip_phases(
                &format!("Strict native {rows}x{features} two-step delta HIP"),
                &native_phase_samples,
                |timings| (timings.delta_launch, timings.delta_wait),
            );
            print_native_hip_phases(
                &format!("Strict native {rows}x{features} two-step scale HIP"),
                &native_phase_samples,
                |timings| (timings.scale_launch, timings.scale_wait),
            );
            print_native_hip_phases(
                &format!("Strict native {rows}x{features} two-step update HIP"),
                &native_phase_samples,
                |timings| (timings.update_launch, timings.update_wait),
            );
            print_sgemm_host_phases(
                &format!("Strict native {rows}x{features} two-step forward SGEMM"),
                &native_phase_samples,
                |timings| timings.forward_sgemm_host,
            );
            print_sgemm_host_phases(
                &format!("Strict native {rows}x{features} two-step gradient SGEMM"),
                &native_phase_samples,
                |timings| timings.gradient_sgemm_host,
            );
            print_duration_summary_with_context(
                &format!("Strict PCU {rows}x{features} profiled whole-step host wall"),
                step_wall_samples.iter().copied(),
                "32 profiled steps",
            );
            print_duration_summary_with_context(
                &format!("Strict PCU {rows}x{features} profiled sum of node timings"),
                node_sum_samples.iter().copied(),
                "32 profiled steps",
            );
            print_duration_summary_with_context(
                &format!(
                    "Strict PCU {rows}x{features} estimated non-node host overhead (includes profiler collection itself)"
                ),
                non_node_samples.iter().copied(),
                "32 profiled steps",
            );
        }
    }
    Ok(())
}

fn time_execution(
    execute: impl FnOnce() -> Result<Vec<f32>, Box<dyn Error>>,
) -> Result<f64, Box<dyn Error>> {
    let started = Instant::now();
    drop(black_box(execute()?));
    Ok(started.elapsed().as_secs_f64())
}

fn print_duration_summary(label: &str, durations: impl Iterator<Item = std::time::Duration>) {
    print_duration_summary_with_context(label, durations, "16 alternating pairs");
}

fn print_duration_summary_with_context(
    label: &str,
    durations: impl Iterator<Item = std::time::Duration>,
    context: &str,
) {
    let mut seconds = durations
        .map(|duration| duration.as_secs_f64())
        .collect::<Vec<_>>();
    let median_seconds = median(&mut seconds);
    println!(
        "{label} ({context}): median {:.3} us, range {:.3}–{:.3} us",
        median_seconds * 1_000_000.0,
        min(&seconds) * 1_000_000.0,
        max(&seconds) * 1_000_000.0,
    );
}

fn print_sgemm_host_phases(
    label: &str,
    samples: &[native::NativeTrainTimings],
    select: impl Fn(&native::NativeTrainTimings) -> fusion_pcu_rocm::RocblasSgemmHostTiming,
) {
    print_duration_summary(
        &format!("{label} preflight"),
        samples.iter().map(|sample| select(sample).preflight),
    );
    print_duration_summary(
        &format!("{label} rocBLAS C call"),
        samples.iter().map(|sample| select(sample).rocblas_call),
    );
    print_duration_summary(
        &format!("{label} device synchronize"),
        samples
            .iter()
            .map(|sample| select(sample).device_synchronize),
    );
    print_duration_summary(
        &format!("{label} cleanup"),
        samples.iter().map(|sample| select(sample).cleanup),
    );
}

fn print_pcu_sgemm_host_phases(label: &str, samples: &[fusion_pcu_rocm::RocblasSgemmHostTiming]) {
    print_duration_summary_with_context(
        &format!("{label} preflight"),
        samples.iter().map(|sample| sample.preflight),
        "32 profiled calls",
    );
    print_duration_summary_with_context(
        &format!("{label} rocBLAS C call"),
        samples.iter().map(|sample| sample.rocblas_call),
        "32 profiled calls",
    );
    print_duration_summary_with_context(
        &format!("{label} device synchronize"),
        samples.iter().map(|sample| sample.device_synchronize),
        "32 profiled calls",
    );
    print_duration_summary_with_context(
        &format!("{label} cleanup"),
        samples.iter().map(|sample| sample.cleanup),
        "32 profiled calls",
    );
}

fn accumulate_pair_node_timings(
    timings: &[fusion_pcu_rocm::RocmTensorNodeTiming],
    node_keys: &[(&'static str, ValueId)],
    node_totals: &mut [std::time::Duration],
    matmul_index: &mut usize,
    forward_host: &mut fusion_pcu_rocm::RocblasSgemmHostTiming,
    gradient_host: &mut fusion_pcu_rocm::RocblasSgemmHostTiming,
) {
    for timing in timings {
        if let Some((ordinal, _)) = node_keys
            .iter()
            .enumerate()
            .find(|(_, key)| **key == (timing.operation, timing.value))
        {
            node_totals[ordinal] += timing.elapsed;
        }
        if timing.operation == "MatMul" {
            if let Some(host) = timing.sgemm_host {
                add_sgemm_host_timing(
                    if matmul_index.is_multiple_of(2) {
                        forward_host
                    } else {
                        gradient_host
                    },
                    host,
                );
            }
            *matmul_index += 1;
        }
    }
}

fn add_sgemm_host_timing(
    total: &mut fusion_pcu_rocm::RocblasSgemmHostTiming,
    sample: fusion_pcu_rocm::RocblasSgemmHostTiming,
) {
    total.preflight += sample.preflight;
    total.rocblas_call += sample.rocblas_call;
    total.device_synchronize += sample.device_synchronize;
    total.cleanup += sample.cleanup;
}

fn print_pcu_sgemm_host_pair_phases(
    label: &str,
    samples: &[fusion_pcu_rocm::RocblasSgemmHostTiming],
) {
    let context = "16 alternating pairs (two steps)";
    print_duration_summary_with_context(
        &format!("{label} preflight"),
        samples.iter().map(|sample| sample.preflight),
        context,
    );
    print_duration_summary_with_context(
        &format!("{label} rocBLAS C call"),
        samples.iter().map(|sample| sample.rocblas_call),
        context,
    );
    print_duration_summary_with_context(
        &format!("{label} device synchronize"),
        samples.iter().map(|sample| sample.device_synchronize),
        context,
    );
    print_duration_summary_with_context(
        &format!("{label} cleanup"),
        samples.iter().map(|sample| sample.cleanup),
        context,
    );
}

fn print_elementwise_host_phases(
    label: &str,
    samples: &[fusion_pcu_rocm::RocmTensorElementwiseHostTiming],
) {
    print_duration_summary_with_context(
        &format!("{label} cache and bind"),
        samples.iter().map(|sample| sample.cache_and_bind),
        "32 profiled calls",
    );
    print_duration_summary_with_context(
        &format!("{label} submit"),
        samples.iter().map(|sample| sample.submit),
        "32 profiled calls",
    );
    print_duration_summary_with_context(
        &format!("{label} wait"),
        samples.iter().map(|sample| sample.wait),
        "32 profiled calls",
    );
}

fn print_native_hip_phases(
    label: &str,
    samples: &[native::NativeTrainTimings],
    select: impl Fn(&native::NativeTrainTimings) -> (std::time::Duration, std::time::Duration),
) {
    print_duration_summary(
        &format!("{label} launch return"),
        samples.iter().map(|sample| select(sample).0),
    );
    print_duration_summary(
        &format!("{label} completion wait"),
        samples.iter().map(|sample| select(sample).1),
    );
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    if values.len().is_multiple_of(2) {
        f64::midpoint(values[middle - 1], values[middle])
    } else {
        values[middle]
    }
}

fn min(values: &[f64]) -> f64 {
    values.iter().copied().fold(f64::INFINITY, f64::min)
}

fn max(values: &[f64]) -> f64 {
    values.iter().copied().fold(0.0_f64, f64::max)
}

fn median_allocation_field(
    samples: &[AllocationCounts; 16],
    field: fn(AllocationCounts) -> usize,
) -> usize {
    let mut values = [0; 16];
    let mut index = 0;
    while index < samples.len() {
        values[index] = field(samples[index]);
        index += 1;
    }
    values.sort_unstable();
    usize::midpoint(values[7], values[8])
}

fn small_integer(value: usize) -> f32 {
    f32::from(u8::try_from(value).expect("bounded benchmark pattern fits in u8"))
}

criterion_group! { name = benches; config = support::criterion_config(); targets = bench }
criterion_main!(benches);
