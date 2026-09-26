//! Resident PCU training benchmark for a three-layer f32 MLP.

mod support;

use std::{
    cell::RefCell,
    error::Error,
    hint::black_box,
    time::Instant,
};

use criterion::{
    criterion_group,
    criterion_main,
    BenchmarkId,
    Criterion,
    Throughput,
};
use fusion_pcu::{
    PcuMemoryProvider,
    PcuMemoryUsage,
    PcuOwnedDispatchMemorySession,
};
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
use support::selection;
#[path = "support/memory_profile.rs"]
mod memory_profile;
use memory_profile::ProfiledMemory;

#[path = "support/mlp_train.rs"]
mod native;

const DEFAULT_BATCH: usize = 256;
const INPUT: usize = 1024;
const HIDDEN: usize = 2048;
const OUTPUT: usize = 1024;
const RATE: f32 = 1.0e-6;

struct Program {
    graph: Graph,
    samples: ValueId,
    targets: ValueId,
    weights: [ValueId; 3],
    updated: [ValueId; 3],
    loss: ValueId,
    backward_start: usize,
    update_start: usize,
}

fn program(batch: usize) -> Result<Program, Box<dyn Error>> {
    let mut graph = Graph::default();
    let samples = graph.input([batch, INPUT])?;
    let w1 = graph.input([INPUT, HIDDEN])?;
    let w2 = graph.input([HIDDEN, HIDDEN])?;
    let w3 = graph.input([HIDDEN, OUTPUT])?;
    let targets = graph.input([batch, OUTPUT])?;
    let z1 = graph.matmul(samples, w1)?;
    let h1 = graph.relu(z1)?;
    let z2 = graph.matmul(h1, w2)?;
    let h2 = graph.relu(z2)?;
    let prediction = graph.matmul(h2, w3)?;
    let loss = graph.mean_squared_error(prediction, targets)?;
    let backward_start = graph.nodes().count();
    let gradients = graph.backward_mse(loss)?;
    let update_start = graph.nodes().count();
    let weights = [w1, w2, w3];
    let mut updated = Vec::with_capacity(3);
    for weight in weights {
        let index = graph
            .nodes()
            .position(|node| node.value == weight)
            .ok_or("weight input missing")?;
        let gradient = gradients[index].ok_or("weight gradient missing")?;
        updated.push(graph.sgd_update(weight, gradient, RATE)?);
    }
    Ok(Program {
        graph,
        samples,
        targets,
        weights,
        updated: updated.try_into().map_err(|_| "expected three updates")?,
        loss,
        backward_start,
        update_start,
    })
}

fn values(count: usize, modulus: usize, scale: f32, shift: i32) -> Vec<f32> {
    let shift = f32::from(i16::try_from(shift).expect("small data pattern shift fits in i16"));
    (0..count)
        .map(|i| {
            let value = i.wrapping_mul(17).wrapping_add(11) % modulus;
            f32::from(u16::try_from(value).expect("data pattern fits in u16")) - shift
        })
        .map(|value| value * scale)
        .collect()
}

fn bench(c: &mut Criterion) {
    run(c).expect("ROCm MLP training benchmark failed");
}

fn configured_batch() -> Result<usize, Box<dyn Error>> {
    let Some(raw) = std::env::var_os("FUSION_PCU_MLP_BATCH") else {
        return Ok(DEFAULT_BATCH);
    };
    let value = raw.to_string_lossy();
    let batch = value.parse::<usize>()?;
    if batch == 0 {
        return Err("FUSION_PCU_MLP_BATCH must be greater than zero".into());
    }
    Ok(batch)
}

fn run(c: &mut Criterion) -> Result<(), Box<dyn Error>> {
    let discovery = RocmDiscovery::new();
    let mut failures = Vec::new();
    for candidate in support::selected_candidates(&discovery)? {
        match run_on(c, &discovery, &candidate) {
            Ok(()) => return Ok(()),
            Err(error) => failures.push(format!("{}: {error}", candidate.name)),
        }
    }
    Err(format!(
        "no ROCm device completed MLP benchmark: {}",
        failures.join("; ")
    )
    .into())
}

#[allow(
    clippy::too_many_lines,
    clippy::explicit_auto_deref,
    clippy::explicit_deref_methods
)]
fn run_on(
    c: &mut Criterion,
    discovery: &RocmDiscovery,
    selected: &selection::Candidate,
) -> Result<(), Box<dyn Error>> {
    let batch = configured_batch()?;
    let session = RocmOwnedDispatchBackend::open(discovery, selected.device, 128)?;
    let assessor = RocmTensorAssessor::new(&session)?;
    println!(
        "Device: {}; {INPUT}->{HIDDEN}->{HIDDEN}->{OUTPUT}, batch {batch}",
        selected.name
    );
    let program = program(batch)?;
    let samples = values(batch * INPUT, 31, 1.0 / 64.0, 15);
    let targets = values(batch * OUTPUT, 19, 1.0 / 32.0, 9);
    let w1 = values(INPUT * HIDDEN, 23, 1.0 / 256.0, 11);
    let w2 = values(HIDDEN * HIDDEN, 23, 1.0 / 256.0, 11);
    let w3 = values(HIDDEN * OUTPUT, 23, 1.0 / 256.0, 11);
    let host_samples = Tensor::new([batch, INPUT], samples.clone())?;
    let host_targets = Tensor::new([batch, OUTPUT], targets.clone())?;
    let host_weights = [
        Tensor::new([INPUT, HIDDEN], w1.clone())?,
        Tensor::new([HIDDEN, HIDDEN], w2.clone())?,
        Tensor::new([HIDDEN, OUTPUT], w3.clone())?,
    ];
    let outputs = [
        program.updated[0],
        program.updated[1],
        program.updated[2],
        program.loss,
    ];
    let prepared = support::cold_once("PCU MLP graph preparation", || {
        assessor.prepare_graph_outputs(&program.graph, &outputs)
    })?;
    let feedback = prepared.tensor_plan().feedback_plan(&[
        (program.updated[0], program.weights[0]),
        (program.updated[1], program.weights[1]),
        (program.updated[2], program.weights[2]),
    ])?;
    let mut memory = PcuOwnedDispatchMemorySession::memory_provider(&session, selected.pool);
    let device_samples = assessor.upload_input(&host_samples, selected.pool, &mut memory)?;
    let device_targets = assessor.upload_input(&host_targets, selected.pool, &mut memory)?;
    let device_weights = host_weights
        .each_ref()
        .map(|w| assessor.upload_input(w, selected.pool, &mut memory))
        .into_iter()
        .collect::<Result<Vec<_>, _>>()?;
    let scratch = support::cold_once("PCU MLP scratch preparation", || {
        assessor.prepare_scratch(&prepared, selected.pool, &mut memory)
    })?;
    let feedback_resources =
        assessor.prepare_feedback_resources(&prepared, selected.pool, &mut memory)?;
    let native_inputs = native::TrainInputs {
        batch,
        samples: &samples,
        targets: &targets,
        initial_w1: &w1,
        initial_w2: &w2,
        initial_w3: &w3,
        learning_rate: RATE,
    };
    let mut native = support::cold_once("native HIP/rocBLAS MLP preparation", || {
        native::NativeMlpTrain::prepare(discovery, selected.device, &native_inputs)
    })?;
    let scratch = RefCell::new(scratch);
    let memory = RefCell::new(memory);
    let feedback_resources = RefCell::new(feedback_resources);

    let run_pcu = || -> Result<native::MlpWeights, Box<dyn Error>> {
        let mut scratch = scratch.borrow_mut();
        let mut memory = memory.borrow_mut();
        let mut current = Vec::new();
        let mut loss_resources = Vec::with_capacity(2);
        for step in 0..2 {
            let weights = if step == 0 { &device_weights } else { &current };
            let inputs = vec![
                (program.samples, &device_samples),
                (program.weights[0], &weights[0]),
                (program.weights[1], &weights[1]),
                (program.weights[2], &weights[2]),
                (program.targets, &device_targets),
            ];
            let mut next = assessor.execute_prepared_outputs_with_resources_and_scratch_resident(
                &prepared,
                &inputs,
                &mut *scratch,
                &mut *memory,
            )?;
            loss_resources.push(next.pop().ok_or("missing loss output")?);
            current = next;
        }
        let losses = [
            assessor
                .download_output(&loss_resources[0], selected.pool, &mut *memory)?
                .data()[0],
            assessor
                .download_output(&loss_resources[1], selected.pool, &mut *memory)?
                .data()[0],
        ];
        Ok(native::MlpWeights {
            w1: assessor
                .download_output(&current[0], selected.pool, &mut *memory)?
                .into_data(),
            w2: assessor
                .download_output(&current[1], selected.pool, &mut *memory)?
                .into_data(),
            w3: assessor
                .download_output(&current[2], selected.pool, &mut *memory)?
                .into_data(),
            losses,
        })
    };
    let run_pcu_output_bank = || -> Result<native::MlpWeights, Box<dyn Error>> {
        let mut memory = memory.borrow_mut();
        let mut resources = feedback_resources.borrow_mut();
        let initial_inputs = [
            (program.samples, &device_samples),
            (program.weights[0], &device_weights[0]),
            (program.weights[1], &device_weights[1]),
            (program.weights[2], &device_weights[2]),
            (program.targets, &device_targets),
        ];
        assessor.execute_feedback_steps(
            &feedback,
            &initial_inputs,
            &mut *resources,
            std::num::NonZeroUsize::new(2).expect("two feedback steps"),
            &mut *memory,
        )?;
        let first_outputs = resources.outputs_for_step(0)?;
        let second_outputs = resources.outputs_for_step(1)?;
        let losses = [
            assessor
                .download_output(&first_outputs[3], selected.pool, &mut *memory)?
                .data()[0],
            assessor
                .download_output(&second_outputs[3], selected.pool, &mut *memory)?
                .data()[0],
        ];
        Ok(native::MlpWeights {
            w1: assessor
                .download_output(&second_outputs[0], selected.pool, &mut *memory)?
                .into_data(),
            w2: assessor
                .download_output(&second_outputs[1], selected.pool, &mut *memory)?
                .into_data(),
            w3: assessor
                .download_output(&second_outputs[2], selected.pool, &mut *memory)?
                .into_data(),
            losses,
        })
    };
    let run_pcu_output_bank_batched = || -> Result<native::MlpWeights, Box<dyn Error>> {
        let mut memory = memory.borrow_mut();
        let mut resources = feedback_resources.borrow_mut();
        let initial_inputs = [
            (program.samples, &device_samples),
            (program.weights[0], &device_weights[0]),
            (program.weights[1], &device_weights[1]),
            (program.weights[2], &device_weights[2]),
            (program.targets, &device_targets),
        ];
        assessor.execute_feedback_steps_batched(
            &feedback,
            &initial_inputs,
            &mut *resources,
            std::num::NonZeroUsize::new(2).expect("two feedback steps"),
            &mut *memory,
        )?;
        let first_outputs = resources.outputs_for_step(0)?;
        let second_outputs = resources.outputs_for_step(1)?;
        let losses = [
            assessor
                .download_output(&first_outputs[3], selected.pool, &mut *memory)?
                .data()[0],
            assessor
                .download_output(&second_outputs[3], selected.pool, &mut *memory)?
                .data()[0],
        ];
        Ok(native::MlpWeights {
            w1: assessor
                .download_output(&second_outputs[0], selected.pool, &mut *memory)?
                .into_data(),
            w2: assessor
                .download_output(&second_outputs[1], selected.pool, &mut *memory)?
                .into_data(),
            w3: assessor
                .download_output(&second_outputs[2], selected.pool, &mut *memory)?
                .into_data(),
            losses,
        })
    };
    let pcu_cold = run_pcu()?;
    let native_result = native.execute_two_steps()?;
    verify_weights(&native_result, &pcu_cold)?;
    let bank_cold = run_pcu_output_bank()?;
    verify_weights(&native_result, &bank_cold)?;
    let batched_bank_cold = run_pcu_output_bank_batched()?;
    verify_weights(&native_result, &batched_bank_cold)?;
    let cpu = cpu_small_check()?;
    // A small independent CPU graph check validates gradient construction without running a
    // multi-billion-operation oracle for the full-size workload.
    if !cpu {
        return Err(
            "independent small CPU gradient fixture disagreed with Graph::backward_mse".into(),
        );
    }
    let macs_per_batch = [INPUT * HIDDEN, HIDDEN * HIDDEN, HIDDEN * OUTPUT]
        .into_iter()
        .sum::<usize>()
        .checked_mul(batch)
        .ok_or("MLP throughput count overflow")?;
    {
        let mut group = c.benchmark_group("tensor_mlp_train");
        group.throughput(Throughput::Elements(u64::try_from(
            macs_per_batch
                .checked_mul(2)
                .ok_or("MLP throughput count overflow")?,
        )?));
        group.bench_function(
            BenchmarkId::new(
                "pcu_prepared_resident",
                format!("{batch}x{INPUT}-{HIDDEN}-{HIDDEN}-{OUTPUT}"),
            ),
            |b| b.iter(|| black_box(run_pcu().expect("PCU two-step MLP failed"))),
        );
        group.bench_function(
            BenchmarkId::new(
                "pcu_prepared_resident_output_bank_batched",
                format!("{batch}x{INPUT}-{HIDDEN}-{HIDDEN}-{OUTPUT}"),
            ),
            |b| {
                b.iter(|| {
                    black_box(
                        run_pcu_output_bank_batched()
                            .expect("batched PCU output-bank training failed"),
                    )
                });
            },
        );
        group.bench_function(
            BenchmarkId::new(
                "native_hip_rocblas",
                format!("{batch}x{INPUT}-{HIDDEN}-{HIDDEN}-{OUTPUT}"),
            ),
            |b| {
                b.iter(|| {
                    black_box(
                        native
                            .execute_two_steps()
                            .expect("native two-step MLP failed"),
                    );
                });
            },
        );
        group.bench_function(
            BenchmarkId::new(
                "pcu_prepared_resident_output_bank",
                format!("{batch}x{INPUT}-{HIDDEN}-{HIDDEN}-{OUTPUT}"),
            ),
            |b| {
                b.iter(|| {
                    black_box(run_pcu_output_bank().expect("PCU output-bank training failed"))
                });
            },
        );
        group.finish();
    }
    let pcu_result = run_pcu()?;
    verify_weights(&native_result, &pcu_result)?;
    let bank_result = run_pcu_output_bank()?;
    verify_weights(&native_result, &bank_result)?;
    let batched_bank_result = run_pcu_output_bank_batched()?;
    verify_weights(&native_result, &batched_bank_result)?;

    // Order-balanced paired measurements help reveal clock or load drift between routes. These
    // host-wall diagnostics are separate from Criterion and intentionally do not alter its data.
    let mut paired_pcu = Vec::with_capacity(16);
    let mut paired_native = Vec::with_capacity(16);
    let mut paired_ratios = Vec::with_capacity(16);
    for pair in 0..16 {
        let (pcu_elapsed, native_elapsed) = if pair % 2 == 0 {
            let pcu_elapsed = measure_host_wall(|| {
                drop(black_box(run_pcu()?));
                Ok(())
            })?;
            let native_elapsed = measure_host_wall(|| {
                drop(black_box(native.execute_two_steps()?));
                Ok(())
            })?;
            (pcu_elapsed, native_elapsed)
        } else {
            let native_elapsed = measure_host_wall(|| {
                drop(black_box(native.execute_two_steps()?));
                Ok(())
            })?;
            let pcu_elapsed = measure_host_wall(|| {
                drop(black_box(run_pcu()?));
                Ok(())
            })?;
            (pcu_elapsed, native_elapsed)
        };
        let pcu_seconds = pcu_elapsed.as_secs_f64();
        let native_seconds = native_elapsed.as_secs_f64();
        paired_pcu.push(pcu_seconds);
        paired_native.push(native_seconds);
        paired_ratios.push(pcu_seconds / native_seconds);
    }
    println!(
        "MLP batch {batch} order-balanced paired host-wall diagnostic (16 pairs; alternate PCU/native order): median PCU {:.3} ms, median native {:.3} ms, median paired PCU/native {:.3}x, paired ratio range {:.3}–{:.3}x; separate from Criterion",
        median(&mut paired_pcu) * 1_000.0,
        median(&mut paired_native) * 1_000.0,
        median(&mut paired_ratios),
        paired_ratios.iter().copied().fold(f64::INFINITY, f64::min),
        paired_ratios.iter().copied().fold(0.0_f64, f64::max),
    );
    let mut paired_bank = Vec::with_capacity(16);
    let mut paired_bank_native = Vec::with_capacity(16);
    let mut paired_bank_ratios = Vec::with_capacity(16);
    for pair in 0..16 {
        let (bank_elapsed, native_elapsed) = if pair % 2 == 0 {
            let bank_elapsed = measure_host_wall(|| {
                drop(black_box(run_pcu_output_bank()?));
                Ok(())
            })?;
            let native_elapsed = measure_host_wall(|| {
                drop(black_box(native.execute_two_steps()?));
                Ok(())
            })?;
            (bank_elapsed, native_elapsed)
        } else {
            let native_elapsed = measure_host_wall(|| {
                drop(black_box(native.execute_two_steps()?));
                Ok(())
            })?;
            let bank_elapsed = measure_host_wall(|| {
                drop(black_box(run_pcu_output_bank()?));
                Ok(())
            })?;
            (bank_elapsed, native_elapsed)
        };
        let bank_seconds = bank_elapsed.as_secs_f64();
        let native_seconds = native_elapsed.as_secs_f64();
        paired_bank.push(bank_seconds);
        paired_bank_native.push(native_seconds);
        paired_bank_ratios.push(bank_seconds / native_seconds);
    }
    println!(
        "MLP batch {batch} order-balanced banked/native host-wall diagnostic (16 pairs): median banked PCU {:.3} ms, median native {:.3} ms, median paired PCU/native {:.3}x, paired ratio range {:.3}–{:.3}x; separate from Criterion",
        median(&mut paired_bank) * 1_000.0,
        median(&mut paired_bank_native) * 1_000.0,
        median(&mut paired_bank_ratios),
        paired_bank_ratios
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min),
        paired_bank_ratios.iter().copied().fold(0.0_f64, f64::max),
    );

    let mut paired_banked_sync = Vec::with_capacity(16);
    let mut paired_banked_batch = Vec::with_capacity(16);
    let mut paired_banked_batch_ratios = Vec::with_capacity(16);
    for pair in 0..16 {
        let (sync_elapsed, batch_elapsed) = if pair % 2 == 0 {
            let sync_elapsed = measure_host_wall(|| {
                drop(black_box(run_pcu_output_bank()?));
                Ok(())
            })?;
            let batch_elapsed = measure_host_wall(|| {
                drop(black_box(run_pcu_output_bank_batched()?));
                Ok(())
            })?;
            (sync_elapsed, batch_elapsed)
        } else {
            let batch_elapsed = measure_host_wall(|| {
                drop(black_box(run_pcu_output_bank_batched()?));
                Ok(())
            })?;
            let sync_elapsed = measure_host_wall(|| {
                drop(black_box(run_pcu_output_bank()?));
                Ok(())
            })?;
            (sync_elapsed, batch_elapsed)
        };
        let sync_seconds = sync_elapsed.as_secs_f64();
        let batch_seconds = batch_elapsed.as_secs_f64();
        paired_banked_sync.push(sync_seconds);
        paired_banked_batch.push(batch_seconds);
        paired_banked_batch_ratios.push(batch_seconds / sync_seconds);
    }
    println!(
        "MLP batch {batch} order-balanced banked sync/batched host-wall diagnostic (16 alternating pairs): median sync {:.3} ms, median batched {:.3} ms, paired batched/sync {:.3}x, paired ratio range {:.3}–{:.3}x; separate from Criterion",
        median(&mut paired_banked_sync) * 1_000.0,
        median(&mut paired_banked_batch) * 1_000.0,
        median(&mut paired_banked_batch_ratios),
        paired_banked_batch_ratios
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min),
        paired_banked_batch_ratios
            .iter()
            .copied()
            .fold(0.0_f64, f64::max),
    );

    // A separate diagnostic pass records provider work and pool telemetry without contaminating
    // Criterion samples or the state used for the correctness checks above.
    let mut diagnostic = ProfiledMemory::new(PcuOwnedDispatchMemorySession::memory_provider(
        &session,
        selected.pool,
    ));
    let pool_start = diagnostic
        .snapshot(selected.pool)
        .map_err(|error| format!("{error:?}"))?;
    let uploads_started = Instant::now();
    let diagnostic_samples =
        assessor.upload_input(&host_samples, selected.pool, &mut diagnostic)?;
    let diagnostic_targets =
        assessor.upload_input(&host_targets, selected.pool, &mut diagnostic)?;
    let diagnostic_weights = host_weights
        .each_ref()
        .map(|weight| assessor.upload_input(weight, selected.pool, &mut diagnostic))
        .into_iter()
        .collect::<Result<Vec<_>, _>>()?;
    let upload_elapsed = uploads_started.elapsed();
    let scratch_started = Instant::now();
    let mut diagnostic_scratch =
        assessor.prepare_scratch(&prepared, selected.pool, &mut diagnostic)?;
    let scratch_elapsed = scratch_started.elapsed();
    let prepared_profile = diagnostic.profile();
    let mut peak_process_bytes = known_process_bytes(pool_start);
    let mut loss_resources = Vec::with_capacity(2);
    let mut current = Vec::new();
    let mut phase_times = [[std::time::Duration::ZERO; 3]; 2];
    let mut slowest_nodes = Vec::new();
    let execution_started = Instant::now();
    for (step, phase_time) in phase_times.iter_mut().enumerate() {
        let weights = if step == 0 {
            &diagnostic_weights
        } else {
            &current
        };
        let inputs = vec![
            (program.samples, &diagnostic_samples),
            (program.weights[0], &weights[0]),
            (program.weights[1], &weights[1]),
            (program.weights[2], &weights[2]),
            (program.targets, &diagnostic_targets),
        ];
        let (mut next, node_timings) = assessor
            .execute_prepared_outputs_with_resources_and_scratch_resident_profiled(
                &prepared,
                &inputs,
                &mut diagnostic_scratch,
                &mut diagnostic,
            )?;
        for timing in node_timings {
            let index = program
                .graph
                .nodes()
                .position(|node| node.value == timing.value)
                .ok_or("diagnostic node absent from graph")?;
            let phase = if index < program.backward_start {
                0
            } else if index < program.update_start {
                1
            } else {
                2
            };
            phase_time[phase] += timing.elapsed;
            slowest_nodes.push((step, timing));
        }
        loss_resources.push(next.pop().ok_or("missing diagnostic loss output")?);
        current = next;
        peak_process_bytes = peak_process_bytes.max(known_process_bytes(
            diagnostic
                .snapshot(selected.pool)
                .map_err(|error| format!("{error:?}"))?,
        ));
    }
    let execution_elapsed = execution_started.elapsed();
    let download_started = Instant::now();
    let losses = [
        assessor
            .download_output(&loss_resources[0], selected.pool, &mut diagnostic)?
            .data()[0],
        assessor
            .download_output(&loss_resources[1], selected.pool, &mut diagnostic)?
            .data()[0],
    ];
    let diagnostic_result = native::MlpWeights {
        w1: assessor
            .download_output(&current[0], selected.pool, &mut diagnostic)?
            .into_data(),
        w2: assessor
            .download_output(&current[1], selected.pool, &mut diagnostic)?
            .into_data(),
        w3: assessor
            .download_output(&current[2], selected.pool, &mut diagnostic)?
            .into_data(),
        losses,
    };
    let download_elapsed = download_started.elapsed();
    verify_weights(&native_result, &diagnostic_result)?;
    let pool_end = diagnostic
        .snapshot(selected.pool)
        .map_err(|error| format!("{error:?}"))?;
    peak_process_bytes = peak_process_bytes.max(known_process_bytes(pool_end));
    let profile = diagnostic.profile();
    println!(
        "PCU batch {batch} diagnostic stages: uploads {upload_elapsed:?}; scratch {scratch_elapsed:?}; two-step execute incl. losses {execution_elapsed:?}; final output downloads {download_elapsed:?}"
    );
    println!(
        "PCU batch {batch} node wall times: forward {:?}/{:?}; backward {:?}/{:?}; update {:?}/{:?}",
        phase_times[0][0],
        phase_times[1][0],
        phase_times[0][1],
        phase_times[1][1],
        phase_times[0][2],
        phase_times[1][2],
    );
    slowest_nodes.sort_unstable_by_key(|(_, timing)| std::cmp::Reverse(timing.elapsed));
    for (step, timing) in slowest_nodes.iter().take(6) {
        println!(
            "PCU batch {batch} slow node step {step}: {:?} {} {:?}",
            timing.value, timing.operation, timing.elapsed
        );
    }
    println!(
        "PCU batch {batch} provider: {} allocations ({} bytes, {:?}); {} uploads ({} bytes, {:?}); {} downloads ({} bytes, {:?})",
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
    println!(
        "PCU batch {batch} warm two-step provider delta: {} allocations ({} bytes, {:?}); {} uploads ({} bytes, {:?}); {} downloads ({} bytes, {:?})",
        profile.allocations - prepared_profile.allocations,
        profile.allocated_bytes - prepared_profile.allocated_bytes,
        profile.allocation_time - prepared_profile.allocation_time,
        profile.uploads - prepared_profile.uploads,
        profile.uploaded_bytes - prepared_profile.uploaded_bytes,
        profile.upload_time - prepared_profile.upload_time,
        profile.downloads - prepared_profile.downloads,
        profile.downloaded_bytes - prepared_profile.downloaded_bytes,
        profile.download_time - prepared_profile.download_time,
    );
    println!(
        "PCU batch {batch} pool: process used start {}, peak observed {}, end {}; peak telemetry is sampled at setup/end-of-step boundaries; cumulative allocated bytes are a separate upper bound",
        display_bytes(pool_start.process_used_bytes),
        display_bytes_opt(peak_process_bytes),
        display_bytes(pool_end.process_used_bytes),
    );

    let mut bank_diagnostic = ProfiledMemory::new(PcuOwnedDispatchMemorySession::memory_provider(
        &session,
        selected.pool,
    ));
    let mut diagnostic_resources =
        assessor.prepare_feedback_resources(&prepared, selected.pool, &mut bank_diagnostic)?;
    let bank_setup_profile = bank_diagnostic.profile();
    let initial_inputs = [
        (program.samples, &device_samples),
        (program.weights[0], &device_weights[0]),
        (program.weights[1], &device_weights[1]),
        (program.weights[2], &device_weights[2]),
        (program.targets, &device_targets),
    ];
    let device_timing_start = assessor.begin_device_timing()?;
    assessor.execute_feedback_steps(
        &feedback,
        &initial_inputs,
        &mut diagnostic_resources,
        std::num::NonZeroUsize::new(2).expect("two feedback steps"),
        &mut bank_diagnostic,
    )?;
    let banked_device_timeline_ms = assessor.finish_device_timing(device_timing_start)?;
    let diagnostic_first_outputs = diagnostic_resources.outputs_for_step(0)?;
    let diagnostic_second_outputs = diagnostic_resources.outputs_for_step(1)?;
    let bank_diagnostic_result = native::MlpWeights {
        w1: assessor
            .download_output(
                &diagnostic_second_outputs[0],
                selected.pool,
                &mut bank_diagnostic,
            )?
            .into_data(),
        w2: assessor
            .download_output(
                &diagnostic_second_outputs[1],
                selected.pool,
                &mut bank_diagnostic,
            )?
            .into_data(),
        w3: assessor
            .download_output(
                &diagnostic_second_outputs[2],
                selected.pool,
                &mut bank_diagnostic,
            )?
            .into_data(),
        losses: [
            assessor
                .download_output(
                    &diagnostic_first_outputs[3],
                    selected.pool,
                    &mut bank_diagnostic,
                )?
                .data()[0],
            assessor
                .download_output(
                    &diagnostic_second_outputs[3],
                    selected.pool,
                    &mut bank_diagnostic,
                )?
                .data()[0],
        ],
    };
    verify_weights(&native_result, &bank_diagnostic_result)?;
    println!(
        "PCU batch {batch} banked two-step HIP stream timeline: {banked_device_timeline_ms:.3} ms (includes host submission gaps between synchronous nodes)"
    );
    let bank_profile = bank_diagnostic.profile();
    println!(
        "PCU batch {batch} output-bank warm two-step provider delta: {} allocations ({} bytes, {:?}); {} uploads ({} bytes, {:?}); {} downloads ({} bytes, {:?}); setup allocated {} bytes",
        bank_profile.allocations - bank_setup_profile.allocations,
        bank_profile.allocated_bytes - bank_setup_profile.allocated_bytes,
        bank_profile.allocation_time - bank_setup_profile.allocation_time,
        bank_profile.uploads - bank_setup_profile.uploads,
        bank_profile.uploaded_bytes - bank_setup_profile.uploaded_bytes,
        bank_profile.upload_time - bank_setup_profile.upload_time,
        bank_profile.downloads - bank_setup_profile.downloads,
        bank_profile.downloaded_bytes - bank_setup_profile.downloaded_bytes,
        bank_profile.download_time - bank_setup_profile.download_time,
        bank_setup_profile.allocated_bytes,
    );

    let (native_profiled, native_profile) = native.execute_two_steps_profiled()?;
    verify_weights(&pcu_result, &native_profiled)?;
    println!(
        "Native batch {batch} profile: reset {:?}; forward {:?}/{:?}; backward {:?}/{:?}; update {:?}/{:?}; {} allocations ({} bytes; route resident allocation total {}, allocation time {:?}); {} uploads ({} bytes, {:?}); {} downloads ({} bytes, {:?})",
        native_profile.reset_weights,
        native_profile.forward[0],
        native_profile.forward[1],
        native_profile.backward[0],
        native_profile.backward[1],
        native_profile.update[0],
        native_profile.update[1],
        native_profile.allocation_count,
        native_profile.allocated_bytes,
        native_profile.peak_device_bytes,
        native_profile.allocation_time,
        native_profile.upload_count,
        native_profile.uploaded_bytes,
        native_profile.upload_time,
        native_profile.download_count,
        native_profile.downloaded_bytes,
        native_profile.download_time,
    );
    Ok(())
}

const fn known_process_bytes(snapshot: fusion_pcu::PcuMemoryPoolSnapshot) -> Option<u64> {
    match snapshot.process_used_bytes {
        PcuMemoryUsage::Known(bytes) => Some(bytes),
        PcuMemoryUsage::Unknown => None,
    }
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    if values.len().is_multiple_of(2) {
        (values[middle - 1] + values[middle]) * 0.5
    } else {
        values[middle]
    }
}

fn measure_host_wall(
    operation: impl FnOnce() -> Result<(), Box<dyn Error>>,
) -> Result<std::time::Duration, Box<dyn Error>> {
    let started = Instant::now();
    operation()?;
    Ok(started.elapsed())
}

fn display_bytes(usage: PcuMemoryUsage) -> String {
    match usage {
        PcuMemoryUsage::Known(bytes) => format!("{bytes} bytes"),
        PcuMemoryUsage::Unknown => "unavailable".to_owned(),
    }
}

fn display_bytes_opt(bytes: Option<u64>) -> String {
    bytes.map_or_else(
        || "unavailable".to_owned(),
        |value| format!("{value} bytes"),
    )
}

fn cpu_small_check() -> Result<bool, Box<dyn Error>> {
    let mut graph = Graph::default();
    let x = graph.input([2, 3])?;
    let w1 = graph.input([3, 4])?;
    let w2 = graph.input([4, 4])?;
    let w3 = graph.input([4, 2])?;
    let y = graph.input([2, 2])?;
    let z1 = graph.matmul(x, w1)?;
    let a = graph.relu(z1)?;
    let z2 = graph.matmul(a, w2)?;
    let b = graph.relu(z2)?;
    let pred = graph.matmul(b, w3)?;
    let loss = graph.mean_squared_error(pred, y)?;
    let grads = graph.backward_mse(loss)?;
    let updates = [
        updated_for_fixture(&mut graph, w1, grads.as_slice())?,
        updated_for_fixture(&mut graph, w2, grads.as_slice())?,
        updated_for_fixture(&mut graph, w3, grads.as_slice())?,
    ];
    let inputs = [
        (x, Tensor::new([2, 3], vec![0.1; 6])?),
        (w1, Tensor::new([3, 4], vec![0.05; 12])?),
        (w2, Tensor::new([4, 4], vec![0.05; 16])?),
        (w3, Tensor::new([4, 2], vec![0.05; 8])?),
        (y, Tensor::new([2, 2], vec![0.1; 4])?),
    ];
    let execution = graph.evaluate(&inputs)?;
    let independent = execution.gradients(&graph, loss)?;
    for (weight, update) in [w1, w2, w3].into_iter().zip(updates) {
        let index = graph
            .nodes()
            .position(|node| node.value == weight)
            .ok_or("fixture weight missing")?;
        let analytic = independent[index]
            .as_ref()
            .ok_or("independent fixture gradient missing")?;
        let backward = execution.value(grads[index].ok_or("backward fixture gradient missing")?)?;
        if analytic
            .data()
            .iter()
            .zip(backward.data())
            .any(|(a, b)| (a - b).abs() > 1.0e-6)
        {
            return Ok(false);
        }
        let expected = execution.value(update)?;
        let initial = inputs
            .iter()
            .find(|(id, _)| *id == weight)
            .ok_or("fixture weight input missing")?
            .1
            .data();
        for ((value, initial), gradient) in expected.data().iter().zip(initial).zip(analytic.data())
        {
            if (value - (initial - RATE * gradient)).abs() > 1.0e-6 {
                return Ok(false);
            }
        }
    }
    Ok(true)
}

fn updated_for_fixture(
    graph: &mut Graph,
    weight: ValueId,
    gradients: &[Option<ValueId>],
) -> Result<ValueId, Box<dyn Error>> {
    let index = graph
        .nodes()
        .position(|node| node.value == weight)
        .ok_or("fixture weight missing")?;
    let gradient = gradients[index].ok_or("fixture gradient missing")?;
    Ok(graph.sgd_update(weight, gradient, RATE)?)
}

fn verify_weights(
    expected: &native::MlpWeights,
    actual: &native::MlpWeights,
) -> Result<(), Box<dyn Error>> {
    for (name, expected, actual) in [
        ("w1", &expected.w1, &actual.w1),
        ("w2", &expected.w2, &actual.w2),
        ("w3", &expected.w3, &actual.w3),
    ] {
        if expected.len() != actual.len() {
            return Err(format!("{name} length mismatch").into());
        }
        let max_error = expected
            .iter()
            .zip(actual)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        if !max_error.is_finite() || max_error > 2.0e-3 {
            return Err(format!("{name} max absolute error {max_error}").into());
        }
    }
    for (step, (expected, actual)) in expected.losses.iter().zip(actual.losses).enumerate() {
        let error = (expected - actual).abs();
        if !expected.is_finite()
            || !actual.is_finite()
            || error > expected.abs().mul_add(2.0e-3, 2.0e-4)
        {
            return Err(format!(
                "loss at step {step} differs: expected {expected}, got {actual} (abs error {error})"
            )
            .into());
        }
    }
    Ok(())
}

criterion_group! { name = benches; config = support::criterion_config(); targets = bench }
criterion_main!(benches);
