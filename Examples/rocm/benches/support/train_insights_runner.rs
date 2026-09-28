//! Paired raw-tick diagnostics; reporting and verification remain outside measured calls.
#[rustfmt::skip]
use super::{
    native,
    support,
    train_volume,
    trace_markers,
    cpu_counters,
    insights,
};
#[rustfmt::skip]
use std::{
    error::Error,
    num::NonZeroUsize,
    time::{
        Duration,
        Instant,
    },
};
#[rustfmt::skip]
use fusion_pcu::{
    PcuOwnedDispatchMemorySession,
    insights::InsightClock,
};
#[rustfmt::skip]
use fusion_pcu_rocm::{
    RocmDiscovery,
    RocmOwnedDispatchBackend,
    RocmTensorAssessor,
};
#[rustfmt::skip]
use fusion_pcu_tensor::{
    Graph,
    Tensor,
    ValueId,
};
#[rustfmt::skip]
use train_volume::{
    FEATURES,
    LEARNING_RATE,
    ROWS,
};
struct StrictGraph {
    graph: Graph,
    samples: ValueId,
    weights: ValueId,
    target: ValueId,
    output: ValueId,
}

fn strict_graph() -> Result<StrictGraph, Box<dyn Error>> {
    let mut graph = Graph::default();
    let samples = graph.input([ROWS, FEATURES])?;
    let weights = graph.input([FEATURES, 1])?;
    let target = graph.input([ROWS, 1])?;
    let prediction = graph.matmul(samples, weights)?;
    let loss = graph.mean_squared_error(prediction, target)?;
    let gradients = graph.backward_mse(loss)?;
    let weight_index = graph
        .nodes()
        .position(|node| node.value == weights)
        .ok_or("weight input missing")?;
    let gradient = gradients[weight_index].ok_or("weight gradient missing")?;
    let output = graph.sgd_update(weights, gradient, LEARNING_RATE)?;
    Ok(StrictGraph {
        graph,
        samples,
        weights,
        target,
        output,
    })
}

#[allow(clippy::too_many_lines, clippy::cast_precision_loss)] // Clock calibration is a post-run estimate, not integer accounting.
pub fn run() -> Result<(), Box<dyn Error>> {
    let bound_route = std::env::var_os("FUSION_TRAIN_INSIGHTS_BOUND").as_deref()
        == Some(std::ffi::OsStr::new("1"));
    let control = std::env::var_os("FUSION_TRAIN_INSIGHTS_CONTROL").as_deref()
        == Some(std::ffi::OsStr::new("1"));
    if bound_route && !control {
        return Err(
            "bound execution is control-only until its profiling phase is implemented; set FUSION_TRAIN_INSIGHTS_CONTROL=1".into(),
        );
    }
    let discovery = RocmDiscovery::new();
    let candidates = support::selected_candidates(&discovery)?;
    let (session, selected) = support::selection::open_ranked(&discovery, candidates, 64)?;
    let session: &RocmOwnedDispatchBackend = &session;
    let assessor = RocmTensorAssessor::new(session)?;
    let strict = strict_graph()?;
    let prepared = support::cold_once("PCU volume graph preparation", || {
        assessor.prepare_graph(&strict.graph, strict.output)
    })?;
    let mut memory = PcuOwnedDispatchMemorySession::memory_provider(session, selected.pool);
    let seed = train_volume::make_job(0)?;
    let sample_tensor = Tensor::new([ROWS, FEATURES], seed.samples.clone())?;
    let weight_tensor = Tensor::new([FEATURES, 1], seed.initial_weights.clone())?;
    let target_tensor = Tensor::new([ROWS, 1], seed.target.clone())?;
    let mut device_samples = assessor.upload_input(&sample_tensor, selected.pool, &mut memory)?;
    let mut device_weights = assessor.upload_input(&weight_tensor, selected.pool, &mut memory)?;
    let mut device_target = assessor.upload_input(&target_tensor, selected.pool, &mut memory)?;
    let mut bank_memory = PcuOwnedDispatchMemorySession::memory_provider(session, selected.pool);
    let mut scratch = assessor.prepare_scratch(&prepared, selected.pool, &mut bank_memory)?;
    let mut first_bank =
        assessor.prepare_output_bank(&prepared, selected.pool, &mut bank_memory)?;
    let mut second_bank =
        assessor.prepare_output_bank(&prepared, selected.pool, &mut bank_memory)?;
    let mut bound_execution = if bound_route {
        let mut provider = PcuOwnedDispatchMemorySession::memory_provider(session, selected.pool);
        let bound_inputs = vec![
            (
                strict.samples,
                assessor.upload_input(&sample_tensor, selected.pool, &mut provider)?,
            ),
            (
                strict.weights,
                assessor.upload_input(&weight_tensor, selected.pool, &mut provider)?,
            ),
            (
                strict.target,
                assessor.upload_input(&target_tensor, selected.pool, &mut provider)?,
            ),
        ];
        Some(assessor.bind_feedback(
            &prepared,
            &[(strict.output, strict.weights)],
            bound_inputs,
            selected.pool,
            provider,
        )?)
    } else {
        None
    };
    let factor = vec![0.5; ROWS];
    let rate = vec![LEARNING_RATE; FEATURES];
    let mut native = native::NativeTrainStep::prepare_strict(
        &discovery,
        selected.device,
        &native::TrainInputs {
            rows: ROWS,
            features: FEATURES,
            samples: &seed.samples,
            target: &seed.target,
            factor: &factor,
            initial_weights: &seed.initial_weights,
            rate: &rate,
        },
        LEARNING_RATE,
    )?;

    let mut clock = insights::SerializedTscClock;
    let mut empty_probe =
        fusion_pcu::insights::InsightLedger::<_, 1, 1>::new(insights::SerializedTscClock);
    for _ in 0..4096 {
        empty_probe.scope(0, |_| std::hint::black_box(()));
    }
    let mut pcu_wall = Duration::ZERO;
    let mut native_wall = Duration::ZERO;
    let jobs = std::env::var("FUSION_TRAIN_INSIGHTS_JOBS").map_or(Ok(4096_usize), |v| v.parse())?;
    if jobs == 0 || jobs > 1_000_000 {
        return Err("insight jobs must be between 1 and 1,000,000".into());
    }
    let profile_capacity = if control { 0 } else { jobs };
    let mut pcu_profiles = Vec::with_capacity(profile_capacity);
    let mut native_profiles = Vec::with_capacity(profile_capacity);
    let mut readback_ticks = 0_u64;
    let calibration_start = Instant::now();
    let stamp = clock.stamp();
    std::thread::sleep(Duration::from_millis(20));
    let end = clock.stamp();
    if stamp.context != end.context {
        return Err("clock calibration migrated CPUs; pin with taskset".into());
    }
    let calibration_ticks = end
        .ticks
        .checked_sub(stamp.ticks)
        .filter(|ticks| *ticks > 0)
        .ok_or("clock calibration did not advance monotonically")?;
    let ticks_per_second = calibration_ticks as f64 / calibration_start.elapsed().as_secs_f64();
    let counters_enabled = std::env::var_os("FUSION_TRAIN_INSIGHTS_COUNTERS").as_deref()
        == Some(std::ffi::OsStr::new("1"));
    let mut pcu_counters = counters_enabled
        .then(cpu_counters::CpuCounters::open)
        .transpose()?;
    let mut native_counters = counters_enabled
        .then(cpu_counters::CpuCounters::open)
        .transpose()?;
    let mut pcu_counts = cpu_counters::CpuCounterSample::default();
    let mut native_counts = cpu_counters::CpuCounterSample::default();
    let trace = std::env::var_os("FUSION_TRAIN_INSIGHTS_TRACE").as_deref()
        == Some(std::ffi::OsStr::new("1"));
    let markers = trace.then(trace_markers::TraceMarkers::open).transpose()?;
    let warmup = 64_usize;
    let feedback_steps = NonZeroUsize::new(2).ok_or("two-step feedback count must be nonzero")?;
    for id in 1..=u64::try_from(jobs + warmup)? {
        let job = train_volume::make_job(id)?;
        assert_eq!(job.id, id);
        let expected = train_volume::cpu_two_steps(&job);
        let pcu_marker = if bound_route {
            c"PCU bound job"
        } else {
            c"PCU banked job"
        };
        if let Some(execution) = bound_execution.as_mut() {
            execution.update_input(
                strict.samples,
                &Tensor::new([ROWS, FEATURES], job.samples.clone())?,
            )?;
            execution.update_input(
                strict.weights,
                &Tensor::new([FEATURES, 1], job.initial_weights.clone())?,
            )?;
            execution.update_input(strict.target, &Tensor::new([ROWS, 1], job.target.clone())?)?;
        } else {
            assessor.update_input(
                &mut device_samples,
                &Tensor::new([ROWS, FEATURES], job.samples.clone())?,
                &mut memory,
            )?;
            assessor.update_input(
                &mut device_weights,
                &Tensor::new([FEATURES, 1], job.initial_weights.clone())?,
                &mut memory,
            )?;
            assessor.update_input(
                &mut device_target,
                &Tensor::new([ROWS, 1], job.target.clone())?,
                &mut memory,
            )?;
        }
        native.replace_strict_inputs(&job.samples, &job.target, &job.initial_weights)?;
        let mut execute_pcu = || -> Result<_, Box<dyn Error>> {
            if let Some(markers) = &markers
                && markers.push(pcu_marker) < 0
            {
                return Err("PCU trace push failed".into());
            }
            if let Some(counters) = &mut pcu_counters {
                counters.start()?;
            }
            let started = Instant::now();
            let (output, profiles, readback) = if let Some(execution) = bound_execution.as_mut() {
                execution.execute_steps(feedback_steps)?;
                let output = execution.read_output(strict.output)?.into_data();
                (output, [None, None], 0)
            } else {
                let first = if control {
                    assessor.execute_prepared_outputs_into_bank_batched(
                        &prepared,
                        &[
                            (strict.samples, &device_samples),
                            (strict.weights, &device_weights),
                            (strict.target, &device_target),
                        ],
                        &mut scratch,
                        &mut first_bank,
                        &mut bank_memory,
                    )?;
                    None
                } else {
                    Some(
                        assessor.execute_prepared_outputs_into_bank_batched_insights_profiled(
                            &prepared,
                            &[
                                (strict.samples, &device_samples),
                                (strict.weights, &device_weights),
                                (strict.target, &device_target),
                            ],
                            &mut scratch,
                            &mut first_bank,
                            &mut bank_memory,
                            &mut clock,
                        )?,
                    )
                };
                let weights = first_bank.outputs().first().ok_or("empty first bank")?;
                let second = if control {
                    assessor.execute_prepared_outputs_into_bank_batched(
                        &prepared,
                        &[
                            (strict.samples, &device_samples),
                            (strict.weights, weights),
                            (strict.target, &device_target),
                        ],
                        &mut scratch,
                        &mut second_bank,
                        &mut bank_memory,
                    )?;
                    None
                } else {
                    Some(
                        assessor.execute_prepared_outputs_into_bank_batched_insights_profiled(
                            &prepared,
                            &[
                                (strict.samples, &device_samples),
                                (strict.weights, weights),
                                (strict.target, &device_target),
                            ],
                            &mut scratch,
                            &mut second_bank,
                            &mut bank_memory,
                            &mut clock,
                        )?,
                    )
                };
                let weights = second_bank.outputs().first().ok_or("empty second bank")?;
                let readback_start = (!control).then(|| clock.stamp());
                let output = assessor
                    .download_output(weights, selected.pool, &mut bank_memory)?
                    .into_data();
                let readback = if let Some(start) = readback_start {
                    let end = clock.stamp();
                    (start.context == end.context)
                        .then(|| end.ticks.checked_sub(start.ticks))
                        .flatten()
                        .ok_or("invalid readback clock sample")?
                } else {
                    0
                };
                (output, [first, second], readback)
            };
            let elapsed = started.elapsed();
            let counts = pcu_counters
                .as_mut()
                .map(cpu_counters::CpuCounters::stop)
                .transpose()?;
            if let Some(markers) = &markers
                && markers.pop() < 0
            {
                return Err("trace pop failed".into());
            }
            Ok((output, elapsed, profiles, readback, counts))
        };
        let mut execute_native = || -> Result<_, Box<dyn Error>> {
            if let Some(markers) = &markers
                && markers.push(c"Native job") < 0
            {
                return Err("native trace push failed".into());
            }
            if let Some(counters) = &mut native_counters {
                counters.start()?;
            }
            let started = Instant::now();
            let (output, profile) = if control {
                (native.execute_two_fully_batched()?, None)
            } else {
                let (output, profile) =
                    native.execute_two_fully_batched_profiled(&mut insights::SerializedTscClock)?;
                (output, Some(profile))
            };
            let elapsed = started.elapsed();
            let counts = native_counters
                .as_mut()
                .map(cpu_counters::CpuCounters::stop)
                .transpose()?;
            if let Some(markers) = &markers
                && markers.pop() < 0
            {
                return Err("trace pop failed".into());
            }
            Ok((output, elapsed, profile, counts))
        };
        // Alternate whole paired jobs. No phase reporting occurs during measurement.
        let (pcu, peer) = if id.is_multiple_of(2) {
            (execute_pcu()?, execute_native()?)
        } else {
            let peer = execute_native()?;
            (execute_pcu()?, peer)
        };
        train_volume::verify(&expected, &pcu.0)?;
        train_volume::verify(&expected, &peer.0)?;
        // Warmup pairs execute and verify fresh inputs but are excluded from aggregates.
        if id > u64::try_from(warmup)? {
            if let Some(counts) = pcu.4 {
                add_counts(&mut pcu_counts, counts)?;
            }
            if let Some(counts) = peer.3 {
                add_counts(&mut native_counts, counts)?;
            }
            pcu_wall += pcu.1;
            native_wall += peer.1;
            readback_ticks = readback_ticks
                .checked_add(pcu.3)
                .ok_or("readback aggregate overflow")?;
            if let [Some(first), Some(second)] = pcu.2 {
                pcu_profiles.push([first, second]);
            }
            if let Some(profile) = peer.2 {
                native_profiles.push(profile);
            }
        }
    }
    let pcu_route = if bound_route { "bound" } else { "banked" };
    println!(
        "Verified paired diagnostic jobs: {jobs} (PCU route={pcu_route}; control={control}; no confidence interval)"
    );
    println!(
        "Elapsed TSC calibration: {ticks_per_second:.0} ticks/second; ticks are not retired CPU cycles"
    );
    println!(
        "Paired mean execution/readback: PCU {:?}, native {:?}",
        pcu_wall / u32::try_from(jobs)?,
        native_wall / u32::try_from(jobs)?
    );
    if counters_enabled {
        println!(
            "CPU_COUNTERS PCU route={pcu_route}, user benchmark thread only, raw/unscaled; includes probes/timer calls, excludes kernel/GPU/driver worker threads"
        );
        println!("CPU_COUNTERS PCU {pcu_counts:?}; native {native_counts:?}");
        println!(
            "CPU_COUNTERS mean cycles/job PCU {:.1}, native {:.1}; instructions/job PCU {:.1}, native {:.1}",
            pcu_counts.cycles as f64 / jobs as f64,
            native_counts.cycles as f64 / jobs as f64,
            pcu_counts.instructions as f64 / jobs as f64,
            native_counts.instructions as f64 / jobs as f64
        );
    }
    if control {
        return Ok(());
    }
    if pcu_profiles
        .iter()
        .flatten()
        .any(|p| p.invalid_samples != 0 || p.counter_overflow)
        || native_profiles
            .iter()
            .any(|p| p.status != fusion_pcu::insights::InsightStatus::default())
    {
        return Err("invalid insight samples; no timing attribution published".into());
    }
    let pcu_sum = |get: fn(&fusion_pcu_rocm::RocmTensorBatchedExecutionTiming) -> Option<u64>| -> Result<u64, Box<dyn Error>> {
        pcu_profiles.iter().flatten().try_fold(0_u64, |sum, p| {
            sum.checked_add(get(p).ok_or("invalid PCU record")?).ok_or_else(|| "PCU aggregate overflow".into())
        })
    };
    let native_sum =
        |get: fn(&native::NativeFullyBatchedTickProfile) -> u64| -> Result<u64, Box<dyn Error>> {
            native_profiles.iter().try_fold(0_u64, |sum, p| {
                sum.checked_add(get(p))
                    .ok_or_else(|| "native aggregate overflow".into())
            })
        };
    let mean_us = |ticks: u64| ticks as f64 / ticks_per_second * 1_000_000.0 / jobs as f64;
    for (name, ticks) in [
        (
            "PCU entry validation",
            pcu_sum(|p| p.entry_validation.ticks)?,
        ),
        (
            "PCU scheduler validation",
            pcu_sum(|p| p.scheduler_validation.ticks)?,
        ),
        (
            "PCU nodes (inclusive)",
            pcu_sum(|p| p.node_execution.ticks)?,
        ),
        ("PCU final finish", pcu_sum(|p| p.final_batch_finish.ticks)?),
        ("PCU final wait", pcu_sum(|p| p.final_batch_wait.ticks)?),
        ("PCU cleanup", pcu_sum(|p| p.batch_drop_cleanup.ticks)?),
        (
            "PCU scheduler residual/probes",
            pcu_sum(|p| p.scheduler_bookkeeping_ticks)?,
        ),
        ("PCU readback", readback_ticks),
        (
            "Native enqueue adapters",
            native_sum(|p| {
                p.forward_sgemm_enqueue
                    .iter()
                    .chain(&p.delta_enqueue)
                    .chain(&p.scale_enqueue)
                    .chain(&p.gradient_sgemm_enqueue)
                    .chain(&p.update_enqueue)
                    .map(|r| r.inclusive_ticks)
                    .sum()
            })?,
        ),
        (
            "Native finish",
            native_sum(|p| p.batch_finish.iter().map(|r| r.inclusive_ticks).sum())?,
        ),
        (
            "Native wait",
            native_sum(|p| p.completion_wait.iter().map(|r| r.inclusive_ticks).sum())?,
        ),
        (
            "Native cleanup",
            native_sum(|p| p.postwait_drop.iter().map(|r| r.inclusive_ticks).sum())?,
        ),
        (
            "Native readback",
            native_sum(|p| p.final_readback.inclusive_ticks)?,
        ),
        (
            "Native self/probes",
            native_sum(|p| p.whole_call.exclusive_ticks)?,
        ),
    ] {
        println!("PHASE {name}: {:.3} us/job", mean_us(ticks));
    }
    for (index, name) in fusion_pcu_rocm::RocmTensorBatchedExecutionTiming::operation_names()
        .iter()
        .enumerate()
    {
        let ticks = pcu_profiles.iter().flatten().try_fold(0_u64, |sum, p| {
            sum.checked_add(
                p.operations[index]
                    .record
                    .ticks
                    .ok_or("invalid operation record")?,
            )
            .ok_or("operation aggregate overflow")
        })?;
        if ticks != 0 {
            println!(
                "OPERATION PCU {name}: {:.3} us/job (included in nodes)",
                mean_us(ticks)
            );
        }
    }
    println!(
        "Empty scope probe baseline (raw ticks; no automatic subtraction): {:?}, status {:?}",
        empty_probe.records()[0],
        empty_probe.status()
    );
    println!("Last paired PCU step records: {:#?}", pcu_profiles.last());
    println!("Last paired native records: {:#?}", native_profiles.last());
    Ok(())
}

fn add_counts(
    total: &mut cpu_counters::CpuCounterSample,
    sample: cpu_counters::CpuCounterSample,
) -> Result<(), Box<dyn Error>> {
    total.cycles = total
        .cycles
        .checked_add(sample.cycles)
        .ok_or("cycle aggregate overflow")?;
    total.instructions = total
        .instructions
        .checked_add(sample.instructions)
        .ok_or("instruction aggregate overflow")?;
    total.time_enabled = total
        .time_enabled
        .checked_add(sample.time_enabled)
        .ok_or("enabled time aggregate overflow")?;
    total.time_running = total
        .time_running
        .checked_add(sample.time_running)
        .ok_or("running time aggregate overflow")?;
    Ok(())
}
