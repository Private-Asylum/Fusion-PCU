//! Volume comparison for unique-input strict PCU and fully-batched native training jobs.

#[rustfmt::skip]
use super::{
    native,
    support,
    train_volume,
};

#[rustfmt::skip]
use std::{
    error::Error,
    hint::black_box,
    sync::atomic::{
        AtomicU64,
        Ordering,
    },
    num::NonZeroUsize,
    time::{
        Duration,
        Instant,
    },
};

#[rustfmt::skip]
use criterion::{
    BenchmarkGroup,
    BenchmarkId,
    Criterion,
    SamplingMode,
    Throughput,
    measurement::WallTime,
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
#[rustfmt::skip]
use train_volume::{
    JobInputs,
    FEATURES,
    LEARNING_RATE,
    ROWS,
};

static NEXT_JOB_ID: AtomicU64 = AtomicU64::new(0);

struct StrictGraph {
    graph: Graph,
    samples: ValueId,
    weights: ValueId,
    target: ValueId,
    output: ValueId,
}

fn strict_graph() -> Result<StrictGraph, Box<dyn Error>> {
    let mut graph = Graph::default();
    let samples = graph.input([ROWS, FEATURES], fusion_pcu::PcuScalarType::F32)?;
    let weights = graph.input([FEATURES, 1], fusion_pcu::PcuScalarType::F32)?;
    let target = graph.input([ROWS, 1], fusion_pcu::PcuScalarType::F32)?;
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

fn next_id() -> Result<u64, Box<dyn Error>> {
    NEXT_JOB_ID
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
            (current <= train_volume::MAX_JOB_ID).then_some(current + 1)
        })
        .map_err(|_| "unique f32 job ID space exhausted".into())
}

fn corpus(count: u64) -> Result<Vec<JobInputs>, Box<dyn Error>> {
    let remaining =
        (train_volume::MAX_JOB_ID + 1).saturating_sub(NEXT_JOB_ID.load(Ordering::Relaxed));
    if count > remaining {
        return Err("requested volume exhausts the remaining unique f32 job IDs".into());
    }
    (0..count)
        .map(|_| train_volume::make_job(next_id()?))
        .collect()
}

pub fn run(criterion: &mut Criterion) -> Result<(), Box<dyn Error>> {
    let discovery = RocmDiscovery::new();
    let candidates = support::selected_candidates(&discovery)?;
    let (session, selected) = support::selection::open_ranked(&discovery, candidates, 64)?;
    run_on(criterion, &discovery, &selected, &session)
}

#[allow(clippy::too_many_lines)]
fn run_on(
    criterion: &mut Criterion,
    discovery: &RocmDiscovery,
    selected: &support::selection::Candidate,
    session: &RocmOwnedDispatchBackend,
) -> Result<(), Box<dyn Error>> {
    let assessor = RocmTensorAssessor::new(session)?;
    let strict = strict_graph()?;
    let prepared = support::cold_once("PCU volume graph preparation", || {
        assessor.prepare_graph(&strict.graph, strict.output)
    })?;
    let mut memory = PcuOwnedDispatchMemorySession::memory_provider(session, selected.pool);
    let seed = train_volume::make_job(next_id()?)?;
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
    let factor = vec![0.5; ROWS];
    let rate = vec![LEARNING_RATE; FEATURES];
    let mut native = native::NativeTrainStep::prepare_strict(
        discovery,
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
    let mut execute_pcu = |job: &JobInputs| -> Result<JobResult, Box<dyn Error>> {
        let refresh_started = Instant::now();
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
        let refresh = refresh_started.elapsed();
        let execution_started = Instant::now();
        let first_inputs = [
            (strict.samples, &device_samples),
            (strict.weights, &device_weights),
            (strict.target, &device_target),
        ];
        assessor.execute_prepared_outputs_into_bank_batched(
            &prepared,
            &first_inputs,
            &mut scratch,
            &mut first_bank,
            &mut bank_memory,
        )?;
        let first_weights = first_bank
            .outputs()
            .first()
            .ok_or("first output bank is empty")?;
        let second_inputs = [
            (strict.samples, &device_samples),
            (strict.weights, first_weights),
            (strict.target, &device_target),
        ];
        assessor.execute_prepared_outputs_into_bank_batched(
            &prepared,
            &second_inputs,
            &mut scratch,
            &mut second_bank,
            &mut bank_memory,
        )?;
        let final_weights = second_bank
            .outputs()
            .first()
            .ok_or("second output bank is empty")?;
        let output = assessor
            .download_output(final_weights, selected.pool, &mut bank_memory)?
            .into_data();
        Ok(JobResult {
            output: output.try_into().map_err(|_| "PCU output shape changed")?,
            refresh,
            execution: execution_started.elapsed(),
        })
    };
    let mut execute_native = |job: &JobInputs| -> Result<JobResult, Box<dyn Error>> {
        let refresh_started = Instant::now();
        native.replace_strict_inputs(&job.samples, &job.target, &job.initial_weights)?;
        let refresh = refresh_started.elapsed();
        let execution_started = Instant::now();
        let output = native.execute_two_fully_batched()?;
        Ok(JobResult {
            output: output
                .try_into()
                .map_err(|_| "native output shape changed")?,
            refresh,
            execution: execution_started.elapsed(),
        })
    };

    // Paired correctness and stale-result rejection use the exact same two input payloads.
    let preflight = corpus(2)?;
    let expected_first = train_volume::cpu_two_steps(&preflight[0]);
    let pcu_first = execute_pcu(&preflight[0])?;
    let native_first = execute_native(&preflight[0])?;
    train_volume::verify(&expected_first, &pcu_first.output)?;
    train_volume::verify(&expected_first, &native_first.output)?;
    let expected_second = train_volume::cpu_two_steps(&preflight[1]);
    let pcu_second = execute_pcu(&preflight[1])?;
    let native_second = execute_native(&preflight[1])?;
    train_volume::verify(&expected_second, &pcu_second.output)?;
    train_volume::verify(&expected_second, &native_second.output)?;
    if train_volume::verify(&expected_second, &pcu_first.output).is_ok()
        || train_volume::verify(&expected_second, &native_first.output).is_ok()
    {
        return Err("preflight failed to reject a previous job's output".into());
    }
    println!(
        "{} strict 4x2 preflight: two distinct payloads passed CPU checks; previous-output reuse rejected",
        selected.name
    );

    let volumes = configured_volumes()?;
    let stress_only = std::env::var_os("FUSION_TRAIN_VOLUME_STRESS").as_deref()
        == Some(std::ffi::OsStr::new("1"));
    if stress_only {
        for &volume in &volumes {
            stress_volume(volume, &mut execute_pcu, &mut execute_native)?;
        }
    }
    if !stress_only {
        // The same bounded, verified pass warms each requested volume before Criterion sampling.
        for &volume in &volumes {
            stress_volume(volume, &mut execute_pcu, &mut execute_native)?;
        }
        for (label, execution_only) in [("end_to_end", false), ("execution_readback", true)] {
            let mut group = criterion.benchmark_group(format!("tensor_train_volume_{label}"));
            group
                .sampling_mode(SamplingMode::Flat)
                .sample_size(10)
                .measurement_time(Duration::from_secs(2));
            for &volume in &volumes {
                group.throughput(Throughput::Elements(u64::try_from(volume)?));
                benchmark_route(
                    &mut group,
                    "pcu_banked",
                    volume,
                    execution_only,
                    &mut execute_pcu,
                );
                benchmark_route(
                    &mut group,
                    "native_fully_batched",
                    volume,
                    execution_only,
                    &mut execute_native,
                );
            }
            group.finish();
        }
    }
    {
        let mut bound_memory =
            PcuOwnedDispatchMemorySession::memory_provider(session, selected.pool);
        let bound_inputs = vec![
            (
                strict.samples,
                assessor.upload_input(&sample_tensor, selected.pool, &mut bound_memory)?,
            ),
            (
                strict.weights,
                assessor.upload_input(&weight_tensor, selected.pool, &mut bound_memory)?,
            ),
            (
                strict.target,
                assessor.upload_input(&target_tensor, selected.pool, &mut bound_memory)?,
            ),
        ];
        let mut bound = assessor.bind_feedback(
            &prepared,
            &[(strict.output, strict.weights)],
            bound_inputs,
            selected.pool,
            bound_memory,
        )?;
        let steps = NonZeroUsize::new(2).ok_or("two-step feedback count must be nonzero")?;
        let mut execute_bound = |job: &JobInputs| -> Result<JobResult, Box<dyn Error>> {
            let refresh_started = Instant::now();
            bound.update_input(
                strict.samples,
                &Tensor::new([ROWS, FEATURES], job.samples.clone())?,
            )?;
            bound.update_input(
                strict.weights,
                &Tensor::new([FEATURES, 1], job.initial_weights.clone())?,
            )?;
            bound.update_input(strict.target, &Tensor::new([ROWS, 1], job.target.clone())?)?;
            let refresh = refresh_started.elapsed();

            let execution_started = Instant::now();
            bound.execute_steps(steps)?;
            let output = bound.read_output(strict.output)?.into_data();
            Ok(JobResult {
                output: output
                    .try_into()
                    .map_err(|_| "bound PCU output shape changed")?,
                refresh,
                execution: execution_started.elapsed(),
            })
        };

        let bound_preflight = corpus(2)?;
        let expected_first = train_volume::cpu_two_steps(&bound_preflight[0]);
        let bound_first = execute_bound(&bound_preflight[0])?;
        train_volume::verify(&expected_first, &bound_first.output)?;
        let expected_second = train_volume::cpu_two_steps(&bound_preflight[1]);
        let bound_second = execute_bound(&bound_preflight[1])?;
        train_volume::verify(&expected_second, &bound_second.output)?;
        if train_volume::verify(&expected_second, &bound_first.output).is_ok() {
            return Err("bound-execution preflight accepted a previous job's output".into());
        }
        println!(
            "{} strict bound-execution preflight: two distinct payloads passed CPU checks; previous-output reuse rejected",
            selected.name
        );
        if stress_only {
            for &volume in &volumes {
                stress_bound_volume(volume, &mut execute_bound)?;
            }
        } else {
            benchmark_bound_route(criterion, &volumes, &mut execute_bound)?;
        }
    }
    println!(
        "End-to-end sums contiguous 4096-job chunk windows including input refresh and loop; execution/readback sums job windows excluding refresh. Chunk generation and verification are outside windows. All timed outputs verified after measured windows; preparation, corpus generation, verification excluded. Jobs run serially, preserving two waits and one readback per job."
    );
    Ok(())
}

fn configured_volumes() -> Result<Vec<usize>, Box<dyn Error>> {
    let Some(raw) = std::env::var_os("FUSION_TRAIN_VOLUME_JOBS") else {
        return Ok(vec![1, 64, 1024, 16_384, 262_144, 1_000_000]);
    };
    let value = raw.to_string_lossy();
    let parsed = value
        .split(',')
        .map(str::parse::<usize>)
        .collect::<Result<Vec<_>, _>>()?;
    if parsed.is_empty() || parsed.contains(&0) {
        return Err("FUSION_TRAIN_VOLUME_JOBS requires positive comma-separated job counts".into());
    }
    Ok(parsed)
}

struct JobResult {
    output: [f32; FEATURES],
    refresh: Duration,
    execution: Duration,
}

const CHUNK_JOBS: usize = 4096;

#[derive(Default)]
struct ChunkTiming {
    refresh: Duration,
    execution: Duration,
    wall: Duration,
}

impl ChunkTiming {
    fn add(&mut self, other: &Self) {
        self.refresh += other.refresh;
        self.execution += other.execution;
        self.wall += other.wall;
    }
}

fn verified_chunk<F>(jobs: &[JobInputs], execute: &mut F) -> Result<ChunkTiming, Box<dyn Error>>
where
    F: FnMut(&JobInputs) -> Result<JobResult, Box<dyn Error>>,
{
    let mut outputs = Vec::with_capacity(jobs.len());
    let mut timing = ChunkTiming::default();
    let started = Instant::now();
    for job in jobs {
        let result = execute(job)?;
        timing.refresh += result.refresh;
        timing.execution += result.execution;
        outputs.push(black_box(result.output));
    }
    timing.wall = started.elapsed();
    for (job, output) in jobs.iter().zip(&outputs) {
        train_volume::verify(&train_volume::cpu_two_steps(job), output)
            .map_err(|error| format!("job {}: {error}", job.id))?;
    }
    Ok(timing)
}

fn sample_volume<F>(
    iterations: u64,
    volume: usize,
    execution_only: bool,
    execute: &mut F,
) -> Result<Duration, Box<dyn Error>>
where
    F: FnMut(&JobInputs) -> Result<JobResult, Box<dyn Error>>,
{
    let mut total = Duration::ZERO;
    for _ in 0..iterations {
        let mut remaining = volume;
        while remaining != 0 {
            let count = remaining.min(CHUNK_JOBS);
            let jobs = corpus(u64::try_from(count)?)?;
            let timing = verified_chunk(&jobs, execute)?;
            total += if execution_only {
                timing.execution
            } else {
                timing.wall
            };
            remaining -= count;
        }
    }
    Ok(total)
}

fn stress_volume<P, N>(volume: usize, pcu: &mut P, native: &mut N) -> Result<(), Box<dyn Error>>
where
    P: FnMut(&JobInputs) -> Result<JobResult, Box<dyn Error>>,
    N: FnMut(&JobInputs) -> Result<JobResult, Box<dyn Error>>,
{
    let mut pcu_total = ChunkTiming::default();
    let mut native_total = ChunkTiming::default();
    let mut completed = 0;
    let mut chunk = 0_usize;
    println!(
        "STRESS_START jobs={volume} chunk_jobs={CHUNK_JOBS} paired_inputs=true confidence_intervals=false"
    );
    while completed < volume {
        let count = (volume - completed).min(CHUNK_JOBS);
        let jobs = corpus(u64::try_from(count)?)?;
        let (pcu_timing, native_timing) = if chunk.is_multiple_of(2) {
            (verified_chunk(&jobs, pcu)?, verified_chunk(&jobs, native)?)
        } else {
            let native_timing = verified_chunk(&jobs, native)?;
            (verified_chunk(&jobs, pcu)?, native_timing)
        };
        pcu_total.add(&pcu_timing);
        native_total.add(&native_timing);
        completed += count;
        chunk += 1;
        if completed.is_multiple_of(65_536) || completed == volume {
            println!(
                "STRESS_PROGRESS jobs={volume} completed={completed} pcu_exec_s={:.6} native_exec_s={:.6}",
                pcu_total.execution.as_secs_f64(),
                native_total.execution.as_secs_f64()
            );
        }
    }
    println!(
        "STRESS_RESULT jobs={volume} verified_per_route={completed} pcu_exec_s={:.9} native_exec_s={:.9} pcu_refresh_s={:.9} native_refresh_s={:.9} pcu_wall_s={:.9} native_wall_s={:.9}",
        pcu_total.execution.as_secs_f64(),
        native_total.execution.as_secs_f64(),
        pcu_total.refresh.as_secs_f64(),
        native_total.refresh.as_secs_f64(),
        pcu_total.wall.as_secs_f64(),
        native_total.wall.as_secs_f64()
    );
    Ok(())
}

fn stress_bound_volume<R>(volume: usize, bound: &mut R) -> Result<(), Box<dyn Error>>
where
    R: FnMut(&JobInputs) -> Result<JobResult, Box<dyn Error>>,
{
    let mut total = ChunkTiming::default();
    let mut completed = 0;
    while completed < volume {
        let count = (volume - completed).min(CHUNK_JOBS);
        let jobs = corpus(u64::try_from(count)?)?;
        let timing = verified_chunk(&jobs, bound)?;
        total.add(&timing);
        completed += count;
    }
    println!(
        "STRESS_RESULT route=pcu_bound jobs={volume} verified={completed} exec_s={:.9} refresh_s={:.9} wall_s={:.9}",
        total.execution.as_secs_f64(),
        total.refresh.as_secs_f64(),
        total.wall.as_secs_f64()
    );
    Ok(())
}

fn benchmark_bound_route<R>(
    criterion: &mut Criterion,
    volumes: &[usize],
    bound: &mut R,
) -> Result<(), Box<dyn Error>>
where
    R: FnMut(&JobInputs) -> Result<JobResult, Box<dyn Error>>,
{
    for &volume in volumes {
        stress_bound_volume(volume, bound)?;
    }
    for (label, execution_only) in [("end_to_end", false), ("execution_readback", true)] {
        let mut group = criterion.benchmark_group(format!("tensor_train_volume_bound_{label}"));
        group
            .sampling_mode(SamplingMode::Flat)
            .sample_size(10)
            .measurement_time(Duration::from_secs(2));
        for &volume in volumes {
            group.throughput(Throughput::Elements(u64::try_from(volume)?));
            benchmark_route(&mut group, "pcu_bound", volume, execution_only, bound);
        }
        group.finish();
    }
    Ok(())
}

fn benchmark_route<F>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    route: &str,
    volume: usize,
    execution_only: bool,
    execute: &mut F,
) where
    F: FnMut(&JobInputs) -> Result<JobResult, Box<dyn Error>>,
{
    group.bench_function(BenchmarkId::new(route, volume), |bencher| {
        bencher.iter_custom(|iterations| {
            sample_volume(iterations, volume, execution_only, execute)
                .expect("unique volume training sample failed")
        });
    });
}
