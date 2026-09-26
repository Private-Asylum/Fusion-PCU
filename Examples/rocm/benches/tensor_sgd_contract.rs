//! Compare strict tensor Mul/Sub to explicitly opted-in contracted SGD.

mod support;

use std::{
    error::Error,
    hint::black_box,
    time::Instant,
};

use criterion::{
    BenchmarkId,
    Criterion,
    Throughput,
    criterion_group,
    criterion_main,
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
    TensorArithmeticRewritePolicy,
    ValueId,
};

#[path = "support/sgd_contract.rs"]
mod native;

const RATE: f32 = f32::from_bits(1.0_f32.to_bits() + 1);
const WEIGHT: f32 = 1.0;
const GRADIENT: f32 = f32::from_bits(1.0_f32.to_bits() - 2);

struct Program {
    graph: Graph,
    weights: ValueId,
    gradient: ValueId,
    updated: ValueId,
}

fn program(elements: usize) -> Result<Program, Box<dyn Error>> {
    let mut graph = Graph::default();
    let weights = graph.input([elements])?;
    let gradient = graph.input([elements])?;
    let rate = graph.constant(Tensor::splat([elements], RATE)?);
    let scaled = graph.mul(rate, gradient)?;
    let updated = graph.sub(weights, scaled)?;
    Ok(Program {
        graph,
        weights,
        gradient,
        updated,
    })
}

fn bench(criterion: &mut Criterion) {
    run(criterion).expect("ROCm SGD contraction benchmark failed");
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
        "no ROCm device completed SGD contraction benchmark: {}",
        failures.join("; ")
    )
    .into())
}

#[allow(clippy::too_many_lines, clippy::significant_drop_tightening)] // Keep paired setup and one Criterion group over all shapes.
fn run_on(
    criterion: &mut Criterion,
    discovery: &RocmDiscovery,
    selected: &support::selection::Candidate,
) -> Result<(), Box<dyn Error>> {
    let session = RocmOwnedDispatchBackend::open(discovery, selected.device, 64)?;
    let assessor = RocmTensorAssessor::new(&session)?;
    println!(
        "Device: {}; strict Mul/Sub vs explicit fmaf SGD",
        selected.name
    );

    let mut group = criterion.benchmark_group("tensor_sgd_contract");
    for elements in [1_usize, 16_384, 1_048_576] {
        let program = program(elements)?;
        let weights = Tensor::new([elements], vec![WEIGHT; elements])?;
        let gradient = Tensor::new([elements], vec![GRADIENT; elements])?;
        let strict =
            support::cold_once(&format!("PCU strict {elements} graph preparation"), || {
                assessor.prepare_graph_outputs(&program.graph, &[program.updated])
            })?;
        let contracted = support::cold_once(
            &format!("PCU contracted {elements} graph preparation"),
            || {
                assessor.prepare_graph_outputs_with_policy(
                    &program.graph,
                    &[program.updated],
                    TensorArithmeticRewritePolicy::AllowContractedArithmetic,
                )
            },
        )?;
        if !strict.lowering_plan().rewrites().is_empty()
            || contracted.lowering_plan().rewrites().len() != 1
        {
            return Err("PCU selected an unexpected strict or contracted schedule".into());
        }

        let mut memory = PcuOwnedDispatchMemorySession::memory_provider(&session, selected.pool);
        let (
            device_weights,
            device_gradient,
            mut strict_scratch,
            mut contracted_scratch,
            mut strict_outputs,
            mut contracted_outputs,
        ) = support::cold_once(
            &format!("PCU {elements} input uploads and scratch/output allocations"),
            || {
                let device_weights = assessor.upload_input(&weights, selected.pool, &mut memory)?;
                let device_gradient =
                    assessor.upload_input(&gradient, selected.pool, &mut memory)?;
                let strict_scratch =
                    assessor.prepare_scratch(&strict, selected.pool, &mut memory)?;
                let contracted_scratch =
                    assessor.prepare_scratch(&contracted, selected.pool, &mut memory)?;
                let strict_outputs =
                    assessor.prepare_output_bank(&strict, selected.pool, &mut memory)?;
                let contracted_outputs =
                    assessor.prepare_output_bank(&contracted, selected.pool, &mut memory)?;
                Ok::<_, Box<dyn Error>>((
                    device_weights,
                    device_gradient,
                    strict_scratch,
                    contracted_scratch,
                    strict_outputs,
                    contracted_outputs,
                ))
            },
        )?;
        let strict_inputs = [
            (program.weights, &device_weights),
            (program.gradient, &device_gradient),
        ];
        let native =
            support::cold_once(&format!("native HIP {elements} compile and setup"), || {
                native::NativeSgd::prepare(
                    discovery,
                    selected.device,
                    weights.data(),
                    gradient.data(),
                    RATE,
                )
            })?;

        let execute_pcu = |prepared: &_, scratch: &mut _, output_bank: &mut _, memory: &mut _| {
            assessor.execute_prepared_outputs_into_bank(
                prepared,
                &strict_inputs,
                scratch,
                output_bank,
                memory,
            )?;
            let output = output_bank
                .outputs()
                .first()
                .expect("single output graph has one output-bank entry");
            assessor.download_output(output, selected.pool, memory)
        };
        let execute_pcu_resident =
            |prepared: &_, scratch: &mut _, output_bank: &mut _, memory: &mut _| {
                assessor.execute_prepared_outputs_into_bank(
                    prepared,
                    &strict_inputs,
                    scratch,
                    output_bank,
                    memory,
                )
            };

        let strict_first = support::cold_once(
            &format!("PCU strict {elements} first execution (lazy kernels)"),
            || {
                execute_pcu(
                    &strict,
                    &mut strict_scratch,
                    &mut strict_outputs,
                    &mut memory,
                )
            },
        )?;
        let contracted_first = support::cold_once(
            &format!("PCU contracted {elements} first execution (lazy kernel)"),
            || {
                execute_pcu(
                    &contracted,
                    &mut contracted_scratch,
                    &mut contracted_outputs,
                    &mut memory,
                )
            },
        )?;
        let native_strict_first = native.execute(false)?;
        let native_contracted_first = native.execute(true)?;
        verify_bits(&strict_first, 0.0_f32.to_bits(), "PCU strict")?;
        verify_bits(&contracted_first, 0x2880_0000, "PCU contracted")?;
        verify_slice_bits(&native_strict_first, 0.0_f32.to_bits(), "native strict")?;
        verify_slice_bits(&native_contracted_first, 0x2880_0000, "native fmaf")?;
        println!(
            "{elements} elements semantic checks passed: strict 0x00000000, contracted 0x28800000"
        );

        if matches!(elements, 16_384 | 1_048_576) {
            let mut ratios = Vec::with_capacity(16);
            let mut pcu_times = Vec::with_capacity(16);
            let mut native_times = Vec::with_capacity(16);
            let mut resident_ratios = Vec::with_capacity(16);
            for pair in 0..16 {
                let pcu_first = pair % 2 == 0;
                let first_start = Instant::now();
                if pcu_first {
                    black_box(
                        execute_pcu(
                            &contracted,
                            &mut contracted_scratch,
                            &mut contracted_outputs,
                            &mut memory,
                        )
                        .expect("paired PCU fmaf execution failed"),
                    );
                } else {
                    black_box(native.execute(true).expect("paired native fmaf failed"));
                }
                let first_elapsed = first_start.elapsed();
                let second_start = Instant::now();
                if pcu_first {
                    black_box(native.execute(true).expect("paired native fmaf failed"));
                } else {
                    black_box(
                        execute_pcu(
                            &contracted,
                            &mut contracted_scratch,
                            &mut contracted_outputs,
                            &mut memory,
                        )
                        .expect("paired PCU fmaf execution failed"),
                    );
                }
                let second_elapsed = second_start.elapsed();
                let (pcu_elapsed, native_elapsed) = if pcu_first {
                    (first_elapsed, second_elapsed)
                } else {
                    (second_elapsed, first_elapsed)
                };
                pcu_times.push(pcu_elapsed.as_secs_f64());
                native_times.push(native_elapsed.as_secs_f64());
                ratios.push(pcu_elapsed.as_secs_f64() / native_elapsed.as_secs_f64());
            }
            ratios.sort_by(f64::total_cmp);
            pcu_times.sort_by(f64::total_cmp);
            native_times.sort_by(f64::total_cmp);
            let median_ratio = f64::midpoint(ratios[7], ratios[8]);
            let median_pcu_ms = f64::midpoint(pcu_times[7], pcu_times[8]) * 1_000.0;
            let median_native_ms = f64::midpoint(native_times[7], native_times[8]) * 1_000.0;
            println!(
                "paired host-wall {elements} elements (16 alternating pairs, contracted): PCU/native median ratio {median_ratio:.3}x, range {:.3}–{:.3}x; median PCU {:.3}us, native {:.3}us",
                ratios[0],
                ratios[15],
                median_pcu_ms * 1_000.0,
                median_native_ms * 1_000.0,
            );

            let mut pcu_times = Vec::with_capacity(16);
            let mut native_times = Vec::with_capacity(16);
            for pair in 0..16 {
                let pcu_first = pair % 2 == 0;
                let first_start = Instant::now();
                if pcu_first {
                    execute_pcu_resident(
                        &contracted,
                        &mut contracted_scratch,
                        &mut contracted_outputs,
                        &mut memory,
                    )
                    .expect("paired resident PCU fmaf execution failed");
                } else {
                    native
                        .execute_resident(true)
                        .expect("paired resident native fmaf execution failed");
                }
                let first_elapsed = first_start.elapsed();
                let second_start = Instant::now();
                if pcu_first {
                    native
                        .execute_resident(true)
                        .expect("paired resident native fmaf execution failed");
                } else {
                    execute_pcu_resident(
                        &contracted,
                        &mut contracted_scratch,
                        &mut contracted_outputs,
                        &mut memory,
                    )
                    .expect("paired resident PCU fmaf execution failed");
                }
                let second_elapsed = second_start.elapsed();
                let (pcu_elapsed, native_elapsed) = if pcu_first {
                    (first_elapsed, second_elapsed)
                } else {
                    (second_elapsed, first_elapsed)
                };
                pcu_times.push(pcu_elapsed.as_secs_f64());
                native_times.push(native_elapsed.as_secs_f64());
                resident_ratios.push(pcu_elapsed.as_secs_f64() / native_elapsed.as_secs_f64());
            }
            pcu_times.sort_by(f64::total_cmp);
            native_times.sort_by(f64::total_cmp);
            resident_ratios.sort_by(f64::total_cmp);
            let resident_pcu_us = f64::midpoint(pcu_times[7], pcu_times[8]) * 1_000_000.0;
            let resident_native_us = f64::midpoint(native_times[7], native_times[8]) * 1_000_000.0;
            println!(
                "paired resident host-wall {elements} elements (16 alternating pairs, contracted; each route waits for completion): paired-ratio median {:.3}x, range {:.3}–{:.3}x; median PCU {:.3}us, native {:.3}us",
                f64::midpoint(resident_ratios[7], resident_ratios[8]),
                resident_ratios[0],
                resident_ratios[15],
                resident_pcu_us,
                resident_native_us,
            );

            let mut readback_pcu_times = Vec::with_capacity(16);
            let mut readback_native_times = Vec::with_capacity(16);
            let mut readback_ratios = Vec::with_capacity(16);
            for pair in 0..16 {
                let pcu_first = pair % 2 == 0;
                let first_start = Instant::now();
                if pcu_first {
                    let output = contracted_outputs
                        .outputs()
                        .first()
                        .expect("single output graph has one output-bank entry");
                    black_box(
                        assessor
                            .download_output(output, selected.pool, &mut memory)
                            .expect("paired PCU output readback failed"),
                    );
                } else {
                    black_box(native.readback().expect("paired native readback failed"));
                }
                let first_elapsed = first_start.elapsed();
                let second_start = Instant::now();
                if pcu_first {
                    black_box(native.readback().expect("paired native readback failed"));
                } else {
                    let output = contracted_outputs
                        .outputs()
                        .first()
                        .expect("single output graph has one output-bank entry");
                    black_box(
                        assessor
                            .download_output(output, selected.pool, &mut memory)
                            .expect("paired PCU output readback failed"),
                    );
                }
                let second_elapsed = second_start.elapsed();
                let (pcu_elapsed, native_elapsed) = if pcu_first {
                    (first_elapsed, second_elapsed)
                } else {
                    (second_elapsed, first_elapsed)
                };
                readback_pcu_times.push(pcu_elapsed.as_secs_f64());
                readback_native_times.push(native_elapsed.as_secs_f64());
                readback_ratios.push(pcu_elapsed.as_secs_f64() / native_elapsed.as_secs_f64());
            }
            readback_pcu_times.sort_by(f64::total_cmp);
            readback_native_times.sort_by(f64::total_cmp);
            readback_ratios.sort_by(f64::total_cmp);
            println!(
                "paired output-readback host-wall {elements} elements (16 alternating pairs; resident outputs already complete): paired-ratio median {:.3}x, range {:.3}–{:.3}x; median PCU {:.3}us, native {:.3}us",
                f64::midpoint(readback_ratios[7], readback_ratios[8]),
                readback_ratios[0],
                readback_ratios[15],
                f64::midpoint(readback_pcu_times[7], readback_pcu_times[8]) * 1_000_000.0,
                f64::midpoint(readback_native_times[7], readback_native_times[8]) * 1_000_000.0,
            );
        }

        group.throughput(Throughput::Elements(u64::try_from(elements)?));
        group.bench_function(BenchmarkId::new("pcu_strict_mul_sub", elements), |b| {
            b.iter(|| {
                black_box(
                    execute_pcu(
                        &strict,
                        &mut strict_scratch,
                        &mut strict_outputs,
                        &mut memory,
                    )
                    .expect("PCU strict Mul/Sub execution failed"),
                );
            });
        });
        group.bench_function(BenchmarkId::new("pcu_opt_in_fmaf", elements), |b| {
            b.iter(|| {
                black_box(
                    execute_pcu(
                        &contracted,
                        &mut contracted_scratch,
                        &mut contracted_outputs,
                        &mut memory,
                    )
                    .expect("PCU opted-in fmaf execution failed"),
                );
            });
        });
        group.bench_function(BenchmarkId::new("native_hip_strict", elements), |b| {
            b.iter(|| black_box(native.execute(false).expect("native strict kernel failed")));
        });
        group.bench_function(BenchmarkId::new("native_hip_fmaf", elements), |b| {
            b.iter(|| black_box(native.execute(true).expect("native fmaf kernel failed")));
        });
        if matches!(elements, 16_384 | 1_048_576) {
            group.bench_function(BenchmarkId::new("pcu_fmaf_resident", elements), |b| {
                b.iter(|| {
                    execute_pcu_resident(
                        &contracted,
                        &mut contracted_scratch,
                        &mut contracted_outputs,
                        &mut memory,
                    )
                    .expect("PCU resident fmaf execution failed");
                });
            });
            group.bench_function(
                BenchmarkId::new("native_hip_fmaf_resident", elements),
                |b| {
                    b.iter(|| {
                        native
                            .execute_resident(true)
                            .expect("native resident fmaf execution failed");
                    });
                },
            );
            group.bench_function(BenchmarkId::new("pcu_fmaf_readback", elements), |b| {
                b.iter(|| {
                    let output = contracted_outputs
                        .outputs()
                        .first()
                        .expect("single output graph has one output-bank entry");
                    black_box(
                        assessor
                            .download_output(output, selected.pool, &mut memory)
                            .expect("PCU output readback failed"),
                    );
                });
            });
            group.bench_function(
                BenchmarkId::new("native_hip_fmaf_readback", elements),
                |b| {
                    b.iter(|| black_box(native.readback().expect("native output readback failed")));
                },
            );
        }
    }
    group.finish();
    Ok(())
}

fn verify_bits(tensor: &Tensor, expected: u32, route: &str) -> Result<(), Box<dyn Error>> {
    if tensor
        .data()
        .iter()
        .all(|value| value.to_bits() == expected)
    {
        Ok(())
    } else {
        let actual = tensor.data()[0].to_bits();
        Err(format!("{route} returned {actual:#010x}, expected {expected:#010x}").into())
    }
}

fn verify_slice_bits(values: &[f32], expected: u32, route: &str) -> Result<(), Box<dyn Error>> {
    if values.iter().all(|value| value.to_bits() == expected) {
        Ok(())
    } else {
        let actual = values
            .first()
            .ok_or("native route returned empty output")?
            .to_bits();
        Err(format!("{route} returned {actual:#010x}, expected {expected:#010x}").into())
    }
}

criterion_group! { name = benches; config = support::criterion_config(); targets = bench }
criterion_main!(benches);
