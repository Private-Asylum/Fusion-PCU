//! Resident comparison of separate PCU Mul-only chain, grouped PCU Mul-only chain, and one native HIP kernel.

mod support;

use std::{
    error::Error,
};

use criterion::{
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
    OpDescriptor,
    Tensor,
    TensorArithmeticRewritePolicy,
    TensorPointwiseGroupingPolicy,
    TensorSelectedOperation,
    ValueId,
};

#[path = "support/mul_chain.rs"]
mod native;
#[path = "support/add_relu_pair.rs"]
mod paired;

fn bench(criterion: &mut Criterion) {
    run(criterion).expect("ROCm Mul-only chain fusion benchmark failed");
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
        "no ROCm device completed Mul-only chain fusion benchmark: {}",
        failures.join("; ")
    )
    .into())
}

#[allow(clippy::too_many_lines, clippy::significant_drop_tightening)] // Keep both schedules and the native peer paired per shape.
fn run_on(
    criterion: &mut Criterion,
    discovery: &RocmDiscovery,
    selected: &support::selection::Candidate,
) -> Result<(), Box<dyn Error>> {
    let session = RocmOwnedDispatchBackend::open(discovery, selected.device, 256)?;
    let assessor = RocmTensorAssessor::new(&session)?;
    println!(
        "Device: {}; resident separate PCU Mul-only chain vs grouped PCU vs native HIP single kernel",
        selected.name
    );

    for elements in [65_usize, 1_048_576] {
        let (left_values, right_values, third_values) = values(elements);
        let expected = expected(&left_values, &right_values, &third_values);
        let mut graph = Graph::default();
        let left_id = graph.input([elements])?;
        let right_id = graph.input([elements])?;
        let third_id = graph.input([elements])?;
        let first = graph.mul(left_id, right_id)?;
        let second = graph.mul(first, third_id)?;
        let third = graph.mul(second, left_id)?;
        let output = third;

        let separate = support::cold_once(
            &format!("PCU separate Mul-only chain {elements} cold graph preparation"),
            || {
                assessor.prepare_graph_outputs_with_policies(
                    &graph,
                    &[output],
                    TensorArithmeticRewritePolicy::Disabled,
                    TensorPointwiseGroupingPolicy::Disabled,
                )
            },
        )?;
        let grouped = support::cold_once(
            &format!("PCU grouped Mul-only chain {elements} cold graph preparation"),
            || {
                assessor.prepare_graph_outputs_with_policies(
                    &graph,
                    &[output],
                    TensorArithmeticRewritePolicy::Disabled,
                    TensorPointwiseGroupingPolicy::BoundedMulIdentity,
                )
            },
        )?;
        let separate_prewarm = support::cold_once(
            &format!("PCU separate {elements} executable prewarm"),
            || assessor.prewarm_prepared_graph(&separate),
        )?;
        let grouped_prewarm = support::cold_once(
            &format!("PCU grouped {elements} executable prewarm"),
            || assessor.prewarm_prepared_graph(&grouped),
        )?;
        println!("{elements} prewarm: separate={separate_prewarm:?}; grouped={grouped_prewarm:?}");
        let grouped_repeat = assessor.prewarm_prepared_graph(&grouped)?;
        if grouped_repeat.compiled_keys != 0
            || grouped_repeat.cache_hits != grouped_repeat.requested_keys
            || grouped_repeat.retained_keys != grouped_repeat.requested_keys
        {
            return Err(format!(
                "{elements} grouped prewarm was not idempotent: {grouped_repeat:?}"
            )
            .into());
        }

        let left = Tensor::new([elements], left_values.clone())?;
        let right = Tensor::new([elements], right_values.clone())?;
        let third = Tensor::new([elements], third_values.clone())?;
        let mut memory = PcuOwnedDispatchMemorySession::memory_provider(&session, selected.pool);
        let (
            device_left,
            device_right,
            device_third,
            mut separate_scratch,
            mut separate_outputs,
            mut grouped_scratch,
            mut grouped_outputs,
        ) = support::cold_once(
            &format!("PCU {elements} resident uploads and output/scratch allocation"),
            || {
                let device_left = assessor.upload_input(&left, selected.pool, &mut memory)?;
                let device_right = assessor.upload_input(&right, selected.pool, &mut memory)?;
                let device_third = assessor.upload_input(&third, selected.pool, &mut memory)?;
                let separate_scratch =
                    assessor.prepare_scratch(&separate, selected.pool, &mut memory)?;
                let separate_outputs =
                    assessor.prepare_output_bank(&separate, selected.pool, &mut memory)?;
                let grouped_scratch =
                    assessor.prepare_scratch(&grouped, selected.pool, &mut memory)?;
                let grouped_outputs =
                    assessor.prepare_output_bank(&grouped, selected.pool, &mut memory)?;
                Ok::<_, Box<dyn Error>>((
                    device_left,
                    device_right,
                    device_third,
                    separate_scratch,
                    separate_outputs,
                    grouped_scratch,
                    grouped_outputs,
                ))
            },
        )?;
        let inputs: [(ValueId, &_); 3] = [
            (left_id, &device_left),
            (right_id, &device_right),
            (third_id, &device_third),
        ];
        let native = support::cold_once(
            &format!("native HIP {elements} compile and resident allocation"),
            || {
                native::NativeMulChain::prepare(
                    discovery,
                    selected.device,
                    &left_values,
                    &right_values,
                    &third_values,
                )
            },
        )?;

        // Verify all schedules before Criterion sampling. Readback is intentionally outside the
        // resident execution samples; after-sample verification catches stale or poisoned state.
        support::cold_once(
            &format!("PCU separate {elements} first execution after prewarm"),
            || {
                assessor.execute_prepared_outputs_into_bank(
                    &separate,
                    &inputs,
                    &mut separate_scratch,
                    &mut separate_outputs,
                    &mut memory,
                )
            },
        )?;
        support::cold_once(
            &format!("PCU grouped {elements} first execution after prewarm"),
            || {
                assessor.execute_prepared_outputs_into_bank(
                    &grouped,
                    &inputs,
                    &mut grouped_scratch,
                    &mut grouped_outputs,
                    &mut memory,
                )
            },
        )?;
        support::cold_once(&format!("native HIP {elements} first execution"), || {
            native.execute_resident()
        })?;
        verify(
            &expected,
            assessor
                .download_output(
                    separate_outputs
                        .outputs()
                        .first()
                        .ok_or("separate output bank is empty")?,
                    selected.pool,
                    &mut memory,
                )?
                .data(),
        )?;
        verify(
            &expected,
            assessor
                .download_output(
                    grouped_outputs
                        .outputs()
                        .first()
                        .ok_or("grouped output bank is empty")?,
                    selected.pool,
                    &mut memory,
                )?
                .data(),
        )?;
        verify(&expected, &native.readback()?)?;
        let separate_scratch_count = scratch_value_count(&separate, output);
        let grouped_scratch_count = scratch_value_count(&grouped, output);
        println!(
            "{elements} elements: PCU separate device allocations=3 inputs + {separate_scratch_count} scratch + {} output; grouped=3 inputs + {grouped_scratch_count} scratch + {} output; native=4 (3 inputs + output)",
            separate_outputs.outputs().len(),
            grouped_outputs.outputs().len(),
        );

        let mut group = criterion.benchmark_group(format!("tensor_mul_chain/{elements}"));
        group.throughput(Throughput::Elements(u64::try_from(elements)?));
        group.bench_function("pcu_prepared_separate_resident", |bencher| {
            bencher.iter(|| {
                assessor
                    .execute_prepared_outputs_into_bank(
                        &separate,
                        &inputs,
                        &mut separate_scratch,
                        &mut separate_outputs,
                        &mut memory,
                    )
                    .expect("separate PCU Mul-only chain execution failed");
            });
        });
        group.bench_function("pcu_prepared_grouped_resident", |bencher| {
            bencher.iter(|| {
                assessor
                    .execute_prepared_outputs_into_bank(
                        &grouped,
                        &inputs,
                        &mut grouped_scratch,
                        &mut grouped_outputs,
                        &mut memory,
                    )
                    .expect("grouped PCU Mul-only chain execution failed");
            });
        });
        group.bench_function("native_hip_single_kernel_resident", |bencher| {
            bencher.iter(|| {
                native
                    .execute_resident()
                    .expect("native HIP Mul-only chain failed");
            });
        });
        group.finish();

        paired::run(
            elements,
            "grouped PCU",
            "native HIP",
            || {
                assessor.execute_prepared_outputs_into_bank(
                    &grouped,
                    &inputs,
                    &mut grouped_scratch,
                    &mut grouped_outputs,
                    &mut memory,
                )?;
                Ok::<_, Box<dyn Error>>(())
            },
            || native.execute_resident(),
        )?;

        assessor.execute_prepared_outputs_into_bank(
            &separate,
            &inputs,
            &mut separate_scratch,
            &mut separate_outputs,
            &mut memory,
        )?;
        assessor.execute_prepared_outputs_into_bank(
            &grouped,
            &inputs,
            &mut grouped_scratch,
            &mut grouped_outputs,
            &mut memory,
        )?;
        native.execute_resident()?;
        verify(
            &expected,
            assessor
                .download_output(
                    separate_outputs
                        .outputs()
                        .first()
                        .ok_or("separate output bank is empty")?,
                    selected.pool,
                    &mut memory,
                )?
                .data(),
        )?;
        verify(
            &expected,
            assessor
                .download_output(
                    grouped_outputs
                        .outputs()
                        .first()
                        .ok_or("grouped output bank is empty")?,
                    selected.pool,
                    &mut memory,
                )?
                .data(),
        )?;
        verify(&expected, &native.readback()?)?;
    }
    Ok(())
}

fn values(elements: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let left = (0..elements)
        .map(|index| if index.is_multiple_of(2) { -1.25 } else { 2.5 })
        .collect::<Vec<_>>();
    let right = vec![0.5; elements];
    let third = vec![0.25; elements];
    (left, right, third)
}

fn expected(left: &[f32], right: &[f32], third: &[f32]) -> Vec<f32> {
    left.iter()
        .zip(right)
        .zip(third)
        .map(|((left, right), third)| {
            let first = left * right;
            let second = first * third;
            second * left
        })
        .collect()
}

fn scratch_value_count(
    prepared: &fusion_pcu_rocm::RocmPreparedTensorGraph<'_>,
    output: ValueId,
) -> usize {
    prepared
        .lowering_plan()
        .operations()
        .iter()
        .filter(|operation| {
            matches!(operation, TensorSelectedOperation::Node(node)
                if node.value != output && !matches!(node.op, OpDescriptor::Input))
        })
        .count()
}

fn verify(expected: &[f32], actual: &[f32]) -> Result<(), Box<dyn Error>> {
    if expected.len() == actual.len()
        && expected
            .iter()
            .zip(actual)
            .all(|(expected, actual)| expected.to_bits() == actual.to_bits())
    {
        Ok(())
    } else {
        Err("Mul-only chain result mismatch".into())
    }
}

criterion_group! { name = benches; config = support::criterion_config(); targets = bench }
criterion_main!(benches);
