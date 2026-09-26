//! Resident comparison of separate PCU Add/ReLU, grouped PCU Add/ReLU, and one native HIP kernel.

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

#[path = "support/add_relu_fusion.rs"]
mod native;
#[path = "support/add_relu_pair.rs"]
mod paired;

fn bench(criterion: &mut Criterion) {
    run(criterion).expect("ROCm Add+ReLU fusion benchmark failed");
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
        "no ROCm device completed Add+ReLU fusion benchmark: {}",
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
        "Device: {}; resident separate PCU Add/ReLU vs grouped PCU vs native HIP single kernel",
        selected.name
    );

    for elements in [65_usize, 1_048_576] {
        let (left_values, right_values) = values(elements);
        let expected = expected(&left_values, &right_values);
        let mut graph = Graph::default();
        let left_id = graph.input([elements])?;
        let right_id = graph.input([elements])?;
        let added = graph.add(left_id, right_id)?;
        let output = graph.relu(added)?;

        let separate = support::cold_once(
            &format!("PCU separate Add/ReLU {elements} cold graph preparation"),
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
            &format!("PCU grouped Add/ReLU {elements} cold graph preparation"),
            || {
                assessor.prepare_graph_outputs_with_policies(
                    &graph,
                    &[output],
                    TensorArithmeticRewritePolicy::Disabled,
                    TensorPointwiseGroupingPolicy::SingleUseAddRelu,
                )
            },
        )?;

        let left = Tensor::new([elements], left_values.clone())?;
        let right = Tensor::new([elements], right_values.clone())?;
        let mut memory = PcuOwnedDispatchMemorySession::memory_provider(&session, selected.pool);
        let (
            device_left,
            device_right,
            mut separate_scratch,
            mut separate_outputs,
            mut grouped_scratch,
            mut grouped_outputs,
        ) = support::cold_once(
            &format!("PCU {elements} resident uploads and output/scratch allocation"),
            || {
                let device_left = assessor.upload_input(&left, selected.pool, &mut memory)?;
                let device_right = assessor.upload_input(&right, selected.pool, &mut memory)?;
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
                    separate_scratch,
                    separate_outputs,
                    grouped_scratch,
                    grouped_outputs,
                ))
            },
        )?;
        let inputs: [(ValueId, &_); 2] = [(left_id, &device_left), (right_id, &device_right)];
        let native = support::cold_once(
            &format!("native HIP {elements} compile and resident allocation"),
            || {
                native::NativeAddRelu::prepare(
                    discovery,
                    selected.device,
                    &left_values,
                    &right_values,
                )
            },
        )?;

        // Verify all schedules before Criterion sampling. Readback is intentionally outside the
        // resident execution samples; after-sample verification catches stale or poisoned state.
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
        let separate_scratch_count = scratch_value_count(&separate, output);
        let grouped_scratch_count = scratch_value_count(&grouped, output);
        println!(
            "{elements} elements: PCU separate device allocations=2 inputs + {separate_scratch_count} scratch + {} output; grouped=2 inputs + {grouped_scratch_count} scratch + {} output; native=3 (2 inputs + output)",
            separate_outputs.outputs().len(),
            grouped_outputs.outputs().len(),
        );

        let mut group = criterion.benchmark_group(format!("tensor_add_relu_fusion/{elements}"));
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
                    .expect("separate PCU Add/ReLU execution failed");
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
                    .expect("grouped PCU Add/ReLU execution failed");
            });
        });
        group.bench_function("native_hip_single_kernel_resident", |bencher| {
            bencher.iter(|| {
                native
                    .execute_resident()
                    .expect("native HIP Add/ReLU failed");
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

fn values(elements: usize) -> (Vec<f32>, Vec<f32>) {
    let left = (0..elements)
        .map(|index| if index.is_multiple_of(2) { -1.25 } else { 2.5 })
        .collect::<Vec<_>>();
    let right = vec![0.5; elements];
    (left, right)
}

fn expected(left: &[f32], right: &[f32]) -> Vec<f32> {
    left.iter()
        .zip(right)
        .map(|(left, right)| (left + right).max(0.0))
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
        Err("Add+ReLU result mismatch".into())
    }
}

criterion_group! { name = benches; config = support::criterion_config(); targets = bench }
criterion_main!(benches);
