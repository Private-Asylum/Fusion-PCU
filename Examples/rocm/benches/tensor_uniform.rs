//! Compare dense splat constants with backend-compact graph Uniform values.

mod support;

use std::{
    error::Error,
    hint::black_box,
};

use criterion::{
    BenchmarkId,
    Criterion,
    Throughput,
    criterion_group,
    criterion_main,
};
use fusion_pcu::{
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
};

#[path = "support/memory_profile.rs"]
mod memory_profile;

const ELEMENTS: usize = 1_048_576;
const UNIFORM_VALUE: f32 = 0.25;

fn bench(criterion: &mut Criterion) {
    run(criterion).expect("ROCm uniform tensor benchmark failed");
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
        "no ROCm device completed Uniform benchmark: {}",
        failures.join("; ")
    )
    .into())
}

#[allow(clippy::significant_drop_tightening)]
fn run_on(
    criterion: &mut Criterion,
    discovery: &RocmDiscovery,
    selected: &support::selection::Candidate,
) -> Result<(), Box<dyn Error>> {
    let session = RocmOwnedDispatchBackend::open(discovery, selected.device, 64)?;
    let assessor = RocmTensorAssessor::new(&session)?;
    let mut group = criterion.benchmark_group("tensor_uniform_1m");
    group.throughput(Throughput::Elements(ELEMENTS as u64));
    println!("Device: {}; dense splat vs compact Uniform", selected.name);

    for (name, uniform) in [("DenseSplatConstant", false), ("GraphUniform", true)] {
        let mut graph = Graph::default();
        let input = graph.input([ELEMENTS])?;
        let value = if uniform {
            graph.uniform([ELEMENTS], UNIFORM_VALUE)?
        } else {
            graph.constant(Tensor::splat([ELEMENTS], UNIFORM_VALUE)?)
        };
        let output = graph.add(input, value)?;
        let prepared = assessor.prepare_graph_outputs(&graph, &[output])?;

        let host_input = Tensor::splat([ELEMENTS], 1.0)?;
        let mut input_provider =
            PcuOwnedDispatchMemorySession::memory_provider(&session, selected.pool);
        let device_input =
            assessor.upload_input(&host_input, selected.pool, &mut input_provider)?;
        let mut memory = memory_profile::ProfiledMemory::new(input_provider);

        let (mut scratch, mut output_bank) =
            support::cold_once(&format!("{name} cold scratch/output preparation"), || {
                Ok::<_, fusion_pcu_rocm::RocmTensorExecutionError>((
                    assessor.prepare_scratch(&prepared, selected.pool, &mut memory)?,
                    assessor.prepare_output_bank(&prepared, selected.pool, &mut memory)?,
                ))
            })?;
        let profile = memory.profile();
        println!(
            "{name} cold provider profile: allocations={} ({} bytes), uploads={} ({} bytes), provider allocation={:?}, upload={:?}",
            profile.allocations,
            profile.allocated_bytes,
            profile.uploads,
            profile.uploaded_bytes,
            profile.allocation_time,
            profile.upload_time,
        );

        let persistent_inputs = [(input, &device_input)];
        assessor.execute_prepared_outputs_into_bank(
            &prepared,
            &persistent_inputs,
            &mut scratch,
            &mut output_bank,
            &mut memory,
        )?;
        let actual =
            assessor.download_output(&output_bank.outputs()[0], selected.pool, &mut memory)?;
        let expected = Tensor::splat([ELEMENTS], 1.0 + UNIFORM_VALUE)?;
        if actual != expected {
            return Err(format!("{name} output differs from the expected dense result").into());
        }

        group.bench_function(BenchmarkId::new(name, ELEMENTS), |bencher| {
            bencher.iter(|| {
                assessor
                    .execute_prepared_outputs_into_bank(
                        &prepared,
                        black_box(&persistent_inputs),
                        &mut scratch,
                        &mut output_bank,
                        &mut memory,
                    )
                    .expect("warm Uniform tensor execution failed");
            });
        });
    }
    group.finish();
    Ok(())
}

criterion_group! {
    name = benches;
    config = support::criterion_config();
    targets = bench
}
criterion_main!(benches);
