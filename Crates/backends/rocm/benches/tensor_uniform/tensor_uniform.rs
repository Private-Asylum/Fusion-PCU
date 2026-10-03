//! Genuine broadcast source and graph constant/Uniform materialization controls.

extern crate pcu_facade as fusion_pcu;

#[cfg(feature = "allocation-census")]
#[path = "../support/alloc.rs"]
mod alloc;
#[path = "driver/driver.rs"]
mod driver;
#[path = "source/source.rs"]
mod source;

#[path = "../support/support.rs"]
mod support;

#[path = "../strict_matmul/activity.rs"]
mod activity;

#[rustfmt::skip]
use std::{
    error::Error,
    hint::black_box,
};

#[rustfmt::skip]
use criterion::{
    BenchmarkId,
    Criterion,
    Throughput,
    criterion_group,
    criterion_main,
};
#[rustfmt::skip]
use fusion_pcu::{
    PcuOwnedDispatchMemorySession,
};
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
};

#[path = "../support/memory_profile.rs"]
mod memory_profile;

const ELEMENTS: usize = 1_048_576;
const UNIFORM_VALUE: f32 = 0.25;

fn bench(criterion: &mut Criterion) {
    if !std::env::args().any(|arg| arg == "--test") {
        activity::guard();
    }
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
    driver::configure(&session);
    driver::case::<65>(criterion, &session);
    driver::case::<ELEMENTS>(criterion, &session);
    let assessor = RocmTensorAssessor::new(&session)?;
    let mut group = criterion.benchmark_group("tensor_uniform_1m");
    group.throughput(Throughput::Elements(ELEMENTS as u64));
    println!(
        "Device: {}; dense constant and graph Uniform storage",
        selected.name
    );

    for (name, uniform) in [("DenseSplatConstant", false), ("GraphUniform", true)] {
        let mut graph = Graph::default();
        let input = graph.input([ELEMENTS], fusion_pcu::PcuScalarType::F32)?;
        let value = if uniform {
            graph.uniform_value(
                [ELEMENTS],
                fusion_pcu::dialect::tensor::TensorScalarValue::F32(UNIFORM_VALUE),
            )?
        } else {
            graph.constant_value(fusion_pcu::dialect::tensor::TensorValue::F32(
                Tensor::splat([ELEMENTS], UNIFORM_VALUE)?,
            ))
        };
        let output = graph.add(input, value)?;
        let prepared = assessor.prepare_graph_outputs(&graph, &[output])?;

        let host_input = Tensor::splat([ELEMENTS], 1.0)?;
        let mut input_provider =
            PcuOwnedDispatchMemorySession::memory_provider(&session, selected.pool);
        let device_inputs = [
            assessor.upload_input(&host_input, selected.pool, &mut input_provider)?,
            assessor.upload_input(
                &Tensor::splat([ELEMENTS], 2.0)?,
                selected.pool,
                &mut input_provider,
            )?,
        ];
        let mut memory = memory_profile::ProfiledMemory::new(input_provider);

        let (mut scratch, mut output_bank) =
            support::cold_once(&format!("{name} cold scratch/output preparation"), || {
                Ok::<_, fusion_pcu_rocm::RocmTensorExecutionError>((
                    assessor.prepare_scratch(&prepared, selected.pool, &mut memory)?,
                    assessor.prepare_output_bank(&prepared, selected.pool, &mut memory)?,
                ))
            })?;
        print_profile(name, &memory.profile());

        let persistent_inputs = device_inputs.each_ref().map(|device| [(input, device)]);
        for (bank, inputs) in persistent_inputs.iter().enumerate() {
            assessor.execute_prepared_outputs_into_bank(
                &prepared,
                inputs,
                &mut scratch,
                &mut output_bank,
                &mut memory,
            )?;
            let actual =
                assessor.download_output(&output_bank.outputs()[0], selected.pool, &mut memory)?;
            let marker = if bank == 0 { 1.0 } else { 2.0 };
            if actual != Tensor::splat([ELEMENTS], marker + UNIFORM_VALUE)? {
                return Err(format!("{name} changing-bank oracle failed").into());
            }
        }
        #[cfg(feature = "allocation-census")]
        {
            census(name, |bank| {
                assessor.execute_prepared_outputs_into_bank(
                    &prepared,
                    &persistent_inputs[bank],
                    &mut scratch,
                    &mut output_bank,
                    &mut memory,
                )
            })?;
            let actual =
                assessor.download_output(&output_bank.outputs()[0], selected.pool, &mut memory)?;
            assert_eq!(actual, Tensor::splat([ELEMENTS], 2.0 + UNIFORM_VALUE)?);
        }
        let mut phase = 0;
        group.bench_function(BenchmarkId::new(name, ELEMENTS), |bencher| {
            bencher.iter(|| {
                phase ^= 1;
                assessor
                    .execute_prepared_outputs_into_bank(
                        &prepared,
                        black_box(&persistent_inputs[phase]),
                        &mut scratch,
                        &mut output_bank,
                        &mut memory,
                    )
                    .expect("warm Uniform tensor execution failed");
            });
        });
        let marker = if phase == 0 { 1.0 } else { 2.0 };
        let actual =
            assessor.download_output(&output_bank.outputs()[0], selected.pool, &mut memory)?;
        assert_eq!(actual, Tensor::splat([ELEMENTS], marker + UNIFORM_VALUE)?);
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

#[cfg(feature = "allocation-census")]
fn census(
    name: &str,
    mut execute: impl FnMut(usize) -> Result<(), fusion_pcu_rocm::RocmTensorExecutionError>,
) -> Result<(), fusion_pcu_rocm::RocmTensorExecutionError> {
    fusion_pcu_rocm::reset_rocm_api_census();
    let capture = alloc::AllocationCapture::start();
    let before = alloc::AllocationCapture::snapshot();
    for iteration in 0..64 {
        execute(iteration % 2)?;
    }
    let rust = alloc::AllocationCapture::finish().since(before);
    drop(capture);
    let api = fusion_pcu_rocm::rocm_api_census();
    assert_eq!(api.symbol_resolutions, 0);
    assert_eq!(api.module_loads, 0);
    eprintln!(
        "census/tensor_uniform/legacy_graph/{name}/64-changing-calls: Rust alloc={} realloc={} frees={} bytes={}; API={api:?}",
        rust.alloc_calls, rust.realloc_calls, rust.dealloc_calls, rust.requested_bytes
    );
    Ok(())
}

fn print_profile(name: &str, profile: &memory_profile::MemoryProfile) {
    println!(
        "{name} cold provider profile: allocations={} ({} bytes), uploads={} ({} bytes), provider allocation={:?}, upload={:?}",
        profile.allocations,
        profile.allocated_bytes,
        profile.uploads,
        profile.uploaded_bytes,
        profile.allocation_time,
        profile.upload_time
    );
}
