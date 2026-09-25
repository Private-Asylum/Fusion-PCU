//! Compare prepared PCU graphs with reusable device inputs against direct rocBLAS.

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
    DeviceBuffer,
    HipRuntime,
    Rocblas,
    RocmDiscovery,
    RocmOwnedDispatchBackend,
    RocmTensorAssessor,
};
use fusion_pcu_tensor::{
    Graph,
    Tensor,
};

fn native_matmul(
    runtime: &HipRuntime,
    blas: &Rocblas,
    left: &DeviceBuffer,
    right: &DeviceBuffer,
    size: usize,
) -> Result<Tensor, Box<dyn Error>> {
    let elements = size.checked_mul(size).ok_or("matrix size overflow")?;
    let bytes = elements
        .checked_mul(size_of::<f32>())
        .ok_or("matrix byte size overflow")?;
    let output = runtime.allocate(bytes)?;
    // Row-major A * B is column-major B^T * A^T in the same contiguous storage.
    blas.sgemm(
        false, false, size, size, size, 1.0, right, size, left, size, 0.0, &output, size,
    )?;
    let mut values = vec![0.0_f32; elements];
    output.copy_to(bytemuck::cast_slice_mut(&mut values))?;
    Ok(Tensor::new([size, size], values)?)
}

fn bench(criterion: &mut Criterion) {
    run(criterion).expect("resident-input ROCm benchmark failed");
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
        "no ROCm device completed resident-input benchmark: {}",
        failures.join("; ")
    )
    .into())
}

fn run_on(
    criterion: &mut Criterion,
    discovery: &RocmDiscovery,
    candidate: &support::selection::Candidate,
) -> Result<(), Box<dyn Error>> {
    let session = RocmOwnedDispatchBackend::open(discovery, candidate.device, 64)?;
    let assessor = RocmTensorAssessor::new(&session)?;
    let runtime = discovery.open_device(candidate.device)?;
    let blas = Rocblas::new(&runtime)?;
    println!("Device: {}; resident-input MatMul", candidate.name);
    for size in [8_usize, 1024] {
        let elements = size.checked_mul(size).ok_or("matrix size overflow")?;
        let bytes = elements
            .checked_mul(size_of::<f32>())
            .ok_or("matrix byte size overflow")?;
        let left = Tensor::new([size, size], vec![2.0_f32; elements])?;
        let right = Tensor::new([size, size], vec![3.0_f32; elements])?;
        let mut graph = Graph::default();
        let left_id = graph.input([size, size])?;
        let right_id = graph.input([size, size])?;
        let output_id = graph.matmul(left_id, right_id)?;
        let prepared = assessor.prepare_graph(&graph, output_id)?;
        let host_inputs = [(left_id, left.clone()), (right_id, right.clone())];
        let reference = graph.evaluate(&host_inputs)?.value(output_id)?.clone();
        let mut memory = PcuOwnedDispatchMemorySession::memory_provider(&session, candidate.pool);
        let device_left = support::cold_once(&format!("{size}x{size} PCU left upload"), || {
            assessor.upload_input(&left, candidate.pool, &mut memory)
        })?;
        let device_right = support::cold_once(&format!("{size}x{size} PCU right upload"), || {
            assessor.upload_input(&right, candidate.pool, &mut memory)
        })?;
        let resident_inputs = [(left_id, &device_left), (right_id, &device_right)];
        let mut native_left = runtime.allocate(bytes)?;
        let mut native_right = runtime.allocate(bytes)?;
        native_left.copy_from(bytemuck::cast_slice(left.data()))?;
        native_right.copy_from(bytemuck::cast_slice(right.data()))?;
        let pcu_cold = support::cold_once(&format!("{size}x{size} PCU resident cold"), || {
            assessor.execute_prepared_with_resources(
                &prepared,
                &resident_inputs,
                candidate.pool,
                &mut memory,
            )
        })?;
        let native_cold =
            support::cold_once(&format!("{size}x{size} native resident cold"), || {
                native_matmul(&runtime, &blas, &native_left, &native_right, size)
            })?;
        verify(&reference, &pcu_cold)?;
        verify(&reference, &native_cold)?;
        {
            let mut group = criterion.benchmark_group("resident_input_matmul");
            group.throughput(Throughput::Elements(u64::try_from(elements)?));
            group.bench_function(BenchmarkId::new("pcu_resident", size), |bencher| {
                bencher.iter(|| {
                    black_box(
                        assessor
                            .execute_prepared_with_resources(
                                &prepared,
                                &resident_inputs,
                                candidate.pool,
                                &mut memory,
                            )
                            .expect("resident PCU MatMul failed"),
                    );
                });
            });
            group.bench_function(BenchmarkId::new("pcu_host_inputs", size), |bencher| {
                bencher.iter(|| {
                    black_box(
                        assessor
                            .execute_prepared(&prepared, &host_inputs, candidate.pool, &mut memory)
                            .expect("host-input PCU MatMul failed"),
                    );
                });
            });
            group.bench_function(BenchmarkId::new("native_rocblas", size), |bencher| {
                bencher.iter(|| {
                    black_box(
                        native_matmul(&runtime, &blas, &native_left, &native_right, size)
                            .expect("resident native MatMul failed"),
                    );
                });
            });
            group.finish();
        }
        verify(
            &reference,
            &assessor.execute_prepared_with_resources(
                &prepared,
                &resident_inputs,
                candidate.pool,
                &mut memory,
            )?,
        )?;
        verify(
            &reference,
            &native_matmul(&runtime, &blas, &native_left, &native_right, size)?,
        )?;
    }
    Ok(())
}

fn verify(expected: &Tensor, actual: &Tensor) -> Result<(), Box<dyn Error>> {
    if expected.shape() == actual.shape()
        && expected.data().len() == actual.data().len()
        && expected
            .data()
            .iter()
            .zip(actual.data())
            .all(|(expected, actual)| (expected - actual).abs() <= 1.0e-3)
    {
        Ok(())
    } else {
        Err("resident MatMul output mismatch".into())
    }
}

criterion_group! {
    name = benches;
    config = support::criterion_config();
    targets = bench
}
criterion_main!(benches);
