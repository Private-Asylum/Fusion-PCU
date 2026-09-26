//! Criterion benchmarks for resident tensor `MatMul` and composed `MatMul` graphs.

mod support;

use std::error::Error;

use std::hint::black_box;

use criterion::{
    criterion_group,
    criterion_main,
    BenchmarkId,
    Criterion,
};
use fusion_pcu::{
    PcuMemoryPoolId,
    PcuMemoryProvider,
    PcuOwnedDispatchMemorySession,
};
use fusion_pcu_rocm::{
    HipRuntime,
    Rocblas,
    RocmDiscovery,
    RocmOwnedDispatchBackend,
    RocmTensorAssessor,
};
use fusion_pcu_tensor::{
    Graph,
    Tensor,
    TensorSynchronousF32MatMulBackend,
};

struct Device {
    candidate: support::selection::Candidate,
    session: RocmOwnedDispatchBackend,
    runtime: HipRuntime,
    blas: Rocblas,
}

fn open_device(discovery: &RocmDiscovery) -> Result<Device, Box<dyn Error>> {
    let mut failures = Vec::new();
    for candidate in support::selected_candidates(discovery)? {
        let candidate_name = candidate.name.clone();
        let result = (|| {
            let session = RocmOwnedDispatchBackend::open(discovery, candidate.device, 64)?;
            let runtime = discovery.open_device(candidate.device)?;
            let blas = Rocblas::new(&runtime)?;
            Ok::<_, Box<dyn Error>>(Device {
                candidate,
                session,
                runtime,
                blas,
            })
        })();
        match result {
            Ok(device) => return Ok(device),
            Err(error) => failures.push(format!("{candidate_name}: {error}")),
        }
    }
    Err(format!(
        "no ROCm device supports both tensor routes: {}",
        failures.join("; ")
    )
    .into())
}

fn benchmarks(criterion: &mut Criterion) {
    if let Err(error) = run(criterion) {
        panic!("tensor benchmark setup failed: {error}");
    }
}

fn run(criterion: &mut Criterion) -> Result<(), Box<dyn Error>> {
    let discovery = RocmDiscovery::new();
    let device = open_device(&discovery)?;
    let assessor = RocmTensorAssessor::new(&device.session)?;
    println!(
        "Device: {}; architecture: {}",
        device.candidate.name,
        device
            .candidate
            .architecture
            .as_deref()
            .unwrap_or("unavailable")
    );
    resident_matmul(criterion, &device, &assessor)?;
    graph_matmul(criterion, &device, &assessor)?;
    graph_elementwise_batch(criterion, &device, &assessor)?;
    Ok(())
}

#[allow(clippy::significant_drop_tightening)]
fn resident_matmul(
    criterion: &mut Criterion,
    device: &Device,
    assessor: &RocmTensorAssessor<'_>,
) -> Result<(), Box<dyn Error>> {
    let mut group = criterion.benchmark_group("resident_matmul");
    for size in [8_usize, 1024] {
        let elements = size.checked_mul(size).ok_or("matrix size overflow")?;
        let bytes = elements
            .checked_mul(size_of::<f32>())
            .ok_or("matrix byte size overflow")?;
        let left = vec![2.0_f32; elements];
        let right = vec![3.0_f32; elements];
        let mut memory =
            PcuOwnedDispatchMemorySession::memory_provider(&device.session, device.candidate.pool);
        let mut pcu_left = memory
            .allocate(request(device.candidate.pool, bytes))
            .map_err(|error| format!("PCU allocation failed: {error:?}"))?;
        let mut pcu_right = memory
            .allocate(request(device.candidate.pool, bytes))
            .map_err(|error| format!("PCU allocation failed: {error:?}"))?;
        let pcu_output = memory
            .allocate(request(device.candidate.pool, bytes))
            .map_err(|error| format!("PCU allocation failed: {error:?}"))?;
        memory
            .transfer_to(&mut pcu_left, 0, bytemuck::cast_slice(&left))
            .map_err(|error| format!("PCU upload failed: {error:?}"))?;
        memory
            .transfer_to(&mut pcu_right, 0, bytemuck::cast_slice(&right))
            .map_err(|error| format!("PCU upload failed: {error:?}"))?;
        let mut native_left = device.runtime.allocate(bytes)?;
        let mut native_right = device.runtime.allocate(bytes)?;
        let native_output = device.runtime.allocate(bytes)?;
        native_left.copy_from(bytemuck::cast_slice(&left))?;
        native_right.copy_from(bytemuck::cast_slice(&right))?;
        support::cold_once(&format!("{size}x{size} PCU resident MatMul check"), || {
            assessor.matmul_row_major(&pcu_left, &pcu_right, &pcu_output, size, size, size)
        })?;
        support::cold_once(
            &format!("{size}x{size} direct resident MatMul check"),
            || {
                device.blas.sgemm(
                    false,
                    false,
                    size,
                    size,
                    size,
                    1.0,
                    &native_right,
                    size,
                    &native_left,
                    size,
                    0.0,
                    &native_output,
                    size,
                )
            },
        )?;
        verify_resident(size, &mut memory, &pcu_output, &native_output)?;

        group.bench_function(BenchmarkId::new("PCU", size), |b| {
            b.iter(|| {
                assessor
                    .matmul_row_major(
                        black_box(&pcu_left),
                        black_box(&pcu_right),
                        black_box(&pcu_output),
                        size,
                        size,
                        size,
                    )
                    .expect("PCU MatMul failed");
            });
        });
        group.bench_function(BenchmarkId::new("direct_rocBLAS", size), |b| {
            b.iter(|| {
                device
                    .blas
                    .sgemm(
                        false,
                        false,
                        size,
                        size,
                        size,
                        1.0,
                        &native_right,
                        size,
                        &native_left,
                        size,
                        0.0,
                        &native_output,
                        size,
                    )
                    .expect("rocBLAS SGEMM failed");
            });
        });
    }
    group.finish();
    Ok(())
}

#[allow(clippy::significant_drop_tightening)]
fn graph_matmul(
    criterion: &mut Criterion,
    device: &Device,
    assessor: &RocmTensorAssessor<'_>,
) -> Result<(), Box<dyn Error>> {
    let mut group = criterion.benchmark_group("two_matmul_graph_end_to_end");
    for size in [8_usize, 256] {
        let bytes = size
            .checked_mul(size)
            .and_then(|n| n.checked_mul(size_of::<f32>()))
            .ok_or("matrix size overflow")?;
        let (a_values, b_values, c_values) = graph_inputs(size);
        let mut graph = Graph::default();
        let a = graph.input([size, size])?;
        let b = graph.input([size, size])?;
        let c = graph.input([size, size])?;
        let first = graph.matmul(a, b)?;
        let output = graph.matmul(first, c)?;
        let inputs = [
            (a, Tensor::new([size, size], a_values.clone())?),
            (b, Tensor::new([size, size], b_values.clone())?),
            (c, Tensor::new([size, size], c_values.clone())?),
        ];
        let reference = graph.evaluate(&inputs)?.value(output)?.clone();
        let mut memory =
            PcuOwnedDispatchMemorySession::memory_provider(&device.session, device.candidate.pool);
        let prepared = support::cold_once(&format!("{size}x{size} graph preparation"), || {
            assessor.prepare_graph(&graph, output)
        })?;
        let pcu_check = support::cold_once(&format!("{size}x{size} PCU cold execution"), || {
            assessor.execute_graph(&graph, &inputs, output, device.candidate.pool, &mut memory)
        })?;
        let native_check = native_graph(
            size,
            bytes,
            &a_values,
            &b_values,
            &c_values,
            &device.runtime,
            &device.blas,
        )?;
        verify_tensor(&reference, &pcu_check)?;
        verify_tensor(&reference, &native_check)?;

        group.bench_function(BenchmarkId::new("PCU", size), |bench| {
            bench.iter(|| {
                black_box(
                    assessor
                        .execute_graph(&graph, &inputs, output, device.candidate.pool, &mut memory)
                        .expect("PCU graph execution failed"),
                );
            });
        });
        group.bench_function(BenchmarkId::new("PCU_prepared", size), |bench| {
            bench.iter(|| {
                black_box(
                    assessor
                        .execute_prepared(&prepared, &inputs, device.candidate.pool, &mut memory)
                        .expect("prepared PCU graph execution failed"),
                );
            });
        });
        group.bench_function(BenchmarkId::new("direct_rocBLAS", size), |bench| {
            bench.iter(|| {
                black_box(
                    native_graph(
                        size,
                        bytes,
                        &a_values,
                        &b_values,
                        &c_values,
                        &device.runtime,
                        &device.blas,
                    )
                    .expect("direct graph execution failed"),
                );
            });
        });
    }
    group.finish();
    Ok(())
}

#[allow(clippy::significant_drop_tightening)]
fn graph_elementwise_batch(
    criterion: &mut Criterion,
    device: &Device,
    assessor: &RocmTensorAssessor<'_>,
) -> Result<(), Box<dyn Error>> {
    let mut group = criterion.benchmark_group("prepared_add_relu_matmul_batch");
    for size in [8_usize, 256] {
        let (a_values, b_values, c_values) = graph_inputs(size);
        let mut graph = Graph::default();
        let a = graph.input([size, size])?;
        let b = graph.input([size, size])?;
        let bias = graph.input([size, size])?;
        let c = graph.input([size, size])?;
        let product = graph.matmul(a, b)?;
        let biased = graph.add(product, bias)?;
        let activated = graph.relu(biased)?;
        let output = graph.matmul(activated, c)?;
        let inputs = [
            (a, Tensor::new([size, size], a_values.clone())?),
            (b, Tensor::new([size, size], b_values.clone())?),
            (
                bias,
                Tensor::new([size, size], vec![0.25_f32; size * size])?,
            ),
            (c, Tensor::new([size, size], c_values.clone())?),
        ];
        let reference = graph.evaluate(&inputs)?.value(output)?.clone();
        let prepared =
            support::cold_once(&format!("{size}x{size} batched graph preparation"), || {
                assessor.prepare_graph(&graph, output)
            })?;
        let mut memory =
            PcuOwnedDispatchMemorySession::memory_provider(&device.session, device.candidate.pool);
        for (mode, result) in [
            (
                "sync",
                assessor.execute_prepared(&prepared, &inputs, device.candidate.pool, &mut memory),
            ),
            (
                "batched",
                assessor.execute_prepared_batched(
                    &prepared,
                    &inputs,
                    device.candidate.pool,
                    &mut memory,
                ),
            ),
        ] {
            let result = result?;
            verify_tensor(&reference, &result)
                .map_err(|error| format!("{mode} graph parity failed: {error}"))?;
        }

        group.bench_function(BenchmarkId::new("PCU_prepared_sync", size), |bench| {
            bench.iter(|| {
                black_box(
                    assessor
                        .execute_prepared(&prepared, &inputs, device.candidate.pool, &mut memory)
                        .expect("synchronous prepared graph failed"),
                );
            });
        });
        group.bench_function(BenchmarkId::new("PCU_prepared_batched", size), |bench| {
            bench.iter(|| {
                black_box(
                    assessor
                        .execute_prepared_batched(
                            &prepared,
                            &inputs,
                            device.candidate.pool,
                            &mut memory,
                        )
                        .expect("batched prepared graph failed"),
                );
            });
        });
    }
    group.finish();
    Ok(())
}

fn native_graph(
    size: usize,
    bytes: usize,
    a: &[f32],
    b: &[f32],
    c: &[f32],
    runtime: &HipRuntime,
    blas: &Rocblas,
) -> Result<Tensor, Box<dyn Error>> {
    let mut da = runtime.allocate(bytes)?;
    da.copy_from(bytemuck::cast_slice(a))?;
    let mut db = runtime.allocate(bytes)?;
    db.copy_from(bytemuck::cast_slice(b))?;
    let mut dc = runtime.allocate(bytes)?;
    dc.copy_from(bytemuck::cast_slice(c))?;
    let intermediate = runtime.allocate(bytes)?;
    blas.sgemm(
        false,
        false,
        size,
        size,
        size,
        1.0,
        &db,
        size,
        &da,
        size,
        0.0,
        &intermediate,
        size,
    )?;
    drop(da);
    drop(db);
    let output = runtime.allocate(bytes)?;
    blas.sgemm(
        false,
        false,
        size,
        size,
        size,
        1.0,
        &dc,
        size,
        &intermediate,
        size,
        0.0,
        &output,
        size,
    )?;
    drop(intermediate);
    drop(dc);
    let mut result = vec![0.0_f32; size * size];
    output.copy_to(bytemuck::cast_slice_mut(&mut result))?;
    Ok(Tensor::new([size, size], result)?)
}

fn graph_inputs(size: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut a = Vec::with_capacity(size * size);
    let mut b = Vec::with_capacity(size * size);
    let mut c = Vec::with_capacity(size * size);
    for row in 0..size {
        for col in 0..size {
            a.push(f32::from(
                u8::try_from(row % 3 + 1).expect("factor fits u8"),
            ));
            b.push(f32::from(
                u8::try_from(col % 3 + 1).expect("factor fits u8"),
            ));
            c.push(f32::from(
                u8::try_from((row + col) % 3 + 1).expect("factor fits u8"),
            ));
        }
    }
    (a, b, c)
}

fn verify_tensor(expected: &Tensor, actual: &Tensor) -> Result<(), Box<dyn Error>> {
    if expected.shape() != actual.shape()
        || expected
            .data()
            .iter()
            .zip(actual.data())
            .any(|(a, b)| a.to_bits() != b.to_bits())
    {
        return Err("tensor output mismatch".into());
    }
    Ok(())
}

fn verify_resident(
    size: usize,
    memory: &mut impl fusion_pcu::PcuMemoryProvider<Resource = fusion_pcu_rocm::RocmMemoryResource>,
    pcu: &fusion_pcu_rocm::RocmMemoryResource,
    native: &fusion_pcu_rocm::DeviceBuffer,
) -> Result<(), Box<dyn Error>> {
    let bytes = size * size * size_of::<f32>();
    let mut pcu_data = vec![0_u8; bytes];
    memory
        .transfer_from(pcu, 0, &mut pcu_data)
        .map_err(|error| format!("PCU readback failed: {error:?}"))?;
    let mut native_data = vec![0_u8; bytes];
    native.copy_to(&mut native_data)?;
    let expected = f32::from(u16::try_from(size * 6).expect("benchmark output fits u16"));
    for (index, (pcu_bytes, native_bytes)) in pcu_data
        .chunks_exact(4)
        .zip(native_data.chunks_exact(4))
        .enumerate()
    {
        let pcu_value = f32::from_ne_bytes(pcu_bytes.try_into()?);
        let native_value = f32::from_ne_bytes(native_bytes.try_into()?);
        if pcu_value.to_bits() != expected.to_bits() || native_value.to_bits() != expected.to_bits()
        {
            return Err(format!("resident output[{index}] mismatch: PCU {pcu_value}, native {native_value}, expected {expected}").into());
        }
    }
    Ok(())
}

const fn request(pool: PcuMemoryPoolId, bytes: usize) -> fusion_pcu::PcuMemoryAllocationRequest {
    fusion_pcu::PcuMemoryAllocationRequest {
        pool,
        size_bytes: bytes as u64,
        alignment_bytes: 4,
        access: fusion_pcu::PcuMemoryAccess::ReadWrite,
        host_access: fusion_pcu::PcuMemoryHostAccess::TransferOnly,
        require_device_local: false,
    }
}

criterion_group! { name = benches; config = support::criterion_config(); targets = benchmarks }
criterion_main!(benches);
