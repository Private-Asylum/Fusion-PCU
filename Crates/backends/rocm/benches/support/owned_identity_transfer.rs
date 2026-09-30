//! Consuming identity transfer against raw assessor and checked native ownership handoffs.

#[path = "alloc.rs"]
#[allow(dead_code)]
mod alloc;

#[rustfmt::skip]
use std::{
    error::Error,
    rc::Rc,
    time::Duration,
};
#[rustfmt::skip]
use criterion::{
    BenchmarkId,
    Criterion,
    Throughput,
};
#[rustfmt::skip]
use fusion_pcu::{
    PcuDeviceTensor,
    PcuExecutionError,
    PcuMemoryPoolId,
    PcuTensor,
};
#[rustfmt::skip]
use fusion_pcu::dialect::tensor::{
    Graph,
    TensorArithmeticCapability,
    TensorArithmeticRewritePolicy,
    TensorPointwiseGroupingPolicy,
};
#[rustfmt::skip]
use fusion_pcu_rocm::{
    RocmDiscovery,
    RocmMemoryResource,
    RocmOwnedDispatchBackend,
    RocmTensorAssessor,
    RocmOwnedTensorAssessor,
};

const SAMPLES: usize = 24;
const ROUTE_ORDERS: [[usize; 3]; 6] = [
    [0, 1, 2],
    [0, 2, 1],
    [1, 0, 2],
    [1, 2, 0],
    [2, 0, 1],
    [2, 1, 0],
];

#[fusion_pcu::pcu]
fn source_identity_consuming(input: PcuTensor<f32>) -> Result<PcuTensor<f32>, PcuExecutionError> {
    Ok(pcu::identity(input)?)
}

#[fusion_pcu::pcu]
fn source_identity_borrowed(input: &[f32]) -> Result<PcuTensor<f32>, PcuExecutionError> {
    pcu::identity(input)
}

#[fusion_pcu::pcu(invocations: N)]
fn source_refresh<const N: usize>(input: &[f32], output: &mut [f32]) {
    let id = pcu::context::global_invocation_id();
    output[id] = input[id];
}

fn refresh_source_for_extent(
    elements: usize,
    input: &[f32],
    output: &mut PcuTensor<f32>,
) -> Result<(), PcuExecutionError> {
    match elements {
        65 => source_refresh::<65>(input, output),
        1_048_576 => source_refresh::<1_048_576>(input, output),
        _ => unreachable!("benchmark declares both extents"),
    }
}

pub fn run(criterion: &mut Criterion) -> Result<(), Box<dyn Error>> {
    let discovery = RocmDiscovery::new();
    let candidates = crate::support::selected_candidates(&discovery)?
        .into_iter()
        .filter(|candidate| candidate.architecture.is_some())
        .collect();
    let (backend, selected) = crate::support::selection::open_ranked(&discovery, candidates, 256)?;
    let backend = Rc::new(backend);
    fusion_pcu::global::configure(fusion_pcu::global::PcuExecutionPolicy {
        backend: fusion_pcu::global::PcuBackendChoice::Rocm,
        device: Some(selected.device.id),
        ..fusion_pcu::global::PcuExecutionPolicy::default()
    })?;
    println!(
        "Owned identity-transfer benchmark device: {}",
        selected.name
    );

    for elements in [65, 1_048_576] {
        run_case(criterion, &backend, selected.pool, elements)?;
    }
    Ok(())
}

#[allow(clippy::too_many_lines)]
#[allow(clippy::significant_drop_tightening)] // Criterion finish consumes the group after all three routes.
fn run_case(
    criterion: &mut Criterion,
    backend: &Rc<RocmOwnedDispatchBackend>,
    pool: PcuMemoryPoolId,
    elements: usize,
) -> Result<(), Box<dyn Error>> {
    let assessor_root = RocmOwnedTensorAssessor::new(Rc::clone(backend))?;
    let assessor = assessor_root.assessor();
    let mut graph = Graph::default();
    let input_id = graph.input([elements], fusion_pcu::PcuScalarType::F32)?;
    let program = graph.into_selected_program(
        &[input_id],
        TensorArithmeticRewritePolicy::Disabled,
        TensorArithmeticCapability::Strict,
        TensorPointwiseGroupingPolicy::Disabled,
    )?;
    let prepared = crate::support::cold_once("identity-transfer graph preparation", || {
        assessor.prepare_owned_program(program)
    })?;
    let mut memory = backend.memory_provider(pool);

    let mut host = vec![0.0_f32; elements];
    let mut observed = vec![0.0_f32; elements];
    fill(&mut host, 0);
    let mut source_owner = Some(source_identity_borrowed(&host)?);
    let mut raw_owner = Some(upload_owner(backend, pool, elements, &host)?);
    let mut native_owner = Some(upload_owner(backend, pool, elements, &host)?);

    // Exercise all three paths cold before collecting timings, including exact-bit readback.
    let cold_source = source_identity_consuming(source_owner.take().expect("source owner"))?;
    source_owner = Some(cold_source);
    read_source(source_owner.as_ref().expect("source owner"), &mut observed)?;
    verify(&host, &observed);

    raw_owner = Some(assessor.execute_owned_program_consuming_input(
        &prepared,
        raw_owner.take().expect("raw owner"),
        pool,
        &mut memory,
    )?);
    read_device(
        backend,
        pool,
        raw_owner.as_ref().expect("raw owner"),
        &mut observed,
    )?;
    verify(&host, &observed);

    native_owner = Some(checked_native_handoff(
        native_owner.take().expect("native owner"),
    )?);
    read_device(
        backend,
        pool,
        native_owner.as_ref().expect("native owner"),
        &mut observed,
    )?;
    verify(&host, &observed);

    print_allocation_census(
        &prepared,
        &assessor,
        pool,
        &mut source_owner,
        &mut raw_owner,
        &mut native_owner,
        &mut memory,
    )?;
    let paired = paired_diagnostic(
        SAMPLES,
        elements,
        &mut host,
        &mut observed,
        backend,
        &prepared,
        &assessor,
        pool,
        &mut source_owner,
        &mut raw_owner,
        &mut native_owner,
        &mut memory,
    )?;
    print_paired_medians(elements, &paired);

    let mut group = criterion.benchmark_group("owned_identity_transfer");
    group.throughput(Throughput::Elements(1));
    group.bench_function(BenchmarkId::new("source_consuming", elements), |bencher| {
        bencher.iter_custom(|iterations| {
            let mut total = Duration::ZERO;
            for iteration in 0..iterations {
                fill(&mut host, iteration.wrapping_add(1));
                refresh_all_inputs(
                    elements,
                    &host,
                    &mut source_owner,
                    &mut raw_owner,
                    &mut native_owner,
                    backend,
                    pool,
                )
                .expect("refresh all identity-transfer inputs");
                let input = source_owner.take().expect("source owner");
                let start = std::time::Instant::now();
                let output = source_identity_consuming(input).expect("source identity transfer");
                total += start.elapsed();
                read_source(&output, &mut observed).expect("source identity readback");
                verify(&host, &observed);
                source_owner = Some(output);
            }
            total
        });
    });
    group.bench_function(
        BenchmarkId::new("raw_assessor_consuming", elements),
        |bencher| {
            bencher.iter_custom(|iterations| {
                let mut total = Duration::ZERO;
                for iteration in 0..iterations {
                    fill(&mut host, iteration.wrapping_add(1));
                    refresh_all_inputs(
                        elements,
                        &host,
                        &mut source_owner,
                        &mut raw_owner,
                        &mut native_owner,
                        backend,
                        pool,
                    )
                    .expect("refresh all identity-transfer inputs");
                    let input = raw_owner.take().expect("raw owner");
                    let start = std::time::Instant::now();
                    let output = assessor
                        .execute_owned_program_consuming_input(&prepared, input, pool, &mut memory)
                        .expect("raw identity transfer");
                    total += start.elapsed();
                    read_device(backend, pool, &output, &mut observed)
                        .expect("raw identity readback");
                    verify(&host, &observed);
                    raw_owner = Some(output);
                }
                total
            });
        },
    );
    group.bench_function(
        BenchmarkId::new("native_checked_handoff", elements),
        |bencher| {
            bencher.iter_custom(|iterations| {
                let mut total = Duration::ZERO;
                for iteration in 0..iterations {
                    fill(&mut host, iteration.wrapping_add(1));
                    refresh_all_inputs(
                        elements,
                        &host,
                        &mut source_owner,
                        &mut raw_owner,
                        &mut native_owner,
                        backend,
                        pool,
                    )
                    .expect("refresh all identity-transfer inputs");
                    let input = native_owner.take().expect("native owner");
                    let start = std::time::Instant::now();
                    let output = checked_native_handoff(input).expect("native checked handoff");
                    total += start.elapsed();
                    read_device(backend, pool, &output, &mut observed)
                        .expect("native handoff readback");
                    verify(&host, &observed);
                    native_owner = Some(output);
                }
                total
            });
        },
    );
    group.finish();
    Ok(())
}

fn upload_owner(
    backend: &RocmOwnedDispatchBackend,
    pool: PcuMemoryPoolId,
    elements: usize,
    values: &[f32],
) -> Result<PcuDeviceTensor<f32, RocmMemoryResource>, Box<dyn Error>> {
    Ok(PcuDeviceTensor::new(
        [elements],
        backend.upload_buffer(pool, values)?,
    )?)
}

fn refresh_device_owner(
    backend: &RocmOwnedDispatchBackend,
    pool: PcuMemoryPoolId,
    elements: usize,
    values: &[f32],
    owner: &mut Option<PcuDeviceTensor<f32, RocmMemoryResource>>,
) -> Result<(), Box<dyn Error>> {
    let mut buffer = owner.take().expect("device owner").into_buffer();
    backend.refresh_buffer(pool, &mut buffer, values)?;
    *owner = Some(PcuDeviceTensor::new([elements], buffer)?);
    Ok(())
}

fn refresh_all_inputs(
    elements: usize,
    values: &[f32],
    source_owner: &mut Option<PcuTensor<f32>>,
    raw_owner: &mut Option<PcuDeviceTensor<f32, RocmMemoryResource>>,
    native_owner: &mut Option<PcuDeviceTensor<f32, RocmMemoryResource>>,
    backend: &RocmOwnedDispatchBackend,
    pool: PcuMemoryPoolId,
) -> Result<(), Box<dyn Error>> {
    refresh_source_for_extent(
        elements,
        values,
        source_owner.as_mut().expect("source owner"),
    )?;
    refresh_device_owner(backend, pool, elements, values, raw_owner)?;
    refresh_device_owner(backend, pool, elements, values, native_owner)?;
    Ok(())
}

fn checked_native_handoff(
    owner: PcuDeviceTensor<f32, RocmMemoryResource>,
) -> Result<PcuDeviceTensor<f32, RocmMemoryResource>, fusion_pcu_rocm::HipError> {
    owner.buffer().resource().validate_access_available()?;
    Ok(owner)
}

fn read_source(owner: &PcuTensor<f32>, observed: &mut [f32]) -> Result<(), PcuExecutionError> {
    owner.read_into(observed)
}

fn read_device(
    backend: &RocmOwnedDispatchBackend,
    pool: PcuMemoryPoolId,
    owner: &PcuDeviceTensor<f32, RocmMemoryResource>,
    observed: &mut [f32],
) -> Result<(), Box<dyn Error>> {
    backend.download_buffer(pool, owner.buffer(), observed)?;
    Ok(())
}

fn print_allocation_census(
    prepared: &fusion_pcu_rocm::RocmOwnedPreparedTensorGraph,
    assessor: &RocmTensorAssessor<'_>,
    pool: PcuMemoryPoolId,
    source_owner: &mut Option<PcuTensor<f32>>,
    raw_owner: &mut Option<PcuDeviceTensor<f32, RocmMemoryResource>>,
    native_owner: &mut Option<PcuDeviceTensor<f32, RocmMemoryResource>>,
    memory: &mut impl fusion_pcu::PcuMemoryProvider<Resource = RocmMemoryResource>,
) -> Result<(), Box<dyn Error>> {
    let capture = alloc::AllocationCapture::start();
    let output = source_identity_consuming(source_owner.take().expect("source owner"))?;
    let source = alloc::AllocationCapture::finish();
    drop(capture);
    *source_owner = Some(output);

    let capture = alloc::AllocationCapture::start();
    let output = assessor.execute_owned_program_consuming_input(
        prepared,
        raw_owner.take().expect("raw owner"),
        pool,
        memory,
    )?;
    let raw = alloc::AllocationCapture::finish();
    drop(capture);
    *raw_owner = Some(output);

    let capture = alloc::AllocationCapture::start();
    let output = checked_native_handoff(native_owner.take().expect("native owner"))?;
    let native = alloc::AllocationCapture::finish();
    drop(capture);
    *native_owner = Some(output);

    println!(
        "Owned identity-transfer Rust heap census: source {}/{}/{} B, raw assessor {}/{}/{} B, native checked handoff {}/{}/{} B allocations/reallocations/requested. This is host allocator accounting only; it does not prove whether storage was physically transferred.",
        source.alloc_calls,
        source.realloc_calls,
        source.requested_bytes,
        raw.alloc_calls,
        raw.realloc_calls,
        raw.requested_bytes,
        native.alloc_calls,
        native.realloc_calls,
        native.requested_bytes,
    );
    Ok(())
}

#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn paired_diagnostic(
    samples: usize,
    elements: usize,
    host: &mut [f32],
    observed: &mut [f32],
    backend: &RocmOwnedDispatchBackend,
    prepared: &fusion_pcu_rocm::RocmOwnedPreparedTensorGraph,
    assessor: &RocmTensorAssessor<'_>,
    pool: PcuMemoryPoolId,
    source_owner: &mut Option<PcuTensor<f32>>,
    raw_owner: &mut Option<PcuDeviceTensor<f32, RocmMemoryResource>>,
    native_owner: &mut Option<PcuDeviceTensor<f32, RocmMemoryResource>>,
    memory: &mut impl fusion_pcu::PcuMemoryProvider<Resource = RocmMemoryResource>,
) -> Result<[[Duration; 3]; SAMPLES], Box<dyn Error>> {
    let mut times = [[Duration::ZERO; 3]; SAMPLES];
    for sample in 0..samples {
        fill(host, u64::try_from(sample + 1)?);
        refresh_all_inputs(
            elements,
            host,
            source_owner,
            raw_owner,
            native_owner,
            backend,
            pool,
        )?;
        for route in ROUTE_ORDERS[sample % ROUTE_ORDERS.len()] {
            match route {
                0 => {
                    let input = source_owner.take().expect("source owner");
                    let start = std::time::Instant::now();
                    let output = source_identity_consuming(input)?;
                    times[sample][0] = start.elapsed();
                    read_source(&output, observed)?;
                    verify(host, observed);
                    *source_owner = Some(output);
                }
                1 => {
                    let input = raw_owner.take().expect("raw owner");
                    let start = std::time::Instant::now();
                    let output = assessor
                        .execute_owned_program_consuming_input(prepared, input, pool, memory)?;
                    times[sample][1] = start.elapsed();
                    read_device(backend, pool, &output, observed)?;
                    verify(host, observed);
                    *raw_owner = Some(output);
                }
                2 => {
                    let input = native_owner.take().expect("native owner");
                    let start = std::time::Instant::now();
                    let output = checked_native_handoff(input)?;
                    times[sample][2] = start.elapsed();
                    read_device(backend, pool, &output, observed)?;
                    verify(host, observed);
                    *native_owner = Some(output);
                }
                _ => unreachable!("three identity-transfer routes"),
            }
        }
    }
    Ok(times)
}

fn fill(values: &mut [f32], job: u64) {
    let phase = f32::from(u8::try_from(job % 17).expect("phase fits")) / 16.0;
    for (index, value) in values.iter_mut().enumerate() {
        let base = f32::from(u16::try_from(index % 1024).expect("sample fits"));
        *value = if (index + usize::try_from(job).expect("job fits")) % 2 == 0 {
            base + phase
        } else {
            -base - phase
        };
    }
}

fn verify(input: &[f32], observed: &[f32]) {
    assert_eq!(input.len(), observed.len());
    assert!(
        input
            .iter()
            .zip(observed)
            .all(|(&expected, &actual)| expected.to_bits() == actual.to_bits()),
        "identity transfer changed one or more f32 bit patterns"
    );
}

fn print_paired_medians(elements: usize, times: &[[Duration; 3]; SAMPLES]) {
    let mut source = [Duration::ZERO; SAMPLES];
    let mut raw = [Duration::ZERO; SAMPLES];
    let mut native = [Duration::ZERO; SAMPLES];
    for (index, [source_time, raw_time, native_time]) in times.iter().copied().enumerate() {
        source[index] = source_time;
        raw[index] = raw_time;
        native[index] = native_time;
    }
    source.sort_unstable();
    raw.sort_unstable();
    native.sort_unstable();
    println!(
        "Owned identity-transfer diagnostic shape {elements}, one ownership handoff per call: source {:?}, raw assessor {:?}, native checked handoff {:?}. Inputs refreshed outside timing; readback/oracles outside timing. Absolute paired host medians only, not GPU-work throughput.",
        (source[11] + source[12]) / 2,
        (raw[11] + raw[12]) / 2,
        (native[11] + native[12]) / 2,
    );
}
