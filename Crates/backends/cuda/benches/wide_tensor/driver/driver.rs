//! Resident inputs to freshly allocated resident output, including output release in timing.
#[rustfmt::skip]
use std::{
    error::Error,
    hint::black_box,
    rc::Rc,
    time::Instant,
    sync::atomic::{
        AtomicUsize,
        Ordering,
    },
};
#[cfg(not(feature = "allocation-census"))]
use std::time::Duration;
#[rustfmt::skip]
use criterion::Criterion;
#[cfg(not(feature = "allocation-census"))]
use criterion::Throughput;
#[rustfmt::skip]
use fusion_pcu::{
    global,
    PcuTensor,
    PcuExecutionError,
    PcuDeviceTensor,
    PcuMemoryPoolId,
    PcuNumericalMode,
    PcuOwnedDispatchBackend,
};
#[rustfmt::skip]
use fusion_pcu::dialect::tensor::{
    Graph,
    TensorArithmeticCapability,
    TensorArithmeticRewritePolicy,
    TensorPointwiseGroupingPolicy,
};
#[rustfmt::skip]
use fusion_pcu_cuda::{
    CudaOwnedDispatchBackend,
    CudaOwnedTensorAssessor,
    CudaRuntime,
};
use super::{
    native::Native,
    oracle::{self, Format},
    source,
};
static SCORES: AtomicUsize = AtomicUsize::new(0);
fn score(_: &global::PcuInvocationCandidate<'_>) -> i128 {
    SCORES.fetch_add(1, Ordering::Relaxed);
    0
}
fn source_call<T: Format>(
    a: &PcuTensor<T>,
    b: &PcuTensor<T>,
    op: u32,
) -> Result<PcuTensor<T>, PcuExecutionError> {
    match op {
        0 => source::add(a, b),
        1 => source::sub(a, b),
        2 => source::mul(a, b),
        _ => unreachable!(),
    }
}
fn verify<T: Format>(expected: &[T], observed: &[T], sentinel: T) {
    assert_eq!(observed.len(), expected.len() + 2);
    for (a, b) in expected.iter().zip(observed) {
        assert_eq!(a.encode_le().as_ref(), b.encode_le().as_ref());
    }
    for tail in &observed[expected.len()..] {
        assert_eq!(tail.encode_le().as_ref(), sentinel.encode_le().as_ref());
    }
}
#[allow(clippy::significant_drop_tightening)]
// Criterion group.finish consumes the group at the last sampling operation; cfg obscures that use.
#[allow(clippy::too_many_lines)] // Paired physical allocation/completion/readback/release boundaries stay together.
#[cfg_attr(feature = "allocation-census", allow(clippy::needless_pass_by_ref_mut))]
// The canonical Criterion interface remains mutable; census builds skip statistical sampling.
fn case<T: Format, const N: usize>(
    criterion: &mut Criterion,
    backend: &Rc<CudaOwnedDispatchBackend>,
    runtime: &CudaRuntime,
    op: u32,
    mode: PcuNumericalMode,
) -> Result<(), Box<dyn Error>> {
    // Semantic --test emits no estimates; the statistical activity guard is unchanged.
    if !std::env::args().any(|argument| argument == "--test") {
        super::activity::activity_guard();
    }
    global::clear_thread_cache()?;
    #[cfg(feature = "allocation-census")]
    let _ = criterion;
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Cuda,
        device: Some(backend.device_identity().device_id()),
        block_size: 256,
        numerical_mode: mode,
        score_invocation: Some(score),
        ..Default::default()
    })?;
    let pool = PcuMemoryPoolId(0x5754_454e);
    let root = CudaOwnedTensorAssessor::new(Rc::clone(backend))?;
    let assessor = root.assessor();
    let mut graph = Graph::default();
    graph.set_numerical_mode(mode);
    let a_id = graph.input([N], T::TYPE)?;
    let b_id = graph.input([N], T::TYPE)?;
    let value = match op {
        0 => graph.add(a_id, b_id)?,
        1 => graph.sub(a_id, b_id)?,
        2 => graph.mul(a_id, b_id)?,
        _ => unreachable!(),
    };
    let mut native = Native::new::<T, N>(runtime, &graph, value)?;
    let borrowed = assessor.prepare_graph(&graph, value)?;
    assessor.prewarm_prepared_graph(&borrowed)?;
    drop(borrowed);
    let program = graph.into_selected_program(
        &[value],
        TensorArithmeticRewritePolicy::Disabled,
        TensorArithmeticCapability::Strict,
        TensorPointwiseGroupingPolicy::Disabled,
    )?;
    let prepared = assessor.prepare_owned_program(program)?;
    let banks = [
        oracle::inputs::<T>(N, 1, op),
        oracle::inputs::<T>(N, 17, op),
    ];
    let source_a = [
        source::identity(banks[0].0.as_slice())?,
        source::identity(banks[1].0.as_slice())?,
    ];
    let source_b = [
        source::identity(banks[0].1.as_slice())?,
        source::identity(banks[1].1.as_slice())?,
    ];
    let raw_a = [
        PcuDeviceTensor::new([N], backend.upload_buffer(pool, banks[0].0.as_slice())?)?,
        PcuDeviceTensor::new([N], backend.upload_buffer(pool, banks[1].0.as_slice())?)?,
    ];
    let raw_b = [
        PcuDeviceTensor::new([N], backend.upload_buffer(pool, banks[0].1.as_slice())?)?,
        PcuDeviceTensor::new([N], backend.upload_buffer(pool, banks[1].1.as_slice())?)?,
    ];
    let left_metadata = [
        assessor.borrow_device_input_ref(&raw_a[0], pool)?,
        assessor.borrow_device_input_ref(&raw_a[1], pool)?,
    ];
    let right_metadata = [
        assessor.borrow_device_input_ref(&raw_b[0], pool)?,
        assessor.borrow_device_input_ref(&raw_b[1], pool)?,
    ];
    for (bank, (a, b, _)) in banks.iter().enumerate() {
        native.upload(bank, a, b)?;
        drop(source_call(&source_a[bank], &source_b[bank], op)?);
    }
    let scores = SCORES.load(Ordering::Relaxed);
    let sentinel = T::sentinel();
    let mut observed = vec![sentinel; N + 2];
    let mut memory = backend.memory_provider(pool);
    #[cfg(not(feature = "allocation-census"))]
    let mut group = criterion.benchmark_group(format!(
        "cuda_wide_tensor/{mode:?}/{}/{op}/resident_to_fresh_resident/{N}",
        T::LABEL
    ));
    #[cfg(not(feature = "allocation-census"))]
    group.sample_size(20);
    #[cfg(not(feature = "allocation-census"))]
    group.warm_up_time(Duration::from_millis(500));
    #[cfg(not(feature = "allocation-census"))]
    group.measurement_time(Duration::from_secs(2));
    #[cfg(not(feature = "allocation-census"))]
    group.throughput(Throughput::Elements(u64::try_from(N)?));
    for route in [
        "actual_source",
        "explicit_graph",
        "independent_native_checker",
    ] {
        let mut phase = 0;
        let mut call = || {
            phase = (phase + 1) % 2;
            let start = Instant::now();
            let total = match route {
                "actual_source" => {
                    let output = source_call(&source_a[phase], &source_b[phase], op).unwrap();
                    let mut elapsed = start.elapsed();
                    output.read_into(&mut observed).unwrap();
                    verify(&banks[phase].2, &observed, sentinel);
                    let release = Instant::now();
                    drop(black_box(output));
                    elapsed += release.elapsed();
                    elapsed
                }
                "explicit_graph" => {
                    let output = assessor
                        .execute_owned_program_output_from_inputs::<T, _>(
                            &prepared,
                            &[
                                (a_id, &left_metadata[phase]),
                                (b_id, &right_metadata[phase]),
                            ],
                            pool,
                            &mut memory,
                        )
                        .unwrap();
                    let mut elapsed = start.elapsed();
                    backend
                        .download_buffer(pool, output.buffer(), &mut observed[..N])
                        .unwrap();
                    verify(&banks[phase].2, &observed, sentinel);
                    let release = Instant::now();
                    drop(black_box(output));
                    elapsed += release.elapsed();
                    elapsed
                }
                _ => {
                    let output = native.submit(phase).unwrap();
                    let mut elapsed = start.elapsed();
                    Native::read(&output, &mut observed).unwrap();
                    verify(&banks[phase].2, &observed, sentinel);
                    let release = Instant::now();
                    drop(black_box(output));
                    elapsed += release.elapsed();
                    elapsed
                }
            };
            assert_eq!(
                SCORES.load(Ordering::Relaxed),
                scores,
                "warm source rescored"
            );
            total
        };
        let _ = call();
        #[cfg(feature = "allocation-census")]
        {
            fusion_pcu_cuda::reset_cuda_api_census();
            let ((), census) = super::allocations::measure(|| {
                for _ in 0..64 {
                    black_box(call());
                }
            });
            let api = fusion_pcu_cuda::cuda_api_census();
            assert_eq!(api.kernel_launches, 64);
            assert_eq!(api.module_loads, 0);
            assert_eq!(api.symbol_resolutions, 0);
            eprintln!(
                "census/{mode:?}/{}/{op}/{N}/{route}/64-changing-calls: alloc={} realloc={} frees={} bytes={}; API={api:?}; outside-timer readback/oracle included, cached sentinel allocates nothing",
                T::LABEL,
                census.alloc_calls,
                census.realloc_calls,
                census.dealloc_calls,
                census.requested_bytes
            );
        }
        #[cfg(not(feature = "allocation-census"))]
        group.bench_function(route, |bench| {
            bench.iter_custom(|iterations| {
                let mut total = Duration::ZERO;
                for _ in 0..iterations {
                    total += call();
                }
                total
            });
        });
    }
    #[cfg(not(feature = "allocation-census"))]
    group.finish();

    Ok(())
}
pub fn run(criterion: &mut Criterion) -> Result<(), Box<dyn Error>> {
    let (_, backend, runtime) = super::selection::selected_device();
    for mode in [PcuNumericalMode::Boundary, PcuNumericalMode::Strict] {
        macro_rules! widths {
            ($ty:ty) => {
                for op in 0..3 {
                    case::<$ty, 65>(criterion, &backend, &runtime, op, mode)?;
                    case::<$ty, 65536>(criterion, &backend, &runtime, op, mode)?;
                }
            };
        }
        widths!(i128);
        widths!(u128);
        widths!(fusion_pcu::PcuI256);
        widths!(fusion_pcu::PcuU256);
        widths!(fusion_pcu::PcuI512);
        widths!(fusion_pcu::PcuU512);
    }
    global::clear_thread_cache()?;
    Ok(())
}
