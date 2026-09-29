//! Shared static driver for fresh owned f32/f64 `MatMul` comparisons.

#[rustfmt::skip]
use std::{
    error::Error,
    hint::black_box,
    mem::size_of,
    rc::Rc,
    time::{
        Duration,
        Instant,
    },
};
#[rustfmt::skip]
use criterion::{
    BenchmarkId,
    Criterion,
    Throughput,
};
#[rustfmt::skip]
use fusion_pcu::{
    PcuDeviceBuffer,
    PcuExecutionError,
    PcuDeviceTensor,
    PcuMemoryPoolId,
    PcuScalar,
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
    DeviceBuffer,
    Rocblas,
    RocmDiscovery,
    RocmMemoryResource,
    RocmOwnedDispatchBackend,
    RocmOwnedTensorAssessor,
};

pub const BALANCED_ORDERS: [[usize; 3]; 6] = [
    [0, 1, 2],
    [0, 2, 1],
    [1, 0, 2],
    [1, 2, 0],
    [2, 0, 1],
    [2, 1, 0],
];

/// Type-level source/native profile. Implementations monomorphize the driver; no trait object or
/// dynamic scalar conversion occurs in the measured route.
pub trait MatMulProfile {
    type Scalar: PcuScalar + bytemuck::Pod + Default;

    const GROUP: &'static str;
    const CENSUS_LABEL: &'static str;
    const DIAGNOSTIC_LABEL: &'static str;
    const PREPARATION_LABEL: &'static str;
    const NATIVE_ROUTE: &'static str;
    const WARM_SOURCE_REFRESH: bool;

    fn fill<const R: usize, const K: usize, const C: usize>(
        lhs: &mut [Self::Scalar],
        rhs: &mut [Self::Scalar],
        job: u64,
    );

    fn oracle<const R: usize, const K: usize, const C: usize>(
        lhs: &[Self::Scalar],
        rhs: &[Self::Scalar],
        output: &mut [Self::Scalar],
    );

    fn verify<const R: usize, const K: usize, const C: usize>(
        expected: &[Self::Scalar],
        actual: &[Self::Scalar],
    );

    fn source_identity<const R: usize, const C: usize>(
        input: &[[Self::Scalar; C]; R],
    ) -> Result<PcuTensor<Self::Scalar>, Box<dyn Error>>;

    fn source_matmul<const R: usize, const K: usize, const C: usize>(
        lhs: &PcuTensor<Self::Scalar>,
        rhs: &PcuTensor<Self::Scalar>,
    ) -> Result<PcuTensor<Self::Scalar>, Box<dyn Error>>;

    fn refresh_source_inputs<
        const R: usize,
        const K: usize,
        const C: usize,
        const NL: usize,
        const NR: usize,
    >(
        host: &HostInputs<Self, R, K, C>,
        lhs: &mut PcuTensor<Self::Scalar>,
        rhs: &mut PcuTensor<Self::Scalar>,
    ) -> Result<(), Box<dyn Error>>
    where
        Self: Sized;

    fn warm_source_refresh<
        const R: usize,
        const K: usize,
        const C: usize,
        const NL: usize,
        const NR: usize,
    >(
        _host: &HostInputs<Self, R, K, C>,
        _lhs: &mut PcuTensor<Self::Scalar>,
        _rhs: &mut PcuTensor<Self::Scalar>,
    ) -> Result<(), Box<dyn Error>>
    where
        Self: Sized,
    {
        Ok(())
    }

    fn require_native_support(blas: &Rocblas) -> Result<(), Box<dyn Error>>;

    fn native_call<const R: usize, const K: usize, const C: usize>(
        runtime: &fusion_pcu_rocm::HipRuntime,
        blas: &Rocblas,
        lhs: &DeviceBuffer,
        rhs: &DeviceBuffer,
        output_bytes: usize,
    ) -> Result<DeviceBuffer, Box<dyn Error>>;
}

#[fusion_pcu::pcu]
pub fn source_identity<T: PcuScalar, const R: usize, const C: usize>(
    input: &[[T; C]; R],
) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::identity(input)
}

#[fusion_pcu::pcu]
pub fn source_matmul<T: PcuScalar, const R: usize, const K: usize, const C: usize>(
    lhs: &[[T; K]; R],
    rhs: &[[T; C]; K],
) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::matmul(lhs, rhs)
}

pub struct HostInputs<P: MatMulProfile, const R: usize, const K: usize, const C: usize> {
    pub lhs: Vec<P::Scalar>,
    pub rhs: Vec<P::Scalar>,
    pub oracle: Vec<P::Scalar>,
    pub source_lhs: Box<[[P::Scalar; K]; R]>,
    pub source_rhs: Box<[[P::Scalar; C]; K]>,
}

impl<P: MatMulProfile, const R: usize, const K: usize, const C: usize> HostInputs<P, R, K, C> {
    fn new(job: u64) -> Self {
        let mut lhs = vec![P::Scalar::default(); R * K];
        let mut rhs = vec![P::Scalar::default(); K * C];
        P::fill::<R, K, C>(&mut lhs, &mut rhs, job);
        let mut oracle = vec![P::Scalar::default(); R * C];
        P::oracle::<R, K, C>(&lhs, &rhs, &mut oracle);
        let source_lhs = box_matrix::<P::Scalar, R, K>(&lhs);
        let source_rhs = box_matrix::<P::Scalar, K, C>(&rhs);
        Self {
            lhs,
            rhs,
            oracle,
            source_lhs,
            source_rhs,
        }
    }

    fn refill(&mut self, job: u64) {
        P::fill::<R, K, C>(&mut self.lhs, &mut self.rhs, job);
        P::oracle::<R, K, C>(&self.lhs, &self.rhs, &mut self.oracle);
        self.source_lhs
            .as_mut()
            .as_flattened_mut()
            .copy_from_slice(&self.lhs);
        self.source_rhs
            .as_mut()
            .as_flattened_mut()
            .copy_from_slice(&self.rhs);
    }
}

struct RetainedInputs<P: MatMulProfile, const R: usize, const K: usize, const C: usize> {
    host: HostInputs<P, R, K, C>,
    raw_lhs: Option<PcuDeviceBuffer<P::Scalar, RocmMemoryResource>>,
    raw_rhs: Option<PcuDeviceBuffer<P::Scalar, RocmMemoryResource>>,
    native_lhs: DeviceBuffer,
    native_rhs: DeviceBuffer,
    source_lhs: PcuTensor<P::Scalar>,
    source_rhs: PcuTensor<P::Scalar>,
}

impl<P: MatMulProfile, const R: usize, const K: usize, const C: usize> RetainedInputs<P, R, K, C> {
    fn refresh<const NL: usize, const NR: usize>(
        &mut self,
        backend: &RocmOwnedDispatchBackend,
        pool: PcuMemoryPoolId,
        job: u64,
    ) -> Result<(), Box<dyn Error>> {
        self.host.refill(job);
        backend.refresh_buffer(
            pool,
            self.raw_lhs.as_mut().expect("left input buffer"),
            &self.host.lhs,
        )?;
        backend.refresh_buffer(
            pool,
            self.raw_rhs.as_mut().expect("right input buffer"),
            &self.host.rhs,
        )?;
        self.native_lhs
            .copy_from(bytemuck::cast_slice(&self.host.lhs))?;
        self.native_rhs
            .copy_from(bytemuck::cast_slice(&self.host.rhs))?;
        P::refresh_source_inputs::<R, K, C, NL, NR>(
            &self.host,
            &mut self.source_lhs,
            &mut self.source_rhs,
        )?;
        Ok(())
    }
}

type SelectedBackend = (
    Rc<RocmOwnedDispatchBackend>,
    fusion_pcu_rocm::HipRuntime,
    PcuMemoryPoolId,
    String,
);

pub fn select_backend() -> Result<SelectedBackend, Box<dyn Error>> {
    let discovery = RocmDiscovery::new();
    let candidates = crate::support::selected_candidates(&discovery)?
        .into_iter()
        .filter(|candidate| candidate.architecture.is_some())
        .collect();
    let (backend, selected) = crate::support::selection::open_ranked(&discovery, candidates, 256)?;
    let backend = Rc::new(backend);
    let runtime = discovery.open_device(selected.device)?;
    fusion_pcu::global::configure(fusion_pcu::global::PcuExecutionPolicy {
        backend: fusion_pcu::global::PcuBackendChoice::Rocm,
        device: Some(selected.device.id),
        ..fusion_pcu::global::PcuExecutionPolicy::default()
    })?;
    Ok((backend, runtime, selected.pool, selected.name))
}

#[allow(clippy::too_many_lines)]
#[allow(clippy::significant_drop_tightening)]
#[allow(clippy::branches_sharing_code)] // Keep route-specific timing scopes intact.
pub fn run_case<
    P: MatMulProfile,
    const R: usize,
    const K: usize,
    const C: usize,
    const NL: usize,
    const NR: usize,
>(
    criterion: &mut Criterion,
    backend: &Rc<RocmOwnedDispatchBackend>,
    runtime: &fusion_pcu_rocm::HipRuntime,
    pool: PcuMemoryPoolId,
) -> Result<(), Box<dyn Error>> {
    let assessor_root = RocmOwnedTensorAssessor::new(Rc::clone(backend))?;
    let assessor = assessor_root.assessor();
    let mut graph = Graph::default();
    let lhs_id = graph.input([R, K], P::Scalar::TYPE)?;
    let rhs_id = graph.input([K, C], P::Scalar::TYPE)?;
    let output_id = graph.matmul(lhs_id, rhs_id)?;
    let program = graph.into_selected_program(
        &[output_id],
        TensorArithmeticRewritePolicy::Disabled,
        TensorArithmeticCapability::Strict,
        TensorPointwiseGroupingPolicy::Disabled,
    )?;
    let prepared = crate::support::cold_once(P::PREPARATION_LABEL, || {
        assessor.prepare_owned_program(program)
    })?;
    let blas = Rocblas::new(runtime)?;
    P::require_native_support(&blas)?;
    // Match the PCU assessor's non-default-stream setup class. The assessor's exact retained
    // stream is private, so this control uses a separate explicit stream on the same runtime.
    let explicit_stream = runtime.create_stream()?;
    let mut explicit_blas = Rocblas::new(runtime)?;
    explicit_blas.bind_stream(&explicit_stream)?;
    P::require_native_support(&explicit_blas)?;

    let output_bytes = R
        .checked_mul(C)
        .and_then(|n| n.checked_mul(size_of::<P::Scalar>()))
        .ok_or("MatMul output size overflow")?;
    let lhs_bytes = R
        .checked_mul(K)
        .and_then(|n| n.checked_mul(size_of::<P::Scalar>()))
        .ok_or("MatMul left input size overflow")?;
    let rhs_bytes = K
        .checked_mul(C)
        .and_then(|n| n.checked_mul(size_of::<P::Scalar>()))
        .ok_or("MatMul right input size overflow")?;
    let host = HostInputs::<P, R, K, C>::new(1);
    let mut observed = vec![P::Scalar::default(); R * C];
    let raw_lhs = backend.upload_buffer(pool, &host.lhs)?;
    let raw_rhs = backend.upload_buffer(pool, &host.rhs)?;
    let mut native_lhs = runtime.allocate(lhs_bytes)?;
    let mut native_rhs = runtime.allocate(rhs_bytes)?;
    native_lhs.copy_from(bytemuck::cast_slice(&host.lhs))?;
    native_rhs.copy_from(bytemuck::cast_slice(&host.rhs))?;
    let mut memory = backend.memory_provider(pool);
    let mut source_lhs = crate::support::cold_once("source left matrix identity", || {
        P::source_identity::<R, K>(&host.source_lhs)
    })?;
    let mut source_rhs = crate::support::cold_once("source right matrix identity", || {
        P::source_identity::<K, C>(&host.source_rhs)
    })?;
    if P::WARM_SOURCE_REFRESH {
        crate::support::cold_once("source retained matrix refresh warmup", || {
            P::warm_source_refresh::<R, K, C, NL, NR>(&host, &mut source_lhs, &mut source_rhs)
        })?;
    }
    let mut inputs = RetainedInputs::<P, R, K, C> {
        host,
        raw_lhs: Some(raw_lhs),
        raw_rhs: Some(raw_rhs),
        native_lhs,
        native_rhs,
        source_lhs,
        source_rhs,
    };

    let lhs_owner =
        PcuDeviceTensor::new([R, K], inputs.raw_lhs.take().expect("left input buffer"))?;
    let rhs_owner =
        PcuDeviceTensor::new([K, C], inputs.raw_rhs.take().expect("right input buffer"))?;
    let raw_first = crate::support::cold_once("owned MatMul first execution", || {
        assessor.execute_owned_program_outputs(
            &prepared,
            &[(lhs_id, &lhs_owner), (rhs_id, &rhs_owner)],
            pool,
            &mut memory,
        )
    })?;
    backend.download_buffer(pool, raw_first[0].1.buffer(), &mut observed)?;
    P::verify::<R, K, C>(&inputs.host.oracle, &observed);
    drop(raw_first);
    let source_first = crate::support::cold_once("source MatMul first execution", || {
        P::source_matmul::<R, K, C>(&inputs.source_lhs, &inputs.source_rhs)
    })?;
    source_first.read_into(&mut observed)?;
    P::verify::<R, K, C>(&inputs.host.oracle, &observed);
    drop(source_first);
    let native_first = crate::support::cold_once("native rocBLAS first execution", || {
        P::native_call::<R, K, C>(
            runtime,
            &blas,
            &inputs.native_lhs,
            &inputs.native_rhs,
            output_bytes,
        )
    })?;
    native_first.copy_to(bytemuck::cast_slice_mut(&mut observed))?;
    P::verify::<R, K, C>(&inputs.host.oracle, &observed);
    drop(native_first);
    let explicit_first =
        crate::support::cold_once("native explicit-stream rocBLAS first execution", || {
            P::native_call::<R, K, C>(
                runtime,
                &explicit_blas,
                &inputs.native_lhs,
                &inputs.native_rhs,
                output_bytes,
            )
        })?;
    explicit_first.copy_to(bytemuck::cast_slice_mut(&mut observed))?;
    P::verify::<R, K, C>(&inputs.host.oracle, &observed);
    drop(explicit_first);

    let raw_counts = {
        let _capture = crate::owned_matmul_support::alloc::AllocationCapture::start();
        let outputs = assessor_root.assessor().execute_owned_program_outputs(
            &prepared,
            &[(lhs_id, &lhs_owner), (rhs_id, &rhs_owner)],
            pool,
            &mut memory,
        )?;
        drop(outputs);
        crate::owned_matmul_support::alloc::AllocationCapture::finish()
    };
    let source_counts = {
        let _capture = crate::owned_matmul_support::alloc::AllocationCapture::start();
        let output = P::source_matmul::<R, K, C>(&inputs.source_lhs, &inputs.source_rhs)?;
        drop(output);
        crate::owned_matmul_support::alloc::AllocationCapture::finish()
    };
    let native_counts = {
        let _capture = crate::owned_matmul_support::alloc::AllocationCapture::start();
        let output = P::native_call::<R, K, C>(
            runtime,
            &blas,
            &inputs.native_lhs,
            &inputs.native_rhs,
            output_bytes,
        )?;
        drop(output);
        crate::owned_matmul_support::alloc::AllocationCapture::finish()
    };
    let explicit_counts = {
        let _capture = crate::owned_matmul_support::alloc::AllocationCapture::start();
        let output = P::native_call::<R, K, C>(
            runtime,
            &explicit_blas,
            &inputs.native_lhs,
            &inputs.native_rhs,
            output_bytes,
        )?;
        drop(output);
        crate::owned_matmul_support::alloc::AllocationCapture::finish()
    };
    println!(
        "{} {R}x{K}x{C}: raw PCU {}/{}/{} B, source {}/{}/{} B, native {}/{}/{} B, explicit-stream native {}/{}/{} B allocations/reallocations/requested. Excludes input refresh, readback, and driver allocations.",
        P::CENSUS_LABEL,
        raw_counts.alloc_calls,
        raw_counts.realloc_calls,
        raw_counts.requested_bytes,
        source_counts.alloc_calls,
        source_counts.realloc_calls,
        source_counts.requested_bytes,
        native_counts.alloc_calls,
        native_counts.realloc_calls,
        native_counts.requested_bytes,
        explicit_counts.alloc_calls,
        explicit_counts.realloc_calls,
        explicit_counts.requested_bytes,
    );
    inputs.raw_lhs = Some(lhs_owner.into_buffer());
    inputs.raw_rhs = Some(rhs_owner.into_buffer());

    let mut raw_times = [Duration::ZERO; 36];
    let mut source_times = [Duration::ZERO; 36];
    let mut native_times = [Duration::ZERO; 36];
    let mut native_control_times = [Duration::ZERO; 36];
    let mut explicit_times = [Duration::ZERO; 36];
    let mut explicit_ratios = [0.0_f64; 36];
    let mut raw_ratios = [0.0_f64; 36];
    let mut source_ratios = [0.0_f64; 36];
    for sample in 0..36 {
        inputs.refresh::<NL, NR>(backend, pool, u64::try_from(sample + 2)?)?;
        let lhs_owner =
            PcuDeviceTensor::new([R, K], inputs.raw_lhs.take().expect("left input buffer"))?;
        let rhs_owner =
            PcuDeviceTensor::new([K, C], inputs.raw_rhs.take().expect("right input buffer"))?;
        for route in BALANCED_ORDERS[sample % BALANCED_ORDERS.len()] {
            if route == 0 {
                let start = Instant::now();
                let outputs = assessor_root.assessor().execute_owned_program_outputs(
                    &prepared,
                    &[(lhs_id, &lhs_owner), (rhs_id, &rhs_owner)],
                    pool,
                    &mut memory,
                )?;
                let execution = start.elapsed();
                backend.download_buffer(pool, outputs[0].1.buffer(), &mut observed)?;
                P::verify::<R, K, C>(&inputs.host.oracle, &observed);
                let release = Instant::now();
                drop(black_box(outputs));
                raw_times[sample] = execution + release.elapsed();
            } else if route == 1 {
                let start = Instant::now();
                let output = P::source_matmul::<R, K, C>(&inputs.source_lhs, &inputs.source_rhs)?;
                let execution = start.elapsed();
                output.read_into(&mut observed)?;
                P::verify::<R, K, C>(&inputs.host.oracle, &observed);
                let release = Instant::now();
                drop(black_box(output));
                source_times[sample] = execution + release.elapsed();
            } else {
                let start = Instant::now();
                let output = P::native_call::<R, K, C>(
                    runtime,
                    &blas,
                    &inputs.native_lhs,
                    &inputs.native_rhs,
                    output_bytes,
                )?;
                let execution = start.elapsed();
                output.copy_to(bytemuck::cast_slice_mut(&mut observed))?;
                P::verify::<R, K, C>(&inputs.host.oracle, &observed);
                let release = Instant::now();
                drop(black_box(output));
                native_times[sample] = execution + release.elapsed();
            }
        }
        raw_ratios[sample] = raw_times[sample].as_secs_f64() / native_times[sample].as_secs_f64();
        source_ratios[sample] =
            source_times[sample].as_secs_f64() / native_times[sample].as_secs_f64();

        // Pair the two native handle configurations on the same fresh inputs, alternating route
        // order to avoid making the explicit stream control systematically second.
        for explicit_first in [sample % 2 == 1, sample % 2 == 0] {
            if explicit_first {
                let start = Instant::now();
                let output = P::native_call::<R, K, C>(
                    runtime,
                    &explicit_blas,
                    &inputs.native_lhs,
                    &inputs.native_rhs,
                    output_bytes,
                )?;
                let execution = start.elapsed();
                output.copy_to(bytemuck::cast_slice_mut(&mut observed))?;
                P::verify::<R, K, C>(&inputs.host.oracle, &observed);
                let release = Instant::now();
                drop(black_box(output));
                explicit_times[sample] = execution + release.elapsed();
            } else {
                let start = Instant::now();
                let output = P::native_call::<R, K, C>(
                    runtime,
                    &blas,
                    &inputs.native_lhs,
                    &inputs.native_rhs,
                    output_bytes,
                )?;
                let execution = start.elapsed();
                output.copy_to(bytemuck::cast_slice_mut(&mut observed))?;
                P::verify::<R, K, C>(&inputs.host.oracle, &observed);
                let release = Instant::now();
                drop(black_box(output));
                native_control_times[sample] = execution + release.elapsed();
            }
        }
        explicit_ratios[sample] =
            explicit_times[sample].as_secs_f64() / native_control_times[sample].as_secs_f64();
        inputs.raw_lhs = Some(lhs_owner.into_buffer());
        inputs.raw_rhs = Some(rhs_owner.into_buffer());
    }
    raw_times.sort_unstable();
    source_times.sort_unstable();
    native_times.sort_unstable();
    native_control_times.sort_unstable();
    explicit_times.sort_unstable();
    explicit_ratios.sort_unstable_by(f64::total_cmp);
    raw_ratios.sort_unstable_by(f64::total_cmp);
    source_ratios.sort_unstable_by(f64::total_cmp);
    println!(
        "{} diagnostic {R}x{K}x{C}, 36 balanced verified triples plus paired native stream control: raw PCU {:?}, source {:?}, triple native {:?}, paired default-stream native {:?}, paired explicit-stream native {:?}; paired raw/triple-native ratio {:.4}, source/triple-native ratio {:.4}, explicit/default native ratio {:.4}. Diagnostic medians, not Criterion intervals.",
        P::DIAGNOSTIC_LABEL,
        midpoint(raw_times[17], raw_times[18]),
        midpoint(source_times[17], source_times[18]),
        midpoint(native_times[17], native_times[18]),
        midpoint(native_control_times[17], native_control_times[18]),
        midpoint(explicit_times[17], explicit_times[18]),
        midpoint_f64(raw_ratios[17], raw_ratios[18]),
        midpoint_f64(source_ratios[17], source_ratios[18]),
        midpoint_f64(explicit_ratios[17], explicit_ratios[18]),
    );

    let mut raw_job = 38_u64;
    let mut source_job = 38_u64;
    let mut native_job = 38_u64;
    let mut explicit_job = 38_u64;
    let criterion_wall_start = Instant::now();
    let mut group = criterion.benchmark_group(P::GROUP);
    group.throughput(Throughput::Elements(u64::try_from(R * K + K * C + R * C)?));
    group.bench_function(
        BenchmarkId::new("raw_owned_graph", format!("{R}x{K}x{C}")),
        |bencher| {
            bencher.iter_custom(|iterations| {
                let mut total = Duration::ZERO;
                for _ in 0..iterations {
                    raw_job = raw_job.wrapping_add(1);
                    inputs
                        .refresh::<NL, NR>(backend, pool, raw_job)
                        .expect("refresh all route inputs");
                    let lhs_owner =
                        PcuDeviceTensor::new([R, K], inputs.raw_lhs.take().expect("left input"))
                            .expect("left tensor shape");
                    let rhs_owner =
                        PcuDeviceTensor::new([K, C], inputs.raw_rhs.take().expect("right input"))
                            .expect("right tensor shape");
                    let start = Instant::now();
                    let outputs = assessor_root
                        .assessor()
                        .execute_owned_program_outputs(
                            &prepared,
                            &[(lhs_id, &lhs_owner), (rhs_id, &rhs_owner)],
                            pool,
                            &mut memory,
                        )
                        .expect("owned graph MatMul");
                    total += start.elapsed();
                    backend
                        .download_buffer(pool, outputs[0].1.buffer(), &mut observed)
                        .expect("owned graph readback");
                    P::verify::<R, K, C>(&inputs.host.oracle, &observed);
                    let start = Instant::now();
                    drop(black_box(outputs));
                    total += start.elapsed();
                    inputs.raw_lhs = Some(lhs_owner.into_buffer());
                    inputs.raw_rhs = Some(rhs_owner.into_buffer());
                }
                total
            });
        },
    );
    group.bench_function(
        BenchmarkId::new("source", format!("{R}x{K}x{C}")),
        |bencher| {
            bencher.iter_custom(|iterations| {
                let mut total = Duration::ZERO;
                for _ in 0..iterations {
                    source_job = source_job.wrapping_add(1);
                    inputs
                        .refresh::<NL, NR>(backend, pool, source_job)
                        .expect("refresh all route inputs");
                    let start = Instant::now();
                    let output =
                        P::source_matmul::<R, K, C>(&inputs.source_lhs, &inputs.source_rhs)
                            .expect("source MatMul");
                    total += start.elapsed();
                    output
                        .read_into(&mut observed)
                        .expect("source MatMul readback");
                    P::verify::<R, K, C>(&inputs.host.oracle, &observed);
                    let start = Instant::now();
                    drop(black_box(output));
                    total += start.elapsed();
                }
                total
            });
        },
    );
    group.bench_function(
        BenchmarkId::new(P::NATIVE_ROUTE, format!("{R}x{K}x{C}")),
        |bencher| {
            bencher.iter_custom(|iterations| {
                let mut total = Duration::ZERO;
                for _ in 0..iterations {
                    native_job = native_job.wrapping_add(1);
                    inputs
                        .refresh::<NL, NR>(backend, pool, native_job)
                        .expect("refresh all route inputs");
                    let start = Instant::now();
                    let output = P::native_call::<R, K, C>(
                        runtime,
                        &blas,
                        &inputs.native_lhs,
                        &inputs.native_rhs,
                        output_bytes,
                    )
                    .expect("native rocBLAS MatMul");
                    total += start.elapsed();
                    output
                        .copy_to(bytemuck::cast_slice_mut(&mut observed))
                        .expect("native readback");
                    P::verify::<R, K, C>(&inputs.host.oracle, &observed);
                    let start = Instant::now();
                    drop(black_box(output));
                    total += start.elapsed();
                }
                total
            });
        },
    );
    group.bench_function(
        BenchmarkId::new("native_explicit_stream", format!("{R}x{K}x{C}")),
        |bencher| {
            bencher.iter_custom(|iterations| {
                let mut total = Duration::ZERO;
                for _ in 0..iterations {
                    explicit_job = explicit_job.wrapping_add(1);
                    inputs
                        .refresh::<NL, NR>(backend, pool, explicit_job)
                        .expect("refresh all route inputs");
                    let start = Instant::now();
                    let output = P::native_call::<R, K, C>(
                        runtime,
                        &explicit_blas,
                        &inputs.native_lhs,
                        &inputs.native_rhs,
                        output_bytes,
                    )
                    .expect("native explicit-stream rocBLAS MatMul");
                    total += start.elapsed();
                    output
                        .copy_to(bytemuck::cast_slice_mut(&mut observed))
                        .expect("native explicit-stream readback");
                    P::verify::<R, K, C>(&inputs.host.oracle, &observed);
                    let start = Instant::now();
                    drop(black_box(output));
                    total += start.elapsed();
                }
                total
            });
        },
    );
    group.finish();
    println!(
        "{} Criterion wall-clock {R}x{K}x{C}: {:?} (includes fresh input generation and CPU verification outside measured intervals).",
        P::CENSUS_LABEL,
        criterion_wall_start.elapsed(),
    );
    Ok(())
}

fn box_matrix<T: Copy, const R: usize, const C: usize>(flat: &[T]) -> Box<[[T; C]; R]> {
    assert_eq!(flat.len(), R * C);
    let (rows, remainder) = flat.as_chunks::<C>();
    assert!(remainder.is_empty());
    rows.to_vec()
        .into_boxed_slice()
        .try_into()
        .unwrap_or_else(|_| panic!("flat matrix has declared row count"))
}

fn midpoint(lhs: Duration, rhs: Duration) -> Duration {
    Duration::from_secs_f64(lhs.as_secs_f64().midpoint(rhs.as_secs_f64()))
}

const fn midpoint_f64(lhs: f64, rhs: f64) -> f64 {
    lhs.midpoint(rhs)
}
