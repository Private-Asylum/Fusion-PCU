//! Matched consuming-donor in-place binary routes with untimed input refresh/readback.

#[path = "alloc.rs"]
#[allow(dead_code)]
mod alloc;

#[rustfmt::skip]
use std::{
    error::Error,
    mem::size_of,
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
    PcuExecutionError,
    PcuTensor,
};
#[rustfmt::skip]
use fusion_pcu_rocm::{
    DeviceBuffer,
    HipKernel,
    HipKernelArgument,
    HipRuntime,
    HipStreamHandle,
    RocmDiscovery,
    compile_hip_source,
};

const BLOCK: u32 = 256;
const PAIRED_SAMPLES: usize = 24;

#[fusion_pcu::pcu]
fn source_add_donor_left(
    donor: PcuTensor<f32>,
    other: &PcuTensor<f32>,
) -> Result<PcuTensor<f32>, PcuExecutionError> {
    Ok(pcu::add(donor, other)?)
}

#[fusion_pcu::pcu]
fn source_sub_donor_left(
    donor: PcuTensor<f32>,
    other: &PcuTensor<f32>,
) -> Result<PcuTensor<f32>, PcuExecutionError> {
    Ok(pcu::sub(donor, other)?)
}

#[fusion_pcu::pcu]
fn source_sub_donor_right(
    donor: PcuTensor<f32>,
    other: &PcuTensor<f32>,
) -> Result<PcuTensor<f32>, PcuExecutionError> {
    Ok(pcu::sub(other, donor)?)
}

#[fusion_pcu::pcu]
fn source_mul_donor_left(
    donor: PcuTensor<f32>,
    other: &PcuTensor<f32>,
) -> Result<PcuTensor<f32>, PcuExecutionError> {
    Ok(pcu::mul(donor, other)?)
}

#[fusion_pcu::pcu]
fn seed(input: &[f32]) -> Result<PcuTensor<f32>, PcuExecutionError> {
    pcu::identity(input)
}

#[fusion_pcu::pcu(invocations: N)]
fn refresh<const N: usize>(input: &[f32], output: &mut [f32]) {
    let id = pcu::context::global_invocation_id();
    output[id] = input[id];
}

#[derive(Clone, Copy, Debug)]
enum BinaryProfile {
    Add,
    SubDonorLeft,
    SubDonorRight,
    Mul,
}

impl BinaryProfile {
    const ALL: [Self; 4] = [
        Self::Add,
        Self::SubDonorLeft,
        Self::SubDonorRight,
        Self::Mul,
    ];

    const fn name(self) -> &'static str {
        match self {
            Self::Add => "add_donor_left",
            Self::SubDonorLeft => "sub_donor_left",
            Self::SubDonorRight => "sub_donor_right",
            Self::Mul => "mul_donor_left",
        }
    }

    const fn donor_is_left(self) -> bool {
        !matches!(self, Self::SubDonorRight)
    }

    fn oracle(self, donor: f32, other: f32) -> f32 {
        match self {
            Self::Add => donor + other,
            Self::SubDonorLeft => donor - other,
            Self::SubDonorRight => other - donor,
            Self::Mul => donor * other,
        }
    }

    fn source_call(
        self,
        donor: PcuTensor<f32>,
        other: &PcuTensor<f32>,
    ) -> Result<PcuTensor<f32>, PcuExecutionError> {
        match self {
            Self::Add => source_add_donor_left(donor, other),
            Self::SubDonorLeft => source_sub_donor_left(donor, other),
            Self::SubDonorRight => source_sub_donor_right(donor, other),
            Self::Mul => source_mul_donor_left(donor, other),
        }
    }

    const fn native_expression(self) -> &'static str {
        match self {
            Self::Add => "left + right",
            Self::SubDonorLeft | Self::SubDonorRight => "left - right",
            Self::Mul => "left * right",
        }
    }
}

struct ResidentInputs {
    donor: Option<PcuTensor<f32>>,
    other: PcuTensor<f32>,
    native_donor: DeviceBuffer,
    native_other: DeviceBuffer,
    host_donor: Vec<f32>,
    host_other: Vec<f32>,
    oracle: Vec<f32>,
    observed: Vec<f32>,
    observed_other: Vec<f32>,
    profile: BinaryProfile,
}

impl ResidentInputs {
    fn new(elements: usize, runtime: &HipRuntime) -> Result<Self, Box<dyn Error>> {
        let mut host_donor = vec![0.0_f32; elements];
        let mut host_other = vec![0.0_f32; elements];
        fill(&mut host_donor, &mut host_other, 0);
        let donor = seed(&host_donor)?;
        let other = seed(&host_other)?;
        let bytes = elements
            .checked_mul(size_of::<f32>())
            .ok_or("donor benchmark byte extent overflow")?;
        let mut native_donor = runtime.allocate(bytes)?;
        native_donor.copy_from(bytemuck::cast_slice(&host_donor))?;
        let mut native_other = runtime.allocate(bytes)?;
        native_other.copy_from(bytemuck::cast_slice(&host_other))?;
        let mut inputs = Self {
            donor: Some(donor),
            other,
            native_donor,
            native_other,
            host_donor,
            host_other,
            oracle: vec![0.0; elements],
            observed: vec![0.0; elements],
            observed_other: vec![0.0; elements],
            profile: BinaryProfile::Add,
        };
        inputs.refresh(0, elements)?;
        Ok(inputs)
    }

    fn refresh(&mut self, job: u64, elements: usize) -> Result<(), Box<dyn Error>> {
        fill(&mut self.host_donor, &mut self.host_other, job);
        for ((expected, &donor), &other) in self
            .oracle
            .iter_mut()
            .zip(&self.host_donor)
            .zip(&self.host_other)
        {
            *expected = self.profile.oracle(donor, other);
        }
        refresh_for_extent(
            elements,
            &self.host_donor,
            self.donor
                .as_mut()
                .expect("donor owner is retained between calls"),
        )?;
        refresh_for_extent(elements, &self.host_other, &mut self.other)?;
        self.native_donor
            .copy_from(bytemuck::cast_slice(&self.host_donor))?;
        self.native_other
            .copy_from(bytemuck::cast_slice(&self.host_other))?;
        Ok(())
    }

    fn run_source(&mut self, profile: BinaryProfile) -> Result<Duration, Box<dyn Error>> {
        let (elapsed, output) = self.execute_source(profile)?;
        self.verify_source(output)?;
        Ok(elapsed)
    }

    fn execute_source(
        &mut self,
        profile: BinaryProfile,
    ) -> Result<(Duration, PcuTensor<f32>), Box<dyn Error>> {
        let donor = self.donor.take().expect("source donor is retained");
        let started = std::time::Instant::now();
        let output = profile.source_call(donor, &self.other)?;
        Ok((started.elapsed(), output))
    }

    fn verify_source(&mut self, output: PcuTensor<f32>) -> Result<(), Box<dyn Error>> {
        output.read_into(&mut self.observed)?;
        self.other.read_into(&mut self.observed_other)?;
        verify(
            &self.oracle,
            &self.host_other,
            &self.observed,
            &self.observed_other,
        );
        self.donor = Some(output);
        Ok(())
    }

    fn run_native(
        &mut self,
        kernel: &HipKernel,
        stream: &HipStreamHandle,
        grid: u32,
    ) -> Result<Duration, Box<dyn Error>> {
        let elapsed = self.execute_native(kernel, stream, grid)?;
        self.verify_native()?;
        Ok(elapsed)
    }

    #[allow(unsafe_code)] // Native control launches the matching in-place HIP kernel.
    fn execute_native(
        &self,
        kernel: &HipKernel,
        stream: &HipStreamHandle,
        grid: u32,
    ) -> Result<Duration, Box<dyn Error>> {
        let started = std::time::Instant::now();
        let arguments = [
            HipKernelArgument::Buffer(&self.native_donor),
            HipKernelArgument::Buffer(&self.native_other),
        ];
        // SAFETY: The native ABI is a writable donor pointer plus a read-only peer pointer, both
        // with exactly `elements` f32 values; the checked grid covers that same extent.
        let mut completion =
            unsafe { kernel.launch(stream, [grid, 1, 1], [BLOCK, 1, 1], 0, &arguments) }?;
        completion.wait()?;
        drop(completion);
        Ok(started.elapsed())
    }

    fn verify_native(&mut self) -> Result<(), Box<dyn Error>> {
        self.native_donor
            .copy_to(bytemuck::cast_slice_mut(&mut self.observed))?;
        self.native_other
            .copy_to(bytemuck::cast_slice_mut(&mut self.observed_other))?;
        verify(
            &self.oracle,
            &self.host_other,
            &self.observed,
            &self.observed_other,
        );
        Ok(())
    }
}

fn refresh_for_extent(
    elements: usize,
    input: &[f32],
    output: &mut PcuTensor<f32>,
) -> Result<(), PcuExecutionError> {
    match elements {
        65 => refresh::<65>(input, output),
        1_048_576 => refresh::<1_048_576>(input, output),
        _ => unreachable!("benchmark declares both extents"),
    }
}

pub fn run(criterion: &mut Criterion) -> Result<(), Box<dyn Error>> {
    let discovery = RocmDiscovery::new();
    let candidates = crate::support::selected_candidates(&discovery)?
        .into_iter()
        .filter(|candidate| candidate.architecture.is_some())
        .collect();
    let (backend, selected) =
        crate::support::selection::open_ranked(&discovery, candidates, BLOCK)?;
    let _backend = Rc::new(backend);
    let runtime = discovery.open_device(selected.device)?;
    let architecture = selected
        .architecture
        .as_deref()
        .ok_or("selected ROCm device has no architecture")?;
    fusion_pcu::global::configure(fusion_pcu::global::PcuExecutionPolicy {
        backend: fusion_pcu::global::PcuBackendChoice::Rocm,
        device: Some(selected.device.id),
        ..fusion_pcu::global::PcuExecutionPolicy::default()
    })?;
    println!("Owned binary donor benchmark device: {}", selected.name);
    for elements in [65, 1_048_576] {
        for profile in BinaryProfile::ALL {
            run_case(criterion, &runtime, architecture, elements, profile)?;
        }
    }
    Ok(())
}

#[allow(clippy::too_many_lines)]
#[allow(clippy::significant_drop_tightening)] // Criterion groups consume their borrowed state.
#[allow(unsafe_code)] // Native control launches the same binary op with an in-place donor.
fn run_case(
    criterion: &mut Criterion,
    runtime: &HipRuntime,
    architecture: &str,
    elements: usize,
    profile: BinaryProfile,
) -> Result<(), Box<dyn Error>> {
    let donor_id = format!("native_{}", profile.name());
    let left = if profile.donor_is_left() {
        "donor[id]"
    } else {
        "other[id]"
    };
    let right = if profile.donor_is_left() {
        "other[id]"
    } else {
        "donor[id]"
    };
    let source = format!(
        "#include <hip/hip_runtime.h>\nextern \"C\" __global__ void {donor_id}(float* donor, const float* other) {{ const unsigned int id = blockIdx.x * blockDim.x + threadIdx.x; if (id < {elements}u) {{ const float left = {left}; const float right = {right}; donor[id] = {}; }} }}\n",
        profile.native_expression()
    );
    let image = crate::support::cold_once("binary donor native HIP compilation", || {
        compile_hip_source(&source, architecture)
    })?;
    let module = runtime.load_module(&image)?;
    let kernel = module.function(&std::ffi::CString::new(donor_id)?)?;
    let stream = runtime.create_stream()?;
    let grid = u32::try_from(elements)?.div_ceil(BLOCK);
    let mut inputs = ResidentInputs::new(elements, runtime)?;
    inputs.profile = profile;
    inputs.refresh(0, elements)?;

    // Cold source compilation/preparation and correctness validation stay outside measurement.
    let source_time = inputs.run_source(profile)?;
    inputs.refresh(1, elements)?;
    let native_time = inputs.run_native(&kernel, &stream, grid)?;
    println!(
        "Owned donor warm-up {} {elements}: source {source_time:?}, matched native in-place {native_time:?}; readback oracle verified both result and untouched peer.",
        profile.name()
    );

    // Record one warm execution's benchmark-thread Rust allocations. HIP/driver allocations are
    // intentionally invisible to this counter; readback and refresh are outside the capture.
    let mut heap = [alloc::AllocationCounts::default(); 2];
    inputs.refresh(2, elements)?;
    let capture = alloc::AllocationCapture::start();
    let (_, source_output) = inputs.execute_source(profile)?;
    heap[0] = alloc::AllocationCapture::finish();
    drop(capture);
    inputs.verify_source(source_output)?;
    inputs.refresh(3, elements)?;
    let capture = alloc::AllocationCapture::start();
    let _ = inputs.execute_native(&kernel, &stream, grid)?;
    heap[1] = alloc::AllocationCapture::finish();
    drop(capture);
    inputs.verify_native()?;
    println!(
        "Owned donor Rust heap census {} {elements}: source {}/{}/{} B, native in-place {}/{}/{} B allocations/reallocations/requested; excludes input refresh, readback, and HIP/driver allocations.",
        profile.name(),
        heap[0].alloc_calls,
        heap[0].realloc_calls,
        heap[0].requested_bytes,
        heap[1].alloc_calls,
        heap[1].realloc_calls,
        heap[1].requested_bytes,
    );

    let mut paired_source = [Duration::ZERO; PAIRED_SAMPLES];
    let mut paired_native = [Duration::ZERO; PAIRED_SAMPLES];
    for sample in 0..PAIRED_SAMPLES {
        let job = u64::try_from(sample + 4)?;
        inputs.refresh(job, elements)?;
        if sample.is_multiple_of(2) {
            paired_source[sample] = inputs.run_source(profile)?;
            inputs.refresh(job, elements)?;
            paired_native[sample] = inputs.run_native(&kernel, &stream, grid)?;
        } else {
            paired_native[sample] = inputs.run_native(&kernel, &stream, grid)?;
            inputs.refresh(job, elements)?;
            paired_source[sample] = inputs.run_source(profile)?;
        }
    }
    let mut paired_ratios = [0.0_f64; PAIRED_SAMPLES];
    for (index, ratio) in paired_ratios.iter_mut().enumerate() {
        *ratio = paired_source[index].as_secs_f64() / paired_native[index].as_secs_f64();
    }
    paired_source.sort_unstable();
    paired_native.sort_unstable();
    paired_ratios.sort_unstable_by(f64::total_cmp);
    let source_median = paired_source[PAIRED_SAMPLES / 2 - 1]
        .as_secs_f64()
        .midpoint(paired_source[PAIRED_SAMPLES / 2].as_secs_f64());
    let native_median = paired_native[PAIRED_SAMPLES / 2 - 1]
        .as_secs_f64()
        .midpoint(paired_native[PAIRED_SAMPLES / 2].as_secs_f64());
    println!(
        "Owned donor paired diagnostic {} {elements}, {PAIRED_SAMPLES} alternating exact-input pairs: source median {:.3} us, native in-place median {:.3} us, median paired source/native ratio {:.4}; host diagnostics, not Criterion confidence intervals.",
        profile.name(),
        source_median * 1.0e6,
        native_median * 1.0e6,
        paired_ratios[PAIRED_SAMPLES / 2 - 1].midpoint(paired_ratios[PAIRED_SAMPLES / 2]),
    );

    let mut job = u64::try_from(PAIRED_SAMPLES + 4)?;
    let mut group = criterion.benchmark_group("owned_binary_donor_inplace");
    group.throughput(Throughput::Elements(u64::try_from(elements)?));
    group.bench_function(
        BenchmarkId::new(format!("source_{}", profile.name()), elements),
        |bencher| {
            bencher.iter_custom(|iterations| {
                let mut total = Duration::ZERO;
                for _ in 0..iterations {
                    job = job.wrapping_add(1);
                    inputs
                        .refresh(job, elements)
                        .expect("refresh source/native peers");
                    total += inputs
                        .run_source(profile)
                        .unwrap_or_else(|error| panic!("source donor call failed: {error}"));
                }
                total
            });
        },
    );
    group.bench_function(
        BenchmarkId::new(format!("native_{}", profile.name()), elements),
        |bencher| {
            bencher.iter_custom(|iterations| {
                let mut total = Duration::ZERO;
                for _ in 0..iterations {
                    job = job.wrapping_add(1);
                    inputs
                        .refresh(job, elements)
                        .expect("refresh source/native peers");
                    total += inputs
                        .run_native(&kernel, &stream, grid)
                        .unwrap_or_else(|error| panic!("native donor dispatch failed: {error}"));
                }
                total
            });
        },
    );
    group.finish();
    Ok(())
}

fn fill(donor: &mut [f32], other: &mut [f32], job: u64) {
    for (index, (donor_value, other_value)) in donor.iter_mut().zip(other).enumerate() {
        let lane = f32::from(u16::try_from(index % 128).expect("lane fits")) * 0.03125;
        let shift = u32::try_from((index % 4) * 16).expect("shift fits");
        let peer_shift = u32::try_from(((index + 1) % 4) * 16).expect("peer shift fits");
        let donor_job = f32::from(u16::try_from((job >> shift) & 0xffff).expect("chunk fits"));
        let other_job = f32::from(u16::try_from((job >> peer_shift) & 0xffff).expect("chunk fits"));
        *donor_value = 1.0 + lane + donor_job / 65_536.0;
        *other_value = lane.mul_add(0.5, 2.0) + other_job / 65_536.0;
    }
}

fn verify(expected: &[f32], expected_other: &[f32], observed: &[f32], observed_other: &[f32]) {
    assert_eq!(expected.len(), expected_other.len());
    assert_eq!(expected.len(), observed.len());
    assert_eq!(expected.len(), observed_other.len());
    for (((&value, &other), &actual), &actual_other) in expected
        .iter()
        .zip(expected_other)
        .zip(observed)
        .zip(observed_other)
    {
        assert_eq!(
            actual.to_bits(),
            value.to_bits(),
            "binary donor oracle mismatch"
        );
        assert_eq!(
            actual_other.to_bits(),
            other.to_bits(),
            "read-only peer changed"
        );
    }
}
