//! Dispatch timing and allocation instrumentation shared by the benchmark composition.

use std::{
    alloc::{
        GlobalAlloc,
        Layout,
        System,
    },
    cell::Cell,
    error::Error,
    time::{
        Duration,
        Instant,
    },
};

use fusion_pcu::{
    PcuBindingAccess,
    PcuBindingRef,
    PcuBindingType,
    PcuCompletionOutcome,
    PcuOwnedCompletion,
    PcuValueType,
};
use fusion_pcu_rocm::{
    HipCompletion,
    HipKernel,
    HipKernelArgument,
    RocmOwnedDispatchBackend,
};

pub const BLOCK_SIZE: u32 = 64;
pub const READBACK_SAMPLES: usize = 64;

// Tracks Rust allocations on the benchmark thread only. HIP/ROCr allocations performed inside
// native libraries do not pass through Rust's global allocator and are intentionally excluded.
#[global_allocator]
static BENCH_ALLOCATOR: TrackingAllocator = TrackingAllocator;

struct TrackingAllocator;

#[derive(Clone, Copy, Default)]
struct AllocationCounts {
    alloc_calls: usize,
    realloc_calls: usize,
    dealloc_calls: usize,
    requested_bytes: usize,
}

impl AllocationCounts {
    const fn since(self, previous: Self) -> Self {
        Self {
            alloc_calls: self.alloc_calls.saturating_sub(previous.alloc_calls),
            realloc_calls: self.realloc_calls.saturating_sub(previous.realloc_calls),
            dealloc_calls: self.dealloc_calls.saturating_sub(previous.dealloc_calls),
            requested_bytes: self
                .requested_bytes
                .saturating_sub(previous.requested_bytes),
        }
    }
}

thread_local! {
    static TRACK_ALLOCATIONS: Cell<bool> = const { Cell::new(false) };
    static ALLOCATION_COUNTS: Cell<AllocationCounts> = const { Cell::new(AllocationCounts {
        alloc_calls: 0,
        realloc_calls: 0,
        dealloc_calls: 0,
        requested_bytes: 0,
    }) };
}

// SAFETY: This allocator delegates every operation to `System` with the exact layout and pointer
// supplied by the caller. Tracking only updates thread-local counters and does not alter memory.
#[allow(unsafe_code)]
unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: Delegated unchanged to the system allocator.
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() {
            record_allocation(layout.size(), false);
        }
        pointer
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        record_deallocation();
        // SAFETY: Delegated unchanged to the system allocator.
        unsafe { System.dealloc(pointer, layout) };
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // SAFETY: Delegated unchanged to the system allocator.
        let replacement = unsafe { System.realloc(pointer, layout, new_size) };
        if !replacement.is_null() {
            record_allocation(new_size, true);
        }
        replacement
    }
}

fn record_allocation(bytes: usize, realloc: bool) {
    let _ = TRACK_ALLOCATIONS.try_with(|tracking| {
        if tracking.get() {
            let _ = ALLOCATION_COUNTS.try_with(|counts_cell| {
                let mut counts = counts_cell.get();
                if realloc {
                    counts.realloc_calls += 1;
                } else {
                    counts.alloc_calls += 1;
                }
                counts.requested_bytes += bytes;
                counts_cell.set(counts);
            });
        }
    });
}

fn record_deallocation() {
    let _ = TRACK_ALLOCATIONS.try_with(|tracking| {
        if tracking.get() {
            let _ = ALLOCATION_COUNTS.try_with(|counts_cell| {
                let mut counts = counts_cell.get();
                counts.dealloc_calls += 1;
                counts_cell.set(counts);
            });
        }
    });
}

struct AllocationCapture;

impl AllocationCapture {
    fn start() -> Self {
        ALLOCATION_COUNTS.with(|counts| counts.set(AllocationCounts::default()));
        TRACK_ALLOCATIONS.with(|tracking| tracking.set(true));
        Self
    }

    fn finish() -> AllocationCounts {
        TRACK_ALLOCATIONS.with(|tracking| tracking.set(false));
        ALLOCATION_COUNTS.with(Cell::get)
    }

    fn snapshot() -> AllocationCounts {
        ALLOCATION_COUNTS.with(Cell::get)
    }
}

impl Drop for AllocationCapture {
    fn drop(&mut self) {
        TRACK_ALLOCATIONS.with(|tracking| tracking.set(false));
    }
}

pub fn timed_copy(
    source: &fusion_pcu_rocm::DeviceBuffer,
    destination: &mut [u8],
) -> Result<Duration, fusion_pcu_rocm::HipError> {
    let started = Instant::now();
    source.copy_to(destination)?;
    Ok(started.elapsed())
}

pub fn run_prepared(
    backend: &RocmOwnedDispatchBackend,
    prepared: &fusion_pcu_rocm::RocmPreparedDispatch<'_>,
    input: &fusion_pcu_rocm::DeviceBuffer,
    output: &fusion_pcu_rocm::DeviceBuffer,
) -> Result<Sample, Box<dyn Error>> {
    let _capture = AllocationCapture::start();
    let total_started = Instant::now();
    let bind_started = Instant::now();
    let bindings = [
        backend.binding(
            PcuBindingRef::new(0, 0),
            PcuBindingAccess::ReadOnly,
            PcuBindingType::Value(PcuValueType::f32()),
            input.clone(),
        )?,
        backend.binding(
            PcuBindingRef::new(0, 1),
            PcuBindingAccess::WriteOnly,
            PcuBindingType::Value(PcuValueType::f32()),
            output.clone(),
        )?,
    ];
    let bind = bind_started.elapsed();
    let after_bind = AllocationCapture::snapshot();
    let launch_started = Instant::now();
    let mut completion = prepared.submit(&bindings)?;
    let launch_return = launch_started.elapsed();
    let after_submit = AllocationCapture::snapshot();
    let wait_started = Instant::now();
    if completion.wait()? != PcuCompletionOutcome::Succeeded {
        return Err("prepared dispatch did not succeed".into());
    }
    let allocations = AllocationCapture::finish();
    Ok(Sample {
        bind: Some(bind),
        launch_return,
        wait: wait_started.elapsed(),
        total: total_started.elapsed(),
        allocations,
        bind_allocations: Some(after_bind),
        submit_allocations: after_submit.since(after_bind),
        wait_allocations: allocations.since(after_submit),
    })
}

pub fn run_direct(
    function: &HipKernel,
    stream: &fusion_pcu_rocm::HipStreamHandle,
    arguments: &[HipKernelArgument<'_>],
    grid: u32,
) -> Result<Sample, Box<dyn Error>> {
    let _capture = AllocationCapture::start();
    let total_started = Instant::now();
    let launch_started = Instant::now();
    let mut completion = direct_launch(function, stream, arguments, grid)?;
    let launch_return = launch_started.elapsed();
    let after_submit = AllocationCapture::snapshot();
    let wait_started = Instant::now();
    completion.wait()?;
    let allocations = AllocationCapture::finish();
    Ok(Sample {
        bind: None,
        launch_return,
        wait: wait_started.elapsed(),
        total: total_started.elapsed(),
        allocations,
        bind_allocations: None,
        submit_allocations: after_submit,
        wait_allocations: allocations.since(after_submit),
    })
}

#[allow(unsafe_code)]
fn direct_launch(
    function: &HipKernel,
    stream: &fusion_pcu_rocm::HipStreamHandle,
    arguments: &[HipKernelArgument<'_>],
    grid: u32,
) -> Result<HipCompletion, fusion_pcu_rocm::HipError> {
    // SAFETY: the handwritten HIP kernel takes input and output pointers in this order, guards
    // every access by its element count, and uses the same grid/block geometry as the PCU route.
    // Both buffers have the declared element count, and the caller waits before reuse.
    unsafe { function.launch(stream, [grid, 1, 1], [BLOCK_SIZE, 1, 1], 0, arguments) }
}

#[derive(Clone, Copy)]
pub struct Sample {
    bind: Option<Duration>,
    launch_return: Duration,
    wait: Duration,
    pub total: Duration,
    allocations: AllocationCounts,
    bind_allocations: Option<AllocationCounts>,
    submit_allocations: AllocationCounts,
    wait_allocations: AllocationCounts,
}

pub fn print_samples(label: &str, samples: &[Sample]) {
    if samples.iter().all(|sample| sample.bind.is_some()) {
        print_metric(label, "binding construction", samples, |sample| {
            sample.bind.unwrap_or_default()
        });
    }
    print_metric(label, "submit/enqueue host return", samples, |sample| {
        sample.launch_return
    });
    print_metric(label, "completion wait", samples, |sample| sample.wait);
    print_metric(label, "end-to-end total", samples, |sample| sample.total);
    if samples
        .iter()
        .all(|sample| sample.bind_allocations.is_some())
    {
        print_allocation_metric(label, "binding", samples, |sample| {
            sample.bind_allocations.unwrap_or_default()
        });
    }
    print_allocation_metric(label, "submit/enqueue", samples, |sample| {
        sample.submit_allocations
    });
    print_allocation_metric(label, "completion wait", samples, |sample| {
        sample.wait_allocations
    });
    print_allocation_metric(label, "total", samples, |sample| sample.allocations);
}

fn print_allocation_metric(
    label: &str,
    stage: &str,
    samples: &[Sample],
    counts: impl Fn(&Sample) -> AllocationCounts,
) {
    let mut values = samples.iter().map(counts).collect::<Vec<_>>();
    values.sort_unstable_by_key(|counts| {
        (
            counts.alloc_calls + counts.realloc_calls,
            counts.requested_bytes,
        )
    });
    let p50 = percentile_index(values.len(), 50);
    let p95 = percentile_index(values.len(), 95);
    let show = |counts: AllocationCounts| {
        format!(
            "{} calls ({} alloc, {} realloc), {} requested B, {} frees",
            counts.alloc_calls + counts.realloc_calls,
            counts.alloc_calls,
            counts.realloc_calls,
            counts.requested_bytes,
            counts.dealloc_calls,
        )
    };
    println!(
        "{label} Rust heap {stage}: p50={} p95={} (benchmark-thread global allocator only; excludes native HIP/driver allocations)",
        show(values[p50]),
        show(values[p95]),
    );
}

fn print_metric(
    label: &str,
    metric: &str,
    samples: &[Sample],
    value: impl Fn(&Sample) -> Duration,
) {
    let mut values = samples.iter().map(value).collect::<Vec<_>>();
    values.sort_unstable();
    let p50 = percentile(&values, 50);
    let p95 = percentile(&values, 95);
    println!(
        "{label} {metric}: p50={:.3} us p95={:.3} us ({} runs)",
        p50.as_secs_f64() * 1_000_000.0,
        p95.as_secs_f64() * 1_000_000.0,
        values.len(),
    );
}

pub fn print_duration_samples(label: &str, samples: &[Duration]) {
    let mut values = samples.to_vec();
    values.sort_unstable();
    println!(
        "{label}: p50={:.3} us p95={:.3} us",
        percentile(&values, 50).as_secs_f64() * 1_000_000.0,
        percentile(&values, 95).as_secs_f64() * 1_000_000.0,
    );
}

fn percentile(samples: &[Duration], percentile: usize) -> Duration {
    samples[percentile_index(samples.len(), percentile)]
}

const fn percentile_index(len: usize, percentile: usize) -> usize {
    len.saturating_mul(percentile)
        .div_ceil(100)
        .saturating_sub(1)
}

pub fn encode_f32(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect()
}

pub fn verify_output(label: &str, bytes: &[u8], input: &[f32]) -> Result<(), Box<dyn Error>> {
    for (index, chunk) in bytes.chunks_exact(size_of::<f32>()).enumerate() {
        let actual = f32::from_ne_bytes(chunk.try_into()?);
        let expected = input[index] + 1.0;
        if (actual - expected).abs() > f32::EPSILON {
            return Err(format!("{label} output[{index}]={actual}; expected {expected}").into());
        }
    }
    Ok(())
}
