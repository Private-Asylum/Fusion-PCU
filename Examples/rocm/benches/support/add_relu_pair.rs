//! Alternating, paired Add -> `ReLU` timing and Rust heap diagnostics.

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

const PAIRS: usize = 64;

#[global_allocator]
static ALLOCATOR: TrackingAllocator = TrackingAllocator;

struct TrackingAllocator;

#[derive(Clone, Copy, Default)]
struct AllocationCounts {
    allocs: usize,
    reallocs: usize,
    frees: usize,
    bytes: usize,
}

thread_local! {
    static TRACKING: Cell<bool> = const { Cell::new(false) };
    static COUNTS: Cell<AllocationCounts> = const { Cell::new(AllocationCounts {
        allocs: 0,
        reallocs: 0,
        frees: 0,
        bytes: 0,
    }) };
}

// SAFETY: Every allocation operation is forwarded to System with its original arguments. The
// counters are thread-local and updated only while the diagnostic explicitly enables tracking.
#[allow(unsafe_code)]
unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: Forwarding the caller-provided layout unchanged is valid.
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() {
            record(layout.size(), false);
        }
        pointer
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        record_free();
        // SAFETY: Forwarding the original pointer and layout is valid.
        unsafe { System.dealloc(pointer, layout) };
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // SAFETY: Forwarding the caller-provided arguments unchanged is valid.
        let replacement = unsafe { System.realloc(pointer, layout, new_size) };
        if !replacement.is_null() {
            record(new_size, true);
        }
        replacement
    }
}

fn record(bytes: usize, realloc: bool) {
    let _ = TRACKING.try_with(|tracking| {
        if tracking.get() {
            let _ = COUNTS.try_with(|counts| {
                let mut snapshot = counts.get();
                if realloc {
                    snapshot.reallocs += 1;
                } else {
                    snapshot.allocs += 1;
                }
                snapshot.bytes += bytes;
                counts.set(snapshot);
            });
        }
    });
}

fn record_free() {
    let _ = TRACKING.try_with(|tracking| {
        if tracking.get() {
            let _ = COUNTS.try_with(|counts| {
                let mut snapshot = counts.get();
                snapshot.frees += 1;
                counts.set(snapshot);
            });
        }
    });
}

struct Capture;

impl Capture {
    fn start() -> Self {
        // Touch both TLS cells before counting so initialization is excluded.
        TRACKING.with(|tracking| tracking.set(false));
        COUNTS.with(|counts| counts.set(AllocationCounts::default()));
        TRACKING.with(|tracking| tracking.set(true));
        Self
    }

    fn finish() -> AllocationCounts {
        TRACKING.with(|tracking| tracking.set(false));
        COUNTS.with(Cell::get)
    }
}

impl Drop for Capture {
    fn drop(&mut self) {
        TRACKING.with(|tracking| tracking.set(false));
    }
}

#[derive(Clone, Copy)]
struct Sample {
    elapsed: Duration,
    allocations: AllocationCounts,
}

fn measure(action: impl FnOnce() -> Result<(), Box<dyn Error>>) -> Result<Sample, Box<dyn Error>> {
    let _capture = Capture::start();
    let started = Instant::now();
    action()?;
    let elapsed = started.elapsed();
    let allocations = Capture::finish();
    Ok(Sample {
        elapsed,
        allocations,
    })
}

/// Measure the two synchronous paths in alternating order. Each action includes submission and
/// terminal completion because the public tensor API exposes only synchronous execution here.
pub fn run(
    elements: usize,
    left_label: &str,
    right_label: &str,
    mut left: impl FnMut() -> Result<(), Box<dyn Error>>,
    mut right: impl FnMut() -> Result<(), Box<dyn Error>>,
) -> Result<(), Box<dyn Error>> {
    // Warm both implementations before collecting pairs and exclude compilation/cold setup.
    left()?;
    right()?;

    let mut left_samples = Vec::with_capacity(PAIRS);
    let mut right_samples = Vec::with_capacity(PAIRS);
    for pair in 0..PAIRS {
        if pair.is_multiple_of(2) {
            left_samples.push(measure(&mut left)?);
            right_samples.push(measure(&mut right)?);
        } else {
            let right_sample = measure(&mut right)?;
            let left_sample = measure(&mut left)?;
            left_samples.push(left_sample);
            right_samples.push(right_sample);
        }
    }

    let mut paired_ratios = left_samples
        .iter()
        .zip(&right_samples)
        .map(|(left, right)| left.elapsed.as_secs_f64() / right.elapsed.as_secs_f64())
        .collect::<Vec<_>>();
    paired_ratios.sort_unstable_by(f64::total_cmp);

    println!(
        "Alternating paired diagnostic ({elements} elements, {PAIRS} pairs): {left_label}/{right_label} median ratio {:.3}x, p10-p90 {:.3}x-{:.3}x",
        percentile(&paired_ratios, 50),
        percentile(&paired_ratios, 10),
        percentile(&paired_ratios, 90),
    );
    print_samples(left_label, &left_samples);
    print_samples(right_label, &right_samples);
    Ok(())
}

fn print_samples(label: &str, samples: &[Sample]) {
    let mut elapsed = samples
        .iter()
        .map(|sample| sample.elapsed)
        .collect::<Vec<_>>();
    elapsed.sort_unstable();
    println!(
        "{label} end-to-end: median {:.3} us, p10-p90 {:.3}-{:.3} us",
        percentile_duration(&elapsed, 50),
        percentile_duration(&elapsed, 10),
        percentile_duration(&elapsed, 90),
    );

    let mut allocations = samples
        .iter()
        .map(|sample| sample.allocations)
        .collect::<Vec<_>>();
    allocations.sort_unstable_by_key(|counts| (counts.allocs + counts.reallocs, counts.bytes));
    let median = allocations[allocations.len() / 2];
    let p90 = allocations[percentile_index(allocations.len(), 90)];
    println!(
        "{label} benchmark-thread Rust heap per call: median {} alloc/realloc ({} alloc, {} realloc), {} requested B, {} frees; p90 {} calls / {} B; HIP/driver allocations excluded",
        median.allocs + median.reallocs,
        median.allocs,
        median.reallocs,
        median.bytes,
        median.frees,
        p90.allocs + p90.reallocs,
        p90.bytes,
    );
}

fn percentile(values: &[f64], percentile: usize) -> f64 {
    values[percentile_index(values.len(), percentile)]
}

fn percentile_duration(values: &[Duration], percentile: usize) -> f64 {
    values[percentile_index(values.len(), percentile)].as_secs_f64() * 1_000_000.0
}

const fn percentile_index(length: usize, percentile: usize) -> usize {
    length.saturating_sub(1) * percentile / 100
}
