//! Benchmark-thread Rust allocation instrumentation.

use std::{
    alloc::{
        GlobalAlloc,
        Layout,
        System,
    },
    cell::Cell,
};

// Tracks Rust allocations on the benchmark thread only. HIP/ROCr allocations performed inside
// native libraries do not pass through Rust's global allocator and are intentionally excluded.
#[global_allocator]
static BENCH_ALLOCATOR: TrackingAllocator = TrackingAllocator;

struct TrackingAllocator;

#[derive(Clone, Copy, Default)]
pub struct AllocationCounts {
    pub alloc_calls: usize,
    pub realloc_calls: usize,
    pub dealloc_calls: usize,
    pub requested_bytes: usize,
}

impl AllocationCounts {
    #[must_use]
    pub const fn since(self, previous: Self) -> Self {
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

pub struct AllocationCapture;

impl AllocationCapture {
    #[must_use]
    pub fn start() -> Self {
        ALLOCATION_COUNTS.with(|counts| counts.set(AllocationCounts::default()));
        TRACK_ALLOCATIONS.with(|tracking| tracking.set(true));
        Self
    }

    pub fn finish() -> AllocationCounts {
        TRACK_ALLOCATIONS.with(|tracking| tracking.set(false));
        ALLOCATION_COUNTS.with(Cell::get)
    }

    pub fn snapshot() -> AllocationCounts {
        ALLOCATION_COUNTS.with(Cell::get)
    }
}

impl Drop for AllocationCapture {
    fn drop(&mut self) {
        TRACK_ALLOCATIONS.with(|tracking| tracking.set(false));
    }
}
