//! Opt-in caller-thread Rust heap census, absent from primary builds.
#![cfg(feature = "allocation-census")]

#[rustfmt::skip]
use std::{
    alloc::{
        GlobalAlloc,
        Layout,
        System,
    },
    cell::Cell,
};

struct CensusAllocator;

#[derive(Clone, Copy, Default)]
pub struct Census {
    pub alloc_calls: usize,
    pub realloc_calls: usize,
    pub dealloc_calls: usize,
    pub requested_bytes: usize,
}

#[derive(Clone, Copy)]
struct State {
    active: bool,
    counts: Census,
}

thread_local! {
    // Constant initialization and a destructor-free Cell avoid allocation or recursion when
    // an allocator callback first touches this thread. No maps, formatting or locks occur here.
    static STATE: Cell<State> = const { Cell::new(State {
        active: false,
        counts: Census {
            alloc_calls: 0,
            realloc_calls: 0,
            dealloc_calls: 0,
            requested_bytes: 0,
        },
    }) };
}

fn record(alloc: usize, realloc: usize, dealloc: usize, bytes: usize) {
    // Ignore unavailable TLS during thread teardown rather than panicking in the allocator.
    let _ = STATE.try_with(|cell| {
        let mut state = cell.get();
        if state.active {
            state.counts.alloc_calls = state.counts.alloc_calls.saturating_add(alloc);
            state.counts.realloc_calls = state.counts.realloc_calls.saturating_add(realloc);
            state.counts.dealloc_calls = state.counts.dealloc_calls.saturating_add(dealloc);
            state.counts.requested_bytes = state.counts.requested_bytes.saturating_add(bytes);
            cell.set(state);
        }
    });
}

// SAFETY: every operation forwards the exact pointer, layout and size contract to System.
// Recording modifies only fixed thread-local counters and never touches the allocated memory.
unsafe impl GlobalAlloc for CensusAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        record(1, 0, 0, layout.size());
        // SAFETY: the caller's valid layout is passed unchanged to System.
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        record(1, 0, 0, layout.size());
        // SAFETY: unchanged layout delegation preserves System's zeroed allocation semantics.
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        record(0, 1, 0, size);
        // SAFETY: pointer, original layout and new size retain the caller's realloc contract.
        unsafe { System.realloc(pointer, layout, size) }
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        record(0, 0, 1, 0);
        // SAFETY: the original pointer and layout retain the caller's deallocation contract.
        unsafe { System.dealloc(pointer, layout) }
    }
}

#[global_allocator]
static ALLOCATOR: CensusAllocator = CensusAllocator;

struct Capture;

impl Capture {
    fn start() -> Self {
        STATE.with(|cell| {
            let state = cell.get();
            assert!(!state.active, "allocation census cannot be nested");
            cell.set(State {
                active: true,
                counts: Census::default(),
            });
        });
        Self
    }

    fn finish(self) -> Census {
        let counts = STATE.with(|cell| {
            let mut state = cell.get();
            state.active = false;
            cell.set(state);
            state.counts
        });
        drop(self);
        counts
    }
}

impl Drop for Capture {
    fn drop(&mut self) {
        let _ = STATE.try_with(|cell| {
            let mut state = cell.get();
            state.active = false;
            cell.set(state);
        });
    }
}

/// Count only the caller thread while `call` executes, disabling capture even on unwind.
/// Requested bytes sum allocation layouts and realloc destination sizes, including attempts.
pub fn measure<T>(call: impl FnOnce() -> T) -> (T, Census) {
    let capture = Capture::start();
    let result = call();
    (result, capture.finish())
}
