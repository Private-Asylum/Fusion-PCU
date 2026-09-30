//! CPU-only regression tests for the opt-in benchmark allocator.
#![cfg(feature = "allocation-census")]

#[path = "../benches/support/allocations/allocations.rs"]
mod allocations;

#[rustfmt::skip]
use std::{
    alloc::{
        Layout,
        alloc,
        dealloc,
        handle_alloc_error,
        realloc,
    },
    hint::black_box,
    sync::{
        Arc,
        Barrier,
    },
};

#[test]
fn counts_allocation_reallocation_and_deallocation_requests() {
    let original = Layout::from_size_align(32, 8).unwrap();
    let replacement_layout = Layout::from_size_align(64, 8).unwrap();
    let ((), counts) = allocations::measure(|| {
        // SAFETY: both layouts are nonzero and share an alignment. A successful allocation
        // supplies the pointer and original layout for realloc; successful realloc supplies
        // the replacement pointer and its exact new layout for dealloc. Null allocation
        // failures abort through handle_alloc_error instead of using invalid pointers.
        unsafe {
            let pointer = black_box(alloc(original));
            if pointer.is_null() {
                handle_alloc_error(original);
            }
            let replacement = black_box(realloc(pointer, original, replacement_layout.size()));
            if replacement.is_null() {
                // Failed realloc leaves the original allocation valid and owned here.
                dealloc(pointer, original);
                handle_alloc_error(replacement_layout);
            }
            dealloc(replacement, replacement_layout);
        }
    });
    assert_eq!(counts.alloc_calls, 1);
    assert_eq!(counts.realloc_calls, 1);
    assert_eq!(counts.dealloc_calls, 1);
    assert_eq!(counts.requested_bytes, 96);
}

#[test]
fn unwind_disables_capture_and_allows_a_fresh_empty_scope() {
    let panic = std::panic::catch_unwind(|| {
        allocations::measure(|| panic!("exercise census unwind cleanup"));
    });
    assert!(panic.is_err());
    let ((), counts) = allocations::measure(|| {});
    assert_empty(counts);
    allocations::census("after_unwind", || {});
}

#[test]
fn allocations_on_another_thread_are_excluded() {
    let start = Arc::new(Barrier::new(2));
    let done = Arc::new(Barrier::new(2));
    let worker_start = Arc::clone(&start);
    let worker_done = Arc::clone(&done);
    // Thread setup and synchronization storage are prepared before caller-thread capture.
    let worker = std::thread::spawn(move || {
        worker_start.wait();
        let bytes = vec![7_u8; 4096];
        black_box(&bytes);
        drop(bytes);
        worker_done.wait();
    });
    let ((), counts) = allocations::measure(|| {
        start.wait();
        done.wait();
    });
    worker.join().unwrap();
    assert_empty(counts);
}

fn assert_empty(counts: allocations::Census) {
    assert_eq!(counts.alloc_calls, 0);
    assert_eq!(counts.realloc_calls, 0);
    assert_eq!(counts.dealloc_calls, 0);
    assert_eq!(counts.requested_bytes, 0);
}
