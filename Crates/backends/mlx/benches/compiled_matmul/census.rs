//! Diagnostic Rust allocator census, separate from primary timings and native SDK allocation.
#[rustfmt::skip]
use std::{
    alloc::{
        GlobalAlloc,
        Layout,
        System,
    },
    cell::Cell,
    sync::atomic::{
        AtomicUsize,
        Ordering,
    },
};

struct Counting;
#[global_allocator]
static ALLOCATOR: Counting = Counting;
static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
static BYTES: AtomicUsize = AtomicUsize::new(0);
static FREES: AtomicUsize = AtomicUsize::new(0);
std::thread_local! { static ACTIVE: Cell<bool> = const { Cell::new(false) }; }

fn active() -> bool {
    ACTIVE.try_with(Cell::get).unwrap_or(false)
}

// SAFETY: the unchanged layouts/pointers are delegated to System; counters never own storage.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: GlobalAlloc caller supplied the validated layout; System owns the result.
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() && active() {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
            BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        }
        pointer
    }
    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        if active() {
            FREES.fetch_add(1, Ordering::Relaxed);
        }
        // SAFETY: GlobalAlloc caller supplied the live System pointer and original layout.
        unsafe { System.dealloc(pointer, layout) };
    }
}

pub fn measure<T>(operation: impl FnOnce() -> T) -> (T, [usize; 3]) {
    let before = [
        ALLOCATIONS.load(Ordering::Relaxed),
        BYTES.load(Ordering::Relaxed),
        FREES.load(Ordering::Relaxed),
    ];
    ACTIVE.with(|active| active.set(true));
    let result = operation();
    ACTIVE.with(|active| active.set(false));
    (
        result,
        [
            ALLOCATIONS.load(Ordering::Relaxed) - before[0],
            BYTES.load(Ordering::Relaxed) - before[1],
            FREES.load(Ordering::Relaxed) - before[2],
        ],
    )
}
