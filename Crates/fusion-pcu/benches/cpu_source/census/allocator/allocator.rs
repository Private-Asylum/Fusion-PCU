//! Current-thread Rust allocator calls only; no allocation in the observer itself.
use std::cell::Cell;
#[rustfmt::skip]
use std::alloc::{
    GlobalAlloc,
    Layout,
    System,
};

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Counts {
    pub allocations: usize,
    pub reallocations: usize,
    pub frees: usize,
    pub requested_bytes: usize,
}

std::thread_local! {
    static ACTIVE: Cell<bool> = const { Cell::new(false) };
    static COUNTS: Cell<Counts> = const { Cell::new(Counts {
        allocations: 0, reallocations: 0, frees: 0, requested_bytes: 0,
    }) };
}

pub struct CountingAllocator;

fn record(update: impl FnOnce(&mut Counts)) {
    if ACTIVE.try_with(Cell::get).unwrap_or(false) {
        let _ = COUNTS.try_with(|cell| {
            let mut counts = cell.get();
            update(&mut counts);
            cell.set(counts);
        });
    }
}

// SAFETY: All pointer/layout/size contracts pass through unchanged to System.
// The observer uses allocation-free constant-initialized thread-local Cells.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        record(|c| {
            c.allocations += 1;
            c.requested_bytes += layout.size();
        });
        // SAFETY: The caller supplied a valid allocation layout.
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        record(|c| {
            c.allocations += 1;
            c.requested_bytes += layout.size();
        });
        // SAFETY: The caller supplied a valid allocation layout.
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        record(|c| c.frees += 1);
        // SAFETY: Pointer and original allocation layout are forwarded unchanged.
        unsafe { System.dealloc(pointer, layout) }
    }
    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        record(|c| {
            c.reallocations += 1;
            c.requested_bytes += size;
        });
        // SAFETY: Original pointer/layout and the caller's nonzero size are unchanged.
        unsafe { System.realloc(pointer, layout, size) }
    }
}

struct Guard;
impl Drop for Guard {
    fn drop(&mut self) {
        ACTIVE.with(|active| active.set(false));
    }
}

pub fn observe<T>(action: impl FnOnce() -> T) -> (T, Counts) {
    ACTIVE.with(|active| assert!(!active.get(), "allocation scopes cannot nest"));
    COUNTS.with(|counts| counts.set(Counts::default()));
    ACTIVE.with(|active| active.set(true));
    let guard = Guard;
    let result = action();
    drop(guard);
    (result, COUNTS.with(Cell::get))
}
