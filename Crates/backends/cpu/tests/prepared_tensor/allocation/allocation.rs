//! Thread-local warm allocation census, excluding unrelated test threads.
#[rustfmt::skip]
use std::{
    alloc::{
        GlobalAlloc,
        Layout,
        System,
    },
    cell::Cell,
};
thread_local! {
    static ACTIVE: Cell<bool> = const { Cell::new(false) };
    static COUNT: Cell<usize> = const { Cell::new(0) };
}
struct Census;
// SAFETY: Every allocation is delegated unchanged to System, preserving its layout contract.
unsafe impl GlobalAlloc for Census {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if ACTIVE.try_with(Cell::get).unwrap_or(false) {
            let _ = COUNT.try_with(|count| count.set(count.get() + 1));
        }
        // SAFETY: Forward the exact allocator layout unchanged.
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        // SAFETY: GlobalAlloc callers provide this allocation's original pointer and layout.
        unsafe { System.dealloc(pointer, layout) }
    }
}
#[global_allocator]
static ALLOCATOR: Census = Census;
pub fn count(operation: impl FnOnce()) -> usize {
    COUNT.with(|count| count.set(0));
    ACTIVE.with(|active| active.set(true));
    operation();
    ACTIVE.with(|active| active.set(false));
    COUNT.with(Cell::get)
}
