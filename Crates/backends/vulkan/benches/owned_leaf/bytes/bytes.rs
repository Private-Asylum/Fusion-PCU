//! Sealed carrier byte views retain the original Rust slice lifetime.
use pcu_facade::PcuScalar;
pub const fn read<T: PcuScalar>(values: &[T]) -> &[u8] {
    // SAFETY: All sealed carriers are padding-free and initialized. This read-only view
    // borrows precisely the complete initialized slice and cannot escape its lifetime.
    unsafe { core::slice::from_raw_parts(values.as_ptr().cast(), core::mem::size_of_val(values)) }
}
pub const fn write<T: PcuScalar>(values: &mut [T]) -> &mut [u8] {
    // SAFETY: Every bit pattern of these sealed carriers is valid. The exclusive borrow
    // covers the exact slice bytes and retains its lifetime and initialized representation.
    unsafe {
        core::slice::from_raw_parts_mut(values.as_mut_ptr().cast(), core::mem::size_of_val(values))
    }
}
