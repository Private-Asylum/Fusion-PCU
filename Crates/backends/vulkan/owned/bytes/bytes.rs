//! Sealed scalar byte borrows. Native foreign calls live in the FFI owner module.
#[rustfmt::skip]
use fusion_pcu::PcuScalar;

pub const fn read<T: PcuScalar>(values: &[T]) -> &[u8] {
    unsafe {
        // SAFETY: Every sealed scalar is padding-free. Native entry validates the little-endian
        // host ABI and exact width; this view borrows only the live immutable slice's storage.
        core::slice::from_raw_parts(values.as_ptr().cast(), core::mem::size_of_val(values))
    }
}

pub(super) const fn write<T: PcuScalar>(values: &mut [T]) -> &mut [u8] {
    unsafe {
        // SAFETY: Every sealed scalar admits all bit patterns and has no padding. The exclusive
        // byte view lasts no longer than this slice borrow and native copies preserve raw bits.
        core::slice::from_raw_parts_mut(values.as_mut_ptr().cast(), core::mem::size_of_val(values))
    }
}
