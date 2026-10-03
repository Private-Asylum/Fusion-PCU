//! Borrowed carrier bytes at the independent native API boundary.
use pcu_facade::PcuScalar;
pub fn input<T: PcuScalar>(values: &[T]) -> &[u8] {
    assert_eq!(core::mem::size_of::<T>(), T::HOST_SIZE);
    const {
        assert!(cfg!(target_endian = "little"));
    }
    // SAFETY: All sealed carriers have initialized bytes, exact fixed HOST_SIZE,
    // and no padding. The shared byte view is bounded by the original borrow.
    unsafe { core::slice::from_raw_parts(values.as_ptr().cast(), core::mem::size_of_val(values)) }
}
pub fn output<T: PcuScalar>(values: &mut [T]) -> &mut [u8] {
    assert_eq!(core::mem::size_of::<T>(), T::HOST_SIZE);
    const {
        assert!(cfg!(target_endian = "little"));
    }
    // SAFETY: Sealed scalar carriers permit every stored raw encoding. This
    // exclusive byte view cannot outlive or alias the original mutable borrow.
    unsafe {
        core::slice::from_raw_parts_mut(values.as_mut_ptr().cast(), core::mem::size_of_val(values))
    }
}
