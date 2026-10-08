//! Native-endian access to the cold-selected padding-free float/integer representations.
#![allow(unsafe_code)]
use fusion_pcu::PcuScalar;
use super::Error;

pub(super) fn read<T: PcuScalar>(bytes: &[u8], offset: usize) -> Result<T, Error> {
    let end = offset
        .checked_add(T::HOST_SIZE)
        .ok_or(Error::ExtentOverflow)?;
    let bytes = bytes.get(offset..end).ok_or(Error::InvalidProgram)?;
    // SAFETY: The only callers use cold-selected checked float/integer carriers,
    // each a sealed padding-free representation; Bool and other carriers never enter here.
    // Every initialized bit pattern is a valid Rust value. The checked byte span covers one
    // complete carrier. read_unaligned creates no reference or retained pointer.
    Ok(unsafe { bytes.as_ptr().cast::<T>().read_unaligned() })
}
/// Per-call resource bases; constructed only after extent and alias preflight.
/// Pointers are never retained in a plan or used after publication begins.
#[derive(Clone, Copy)]
pub(super) struct BoundResource {
    pub(super) read: *const u8,
    pub(super) write: *mut u8,
}
impl BoundResource {
    pub(super) const EMPTY: Self = Self {
        read: core::ptr::null(),
        write: core::ptr::null_mut(),
    };
    /// # Safety
    /// The element must lie in this call's live validated resource span, and T must
    /// be its cold-selected padding-free carrier, valid for all bit patterns.
    pub(super) const unsafe fn read<T: PcuScalar>(self, element: usize) -> T {
        unsafe {
            self.read
                .add(element * T::HOST_SIZE)
                .cast::<T>()
                .read_unaligned()
        }
    }
    /// # Safety
    /// This must be a writable resource with the element in its live private scratch
    /// span, using the cold-selected padding-free T. No scratch reference may alias.
    pub(super) const unsafe fn write<T: PcuScalar>(self, element: usize, value: T) {
        unsafe {
            self.write
                .add(element * T::HOST_SIZE)
                .cast::<T>()
                .write_unaligned(value);
        }
    }
}
