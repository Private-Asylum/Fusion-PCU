//! Native-endian access to the cold-selected padding-free float/integer representations.
#![allow(unsafe_code)]
use fusion_pcu::PcuScalar;
use super::Error;

pub(super) fn read<T: PcuScalar>(bytes: &[u8], offset: usize) -> Result<T, Error> {
    let end = offset
        .checked_add(T::HOST_SIZE)
        .ok_or(Error::ExtentOverflow)?;
    let bytes = bytes.get(offset..end).ok_or(Error::InvalidProgram)?;
    // SAFETY: The only callers use cold-selected checked floats or primitive integers,
    // each a sealed padding-free representation; Bool and other carriers never enter here.
    // Every initialized bit pattern is a valid Rust value. The checked byte span covers one
    // complete carrier. read_unaligned creates no reference or retained pointer.
    Ok(unsafe { bytes.as_ptr().cast::<T>().read_unaligned() })
}
pub(super) fn write<T: PcuScalar>(bytes: &mut [u8], offset: usize, value: T) -> Result<(), Error> {
    let end = offset
        .checked_add(T::HOST_SIZE)
        .ok_or(Error::ExtentOverflow)?;
    let bytes = bytes.get_mut(offset..end).ok_or(Error::InvalidProgram)?;
    // SAFETY: The exclusive checked span covers one complete padding-free carrier.
    // write_unaligned initializes all its bytes without an aligned reference or escape.
    unsafe {
        bytes.as_mut_ptr().cast::<T>().write_unaligned(value);
    }
    Ok(())
}
