//! Cold-frozen checked integer executable without native pointer access.
use super::PcuCpuCheckedIntegerError;
#[path = "ffi/ffi.rs"]
mod ffi;
pub(super) use ffi::prepare;
pub(super) type Executable =
    fn(&[u8], &[u8], &mut [u8], usize) -> Result<(), PcuCpuCheckedIntegerError>;

#[cfg(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64"))]
#[path = "simd/simd.rs"]
mod simd;

pub(super) fn prepare_selected<T: fusion_pcu::PcuCheckedInteger>(
    implementation: crate::PcuCpuImplementation,
    op: fusion_pcu::PcuDispatchIntegerBinaryOp,
    range: fusion_pcu::PcuRangePolicy,
    left: bool,
    right: bool,
) -> Result<Executable, PcuCpuCheckedIntegerError> {
    if implementation == crate::PcuCpuImplementation::Scalar {
        return Ok(prepare::<T>(op, range, left, right));
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64"))]
    {
        simd::prepare::<T>(implementation, op, range, left, right)
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
    {
        Err(PcuCpuCheckedIntegerError::UnsupportedProfile)
    }
}
