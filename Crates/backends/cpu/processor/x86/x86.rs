//! Integer XOR avoids floating rounding, FTZ and DAZ mode dependencies.

#[cfg(target_arch = "x86")]
use core::arch::x86 as arch;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64 as arch;

#[target_feature(enable = "sse2")]
pub(super) unsafe fn sse2(input: &[u8], output: &mut [u8]) -> usize {
    let prefix = input.len() / 16 * 16;
    let mask = arch::_mm_set1_epi32(i32::MIN);
    for offset in (0..prefix).step_by(16) {
        // SAFETY: The guarded caller provides equal, disjoint slices; prefix rounds down to
        // complete 16-byte vectors. Unaligned intrinsics impose no extra alignment requirement.
        unsafe {
            let value = arch::_mm_loadu_si128(input.as_ptr().add(offset).cast());
            arch::_mm_storeu_si128(
                output.as_mut_ptr().add(offset).cast(),
                arch::_mm_xor_si128(value, mask),
            );
        }
    }
    prefix
}

#[target_feature(enable = "avx2")]
pub(super) unsafe fn avx2(input: &[u8], output: &mut [u8]) -> usize {
    let prefix = input.len() / 32 * 32;
    let mask = arch::_mm256_set1_epi32(i32::MIN);
    for offset in (0..prefix).step_by(32) {
        // SAFETY: Same slice bounds and unaligned memory law as the SSE2 entry, in 32-byte lanes.
        unsafe {
            let value = arch::_mm256_loadu_si256(input.as_ptr().add(offset).cast());
            arch::_mm256_storeu_si256(
                output.as_mut_ptr().add(offset).cast(),
                arch::_mm256_xor_si256(value, mask),
            );
        }
    }
    prefix
}

#[target_feature(enable = "sse2")]
pub(super) unsafe fn sse2_f64(input: &[u8], output: &mut [u8]) -> usize {
    let prefix = input.len() / 16 * 16;
    let mask = arch::_mm_set1_epi64x(i64::MIN);
    for offset in (0..prefix).step_by(16) {
        // SAFETY: The guarded caller supplies equal, disjoint slices. Prefix bounds complete
        // 16-byte vectors and the unaligned intrinsics require no typed/aligned reference.
        unsafe {
            let value = arch::_mm_loadu_si128(input.as_ptr().add(offset).cast());
            arch::_mm_storeu_si128(
                output.as_mut_ptr().add(offset).cast(),
                arch::_mm_xor_si128(value, mask),
            );
        }
    }
    prefix
}

#[target_feature(enable = "avx2")]
pub(super) unsafe fn avx2_f64(input: &[u8], output: &mut [u8]) -> usize {
    let prefix = input.len() / 32 * 32;
    let mask = arch::_mm256_set1_epi64x(i64::MIN);
    for offset in (0..prefix).step_by(32) {
        // SAFETY: The same equal/disjoint slice law bounds each complete 32-byte vector.
        unsafe {
            let value = arch::_mm256_loadu_si256(input.as_ptr().add(offset).cast());
            arch::_mm256_storeu_si256(
                output.as_mut_ptr().add(offset).cast(),
                arch::_mm256_xor_si256(value, mask),
            );
        }
    }
    prefix
}
