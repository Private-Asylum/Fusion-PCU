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

#[target_feature(enable = "avx")]
pub(super) unsafe fn avx(input: &[u8], output: &mut [u8]) -> usize {
    let prefix = input.len() / 32 * 32;
    let mask = arch::_mm256_set1_ps(f32::from_bits(0x8000_0000));
    for offset in (0..prefix).step_by(32) {
        // SAFETY: Detection proves AVX/OS YMM state; complete unaligned disjoint spans.
        // XOR is a bit operation, including for NaN/subnormal encodings; no FP arithmetic.
        unsafe {
            let value = arch::_mm256_loadu_ps(input.as_ptr().add(offset).cast());
            arch::_mm256_storeu_ps(
                output.as_mut_ptr().add(offset).cast(),
                arch::_mm256_xor_ps(value, mask),
            );
        }
    }
    prefix
}

// AVX-512: PROVISIONAL, UNTESTED ON ACTUAL AVX-512 HARDWARE.
// This path passes compilation/lint checks only; the development Ryzen 9 5950X
// supports AVX2, not AVX-512. Runtime ISA/OS-state gating remains mandatory.
// Native payload, tail, bounds and fault regression tests must pass on capable
// hardware before this implementation can be described as hardware-qualified.
#[target_feature(enable = "avx512f")]
pub(super) unsafe fn avx512(input: &[u8], output: &mut [u8]) -> usize {
    let prefix = input.len() / 64 * 64;
    let mask = arch::_mm512_set1_epi32(i32::MIN);
    for offset in (0..prefix).step_by(64) {
        // SAFETY: Private OS-aware detection proves AVX-512F, including ZMM/opmask state.
        // Only full64-byte spans are loaded/stored; foundation XOR needs no BW/DQ/VL.
        unsafe {
            let value = arch::_mm512_loadu_si512(input.as_ptr().add(offset).cast());
            arch::_mm512_storeu_si512(
                output.as_mut_ptr().add(offset).cast(),
                arch::_mm512_xor_si512(value, mask),
            );
        }
    }
    prefix
}

#[target_feature(enable = "avx")]
pub(super) unsafe fn avx_f64(input: &[u8], output: &mut [u8]) -> usize {
    let prefix = input.len() / 32 * 32;
    let mask = arch::_mm256_set1_pd(f64::from_bits(0x8000_0000_0000_0000));
    for offset in (0..prefix).step_by(32) {
        // SAFETY: Detection proves AVX/OS YMM state; complete unaligned disjoint spans.
        // XOR is a bit operation, including for NaN/subnormal encodings; no FP arithmetic.
        unsafe {
            let value = arch::_mm256_loadu_pd(input.as_ptr().add(offset).cast());
            arch::_mm256_storeu_pd(
                output.as_mut_ptr().add(offset).cast(),
                arch::_mm256_xor_pd(value, mask),
            );
        }
    }
    prefix
}

// AVX-512: PROVISIONAL, UNTESTED ON ACTUAL AVX-512 HARDWARE.
// The F64 sign transform has compilation/lint coverage only. Keep runtime
// ISA/OS-state gating; require native payload/tail/bounds regression coverage
// on AVX-512 hardware before claiming hardware qualification.
#[target_feature(enable = "avx512f")]
pub(super) unsafe fn avx512_f64(input: &[u8], output: &mut [u8]) -> usize {
    let prefix = input.len() / 64 * 64;
    let mask = arch::_mm512_set1_epi64(i64::MIN);
    for offset in (0..prefix).step_by(64) {
        // SAFETY: Private OS-aware detection proves AVX-512F, including ZMM/opmask state.
        // Only full64-byte spans are loaded/stored; foundation XOR needs no BW/DQ/VL.
        unsafe {
            let value = arch::_mm512_loadu_si512(input.as_ptr().add(offset).cast());
            arch::_mm512_storeu_si512(
                output.as_mut_ptr().add(offset).cast(),
                arch::_mm512_xor_si512(value, mask),
            );
        }
    }
    prefix
}
