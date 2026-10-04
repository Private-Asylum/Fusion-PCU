//! SSE2/AVX2/AVX-512 wrapping lanes and exact carry/sign masks; callers prove required ISA.
//! Inline metadata retains guarded intrinsic bodies across crate boundaries.
#[cfg(target_arch = "x86")]
use core::arch::x86 as arch;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64 as arch;
use super::Vector;
pub(super) struct Sse2;
impl Vector for Sse2 {
    type Bits = arch::__m128i;
    const BYTES: usize = 16;
    #[inline]
    #[target_feature(enable = "sse2")]
    unsafe fn load(p: *const u8) -> Self::Bits {
        // SAFETY: Guarded caller proves ISA and a complete initialized/disjoint16-byte span.
        unsafe { arch::_mm_loadu_si128(p.cast()) }
    }
    #[inline]
    #[target_feature(enable = "sse2")]
    unsafe fn store(p: *mut u8, v: Self::Bits) {
        unsafe {
            arch::_mm_storeu_si128(p.cast(), v);
        }
    }
    #[inline]
    #[target_feature(enable = "sse2")]
    unsafe fn add<const SIZE: usize>(a: Self::Bits, b: Self::Bits) -> Self::Bits {
        match SIZE {
            1 => arch::_mm_add_epi8(a, b),
            2 => arch::_mm_add_epi16(a, b),
            4 => arch::_mm_add_epi32(a, b),
            _ => arch::_mm_add_epi64(a, b),
        }
    }
    #[inline]
    #[target_feature(enable = "sse2")]
    unsafe fn sub<const SIZE: usize>(a: Self::Bits, b: Self::Bits) -> Self::Bits {
        match SIZE {
            1 => arch::_mm_sub_epi8(a, b),
            2 => arch::_mm_sub_epi16(a, b),
            4 => arch::_mm_sub_epi32(a, b),
            _ => arch::_mm_sub_epi64(a, b),
        }
    }
    #[inline]
    #[target_feature(enable = "sse2")]
    unsafe fn and(a: Self::Bits, b: Self::Bits) -> Self::Bits {
        arch::_mm_and_si128(a, b)
    }
    #[inline]
    #[target_feature(enable = "sse2")]
    unsafe fn or(a: Self::Bits, b: Self::Bits) -> Self::Bits {
        arch::_mm_or_si128(a, b)
    }
    #[inline]
    #[target_feature(enable = "sse2")]
    unsafe fn xor(a: Self::Bits, b: Self::Bits) -> Self::Bits {
        arch::_mm_xor_si128(a, b)
    }
    #[inline]
    #[target_feature(enable = "sse2")]
    unsafe fn and_not(a: Self::Bits, b: Self::Bits) -> Self::Bits {
        arch::_mm_andnot_si128(a, b)
    }
    #[inline]
    #[target_feature(enable = "sse2")]
    unsafe fn signs<const SIZE: usize>(v: Self::Bits) -> Self::Bits {
        match SIZE {
            1 => arch::_mm_cmpgt_epi8(arch::_mm_setzero_si128(), v),
            2 => arch::_mm_srai_epi16::<15>(v),
            4 => arch::_mm_srai_epi32::<31>(v),
            _ => arch::_mm_shuffle_epi32::<0xf5>(arch::_mm_srai_epi32::<31>(v)),
        }
    }
    #[inline]
    #[target_feature(enable = "sse2")]
    unsafe fn mask(v: Self::Bits) -> u64 {
        u64::try_from(arch::_mm_movemask_epi8(v)).expect("SSE2 movemask unsigned16bits")
    }
    #[inline]
    #[target_feature(enable = "sse2")]
    unsafe fn execute<
        T: super::PcuCheckedInteger,
        const OP: u8,
        const CLAMP: bool,
        const LB: bool,
        const RB: bool,
    >(
        left: &[u8],
        right: &[u8],
        output: &mut [u8],
        extent: usize,
        broadcast_left: &[u8; 64],
        broadcast_right: &[u8; 64],
    ) -> Result<(), super::PcuCpuCheckedIntegerError> {
        // SAFETY: Private cold admission proves this family's ISA and all complete byte spans.
        unsafe {
            super::execute::<T, Self, OP, CLAMP, LB, RB>(
                left,
                right,
                output,
                extent,
                broadcast_left,
                broadcast_right,
            )
        }
    }
}

pub(super) struct Avx2;
impl Vector for Avx2 {
    type Bits = arch::__m256i;
    const BYTES: usize = 32;
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn load(p: *const u8) -> Self::Bits {
        // SAFETY: Guarded caller proves ISA and a complete initialized/disjoint32-byte span.
        unsafe { arch::_mm256_loadu_si256(p.cast()) }
    }
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn store(p: *mut u8, v: Self::Bits) {
        unsafe {
            arch::_mm256_storeu_si256(p.cast(), v);
        }
    }
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn add<const SIZE: usize>(a: Self::Bits, b: Self::Bits) -> Self::Bits {
        match SIZE {
            1 => arch::_mm256_add_epi8(a, b),
            2 => arch::_mm256_add_epi16(a, b),
            4 => arch::_mm256_add_epi32(a, b),
            _ => arch::_mm256_add_epi64(a, b),
        }
    }
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn sub<const SIZE: usize>(a: Self::Bits, b: Self::Bits) -> Self::Bits {
        match SIZE {
            1 => arch::_mm256_sub_epi8(a, b),
            2 => arch::_mm256_sub_epi16(a, b),
            4 => arch::_mm256_sub_epi32(a, b),
            _ => arch::_mm256_sub_epi64(a, b),
        }
    }
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn and(a: Self::Bits, b: Self::Bits) -> Self::Bits {
        arch::_mm256_and_si256(a, b)
    }
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn or(a: Self::Bits, b: Self::Bits) -> Self::Bits {
        arch::_mm256_or_si256(a, b)
    }
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn xor(a: Self::Bits, b: Self::Bits) -> Self::Bits {
        arch::_mm256_xor_si256(a, b)
    }
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn and_not(a: Self::Bits, b: Self::Bits) -> Self::Bits {
        arch::_mm256_andnot_si256(a, b)
    }
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn signs<const SIZE: usize>(v: Self::Bits) -> Self::Bits {
        match SIZE {
            1 => arch::_mm256_cmpgt_epi8(arch::_mm256_setzero_si256(), v),
            2 => arch::_mm256_srai_epi16::<15>(v),
            4 => arch::_mm256_srai_epi32::<31>(v),
            _ => arch::_mm256_shuffle_epi32::<0xf5>(arch::_mm256_srai_epi32::<31>(v)),
        }
    }
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn mask(v: Self::Bits) -> u64 {
        u64::from(u32::from_ne_bytes(
            arch::_mm256_movemask_epi8(v).to_ne_bytes(),
        ))
    }
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn execute<
        T: super::PcuCheckedInteger,
        const OP: u8,
        const CLAMP: bool,
        const LB: bool,
        const RB: bool,
    >(
        left: &[u8],
        right: &[u8],
        output: &mut [u8],
        extent: usize,
        broadcast_left: &[u8; 64],
        broadcast_right: &[u8; 64],
    ) -> Result<(), super::PcuCpuCheckedIntegerError> {
        // SAFETY: Private cold admission proves this family's ISA and all complete byte spans.
        unsafe {
            super::execute::<T, Self, OP, CLAMP, LB, RB>(
                left,
                right,
                output,
                extent,
                broadcast_left,
                broadcast_right,
            )
        }
    }
}

// AVX-512: PROVISIONAL, UNTESTED ON ACTUAL AVX-512 HARDWARE.
// This entire vector implementation has compilation/lint coverage only; the
// development Ryzen 9 5950X cannot execute it. Runtime AVX-512F/BW and OS-state
// checks remain mandatory. Native signed/unsigned overflow masks, Reject/Clamp,
// broadcasts, rollback and tail tests must pass on capable hardware before
// this implementation can be described as hardware-qualified.
pub(super) struct Avx512;
impl Vector for Avx512 {
    type Bits = arch::__m512i;
    const BYTES: usize = 64;
    #[inline]
    #[target_feature(enable = "avx512f,avx512bw")]
    unsafe fn load(p: *const u8) -> Self::Bits {
        // SAFETY: Guarded caller proves ISA and a complete initialized/disjoint64-byte span.
        unsafe { arch::_mm512_loadu_si512(p.cast()) }
    }
    #[inline]
    #[target_feature(enable = "avx512f,avx512bw")]
    unsafe fn store(p: *mut u8, v: Self::Bits) {
        unsafe {
            arch::_mm512_storeu_si512(p.cast(), v);
        }
    }
    #[inline]
    #[target_feature(enable = "avx512f,avx512bw")]
    unsafe fn add<const SIZE: usize>(a: Self::Bits, b: Self::Bits) -> Self::Bits {
        match SIZE {
            1 => arch::_mm512_add_epi8(a, b),
            2 => arch::_mm512_add_epi16(a, b),
            4 => arch::_mm512_add_epi32(a, b),
            _ => arch::_mm512_add_epi64(a, b),
        }
    }
    #[inline]
    #[target_feature(enable = "avx512f,avx512bw")]
    unsafe fn sub<const SIZE: usize>(a: Self::Bits, b: Self::Bits) -> Self::Bits {
        match SIZE {
            1 => arch::_mm512_sub_epi8(a, b),
            2 => arch::_mm512_sub_epi16(a, b),
            4 => arch::_mm512_sub_epi32(a, b),
            _ => arch::_mm512_sub_epi64(a, b),
        }
    }
    #[inline]
    #[target_feature(enable = "avx512f,avx512bw")]
    unsafe fn and(a: Self::Bits, b: Self::Bits) -> Self::Bits {
        arch::_mm512_and_si512(a, b)
    }
    #[inline]
    #[target_feature(enable = "avx512f,avx512bw")]
    unsafe fn or(a: Self::Bits, b: Self::Bits) -> Self::Bits {
        arch::_mm512_or_si512(a, b)
    }
    #[inline]
    #[target_feature(enable = "avx512f,avx512bw")]
    unsafe fn xor(a: Self::Bits, b: Self::Bits) -> Self::Bits {
        arch::_mm512_xor_si512(a, b)
    }
    #[inline]
    #[target_feature(enable = "avx512f,avx512bw")]
    unsafe fn and_not(a: Self::Bits, b: Self::Bits) -> Self::Bits {
        arch::_mm512_andnot_si512(a, b)
    }
    #[inline]
    #[target_feature(enable = "avx512f,avx512bw")]
    unsafe fn signs<const SIZE: usize>(v: Self::Bits) -> Self::Bits {
        match SIZE {
            1 => arch::_mm512_movm_epi8(arch::_mm512_cmpgt_epi8_mask(
                arch::_mm512_setzero_si512(),
                v,
            )),
            2 => arch::_mm512_srai_epi16::<15>(v),
            4 => arch::_mm512_srai_epi32::<31>(v),
            _ => arch::_mm512_shuffle_epi32::<0xf5>(arch::_mm512_srai_epi32::<31>(v)),
        }
    }
    #[inline]
    #[target_feature(enable = "avx512f,avx512bw")]
    unsafe fn mask(v: Self::Bits) -> u64 {
        arch::_mm512_movepi8_mask(v)
    }
    #[inline]
    #[target_feature(enable = "avx512f,avx512bw")]
    unsafe fn execute<
        T: super::PcuCheckedInteger,
        const OP: u8,
        const CLAMP: bool,
        const LB: bool,
        const RB: bool,
    >(
        left: &[u8],
        right: &[u8],
        output: &mut [u8],
        extent: usize,
        broadcast_left: &[u8; 64],
        broadcast_right: &[u8; 64],
    ) -> Result<(), super::PcuCpuCheckedIntegerError> {
        // SAFETY: Private cold admission proves this family's ISA and all complete byte spans.
        unsafe {
            super::execute::<T, Self, OP, CLAMP, LB, RB>(
                left,
                right,
                output,
                extent,
                broadcast_left,
                broadcast_right,
            )
        }
    }
}
