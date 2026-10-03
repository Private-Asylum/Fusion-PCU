//! SSE2 wrapping lane arithmetic and exact carry/sign masks; caller proves instruction support.
//! Inline metadata retains guarded intrinsic bodies across crate boundaries.
#[cfg(target_arch = "x86")]
use core::arch::x86 as arch;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64 as arch;
use super::Vector;
pub(super) struct Sse2;
impl Vector for Sse2 {
    type Bits = arch::__m128i;
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
    unsafe fn mask(v: Self::Bits) -> u32 {
        u32::try_from(arch::_mm_movemask_epi8(v)).expect("SSE2 movemask unsigned16bits")
    }
}
