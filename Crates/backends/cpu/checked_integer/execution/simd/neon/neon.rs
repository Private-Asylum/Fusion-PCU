//! NEON integer lane bits do not depend on floating rounding/FTZ state.
//! Inline metadata retains guarded intrinsic bodies across crate boundaries.
use core::arch::aarch64 as arch;
use super::Vector;
pub(super) struct Neon;
impl Vector for Neon {
    type Bits = arch::uint8x16_t;
    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn load(p: *const u8) -> Self::Bits {
        // SAFETY: Guarded caller proves ISA and a complete initialized/disjoint16-byte span.
        unsafe { arch::vld1q_u8(p) }
    }
    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn store(p: *mut u8, v: Self::Bits) {
        unsafe {
            arch::vst1q_u8(p, v);
        }
    }
    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn add<const SIZE: usize>(a: Self::Bits, b: Self::Bits) -> Self::Bits {
        match SIZE {
            1 => arch::vaddq_u8(a, b),
            2 => arch::vreinterpretq_u8_u16(arch::vaddq_u16(
                arch::vreinterpretq_u16_u8(a),
                arch::vreinterpretq_u16_u8(b),
            )),
            4 => arch::vreinterpretq_u8_u32(arch::vaddq_u32(
                arch::vreinterpretq_u32_u8(a),
                arch::vreinterpretq_u32_u8(b),
            )),
            _ => arch::vreinterpretq_u8_u64(arch::vaddq_u64(
                arch::vreinterpretq_u64_u8(a),
                arch::vreinterpretq_u64_u8(b),
            )),
        }
    }
    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn sub<const SIZE: usize>(a: Self::Bits, b: Self::Bits) -> Self::Bits {
        match SIZE {
            1 => arch::vsubq_u8(a, b),
            2 => arch::vreinterpretq_u8_u16(arch::vsubq_u16(
                arch::vreinterpretq_u16_u8(a),
                arch::vreinterpretq_u16_u8(b),
            )),
            4 => arch::vreinterpretq_u8_u32(arch::vsubq_u32(
                arch::vreinterpretq_u32_u8(a),
                arch::vreinterpretq_u32_u8(b),
            )),
            _ => arch::vreinterpretq_u8_u64(arch::vsubq_u64(
                arch::vreinterpretq_u64_u8(a),
                arch::vreinterpretq_u64_u8(b),
            )),
        }
    }
    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn and(a: Self::Bits, b: Self::Bits) -> Self::Bits {
        arch::vandq_u8(a, b)
    }
    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn or(a: Self::Bits, b: Self::Bits) -> Self::Bits {
        arch::vorrq_u8(a, b)
    }
    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn xor(a: Self::Bits, b: Self::Bits) -> Self::Bits {
        arch::veorq_u8(a, b)
    }
    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn and_not(a: Self::Bits, b: Self::Bits) -> Self::Bits {
        arch::vbicq_u8(b, a)
    }
    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn signs<const SIZE: usize>(v: Self::Bits) -> Self::Bits {
        match SIZE {
            1 => arch::vcltq_s8(arch::vreinterpretq_s8_u8(v), arch::vdupq_n_s8(0)),
            2 => arch::vreinterpretq_u8_u16(arch::vcltq_s16(
                arch::vreinterpretq_s16_u8(v),
                arch::vdupq_n_s16(0),
            )),
            4 => arch::vreinterpretq_u8_u32(arch::vcltq_s32(
                arch::vreinterpretq_s32_u8(v),
                arch::vdupq_n_s32(0),
            )),
            _ => arch::vreinterpretq_u8_u64(arch::vcltq_s64(
                arch::vreinterpretq_s64_u8(v),
                arch::vdupq_n_s64(0),
            )),
        }
    }
    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn mask(v: Self::Bits) -> u32 {
        let mut bytes = [0u8; 16];
        unsafe {
            arch::vst1q_u8(bytes.as_mut_ptr(), v);
        }
        bytes
            .into_iter()
            .enumerate()
            .fold(0, |mask, (index, byte)| {
                mask | (u32::from(byte >> 7) << index)
            })
    }
}
