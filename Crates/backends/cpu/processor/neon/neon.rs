//! Runtime-guarded `AArch64` integer sign-bit inversion.

#[target_feature(enable = "neon")]
pub(super) unsafe fn negate(input: &[u8], output: &mut [u8]) -> usize {
    use core::arch::aarch64 as arch;
    let prefix = input.len() / 16 * 16;
    let mask = arch::vdupq_n_u32(0x8000_0000);
    for offset in (0..prefix).step_by(16) {
        // SAFETY: The guarded caller supplies disjoint equal-length slices. AArch64 vector
        // loads/stores accept unaligned addresses; prefix bounds each complete 16-byte vector.
        unsafe {
            let value = arch::vld1q_u32(input.as_ptr().add(offset).cast());
            arch::vst1q_u32(
                output.as_mut_ptr().add(offset).cast(),
                arch::veorq_u32(value, mask),
            );
        }
    }
    prefix
}

#[target_feature(enable = "neon")]
pub(super) unsafe fn negate_f64(input: &[u8], output: &mut [u8]) -> usize {
    use core::arch::aarch64 as arch;
    let prefix = input.len() / 16 * 16;
    let mask = arch::vdupq_n_u64(0x8000_0000_0000_0000);
    for offset in (0..prefix).step_by(16) {
        // SAFETY: The detected caller supplies equal, disjoint slices and bounds complete
        // vectors. AArch64 integer loads/stores permit the provided unaligned addresses.
        unsafe {
            let value = arch::vld1q_u64(input.as_ptr().add(offset).cast());
            arch::vst1q_u64(
                output.as_mut_ptr().add(offset).cast(),
                arch::veorq_u64(value, mask),
            );
        }
    }
    prefix
}
