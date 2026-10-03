use super::{count_f32, count_f64};

#[test]
fn counts_have_exact_zero_one_and_nearest_even_binary32_ties() {
    for (count, bits) in [
        (0, 0),
        (1, 0x3f80_0000),
        (3, 0x4040_0000),
        (16_777_215, 0x4b7f_ffff),
        (16_777_217, 0x4b80_0000),
        (16_777_219, 0x4b80_0002),
        (33_554_431, 0x4c00_0000),
    ] {
        assert_eq!(count_f32(count).to_bits(), bits, "count {count}");
    }
    assert_eq!(count_f64(0).to_bits(), 0);
    assert_eq!(count_f64(1).to_bits(), 0x3ff0_0000_0000_0000);
    assert_eq!(count_f64(3).to_bits(), 0x4008_0000_0000_0000);
}

#[cfg(target_pointer_width = "64")]
#[test]
fn binary64_count_ties_and_carry_at_maximum_usize_are_defined() {
    assert_eq!(count_f64((1 << 53) + 1).to_bits(), 0x4340_0000_0000_0000);
    assert_eq!(count_f64((1 << 53) + 3).to_bits(), 0x4340_0000_0000_0002);
    assert_eq!(count_f32(usize::MAX).to_bits(), 0x5f80_0000);
    assert_eq!(count_f64(usize::MAX).to_bits(), 0x43f0_0000_0000_0000);
}
