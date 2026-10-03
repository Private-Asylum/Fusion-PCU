#[rustfmt::skip]
use super::{
    u64_to_f32_nearest_even,
    u64_to_f64_nearest_even,
};

#[test]
fn full_unsigned_range_packs_finite_binary32_with_ties_and_carry() {
    for (integer, bits) in [
        (0, 0),
        (1, 0x3f80_0000),
        ((1 << 24) + 1, 0x4b80_0000),
        ((1 << 24) + 3, 0x4b80_0002),
        ((1 << 25) - 1, 0x4c00_0000),
        (1 << 63, 0x5f00_0000),
        (u64::MAX, 0x5f80_0000),
    ] {
        assert_eq!(
            u64_to_f32_nearest_even(integer).to_bits(),
            bits,
            "{integer}"
        );
    }
}

#[test]
fn full_unsigned_range_packs_finite_binary64_with_ties_and_carry() {
    for (integer, bits) in [
        (0, 0),
        (1, 0x3ff0_0000_0000_0000),
        ((1 << 53) + 1, 0x4340_0000_0000_0000),
        ((1 << 53) + 3, 0x4340_0000_0000_0002),
        ((1 << 54) - 1, 0x4350_0000_0000_0000),
        (1 << 63, 0x43e0_0000_0000_0000),
        (u64::MAX, 0x43f0_0000_0000_0000),
    ] {
        assert_eq!(
            u64_to_f64_nearest_even(integer).to_bits(),
            bits,
            "{integer}"
        );
    }
}
