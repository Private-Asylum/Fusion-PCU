use super::*;
#[rustfmt::skip]
use crate::{
    PcuScalar,
    PcuF128Bits,
    PcuF256Bits,
    PcuValueTypeCaps,
};

#[test]
fn format_identity_and_capabilities_do_not_alias_or_truncate() {
    let mut occupied = 0_u32;
    for scalar in PcuScalarType::ALL {
        let cap = PcuValueTypeCaps::for_scalar(scalar).bits();
        assert_eq!(cap.count_ones(), 1);
        assert_eq!(occupied & cap, 0);
        occupied |= cap;
        if let Some(format) = scalar.binary_float_format() {
            assert_eq!(format.storage_bits, scalar.bit_width());
            assert_eq!(
                u16::from(format.exponent_bits) + format.precision_bits,
                format.storage_bits
            );
        }
    }
    assert_eq!(PcuScalarType::U512.bit_width(), 512);
    assert_eq!(
        PcuScalarType::F256
            .binary_float_format()
            .unwrap()
            .precision_bits,
        237
    );
}

#[test]
fn every_fp8_pattern_is_preserved_with_named_exception_classes() {
    for bits in 0..=u8::MAX {
        let e4 = PcuF8E4M3FnBits::from_bits(bits);
        let e5 = PcuF8E5M2Bits::from_bits(bits);
        assert_eq!(PcuF8E4M3FnBits::decode_le(e4.encode_le()), e4);
        assert_eq!(PcuF8E5M2Bits::decode_le(e5.encode_le()), e5);
        assert_ne!(e4.classify(), PcuFloatClass::Infinite);
        assert_eq!(
            e4.classify() == PcuFloatClass::QuietNaN,
            bits & 0x7f == 0x7f
        );
    }
    assert_eq!(
        PcuF8E5M2Bits::from_bits(0xfc).classify(),
        PcuFloatClass::Infinite
    );
    assert_eq!(
        PcuF8E5M2Bits::from_bits(0x7d).classify(),
        PcuFloatClass::SignalingNaN
    );
    assert_eq!(
        PcuF8E4M3FnBits::from_bits(0x7e).classify(),
        PcuFloatClass::Normal
    );
}

#[test]
fn wide_float_classes_preserve_sign_and_payload_without_host_arithmetic() {
    for (words, expected) in [
        ([0, 0], PcuFloatClass::Zero),
        ([0, 1 << 63], PcuFloatClass::Zero),
        ([1, 0], PcuFloatClass::Subnormal),
        ([0, 1 << 48], PcuFloatClass::Normal),
        ([0, 0x7fff << 48], PcuFloatClass::Infinite),
        ([1, 0x7fff << 48], PcuFloatClass::SignalingNaN),
        ([0, (0x7fff << 48) | (1 << 47)], PcuFloatClass::QuietNaN),
    ] {
        assert_eq!(PcuF128Bits::from_limbs_le(words).classify(), expected);
    }
    assert_eq!(
        PcuF256Bits::from_limbs_le([0, 0, 0, 0x7ffff << 44]).classify(),
        PcuFloatClass::Infinite
    );
    assert_eq!(
        PcuF256Bits::from_limbs_le([1, 0, 0, 0x7ffff << 44]).classify(),
        PcuFloatClass::SignalingNaN
    );
}
