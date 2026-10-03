//! Corrupted but family-valid status must never reach recovered publication.
use super::*;
#[rustfmt::skip]
use fusion_pcu::{
    PcuDispatchFloatBinaryOp as Binary,
    PcuDispatchFloatUnaryOp as Unary,
    PcuDispatchIntegerBinaryOp as Integer,
    PcuFloatUnderflowPolicy as Underflow,
    PcuRangePolicy as Range,
    PcuScalarType as Scalar,
};
#[test]
fn impossible_operation_range_and_tininess_records_reject_before_arbitration() {
    let add = PcuCheckedScalarFaultLaw::float_binary(
        Scalar::F32,
        Binary::Add,
        Range::Reject,
        Underflow::IeeeAfterRounding,
    )
    .unwrap();
    for record in [2, 3, 0x101, 0x103] {
        assert!(validate(&[0, record], 2, add, Encoding::Scalar).is_err());
    }
    let div = PcuCheckedScalarFaultLaw::float_binary(
        Scalar::F64,
        Binary::Div,
        Range::Clamp,
        Underflow::AllowGradualUnderflow,
    )
    .unwrap();
    // The valid fatal divisor fault must not conceal a later illegal underflow notice.
    assert!(validate(&[2, 0x103], 2, div, Encoding::Scalar).is_err());
    assert!(validate(&[2, 0x101], 2, div, Encoding::Scalar).is_ok());
    let unary = PcuCheckedScalarFaultLaw::float_unary(
        Scalar::F16,
        Unary::Neg,
        Range::Clamp,
        Underflow::IeeeAfterRounding,
    )
    .unwrap();
    for record in [1, 2, 3, 0x101, 0x103] {
        assert!(validate(&[record], 1, unary, Encoding::Scalar).is_err());
    }
    let subtract =
        PcuCheckedScalarFaultLaw::integer_binary(Scalar::U256, Integer::Sub, Range::Clamp).unwrap();
    assert!(validate(&[0x101], 1, subtract, Encoding::Scalar).is_err());
    assert!(validate(&[0x103], 1, subtract, Encoding::Scalar).is_ok());
}
#[test]
fn division_signedness_raw_encoding_and_exact_status_extent_are_independent_guards() {
    let unsigned = PcuCheckedScalarFaultLaw::integer_div_rem(Scalar::U512).unwrap();
    let signed = PcuCheckedScalarFaultLaw::integer_div_rem(Scalar::I512).unwrap();
    assert!(validate(&[0, 1], 2, unsigned, Encoding::DivRem).is_err());
    assert!(validate(&[0, 1], 2, signed, Encoding::DivRem).is_ok());
    assert!(validate(&[4, 1], 2, unsigned, Encoding::DivRem).is_err());
    assert!(validate(&[4], 2, unsigned, Encoding::DivRem).is_err());
    for record in [2, 3, 0x101, 0x104, u32::MAX] {
        assert!(validate(&[record], 1, signed, Encoding::DivRem).is_err());
    }
    assert!(validate(&[0, 4], 2, unsigned, Encoding::DivRem).is_ok());
}
