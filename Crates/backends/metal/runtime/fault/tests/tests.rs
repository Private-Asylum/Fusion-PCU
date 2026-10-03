//! Valid family encodings remain insufficient for useful payload publication.
use super::*;
#[rustfmt::skip]
use fusion_pcu::{
    PcuDispatchFloatBinaryOp as Binary,
    PcuDispatchFloatUnaryOp as Unary,
    PcuFloatUnderflowPolicy as Underflow,
};
#[test]
fn every_lane_checks_exact_operation_range_underflow_and_transport_law() {
    let add = PcuCheckedScalarFaultLaw::float_binary(
        PcuScalarType::F64,
        Binary::Add,
        PcuRangePolicy::Reject,
        Underflow::IeeeAfterRounding,
    )
    .unwrap();
    for record in [2, 3, 0x101, 0x103] {
        assert!(validate(&[record], 1, Some(add), Encoding::Scalar).is_err());
    }
    let div = PcuCheckedScalarFaultLaw::float_binary(
        PcuScalarType::F16,
        Binary::Div,
        PcuRangePolicy::Clamp,
        Underflow::AllowGradualUnderflow,
    )
    .unwrap();
    assert!(validate(&[2, 0x103], 2, Some(div), Encoding::Scalar).is_err());
    assert!(validate(&[2, 0x101], 2, Some(div), Encoding::Scalar).is_ok());
    let unary = PcuCheckedScalarFaultLaw::float_unary(
        PcuScalarType::F32,
        Unary::Relu,
        PcuRangePolicy::Reject,
        Underflow::RejectSubnormalResult,
    )
    .unwrap();
    assert!(validate(&[4, 1], 2, Some(unary), Encoding::Scalar).is_err());
    assert!(validate(&[4, 3], 2, Some(unary), Encoding::Scalar).is_ok());
    assert!(validate(&[0, 1], 2, None, Encoding::Scalar).is_err());
    assert!(validate(&[0, 0], 2, None, Encoding::Scalar).is_ok());
}
#[test]
fn joint_unsigned_signed_code_and_status_extent_cannot_be_forged() {
    let unsigned = PcuCheckedScalarFaultLaw::integer_div_rem(PcuScalarType::U512).unwrap();
    let signed = PcuCheckedScalarFaultLaw::integer_div_rem(PcuScalarType::I512).unwrap();
    assert!(validate(&[2, 5], 2, Some(unsigned), Encoding::DivRem).is_err());
    assert!(validate(&[2, 5], 2, Some(signed), Encoding::DivRem).is_ok());
    assert!(validate(&[0, 2], 1, Some(unsigned), Encoding::DivRem).is_err());
    for record in [1, 3, 0x105, u32::MAX] {
        assert!(validate(&[record], 1, Some(signed), Encoding::DivRem).is_err());
    }
}
