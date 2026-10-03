#[rustfmt::skip]
use crate::{
    PcuCheckedFloat,
    PcuCheckedInteger,
    PcuCheckedScalarFaultLaw as Law,
    PcuClampedError,
    PcuClampedFloat,
    PcuClampedInteger,
    PcuExecutionFault,
    PcuExecutionFaultKind as Kind,
    PcuFloatUnderflowPolicy as Underflow,
    PcuRangePolicy as Range,
    PcuScalarType as Scalar,
    PcuF8E4M3FnBits,
    PcuF8E5M2Bits,
    model::{
        PcuDispatchCheckedFloatConversion as Conversion,
        PcuDispatchFloatBinaryOp as Float,
        PcuDispatchFloatUnaryOp as Unary,
        PcuDispatchIntegerBinaryOp as Integer,
    },
};

const SIGNED: [Scalar; 7] = [
    Scalar::I8,
    Scalar::I16,
    Scalar::I32,
    Scalar::I64,
    Scalar::I128,
    Scalar::I256,
    Scalar::I512,
];
const UNSIGNED: [Scalar; 7] = [
    Scalar::U8,
    Scalar::U16,
    Scalar::U32,
    Scalar::U64,
    Scalar::U128,
    Scalar::U256,
    Scalar::U512,
];
const FLOATS: [Scalar; 6] = [
    Scalar::F16,
    Scalar::BF16,
    Scalar::F32,
    Scalar::F64,
    Scalar::F8E4M3FN,
    Scalar::F8E5M2,
];
const POLICIES: [Underflow; 3] = [
    Underflow::IeeeAfterRounding,
    Underflow::RejectSubnormalResult,
    Underflow::AllowGradualUnderflow,
];

#[test]
fn division_rejects_wrong_signedness_range_and_recovery_codes_at_every_width() {
    for scalar in SIGNED.into_iter().chain(UNSIGNED) {
        let law = Law::integer_div_rem(scalar).unwrap();
        assert!(law.allows(Kind::DivideByZero, false));
        assert_eq!(
            law.allows(Kind::SignedDivisionOverflow, false),
            SIGNED.contains(&scalar)
        );
        for kind in [
            Kind::ArithmeticOverflow,
            Kind::ArithmeticUnderflow,
            Kind::InvalidFloatingOperand,
        ] {
            assert!(!law.allows(kind, false));
        }
        for kind in [
            Kind::DivideByZero,
            Kind::SignedDivisionOverflow,
            Kind::ArithmeticOverflow,
            Kind::ArithmeticUnderflow,
            Kind::InvalidFloatingOperand,
        ] {
            assert!(!law.allows(kind, true));
        }
    }
}

#[test]
fn unsigned_add_and_sub_do_not_accept_the_other_range_endpoint() {
    for scalar in UNSIGNED {
        for range in [Range::Reject, Range::Clamp] {
            let recovered = range == Range::Clamp;
            for op in [Integer::Add, Integer::Mul] {
                let law = Law::integer_binary(scalar, op, range).unwrap();
                assert!(law.allows(Kind::ArithmeticOverflow, recovered));
                assert!(!law.allows(Kind::ArithmeticOverflow, !recovered));
                assert!(!law.allows(Kind::ArithmeticUnderflow, false));
                assert!(!law.allows(Kind::ArithmeticUnderflow, true));
                assert!(!law.allows(Kind::DivideByZero, false));
            }
            let law = Law::integer_binary(scalar, Integer::Sub, range).unwrap();
            assert!(law.allows(Kind::ArithmeticUnderflow, recovered));
            assert!(!law.allows(Kind::ArithmeticUnderflow, !recovered));
            assert!(!law.allows(Kind::ArithmeticOverflow, false));
            assert!(!law.allows(Kind::ArithmeticOverflow, true));
        }
    }
}

#[test]
fn checked_and_clamped_integer_reference_errors_obey_the_same_law() {
    let underflow = i8::MIN.pcu_checked_add(-1).unwrap_err();
    let overflow = i8::MAX.pcu_checked_mul(2).unwrap_err();
    for kind in [underflow, overflow] {
        assert!(
            Law::integer_binary(Scalar::I8, Integer::Add, Range::Reject)
                .unwrap()
                .allows(kind, false)
        );
        assert!(
            !Law::integer_binary(Scalar::I8, Integer::Add, Range::Reject)
                .unwrap()
                .allows(kind, true)
        );
    }
    let fault = u8::MAX.pcu_clamped_add(1).unwrap_err();
    assert_eq!(fault.clamped_value(), u8::MAX);
    assert!(
        Law::integer_binary(Scalar::U8, Integer::Add, Range::Clamp)
            .unwrap()
            .allows(Kind::ArithmeticOverflow, true)
    );
}

#[test]
fn floating_law_rejects_zero_divisor_on_add_and_recovered_operand_faults() {
    for scalar in FLOATS {
        for range in [Range::Reject, Range::Clamp] {
            for policy in POLICIES {
                for op in [Float::Add, Float::Sub, Float::Mul] {
                    let law = Law::float_binary(scalar, op, range, policy).unwrap();
                    assert!(!law.allows(Kind::DivideByZero, false));
                    assert!(!law.allows(Kind::DivideByZero, true));
                    assert!(law.allows(Kind::InvalidFloatingOperand, false));
                    assert!(!law.allows(Kind::InvalidFloatingOperand, true));
                    assert!(!law.allows(Kind::SignedDivisionOverflow, false));
                }
                let law = Law::float_binary(scalar, Float::Div, range, policy).unwrap();
                assert!(law.allows(Kind::DivideByZero, false));
                assert!(!law.allows(Kind::DivideByZero, true));
                assert_eq!(
                    law.allows(Kind::ArithmeticUnderflow, range == Range::Clamp),
                    policy != Underflow::AllowGradualUnderflow
                );
                assert!(!law.allows(Kind::ArithmeticUnderflow, range != Range::Clamp));
            }
        }
    }
}

#[test]
fn same_format_add_sub_have_no_ieee_tiny_inexact_fault() {
    for scalar in FLOATS {
        for op in [Float::Add, Float::Sub] {
            for range in [Range::Reject, Range::Clamp] {
                let law =
                    Law::float_binary(scalar, op, range, Underflow::IeeeAfterRounding).unwrap();
                assert!(!law.allows(Kind::ArithmeticUnderflow, false));
                assert!(!law.allows(Kind::ArithmeticUnderflow, true));
                let tight =
                    Law::float_binary(scalar, op, range, Underflow::RejectSubnormalResult).unwrap();
                assert!(tight.allows(Kind::ArithmeticUnderflow, range == Range::Clamp));
            }
        }
    }
    // Exhaust every encoded FP8 pair, including nonfinite encodings. Their sealed
    // integer-significand reference must never invent IEEE Add/Sub underflow.
    for lhs in 0..=u8::MAX {
        for rhs in 0..=u8::MAX {
            let a = PcuF8E4M3FnBits::from_bits(lhs);
            let b = PcuF8E4M3FnBits::from_bits(rhs);
            assert_ne!(a.pcu_checked_add(b), Err(Kind::ArithmeticUnderflow));
            assert_ne!(a.pcu_checked_sub(b), Err(Kind::ArithmeticUnderflow));
            let a = PcuF8E5M2Bits::from_bits(lhs);
            let b = PcuF8E5M2Bits::from_bits(rhs);
            assert_ne!(a.pcu_checked_add(b), Err(Kind::ArithmeticUnderflow));
            assert_ne!(a.pcu_checked_sub(b), Err(Kind::ArithmeticUnderflow));
        }
    }
    for tiny in [
        f32::from_bits(1),
        f32::from_bits(2),
        f32::from_bits(0x7f_ffff),
    ] {
        assert!(tiny.pcu_checked_add(f32::from_bits(1)).is_ok());
        assert!(tiny.pcu_checked_sub(f32::from_bits(1)).is_ok());
    }
}

#[test]
fn exact_float_selection_only_allows_tightened_subnormal_faults() {
    for scalar in FLOATS {
        for range in [Range::Reject, Range::Clamp] {
            for policy in POLICIES {
                for op in [Unary::Neg, Unary::Relu] {
                    let law = Law::float_unary(scalar, op, range, policy).unwrap();
                    assert!(!law.allows(Kind::ArithmeticOverflow, false));
                    assert!(!law.allows(Kind::ArithmeticOverflow, true));
                    assert!(!law.allows(Kind::DivideByZero, false));
                    assert_eq!(
                        law.allows(Kind::ArithmeticUnderflow, range == Range::Clamp),
                        policy == Underflow::RejectSubnormalResult
                    );
                    assert_eq!(
                        law,
                        Law::float_relu_backward(scalar, range, policy).unwrap()
                    );
                }
            }
        }
    }
}

#[test]
fn ieee_exact_tiny_reference_and_clamp_notice_match_the_selection_law() {
    let tiny = f32::from_bits(1);
    assert_eq!(
        tiny.pcu_checked_neg().unwrap().to_bits(),
        tiny.to_bits() | (1 << 31)
    );
    let kind = tiny
        .pcu_checked_neg_with_policy(Underflow::RejectSubnormalResult)
        .unwrap_err();
    assert!(
        Law::float_unary(
            Scalar::F32,
            Unary::Neg,
            Range::Reject,
            Underflow::RejectSubnormalResult
        )
        .unwrap()
        .allows(kind, false)
    );
    let Err(PcuClampedError::Range(fault)) =
        tiny.pcu_clamped_neg_with_policy(Underflow::RejectSubnormalResult)
    else {
        panic!("expected observable range recovery")
    };
    assert!(
        Law::float_unary(
            Scalar::F32,
            Unary::Neg,
            Range::Clamp,
            Underflow::RejectSubnormalResult
        )
        .unwrap()
        .allows(fault.kind(), true)
    );
}

#[test]
fn finite_widening_has_no_range_status_under_any_policy() {
    for range in [Range::Reject, Range::Clamp] {
        for policy in POLICIES {
            let law = Law::float_conversion(Conversion::F32ToF64, range, policy);
            assert!(law.allows(Kind::InvalidFloatingOperand, false));
            for kind in [
                Kind::ArithmeticOverflow,
                Kind::ArithmeticUnderflow,
                Kind::DivideByZero,
            ] {
                assert!(!law.allows(kind, false));
                assert!(!law.allows(kind, true));
            }
            let narrow = Law::float_conversion(Conversion::F64ToF32, range, policy);
            assert!(narrow.allows(Kind::ArithmeticOverflow, range == Range::Clamp));
        }
    }
}

#[test]
fn fault_class_and_logical_domain_are_both_required() {
    let law = Law::integer_div_rem(Scalar::U512).unwrap();
    let fault = PcuExecutionFault {
        kind: Kind::DivideByZero,
        invocation_id: 8,
        recovered: false,
    };
    assert!(!law.accepts(fault, 4));
    assert!(law.accepts(fault, 19));
    assert!(!law.accepts(
        PcuExecutionFault {
            recovered: true,
            ..fault
        },
        19
    ));
    assert!(!law.accepts(
        PcuExecutionFault {
            kind: Kind::SignedDivisionOverflow,
            ..fault
        },
        19
    ));
}

#[test]
fn reserved_and_unrelated_scalar_formats_do_not_get_an_arithmetic_law() {
    for scalar in Scalar::ALL {
        let integer = SIGNED.contains(&scalar) || UNSIGNED.contains(&scalar);
        assert_eq!(Law::integer_div_rem(scalar).is_some(), integer);
        assert_eq!(
            Law::integer_binary(scalar, Integer::Add, Range::Reject).is_some(),
            integer
        );
        assert_eq!(
            Law::float_binary(
                scalar,
                Float::Add,
                Range::Reject,
                Underflow::IeeeAfterRounding
            )
            .is_some(),
            FLOATS.contains(&scalar)
        );
    }
}

#[test]
fn composed_float_laws_retain_each_operation_without_inventing_faults() {
    for scalar in FLOATS {
        for policy in POLICIES {
            let add = Law::float_binary(scalar, Float::Add, Range::Reject, policy).unwrap();
            let multiply = Law::float_binary(scalar, Float::Mul, Range::Reject, policy).unwrap();
            let combined = add.union(multiply);
            assert_eq!(combined, multiply.union(add));
            assert_eq!(combined.union(add), combined);
            assert!(!combined.allows(Kind::DivideByZero, false));
            assert!(!combined.allows(Kind::SignedDivisionOverflow, false));
            assert!(combined.allows(Kind::InvalidFloatingOperand, false));
            assert!(combined.allows(Kind::ArithmeticOverflow, false));
            assert_eq!(
                combined.allows(Kind::ArithmeticUnderflow, false),
                policy != Underflow::AllowGradualUnderflow
            );
            assert!(!combined.allows(Kind::ArithmeticUnderflow, true));
            let divide = Law::float_binary(scalar, Float::Div, Range::Reject, policy).unwrap();
            assert!(combined.union(divide).allows(Kind::DivideByZero, false));
        }
    }
}

#[test]
fn union_preserves_distinct_fatal_and_recovered_dispositions() {
    const CHECKED: Law = match Law::integer_binary(Scalar::U32, Integer::Sub, Range::Reject) {
        Some(law) => law,
        None => panic!("supported scalar"),
    };
    const CLAMPED: Law = match Law::integer_binary(Scalar::U32, Integer::Add, Range::Clamp) {
        Some(law) => law,
        None => panic!("supported scalar"),
    };
    const COMBINED: Law = CHECKED.union(CLAMPED);
    assert!(COMBINED.allows(Kind::ArithmeticUnderflow, false));
    assert!(!COMBINED.allows(Kind::ArithmeticUnderflow, true));
    assert!(COMBINED.allows(Kind::ArithmeticOverflow, true));
    assert!(!COMBINED.allows(Kind::ArithmeticOverflow, false));
    assert!(!COMBINED.allows(Kind::DivideByZero, false));
    assert!(!COMBINED.accepts(
        PcuExecutionFault {
            invocation_id: 4,
            kind: Kind::ArithmeticUnderflow,
            recovered: false
        },
        4
    ));
}
