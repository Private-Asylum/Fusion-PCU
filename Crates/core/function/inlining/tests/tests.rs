use super::*;
use crate::model::PcuIntegerDivFlags;
#[rustfmt::skip]
use crate::{
    PcuDispatchCheckedFloatConversion as CheckedConversion,
    PcuDispatchConversion as Conversion,
    PcuDispatchFloatBinaryOp as FloatBinary,
    PcuDispatchIntegerBinaryOp as IntegerBinary,
    PcuFloatUnderflowPolicy as Underflow,
    PcuRangePolicy as Range,
    PcuScalarType as Scalar,
};

const INTEGERS: [Scalar; 14] = [
    Scalar::I8,
    Scalar::U8,
    Scalar::I16,
    Scalar::U16,
    Scalar::I32,
    Scalar::U32,
    Scalar::I64,
    Scalar::U64,
    Scalar::I128,
    Scalar::U128,
    Scalar::I256,
    Scalar::U256,
    Scalar::I512,
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

fn run<const P: usize, const R: usize, const O: usize>(
    parameter_types: [PcuValueType; P],
    result_types: [PcuValueType; R],
    local_results: [PcuDispatchValueId; R],
    operations: [PcuDispatchDataOp; O],
    first_fresh: u16,
) -> Result<([PcuDispatchDataOp; O], [PcuDispatchValueId; R]), PcuFunctionInlineError> {
    let parameters = core::array::from_fn::<_, P, _>(|index| {
        PcuDispatchValueId(u16::try_from(index + 1).unwrap())
    });
    let actual_arguments = core::array::from_fn::<_, P, _>(|index| {
        PcuDispatchValueId(u16::try_from(index + 200).unwrap())
    });
    let functions = [
        PcuFunctionSignature {
            id: PcuFunctionId(7),
            name: "scalar_helper",
            parameters: &parameter_types,
            results: &result_types,
            effects: super::super::PcuFunctionEffects::PURE,
        },
        PcuFunctionSignature {
            id: PcuFunctionId(8),
            name: "caller",
            parameters: &[],
            results: &[],
            effects: super::super::PcuFunctionEffects::PURE,
        },
    ];
    let body = PcuDispatchFunctionBody {
        function: PcuFunctionId(7),
        parameters: &parameters,
        operations: &operations,
        results: &local_results,
    };
    let mut result_ids = [PcuDispatchValueId(0); R];
    let mut operands = PcuDispatchFunctionOperands {
        signature: PcuFunctionCall {
            caller: PcuFunctionId(8),
            callee: PcuFunctionId(7),
            arguments: &parameter_types,
            results: &result_types,
        },
        arguments: &actual_arguments,
        results: &mut result_ids,
    };
    let mut value_map = [PcuDispatchValueId(0); 64];
    let mut type_map = [None; 64];
    let mut output = operations;
    assert_eq!(
        inline_pcu_dispatch_function(
            &functions,
            &body,
            &mut operands,
            first_fresh,
            &mut value_map,
            &mut type_map,
            &mut output,
        )?,
        O,
    );
    Ok((output, result_ids))
}

#[test]
fn checked_binary_remapping_retains_each_numerical_policy() {
    for scalar in FLOATS {
        let ty = PcuValueType::Scalar(scalar);
        for op in [
            FloatBinary::Add,
            FloatBinary::Sub,
            FloatBinary::Mul,
            FloatBinary::Div,
        ] {
            for underflow_policy in [
                Underflow::IeeeAfterRounding,
                Underflow::RejectSubnormalResult,
                Underflow::AllowGradualUnderflow,
            ] {
                for range_policy in [Range::Reject, Range::Clamp] {
                    let instruction = PcuDispatchDataOp::CheckedFloatBinary {
                        value_type: ty,
                        op,
                        underflow_policy,
                        range_policy,
                        result: PcuDispatchValueId(31),
                        lhs: PcuDispatchValueId(1),
                        rhs: PcuDispatchValueId(2),
                    };
                    let (output, results) =
                        run([ty; 2], [ty], [PcuDispatchValueId(31)], [instruction], 1000).unwrap();
                    assert_eq!(
                        output,
                        [PcuDispatchDataOp::CheckedFloatBinary {
                            value_type: ty,
                            op,
                            underflow_policy,
                            range_policy,
                            result: PcuDispatchValueId(1000),
                            lhs: PcuDispatchValueId(200),
                            rhs: PcuDispatchValueId(201),
                        }]
                    );
                    assert_eq!(results, [PcuDispatchValueId(1000)]);
                }
            }
        }
    }
    for scalar in INTEGERS {
        let ty = PcuValueType::Scalar(scalar);
        for op in [IntegerBinary::Add, IntegerBinary::Sub, IntegerBinary::Mul] {
            for range_policy in [Range::Reject, Range::Clamp] {
                let instruction = PcuDispatchDataOp::CheckedIntegerBinary {
                    value_type: ty,
                    op,
                    range_policy,
                    result: PcuDispatchValueId(31),
                    lhs: PcuDispatchValueId(1),
                    rhs: PcuDispatchValueId(2),
                };
                let (output, results) =
                    run([ty; 2], [ty], [PcuDispatchValueId(31)], [instruction], 1000).unwrap();
                assert_eq!(
                    output,
                    [PcuDispatchDataOp::CheckedIntegerBinary {
                        value_type: ty,
                        op,
                        range_policy,
                        result: PcuDispatchValueId(1000),
                        lhs: PcuDispatchValueId(200),
                        rhs: PcuDispatchValueId(201),
                    }]
                );
                assert_eq!(results, [PcuDispatchValueId(1000)]);
            }
        }
    }
}

#[test]
fn joint_division_keeps_both_results_and_their_caller_order() {
    for scalar in INTEGERS {
        let ty = PcuValueType::Scalar(scalar);
        let division = PcuDispatchDataOp::CheckedDivRem {
            value_type: ty,
            flags: PcuIntegerDivFlags::CHECKED,
            quotient: PcuDispatchValueId(3),
            remainder: PcuDispatchValueId(63),
            lhs: PcuDispatchValueId(1),
            rhs: PcuDispatchValueId(2),
        };
        let combine = PcuDispatchDataOp::CheckedIntegerBinary {
            value_type: ty,
            op: IntegerBinary::Add,
            range_policy: Range::Reject,
            result: PcuDispatchValueId(4),
            lhs: PcuDispatchValueId(63),
            rhs: PcuDispatchValueId(3),
        };
        let (output, results) = run(
            [ty; 2],
            [ty; 3],
            [
                PcuDispatchValueId(63),
                PcuDispatchValueId(4),
                PcuDispatchValueId(3),
            ],
            [division, combine],
            1000,
        )
        .unwrap();
        assert_eq!(
            output,
            [
                PcuDispatchDataOp::CheckedDivRem {
                    value_type: ty,
                    flags: PcuIntegerDivFlags::CHECKED,
                    quotient: PcuDispatchValueId(1000),
                    remainder: PcuDispatchValueId(1001),
                    lhs: PcuDispatchValueId(200),
                    rhs: PcuDispatchValueId(201),
                },
                PcuDispatchDataOp::CheckedIntegerBinary {
                    value_type: ty,
                    op: IntegerBinary::Add,
                    range_policy: Range::Reject,
                    result: PcuDispatchValueId(1002),
                    lhs: PcuDispatchValueId(1001),
                    rhs: PcuDispatchValueId(1000),
                },
            ]
        );
        assert_eq!(
            results,
            [
                PcuDispatchValueId(1001),
                PcuDispatchValueId(1002),
                PcuDispatchValueId(1000)
            ]
        );
    }
}

#[test]
fn joint_division_rejects_reserved_flags_duplicate_results_and_short_maps() {
    let ty = PcuValueType::i32();
    let division = PcuDispatchDataOp::CheckedDivRem {
        value_type: ty,
        flags: PcuIntegerDivFlags::CHECKED,
        quotient: PcuDispatchValueId(3),
        remainder: PcuDispatchValueId(4),
        lhs: PcuDispatchValueId(1),
        rhs: PcuDispatchValueId(2),
    };
    for (flags, quotient, remainder, expected) in [
        (
            PcuIntegerDivFlags::DIV_OR_ZERO,
            3,
            4,
            PcuFunctionInlineError::UnsupportedOperation,
        ),
        (
            PcuIntegerDivFlags::CHECKED,
            3,
            3,
            PcuFunctionInlineError::DuplicateDefinition(PcuDispatchValueId(3)),
        ),
        (
            PcuIntegerDivFlags::CHECKED,
            3,
            64,
            PcuFunctionInlineError::ValueMapTooSmall,
        ),
        (
            PcuIntegerDivFlags::CHECKED,
            64,
            3,
            PcuFunctionInlineError::ValueMapTooSmall,
        ),
        (
            PcuIntegerDivFlags::CHECKED,
            3,
            0,
            PcuFunctionInlineError::DuplicateDefinition(PcuDispatchValueId(0)),
        ),
    ] {
        let mut modified = division;
        if let PcuDispatchDataOp::CheckedDivRem {
            flags: f,
            quotient: q,
            remainder: r,
            ..
        } = &mut modified
        {
            *f = flags;
            *q = PcuDispatchValueId(quotient);
            *r = PcuDispatchValueId(remainder);
        }
        assert_eq!(
            run([ty; 2], [ty], [PcuDispatchValueId(3)], [modified], 1000),
            Err(expected)
        );
    }
    assert_eq!(
        run(
            [ty; 2],
            [ty],
            [PcuDispatchValueId(3)],
            [division],
            u16::MAX - 1
        ),
        Err(PcuFunctionInlineError::ValueIdOverflow)
    );
}

#[test]
fn conversions_keep_exact_source_and_destination_types() {
    for conversion in [
        Conversion::I8ToI16,
        Conversion::U8ToU16,
        Conversion::I16ToI32,
        Conversion::U16ToU32,
        Conversion::I32ToI64,
        Conversion::U32ToU64,
        Conversion::F32ToF16Bits,
        Conversion::F16BitsToF32,
        Conversion::F32ToBf16Bits,
        Conversion::Bf16BitsToF32,
        Conversion::F32ToF64Exact,
    ] {
        let operation = PcuDispatchDataOp::Convert {
            conversion,
            result: PcuDispatchValueId(3),
            value: PcuDispatchValueId(1),
        };
        let (output, results) = run(
            [conversion.source_type()],
            [conversion.target_type()],
            [PcuDispatchValueId(3)],
            [operation],
            1000,
        )
        .unwrap();
        assert_eq!(
            output,
            [PcuDispatchDataOp::Convert {
                conversion,
                result: PcuDispatchValueId(1000),
                value: PcuDispatchValueId(200)
            }]
        );
        assert_eq!(results, [PcuDispatchValueId(1000)]);
        assert_eq!(
            run(
                [PcuValueType::Scalar(Scalar::Bool)],
                [conversion.target_type()],
                [PcuDispatchValueId(3)],
                [operation],
                1000
            ),
            Err(PcuFunctionInlineError::TypeMismatch(PcuDispatchValueId(1)))
        );
    }
    for conversion in [CheckedConversion::F64ToF32, CheckedConversion::F32ToF64] {
        for underflow_policy in [
            Underflow::IeeeAfterRounding,
            Underflow::RejectSubnormalResult,
            Underflow::AllowGradualUnderflow,
        ] {
            for range_policy in [Range::Reject, Range::Clamp] {
                let operation = PcuDispatchDataOp::CheckedFloatConvert {
                    conversion,
                    underflow_policy,
                    range_policy,
                    result: PcuDispatchValueId(3),
                    value: PcuDispatchValueId(1),
                };
                let (output, results) = run(
                    [conversion.source_type()],
                    [conversion.target_type()],
                    [PcuDispatchValueId(3)],
                    [operation],
                    1000,
                )
                .unwrap();
                assert_eq!(
                    output,
                    [PcuDispatchDataOp::CheckedFloatConvert {
                        conversion,
                        underflow_policy,
                        range_policy,
                        result: PcuDispatchValueId(1000),
                        value: PcuDispatchValueId(200)
                    }]
                );
                assert_eq!(results, [PcuDispatchValueId(1000)]);
                assert_eq!(
                    run(
                        [conversion.source_type()],
                        [PcuValueType::Scalar(Scalar::Bool)],
                        [PcuDispatchValueId(3)],
                        [operation],
                        1000
                    ),
                    Err(PcuFunctionInlineError::TypeMismatch(PcuDispatchValueId(3)))
                );
            }
        }
    }
}
