use super::super::lower_dispatch_to_cuda_source;
use fusion_pcu::{
    PcuBinding, PcuBindingAccess, PcuBindingRef, PcuBindingStorageClass, PcuDispatchControlOp,
    PcuDispatchDataOp, PcuDispatchEntryPoint, PcuDispatchFeatureCaps, PcuDispatchIndex,
    PcuDispatchKernelIr, PcuDispatchOp, PcuDispatchValueId, PcuKernelId, PcuValueType,
    PcuValueTypeCaps,
};
use fusion_pcu::model::PcuDispatchIntegerBinaryOp;

fn kernel<'a>(
    ops: &'a [PcuDispatchOp<'a>],
    bindings: &'a [PcuBinding<'a>],
) -> PcuDispatchKernelIr<'a> {
    PcuDispatchKernelIr {
        id: PcuKernelId(1),
        entry: PcuDispatchEntryPoint {
            name: "checked_integer_test",
            logical_shape: [64, 1, 1],
        },
        bindings,
        ports: &[],
        parameters: &[],
        ops,
        type_caps: PcuValueTypeCaps::empty(),
        feature_caps: PcuDispatchFeatureCaps::empty(),
    }
}

#[test]
fn lowers_checked_integer_binary_with_width_safe_guards_for_all_integer_widths() {
    use fusion_pcu::PcuScalarType;

    for scalar in [
        PcuScalarType::U8,
        PcuScalarType::I8,
        PcuScalarType::U16,
        PcuScalarType::I16,
        PcuScalarType::U32,
        PcuScalarType::I32,
        PcuScalarType::U64,
        PcuScalarType::I64,
    ] {
        for op in [
            PcuDispatchIntegerBinaryOp::Add,
            PcuDispatchIntegerBinaryOp::Sub,
            PcuDispatchIntegerBinaryOp::Mul,
        ] {
            let source = lower_checked_map(scalar, op);
            assert_fault_tags(&source, scalar, op);
            assert_width_guard(&source, scalar, op);
        }
    }
}

fn lower_checked_map(scalar: fusion_pcu::PcuScalarType, op: PcuDispatchIntegerBinaryOp) -> String {
    let ty = PcuValueType::Scalar(scalar);
    let bindings = [
        PcuBinding::value(
            Some("lhs"),
            0,
            0,
            PcuBindingStorageClass::Storage,
            PcuBindingAccess::ReadOnly,
            ty,
        ),
        PcuBinding::value(
            Some("rhs"),
            0,
            1,
            PcuBindingStorageClass::Storage,
            PcuBindingAccess::ReadOnly,
            ty,
        ),
        PcuBinding::value(
            Some("out"),
            0,
            2,
            PcuBindingStorageClass::Storage,
            PcuBindingAccess::WriteOnly,
            ty,
        ),
    ];
    let ops = [
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
            result: PcuDispatchValueId(1),
            binding: PcuBindingRef::new(0, 0),
            index: PcuDispatchIndex::InvocationId,
        }),
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
            result: PcuDispatchValueId(2),
            binding: PcuBindingRef::new(0, 1),
            index: PcuDispatchIndex::InvocationId,
        }),
        PcuDispatchOp::Data(PcuDispatchDataOp::CheckedIntegerBinary {
            value_type: ty,
            op,
            result: PcuDispatchValueId(3),
            lhs: PcuDispatchValueId(1),
            rhs: PcuDispatchValueId(2),
        }),
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
            binding: PcuBindingRef::new(0, 2),
            index: PcuDispatchIndex::InvocationId,
            value: PcuDispatchValueId(3),
        }),
        PcuDispatchOp::Control(PcuDispatchControlOp::Return),
    ];
    let source = lower_dispatch_to_cuda_source(&kernel(&ops, &bindings))
        .unwrap_or_else(|error| panic!("{scalar:?} checked {op:?} rejected: {error:?}"));
    assert!(source.contains("unsigned long long* fusion_fault_word"));
    assert!(source.contains("return;"));
    assert!(source.contains("binding_0_2[fusion_gid] = v3;"));
    source
}

fn assert_fault_tags(
    source: &str,
    scalar: fusion_pcu::PcuScalarType,
    op: PcuDispatchIntegerBinaryOp,
) {
    use fusion_pcu::PcuScalarType;
    let signed = matches!(
        scalar,
        PcuScalarType::I8 | PcuScalarType::I16 | PcuScalarType::I32 | PcuScalarType::I64
    );
    if signed
        || matches!(
            op,
            PcuDispatchIntegerBinaryOp::Add | PcuDispatchIntegerBinaryOp::Mul
        )
    {
        assert!(source.contains("<< 3u) | 3ull"));
    }
    if signed || op == PcuDispatchIntegerBinaryOp::Sub {
        assert!(source.contains("<< 3u) | 4ull"));
    }
}

fn assert_width_guard(
    source: &str,
    scalar: fusion_pcu::PcuScalarType,
    op: PcuDispatchIntegerBinaryOp,
) {
    use fusion_pcu::PcuScalarType;
    let narrow_unsigned = matches!(
        scalar,
        PcuScalarType::U8 | PcuScalarType::U16 | PcuScalarType::U32
    );
    let narrow_signed = matches!(
        scalar,
        PcuScalarType::I8 | PcuScalarType::I16 | PcuScalarType::I32
    );
    if (narrow_unsigned && op != PcuDispatchIntegerBinaryOp::Sub) || narrow_signed {
        let wide = if narrow_signed {
            "const long long fusion_checked_wide_3"
        } else {
            "const unsigned long long fusion_checked_wide_3"
        };
        assert!(source.contains(wide));
        return;
    }
    match scalar {
        PcuScalarType::U64 if op == PcuDispatchIntegerBinaryOp::Mul => {
            assert!(source.contains("18446744073709551615ull / fusion_checked_rhs_3"));
        }
        PcuScalarType::I64 if op == PcuDispatchIntegerBinaryOp::Mul => {
            assert!(source.contains("9223372036854775807ll / fusion_checked_rhs_3"));
        }
        _ => {}
    }
}

#[test]
fn checked_integer_binary_rejects_type_and_binding_mismatches() {
    use fusion_pcu::PcuScalarType;
    let ty = PcuValueType::i32();
    let bindings = [
        PcuBinding::value(
            Some("lhs"),
            0,
            0,
            PcuBindingStorageClass::Storage,
            PcuBindingAccess::ReadOnly,
            ty,
        ),
        PcuBinding::value(
            Some("rhs"),
            0,
            1,
            PcuBindingStorageClass::Storage,
            PcuBindingAccess::ReadOnly,
            ty,
        ),
        PcuBinding::value(
            Some("out"),
            0,
            2,
            PcuBindingStorageClass::Storage,
            PcuBindingAccess::WriteOnly,
            ty,
        ),
    ];
    let valid_shape = [
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
            result: PcuDispatchValueId(1),
            binding: PcuBindingRef::new(0, 0),
            index: PcuDispatchIndex::InvocationId,
        }),
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
            result: PcuDispatchValueId(2),
            binding: PcuBindingRef::new(0, 1),
            index: PcuDispatchIndex::InvocationId,
        }),
        PcuDispatchOp::Data(PcuDispatchDataOp::CheckedIntegerBinary {
            value_type: PcuValueType::Scalar(PcuScalarType::U32),
            op: PcuDispatchIntegerBinaryOp::Add,
            result: PcuDispatchValueId(3),
            lhs: PcuDispatchValueId(1),
            rhs: PcuDispatchValueId(2),
        }),
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
            binding: PcuBindingRef::new(0, 2),
            index: PcuDispatchIndex::InvocationId,
            value: PcuDispatchValueId(3),
        }),
        PcuDispatchOp::Control(PcuDispatchControlOp::Return),
    ];
    assert!(lower_dispatch_to_cuda_source(&kernel(&valid_shape, &bindings)).is_err());

    let malformed = [
        valid_shape[0],
        valid_shape[1],
        valid_shape[2],
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
            binding: PcuBindingRef::new(0, 1),
            index: PcuDispatchIndex::InvocationId,
            value: PcuDispatchValueId(3),
        }),
        valid_shape[4],
    ];
    assert!(lower_dispatch_to_cuda_source(&kernel(&malformed, &bindings)).is_err());
}

#[test]
fn guarded_i64_and_u64_classification_matches_checked_arithmetic_at_boundaries() {
    use fusion_pcu::{PcuCheckedInteger, PcuExecutionFaultKind};

    fn i64_tag(lhs: i64, rhs: i64, op: PcuDispatchIntegerBinaryOp) -> Option<u64> {
        const MIN: i64 = i64::MIN;
        const MAX: i64 = i64::MAX;
        match op {
            PcuDispatchIntegerBinaryOp::Add if rhs > 0 && lhs > MAX - rhs => Some(3),
            PcuDispatchIntegerBinaryOp::Add if rhs < 0 && lhs < MIN - rhs => Some(4),
            PcuDispatchIntegerBinaryOp::Sub if rhs < 0 && lhs > MAX + rhs => Some(3),
            PcuDispatchIntegerBinaryOp::Sub if rhs > 0 && lhs < MIN + rhs => Some(4),
            PcuDispatchIntegerBinaryOp::Mul if lhs > 0 && rhs > 0 && lhs > MAX / rhs => Some(3),
            PcuDispatchIntegerBinaryOp::Mul if lhs > 0 && rhs < 0 && rhs < MIN / lhs => Some(4),
            PcuDispatchIntegerBinaryOp::Mul if lhs < 0 && rhs > 0 && lhs < MIN / rhs => Some(4),
            PcuDispatchIntegerBinaryOp::Mul if lhs < 0 && rhs < 0 && lhs < MAX / rhs => Some(3),
            _ => None,
        }
    }

    fn u64_tag(lhs: u64, rhs: u64, op: PcuDispatchIntegerBinaryOp) -> Option<u64> {
        match op {
            PcuDispatchIntegerBinaryOp::Add if lhs > u64::MAX - rhs => Some(3),
            PcuDispatchIntegerBinaryOp::Sub if lhs < rhs => Some(4),
            PcuDispatchIntegerBinaryOp::Mul if rhs != 0 && lhs > u64::MAX / rhs => Some(3),
            _ => None,
        }
    }

    let ops = [
        PcuDispatchIntegerBinaryOp::Add,
        PcuDispatchIntegerBinaryOp::Sub,
        PcuDispatchIntegerBinaryOp::Mul,
    ];
    let signed_values = [
        i64::MIN,
        i64::MIN + 1,
        -2,
        -1,
        0,
        1,
        2,
        i64::MAX - 1,
        i64::MAX,
    ];
    for lhs in signed_values {
        for rhs in signed_values {
            for op in ops {
                let expected = match op {
                    PcuDispatchIntegerBinaryOp::Add => lhs.pcu_checked_add(rhs),
                    PcuDispatchIntegerBinaryOp::Sub => lhs.pcu_checked_sub(rhs),
                    PcuDispatchIntegerBinaryOp::Mul => lhs.pcu_checked_mul(rhs),
                }
                .err()
                .map(|kind| match kind {
                    PcuExecutionFaultKind::ArithmeticOverflow => 3,
                    PcuExecutionFaultKind::ArithmeticUnderflow => 4,
                    _ => unreachable!(),
                });
                assert_eq!(i64_tag(lhs, rhs, op), expected, "i64 {lhs} {op:?} {rhs}");
            }
        }
    }
    let unsigned_values = [0, 1, 2, 1_u64 << 63, u64::MAX - 1, u64::MAX];
    for lhs in unsigned_values {
        for rhs in unsigned_values {
            for op in ops {
                let expected = match op {
                    PcuDispatchIntegerBinaryOp::Add => lhs.pcu_checked_add(rhs),
                    PcuDispatchIntegerBinaryOp::Sub => lhs.pcu_checked_sub(rhs),
                    PcuDispatchIntegerBinaryOp::Mul => lhs.pcu_checked_mul(rhs),
                }
                .err()
                .map(|kind| match kind {
                    PcuExecutionFaultKind::ArithmeticOverflow => 3,
                    PcuExecutionFaultKind::ArithmeticUnderflow => 4,
                    _ => unreachable!(),
                });
                assert_eq!(u64_tag(lhs, rhs, op), expected, "u64 {lhs} {op:?} {rhs}");
            }
        }
    }
}
