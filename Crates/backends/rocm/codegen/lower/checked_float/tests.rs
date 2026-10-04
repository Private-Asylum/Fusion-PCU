use super::super::lower_dispatch_to_hip_source;
#[rustfmt::skip]
use fusion_pcu::{
    PcuBinding,
    PcuBindingAccess,
    PcuBindingRef,
    PcuBindingStorageClass,
    PcuDispatchControlOp,
    PcuDispatchCheckedFloatConversion,
    PcuDispatchDataOp,
    PcuDispatchEntryPoint,
    PcuDispatchFeatureCaps,
    PcuDispatchIndex,
    PcuDispatchKernelIr,
    PcuDispatchOp,
    PcuDispatchValueId,
    PcuFloatUnderflowPolicy,
    PcuRangePolicy,
    PcuKernelId,
    PcuValueType,
    PcuValueTypeCaps,
};
use fusion_pcu::model::{PcuDispatchFloatBinaryOp, PcuDispatchFloatUnaryOp};

#[test]
fn checked_relu_lowers_direct_dispatch_to_policy_aware_bit_classification() {
    let bindings = checked_relu_bindings();
    let ops = [
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
            result: PcuDispatchValueId(1),
            binding: PcuBindingRef::new(0, 0),
            index: PcuDispatchIndex::InvocationId,
        }),
        PcuDispatchOp::Data(PcuDispatchDataOp::CheckedFloatUnary {
            value_type: PcuValueType::f32(),
            op: PcuDispatchFloatUnaryOp::Relu,
            underflow_policy: PcuFloatUnderflowPolicy::RejectSubnormalResult,
            range_policy: PcuRangePolicy::Clamp,
            result: PcuDispatchValueId(2),
            value: PcuDispatchValueId(1),
        }),
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
            binding: PcuBindingRef::new(0, 1),
            index: PcuDispatchIndex::InvocationId,
            value: PcuDispatchValueId(2),
        }),
        PcuDispatchOp::Control(PcuDispatchControlOp::Return),
    ];
    let source = lower_dispatch_to_hip_source(&PcuDispatchKernelIr {
        numerical_requirements: PcuDispatchKernelIr::DEFAULT_REQUIREMENTS,
        id: PcuKernelId(91),
        entry: PcuDispatchEntryPoint {
            name: "checked_relu_test",
            logical_shape: [64, 1, 1],
        },
        bindings: &bindings,
        ports: &[],
        parameters: &[],
        ops: &ops,
        type_caps: PcuValueTypeCaps::FLOAT32,
        feature_caps: PcuDispatchFeatureCaps::RANGE_CLAMP,
    })
    .expect("checked ReLU kernel");
    assert!(source.contains("fusion_checked_f32_relu(__builtin_bit_cast(unsigned int, v1), 1u)"));
    assert!(source.contains("fusion_fault_word"));
    assert!(source.contains("bool fusion_range_fault_recorded = false;"));
}

#[test]
fn checked_relu_lowers_grid_stride_dispatch_with_logical_fault_indices() {
    let bindings = checked_relu_bindings();
    let grid_body = [
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
            result: PcuDispatchValueId(1),
            binding: PcuBindingRef::new(0, 0),
            index: PcuDispatchIndex::GridStrideId,
        }),
        PcuDispatchOp::Data(PcuDispatchDataOp::CheckedFloatUnary {
            value_type: PcuValueType::f32(),
            op: PcuDispatchFloatUnaryOp::Relu,
            underflow_policy: PcuFloatUnderflowPolicy::RejectSubnormalResult,
            range_policy: PcuRangePolicy::Clamp,
            result: PcuDispatchValueId(2),
            value: PcuDispatchValueId(1),
        }),
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
            binding: PcuBindingRef::new(0, 1),
            index: PcuDispatchIndex::GridStrideId,
            value: PcuDispatchValueId(2),
        }),
    ];
    let grid_ops = [
        PcuDispatchOp::GridStrideLoop {
            extent: 64,
            body: &grid_body,
        },
        PcuDispatchOp::Control(PcuDispatchControlOp::Return),
    ];
    let grid_source = lower_dispatch_to_hip_source(&PcuDispatchKernelIr {
        numerical_requirements: PcuDispatchKernelIr::DEFAULT_REQUIREMENTS,
        id: PcuKernelId(92),
        entry: PcuDispatchEntryPoint {
            name: "checked_relu_grid_test",
            logical_shape: [1, 1, 1],
        },
        bindings: &bindings,
        ports: &[],
        parameters: &[],
        ops: &grid_ops,
        type_caps: PcuValueTypeCaps::FLOAT32,
        feature_caps: PcuDispatchFeatureCaps::RANGE_CLAMP,
    })
    .expect("checked grid-stride ReLU kernel");
    assert!(
        grid_source.contains("fusion_checked_f32_relu(__builtin_bit_cast(unsigned int, v1), 1u)")
    );
    assert!(grid_source.contains("bool fusion_range_fault_recorded = false;"));
    assert!(grid_source.contains("static_cast<unsigned long long>(fusion_idx)"));
}

fn checked_relu_bindings() -> [PcuBinding<'static>; 2] {
    [
        PcuBinding::value(
            Some("input"),
            0,
            0,
            PcuBindingStorageClass::Storage,
            PcuBindingAccess::ReadOnly,
            PcuValueType::f32(),
        ),
        PcuBinding::value(
            Some("output"),
            0,
            1,
            PcuBindingStorageClass::Storage,
            PcuBindingAccess::WriteOnly,
            PcuValueType::f32(),
        ),
    ]
}

fn checked_kernel(op: PcuDispatchFloatBinaryOp, policy: PcuFloatUnderflowPolicy) -> String {
    checked_kernel_for_type(PcuValueType::f32(), PcuValueTypeCaps::FLOAT32, op, policy)
}

fn checked_kernel_for_type(
    ty: PcuValueType,
    caps: PcuValueTypeCaps,
    op: PcuDispatchFloatBinaryOp,
    policy: PcuFloatUnderflowPolicy,
) -> String {
    checked_kernel_for_type_with_range_policy(ty, caps, op, policy, PcuRangePolicy::Reject)
}

fn checked_kernel_for_type_with_range_policy(
    ty: PcuValueType,
    caps: PcuValueTypeCaps,
    op: PcuDispatchFloatBinaryOp,
    policy: PcuFloatUnderflowPolicy,
    range_policy: PcuRangePolicy,
) -> String {
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
        PcuDispatchOp::Data(PcuDispatchDataOp::CheckedFloatBinary {
            value_type: ty,
            op,
            underflow_policy: policy,
            range_policy,
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
    lower_dispatch_to_hip_source(&PcuDispatchKernelIr {
        numerical_requirements: PcuDispatchKernelIr::DEFAULT_REQUIREMENTS,
        id: PcuKernelId(91),
        entry: PcuDispatchEntryPoint {
            name: "checked_f32_test",
            logical_shape: [64, 1, 1],
        },
        bindings: &bindings,
        ports: &[],
        parameters: &[],
        ops: &ops,
        type_caps: caps,
        feature_caps: if range_policy == PcuRangePolicy::Clamp {
            PcuDispatchFeatureCaps::RANGE_CLAMP
        } else {
            PcuDispatchFeatureCaps::empty()
        },
    })
    .expect("bounded checked float kernel")
}

#[test]
fn range_clamp_emits_recovered_fault_and_continuation_for_direct_and_grid_kernels() {
    let direct = checked_kernel_for_type_with_range_policy(
        PcuValueType::f32(),
        PcuValueTypeCaps::FLOAT32,
        PcuDispatchFloatBinaryOp::Mul,
        PcuFloatUnderflowPolicy::IeeeAfterRounding,
        PcuRangePolicy::Clamp,
    );
    assert!(direct.contains("bool fusion_range_fault_recorded = false;"));
    assert!(
        direct.contains(
            "0x8000000000000000ull | (static_cast<unsigned long long>(fusion_gid) << 3u)"
        )
    );
    assert!(direct.contains(
        "fusion_recovered_bits_3 = (fusion_recovered_bits_3 & 0x80000000u) | 0x7f7fffffu"
    ));
    assert!(direct.contains("float v3 = __builtin_bit_cast(float, fusion_recovered_bits_3)"));
    assert!(
        !checked_kernel(
            PcuDispatchFloatBinaryOp::Mul,
            PcuFloatUnderflowPolicy::IeeeAfterRounding
        )
        .contains("fusion_range_fault_recorded")
    );

    let grid = checked_f64_to_f32_kernel_with_range_policy(
        PcuFloatUnderflowPolicy::IeeeAfterRounding,
        PcuRangePolicy::Clamp,
        true,
    );
    assert!(grid.contains("for (unsigned int fusion_idx = fusion_gid; fusion_idx < 64u; fusion_idx += 64u) {\n        bool fusion_range_fault_recorded = false;"));
    assert!(
        grid.contains(
            "0x8000000000000000ull | (static_cast<unsigned long long>(fusion_idx) << 3u)"
        )
    );
    assert!(grid.contains(
        "fusion_recovered_bits_2 = (fusion_recovered_bits_2 & 0x80000000u) | 0x7f7fffffu"
    ));
    assert!(grid.contains(
        "fusion_f64_to_f32_checked_2.fault == 3u || fusion_f64_to_f32_checked_2.fault == 4u"
    ));
    assert!(grid.contains("FusionF64ToF32Result{result, 4u}"));
    assert!(grid.contains("float v2 = __builtin_bit_cast(float, fusion_recovered_bits_2)"));
    assert!(grid.contains("binding_0_1[fusion_idx] = v2;"));
}

#[test]
fn checked_float64_routes_through_integer_only_binary64_helper() {
    for (op, op_tag) in [
        (PcuDispatchFloatBinaryOp::Add, "0u"),
        (PcuDispatchFloatBinaryOp::Sub, "1u"),
        (PcuDispatchFloatBinaryOp::Mul, "2u"),
        (PcuDispatchFloatBinaryOp::Div, "3u"),
    ] {
        let source = checked_kernel_for_type(
            PcuValueType::f64(),
            PcuValueTypeCaps::FLOAT64,
            op,
            PcuFloatUnderflowPolicy::IeeeAfterRounding,
        );
        assert!(source.contains("fusion_checked_f64_binary(__builtin_bit_cast(unsigned long long, v1), __builtin_bit_cast(unsigned long long, v2),"));
        assert!(source.contains(&format!("), {op_tag}, 0u)")));
        assert!(
            source.contains("double v3 = __builtin_bit_cast(double, fusion_f64_checked_3.bits)")
        );
        assert!(source.contains("unsigned int product[4] = {}"));
        assert!(source.contains("return {sign_bits | 0x7ff0000000000000ull, 3u}"));
        assert!(source.contains("return {bits, 4u}"));
        assert!(source.contains("return {0ull, 5u}"));
        assert!(!source.contains("FusionF32CheckedResult"));
        assert!(!source.contains("double v3 = v1 + v2"));
        assert!(source.contains("for (int i = 0; i < 55; ++i)"));
        assert!(source.contains("if (remainder != 0ull) ext |= 1ull"));
        assert!(source.contains("if ((y << 1) == 0ull) return {0ull, 1u}"));
        assert!(!source.contains("double v3 = v1 / v2"));
    }
}

#[test]
#[allow(clippy::cognitive_complexity)] // Exhaustive typed fault/ownership matrix retains its exact witnesses.
fn checked_float_uses_bit_integer_rounding_and_fault_word_tags() {
    for (op, op_tag) in [
        (PcuDispatchFloatBinaryOp::Add, "0u"),
        (PcuDispatchFloatBinaryOp::Sub, "1u"),
        (PcuDispatchFloatBinaryOp::Mul, "2u"),
        (PcuDispatchFloatBinaryOp::Div, "3u"),
    ] {
        let source = checked_kernel(op, PcuFloatUnderflowPolicy::IeeeAfterRounding);
        assert!(source.contains("fusion_checked_f32_binary(__builtin_bit_cast(unsigned int, v1), __builtin_bit_cast(unsigned int, v2),"));
        assert!(source.contains(&format!("), {op_tag}, 0u)")));
        assert!(source.contains("| fusion_f32_checked_3.fault"));
        assert!(source.contains("return {0u, 5u}"));
        assert!(source.contains("return {bits, 4u}"));
        assert!(source.contains("return {(sign << 31) | 0x7f800000u, 3u}"));
        assert!(!source.contains("float v3 = v1 + v2"));
        assert!(!source.contains("--use_fast_math"));
        assert!(source.contains("unsigned long long product = static_cast<unsigned long long>(xm) * static_cast<unsigned long long>(ym)"));
        assert!(source.contains(
            "fusion_f32_shift_jam(static_cast<unsigned long long>(ym) << 3, xexp - yexp)"
        ));
        assert!(source.contains("while (*ext >= (1ull << 27))"));
        assert!(source.contains("while (*ext < (1ull << 26))"));
        assert!(source.contains("(sign << 31) | static_cast<unsigned int>(sig)"));
        assert!(source.contains("(static_cast<unsigned int>(xs == ys) * xs) << 31"));
        assert!(source.contains("if (exponent < -126)"));
        assert!(source.contains("bool tiny = unbounded_exp < -126"));
        assert!(source.contains("(policy == 0u && tiny && inexact)"));
        assert!(source.contains("(policy == 1u && (subnormal || (tiny && inexact)))"));
        assert!(!source.contains("unsigned long long a[5]"));
        assert!(!source.contains("fusion_f32_place"));
        assert!(!source.contains("fusion_f32_add("));
        assert!(!source.contains("fusion_f32_sub("));
        assert!(!source.contains("float v3 = v1 * v2"));
        assert!(source.contains("for (int i = 0; i < 26; ++i)"));
        assert!(source.contains("if (remainder != 0ull) ext |= 1ull"));
        assert!(source.contains("if ((y << 1) == 0u) return {0u, 1u}"));
        assert!(!source.contains("float v3 = v1 / v2"));
    }
}

#[test]
fn checked_float_emits_the_selected_underflow_policy() {
    for (policy, tag) in [
        (PcuFloatUnderflowPolicy::IeeeAfterRounding, "0u"),
        (PcuFloatUnderflowPolicy::RejectSubnormalResult, "1u"),
        (PcuFloatUnderflowPolicy::AllowGradualUnderflow, "2u"),
    ] {
        let source = checked_kernel(PcuDispatchFloatBinaryOp::Mul, policy);
        assert!(source.contains(&format!("), 2u, {tag})")));
        let division = checked_kernel(PcuDispatchFloatBinaryOp::Div, policy);
        assert!(division.contains(&format!("), 3u, {tag})")));
    }
}

fn checked_f64_to_f32_kernel(policy: PcuFloatUnderflowPolicy, grid: bool) -> String {
    checked_f64_to_f32_kernel_with_range_policy(policy, PcuRangePolicy::Reject, grid)
}

fn checked_f64_to_f32_kernel_with_range_policy(
    policy: PcuFloatUnderflowPolicy,
    range_policy: PcuRangePolicy,
    grid: bool,
) -> String {
    let bindings = [
        PcuBinding::value(
            Some("input"),
            0,
            0,
            PcuBindingStorageClass::Storage,
            PcuBindingAccess::ReadOnly,
            PcuValueType::f64(),
        ),
        PcuBinding::value(
            Some("output"),
            0,
            1,
            PcuBindingStorageClass::Storage,
            PcuBindingAccess::WriteOnly,
            PcuValueType::f32(),
        ),
    ];
    let body = [
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
            result: PcuDispatchValueId(1),
            binding: PcuBindingRef::new(0, 0),
            index: if grid {
                PcuDispatchIndex::GridStrideId
            } else {
                PcuDispatchIndex::InvocationId
            },
        }),
        PcuDispatchOp::Data(PcuDispatchDataOp::CheckedFloatConvert {
            conversion: PcuDispatchCheckedFloatConversion::F64ToF32,
            underflow_policy: policy,
            range_policy,
            result: PcuDispatchValueId(2),
            value: PcuDispatchValueId(1),
        }),
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
            binding: PcuBindingRef::new(0, 1),
            index: if grid {
                PcuDispatchIndex::GridStrideId
            } else {
                PcuDispatchIndex::InvocationId
            },
            value: PcuDispatchValueId(2),
        }),
    ];
    let grid_ops = [
        PcuDispatchOp::GridStrideLoop {
            extent: 64,
            body: &body,
        },
        PcuDispatchOp::Control(PcuDispatchControlOp::Return),
    ];
    let direct_ops = [
        body[0],
        body[1],
        body[2],
        PcuDispatchOp::Control(PcuDispatchControlOp::Return),
    ];
    lower_dispatch_to_hip_source(&PcuDispatchKernelIr {
        numerical_requirements: PcuDispatchKernelIr::DEFAULT_REQUIREMENTS,
        id: PcuKernelId(92),
        entry: PcuDispatchEntryPoint {
            name: "checked_f64_to_f32_test",
            logical_shape: [64, 1, 1],
        },
        bindings: &bindings,
        ports: &[],
        parameters: &[],
        ops: if grid { &grid_ops } else { &direct_ops },
        type_caps: PcuValueTypeCaps::FLOAT32 | PcuValueTypeCaps::FLOAT64,
        feature_caps: if range_policy == PcuRangePolicy::Clamp {
            PcuDispatchFeatureCaps::RANGE_CLAMP
        } else {
            PcuDispatchFeatureCaps::empty()
        },
    })
    .expect("bounded checked conversion kernel")
}

#[test]
fn checked_f64_to_f32_uses_integer_rounding_and_checked_fault_channel() {
    for (policy, tag) in [
        (PcuFloatUnderflowPolicy::IeeeAfterRounding, "0u"),
        (PcuFloatUnderflowPolicy::RejectSubnormalResult, "1u"),
        (PcuFloatUnderflowPolicy::AllowGradualUnderflow, "2u"),
    ] {
        for grid in [false, true] {
            let source = checked_f64_to_f32_kernel(policy, grid);
            assert!(source.contains(&format!(
                "fusion_checked_f64_to_f32(__builtin_bit_cast(unsigned long long, v1), {tag})"
            )));
            assert!(source.contains("bool tiny_after_rounding = rounded_exponent < -126"));
            assert!(source.contains("bool increment = remainder > halfway || (remainder == halfway && (quotient & 1ull) != 0ull)"));
            assert!(source.contains("precision_inexact"));
            assert!(source.contains("| fusion_f64_to_f32_checked_2.fault"));
            assert!(source.contains(
                "float v2 = __builtin_bit_cast(float, fusion_f64_to_f32_checked_2.bits)"
            ));
            assert!(source.contains("return {0u, 5u}"));
            assert!(source.contains("{(sign | 0x7f800000u), 3u}"));
            assert!(!source.contains("static_cast<float>(v1)"));
            assert!(!source.contains("float v2 = v1"));
        }
    }
}

fn checked_f32_to_f64_kernel(grid: bool) -> String {
    let bindings = [
        PcuBinding::value(
            Some("input"),
            0,
            0,
            PcuBindingStorageClass::Storage,
            PcuBindingAccess::ReadOnly,
            PcuValueType::f32(),
        ),
        PcuBinding::value(
            Some("output"),
            0,
            1,
            PcuBindingStorageClass::Storage,
            PcuBindingAccess::WriteOnly,
            PcuValueType::f64(),
        ),
    ];
    let body = [
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
            result: PcuDispatchValueId(1),
            binding: PcuBindingRef::new(0, 0),
            index: if grid {
                PcuDispatchIndex::GridStrideId
            } else {
                PcuDispatchIndex::InvocationId
            },
        }),
        PcuDispatchOp::Data(PcuDispatchDataOp::CheckedFloatConvert {
            conversion: PcuDispatchCheckedFloatConversion::F32ToF64,
            underflow_policy: PcuFloatUnderflowPolicy::IeeeAfterRounding,
            range_policy: PcuRangePolicy::Reject,
            result: PcuDispatchValueId(2),
            value: PcuDispatchValueId(1),
        }),
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
            binding: PcuBindingRef::new(0, 1),
            index: if grid {
                PcuDispatchIndex::GridStrideId
            } else {
                PcuDispatchIndex::InvocationId
            },
            value: PcuDispatchValueId(2),
        }),
    ];
    let grid_ops = [
        PcuDispatchOp::GridStrideLoop {
            extent: 64,
            body: &body,
        },
        PcuDispatchOp::Control(PcuDispatchControlOp::Return),
    ];
    let direct_ops = [
        body[0],
        body[1],
        body[2],
        PcuDispatchOp::Control(PcuDispatchControlOp::Return),
    ];
    lower_dispatch_to_hip_source(&PcuDispatchKernelIr {
        numerical_requirements: PcuDispatchKernelIr::DEFAULT_REQUIREMENTS,
        id: PcuKernelId(94),
        entry: PcuDispatchEntryPoint {
            name: "checked_f32_to_f64_test",
            logical_shape: [64, 1, 1],
        },
        bindings: &bindings,
        ports: &[],
        parameters: &[],
        ops: if grid { &grid_ops } else { &direct_ops },
        type_caps: PcuValueTypeCaps::FLOAT32 | PcuValueTypeCaps::FLOAT64,
        feature_caps: PcuDispatchFeatureCaps::empty(),
    })
    .expect("bounded checked widening kernel")
}

#[test]
fn checked_f32_to_f64_widens_finite_bits_with_integer_only_logic() {
    for grid in [false, true] {
        let source = checked_f32_to_f64_kernel(grid);
        assert!(source.contains("fusion_checked_f32_to_f64(__builtin_bit_cast(unsigned int, v1))"));
        assert!(source.contains("int top = 31 - __builtin_clz(fraction)"));
        assert!(source.contains("exponent_field + 896u"));
        assert!(source.contains("static_cast<unsigned long long>(fraction) << 29u"));
        assert!(source.contains("| fusion_f32_to_f64_checked_2.fault"));
        assert!(
            source.contains(
                "double v2 = __builtin_bit_cast(double, fusion_f32_to_f64_checked_2.bits)"
            )
        );
        assert!(source.contains("return {0ull, 5u}"));
        assert!(!source.contains("static_cast<double>(v1)"));
        assert!(!source.contains("double v2 = v1"));
    }
}

#[test]
#[allow(clippy::too_many_lines)] // Explicit IR is the subject of this mixed-width lowering test.
fn checked_conversion_composes_with_checked_math_on_both_widths() {
    let bindings = [
        PcuBinding::value(
            Some("wide_lhs"),
            0,
            0,
            PcuBindingStorageClass::Storage,
            PcuBindingAccess::ReadOnly,
            PcuValueType::f64(),
        ),
        PcuBinding::value(
            Some("wide_rhs"),
            0,
            1,
            PcuBindingStorageClass::Storage,
            PcuBindingAccess::ReadOnly,
            PcuValueType::f64(),
        ),
        PcuBinding::value(
            Some("narrow_rhs"),
            0,
            2,
            PcuBindingStorageClass::Storage,
            PcuBindingAccess::ReadOnly,
            PcuValueType::f32(),
        ),
        PcuBinding::value(
            Some("output"),
            0,
            3,
            PcuBindingStorageClass::Storage,
            PcuBindingAccess::WriteOnly,
            PcuValueType::f32(),
        ),
    ];
    let index = PcuDispatchIndex::InvocationId;
    let ops = [
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
            result: PcuDispatchValueId(1),
            binding: PcuBindingRef::new(0, 0),
            index,
        }),
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
            result: PcuDispatchValueId(2),
            binding: PcuBindingRef::new(0, 1),
            index,
        }),
        PcuDispatchOp::Data(PcuDispatchDataOp::CheckedFloatBinary {
            value_type: PcuValueType::f64(),
            op: PcuDispatchFloatBinaryOp::Add,
            underflow_policy: PcuFloatUnderflowPolicy::IeeeAfterRounding,
            range_policy: PcuRangePolicy::Reject,
            result: PcuDispatchValueId(3),
            lhs: PcuDispatchValueId(1),
            rhs: PcuDispatchValueId(2),
        }),
        PcuDispatchOp::Data(PcuDispatchDataOp::CheckedFloatConvert {
            conversion: PcuDispatchCheckedFloatConversion::F64ToF32,
            underflow_policy: PcuFloatUnderflowPolicy::AllowGradualUnderflow,
            range_policy: PcuRangePolicy::Reject,
            result: PcuDispatchValueId(4),
            value: PcuDispatchValueId(3),
        }),
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
            result: PcuDispatchValueId(5),
            binding: PcuBindingRef::new(0, 2),
            index,
        }),
        PcuDispatchOp::Data(PcuDispatchDataOp::CheckedFloatBinary {
            value_type: PcuValueType::f32(),
            op: PcuDispatchFloatBinaryOp::Mul,
            underflow_policy: PcuFloatUnderflowPolicy::RejectSubnormalResult,
            range_policy: PcuRangePolicy::Reject,
            result: PcuDispatchValueId(6),
            lhs: PcuDispatchValueId(4),
            rhs: PcuDispatchValueId(5),
        }),
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
            binding: PcuBindingRef::new(0, 3),
            index,
            value: PcuDispatchValueId(6),
        }),
        PcuDispatchOp::Control(PcuDispatchControlOp::Return),
    ];
    let source = lower_dispatch_to_hip_source(&PcuDispatchKernelIr {
        numerical_requirements: PcuDispatchKernelIr::DEFAULT_REQUIREMENTS,
        id: PcuKernelId(93),
        entry: PcuDispatchEntryPoint {
            name: "mixed_checked_float_test",
            logical_shape: [64, 1, 1],
        },
        bindings: &bindings,
        ports: &[],
        parameters: &[],
        ops: &ops,
        type_caps: PcuValueTypeCaps::FLOAT32 | PcuValueTypeCaps::FLOAT64,
        feature_caps: PcuDispatchFeatureCaps::empty(),
    })
    .expect("mixed-width checked conversion map");
    assert!(source.contains("fusion_checked_f64_binary"));
    assert!(source.contains("fusion_checked_f64_to_f32"));
    assert!(source.contains("fusion_checked_f32_binary"));
    assert!(source.contains("| fusion_f64_checked_3.fault"));
    assert!(source.contains("| fusion_f64_to_f32_checked_4.fault"));
    assert!(source.contains("| fusion_f32_checked_6.fault"));
    assert!(!source.contains("double v3 = v1 + v2"));
    assert!(!source.contains("float v6 = v4 * v5"));
}
