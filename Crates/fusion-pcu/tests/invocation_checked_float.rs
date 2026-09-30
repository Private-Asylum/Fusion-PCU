//! Checked invocation arithmetic shares the owned-source fault and policy contract.
#![cfg(feature = "rocm")]

#[rustfmt::skip]
use fusion_pcu::{
    global,
    pcu,
    PcuCheckedFloat,
    PcuCheckedFloatWidening,
    PcuExecutionError,
    PcuExecutionFaultKind,
    PcuFloatUnderflowPolicy,
};

macro_rules! define_profile {
    ($module:ident, $scalar:ident) => {
        mod $module {
            use super::*;

            #[pcu(invocations: N)]
            fn add<const N: usize>(lhs: &[$scalar; N], rhs: &[$scalar; N], out: &mut [$scalar; N]) {
                let id = pcu::context::global_invocation_id();
                out[id] = lhs[id] + rhs[id];
            }
            #[pcu(invocations: N)]
            fn sub<const N: usize>(lhs: &[$scalar; N], rhs: &[$scalar; N], out: &mut [$scalar; N]) {
                let id = pcu::context::global_invocation_id();
                out[id] = lhs[id] - rhs[id];
            }
            #[pcu(invocations: N)]
            fn mul<const N: usize>(lhs: &[$scalar; N], rhs: &[$scalar; N], out: &mut [$scalar; N]) {
                let id = pcu::context::global_invocation_id();
                out[id] = lhs[id] * rhs[id];
            }
            #[pcu(invocations: N)]
            fn div<const N: usize>(lhs: &[$scalar; N], rhs: &[$scalar; N], out: &mut [$scalar; N]) {
                let id = pcu::context::global_invocation_id();
                out[id] = lhs[id] / rhs[id];
            }
            #[pcu(invocations: N, flag(allow_gradual_underflow))]
            fn gradual<const N: usize>(
                lhs: &[$scalar; N],
                rhs: &[$scalar; N],
                out: &mut [$scalar; N],
            ) {
                let id = pcu::context::global_invocation_id();
                out[id] = lhs[id] / rhs[id];
            }
            #[pcu(invocations: N, flag(ieee_underflow))]
            fn ieee<const N: usize>(
                lhs: &[$scalar; N],
                rhs: &[$scalar; N],
                out: &mut [$scalar; N],
            ) {
                let id = pcu::context::global_invocation_id();
                out[id] = lhs[id] / rhs[id];
            }
            #[pcu(invocations: N, flag(reject_subnormal_result))]
            fn strict<const N: usize>(
                lhs: &[$scalar; N],
                rhs: &[$scalar; N],
                out: &mut [$scalar; N],
            ) {
                let id = pcu::context::global_invocation_id();
                out[id] = lhs[id] / rhs[id];
            }

            #[test]
            fn manual_ir_default_and_explicit_policy_are_frozen() {
                let bindings = div_bindings();
                let default = div_ir::<4>(&bindings).unwrap();
                let gradual = gradual_ir::<4>(&bindings).unwrap();
                for (builder, policy) in [
                    (&default, PcuFloatUnderflowPolicy::IeeeAfterRounding),
                    (&gradual, PcuFloatUnderflowPolicy::AllowGradualUnderflow),
                ] {
                    let ir = builder.ir();
                    assert!(ir.ops.iter().any(|op| matches!(op,
                        fusion_pcu::PcuDispatchOp::Data(fusion_pcu::PcuDispatchDataOp::CheckedFloatBinary {
                            op: fusion_pcu::PcuDispatchFloatBinaryOp::Div,
                            underflow_policy,
                            ..
                        }) if *underflow_policy == policy
                    )));
                    assert!(!ir.ops.iter().any(|op| matches!(op,
                        fusion_pcu::PcuDispatchOp::Data(fusion_pcu::PcuDispatchDataOp::Alu { .. })
                    )));
                }
            }

            fn fault(result: Result<(), PcuExecutionError>, kind: PcuExecutionFaultKind) {
                match result {
                    Err(PcuExecutionError::ArithmeticFault(record)) => {
                        assert_eq!(record.kind, kind);
                        assert_eq!(record.invocation_id, 1);
                    }
                    other => panic!("expected {kind:?}, got {other:?}"),
                }
            }

            #[test]
            #[ignore = "requires a working ROCm device"]
            fn checked_operations_fault_and_retry() {
                global::use_defaults().unwrap();
                global::clear_thread_cache().unwrap();
                let lhs = [3.0, -4.0, $scalar::MIN_POSITIVE, $scalar::from_bits(1)];
                let rhs = [2.0, 3.0, 2.0, 1.0];
                let mut out = [0.0; 4];
                add(&lhs, &rhs, &mut out).unwrap();
                assert_eq!(
                    out.map($scalar::to_bits),
                    core::array::from_fn(|i| lhs[i].pcu_checked_add(rhs[i]).unwrap().to_bits())
                );
                sub(&lhs, &rhs, &mut out).unwrap();
                assert_eq!(
                    out.map($scalar::to_bits),
                    core::array::from_fn(|i| lhs[i].pcu_checked_sub(rhs[i]).unwrap().to_bits())
                );
                mul(&lhs, &rhs, &mut out).unwrap();
                assert_eq!(
                    out.map($scalar::to_bits),
                    core::array::from_fn(|i| lhs[i].pcu_checked_mul(rhs[i]).unwrap().to_bits())
                );
                div(&lhs, &rhs, &mut out).unwrap();
                assert_eq!(
                    out.map($scalar::to_bits),
                    core::array::from_fn(|i| lhs[i].pcu_checked_div(rhs[i]).unwrap().to_bits())
                );

                fault(
                    add(&[1.0, $scalar::MAX], &[1.0, $scalar::MAX], &mut [0.0; 2]),
                    PcuExecutionFaultKind::ArithmeticOverflow,
                );
                fault(
                    sub(&[1.0, -$scalar::MAX], &[1.0, $scalar::MAX], &mut [0.0; 2]),
                    PcuExecutionFaultKind::ArithmeticOverflow,
                );
                fault(
                    mul(&[1.0, $scalar::INFINITY], &[1.0, 0.0], &mut [0.0; 2]),
                    PcuExecutionFaultKind::InvalidFloatingOperand,
                );
                fault(
                    div(&[1.0, 0.0], &[1.0, 0.0], &mut [0.0; 2]),
                    PcuExecutionFaultKind::DivideByZero,
                );
                // The same cached callsite must clear its previous terminal fault before reuse.
                let mut retry = [0.0; 2];
                div(&[1.0, 6.0], &[1.0, 2.0], &mut retry).unwrap();
                assert_eq!(retry.map($scalar::to_bits), [1.0 as $scalar, 3.0 as $scalar].map($scalar::to_bits));
                global::clear_thread_cache().unwrap();
            }

            #[test]
            #[ignore = "requires a working ROCm device"]
            fn global_snapshot_and_explicit_policies_are_distinct() {
                global::use_defaults().unwrap();
                let tiny = [1.0, $scalar::from_bits(1)];
                let halves = [1.0, 2.0];
                let mut out = [99.0; 2];
                fault(
                    div(&tiny, &halves, &mut out),
                    PcuExecutionFaultKind::ArithmeticUnderflow,
                );
                gradual(&tiny, &halves, &mut out).unwrap();
                assert_eq!(out.map($scalar::to_bits), [1.0 as $scalar, 0.0 as $scalar].map($scalar::to_bits));
                global::configure(global::PcuExecutionPolicy {
                    float_underflow: PcuFloatUnderflowPolicy::AllowGradualUnderflow,
                    ..Default::default()
                })
                .unwrap();
                div(&tiny, &halves, &mut out).unwrap();
                assert_eq!(out.map($scalar::to_bits), [1.0 as $scalar, 0.0 as $scalar].map($scalar::to_bits));
                fault(
                    ieee(&tiny, &halves, &mut out),
                    PcuExecutionFaultKind::ArithmeticUnderflow,
                );
                fault(
                    strict(&[1.0, $scalar::MIN_POSITIVE], &halves, &mut out),
                    PcuExecutionFaultKind::ArithmeticUnderflow,
                );
                global::use_defaults().unwrap();
                fault(
                    div(&tiny, &halves, &mut out),
                    PcuExecutionFaultKind::ArithmeticUnderflow,
                );
                global::clear_thread_cache().unwrap();
            }
        }
    };
}

define_profile!(binary32, f32);
define_profile!(binary64, f64);

#[pcu]
fn divide_helper(value: f64, denominator: f64) -> f64 {
    value / denominator
}

#[allow(clippy::suboptimal_flops)] // Separate checked multiply and add must preserve both fault boundaries.
#[pcu]
fn affine_helper(value: f64, denominator: f64) -> f64 {
    divide_helper(value, denominator) * 2.0 + 1.0
}

#[pcu(invocations: 4)]
fn helper_grid<const N: usize>(denominator: &f64, input: &[f64; N], output: &mut [f64; N]) {
    let mut id = pcu::context::global_invocation_id();
    let stride = pcu::context::invocation_count();
    while id < N {
        output[id] = affine_helper(input[id], *denominator);
        id += stride;
    }
}

#[test]
fn nested_helper_grid_is_checked_and_admitted() {
    let bindings = helper_grid_bindings();
    let builder = helper_grid_ir::<17>(&bindings).unwrap();
    builder.with_ir(|ir| {
        fusion_pcu::validate_checked_float_map_kernel(
            ir,
            fusion_pcu::PcuValueType::f64(),
            fusion_pcu::PcuValueTypeCaps::FLOAT64,
        )
        .unwrap();
        let body = ir
            .ops
            .iter()
            .find_map(|op| match op {
                fusion_pcu::PcuDispatchOp::GridStrideLoop { body, .. } => Some(*body),
                _ => None,
            })
            .unwrap();
        let checked = body
            .iter()
            .filter(|op| {
                matches!(
                    op,
                    fusion_pcu::PcuDispatchOp::Data(
                        fusion_pcu::PcuDispatchDataOp::CheckedFloatBinary { .. }
                    )
                )
            })
            .count();
        assert_eq!(checked, 3);
        assert!(!body.iter().any(|op| matches!(
            op,
            fusion_pcu::PcuDispatchOp::Data(fusion_pcu::PcuDispatchDataOp::Alu { .. })
        )));
    });
}

#[test]
#[ignore = "requires a working ROCm device"]
fn nested_helpers_and_grid_stride_report_faults() {
    global::use_defaults().unwrap();
    let input =
        core::array::from_fn::<_, 17, _>(|index| f64::from(u32::try_from(index).unwrap()) - 8.0);
    let mut output = [0.0_f64; 17];
    helper_grid(&2.0, &input, &mut output).unwrap();
    assert_eq!(
        output.map(f64::to_bits),
        input.map(|value| (value + 1.0).to_bits())
    );
    assert!(matches!(
        helper_grid(&0.0, &input, &mut output),
        Err(PcuExecutionError::ArithmeticFault(fault))
            if fault.kind == PcuExecutionFaultKind::DivideByZero && fault.invocation_id == 0
    ));
    helper_grid(&2.0, &input, &mut output).unwrap();
    assert_eq!(
        output.map(f64::to_bits),
        input.map(|value| (value + 1.0).to_bits())
    );
    global::clear_thread_cache().unwrap();
}

#[pcu(invocations: N)]
fn cast_f64_to_f32<const N: usize>(input: &[f64; N], output: &mut [f32; N]) {
    let id = pcu::context::global_invocation_id();
    output[id] = input[id] as f32;
}

#[pcu(invocations: N, flag(allow_gradual_underflow))]
fn cast_f64_to_f32_gradual<const N: usize>(input: &[f64; N], output: &mut [f32; N]) {
    let id = pcu::context::global_invocation_id();
    output[id] = input[id] as f32;
}

#[pcu(invocations: N, flag(ieee_underflow))]
fn cast_f64_to_f32_ieee<const N: usize>(input: &[f64; N], output: &mut [f32; N]) {
    let id = pcu::context::global_invocation_id();
    output[id] = input[id] as f32;
}

#[pcu(invocations: N, flag(reject_subnormal_result))]
fn cast_f64_to_f32_strict<const N: usize>(input: &[f64; N], output: &mut [f32; N]) {
    let id = pcu::context::global_invocation_id();
    output[id] = input[id] as f32;
}

#[pcu(invocations: N, flag(reject_subnormal_result))]
fn cast_f32_to_f64<const N: usize>(input: &[f32; N], output: &mut [f64; N]) {
    let id = pcu::context::global_invocation_id();
    output[id] = input[id] as f64;
}

macro_rules! define_macro_forwarded_cast {
    ($name:ident, $source:ty, $target:ty) => {
        #[pcu(invocations: N)]
        fn $name<const N: usize>(input: &[$source; N], output: &mut [$target; N]) {
            let id = pcu::context::global_invocation_id();
            output[id] = input[id] as $target;
        }
    };
}

define_macro_forwarded_cast!(forwarded_f32_to_f64, f32, f64);
define_macro_forwarded_cast!(forwarded_f64_to_f32, f64, f32);

#[allow(clippy::cast_lossless, clippy::cast_possible_truncation)] // Both casts use checked PCU conversion instructions.
#[pcu]
fn widen_add_narrow_helper(value: f32) -> f32 {
    ((value as f64) + 0.0_f64) as f32
}

#[pcu(invocations: 4, flag(allow_gradual_underflow))]
fn widen_narrow_gradual_grid<const N: usize>(input: &[f32; N], output: &mut [f32; N]) {
    let mut id = pcu::context::global_invocation_id();
    let stride = pcu::context::invocation_count();
    while id < N {
        output[id] = widen_add_narrow_helper(input[id]);
        id += stride;
    }
}

#[pcu(invocations: 4)]
fn widen_narrow_default_grid<const N: usize>(input: &[f32; N], output: &mut [f32; N]) {
    let mut id = pcu::context::global_invocation_id();
    let stride = pcu::context::invocation_count();
    while id < N {
        output[id] = widen_add_narrow_helper(input[id]);
        id += stride;
    }
}

#[allow(clippy::cast_possible_truncation)] // The cast is routed through the checked PCU conversion lowering.
#[pcu]
fn cast_and_add_helper(value: f32) -> f32 {
    1.0_f64 as f32 + value
}

#[pcu(invocations: 4)]
fn cast_grid_with_helper<const N: usize>(input: &[f64; N], output: &mut [f32; N]) {
    let mut id = pcu::context::global_invocation_id();
    let stride = pcu::context::invocation_count();
    while id < N {
        output[id] = cast_and_add_helper((input[id] + 1.0_f64) as f32);
        id += stride;
    }
}

#[test]
#[allow(clippy::too_many_lines)]
fn invocation_cast_ir_is_typed_checked_and_policy_aware() {
    // Forwarded `$ty` fragments are opaque groups to syn until transparent wrappers are peeled.
    let bindings = forwarded_f32_to_f64_bindings();
    let forwarded_widen = forwarded_f32_to_f64_ir::<4>(&bindings).unwrap();
    forwarded_widen.with_ir(|ir| {
        assert!(ir.ops.iter().any(|op| matches!(
            op,
            fusion_pcu::PcuDispatchOp::Data(fusion_pcu::PcuDispatchDataOp::CheckedFloatConvert {
                conversion: fusion_pcu::PcuDispatchCheckedFloatConversion::F32ToF64,
                ..
            })
        )));
    });
    let bindings = forwarded_f64_to_f32_bindings();
    let forwarded_narrow = forwarded_f64_to_f32_ir::<4>(&bindings).unwrap();
    forwarded_narrow.with_ir(|ir| {
        assert!(ir.ops.iter().any(|op| matches!(
            op,
            fusion_pcu::PcuDispatchOp::Data(fusion_pcu::PcuDispatchDataOp::CheckedFloatConvert {
                conversion: fusion_pcu::PcuDispatchCheckedFloatConversion::F64ToF32,
                ..
            })
        )));
    });

    let bindings = cast_f64_to_f32_bindings();
    let default = cast_f64_to_f32_ir::<4>(&bindings).unwrap();
    default.with_ir(|ir| {
        assert!(ir.ops.iter().any(|op| matches!(
            op,
            fusion_pcu::PcuDispatchOp::Data(fusion_pcu::PcuDispatchDataOp::CheckedFloatConvert {
                underflow_policy: PcuFloatUnderflowPolicy::IeeeAfterRounding,
                ..
            })
        )));
        assert!(!ir.ops.iter().any(|op| matches!(
            op,
            fusion_pcu::PcuDispatchOp::Data(
                fusion_pcu::PcuDispatchDataOp::Convert { .. }
                    | fusion_pcu::PcuDispatchDataOp::Alu { .. }
            )
        )));
    });

    let bindings = cast_f64_to_f32_gradual_bindings();
    let gradual = cast_f64_to_f32_gradual_ir::<4>(&bindings).unwrap();
    gradual.with_ir(|ir| {
        assert!(ir.ops.iter().any(|op| matches!(
            op,
            fusion_pcu::PcuDispatchOp::Data(fusion_pcu::PcuDispatchDataOp::CheckedFloatConvert {
                underflow_policy: PcuFloatUnderflowPolicy::AllowGradualUnderflow,
                ..
            })
        )));
    });

    let bindings = cast_grid_with_helper_bindings();
    let grid = cast_grid_with_helper_ir::<17>(&bindings).unwrap();
    grid.with_ir(|ir| {
        let body = ir
            .ops
            .iter()
            .find_map(|op| match op {
                fusion_pcu::PcuDispatchOp::GridStrideLoop { body, .. } => Some(*body),
                _ => None,
            })
            .expect("grid-stride body exists");
        assert!(body.iter().any(|op| matches!(
            op,
            fusion_pcu::PcuDispatchOp::Data(
                fusion_pcu::PcuDispatchDataOp::CheckedFloatConvert { .. }
            )
        )));
        assert!(
            body.iter()
                .filter(|op| matches!(
                    op,
                    fusion_pcu::PcuDispatchOp::Data(
                        fusion_pcu::PcuDispatchDataOp::CheckedFloatBinary { .. }
                    )
                ))
                .count()
                >= 2
        );
    });

    let bindings = cast_f32_to_f64_bindings();
    let widened = cast_f32_to_f64_ir::<4>(&bindings).unwrap();
    widened.with_ir(|ir| {
        assert!(ir.ops.iter().any(|op| matches!(
            op,
            fusion_pcu::PcuDispatchOp::Data(fusion_pcu::PcuDispatchDataOp::CheckedFloatConvert {
                conversion: fusion_pcu::PcuDispatchCheckedFloatConversion::F32ToF64,
                underflow_policy: PcuFloatUnderflowPolicy::RejectSubnormalResult,
                ..
            })
        )));
    });

    let bindings = widen_narrow_gradual_grid_bindings();
    let round_trip = widen_narrow_gradual_grid_ir::<17>(&bindings).unwrap();
    round_trip.with_ir(|ir| {
        let body = ir
            .ops
            .iter()
            .find_map(|op| match op {
                fusion_pcu::PcuDispatchOp::GridStrideLoop { body, .. } => Some(*body),
                _ => None,
            })
            .expect("widen-narrow grid body exists");
        assert_eq!(
            body.iter()
                .filter(|op| matches!(
                    op,
                    fusion_pcu::PcuDispatchOp::Data(
                        fusion_pcu::PcuDispatchDataOp::CheckedFloatConvert { .. }
                    )
                ))
                .count(),
            2
        );
        assert!(body.iter().any(|op| matches!(
            op,
            fusion_pcu::PcuDispatchOp::Data(fusion_pcu::PcuDispatchDataOp::CheckedFloatConvert {
                conversion: fusion_pcu::PcuDispatchCheckedFloatConversion::F32ToF64,
                underflow_policy: PcuFloatUnderflowPolicy::AllowGradualUnderflow,
                ..
            })
        )));
        assert!(body.iter().any(|op| matches!(
            op,
            fusion_pcu::PcuDispatchOp::Data(fusion_pcu::PcuDispatchDataOp::CheckedFloatConvert {
                conversion: fusion_pcu::PcuDispatchCheckedFloatConversion::F64ToF32,
                underflow_policy: PcuFloatUnderflowPolicy::AllowGradualUnderflow,
                ..
            })
        )));
        assert!(body.iter().any(|op| matches!(
            op,
            fusion_pcu::PcuDispatchOp::Data(fusion_pcu::PcuDispatchDataOp::CheckedFloatBinary {
                value_type: fusion_pcu::PcuValueType::Scalar(fusion_pcu::PcuScalarType::F64),
                ..
            })
        )));
    });
}

#[test]
#[ignore = "requires a working ROCm device"]
#[allow(clippy::too_many_lines)]
fn invocation_f64_to_f32_faults_retry_and_policy_overrides() {
    use fusion_pcu::PcuCheckedFloatConversion;

    global::use_defaults().unwrap();
    global::clear_thread_cache().unwrap();
    let input = [1.0_f64, f64::from_bits(1)];
    let mut output = [0.0_f32; 2];
    match cast_f64_to_f32(&input, &mut output) {
        Err(PcuExecutionError::ArithmeticFault(record)) => {
            assert_eq!(record.kind, PcuExecutionFaultKind::ArithmeticUnderflow);
            assert_eq!(record.invocation_id, 1);
        }
        other => panic!("expected conversion underflow, got {other:?}"),
    }

    cast_f64_to_f32_gradual(&input, &mut output).unwrap();
    assert_eq!(
        output.map(f32::to_bits),
        input.map(|value| value
            .pcu_checked_to_f32_with_policy(PcuFloatUnderflowPolicy::AllowGradualUnderflow)
            .unwrap()
            .to_bits())
    );

    global::configure(global::PcuExecutionPolicy {
        float_underflow: PcuFloatUnderflowPolicy::AllowGradualUnderflow,
        ..Default::default()
    })
    .unwrap();
    cast_f64_to_f32(&input, &mut output).unwrap();
    assert_eq!(output[1].to_bits(), 0);
    match cast_f64_to_f32_ieee(&input, &mut output) {
        Err(PcuExecutionError::ArithmeticFault(record)) => {
            assert_eq!(record.kind, PcuExecutionFaultKind::ArithmeticUnderflow);
            assert_eq!(record.invocation_id, 1);
        }
        other => panic!("explicit IEEE policy must override global gradual policy: {other:?}"),
    }

    // An exact f32 subnormal is accepted by IEEE policy and rejected only by the stricter flag.
    let exact_subnormal = [f64::from_bits(0x36a0_0000_0000_0000)];
    let mut exact_output = [0.0_f32; 1];
    cast_f64_to_f32_ieee(&exact_subnormal, &mut exact_output).unwrap();
    assert_eq!(exact_output[0].to_bits(), 1);
    match cast_f64_to_f32_strict(&exact_subnormal, &mut exact_output) {
        Err(PcuExecutionError::ArithmeticFault(record)) => {
            assert_eq!(record.kind, PcuExecutionFaultKind::ArithmeticUnderflow);
            assert_eq!(record.invocation_id, 0);
        }
        other => panic!("strict policy must reject exact subnormal output: {other:?}"),
    }

    // Exact widening preserves every finite F32 bit pattern, including signed zero.
    global::configure(global::PcuExecutionPolicy {
        float_underflow: PcuFloatUnderflowPolicy::RejectSubnormalResult,
        ..Default::default()
    })
    .unwrap();
    let widen_input = [
        1.0_f32,
        f32::from_bits(0x3f80_0001),
        -0.0,
        f32::MIN_POSITIVE,
    ];
    let mut widen_output = [0.0_f64; 4];
    cast_f32_to_f64(&widen_input, &mut widen_output).unwrap();
    assert_eq!(
        widen_output.map(f64::to_bits),
        widen_input.map(|value| value.pcu_checked_to_f64().unwrap().to_bits())
    );
    let mut invalid_widen_output = [0.0_f64; 2];
    match cast_f32_to_f64(&[1.0_f32, f32::INFINITY], &mut invalid_widen_output) {
        Err(PcuExecutionError::ArithmeticFault(record)) => {
            assert_eq!(record.kind, PcuExecutionFaultKind::InvalidFloatingOperand);
            assert_eq!(record.invocation_id, 1);
        }
        other => panic!("non-finite widening input must fault: {other:?}"),
    }

    // The default kernel snapshots global strict policy and rejects an exact subnormal
    // after narrowing; the explicit gradual grid/helper overrides it and preserves the value.
    let subnormal = [f32::from_bits(1); 17];
    let mut round_trip = [0.0_f32; 17];
    match widen_narrow_default_grid(&subnormal, &mut round_trip) {
        Err(PcuExecutionError::ArithmeticFault(record)) => {
            assert_eq!(record.kind, PcuExecutionFaultKind::ArithmeticUnderflow);
        }
        other => panic!("global strict policy must reject exact subnormal: {other:?}"),
    }
    widen_narrow_gradual_grid(&subnormal, &mut round_trip).unwrap();
    assert_eq!(round_trip.map(f32::to_bits), subnormal.map(f32::to_bits));

    // Exercise F64 checked arithmetic before the cast, then the cast and checked F32 math
    // inside a nested scalar helper in a grid-stride loop.
    let grid_input =
        core::array::from_fn::<_, 17, _>(|index| f64::from(u32::try_from(index).unwrap()) - 8.0);
    let mut grid_output = [0.0_f32; 17];
    cast_grid_with_helper(&grid_input, &mut grid_output).unwrap();
    assert_eq!(
        grid_output.map(f32::to_bits),
        grid_input.map(|value| {
            value
                .pcu_checked_add(1.0)
                .unwrap()
                .pcu_checked_to_f32_with_policy(PcuFloatUnderflowPolicy::AllowGradualUnderflow)
                .unwrap()
                .pcu_checked_add(1.0)
                .unwrap()
                .to_bits()
        })
    );

    // A successful call after a fault verifies that the same callsite clears its fault record.
    cast_f64_to_f32_gradual(&[2.0, 3.0], &mut output).unwrap();
    assert_eq!(
        output.map(f32::to_bits),
        [2.0_f32.to_bits(), 3.0_f32.to_bits()]
    );
    global::use_defaults().unwrap();
    global::clear_thread_cache().unwrap();
}
