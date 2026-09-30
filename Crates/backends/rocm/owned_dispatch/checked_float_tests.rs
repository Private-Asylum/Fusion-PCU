//! Hardware acceptance for checked binary32 policy, bits and completion.
use super::*;
#[rustfmt::skip]
use fusion_pcu::{
    PcuBinding,
    PcuBindingStorageClass,
    PcuCheckedFloat,
    PcuCompletionOutcome,
    PcuDispatchControlOp,
    PcuDispatchEntryPoint,
    PcuDispatchFloatBinaryOp,
    PcuDispatchIndex,
    PcuDispatchValueId,
    PcuFloatUnderflowPolicy,
    PcuRangePolicy,
    PcuInvocationShape,
    PcuValueType,
    PcuValueTypeCaps,
};
use std::num::NonZeroU32;

fn prepare(
    backend: &RocmOwnedDispatchBackend,
    op: PcuDispatchFloatBinaryOp,
    policy: PcuFloatUnderflowPolicy,
    grid: bool,
    extent: u32,
) -> RocmPreparedDispatch {
    prepare_with_range_policy(backend, op, policy, PcuRangePolicy::Reject, grid, extent)
}

fn prepare_with_range_policy(
    backend: &RocmOwnedDispatchBackend,
    op: PcuDispatchFloatBinaryOp,
    policy: PcuFloatUnderflowPolicy,
    range_policy: PcuRangePolicy,
    grid: bool,
    extent: u32,
) -> RocmPreparedDispatch {
    let bindings = [
        PcuBinding::value(
            None,
            0,
            0,
            PcuBindingStorageClass::Storage,
            PcuBindingAccess::ReadOnly,
            PcuValueType::f32(),
        ),
        PcuBinding::value(
            None,
            0,
            1,
            PcuBindingStorageClass::Storage,
            PcuBindingAccess::ReadOnly,
            PcuValueType::f32(),
        ),
        PcuBinding::value(
            None,
            0,
            2,
            PcuBindingStorageClass::Storage,
            PcuBindingAccess::WriteOnly,
            PcuValueType::f32(),
        ),
    ];
    let index = if grid {
        PcuDispatchIndex::GridStrideId
    } else {
        PcuDispatchIndex::InvocationId
    };
    let body = [
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
            value_type: PcuValueType::f32(),
            op,
            underflow_policy: policy,
            range_policy,
            result: PcuDispatchValueId(3),
            lhs: PcuDispatchValueId(1),
            rhs: PcuDispatchValueId(2),
        }),
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
            binding: PcuBindingRef::new(0, 2),
            index,
            value: PcuDispatchValueId(3),
        }),
        PcuDispatchOp::Control(PcuDispatchControlOp::Return),
    ];
    let loop_ops = [
        PcuDispatchOp::GridStrideLoop {
            extent,
            body: &body[..4],
        },
        PcuDispatchOp::Control(PcuDispatchControlOp::Return),
    ];
    let kernel = PcuDispatchKernelIr {
        id: fusion_pcu::PcuKernelId(0xf103),
        entry: PcuDispatchEntryPoint {
            name: "checked_float_acceptance",
            logical_shape: [if grid { 1 } else { extent }, 1, 1],
        },
        bindings: &bindings,
        ports: &[],
        parameters: &[],
        ops: if grid { &loop_ops } else { &body },
        type_caps: PcuValueTypeCaps::FLOAT32,
        feature_caps: PcuDispatchFeatureCaps::READ_ONLY_RESOURCES
            .union(PcuDispatchFeatureCaps::MUTABLE_RESOURCES)
            .union(if range_policy == PcuRangePolicy::Clamp {
                PcuDispatchFeatureCaps::RANGE_CLAMP
            } else {
                PcuDispatchFeatureCaps::empty()
            }),
    };
    backend
        .prepare_dispatch(PcuDispatchSubmission {
            kernel: &kernel,
            shape: PcuInvocationShape::invocations(
                NonZeroU32::new(if grid { 1 } else { extent }).unwrap(),
            ),
        })
        .unwrap()
}

fn prepare_f64(
    backend: &RocmOwnedDispatchBackend,
    op: PcuDispatchFloatBinaryOp,
    policy: PcuFloatUnderflowPolicy,
    grid: bool,
    extent: u32,
) -> RocmPreparedDispatch {
    prepare_f64_with_range_policy(backend, op, policy, PcuRangePolicy::Reject, grid, extent)
}

fn prepare_f64_with_range_policy(
    backend: &RocmOwnedDispatchBackend,
    op: PcuDispatchFloatBinaryOp,
    policy: PcuFloatUnderflowPolicy,
    range_policy: PcuRangePolicy,
    grid: bool,
    extent: u32,
) -> RocmPreparedDispatch {
    let bindings = [
        PcuBinding::value(
            None,
            0,
            0,
            PcuBindingStorageClass::Storage,
            PcuBindingAccess::ReadOnly,
            PcuValueType::f64(),
        ),
        PcuBinding::value(
            None,
            0,
            1,
            PcuBindingStorageClass::Storage,
            PcuBindingAccess::ReadOnly,
            PcuValueType::f64(),
        ),
        PcuBinding::value(
            None,
            0,
            2,
            PcuBindingStorageClass::Storage,
            PcuBindingAccess::WriteOnly,
            PcuValueType::f64(),
        ),
    ];
    let index = if grid {
        PcuDispatchIndex::GridStrideId
    } else {
        PcuDispatchIndex::InvocationId
    };
    let body = [
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
            op,
            underflow_policy: policy,
            range_policy,
            result: PcuDispatchValueId(3),
            lhs: PcuDispatchValueId(1),
            rhs: PcuDispatchValueId(2),
        }),
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
            binding: PcuBindingRef::new(0, 2),
            index,
            value: PcuDispatchValueId(3),
        }),
        PcuDispatchOp::Control(PcuDispatchControlOp::Return),
    ];
    let loop_ops = [
        PcuDispatchOp::GridStrideLoop {
            extent,
            body: &body[..4],
        },
        PcuDispatchOp::Control(PcuDispatchControlOp::Return),
    ];
    let kernel = PcuDispatchKernelIr {
        id: fusion_pcu::PcuKernelId(0xf164),
        entry: PcuDispatchEntryPoint {
            name: "checked_float64_acceptance",
            logical_shape: [if grid { 1 } else { extent }, 1, 1],
        },
        bindings: &bindings,
        ports: &[],
        parameters: &[],
        ops: if grid { &loop_ops } else { &body },
        type_caps: PcuValueTypeCaps::FLOAT64,
        feature_caps: PcuDispatchFeatureCaps::READ_ONLY_RESOURCES
            .union(PcuDispatchFeatureCaps::MUTABLE_RESOURCES)
            .union(if range_policy == PcuRangePolicy::Clamp {
                PcuDispatchFeatureCaps::RANGE_CLAMP
            } else {
                PcuDispatchFeatureCaps::empty()
            }),
    };
    backend
        .prepare_dispatch(PcuDispatchSubmission {
            kernel: &kernel,
            shape: PcuInvocationShape::invocations(
                NonZeroU32::new(if grid { 1 } else { extent }).unwrap(),
            ),
        })
        .unwrap()
}

fn run_case(
    backend: &RocmOwnedDispatchBackend,
    prepared: &RocmPreparedDispatch,
    fault_word: &mut DeviceBuffer,
    left: f32,
    right: f32,
    expected: Result<u32, PcuExecutionFaultKind>,
) {
    let mut lhs = backend.allocate(8).unwrap();
    let mut rhs = backend.allocate(8).unwrap();
    let output = backend.allocate(8).unwrap();
    lhs.copy_from(&[1.0_f32.to_ne_bytes(), left.to_ne_bytes()].concat())
        .unwrap();
    rhs.copy_from(&[1.0_f32.to_ne_bytes(), right.to_ne_bytes()].concat())
        .unwrap();
    let bindings = [
        backend
            .binding(
                PcuBindingRef::new(0, 0),
                PcuBindingAccess::ReadOnly,
                PcuBindingType::Value(PcuValueType::f32()),
                lhs,
            )
            .unwrap(),
        backend
            .binding(
                PcuBindingRef::new(0, 1),
                PcuBindingAccess::ReadOnly,
                PcuBindingType::Value(PcuValueType::f32()),
                rhs,
            )
            .unwrap(),
        backend
            .binding(
                PcuBindingRef::new(0, 2),
                PcuBindingAccess::WriteOnly,
                PcuBindingType::Value(PcuValueType::f32()),
                output.clone(),
            )
            .unwrap(),
    ];
    let mut completion = prepared
        .submit_with_fault_word(&bindings, fault_word)
        .unwrap();
    let outcome = completion.wait().unwrap();
    match expected {
        Ok(bits) => {
            assert_eq!(outcome, PcuCompletionOutcome::Succeeded);
            let mut bytes = [0_u8; 8];
            output.copy_to(&mut bytes).unwrap();
            assert_eq!(u32::from_ne_bytes(bytes[4..].try_into().unwrap()), bits);
        }
        Err(kind) => assert_eq!(
            outcome,
            PcuCompletionOutcome::Fault(PcuExecutionFault {
                kind,
                invocation_id: 1,
                recovered: false,
            })
        ),
    }
}

fn run_case_f64(
    backend: &RocmOwnedDispatchBackend,
    prepared: &RocmPreparedDispatch,
    fault_word: &mut DeviceBuffer,
    left: f64,
    right: f64,
    expected: Result<u64, PcuExecutionFaultKind>,
) {
    let mut lhs = backend.allocate(16).unwrap();
    let mut rhs = backend.allocate(16).unwrap();
    let output = backend.allocate(16).unwrap();
    lhs.copy_from(&[1.0_f64.to_ne_bytes(), left.to_ne_bytes()].concat())
        .unwrap();
    rhs.copy_from(&[1.0_f64.to_ne_bytes(), right.to_ne_bytes()].concat())
        .unwrap();
    let bindings = [
        backend
            .binding(
                PcuBindingRef::new(0, 0),
                PcuBindingAccess::ReadOnly,
                PcuBindingType::Value(PcuValueType::f64()),
                lhs,
            )
            .unwrap(),
        backend
            .binding(
                PcuBindingRef::new(0, 1),
                PcuBindingAccess::ReadOnly,
                PcuBindingType::Value(PcuValueType::f64()),
                rhs,
            )
            .unwrap(),
        backend
            .binding(
                PcuBindingRef::new(0, 2),
                PcuBindingAccess::WriteOnly,
                PcuBindingType::Value(PcuValueType::f64()),
                output.clone(),
            )
            .unwrap(),
    ];
    let mut completion = prepared
        .submit_with_fault_word(&bindings, fault_word)
        .unwrap();
    let outcome = completion.wait().unwrap();
    match expected {
        Ok(bits) => {
            assert_eq!(outcome, PcuCompletionOutcome::Succeeded);
            let mut bytes = [0_u8; 16];
            output.copy_to(&mut bytes).unwrap();
            assert_eq!(u64::from_ne_bytes(bytes[8..].try_into().unwrap()), bits);
        }
        Err(kind) => assert_eq!(
            outcome,
            PcuCompletionOutcome::Fault(PcuExecutionFault {
                kind,
                invocation_id: 1,
                recovered: false,
            })
        ),
    }
}

fn run_clamped_case_f32(
    backend: &RocmOwnedDispatchBackend,
    prepared: &RocmPreparedDispatch,
    fault_word: &mut DeviceBuffer,
    lhs_values: [f32; 2],
    rhs_values: [f32; 2],
    expected_kind: PcuExecutionFaultKind,
    expected_value_bits: u32,
) {
    let mut lhs = backend.allocate(8).unwrap();
    let mut rhs = backend.allocate(8).unwrap();
    let output = backend.allocate(8).unwrap();
    lhs.copy_from(&[lhs_values[0].to_ne_bytes(), lhs_values[1].to_ne_bytes()].concat())
        .unwrap();
    rhs.copy_from(&[rhs_values[0].to_ne_bytes(), rhs_values[1].to_ne_bytes()].concat())
        .unwrap();
    let bindings = [
        backend
            .binding(
                PcuBindingRef::new(0, 0),
                PcuBindingAccess::ReadOnly,
                PcuBindingType::Value(PcuValueType::f32()),
                lhs,
            )
            .unwrap(),
        backend
            .binding(
                PcuBindingRef::new(0, 1),
                PcuBindingAccess::ReadOnly,
                PcuBindingType::Value(PcuValueType::f32()),
                rhs,
            )
            .unwrap(),
        backend
            .binding(
                PcuBindingRef::new(0, 2),
                PcuBindingAccess::WriteOnly,
                PcuBindingType::Value(PcuValueType::f32()),
                output.clone(),
            )
            .unwrap(),
    ];
    let mut completion = prepared
        .submit_with_fault_word(&bindings, fault_word)
        .unwrap();
    assert_eq!(
        completion.wait().unwrap(),
        PcuCompletionOutcome::Fault(PcuExecutionFault {
            kind: expected_kind,
            invocation_id: 1,
            recovered: true,
        })
    );
    let mut bytes = [0_u8; 8];
    output.copy_to(&mut bytes).unwrap();
    assert_eq!(
        u32::from_ne_bytes(bytes[..4].try_into().unwrap()),
        6.0_f32.to_bits()
    );
    assert_eq!(
        u32::from_ne_bytes(bytes[4..].try_into().unwrap()),
        expected_value_bits
    );
}

fn run_clamped_case_f64(
    backend: &RocmOwnedDispatchBackend,
    prepared: &RocmPreparedDispatch,
    fault_word: &mut DeviceBuffer,
    lhs_values: [f64; 2],
    rhs_values: [f64; 2],
    expected_kind: PcuExecutionFaultKind,
    expected_value_bits: u64,
) {
    let mut lhs = backend.allocate(16).unwrap();
    let mut rhs = backend.allocate(16).unwrap();
    let output = backend.allocate(16).unwrap();
    lhs.copy_from(&[lhs_values[0].to_ne_bytes(), lhs_values[1].to_ne_bytes()].concat())
        .unwrap();
    rhs.copy_from(&[rhs_values[0].to_ne_bytes(), rhs_values[1].to_ne_bytes()].concat())
        .unwrap();
    let bindings = [
        backend
            .binding(
                PcuBindingRef::new(0, 0),
                PcuBindingAccess::ReadOnly,
                PcuBindingType::Value(PcuValueType::f64()),
                lhs,
            )
            .unwrap(),
        backend
            .binding(
                PcuBindingRef::new(0, 1),
                PcuBindingAccess::ReadOnly,
                PcuBindingType::Value(PcuValueType::f64()),
                rhs,
            )
            .unwrap(),
        backend
            .binding(
                PcuBindingRef::new(0, 2),
                PcuBindingAccess::WriteOnly,
                PcuBindingType::Value(PcuValueType::f64()),
                output.clone(),
            )
            .unwrap(),
    ];
    let mut completion = prepared
        .submit_with_fault_word(&bindings, fault_word)
        .unwrap();
    assert_eq!(
        completion.wait().unwrap(),
        PcuCompletionOutcome::Fault(PcuExecutionFault {
            kind: expected_kind,
            invocation_id: 1,
            recovered: true,
        })
    );
    let mut bytes = [0_u8; 16];
    output.copy_to(&mut bytes).unwrap();
    assert_eq!(
        u64::from_ne_bytes(bytes[..8].try_into().unwrap()),
        6.0_f64.to_bits()
    );
    assert_eq!(
        u64::from_ne_bytes(bytes[8..].try_into().unwrap()),
        expected_value_bits
    );
}

#[test]
#[ignore = "requires a working ROCm device and HIPRTC"]
fn range_clamp_reports_recovery_and_stores_all_direct_and_grid_results() {
    let (_discovery, backend) = super::checked_integer_tests::selected_device();
    let mut fault_word = backend.allocate(8).unwrap();
    for grid in [false, true] {
        let f32_mul = prepare_with_range_policy(
            &backend,
            PcuDispatchFloatBinaryOp::Mul,
            PcuFloatUnderflowPolicy::IeeeAfterRounding,
            PcuRangePolicy::Clamp,
            grid,
            2,
        );
        run_clamped_case_f32(
            &backend,
            &f32_mul,
            &mut fault_word,
            [2.0, f32::MAX],
            [3.0, 2.0],
            PcuExecutionFaultKind::ArithmeticOverflow,
            f32::MAX.to_bits(),
        );
        let f32_div = prepare_with_range_policy(
            &backend,
            PcuDispatchFloatBinaryOp::Div,
            PcuFloatUnderflowPolicy::IeeeAfterRounding,
            PcuRangePolicy::Clamp,
            grid,
            2,
        );
        run_clamped_case_f32(
            &backend,
            &f32_div,
            &mut fault_word,
            [6.0, f32::from_bits(1)],
            [1.0, 2.0],
            PcuExecutionFaultKind::ArithmeticUnderflow,
            0,
        );

        let f64_mul = prepare_f64_with_range_policy(
            &backend,
            PcuDispatchFloatBinaryOp::Mul,
            PcuFloatUnderflowPolicy::IeeeAfterRounding,
            PcuRangePolicy::Clamp,
            grid,
            2,
        );
        run_clamped_case_f64(
            &backend,
            &f64_mul,
            &mut fault_word,
            [2.0, f64::MAX],
            [3.0, 2.0],
            PcuExecutionFaultKind::ArithmeticOverflow,
            f64::MAX.to_bits(),
        );
        let f64_div = prepare_f64_with_range_policy(
            &backend,
            PcuDispatchFloatBinaryOp::Div,
            PcuFloatUnderflowPolicy::IeeeAfterRounding,
            PcuRangePolicy::Clamp,
            grid,
            2,
        );
        run_clamped_case_f64(
            &backend,
            &f64_div,
            &mut fault_word,
            [6.0, f64::from_bits(1)],
            [1.0, 2.0],
            PcuExecutionFaultKind::ArithmeticUnderflow,
            0,
        );
    }
}

#[test]
#[ignore = "requires a working ROCm device and HIPRTC"]
#[allow(clippy::too_many_lines)] // Keep the policy/operation oracle matrix visible in one hardware fixture.
fn binary64_all_ops_match_policy_boundaries_and_report_terminal_faults() {
    let (_discovery, backend) = super::checked_integer_tests::selected_device();
    let mut fault = backend.allocate(8).unwrap();
    for grid in [false, true] {
        for policy in [
            PcuFloatUnderflowPolicy::IeeeAfterRounding,
            PcuFloatUnderflowPolicy::RejectSubnormalResult,
            PcuFloatUnderflowPolicy::AllowGradualUnderflow,
        ] {
            for (op, lhs, rhs, expected) in [
                (
                    PcuDispatchFloatBinaryOp::Add,
                    2.0,
                    3.0,
                    Ok(5.0_f64.to_bits()),
                ),
                (
                    PcuDispatchFloatBinaryOp::Sub,
                    2.0,
                    3.0,
                    Ok((-1.0_f64).to_bits()),
                ),
                (
                    PcuDispatchFloatBinaryOp::Mul,
                    -0.0,
                    2.0,
                    Ok((-0.0_f64).to_bits()),
                ),
                (
                    PcuDispatchFloatBinaryOp::Div,
                    7.5,
                    2.5,
                    Ok(3.0_f64.to_bits()),
                ),
                (
                    PcuDispatchFloatBinaryOp::Div,
                    1.0,
                    3.0,
                    Ok((1.0_f64 / 3.0).to_bits()),
                ),
            ] {
                let prepared = prepare_f64(&backend, op, policy, grid, 2);
                run_case_f64(&backend, &prepared, &mut fault, lhs, rhs, expected);
            }
            let mul = prepare_f64(&backend, PcuDispatchFloatBinaryOp::Mul, policy, grid, 2);
            let subnormal = if policy == PcuFloatUnderflowPolicy::RejectSubnormalResult {
                Err(PcuExecutionFaultKind::ArithmeticUnderflow)
            } else {
                Ok((f64::MIN_POSITIVE * 0.5).to_bits())
            };
            run_case_f64(
                &backend,
                &mul,
                &mut fault,
                f64::MIN_POSITIVE,
                0.5,
                subnormal,
            );
            let rounded_zero = if policy == PcuFloatUnderflowPolicy::AllowGradualUnderflow {
                Ok(0)
            } else {
                Err(PcuExecutionFaultKind::ArithmeticUnderflow)
            };
            run_case_f64(
                &backend,
                &mul,
                &mut fault,
                f64::from_bits(1),
                0.5,
                rounded_zero,
            );
            let rounded_min_normal = if policy == PcuFloatUnderflowPolicy::AllowGradualUnderflow {
                Ok(f64::MIN_POSITIVE.to_bits())
            } else {
                Err(PcuExecutionFaultKind::ArithmeticUnderflow)
            };
            run_case_f64(
                &backend,
                &mul,
                &mut fault,
                f64::MIN_POSITIVE,
                f64::from_bits(0x3fef_ffff_ffff_ffff),
                rounded_min_normal,
            );
            run_case_f64(
                &backend,
                &mul,
                &mut fault,
                f64::MAX,
                2.0,
                Err(PcuExecutionFaultKind::ArithmeticOverflow),
            );
            run_case_f64(
                &backend,
                &mul,
                &mut fault,
                f64::INFINITY,
                1.0,
                Err(PcuExecutionFaultKind::InvalidFloatingOperand),
            );
            let div = prepare_f64(&backend, PcuDispatchFloatBinaryOp::Div, policy, grid, 2);
            let exact_subnormal = if policy == PcuFloatUnderflowPolicy::RejectSubnormalResult {
                Err(PcuExecutionFaultKind::ArithmeticUnderflow)
            } else {
                Ok(1)
            };
            run_case_f64(
                &backend,
                &div,
                &mut fault,
                f64::from_bits(1),
                1.0,
                exact_subnormal,
            );
            let inexact_tiny = if policy == PcuFloatUnderflowPolicy::AllowGradualUnderflow {
                Ok(0)
            } else {
                Err(PcuExecutionFaultKind::ArithmeticUnderflow)
            };
            // Half of the minimum subnormal is an exact halfway case and rounds to even zero.
            run_case_f64(
                &backend,
                &div,
                &mut fault,
                f64::from_bits(1),
                2.0,
                inexact_tiny,
            );
            let half_subnormal = if policy == PcuFloatUnderflowPolicy::AllowGradualUnderflow {
                Ok((f64::MIN_POSITIVE / 2.0).to_bits())
            } else if policy == PcuFloatUnderflowPolicy::RejectSubnormalResult {
                Err(PcuExecutionFaultKind::ArithmeticUnderflow)
            } else {
                Ok((f64::MIN_POSITIVE / 2.0).to_bits())
            };
            run_case_f64(
                &backend,
                &div,
                &mut fault,
                f64::MIN_POSITIVE,
                2.0,
                half_subnormal,
            );
            run_case_f64(&backend, &div, &mut fault, -0.0, 2.0, Ok(1u64 << 63));
            run_case_f64(
                &backend,
                &div,
                &mut fault,
                f64::MAX,
                f64::from_bits(1),
                Err(PcuExecutionFaultKind::ArithmeticOverflow),
            );
            run_case_f64(
                &backend,
                &div,
                &mut fault,
                0.0,
                -0.0,
                Err(PcuExecutionFaultKind::DivideByZero),
            );
            run_case_f64(
                &backend,
                &div,
                &mut fault,
                f64::INFINITY,
                0.0,
                Err(PcuExecutionFaultKind::InvalidFloatingOperand),
            );
            run_case_f64(
                &backend,
                &div,
                &mut fault,
                0.0,
                f64::NAN,
                Err(PcuExecutionFaultKind::InvalidFloatingOperand),
            );
            let asymmetric = f64::from_bits(0x000f_ffff_ffff_ffff);
            let asymmetric_expected = asymmetric
                .pcu_checked_div_with_policy(3.0, policy)
                .map(f64::to_bits);
            run_case_f64(
                &backend,
                &div,
                &mut fault,
                asymmetric,
                3.0,
                asymmetric_expected,
            );
        }
        let add = prepare_f64(
            &backend,
            PcuDispatchFloatBinaryOp::Add,
            PcuFloatUnderflowPolicy::AllowGradualUnderflow,
            grid,
            2,
        );
        run_case_f64(&backend, &add, &mut fault, -2.0, 2.0, Ok(0));
    }
}

#[test]
#[ignore = "requires a working ROCm device and HIPRTC"]
#[allow(clippy::too_many_lines)] // Keep generated inputs, GPU dispatch, and checked-core comparison visible together.
fn binary64_randomized_result_bits_match_checked_core_oracle() {
    const COUNT: u32 = 4096;
    let (_discovery, backend) = super::checked_integer_tests::selected_device();
    let mut seed = 0x749b_6153_u32;
    let mut next = || {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        let fraction = (u64::from(seed) * 0x0000_0001_0000_0001) & 0x000f_ffff_ffff_ffff;
        // Keep this bitwise-result cohort finite; separate boundary fixtures assert faults.
        let exponent = u64::from((seed >> 23) % 1024 + 511);
        f64::from_bits((u64::from(seed & 1) << 63) | (exponent << 52) | fraction)
    };
    let left: Vec<_> = (0..COUNT).map(|_| next()).collect();
    let right: Vec<_> = (0..COUNT).map(|_| next()).collect();
    let right: Vec<_> = left
        .iter()
        .zip(right)
        .map(|(lhs, rhs)| {
            if rhs == 0.0 || !(lhs / rhs).is_finite() {
                1.0
            } else {
                rhs
            }
        })
        .collect();
    let mut lhs = backend.allocate(left.len() * 8).unwrap();
    let mut rhs = backend.allocate(right.len() * 8).unwrap();
    lhs.copy_from(
        &left
            .iter()
            .flat_map(|value| value.to_ne_bytes())
            .collect::<Vec<_>>(),
    )
    .unwrap();
    rhs.copy_from(
        &right
            .iter()
            .flat_map(|value| value.to_ne_bytes())
            .collect::<Vec<_>>(),
    )
    .unwrap();
    let output = backend.allocate(left.len() * 8).unwrap();
    let bindings = [
        backend
            .binding(
                PcuBindingRef::new(0, 0),
                PcuBindingAccess::ReadOnly,
                PcuBindingType::Value(PcuValueType::f64()),
                lhs,
            )
            .unwrap(),
        backend
            .binding(
                PcuBindingRef::new(0, 1),
                PcuBindingAccess::ReadOnly,
                PcuBindingType::Value(PcuValueType::f64()),
                rhs,
            )
            .unwrap(),
        backend
            .binding(
                PcuBindingRef::new(0, 2),
                PcuBindingAccess::WriteOnly,
                PcuBindingType::Value(PcuValueType::f64()),
                output.clone(),
            )
            .unwrap(),
    ];
    let mut fault = backend.allocate(8).unwrap();
    for grid in [false, true] {
        for op in [
            PcuDispatchFloatBinaryOp::Add,
            PcuDispatchFloatBinaryOp::Sub,
            PcuDispatchFloatBinaryOp::Mul,
            PcuDispatchFloatBinaryOp::Div,
        ] {
            let prepared = prepare_f64(
                &backend,
                op,
                PcuFloatUnderflowPolicy::AllowGradualUnderflow,
                grid,
                COUNT,
            );
            let mut completion = prepared
                .submit_with_fault_word(&bindings, &mut fault)
                .unwrap();
            assert_eq!(completion.wait().unwrap(), PcuCompletionOutcome::Succeeded);
            let mut bytes = vec![0_u8; left.len() * 8];
            output.copy_to(&mut bytes).unwrap();
            for ((lhs, rhs), chunk) in left.iter().zip(&right).zip(bytes.as_chunks::<8>().0) {
                let expected = match op {
                    PcuDispatchFloatBinaryOp::Add => lhs
                        .pcu_checked_add_with_policy(
                            *rhs,
                            PcuFloatUnderflowPolicy::AllowGradualUnderflow,
                        )
                        .unwrap(),
                    PcuDispatchFloatBinaryOp::Sub => lhs
                        .pcu_checked_sub_with_policy(
                            *rhs,
                            PcuFloatUnderflowPolicy::AllowGradualUnderflow,
                        )
                        .unwrap(),
                    PcuDispatchFloatBinaryOp::Mul => lhs
                        .pcu_checked_mul_with_policy(
                            *rhs,
                            PcuFloatUnderflowPolicy::AllowGradualUnderflow,
                        )
                        .unwrap(),
                    PcuDispatchFloatBinaryOp::Div => lhs
                        .pcu_checked_div_with_policy(
                            *rhs,
                            PcuFloatUnderflowPolicy::AllowGradualUnderflow,
                        )
                        .unwrap(),
                };
                assert_eq!(
                    u64::from_ne_bytes(*chunk),
                    expected.to_bits(),
                    "{lhs:?} {op:?} {rhs:?}"
                );
            }
        }
    }
}

#[test]
#[ignore = "requires a working ROCm device and HIPRTC"]
#[allow(clippy::too_many_lines)] // Keep the policy/operation oracle matrix visible in one hardware fixture.
fn binary32_ieee_policy_faults_and_exact_bits_are_terminal_and_retryable() {
    let (_discovery, backend) = super::checked_integer_tests::selected_device();
    let mut fault = backend.allocate(8).unwrap();
    for grid in [false, true] {
        for policy in [
            PcuFloatUnderflowPolicy::IeeeAfterRounding,
            PcuFloatUnderflowPolicy::RejectSubnormalResult,
            PcuFloatUnderflowPolicy::AllowGradualUnderflow,
        ] {
            let mul = prepare(&backend, PcuDispatchFloatBinaryOp::Mul, policy, grid, 2);
            let exact = if policy == PcuFloatUnderflowPolicy::RejectSubnormalResult {
                Err(PcuExecutionFaultKind::ArithmeticUnderflow)
            } else {
                Ok(0x0040_0000)
            };
            run_case(&backend, &mul, &mut fault, f32::MIN_POSITIVE, 0.5, exact);
            let tiny = if policy == PcuFloatUnderflowPolicy::AllowGradualUnderflow {
                Ok(0)
            } else {
                Err(PcuExecutionFaultKind::ArithmeticUnderflow)
            };
            run_case(&backend, &mul, &mut fault, f32::from_bits(1), 0.5, tiny);
            let boundary = if policy == PcuFloatUnderflowPolicy::AllowGradualUnderflow {
                Ok(f32::MIN_POSITIVE.to_bits())
            } else {
                Err(PcuExecutionFaultKind::ArithmeticUnderflow)
            };
            run_case(
                &backend,
                &mul,
                &mut fault,
                f32::MIN_POSITIVE,
                f32::from_bits(0x3f7f_ffff),
                boundary,
            );
            run_case(
                &backend,
                &mul,
                &mut fault,
                f32::MAX,
                2.0,
                Err(PcuExecutionFaultKind::ArithmeticOverflow),
            );
            run_case(
                &backend,
                &mul,
                &mut fault,
                f32::INFINITY,
                0.0,
                Err(PcuExecutionFaultKind::InvalidFloatingOperand),
            );
            run_case(&backend, &mul, &mut fault, -0.0, 2.0, Ok(0x8000_0000));
            run_case(&backend, &mul, &mut fault, 2.0, 3.0, Ok(6.0_f32.to_bits()));
            let add = prepare(&backend, PcuDispatchFloatBinaryOp::Add, policy, grid, 2);
            run_case(
                &backend,
                &add,
                &mut fault,
                f32::MAX,
                f32::MAX,
                Err(PcuExecutionFaultKind::ArithmeticOverflow),
            );
            run_case(
                &backend,
                &add,
                &mut fault,
                f32::from_bits(0x7fc0_1234),
                1.0,
                Err(PcuExecutionFaultKind::InvalidFloatingOperand),
            );
            run_case(&backend, &add, &mut fault, 2.0, 3.0, Ok(5.0_f32.to_bits()));
            run_case(&backend, &add, &mut fault, -2.0, 2.0, Ok(0));
            let sub = prepare(&backend, PcuDispatchFloatBinaryOp::Sub, policy, grid, 2);
            run_case(
                &backend,
                &sub,
                &mut fault,
                -f32::MAX,
                f32::MAX,
                Err(PcuExecutionFaultKind::ArithmeticOverflow),
            );
            run_case(&backend, &sub, &mut fault, -0.0, 0.0, Ok(0x8000_0000));
            run_case(&backend, &sub, &mut fault, -1.0, -1.0, Ok(0));
            run_case(
                &backend,
                &sub,
                &mut fault,
                2.0,
                3.0,
                Ok((-1.0_f32).to_bits()),
            );
            let div = prepare(&backend, PcuDispatchFloatBinaryOp::Div, policy, grid, 2);
            for (lhs, rhs) in [(7.5_f32, 2.5_f32), (1.0, 3.0), (1.0, 10.0)] {
                let expected = lhs
                    .pcu_checked_div_with_policy(rhs, policy)
                    .map(f32::to_bits);
                run_case(&backend, &div, &mut fault, lhs, rhs, expected);
            }
            for (lhs, rhs) in [
                (f32::from_bits(1), 1.0),
                (f32::from_bits(1), 2.0),
                (f32::MIN_POSITIVE, 2.0),
                (f32::from_bits(0x007f_ffff), 3.0),
            ] {
                let expected = lhs
                    .pcu_checked_div_with_policy(rhs, policy)
                    .map(f32::to_bits);
                run_case(&backend, &div, &mut fault, lhs, rhs, expected);
            }
            run_case(&backend, &div, &mut fault, -0.0, 2.0, Ok(0x8000_0000));
            run_case(
                &backend,
                &div,
                &mut fault,
                0.0,
                -0.0,
                Err(PcuExecutionFaultKind::DivideByZero),
            );
            run_case(
                &backend,
                &div,
                &mut fault,
                f32::INFINITY,
                0.0,
                Err(PcuExecutionFaultKind::InvalidFloatingOperand),
            );
            run_case(
                &backend,
                &div,
                &mut fault,
                0.0,
                f32::NAN,
                Err(PcuExecutionFaultKind::InvalidFloatingOperand),
            );
            run_case(
                &backend,
                &div,
                &mut fault,
                f32::MAX,
                f32::from_bits(1),
                Err(PcuExecutionFaultKind::ArithmeticOverflow),
            );
        }
    }
}

#[test]
#[ignore = "requires a working ROCm device and HIPRTC"]
#[allow(clippy::too_many_lines)] // Keep the policy/operation oracle matrix visible in one hardware fixture.
fn binary32_randomized_result_bits_match_host_gradual_rounding() {
    const COUNT: u32 = 4096;
    let (_discovery, backend) = super::checked_integer_tests::selected_device();
    let mut seed = 0x749b_6153_u32;
    let mut next = || {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        // Bound magnitudes so no operation overflows; preserve signs and subnormals.
        f32::from_bits((seed & 0x807f_ffff) | (((seed >> 23) % 181) << 23))
    };
    let left: Vec<_> = (0..COUNT).map(|_| next()).collect();
    let right: Vec<_> = (0..COUNT).map(|_| next()).collect();
    let right: Vec<_> = left
        .iter()
        .zip(right)
        .map(|(lhs, rhs)| {
            if rhs == 0.0 || !(lhs / rhs).is_finite() {
                1.0
            } else {
                rhs
            }
        })
        .collect();
    let mut lhs = backend.allocate(left.len() * 4).unwrap();
    let mut rhs = backend.allocate(right.len() * 4).unwrap();
    lhs.copy_from(
        &left
            .iter()
            .flat_map(|v| v.to_ne_bytes())
            .collect::<Vec<_>>(),
    )
    .unwrap();
    rhs.copy_from(
        &right
            .iter()
            .flat_map(|v| v.to_ne_bytes())
            .collect::<Vec<_>>(),
    )
    .unwrap();
    let output = backend.allocate(left.len() * 4).unwrap();
    let bindings = [
        backend
            .binding(
                PcuBindingRef::new(0, 0),
                PcuBindingAccess::ReadOnly,
                PcuBindingType::Value(PcuValueType::f32()),
                lhs,
            )
            .unwrap(),
        backend
            .binding(
                PcuBindingRef::new(0, 1),
                PcuBindingAccess::ReadOnly,
                PcuBindingType::Value(PcuValueType::f32()),
                rhs,
            )
            .unwrap(),
        backend
            .binding(
                PcuBindingRef::new(0, 2),
                PcuBindingAccess::WriteOnly,
                PcuBindingType::Value(PcuValueType::f32()),
                output.clone(),
            )
            .unwrap(),
    ];
    let mut fault = backend.allocate(8).unwrap();
    for grid in [false, true] {
        for op in [
            PcuDispatchFloatBinaryOp::Add,
            PcuDispatchFloatBinaryOp::Sub,
            PcuDispatchFloatBinaryOp::Mul,
            PcuDispatchFloatBinaryOp::Div,
        ] {
            let prepared = prepare(
                &backend,
                op,
                PcuFloatUnderflowPolicy::AllowGradualUnderflow,
                grid,
                COUNT,
            );
            let mut completion = prepared
                .submit_with_fault_word(&bindings, &mut fault)
                .unwrap();
            assert_eq!(completion.wait().unwrap(), PcuCompletionOutcome::Succeeded);
            let mut bytes = vec![0_u8; left.len() * 4];
            output.copy_to(&mut bytes).unwrap();
            for ((lhs, rhs), chunk) in left.iter().zip(&right).zip(bytes.as_chunks::<4>().0) {
                let expected = match op {
                    PcuDispatchFloatBinaryOp::Add => lhs + rhs,
                    PcuDispatchFloatBinaryOp::Sub => lhs - rhs,
                    PcuDispatchFloatBinaryOp::Mul => lhs * rhs,
                    PcuDispatchFloatBinaryOp::Div => lhs
                        .pcu_checked_div_with_policy(
                            *rhs,
                            PcuFloatUnderflowPolicy::AllowGradualUnderflow,
                        )
                        .unwrap(),
                };
                assert_eq!(
                    u32::from_ne_bytes(*chunk),
                    expected.to_bits(),
                    "{lhs:?} {op:?} {rhs:?}"
                );
            }
        }
    }
}
