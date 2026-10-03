#[rustfmt::skip]
use super::{
    PcuCheckedFloatReference,
    PcuCheckedFloatReferenceError,
    PcuCheckedF32Reference,
    PcuCheckedF64Reference,
};
#[rustfmt::skip]
use fusion_pcu::{
    PcuBinding,
    PcuBindingAccess,
    PcuBindingRef,
    PcuBindingStorageClass,
    PcuClampedFloat,
    PcuCheckedFloat,
    PcuDispatchAluOp,
    PcuDispatchCheckedFloatConversion,
    PcuDispatchControlOp,
    PcuDispatchDataOp,
    PcuDispatchEntryPoint,
    PcuDispatchFeatureCaps,
    PcuDispatchFloatBinaryOp,
    PcuDispatchFloatUnaryOp,
    PcuDispatchIndex,
    PcuDispatchKernelIr,
    PcuDispatchOp,
    PcuDispatchSubmission,
    PcuDispatchValueId,
    PcuExecutionFault,
    PcuExecutionFaultKind,
    PcuFloatUnderflowPolicy,
    PcuHostScalarBinding,
    PcuHostScalarSlice,
    PcuInvocationParameters,
    PcuInvocationShape,
    PcuKernelId,
    PcuRangePolicy,
    PcuSynchronousHostDispatchBackend,
    PcuValueType,
    PcuValueTypeCaps,
};
#[rustfmt::skip]
use crate::{
    PcuCpuTypedBinding,
    PcuCpuTypedSlice,
};
use core::num::NonZeroU32;
use std::boxed::Box;
use std::vec;
use std::vec::Vec;

const INPUT: PcuBindingRef = PcuBindingRef::new(0, 0);
const RHS: PcuBindingRef = PcuBindingRef::new(0, 1);
const OUTPUT: PcuBindingRef = PcuBindingRef::new(0, 2);

fn kernel(
    width: PcuValueType,
    op: PcuDispatchFloatBinaryOp,
    underflow_policy: PcuFloatUnderflowPolicy,
    range_policy: PcuRangePolicy,
    grid: Option<u32>,
    launch: u32,
) -> PcuDispatchKernelIr<'static> {
    let bindings = Box::leak(Box::new([
        PcuBinding::value(
            None,
            0,
            0,
            PcuBindingStorageClass::Storage,
            PcuBindingAccess::ReadOnly,
            width,
        ),
        PcuBinding::value(
            None,
            0,
            1,
            PcuBindingStorageClass::Storage,
            PcuBindingAccess::ReadOnly,
            width,
        ),
        PcuBinding::value(
            None,
            0,
            2,
            PcuBindingStorageClass::Storage,
            PcuBindingAccess::WriteOnly,
            width,
        ),
    ]));
    let index = if grid.is_some() {
        PcuDispatchIndex::GridStrideId
    } else {
        PcuDispatchIndex::InvocationId
    };
    let body = Box::leak(Box::new([
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
            result: PcuDispatchValueId(1),
            binding: INPUT,
            index,
        }),
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
            result: PcuDispatchValueId(2),
            binding: RHS,
            index,
        }),
        PcuDispatchOp::Data(PcuDispatchDataOp::CheckedFloatBinary {
            value_type: width,
            op,
            underflow_policy,
            range_policy,
            result: PcuDispatchValueId(3),
            lhs: PcuDispatchValueId(1),
            rhs: PcuDispatchValueId(2),
        }),
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
            binding: OUTPUT,
            index,
            value: PcuDispatchValueId(3),
        }),
    ]));
    let ops = grid.map_or_else(
        || {
            let mut ops = body.to_vec();
            ops.push(PcuDispatchOp::Control(PcuDispatchControlOp::Return));
            ops
        },
        |extent| {
            vec![
                PcuDispatchOp::GridStrideLoop { extent, body },
                PcuDispatchOp::Control(PcuDispatchControlOp::Return),
            ]
        },
    );
    PcuDispatchKernelIr {
        numerical_requirements: PcuDispatchKernelIr::DEFAULT_REQUIREMENTS,
        id: PcuKernelId(501),
        entry: PcuDispatchEntryPoint {
            name: "checked_cpu",
            logical_shape: [launch, 1, 1],
        },
        bindings,
        ports: &[],
        parameters: &[],
        ops: Box::leak(ops.into_boxed_slice()),
        type_caps: PcuValueTypeCaps::FLOAT32 | PcuValueTypeCaps::FLOAT64,
        feature_caps: PcuDispatchFeatureCaps::default(),
    }
}
fn submission<'a>(kernel: &'a PcuDispatchKernelIr<'a>) -> PcuDispatchSubmission<'a> {
    PcuDispatchSubmission {
        kernel,
        shape: PcuInvocationShape::invocations(
            NonZeroU32::new(kernel.entry.logical_shape[0]).unwrap(),
        ),
    }
}
fn run_f32(
    kernel: &PcuDispatchKernelIr<'_>,
    lhs: &[f32],
    rhs: &[f32],
) -> (Vec<f32>, Result<(), PcuCheckedFloatReferenceError>) {
    let mut output = vec![19.0; lhs.len()];
    let result = PcuCheckedF32Reference.run_host_direct(
        submission(kernel),
        &mut [
            PcuHostScalarBinding {
                target: INPUT,
                slice: PcuHostScalarSlice::Read(lhs),
            },
            PcuHostScalarBinding {
                target: RHS,
                slice: PcuHostScalarSlice::Read(rhs),
            },
            PcuHostScalarBinding {
                target: OUTPUT,
                slice: PcuHostScalarSlice::ReadWrite(&mut output),
            },
        ],
        PcuInvocationParameters::empty(),
    );
    (output, result)
}
fn fault(
    kind: PcuExecutionFaultKind,
    invocation_id: u64,
    recovered: bool,
) -> Result<(), PcuCheckedFloatReferenceError> {
    Err(PcuCheckedFloatReferenceError::Fault(PcuExecutionFault {
        kind,
        invocation_id,
        recovered,
    }))
}

#[test]
fn checked_arithmetic_bits_and_signed_zero_both_widths_direct_and_grid() {
    let lhs = [1.25_f32, -0.0, 7.0, -3.5, f32::MIN_POSITIVE];
    let rhs = [2.0_f32, 2.0, 3.0, 4.0, 0.5];
    for grid in [None, Some(5)] {
        let count = if grid.is_some() { 5 } else { 2 };
        for op in [
            PcuDispatchFloatBinaryOp::Add,
            PcuDispatchFloatBinaryOp::Sub,
            PcuDispatchFloatBinaryOp::Mul,
            PcuDispatchFloatBinaryOp::Div,
        ] {
            let program = kernel(
                PcuValueType::f32(),
                op,
                PcuFloatUnderflowPolicy::AllowGradualUnderflow,
                PcuRangePolicy::Reject,
                grid,
                2,
            );
            let (actual, result) = run_f32(&program, &lhs[..count], &rhs[..count]);
            result.unwrap();
            for i in 0..count {
                let expected = match op {
                    PcuDispatchFloatBinaryOp::Add => lhs[i].pcu_checked_add(rhs[i]),
                    PcuDispatchFloatBinaryOp::Sub => lhs[i].pcu_checked_sub(rhs[i]),
                    PcuDispatchFloatBinaryOp::Mul => lhs[i].pcu_checked_mul(rhs[i]),
                    PcuDispatchFloatBinaryOp::Div => lhs[i].pcu_checked_div(rhs[i]),
                }
                .unwrap();
                assert_eq!(actual[i].to_bits(), expected.to_bits());
            }
            let program = kernel(
                PcuValueType::f64(),
                op,
                PcuFloatUnderflowPolicy::AllowGradualUnderflow,
                PcuRangePolicy::Reject,
                grid,
                2,
            );
            let lhs64: Vec<_> = lhs[..count].iter().map(|&x| f64::from(x)).collect();
            let rhs64: Vec<_> = rhs[..count].iter().map(|&x| f64::from(x)).collect();
            let mut output = vec![19.0_f64; count];
            PcuCheckedF64Reference
                .run_host_direct(
                    submission(&program),
                    &mut [
                        PcuHostScalarBinding {
                            target: INPUT,
                            slice: PcuHostScalarSlice::Read(&lhs64),
                        },
                        PcuHostScalarBinding {
                            target: RHS,
                            slice: PcuHostScalarSlice::Read(&rhs64),
                        },
                        PcuHostScalarBinding {
                            target: OUTPUT,
                            slice: PcuHostScalarSlice::ReadWrite(&mut output),
                        },
                    ],
                    PcuInvocationParameters::empty(),
                )
                .unwrap();
            for i in 0..count {
                let expected = match op {
                    PcuDispatchFloatBinaryOp::Add => lhs64[i].pcu_checked_add(rhs64[i]),
                    PcuDispatchFloatBinaryOp::Sub => lhs64[i].pcu_checked_sub(rhs64[i]),
                    PcuDispatchFloatBinaryOp::Mul => lhs64[i].pcu_checked_mul(rhs64[i]),
                    PcuDispatchFloatBinaryOp::Div => lhs64[i].pcu_checked_div(rhs64[i]),
                }
                .unwrap();
                assert_eq!(output[i].to_bits(), expected.to_bits());
            }
        }
    }
    let program = kernel(
        PcuValueType::f32(),
        PcuDispatchFloatBinaryOp::Mul,
        PcuFloatUnderflowPolicy::AllowGradualUnderflow,
        PcuRangePolicy::Reject,
        None,
        2,
    );
    assert_eq!(
        run_f32(&program, &[-0.0, 0.0], &[2.0, -2.0])
            .0
            .iter()
            .map(|x| x.to_bits())
            .collect::<Vec<_>>(),
        vec![0x8000_0000; 2]
    );
}

#[test]
fn strict_fault_discard_clamp_complete_and_retry_are_explicit() {
    let mut program = kernel(
        PcuValueType::f32(),
        PcuDispatchFloatBinaryOp::Mul,
        PcuFloatUnderflowPolicy::AllowGradualUnderflow,
        PcuRangePolicy::Reject,
        Some(5),
        2,
    );
    let lhs = [1.0, 2.0, f32::MAX, 4.0, 5.0];
    let rhs = [2.0; 5];
    let (output, result) = run_f32(&program, &lhs, &rhs);
    assert_eq!(
        result,
        fault(PcuExecutionFaultKind::ArithmeticOverflow, 2, false)
    );
    assert_eq!(output, [2.0, 4.0, 19.0, 19.0, 19.0]); // Failed output is not rolled back.
    program = kernel(
        PcuValueType::f32(),
        PcuDispatchFloatBinaryOp::Mul,
        PcuFloatUnderflowPolicy::AllowGradualUnderflow,
        PcuRangePolicy::Clamp,
        Some(5),
        2,
    );
    let (output, result) = run_f32(&program, &lhs, &rhs);
    assert_eq!(
        result,
        fault(PcuExecutionFaultKind::ArithmeticOverflow, 2, true)
    );
    assert_eq!(output, [2.0, 4.0, f32::MAX, 8.0, 10.0]);
    let (output, result) = run_f32(&program, &[1.0; 5], &rhs);
    result.unwrap();
    assert_eq!(output, [2.0; 5]);
    // Lane-major traversal would report invocation 4 before 3. Logical ordering must win.
    let lhs = [1.0, 2.0, f32::MAX, f32::NAN, f32::NAN];
    assert_eq!(
        run_f32(&program, &lhs, &rhs).1,
        fault(PcuExecutionFaultKind::InvalidFloatingOperand, 3, false)
    );
}

#[test]
fn underflow_uses_ieee_tininess_and_core_recovery_payloads() {
    // Exact subnormal: IEEE tininess accepts it, while RejectSubnormal rejects it.
    for (policy, expected) in [
        (PcuFloatUnderflowPolicy::AllowGradualUnderflow, Ok(())),
        (PcuFloatUnderflowPolicy::IeeeAfterRounding, Ok(())),
        (
            PcuFloatUnderflowPolicy::RejectSubnormalResult,
            fault(PcuExecutionFaultKind::ArithmeticUnderflow, 0, false),
        ),
    ] {
        let program = kernel(
            PcuValueType::f32(),
            PcuDispatchFloatBinaryOp::Mul,
            policy,
            PcuRangePolicy::Reject,
            None,
            1,
        );
        let (output, result) = run_f32(&program, &[f32::MIN_POSITIVE], &[0.5]);
        assert_eq!(result, expected);
        if result.is_ok() {
            assert_eq!(output[0].to_bits(), 0x0040_0000);
        }
    }
    let program = kernel(
        PcuValueType::f32(),
        PcuDispatchFloatBinaryOp::Div,
        PcuFloatUnderflowPolicy::IeeeAfterRounding,
        PcuRangePolicy::Clamp,
        None,
        1,
    );
    let (output, result) = run_f32(&program, &[-f32::from_bits(1)], &[2.0]);
    assert_eq!(
        result,
        fault(PcuExecutionFaultKind::ArithmeticUnderflow, 0, true)
    );
    assert_eq!(output[0].to_bits(), 0x8000_0000);
    let expected = (-f32::from_bits(1))
        .pcu_clamped_div_with_policy(2.0, PcuFloatUnderflowPolicy::IeeeAfterRounding)
        .unwrap_err();
    assert_eq!(expected.kind(), PcuExecutionFaultKind::ArithmeticUnderflow);
}

#[test]
fn unary_relu_canonicalizes_zeros_and_rejects_nonfinite() {
    let program = kernel(
        PcuValueType::f32(),
        PcuDispatchFloatBinaryOp::Add,
        PcuFloatUnderflowPolicy::AllowGradualUnderflow,
        PcuRangePolicy::Reject,
        None,
        5,
    );
    let mut ops = program.ops.to_vec();
    ops[2] = PcuDispatchOp::Data(PcuDispatchDataOp::CheckedFloatUnary {
        value_type: PcuValueType::f32(),
        op: PcuDispatchFloatUnaryOp::Relu,
        underflow_policy: PcuFloatUnderflowPolicy::AllowGradualUnderflow,
        range_policy: PcuRangePolicy::Reject,
        result: PcuDispatchValueId(3),
        value: PcuDispatchValueId(1),
    });
    let program = PcuDispatchKernelIr {
        ops: &ops,
        ..program
    };
    let (output, result) = run_f32(
        &program,
        &[-0.0, 0.0, -2.0, f32::from_bits(1), 3.0],
        &[0.0; 5],
    );
    result.unwrap();
    assert_eq!(
        output.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        [0, 0, 0, 1, 3.0_f32.to_bits()]
    );
    assert_eq!(
        run_f32(&program, &[f32::INFINITY; 5], &[0.0; 5]).1,
        fault(PcuExecutionFaultKind::InvalidFloatingOperand, 0, false)
    );
}

#[test]
#[allow(clippy::too_many_lines)] // Keep the complete mixed-width fixture next to both policy outcomes.
fn mixed_typed_ssa_widen_arithmetic_narrow_grid_and_clamp() {
    let bindings = [
        PcuBinding::scalar::<f32>(
            None,
            0,
            0,
            PcuBindingStorageClass::Storage,
            PcuBindingAccess::ReadOnly,
        ),
        PcuBinding::scalar::<f64>(
            None,
            0,
            1,
            PcuBindingStorageClass::Storage,
            PcuBindingAccess::ReadOnly,
        ),
        PcuBinding::scalar::<f32>(
            None,
            0,
            2,
            PcuBindingStorageClass::Storage,
            PcuBindingAccess::WriteOnly,
        ),
    ];
    for range_policy in [PcuRangePolicy::Reject, PcuRangePolicy::Clamp] {
        let body = [
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                binding: INPUT,
                result: PcuDispatchValueId(1),
                index: PcuDispatchIndex::GridStrideId,
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::CheckedFloatConvert {
                conversion: PcuDispatchCheckedFloatConversion::F32ToF64,
                result: PcuDispatchValueId(2),
                value: PcuDispatchValueId(1),
                underflow_policy: PcuFloatUnderflowPolicy::AllowGradualUnderflow,
                range_policy,
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                binding: RHS,
                result: PcuDispatchValueId(3),
                index: PcuDispatchIndex::BindingElementZero,
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::CheckedFloatBinary {
                value_type: PcuValueType::f64(),
                op: PcuDispatchFloatBinaryOp::Mul,
                underflow_policy: PcuFloatUnderflowPolicy::AllowGradualUnderflow,
                range_policy,
                result: PcuDispatchValueId(4),
                lhs: PcuDispatchValueId(2),
                rhs: PcuDispatchValueId(3),
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::CheckedFloatConvert {
                conversion: PcuDispatchCheckedFloatConversion::F64ToF32,
                result: PcuDispatchValueId(5),
                value: PcuDispatchValueId(4),
                underflow_policy: PcuFloatUnderflowPolicy::AllowGradualUnderflow,
                range_policy,
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                binding: OUTPUT,
                index: PcuDispatchIndex::GridStrideId,
                value: PcuDispatchValueId(5),
            }),
        ];
        let ops = [
            PcuDispatchOp::GridStrideLoop {
                extent: 4,
                body: &body,
            },
            PcuDispatchOp::Control(PcuDispatchControlOp::Return),
        ];
        let program = PcuDispatchKernelIr {
            numerical_requirements: PcuDispatchKernelIr::DEFAULT_REQUIREMENTS,
            id: PcuKernelId(502),
            entry: PcuDispatchEntryPoint {
                name: "mixed",
                logical_shape: [2, 1, 1],
            },
            bindings: &bindings,
            ports: &[],
            parameters: &[],
            ops: &ops,
            type_caps: PcuValueTypeCaps::FLOAT32 | PcuValueTypeCaps::FLOAT64,
            feature_caps: PcuDispatchFeatureCaps::default(),
        };
        let lhs = [-0.0, 1.25, f32::MAX, 3.0];
        let rhs = [2.0_f64];
        let mut output = [19.0; 4];
        let result = PcuCheckedFloatReference.run_host_direct(
            submission(&program),
            &mut [
                PcuCpuTypedBinding {
                    target: INPUT,
                    slice: PcuCpuTypedSlice::ReadF32(&lhs),
                },
                PcuCpuTypedBinding {
                    target: RHS,
                    slice: PcuCpuTypedSlice::ReadF64(&rhs),
                },
                PcuCpuTypedBinding {
                    target: OUTPUT,
                    slice: PcuCpuTypedSlice::ReadWriteF32(&mut output),
                },
            ],
        );
        assert_eq!(
            result,
            fault(
                PcuExecutionFaultKind::ArithmeticOverflow,
                2,
                range_policy == PcuRangePolicy::Clamp
            )
        );
        assert_eq!(output[0].to_bits(), 0x8000_0000);
        assert_eq!(output[1].to_bits(), 2.5_f32.to_bits());
        if range_policy == PcuRangePolicy::Clamp {
            assert_eq!(output[2..], [f32::MAX, 6.0]);
        } else {
            assert_eq!(output[2..], [19.0, 19.0]);
        }
    }
}

#[test]
fn raw_alu_and_out_of_range_ssa_reject_before_any_write() {
    let program = kernel(
        PcuValueType::f32(),
        PcuDispatchFloatBinaryOp::Add,
        PcuFloatUnderflowPolicy::AllowGradualUnderflow,
        PcuRangePolicy::Reject,
        None,
        1,
    );
    let mut ops = program.ops.to_vec();
    ops[2] = PcuDispatchOp::Data(PcuDispatchDataOp::Alu {
        value_type: PcuValueType::f32(),
        op: PcuDispatchAluOp::Add,
        result: PcuDispatchValueId(3),
        lhs: PcuDispatchValueId(1),
        rhs: PcuDispatchValueId(2),
    });
    let raw = PcuDispatchKernelIr {
        ops: &ops,
        ..program
    };
    let (output, result) = run_f32(&raw, &[1.0], &[2.0]);
    assert_eq!(
        result,
        Err(PcuCheckedFloatReferenceError::UnsupportedProfile)
    );
    assert_eq!(output, [19.0]);
    ops[2] = program.ops[2];
    ops[0] = PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
        binding: INPUT,
        result: PcuDispatchValueId(256),
        index: PcuDispatchIndex::InvocationId,
    });
    let malformed = PcuDispatchKernelIr {
        ops: &ops,
        ..program
    };
    let (output, result) = run_f32(&malformed, &[1.0], &[2.0]);
    assert!(result.is_err());
    assert_eq!(output, [19.0]);
    assert!(
        !PcuCheckedFloatReference::SUPPORTED_INSTRUCTIONS
            .contains(fusion_pcu::PcuDispatchOpCaps::ALU_ADD)
    );
}

#[test]
fn binary64_underflow_relu_and_invalid_operand_priority() {
    let mut program = kernel(
        PcuValueType::f64(),
        PcuDispatchFloatBinaryOp::Div,
        PcuFloatUnderflowPolicy::default(),
        PcuRangePolicy::Clamp,
        None,
        3,
    );
    let lhs = [-f64::from_bits(1), 4.0, f64::NAN];
    let rhs = [2.0, 0.0, 0.0];
    let mut output = [19.0_f64; 3];
    let result = PcuCheckedF64Reference.run_host_direct(
        submission(&program),
        &mut [
            PcuHostScalarBinding {
                target: INPUT,
                slice: PcuHostScalarSlice::Read(&lhs),
            },
            PcuHostScalarBinding {
                target: RHS,
                slice: PcuHostScalarSlice::Read(&rhs),
            },
            PcuHostScalarBinding {
                target: OUTPUT,
                slice: PcuHostScalarSlice::ReadWrite(&mut output),
            },
        ],
        PcuInvocationParameters::empty(),
    );
    assert_eq!(result, fault(PcuExecutionFaultKind::DivideByZero, 1, false));
    assert_eq!(output[0].to_bits(), 0x8000_0000_0000_0000);
    let mut ops = program.ops.to_vec();
    ops[2] = PcuDispatchOp::Data(PcuDispatchDataOp::CheckedFloatUnary {
        value_type: PcuValueType::f64(),
        op: PcuDispatchFloatUnaryOp::Relu,
        underflow_policy: PcuFloatUnderflowPolicy::RejectSubnormalResult,
        range_policy: PcuRangePolicy::Clamp,
        result: PcuDispatchValueId(3),
        value: PcuDispatchValueId(1),
    });
    program = PcuDispatchKernelIr {
        ops: &ops,
        ..program
    };
    let lhs = [-0.0, f64::from_bits(1), 3.0];
    let result = PcuCheckedF64Reference.run_host_direct(
        submission(&program),
        &mut [
            PcuHostScalarBinding {
                target: INPUT,
                slice: PcuHostScalarSlice::Read(&lhs),
            },
            PcuHostScalarBinding {
                target: RHS,
                slice: PcuHostScalarSlice::Read(&rhs),
            },
            PcuHostScalarBinding {
                target: OUTPUT,
                slice: PcuHostScalarSlice::ReadWrite(&mut output),
            },
        ],
        PcuInvocationParameters::empty(),
    );
    assert_eq!(
        result,
        fault(PcuExecutionFaultKind::ArithmeticUnderflow, 1, true)
    );
    assert_eq!(output.map(f64::to_bits), [0, 1, 3.0_f64.to_bits()]);
    let lhs = [f64::NAN, 1.0, 1.0];
    let program = kernel(
        PcuValueType::f64(),
        PcuDispatchFloatBinaryOp::Div,
        PcuFloatUnderflowPolicy::default(),
        PcuRangePolicy::Reject,
        None,
        3,
    );
    let result = PcuCheckedF64Reference.run_host_direct(
        submission(&program),
        &mut [
            PcuHostScalarBinding {
                target: INPUT,
                slice: PcuHostScalarSlice::Read(&lhs),
            },
            PcuHostScalarBinding {
                target: RHS,
                slice: PcuHostScalarSlice::Read(&[0.0; 3]),
            },
            PcuHostScalarBinding {
                target: OUTPUT,
                slice: PcuHostScalarSlice::ReadWrite(&mut output),
            },
        ],
        PcuInvocationParameters::empty(),
    );
    assert_eq!(
        result,
        fault(PcuExecutionFaultKind::InvalidFloatingOperand, 0, false)
    );
}

#[test]
#[allow(clippy::too_many_lines)] // Keep the rounding vectors and complete mixed SSA fixture together.
fn mixed_narrowing_rounds_ties_and_default_tininess_without_stale_ssa() {
    let bindings = [
        PcuBinding::scalar::<f64>(
            None,
            0,
            0,
            PcuBindingStorageClass::Storage,
            PcuBindingAccess::ReadOnly,
        ),
        PcuBinding::scalar::<f32>(
            None,
            0,
            2,
            PcuBindingStorageClass::Storage,
            PcuBindingAccess::WriteOnly,
        ),
    ];
    let body = [
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
            binding: INPUT,
            index: PcuDispatchIndex::GridStrideId,
            result: PcuDispatchValueId(1),
        }),
        PcuDispatchOp::Data(PcuDispatchDataOp::Constant {
            result: PcuDispatchValueId(2),
            value: fusion_pcu::PcuParameterValue::F64(1.0_f64.to_bits()),
        }),
        PcuDispatchOp::Data(PcuDispatchDataOp::CheckedFloatBinary {
            value_type: PcuValueType::f64(),
            op: PcuDispatchFloatBinaryOp::Mul,
            underflow_policy: PcuFloatUnderflowPolicy::default(),
            range_policy: PcuRangePolicy::Clamp,
            result: PcuDispatchValueId(3),
            lhs: PcuDispatchValueId(1),
            rhs: PcuDispatchValueId(2),
        }),
        PcuDispatchOp::Data(PcuDispatchDataOp::CheckedFloatConvert {
            conversion: PcuDispatchCheckedFloatConversion::F64ToF32,
            underflow_policy: PcuFloatUnderflowPolicy::default(),
            range_policy: PcuRangePolicy::Clamp,
            result: PcuDispatchValueId(4),
            value: PcuDispatchValueId(3),
        }),
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
            binding: OUTPUT,
            index: PcuDispatchIndex::GridStrideId,
            value: PcuDispatchValueId(4),
        }),
    ];
    let ops = [
        PcuDispatchOp::GridStrideLoop {
            extent: 6,
            body: &body,
        },
        PcuDispatchOp::Control(PcuDispatchControlOp::Return),
    ];
    let program = PcuDispatchKernelIr {
        numerical_requirements: PcuDispatchKernelIr::DEFAULT_REQUIREMENTS,
        id: PcuKernelId(504),
        entry: PcuDispatchEntryPoint {
            name: "narrow",
            logical_shape: [2, 1, 1],
        },
        bindings: &bindings,
        ports: &[],
        parameters: &[],
        ops: &ops,
        type_caps: PcuValueTypeCaps::FLOAT32 | PcuValueTypeCaps::FLOAT64,
        feature_caps: PcuDispatchFeatureCaps::default(),
    };
    let a = f64::from(f32::from_bits(0x3f80_0000));
    let b = f64::from(f32::from_bits(0x3f80_0001));
    let c = f64::from(f32::from_bits(0x3f80_0002));
    let lhs = [
        -0.0,
        a.midpoint(b),
        b.midpoint(c),
        f64::from(f32::from_bits(1)),
        -f64::from(f32::from_bits(1)) * 0.5,
        7.0,
    ];
    let mut output = [19.0; 6];
    let result = PcuCheckedFloatReference.run_host_direct(
        submission(&program),
        &mut [
            PcuCpuTypedBinding {
                target: INPUT,
                slice: PcuCpuTypedSlice::ReadF64(&lhs),
            },
            PcuCpuTypedBinding {
                target: OUTPUT,
                slice: PcuCpuTypedSlice::ReadWriteF32(&mut output),
            },
        ],
    );
    assert_eq!(
        result,
        fault(PcuExecutionFaultKind::ArithmeticUnderflow, 4, true)
    );
    assert_eq!(
        output.map(f32::to_bits),
        [
            0x8000_0000,
            0x3f80_0000,
            0x3f80_0002,
            1,
            0x8000_0000,
            7.0_f32.to_bits()
        ]
    );
    let lhs = [1.0_f64; 6];
    PcuCheckedFloatReference
        .run_host_direct(
            submission(&program),
            &mut [
                PcuCpuTypedBinding {
                    target: INPUT,
                    slice: PcuCpuTypedSlice::ReadF64(&lhs),
                },
                PcuCpuTypedBinding {
                    target: OUTPUT,
                    slice: PcuCpuTypedSlice::ReadWriteF32(&mut output),
                },
            ],
        )
        .unwrap();
    assert_eq!(output.map(f32::to_bits), [1.0_f32.to_bits(); 6]);
}
