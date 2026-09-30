//! Checked unary CUDA acceptance, including exact encodings and terminal publication.

use super::*;
#[rustfmt::skip]
use fusion_pcu::{
    PcuBinding,
    PcuBindingStorageClass,
    
    PcuDispatchFloatUnaryOp,
    PcuDispatchControlOp,
    PcuDispatchEntryPoint,
    PcuDispatchIndex,
    PcuDispatchValueId,
    PcuFloatUnderflowPolicy,
    PcuHostArgument,
    PcuHostKernelBackend,
    PcuPreparedHostKernel,
    PcuRangePolicy,
    PcuValueType,
};

fn prepare_unary(
    backend: &CudaOwnedDispatchBackend,
    op: PcuDispatchFloatUnaryOp,
    value_type: PcuValueType,
    policy: PcuFloatUnderflowPolicy,
    grid: bool,
    range_policy: PcuRangePolicy,
) -> crate::CudaPreparedHostKernel {
    let bindings = [
        PcuBinding::value(
            None,
            0,
            0,
            PcuBindingStorageClass::Storage,
            PcuBindingAccess::ReadOnly,
            value_type,
        ),
        PcuBinding::value(
            None,
            0,
            1,
            PcuBindingStorageClass::Storage,
            PcuBindingAccess::WriteOnly,
            value_type,
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
        PcuDispatchOp::Data(PcuDispatchDataOp::CheckedFloatUnary {
            value_type,
            op,
            underflow_policy: policy,
            range_policy,
            result: PcuDispatchValueId(2),
            value: PcuDispatchValueId(1),
        }),
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
            binding: PcuBindingRef::new(0, 1),
            index,
            value: PcuDispatchValueId(2),
        }),
    ];
    let direct = [
        body[0],
        body[1],
        body[2],
        PcuDispatchOp::Control(PcuDispatchControlOp::Return),
    ];
    let loop_ops = [
        PcuDispatchOp::GridStrideLoop {
            extent: 4,
            body: &body,
        },
        PcuDispatchOp::Control(PcuDispatchControlOp::Return),
    ];
    backend
        .prepare_host_kernel(&PcuDispatchKernelIr {
            id: fusion_pcu::PcuKernelId(0xc070),
            entry: PcuDispatchEntryPoint {
                name: "cuda_checked_relu",
                logical_shape: [if grid { 1 } else { 4 }, 1, 1],
            },
            bindings: &bindings,
            ports: &[],
            parameters: &[],
            ops: if grid { &loop_ops } else { &direct },
            type_caps: fusion_pcu::PcuValueTypeCaps::for_scalar(value_type.scalar_type())
                .union(fusion_pcu::PcuValueTypeCaps::SCALAR_VALUES),
            feature_caps: fusion_pcu::PcuDispatchFeatureCaps::READ_ONLY_RESOURCES
                .union(fusion_pcu::PcuDispatchFeatureCaps::MUTABLE_RESOURCES)
                .union(if range_policy == PcuRangePolicy::Clamp {
                    fusion_pcu::PcuDispatchFeatureCaps::RANGE_CLAMP
                } else {
                    fusion_pcu::PcuDispatchFeatureCaps::empty()
                }),
        })
        .expect("prepare checked unary")
}

#[test]
#[ignore = "requires a working CUDA device"]
fn checked_relu_direct_and_grid_preserve_finite_bits_discard_faults_and_retry() {
    let (_discovery, backend) = super::checked_integer_tests::selected_device();
    for grid in [false, true] {
        for policy in [
            PcuFloatUnderflowPolicy::IeeeAfterRounding,
            PcuFloatUnderflowPolicy::AllowGradualUnderflow,
            PcuFloatUnderflowPolicy::RejectSubnormalResult,
        ] {
            let mut kernel = prepare_unary(
                &backend,
                PcuDispatchFloatUnaryOp::Relu,
                PcuValueType::f32(),
                policy,
                grid,
                PcuRangePolicy::Reject,
            );
            let input = [-0.0_f32, -3.0, 2.5, f32::MAX];
            let mut output = [13.0_f32; 4];
            kernel
                .call(&mut [
                    PcuHostArgument::read(PcuBindingRef::new(0, 0), &input),
                    PcuHostArgument::read_write(PcuBindingRef::new(0, 1), &mut output),
                ])
                .unwrap();
            assert_eq!(
                output.map(f32::to_bits),
                [0, 0, 2.5_f32.to_bits(), f32::MAX.to_bits()]
            );
            for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
                output.fill(13.0);
                let input = [1.0, bad, 2.0, 3.0];
                assert!(
                    kernel
                        .call(&mut [
                            PcuHostArgument::read(PcuBindingRef::new(0, 0), &input),
                            PcuHostArgument::read_write(PcuBindingRef::new(0, 1), &mut output)
                        ])
                        .is_err()
                );
                assert_eq!(output.map(f32::to_bits), [13.0_f32.to_bits(); 4]);
            }
            let input = [f32::from_bits(1), -f32::from_bits(1), 1.0, -1.0];
            output.fill(13.0);
            let result = kernel.call(&mut [
                PcuHostArgument::read(PcuBindingRef::new(0, 0), &input),
                PcuHostArgument::read_write(PcuBindingRef::new(0, 1), &mut output),
            ]);
            if policy == PcuFloatUnderflowPolicy::RejectSubnormalResult {
                assert!(result.is_err());
                assert_eq!(output.map(f32::to_bits), [13.0_f32.to_bits(); 4]);
            } else {
                result.unwrap();
                assert_eq!(output.map(f32::to_bits), [1, 0, 1.0_f32.to_bits(), 0]);
            }
            let mut wide = prepare_unary(
                &backend,
                PcuDispatchFloatUnaryOp::Relu,
                PcuValueType::f64(),
                policy,
                grid,
                PcuRangePolicy::Reject,
            );
            let input = [-0.0_f64, -3.0, 2.5, f64::MAX];
            let mut output = [13.0_f64; 4];
            wide.call(&mut [
                PcuHostArgument::read(PcuBindingRef::new(0, 0), &input),
                PcuHostArgument::read_write(PcuBindingRef::new(0, 1), &mut output),
            ])
            .unwrap();
            assert_eq!(
                output.map(f64::to_bits),
                [0, 0, 2.5_f64.to_bits(), f64::MAX.to_bits()]
            );
        }
    }
}

#[test]
#[ignore = "requires a working CUDA device"]
fn checked_f64_relu_range_clamp_and_fatal_priority_preserve_terminal_output_contract() {
    let (_discovery, backend) = super::checked_integer_tests::selected_device();
    for grid in [false, true] {
        let mut strict = prepare_unary(
            &backend,
            PcuDispatchFloatUnaryOp::Relu,
            PcuValueType::f64(),
            PcuFloatUnderflowPolicy::RejectSubnormalResult,
            grid,
            PcuRangePolicy::Reject,
        );
        let input = [f64::from_bits(1), -f64::from_bits(1), 1.0, -0.0];
        let mut output = [13.0_f64; 4];
        assert!(
            strict
                .call(&mut [
                    PcuHostArgument::read(PcuBindingRef::new(0, 0), &input),
                    PcuHostArgument::read_write(PcuBindingRef::new(0, 1), &mut output)
                ])
                .is_err()
        );
        assert_eq!(output.map(f64::to_bits), [13.0_f64.to_bits(); 4]);
        let mut clamp = prepare_unary(
            &backend,
            PcuDispatchFloatUnaryOp::Relu,
            PcuValueType::f64(),
            PcuFloatUnderflowPolicy::RejectSubnormalResult,
            grid,
            PcuRangePolicy::Clamp,
        );
        let error = clamp
            .call(&mut [
                PcuHostArgument::read(PcuBindingRef::new(0, 0), &input),
                PcuHostArgument::read_write(PcuBindingRef::new(0, 1), &mut output),
            ])
            .unwrap_err();
        assert!(matches!(
            error,
            crate::CudaHostKernelError::CheckedExecutionFault(fusion_pcu::PcuExecutionFault {
                kind: fusion_pcu::PcuExecutionFaultKind::ArithmeticUnderflow,
                invocation_id: 0,
                recovered: true
            })
        ));
        assert_eq!(output.map(f64::to_bits), [1, 0, 1.0_f64.to_bits(), 0]);
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let input = [f64::from_bits(1), 1.0, bad, 2.0];
            output.fill(13.0);
            let error = clamp
                .call(&mut [
                    PcuHostArgument::read(PcuBindingRef::new(0, 0), &input),
                    PcuHostArgument::read_write(PcuBindingRef::new(0, 1), &mut output),
                ])
                .unwrap_err();
            assert!(matches!(
                error,
                crate::CudaHostKernelError::CheckedExecutionFault(fusion_pcu::PcuExecutionFault {
                    kind: fusion_pcu::PcuExecutionFaultKind::InvalidFloatingOperand,
                    invocation_id: 2,
                    recovered: false
                })
            ));
            assert_eq!(output.map(f64::to_bits), [13.0_f64.to_bits(); 4]);
        }
        let input = [2.0_f64; 4];
        clamp
            .call(&mut [
                PcuHostArgument::read(PcuBindingRef::new(0, 0), &input),
                PcuHostArgument::read_write(PcuBindingRef::new(0, 1), &mut output),
            ])
            .unwrap();
        assert_eq!(output.map(f64::to_bits), [2.0_f64.to_bits(); 4]);
    }
}

macro_rules! neg_acceptance {
    ($name:ident, $float:ty, $bits:ty, $ty:expr, $sign:expr) => {
        #[test]
        #[ignore = "requires a working CUDA device"]
        #[allow(clippy::too_many_lines)] // Keep each width's fault, recovery and retry sequence together.
        fn $name() {
            let (_discovery, backend) = super::checked_integer_tests::selected_device();
            for grid in [false, true] {
                for policy in [
                    PcuFloatUnderflowPolicy::IeeeAfterRounding,
                    PcuFloatUnderflowPolicy::AllowGradualUnderflow,
                    PcuFloatUnderflowPolicy::RejectSubnormalResult,
                ] {
                    let mut kernel = prepare_unary(
                        &backend,
                        PcuDispatchFloatUnaryOp::Neg,
                        $ty,
                        policy,
                        grid,
                        PcuRangePolicy::Reject,
                    );
                    let mut output = [13.0 as $float; 4];
                    let input = [0.0 as $float, -0.0 as $float, <$float>::MAX, -<$float>::MAX];
                    kernel
                        .call(&mut [
                            PcuHostArgument::read(PcuBindingRef::new(0, 0), &input),
                            PcuHostArgument::read_write(PcuBindingRef::new(0, 1), &mut output),
                        ])
                        .unwrap();
                    assert_eq!(
                        output.map(<$float>::to_bits),
                        input.map(|value| value.to_bits() ^ $sign)
                    );

                    let subnormal = <$float>::from_bits(1);
                    let tiny_input = [
                        subnormal,
                        -subnormal,
                        <$float>::MIN_POSITIVE,
                        -<$float>::MIN_POSITIVE,
                    ];
                    output.fill(13.0);
                    let result = kernel.call(&mut [
                        PcuHostArgument::read(PcuBindingRef::new(0, 0), &tiny_input),
                        PcuHostArgument::read_write(PcuBindingRef::new(0, 1), &mut output),
                    ]);
                    if policy == PcuFloatUnderflowPolicy::RejectSubnormalResult {
                        assert!(matches!(
                            result,
                            Err(crate::CudaHostKernelError::CheckedExecutionFault(
                                fusion_pcu::PcuExecutionFault {
                                    kind: fusion_pcu::PcuExecutionFaultKind::ArithmeticUnderflow,
                                    invocation_id: 0,
                                    recovered: false,
                                }
                            ))
                        ));
                        assert_eq!(
                            output.map(<$float>::to_bits),
                            [13.0 as $float; 4].map(<$float>::to_bits)
                        );
                    } else {
                        result.unwrap();
                        assert_eq!(
                            output.map(<$float>::to_bits),
                            tiny_input.map(|value| value.to_bits() ^ $sign)
                        );
                    }
                    for bad in [<$float>::NAN, <$float>::INFINITY, <$float>::NEG_INFINITY] {
                        let bad_input = [1.0 as $float, bad, bad, 2.0 as $float];
                        output.fill(13.0);
                        let error = kernel
                            .call(&mut [
                                PcuHostArgument::read(PcuBindingRef::new(0, 0), &bad_input),
                                PcuHostArgument::read_write(PcuBindingRef::new(0, 1), &mut output),
                            ])
                            .unwrap_err();
                        assert!(matches!(
                            error,
                            crate::CudaHostKernelError::CheckedExecutionFault(
                                fusion_pcu::PcuExecutionFault {
                                    kind: fusion_pcu::PcuExecutionFaultKind::InvalidFloatingOperand,
                                    invocation_id: 1,
                                    recovered: false,
                                }
                            )
                        ));
                        assert_eq!(
                            output.map(<$float>::to_bits),
                            [13.0 as $float; 4].map(<$float>::to_bits)
                        );
                        kernel
                            .call(&mut [
                                PcuHostArgument::read(PcuBindingRef::new(0, 0), &input),
                                PcuHostArgument::read_write(PcuBindingRef::new(0, 1), &mut output),
                            ])
                            .unwrap();
                        assert_eq!(
                            output.map(<$float>::to_bits),
                            input.map(|value| value.to_bits() ^ $sign)
                        );
                    }
                }
                let mut clamp = prepare_unary(
                    &backend,
                    PcuDispatchFloatUnaryOp::Neg,
                    $ty,
                    PcuFloatUnderflowPolicy::RejectSubnormalResult,
                    grid,
                    PcuRangePolicy::Clamp,
                );
                let tiny = <$float>::from_bits(1);
                let input = [tiny, -tiny, -0.0 as $float, 3.0 as $float];
                let mut output = [13.0 as $float; 4];
                let error = clamp
                    .call(&mut [
                        PcuHostArgument::read(PcuBindingRef::new(0, 0), &input),
                        PcuHostArgument::read_write(PcuBindingRef::new(0, 1), &mut output),
                    ])
                    .unwrap_err();
                assert!(matches!(
                    error,
                    crate::CudaHostKernelError::CheckedExecutionFault(
                        fusion_pcu::PcuExecutionFault {
                            kind: fusion_pcu::PcuExecutionFaultKind::ArithmeticUnderflow,
                            invocation_id: 0,
                            recovered: true,
                        }
                    )
                ));
                assert_eq!(
                    output.map(<$float>::to_bits),
                    input.map(|value| value.to_bits() ^ $sign)
                );
                let input = [tiny, 1.0 as $float, <$float>::INFINITY, 2.0 as $float];
                output.fill(13.0);
                let error = clamp
                    .call(&mut [
                        PcuHostArgument::read(PcuBindingRef::new(0, 0), &input),
                        PcuHostArgument::read_write(PcuBindingRef::new(0, 1), &mut output),
                    ])
                    .unwrap_err();
                assert!(matches!(
                    error,
                    crate::CudaHostKernelError::CheckedExecutionFault(
                        fusion_pcu::PcuExecutionFault {
                            kind: fusion_pcu::PcuExecutionFaultKind::InvalidFloatingOperand,
                            invocation_id: 2,
                            recovered: false,
                        }
                    )
                ));
                assert_eq!(
                    output.map(<$float>::to_bits),
                    [13.0 as $float; 4].map(<$float>::to_bits)
                );

                // Encoding oracle is independent of arithmetic helpers. The deterministic corpus
                // spans signs/exponents/mantissas, with nonfinite values replaced before admission.
                let mut kernel = prepare_unary(
                    &backend,
                    PcuDispatchFloatUnaryOp::Neg,
                    $ty,
                    PcuFloatUnderflowPolicy::IeeeAfterRounding,
                    grid,
                    PcuRangePolicy::Reject,
                );
                let mut seed = 0x8ad7_415e_c62b_193fu64;
                for _ in 0..32 {
                    let input: [$float; 4] = std::array::from_fn(|_| {
                        seed ^= seed << 13;
                        seed ^= seed >> 7;
                        seed ^= seed << 17;
                        #[allow(clippy::cast_possible_truncation)]
                        // Select the tested encoding width.
                        let bits = seed as $bits;
                        let value = <$float>::from_bits(bits);
                        if value.is_finite() {
                            value
                        } else {
                            1.0 as $float
                        }
                    });
                    kernel
                        .call(&mut [
                            PcuHostArgument::read(PcuBindingRef::new(0, 0), &input),
                            PcuHostArgument::read_write(PcuBindingRef::new(0, 1), &mut output),
                        ])
                        .unwrap();
                    assert_eq!(
                        output.map(<$float>::to_bits),
                        input.map(|value| value.to_bits() ^ $sign)
                    );
                }
            }
        }
    };
}

neg_acceptance!(
    checked_f32_neg_exact_bits_fault_clamp_and_retry,
    f32,
    u32,
    PcuValueType::f32(),
    0x8000_0000u32
);
neg_acceptance!(
    checked_f64_neg_exact_bits_fault_clamp_and_retry,
    f64,
    u64,
    PcuValueType::f64(),
    0x8000_0000_0000_0000u64
);
