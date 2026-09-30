//! Mixed-width host-call CUDA acceptance against the checked core oracle.

use super::*;
#[rustfmt::skip]
use fusion_pcu::{
    PcuBinding,
    PcuBindingStorageClass,
    PcuCheckedFloatConversion,
    PcuDispatchCheckedFloatConversion,
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

fn prepare_conversion(
    backend: &CudaOwnedDispatchBackend,
    conversion: PcuDispatchCheckedFloatConversion,
    grid: bool,
    range_policy: PcuRangePolicy,
) -> crate::CudaPreparedHostKernel {
    let (source, target) = match conversion {
        PcuDispatchCheckedFloatConversion::F32ToF64 => (PcuValueType::f32(), PcuValueType::f64()),
        PcuDispatchCheckedFloatConversion::F64ToF32 => (PcuValueType::f64(), PcuValueType::f32()),
    };
    let bindings = [
        PcuBinding::value(
            None,
            0,
            0,
            PcuBindingStorageClass::Storage,
            PcuBindingAccess::ReadOnly,
            source,
        ),
        PcuBinding::value(
            None,
            0,
            1,
            PcuBindingStorageClass::Storage,
            PcuBindingAccess::WriteOnly,
            target,
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
        PcuDispatchOp::Data(PcuDispatchDataOp::CheckedFloatConvert {
            conversion,
            underflow_policy: PcuFloatUnderflowPolicy::IeeeAfterRounding,
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
                name: "cuda_checked_conversion",
                logical_shape: [if grid { 1 } else { 4 }, 1, 1],
            },
            bindings: &bindings,
            ports: &[],
            parameters: &[],
            ops: if grid { &loop_ops } else { &direct },
            type_caps: fusion_pcu::PcuValueTypeCaps::FLOAT32
                .union(fusion_pcu::PcuValueTypeCaps::FLOAT64),
            feature_caps: fusion_pcu::PcuDispatchFeatureCaps::READ_ONLY_RESOURCES
                .union(fusion_pcu::PcuDispatchFeatureCaps::MUTABLE_RESOURCES)
                .union(if range_policy == PcuRangePolicy::Clamp {
                    fusion_pcu::PcuDispatchFeatureCaps::RANGE_CLAMP
                } else {
                    fusion_pcu::PcuDispatchFeatureCaps::empty()
                }),
        })
        .expect("prepare mixed-width checked conversion")
}

#[test]
#[ignore = "requires a working CUDA device"]
fn mixed_width_host_calls_preserve_bits_discard_fatal_output_and_retry() {
    let (_discovery, backend) = super::checked_integer_tests::selected_device();
    for grid in [false, true] {
        let mut narrow = prepare_conversion(
            &backend,
            PcuDispatchCheckedFloatConversion::F64ToF32,
            grid,
            PcuRangePolicy::Reject,
        );
        let source = [-0.0_f64, f64::from(f32::from_bits(1)), 1.25, -3.5];
        let mut output = [13.0_f32; 4];
        narrow
            .call(&mut [
                PcuHostArgument::read(PcuBindingRef::new(0, 0), &source),
                PcuHostArgument::read_write(PcuBindingRef::new(0, 1), &mut output),
            ])
            .unwrap();
        for (actual, input) in output.into_iter().zip(source) {
            assert_eq!(
                actual.to_bits(),
                input.pcu_checked_to_f32().unwrap().to_bits()
            );
        }
        for bad in [
            f64::NAN,
            f64::INFINITY,
            f64::MAX,
            f64::from(f32::from_bits(1)) * 0.5,
        ] {
            let source = [1.0, bad, 2.0, 3.0];
            let mut output = [13.0_f32; 4];
            assert!(
                narrow
                    .call(&mut [
                        PcuHostArgument::read(PcuBindingRef::new(0, 0), &source),
                        PcuHostArgument::read_write(PcuBindingRef::new(0, 1), &mut output)
                    ])
                    .is_err()
            );
            assert_eq!(output.map(f32::to_bits), [13.0_f32.to_bits(); 4]);
        }
        let source = [2.0_f64; 4];
        narrow
            .call(&mut [
                PcuHostArgument::read(PcuBindingRef::new(0, 0), &source),
                PcuHostArgument::read_write(PcuBindingRef::new(0, 1), &mut output),
            ])
            .unwrap();
        assert_eq!(output.map(f32::to_bits), [2.0_f32.to_bits(); 4]);

        let mut wide = prepare_conversion(
            &backend,
            PcuDispatchCheckedFloatConversion::F32ToF64,
            grid,
            PcuRangePolicy::Reject,
        );
        let source = [-0.0_f32, f32::from_bits(1), f32::MAX, -1.25];
        let mut output = [13.0_f64; 4];
        wide.call(&mut [
            PcuHostArgument::read(PcuBindingRef::new(0, 0), &source),
            PcuHostArgument::read_write(PcuBindingRef::new(0, 1), &mut output),
        ])
        .unwrap();
        for (actual, input) in output.into_iter().zip(source) {
            assert_eq!(actual.to_bits(), f64::from(input).to_bits());
        }
        let bad = [1.0_f32, f32::NAN, 2.0, 3.0];
        output.fill(13.0);
        assert!(
            wide.call(&mut [
                PcuHostArgument::read(PcuBindingRef::new(0, 0), &bad),
                PcuHostArgument::read_write(PcuBindingRef::new(0, 1), &mut output)
            ])
            .is_err()
        );
        assert_eq!(output.map(f64::to_bits), [13.0_f64.to_bits(); 4]);
        wide.call(&mut [
            PcuHostArgument::read(PcuBindingRef::new(0, 0), &source),
            PcuHostArgument::read_write(PcuBindingRef::new(0, 1), &mut output),
        ])
        .unwrap();
    }
}

#[test]
#[ignore = "requires a working CUDA device"]
fn clamped_narrowing_copies_complete_output_before_range_error_and_discards_fatal_output() {
    let (_discovery, backend) = super::checked_integer_tests::selected_device();
    for grid in [false, true] {
        let mut kernel = prepare_conversion(
            &backend,
            PcuDispatchCheckedFloatConversion::F64ToF32,
            grid,
            PcuRangePolicy::Clamp,
        );
        let source = [
            f64::MAX,
            -f64::MAX,
            f64::from(f32::from_bits(1)) * 0.5,
            -0.0,
        ];
        let mut output = [13.0_f32; 4];
        let error = kernel
            .call(&mut [
                PcuHostArgument::read(PcuBindingRef::new(0, 0), &source),
                PcuHostArgument::read_write(PcuBindingRef::new(0, 1), &mut output),
            ])
            .unwrap_err();
        assert!(matches!(
            error,
            crate::CudaHostKernelError::CheckedExecutionFault(fusion_pcu::PcuExecutionFault {
                kind: fusion_pcu::PcuExecutionFaultKind::ArithmeticOverflow,
                invocation_id: 0,
                recovered: true,
            })
        ));
        assert_eq!(
            output.map(f32::to_bits),
            [
                f32::MAX.to_bits(),
                (-f32::MAX).to_bits(),
                0,
                (-0.0_f32).to_bits()
            ]
        );
        let fatal = [f64::MAX, 2.0, f64::NAN, 3.0];
        output.fill(13.0);
        let error = kernel
            .call(&mut [
                PcuHostArgument::read(PcuBindingRef::new(0, 0), &fatal),
                PcuHostArgument::read_write(PcuBindingRef::new(0, 1), &mut output),
            ])
            .unwrap_err();
        assert!(matches!(
            error,
            crate::CudaHostKernelError::CheckedExecutionFault(fusion_pcu::PcuExecutionFault {
                kind: fusion_pcu::PcuExecutionFaultKind::InvalidFloatingOperand,
                invocation_id: 2,
                recovered: false,
            })
        ));
        assert_eq!(output.map(f32::to_bits), [13.0_f32.to_bits(); 4]);
    }
}
