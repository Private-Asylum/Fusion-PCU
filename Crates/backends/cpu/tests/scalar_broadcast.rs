use core::num::NonZeroU32;

use fusion_pcu_core::{
    PcuBinding, PcuBindingAccess, PcuBindingRef, PcuBindingStorageClass, PcuDispatchAluOp,
    PcuDispatchDataOp, PcuDispatchEntryPoint, PcuDispatchFeatureCaps, PcuDispatchIndex,
    PcuDispatchKernelIr, PcuDispatchOp, PcuDispatchSubmission, PcuDispatchValueId,
    PcuHostScalarBinding, PcuHostScalarSlice, PcuInvocationParameters, PcuInvocationShape,
    PcuKernelId, PcuSynchronousHostDispatchBackend, PcuValueType, PcuValueTypeCaps,
    validate_f32_map_kernel, validate_f64_map_kernel,
};
use fusion_pcu_cpu::{PcuF32Reference, PcuF64Reference};

#[test]
fn f32_coefficient_scalar_broadcast_maps_over_vector() {
    let bindings = [
        PcuBinding::scalar::<f32>(
            Some("coefficient"),
            0,
            0,
            PcuBindingStorageClass::Storage,
            PcuBindingAccess::ReadOnly,
        ),
        PcuBinding::value(
            Some("input"),
            0,
            1,
            PcuBindingStorageClass::Storage,
            PcuBindingAccess::ReadOnly,
            PcuValueType::f32(),
        ),
        PcuBinding::value(
            Some("output"),
            0,
            2,
            PcuBindingStorageClass::Storage,
            PcuBindingAccess::WriteOnly,
            PcuValueType::f32(),
        ),
    ];
    let ops = [
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
            result: PcuDispatchValueId(1),
            binding: PcuBindingRef::new(0, 0),
            index: PcuDispatchIndex::BindingElementZero,
        }),
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
            result: PcuDispatchValueId(2),
            binding: PcuBindingRef::new(0, 1),
            index: PcuDispatchIndex::InvocationId,
        }),
        PcuDispatchOp::Data(PcuDispatchDataOp::Alu {
            value_type: PcuValueType::f32(),
            result: PcuDispatchValueId(3),
            op: PcuDispatchAluOp::Mul,
            lhs: PcuDispatchValueId(1),
            rhs: PcuDispatchValueId(2),
        }),
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
            binding: PcuBindingRef::new(0, 2),
            index: PcuDispatchIndex::InvocationId,
            value: PcuDispatchValueId(3),
        }),
        PcuDispatchOp::Control(fusion_pcu_core::PcuDispatchControlOp::Return),
    ];
    let kernel = PcuDispatchKernelIr {
        id: PcuKernelId(1),
        entry: PcuDispatchEntryPoint {
            name: "coefficient_map",
            logical_shape: [3, 1, 1],
        },
        bindings: &bindings,
        ports: &[],
        parameters: &[],
        ops: &ops,
        type_caps: PcuValueTypeCaps::for_scalar(fusion_pcu_core::PcuScalarType::F32),
        feature_caps: PcuDispatchFeatureCaps::default(),
    };
    assert_eq!(validate_f32_map_kernel(&kernel), Ok(()));

    let coefficient = 2.5_f32;
    let input = [1.0_f32, -2.0, 4.0];
    let mut output = [0.0_f32; 3];
    let mut host = [
        PcuHostScalarBinding {
            target: PcuBindingRef::new(0, 0),
            slice: PcuHostScalarSlice::Read(core::slice::from_ref(&coefficient)),
        },
        PcuHostScalarBinding {
            target: PcuBindingRef::new(0, 1),
            slice: PcuHostScalarSlice::Read(&input),
        },
        PcuHostScalarBinding {
            target: PcuBindingRef::new(0, 2),
            slice: PcuHostScalarSlice::ReadWrite(&mut output),
        },
    ];
    PcuF32Reference
        .run_host_direct(
            PcuDispatchSubmission {
                kernel: &kernel,
                shape: PcuInvocationShape::invocations(NonZeroU32::new(3).unwrap()),
            },
            &mut host,
            PcuInvocationParameters::empty(),
        )
        .unwrap();
    assert_eq!(
        output.map(f32::to_bits),
        [2.5_f32, -5.0, 10.0].map(f32::to_bits)
    );
}

#[test]
fn f64_seed_scalar_broadcast_maps_over_vector() {
    let bindings = [
        PcuBinding::scalar::<f64>(
            Some("seed"),
            0,
            0,
            PcuBindingStorageClass::Storage,
            PcuBindingAccess::ReadOnly,
        ),
        PcuBinding::value(
            Some("input"),
            0,
            1,
            PcuBindingStorageClass::Storage,
            PcuBindingAccess::ReadOnly,
            PcuValueType::f64(),
        ),
        PcuBinding::value(
            Some("output"),
            0,
            2,
            PcuBindingStorageClass::Storage,
            PcuBindingAccess::WriteOnly,
            PcuValueType::f64(),
        ),
    ];
    let ops = [
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
            result: PcuDispatchValueId(1),
            binding: PcuBindingRef::new(0, 0),
            index: PcuDispatchIndex::BindingElementZero,
        }),
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
            result: PcuDispatchValueId(2),
            binding: PcuBindingRef::new(0, 1),
            index: PcuDispatchIndex::InvocationId,
        }),
        PcuDispatchOp::Data(PcuDispatchDataOp::Alu {
            value_type: PcuValueType::f64(),
            result: PcuDispatchValueId(3),
            op: PcuDispatchAluOp::Add,
            lhs: PcuDispatchValueId(1),
            rhs: PcuDispatchValueId(2),
        }),
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
            binding: PcuBindingRef::new(0, 2),
            index: PcuDispatchIndex::InvocationId,
            value: PcuDispatchValueId(3),
        }),
        PcuDispatchOp::Control(fusion_pcu_core::PcuDispatchControlOp::Return),
    ];
    let kernel = PcuDispatchKernelIr {
        id: PcuKernelId(2),
        entry: PcuDispatchEntryPoint {
            name: "seed_map",
            logical_shape: [3, 1, 1],
        },
        bindings: &bindings,
        ports: &[],
        parameters: &[],
        ops: &ops,
        type_caps: PcuValueTypeCaps::for_scalar(fusion_pcu_core::PcuScalarType::F64),
        feature_caps: PcuDispatchFeatureCaps::default(),
    };
    assert_eq!(validate_f64_map_kernel(&kernel), Ok(()));

    let seed = -0.5_f64;
    let input = [1.0_f64, 2.0, 4.0];
    let mut output = [0.0_f64; 3];
    let mut host = [
        PcuHostScalarBinding {
            target: PcuBindingRef::new(0, 0),
            slice: PcuHostScalarSlice::Read(core::slice::from_ref(&seed)),
        },
        PcuHostScalarBinding {
            target: PcuBindingRef::new(0, 1),
            slice: PcuHostScalarSlice::Read(&input),
        },
        PcuHostScalarBinding {
            target: PcuBindingRef::new(0, 2),
            slice: PcuHostScalarSlice::ReadWrite(&mut output),
        },
    ];
    PcuF64Reference
        .run_host_direct(
            PcuDispatchSubmission {
                kernel: &kernel,
                shape: PcuInvocationShape::invocations(NonZeroU32::new(3).unwrap()),
            },
            &mut host,
            PcuInvocationParameters::empty(),
        )
        .unwrap();
    assert_eq!(
        output.map(f64::to_bits),
        [0.5_f64, 1.5, 3.5].map(f64::to_bits)
    );
}

#[test]
#[allow(clippy::too_many_lines)] // Keep the complete direct-map arithmetic contract visible.
fn f64_min_max_keep_double_precision_nan_and_signed_zero_semantics() {
    let bindings = [
        PcuBinding::value(
            Some("left"),
            0,
            0,
            PcuBindingStorageClass::Storage,
            PcuBindingAccess::ReadOnly,
            PcuValueType::f64(),
        ),
        PcuBinding::value(
            Some("right"),
            0,
            1,
            PcuBindingStorageClass::Storage,
            PcuBindingAccess::ReadOnly,
            PcuValueType::f64(),
        ),
        PcuBinding::value(
            Some("output"),
            0,
            2,
            PcuBindingStorageClass::Storage,
            PcuBindingAccess::WriteOnly,
            PcuValueType::f64(),
        ),
    ];
    let left = [16_777_217.0_f64, f64::NAN, -0.0, f64::NAN];
    let right = [1.0_f64, 2.0, 0.0, f64::NAN];

    for (operation, expected) in [
        (
            PcuDispatchAluOp::Max,
            [16_777_217.0_f64, 2.0, 0.0, f64::NAN],
        ),
        (PcuDispatchAluOp::Min, [1.0_f64, 2.0, -0.0, f64::NAN]),
    ] {
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
            PcuDispatchOp::Data(PcuDispatchDataOp::Alu {
                value_type: PcuValueType::f64(),
                result: PcuDispatchValueId(3),
                op: operation,
                lhs: PcuDispatchValueId(1),
                rhs: PcuDispatchValueId(2),
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                binding: PcuBindingRef::new(0, 2),
                index: PcuDispatchIndex::InvocationId,
                value: PcuDispatchValueId(3),
            }),
            PcuDispatchOp::Control(fusion_pcu_core::PcuDispatchControlOp::Return),
        ];
        let kernel = PcuDispatchKernelIr {
            id: PcuKernelId(3),
            entry: PcuDispatchEntryPoint {
                name: "f64_min_max",
                logical_shape: [4, 1, 1],
            },
            bindings: &bindings,
            ports: &[],
            parameters: &[],
            ops: &ops,
            type_caps: PcuValueTypeCaps::for_scalar(fusion_pcu_core::PcuScalarType::F64),
            feature_caps: PcuDispatchFeatureCaps::default(),
        };
        assert_eq!(validate_f64_map_kernel(&kernel), Ok(()));

        let mut output = [0.0_f64; 4];
        let mut host = [
            PcuHostScalarBinding {
                target: PcuBindingRef::new(0, 0),
                slice: PcuHostScalarSlice::Read(&left),
            },
            PcuHostScalarBinding {
                target: PcuBindingRef::new(0, 1),
                slice: PcuHostScalarSlice::Read(&right),
            },
            PcuHostScalarBinding {
                target: PcuBindingRef::new(0, 2),
                slice: PcuHostScalarSlice::ReadWrite(&mut output),
            },
        ];
        PcuF64Reference
            .run_host_direct(
                PcuDispatchSubmission {
                    kernel: &kernel,
                    shape: PcuInvocationShape::invocations(NonZeroU32::new(4).unwrap()),
                },
                &mut host,
                PcuInvocationParameters::empty(),
            )
            .unwrap();
        for index in 0..3 {
            assert_eq!(output[index].to_bits(), expected[index].to_bits());
        }
        assert!(output[3].is_nan());
    }
}

#[test]
#[allow(clippy::too_many_lines)] // This oracle pins the full F64 Add+ReLU IR and exact output.
fn f64_add_relu_oracle_preserves_bits_beyond_f32_integer_precision() {
    let bindings = [
        PcuBinding::value(
            Some("left"),
            0,
            0,
            PcuBindingStorageClass::Storage,
            PcuBindingAccess::ReadOnly,
            PcuValueType::f64(),
        ),
        PcuBinding::value(
            Some("right"),
            0,
            1,
            PcuBindingStorageClass::Storage,
            PcuBindingAccess::ReadOnly,
            PcuValueType::f64(),
        ),
        PcuBinding::value(
            Some("output"),
            0,
            2,
            PcuBindingStorageClass::Storage,
            PcuBindingAccess::WriteOnly,
            PcuValueType::f64(),
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
        PcuDispatchOp::Data(PcuDispatchDataOp::Alu {
            value_type: PcuValueType::f64(),
            result: PcuDispatchValueId(3),
            op: PcuDispatchAluOp::Add,
            lhs: PcuDispatchValueId(1),
            rhs: PcuDispatchValueId(2),
        }),
        PcuDispatchOp::Data(PcuDispatchDataOp::Constant {
            result: PcuDispatchValueId(4),
            value: fusion_pcu_core::PcuParameterValue::F64(0.0_f64.to_bits()),
        }),
        PcuDispatchOp::Data(PcuDispatchDataOp::Alu {
            value_type: PcuValueType::f64(),
            result: PcuDispatchValueId(5),
            op: PcuDispatchAluOp::Max,
            lhs: PcuDispatchValueId(3),
            rhs: PcuDispatchValueId(4),
        }),
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
            binding: PcuBindingRef::new(0, 2),
            index: PcuDispatchIndex::InvocationId,
            value: PcuDispatchValueId(5),
        }),
        PcuDispatchOp::Control(fusion_pcu_core::PcuDispatchControlOp::Return),
    ];
    let kernel = PcuDispatchKernelIr {
        id: PcuKernelId(4),
        entry: PcuDispatchEntryPoint {
            name: "f64_add_relu",
            logical_shape: [3, 1, 1],
        },
        bindings: &bindings,
        ports: &[],
        parameters: &[],
        ops: &ops,
        type_caps: PcuValueTypeCaps::for_scalar(fusion_pcu_core::PcuScalarType::F64),
        feature_caps: PcuDispatchFeatureCaps::default(),
    };
    assert_eq!(validate_f64_map_kernel(&kernel), Ok(()));

    let left = [16_777_217.0_f64, -16_777_219.0, -2.0];
    let right = [1.0_f64; 3];
    let mut output = [0.0_f64; 3];
    let mut host = [
        PcuHostScalarBinding {
            target: PcuBindingRef::new(0, 0),
            slice: PcuHostScalarSlice::Read(&left),
        },
        PcuHostScalarBinding {
            target: PcuBindingRef::new(0, 1),
            slice: PcuHostScalarSlice::Read(&right),
        },
        PcuHostScalarBinding {
            target: PcuBindingRef::new(0, 2),
            slice: PcuHostScalarSlice::ReadWrite(&mut output),
        },
    ];
    PcuF64Reference
        .run_host_direct(
            PcuDispatchSubmission {
                kernel: &kernel,
                shape: PcuInvocationShape::invocations(NonZeroU32::new(3).unwrap()),
            },
            &mut host,
            PcuInvocationParameters::empty(),
        )
        .unwrap();
    assert_eq!(
        output.map(f64::to_bits),
        [16_777_218.0, 0.0, 0.0].map(f64::to_bits)
    );
}
