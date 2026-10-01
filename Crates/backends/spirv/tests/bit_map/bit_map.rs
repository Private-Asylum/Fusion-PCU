//! Bounded bit-map admission and exact Vulkan SPIR-V environment validation.

#[rustfmt::skip]
use fusion_pcu_core::{
    PcuBinding,
    PcuBindingAccess,
    PcuBindingRef,
    PcuBindingStorageClass,
    PcuDispatchControlOp,
    PcuDispatchDataOp,
    PcuDispatchEntryPoint,
    PcuDispatchFeatureCaps,
    PcuDispatchFloatUnaryOp,
    PcuDispatchIndex,
    PcuDispatchKernelIr,
    PcuDispatchOp,
    PcuDispatchValueId,
    PcuFloatUnderflowPolicy,
    PcuKernelId,
    PcuRangePolicy,
    PcuScalarType,
    PcuValueType,
    PcuValueTypeCaps,
};
#[rustfmt::skip]
use fusion_pcu_spirv::{
    lower_float_bit_map_to_spirv,
    PcuSpirvError,
    PcuSpirvCapability,
    PcuSpirvCapabilityCaps,
    PcuSpirvBitOperation,
    PcuSpirvFixedSink,
    PcuSpirvLoweringOptions,
    PcuSpirvVersion,
};

fn with_kernel<R>(
    operation: PcuSpirvBitOperation,
    range: PcuRangePolicy,
    grid: bool,
    run: impl FnOnce(&PcuDispatchKernelIr<'_>) -> R,
) -> R {
    with_typed_kernel(operation, range, grid, PcuScalarType::F32, run)
}

fn with_typed_kernel<R>(
    operation: PcuSpirvBitOperation,
    range: PcuRangePolicy,
    grid: bool,
    scalar: PcuScalarType,
    run: impl FnOnce(&PcuDispatchKernelIr<'_>) -> R,
) -> R {
    let bindings = [
        PcuBinding::value(
            Some("input"),
            0,
            0,
            PcuBindingStorageClass::Storage,
            PcuBindingAccess::ReadOnly,
            PcuValueType::Scalar(scalar),
        ),
        PcuBinding::value(
            Some("output"),
            0,
            1,
            PcuBindingStorageClass::Storage,
            PcuBindingAccess::WriteOnly,
            PcuValueType::Scalar(scalar),
        ),
    ];
    let index = if grid {
        PcuDispatchIndex::GridStrideId
    } else {
        PcuDispatchIndex::InvocationId
    };
    let load = PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
        result: PcuDispatchValueId(1),
        binding: PcuBindingRef::new(0, 0),
        index,
    });
    let copy_body = [
        load,
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
            binding: PcuBindingRef::new(0, 1),
            index,
            value: PcuDispatchValueId(1),
        }),
    ];
    let neg_body = [
        load,
        PcuDispatchOp::Data(PcuDispatchDataOp::CheckedFloatUnary {
            value_type: PcuValueType::Scalar(scalar),
            op: PcuDispatchFloatUnaryOp::Neg,
            underflow_policy: match operation {
                PcuSpirvBitOperation::Copy => PcuFloatUnderflowPolicy::default(),
                PcuSpirvBitOperation::CheckedNeg(policy) => policy,
            },
            range_policy: range,
            result: PcuDispatchValueId(2),
            value: PcuDispatchValueId(1),
        }),
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
            binding: PcuBindingRef::new(0, 1),
            index,
            value: PcuDispatchValueId(2),
        }),
    ];
    let body = if operation == PcuSpirvBitOperation::Copy {
        &copy_body[..]
    } else {
        &neg_body[..]
    };
    let mut direct = body.to_vec();
    direct.push(PcuDispatchOp::Control(PcuDispatchControlOp::Return));
    let grid_ops = [
        PcuDispatchOp::GridStrideLoop { extent: 257, body },
        PcuDispatchOp::Control(PcuDispatchControlOp::Return),
    ];
    run(&PcuDispatchKernelIr {
        id: PcuKernelId(1),
        entry: PcuDispatchEntryPoint {
            name: "bits",
            logical_shape: [if grid { 2 } else { 257 }, 1, 1],
        },
        bindings: &bindings,
        ports: &[],
        parameters: &[],
        ops: if grid { &grid_ops } else { &direct },
        type_caps: if scalar == PcuScalarType::F64 {
            PcuValueTypeCaps::FLOAT64
        } else {
            PcuValueTypeCaps::FLOAT32
        },
        feature_caps: PcuDispatchFeatureCaps::default(),
    })
}

#[test]
fn profile_is_explicit_versioned_and_unsupported_admission_writes_no_words() {
    for operation in [
        PcuSpirvBitOperation::Copy,
        PcuSpirvBitOperation::CheckedNeg(PcuFloatUnderflowPolicy::IeeeAfterRounding),
        PcuSpirvBitOperation::CheckedNeg(PcuFloatUnderflowPolicy::RejectSubnormalResult),
    ] {
        for grid in [false, true] {
            with_kernel(operation, PcuRangePolicy::Reject, grid, |kernel| {
                let mut sink = PcuSpirvFixedSink::<512>::new();
                let (info, profile) = lower_float_bit_map_to_spirv(
                    kernel,
                    PcuSpirvLoweringOptions::default(),
                    &mut sink,
                )
                .unwrap();
                assert_eq!(profile.operation, operation);
                assert_eq!(profile.extent, 257);
                assert_eq!(profile.local_size, [64, 1, 1]);
                assert_eq!(info.word_count, sink.len());
                let mut empty = PcuSpirvFixedSink::<512>::new();
                assert!(matches!(
                    lower_float_bit_map_to_spirv(
                        kernel,
                        PcuSpirvLoweringOptions::default()
                            .with_version(PcuSpirvVersion(0x0001_0700)),
                        &mut empty
                    ),
                    Err(PcuSpirvError::UnsupportedVersion(_))
                ));
                assert!(empty.is_empty());
            });
        }
    }
    with_kernel(
        PcuSpirvBitOperation::CheckedNeg(PcuFloatUnderflowPolicy::default()),
        PcuRangePolicy::Clamp,
        false,
        |kernel| {
            let mut sink = PcuSpirvFixedSink::<512>::new();
            assert!(
                lower_float_bit_map_to_spirv(kernel, PcuSpirvLoweringOptions::default(), &mut sink)
                    .is_err()
            );
            assert!(sink.is_empty());
        },
    );
}

#[test]
#[ignore = "requires the official SPIRV-Tools spirv-val executable"]
fn emitted_copy_and_every_neg_policy_validate_for_vulkan_versions() {
    let directory = std::env::temp_dir().join(format!("pcu-spirv-bits-{}", std::process::id()));
    std::fs::create_dir_all(&directory).unwrap();
    for scalar in [PcuScalarType::F32, PcuScalarType::F64] {
        for operation in [
            PcuSpirvBitOperation::Copy,
            PcuSpirvBitOperation::CheckedNeg(PcuFloatUnderflowPolicy::IeeeAfterRounding),
            PcuSpirvBitOperation::CheckedNeg(PcuFloatUnderflowPolicy::AllowGradualUnderflow),
            PcuSpirvBitOperation::CheckedNeg(PcuFloatUnderflowPolicy::RejectSubnormalResult),
        ] {
            for (version, environment) in [
                (PcuSpirvVersion::V1_0, "vulkan1.0"),
                (PcuSpirvVersion::V1_3, "vulkan1.1"),
                (PcuSpirvVersion::V1_5, "vulkan1.2"),
                (PcuSpirvVersion::V1_6, "vulkan1.3"),
            ] {
                with_typed_kernel(operation, PcuRangePolicy::Reject, false, scalar, |kernel| {
                    let mut sink = PcuSpirvFixedSink::<512>::new();
                    lower_float_bit_map_to_spirv(
                        kernel,
                        PcuSpirvLoweringOptions::default()
                            .with_version(version)
                            .with_capabilities(
                                PcuSpirvCapabilityCaps::SHADER
                                    .union(PcuSpirvCapabilityCaps::FLOAT64),
                            ),
                        &mut sink,
                    )
                    .unwrap();
                    let bytes: Vec<u8> = sink
                        .as_slice()
                        .iter()
                        .flat_map(|word| word.to_le_bytes())
                        .collect();
                    let path = directory.join("bits.spv");
                    std::fs::write(&path, bytes).unwrap();
                    let result = std::process::Command::new("spirv-val")
                        .args(["--target-env", environment])
                        .arg(path)
                        .output()
                        .unwrap();
                    assert!(
                        result.status.success(),
                        "{operation:?} {version:?}: {}",
                        String::from_utf8_lossy(&result.stderr)
                    );
                });
            }
        }
    }
    std::fs::remove_dir_all(directory).unwrap();
}

#[test]
fn f64_admission_requires_explicit_float64_before_writing_and_both_schemas_are_typed() {
    for operation in [
        PcuSpirvBitOperation::Copy,
        PcuSpirvBitOperation::CheckedNeg(PcuFloatUnderflowPolicy::RejectSubnormalResult),
    ] {
        for grid in [false, true] {
            with_typed_kernel(
                operation,
                PcuRangePolicy::Reject,
                grid,
                PcuScalarType::F64,
                |kernel| {
                    let mut sink = PcuSpirvFixedSink::<512>::new();
                    assert_eq!(
                        lower_float_bit_map_to_spirv(
                            kernel,
                            PcuSpirvLoweringOptions::default(),
                            &mut sink
                        ),
                        Err(PcuSpirvError::UnsupportedCapability(
                            PcuSpirvCapability::Float64
                        ))
                    );
                    assert!(sink.is_empty());
                    let (info, profile) = lower_float_bit_map_to_spirv(
                        kernel,
                        PcuSpirvLoweringOptions::default().with_capabilities(
                            PcuSpirvCapabilityCaps::SHADER.union(PcuSpirvCapabilityCaps::FLOAT64),
                        ),
                        &mut sink,
                    )
                    .unwrap();
                    assert_eq!(profile.scalar, PcuScalarType::F64);
                    assert!(info.capabilities.contains(PcuSpirvCapabilityCaps::FLOAT64));
                    assert_eq!(info.bound, 59);
                },
            );
        }
    }
}
