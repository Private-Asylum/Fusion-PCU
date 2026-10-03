//! Shader/U32-only raw carrier transport; separate broadcast/dense cold admission and byte bounds.
#[rustfmt::skip]
use fusion_pcu_core::{PcuBinding,PcuBindingAccess,PcuBindingRef,PcuBindingStorageClass,PcuDispatchControlOp,PcuDispatchDataOp,PcuDispatchEntryPoint,PcuDispatchFeatureCaps,PcuDispatchIndex,PcuDispatchKernelIr,PcuDispatchOp,PcuDispatchValueId,PcuKernelId,PcuScalarType,PcuValueType,PcuValueTypeCaps,PcuReproducibility};
#[rustfmt::skip]
use fusion_pcu_spirv::{lower_scalar_transport_to_spirv,PcuSpirvCapabilityCaps,PcuSpirvLoweringOptions,PcuSpirvVersion};
const TYPES: [PcuScalarType; 22] = [
    PcuScalarType::I8,
    PcuScalarType::U8,
    PcuScalarType::I16,
    PcuScalarType::U16,
    PcuScalarType::I32,
    PcuScalarType::U32,
    PcuScalarType::I64,
    PcuScalarType::U64,
    PcuScalarType::I128,
    PcuScalarType::U128,
    PcuScalarType::I256,
    PcuScalarType::U256,
    PcuScalarType::I512,
    PcuScalarType::U512,
    PcuScalarType::F16,
    PcuScalarType::BF16,
    PcuScalarType::F8E4M3FN,
    PcuScalarType::F8E5M2,
    PcuScalarType::F32,
    PcuScalarType::F64,
    PcuScalarType::F128,
    PcuScalarType::F256,
];
fn with(
    scalar: PcuScalarType,
    extent: u32,
    grid: bool,
    broadcast: bool,
    run: impl FnOnce(&PcuDispatchKernelIr<'_>),
) {
    // Deliberately noncanonical binding IDs/order: physical shader slots must not leak into IR.
    let source = PcuBindingRef::new(7, 9);
    let destination = PcuBindingRef::new(8, 11);
    let bindings = [
        PcuBinding::value(
            None,
            8,
            11,
            PcuBindingStorageClass::Storage,
            PcuBindingAccess::WriteOnly,
            PcuValueType::Scalar(scalar),
        ),
        PcuBinding::value(
            None,
            7,
            9,
            PcuBindingStorageClass::Storage,
            PcuBindingAccess::ReadOnly,
            PcuValueType::Scalar(scalar),
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
            binding: source,
            index: if broadcast {
                PcuDispatchIndex::BindingElementZero
            } else {
                index
            },
        }),
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
            binding: destination,
            index,
            value: PcuDispatchValueId(1),
        }),
    ];
    let direct = [
        body[0],
        body[1],
        PcuDispatchOp::Control(PcuDispatchControlOp::Return),
    ];
    let loop_ops = [
        PcuDispatchOp::GridStrideLoop {
            extent,
            body: &body,
        },
        PcuDispatchOp::Control(PcuDispatchControlOp::Return),
    ];
    run(&PcuDispatchKernelIr {
        numerical_requirements: PcuDispatchKernelIr::DEFAULT_REQUIREMENTS,
        id: PcuKernelId(1),
        entry: PcuDispatchEntryPoint {
            name: "transport",
            logical_shape: [if grid { 17 } else { extent }, 1, 1],
        },
        bindings: &bindings,
        ports: &[],
        parameters: &[],
        ops: if grid { &loop_ops } else { &direct },
        type_caps: PcuValueTypeCaps::for_scalar(scalar),
        feature_caps: PcuDispatchFeatureCaps::default(),
    });
}
#[test]
fn closed_carriers_and_noncanonical_logical_roles_need_shader_only() {
    for scalar in TYPES {
        for grid in [false, true] {
            for broadcast in [false, true] {
                with(scalar, 65, grid, broadcast, |kernel| {
                    let mut words = Vec::new();
                    let (info, profile) = lower_scalar_transport_to_spirv(
                        kernel,
                        PcuSpirvLoweringOptions::minimal_shader(),
                        &mut words,
                    )
                    .unwrap();
                    assert_eq!(info.capabilities, PcuSpirvCapabilityCaps::SHADER);
                    assert_eq!(profile.input, PcuBindingRef::new(7, 9));
                    assert_eq!(profile.output, PcuBindingRef::new(8, 11));
                    assert_eq!(profile.input_extent(), if broadcast { 1 } else { 65 });
                    assert_eq!(
                        profile.local_id(),
                        Some(if broadcast { 160 } else { 128 } + scalar as u32)
                    );
                    assert_eq!(
                        profile.dispatch_extent(),
                        Some((65 * u32::from(scalar.bit_width()) / 8).div_ceil(4))
                    );
                });
            }
        }
    }
}
#[test]
fn unsupported_profiles_and_overlarge_byte_products_write_no_words() {
    for scalar in [PcuScalarType::Bool, PcuScalarType::I4, PcuScalarType::U4] {
        with(scalar, 65, false, false, |kernel| {
            let mut words = Vec::new();
            assert!(
                lower_scalar_transport_to_spirv(
                    kernel,
                    PcuSpirvLoweringOptions::minimal_shader(),
                    &mut words
                )
                .is_err()
            );
            assert!(words.is_empty());
        });
    }
    for scalar in TYPES {
        with(scalar, u32::MAX, false, true, |kernel| {
            let mut words = Vec::new();
            let result = lower_scalar_transport_to_spirv(
                kernel,
                PcuSpirvLoweringOptions::minimal_shader(),
                &mut words,
            );
            assert_eq!(result.is_ok(), scalar.bit_width() == 8);
            if result.is_err() {
                assert!(words.is_empty());
            }
            let mut kernel = *kernel;
            kernel
                .numerical_requirements
                .numerical_options
                .reproducibility = PcuReproducibility::PortableV1;
            let mut words = Vec::new();
            assert!(
                lower_scalar_transport_to_spirv(
                    &kernel,
                    PcuSpirvLoweringOptions::minimal_shader(),
                    &mut words
                )
                .is_err()
            );
            assert!(words.is_empty());
        });
    }
}
#[test]
#[ignore = "requires official spirv-val executable"]
fn official_validator_all22_profiles_versions() {
    let mut count = 0;
    for scalar in TYPES {
        for grid in [false, true] {
            for broadcast in [false, true] {
                for version in [PcuSpirvVersion::V1_0, PcuSpirvVersion::V1_3] {
                    with(scalar, 65, grid, broadcast, |kernel| {
                        let mut words = Vec::new();
                        lower_scalar_transport_to_spirv(
                            kernel,
                            PcuSpirvLoweringOptions::minimal_shader().with_version(version),
                            &mut words,
                        )
                        .unwrap();
                        let path = std::env::temp_dir().join(format!(
                            "pcu-transport-val-{}-{count}.spv",
                            std::process::id()
                        ));
                        std::fs::write(
                            &path,
                            words
                                .iter()
                                .flat_map(|word| word.to_le_bytes())
                                .collect::<Vec<_>>(),
                        )
                        .unwrap();
                        let result = std::process::Command::new("spirv-val")
                            .args([
                                "--target-env",
                                if version == PcuSpirvVersion::V1_0 {
                                    "vulkan1.0"
                                } else {
                                    "vulkan1.1"
                                },
                            ])
                            .arg(&path)
                            .output()
                            .unwrap();
                        std::fs::remove_file(path).unwrap();
                        assert!(
                            result.status.success(),
                            "{}",
                            String::from_utf8_lossy(&result.stderr)
                        );
                        count += 1;
                    });
                }
            }
        }
    }
    assert_eq!(count, 176);
    println!("official spirv-val:{count} transport modules");
}
