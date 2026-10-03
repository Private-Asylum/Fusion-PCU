//! Exact binary profile admission, integer-only code generation and native validator proof.
#[path = "support/support.rs"]
mod support;
#[rustfmt::skip]
use fusion_pcu_core::{
    PcuDispatchFloatBinaryOp,
    PcuFloatUnderflowPolicy,
    PcuRangePolicy,
    PcuScalarType,
};
#[rustfmt::skip]
use fusion_pcu_spirv::{
    lower_checked_float_binary_to_spirv,
    PcuSpirvCapabilityCaps,
    PcuSpirvLoweringOptions,
    PcuSpirvVersion,
};

#[test]
fn unsupported_profiles_write_no_words() {
    for scalar in [PcuScalarType::F128, PcuScalarType::F256] {
        let mut graph = support::Graph::new(
            65,
            PcuDispatchFloatBinaryOp::Add,
            PcuFloatUnderflowPolicy::default(),
        );
        graph.scalar = scalar;
        graph.with(|kernel| {
            let mut words = Vec::new();
            assert!(
                lower_checked_float_binary_to_spirv(
                    kernel,
                    PcuSpirvLoweringOptions::default(),
                    &mut words
                )
                .is_err()
            );
            assert!(words.is_empty());
        });
    }
}

#[test]
fn exact_header_policies_are_required_before_any_emission() {
    for scalar in [
        PcuScalarType::F32,
        PcuScalarType::F64,
        PcuScalarType::F16,
        PcuScalarType::BF16,
        PcuScalarType::F8E4M3FN,
        PcuScalarType::F8E5M2,
    ] {
        let mut graph = support::Graph::new(
            65,
            PcuDispatchFloatBinaryOp::Add,
            PcuFloatUnderflowPolicy::IeeeAfterRounding,
        );
        graph.scalar = scalar;
        graph.with(|kernel| {
            for mismatch_range in [false, true] {
                let mut kernel = *kernel;
                if mismatch_range {
                    kernel.numerical_requirements.range_policy = PcuRangePolicy::Clamp;
                } else {
                    kernel.numerical_requirements.float_underflow =
                        PcuFloatUnderflowPolicy::RejectSubnormalResult;
                }
                let mut words = Vec::new();
                assert_eq!(
                    lower_checked_float_binary_to_spirv(
                        &kernel,
                        PcuSpirvLoweringOptions::default(),
                        &mut words
                    ),
                    Err(fusion_pcu_spirv::PcuSpirvError::UnsupportedNumericalRequirements)
                );
                assert!(words.is_empty());
            }
        });
    }
}

#[test]
#[allow(clippy::too_many_lines)] // Full six-format template/op/policy ABI audit remains one executable matrix.
fn every_template_is_integer_only_with_exact_specialization_defaults() {
    for scalar in [
        PcuScalarType::F32,
        PcuScalarType::F64,
        PcuScalarType::F16,
        PcuScalarType::BF16,
        PcuScalarType::F8E4M3FN,
        PcuScalarType::F8E5M2,
    ] {
        for op in [
            PcuDispatchFloatBinaryOp::Add,
            PcuDispatchFloatBinaryOp::Sub,
            PcuDispatchFloatBinaryOp::Mul,
            PcuDispatchFloatBinaryOp::Div,
        ] {
            for (policy_code, policy) in [
                PcuFloatUnderflowPolicy::IeeeAfterRounding,
                PcuFloatUnderflowPolicy::RejectSubnormalResult,
                PcuFloatUnderflowPolicy::AllowGradualUnderflow,
            ]
            .into_iter()
            .enumerate()
            {
                let mut graph = support::Graph::new(65, op, policy);
                graph.scalar = scalar;
                graph.grid = true;
                graph.broadcast = [true, false];
                graph.operands = [2, 2];
                graph.with(|kernel| {
                    let mut words = Vec::new();
                    let (info, profile) = lower_checked_float_binary_to_spirv(
                        kernel,
                        PcuSpirvLoweringOptions::default(),
                        &mut words,
                    )
                    .unwrap();
                    assert_eq!(info.word_count, words.len());
                    assert_eq!(info.capabilities, PcuSpirvCapabilityCaps::SHADER);
                    assert_eq!(profile.extent, 65);
                    assert_eq!(profile.scalar, scalar);
                    assert_eq!(profile.element_bytes(), usize::from(scalar.bit_width() / 8));
                    assert_eq!(profile.operands, [1, 1]);
                    assert_eq!(profile.broadcast, [false, false]);
                    assert_eq!(profile.input_extents, [1, 65]);
                    let mut ids = [0; 8];
                    let mut defaults = [0; 8];
                    let mut cursor = 5;
                    while cursor < words.len() {
                        let instruction = &words[cursor..];
                        let opcode = instruction[0] & 0xffff;
                        let count = (instruction[0] >> 16) as usize;
                        assert_ne!(opcode, 22, "no OpTypeFloat");
                        if opcode == 21 {
                            assert_eq!(instruction[2], 32, "only U32/I32 integer widths");
                        }
                        assert!(
                            ![129, 131, 133, 136].contains(&opcode),
                            "no floating arithmetic"
                        );
                        if opcode == 17 {
                            assert_eq!(instruction[1], 1, "Shader is the only capability");
                        }
                        if opcode == 71 && count == 4 && instruction[2] == 1 {
                            ids[instruction[3] as usize] = instruction[1];
                        }
                        if opcode == 50
                            && let Some(slot) = ids.iter().position(|id| *id == instruction[2])
                        {
                            defaults[slot] = instruction[3];
                        }
                        cursor += count;
                    }
                    let op_code = match op {
                        PcuDispatchFloatBinaryOp::Add => 0,
                        PcuDispatchFloatBinaryOp::Sub => 1,
                        PcuDispatchFloatBinaryOp::Mul => 2,
                        PcuDispatchFloatBinaryOp::Div => 3,
                    };
                    assert_eq!(
                        defaults[..6],
                        [op_code, u32::try_from(policy_code).unwrap(), 1, 1, 0, 0]
                    );
                    if profile.packing_lanes() > 1 {
                        assert_eq!(
                            defaults[6],
                            match scalar {
                                PcuScalarType::F16 => 0,
                                PcuScalarType::BF16 => 1,
                                PcuScalarType::F8E4M3FN => 2,
                                PcuScalarType::F8E5M2 => 3,
                                _ => unreachable!(),
                            }
                        );
                        assert_eq!(defaults[7], 65);
                        assert_eq!(
                            profile.dispatch_extent(),
                            65u32.div_ceil(profile.packing_lanes())
                        );
                    }
                });
            }
        }
    }
}

#[test]
#[ignore = "requires installed SPIR-V Tools"]
fn native_validator_accepts_all_operations_policies_versions_and_layouts() {
    let directory = std::env::temp_dir().join(format!("pcu-spirv-binary-{}", std::process::id()));
    std::fs::create_dir_all(&directory).unwrap();
    let mut serial = 0;
    for scalar in [
        PcuScalarType::F32,
        PcuScalarType::F64,
        PcuScalarType::F16,
        PcuScalarType::BF16,
        PcuScalarType::F8E4M3FN,
        PcuScalarType::F8E5M2,
    ] {
        for op in [
            PcuDispatchFloatBinaryOp::Add,
            PcuDispatchFloatBinaryOp::Sub,
            PcuDispatchFloatBinaryOp::Mul,
            PcuDispatchFloatBinaryOp::Div,
        ] {
            for policy in [
                PcuFloatUnderflowPolicy::IeeeAfterRounding,
                PcuFloatUnderflowPolicy::RejectSubnormalResult,
                PcuFloatUnderflowPolicy::AllowGradualUnderflow,
            ] {
                for version in [PcuSpirvVersion::V1_0, PcuSpirvVersion::V1_3] {
                    for grid in [false, true] {
                        let mut graph = support::Graph::new(257, op, policy);
                        graph.scalar = scalar;
                        for range in [PcuRangePolicy::Reject, PcuRangePolicy::Clamp] {
                            graph.range = range;
                            graph.grid = grid;
                            graph.broadcast = [grid, !grid];
                            graph.operands = if grid { [2, 1] } else { [1, 1] };
                            graph.with(|kernel| {
                                let mut words = Vec::new();
                                lower_checked_float_binary_to_spirv(
                                    kernel,
                                    PcuSpirvLoweringOptions::default().with_version(version),
                                    &mut words,
                                )
                                .unwrap();
                                let file = directory.join(format!("{serial}.spv"));
                                let bytes: Vec<u8> =
                                    words.iter().flat_map(|word| word.to_le_bytes()).collect();
                                std::fs::write(&file, bytes).unwrap();
                                let result = std::process::Command::new("spirv-val")
                                    .args([
                                        "--target-env",
                                        if version == PcuSpirvVersion::V1_0 {
                                            "vulkan1.0"
                                        } else {
                                            "vulkan1.1"
                                        },
                                    ])
                                    .arg(&file)
                                    .output()
                                    .unwrap();
                                assert!(
                                    result.status.success(),
                                    "{}",
                                    String::from_utf8_lossy(&result.stderr)
                                );
                                serial += 1;
                            });
                        }
                    }
                }
            }
        }
    }
    assert_eq!(serial, 576);
    std::fs::remove_dir_all(directory).unwrap();
}
#[test]
fn portable_metadata_rejects_before_any_bytecode_and_permissions_keep_exact_words() {
    for op in [
        PcuDispatchFloatBinaryOp::Add,
        PcuDispatchFloatBinaryOp::Sub,
        PcuDispatchFloatBinaryOp::Mul,
        PcuDispatchFloatBinaryOp::Div,
    ] {
        support::Graph::new(7, op, PcuFloatUnderflowPolicy::default()).with(|kernel| {
            let mut reference = Vec::new();
            lower_checked_float_binary_to_spirv(
                kernel,
                PcuSpirvLoweringOptions::default(),
                &mut reference,
            )
            .unwrap();
            let mut permitted = *kernel;
            permitted
                .numerical_requirements
                .numerical_options
                .compound_arithmetic = fusion_pcu_core::PcuCompoundArithmeticPolicy::BackendDefined;
            permitted.numerical_requirements.numerical_options.precision =
                fusion_pcu_core::PcuPrecisionPolicy::BackendOptimized;
            let mut words = Vec::new();
            lower_checked_float_binary_to_spirv(
                &permitted,
                PcuSpirvLoweringOptions::default(),
                &mut words,
            )
            .unwrap();
            assert_eq!(words, reference);
            permitted
                .numerical_requirements
                .numerical_options
                .reproducibility = fusion_pcu_core::PcuReproducibility::PortableV1;
            let mut words = Vec::new();
            assert_eq!(
                lower_checked_float_binary_to_spirv(
                    &permitted,
                    PcuSpirvLoweringOptions::default(),
                    &mut words
                ),
                Err(fusion_pcu_spirv::PcuSpirvError::UnsupportedNumericalRequirements)
            );
            assert!(words.is_empty());
            assert_eq!(
                fusion_pcu_spirv::lower_dispatch_to_spirv(
                    &permitted,
                    PcuSpirvLoweringOptions::default(),
                    &mut words
                ),
                Err(fusion_pcu_spirv::PcuSpirvError::UnsupportedNumericalRequirements)
            );
            assert!(words.is_empty());
            assert_eq!(
                fusion_pcu_spirv::validate_float_bit_map(&permitted),
                Err(fusion_pcu_spirv::PcuSpirvError::UnsupportedNumericalRequirements)
            );
        });
    }
}

#[test]
fn clamp_retains_same_integer_only_module_with_explicit_publication_profile() {
    for op in [
        PcuDispatchFloatBinaryOp::Add,
        PcuDispatchFloatBinaryOp::Sub,
        PcuDispatchFloatBinaryOp::Mul,
        PcuDispatchFloatBinaryOp::Div,
    ] {
        let mut normal = Vec::new();
        let mut clamp = Vec::new();
        let mut graph = support::Graph::new(7, op, PcuFloatUnderflowPolicy::default());
        graph.with(|kernel| {
            lower_checked_float_binary_to_spirv(
                kernel,
                PcuSpirvLoweringOptions::default(),
                &mut normal,
            )
            .unwrap()
        });
        graph.range = PcuRangePolicy::Clamp;
        graph.with(|kernel| {
            let (_, profile) = lower_checked_float_binary_to_spirv(
                kernel,
                PcuSpirvLoweringOptions::default(),
                &mut clamp,
            )
            .unwrap();
            assert_eq!(profile.range, PcuRangePolicy::Clamp);
        });
        assert_eq!(normal, clamp);
    }
}

#[test]
fn qualified_low_portable_descriptor_keeps_identical_u32_words_and_closed_negatives() {
    for scalar in [
        PcuScalarType::F16,
        PcuScalarType::BF16,
        PcuScalarType::F8E4M3FN,
        PcuScalarType::F8E5M2,
    ] {
        for op in [
            PcuDispatchFloatBinaryOp::Add,
            PcuDispatchFloatBinaryOp::Sub,
            PcuDispatchFloatBinaryOp::Mul,
            PcuDispatchFloatBinaryOp::Div,
        ] {
            for policy in [
                PcuFloatUnderflowPolicy::IeeeAfterRounding,
                PcuFloatUnderflowPolicy::RejectSubnormalResult,
                PcuFloatUnderflowPolicy::AllowGradualUnderflow,
            ] {
                for grid in [false, true] {
                    let mut graph = support::Graph::new(7, op, policy);
                    graph.scalar = scalar;
                    graph.grid = grid;
                    graph.reverse_loads = grid;
                    graph.broadcast = [grid, !grid];
                    graph.operands = [2, 1];
                    graph.with(|kernel| {
                        let mut normal = Vec::new();
                        lower_checked_float_binary_to_spirv(
                            kernel,
                            PcuSpirvLoweringOptions::default(),
                            &mut normal,
                        )
                        .unwrap();
                        let mut portable = *kernel;
                        portable
                            .numerical_requirements
                            .numerical_options
                            .reproducibility = fusion_pcu_core::PcuReproducibility::PortableV1;
                        fusion_pcu_core::describe_portable_v1_map(&portable).unwrap();
                        for mode in [
                            fusion_pcu_core::PcuNumericalMode::Boundary,
                            fusion_pcu_core::PcuNumericalMode::Strict,
                        ] {
                            portable.numerical_requirements.numerical_mode = mode;
                            portable
                                .numerical_requirements
                                .numerical_options
                                .compound_arithmetic =
                                fusion_pcu_core::PcuCompoundArithmeticPolicy::BackendDefined;
                            portable.numerical_requirements.numerical_options.precision =
                                fusion_pcu_core::PcuPrecisionPolicy::BackendOptimized;
                            let mut words = Vec::new();
                            lower_checked_float_binary_to_spirv(
                                &portable,
                                PcuSpirvLoweringOptions::default(),
                                &mut words,
                            )
                            .unwrap();
                            assert_eq!(words, normal);
                        }
                        portable.numerical_requirements.range_policy = PcuRangePolicy::Clamp;
                        let mut words = Vec::new();
                        assert_eq!(
                            lower_checked_float_binary_to_spirv(
                                &portable,
                                PcuSpirvLoweringOptions::default(),
                                &mut words
                            ),
                            Err(fusion_pcu_spirv::PcuSpirvError::UnsupportedNumericalRequirements)
                        );
                        assert!(words.is_empty());
                    });
                }
            }
        }
    }
}

#[test]
#[ignore = "requires installed SPIR-V Tools"]
fn native_validator_accepts_requested_portable_low_maps() {
    let directory = std::env::temp_dir().join(format!("pcu-spirv-portable-{}", std::process::id()));
    std::fs::create_dir_all(&directory).unwrap();
    let mut serial = 0;
    for scalar in [
        PcuScalarType::F16,
        PcuScalarType::BF16,
        PcuScalarType::F8E4M3FN,
        PcuScalarType::F8E5M2,
    ] {
        for op in [
            PcuDispatchFloatBinaryOp::Add,
            PcuDispatchFloatBinaryOp::Sub,
            PcuDispatchFloatBinaryOp::Mul,
            PcuDispatchFloatBinaryOp::Div,
        ] {
            for policy in [
                PcuFloatUnderflowPolicy::IeeeAfterRounding,
                PcuFloatUnderflowPolicy::RejectSubnormalResult,
                PcuFloatUnderflowPolicy::AllowGradualUnderflow,
            ] {
                for version in [PcuSpirvVersion::V1_0, PcuSpirvVersion::V1_3] {
                    for grid in [false, true] {
                        let mut graph = support::Graph::new(257, op, policy);
                        graph.scalar = scalar;
                        {
                            let range = PcuRangePolicy::Reject;
                            graph.range = range;
                            graph.grid = grid;
                            graph.broadcast = [grid, !grid];
                            graph.operands = if grid { [2, 1] } else { [1, 1] };
                            graph.with(|kernel| {
                                let mut portable = *kernel;
                                portable
                                    .numerical_requirements
                                    .numerical_options
                                    .reproducibility =
                                    fusion_pcu_core::PcuReproducibility::PortableV1;
                                let kernel = &portable;
                                fusion_pcu_core::describe_portable_v1_map(kernel).unwrap();
                                let mut words = Vec::new();
                                lower_checked_float_binary_to_spirv(
                                    kernel,
                                    PcuSpirvLoweringOptions::default().with_version(version),
                                    &mut words,
                                )
                                .unwrap();
                                let file = directory.join(format!("{serial}.spv"));
                                let bytes: Vec<u8> =
                                    words.iter().flat_map(|word| word.to_le_bytes()).collect();
                                std::fs::write(&file, bytes).unwrap();
                                let result = std::process::Command::new("spirv-val")
                                    .args([
                                        "--target-env",
                                        if version == PcuSpirvVersion::V1_0 {
                                            "vulkan1.0"
                                        } else {
                                            "vulkan1.1"
                                        },
                                    ])
                                    .arg(&file)
                                    .output()
                                    .unwrap();
                                assert!(
                                    result.status.success(),
                                    "{}",
                                    String::from_utf8_lossy(&result.stderr)
                                );
                                serial += 1;
                            });
                        }
                    }
                }
            }
        }
    }
    assert_eq!(serial, 192);
    std::fs::remove_dir_all(directory).unwrap();
}
