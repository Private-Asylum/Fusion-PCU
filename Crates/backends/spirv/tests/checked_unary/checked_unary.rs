//! Exact six-format integer-only unary admission and offline validator matrix.
extern crate fusion_pcu_core as pcu_facade;
#[path = "../../../cpu/tests/low_unary/graph/graph.rs"]
mod graph;
#[rustfmt::skip]
use fusion_pcu_core::{PcuDispatchFloatUnaryOp,PcuFloatUnderflowPolicy,PcuRangePolicy,PcuScalarType,PcuReproducibility};
#[rustfmt::skip]
use fusion_pcu_spirv::{lower_checked_float_unary_to_spirv,PcuSpirvCapabilityCaps,PcuSpirvLoweringOptions,PcuSpirvVersion};
const TYPES: [PcuScalarType; 6] = [
    PcuScalarType::F16,
    PcuScalarType::BF16,
    PcuScalarType::F8E4M3FN,
    PcuScalarType::F8E5M2,
    PcuScalarType::F32,
    PcuScalarType::F64,
];
const OPS: [PcuDispatchFloatUnaryOp; 2] =
    [PcuDispatchFloatUnaryOp::Neg, PcuDispatchFloatUnaryOp::Relu];
const POLICIES: [PcuFloatUnderflowPolicy; 3] = [
    PcuFloatUnderflowPolicy::IeeeAfterRounding,
    PcuFloatUnderflowPolicy::RejectSubnormalResult,
    PcuFloatUnderflowPolicy::AllowGradualUnderflow,
];
#[test]
fn portable_header_mismatches_write_no_words() {
    for scalar in TYPES {
        for op in OPS {
            let g = graph::Graph::new(
                scalar,
                65,
                op,
                PcuFloatUnderflowPolicy::default(),
                PcuRangePolicy::Reject,
            );
            g.with(|kernel| {
                for mismatch in 0..2 {
                    let mut k = *kernel;
                    k.numerical_requirements.numerical_options.reproducibility =
                        PcuReproducibility::PortableV1;
                    match mismatch {
                        0 => k.numerical_requirements.range_policy = PcuRangePolicy::Clamp,
                        1 => {
                            k.numerical_requirements.float_underflow =
                                PcuFloatUnderflowPolicy::RejectSubnormalResult;
                        }
                        _ => {
                            k.numerical_requirements.numerical_options.reproducibility =
                                PcuReproducibility::PortableV1;
                        }
                    }
                    let mut words = Vec::new();
                    assert!(
                        lower_checked_float_unary_to_spirv(
                            &k,
                            PcuSpirvLoweringOptions::default(),
                            &mut words
                        )
                        .is_err()
                    );
                    assert!(words.is_empty());
                }
            });
        }
    }
}
#[test]
fn integer_only_exact_defaults_and_disjoint_ids() {
    for (format, scalar) in TYPES.into_iter().enumerate() {
        for (operation, op) in OPS.into_iter().enumerate() {
            for (policy, uf) in POLICIES.into_iter().enumerate() {
                for (range_code, range) in [PcuRangePolicy::Reject, PcuRangePolicy::Clamp]
                    .into_iter()
                    .enumerate()
                {
                    let mut g = graph::Graph::new(scalar, 65, op, uf, range);
                    g.grid = true;
                    g.broadcast = true;
                    g.with(|k| {
                        let mut words = Vec::new();
                        let (info, p) = lower_checked_float_unary_to_spirv(
                            k,
                            PcuSpirvLoweringOptions::default(),
                            &mut words,
                        )
                        .unwrap();
                        assert_eq!(info.capabilities, PcuSpirvCapabilityCaps::SHADER);
                        assert_eq!(
                            p.local_id(),
                            Some(
                                96 + u32::try_from(format * 4 + range_code * 2 + operation)
                                    .unwrap()
                            )
                        );
                        assert_eq!(p.input_extent(), 1);
                        assert_eq!(p.dispatch_extent(), 65u32.div_ceil(p.packing_lanes()));
                        let mut ids = [0u32; 5];
                        let mut defaults = [0u32; 5];
                        let mut cursor = 5;
                        while cursor < words.len() {
                            let i = &words[cursor..];
                            let code = i[0] & 65535;
                            let n = usize::try_from(i[0] >> 16).unwrap();
                            assert_ne!(code, 22, "no floating types");
                            if code == 21 {
                                assert_eq!(i[2], 32, "U32/I32 only");
                            }
                            if code == 17 {
                                assert_eq!(i[1], 1, "only Shader capability");
                            }
                            if code == 71 && n == 4 && i[2] == 1 {
                                ids[usize::try_from(i[3]).unwrap()] = i[1];
                            }
                            if code == 50
                                && let Some(slot) = ids.iter().position(|id| *id == i[2])
                            {
                                defaults[slot] = i[3];
                            }
                            cursor += n;
                        }
                        assert_eq!(
                            defaults,
                            [
                                u32::try_from(format).unwrap(),
                                u32::try_from(operation).unwrap(),
                                u32::try_from(policy).unwrap(),
                                65,
                                1
                            ]
                        );
                    });
                }
            }
        }
    }
}
fn validate_modules<const PORTABLE: bool>() {
    let dir =
        std::env::temp_dir().join(format!("pcu-spirv-unary-{PORTABLE}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let mut count = 0;
    for scalar in TYPES {
        for op in OPS {
            for uf in POLICIES {
                for range in [PcuRangePolicy::Reject, PcuRangePolicy::Clamp] {
                    for grid in [false, true] {
                        for broadcast in [false, true] {
                            for version in [PcuSpirvVersion::V1_0, PcuSpirvVersion::V1_3] {
                                let mut g = graph::Graph::new(scalar, 65, op, uf, range);
                                g.grid = grid;
                                g.broadcast = broadcast;
                                g.with(|k| {
                                    let mut k = *k;
                                    if PORTABLE {
                                        k.numerical_requirements
                                            .numerical_options
                                            .reproducibility = PcuReproducibility::PortableV1;
                                    }
                                    let k = &k;
                                    let mut words = Vec::new();
                                    lower_checked_float_unary_to_spirv(
                                        k,
                                        PcuSpirvLoweringOptions::default().with_version(version),
                                        &mut words,
                                    )
                                    .unwrap();
                                    let file = dir.join(format!("{count}.spv"));
                                    std::fs::write(
                                        &file,
                                        words
                                            .iter()
                                            .flat_map(|w| w.to_le_bytes())
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
                                        .arg(&file)
                                        .output()
                                        .unwrap();
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
            }
        }
    }
    assert_eq!(count, 576);
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn portable_exact_profile_keeps_all_normal_shader_words() {
    let mut profiles = 0;
    for (format, scalar) in TYPES.into_iter().enumerate() {
        for (operation, op) in OPS.into_iter().enumerate() {
            for uf in POLICIES {
                for (range_code, range) in [PcuRangePolicy::Reject, PcuRangePolicy::Clamp]
                    .into_iter()
                    .enumerate()
                {
                    for grid in [false, true] {
                        for broadcast in [false, true] {
                            for version in [PcuSpirvVersion::V1_0, PcuSpirvVersion::V1_3] {
                                let mut graph = graph::Graph::new(scalar, 65, op, uf, range);
                                graph.grid = grid;
                                graph.broadcast = broadcast;
                                graph.with(|kernel| {
                                    let options =
                                        PcuSpirvLoweringOptions::default().with_version(version);
                                    let mut normal = Vec::new();
                                    let (_, profile) = lower_checked_float_unary_to_spirv(
                                        kernel,
                                        options,
                                        &mut normal,
                                    )
                                    .unwrap();
                                    assert!(!profile.portable);
                                    let mut portable = *kernel;
                                    portable
                                        .numerical_requirements
                                        .numerical_options
                                        .reproducibility = PcuReproducibility::PortableV1;
                                    let descriptor =
                                        fusion_pcu_core::describe_portable_v1_unary_map(&portable)
                                            .unwrap();
                                    let mut words = Vec::new();
                                    let (_, profile) = lower_checked_float_unary_to_spirv(
                                        &portable, options, &mut words,
                                    )
                                    .unwrap();
                                    assert!(profile.portable);
                                    assert_eq!(
                                        descriptor.requirements,
                                        portable.numerical_requirements
                                    );
                                    assert_eq!(
                                        profile.local_id(),
                                        Some(
                                            16640
                                                + u32::try_from(
                                                    format * 4 + range_code * 2 + operation
                                                )
                                                .unwrap()
                                        )
                                    );
                                    assert_eq!(
                                        words, normal,
                                        "the original proven integer shader is unchanged"
                                    );
                                    profiles += 1;
                                });
                            }
                        }
                    }
                }
            }
        }
    }
    assert_eq!(profiles, 576);
}

#[test]
#[ignore = "requires installed SPIR-V Tools"]
fn native_validator_accepts_576_exact_profiles() {
    validate_modules::<false>();
}
#[test]
#[ignore = "requires installed SPIR-V Tools"]
fn native_validator_accepts_576_portable_profiles() {
    validate_modules::<true>();
}
