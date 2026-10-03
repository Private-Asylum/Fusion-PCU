//! Exact fourteen-width lowerer schema, closed headers and official SPIR-V validation.
#[path = "graph/graph.rs"]
mod graph;
#[rustfmt::skip]
use fusion_pcu_core::{PcuScalarType,PcuRangePolicy,PcuReproducibility};
#[rustfmt::skip]
use fusion_pcu_spirv::{validate_checked_div_rem_map,lower_checked_div_rem_to_spirv,PcuSpirvLoweringOptions,PcuSpirvVersion,PcuSpirvCapabilityCaps};
const TYPES: [PcuScalarType; 14] = [
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
];
#[test]
fn exact_profiles_routing_and_closed_policies() {
    for (ordinal, scalar) in TYPES.into_iter().enumerate() {
        for grid in [false, true] {
            for routing in 0..3 {
                let mut g = graph::Graph::new(scalar, 65);
                g.grid = grid;
                g.routing = routing;
                g.with(|kernel| {
                    let mut words = Vec::new();
                    let (info, profile) = lower_checked_div_rem_to_spirv(
                        kernel,
                        PcuSpirvLoweringOptions::minimal_shader(),
                        &mut words,
                    )
                    .unwrap();
                    assert_eq!(
                        profile.local_id(),
                        Some(608 + u32::try_from(ordinal).unwrap())
                    );
                    assert_eq!(profile.inputs, [graph::INPUTS[1], graph::INPUTS[0]]);
                    assert_eq!(profile.outputs, graph::OUTPUTS);
                    assert_eq!(
                        profile.operands,
                        match routing {
                            0 => [1, 0],
                            1 => [0, 1],
                            _ => [1, 1],
                        }
                    );
                    assert_eq!(info.capabilities, PcuSpirvCapabilityCaps::SHADER);
                    assert!(profile.logical_bytes().is_some());
                    let mut bad = *kernel;
                    bad.numerical_requirements.range_policy = PcuRangePolicy::Clamp;
                    let mut empty = Vec::new();
                    assert!(
                        lower_checked_div_rem_to_spirv(
                            &bad,
                            PcuSpirvLoweringOptions::minimal_shader(),
                            &mut empty
                        )
                        .is_err()
                    );
                    assert!(empty.is_empty());
                    bad.numerical_requirements.range_policy = PcuRangePolicy::Reject;
                    bad.numerical_requirements.numerical_options.reproducibility =
                        PcuReproducibility::PortableV1;
                    let portable = validate_checked_div_rem_map(&bad).unwrap();
                    assert!(portable.portable);
                    assert_eq!(
                        portable.local_id(),
                        Some(12544 + u32::try_from(ordinal).unwrap())
                    );
                });
            }
        }
    }
    for scalar in [
        PcuScalarType::Bool,
        PcuScalarType::F128,
        PcuScalarType::F256,
        PcuScalarType::F32,
    ] {
        graph::Graph::new(scalar, 1).with(|k| assert!(validate_checked_div_rem_map(k).is_err()));
    }
}
#[test]
#[ignore = "requires official spirv-val executable"]
fn official_validator_all_profiles() {
    let dir = std::env::temp_dir().join(format!("pcu-div-rem-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("module.spv");
    let mut count = 0;
    for scalar in TYPES {
        for grid in [false, true] {
            for routing in 0..3 {
                for version in [PcuSpirvVersion::V1_0, PcuSpirvVersion::V1_3] {
                    let mut g = graph::Graph::new(scalar, 65);
                    g.grid = grid;
                    g.routing = routing;
                    g.with(|kernel| {
                        let mut words = Vec::new();
                        lower_checked_div_rem_to_spirv(
                            kernel,
                            PcuSpirvLoweringOptions {
                                version,
                                ..PcuSpirvLoweringOptions::minimal_shader()
                            },
                            &mut words,
                        )
                        .unwrap();
                        let bytes = words
                            .iter()
                            .flat_map(|word| word.to_le_bytes())
                            .collect::<Vec<_>>();
                        std::fs::write(&path, bytes).unwrap();
                        let status = std::process::Command::new("spirv-val")
                            .args([
                                "--target-env",
                                if version == PcuSpirvVersion::V1_0 {
                                    "vulkan1.0"
                                } else {
                                    "vulkan1.1"
                                },
                            ])
                            .arg(&path)
                            .status()
                            .unwrap();
                        assert!(status.success());
                        count += 1;
                    });
                }
            }
        }
    }
    assert_eq!(count, 168);
    std::fs::remove_file(path).unwrap();
    std::fs::remove_dir(dir).unwrap();
}

#[test]
fn wide_byte_dimensions_and_shader_only_capabilities() {
    for (scalar, bytes) in [
        (PcuScalarType::I128, 16),
        (PcuScalarType::U128, 16),
        (PcuScalarType::I256, 32),
        (PcuScalarType::U256, 32),
        (PcuScalarType::I512, 64),
        (PcuScalarType::U512, 64),
    ] {
        graph::Graph::new(scalar, 65).with(|kernel| {
            let mut words = Vec::new();
            let (_, profile) = lower_checked_div_rem_to_spirv(
                kernel,
                PcuSpirvLoweringOptions::minimal_shader(),
                &mut words,
            )
            .unwrap();
            assert_eq!(profile.element_bytes(), bytes);
            assert_eq!(
                profile.logical_bytes(),
                Some(u32::try_from(bytes * 65).unwrap())
            );
            assert_eq!(
                (profile.packing_lanes(), profile.dispatch_extent()),
                (1, 65)
            );
            let mut cursor = 5;
            let mut capabilities = Vec::new();
            while cursor < words.len() {
                if words[cursor] & 0xffff == 17 {
                    capabilities.push(words[cursor + 1]);
                }
                cursor += usize::try_from(words[cursor] >> 16).unwrap();
            }
            assert_eq!(capabilities, [1], "only Shader; no native Int64/Float64");
        });
        for extent in [0, u32::MAX] {
            graph::Graph::new(scalar, extent).with(|kernel| {
                let mut empty = Vec::new();
                assert!(
                    lower_checked_div_rem_to_spirv(
                        kernel,
                        PcuSpirvLoweringOptions::minimal_shader(),
                        &mut empty
                    )
                    .is_err()
                );
                assert!(empty.is_empty(), "invalid dimension writes no module words");
            });
        }
    }
}
