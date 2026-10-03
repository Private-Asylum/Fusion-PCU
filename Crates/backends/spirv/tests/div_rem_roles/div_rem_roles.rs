//! New operand routes retain all declarations and specialize exact read spans before lowering.
#[path = "graph/graph.rs"]
mod graph;
#[rustfmt::skip]
use fusion_pcu_core::{PcuScalarType,PcuRangePolicy};
#[rustfmt::skip]
use fusion_pcu_spirv::{lower_checked_div_rem_to_spirv,PcuSpirvLoweringOptions,PcuSpirvVersion,PcuSpirvCapabilityCaps};
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
fn exact_roles_spans_ids_and_rejection_before_words() {
    for (ordinal, scalar) in TYPES.into_iter().enumerate() {
        for kind in 0..6 {
            let mut g = graph::Graph::new(scalar, 65);
            g.kind = kind;
            g.grid = kind == 4;
            g.with(|kernel| {
                let mut words = Vec::new();
                let (info, p) = lower_checked_div_rem_to_spirv(
                    kernel,
                    PcuSpirvLoweringOptions::minimal_shader(),
                    &mut words,
                )
                .unwrap();
                assert_eq!(p.local_id(), Some(8448 + u32::try_from(ordinal).unwrap()));
                assert!(p.operand_profile);
                assert_eq!(p.extent, 65);
                assert_eq!(p.outputs, graph::OUTPUTS);
                assert_eq!(p.input_count, if kind == 2 || kind == 5 { 2 } else { 1 });
                assert_eq!(
                    p.input_extents,
                    if kind == 5 {
                        [1, 1]
                    } else if kind == 2 {
                        [65, 65]
                    } else {
                        [65, 0]
                    }
                );
                assert_eq!(
                    p.operands,
                    if kind == 2 || kind == 5 {
                        [0, 1]
                    } else {
                        [0, 0]
                    }
                );
                assert_eq!(
                    p.operand_broadcast,
                    [kind == 4 || kind == 5, kind == 3 || kind == 5]
                );
                assert_eq!(p.declaration_count, kernel.bindings.len());
                for (original, frozen) in kernel.bindings.iter().zip(p.declarations) {
                    assert_eq!(original.reference(), frozen);
                }
                assert_eq!(info.capabilities, PcuSpirvCapabilityCaps::SHADER);
                let mut cursor = 5;
                let mut capabilities = Vec::new();
                while cursor < words.len() {
                    if words[cursor] & 0xffff == 17 {
                        capabilities.push(words[cursor + 1]);
                    }
                    cursor += usize::try_from(words[cursor] >> 16).unwrap();
                }
                assert_eq!(capabilities, [1]);
                for invalid in 0..3 {
                    let mut bad = *kernel;
                    match invalid {
                        0 => bad.numerical_requirements.range_policy = PcuRangePolicy::Clamp,
                        1 => bad.entry.logical_shape[1] = 2,
                        _ => bad.bindings = &[],
                    }
                    let mut untouched = vec![0xfeed_beef];
                    assert!(
                        lower_checked_div_rem_to_spirv(
                            &bad,
                            PcuSpirvLoweringOptions::minimal_shader(),
                            &mut untouched
                        )
                        .is_err()
                    );
                    assert_eq!(untouched, [0xfeed_beef]);
                }
            });
        }
    }
}
#[test]
#[ignore = "requires official SPIR-V Tools"]
fn official_validator_all_role_profiles() {
    let dir = std::env::temp_dir().join(format!("pcu-div-rem-roles-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("module.spv");
    let mut count = 0;
    for scalar in TYPES {
        for kind in 0..6 {
            for version in [PcuSpirvVersion::V1_0, PcuSpirvVersion::V1_3] {
                let mut g = graph::Graph::new(scalar, 65);
                g.kind = kind;
                g.grid = kind == 4;
                g.with(|kernel| {
                    let mut words = Vec::new();
                    lower_checked_div_rem_to_spirv(
                        kernel,
                        PcuSpirvLoweringOptions::minimal_shader().with_version(version),
                        &mut words,
                    )
                    .unwrap();
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
    assert_eq!(count, 168);
    std::fs::remove_dir_all(dir).unwrap();
}
