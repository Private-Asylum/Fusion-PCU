//! Actual-read specialization keeps original declarations while indexing each mathematical operand.
#[path = "graph/graph.rs"]
mod graph;
#[rustfmt::skip]
use fusion_pcu_core::{
    PcuRangePolicy,
    PcuReproducibility,
    PcuScalarType,
};
#[rustfmt::skip]
use fusion_pcu_spirv::{
    lower_checked_integer_to_spirv,
    PcuSpirvLoweringOptions,
    PcuSpirvVersion,
};
const FORMATS: [PcuScalarType; 14] = [
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
fn repeated_read_aliases_one_input_and_preserves_typed_metadata() {
    for (scalar, range) in FORMATS
        .into_iter()
        .flat_map(|s| [PcuRangePolicy::Reject, PcuRangePolicy::Clamp].map(|r| (s, r)))
    {
        for schema in 0..5 {
            graph::with_range(scalar, 65, schema, range, |kernel| {
                let mut words = Vec::new();
                let (_, profile) = lower_checked_integer_to_spirv(
                    kernel,
                    PcuSpirvLoweringOptions::default(),
                    &mut words,
                )
                .unwrap();
                assert_eq!(profile.input_count, 1);
                assert_eq!(profile.inputs, [graph::INPUT; 2]);
                assert_eq!(profile.input_extents, [65, 0]);
                assert!(profile.operand_profile);
                assert!(profile.local_id().unwrap() >= 2048);
                assert_eq!(profile.operands, [0, 0]);
                assert_eq!(
                    profile.operand_broadcast,
                    if schema == 4 {
                        [true, false]
                    } else {
                        [false, schema == 3]
                    }
                );
                assert_eq!(profile.output, graph::OUTPUT);
                assert_eq!(profile.declaration_count, kernel.bindings.len());
                for (original, frozen) in kernel.bindings.iter().zip(profile.declarations) {
                    assert_eq!(original.reference(), frozen);
                }
                assert!(!words.is_empty());
                // Portable is a separate cold profile using the identical exact shader realization.
                let mut portable = *kernel;
                portable
                    .numerical_requirements
                    .numerical_options
                    .reproducibility = PcuReproducibility::PortableV1;
                let mut portable_words = Vec::new();
                let (_, admitted) = lower_checked_integer_to_spirv(
                    &portable,
                    PcuSpirvLoweringOptions::default(),
                    &mut portable_words,
                )
                .unwrap();
                assert!(admitted.portable);
                assert_eq!(admitted.local_id(), profile.local_id().map(|id| id + 2048));
                assert_eq!(portable_words, words);
            });
        }
    }
}
#[test]
#[ignore = "requires installed SPIR-V Tools"]
fn official_validator_accepts_every_operand_specialization() {
    let directory = std::env::temp_dir().join(format!("pcu-spirv-operands-{}", std::process::id()));
    std::fs::create_dir_all(&directory).unwrap();
    let mut serial = 0;
    for (scalar, range) in FORMATS
        .into_iter()
        .flat_map(|s| [PcuRangePolicy::Reject, PcuRangePolicy::Clamp].map(|r| (s, r)))
    {
        for schema in 0..5 {
            for version in [PcuSpirvVersion::V1_0, PcuSpirvVersion::V1_3] {
                graph::with_range(scalar, 65, schema, range, |kernel| {
                    for reproducibility in [
                        PcuReproducibility::Unspecified,
                        PcuReproducibility::PortableV1,
                    ] {
                        let mut kernel = *kernel;
                        kernel
                            .numerical_requirements
                            .numerical_options
                            .reproducibility = reproducibility;
                        let mut words = Vec::new();
                        lower_checked_integer_to_spirv(
                            &kernel,
                            PcuSpirvLoweringOptions::default().with_version(version),
                            &mut words,
                        )
                        .unwrap();
                        let file = directory.join(format!("{serial}.spv"));
                        std::fs::write(
                            &file,
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
                            .arg(file)
                            .output()
                            .unwrap();
                        assert!(
                            result.status.success(),
                            "{}",
                            String::from_utf8_lossy(&result.stderr)
                        );
                        serial += 1;
                    }
                });
            }
        }
    }
    assert_eq!(serial, 560);
    std::fs::remove_dir_all(directory).unwrap();
}

#[test]
fn portable_nonmember_refusal_writes_no_words() {
    graph::with_range(
        PcuScalarType::U512,
        65,
        0,
        PcuRangePolicy::Reject,
        |kernel| {
            for invalid in 0..3 {
                let mut candidate = *kernel;
                candidate
                    .numerical_requirements
                    .numerical_options
                    .reproducibility = PcuReproducibility::PortableV1;
                match invalid {
                    0 => candidate.entry.logical_shape[1] = 2,
                    1 => candidate.numerical_requirements.range_policy = PcuRangePolicy::Clamp,
                    _ => candidate.ops = &[],
                }
                let mut words = vec![0xfeed_beef];
                assert!(
                    lower_checked_integer_to_spirv(
                        &candidate,
                        PcuSpirvLoweringOptions::default(),
                        &mut words
                    )
                    .is_err()
                );
                assert_eq!(words, [0xfeed_beef]);
            }
        },
    );
}
