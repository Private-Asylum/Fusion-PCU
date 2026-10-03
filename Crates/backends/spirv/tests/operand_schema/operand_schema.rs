//! Actual-read specialization keeps original declarations while indexing each mathematical operand.
#[path = "graph/graph.rs"]
mod graph;
#[rustfmt::skip]
use fusion_pcu_core::{PcuReproducibility,PcuScalarType};
#[rustfmt::skip]
use fusion_pcu_spirv::{lower_checked_float_binary_to_spirv,PcuSpirvLoweringOptions,PcuSpirvVersion};
const FORMATS: [PcuScalarType; 6] = [
    PcuScalarType::F32,
    PcuScalarType::F64,
    PcuScalarType::F16,
    PcuScalarType::BF16,
    PcuScalarType::F8E4M3FN,
    PcuScalarType::F8E5M2,
];
#[test]
fn repeated_read_aliases_one_input_and_preserves_typed_metadata() {
    for scalar in FORMATS {
        for schema in 0..5 {
            graph::with(scalar, 65, schema, |kernel| {
                let mut words = Vec::new();
                let (_, profile) = lower_checked_float_binary_to_spirv(
                    kernel,
                    PcuSpirvLoweringOptions::default(),
                    &mut words,
                )
                .unwrap();
                assert_eq!(profile.input_count, 1);
                assert_eq!(profile.inputs, [graph::INPUT; 2]);
                assert_eq!(profile.input_extents, [65, 0]);
                assert_eq!(profile.operands, [0, 0]);
                assert_eq!(profile.broadcast, [false, schema == 3]);
                assert_eq!(profile.output, graph::OUTPUT);
                assert_eq!(profile.declaration_count, kernel.bindings.len());
                for (original, frozen) in kernel.bindings.iter().zip(profile.declarations) {
                    assert_eq!(original.reference(), frozen);
                }
                assert!(!words.is_empty());
                // The Portable membership descriptor remains its separately qualified binary profile.
                let mut portable = *kernel;
                portable
                    .numerical_requirements
                    .numerical_options
                    .reproducibility = PcuReproducibility::PortableV1;
                let mut untouched = vec![0xfeed_beef];
                assert!(
                    lower_checked_float_binary_to_spirv(
                        &portable,
                        PcuSpirvLoweringOptions::default(),
                        &mut untouched
                    )
                    .is_err()
                );
                assert_eq!(untouched, [0xfeed_beef]);
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
    for scalar in FORMATS {
        for schema in 0..5 {
            for version in [PcuSpirvVersion::V1_0, PcuSpirvVersion::V1_3] {
                graph::with(scalar, 65, schema, |kernel| {
                    let mut words = Vec::new();
                    lower_checked_float_binary_to_spirv(
                        kernel,
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
                });
            }
        }
    }
    assert_eq!(serial, 60);
    std::fs::remove_dir_all(directory).unwrap();
}
