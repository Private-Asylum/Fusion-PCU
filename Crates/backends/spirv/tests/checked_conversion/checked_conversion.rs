//! Exact mixed-width cold admission and official U32-only module proof.
#[path = "graph/graph.rs"]
mod graph;
#[rustfmt::skip]
use fusion_pcu_core::{PcuDispatchCheckedFloatConversion,PcuFloatUnderflowPolicy,PcuRangePolicy,PcuReproducibility};
#[rustfmt::skip]
use fusion_pcu_spirv::{lower_checked_float_conversion_to_spirv,PcuSpirvLoweringOptions,PcuSpirvVersion,PcuSpirvCapabilityCaps};
fn profiles(
    mut run: impl FnMut(
        PcuDispatchCheckedFloatConversion,
        PcuRangePolicy,
        PcuFloatUnderflowPolicy,
        bool,
        bool,
    ),
) {
    for conversion in [
        PcuDispatchCheckedFloatConversion::F32ToF64,
        PcuDispatchCheckedFloatConversion::F64ToF32,
    ] {
        for range in [PcuRangePolicy::Reject, PcuRangePolicy::Clamp] {
            for policy in [
                PcuFloatUnderflowPolicy::IeeeAfterRounding,
                PcuFloatUnderflowPolicy::AllowGradualUnderflow,
                PcuFloatUnderflowPolicy::RejectSubnormalResult,
            ] {
                for broadcast in [false, true] {
                    for grid in [false, true] {
                        run(conversion, range, policy, broadcast, grid);
                    }
                }
            }
        }
    }
}
#[test]
fn all_cold_profiles_preserve_roles_and_emit_only_u32_shader() {
    profiles(|conversion, range, policy, broadcast, grid| {
        graph::with(conversion, 65, range, policy, broadcast, grid, |kernel| {
            let mut words = Vec::new();
            let (info, profile) = lower_checked_float_conversion_to_spirv(
                kernel,
                PcuSpirvLoweringOptions::minimal_shader(),
                &mut words,
            )
            .unwrap();
            assert_eq!(info.capabilities, PcuSpirvCapabilityCaps::SHADER);
            assert_eq!(profile.input, graph::INPUT);
            assert_eq!(profile.output, graph::OUTPUT);
            assert_eq!(profile.input_extent(), if broadcast { 1 } else { 65 });
            assert_eq!(profile.extent, 65);
            assert_eq!(profile.range, range);
            assert_eq!(profile.underflow, policy);
            assert!((640..644).contains(&profile.local_id()));
            let mut cursor = 5;
            while cursor < words.len() {
                let instruction = &words[cursor..];
                let opcode = instruction[0] & 0xffff;
                let count = (instruction[0] >> 16) as usize;
                assert_ne!(opcode, 22, "no OpTypeFloat");
                if opcode == 21 {
                    assert_eq!(instruction[2], 32, "only 32-bit integers");
                }
                if opcode == 17 {
                    assert_eq!(instruction[1], 1, "Shader only");
                }
                cursor += count;
            }
            let mut invalid = *kernel;
            invalid
                .numerical_requirements
                .numerical_options
                .reproducibility = PcuReproducibility::PortableV1;
            let mut untouched = vec![0xfeed_beef];
            assert!(
                lower_checked_float_conversion_to_spirv(
                    &invalid,
                    PcuSpirvLoweringOptions::minimal_shader(),
                    &mut untouched
                )
                .is_err()
            );
            assert_eq!(untouched, [0xfeed_beef]);
            invalid = *kernel;
            invalid.numerical_requirements.range_policy = if range == PcuRangePolicy::Reject {
                PcuRangePolicy::Clamp
            } else {
                PcuRangePolicy::Reject
            };
            assert!(
                lower_checked_float_conversion_to_spirv(
                    &invalid,
                    PcuSpirvLoweringOptions::minimal_shader(),
                    &mut untouched
                )
                .is_err()
            );
            assert_eq!(untouched, [0xfeed_beef]);
        });
    });
}
#[test]
#[ignore = "requires installed SPIR-V Tools"]
fn official_validator_accepts_all_96_policy_index_version_modules() {
    let directory =
        std::env::temp_dir().join(format!("pcu-spirv-conversion-{}", std::process::id()));
    std::fs::create_dir_all(&directory).unwrap();
    let mut serial = 0;
    profiles(|conversion, range, policy, broadcast, grid| {
        graph::with(conversion, 65, range, policy, broadcast, grid, |kernel| {
            for version in [PcuSpirvVersion::V1_0, PcuSpirvVersion::V1_3] {
                let mut words = Vec::new();
                lower_checked_float_conversion_to_spirv(
                    kernel,
                    PcuSpirvLoweringOptions::minimal_shader().with_version(version),
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
    });
    assert_eq!(serial, 96);
    std::fs::remove_dir_all(directory).unwrap();
}
