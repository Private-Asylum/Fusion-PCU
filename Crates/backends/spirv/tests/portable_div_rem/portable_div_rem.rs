//! Portable joint admission preserves the independently qualified exact integer modules.
#[path = "../div_rem_roles/graph/graph.rs"]
mod graph;
#[path = "../checked_div_rem/graph/graph.rs"]
mod legacy;
#[rustfmt::skip]
use fusion_pcu_core::{PcuScalarType,PcuRangePolicy,PcuReproducibility,PcuNumericalMode,PcuCompoundArithmeticPolicy,PcuPrecisionPolicy,PcuFloatUnderflowPolicy,PcuDispatchKernelIr,PcuDispatchOp,PcuDispatchDataOp};
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

fn compare(kernel: &PcuDispatchKernelIr<'_>, ordinal: u32, version: PcuSpirvVersion) {
    let options = PcuSpirvLoweringOptions::minimal_shader().with_version(version);
    let mut normal = Vec::new();
    let (_, old) = lower_checked_div_rem_to_spirv(kernel, options, &mut normal).unwrap();
    assert!(!old.portable);
    for mode in [PcuNumericalMode::Boundary, PcuNumericalMode::Strict] {
        for compound in [
            PcuCompoundArithmeticPolicy::Checked,
            PcuCompoundArithmeticPolicy::BackendDefined,
        ] {
            for precision in [
                PcuPrecisionPolicy::Preserve,
                PcuPrecisionPolicy::BackendOptimized,
            ] {
                for underflow in [
                    PcuFloatUnderflowPolicy::AllowGradualUnderflow,
                    PcuFloatUnderflowPolicy::RejectSubnormalResult,
                    PcuFloatUnderflowPolicy::IeeeAfterRounding,
                ] {
                    let mut portable = *kernel;
                    portable.numerical_requirements.numerical_mode = mode;
                    portable.numerical_requirements.float_underflow = underflow;
                    portable
                        .numerical_requirements
                        .numerical_options
                        .compound_arithmetic = compound;
                    portable.numerical_requirements.numerical_options.precision = precision;
                    portable
                        .numerical_requirements
                        .numerical_options
                        .reproducibility = PcuReproducibility::PortableV1;
                    let original = portable.numerical_requirements;
                    let mut words = Vec::new();
                    let (info, profile) =
                        lower_checked_div_rem_to_spirv(&portable, options, &mut words).unwrap();
                    assert!(profile.portable);
                    assert_eq!(profile.local_id(), Some(12544 + ordinal));
                    assert_eq!(profile.operand_profile, old.operand_profile);
                    assert_eq!(profile.inputs, old.inputs);
                    assert_eq!(profile.outputs, old.outputs);
                    assert_eq!(profile.input_extents, old.input_extents);
                    assert_eq!(profile.operand_broadcast, old.operand_broadcast);
                    assert_eq!(profile.declarations, old.declarations);
                    assert_eq!(profile.declaration_count, old.declaration_count);
                    assert_eq!(portable.numerical_requirements, original);
                    assert_eq!(info.capabilities, PcuSpirvCapabilityCaps::SHADER);
                    assert_eq!(
                        words, normal,
                        "Portable opt-in changes identity, not arithmetic bytes"
                    );
                }
            }
        }
    }
}

#[test]
fn exact_fourteen_width_twenty_four_tuples_keep_original_header_and_module_bytes() {
    for (ordinal, scalar) in TYPES.into_iter().enumerate() {
        let ordinal = u32::try_from(ordinal).unwrap();
        for version in [PcuSpirvVersion::V1_0, PcuSpirvVersion::V1_3] {
            for kind in 0..6 {
                let mut graph = graph::Graph::new(scalar, 65);
                graph.kind = kind;
                graph.grid = kind == 4;
                graph.with(|kernel| compare(kernel, ordinal, version));
            }
            for grid in [false, true] {
                for routing in 0..3 {
                    let mut graph = legacy::Graph::new(scalar, 65);
                    graph.grid = grid;
                    graph.routing = routing;
                    graph.with(|kernel| compare(kernel, ordinal, version));
                }
            }
        }
    }
}

#[test]
fn nonmember_portable_profiles_emit_no_words() {
    for scalar in TYPES {
        graph::Graph::new(scalar, 5).with(|kernel| {
            assert!(fusion_pcu_core::describe_portable_v1_integer_div_rem_map(kernel).is_err());
            let mut portable = *kernel;
            portable
                .numerical_requirements
                .numerical_options
                .reproducibility = PcuReproducibility::PortableV1;
            for invalid in 0..5 {
                let mut bad = portable;
                match invalid {
                    0 => bad.numerical_requirements.range_policy = PcuRangePolicy::Clamp,
                    1 => bad.entry.logical_shape[0] = 0,
                    2 => bad.entry.logical_shape[1] = 2,
                    3 => bad.bindings = &[],
                    _ => bad.ops = &[],
                }
                no_words(&bad);
            }
            let mut operations = kernel.ops.to_vec();
            let PcuDispatchOp::Data(PcuDispatchDataOp::CheckedDivRem { flags, .. }) =
                &mut operations[2]
            else {
                panic!("joint operation")
            };
            *flags = fusion_pcu_core::model::PcuIntegerDivFlags::DIV_OR_ZERO;
            portable.ops = &operations;
            no_words(&portable);
        });
    }
}

fn no_words(kernel: &PcuDispatchKernelIr<'_>) {
    let mut words = vec![0xfeed_beef];
    assert!(
        lower_checked_div_rem_to_spirv(
            kernel,
            PcuSpirvLoweringOptions::minimal_shader(),
            &mut words
        )
        .is_err()
    );
    assert_eq!(words, [0xfeed_beef]);
}

#[test]
#[ignore = "requires official SPIR-V Tools"]
fn official_portable_modules_keep_shader_only_capability() {
    let dir = std::env::temp_dir().join(format!("pcu-portable-div-rem-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("module.spv");
    let mut count = 0;
    for scalar in TYPES {
        for kind in 0..6 {
            for version in [PcuSpirvVersion::V1_0, PcuSpirvVersion::V1_3] {
                let mut graph = graph::Graph::new(scalar, 65);
                graph.kind = kind;
                graph.grid = kind == 4;
                graph.with(|kernel| {
                    let mut kernel = *kernel;
                    kernel
                        .numerical_requirements
                        .numerical_options
                        .reproducibility = PcuReproducibility::PortableV1;
                    let mut words = Vec::new();
                    let (info, _) = lower_checked_div_rem_to_spirv(
                        &kernel,
                        PcuSpirvLoweringOptions::minimal_shader().with_version(version),
                        &mut words,
                    )
                    .unwrap();
                    assert_eq!(info.capabilities, PcuSpirvCapabilityCaps::SHADER);
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
