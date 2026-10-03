//! Pure static compiler proof; no Vulkan offer or hardware admission follows from these modules.
#[path = "graph/graph.rs"]
mod graph;
#[rustfmt::skip]
use fusion_pcu_core::{
    PcuCompoundArithmeticPolicy,
    PcuFloatUnderflowPolicy,
    PcuImplementationRequirements,
    PcuNumericalMode,
    PcuPrecisionPolicy,
    PcuRangePolicy,
    PcuReproducibility,
    PcuScalarType,
};
#[rustfmt::skip]
use fusion_pcu_spirv::{
    lower_ordered_scalar_transport_to_spirv,
    PcuSpirvLoweringOptions,
};
fn calls(words: &[u32]) -> usize {
    let mut ids = Vec::new();
    let mut cursor = 5;
    while cursor < words.len() {
        let count = usize::try_from(words[cursor] >> 16).unwrap();
        let opcode = words[cursor] & 0xffff;
        if opcode == 5 {
            let bytes: Vec<_> = words[cursor + 2..cursor + count]
                .iter()
                .flat_map(|w| w.to_le_bytes())
                .collect();
            let name = String::from_utf8(bytes).unwrap();
            if [
                "load_indexed(",
                "load_readonly_zero(",
                "load_mutable_zero(",
                "store_indexed(",
            ]
            .iter()
            .any(|prefix| name.starts_with(prefix))
            {
                ids.push(words[cursor + 1]);
            }
        }
        cursor += count;
    }
    assert_eq!(ids.len(), 4);
    let mut calls = 0;
    cursor = 5;
    while cursor < words.len() {
        let count = usize::try_from(words[cursor] >> 16).unwrap();
        let opcode = words[cursor] & 0xffff;
        if opcode == 57 && ids.contains(&words[cursor + 3]) {
            calls += 1;
        }
        assert_ne!(opcode, 50, "all specialization constants must be frozen");
        cursor += count;
    }
    calls
}
#[test]
fn all_twenty_two_modules_retain_only_six_real_calls_and_exact_inert_headers() {
    for scalar in PcuScalarType::ALL {
        if matches!(
            scalar,
            PcuScalarType::Bool | PcuScalarType::I4 | PcuScalarType::U4
        ) {
            continue;
        }
        for grid in [false, true] {
            let mut baseline = Vec::new();
            graph::with(
                scalar,
                47,
                grid,
                PcuImplementationRequirements::default(),
                |kernel| {
                    lower_ordered_scalar_transport_to_spirv(
                        kernel,
                        PcuSpirvLoweringOptions::minimal_shader(),
                        &mut baseline,
                    )
                    .unwrap();
                },
            );
            assert_eq!(calls(&baseline), 6);
            for numerical_mode in [PcuNumericalMode::Boundary, PcuNumericalMode::Strict] {
                for compound_arithmetic in [
                    PcuCompoundArithmeticPolicy::Checked,
                    PcuCompoundArithmeticPolicy::BackendDefined,
                ] {
                    for precision in [
                        PcuPrecisionPolicy::Preserve,
                        PcuPrecisionPolicy::BackendOptimized,
                    ] {
                        for float_underflow in [
                            PcuFloatUnderflowPolicy::IeeeAfterRounding,
                            PcuFloatUnderflowPolicy::AllowGradualUnderflow,
                            PcuFloatUnderflowPolicy::RejectSubnormalResult,
                        ] {
                            for range_policy in [PcuRangePolicy::Reject, PcuRangePolicy::Clamp] {
                                let requirements = PcuImplementationRequirements {
                                    numerical_mode,
                                    float_underflow,
                                    range_policy,
                                    numerical_options: fusion_pcu_core::PcuNumericalOptions {
                                        compound_arithmetic,
                                        precision,
                                        ..Default::default()
                                    },
                                };
                                graph::with(scalar, 47, grid, requirements, |kernel| {
                                    let mut words = Vec::new();
                                    let (info, profile) = lower_ordered_scalar_transport_to_spirv(
                                        kernel,
                                        PcuSpirvLoweringOptions::minimal_shader(),
                                        &mut words,
                                    )
                                    .unwrap();
                                    assert_eq!(profile.requirements(), requirements);
                                    assert_eq!(profile.step_count(), 6);
                                    assert_eq!(profile.resources().len(), 4);
                                    assert_eq!(
                                        profile.local_id(),
                                        Some(
                                            19456
                                                + u32::try_from(
                                                    PcuScalarType::ALL
                                                        .iter()
                                                        .position(|t| *t == scalar)
                                                        .unwrap()
                                                )
                                                .unwrap()
                                        )
                                    );
                                    assert_eq!(info.word_count, words.len());
                                    assert_eq!(words, baseline);
                                });
                            }
                        }
                    }
                }
            }
            if let Some(directory) = std::env::var_os("PCU_ORDERED_TRANSPORT_MODULE_DIR") {
                std::fs::create_dir_all(&directory).unwrap();
                let file = std::path::Path::new(&directory).join(format!(
                    "{scalar:?}-{}.spv",
                    if grid { "grid" } else { "direct" }
                ));
                std::fs::write(
                    file,
                    baseline
                        .iter()
                        .flat_map(|word| word.to_le_bytes())
                        .collect::<Vec<_>>(),
                )
                .unwrap();
            }
        }
    }
}
#[test]
fn portable_and_cross_lane_mutable_zero_are_refused_while_one_lane_is_static() {
    graph::with(
        PcuScalarType::U8,
        47,
        false,
        PcuImplementationRequirements::default(),
        |kernel| {
            let mut changed = *kernel;
            changed
                .numerical_requirements
                .numerical_options
                .reproducibility = PcuReproducibility::PortableV1;
            assert!(
                lower_ordered_scalar_transport_to_spirv(
                    &changed,
                    PcuSpirvLoweringOptions::minimal_shader(),
                    &mut Vec::new()
                )
                .is_err()
            );
        },
    );
    for extent in [1, 47] {
        graph::with(
            PcuScalarType::U512,
            extent,
            false,
            PcuImplementationRequirements::default(),
            |kernel| {
                let mut ops = kernel.ops.to_vec();
                let fusion_pcu_core::PcuDispatchOp::Data(
                    fusion_pcu_core::PcuDispatchDataOp::BindingLoad { index, .. },
                ) = &mut ops[2]
                else {
                    panic!("stage load")
                };
                *index = fusion_pcu_core::PcuDispatchIndex::BindingElementZero;
                let mut changed = *kernel;
                changed.ops = &ops;
                let mut words = Vec::new();
                let result = lower_ordered_scalar_transport_to_spirv(
                    &changed,
                    PcuSpirvLoweringOptions::minimal_shader(),
                    &mut words,
                );
                if extent == 1 {
                    let (_, profile) = result.unwrap();
                    assert_eq!(profile.dispatch_extent(), 16);
                    assert_eq!(calls(&words), 6);
                } else {
                    assert!(result.is_err());
                }
            },
        );
    }
}
