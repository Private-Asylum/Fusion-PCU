//! U32-only checked integer cold admission, exact defaults and official validator matrix.
#[path = "graph/graph.rs"]
mod graph;
#[rustfmt::skip]
use fusion_pcu_core::{PcuDispatchIntegerBinaryOp,PcuScalarType,PcuRangePolicy,PcuReproducibility};
#[rustfmt::skip]
use fusion_pcu_spirv::{lower_checked_integer_to_spirv,PcuSpirvCapabilityCaps,PcuSpirvLoweringOptions,PcuSpirvVersion};
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
const OPS: [PcuDispatchIntegerBinaryOp; 3] = [
    PcuDispatchIntegerBinaryOp::Add,
    PcuDispatchIntegerBinaryOp::Sub,
    PcuDispatchIntegerBinaryOp::Mul,
];
#[test]
fn disjoint_ids_exact_logical_roles_and_u32_only_capabilities() {
    for (format, scalar) in TYPES.into_iter().enumerate() {
        for (operation, op) in OPS.into_iter().enumerate() {
            for (range_code, range) in [PcuRangePolicy::Reject, PcuRangePolicy::Clamp]
                .into_iter()
                .enumerate()
            {
                for grid in [false, true] {
                    for broadcast in [false, true] {
                        for swapped in [false, true] {
                            let mut g = graph::Graph::new(scalar, 65, op, range);
                            g.grid = grid;
                            g.broadcast = broadcast;
                            g.swapped = swapped;
                            g.with(|kernel| {
                                let mut words = Vec::new();
                                let (info, p) = lower_checked_integer_to_spirv(
                                    kernel,
                                    PcuSpirvLoweringOptions::default(),
                                    &mut words,
                                )
                                .unwrap();
                                assert_eq!(info.capabilities, PcuSpirvCapabilityCaps::SHADER);
                                assert_eq!(
                                    p.local_id(),
                                    Some(
                                        512 + u32::try_from(
                                            format * 6 + range_code * 3 + operation
                                        )
                                        .unwrap()
                                    )
                                );
                                assert_eq!(p.inputs, graph::INPUTS);
                                assert_eq!(p.output, graph::OUTPUT);
                                assert_eq!(p.broadcast, [false, broadcast]);
                                assert_eq!(p.operands, if swapped { [1, 0] } else { [0, 1] });
                                assert_eq!(p.input_extent(1), if broadcast { 1 } else { 65 });
                                assert_eq!(p.dispatch_extent(), 65u32.div_ceil(p.packing_lanes()));
                                assert_eq!(info.word_count, words.len());
                            });
                        }
                    }
                }
            }
        }
    }
}
#[test]
fn unsupported_and_mismatched_headers_write_no_words() {
    for scalar in TYPES.into_iter().chain([
        PcuScalarType::I4,
        PcuScalarType::U4,
        PcuScalarType::F16,
        PcuScalarType::F128,
    ]) {
        graph::Graph::new(scalar, 65, OPS[0], PcuRangePolicy::Reject).with(|kernel| {
            for mismatch in [0, 2] {
                let mut kernel = *kernel;
                match mismatch {
                    0 => kernel.numerical_requirements.range_policy = PcuRangePolicy::Clamp,
                    1 => {
                        kernel
                            .numerical_requirements
                            .numerical_options
                            .reproducibility = PcuReproducibility::PortableV1;
                    }
                    _ => kernel.entry.logical_shape[1] = 2,
                }
                let mut words = Vec::new();
                assert!(
                    lower_checked_integer_to_spirv(
                        &kernel,
                        PcuSpirvLoweringOptions::default(),
                        &mut words
                    )
                    .is_err()
                );
                assert!(words.is_empty());
            }
        });
    }
    graph::Graph::new(
        PcuScalarType::U512,
        u32::MAX,
        OPS[0],
        PcuRangePolicy::Reject,
    )
    .with(|kernel| {
        let mut words = Vec::new();
        assert!(
            lower_checked_integer_to_spirv(kernel, PcuSpirvLoweringOptions::default(), &mut words)
                .is_err()
        );
        assert!(words.is_empty());
    });
}
#[test]
#[ignore = "requires installed SPIR-V Tools"]
fn native_validator_accepts_672_exact_profiles() {
    let dir = std::env::temp_dir().join(format!("pcu-spirv-integer-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let mut count = 0;
    for scalar in TYPES {
        for op in OPS {
            for range in [PcuRangePolicy::Reject, PcuRangePolicy::Clamp] {
                for grid in [false, true] {
                    for broadcast in [false, true] {
                        for version in [PcuSpirvVersion::V1_0, PcuSpirvVersion::V1_3] {
                            let mut g = graph::Graph::new(scalar, 65, op, range);
                            g.grid = grid;
                            g.broadcast = broadcast;
                            g.with(|kernel| {
                                let mut words = Vec::new();
                                lower_checked_integer_to_spirv(
                                    kernel,
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
    assert_eq!(count, 672);
    std::fs::remove_dir_all(dir).unwrap();
}
