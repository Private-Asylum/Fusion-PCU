//! Official validation of separately lowered single effects; no runtime claim.
use super::*;

fn module(kernel: &PcuDispatchKernelIr<'_>, division: bool) -> Vec<u32> {
    let mut ops = kernel.ops.to_vec();
    ops.remove(4);
    if division {
        let PcuDispatchOp::Data(PcuDispatchDataOp::CheckedFloatBinary { op, .. }) = &mut ops[2]
        else {
            panic!("fixture arithmetic position changed");
        };
        *op = PcuDispatchFloatBinaryOp::Div;
    }
    ops[4] = PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
        binding: PcuBindingRef::new(0, 0),
        value: PcuDispatchValueId(29),
        index: PcuDispatchIndex::InvocationId,
    });
    let candidate = PcuDispatchKernelIr {
        ops: &ops,
        ..*kernel
    };
    let mut words = Vec::new();
    let (_, plan) = lower_one_effect_float_to_spirv(
        &candidate,
        PcuSpirvLoweringOptions::minimal_shader(),
        &mut words,
    )
    .unwrap();
    assert_eq!(plan.step_count(), 5);
    let instructions = instructions(&words);
    let names = instructions
        .iter()
        .filter(|instruction| instruction[0] & 0xffff == 5)
        .filter_map(|instruction| {
            let bytes = instruction[2..]
                .iter()
                .flat_map(|x| x.to_le_bytes())
                .collect::<Vec<_>>();
            bytes.starts_with(b"native_").then_some(instruction[1])
        })
        .collect::<Vec<_>>();
    assert_eq!(
        instructions
            .iter()
            .filter(|instruction| instruction[0] & 0xffff == 57 && names.contains(&instruction[3]))
            .count(),
        5
    );
    words
}

#[test]
#[ignore = "requires installed SPIR-V Tools; separate one-effect compiler modules"]
fn official_validator_accepts_all_separate_one_effect_modules() {
    let supplied = std::env::var_os("PCU_ONE_EFFECT_MODULE_DIR");
    let root = supplied.as_ref().map_or_else(
        || std::env::temp_dir().join(std::format!("pcu-one-effect-{}", std::process::id())),
        std::path::PathBuf::from,
    );
    std::fs::create_dir_all(&root).unwrap();
    let mut count = 0;
    for scalar in TYPES {
        for uf in [
            PcuFloatUnderflowPolicy::IeeeAfterRounding,
            PcuFloatUnderflowPolicy::RejectSubnormalResult,
            PcuFloatUnderflowPolicy::AllowGradualUnderflow,
        ] {
            for range in [PcuRangePolicy::Reject, PcuRangePolicy::Clamp] {
                with_map(scalar, uf, range, |kernel| {
                    for division in [false, true] {
                        let words = module(kernel, division);
                        let file = root.join(std::format!("{count}.spv"));
                        std::fs::write(
                            &file,
                            words
                                .iter()
                                .flat_map(|word| word.to_le_bytes())
                                .collect::<Vec<_>>(),
                        )
                        .unwrap();
                        let output = std::process::Command::new("spirv-val")
                            .args(["--target-env", "vulkan1.0"])
                            .arg(&file)
                            .output()
                            .unwrap();
                        assert!(
                            output.status.success(),
                            "{}",
                            std::string::String::from_utf8_lossy(&output.stderr)
                        );
                        count += 1;
                    }
                });
            }
        }
    }
    assert_eq!(count, 72);
    if supplied.is_none() {
        std::fs::remove_dir_all(root).unwrap();
    }
}
