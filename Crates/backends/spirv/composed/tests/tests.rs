//! Static call and status-policy structure, without a hardware execution claim.
use super::*;
#[rustfmt::skip]
use fusion_pcu::{
    PcuBinding,
    PcuBindingStorageClass,
    PcuDispatchControlOp,
    PcuDispatchDataOp,
    PcuDispatchEntryPoint,
    PcuDispatchFeatureCaps,
    PcuDispatchFloatBinaryOp,
    PcuDispatchIndex,
    PcuDispatchOp,
    PcuDispatchValueId,
    PcuExecutionFaultKind,
    PcuFloatUnderflowPolicy,
    PcuKernelId,
    PcuRangePolicy,
    PcuReproducibility,
    PcuValueType,
    PcuValueTypeCaps,
};
use std::vec::Vec;

const TYPES: [PcuScalarType; 6] = [
    PcuScalarType::F16,
    PcuScalarType::BF16,
    PcuScalarType::F8E4M3FN,
    PcuScalarType::F8E5M2,
    PcuScalarType::F32,
    PcuScalarType::F64,
];
fn with_map<R>(
    scalar: PcuScalarType,
    uf: PcuFloatUnderflowPolicy,
    range: PcuRangePolicy,
    visit: impl FnOnce(&PcuDispatchKernelIr<'_>) -> R,
) -> R {
    let value_type = PcuValueType::Scalar(scalar);
    let bindings = [
        PcuBinding::value(
            None,
            0,
            0,
            PcuBindingStorageClass::Storage,
            PcuBindingAccess::ReadWrite,
            value_type,
        ),
        PcuBinding::value(
            None,
            0,
            1,
            PcuBindingStorageClass::Storage,
            PcuBindingAccess::ReadOnly,
            value_type,
        ),
    ];
    let load = |id| {
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
            result: PcuDispatchValueId(id),
            binding: PcuBindingRef::new(0, 1),
            index: PcuDispatchIndex::InvocationId,
        })
    };
    let binary = |id, left, right, op| {
        PcuDispatchOp::Data(PcuDispatchDataOp::CheckedFloatBinary {
            result: PcuDispatchValueId(id),
            lhs: PcuDispatchValueId(left),
            rhs: PcuDispatchValueId(right),
            op,
            value_type,
            range_policy: range,
            underflow_policy: uf,
        })
    };
    let ops = [
        load(7),
        load(13),
        binary(17, 7, 13, PcuDispatchFloatBinaryOp::Add),
        load(29),
        binary(31, 17, 29, PcuDispatchFloatBinaryOp::Mul),
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
            binding: PcuBindingRef::new(0, 0),
            value: PcuDispatchValueId(31),
            index: PcuDispatchIndex::InvocationId,
        }),
        PcuDispatchOp::Control(PcuDispatchControlOp::Return),
    ];
    visit(&PcuDispatchKernelIr {
        id: PcuKernelId(1),
        entry: PcuDispatchEntryPoint {
            name: "static_composed",
            logical_shape: [65, 1, 1],
        },
        bindings: &bindings,
        ports: &[],
        parameters: &[],
        ops: &ops,
        type_caps: PcuValueTypeCaps::empty(),
        feature_caps: PcuDispatchFeatureCaps::empty(),
        numerical_requirements: PcuImplementationRequirements {
            float_underflow: uf,
            range_policy: range,
            ..PcuDispatchKernelIr::DEFAULT_REQUIREMENTS
        },
    })
}

fn instructions(words: &[u32]) -> Vec<&[u32]> {
    let mut result = Vec::new();
    let mut offset = 5;
    while offset < words.len() {
        let count = (words[offset] >> 16) as usize;
        assert!(count != 0 && offset + count <= words.len());
        result.push(&words[offset..offset + count]);
        offset += count;
    }
    result
}

#[test]
fn single_lane_element_zero_reload_observes_the_ordered_private_store() {
    with_map(
        PcuScalarType::F32,
        PcuFloatUnderflowPolicy::IeeeAfterRounding,
        PcuRangePolicy::Reject,
        |kernel| {
            let mut ops = kernel.ops.to_vec();
            ops.insert(
                3,
                PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                    binding: PcuBindingRef::new(0, 0),
                    value: PcuDispatchValueId(17),
                    index: PcuDispatchIndex::InvocationId,
                }),
            );
            ops[4] = PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result: PcuDispatchValueId(29),
                binding: PcuBindingRef::new(0, 0),
                index: PcuDispatchIndex::BindingElementZero,
            });
            let mut candidate = *kernel;
            candidate.entry.logical_shape = [1, 1, 1];
            candidate.ops = &ops;
            let plan = validate_composed_float_map(&candidate).unwrap();
            assert_eq!(
                plan.steps[4].arguments[2], 0,
                "N=1 element-zero reload must read the private word, including an earlier store"
            );
            candidate.entry.logical_shape = [65, 1, 1];
            assert!(
                validate_composed_float_map(&candidate).is_err(),
                "multiple logical lanes still require an unimplemented cross-index snapshot"
            );
        },
    );
}

#[test]
fn frozen_calls_have_exact_roles_and_instruction_local_laws() {
    for scalar in TYPES {
        with_map(
            scalar,
            PcuFloatUnderflowPolicy::IeeeAfterRounding,
            PcuRangePolicy::Clamp,
            |kernel| {
                let mut words = Vec::new();
                let (info, plan) = lower_composed_float_to_spirv(
                    kernel,
                    PcuSpirvLoweringOptions::default(),
                    &mut words,
                )
                .unwrap();
                assert_eq!(info.word_count, words.len());
                assert_eq!(info.capabilities, crate::PcuSpirvCapabilityCaps::SHADER);
                assert_eq!(plan.step_count(), 6);
                assert_eq!(plan.resources().len(), 2);
                assert_eq!(plan.resources()[0].binding, PcuBindingRef::new(0, 1));
                assert_eq!(plan.resources()[0].read_elements, 65);
                assert_eq!(plan.resources()[1].write_elements, 65);
                assert_eq!(
                    plan.declarations(),
                    [PcuBindingRef::new(0, 0), PcuBindingRef::new(0, 1)]
                );
                assert_eq!(
                    plan.steps[..6]
                        .iter()
                        .map(|step| step.function)
                        .collect::<Vec<_>>(),
                    [1, 1, 4, 1, 6, 2]
                );
                assert!(plan.fault_law(0).is_none());
                assert!(
                    !plan
                        .fault_law(2)
                        .unwrap()
                        .allows(PcuExecutionFaultKind::ArithmeticUnderflow, true)
                );
                assert!(
                    plan.fault_law(4)
                        .unwrap()
                        .allows(PcuExecutionFaultKind::ArithmeticUnderflow, true)
                );
                let instructions = instructions(&words);
                assert!(
                    !instructions
                        .iter()
                        .any(|instruction| instruction[0] & 0xffff == 50)
                );
                assert!(
                    !instructions
                        .iter()
                        .any(|instruction| instruction[0] & 0xffff == 71
                            && instruction.len() == 4
                            && instruction[2] == 1)
                );
                let mut native = Vec::new();
                for instruction in &instructions {
                    if instruction[0] & 0xffff == 5 {
                        let bytes = instruction[2..]
                            .iter()
                            .flat_map(|word| word.to_le_bytes())
                            .collect::<Vec<_>>();
                        if bytes.starts_with(b"native_") {
                            native.push(instruction[1]);
                        }
                    }
                }
                assert_eq!(
                    instructions
                        .iter()
                        .filter(|instruction| instruction[0] & 0xffff == 57
                            && native.contains(&instruction[3]))
                        .count(),
                    6
                );
            },
        );
    }
}

#[test]
fn unsupported_requests_preserve_sink_before_emission() {
    with_map(
        PcuScalarType::F32,
        PcuFloatUnderflowPolicy::IeeeAfterRounding,
        PcuRangePolicy::Reject,
        |kernel| {
            let mut portable = *kernel;
            portable
                .numerical_requirements
                .numerical_options
                .reproducibility = PcuReproducibility::PortableV1;
            let mut sink = std::vec![0xfeed_beef];
            assert_eq!(
                lower_composed_float_to_spirv(
                    &portable,
                    PcuSpirvLoweringOptions::default(),
                    &mut sink
                ),
                Err(PcuSpirvError::UnsupportedNumericalRequirements)
            );
            assert_eq!(sink, [0xfeed_beef]);
        },
    );
}

#[test]
fn one_effect_is_separate_and_keeps_the_observable_unused_operation() {
    for scalar in TYPES {
        with_map(
            scalar,
            PcuFloatUnderflowPolicy::RejectSubnormalResult,
            PcuRangePolicy::Clamp,
            |kernel| {
                assert!(validate_one_effect_float_map(kernel).is_err());
                let mut ops = kernel.ops.to_vec();
                // Delete Mul but preserve Add and the subsequent fresh input load.
                // The output ignores Add's result; its checked effect is still required.
                ops.remove(4);
                ops[4] = PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                    binding: PcuBindingRef::new(0, 0),
                    value: PcuDispatchValueId(29),
                    index: PcuDispatchIndex::InvocationId,
                });
                let mut one = *kernel;
                one.ops = &ops;
                assert!(validate_composed_float_map(&one).is_err());
                let mut words = Vec::new();
                let (_, plan) = lower_one_effect_float_to_spirv(
                    &one,
                    PcuSpirvLoweringOptions::minimal_shader(),
                    &mut words,
                )
                .unwrap();
                assert!(plan.is_one_effect());
                assert_eq!(plan.step_count(), 5);
                assert_eq!(plan.steps[2].function, 4);
                assert!(
                    plan.fault_law(2)
                        .unwrap()
                        .allows(PcuExecutionFaultKind::ArithmeticUnderflow, true)
                );
                assert!(plan.fault_law(3).is_none());
                let mut zero_ops = ops.clone();
                zero_ops.remove(2);
                let zero = PcuDispatchKernelIr {
                    ops: &zero_ops,
                    ..*kernel
                };
                assert!(validate_one_effect_float_map(&zero).is_err());
            },
        );
    }
}

#[test]
#[ignore = "requires installed SPIR-V Tools; compiler structure only"]
fn official_validator_accepts_compacted_static_programs() {
    let root =
        std::env::temp_dir().join(std::format!("pcu-static-composed-{}", std::process::id()));
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
                    let mut words = Vec::new();
                    lower_composed_float_to_spirv(
                        kernel,
                        PcuSpirvLoweringOptions::default(),
                        &mut words,
                    )
                    .unwrap();
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
                        .arg(file)
                        .output()
                        .unwrap();
                    assert!(
                        output.status.success(),
                        "{}",
                        std::string::String::from_utf8_lossy(&output.stderr)
                    );
                });
                count += 1;
            }
        }
    }
    assert_eq!(count, 36);
    std::fs::remove_dir_all(root).unwrap();
}

#[path = "one_effect_validator/one_effect_validator.rs"]
mod one_effect_validator;
