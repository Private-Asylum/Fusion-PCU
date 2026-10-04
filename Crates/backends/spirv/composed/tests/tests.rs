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

#[cfg(feature = "embedded-composed")]
#[test]
fn exact_external_templates_preserve_emission_and_refuse_mutated_assets() {
    for family in [
        PcuSpirvComposedTemplateFamily::F32,
        PcuSpirvComposedTemplateFamily::F64,
        PcuSpirvComposedTemplateFamily::Low,
        PcuSpirvComposedTemplateFamily::IntegerWide,
        PcuSpirvComposedTemplateFamily::IntegerNarrow,
    ] {
        let data = embedded_composed_template(family);
        PcuSpirvComposedTemplate::validate(data).unwrap();
        let mut words = data.words.to_vec();
        words[2] ^= 1;
        assert!(
            PcuSpirvComposedTemplate::validate(PcuSpirvComposedTemplateData {
                words: &words,
                ..data
            })
            .is_err()
        );
        let mut offsets = data.call_target_offsets.to_vec();
        offsets[0] += 1;
        assert!(
            PcuSpirvComposedTemplate::validate(PcuSpirvComposedTemplateData {
                call_target_offsets: &offsets,
                ..data
            })
            .is_err()
        );
    }
    for scalar in TYPES {
        let family = match scalar {
            PcuScalarType::F32 => PcuSpirvComposedTemplateFamily::F32,
            PcuScalarType::F64 => PcuSpirvComposedTemplateFamily::F64,
            _ => PcuSpirvComposedTemplateFamily::Low,
        };
        let template =
            PcuSpirvComposedTemplate::validate(embedded_composed_template(family)).unwrap();
        for uf in [
            PcuFloatUnderflowPolicy::IeeeAfterRounding,
            PcuFloatUnderflowPolicy::RejectSubnormalResult,
            PcuFloatUnderflowPolicy::AllowGradualUnderflow,
        ] {
            for range in [PcuRangePolicy::Reject, PcuRangePolicy::Clamp] {
                with_map(scalar, uf, range, |kernel| {
                    let mut embedded = Vec::new();
                    let mut external = Vec::new();
                    let options = PcuSpirvLoweringOptions::default();
                    let expected =
                        lower_composed_float_to_spirv(kernel, options, &mut embedded).unwrap();
                    let actual = lower_composed_float_with_template(
                        kernel,
                        options,
                        template,
                        false,
                        &mut external,
                    )
                    .unwrap();
                    assert_eq!(actual, expected);
                    assert_eq!(external, embedded);
                });
            }
        }
    }
}

fn integer_ops<'a>(kernel: &PcuDispatchKernelIr<'a>) -> Vec<fusion_pcu::PcuDispatchOp<'a>> {
    kernel
        .ops
        .iter()
        .map(|operation| match *operation {
            PcuDispatchOp::Data(PcuDispatchDataOp::CheckedFloatBinary {
                result,
                lhs,
                rhs,
                op,
                value_type,
                range_policy,
                ..
            }) => PcuDispatchOp::Data(PcuDispatchDataOp::CheckedIntegerBinary {
                result,
                lhs,
                rhs,
                value_type,
                range_policy,
                op: match op {
                    PcuDispatchFloatBinaryOp::Add => fusion_pcu::PcuDispatchIntegerBinaryOp::Add,
                    PcuDispatchFloatBinaryOp::Mul => fusion_pcu::PcuDispatchIntegerBinaryOp::Mul,
                    _ => unreachable!(),
                },
            }),
            other => other,
        })
        .collect()
}

#[cfg(feature = "embedded-composed")]
#[test]
fn integer_metadata_and_external_template_are_exact() {
    for scalar in [
        PcuScalarType::U8,
        PcuScalarType::I8,
        PcuScalarType::U16,
        PcuScalarType::I16,
        PcuScalarType::U32,
        PcuScalarType::I32,
        PcuScalarType::U64,
        PcuScalarType::I64,
        PcuScalarType::U128,
        PcuScalarType::I128,
        PcuScalarType::U256,
        PcuScalarType::I256,
        PcuScalarType::U512,
        PcuScalarType::I512,
    ] {
        for range in [PcuRangePolicy::Reject, PcuRangePolicy::Clamp] {
            with_map(
                scalar,
                PcuFloatUnderflowPolicy::IeeeAfterRounding,
                range,
                |kernel| {
                    let ops = integer_ops(kernel);
                    let mut candidate = *kernel;
                    candidate.ops = &ops;
                    let profile = validate_composed_integer_map(&candidate).unwrap();
                    assert!(profile.is_integer());
                    assert_eq!(profile.element_bytes(), usize::from(scalar.bit_width() / 8));
                    assert!(profile.fault_law(2).is_some());
                    assert!(profile.fault_law(4).is_some());
                    let options = PcuSpirvLoweringOptions::minimal_shader();
                    let mut embedded = Vec::new();
                    let mut external = Vec::new();
                    lower_composed_integer_to_spirv(&candidate, options, &mut embedded).unwrap();
                    let raw = embedded_composed_template(if scalar.bit_width() <= 128 {
                        PcuSpirvComposedTemplateFamily::IntegerNarrow
                    } else {
                        PcuSpirvComposedTemplateFamily::IntegerWide
                    });
                    let sealed = PcuSpirvComposedTemplate::validate(raw).unwrap();
                    lower_composed_integer_with_template(
                        &candidate,
                        options,
                        sealed,
                        false,
                        &mut external,
                    )
                    .unwrap();
                    assert_eq!(embedded, external);
                    assert!(validate_composed_float_map(&candidate).is_err());
                    let mut portable = candidate;
                    portable
                        .numerical_requirements
                        .numerical_options
                        .reproducibility = PcuReproducibility::PortableV1;
                    let mut portable_words = Vec::new();
                    lower_composed_integer_to_spirv(&portable, options, &mut portable_words)
                        .unwrap();
                    assert_eq!(embedded, portable_words);
                    assert_eq!(
                        validate_composed_integer_map(&portable)
                            .unwrap()
                            .requirements(),
                        portable.numerical_requirements
                    );
                    assert_eq!(
                        profile.packing_lanes(),
                        if scalar.bit_width() < 32 {
                            32 / u32::from(scalar.bit_width())
                        } else {
                            1
                        }
                    );
                    assert_eq!(
                        profile.dispatch_extent(),
                        65_u32.div_ceil(profile.packing_lanes())
                    );
                },
            );
        }
    }
}

#[test]
fn integer_byte_address_overflow_is_refused_cold() {
    with_map(
        PcuScalarType::U128,
        PcuFloatUnderflowPolicy::IeeeAfterRounding,
        PcuRangePolicy::Reject,
        |kernel| {
            let ops = integer_ops(kernel);
            let mut oversized = *kernel;
            oversized.ops = &ops;
            oversized.entry.logical_shape[0] = u32::MAX / 16 + 1;
            assert!(validate_composed_integer_map(&oversized).is_err());
        },
    );
}

#[path = "negative_integer_contracts/negative_integer_contracts.rs"]
mod negative_integer_contracts;
