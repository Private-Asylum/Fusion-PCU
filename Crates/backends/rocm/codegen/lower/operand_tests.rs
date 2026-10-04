//! Detached repeated/broadcast SSA roles and physical device ABI projection.
use super::*;
#[rustfmt::skip]
use fusion_pcu::{
    PcuBinding, PcuDispatchEntryPoint, PcuDispatchFloatBinaryOp,
    PcuFloatUnderflowPolicy, PcuRangePolicy, PcuKernelId,
};

#[test]
#[allow(clippy::too_many_lines)] // One fixture keeps each cold tuple, exact SSA and ABI witness together.
#[allow(clippy::cognitive_complexity)] // Exhaustive typed fault/ownership matrix retains its exact witnesses.
fn six_format_operand_schema_preserves_roles_and_projects_only_unread_declarations() {
    for scalar in [
        PcuScalarType::F16,
        PcuScalarType::BF16,
        PcuScalarType::F8E4M3FN,
        PcuScalarType::F8E5M2,
        PcuScalarType::F32,
        PcuScalarType::F64,
    ] {
        let value_type = PcuValueType::Scalar(scalar);
        for op in [
            PcuDispatchFloatBinaryOp::Add,
            PcuDispatchFloatBinaryOp::Sub,
            PcuDispatchFloatBinaryOp::Mul,
            PcuDispatchFloatBinaryOp::Div,
        ] {
            for policy in [
                PcuFloatUnderflowPolicy::IeeeAfterRounding,
                PcuFloatUnderflowPolicy::AllowGradualUnderflow,
                PcuFloatUnderflowPolicy::RejectSubnormalResult,
            ] {
                for range in [PcuRangePolicy::Reject, PcuRangePolicy::Clamp] {
                    for grid in [false, true] {
                        for roles in 0..5 {
                            let index = if grid {
                                PcuDispatchIndex::GridStrideId
                            } else {
                                PcuDispatchIndex::InvocationId
                            };
                            // Repeat input, repeat SSA, reversed SSA, mixed zero/indexed same
                            // input, or distinct indexed+zero broadcast input.
                            let second_binding = u32::from(!matches!(roles, 0 | 3));
                            let second_index = if roles >= 3 {
                                PcuDispatchIndex::BindingElementZero
                            } else {
                                index
                            };
                            let (lhs, rhs) = match roles {
                                1 => (1, 1),
                                2 => (2, 1),
                                _ => (1, 2),
                            };
                            let binding = |slot, access| {
                                PcuBinding::value(
                                    None,
                                    0,
                                    slot,
                                    PcuBindingStorageClass::Storage,
                                    access,
                                    value_type,
                                )
                            };
                            let mut bindings = [
                                binding(2, PcuBindingAccess::WriteOnly),
                                binding(1, PcuBindingAccess::ReadOnly),
                                binding(0, PcuBindingAccess::ReadOnly),
                            ];
                            if roles == 0 {
                                bindings.swap(0, 2);
                            }
                            let body = [
                                PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                                    result: PcuDispatchValueId(1),
                                    binding: PcuBindingRef::new(0, 0),
                                    index,
                                }),
                                PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                                    result: PcuDispatchValueId(2),
                                    binding: PcuBindingRef::new(0, second_binding),
                                    index: second_index,
                                }),
                                PcuDispatchOp::Data(PcuDispatchDataOp::CheckedFloatBinary {
                                    value_type,
                                    op,
                                    underflow_policy: policy,
                                    range_policy: range,
                                    result: PcuDispatchValueId(3),
                                    lhs: PcuDispatchValueId(lhs),
                                    rhs: PcuDispatchValueId(rhs),
                                }),
                                PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                                    binding: PcuBindingRef::new(0, 2),
                                    index,
                                    value: PcuDispatchValueId(3),
                                }),
                                PcuDispatchOp::Control(PcuDispatchControlOp::Return),
                            ];
                            let loop_ops = [
                                PcuDispatchOp::GridStrideLoop {
                                    extent: 65,
                                    body: &body[..4],
                                },
                                PcuDispatchOp::Control(PcuDispatchControlOp::Return),
                            ];
                            let mut requirements = PcuDispatchKernelIr::DEFAULT_REQUIREMENTS;
                            requirements.float_underflow = policy;
                            requirements.range_policy = range;
                            let kernel = PcuDispatchKernelIr {
                                numerical_requirements: requirements,
                                id: PcuKernelId(0x4f50_5343),
                                entry: PcuDispatchEntryPoint {
                                    name: "operand_schema",
                                    logical_shape: [if grid { 17 } else { 65 }, 1, 1],
                                },
                                bindings: &bindings,
                                ports: &[],
                                parameters: &[],
                                ops: if grid { &loop_ops } else { &body },
                                type_caps: PcuValueTypeCaps::for_scalar(scalar),
                                feature_caps: PcuDispatchFeatureCaps::READ_ONLY_RESOURCES
                                    .union(PcuDispatchFeatureCaps::MUTABLE_RESOURCES)
                                    .union(if range == PcuRangePolicy::Clamp {
                                        PcuDispatchFeatureCaps::RANGE_CLAMP
                                    } else {
                                        PcuDispatchFeatureCaps::empty()
                                    }),
                            };
                            let schema = checked_float_binary_operand_schema(&kernel).unwrap();
                            let repeated = second_binding == 0;
                            assert_eq!(schema.input_bindings().len(), if repeated { 1 } else { 2 });
                            assert_eq!(
                                schema.input_element_counts(65),
                                if repeated {
                                    [65, 0]
                                } else if roles == 4 {
                                    [65, 1]
                                } else {
                                    [65, 65]
                                }
                            );
                            assert_eq!(
                                schema.operand_inputs(),
                                match roles {
                                    1 => [0, 0],
                                    2 => [1, 0],
                                    _ if repeated => [0, 0],
                                    _ => [0, 1],
                                }
                            );
                            let source = lower_dispatch_to_hip_source(&kernel).unwrap();
                            let args = source
                                .split("fusion_kernel(")
                                .nth(1)
                                .unwrap()
                                .split(')')
                                .next()
                                .unwrap();
                            assert!(args.contains("binding_0_0"));
                            assert!(args.contains("binding_0_2"));
                            assert_eq!(args.contains("binding_0_1"), !repeated);
                            if repeated {
                                assert!(!source.contains("binding_0_1"));
                            }
                            let mut portable = kernel;
                            portable
                                .numerical_requirements
                                .numerical_options
                                .reproducibility = fusion_pcu::PcuReproducibility::PortableV1;
                            // No extension of the separately qualified distinct-operand Portable profile.
                            if repeated || range == PcuRangePolicy::Clamp {
                                assert!(lower_dispatch_to_hip_source(&portable).is_err());
                            }
                        }
                    }
                }
            }
        }
    }
}
