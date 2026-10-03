//! Cold valid/negative schema, SSA, policy and shape admission independent of Metal loading.
use super::*;
#[rustfmt::skip]
use fusion_pcu::{
    PcuBinding,
    PcuBindingStorageClass,
    PcuDispatchEntryPoint,
    PcuDispatchFeatureCaps,
    PcuDispatchFloatBinaryOp as Op,
    PcuDispatchValueId,
    PcuFloatUnderflowPolicy as Policy,
    PcuKernelId,
};
#[derive(Clone, Copy, Debug)]
pub enum Variant {
    Plain,
    Swap,
    Repeat,
    Broadcast,
    Clamp,
    WrongWidth,
    InvalidSsa,
    WrongOutput,
    WrongShape,
    Portable,
    PortableBroadcast,
}
#[allow(clippy::too_many_lines)] // One canonical fixture deliberately mutates independent admission facts.
pub fn fixture<T: fusion_pcu::PcuScalar>(
    grid: bool,
    op: Op,
    policy: Policy,
    variant: Variant,
    visit: impl FnOnce(&PcuDispatchKernelIr<'_>),
) {
    let bindings = [
        PcuBinding::scalar::<T>(
            Some("a"),
            2,
            3,
            PcuBindingStorageClass::Storage,
            PcuBindingAccess::ReadOnly,
        ),
        PcuBinding::scalar::<T>(
            Some("b"),
            2,
            7,
            PcuBindingStorageClass::Storage,
            PcuBindingAccess::ReadOnly,
        ),
        PcuBinding::scalar::<T>(
            Some("out"),
            4,
            1,
            PcuBindingStorageClass::Storage,
            PcuBindingAccess::ReadWrite,
        ),
    ];
    let index = if grid {
        PcuDispatchIndex::GridStrideId
    } else {
        PcuDispatchIndex::InvocationId
    };
    let first = PcuDispatchValueId(1);
    let second = PcuDispatchValueId(2);
    let result = PcuDispatchValueId(3);
    let (lhs, rhs) = match variant {
        Variant::Swap => (second, first),
        Variant::Repeat => (first, first),
        Variant::InvalidSsa => (PcuDispatchValueId(9), first),
        _ => (first, second),
    };
    let body = [
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
            result: first,
            binding: bindings[0].reference(),
            index: if matches!(variant, Variant::Broadcast | Variant::PortableBroadcast) {
                PcuDispatchIndex::BindingElementZero
            } else {
                index
            },
        }),
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
            result: second,
            binding: bindings[1].reference(),
            index,
        }),
        PcuDispatchOp::Data(PcuDispatchDataOp::CheckedFloatBinary {
            value_type: if matches!(variant, Variant::WrongWidth) {
                if T::TYPE == fusion_pcu::PcuScalarType::F32 {
                    PcuValueType::f64()
                } else {
                    PcuValueType::f32()
                }
            } else {
                PcuValueType::Scalar(T::TYPE)
            },
            op,
            underflow_policy: policy,
            range_policy: if matches!(variant, Variant::Clamp) {
                PcuRangePolicy::Clamp
            } else {
                PcuRangePolicy::Reject
            },
            result,
            lhs,
            rhs,
        }),
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
            binding: if matches!(variant, Variant::WrongOutput) {
                bindings[0].reference()
            } else {
                bindings[2].reference()
            },
            index,
            value: result,
        }),
    ];
    let direct = [
        body[0],
        body[1],
        body[2],
        body[3],
        PcuDispatchOp::Control(PcuDispatchControlOp::Return),
    ];
    let grid_ops = [
        PcuDispatchOp::GridStrideLoop {
            extent: 3,
            body: &body,
        },
        PcuDispatchOp::Control(PcuDispatchControlOp::Return),
    ];
    let kernel = PcuDispatchKernelIr {
        numerical_requirements: fusion_pcu::PcuImplementationRequirements {
            float_underflow: policy,
            range_policy: if matches!(variant, Variant::Clamp) {
                PcuRangePolicy::Clamp
            } else {
                PcuRangePolicy::Reject
            },
            ..PcuDispatchKernelIr::DEFAULT_REQUIREMENTS
        },
        id: PcuKernelId(1),
        entry: PcuDispatchEntryPoint {
            name: "f64-map",
            logical_shape: if matches!(variant, Variant::WrongShape) {
                [3, 2, 1]
            } else {
                [3, 1, 1]
            },
        },
        bindings: &bindings,
        ports: &[],
        parameters: &[],
        ops: if grid { &grid_ops } else { &direct },
        type_caps: PcuValueTypeCaps::for_value_type(PcuValueType::Scalar(T::TYPE)),
        feature_caps: PcuDispatchFeatureCaps::empty(),
    };
    let mut kernel = kernel;
    if matches!(variant, Variant::Portable | Variant::PortableBroadcast) {
        kernel.numerical_requirements.float_underflow = policy;
        kernel
            .numerical_requirements
            .numerical_options
            .reproducibility = fusion_pcu::PcuReproducibility::PortableV1;
    }
    visit(&kernel);
}
fn complete<T: fusion_pcu::PcuScalar>() {
    for grid in [false, true] {
        for op in [Op::Add, Op::Sub, Op::Mul, Op::Div] {
            for policy in [
                Policy::IeeeAfterRounding,
                Policy::RejectSubnormalResult,
                Policy::AllowGradualUnderflow,
            ] {
                for variant in [
                    Variant::Plain,
                    Variant::Swap,
                    Variant::Repeat,
                    Variant::Clamp,
                    Variant::Broadcast,
                ] {
                    fixture::<T>(grid, op, policy, variant, |kernel| {
                        let profile = Profile::admit(kernel).unwrap();
                        assert_eq!(profile.operation, op);
                        assert_eq!(profile.underflow, policy);
                        assert_eq!(profile.extent, 3);
                        assert_eq!(profile.bytes, 3 * usize::from(T::TYPE.bit_width()) / 8);
                        assert_eq!(profile.output, PcuBindingRef::new(4, 1));
                        let expected = match variant {
                            Variant::Swap => [PcuBindingRef::new(2, 7), PcuBindingRef::new(2, 3)],
                            Variant::Repeat => [PcuBindingRef::new(2, 3); 2],
                            _ => [PcuBindingRef::new(2, 3), PcuBindingRef::new(2, 7)],
                        };
                        assert_eq!(profile.inputs, expected);
                        let broadcast = matches!(variant, Variant::Broadcast);
                        assert_eq!(profile.broadcast, [broadcast, false]);
                        assert_eq!(
                            profile.input_bytes[0],
                            if broadcast {
                                usize::from(T::TYPE.bit_width()) / 8
                            } else {
                                profile.bytes
                            }
                        );
                        assert_eq!(profile.input_bytes[1], profile.bytes);
                    });
                }
            }
        }
    }
}
fn negative<T: fusion_pcu::PcuScalar>() {
    for grid in [false, true] {
        for variant in [
            Variant::WrongWidth,
            Variant::InvalidSsa,
            Variant::WrongOutput,
            Variant::WrongShape,
            Variant::Portable,
        ] {
            fixture::<T>(
                grid,
                Op::Add,
                Policy::IeeeAfterRounding,
                variant,
                |kernel| {
                    let admitted_portable = matches!(variant, Variant::Portable)
                        && matches!(
                            T::TYPE,
                            fusion_pcu::PcuScalarType::F16
                                | fusion_pcu::PcuScalarType::BF16
                                | fusion_pcu::PcuScalarType::F8E4M3FN
                                | fusion_pcu::PcuScalarType::F8E5M2
                        );
                    if admitted_portable {
                        assert!(Profile::admit(kernel).is_ok());
                    } else {
                        assert!(
                            matches!(Profile::admit(kernel), Err(MetalError::Unsupported)),
                            "{variant:?}, grid={grid}, type={:?}",
                            T::TYPE
                        );
                    }
                },
            );
        }
    }
}

#[test]
fn complete_direct_grid_operation_policy_and_operand_admission() {
    complete::<f32>();
    complete::<f64>();
    complete::<fusion_pcu::PcuF16Bits>();
    complete::<fusion_pcu::PcuBf16Bits>();
    complete::<fusion_pcu::PcuF8E4M3FnBits>();
    complete::<fusion_pcu::PcuF8E5M2Bits>();
}
#[test]
fn incompatible_width_ssa_output_and_shape_reject_cold() {
    negative::<f32>();
    negative::<f64>();
    negative::<fusion_pcu::PcuF16Bits>();
    negative::<fusion_pcu::PcuBf16Bits>();
    negative::<fusion_pcu::PcuF8E4M3FnBits>();
    negative::<fusion_pcu::PcuF8E5M2Bits>();
}

fn portable<T: fusion_pcu::PcuScalar>() {
    for grid in [false, true] {
        for op in [Op::Add, Op::Sub, Op::Mul, Op::Div] {
            for policy in [
                Policy::IeeeAfterRounding,
                Policy::RejectSubnormalResult,
                Policy::AllowGradualUnderflow,
            ] {
                for variant in [Variant::Portable, Variant::PortableBroadcast] {
                    fixture::<T>(grid, op, policy, variant, |kernel| {
                        let description = fusion_pcu::describe_portable_v1_map(kernel).unwrap();
                        let profile = Profile::admit(kernel).unwrap();
                        assert_eq!(profile.broadcast, description.broadcast_loads);
                        assert_eq!(
                            profile.input_bytes[0],
                            usize::from(T::TYPE.bit_width()) / 8
                                * if matches!(variant, Variant::PortableBroadcast) {
                                    1
                                } else {
                                    3
                                }
                        );
                        let mut invalid = *kernel;
                        invalid.numerical_requirements.float_underflow =
                            if policy == Policy::IeeeAfterRounding {
                                Policy::AllowGradualUnderflow
                            } else {
                                Policy::IeeeAfterRounding
                            };
                        assert!(matches!(
                            Profile::admit(&invalid),
                            Err(MetalError::Unsupported)
                        ));
                    });
                }
            }
        }
    }
}
#[test]
fn portable_admits_only_frozen_low_format_map_and_matching_header() {
    portable::<fusion_pcu::PcuF16Bits>();
    portable::<fusion_pcu::PcuBf16Bits>();
    portable::<fusion_pcu::PcuF8E4M3FnBits>();
    portable::<fusion_pcu::PcuF8E5M2Bits>();
}

#[cfg(target_os = "macos")]
pub fn prepared_portable<T: fusion_pcu::PcuScalar>(
    session: &MetalSession,
    extent: u32,
    op: Op,
    policy: Policy,
    broadcast: [bool; 2],
) -> MetalPreparedFloatBinaryKernel {
    let mut prepared = None;
    fixture::<T>(false, op, policy, Variant::Portable, |template| {
        let mut kernel = *template;
        kernel.entry.logical_shape[0] = extent;
        let mut ops = template.ops.to_vec();
        for (index, is_broadcast) in broadcast.into_iter().enumerate() {
            if is_broadcast
                && let PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad { index, .. }) =
                    &mut ops[index]
            {
                *index = PcuDispatchIndex::BindingElementZero;
            }
        }
        kernel.ops = &ops;
        prepared = Some(session.prepare_float_binary_kernel(&kernel).unwrap());
    });
    prepared.unwrap()
}

#[path = "roles/roles.rs"]
mod roles;
