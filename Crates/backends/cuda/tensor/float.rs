//! Fixed checked f32/f64 tensor Dispatch factories.

#[rustfmt::skip]
use fusion_pcu::{
    PcuBinding,
    PcuBindingAccess,
    PcuBindingRef,
    PcuBindingStorageClass,
    PcuDispatchControlOp,
    PcuDispatchDataOp,
    PcuDispatchEntryPoint,
    PcuDispatchFeatureCaps,
    PcuDispatchFloatBinaryOp,
    PcuDispatchIndex,
    PcuDispatchKernelIr,
    PcuDispatchOp,
    PcuDispatchValueId,
    PcuFloatUnderflowPolicy,
    PcuKernelId,
    PcuValueType,
    PcuValueTypeCaps,
};

const LEFT: PcuDispatchValueId = PcuDispatchValueId(1);
const RIGHT: PcuDispatchValueId = PcuDispatchValueId(2);
const RESULT: PcuDispatchValueId = PcuDispatchValueId(3);
const LEFT_REF: PcuBindingRef = PcuBindingRef::new(0, 0);
const RIGHT_REF: PcuBindingRef = PcuBindingRef::new(0, 1);
const OUTPUT_REF: PcuBindingRef = PcuBindingRef::new(0, 2);
const VALUE_TYPE: PcuValueType = PcuValueType::f32();

const BINDINGS: &[PcuBinding<'static>] = &[
    PcuBinding::value(
        Some("left"),
        0,
        0,
        PcuBindingStorageClass::Storage,
        PcuBindingAccess::ReadOnly,
        VALUE_TYPE,
    ),
    PcuBinding::value(
        Some("right"),
        0,
        1,
        PcuBindingStorageClass::Storage,
        PcuBindingAccess::ReadOnly,
        VALUE_TYPE,
    ),
    PcuBinding::value(
        Some("output"),
        0,
        2,
        PcuBindingStorageClass::Storage,
        PcuBindingAccess::WriteOnly,
        VALUE_TYPE,
    ),
];

const F64_BINDINGS: &[PcuBinding<'static>] = &[
    PcuBinding::value(
        Some("left"),
        0,
        0,
        PcuBindingStorageClass::Storage,
        PcuBindingAccess::ReadOnly,
        PcuValueType::f64(),
    ),
    PcuBinding::value(
        Some("right"),
        0,
        1,
        PcuBindingStorageClass::Storage,
        PcuBindingAccess::ReadOnly,
        PcuValueType::f64(),
    ),
    PcuBinding::value(
        Some("output"),
        0,
        2,
        PcuBindingStorageClass::Storage,
        PcuBindingAccess::WriteOnly,
        PcuValueType::f64(),
    ),
];

macro_rules! binary_ops {
    ($value_type:expr, $op:expr, $policy:expr) => {
        &[
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result: LEFT,
                binding: LEFT_REF,
                index: PcuDispatchIndex::InvocationId,
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result: RIGHT,
                binding: RIGHT_REF,
                index: PcuDispatchIndex::InvocationId,
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::CheckedFloatBinary {
                value_type: $value_type,
                op: $op,
                underflow_policy: $policy,
                range_policy: fusion_pcu::PcuRangePolicy::Reject,
                result: RESULT,
                lhs: LEFT,
                rhs: RIGHT,
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                binding: OUTPUT_REF,
                index: PcuDispatchIndex::InvocationId,
                value: RESULT,
            }),
            PcuDispatchOp::Control(PcuDispatchControlOp::Return),
        ]
    };
}

macro_rules! declare_ops {
    ($name:ident, $value_type:expr, $op:expr, $policy:expr) => {
        const $name: &[PcuDispatchOp<'static>] = binary_ops!($value_type, $op, $policy);
    };
}

const ADD_IEEE_OPS: &[PcuDispatchOp<'static>] = binary_ops!(
    PcuValueType::f32(),
    PcuDispatchFloatBinaryOp::Add,
    PcuFloatUnderflowPolicy::IeeeAfterRounding
);
const SUB_IEEE_OPS: &[PcuDispatchOp<'static>] = binary_ops!(
    PcuValueType::f32(),
    PcuDispatchFloatBinaryOp::Sub,
    PcuFloatUnderflowPolicy::IeeeAfterRounding
);
const MUL_IEEE_OPS: &[PcuDispatchOp<'static>] = binary_ops!(
    PcuValueType::f32(),
    PcuDispatchFloatBinaryOp::Mul,
    PcuFloatUnderflowPolicy::IeeeAfterRounding
);
const ADD_REJECT_OPS: &[PcuDispatchOp<'static>] = binary_ops!(
    PcuValueType::f32(),
    PcuDispatchFloatBinaryOp::Add,
    PcuFloatUnderflowPolicy::RejectSubnormalResult
);
const SUB_REJECT_OPS: &[PcuDispatchOp<'static>] = binary_ops!(
    PcuValueType::f32(),
    PcuDispatchFloatBinaryOp::Sub,
    PcuFloatUnderflowPolicy::RejectSubnormalResult
);
const MUL_REJECT_OPS: &[PcuDispatchOp<'static>] = binary_ops!(
    PcuValueType::f32(),
    PcuDispatchFloatBinaryOp::Mul,
    PcuFloatUnderflowPolicy::RejectSubnormalResult
);
const ADD_GRADUAL_OPS: &[PcuDispatchOp<'static>] = binary_ops!(
    PcuValueType::f32(),
    PcuDispatchFloatBinaryOp::Add,
    PcuFloatUnderflowPolicy::AllowGradualUnderflow
);
const SUB_GRADUAL_OPS: &[PcuDispatchOp<'static>] = binary_ops!(
    PcuValueType::f32(),
    PcuDispatchFloatBinaryOp::Sub,
    PcuFloatUnderflowPolicy::AllowGradualUnderflow
);
const MUL_GRADUAL_OPS: &[PcuDispatchOp<'static>] = binary_ops!(
    PcuValueType::f32(),
    PcuDispatchFloatBinaryOp::Mul,
    PcuFloatUnderflowPolicy::AllowGradualUnderflow
);

declare_ops!(
    ADD_IEEE_F64_OPS,
    PcuValueType::f64(),
    PcuDispatchFloatBinaryOp::Add,
    PcuFloatUnderflowPolicy::IeeeAfterRounding
);
declare_ops!(
    SUB_IEEE_F64_OPS,
    PcuValueType::f64(),
    PcuDispatchFloatBinaryOp::Sub,
    PcuFloatUnderflowPolicy::IeeeAfterRounding
);
declare_ops!(
    MUL_IEEE_F64_OPS,
    PcuValueType::f64(),
    PcuDispatchFloatBinaryOp::Mul,
    PcuFloatUnderflowPolicy::IeeeAfterRounding
);
declare_ops!(
    ADD_REJECT_F64_OPS,
    PcuValueType::f64(),
    PcuDispatchFloatBinaryOp::Add,
    PcuFloatUnderflowPolicy::RejectSubnormalResult
);
declare_ops!(
    SUB_REJECT_F64_OPS,
    PcuValueType::f64(),
    PcuDispatchFloatBinaryOp::Sub,
    PcuFloatUnderflowPolicy::RejectSubnormalResult
);
declare_ops!(
    MUL_REJECT_F64_OPS,
    PcuValueType::f64(),
    PcuDispatchFloatBinaryOp::Mul,
    PcuFloatUnderflowPolicy::RejectSubnormalResult
);
declare_ops!(
    ADD_GRADUAL_F64_OPS,
    PcuValueType::f64(),
    PcuDispatchFloatBinaryOp::Add,
    PcuFloatUnderflowPolicy::AllowGradualUnderflow
);
declare_ops!(
    SUB_GRADUAL_F64_OPS,
    PcuValueType::f64(),
    PcuDispatchFloatBinaryOp::Sub,
    PcuFloatUnderflowPolicy::AllowGradualUnderflow
);
declare_ops!(
    MUL_GRADUAL_F64_OPS,
    PcuValueType::f64(),
    PcuDispatchFloatBinaryOp::Mul,
    PcuFloatUnderflowPolicy::AllowGradualUnderflow
);

declare_ops!(
    DIV_IEEE_OPS,
    PcuValueType::f32(),
    PcuDispatchFloatBinaryOp::Div,
    PcuFloatUnderflowPolicy::IeeeAfterRounding
);

declare_ops!(
    DIV_REJECT_OPS,
    PcuValueType::f32(),
    PcuDispatchFloatBinaryOp::Div,
    PcuFloatUnderflowPolicy::RejectSubnormalResult
);

declare_ops!(
    DIV_GRADUAL_OPS,
    PcuValueType::f32(),
    PcuDispatchFloatBinaryOp::Div,
    PcuFloatUnderflowPolicy::AllowGradualUnderflow
);

declare_ops!(
    DIV_IEEE_F64_OPS,
    PcuValueType::f64(),
    PcuDispatchFloatBinaryOp::Div,
    PcuFloatUnderflowPolicy::IeeeAfterRounding
);

declare_ops!(
    DIV_REJECT_F64_OPS,
    PcuValueType::f64(),
    PcuDispatchFloatBinaryOp::Div,
    PcuFloatUnderflowPolicy::RejectSubnormalResult
);

declare_ops!(
    DIV_GRADUAL_F64_OPS,
    PcuValueType::f64(),
    PcuDispatchFloatBinaryOp::Div,
    PcuFloatUnderflowPolicy::AllowGradualUnderflow
);

#[derive(Clone, Copy)]
struct CheckedFloatProfile {
    bindings: &'static [PcuBinding<'static>],
    type_caps: PcuValueTypeCaps,
    kernel_prefix: u32,
    names: [[&'static str; 4]; 3],
    ops: [[&'static [PcuDispatchOp<'static>]; 4]; 3],
}

const F32_PROFILE: CheckedFloatProfile = CheckedFloatProfile {
    bindings: BINDINGS,
    type_caps: PcuValueTypeCaps::FLOAT32.union(PcuValueTypeCaps::SCALAR_VALUES),
    kernel_prefix: 0x4643_3200,
    names: [
        [
            "tensor_checked_f32_add_ieee",
            "tensor_checked_f32_sub_ieee",
            "tensor_checked_f32_mul_ieee",
            "tensor_checked_f32_div_ieee",
        ],
        [
            "tensor_checked_f32_add_reject_subnormal",
            "tensor_checked_f32_sub_reject_subnormal",
            "tensor_checked_f32_mul_reject_subnormal",
            "tensor_checked_f32_div_reject_subnormal",
        ],
        [
            "tensor_checked_f32_add_gradual",
            "tensor_checked_f32_sub_gradual",
            "tensor_checked_f32_mul_gradual",
            "tensor_checked_f32_div_gradual",
        ],
    ],
    ops: [
        [ADD_IEEE_OPS, SUB_IEEE_OPS, MUL_IEEE_OPS, DIV_IEEE_OPS],
        [
            ADD_REJECT_OPS,
            SUB_REJECT_OPS,
            MUL_REJECT_OPS,
            DIV_REJECT_OPS,
        ],
        [
            ADD_GRADUAL_OPS,
            SUB_GRADUAL_OPS,
            MUL_GRADUAL_OPS,
            DIV_GRADUAL_OPS,
        ],
    ],
};

const F64_PROFILE: CheckedFloatProfile = CheckedFloatProfile {
    bindings: F64_BINDINGS,
    type_caps: PcuValueTypeCaps::FLOAT64.union(PcuValueTypeCaps::SCALAR_VALUES),
    kernel_prefix: 0x4644_3200,
    names: [
        [
            "tensor_checked_f64_add_ieee",
            "tensor_checked_f64_sub_ieee",
            "tensor_checked_f64_mul_ieee",
            "tensor_checked_f64_div_ieee",
        ],
        [
            "tensor_checked_f64_add_reject_subnormal",
            "tensor_checked_f64_sub_reject_subnormal",
            "tensor_checked_f64_mul_reject_subnormal",
            "tensor_checked_f64_div_reject_subnormal",
        ],
        [
            "tensor_checked_f64_add_gradual",
            "tensor_checked_f64_sub_gradual",
            "tensor_checked_f64_mul_gradual",
            "tensor_checked_f64_div_gradual",
        ],
    ],
    ops: [
        [
            ADD_IEEE_F64_OPS,
            SUB_IEEE_F64_OPS,
            MUL_IEEE_F64_OPS,
            DIV_IEEE_F64_OPS,
        ],
        [
            ADD_REJECT_F64_OPS,
            SUB_REJECT_F64_OPS,
            MUL_REJECT_F64_OPS,
            DIV_REJECT_F64_OPS,
        ],
        [
            ADD_GRADUAL_F64_OPS,
            SUB_GRADUAL_F64_OPS,
            MUL_GRADUAL_F64_OPS,
            DIV_GRADUAL_F64_OPS,
        ],
    ],
};

const fn operation_index_and_tag(op: PcuDispatchFloatBinaryOp) -> (usize, u32) {
    match op {
        PcuDispatchFloatBinaryOp::Add => (0, 1),
        PcuDispatchFloatBinaryOp::Sub => (1, 2),
        PcuDispatchFloatBinaryOp::Mul => (2, 3),
        PcuDispatchFloatBinaryOp::Div => (3, 4),
    }
}

const fn policy_index_and_offset(policy: PcuFloatUnderflowPolicy) -> (usize, u32) {
    match policy {
        PcuFloatUnderflowPolicy::IeeeAfterRounding => (0, 0),
        PcuFloatUnderflowPolicy::RejectSubnormalResult => (1, 0x10),
        PcuFloatUnderflowPolicy::AllowGradualUnderflow => (2, 0x20),
    }
}

/// Builds a fixed dense checked f32/f64 binary kernel with its per-node policy.
pub(super) const fn kernel(
    op: PcuDispatchFloatBinaryOp,
    scalar_type: fusion_pcu::PcuScalarType,
    policy: PcuFloatUnderflowPolicy,
    logical_count: u32,
) -> Result<PcuDispatchKernelIr<'static>, super::CudaTensorExecutionError> {
    let profile = match scalar_type {
        fusion_pcu::PcuScalarType::F32 => F32_PROFILE,
        fusion_pcu::PcuScalarType::F64 => F64_PROFILE,
        _ => return Err(super::CudaTensorExecutionError::InvalidPointwiseProfile),
    };
    let (op_index, op_tag) = operation_index_and_tag(op);
    let (policy_index, policy_offset) = policy_index_and_offset(policy);
    let id = profile.kernel_prefix + policy_offset + op_tag;
    Ok(PcuDispatchKernelIr {
        id: PcuKernelId(id),
        entry: PcuDispatchEntryPoint {
            name: profile.names[policy_index][op_index],
            logical_shape: [logical_count, 1, 1],
        },
        bindings: profile.bindings,
        ports: &[],
        parameters: &[],
        ops: profile.ops[policy_index][op_index],
        type_caps: profile.type_caps,
        feature_caps: PcuDispatchFeatureCaps::empty(),
    })
}

macro_rules! unary_ops {
    ($type:expr, $policy:expr) => {
        &[
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result: LEFT,
                binding: LEFT_REF,
                index: PcuDispatchIndex::InvocationId,
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::CheckedFloatUnary {
                value_type: $type,
                op: fusion_pcu::PcuDispatchFloatUnaryOp::Relu,
                underflow_policy: $policy,
                range_policy: fusion_pcu::PcuRangePolicy::Reject,
                result: RESULT,
                value: LEFT,
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                binding: PcuBindingRef::new(0, 1),
                index: PcuDispatchIndex::InvocationId,
                value: RESULT,
            }),
            PcuDispatchOp::Control(PcuDispatchControlOp::Return),
        ]
    };
}
const RELU_F32_BINDINGS: &[PcuBinding<'static>] = &[
    PcuBinding::value(
        Some("input"),
        0,
        0,
        PcuBindingStorageClass::Storage,
        PcuBindingAccess::ReadOnly,
        PcuValueType::f32(),
    ),
    PcuBinding::value(
        Some("output"),
        0,
        1,
        PcuBindingStorageClass::Storage,
        PcuBindingAccess::WriteOnly,
        PcuValueType::f32(),
    ),
];
const RELU_F64_BINDINGS: &[PcuBinding<'static>] = &[
    PcuBinding::value(
        Some("input"),
        0,
        0,
        PcuBindingStorageClass::Storage,
        PcuBindingAccess::ReadOnly,
        PcuValueType::f64(),
    ),
    PcuBinding::value(
        Some("output"),
        0,
        1,
        PcuBindingStorageClass::Storage,
        PcuBindingAccess::WriteOnly,
        PcuValueType::f64(),
    ),
];
const RELU_F32_OPS: [&[PcuDispatchOp<'static>]; 3] = [
    unary_ops!(
        PcuValueType::f32(),
        PcuFloatUnderflowPolicy::IeeeAfterRounding
    ),
    unary_ops!(
        PcuValueType::f32(),
        PcuFloatUnderflowPolicy::RejectSubnormalResult
    ),
    unary_ops!(
        PcuValueType::f32(),
        PcuFloatUnderflowPolicy::AllowGradualUnderflow
    ),
];
const RELU_F64_OPS: [&[PcuDispatchOp<'static>]; 3] = [
    unary_ops!(
        PcuValueType::f64(),
        PcuFloatUnderflowPolicy::IeeeAfterRounding
    ),
    unary_ops!(
        PcuValueType::f64(),
        PcuFloatUnderflowPolicy::RejectSubnormalResult
    ),
    unary_ops!(
        PcuValueType::f64(),
        PcuFloatUnderflowPolicy::AllowGradualUnderflow
    ),
];
pub(super) fn relu_kernel(
    scalar_type: fusion_pcu::PcuScalarType,
    policy: PcuFloatUnderflowPolicy,
    logical_count: u32,
) -> Result<PcuDispatchKernelIr<'static>, super::CudaTensorExecutionError> {
    let (index, offset) = policy_index_and_offset(policy);
    let (bindings, ops, caps, prefix) = match scalar_type {
        fusion_pcu::PcuScalarType::F32 => (
            RELU_F32_BINDINGS,
            RELU_F32_OPS[index],
            PcuValueTypeCaps::FLOAT32,
            0x5243_3200,
        ),
        fusion_pcu::PcuScalarType::F64 => (
            RELU_F64_BINDINGS,
            RELU_F64_OPS[index],
            PcuValueTypeCaps::FLOAT64,
            0x5243_6400,
        ),
        _ => return Err(super::CudaTensorExecutionError::InvalidPointwiseProfile),
    };
    Ok(PcuDispatchKernelIr {
        id: PcuKernelId(prefix + offset),
        entry: PcuDispatchEntryPoint {
            name: "tensor_checked_relu",
            logical_shape: [logical_count, 1, 1],
        },
        bindings,
        ports: &[],
        parameters: &[],
        ops,
        type_caps: caps.union(PcuValueTypeCaps::SCALAR_VALUES),
        feature_caps: PcuDispatchFeatureCaps::default(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checked_float_factory_preserves_dtype_policy_and_unique_kernel_identity() {
        let mut identities = Vec::new();
        for (scalar_type, value_type, type_cap) in [
            (
                fusion_pcu::PcuScalarType::F32,
                PcuValueType::f32(),
                PcuValueTypeCaps::FLOAT32,
            ),
            (
                fusion_pcu::PcuScalarType::F64,
                PcuValueType::f64(),
                PcuValueTypeCaps::FLOAT64,
            ),
        ] {
            for (op, op_name) in [
                (PcuDispatchFloatBinaryOp::Add, "add"),
                (PcuDispatchFloatBinaryOp::Sub, "sub"),
                (PcuDispatchFloatBinaryOp::Mul, "mul"),
                (PcuDispatchFloatBinaryOp::Div, "div"),
            ] {
                for (policy, policy_name) in [
                    (PcuFloatUnderflowPolicy::IeeeAfterRounding, "ieee"),
                    (
                        PcuFloatUnderflowPolicy::RejectSubnormalResult,
                        "reject_subnormal",
                    ),
                    (PcuFloatUnderflowPolicy::AllowGradualUnderflow, "gradual"),
                ] {
                    let kernel =
                        kernel(op, scalar_type, policy, 17).expect("supported float profile");
                    assert!(!identities.contains(&kernel.id));
                    identities.push(kernel.id);
                    assert!(kernel.type_caps.contains(type_cap));
                    assert_eq!(kernel.bindings.len(), 3);
                    assert!(
                        kernel.bindings.iter().all(|binding| {
                            binding.binding_type.value_type() == Some(value_type)
                        })
                    );
                    assert!(kernel.entry.name.contains(op_name));
                    assert!(kernel.entry.name.contains(policy_name));
                    assert!(kernel.ops.iter().any(|instruction| matches!(
                        instruction,
                        PcuDispatchOp::Data(PcuDispatchDataOp::CheckedFloatBinary {
                            value_type: actual_type,
                            op: actual_op,
                            underflow_policy: actual_policy,
                            ..
                        }) if *actual_type == value_type && *actual_op == op && *actual_policy == policy
                    )));
                    assert!(
                        fusion_pcu::validate_checked_float_binary_kernel(
                            &kernel, value_type, op, policy, type_cap,
                        )
                        .is_ok()
                    );
                }
            }
        }
        assert_eq!(identities.len(), 24);
    }

    #[test]
    fn checked_relu_factory_carries_finite_policy_at_both_widths() {
        for scalar in [
            fusion_pcu::PcuScalarType::F32,
            fusion_pcu::PcuScalarType::F64,
        ] {
            for policy in [
                PcuFloatUnderflowPolicy::IeeeAfterRounding,
                PcuFloatUnderflowPolicy::RejectSubnormalResult,
                PcuFloatUnderflowPolicy::AllowGradualUnderflow,
            ] {
                let kernel = relu_kernel(scalar, policy, 8).unwrap();
                assert_eq!(kernel.bindings.len(), 2);
                assert!(kernel.ops.iter().any(|op| matches!(op, PcuDispatchOp::Data(PcuDispatchDataOp::CheckedFloatUnary {op: fusion_pcu::PcuDispatchFloatUnaryOp::Relu, underflow_policy, range_policy: fusion_pcu::PcuRangePolicy::Reject, ..}) if *underflow_policy == policy)));
                assert!(
                    fusion_pcu::validate_checked_float_map_kernel(
                        &kernel,
                        PcuValueType::Scalar(scalar),
                        PcuValueTypeCaps::for_scalar(scalar)
                    )
                    .is_ok()
                );
            }
        }
    }

    #[test]
    fn checked_float_factory_rejects_unimplemented_scalar_types() {
        assert!(matches!(
            kernel(
                PcuDispatchFloatBinaryOp::Add,
                fusion_pcu::PcuScalarType::F16,
                PcuFloatUnderflowPolicy::IeeeAfterRounding,
                1,
            ),
            Err(super::super::CudaTensorExecutionError::InvalidPointwiseProfile)
        ));
    }
}
