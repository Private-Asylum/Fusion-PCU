//! Fixed checked six-format floating tensor Dispatch factories.

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

const F16_BINDINGS: &[PcuBinding<'static>] = &[
    PcuBinding::value(
        Some("left"),
        0,
        0,
        PcuBindingStorageClass::Storage,
        PcuBindingAccess::ReadOnly,
        PcuValueType::Scalar(fusion_pcu::PcuScalarType::F16),
    ),
    PcuBinding::value(
        Some("right"),
        0,
        1,
        PcuBindingStorageClass::Storage,
        PcuBindingAccess::ReadOnly,
        PcuValueType::Scalar(fusion_pcu::PcuScalarType::F16),
    ),
    PcuBinding::value(
        Some("output"),
        0,
        2,
        PcuBindingStorageClass::Storage,
        PcuBindingAccess::WriteOnly,
        PcuValueType::Scalar(fusion_pcu::PcuScalarType::F16),
    ),
];
const F16_PROFILE: CheckedFloatProfile = CheckedFloatProfile {
    bindings: F16_BINDINGS,
    type_caps: PcuValueTypeCaps::for_scalar(fusion_pcu::PcuScalarType::F16)
        .union(PcuValueTypeCaps::SCALAR_VALUES),
    kernel_prefix: 0x4645_1000,
    names: [
        [
            "tensor_checked_f16_add_ieee",
            "tensor_checked_f16_sub_ieee",
            "tensor_checked_f16_mul_ieee",
            "tensor_checked_f16_div_ieee",
        ],
        [
            "tensor_checked_f16_add_reject_subnormal",
            "tensor_checked_f16_sub_reject_subnormal",
            "tensor_checked_f16_mul_reject_subnormal",
            "tensor_checked_f16_div_reject_subnormal",
        ],
        [
            "tensor_checked_f16_add_gradual",
            "tensor_checked_f16_sub_gradual",
            "tensor_checked_f16_mul_gradual",
            "tensor_checked_f16_div_gradual",
        ],
    ],
    ops: [
        [
            binary_ops!(
                PcuValueType::Scalar(fusion_pcu::PcuScalarType::F16),
                PcuDispatchFloatBinaryOp::Add,
                PcuFloatUnderflowPolicy::IeeeAfterRounding
            ),
            binary_ops!(
                PcuValueType::Scalar(fusion_pcu::PcuScalarType::F16),
                PcuDispatchFloatBinaryOp::Sub,
                PcuFloatUnderflowPolicy::IeeeAfterRounding
            ),
            binary_ops!(
                PcuValueType::Scalar(fusion_pcu::PcuScalarType::F16),
                PcuDispatchFloatBinaryOp::Mul,
                PcuFloatUnderflowPolicy::IeeeAfterRounding
            ),
            binary_ops!(
                PcuValueType::Scalar(fusion_pcu::PcuScalarType::F16),
                PcuDispatchFloatBinaryOp::Div,
                PcuFloatUnderflowPolicy::IeeeAfterRounding
            ),
        ],
        [
            binary_ops!(
                PcuValueType::Scalar(fusion_pcu::PcuScalarType::F16),
                PcuDispatchFloatBinaryOp::Add,
                PcuFloatUnderflowPolicy::RejectSubnormalResult
            ),
            binary_ops!(
                PcuValueType::Scalar(fusion_pcu::PcuScalarType::F16),
                PcuDispatchFloatBinaryOp::Sub,
                PcuFloatUnderflowPolicy::RejectSubnormalResult
            ),
            binary_ops!(
                PcuValueType::Scalar(fusion_pcu::PcuScalarType::F16),
                PcuDispatchFloatBinaryOp::Mul,
                PcuFloatUnderflowPolicy::RejectSubnormalResult
            ),
            binary_ops!(
                PcuValueType::Scalar(fusion_pcu::PcuScalarType::F16),
                PcuDispatchFloatBinaryOp::Div,
                PcuFloatUnderflowPolicy::RejectSubnormalResult
            ),
        ],
        [
            binary_ops!(
                PcuValueType::Scalar(fusion_pcu::PcuScalarType::F16),
                PcuDispatchFloatBinaryOp::Add,
                PcuFloatUnderflowPolicy::AllowGradualUnderflow
            ),
            binary_ops!(
                PcuValueType::Scalar(fusion_pcu::PcuScalarType::F16),
                PcuDispatchFloatBinaryOp::Sub,
                PcuFloatUnderflowPolicy::AllowGradualUnderflow
            ),
            binary_ops!(
                PcuValueType::Scalar(fusion_pcu::PcuScalarType::F16),
                PcuDispatchFloatBinaryOp::Mul,
                PcuFloatUnderflowPolicy::AllowGradualUnderflow
            ),
            binary_ops!(
                PcuValueType::Scalar(fusion_pcu::PcuScalarType::F16),
                PcuDispatchFloatBinaryOp::Div,
                PcuFloatUnderflowPolicy::AllowGradualUnderflow
            ),
        ],
    ],
};
const BF16_BINDINGS: &[PcuBinding<'static>] = &[
    PcuBinding::value(
        Some("left"),
        0,
        0,
        PcuBindingStorageClass::Storage,
        PcuBindingAccess::ReadOnly,
        PcuValueType::Scalar(fusion_pcu::PcuScalarType::BF16),
    ),
    PcuBinding::value(
        Some("right"),
        0,
        1,
        PcuBindingStorageClass::Storage,
        PcuBindingAccess::ReadOnly,
        PcuValueType::Scalar(fusion_pcu::PcuScalarType::BF16),
    ),
    PcuBinding::value(
        Some("output"),
        0,
        2,
        PcuBindingStorageClass::Storage,
        PcuBindingAccess::WriteOnly,
        PcuValueType::Scalar(fusion_pcu::PcuScalarType::BF16),
    ),
];
const BF16_PROFILE: CheckedFloatProfile = CheckedFloatProfile {
    bindings: BF16_BINDINGS,
    type_caps: PcuValueTypeCaps::for_scalar(fusion_pcu::PcuScalarType::BF16)
        .union(PcuValueTypeCaps::SCALAR_VALUES),
    kernel_prefix: 0x4645_b000,
    names: [
        [
            "tensor_checked_bf16_add_ieee",
            "tensor_checked_bf16_sub_ieee",
            "tensor_checked_bf16_mul_ieee",
            "tensor_checked_bf16_div_ieee",
        ],
        [
            "tensor_checked_bf16_add_reject_subnormal",
            "tensor_checked_bf16_sub_reject_subnormal",
            "tensor_checked_bf16_mul_reject_subnormal",
            "tensor_checked_bf16_div_reject_subnormal",
        ],
        [
            "tensor_checked_bf16_add_gradual",
            "tensor_checked_bf16_sub_gradual",
            "tensor_checked_bf16_mul_gradual",
            "tensor_checked_bf16_div_gradual",
        ],
    ],
    ops: [
        [
            binary_ops!(
                PcuValueType::Scalar(fusion_pcu::PcuScalarType::BF16),
                PcuDispatchFloatBinaryOp::Add,
                PcuFloatUnderflowPolicy::IeeeAfterRounding
            ),
            binary_ops!(
                PcuValueType::Scalar(fusion_pcu::PcuScalarType::BF16),
                PcuDispatchFloatBinaryOp::Sub,
                PcuFloatUnderflowPolicy::IeeeAfterRounding
            ),
            binary_ops!(
                PcuValueType::Scalar(fusion_pcu::PcuScalarType::BF16),
                PcuDispatchFloatBinaryOp::Mul,
                PcuFloatUnderflowPolicy::IeeeAfterRounding
            ),
            binary_ops!(
                PcuValueType::Scalar(fusion_pcu::PcuScalarType::BF16),
                PcuDispatchFloatBinaryOp::Div,
                PcuFloatUnderflowPolicy::IeeeAfterRounding
            ),
        ],
        [
            binary_ops!(
                PcuValueType::Scalar(fusion_pcu::PcuScalarType::BF16),
                PcuDispatchFloatBinaryOp::Add,
                PcuFloatUnderflowPolicy::RejectSubnormalResult
            ),
            binary_ops!(
                PcuValueType::Scalar(fusion_pcu::PcuScalarType::BF16),
                PcuDispatchFloatBinaryOp::Sub,
                PcuFloatUnderflowPolicy::RejectSubnormalResult
            ),
            binary_ops!(
                PcuValueType::Scalar(fusion_pcu::PcuScalarType::BF16),
                PcuDispatchFloatBinaryOp::Mul,
                PcuFloatUnderflowPolicy::RejectSubnormalResult
            ),
            binary_ops!(
                PcuValueType::Scalar(fusion_pcu::PcuScalarType::BF16),
                PcuDispatchFloatBinaryOp::Div,
                PcuFloatUnderflowPolicy::RejectSubnormalResult
            ),
        ],
        [
            binary_ops!(
                PcuValueType::Scalar(fusion_pcu::PcuScalarType::BF16),
                PcuDispatchFloatBinaryOp::Add,
                PcuFloatUnderflowPolicy::AllowGradualUnderflow
            ),
            binary_ops!(
                PcuValueType::Scalar(fusion_pcu::PcuScalarType::BF16),
                PcuDispatchFloatBinaryOp::Sub,
                PcuFloatUnderflowPolicy::AllowGradualUnderflow
            ),
            binary_ops!(
                PcuValueType::Scalar(fusion_pcu::PcuScalarType::BF16),
                PcuDispatchFloatBinaryOp::Mul,
                PcuFloatUnderflowPolicy::AllowGradualUnderflow
            ),
            binary_ops!(
                PcuValueType::Scalar(fusion_pcu::PcuScalarType::BF16),
                PcuDispatchFloatBinaryOp::Div,
                PcuFloatUnderflowPolicy::AllowGradualUnderflow
            ),
        ],
    ],
};
const F8E4M3FN_BINDINGS: &[PcuBinding<'static>] = &[
    PcuBinding::value(
        Some("left"),
        0,
        0,
        PcuBindingStorageClass::Storage,
        PcuBindingAccess::ReadOnly,
        PcuValueType::Scalar(fusion_pcu::PcuScalarType::F8E4M3FN),
    ),
    PcuBinding::value(
        Some("right"),
        0,
        1,
        PcuBindingStorageClass::Storage,
        PcuBindingAccess::ReadOnly,
        PcuValueType::Scalar(fusion_pcu::PcuScalarType::F8E4M3FN),
    ),
    PcuBinding::value(
        Some("output"),
        0,
        2,
        PcuBindingStorageClass::Storage,
        PcuBindingAccess::WriteOnly,
        PcuValueType::Scalar(fusion_pcu::PcuScalarType::F8E4M3FN),
    ),
];
const F8E4M3FN_PROFILE: CheckedFloatProfile = CheckedFloatProfile {
    bindings: F8E4M3FN_BINDINGS,
    type_caps: PcuValueTypeCaps::for_scalar(fusion_pcu::PcuScalarType::F8E4M3FN)
        .union(PcuValueTypeCaps::SCALAR_VALUES),
    kernel_prefix: 0x4645_e400,
    names: [
        [
            "tensor_checked_e4m3fn_add_ieee",
            "tensor_checked_e4m3fn_sub_ieee",
            "tensor_checked_e4m3fn_mul_ieee",
            "tensor_checked_e4m3fn_div_ieee",
        ],
        [
            "tensor_checked_e4m3fn_add_reject_subnormal",
            "tensor_checked_e4m3fn_sub_reject_subnormal",
            "tensor_checked_e4m3fn_mul_reject_subnormal",
            "tensor_checked_e4m3fn_div_reject_subnormal",
        ],
        [
            "tensor_checked_e4m3fn_add_gradual",
            "tensor_checked_e4m3fn_sub_gradual",
            "tensor_checked_e4m3fn_mul_gradual",
            "tensor_checked_e4m3fn_div_gradual",
        ],
    ],
    ops: [
        [
            binary_ops!(
                PcuValueType::Scalar(fusion_pcu::PcuScalarType::F8E4M3FN),
                PcuDispatchFloatBinaryOp::Add,
                PcuFloatUnderflowPolicy::IeeeAfterRounding
            ),
            binary_ops!(
                PcuValueType::Scalar(fusion_pcu::PcuScalarType::F8E4M3FN),
                PcuDispatchFloatBinaryOp::Sub,
                PcuFloatUnderflowPolicy::IeeeAfterRounding
            ),
            binary_ops!(
                PcuValueType::Scalar(fusion_pcu::PcuScalarType::F8E4M3FN),
                PcuDispatchFloatBinaryOp::Mul,
                PcuFloatUnderflowPolicy::IeeeAfterRounding
            ),
            binary_ops!(
                PcuValueType::Scalar(fusion_pcu::PcuScalarType::F8E4M3FN),
                PcuDispatchFloatBinaryOp::Div,
                PcuFloatUnderflowPolicy::IeeeAfterRounding
            ),
        ],
        [
            binary_ops!(
                PcuValueType::Scalar(fusion_pcu::PcuScalarType::F8E4M3FN),
                PcuDispatchFloatBinaryOp::Add,
                PcuFloatUnderflowPolicy::RejectSubnormalResult
            ),
            binary_ops!(
                PcuValueType::Scalar(fusion_pcu::PcuScalarType::F8E4M3FN),
                PcuDispatchFloatBinaryOp::Sub,
                PcuFloatUnderflowPolicy::RejectSubnormalResult
            ),
            binary_ops!(
                PcuValueType::Scalar(fusion_pcu::PcuScalarType::F8E4M3FN),
                PcuDispatchFloatBinaryOp::Mul,
                PcuFloatUnderflowPolicy::RejectSubnormalResult
            ),
            binary_ops!(
                PcuValueType::Scalar(fusion_pcu::PcuScalarType::F8E4M3FN),
                PcuDispatchFloatBinaryOp::Div,
                PcuFloatUnderflowPolicy::RejectSubnormalResult
            ),
        ],
        [
            binary_ops!(
                PcuValueType::Scalar(fusion_pcu::PcuScalarType::F8E4M3FN),
                PcuDispatchFloatBinaryOp::Add,
                PcuFloatUnderflowPolicy::AllowGradualUnderflow
            ),
            binary_ops!(
                PcuValueType::Scalar(fusion_pcu::PcuScalarType::F8E4M3FN),
                PcuDispatchFloatBinaryOp::Sub,
                PcuFloatUnderflowPolicy::AllowGradualUnderflow
            ),
            binary_ops!(
                PcuValueType::Scalar(fusion_pcu::PcuScalarType::F8E4M3FN),
                PcuDispatchFloatBinaryOp::Mul,
                PcuFloatUnderflowPolicy::AllowGradualUnderflow
            ),
            binary_ops!(
                PcuValueType::Scalar(fusion_pcu::PcuScalarType::F8E4M3FN),
                PcuDispatchFloatBinaryOp::Div,
                PcuFloatUnderflowPolicy::AllowGradualUnderflow
            ),
        ],
    ],
};
const F8E5M2_BINDINGS: &[PcuBinding<'static>] = &[
    PcuBinding::value(
        Some("left"),
        0,
        0,
        PcuBindingStorageClass::Storage,
        PcuBindingAccess::ReadOnly,
        PcuValueType::Scalar(fusion_pcu::PcuScalarType::F8E5M2),
    ),
    PcuBinding::value(
        Some("right"),
        0,
        1,
        PcuBindingStorageClass::Storage,
        PcuBindingAccess::ReadOnly,
        PcuValueType::Scalar(fusion_pcu::PcuScalarType::F8E5M2),
    ),
    PcuBinding::value(
        Some("output"),
        0,
        2,
        PcuBindingStorageClass::Storage,
        PcuBindingAccess::WriteOnly,
        PcuValueType::Scalar(fusion_pcu::PcuScalarType::F8E5M2),
    ),
];
const F8E5M2_PROFILE: CheckedFloatProfile = CheckedFloatProfile {
    bindings: F8E5M2_BINDINGS,
    type_caps: PcuValueTypeCaps::for_scalar(fusion_pcu::PcuScalarType::F8E5M2)
        .union(PcuValueTypeCaps::SCALAR_VALUES),
    kernel_prefix: 0x4645_e500,
    names: [
        [
            "tensor_checked_e5m2_add_ieee",
            "tensor_checked_e5m2_sub_ieee",
            "tensor_checked_e5m2_mul_ieee",
            "tensor_checked_e5m2_div_ieee",
        ],
        [
            "tensor_checked_e5m2_add_reject_subnormal",
            "tensor_checked_e5m2_sub_reject_subnormal",
            "tensor_checked_e5m2_mul_reject_subnormal",
            "tensor_checked_e5m2_div_reject_subnormal",
        ],
        [
            "tensor_checked_e5m2_add_gradual",
            "tensor_checked_e5m2_sub_gradual",
            "tensor_checked_e5m2_mul_gradual",
            "tensor_checked_e5m2_div_gradual",
        ],
    ],
    ops: [
        [
            binary_ops!(
                PcuValueType::Scalar(fusion_pcu::PcuScalarType::F8E5M2),
                PcuDispatchFloatBinaryOp::Add,
                PcuFloatUnderflowPolicy::IeeeAfterRounding
            ),
            binary_ops!(
                PcuValueType::Scalar(fusion_pcu::PcuScalarType::F8E5M2),
                PcuDispatchFloatBinaryOp::Sub,
                PcuFloatUnderflowPolicy::IeeeAfterRounding
            ),
            binary_ops!(
                PcuValueType::Scalar(fusion_pcu::PcuScalarType::F8E5M2),
                PcuDispatchFloatBinaryOp::Mul,
                PcuFloatUnderflowPolicy::IeeeAfterRounding
            ),
            binary_ops!(
                PcuValueType::Scalar(fusion_pcu::PcuScalarType::F8E5M2),
                PcuDispatchFloatBinaryOp::Div,
                PcuFloatUnderflowPolicy::IeeeAfterRounding
            ),
        ],
        [
            binary_ops!(
                PcuValueType::Scalar(fusion_pcu::PcuScalarType::F8E5M2),
                PcuDispatchFloatBinaryOp::Add,
                PcuFloatUnderflowPolicy::RejectSubnormalResult
            ),
            binary_ops!(
                PcuValueType::Scalar(fusion_pcu::PcuScalarType::F8E5M2),
                PcuDispatchFloatBinaryOp::Sub,
                PcuFloatUnderflowPolicy::RejectSubnormalResult
            ),
            binary_ops!(
                PcuValueType::Scalar(fusion_pcu::PcuScalarType::F8E5M2),
                PcuDispatchFloatBinaryOp::Mul,
                PcuFloatUnderflowPolicy::RejectSubnormalResult
            ),
            binary_ops!(
                PcuValueType::Scalar(fusion_pcu::PcuScalarType::F8E5M2),
                PcuDispatchFloatBinaryOp::Div,
                PcuFloatUnderflowPolicy::RejectSubnormalResult
            ),
        ],
        [
            binary_ops!(
                PcuValueType::Scalar(fusion_pcu::PcuScalarType::F8E5M2),
                PcuDispatchFloatBinaryOp::Add,
                PcuFloatUnderflowPolicy::AllowGradualUnderflow
            ),
            binary_ops!(
                PcuValueType::Scalar(fusion_pcu::PcuScalarType::F8E5M2),
                PcuDispatchFloatBinaryOp::Sub,
                PcuFloatUnderflowPolicy::AllowGradualUnderflow
            ),
            binary_ops!(
                PcuValueType::Scalar(fusion_pcu::PcuScalarType::F8E5M2),
                PcuDispatchFloatBinaryOp::Mul,
                PcuFloatUnderflowPolicy::AllowGradualUnderflow
            ),
            binary_ops!(
                PcuValueType::Scalar(fusion_pcu::PcuScalarType::F8E5M2),
                PcuDispatchFloatBinaryOp::Div,
                PcuFloatUnderflowPolicy::AllowGradualUnderflow
            ),
        ],
    ],
};

/// Builds a fixed dense checked f32/f64 binary kernel with its per-node policy.
pub(super) const fn kernel(
    op: PcuDispatchFloatBinaryOp,
    scalar_type: fusion_pcu::PcuScalarType,
    policy: PcuFloatUnderflowPolicy,
    logical_count: u32,
) -> Result<PcuDispatchKernelIr<'static>, super::CudaTensorExecutionError> {
    let profile = match scalar_type {
        fusion_pcu::PcuScalarType::F16 => F16_PROFILE,
        fusion_pcu::PcuScalarType::BF16 => BF16_PROFILE,
        fusion_pcu::PcuScalarType::F8E4M3FN => F8E4M3FN_PROFILE,
        fusion_pcu::PcuScalarType::F8E5M2 => F8E5M2_PROFILE,
        fusion_pcu::PcuScalarType::F32 => F32_PROFILE,
        fusion_pcu::PcuScalarType::F64 => F64_PROFILE,
        _ => return Err(super::CudaTensorExecutionError::InvalidPointwiseProfile),
    };
    let (op_index, op_tag) = operation_index_and_tag(op);
    let (policy_index, policy_offset) = policy_index_and_offset(policy);
    let id = profile.kernel_prefix + policy_offset + op_tag;
    Ok(PcuDispatchKernelIr {
        numerical_requirements: fusion_pcu::PcuImplementationRequirements {
            float_underflow: policy,
            ..PcuDispatchKernelIr::DEFAULT_REQUIREMENTS
        },
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
const RELU_F16_BINDINGS: &[PcuBinding<'static>] = &[
    PcuBinding::value(
        Some("input"),
        0,
        0,
        PcuBindingStorageClass::Storage,
        PcuBindingAccess::ReadOnly,
        PcuValueType::Scalar(fusion_pcu::PcuScalarType::F16),
    ),
    PcuBinding::value(
        Some("output"),
        0,
        1,
        PcuBindingStorageClass::Storage,
        PcuBindingAccess::WriteOnly,
        PcuValueType::Scalar(fusion_pcu::PcuScalarType::F16),
    ),
];
const RELU_F16_OPS: [&[PcuDispatchOp<'static>]; 3] = [
    unary_ops!(
        PcuValueType::Scalar(fusion_pcu::PcuScalarType::F16),
        PcuFloatUnderflowPolicy::IeeeAfterRounding
    ),
    unary_ops!(
        PcuValueType::Scalar(fusion_pcu::PcuScalarType::F16),
        PcuFloatUnderflowPolicy::RejectSubnormalResult
    ),
    unary_ops!(
        PcuValueType::Scalar(fusion_pcu::PcuScalarType::F16),
        PcuFloatUnderflowPolicy::AllowGradualUnderflow
    ),
];

const RELU_BF16_BINDINGS: &[PcuBinding<'static>] = &[
    PcuBinding::value(
        Some("input"),
        0,
        0,
        PcuBindingStorageClass::Storage,
        PcuBindingAccess::ReadOnly,
        PcuValueType::Scalar(fusion_pcu::PcuScalarType::BF16),
    ),
    PcuBinding::value(
        Some("output"),
        0,
        1,
        PcuBindingStorageClass::Storage,
        PcuBindingAccess::WriteOnly,
        PcuValueType::Scalar(fusion_pcu::PcuScalarType::BF16),
    ),
];
const RELU_BF16_OPS: [&[PcuDispatchOp<'static>]; 3] = [
    unary_ops!(
        PcuValueType::Scalar(fusion_pcu::PcuScalarType::BF16),
        PcuFloatUnderflowPolicy::IeeeAfterRounding
    ),
    unary_ops!(
        PcuValueType::Scalar(fusion_pcu::PcuScalarType::BF16),
        PcuFloatUnderflowPolicy::RejectSubnormalResult
    ),
    unary_ops!(
        PcuValueType::Scalar(fusion_pcu::PcuScalarType::BF16),
        PcuFloatUnderflowPolicy::AllowGradualUnderflow
    ),
];

const RELU_F8E4M3FN_BINDINGS: &[PcuBinding<'static>] = &[
    PcuBinding::value(
        Some("input"),
        0,
        0,
        PcuBindingStorageClass::Storage,
        PcuBindingAccess::ReadOnly,
        PcuValueType::Scalar(fusion_pcu::PcuScalarType::F8E4M3FN),
    ),
    PcuBinding::value(
        Some("output"),
        0,
        1,
        PcuBindingStorageClass::Storage,
        PcuBindingAccess::WriteOnly,
        PcuValueType::Scalar(fusion_pcu::PcuScalarType::F8E4M3FN),
    ),
];
const RELU_F8E4M3FN_OPS: [&[PcuDispatchOp<'static>]; 3] = [
    unary_ops!(
        PcuValueType::Scalar(fusion_pcu::PcuScalarType::F8E4M3FN),
        PcuFloatUnderflowPolicy::IeeeAfterRounding
    ),
    unary_ops!(
        PcuValueType::Scalar(fusion_pcu::PcuScalarType::F8E4M3FN),
        PcuFloatUnderflowPolicy::RejectSubnormalResult
    ),
    unary_ops!(
        PcuValueType::Scalar(fusion_pcu::PcuScalarType::F8E4M3FN),
        PcuFloatUnderflowPolicy::AllowGradualUnderflow
    ),
];

const RELU_F8E5M2_BINDINGS: &[PcuBinding<'static>] = &[
    PcuBinding::value(
        Some("input"),
        0,
        0,
        PcuBindingStorageClass::Storage,
        PcuBindingAccess::ReadOnly,
        PcuValueType::Scalar(fusion_pcu::PcuScalarType::F8E5M2),
    ),
    PcuBinding::value(
        Some("output"),
        0,
        1,
        PcuBindingStorageClass::Storage,
        PcuBindingAccess::WriteOnly,
        PcuValueType::Scalar(fusion_pcu::PcuScalarType::F8E5M2),
    ),
];
const RELU_F8E5M2_OPS: [&[PcuDispatchOp<'static>]; 3] = [
    unary_ops!(
        PcuValueType::Scalar(fusion_pcu::PcuScalarType::F8E5M2),
        PcuFloatUnderflowPolicy::IeeeAfterRounding
    ),
    unary_ops!(
        PcuValueType::Scalar(fusion_pcu::PcuScalarType::F8E5M2),
        PcuFloatUnderflowPolicy::RejectSubnormalResult
    ),
    unary_ops!(
        PcuValueType::Scalar(fusion_pcu::PcuScalarType::F8E5M2),
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
        fusion_pcu::PcuScalarType::F16 => (
            RELU_F16_BINDINGS,
            RELU_F16_OPS[index],
            PcuValueTypeCaps::for_scalar(fusion_pcu::PcuScalarType::F16),
            0x5245_1000,
        ),
        fusion_pcu::PcuScalarType::BF16 => (
            RELU_BF16_BINDINGS,
            RELU_BF16_OPS[index],
            PcuValueTypeCaps::for_scalar(fusion_pcu::PcuScalarType::BF16),
            0x5245_b000,
        ),
        fusion_pcu::PcuScalarType::F8E4M3FN => (
            RELU_F8E4M3FN_BINDINGS,
            RELU_F8E4M3FN_OPS[index],
            PcuValueTypeCaps::for_scalar(fusion_pcu::PcuScalarType::F8E4M3FN),
            0x5245_e400,
        ),
        fusion_pcu::PcuScalarType::F8E5M2 => (
            RELU_F8E5M2_BINDINGS,
            RELU_F8E5M2_OPS[index],
            PcuValueTypeCaps::for_scalar(fusion_pcu::PcuScalarType::F8E5M2),
            0x5245_e500,
        ),
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
        numerical_requirements: fusion_pcu::PcuImplementationRequirements {
            float_underflow: policy,
            ..PcuDispatchKernelIr::DEFAULT_REQUIREMENTS
        },
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

/// Lowers one dense checked floating Add/Sub/Mul/Div or `ReLU` tensor node.
///
/// This is the executor's integer-encoding runtime-compiler body with the captured
/// numerical tuple. Binary ABI: left, right, fresh output, u64 fault; `ReLU` ABI:
/// input, fresh output, u64 fault. Initialize fault to MAX, wait and inspect status
/// before publishing. No tensor Clamp, Uniform, training or `PortableV1` is offered.
///
/// # Errors
/// Returns unsupported graph operation, dtype, dense shape or numerical requirements.
pub fn lower_checked_float_tensor_to_cuda_source(
    graph: &fusion_pcu::dialect::tensor::Graph,
    value: fusion_pcu::dialect::tensor::ValueId,
) -> Result<String, super::CudaTensorExecutionError> {
    use fusion_pcu::dialect::tensor::{OpDescriptor, TensorOperationSupport, TensorUnsupportedReason};
    let node = graph.node(value)?;
    if node.numerical_options.reproducibility != fusion_pcu::PcuReproducibility::Unspecified {
        return Err(super::CudaTensorExecutionError::Unsupported {
            value,
            reason: TensorUnsupportedReason::NumericalPolicy {
                requirement: fusion_pcu::PcuNumericalRequirement::Reproducibility,
                options: node.numerical_options,
            },
        });
    }
    if !super::is_checked_float_type(node.scalar_type) {
        return Err(super::CudaTensorExecutionError::UnsupportedScalarType(
            node.scalar_type,
        ));
    }
    if let TensorOperationSupport::Unsupported { reason } =
        super::assess_low_float_node(graph, node)
    {
        return Err(super::CudaTensorExecutionError::Unsupported { value, reason });
    }
    let count = node
        .shape
        .iter()
        .try_fold(1u32, |count, &dim| {
            u32::try_from(dim)
                .ok()
                .and_then(|dim| count.checked_mul(dim))
        })
        .ok_or(super::CudaTensorExecutionError::SizeOverflow)?;
    let requirements = super::fixed_numerical_requirements(node);
    let base = match node.op {
        OpDescriptor::Relu { .. } => {
            relu_kernel(node.scalar_type, requirements.float_underflow, count)?
        }
        OpDescriptor::Add { .. } => kernel(
            PcuDispatchFloatBinaryOp::Add,
            node.scalar_type,
            requirements.float_underflow,
            count,
        )?,
        OpDescriptor::Sub { .. } => kernel(
            PcuDispatchFloatBinaryOp::Sub,
            node.scalar_type,
            requirements.float_underflow,
            count,
        )?,
        OpDescriptor::Mul { .. } => kernel(
            PcuDispatchFloatBinaryOp::Mul,
            node.scalar_type,
            requirements.float_underflow,
            count,
        )?,
        OpDescriptor::Div { .. } => kernel(
            PcuDispatchFloatBinaryOp::Div,
            node.scalar_type,
            requirements.float_underflow,
            count,
        )?,
        _ => return Err(super::CudaTensorExecutionError::InvalidPointwiseProfile),
    };
    let ir = PcuDispatchKernelIr {
        numerical_requirements: requirements,
        ..base
    };
    crate::lower_dispatch_to_cuda_rtc_source(&ir)
        .map_err(|error| super::CudaTensorExecutionError::Backend(error.into()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checked_float_factory_preserves_dtype_policy_and_unique_kernel_identity() {
        let mut identities = Vec::new();
        for (scalar_type, value_type, type_cap) in [
            (
                fusion_pcu::PcuScalarType::F16,
                PcuValueType::Scalar(fusion_pcu::PcuScalarType::F16),
                PcuValueTypeCaps::for_scalar(fusion_pcu::PcuScalarType::F16),
            ),
            (
                fusion_pcu::PcuScalarType::BF16,
                PcuValueType::Scalar(fusion_pcu::PcuScalarType::BF16),
                PcuValueTypeCaps::for_scalar(fusion_pcu::PcuScalarType::BF16),
            ),
            (
                fusion_pcu::PcuScalarType::F8E4M3FN,
                PcuValueType::Scalar(fusion_pcu::PcuScalarType::F8E4M3FN),
                PcuValueTypeCaps::for_scalar(fusion_pcu::PcuScalarType::F8E4M3FN),
            ),
            (
                fusion_pcu::PcuScalarType::F8E5M2,
                PcuValueType::Scalar(fusion_pcu::PcuScalarType::F8E5M2),
                PcuValueTypeCaps::for_scalar(fusion_pcu::PcuScalarType::F8E5M2),
            ),
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
        assert_eq!(identities.len(), 72);
    }

    #[test]
    fn checked_relu_factory_carries_finite_policy_at_all_six_formats() {
        for scalar in [
            fusion_pcu::PcuScalarType::F16,
            fusion_pcu::PcuScalarType::BF16,
            fusion_pcu::PcuScalarType::F8E4M3FN,
            fusion_pcu::PcuScalarType::F8E5M2,
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
                fusion_pcu::PcuScalarType::F128,
                PcuFloatUnderflowPolicy::IeeeAfterRounding,
                1,
            ),
            Err(super::super::CudaTensorExecutionError::InvalidPointwiseProfile)
        ));
    }
}
