//! Fixed scalar-typed pointwise PCU Dispatch factories.
//!
//! The F32 factories remain in the parent module to preserve their existing IR and identifiers.
//! This module adds the separately typed F64 fixed profile for owned tensor execution.

#[rustfmt::skip]
use fusion_pcu::{
    PcuBinding,
    PcuBindingAccess,
    PcuBindingRef,
    PcuBindingStorageClass,
    PcuDispatchAluOp,
    PcuDispatchControlOp,
    PcuDispatchDataOp,
    PcuDispatchEntryPoint,
    PcuDispatchFeatureCaps,
    PcuDispatchIndex,
    PcuDispatchKernelIr,
    PcuDispatchOp,
    PcuDispatchValueId,
    PcuParameterValue,
    PcuValueType,
    PcuValueTypeCaps,
};

#[rustfmt::skip]
use super::{
    CudaTensorExecutionError,
    TensorDispatchKind,
    TensorPointwiseScalarType,
};
#[rustfmt::skip]
use fusion_pcu::dialect::tensor::{
    TensorBinaryOperand,
    TensorBinaryOperation,
};

const LEFT: PcuDispatchValueId = PcuDispatchValueId(1);
const RIGHT: PcuDispatchValueId = PcuDispatchValueId(2);
const RESULT: PcuDispatchValueId = PcuDispatchValueId(3);
const ZERO: PcuDispatchValueId = PcuDispatchValueId(4);
const EPILOGUE_RESULT: PcuDispatchValueId = PcuDispatchValueId(5);
const LEFT_REF: PcuBindingRef = PcuBindingRef::new(0, 0);
const RIGHT_REF: PcuBindingRef = PcuBindingRef::new(0, 1);
const OUTPUT_REF: PcuBindingRef = PcuBindingRef::new(0, 2);
const INPUT_REF: PcuBindingRef = PcuBindingRef::new(0, 0);
const RELU_OUTPUT_REF: PcuBindingRef = PcuBindingRef::new(0, 1);
const DONOR_REF: PcuBindingRef = PcuBindingRef::new(0, 0);
const OTHER_REF: PcuBindingRef = PcuBindingRef::new(0, 1);

const BINARY_BINDINGS: &[PcuBinding<'static>] = &[
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

const RELU_BINDINGS: &[PcuBinding<'static>] = &[
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

const CONSUMING_RELU_BINDINGS: &[PcuBinding<'static>] = &[PcuBinding::value(
    Some("inout"),
    0,
    0,
    PcuBindingStorageClass::Storage,
    PcuBindingAccess::ReadWrite,
    PcuValueType::f32(),
)];

const CONSUMING_RELU_OPS: &[PcuDispatchOp<'static>] = &[
    PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
        result: LEFT,
        binding: INPUT_REF,
        index: PcuDispatchIndex::InvocationId,
    }),
    PcuDispatchOp::Data(PcuDispatchDataOp::Constant {
        result: ZERO,
        value: PcuParameterValue::F32(0.0_f32.to_bits()),
    }),
    PcuDispatchOp::Data(PcuDispatchDataOp::Alu {
        value_type: PcuValueType::f32(),
        result: RESULT,
        op: PcuDispatchAluOp::Max,
        lhs: LEFT,
        rhs: ZERO,
    }),
    PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
        binding: INPUT_REF,
        index: PcuDispatchIndex::InvocationId,
        value: RESULT,
    }),
    PcuDispatchOp::Control(PcuDispatchControlOp::Return),
];

const CONSUMING_RELU_F64_BINDINGS: &[PcuBinding<'static>] = &[PcuBinding::value(
    Some("inout"),
    0,
    0,
    PcuBindingStorageClass::Storage,
    PcuBindingAccess::ReadWrite,
    PcuValueType::f64(),
)];

const CONSUMING_RELU_F64_OPS: &[PcuDispatchOp<'static>] = &[
    PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
        result: LEFT,
        binding: INPUT_REF,
        index: PcuDispatchIndex::InvocationId,
    }),
    PcuDispatchOp::Data(PcuDispatchDataOp::Constant {
        result: ZERO,
        value: PcuParameterValue::F64(0.0_f64.to_bits()),
    }),
    PcuDispatchOp::Data(PcuDispatchDataOp::Alu {
        value_type: PcuValueType::f64(),
        result: RESULT,
        op: PcuDispatchAluOp::Max,
        lhs: LEFT,
        rhs: ZERO,
    }),
    PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
        binding: INPUT_REF,
        index: PcuDispatchIndex::InvocationId,
        value: RESULT,
    }),
    PcuDispatchOp::Control(PcuDispatchControlOp::Return),
];

const CONSUMING_BINARY_F32_BINDINGS: &[PcuBinding<'static>] = &[
    PcuBinding::value(
        Some("donor"),
        0,
        0,
        PcuBindingStorageClass::Storage,
        PcuBindingAccess::ReadWrite,
        PcuValueType::f32(),
    ),
    PcuBinding::value(
        Some("other"),
        0,
        1,
        PcuBindingStorageClass::Storage,
        PcuBindingAccess::ReadOnly,
        PcuValueType::f32(),
    ),
];

const CONSUMING_BINARY_F64_BINDINGS: &[PcuBinding<'static>] = &[
    PcuBinding::value(
        Some("donor"),
        0,
        0,
        PcuBindingStorageClass::Storage,
        PcuBindingAccess::ReadWrite,
        PcuValueType::f64(),
    ),
    PcuBinding::value(
        Some("other"),
        0,
        1,
        PcuBindingStorageClass::Storage,
        PcuBindingAccess::ReadOnly,
        PcuValueType::f64(),
    ),
];

macro_rules! consuming_binary_ops {
    ($name:ident, $value_type:expr, $operation:expr, $donor_operand:expr) => {
        const $name: &[PcuDispatchOp<'static>] = &[
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result: LEFT,
                binding: if matches!($donor_operand, TensorBinaryOperand::Left) {
                    DONOR_REF
                } else {
                    OTHER_REF
                },
                index: PcuDispatchIndex::InvocationId,
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result: RIGHT,
                binding: if matches!($donor_operand, TensorBinaryOperand::Right) {
                    DONOR_REF
                } else {
                    OTHER_REF
                },
                index: PcuDispatchIndex::InvocationId,
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::Alu {
                value_type: $value_type,
                result: RESULT,
                op: $operation,
                lhs: LEFT,
                rhs: RIGHT,
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                binding: DONOR_REF,
                index: PcuDispatchIndex::InvocationId,
                value: RESULT,
            }),
            PcuDispatchOp::Control(PcuDispatchControlOp::Return),
        ];
    };
}

consuming_binary_ops!(
    CONSUMING_ADD_F32_LEFT,
    PcuValueType::f32(),
    PcuDispatchAluOp::Add,
    TensorBinaryOperand::Left
);
consuming_binary_ops!(
    CONSUMING_ADD_F32_RIGHT,
    PcuValueType::f32(),
    PcuDispatchAluOp::Add,
    TensorBinaryOperand::Right
);
consuming_binary_ops!(
    CONSUMING_SUB_F32_LEFT,
    PcuValueType::f32(),
    PcuDispatchAluOp::Sub,
    TensorBinaryOperand::Left
);
consuming_binary_ops!(
    CONSUMING_SUB_F32_RIGHT,
    PcuValueType::f32(),
    PcuDispatchAluOp::Sub,
    TensorBinaryOperand::Right
);
consuming_binary_ops!(
    CONSUMING_MUL_F32_LEFT,
    PcuValueType::f32(),
    PcuDispatchAluOp::Mul,
    TensorBinaryOperand::Left
);
consuming_binary_ops!(
    CONSUMING_MUL_F32_RIGHT,
    PcuValueType::f32(),
    PcuDispatchAluOp::Mul,
    TensorBinaryOperand::Right
);
consuming_binary_ops!(
    CONSUMING_ADD_F64_LEFT,
    PcuValueType::f64(),
    PcuDispatchAluOp::Add,
    TensorBinaryOperand::Left
);
consuming_binary_ops!(
    CONSUMING_ADD_F64_RIGHT,
    PcuValueType::f64(),
    PcuDispatchAluOp::Add,
    TensorBinaryOperand::Right
);
consuming_binary_ops!(
    CONSUMING_SUB_F64_LEFT,
    PcuValueType::f64(),
    PcuDispatchAluOp::Sub,
    TensorBinaryOperand::Left
);
consuming_binary_ops!(
    CONSUMING_SUB_F64_RIGHT,
    PcuValueType::f64(),
    PcuDispatchAluOp::Sub,
    TensorBinaryOperand::Right
);
consuming_binary_ops!(
    CONSUMING_MUL_F64_LEFT,
    PcuValueType::f64(),
    PcuDispatchAluOp::Mul,
    TensorBinaryOperand::Left
);
consuming_binary_ops!(
    CONSUMING_MUL_F64_RIGHT,
    PcuValueType::f64(),
    PcuDispatchAluOp::Mul,
    TensorBinaryOperand::Right
);

pub(super) fn consuming_binary_kernel(
    scalar_type: TensorPointwiseScalarType,
    operation: TensorBinaryOperation,
    donor_operand: TensorBinaryOperand,
    logical_count: u32,
) -> PcuDispatchKernelIr<'static> {
    let (name, base_id, bindings, ops, type_caps) =
        consuming_binary_profile(scalar_type, operation, donor_operand);
    PcuDispatchKernelIr {
        id: fusion_pcu::PcuKernelId(base_id),
        entry: PcuDispatchEntryPoint {
            name,
            logical_shape: [logical_count, 1, 1],
        },
        bindings,
        ports: &[],
        parameters: &[],
        ops,
        type_caps: type_caps.union(PcuValueTypeCaps::SCALAR_VALUES),
        feature_caps: PcuDispatchFeatureCaps::default(),
    }
}

type ConsumingBinaryProfile = (
    &'static str,
    u32,
    &'static [PcuBinding<'static>],
    &'static [PcuDispatchOp<'static>],
    PcuValueTypeCaps,
);

// Keep the finite scalar/op/operand manifest together so the binding, kernel ID, and name
// remain visibly one-to-one; splitting its match arms would make this static routing harder to audit.
#[allow(clippy::too_many_lines)]
const fn consuming_binary_profile(
    scalar_type: TensorPointwiseScalarType,
    operation: TensorBinaryOperation,
    donor_operand: TensorBinaryOperand,
) -> ConsumingBinaryProfile {
    match (scalar_type, operation, donor_operand) {
        (TensorPointwiseScalarType::F32, TensorBinaryOperation::Add, TensorBinaryOperand::Left) => {
            (
                "tensor_consuming_add_f32_left",
                0x4353_1000,
                CONSUMING_BINARY_F32_BINDINGS,
                CONSUMING_ADD_F32_LEFT,
                PcuValueTypeCaps::FLOAT32,
            )
        }
        (
            TensorPointwiseScalarType::F32,
            TensorBinaryOperation::Add,
            TensorBinaryOperand::Right,
        ) => (
            "tensor_consuming_add_f32_right",
            0x4353_1001,
            CONSUMING_BINARY_F32_BINDINGS,
            CONSUMING_ADD_F32_RIGHT,
            PcuValueTypeCaps::FLOAT32,
        ),
        (TensorPointwiseScalarType::F32, TensorBinaryOperation::Sub, TensorBinaryOperand::Left) => {
            (
                "tensor_consuming_sub_f32_left",
                0x4353_1002,
                CONSUMING_BINARY_F32_BINDINGS,
                CONSUMING_SUB_F32_LEFT,
                PcuValueTypeCaps::FLOAT32,
            )
        }
        (
            TensorPointwiseScalarType::F32,
            TensorBinaryOperation::Sub,
            TensorBinaryOperand::Right,
        ) => (
            "tensor_consuming_sub_f32_right",
            0x4353_1003,
            CONSUMING_BINARY_F32_BINDINGS,
            CONSUMING_SUB_F32_RIGHT,
            PcuValueTypeCaps::FLOAT32,
        ),
        (TensorPointwiseScalarType::F32, TensorBinaryOperation::Mul, TensorBinaryOperand::Left) => {
            (
                "tensor_consuming_mul_f32_left",
                0x4353_1004,
                CONSUMING_BINARY_F32_BINDINGS,
                CONSUMING_MUL_F32_LEFT,
                PcuValueTypeCaps::FLOAT32,
            )
        }
        (
            TensorPointwiseScalarType::F32,
            TensorBinaryOperation::Mul,
            TensorBinaryOperand::Right,
        ) => (
            "tensor_consuming_mul_f32_right",
            0x4353_1005,
            CONSUMING_BINARY_F32_BINDINGS,
            CONSUMING_MUL_F32_RIGHT,
            PcuValueTypeCaps::FLOAT32,
        ),
        (TensorPointwiseScalarType::F64, TensorBinaryOperation::Add, TensorBinaryOperand::Left) => {
            (
                "tensor_consuming_add_f64_left",
                0x4353_2000,
                CONSUMING_BINARY_F64_BINDINGS,
                CONSUMING_ADD_F64_LEFT,
                PcuValueTypeCaps::FLOAT64,
            )
        }
        (
            TensorPointwiseScalarType::F64,
            TensorBinaryOperation::Add,
            TensorBinaryOperand::Right,
        ) => (
            "tensor_consuming_add_f64_right",
            0x4353_2001,
            CONSUMING_BINARY_F64_BINDINGS,
            CONSUMING_ADD_F64_RIGHT,
            PcuValueTypeCaps::FLOAT64,
        ),
        (TensorPointwiseScalarType::F64, TensorBinaryOperation::Sub, TensorBinaryOperand::Left) => {
            (
                "tensor_consuming_sub_f64_left",
                0x4353_2002,
                CONSUMING_BINARY_F64_BINDINGS,
                CONSUMING_SUB_F64_LEFT,
                PcuValueTypeCaps::FLOAT64,
            )
        }
        (
            TensorPointwiseScalarType::F64,
            TensorBinaryOperation::Sub,
            TensorBinaryOperand::Right,
        ) => (
            "tensor_consuming_sub_f64_right",
            0x4353_2003,
            CONSUMING_BINARY_F64_BINDINGS,
            CONSUMING_SUB_F64_RIGHT,
            PcuValueTypeCaps::FLOAT64,
        ),
        (TensorPointwiseScalarType::F64, TensorBinaryOperation::Mul, TensorBinaryOperand::Left) => {
            (
                "tensor_consuming_mul_f64_left",
                0x4353_2004,
                CONSUMING_BINARY_F64_BINDINGS,
                CONSUMING_MUL_F64_LEFT,
                PcuValueTypeCaps::FLOAT64,
            )
        }
        (
            TensorPointwiseScalarType::F64,
            TensorBinaryOperation::Mul,
            TensorBinaryOperand::Right,
        ) => (
            "tensor_consuming_mul_f64_right",
            0x4353_2005,
            CONSUMING_BINARY_F64_BINDINGS,
            CONSUMING_MUL_F64_RIGHT,
            PcuValueTypeCaps::FLOAT64,
        ),
        _ => panic!("checked integer tensors cannot use consuming binary profiles"),
    }
}

pub(super) fn consuming_relu_kernel(
    scalar_type: TensorPointwiseScalarType,
    logical_count: u32,
) -> PcuDispatchKernelIr<'static> {
    let (name, id, bindings, ops, type_caps) = match scalar_type {
        TensorPointwiseScalarType::F32 => (
            "tensor_consuming_relu_f32",
            0x4352_0001,
            CONSUMING_RELU_BINDINGS,
            CONSUMING_RELU_OPS,
            PcuValueTypeCaps::FLOAT32,
        ),
        TensorPointwiseScalarType::F64 => (
            "tensor_consuming_relu_f64",
            0x4352_0002,
            CONSUMING_RELU_F64_BINDINGS,
            CONSUMING_RELU_F64_OPS,
            PcuValueTypeCaps::FLOAT64,
        ),
        _ => panic!("checked integer tensors cannot use consuming ReLU profiles"),
    };
    PcuDispatchKernelIr {
        id: fusion_pcu::PcuKernelId(id),
        entry: PcuDispatchEntryPoint {
            name,
            logical_shape: [logical_count, 1, 1],
        },
        bindings,
        ports: &[],
        parameters: &[],
        ops,
        type_caps: type_caps.union(PcuValueTypeCaps::SCALAR_VALUES),
        feature_caps: PcuDispatchFeatureCaps::default(),
    }
}

macro_rules! binary_ops {
    ($name:ident, $alu:expr, $left_index:expr, $right_index:expr) => {
        const $name: &[PcuDispatchOp<'static>] = &[
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result: LEFT,
                binding: LEFT_REF,
                index: $left_index,
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result: RIGHT,
                binding: RIGHT_REF,
                index: $right_index,
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::Alu {
                value_type: PcuValueType::f64(),
                result: RESULT,
                op: $alu,
                lhs: LEFT,
                rhs: RIGHT,
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                binding: OUTPUT_REF,
                index: PcuDispatchIndex::InvocationId,
                value: RESULT,
            }),
            PcuDispatchOp::Control(PcuDispatchControlOp::Return),
        ];
    };
}

macro_rules! add_relu_ops {
    ($name:ident, $left_index:expr, $right_index:expr) => {
        const $name: &[PcuDispatchOp<'static>] = &[
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result: LEFT,
                binding: LEFT_REF,
                index: $left_index,
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result: RIGHT,
                binding: RIGHT_REF,
                index: $right_index,
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::Alu {
                value_type: PcuValueType::f64(),
                result: RESULT,
                op: PcuDispatchAluOp::Add,
                lhs: LEFT,
                rhs: RIGHT,
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::Constant {
                result: ZERO,
                value: PcuParameterValue::F64(0.0_f64.to_bits()),
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::Alu {
                value_type: PcuValueType::f64(),
                result: EPILOGUE_RESULT,
                op: PcuDispatchAluOp::Max,
                lhs: RESULT,
                rhs: ZERO,
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                binding: OUTPUT_REF,
                index: PcuDispatchIndex::InvocationId,
                value: EPILOGUE_RESULT,
            }),
            PcuDispatchOp::Control(PcuDispatchControlOp::Return),
        ];
    };
}

binary_ops!(
    ADD,
    PcuDispatchAluOp::Add,
    PcuDispatchIndex::InvocationId,
    PcuDispatchIndex::InvocationId
);
binary_ops!(
    ADD_LEFT_UNIFORM,
    PcuDispatchAluOp::Add,
    PcuDispatchIndex::BindingElementZero,
    PcuDispatchIndex::InvocationId
);
binary_ops!(
    ADD_RIGHT_UNIFORM,
    PcuDispatchAluOp::Add,
    PcuDispatchIndex::InvocationId,
    PcuDispatchIndex::BindingElementZero
);
binary_ops!(
    ADD_BOTH_UNIFORM,
    PcuDispatchAluOp::Add,
    PcuDispatchIndex::BindingElementZero,
    PcuDispatchIndex::BindingElementZero
);
binary_ops!(
    SUB,
    PcuDispatchAluOp::Sub,
    PcuDispatchIndex::InvocationId,
    PcuDispatchIndex::InvocationId
);
binary_ops!(
    SUB_LEFT_UNIFORM,
    PcuDispatchAluOp::Sub,
    PcuDispatchIndex::BindingElementZero,
    PcuDispatchIndex::InvocationId
);
binary_ops!(
    SUB_RIGHT_UNIFORM,
    PcuDispatchAluOp::Sub,
    PcuDispatchIndex::InvocationId,
    PcuDispatchIndex::BindingElementZero
);
binary_ops!(
    SUB_BOTH_UNIFORM,
    PcuDispatchAluOp::Sub,
    PcuDispatchIndex::BindingElementZero,
    PcuDispatchIndex::BindingElementZero
);
binary_ops!(
    MUL,
    PcuDispatchAluOp::Mul,
    PcuDispatchIndex::InvocationId,
    PcuDispatchIndex::InvocationId
);
binary_ops!(
    MUL_LEFT_UNIFORM,
    PcuDispatchAluOp::Mul,
    PcuDispatchIndex::BindingElementZero,
    PcuDispatchIndex::InvocationId
);
binary_ops!(
    MUL_RIGHT_UNIFORM,
    PcuDispatchAluOp::Mul,
    PcuDispatchIndex::InvocationId,
    PcuDispatchIndex::BindingElementZero
);
binary_ops!(
    MUL_BOTH_UNIFORM,
    PcuDispatchAluOp::Mul,
    PcuDispatchIndex::BindingElementZero,
    PcuDispatchIndex::BindingElementZero
);

add_relu_ops!(
    ADD_RELU,
    PcuDispatchIndex::InvocationId,
    PcuDispatchIndex::InvocationId
);
add_relu_ops!(
    ADD_RELU_LEFT_UNIFORM,
    PcuDispatchIndex::BindingElementZero,
    PcuDispatchIndex::InvocationId
);
add_relu_ops!(
    ADD_RELU_RIGHT_UNIFORM,
    PcuDispatchIndex::InvocationId,
    PcuDispatchIndex::BindingElementZero
);
add_relu_ops!(
    ADD_RELU_BOTH_UNIFORM,
    PcuDispatchIndex::BindingElementZero,
    PcuDispatchIndex::BindingElementZero
);

const RELU_OPS: &[PcuDispatchOp<'static>] = &[
    PcuDispatchOp::Data(PcuDispatchDataOp::Constant {
        result: ZERO,
        value: PcuParameterValue::F64(0.0_f64.to_bits()),
    }),
    PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
        result: LEFT,
        binding: INPUT_REF,
        index: PcuDispatchIndex::InvocationId,
    }),
    PcuDispatchOp::Data(PcuDispatchDataOp::Alu {
        value_type: PcuValueType::f64(),
        result: RESULT,
        op: PcuDispatchAluOp::Max,
        lhs: LEFT,
        rhs: ZERO,
    }),
    PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
        binding: RELU_OUTPUT_REF,
        index: PcuDispatchIndex::InvocationId,
        value: RESULT,
    }),
    PcuDispatchOp::Control(PcuDispatchControlOp::Return),
];

pub(super) fn kernel(
    kind: TensorDispatchKind,
    logical_count: u32,
    scalar_mask: u8,
) -> Result<PcuDispatchKernelIr<'static>, CudaTensorExecutionError> {
    let (name, id, bindings, ops) = match kind {
        TensorDispatchKind::Add => (
            "tensor_add_f64",
            0x4644_0000 + u32::from(scalar_mask),
            BINARY_BINDINGS,
            binary_mask(
                scalar_mask,
                ADD,
                ADD_LEFT_UNIFORM,
                ADD_RIGHT_UNIFORM,
                ADD_BOTH_UNIFORM,
            )?,
        ),
        TensorDispatchKind::Sub => (
            "tensor_sub_f64",
            0x4644_0010 + u32::from(scalar_mask),
            BINARY_BINDINGS,
            binary_mask(
                scalar_mask,
                SUB,
                SUB_LEFT_UNIFORM,
                SUB_RIGHT_UNIFORM,
                SUB_BOTH_UNIFORM,
            )?,
        ),
        TensorDispatchKind::Mul => (
            "tensor_mul_f64",
            0x4644_0020 + u32::from(scalar_mask),
            BINARY_BINDINGS,
            binary_mask(
                scalar_mask,
                MUL,
                MUL_LEFT_UNIFORM,
                MUL_RIGHT_UNIFORM,
                MUL_BOTH_UNIFORM,
            )?,
        ),
        TensorDispatchKind::AddRelu => (
            "tensor_add_relu_f64",
            0x4644_0030 + u32::from(scalar_mask),
            BINARY_BINDINGS,
            add_relu_mask(scalar_mask)?,
        ),
        TensorDispatchKind::Relu if scalar_mask == 0 => {
            ("tensor_relu_f64", 0x4644_0040, RELU_BINDINGS, RELU_OPS)
        }
        TensorDispatchKind::Relu => return Err(CudaTensorExecutionError::InvalidPointwiseProfile),
        TensorDispatchKind::SquaredDifference => {
            return Err(CudaTensorExecutionError::UnsupportedScalarType(
                fusion_pcu::PcuScalarType::F64,
            ));
        }
        TensorDispatchKind::CheckedIntegerAdd
        | TensorDispatchKind::CheckedIntegerSub
        | TensorDispatchKind::CheckedIntegerMul
        | TensorDispatchKind::CheckedFloatAdd
        | TensorDispatchKind::CheckedFloatSub
        | TensorDispatchKind::CheckedFloatMul
        | TensorDispatchKind::CheckedFloatDiv => {
            return Err(CudaTensorExecutionError::InvalidPointwiseProfile);
        }
    };

    Ok(PcuDispatchKernelIr {
        id: fusion_pcu::PcuKernelId(id),
        entry: PcuDispatchEntryPoint {
            name,
            logical_shape: [logical_count, 1, 1],
        },
        bindings,
        ports: &[],
        parameters: &[],
        ops,
        type_caps: PcuValueTypeCaps::FLOAT64.union(PcuValueTypeCaps::SCALAR_VALUES),
        feature_caps: PcuDispatchFeatureCaps::default(),
    })
}

const fn binary_mask(
    mask: u8,
    ordinary: &'static [PcuDispatchOp<'static>],
    left_uniform: &'static [PcuDispatchOp<'static>],
    right_uniform: &'static [PcuDispatchOp<'static>],
    both_uniform: &'static [PcuDispatchOp<'static>],
) -> Result<&'static [PcuDispatchOp<'static>], CudaTensorExecutionError> {
    match mask {
        0 => Ok(ordinary),
        1 => Ok(left_uniform),
        2 => Ok(right_uniform),
        3 => Ok(both_uniform),
        _ => Err(CudaTensorExecutionError::InvalidPointwiseProfile),
    }
}

const fn add_relu_mask(
    mask: u8,
) -> Result<&'static [PcuDispatchOp<'static>], CudaTensorExecutionError> {
    match mask {
        0 => Ok(ADD_RELU),
        1 => Ok(ADD_RELU_LEFT_UNIFORM),
        2 => Ok(ADD_RELU_RIGHT_UNIFORM),
        3 => Ok(ADD_RELU_BOTH_UNIFORM),
        _ => Err(CudaTensorExecutionError::InvalidPointwiseProfile),
    }
}

#[cfg(test)]
mod tests {
    #[rustfmt::skip]
    use fusion_pcu::{
        PcuBf16Bits,
        PcuDeviceTensor,
        PcuDeviceBuffer,
        PcuF16Bits,
        PcuBindingAccess,
        PcuBindingRef,
        PcuBindingType,
        PcuDeviceDescriptor,
        PcuDispatchSubmission,
        PcuObjectKind,
        PcuObjectRef,
        PcuProviderDescriptor,
        PcuProviderId,
        PcuProviderReadiness,
        PcuProviderStatus,
        PcuOwnedCompletion,
        PcuRuntimeDiscovery,
        PcuMemoryPoolId,
        PcuScalarType,
        PcuScalar,
        PcuTargetDescriptor,
        PcuValueType,
        PcuValueTypeCaps,
    };
    use core::num::NonZeroU32;

    #[rustfmt::skip]
    use crate::{
        DeviceBuffer,
        CudaDiscovery,
    };
    #[rustfmt::skip]
    use super::super::{
        CudaTensorAssessor,
        CudaOwnedDispatchBackend,
        CudaOwnedPreparedTensorGraph,
        TensorDispatchKind,
    };
    use fusion_pcu::dialect::tensor::{
        Graph, TensorArithmeticCapability, TensorArithmeticRewritePolicy, TensorBinaryOperand,
        TensorBinaryOperation, TensorPointwiseGroupingPolicy, ValueId,
    };
    #[rustfmt::skip]
    use super::{
        PcuDispatchAluOp,
        PcuDispatchDataOp,
        PcuDispatchOp,
        TensorPointwiseScalarType,
    };

    const INVALID_REFERENCE: PcuObjectRef = PcuObjectRef {
        provider: PcuProviderId(0),
        generation: 0,
        kind: PcuObjectKind::Target,
        id: 0,
    };

    const fn empty_provider() -> PcuProviderDescriptor<'static> {
        PcuProviderDescriptor {
            id: PcuProviderId(0),
            generation: 0,
            backend: "",
            readiness: PcuProviderReadiness {
                status: PcuProviderStatus::Unavailable,
                reason: None,
            },
        }
    }

    const fn empty_target() -> PcuTargetDescriptor<'static> {
        PcuTargetDescriptor {
            reference: INVALID_REFERENCE,
            name: "",
            readiness: PcuProviderReadiness {
                status: PcuProviderStatus::Unavailable,
                reason: None,
            },
        }
    }

    const fn empty_device() -> PcuDeviceDescriptor<'static> {
        PcuDeviceDescriptor {
            reference: INVALID_REFERENCE,
            target: INVALID_REFERENCE,
            name: "",
            class: fusion_pcu::PcuDeviceClass::Other,
            vendor: None,
            architecture: None,
            generation: None,
            location: None,
        }
    }

    fn device_session() -> (CudaDiscovery, CudaOwnedDispatchBackend) {
        let discovery = CudaDiscovery::new();
        let mut providers = [empty_provider()];
        assert_eq!(discovery.providers(&mut providers).unwrap(), 1);
        let mut targets = [empty_target()];
        assert_eq!(
            discovery
                .targets(providers[0].id, providers[0].generation, &mut targets)
                .unwrap(),
            1
        );
        let count = discovery.devices(targets[0].reference, &mut []).unwrap();
        let mut devices = vec![empty_device(); count];
        discovery
            .devices(targets[0].reference, &mut devices)
            .unwrap();
        let device = devices
            .first()
            .expect("ignored hardware test requires a visible CUDA device")
            .reference;
        let session = CudaOwnedDispatchBackend::open(&discovery, device, 64)
            .expect("open selected CUDA device");
        (discovery, session)
    }

    fn prepared_f64_subtraction(
        assessor: &CudaTensorAssessor<'_>,
    ) -> (ValueId, ValueId, CudaOwnedPreparedTensorGraph) {
        let mut graph = Graph::default();
        let left = graph
            .input([4], PcuScalarType::F64)
            .expect("left f64 input");
        let right = graph
            .input([4], PcuScalarType::F64)
            .expect("right f64 input");
        let output = graph.sub(left, right).expect("subtraction");
        let program = graph
            .into_selected_program(
                &[output],
                TensorArithmeticRewritePolicy::Disabled,
                TensorArithmeticCapability::Strict,
                TensorPointwiseGroupingPolicy::Disabled,
            )
            .expect("selected binary graph");
        let prepared = assessor
            .prepare_owned_program(program)
            .expect("backend preparation");
        (left, right, prepared)
    }

    fn f64_bytes(values: &[f64]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(core::mem::size_of_val(values));
        for value in values {
            bytes.extend_from_slice(&value.to_ne_bytes());
        }
        bytes
    }

    fn read_f64_output(output: &DeviceBuffer, count: usize) -> Vec<f64> {
        let mut bytes = vec![0; count * 8];
        output.copy_to(&mut bytes).unwrap();
        bytes
            .as_chunks::<8>()
            .0
            .iter()
            .map(|chunk| f64::from_ne_bytes(*chunk))
            .collect()
    }

    fn dispatch_binary(
        session: &CudaOwnedDispatchBackend,
        kind: TensorDispatchKind,
        scalar_mask: u8,
        left: &[f64],
        right: &[f64],
    ) -> Vec<f64> {
        let logical_count = 3_u32;
        let left_elements = if scalar_mask & 0b01 == 0 { 3 } else { 1 };
        let right_elements = if scalar_mask & 0b10 == 0 { 3 } else { 1 };
        assert_eq!(left.len(), left_elements);
        assert_eq!(right.len(), right_elements);
        let op = match kind {
            TensorDispatchKind::Add => fusion_pcu::PcuDispatchFloatBinaryOp::Add,
            TensorDispatchKind::Sub => fusion_pcu::PcuDispatchFloatBinaryOp::Sub,
            TensorDispatchKind::Mul => fusion_pcu::PcuDispatchFloatBinaryOp::Mul,
            _ => panic!("test requires checked binary operation"),
        };
        let kernel = super::super::checked_float::kernel(
            op,
            PcuScalarType::F64,
            fusion_pcu::PcuFloatUnderflowPolicy::default(),
            logical_count,
        )
        .unwrap();
        let left = (0..3)
            .map(|index| left[if scalar_mask & 1 == 0 { index } else { 0 }])
            .collect::<Vec<_>>();
        let right = (0..3)
            .map(|index| right[if scalar_mask & 2 == 0 { index } else { 0 }])
            .collect::<Vec<_>>();
        let mut left_buffer = session.allocate(3 * 8).unwrap();
        left_buffer.copy_from(&f64_bytes(&left)).unwrap();
        let mut right_buffer = session.allocate(3 * 8).unwrap();
        right_buffer.copy_from(&f64_bytes(&right)).unwrap();
        let output_buffer = session.allocate(3 * 8).unwrap();
        let bindings = [
            session
                .binding(
                    PcuBindingRef::new(0, 0),
                    PcuBindingAccess::ReadOnly,
                    PcuBindingType::Value(PcuValueType::f64()),
                    left_buffer,
                )
                .unwrap(),
            session
                .binding(
                    PcuBindingRef::new(0, 1),
                    PcuBindingAccess::ReadOnly,
                    PcuBindingType::Value(PcuValueType::f64()),
                    right_buffer,
                )
                .unwrap(),
            session
                .binding(
                    PcuBindingRef::new(0, 2),
                    PcuBindingAccess::WriteOnly,
                    PcuBindingType::Value(PcuValueType::f64()),
                    output_buffer,
                )
                .unwrap(),
        ];
        let invocations = NonZeroU32::new(logical_count).unwrap();
        let prepared = session
            .prepare_dispatch(PcuDispatchSubmission {
                kernel: &kernel,
                shape: fusion_pcu::PcuInvocationShape::invocations(invocations),
            })
            .unwrap();
        let mut completion = prepared.submit(&bindings).unwrap();
        completion.wait().unwrap();
        let output = bindings.into_iter().nth(2).unwrap().into_resource();
        read_f64_output(&output, 3)
    }

    fn dispatch_relu(session: &CudaOwnedDispatchBackend, values: &[f64]) -> Vec<f64> {
        assert_eq!(values.len(), 3);
        let kernel = super::super::checked_float::relu_kernel(
            PcuScalarType::F64,
            fusion_pcu::PcuFloatUnderflowPolicy::default(),
            3,
        )
        .unwrap();
        let mut input = session.allocate(values.len() * 8).unwrap();
        input.copy_from(&f64_bytes(values)).unwrap();
        let output = session.allocate(values.len() * 8).unwrap();
        let bindings = [
            session
                .binding(
                    PcuBindingRef::new(0, 0),
                    PcuBindingAccess::ReadOnly,
                    PcuBindingType::Value(PcuValueType::f64()),
                    input,
                )
                .unwrap(),
            session
                .binding(
                    PcuBindingRef::new(0, 1),
                    PcuBindingAccess::WriteOnly,
                    PcuBindingType::Value(PcuValueType::f64()),
                    output,
                )
                .unwrap(),
        ];
        let prepared = session
            .prepare_dispatch(PcuDispatchSubmission {
                kernel: &kernel,
                shape: fusion_pcu::PcuInvocationShape::invocations(NonZeroU32::new(3).unwrap()),
            })
            .unwrap();
        let mut completion = prepared.submit(&bindings).unwrap();
        completion.wait().unwrap();
        let output = bindings.into_iter().nth(1).unwrap().into_resource();
        read_f64_output(&output, 3)
    }

    #[allow(clippy::too_many_lines)] // Keep the explicit F64 Min IR and real dispatch lifecycle together.
    fn assert_unchecked_min_rejected(session: &CudaOwnedDispatchBackend, values: &[f64]) {
        assert_eq!(values.len(), 3);
        let bindings_ir = [
            fusion_pcu::PcuBinding::value(
                Some("input"),
                0,
                0,
                fusion_pcu::PcuBindingStorageClass::Storage,
                PcuBindingAccess::ReadOnly,
                PcuValueType::f64(),
            ),
            fusion_pcu::PcuBinding::value(
                Some("output"),
                0,
                1,
                fusion_pcu::PcuBindingStorageClass::Storage,
                PcuBindingAccess::WriteOnly,
                PcuValueType::f64(),
            ),
        ];
        let ops = [
            fusion_pcu::PcuDispatchOp::Data(fusion_pcu::PcuDispatchDataOp::Constant {
                result: fusion_pcu::PcuDispatchValueId(4),
                value: fusion_pcu::PcuParameterValue::F64(0.0_f64.to_bits()),
            }),
            fusion_pcu::PcuDispatchOp::Data(fusion_pcu::PcuDispatchDataOp::BindingLoad {
                result: fusion_pcu::PcuDispatchValueId(1),
                binding: PcuBindingRef::new(0, 0),
                index: fusion_pcu::PcuDispatchIndex::InvocationId,
            }),
            fusion_pcu::PcuDispatchOp::Data(fusion_pcu::PcuDispatchDataOp::Alu {
                value_type: PcuValueType::f64(),
                result: fusion_pcu::PcuDispatchValueId(3),
                op: fusion_pcu::PcuDispatchAluOp::Min,
                lhs: fusion_pcu::PcuDispatchValueId(1),
                rhs: fusion_pcu::PcuDispatchValueId(4),
            }),
            fusion_pcu::PcuDispatchOp::Data(fusion_pcu::PcuDispatchDataOp::BindingStore {
                binding: PcuBindingRef::new(0, 1),
                index: fusion_pcu::PcuDispatchIndex::InvocationId,
                value: fusion_pcu::PcuDispatchValueId(3),
            }),
            fusion_pcu::PcuDispatchOp::Control(fusion_pcu::PcuDispatchControlOp::Return),
        ];
        let kernel = fusion_pcu::PcuDispatchKernelIr {
            id: fusion_pcu::PcuKernelId(0x4644_0050),
            entry: fusion_pcu::PcuDispatchEntryPoint {
                name: "tensor_min_f64_hardware_test",
                logical_shape: [3, 1, 1],
            },
            bindings: &bindings_ir,
            ports: &[],
            parameters: &[],
            ops: &ops,
            type_caps: PcuValueTypeCaps::FLOAT64.union(PcuValueTypeCaps::SCALAR_VALUES),
            feature_caps: fusion_pcu::PcuDispatchFeatureCaps::default(),
        };
        assert!(
            session
                .prepare_dispatch(PcuDispatchSubmission {
                    kernel: &kernel,
                    shape: fusion_pcu::PcuInvocationShape::invocations(NonZeroU32::new(3).unwrap()),
                })
                .is_err()
        );
    }

    fn assert_bits(actual: &[f64], expected: &[f64]) {
        assert_eq!(
            actual
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            expected
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn consuming_binary_factories_use_rw_donor_and_preserve_operand_order() {
        for (scalar_type, value_type) in [
            (TensorPointwiseScalarType::F32, PcuValueType::f32()),
            (TensorPointwiseScalarType::F64, PcuValueType::f64()),
        ] {
            for operation in [
                TensorBinaryOperation::Add,
                TensorBinaryOperation::Sub,
                TensorBinaryOperation::Mul,
            ] {
                for donor_operand in [TensorBinaryOperand::Left, TensorBinaryOperand::Right] {
                    let kernel =
                        super::consuming_binary_kernel(scalar_type, operation, donor_operand, 7);
                    assert_eq!(kernel.entry.logical_shape, [7, 1, 1]);
                    assert_eq!(kernel.bindings.len(), 2);
                    assert_eq!(kernel.bindings[0].access, PcuBindingAccess::ReadWrite);
                    assert_eq!(kernel.bindings[0].value_type(), Some(value_type));
                    assert_eq!(kernel.bindings[1].access, PcuBindingAccess::ReadOnly);
                    assert_eq!(kernel.bindings[1].value_type(), Some(value_type));

                    let donor_is_left = donor_operand == TensorBinaryOperand::Left;
                    let expected_left = if donor_is_left {
                        super::DONOR_REF
                    } else {
                        super::OTHER_REF
                    };
                    let expected_right = if donor_is_left {
                        super::OTHER_REF
                    } else {
                        super::DONOR_REF
                    };
                    assert!(matches!(
                        kernel.ops[0],
                        PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                            result: super::LEFT,
                            binding,
                            index: fusion_pcu::PcuDispatchIndex::InvocationId,
                        }) if binding == expected_left
                    ));
                    assert!(matches!(
                        kernel.ops[1],
                        PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                            result: super::RIGHT,
                            binding,
                            index: fusion_pcu::PcuDispatchIndex::InvocationId,
                        }) if binding == expected_right
                    ));
                    assert!(matches!(
                        kernel.ops[2],
                        PcuDispatchOp::Data(PcuDispatchDataOp::Alu {
                            value_type: actual_type,
                            result: super::RESULT,
                            op,
                            lhs: super::LEFT,
                            rhs: super::RIGHT,
                        }) if actual_type == value_type && op == match operation {
                            TensorBinaryOperation::Add => PcuDispatchAluOp::Add,
                            TensorBinaryOperation::Sub => PcuDispatchAluOp::Sub,
                            TensorBinaryOperation::Mul => PcuDispatchAluOp::Mul,
                        }
                    ));
                    assert!(matches!(
                        kernel.ops[3],
                        PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                            binding: super::DONOR_REF,
                            index: fusion_pcu::PcuDispatchIndex::InvocationId,
                            value: super::RESULT,
                        })
                    ));
                }
            }
        }
    }

    #[test]
    #[ignore = "requires a working CUDA device and F64 pointwise support"]
    fn checked_binary_keeps_fresh_output_for_exclusive_rhs() {
        let (_discovery, session) = device_session();
        let pool = PcuMemoryPoolId(0x4352_0002);
        let assessor = CudaTensorAssessor::new(&session).expect("tensor assessor");
        let (left, right, prepared) = prepared_f64_subtraction(&assessor);
        let mut memory = session.memory_provider(pool);

        let left_tensor = PcuDeviceTensor::new(
            [4],
            session
                .upload_buffer(pool, &[3.0_f64, 4.0, 2.0, -3.0])
                .expect("left upload"),
        )
        .expect("left tensor");
        let right_tensor = PcuDeviceTensor::new(
            [4],
            session
                .upload_buffer(pool, &[10.0_f64, 20.0, -5.0, 1.0])
                .expect("right upload"),
        )
        .expect("right tensor");
        let donor_identity = right_tensor
            .buffer()
            .resource()
            .allocation_identity_for_test();
        let left_input = assessor
            .borrow_device_input_ref(&left_tensor, pool)
            .expect("left descriptor");
        let result = assessor
            .execute_owned_program_consuming_binary_input(
                &prepared,
                right,
                right_tensor,
                &[(left, &left_input)],
                pool,
                &mut memory,
            )
            .expect("consume right operand");
        assert_ne!(
            result.buffer().resource().allocation_identity_for_test(),
            donor_identity,
            "checked binary output must remain distinct from its input allocation"
        );
        let mut observed = [0.0_f64; 4];
        session
            .download_buffer(pool, result.buffer(), &mut observed)
            .expect("read reused result");
        assert_bits(&observed, &[-7.0, -16.0, 7.0, -4.0]);
        let mut unchanged_left = [0.0_f64; 4];
        session
            .download_buffer(pool, left_tensor.buffer(), &mut unchanged_left)
            .expect("read borrowed input");
        assert_bits(&unchanged_left, &[3.0, 4.0, 2.0, -3.0]);
    }

    #[test]
    #[ignore = "requires a working CUDA device and F64 pointwise support"]
    fn consuming_binary_falls_back_for_shared_backing() {
        let (_discovery, session) = device_session();
        let pool = PcuMemoryPoolId(0x4352_0003);
        let assessor = CudaTensorAssessor::new(&session).expect("tensor assessor");
        let (left, right, prepared) = prepared_f64_subtraction(&assessor);
        let mut memory = session.memory_provider(pool);
        let shared_donor = PcuDeviceTensor::new(
            [4],
            session
                .upload_buffer(pool, &[10.0_f64, 20.0, -5.0, 1.0])
                .expect("shared donor upload"),
        )
        .expect("shared donor tensor");
        let shared_identity = shared_donor
            .buffer()
            .resource()
            .allocation_identity_for_test();
        let alias = PcuDeviceTensor::new(
            [4],
            PcuDeviceBuffer::new(shared_donor.buffer().resource().clone_for_tensor_input(), 4),
        )
        .expect("retained alias tensor");
        let alias_input = assessor
            .borrow_device_input_ref(&alias, pool)
            .expect("alias descriptor");
        let fresh = assessor
            .execute_owned_program_consuming_binary_input(
                &prepared,
                right,
                shared_donor,
                &[(left, &alias_input)],
                pool,
                &mut memory,
            )
            .expect("shared donor should use fresh output");
        assert_ne!(
            fresh.buffer().resource().allocation_identity_for_test(),
            shared_identity,
            "shared backing must not be donated"
        );
        let mut observed = [0.0_f64; 4];
        session
            .download_buffer(pool, fresh.buffer(), &mut observed)
            .expect("read fresh result");
        assert_bits(&observed, &[0.0; 4]);
        session
            .download_buffer(pool, alias.buffer(), &mut observed)
            .expect("read unchanged alias");
        assert_bits(&observed, &[10.0, 20.0, -5.0, 1.0]);
    }

    #[test]
    #[ignore = "requires a working CUDA device and F64 pointwise support"]
    fn checked_binary_pair_keeps_fresh_output_with_exclusive_second_input() {
        let (_discovery, session) = device_session();
        let pool = PcuMemoryPoolId(0x4352_0004);
        let assessor = CudaTensorAssessor::new(&session).expect("tensor assessor");
        let (left, right, prepared) = prepared_f64_subtraction(&assessor);
        let mut memory = session.memory_provider(pool);
        let first = PcuDeviceTensor::new(
            [4],
            session
                .upload_buffer(pool, &[10.0_f64, 20.0, 30.0, 40.0])
                .expect("first upload"),
        )
        .expect("first tensor");
        let alias = PcuDeviceBuffer::new(first.buffer().resource().clone_for_tensor_input(), 4);
        let second = PcuDeviceTensor::new(
            [4],
            session
                .upload_buffer(pool, &[1.0_f64, 2.0, 3.0, 4.0])
                .expect("second upload"),
        )
        .expect("second tensor");
        let second_identity = second.buffer().resource().allocation_identity_for_test();
        let result = assessor
            .execute_owned_program_consuming_binary_pair(
                &prepared,
                [(left, first), (right, second)],
                pool,
                &mut memory,
            )
            .expect("second exclusive input should be selected as donor");
        assert_ne!(
            result.buffer().resource().allocation_identity_for_test(),
            second_identity,
            "checked binary output must remain distinct from both input allocations"
        );
        let mut observed = [0.0_f64; 4];
        session
            .download_buffer(pool, result.buffer(), &mut observed)
            .expect("read reused result");
        assert_bits(&observed, &[9.0, 18.0, 27.0, 36.0]);
        let mut unchanged_alias = [0.0_f64; 4];
        session
            .download_buffer(pool, &alias, &mut unchanged_alias)
            .expect("read retained first backing");
        assert_bits(&unchanged_alias, &[10.0, 20.0, 30.0, 40.0]);
    }

    #[test]
    #[ignore = "requires a working CUDA device and F64 pointwise support"]
    fn consuming_binary_pair_falls_back_when_both_inputs_are_shared() {
        let (_discovery, session) = device_session();
        let pool = PcuMemoryPoolId(0x4352_0005);
        let assessor = CudaTensorAssessor::new(&session).expect("tensor assessor");
        let (left, right, prepared) = prepared_f64_subtraction(&assessor);
        let mut memory = session.memory_provider(pool);
        let first = PcuDeviceTensor::new(
            [4],
            session
                .upload_buffer(pool, &[10.0_f64, 20.0, 30.0, 40.0])
                .expect("first upload"),
        )
        .expect("first tensor");
        let first_alias =
            PcuDeviceBuffer::new(first.buffer().resource().clone_for_tensor_input(), 4);
        let second = PcuDeviceTensor::new(
            [4],
            session
                .upload_buffer(pool, &[1.0_f64, 2.0, 3.0, 4.0])
                .expect("second upload"),
        )
        .expect("second tensor");
        let second_identity = second.buffer().resource().allocation_identity_for_test();
        let second_alias =
            PcuDeviceBuffer::new(second.buffer().resource().clone_for_tensor_input(), 4);
        let result = assessor
            .execute_owned_program_consuming_binary_pair(
                &prepared,
                [(left, first), (right, second)],
                pool,
                &mut memory,
            )
            .expect("shared inputs should use fresh output");
        let result_identity = result.buffer().resource().allocation_identity_for_test();
        assert_ne!(result_identity, second_identity);
        assert_ne!(
            result_identity,
            first_alias.resource().allocation_identity_for_test()
        );
        let mut observed = [0.0_f64; 4];
        session
            .download_buffer(pool, result.buffer(), &mut observed)
            .expect("read fresh result");
        assert_bits(&observed, &[9.0, 18.0, 27.0, 36.0]);
        session
            .download_buffer(pool, &first_alias, &mut observed)
            .expect("read first alias");
        assert_bits(&observed, &[10.0, 20.0, 30.0, 40.0]);
        session
            .download_buffer(pool, &second_alias, &mut observed)
            .expect("read second alias");
        assert_bits(&observed, &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    #[ignore = "requires CUDA hardware and an F64-capable device"]
    fn f64_fixed_pointwise_dispatch_preserves_precision_and_uniform_operands() {
        let (_discovery, session) = device_session();
        let exact = 16_777_217.0_f64;

        for mask in 0..=3 {
            let left = if mask & 1 == 0 {
                vec![exact, -3.5, 4.25]
            } else {
                vec![exact]
            };
            let right = if mask & 2 == 0 {
                vec![1.0, 2.25, -8.0]
            } else {
                vec![1.0]
            };
            let actual = dispatch_binary(&session, TensorDispatchKind::Add, mask, &left, &right);
            let expected = (0..3)
                .map(|index| {
                    left[if mask & 1 == 0 { index } else { 0 }]
                        + right[if mask & 2 == 0 { index } else { 0 }]
                })
                .collect::<Vec<_>>();
            assert_bits(&actual, &expected);
        }

        let actual = dispatch_binary(
            &session,
            TensorDispatchKind::Sub,
            0b10,
            &[exact, -3.5, 4.25],
            &[16_777_216.0],
        );
        assert_bits(&actual, &[1.0, -16_777_219.5, -16_777_211.75]);

        let actual = dispatch_binary(
            &session,
            TensorDispatchKind::Mul,
            0,
            &[exact, 0.5, -1.25],
            &[1.0, 2.0, -4.0],
        );
        assert_bits(&actual, &[exact, 1.0, 5.0]);

        let actual = dispatch_binary(&session, TensorDispatchKind::Mul, 0b11, &[1.5], &[2.0]);
        assert_bits(&actual, &[3.0, 3.0, 3.0]);

        let actual = dispatch_relu(&session, &[-exact, exact, -1.0]);
        assert_bits(&actual, &[0.0, exact, 0.0]);

        let unchecked = super::kernel(TensorDispatchKind::AddRelu, 3, 0).unwrap();
        assert!(
            session
                .prepare_dispatch(PcuDispatchSubmission {
                    kernel: &unchecked,
                    shape: fusion_pcu::PcuInvocationShape::invocations(NonZeroU32::new(3).unwrap()),
                })
                .is_err()
        );
        assert_unchecked_min_rejected(&session, &[-16_777_217.0, -1.0e-10, 3.0]);
    }

    #[test]
    #[ignore = "requires a working CUDA device"]
    fn checked_relu_keeps_fresh_output_for_exclusive_and_shared_input_storage() {
        let (_discovery, session) = device_session();
        let pool = PcuMemoryPoolId(0x4352_0001);
        let assessor = CudaTensorAssessor::new(&session).expect("tensor assessor");
        let mut graph = Graph::default();
        let input = graph.input([4], PcuScalarType::F32).expect("f32 input");
        let output = graph.relu(input).expect("terminal relu");
        let program = graph
            .into_selected_program(
                &[output],
                TensorArithmeticRewritePolicy::Disabled,
                TensorArithmeticCapability::Strict,
                TensorPointwiseGroupingPolicy::Disabled,
            )
            .expect("selected relu graph");
        let prepared = assessor
            .prepare_owned_program(program)
            .expect("backend preparation");
        let mut memory = session.memory_provider(pool);

        let source_values = [-2.0_f32, 3.0, -4.0, 5.0];
        let source = PcuDeviceTensor::new(
            [4],
            session
                .upload_buffer(pool, &source_values)
                .expect("exclusive input upload"),
        )
        .expect("exclusive input tensor");
        let source_identity = source.buffer().resource().allocation_identity_for_test();
        let result = assessor
            .execute_owned_program_consuming_input(&prepared, source, pool, &mut memory)
            .expect("consume exclusive input");
        assert_ne!(
            result.buffer().resource().allocation_identity_for_test(),
            source_identity,
            "checked ReLU keeps fresh output so a fatal fault cannot publish partial mutation"
        );
        let mut observed = [0.0_f32; 4];
        session
            .download_buffer(pool, result.buffer(), &mut observed)
            .expect("read consumed result");
        assert_eq!(
            observed.map(f32::to_bits),
            [0.0_f32, 3.0, 0.0, 5.0].map(f32::to_bits)
        );

        let source_values = [9.0_f32, -10.0, 11.0, -12.0];
        let shared_source = PcuDeviceTensor::new(
            [4],
            session
                .upload_buffer(pool, &source_values)
                .expect("shared input upload"),
        )
        .expect("shared input tensor");
        let shared_identity = shared_source
            .buffer()
            .resource()
            .allocation_identity_for_test();
        let retained_alias = PcuDeviceBuffer::new(
            shared_source.buffer().resource().clone_for_tensor_input(),
            4,
        );
        let fresh_result = assessor
            .execute_owned_program_consuming_input(&prepared, shared_source, pool, &mut memory)
            .expect("shared input must use fresh output storage");
        assert_ne!(
            fresh_result
                .buffer()
                .resource()
                .allocation_identity_for_test(),
            shared_identity,
            "a shared backing must never be donated"
        );
        let mut observed = [0.0_f32; 4];
        session
            .download_buffer(pool, fresh_result.buffer(), &mut observed)
            .expect("read fresh result");
        assert_eq!(
            observed.map(f32::to_bits),
            [9.0_f32, 0.0, 11.0, 0.0].map(f32::to_bits)
        );
        let mut retained_values = [0.0_f32; 4];
        session
            .download_buffer(pool, &retained_alias, &mut retained_values)
            .expect("shared source stays intact");
        assert_eq!(
            retained_values.map(f32::to_bits),
            source_values.map(f32::to_bits)
        );
    }

    #[test]
    #[ignore = "requires a working CUDA device"]
    fn consuming_identity_transfers_exact_storage_with_shared_aliases() {
        let (_discovery, session) = device_session();
        let pool = PcuMemoryPoolId(0x4352_0002);
        let assessor = CudaTensorAssessor::new(&session).expect("tensor assessor");
        let mut graph = Graph::default();
        let input = graph.input([3], PcuScalarType::F64).expect("f64 input");
        let program = graph
            .into_selected_program(
                &[input],
                TensorArithmeticRewritePolicy::Disabled,
                TensorArithmeticCapability::Strict,
                TensorPointwiseGroupingPolicy::Disabled,
            )
            .expect("identity program");
        let prepared = assessor
            .prepare_owned_program(program)
            .expect("backend preparation");
        let values = [16_777_217.0_f64, 1.0e-10, -0.0];
        let source = PcuDeviceTensor::new(
            [3],
            session.upload_buffer(pool, &values).expect("f64 upload"),
        )
        .expect("typed input");
        let source_identity = source.buffer().resource().allocation_identity_for_test();
        let mut memory = session.memory_provider(pool);
        let result = assessor
            .execute_owned_program_consuming_input(&prepared, source, pool, &mut memory)
            .expect("identity transfers the moved owner directly");
        assert_eq!(
            result.buffer().resource().allocation_identity_for_test(),
            source_identity
        );
        let mut observed = [0.0_f64; 3];
        session
            .download_buffer(pool, result.buffer(), &mut observed)
            .expect("f64 identity readback");
        assert_eq!(observed.map(f64::to_bits), values.map(f64::to_bits));

        let shared = PcuDeviceTensor::new(
            [3],
            session
                .upload_buffer(pool, &values)
                .expect("shared f64 upload"),
        )
        .expect("shared typed input");
        let shared_identity = shared.buffer().resource().allocation_identity_for_test();
        let retained_alias =
            PcuDeviceBuffer::new(shared.buffer().resource().clone_for_tensor_input(), 3);
        let shared_result = assessor
            .execute_owned_program_consuming_input(&prepared, shared, pool, &mut memory)
            .expect("shared identity transfer is read-only");
        assert_eq!(
            shared_result
                .buffer()
                .resource()
                .allocation_identity_for_test(),
            shared_identity
        );
        let mut retained_values = [0.0_f64; 3];
        session
            .download_buffer(pool, &retained_alias, &mut retained_values)
            .expect("retained identity alias readback");
        assert_eq!(retained_values.map(f64::to_bits), values.map(f64::to_bits));
    }

    #[test]
    #[ignore = "requires a working CUDA device"]
    fn identity_transport_preserves_all_typed_scalar_bits_and_storage_laws() {
        let (_discovery, session) = device_session();
        let pool = PcuMemoryPoolId(0x4352_0004);
        let assessor = CudaTensorAssessor::new(&session).expect("tensor assessor");

        assert_typed_identity_transport(
            &session,
            &assessor,
            pool,
            &[f32::from_bits(0x7fc1_2345), -0.0_f32, f32::MIN, f32::MAX],
        );
        assert_typed_identity_transport(
            &session,
            &assessor,
            pool,
            &[
                f64::from_bits(0x7ff8_1234_5678_9abc),
                -0.0_f64,
                f64::MIN,
                f64::MAX,
            ],
        );
        assert_typed_identity_transport(&session, &assessor, pool, &[u8::MIN, 0, 1, u8::MAX]);
        assert_typed_identity_transport(
            &session,
            &assessor,
            pool,
            &[u16::MIN, 1, 0x8000, u16::MAX],
        );
        assert_typed_identity_transport(
            &session,
            &assessor,
            pool,
            &[u32::MIN, 1, 0x8000_0000, u32::MAX],
        );
        assert_typed_identity_transport(
            &session,
            &assessor,
            pool,
            &[u64::MIN, 1, 0x8000_0000_0000_0000, u64::MAX],
        );
        assert_typed_identity_transport(&session, &assessor, pool, &[i8::MIN, -1, 0, i8::MAX]);
        assert_typed_identity_transport(&session, &assessor, pool, &[i16::MIN, -1, 0, i16::MAX]);
        assert_typed_identity_transport(&session, &assessor, pool, &[i32::MIN, -1, 0, i32::MAX]);
        assert_typed_identity_transport(&session, &assessor, pool, &[i64::MIN, -1, 0, i64::MAX]);
        assert_typed_identity_transport(
            &session,
            &assessor,
            pool,
            &[
                PcuF16Bits::from_bits(0x7e55),
                PcuF16Bits::from_bits(0x8000),
                PcuF16Bits::from_bits(0x7bff),
                PcuF16Bits::from_bits(0x0001),
            ],
        );
        assert_typed_identity_transport(
            &session,
            &assessor,
            pool,
            &[
                PcuBf16Bits::from_bits(0x7fc1),
                PcuBf16Bits::from_bits(0x8000),
                PcuBf16Bits::from_bits(0x7f7f),
                PcuBf16Bits::from_bits(0x0001),
            ],
        );
    }

    fn assert_typed_identity_transport<T: PcuScalar>(
        session: &CudaOwnedDispatchBackend,
        assessor: &CudaTensorAssessor<'_>,
        pool: PcuMemoryPoolId,
        values: &[T],
    ) {
        let mut graph = Graph::default();
        let input = graph
            .input([values.len()], T::TYPE)
            .expect("typed identity input");
        let program = graph
            .into_selected_program(
                &[input],
                TensorArithmeticRewritePolicy::Disabled,
                TensorArithmeticCapability::Strict,
                TensorPointwiseGroupingPolicy::Disabled,
            )
            .expect("identity transport program");
        let prepared = assessor
            .prepare_owned_program(program)
            .expect("backend identity preparation");
        let mut memory = session.memory_provider(pool);
        assert_borrowed_identity_is_fresh(
            session,
            assessor,
            pool,
            &prepared,
            input,
            values,
            &mut memory,
        );
        assert_consuming_identity_moves_storage(
            session,
            assessor,
            pool,
            &prepared,
            values,
            &mut memory,
        );
        assert_shared_identity_preserves_alias(
            session,
            assessor,
            pool,
            &prepared,
            values,
            &mut memory,
        );
    }

    fn assert_borrowed_identity_is_fresh<T: PcuScalar>(
        session: &CudaOwnedDispatchBackend,
        assessor: &CudaTensorAssessor<'_>,
        pool: PcuMemoryPoolId,
        prepared: &super::super::CudaOwnedPreparedTensorGraph,
        input: ValueId,
        values: &[T],
        memory: &mut impl fusion_pcu::PcuMemoryProvider<Resource = super::super::CudaMemoryResource>,
    ) {
        let borrowed_source = upload_typed_source(session, pool, values);
        let borrowed_identity = borrowed_source
            .buffer()
            .resource()
            .allocation_identity_for_test();
        let borrowed_outputs = assessor
            .execute_owned_program_outputs(prepared, &[(input, &borrowed_source)], pool, memory)
            .expect("borrowed identity copies to fresh output storage");
        let (_, borrowed_result) = borrowed_outputs
            .into_iter()
            .next()
            .expect("single borrowed identity output");
        assert_ne!(
            borrowed_result
                .buffer()
                .resource()
                .allocation_identity_for_test(),
            borrowed_identity,
            "borrowed identity must retain the fresh-output law"
        );
        assert_device_values(session, pool, &borrowed_result, values);
        assert_device_values(session, pool, &borrowed_source, values);
    }

    fn assert_consuming_identity_moves_storage<T: PcuScalar>(
        session: &CudaOwnedDispatchBackend,
        assessor: &CudaTensorAssessor<'_>,
        pool: PcuMemoryPoolId,
        prepared: &super::super::CudaOwnedPreparedTensorGraph,
        values: &[T],
        memory: &mut impl fusion_pcu::PcuMemoryProvider<Resource = super::super::CudaMemoryResource>,
    ) {
        let moved_source = upload_typed_source(session, pool, values);
        let moved_identity = moved_source
            .buffer()
            .resource()
            .allocation_identity_for_test();
        let moved_result = assessor
            .execute_owned_program_consuming_input(prepared, moved_source, pool, memory)
            .expect("identity handoff moves the exact owner");
        assert_eq!(
            moved_result
                .buffer()
                .resource()
                .allocation_identity_for_test(),
            moved_identity,
            "consuming identity must preserve allocation identity"
        );
        assert_device_values(session, pool, &moved_result, values);
    }

    fn assert_shared_identity_preserves_alias<T: PcuScalar>(
        session: &CudaOwnedDispatchBackend,
        assessor: &CudaTensorAssessor<'_>,
        pool: PcuMemoryPoolId,
        prepared: &super::super::CudaOwnedPreparedTensorGraph,
        values: &[T],
        memory: &mut impl fusion_pcu::PcuMemoryProvider<Resource = super::super::CudaMemoryResource>,
    ) {
        let shared_source = upload_typed_source(session, pool, values);
        let shared_identity = shared_source
            .buffer()
            .resource()
            .allocation_identity_for_test();
        let retained_alias = PcuDeviceBuffer::new(
            shared_source.buffer().resource().clone_for_tensor_input(),
            values.len(),
        );
        let shared_result = assessor
            .execute_owned_program_consuming_input(prepared, shared_source, pool, memory)
            .expect("identity handoff remains lawful with a read alias");
        assert_eq!(
            shared_result
                .buffer()
                .resource()
                .allocation_identity_for_test(),
            shared_identity,
            "read-only identity transfer preserves the shared allocation"
        );
        assert_eq!(
            retained_alias.resource().allocation_identity_for_test(),
            shared_identity,
            "retained alias continues to refer to the transferred allocation"
        );
        assert_device_values(session, pool, &shared_result, values);
        let mut alias_observed = vec![T::decode_le(values[0].encode_le()); values.len()];
        session
            .download_buffer(pool, &retained_alias, &mut alias_observed)
            .expect("read retained shared alias");
        assert_scalar_bits(values, &alias_observed);
    }

    fn upload_typed_source<T: PcuScalar>(
        session: &CudaOwnedDispatchBackend,
        pool: PcuMemoryPoolId,
        values: &[T],
    ) -> PcuDeviceTensor<T, super::super::CudaMemoryResource> {
        PcuDeviceTensor::new(
            [values.len()],
            session
                .upload_buffer(pool, values)
                .expect("typed identity upload"),
        )
        .expect("typed identity source")
    }

    fn assert_device_values<T: PcuScalar>(
        session: &CudaOwnedDispatchBackend,
        pool: PcuMemoryPoolId,
        tensor: &PcuDeviceTensor<T, super::super::CudaMemoryResource>,
        expected: &[T],
    ) {
        let mut observed = vec![T::decode_le(expected[0].encode_le()); expected.len()];
        session
            .download_buffer(pool, tensor.buffer(), &mut observed)
            .expect("typed identity readback");
        assert_scalar_bits(expected, &observed);
    }

    fn assert_scalar_bits<T: PcuScalar>(expected: &[T], observed: &[T]) {
        assert_eq!(expected.len(), observed.len());
        for (&expected, &observed) in expected.iter().zip(observed) {
            assert_eq!(
                expected.encode_le().as_ref(),
                observed.encode_le().as_ref(),
                "typed identity changed the scalar transport bits"
            );
        }
    }

    #[test]
    #[ignore = "requires a working CUDA device"]
    fn consuming_nonterminal_relu_graph_keeps_fresh_output_storage() {
        let (_discovery, session) = device_session();
        let pool = PcuMemoryPoolId(0x4352_0003);
        let assessor = CudaTensorAssessor::new(&session).expect("tensor assessor");
        let mut graph = Graph::default();
        let input = graph.input([4], PcuScalarType::F32).expect("input");
        let first_relu = graph.relu(input).expect("first relu");
        let output = graph.relu(first_relu).expect("second relu");
        let program = graph
            .into_selected_program(
                &[output],
                TensorArithmeticRewritePolicy::Disabled,
                TensorArithmeticCapability::Strict,
                TensorPointwiseGroupingPolicy::Disabled,
            )
            .expect("nonterminal relu program");
        let prepared = assessor
            .prepare_owned_program(program)
            .expect("backend preparation");
        let values = [-9.0_f32, 3.0, -4.0, 12.0];
        let source = PcuDeviceTensor::new(
            [4],
            session.upload_buffer(pool, &values).expect("input upload"),
        )
        .expect("input tensor");
        let source_identity = source.buffer().resource().allocation_identity_for_test();
        let mut memory = session.memory_provider(pool);
        let result = assessor
            .execute_owned_program_consuming_input(&prepared, source, pool, &mut memory)
            .expect("ineligible destructive profile must use shared scheduler");
        assert_ne!(
            result.buffer().resource().allocation_identity_for_test(),
            source_identity
        );
        let mut observed = [0.0_f32; 4];
        session
            .download_buffer(pool, result.buffer(), &mut observed)
            .expect("nonterminal relu output readback");
        assert_eq!(
            observed.map(f32::to_bits),
            [0.0_f32, 3.0, 0.0, 12.0].map(f32::to_bits)
        );
    }
}
