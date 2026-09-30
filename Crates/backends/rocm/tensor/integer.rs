//! Fixed checked integer tensor Dispatch factories.

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
    PcuDispatchIndex,
    PcuDispatchIntegerBinaryOp,
    PcuDispatchKernelIr,
    PcuDispatchOp,
    PcuDispatchValueId,
    PcuKernelId,
    PcuValueType,
    PcuValueTypeCaps,
};

#[rustfmt::skip]
use super::{
    RocmTensorExecutionError,
    TensorPointwiseScalarType,
};

const LEFT: PcuDispatchValueId = PcuDispatchValueId(1);
const RIGHT: PcuDispatchValueId = PcuDispatchValueId(2);
const RESULT: PcuDispatchValueId = PcuDispatchValueId(3);
const LEFT_REF: PcuBindingRef = PcuBindingRef::new(0, 0);
const RIGHT_REF: PcuBindingRef = PcuBindingRef::new(0, 1);
const OUTPUT_REF: PcuBindingRef = PcuBindingRef::new(0, 2);

macro_rules! integer_profile {
    ($factory:ident, $bindings:ident, $add_ops:ident, $sub_ops:ident, $mul_ops:ident,
        $value_type:expr, $caps:expr, $kernel_id:expr) => {
        const $bindings: &[PcuBinding<'static>] = &[
            PcuBinding::value(
                Some("left"),
                0,
                0,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::ReadOnly,
                $value_type,
            ),
            PcuBinding::value(
                Some("right"),
                0,
                1,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::ReadOnly,
                $value_type,
            ),
            PcuBinding::value(
                Some("output"),
                0,
                2,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::WriteOnly,
                $value_type,
            ),
        ];
        const $add_ops: &[PcuDispatchOp<'static>] =
            binary_ops!($value_type, PcuDispatchIntegerBinaryOp::Add);
        const $sub_ops: &[PcuDispatchOp<'static>] =
            binary_ops!($value_type, PcuDispatchIntegerBinaryOp::Sub);
        const $mul_ops: &[PcuDispatchOp<'static>] =
            binary_ops!($value_type, PcuDispatchIntegerBinaryOp::Mul);

        const fn $factory(
            op: PcuDispatchIntegerBinaryOp,
            logical_count: u32,
        ) -> PcuDispatchKernelIr<'static> {
            let ops = match op {
                PcuDispatchIntegerBinaryOp::Add => $add_ops,
                PcuDispatchIntegerBinaryOp::Sub => $sub_ops,
                PcuDispatchIntegerBinaryOp::Mul => $mul_ops,
            };
            let op_tag = match op {
                PcuDispatchIntegerBinaryOp::Add => 1,
                PcuDispatchIntegerBinaryOp::Sub => 2,
                PcuDispatchIntegerBinaryOp::Mul => 3,
            };
            PcuDispatchKernelIr {
                id: PcuKernelId($kernel_id + op_tag),
                entry: PcuDispatchEntryPoint {
                    name: "tensor_checked_integer_binary",
                    logical_shape: [logical_count, 1, 1],
                },
                bindings: $bindings,
                ports: &[],
                parameters: &[],
                ops,
                type_caps: $caps.union(PcuValueTypeCaps::SCALAR_VALUES),
                feature_caps: PcuDispatchFeatureCaps::empty(),
            }
        }
    };
}

macro_rules! binary_ops {
    ($value_type:expr, $op:expr) => {
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
            PcuDispatchOp::Data(PcuDispatchDataOp::CheckedIntegerBinary {
                value_type: $value_type,
                op: $op,
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

integer_profile!(
    i8_kernel,
    I8_BINDINGS,
    I8_ADD,
    I8_SUB,
    I8_MUL,
    PcuValueType::i8(),
    PcuValueTypeCaps::INT8,
    0x494e_5000
);
integer_profile!(
    u8_kernel,
    U8_BINDINGS,
    U8_ADD,
    U8_SUB,
    U8_MUL,
    PcuValueType::u8(),
    PcuValueTypeCaps::UINT8,
    0x494e_5100
);
integer_profile!(
    i16_kernel,
    I16_BINDINGS,
    I16_ADD,
    I16_SUB,
    I16_MUL,
    PcuValueType::i16(),
    PcuValueTypeCaps::INT16,
    0x494e_5200
);
integer_profile!(
    u16_kernel,
    U16_BINDINGS,
    U16_ADD,
    U16_SUB,
    U16_MUL,
    PcuValueType::u16(),
    PcuValueTypeCaps::UINT16,
    0x494e_5300
);
integer_profile!(
    i32_kernel,
    I32_BINDINGS,
    I32_ADD,
    I32_SUB,
    I32_MUL,
    PcuValueType::i32(),
    PcuValueTypeCaps::INT32,
    0x494e_5400
);
integer_profile!(
    u32_kernel,
    U32_BINDINGS,
    U32_ADD,
    U32_SUB,
    U32_MUL,
    PcuValueType::u32(),
    PcuValueTypeCaps::UINT32,
    0x494e_5500
);
integer_profile!(
    i64_kernel,
    I64_BINDINGS,
    I64_ADD,
    I64_SUB,
    I64_MUL,
    PcuValueType::i64(),
    PcuValueTypeCaps::INT64,
    0x494e_5600
);
integer_profile!(
    u64_kernel,
    U64_BINDINGS,
    U64_ADD,
    U64_SUB,
    U64_MUL,
    PcuValueType::u64(),
    PcuValueTypeCaps::UINT64,
    0x494e_5700
);

pub(super) fn kernel(
    scalar_type: TensorPointwiseScalarType,
    op: PcuDispatchIntegerBinaryOp,
    logical_count: u32,
) -> Result<PcuDispatchKernelIr<'static>, RocmTensorExecutionError> {
    let build = match scalar_type {
        TensorPointwiseScalarType::I8 => i8_kernel,
        TensorPointwiseScalarType::U8 => u8_kernel,
        TensorPointwiseScalarType::I16 => i16_kernel,
        TensorPointwiseScalarType::U16 => u16_kernel,
        TensorPointwiseScalarType::I32 => i32_kernel,
        TensorPointwiseScalarType::U32 => u32_kernel,
        TensorPointwiseScalarType::I64 => i64_kernel,
        TensorPointwiseScalarType::U64 => u64_kernel,
        TensorPointwiseScalarType::F32 | TensorPointwiseScalarType::F64 => {
            return Err(RocmTensorExecutionError::InvalidPointwiseProfile);
        }
    };
    Ok(build(op, logical_count))
}
