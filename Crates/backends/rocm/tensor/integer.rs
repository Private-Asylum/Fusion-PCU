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
                numerical_requirements: PcuDispatchKernelIr::DEFAULT_REQUIREMENTS,
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
                range_policy: fusion_pcu::PcuRangePolicy::Reject,
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

integer_profile!(
    i128_kernel,
    I128_BINDINGS,
    I128_ADD,
    I128_SUB,
    I128_MUL,
    PcuValueType::Scalar(fusion_pcu::PcuScalarType::I128),
    PcuValueTypeCaps::for_scalar(fusion_pcu::PcuScalarType::I128),
    0x494e_5800
);
integer_profile!(
    u128_kernel,
    U128_BINDINGS,
    U128_ADD,
    U128_SUB,
    U128_MUL,
    PcuValueType::Scalar(fusion_pcu::PcuScalarType::U128),
    PcuValueTypeCaps::for_scalar(fusion_pcu::PcuScalarType::U128),
    0x494e_5900
);
integer_profile!(
    i256_kernel,
    I256_BINDINGS,
    I256_ADD,
    I256_SUB,
    I256_MUL,
    PcuValueType::Scalar(fusion_pcu::PcuScalarType::I256),
    PcuValueTypeCaps::for_scalar(fusion_pcu::PcuScalarType::I256),
    0x494e_5a00
);
integer_profile!(
    u256_kernel,
    U256_BINDINGS,
    U256_ADD,
    U256_SUB,
    U256_MUL,
    PcuValueType::Scalar(fusion_pcu::PcuScalarType::U256),
    PcuValueTypeCaps::for_scalar(fusion_pcu::PcuScalarType::U256),
    0x494e_5b00
);
integer_profile!(
    i512_kernel,
    I512_BINDINGS,
    I512_ADD,
    I512_SUB,
    I512_MUL,
    PcuValueType::Scalar(fusion_pcu::PcuScalarType::I512),
    PcuValueTypeCaps::for_scalar(fusion_pcu::PcuScalarType::I512),
    0x494e_5c00
);
integer_profile!(
    u512_kernel,
    U512_BINDINGS,
    U512_ADD,
    U512_SUB,
    U512_MUL,
    PcuValueType::Scalar(fusion_pcu::PcuScalarType::U512),
    PcuValueTypeCaps::for_scalar(fusion_pcu::PcuScalarType::U512),
    0x494e_5d00
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
        TensorPointwiseScalarType::I128 => i128_kernel,
        TensorPointwiseScalarType::U128 => u128_kernel,
        TensorPointwiseScalarType::I256 => i256_kernel,
        TensorPointwiseScalarType::U256 => u256_kernel,
        TensorPointwiseScalarType::I512 => i512_kernel,
        TensorPointwiseScalarType::U512 => u512_kernel,

        TensorPointwiseScalarType::F16
        | TensorPointwiseScalarType::BF16
        | TensorPointwiseScalarType::F8E4M3FN
        | TensorPointwiseScalarType::F8E5M2
        | TensorPointwiseScalarType::F32
        | TensorPointwiseScalarType::F64 => {
            return Err(RocmTensorExecutionError::InvalidPointwiseProfile);
        }
    };
    Ok(build(op, logical_count))
}

/// Lowers one validated dense checked integer `Add`/`Sub`/`Mul` node to its executor kernel.
///
/// The header-free runtime-compiler source uses the same fixed kernel body.
/// The four-pointer ABI is left, right, fresh output, and a u64 fault word initialized
/// to `u64::MAX`. Wait for completion and inspect status before publishing useful output.
/// This is exact rejecting scalar arithmetic; it offers no tensor Clamp or `PortableV1` profile.
///
/// # Errors
/// Returns invalid graph, dtype, operation, shape or unproved reproducibility errors.
pub fn lower_checked_integer_tensor_to_hip_source(
    graph: &fusion_pcu::dialect::tensor::Graph,
    value: fusion_pcu::dialect::tensor::ValueId,
) -> Result<String, RocmTensorExecutionError> {
    use fusion_pcu::dialect::tensor::{OpDescriptor, TensorOperationSupport, TensorUnsupportedReason};
    let node = graph.node(value)?;
    if node.numerical_options.reproducibility != fusion_pcu::PcuReproducibility::Unspecified {
        return Err(RocmTensorExecutionError::Unsupported {
            value,
            reason: TensorUnsupportedReason::NumericalPolicy {
                requirement: fusion_pcu::PcuNumericalRequirement::Reproducibility,
                options: node.numerical_options,
            },
        });
    }
    let op = match node.op {
        OpDescriptor::Add { .. } => PcuDispatchIntegerBinaryOp::Add,
        OpDescriptor::Sub { .. } => PcuDispatchIntegerBinaryOp::Sub,
        OpDescriptor::Mul { .. } => PcuDispatchIntegerBinaryOp::Mul,
        _ => return Err(RocmTensorExecutionError::InvalidPointwiseProfile),
    };
    if !super::is_checked_integer_scalar(node.scalar_type) {
        return Err(RocmTensorExecutionError::UnsupportedScalarType(
            node.scalar_type,
        ));
    }
    if let TensorOperationSupport::Unsupported { reason } =
        super::assess_checked_integer_node(graph, node)
    {
        return Err(RocmTensorExecutionError::Unsupported { value, reason });
    }
    let scalar = TensorPointwiseScalarType::try_from(node.scalar_type)?;
    let count = node
        .shape
        .iter()
        .try_fold(1u32, |count, &dim| {
            u32::try_from(dim)
                .ok()
                .and_then(|dim| count.checked_mul(dim))
        })
        .ok_or(RocmTensorExecutionError::SizeOverflow)?;
    let ir = PcuDispatchKernelIr {
        numerical_requirements: super::fixed_numerical_requirements(node),
        ..kernel(scalar, op, count)?
    };
    crate::lower_dispatch_to_hip_rtc_source(&ir)
        .map_err(|error| RocmTensorExecutionError::Backend(error.into()))
}
