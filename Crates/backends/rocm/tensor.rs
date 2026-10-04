//! Opt-in tensor operation assessment for an explicitly selected `ROCm` session.
//!
//! This adapter transports typed owned inputs through selected identity graphs and synthesizes
//! strictly ordered checked row-major f32/f64 matrix multiplication. Selected checked f32/f64
//! pointwise operations and dense integer Add/Sub/Mul use owned PCU Dispatch. Ordinary boundary
//! matrix multiplication requires explicit backend-defined compound arithmetic permission;
//! unproved checked boundary compounds remain unsupported.

#[path = "tensor/float.rs"]
mod checked_float;
pub use checked_float::lower_checked_float_tensor_to_hip_source;
#[path = "tensor/consuming.rs"]
mod consuming;
mod feedback_runtime;
mod integer;
pub use integer::lower_checked_integer_tensor_to_hip_source;
#[path = "tensor/native_execution/native_execution.rs"]
mod native_execution;
#[path = "tensor/native_loss/native_loss.rs"]
mod native_loss;
#[path = "tensor/native_policy/native_policy.rs"]
mod native_policy;
#[path = "tensor/native_sgd/native_sgd.rs"]
mod native_sgd;
pub use native_sgd::lower_native_sgd_to_hip_source;
pub use native_loss::lower_native_mse_to_hip_source;
#[path = "tensor/relu_backward/relu_backward.rs"]
#[allow(clippy::redundant_pub_crate)] // Keep the generated-source factory internal to this backend.
pub(crate) mod relu_backward;
#[path = "tensor/strict_mse/strict_mse.rs"]
#[allow(clippy::redundant_pub_crate)] // Private admitted source factory.
pub(crate) mod strict_mse;
pub use strict_mse::lower_strict_mse_to_hip_source;
pub use relu_backward::lower_relu_backward_to_hip_source;
use native_sgd::SgdUpdateMode;
#[path = "tensor/literal/literal.rs"]
mod literal;
#[cfg(test)]
#[path = "tensor/low_literals/low_literals.rs"]
mod low_literals;
#[path = "tensor/owned_scratch/owned_scratch.rs"]
mod owned_scratch;
#[path = "tensor/output_array/output_array.rs"]
mod output_array;

#[path = "tensor/pointwise.rs"]
mod pointwise;
#[path = "tensor/strict_matmul/strict_matmul.rs"]
#[allow(clippy::redundant_pub_crate)] // Typed source factory must remain backend-internal.
pub(crate) mod strict_matmul;
#[path = "tensor/strict_sgd/strict_sgd.rs"]
#[allow(clippy::redundant_pub_crate)] // Keep generated-source admission backend-private.
pub(crate) mod strict_sgd;

/// Lowers a validated strict `MatMul` to the exact HIP source used by the tensor executor.
///
/// This diagnostic/native-comparison surface does not admit an unchecked tensor route. The
/// `fusion_kernel` ABI is left, right and output scalar-storage pointers, followed by a u64
/// fault pointer initialized to `u64::MAX`. Launch one invocation per output cell, bounds
/// rounded blocks, wait for completion, then inspect status before reading useful output.
/// Fault indices encode `(cell * K + reduction_index) * 2 + step`, where multiply is step 0
/// and add is step 1; the ordinary Dispatch fault-kind packing occupies the low three bits.
///
/// # Errors
/// Returns graph provenance or unsupported profile/shape errors before source generation.
pub fn lower_strict_matmul_to_hip_source(
    graph: &Graph,
    value: ValueId,
) -> Result<String, RocmTensorExecutionError> {
    let node = graph.node(value)?;
    let spec = strict_matmul::StrictMatMulSpec::from_node(graph, node).ok_or_else(|| {
        RocmTensorExecutionError::Unsupported {
            value,
            reason: TensorUnsupportedReason::Other(
                "strict MatMul profile or shape is unsupported".into(),
            ),
        }
    })?;
    Ok(spec.source())
}

/// Generates the exact admitted strict F32/F64 SGD checker for a native comparison.
///
/// ABI: `(weights, gradient, output, u64 fault)`, with scalar-storage pointers and a fault
/// word initialized to `u64::MAX`. Wait for terminal completion and inspect the fault before
/// publishing output. Fault events encode `(element * 2 + step) << 3 | kind`, with multiply
/// step zero and subtraction step one. The safe executor accepts only its private factory.
///
/// # Errors
/// Rejects unproved provenance, shape, scalar, rate and numerical contracts before lowering.
pub fn lower_strict_sgd_to_hip_source(
    graph: &Graph,
    value: ValueId,
) -> Result<String, RocmTensorExecutionError> {
    let node = graph.node(value)?;
    let spec = strict_sgd::StrictSgdSpec::from_node(graph, node).ok_or_else(|| {
        RocmTensorExecutionError::Unsupported {
            value,
            reason: TensorUnsupportedReason::Other(
                "strict SGD profile or shape is unsupported".into(),
            ),
        }
    })?;
    Ok(spec.source())
}
pub use feedback_runtime::RocmTensorExecution;
#[rustfmt::skip]
pub use feedback_runtime::{
    RocmAdmittedTensorFeedbackResources,
    RocmTensorFeedbackPrepareError,
    RocmTensorFeedbackReleaseError,
    RocmTensorFeedbackResources,
};

#[rustfmt::skip]
use std::{
    cell::RefCell,
    cell::OnceCell,
    collections::{
        HashMap,
        HashSet,
        VecDeque,
    },
    error::Error,
    fmt,
    mem::{
        align_of,
        size_of,
    },
    ops::Deref,
    rc::Rc,
    sync::Arc,
    time::{
        Duration,
        Instant,
    },
};
use smallvec::SmallVec;

#[rustfmt::skip]
use fusion_pcu::{
    PcuBinding,
    PcuBindingAccess,
    PcuBindingRef,
    PcuBindingStorageClass,
    PcuBindingType,
    PcuCompletionOutcome,
    PcuDeviceBuffer,
    PcuDeviceTensor,
    PcuDispatchAluOp,
    PcuDispatchControlOp,
    PcuDispatchDataOp,
    PcuDispatchEntryPoint,
    PcuDispatchFeatureCaps,
    PcuDispatchIndex,
    PcuDispatchKernelIr,
    PcuDispatchOp,
    PcuDispatchValueId,
    PcuFloatUnderflowPolicy,
    PcuInvocationShape,
    PcuMemoryAccess,
    PcuMemoryAllocationRequest,
    PcuMemoryBackingOwnership,
    PcuMemoryHostAccess,
    PcuMemoryMemberRequirement,
    PcuMemoryPoolId,
    PcuMemoryProvider,
    PcuMemoryProviderError,
    PcuMemoryResource,
    PcuMemoryResourceCapability,
    PcuOwnedCompletion,
    PcuOwnedDispatchMemorySession,
    PcuOwnedShape,
    PcuParameterValue,
    PcuValueType,
    PcuValueTypeCaps,
    validate_reusable_memory_bank_members_by,
};
use std::num::NonZeroU32;
#[rustfmt::skip]
use fusion_pcu::dialect::tensor::{
    Graph,
    NodeDescriptor,
    OpDescriptor,
    Tensor,
    TensorError,
    TensorExecutionPlan,
    TensorExecutionRoute,
    TensorOperationAssessor,
    TensorOperationSupport,
    TensorOperandRepresentation,
    TensorPointwiseGroupingPolicy,
    TensorSynchronousF32MatMulBackend,
    TensorArithmeticCapability,
    TensorArithmeticRewritePolicy,
    TensorBoundedPointwiseFusionGroup,
    TensorBoundedMulFusionGroup,
    TensorPointwiseEpilogue,
    TensorPointwiseArithmeticOp,
    TensorPointwiseOperand,
    TensorSelectedLoweringPlan,
    TensorScratchStoragePlan,
    TensorStorageConstraint,
    TensorStorageValidationError,
    TensorInputReuseProof,
    TerminalBinaryDonorProof,
    TensorBinaryOperand,
    TensorBinaryOperation,
    TensorSgdRewriteCandidate,
    TensorUnsupportedReason,
    ValueId,
};

#[rustfmt::skip]
use crate::{
    HipCompletionBatch,
    HipKernel,
    HipKernelArgument,
    HipStreamHandle,
    HipTimingEventHandle,
    Rocblas,
    RocblasError,
    RocblasSgemmHostTiming,
    RocmMemoryResource,
    RocmOwnedDispatchBackend,
    RocmPreparedDispatch,
};

const ADD_DISPATCH_CACHE_CAPACITY: usize = 32;
// Covers the common small training graph without heap-spilling per-step resource/use slots.
// Larger graphs still spill rather than imposing their full working set on every stack frame.
const TENSOR_EXECUTION_INLINE_NODES: usize = 12;

type TensorExecutionResources =
    SmallVec<[Option<RocmMemoryResource>; TENSOR_EXECUTION_INLINE_NODES]>;
type TensorExecutionUseCounts = SmallVec<[usize; TENSOR_EXECUTION_INLINE_NODES]>;
type TensorExecutionOutputs<'session> = SmallVec<[RocmTensorInput<'session>; 4]>;

// Feature-off expansion preserves the original one-argument finish/wait call.
#[cfg(feature = "insights")]
macro_rules! tensor_flush_batch {
    ($batch:expr, $timings:expr, $is_final:expr) => {
        flush_hip_batch($batch, $timings, $is_final)
    };
}
#[cfg(not(feature = "insights"))]
macro_rules! tensor_flush_batch {
    ($batch:expr, $timings:expr, $is_final:expr) => {
        flush_hip_batch($batch)
    };
}

#[cfg(feature = "insights")]
fn flush_hip_batch<T: NodeTimingSink>(
    batch: &mut HipCompletionBatch,
    timings: &mut T,
    is_final: bool,
) -> Result<(), RocmTensorExecutionError> {
    if batch.is_empty() {
        return Ok(());
    }
    let finish_mark = timings.begin_batch_finish();
    let completion = batch.finish();
    timings.finish_batch_finish(finish_mark, is_final);
    let mut completion = completion.map_err(RocmTensorExecutionError::Completion)?;
    let wait_mark = timings.begin_batch_wait();
    let result = completion.wait();
    timings.finish_batch_wait(wait_mark, is_final);
    result.map_err(RocmTensorExecutionError::Completion)
}

#[cfg(not(feature = "insights"))]
fn flush_hip_batch(batch: &mut HipCompletionBatch) -> Result<(), RocmTensorExecutionError> {
    if batch.is_empty() {
        return Ok(());
    }
    batch
        .finish()
        .map_err(RocmTensorExecutionError::Completion)?
        .wait()
        .map_err(RocmTensorExecutionError::Completion)
}

const ADD_BINDINGS: &[PcuBinding<'static>] = &[
    PcuBinding::value(
        Some("left"),
        0,
        0,
        PcuBindingStorageClass::Storage,
        PcuBindingAccess::ReadOnly,
        PcuValueType::f32(),
    ),
    PcuBinding::value(
        Some("right"),
        0,
        1,
        PcuBindingStorageClass::Storage,
        PcuBindingAccess::ReadOnly,
        PcuValueType::f32(),
    ),
    PcuBinding::value(
        Some("output"),
        0,
        2,
        PcuBindingStorageClass::Storage,
        PcuBindingAccess::WriteOnly,
        PcuValueType::f32(),
    ),
];

const ADD_LEFT: PcuDispatchValueId = PcuDispatchValueId(1);
const ADD_RIGHT: PcuDispatchValueId = PcuDispatchValueId(2);
const ADD_SUM: PcuDispatchValueId = PcuDispatchValueId(3);
const ADD_LEFT_REF: PcuBindingRef = PcuBindingRef::new(0, 0);
const ADD_RIGHT_REF: PcuBindingRef = PcuBindingRef::new(0, 1);
const ADD_OUTPUT_REF: PcuBindingRef = PcuBindingRef::new(0, 2);

const ADD_OPS: &[PcuDispatchOp<'static>] = &[
    PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
        result: ADD_LEFT,
        binding: ADD_LEFT_REF,
        index: PcuDispatchIndex::InvocationId,
    }),
    PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
        result: ADD_RIGHT,
        binding: ADD_RIGHT_REF,
        index: PcuDispatchIndex::InvocationId,
    }),
    PcuDispatchOp::Data(PcuDispatchDataOp::Alu {
        value_type: fusion_pcu::PcuValueType::f32(),
        result: ADD_SUM,
        op: fusion_pcu::PcuDispatchAluOp::Add,
        lhs: ADD_LEFT,
        rhs: ADD_RIGHT,
    }),
    PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
        binding: ADD_OUTPUT_REF,
        index: PcuDispatchIndex::InvocationId,
        value: ADD_SUM,
    }),
    PcuDispatchOp::Control(PcuDispatchControlOp::Return),
];

macro_rules! binary_operand_ops {
    ($name:ident, $alu:expr, $left:expr, $right:expr) => {
        const $name: &[PcuDispatchOp<'static>] = &[
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result: ADD_LEFT,
                binding: ADD_LEFT_REF,
                index: $left,
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result: ADD_RIGHT,
                binding: ADD_RIGHT_REF,
                index: $right,
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::Alu {
                value_type: fusion_pcu::PcuValueType::f32(),
                result: ADD_SUM,
                op: $alu,
                lhs: ADD_LEFT,
                rhs: ADD_RIGHT,
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                binding: ADD_OUTPUT_REF,
                index: PcuDispatchIndex::InvocationId,
                value: ADD_SUM,
            }),
            PcuDispatchOp::Control(PcuDispatchControlOp::Return),
        ];
    };
}

binary_operand_ops!(
    ADD_LEFT_UNIFORM_OPS,
    fusion_pcu::PcuDispatchAluOp::Add,
    PcuDispatchIndex::BindingElementZero,
    PcuDispatchIndex::InvocationId
);
binary_operand_ops!(
    ADD_RIGHT_UNIFORM_OPS,
    fusion_pcu::PcuDispatchAluOp::Add,
    PcuDispatchIndex::InvocationId,
    PcuDispatchIndex::BindingElementZero
);
binary_operand_ops!(
    ADD_BOTH_UNIFORM_OPS,
    fusion_pcu::PcuDispatchAluOp::Add,
    PcuDispatchIndex::BindingElementZero,
    PcuDispatchIndex::BindingElementZero
);

macro_rules! add_relu_operand_ops {
    ($name:ident, $left:expr, $right:expr) => {
        const $name: &[PcuDispatchOp<'static>] = &[
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result: ADD_LEFT,
                binding: ADD_LEFT_REF,
                index: $left,
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result: ADD_RIGHT,
                binding: ADD_RIGHT_REF,
                index: $right,
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::Alu {
                value_type: fusion_pcu::PcuValueType::f32(),
                result: ADD_SUM,
                op: fusion_pcu::PcuDispatchAluOp::Add,
                lhs: ADD_LEFT,
                rhs: ADD_RIGHT,
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::Constant {
                result: ADD_RELU_ZERO,
                value: PcuParameterValue::F32(0.0_f32.to_bits()),
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::Alu {
                value_type: fusion_pcu::PcuValueType::f32(),
                result: ADD_RELU_RESULT,
                op: fusion_pcu::PcuDispatchAluOp::Max,
                lhs: ADD_SUM,
                rhs: ADD_RELU_ZERO,
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                binding: ADD_OUTPUT_REF,
                index: PcuDispatchIndex::InvocationId,
                value: ADD_RELU_RESULT,
            }),
            PcuDispatchOp::Control(PcuDispatchControlOp::Return),
        ];
    };
}

const ADD_RELU_ZERO: PcuDispatchValueId = PcuDispatchValueId(4);
const ADD_RELU_RESULT: PcuDispatchValueId = PcuDispatchValueId(5);

add_relu_operand_ops!(
    ADD_RELU_OPS,
    PcuDispatchIndex::InvocationId,
    PcuDispatchIndex::InvocationId
);
add_relu_operand_ops!(
    ADD_RELU_LEFT_UNIFORM_OPS,
    PcuDispatchIndex::BindingElementZero,
    PcuDispatchIndex::InvocationId
);
add_relu_operand_ops!(
    ADD_RELU_RIGHT_UNIFORM_OPS,
    PcuDispatchIndex::InvocationId,
    PcuDispatchIndex::BindingElementZero
);
add_relu_operand_ops!(
    ADD_RELU_BOTH_UNIFORM_OPS,
    PcuDispatchIndex::BindingElementZero,
    PcuDispatchIndex::BindingElementZero
);

const SUB_OPS: &[PcuDispatchOp<'static>] = &[
    PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
        result: ADD_LEFT,
        binding: ADD_LEFT_REF,
        index: PcuDispatchIndex::InvocationId,
    }),
    PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
        result: ADD_RIGHT,
        binding: ADD_RIGHT_REF,
        index: PcuDispatchIndex::InvocationId,
    }),
    PcuDispatchOp::Data(PcuDispatchDataOp::Alu {
        value_type: fusion_pcu::PcuValueType::f32(),
        result: ADD_SUM,
        op: fusion_pcu::PcuDispatchAluOp::Sub,
        lhs: ADD_LEFT,
        rhs: ADD_RIGHT,
    }),
    PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
        binding: ADD_OUTPUT_REF,
        index: PcuDispatchIndex::InvocationId,
        value: ADD_SUM,
    }),
    PcuDispatchOp::Control(PcuDispatchControlOp::Return),
];
binary_operand_ops!(
    SUB_LEFT_UNIFORM_OPS,
    fusion_pcu::PcuDispatchAluOp::Sub,
    PcuDispatchIndex::BindingElementZero,
    PcuDispatchIndex::InvocationId
);
binary_operand_ops!(
    SUB_RIGHT_UNIFORM_OPS,
    fusion_pcu::PcuDispatchAluOp::Sub,
    PcuDispatchIndex::InvocationId,
    PcuDispatchIndex::BindingElementZero
);
binary_operand_ops!(
    SUB_BOTH_UNIFORM_OPS,
    fusion_pcu::PcuDispatchAluOp::Sub,
    PcuDispatchIndex::BindingElementZero,
    PcuDispatchIndex::BindingElementZero
);

const MUL_OPS: &[PcuDispatchOp<'static>] = &[
    PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
        result: ADD_LEFT,
        binding: ADD_LEFT_REF,
        index: PcuDispatchIndex::InvocationId,
    }),
    PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
        result: ADD_RIGHT,
        binding: ADD_RIGHT_REF,
        index: PcuDispatchIndex::InvocationId,
    }),
    PcuDispatchOp::Data(PcuDispatchDataOp::Alu {
        value_type: fusion_pcu::PcuValueType::f32(),
        result: ADD_SUM,
        op: fusion_pcu::PcuDispatchAluOp::Mul,
        lhs: ADD_LEFT,
        rhs: ADD_RIGHT,
    }),
    PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
        binding: ADD_OUTPUT_REF,
        index: PcuDispatchIndex::InvocationId,
        value: ADD_SUM,
    }),
    PcuDispatchOp::Control(PcuDispatchControlOp::Return),
];
binary_operand_ops!(
    MUL_LEFT_UNIFORM_OPS,
    fusion_pcu::PcuDispatchAluOp::Mul,
    PcuDispatchIndex::BindingElementZero,
    PcuDispatchIndex::InvocationId
);
binary_operand_ops!(
    MUL_RIGHT_UNIFORM_OPS,
    fusion_pcu::PcuDispatchAluOp::Mul,
    PcuDispatchIndex::InvocationId,
    PcuDispatchIndex::BindingElementZero
);
binary_operand_ops!(
    MUL_BOTH_UNIFORM_OPS,
    fusion_pcu::PcuDispatchAluOp::Mul,
    PcuDispatchIndex::BindingElementZero,
    PcuDispatchIndex::BindingElementZero
);

const RELU_BINDINGS: &[PcuBinding<'static>] = &[
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

const RELU_INPUT: PcuDispatchValueId = PcuDispatchValueId(2);
const RELU_ZERO: PcuDispatchValueId = PcuDispatchValueId(1);
const RELU_RESULT: PcuDispatchValueId = PcuDispatchValueId(3);
const RELU_INPUT_REF: PcuBindingRef = PcuBindingRef::new(0, 0);
const RELU_OUTPUT_REF: PcuBindingRef = PcuBindingRef::new(0, 1);
const RELU_OPS: &[PcuDispatchOp<'static>] = &[
    PcuDispatchOp::Data(PcuDispatchDataOp::Constant {
        result: RELU_ZERO,
        value: PcuParameterValue::F32(0.0_f32.to_bits()),
    }),
    PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
        result: RELU_INPUT,
        binding: RELU_INPUT_REF,
        index: PcuDispatchIndex::InvocationId,
    }),
    PcuDispatchOp::Data(PcuDispatchDataOp::Alu {
        value_type: fusion_pcu::PcuValueType::f32(),
        result: RELU_RESULT,
        op: fusion_pcu::PcuDispatchAluOp::Max,
        lhs: RELU_INPUT,
        rhs: RELU_ZERO,
    }),
    PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
        binding: RELU_OUTPUT_REF,
        index: PcuDispatchIndex::InvocationId,
        value: RELU_RESULT,
    }),
    PcuDispatchOp::Control(PcuDispatchControlOp::Return),
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TensorDispatchKind {
    Add,
    AddRelu,
    Sub,
    Mul,
    Relu,
    CheckedIntegerAdd,
    CheckedIntegerSub,
    CheckedIntegerMul,
    CheckedFloatAdd,
    CheckedFloatSub,
    CheckedFloatMul,
    CheckedFloatDiv,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum TensorPointwiseScalarType {
    F16,
    BF16,
    F8E4M3FN,
    F8E5M2,
    F32,
    F64,
    I8,
    U8,
    I16,
    U16,
    I32,
    U32,
    I64,
    U64,
    I128,
    U128,
    I256,
    U256,
    I512,
    U512,
}

impl TryFrom<fusion_pcu::PcuScalarType> for TensorPointwiseScalarType {
    type Error = RocmTensorExecutionError;

    fn try_from(scalar_type: fusion_pcu::PcuScalarType) -> Result<Self, Self::Error> {
        match scalar_type {
            fusion_pcu::PcuScalarType::F16 => Ok(Self::F16),
            fusion_pcu::PcuScalarType::BF16 => Ok(Self::BF16),
            fusion_pcu::PcuScalarType::F8E4M3FN => Ok(Self::F8E4M3FN),
            fusion_pcu::PcuScalarType::F8E5M2 => Ok(Self::F8E5M2),
            fusion_pcu::PcuScalarType::F32 => Ok(Self::F32),
            fusion_pcu::PcuScalarType::F64 => Ok(Self::F64),
            fusion_pcu::PcuScalarType::I8 => Ok(Self::I8),
            fusion_pcu::PcuScalarType::U8 => Ok(Self::U8),
            fusion_pcu::PcuScalarType::I16 => Ok(Self::I16),
            fusion_pcu::PcuScalarType::U16 => Ok(Self::U16),
            fusion_pcu::PcuScalarType::I32 => Ok(Self::I32),
            fusion_pcu::PcuScalarType::U32 => Ok(Self::U32),
            fusion_pcu::PcuScalarType::I64 => Ok(Self::I64),
            fusion_pcu::PcuScalarType::U64 => Ok(Self::U64),
            fusion_pcu::PcuScalarType::I128 => Ok(Self::I128),
            fusion_pcu::PcuScalarType::U128 => Ok(Self::U128),
            fusion_pcu::PcuScalarType::I256 => Ok(Self::I256),
            fusion_pcu::PcuScalarType::U256 => Ok(Self::U256),
            fusion_pcu::PcuScalarType::I512 => Ok(Self::I512),
            fusion_pcu::PcuScalarType::U512 => Ok(Self::U512),
            unsupported => Err(RocmTensorExecutionError::UnsupportedScalarType(unsupported)),
        }
    }
}

impl TensorPointwiseScalarType {
    const fn scalar_type(self) -> fusion_pcu::PcuScalarType {
        match self {
            Self::F16 => fusion_pcu::PcuScalarType::F16,
            Self::BF16 => fusion_pcu::PcuScalarType::BF16,
            Self::F8E4M3FN => fusion_pcu::PcuScalarType::F8E4M3FN,
            Self::F8E5M2 => fusion_pcu::PcuScalarType::F8E5M2,
            Self::F32 => fusion_pcu::PcuScalarType::F32,
            Self::F64 => fusion_pcu::PcuScalarType::F64,
            Self::I8 => fusion_pcu::PcuScalarType::I8,
            Self::U8 => fusion_pcu::PcuScalarType::U8,
            Self::I16 => fusion_pcu::PcuScalarType::I16,
            Self::U16 => fusion_pcu::PcuScalarType::U16,
            Self::I32 => fusion_pcu::PcuScalarType::I32,
            Self::U32 => fusion_pcu::PcuScalarType::U32,
            Self::I64 => fusion_pcu::PcuScalarType::I64,
            Self::U64 => fusion_pcu::PcuScalarType::U64,
            Self::I128 => fusion_pcu::PcuScalarType::I128,
            Self::U128 => fusion_pcu::PcuScalarType::U128,
            Self::I256 => fusion_pcu::PcuScalarType::I256,
            Self::U256 => fusion_pcu::PcuScalarType::U256,
            Self::I512 => fusion_pcu::PcuScalarType::I512,
            Self::U512 => fusion_pcu::PcuScalarType::U512,
        }
    }

    const fn value_type(self) -> PcuValueType {
        match self {
            Self::F16 => PcuValueType::Scalar(fusion_pcu::PcuScalarType::F16),
            Self::BF16 => PcuValueType::Scalar(fusion_pcu::PcuScalarType::BF16),
            Self::F8E4M3FN => PcuValueType::Scalar(fusion_pcu::PcuScalarType::F8E4M3FN),
            Self::F8E5M2 => PcuValueType::Scalar(fusion_pcu::PcuScalarType::F8E5M2),
            Self::F32 => PcuValueType::f32(),
            Self::F64 => PcuValueType::f64(),
            Self::I8 => PcuValueType::i8(),
            Self::U8 => PcuValueType::u8(),
            Self::I16 => PcuValueType::i16(),
            Self::U16 => PcuValueType::u16(),
            Self::I32 => PcuValueType::i32(),
            Self::U32 => PcuValueType::u32(),
            Self::I64 => PcuValueType::i64(),
            Self::U64 => PcuValueType::u64(),
            Self::I128 => PcuValueType::Scalar(fusion_pcu::PcuScalarType::I128),
            Self::U128 => PcuValueType::Scalar(fusion_pcu::PcuScalarType::U128),
            Self::I256 => PcuValueType::Scalar(fusion_pcu::PcuScalarType::I256),
            Self::U256 => PcuValueType::Scalar(fusion_pcu::PcuScalarType::U256),
            Self::I512 => PcuValueType::Scalar(fusion_pcu::PcuScalarType::I512),
            Self::U512 => PcuValueType::Scalar(fusion_pcu::PcuScalarType::U512),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum TensorDispatchCacheKey {
    ReluBackward(relu_backward::Profile),
    StrictMse(strict_mse::Profile),
    StrictMatMul(strict_matmul::StrictMatMulSpec),
    StrictSgd(strict_sgd::StrictSgdSpec),
    Fixed(
        TensorDispatchKind,
        TensorPointwiseScalarType,
        u32,
        u8,
        fusion_pcu::PcuImplementationRequirements,
    ),
    ConsumingRelu(TensorPointwiseScalarType, u32),
    ConsumingBinary(
        TensorBinaryOperation,
        TensorBinaryOperand,
        TensorPointwiseScalarType,
        u32,
    ),
    Pointwise {
        invocation_count: u32,
        scalar_mask: u8,
        topology: SmallVec<[u8; 64]>,
    },
}

type TensorDispatchCache = VecDeque<(TensorDispatchCacheKey, RocmPreparedDispatch)>;

/// Fixed per-node kernel and binding facts selected during graph preparation.
#[derive(Clone, Debug)]
struct PreparedFixedTensorDispatch {
    value: ValueId,
    cache_key: TensorDispatchCacheKey,
    kernel: PcuDispatchKernelIr<'static>,
    invocation_shape: PcuInvocationShape,
    scalar_type: TensorPointwiseScalarType,
    value_type: PcuValueType,
    left_binding: PcuBindingRef,
    right_binding: Option<PcuBindingRef>,
    output_binding: PcuBindingRef,
}

#[derive(Clone, Debug)]
struct PreparedConsumingRelu {
    proof: TensorInputReuseProof,
    kernel: PcuDispatchKernelIr<'static>,
    invocation_shape: PcuInvocationShape,
    scalar_type: TensorPointwiseScalarType,
    value_type: PcuValueType,
    binding: PcuBindingRef,
    logical_count: u32,
}

#[derive(Clone, Debug)]
struct PreparedConsumingBinary {
    proof: TerminalBinaryDonorProof,
    kernel: PcuDispatchKernelIr<'static>,
    invocation_shape: PcuInvocationShape,
    scalar_type: TensorPointwiseScalarType,
    value_type: PcuValueType,
    donor_binding: PcuBindingRef,
    other_binding: PcuBindingRef,
    logical_count: u32,
}

#[derive(Clone, Debug)]
enum PreparedConsumingAction {
    IdentityTransfer(ValueId),
    TerminalRelu(PreparedConsumingRelu),
    TerminalBinary(Box<[PreparedConsumingBinary; 2]>),
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct TensorDispatchCacheAdmission {
    compiled: bool,
    evicted: bool,
}

#[derive(Clone, Debug)]
enum TensorDispatchRequest<'graph> {
    ReluBackward(relu_backward::Profile),
    StrictMse(strict_mse::Profile),
    StrictMatMul(strict_matmul::StrictMatMulSpec),
    StrictSgd(strict_sgd::StrictSgdSpec),
    Fixed {
        kind: TensorDispatchKind,
        scalar_type: TensorPointwiseScalarType,
        logical_count: u32,
        scalar_mask: u8,
        numerical_requirements: fusion_pcu::PcuImplementationRequirements,
    },
    BoundedPointwise {
        group: &'graph TensorBoundedPointwiseFusionGroup,
        logical_count: u32,
        scalar_mask: u8,
        topology: SmallVec<[u8; 64]>,
    },
    BoundedMul {
        group: &'graph TensorBoundedMulFusionGroup,
        logical_count: u32,
        scalar_mask: u8,
        topology: SmallVec<[u8; 64]>,
    },
}

impl TensorDispatchRequest<'_> {
    fn key(&self) -> TensorDispatchCacheKey {
        match self {
            Self::StrictMse(profile) => TensorDispatchCacheKey::StrictMse(*profile),
            Self::ReluBackward(profile) => TensorDispatchCacheKey::ReluBackward(*profile),
            Self::StrictMatMul(spec) => TensorDispatchCacheKey::StrictMatMul(*spec),
            Self::StrictSgd(spec) => TensorDispatchCacheKey::StrictSgd(*spec),
            Self::Fixed {
                kind,
                scalar_type,
                logical_count,
                scalar_mask,
                numerical_requirements,
            } => TensorDispatchCacheKey::Fixed(
                *kind,
                *scalar_type,
                *logical_count,
                *scalar_mask,
                *numerical_requirements,
            ),
            Self::BoundedPointwise {
                logical_count,
                scalar_mask,
                topology,
                ..
            }
            | Self::BoundedMul {
                logical_count,
                scalar_mask,
                topology,
                ..
            } => TensorDispatchCacheKey::Pointwise {
                invocation_count: *logical_count,
                scalar_mask: *scalar_mask,
                topology: topology.clone(),
            },
        }
    }
}

#[derive(Clone, Copy)]
struct ElementwiseOperands<'a> {
    left: &'a RocmMemoryResource,
    right: Option<&'a RocmMemoryResource>,
    output: &'a RocmMemoryResource,
}

impl TensorDispatchKind {
    fn kernel(
        self,
        scalar_type: TensorPointwiseScalarType,
        logical_count: u32,
        scalar_mask: u8,
        float_underflow_policy: Option<PcuFloatUnderflowPolicy>,
    ) -> Result<PcuDispatchKernelIr<'static>, RocmTensorExecutionError> {
        if let Some(op) = self.checked_integer_op() {
            return integer::kernel(scalar_type, op, logical_count);
        }
        if matches!(self, Self::Relu) {
            return checked_float::relu_kernel(
                scalar_type.scalar_type(),
                float_underflow_policy.unwrap_or_default(),
                logical_count,
            );
        }
        if matches!(self, Self::AddRelu) {
            return checked_float::add_relu_kernel(
                scalar_type.scalar_type(),
                float_underflow_policy.unwrap_or_default(),
                logical_count,
                scalar_mask,
            );
        }
        if let Some(op) = self.checked_float_op() {
            return checked_float::kernel(
                op,
                scalar_type.scalar_type(),
                float_underflow_policy.unwrap_or_default(),
                logical_count,
            );
        }
        match scalar_type {
            TensorPointwiseScalarType::F32 => Ok(match self {
                Self::Add => add_kernel(logical_count, scalar_mask),
                Self::AddRelu => add_relu_kernel(logical_count, scalar_mask),
                Self::Sub => sub_kernel(logical_count, scalar_mask),
                Self::Mul => mul_kernel(logical_count, scalar_mask),
                Self::Relu => relu_kernel(logical_count),
                Self::CheckedIntegerAdd
                | Self::CheckedIntegerSub
                | Self::CheckedIntegerMul
                | Self::CheckedFloatAdd
                | Self::CheckedFloatSub
                | Self::CheckedFloatMul
                | Self::CheckedFloatDiv => {
                    return Err(RocmTensorExecutionError::InvalidPointwiseProfile);
                }
            }),
            TensorPointwiseScalarType::F64 => pointwise::kernel(self, logical_count, scalar_mask),
            _ => Err(RocmTensorExecutionError::InvalidPointwiseProfile),
        }
    }

    const fn checked_integer_op(self) -> Option<fusion_pcu::PcuDispatchIntegerBinaryOp> {
        match self {
            Self::CheckedIntegerAdd => Some(fusion_pcu::PcuDispatchIntegerBinaryOp::Add),
            Self::CheckedIntegerSub => Some(fusion_pcu::PcuDispatchIntegerBinaryOp::Sub),
            Self::CheckedIntegerMul => Some(fusion_pcu::PcuDispatchIntegerBinaryOp::Mul),
            _ => None,
        }
    }

    const fn checked_float_op(self) -> Option<fusion_pcu::PcuDispatchFloatBinaryOp> {
        match self {
            Self::CheckedFloatAdd => Some(fusion_pcu::PcuDispatchFloatBinaryOp::Add),
            Self::CheckedFloatSub => Some(fusion_pcu::PcuDispatchFloatBinaryOp::Sub),
            Self::CheckedFloatMul => Some(fusion_pcu::PcuDispatchFloatBinaryOp::Mul),
            Self::CheckedFloatDiv => Some(fusion_pcu::PcuDispatchFloatBinaryOp::Div),
            _ => None,
        }
    }
}

const fn binary_dispatch_kind(
    float_kind: TensorDispatchKind,
    scalar_type: TensorPointwiseScalarType,
) -> TensorDispatchKind {
    match (float_kind, scalar_type) {
        (
            TensorDispatchKind::Add,
            TensorPointwiseScalarType::F16
            | TensorPointwiseScalarType::BF16
            | TensorPointwiseScalarType::F8E4M3FN
            | TensorPointwiseScalarType::F8E5M2
            | TensorPointwiseScalarType::F32
            | TensorPointwiseScalarType::F64,
        ) => TensorDispatchKind::CheckedFloatAdd,
        (
            TensorDispatchKind::Sub,
            TensorPointwiseScalarType::F16
            | TensorPointwiseScalarType::BF16
            | TensorPointwiseScalarType::F8E4M3FN
            | TensorPointwiseScalarType::F8E5M2
            | TensorPointwiseScalarType::F32
            | TensorPointwiseScalarType::F64,
        ) => TensorDispatchKind::CheckedFloatSub,
        (
            TensorDispatchKind::Mul,
            TensorPointwiseScalarType::F16
            | TensorPointwiseScalarType::BF16
            | TensorPointwiseScalarType::F8E4M3FN
            | TensorPointwiseScalarType::F8E5M2
            | TensorPointwiseScalarType::F32
            | TensorPointwiseScalarType::F64,
        ) => TensorDispatchKind::CheckedFloatMul,
        (
            TensorDispatchKind::Add,
            TensorPointwiseScalarType::I8
            | TensorPointwiseScalarType::U8
            | TensorPointwiseScalarType::I16
            | TensorPointwiseScalarType::U16
            | TensorPointwiseScalarType::I32
            | TensorPointwiseScalarType::U32
            | TensorPointwiseScalarType::I64
            | TensorPointwiseScalarType::U64
            | TensorPointwiseScalarType::I128
            | TensorPointwiseScalarType::U128
            | TensorPointwiseScalarType::I256
            | TensorPointwiseScalarType::U256
            | TensorPointwiseScalarType::I512
            | TensorPointwiseScalarType::U512,
        ) => TensorDispatchKind::CheckedIntegerAdd,
        (
            TensorDispatchKind::Sub,
            TensorPointwiseScalarType::I8
            | TensorPointwiseScalarType::U8
            | TensorPointwiseScalarType::I16
            | TensorPointwiseScalarType::U16
            | TensorPointwiseScalarType::I32
            | TensorPointwiseScalarType::U32
            | TensorPointwiseScalarType::I64
            | TensorPointwiseScalarType::U64
            | TensorPointwiseScalarType::I128
            | TensorPointwiseScalarType::U128
            | TensorPointwiseScalarType::I256
            | TensorPointwiseScalarType::U256
            | TensorPointwiseScalarType::I512
            | TensorPointwiseScalarType::U512,
        ) => TensorDispatchKind::CheckedIntegerSub,
        (
            TensorDispatchKind::Mul,
            TensorPointwiseScalarType::I8
            | TensorPointwiseScalarType::U8
            | TensorPointwiseScalarType::I16
            | TensorPointwiseScalarType::U16
            | TensorPointwiseScalarType::I32
            | TensorPointwiseScalarType::U32
            | TensorPointwiseScalarType::I64
            | TensorPointwiseScalarType::U64
            | TensorPointwiseScalarType::I128
            | TensorPointwiseScalarType::U128
            | TensorPointwiseScalarType::I256
            | TensorPointwiseScalarType::U256
            | TensorPointwiseScalarType::I512
            | TensorPointwiseScalarType::U512,
        ) => TensorDispatchKind::CheckedIntegerMul,
        _ => float_kind,
    }
}

fn add_kernel(logical_count: u32, scalar_mask: u8) -> PcuDispatchKernelIr<'static> {
    PcuDispatchKernelIr {
        numerical_requirements: PcuDispatchKernelIr::DEFAULT_REQUIREMENTS,
        id: fusion_pcu::PcuKernelId(0x5445_4e53 + u32::from(scalar_mask)),
        entry: PcuDispatchEntryPoint {
            name: "tensor_add",
            logical_shape: [logical_count, 1, 1],
        },
        bindings: ADD_BINDINGS,
        ports: &[],
        parameters: &[],
        ops: match scalar_mask {
            0 => ADD_OPS,
            1 => ADD_LEFT_UNIFORM_OPS,
            2 => ADD_RIGHT_UNIFORM_OPS,
            3 => ADD_BOTH_UNIFORM_OPS,
            _ => unreachable!("two binary operands have a two-bit scalar mask"),
        },
        type_caps: PcuValueTypeCaps::FLOAT32.union(PcuValueTypeCaps::SCALAR_VALUES),
        feature_caps: PcuDispatchFeatureCaps::default(),
    }
}

fn add_relu_kernel(logical_count: u32, scalar_mask: u8) -> PcuDispatchKernelIr<'static> {
    PcuDispatchKernelIr {
        numerical_requirements: PcuDispatchKernelIr::DEFAULT_REQUIREMENTS,
        id: fusion_pcu::PcuKernelId(0x4144_4452 + u32::from(scalar_mask)),
        entry: PcuDispatchEntryPoint {
            name: "tensor_add_relu",
            logical_shape: [logical_count, 1, 1],
        },
        bindings: ADD_BINDINGS,
        ports: &[],
        parameters: &[],
        ops: match scalar_mask {
            0 => ADD_RELU_OPS,
            1 => ADD_RELU_LEFT_UNIFORM_OPS,
            2 => ADD_RELU_RIGHT_UNIFORM_OPS,
            3 => ADD_RELU_BOTH_UNIFORM_OPS,
            _ => unreachable!("two binary operands have a two-bit scalar mask"),
        },
        type_caps: PcuValueTypeCaps::FLOAT32.union(PcuValueTypeCaps::SCALAR_VALUES),
        feature_caps: PcuDispatchFeatureCaps::default(),
    }
}

fn sub_kernel(logical_count: u32, scalar_mask: u8) -> PcuDispatchKernelIr<'static> {
    PcuDispatchKernelIr {
        numerical_requirements: PcuDispatchKernelIr::DEFAULT_REQUIREMENTS,
        id: fusion_pcu::PcuKernelId(0x5355_4221 + u32::from(scalar_mask)),
        entry: PcuDispatchEntryPoint {
            name: "tensor_sub",
            logical_shape: [logical_count, 1, 1],
        },
        bindings: ADD_BINDINGS,
        ports: &[],
        parameters: &[],
        ops: match scalar_mask {
            0 => SUB_OPS,
            1 => SUB_LEFT_UNIFORM_OPS,
            2 => SUB_RIGHT_UNIFORM_OPS,
            3 => SUB_BOTH_UNIFORM_OPS,
            _ => unreachable!("two binary operands have a two-bit scalar mask"),
        },
        type_caps: PcuValueTypeCaps::FLOAT32.union(PcuValueTypeCaps::SCALAR_VALUES),
        feature_caps: PcuDispatchFeatureCaps::default(),
    }
}

fn mul_kernel(logical_count: u32, scalar_mask: u8) -> PcuDispatchKernelIr<'static> {
    PcuDispatchKernelIr {
        numerical_requirements: PcuDispatchKernelIr::DEFAULT_REQUIREMENTS,
        id: fusion_pcu::PcuKernelId(0x4d55_4c21 + u32::from(scalar_mask)),
        entry: PcuDispatchEntryPoint {
            name: "tensor_mul",
            logical_shape: [logical_count, 1, 1],
        },
        bindings: ADD_BINDINGS,
        ports: &[],
        parameters: &[],
        ops: match scalar_mask {
            0 => MUL_OPS,
            1 => MUL_LEFT_UNIFORM_OPS,
            2 => MUL_RIGHT_UNIFORM_OPS,
            3 => MUL_BOTH_UNIFORM_OPS,
            _ => unreachable!("two binary operands have a two-bit scalar mask"),
        },
        type_caps: PcuValueTypeCaps::FLOAT32.union(PcuValueTypeCaps::SCALAR_VALUES),
        feature_caps: PcuDispatchFeatureCaps::default(),
    }
}

fn relu_kernel(logical_count: u32) -> PcuDispatchKernelIr<'static> {
    PcuDispatchKernelIr {
        numerical_requirements: PcuDispatchKernelIr::DEFAULT_REQUIREMENTS,
        id: fusion_pcu::PcuKernelId(0x5245_4c55),
        entry: PcuDispatchEntryPoint {
            name: "tensor_relu",
            logical_shape: [logical_count, 1, 1],
        },
        bindings: RELU_BINDINGS,
        ports: &[],
        parameters: &[],
        ops: RELU_OPS,
        type_caps: PcuValueTypeCaps::FLOAT32.union(PcuValueTypeCaps::SCALAR_VALUES),
        feature_caps: PcuDispatchFeatureCaps::default(),
    }
}

/// Encodes only the expression shape and operand order, never graph-local `ValueId`s.
fn bounded_pointwise_topology(group: &TensorBoundedPointwiseFusionGroup) -> SmallVec<[u8; 64]> {
    let mut topology = SmallVec::new();
    topology.push(match group.epilogue {
        TensorPointwiseEpilogue::Identity => 0,
        TensorPointwiseEpilogue::Relu => 1,
    });
    topology.push(u8::try_from(group.leaves.len()).unwrap_or(u8::MAX));
    topology.push(u8::try_from(group.steps.len()).unwrap_or(u8::MAX));
    for step in &group.steps {
        topology.push(match step.op {
            TensorPointwiseArithmeticOp::Add => 0,
            TensorPointwiseArithmeticOp::Sub => 1,
        });
        encode_pointwise_operand(&mut topology, step.left);
        encode_pointwise_operand(&mut topology, step.right);
    }
    topology
}

fn bounded_mul_topology(group: &TensorBoundedMulFusionGroup) -> SmallVec<[u8; 64]> {
    let mut topology = SmallVec::new();
    // Distinct tag from Add/Sub groups so cache identity includes the arithmetic opcode.
    topology.extend([
        2,
        0,
        u8::try_from(group.leaves.len()).unwrap_or(u8::MAX),
        u8::try_from(group.steps.len()).unwrap_or(u8::MAX),
    ]);
    for step in &group.steps {
        topology.push(2); // Mul
        encode_pointwise_operand(&mut topology, step.left);
        encode_pointwise_operand(&mut topology, step.right);
    }
    topology
}

fn bounded_mul_program(
    group: &TensorBoundedMulFusionGroup,
    scalar_mask: u8,
) -> Result<(Vec<PcuBinding<'static>>, Vec<PcuDispatchOp<'static>>), RocmTensorExecutionError> {
    if group.leaves.is_empty()
        || group.leaves.len() > 4
        || group.steps.is_empty()
        || group.steps.len() > 8
        || usize::from(scalar_mask) >= (1_usize << group.leaves.len())
    {
        return Err(RocmTensorExecutionError::SizeOverflow);
    }
    let mut bindings = Vec::with_capacity(group.leaves.len() + 1);
    for index in 0..group.leaves.len() {
        bindings.push(PcuBinding::value(
            None,
            0,
            u32::try_from(index).map_err(|_| RocmTensorExecutionError::SizeOverflow)?,
            PcuBindingStorageClass::Storage,
            PcuBindingAccess::ReadOnly,
            PcuValueType::f32(),
        ));
    }
    bindings.push(PcuBinding::value(
        None,
        0,
        u32::try_from(group.leaves.len()).map_err(|_| RocmTensorExecutionError::SizeOverflow)?,
        PcuBindingStorageClass::Storage,
        PcuBindingAccess::WriteOnly,
        PcuValueType::f32(),
    ));
    let mut ops = Vec::with_capacity(group.leaves.len() + group.steps.len() + 2);
    for index in 0..group.leaves.len() {
        ops.push(PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
            result: PcuDispatchValueId(
                u16::try_from(index + 1).map_err(|_| RocmTensorExecutionError::SizeOverflow)?,
            ),
            binding: PcuBindingRef::new(
                0,
                u32::try_from(index).map_err(|_| RocmTensorExecutionError::SizeOverflow)?,
            ),
            index: if scalar_mask & (1 << index) != 0 {
                PcuDispatchIndex::BindingElementZero
            } else {
                PcuDispatchIndex::InvocationId
            },
        }));
    }
    for (index, step) in group.steps.iter().enumerate() {
        let lhs = pointwise_value_id(group.leaves.len(), step.left)?;
        let rhs = pointwise_value_id(group.leaves.len(), step.right)?;
        ops.push(PcuDispatchOp::Data(PcuDispatchDataOp::Alu {
            value_type: fusion_pcu::PcuValueType::f32(),
            result: PcuDispatchValueId(
                u16::try_from(group.leaves.len() + index + 1)
                    .map_err(|_| RocmTensorExecutionError::SizeOverflow)?,
            ),
            op: PcuDispatchAluOp::Mul,
            lhs,
            rhs,
        }));
    }
    let terminal = PcuDispatchValueId(
        u16::try_from(group.leaves.len() + group.steps.len())
            .map_err(|_| RocmTensorExecutionError::SizeOverflow)?,
    );
    ops.push(PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
        binding: PcuBindingRef::new(
            0,
            u32::try_from(group.leaves.len())
                .map_err(|_| RocmTensorExecutionError::SizeOverflow)?,
        ),
        index: PcuDispatchIndex::InvocationId,
        value: terminal,
    }));
    ops.push(PcuDispatchOp::Control(PcuDispatchControlOp::Return));
    Ok((bindings, ops))
}

fn encode_pointwise_operand(topology: &mut SmallVec<[u8; 64]>, operand: TensorPointwiseOperand) {
    match operand {
        TensorPointwiseOperand::Leaf(index) => {
            topology.push(0);
            topology.push(u8::try_from(index).unwrap_or(u8::MAX));
        }
        TensorPointwiseOperand::Step(index) => {
            topology.push(1);
            topology.push(u8::try_from(index).unwrap_or(u8::MAX));
        }
    }
}

fn bounded_pointwise_kernel_id(topology: &[u8], scalar_mask: u8) -> fusion_pcu::PcuKernelId {
    // Stable FNV-1a identity for diagnostics/cache identity. The prepared executable cache also
    // retains the full topology bytes, so hash collisions cannot alias two kernels.
    let hash = topology
        .iter()
        .copied()
        .chain([scalar_mask])
        .fold(0x811c_9dc5_u32, |hash, byte| {
            (hash ^ u32::from(byte)).wrapping_mul(0x0100_0193)
        });
    fusion_pcu::PcuKernelId(0x4250_0000 | (hash & 0x00ff_ffff))
}

/// Builds a Dispatch program whose ordered ALU instructions preserve each source Add/Sub
/// rounding point and apply the group's selected terminal epilogue.
fn bounded_pointwise_program(
    group: &TensorBoundedPointwiseFusionGroup,
    scalar_mask: u8,
) -> Result<(Vec<PcuBinding<'static>>, Vec<PcuDispatchOp<'static>>), RocmTensorExecutionError> {
    if group.leaves.is_empty()
        || group.leaves.len() > 4
        || group.steps.is_empty()
        || group.steps.len() > 8
        || usize::from(scalar_mask) >= (1_usize << group.leaves.len())
    {
        return Err(RocmTensorExecutionError::SizeOverflow);
    }
    let mut bindings = Vec::with_capacity(group.leaves.len() + 1);
    for index in 0..group.leaves.len() {
        bindings.push(PcuBinding::value(
            None,
            0,
            u32::try_from(index).map_err(|_| RocmTensorExecutionError::SizeOverflow)?,
            PcuBindingStorageClass::Storage,
            PcuBindingAccess::ReadOnly,
            PcuValueType::f32(),
        ));
    }
    let output_binding_index =
        u32::try_from(group.leaves.len()).map_err(|_| RocmTensorExecutionError::SizeOverflow)?;
    bindings.push(PcuBinding::value(
        None,
        0,
        output_binding_index,
        PcuBindingStorageClass::Storage,
        PcuBindingAccess::WriteOnly,
        PcuValueType::f32(),
    ));

    let has_relu = group.epilogue == TensorPointwiseEpilogue::Relu;
    let mut ops =
        Vec::with_capacity(group.leaves.len() + group.steps.len() + usize::from(has_relu) * 2 + 2);
    for index in 0..group.leaves.len() {
        let binding_index =
            u32::try_from(index).map_err(|_| RocmTensorExecutionError::SizeOverflow)?;
        ops.push(PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
            result: PcuDispatchValueId(
                u16::try_from(index + 1).map_err(|_| RocmTensorExecutionError::SizeOverflow)?,
            ),
            binding: PcuBindingRef::new(0, binding_index),
            index: if scalar_mask & (1 << index) != 0 {
                PcuDispatchIndex::BindingElementZero
            } else {
                PcuDispatchIndex::InvocationId
            },
        }));
    }
    for (index, step) in group.steps.iter().enumerate() {
        let lhs = pointwise_value_id(group.leaves.len(), step.left)?;
        let rhs = pointwise_value_id(group.leaves.len(), step.right)?;
        let result = PcuDispatchValueId(
            u16::try_from(group.leaves.len() + index + 1)
                .map_err(|_| RocmTensorExecutionError::SizeOverflow)?,
        );
        let op = match step.op {
            TensorPointwiseArithmeticOp::Add => PcuDispatchAluOp::Add,
            TensorPointwiseArithmeticOp::Sub => PcuDispatchAluOp::Sub,
        };
        ops.push(PcuDispatchOp::Data(PcuDispatchDataOp::Alu {
            value_type: fusion_pcu::PcuValueType::f32(),
            result,
            op,
            lhs,
            rhs,
        }));
    }
    let terminal = PcuDispatchValueId(
        u16::try_from(group.leaves.len() + group.steps.len())
            .map_err(|_| RocmTensorExecutionError::SizeOverflow)?,
    );
    let result = if has_relu {
        let relu_zero = PcuDispatchValueId(terminal.0 + 1);
        let relu_result = PcuDispatchValueId(terminal.0 + 2);
        ops.push(PcuDispatchOp::Data(PcuDispatchDataOp::Constant {
            result: relu_zero,
            value: PcuParameterValue::F32(0.0_f32.to_bits()),
        }));
        ops.push(PcuDispatchOp::Data(PcuDispatchDataOp::Alu {
            value_type: fusion_pcu::PcuValueType::f32(),
            result: relu_result,
            op: PcuDispatchAluOp::Max,
            lhs: terminal,
            rhs: relu_zero,
        }));
        relu_result
    } else {
        terminal
    };
    ops.push(PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
        binding: PcuBindingRef::new(0, output_binding_index),
        index: PcuDispatchIndex::InvocationId,
        value: result,
    }));
    ops.push(PcuDispatchOp::Control(PcuDispatchControlOp::Return));
    Ok((bindings, ops))
}

fn pointwise_value_id(
    leaf_count: usize,
    operand: TensorPointwiseOperand,
) -> Result<PcuDispatchValueId, RocmTensorExecutionError> {
    let index = match operand {
        TensorPointwiseOperand::Leaf(index) if index < leaf_count => index,
        TensorPointwiseOperand::Step(index) => leaf_count
            .checked_add(index)
            .ok_or(RocmTensorExecutionError::SizeOverflow)?,
        TensorPointwiseOperand::Leaf(_) => return Err(RocmTensorExecutionError::SizeOverflow),
    };
    Ok(PcuDispatchValueId(
        u16::try_from(index + 1).map_err(|_| RocmTensorExecutionError::SizeOverflow)?,
    ))
}

/// Errors from the bounded `ROCm` tensor `MatMul` operation.
#[derive(Debug)]
pub enum RocmTensorError {
    Rocblas(RocblasError),
    InvalidShape,
    InvalidMemoryAccess,
    DimensionOverflow,
}

impl fmt::Display for RocmTensorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Rocblas(error) => error.fmt(f),
            Self::InvalidShape => {
                f.write_str("ROCm tensor MatMul requires nonempty rank-two matrices")
            }
            Self::InvalidMemoryAccess => {
                f.write_str("ROCm tensor MatMul memory access does not match its role")
            }
            Self::DimensionOverflow => {
                f.write_str("ROCm tensor MatMul byte size or dimension overflows")
            }
        }
    }
}

impl Error for RocmTensorError {}

impl From<RocblasError> for RocmTensorError {
    fn from(error: RocblasError) -> Self {
        Self::Rocblas(error)
    }
}

/// Failure while executing a requested output from a graph on the selected `ROCm` session.
#[derive(Debug)]
pub enum RocmTensorExecutionError {
    Graph(TensorError),
    Unsupported {
        value: ValueId,
        reason: TensorUnsupportedReason,
    },
    UnsupportedScalarType(fusion_pcu::PcuScalarType),
    InvalidPointwiseProfile,
    Memory(PcuMemoryProviderError),
    MemoryAdmission(fusion_pcu::PcuMemoryAllocateWithPolicyError),
    Operation(RocmTensorError),
    Backend(crate::RocmOwnedDispatchError),
    Completion(crate::HipError),
    FailedCompletion,
    ExecutionFault(fusion_pcu::PcuExecutionFault),
    SizeOverflow,
    EmptyOutputs,
    DuplicateOutput(ValueId),
    OutputCountMismatch {
        expected: usize,
        actual: usize,
    },
    InvalidPlan(ValueId),
    MissingResource(ValueId),
    InputResourceMismatch,
    InputPoolMismatch,
    InputAccessMismatch,
    BorrowedInputUpdate,
    BorrowedInputEscape,
    OutputResourceMismatch,
    ScratchMismatch,
    ScratchBusy,
    FeedbackPlanMismatch,
    FeedbackStepUnavailable {
        requested: usize,
        first_available: usize,
        completed: usize,
    },
    StorageConstraint(TensorStorageValidationError),
}

impl fmt::Display for RocmTensorExecutionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl Error for RocmTensorExecutionError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Graph(error) => Some(error),
            Self::Operation(error) => Some(error),
            Self::Backend(error) => Some(error),
            Self::Completion(error) => Some(error),
            _ => None,
        }
    }
}

impl From<TensorError> for RocmTensorExecutionError {
    fn from(error: TensorError) -> Self {
        Self::Graph(error)
    }
}

impl From<PcuMemoryProviderError> for RocmTensorExecutionError {
    fn from(error: PcuMemoryProviderError) -> Self {
        Self::Memory(error)
    }
}

impl From<RocmTensorError> for RocmTensorExecutionError {
    fn from(error: RocmTensorError) -> Self {
        Self::Operation(error)
    }
}

/// Assessor and bounded tensor executor tied to an explicitly opened `ROCm` dispatch session.
///
/// Construction creates the reusable tensor stream. rocBLAS is loaded and bound only when a graph
/// requests a rocBLAS operation, so Dispatch-only graphs do not depend on the BLAS runtime.
pub struct RocmTensorAssessor<'session> {
    session: &'session RocmOwnedDispatchBackend,
    state: RocmTensorAssessorStateSource<'session>,
}

#[allow(clippy::large_enum_variant)] // Direct assessors own inline state; retained warm views borrow without another allocation.
enum RocmTensorAssessorStateSource<'state> {
    Owned(RocmTensorAssessorState),
    Borrowed(&'state RocmTensorAssessorState),
}

struct RocmTensorAssessorState {
    rocblas: OnceCell<Result<Rocblas, RocblasError>>,
    stream: HipStreamHandle,
    add_dispatches: RefCell<TensorDispatchCache>,
    native_sgd: native_sgd::NativeSgdKernels,
    native_mse: [RefCell<Option<HipKernel>>; 2],
}

/// Rc-retained backend and warm tensor-assessor state for hosted per-function execution.
///
/// The backend and assessor state have one shared lifetime root. Cloning the `Rc` preserves both
/// the exact HIP session and its compiled kernel caches across hosted cache eviction. Borrowed
/// assessor views do not clone this root or allocate.
pub struct RocmOwnedTensorAssessor {
    // Drop handles, streams, and compiled dispatch state before releasing the backend runtime.
    state: RocmTensorAssessorState,
    backend: Rc<RocmOwnedDispatchBackend>,
}

impl RocmTensorAssessorState {
    fn new(session: &RocmOwnedDispatchBackend) -> Result<Self, RocblasError> {
        let stream = session.tensor_runtime().create_stream()?;
        Ok(Self {
            rocblas: OnceCell::new(),
            stream,
            add_dispatches: RefCell::new(VecDeque::new()),
            native_sgd: native_sgd::NativeSgdKernels::default(),
            native_mse: core::array::from_fn(|_| RefCell::new(None)),
        })
    }
}

impl RocmOwnedTensorAssessor {
    /// Creates retained Dispatch tensor state for an already selected HIP backend.
    ///
    /// # Errors
    ///
    /// Returns an error if the HIP tensor stream cannot be created. rocBLAS initialization is
    /// deferred until a prepared graph actually requests a BLAS operation.
    pub fn new(backend: Rc<RocmOwnedDispatchBackend>) -> Result<Self, RocblasError> {
        let state = RocmTensorAssessorState::new(&backend)?;
        Ok(Self { state, backend })
    }

    /// Exact selected backend retained by this tensor assessor root.
    #[must_use]
    pub fn backend(&self) -> &RocmOwnedDispatchBackend {
        &self.backend
    }

    /// Borrows this root as a per-call tensor assessor without cloning state or resetting caches.
    #[must_use]
    pub fn assessor(&self) -> RocmTensorAssessor<'_> {
        RocmTensorAssessor {
            session: &self.backend,
            state: RocmTensorAssessorStateSource::Borrowed(&self.state),
        }
    }
}

/// Summary of explicit prepared-graph dispatch prewarming.
///
/// `requested_keys` counts unique selected dispatch keys in the graph. `retained_keys` reports
/// how many of those keys remain in the assessor's bounded FIFO cache when prewarming returns.
/// Existing unrelated entries can consume cache capacity, so prewarming a graph with fewer than
/// 32 keys does not guarantee that every key is retained. `compiled_keys`, `cache_hits`, and
/// `evictions` describe the operations performed by this call.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RocmTensorPrewarmReport {
    /// Number of unique fixed or dynamic PCU dispatch keys selected by the graph.
    pub requested_keys: usize,
    /// Number of selected keys compiled and inserted during this call.
    pub compiled_keys: usize,
    /// Number of selected keys already present in the cache when visited.
    pub cache_hits: usize,
    /// Number of cache entries evicted while admitting selected keys.
    pub evictions: usize,
    /// Number of this graph's unique keys retained after the prewarm pass.
    pub retained_keys: usize,
}

/// Caller-owned device allocation reusable as an immutable graph input.
///
/// The session borrow prevents use after backend destruction; execution also checks backend and
/// pool identity before launching work. Uploaded inputs can be changed through
/// [`RocmTensorAssessor::update_input`]; borrowed device inputs are immutable for their borrow.
pub struct RocmTensorInput<'session> {
    session: &'session RocmOwnedDispatchBackend,
    shape: PcuOwnedShape,
    resource: RocmMemoryResource,
    scalar_type: fusion_pcu::PcuScalarType,
    updateable: bool,
}

/// Borrowed, allocation-free descriptor for an already-owned device resource input.
pub struct RocmTensorInputRef<'input> {
    session: &'input RocmOwnedDispatchBackend,
    shape: &'input [usize],
    resource: &'input RocmMemoryResource,
    scalar_type: fusion_pcu::PcuScalarType,
}

/// Fresh output storage borrows its shape from the prepared graph. Execution is synchronous, so
/// this descriptor lives only through scheduling and completion quiescence.
struct FreshTensorOutput<'graph> {
    shape: &'graph [usize],
    resource: RocmMemoryResource,
    scalar_type: fusion_pcu::PcuScalarType,
}

#[derive(Clone, Copy)]
struct TensorOutputView<'output> {
    shape: &'output [usize],
    resource: &'output RocmMemoryResource,
    scalar_type: fusion_pcu::PcuScalarType,
}

impl<'output> From<&'output RocmTensorInput<'_>> for TensorOutputView<'output> {
    fn from(output: &'output RocmTensorInput<'_>) -> Self {
        Self {
            shape: output.shape.as_slice(),
            resource: &output.resource,
            scalar_type: output.scalar_type,
        }
    }
}

impl<'output> From<&'output FreshTensorOutput<'_>> for TensorOutputView<'output> {
    fn from(output: &'output FreshTensorOutput<'_>) -> Self {
        Self {
            shape: output.shape,
            resource: &output.resource,
            scalar_type: output.scalar_type,
        }
    }
}

trait RocmTensorInputDescriptor {
    fn session(&self) -> &RocmOwnedDispatchBackend;
    fn shape(&self) -> &[usize];
    fn resource(&self) -> &RocmMemoryResource;
    fn scalar_type(&self) -> fusion_pcu::PcuScalarType;
}

impl RocmTensorInputDescriptor for RocmTensorInput<'_> {
    fn session(&self) -> &RocmOwnedDispatchBackend {
        self.session
    }

    fn shape(&self) -> &[usize] {
        self.shape.as_slice()
    }

    fn resource(&self) -> &RocmMemoryResource {
        &self.resource
    }

    fn scalar_type(&self) -> fusion_pcu::PcuScalarType {
        self.scalar_type
    }
}

impl RocmTensorInputDescriptor for RocmTensorInputRef<'_> {
    fn session(&self) -> &RocmOwnedDispatchBackend {
        self.session
    }

    fn shape(&self) -> &[usize] {
        self.shape
    }

    fn resource(&self) -> &RocmMemoryResource {
        self.resource
    }

    fn scalar_type(&self) -> fusion_pcu::PcuScalarType {
        self.scalar_type
    }
}

/// A selected tensor graph output and its moved, typed device allocation.
pub type RocmTensorOwnedOutput<T = f32> = (ValueId, PcuDeviceTensor<T, RocmMemoryResource>);

impl RocmTensorInput<'_> {
    fn validate_session(
        &self,
        session: &RocmOwnedDispatchBackend,
    ) -> Result<(), RocmTensorExecutionError> {
        if !std::ptr::eq(self.session, session) {
            return Err(RocmTensorExecutionError::InputResourceMismatch);
        }
        if !self.resource.belongs_to_runtime(session.tensor_runtime()) {
            return Err(RocmTensorExecutionError::InputResourceMismatch);
        }
        Ok(())
    }

    fn into_device_tensor<T: fusion_pcu::PcuScalar>(
        self,
    ) -> Result<PcuDeviceTensor<T, RocmMemoryResource>, RocmTensorExecutionError> {
        if !self.updateable {
            return Err(RocmTensorExecutionError::BorrowedInputEscape);
        }
        validate_tensor_scalar_tag::<T>(self.scalar_type)?;
        let elements = self
            .shape
            .as_slice()
            .iter()
            .try_fold(1_usize, |count, &dimension| count.checked_mul(dimension))
            .ok_or(RocmTensorExecutionError::SizeOverflow)?;
        let required_bytes = byte_len_for::<T>(self.shape.as_slice())?;
        if u64::try_from(required_bytes).map_err(|_| RocmTensorExecutionError::SizeOverflow)?
            > self.resource.size_bytes()
            || !alignment_satisfies(self.resource.alignment_bytes(), scalar_layout(T::TYPE)?.1)
        {
            return Err(RocmTensorExecutionError::OutputResourceMismatch);
        }
        let Self {
            shape, resource, ..
        } = self;
        PcuDeviceTensor::new(shape, PcuDeviceBuffer::new(resource, elements))
            .map_err(|_| RocmTensorExecutionError::OutputResourceMismatch)
    }
}

impl<'session> RocmTensorAssessor<'session> {
    const fn state(&self) -> &RocmTensorAssessorState {
        match &self.state {
            RocmTensorAssessorStateSource::Owned(state) => state,
            RocmTensorAssessorStateSource::Borrowed(state) => state,
        }
    }

    fn rocblas(&self) -> Result<&Rocblas, RocblasError> {
        self.state()
            .rocblas
            .get_or_init(|| {
                let mut handle = Rocblas::new(self.session.tensor_runtime())?;
                handle.bind_stream(&self.state().stream)?;
                Ok(handle)
            })
            .as_ref()
            .map_err(Clone::clone)
    }

    fn ensure_dispatch_cached(
        &self,
        key: TensorDispatchCacheKey,
        kernel: PcuDispatchKernelIr<'_>,
        shape: PcuInvocationShape,
        dynamic_tensor_kernel: bool,
    ) -> Result<TensorDispatchCacheAdmission, RocmTensorExecutionError> {
        let mut cache = self.state().add_dispatches.borrow_mut();
        if cache.iter().any(|(cached_key, _)| *cached_key == key) {
            return Ok(TensorDispatchCacheAdmission::default());
        }
        let prepared = if dynamic_tensor_kernel {
            self.session.prepare_dynamic_tensor_kernel_on_stream(
                kernel,
                shape,
                &self.state().stream,
            )
        } else {
            self.session.prepare_dispatch_owned_kernel_on_stream(
                kernel,
                shape,
                &self.state().stream,
            )
        }
        .map_err(RocmTensorExecutionError::Backend)?;
        let evicted = cache.len() == ADD_DISPATCH_CACHE_CAPACITY;
        if evicted {
            cache.pop_front();
        }
        cache.push_back((key, prepared));
        Ok(TensorDispatchCacheAdmission {
            compiled: true,
            evicted,
        })
    }

    fn ensure_fixed_dispatch_cached(
        &self,
        kind: TensorDispatchKind,
        scalar_type: TensorPointwiseScalarType,
        logical_count: u32,
        scalar_mask: u8,
        numerical_requirements: fusion_pcu::PcuImplementationRequirements,
    ) -> Result<TensorDispatchCacheAdmission, RocmTensorExecutionError> {
        let invocations =
            NonZeroU32::new(logical_count).ok_or(RocmTensorExecutionError::SizeOverflow)?;
        self.ensure_dispatch_cached(
            TensorDispatchCacheKey::Fixed(
                kind,
                scalar_type,
                logical_count,
                scalar_mask,
                numerical_requirements,
            ),
            PcuDispatchKernelIr {
                numerical_requirements,
                ..kind.kernel(
                    scalar_type,
                    logical_count,
                    scalar_mask,
                    Some(numerical_requirements.float_underflow),
                )?
            },
            PcuInvocationShape::invocations(invocations),
            false,
        )
    }

    fn ensure_bounded_pointwise_dispatch_cached(
        &self,
        group: &TensorBoundedPointwiseFusionGroup,
        logical_count: u32,
        scalar_mask: u8,
        topology: &SmallVec<[u8; 64]>,
    ) -> Result<TensorDispatchCacheAdmission, RocmTensorExecutionError> {
        let invocations =
            NonZeroU32::new(logical_count).ok_or(RocmTensorExecutionError::SizeOverflow)?;
        let key = TensorDispatchCacheKey::Pointwise {
            invocation_count: logical_count,
            scalar_mask,
            topology: topology.clone(),
        };
        // Keep the execution hit path allocation-free: the dynamic program owns temporary
        // binding/op vectors, which must only be built after a cache miss.
        if self
            .state()
            .add_dispatches
            .borrow()
            .iter()
            .any(|(cached_key, _)| *cached_key == key)
        {
            return Ok(TensorDispatchCacheAdmission::default());
        }
        let (kernel_bindings, kernel_ops) = bounded_pointwise_program(group, scalar_mask)?;
        let kernel = PcuDispatchKernelIr {
            numerical_requirements: PcuDispatchKernelIr::DEFAULT_REQUIREMENTS,
            id: bounded_pointwise_kernel_id(topology, scalar_mask),
            entry: PcuDispatchEntryPoint {
                name: match group.epilogue {
                    TensorPointwiseEpilogue::Identity => "tensor_bounded_add_sub",
                    TensorPointwiseEpilogue::Relu => "tensor_bounded_add_sub_relu",
                },
                logical_shape: [logical_count, 1, 1],
            },
            bindings: &kernel_bindings,
            ports: &[],
            parameters: &[],
            ops: &kernel_ops,
            type_caps: PcuValueTypeCaps::FLOAT32.union(PcuValueTypeCaps::SCALAR_VALUES),
            feature_caps: PcuDispatchFeatureCaps::default(),
        };
        self.ensure_dispatch_cached(
            key,
            kernel,
            PcuInvocationShape::invocations(invocations),
            true,
        )
    }

    fn ensure_bounded_mul_dispatch_cached(
        &self,
        group: &TensorBoundedMulFusionGroup,
        logical_count: u32,
        scalar_mask: u8,
        topology: &SmallVec<[u8; 64]>,
    ) -> Result<TensorDispatchCacheAdmission, RocmTensorExecutionError> {
        let invocations =
            NonZeroU32::new(logical_count).ok_or(RocmTensorExecutionError::SizeOverflow)?;
        let key = TensorDispatchCacheKey::Pointwise {
            invocation_count: logical_count,
            scalar_mask,
            topology: topology.clone(),
        };
        if self
            .state()
            .add_dispatches
            .borrow()
            .iter()
            .any(|(cached_key, _)| *cached_key == key)
        {
            return Ok(TensorDispatchCacheAdmission::default());
        }
        let (kernel_bindings, kernel_ops) = bounded_mul_program(group, scalar_mask)?;
        let kernel = PcuDispatchKernelIr {
            numerical_requirements: PcuDispatchKernelIr::DEFAULT_REQUIREMENTS,
            id: bounded_pointwise_kernel_id(topology, scalar_mask),
            entry: PcuDispatchEntryPoint {
                name: "tensor_bounded_mul",
                logical_shape: [logical_count, 1, 1],
            },
            bindings: &kernel_bindings,
            ports: &[],
            parameters: &[],
            ops: &kernel_ops,
            type_caps: PcuValueTypeCaps::FLOAT32.union(PcuValueTypeCaps::SCALAR_VALUES),
            feature_caps: PcuDispatchFeatureCaps::default(),
        };
        self.ensure_dispatch_cached(
            key,
            kernel,
            PcuInvocationShape::invocations(invocations),
            true,
        )
    }

    /// Records a timing event on this assessor's selected tensor stream.
    ///
    /// # Errors
    ///
    /// Returns a HIP event creation or recording error.
    pub fn begin_device_timing(&self) -> Result<HipTimingEventHandle, RocmTensorExecutionError> {
        let event = self
            .session
            .tensor_runtime()
            .create_timing_event()
            .map_err(RocmTensorExecutionError::Completion)?;
        if let Err(error) = self.state().stream.record_timing(&event) {
            // A failed record cannot prove whether HIP retained the event reference.
            std::mem::forget(event);
            return Err(RocmTensorExecutionError::Completion(error));
        }
        Ok(event)
    }

    /// Records a final event and returns elapsed milliseconds on the selected tensor stream.
    ///
    /// A synchronous graph can leave host submission gaps between operations, so this is a
    /// device timeline interval, not a sum of GPU-busy kernel durations.
    ///
    /// # Errors
    ///
    /// Returns an event creation, recording, completion, or elapsed-time query error.
    pub fn finish_device_timing(
        &self,
        start: HipTimingEventHandle,
    ) -> Result<f32, RocmTensorExecutionError> {
        let runtime = self.session.tensor_runtime();
        let end = runtime
            .create_timing_event()
            .map_err(RocmTensorExecutionError::Completion)?;
        if let Err(error) = self.state().stream.record_timing(&end) {
            std::mem::forget(start);
            std::mem::forget(end);
            return Err(RocmTensorExecutionError::Completion(error));
        }
        match runtime.elapsed_time_ms(&start, &end) {
            Ok(milliseconds) => Ok(milliseconds),
            Err(error) => {
                // Completion may be unknown: keep both event owners until a future recovery
                // mechanism can prove the device no longer references them.
                std::mem::forget(start);
                std::mem::forget(end);
                Err(RocmTensorExecutionError::Completion(error))
            }
        }
    }

    /// Creates tensor execution state for the backend's already selected device.
    ///
    /// # Errors
    ///
    /// Returns an error if the HIP tensor stream cannot be created. rocBLAS initialization is
    /// deferred until a prepared graph requests a BLAS operation.
    pub fn new(session: &'session RocmOwnedDispatchBackend) -> Result<Self, RocblasError> {
        Ok(Self {
            session,
            state: RocmTensorAssessorStateSource::Owned(RocmTensorAssessorState::new(session)?),
        })
    }

    /// Compile the PCU dispatches selected by a prepared graph on this assessor's stream.
    ///
    /// This is an explicit, idempotent warm-up operation: it compiles selected fixed, dynamic,
    /// and typed strict `MatMul` Dispatch kernels without allocating, binding, uploading, or requiring provider resources.
    /// A compilation failure may leave earlier keys cached. The per-assessor FIFO cache holds 32
    /// entries; evictions are reported, and `retained_keys` reports how many unique keys from this
    /// graph remain after the pass. Existing unrelated entries may occupy slots. Custom HIP
    /// kernels and rocBLAS calls are outside the PCU dispatch cache and are not prewarmed here.
    ///
    /// # Errors
    ///
    /// Returns the same backend compilation or shape errors that execution would return when it
    /// first encounters a selected Dispatch kernel.
    pub fn prewarm_prepared_graph(
        &self,
        prepared: &RocmPreparedTensorGraph<'_>,
    ) -> Result<RocmTensorPrewarmReport, RocmTensorExecutionError> {
        let requests = collect_dispatch_requests(prepared)?;
        let mut report = RocmTensorPrewarmReport {
            requested_keys: requests.len(),
            ..RocmTensorPrewarmReport::default()
        };

        for request in &requests {
            let admission = match request {
                TensorDispatchRequest::StrictMse(profile) => {
                    self.ensure_strict_mse_cached(*profile)?
                }
                TensorDispatchRequest::ReluBackward(profile) => {
                    self.ensure_relu_backward_cached(*profile)?
                }
                TensorDispatchRequest::StrictMatMul(spec) => {
                    self.ensure_strict_matmul_cached(*spec)?
                }
                TensorDispatchRequest::StrictSgd(spec) => self.ensure_strict_sgd_cached(*spec)?,
                TensorDispatchRequest::Fixed {
                    kind,
                    scalar_type,
                    logical_count,
                    scalar_mask,
                    numerical_requirements,
                } => self.ensure_fixed_dispatch_cached(
                    *kind,
                    *scalar_type,
                    *logical_count,
                    *scalar_mask,
                    *numerical_requirements,
                )?,
                TensorDispatchRequest::BoundedPointwise {
                    group,
                    logical_count,
                    scalar_mask,
                    topology,
                } => self.ensure_bounded_pointwise_dispatch_cached(
                    group,
                    *logical_count,
                    *scalar_mask,
                    topology,
                )?,
                TensorDispatchRequest::BoundedMul {
                    group,
                    logical_count,
                    scalar_mask,
                    topology,
                } => self.ensure_bounded_mul_dispatch_cached(
                    group,
                    *logical_count,
                    *scalar_mask,
                    topology,
                )?,
            };
            report.compiled_keys += usize::from(admission.compiled);
            report.cache_hits += usize::from(!admission.compiled);
            report.evictions += usize::from(admission.evicted);
        }

        let cache = self.state().add_dispatches.borrow();
        let requested_keys = requests
            .iter()
            .map(TensorDispatchRequest::key)
            .collect::<Vec<_>>();
        let cached_keys = cache.iter().map(|(key, _)| key.clone()).collect::<Vec<_>>();
        report.retained_keys = retained_requested_key_count(&requested_keys, &cached_keys);
        Ok(report)
    }

    /// Executes the dependency closure of `output` using this selected `ROCm` tensor route.
    ///
    /// Inputs and constants use the supplied provider. Explicit strict dense `MatMul` nodes use
    /// ordered checked synthesis; Boundary `MatMul` is rejected. Same-shape nonempty floating
    /// `Add` and `ReLU` nodes use owned PCU Dispatch; dense integer
    /// `Add`, `Sub` and `Mul` nodes use checked Dispatch with a synchronous completion gate.
    /// Elementwise executables are cached by operation, type and flattened element count in a
    /// bounded per-assessor cache. The
    /// entire requested closure and its host inputs are validated before the first allocation. No
    /// reference or other-backend fallback is attempted.
    ///
    /// # Errors
    ///
    /// Returns a graph/input error, an explicit unsupported-node error, a memory-provider error,
    /// or a `ROCm` operation error.
    pub fn execute_graph<P>(
        &self,
        graph: &Graph,
        inputs: &[(ValueId, Tensor)],
        output: ValueId,
        pool: PcuMemoryPoolId,
        memory: &mut P,
    ) -> Result<Tensor, RocmTensorExecutionError>
    where
        P: PcuMemoryProvider<Resource = RocmMemoryResource>,
    {
        let prepared = self.prepare_graph(graph, output)?;
        self.execute_prepared(&prepared, inputs, pool, memory)
    }

    /// Prepare the requested output dependency closure for repeated execution.
    ///
    /// The returned value borrows `graph`, which must remain unchanged for its lifetime. It
    /// contains the selected node order and resource use counts; host input validation and device
    /// allocation are deferred until [`Self::execute_prepared`].
    ///
    /// # Errors
    ///
    /// Returns an error if the output is unknown, its dependency closure contains an operation
    /// this adapter cannot execute, or the selected node shapes cannot be represented.
    pub fn prepare_graph<'graph>(
        &self,
        graph: &'graph Graph,
        output: ValueId,
    ) -> Result<RocmPreparedTensorGraph<'graph>, RocmTensorExecutionError> {
        self.prepare_graph_outputs(graph, &[output])
    }

    /// Prepare the union of the dependency closures for requested outputs.
    ///
    /// Outputs retain the caller's order. Empty and duplicate lists are rejected.
    ///
    /// # Errors
    ///
    /// Returns an error for an empty or duplicate output list, unknown outputs, unsupported
    /// operations in any dependency closure, or unrepresentable node shapes.
    pub fn prepare_graph_outputs<'graph>(
        &self,
        graph: &'graph Graph,
        outputs: &[ValueId],
    ) -> Result<RocmPreparedTensorGraph<'graph>, RocmTensorExecutionError> {
        self.prepare_graph_outputs_with_policy(
            graph,
            outputs,
            TensorArithmeticRewritePolicy::Disabled,
        )
    }

    /// Prepare requested outputs with an explicit arithmetic rewrite policy.
    ///
    /// `ROCm` declares its explicit `fmaf` implementation as supporting contracted multiply-add.
    /// Contracted SGD is selected only when `AllowContractedArithmetic` is supplied. The ordinary
    /// preparation methods always preserve the graph's separate Mul/Sub rounding.
    ///
    /// # Errors
    ///
    /// Returns the same validation errors as [`Self::prepare_graph_outputs`].
    pub fn prepare_graph_outputs_with_policy<'graph>(
        &self,
        graph: &'graph Graph,
        outputs: &[ValueId],
        policy: TensorArithmeticRewritePolicy,
    ) -> Result<RocmPreparedTensorGraph<'graph>, RocmTensorExecutionError> {
        self.prepare_graph_outputs_with_policies(
            graph,
            outputs,
            policy,
            TensorPointwiseGroupingPolicy::Disabled,
        )
    }

    /// Prepare an opt-in pointwise grouping alongside the arithmetic rewrite policy.
    ///
    /// The ordinary preparation methods leave pointwise operations separate. A grouped
    /// Add followed by `ReLU` keeps the source graph unchanged and omits the Add allocation.
    ///
    /// # Errors
    ///
    /// Returns the same validation errors as [`Self::prepare_graph_outputs`].
    pub fn prepare_graph_outputs_with_policies<'graph>(
        &self,
        graph: &'graph Graph,
        outputs: &[ValueId],
        policy: TensorArithmeticRewritePolicy,
        grouping: TensorPointwiseGroupingPolicy,
    ) -> Result<RocmPreparedTensorGraph<'graph>, RocmTensorExecutionError> {
        let arithmetic = if policy == TensorArithmeticRewritePolicy::AllowContractedArithmetic {
            TensorArithmeticCapability::ContractedMultiplyAdd
        } else {
            TensorArithmeticCapability::Strict
        };
        let plan = prepare_graph_outputs_plan_with_policies(
            graph, outputs, self, policy, arithmetic, grouping,
        )?;
        for node in &plan.nodes {
            if matches!(node.op, OpDescriptor::MeanSquaredError { .. })
                && node.numerical_mode != Some(fusion_pcu::PcuNumericalMode::Strict)
            {
                self.prepare_native_mse(node.scalar_type)?;
            }
            if matches!(node.op, OpDescriptor::SgdUpdate { .. })
                && node.numerical_mode != Some(fusion_pcu::PcuNumericalMode::Strict)
            {
                self.prepare_native_sgd(
                    node.scalar_type,
                    node.numerical_options.precision
                        == fusion_pcu::PcuPrecisionPolicy::BackendOptimized
                        || plan
                            .lowering_plan
                            .rewrites()
                            .iter()
                            .any(|rewrite| rewrite.output == node.value),
                )?;
            }
        }
        let data = RocmPreparedGraphData {
            scalar_type: homogeneous_scalar_type(&plan.nodes),
            requires_blas: nodes_require_blas(&plan.nodes),
            batch_native_matmuls: native_execution::batch_native_matmuls(&plan.nodes, outputs),
            transport_only_inputs: false,
            consuming_action: None,
            node_values: plan.nodes.iter().map(|node| node.value).collect(),
            output: outputs[0],
            outputs: outputs.to_vec(),
            index_by_value: plan.index_by_value,
            use_counts: plan.use_counts,
            fused_add_by_relu: plan.fused_add_by_relu,
            bounded_pointwise_by_output: plan.bounded_pointwise_by_output,
            bounded_mul_by_output: plan.bounded_mul_by_output,
            fixed_dispatches: plan.fixed_dispatches,
            suppressed_adds: plan.suppressed_adds,
            indexed_storage_constraints: plan.indexed_storage_constraints,
            matmul_operands: plan.matmul_operands,
            strict_sgd: plan.strict_sgd,
            physical_layouts: plan.physical_layouts,
            rewrites: plan.lowering_plan.rewrites().to_vec(),
            input_values: plan.tensor_plan.input_values().to_vec(),
        };
        Ok(RocmPreparedTensorGraph {
            graph,
            plan: plan.tensor_plan,
            lowering_plan: plan.lowering_plan,
            data,
            nodes: plan.nodes,
        })
    }

    /// Prepares a graph-owning selected program without retaining a self-reference into its graph.
    ///
    /// This cold operation derives `ROCm` layouts and validation indexes from the already selected
    /// core program. Later executions use the captured metadata and do not reconstruct a core
    /// execution or lowering plan.
    ///
    /// # Errors
    ///
    /// Returns an unsupported-operation or storage-layout error if the selected program cannot
    /// be executed by this assessor.
    pub fn prepare_owned_program(
        &self,
        program: fusion_pcu::dialect::tensor::TensorOwnedSelectedProgram,
    ) -> Result<RocmOwnedPreparedTensorGraph, RocmTensorExecutionError> {
        self.prepare_shared_owned_program(Arc::new(program))
    }

    /// Prepares an immutable captured program shared across cold device-admission attempts.
    ///
    /// Each candidate derives its own backend facts; rejection leaves the caller's program
    /// available for another candidate. Capturing source functions and cloning graph contents
    /// are unnecessary. Executions borrow this retained program without cloning its Arc.
    ///
    /// # Errors
    /// Returns unsupported operation or storage-layout errors for this selected device.
    pub fn prepare_shared_owned_program(
        &self,
        program: Arc<fusion_pcu::dialect::tensor::TensorOwnedSelectedProgram>,
    ) -> Result<RocmOwnedPreparedTensorGraph, RocmTensorExecutionError> {
        let data = prepare_owned_graph_data(&program, self)?;
        for &value in program.selected_nodes() {
            let node = program.graph().node(value)?;
            if matches!(node.op, OpDescriptor::MeanSquaredError { .. })
                && node.numerical_mode != Some(fusion_pcu::PcuNumericalMode::Strict)
            {
                self.prepare_native_mse(node.scalar_type)?;
            }
            if matches!(node.op, OpDescriptor::SgdUpdate { .. })
                && node.numerical_mode != Some(fusion_pcu::PcuNumericalMode::Strict)
            {
                self.prepare_native_sgd(
                    node.scalar_type,
                    node.numerical_options.precision
                        == fusion_pcu::PcuPrecisionPolicy::BackendOptimized
                        || data
                            .rewrites
                            .iter()
                            .any(|rewrite| rewrite.output == node.value),
                )?;
            }
        }
        RocmOwnedPreparedTensorGraph::from_parts(program, data)
    }

    /// Executes a graph-owning selected program from typed device inputs and returns fresh,
    /// independently owned typed device outputs.
    ///
    /// This first owned execution path deliberately allocates output storage per call. It does
    /// not infer storage donation or reuse an input allocation as an output. Every output is
    /// published only after the shared scheduler and its terminal stream completion succeed.
    ///
    /// # Errors
    ///
    /// Returns an input validation error, allocation/provider error, or a `ROCm` operation or
    /// completion error. On failure no output tensor is returned.
    pub fn execute_owned_program_outputs<T: fusion_pcu::PcuScalar, P>(
        &self,
        prepared: &RocmOwnedPreparedTensorGraph,
        inputs: &[(ValueId, &PcuDeviceTensor<T, RocmMemoryResource>)],
        pool: PcuMemoryPoolId,
        memory: &mut P,
    ) -> Result<Vec<RocmTensorOwnedOutput<T>>, RocmTensorExecutionError>
    where
        P: PcuMemoryProvider<Resource = RocmMemoryResource>,
    {
        let mut borrowed_inputs =
            SmallVec::<[(ValueId, RocmTensorInputRef<'_>); 8]>::with_capacity(inputs.len());
        for &(value, tensor) in inputs {
            borrowed_inputs.push((value, self.borrow_device_input_ref(tensor, pool)?));
        }
        let resource_inputs = borrowed_inputs
            .iter()
            .map(|(value, input)| (*value, input))
            .collect::<SmallVec<[_; 8]>>();
        self.execute_owned_program_outputs_from_inputs(prepared, &resource_inputs, pool, memory)
    }

    /// Executes a selected graph from one moved input. A selected identity graph transfers the
    /// owner unchanged without dispatch, and a prepared terminal same-index `ReLU` may reuse its
    /// allocation when graph proof and exclusive physical ownership permit it. Other supported
    /// single-input/single-output graphs use the ordinary fresh-output scheduler.
    ///
    /// The reuse route is never requested by a caller flag. Identity requires no mutation or
    /// uniqueness; in-place `ReLU` uses a prepared core graph proof and fixed read/write kernel,
    /// while execution checks the live resource lease and quiescence before dispatch. Its owner
    /// is returned only after terminal completion succeeds.
    ///
    /// # Errors
    ///
    /// Returns ordinary input, allocation, provider, dispatch, or completion errors. No tensor is
    /// returned after an unsuccessful in-place dispatch.
    pub fn execute_owned_program_consuming_input<T, P>(
        &self,
        prepared: &RocmOwnedPreparedTensorGraph,
        input: PcuDeviceTensor<T, RocmMemoryResource>,
        pool: PcuMemoryPoolId,
        memory: &mut P,
    ) -> Result<PcuDeviceTensor<T, RocmMemoryResource>, RocmTensorExecutionError>
    where
        T: fusion_pcu::PcuScalar,
        P: PcuMemoryProvider<Resource = RocmMemoryResource>,
    {
        if prepared.data.input_values.len() != 1 || prepared.data.outputs.len() != 1 {
            return Err(RocmTensorExecutionError::InvalidPlan(prepared.data.output));
        }
        let input_value = prepared.data.input_values[0];
        let borrowed = self.borrow_device_input_ref(&input, pool)?;
        let resource_inputs = [(input_value, &borrowed)];
        let view = prepared.view();
        self.validate_execution_sources_view(&view, &[], &resource_inputs, pool)?;

        if let Some(PreparedConsumingAction::IdentityTransfer(value)) =
            prepared.data.consuming_action.as_ref()
            && *value == input_value
            && prepared.data.output == input_value
        {
            validate_owned_scalar_profile::<T>(prepared.data.scalar_type)?;
            input
                .buffer()
                .resource()
                .validate_access_available()
                .map_err(|_| RocmTensorExecutionError::InputResourceMismatch)?;
            return Ok(input);
        }
        let Some(PreparedConsumingAction::TerminalRelu(reuse)) =
            prepared.data.consuming_action.as_ref()
        else {
            return self.execute_fresh_owned_single_output(
                prepared,
                input_value,
                &input,
                pool,
                memory,
            );
        };
        let resource = input.buffer().resource();
        let proof = reuse.proof;
        let required_alignment = scalar_layout(T::TYPE)?.1;
        let required_bytes = byte_len_for::<T>(input.shape())?;
        let can_reuse = proof.input() == input_value
            && proof.output() == prepared.data.output
            && proof.scalar_type() == T::TYPE
            && u64::try_from(required_bytes).ok() == Some(proof.bytes())
            && alignment_satisfies(resource.alignment_bytes(), required_alignment)
            && resource.access() == PcuMemoryAccess::ReadWrite
            && resource.backing_ownership() == PcuMemoryBackingOwnership::Exclusive
            && resource.validate_access_available().is_ok()
            && usize::try_from(reuse.logical_count).ok() == Some(input.buffer().len())
            && reuse.scalar_type.value_type() == reuse.value_type;
        if !can_reuse {
            return self.execute_fresh_owned_single_output(
                prepared,
                input_value,
                &input,
                pool,
                memory,
            );
        }

        let invocation_count =
            NonZeroU32::new(reuse.logical_count).ok_or(RocmTensorExecutionError::SizeOverflow)?;
        debug_assert_eq!(
            reuse.invocation_shape,
            PcuInvocationShape::invocations(invocation_count)
        );
        let cache_key =
            TensorDispatchCacheKey::ConsumingRelu(reuse.scalar_type, reuse.logical_count);
        self.ensure_dispatch_cached(cache_key, reuse.kernel, reuse.invocation_shape, false)?;
        let binding = self
            .session
            .bind(
                reuse.binding,
                PcuBindingAccess::ReadWrite,
                PcuBindingType::Value(reuse.value_type),
                resource,
            )
            .map_err(RocmTensorExecutionError::Backend)?;
        let mut completion = {
            let cache = self.state().add_dispatches.borrow();
            let executable = cache
                .iter()
                .find(|(key, _)| {
                    *key == TensorDispatchCacheKey::ConsumingRelu(
                        reuse.scalar_type,
                        reuse.logical_count,
                    )
                })
                .map(|(_, executable)| executable)
                .ok_or_else(|| RocmTensorExecutionError::InvalidPlan(reuse.proof.output()))?;
            executable
                .submit(std::slice::from_ref(&binding))
                .map_err(RocmTensorExecutionError::Backend)?
        };
        match completion
            .wait()
            .map_err(RocmTensorExecutionError::Completion)?
        {
            PcuCompletionOutcome::Succeeded => Ok(input),
            PcuCompletionOutcome::Failed => Err(RocmTensorExecutionError::FailedCompletion),
            PcuCompletionOutcome::Fault(fault) => {
                Err(RocmTensorExecutionError::ExecutionFault(fault))
            }
        }
    }

    fn execute_fresh_owned_single_output<T, P>(
        &self,
        prepared: &RocmOwnedPreparedTensorGraph,
        input_value: ValueId,
        input: &PcuDeviceTensor<T, RocmMemoryResource>,
        pool: PcuMemoryPoolId,
        memory: &mut P,
    ) -> Result<PcuDeviceTensor<T, RocmMemoryResource>, RocmTensorExecutionError>
    where
        T: fusion_pcu::PcuScalar,
        P: PcuMemoryProvider<Resource = RocmMemoryResource>,
    {
        let borrowed = self.borrow_device_input_ref(input, pool)?;
        self.execute_owned_program_output_from_inputs(
            prepared,
            &[(input_value, &borrowed)],
            pool,
            memory,
        )
    }

    /// Executes an owned program from erased, validated-layout resource inputs.
    ///
    /// Each input carries its scalar type explicitly; the selected graph profile and every
    /// input shape/type are checked before output allocation or device work begins.
    ///
    /// # Errors
    ///
    /// Returns an input validation error, allocation/provider error, or a `ROCm` operation or
    /// completion error. On failure no output tensor is returned.
    pub fn execute_owned_program_outputs_from_inputs<'input, T, P>(
        &'input self,
        prepared: &RocmOwnedPreparedTensorGraph,
        inputs: &[(ValueId, &RocmTensorInputRef<'input>)],
        pool: PcuMemoryPoolId,
        memory: &mut P,
    ) -> Result<Vec<RocmTensorOwnedOutput<T>>, RocmTensorExecutionError>
    where
        T: fusion_pcu::PcuScalar,
        P: PcuMemoryProvider<Resource = RocmMemoryResource>,
        'session: 'input,
    {
        let outputs =
            self.execute_owned_program_outputs_inline_from_inputs(prepared, inputs, pool, memory)?;
        if outputs.spilled() {
            // Arbitrary multi-output graphs already own a Vec-compatible backing allocation.
            return Ok(outputs.into_vec());
        }
        // The public Vec is unavoidable here, but collecting an inline single output through
        // SmallVec::into_vec would reserve the iterator's minimum capacity rather than one.
        let mut output_vec = Vec::with_capacity(outputs.len());
        output_vec.extend(outputs);
        Ok(output_vec)
    }

    /// Executes an owned graph prepared with exactly one output, returning that tensor directly.
    ///
    /// Output cardinality is checked before input validation, allocation, or dispatch. The shared
    /// scheduler and all completion/quiescence checks are the same as the multi-output adapter.
    ///
    /// # Errors
    /// Returns an output-count, input validation, allocation/provider, operation, or completion
    /// error. No tensor is returned unless terminal completion succeeds.
    pub fn execute_owned_program_output_from_inputs<'input, T, P>(
        &'input self,
        prepared: &RocmOwnedPreparedTensorGraph,
        inputs: &[(ValueId, &RocmTensorInputRef<'input>)],
        pool: PcuMemoryPoolId,
        memory: &mut P,
    ) -> Result<PcuDeviceTensor<T, RocmMemoryResource>, RocmTensorExecutionError>
    where
        T: fusion_pcu::PcuScalar,
        P: PcuMemoryProvider<Resource = RocmMemoryResource>,
        'session: 'input,
    {
        with_single_output_plan(prepared, || {
            let mut outputs = self
                .execute_owned_program_outputs_inline_from_inputs(prepared, inputs, pool, memory)?;
            if outputs.len() != 1 {
                return Err(RocmTensorExecutionError::OutputCountMismatch {
                    expected: 1,
                    actual: outputs.len(),
                });
            }
            outputs.pop().map(|(_, tensor)| tensor).ok_or(
                RocmTensorExecutionError::OutputCountMismatch {
                    expected: 1,
                    actual: 0,
                },
            )
        })
    }

    fn execute_owned_program_outputs_inline_from_inputs<'input, T, P>(
        &'input self,
        prepared: &RocmOwnedPreparedTensorGraph,
        inputs: &[(ValueId, &RocmTensorInputRef<'input>)],
        pool: PcuMemoryPoolId,
        memory: &mut P,
    ) -> Result<SmallVec<[RocmTensorOwnedOutput<T>; 1]>, RocmTensorExecutionError>
    where
        T: fusion_pcu::PcuScalar,
        P: PcuMemoryProvider<Resource = RocmMemoryResource>,
        'session: 'input,
    {
        if !is_checked_float_type(T::TYPE)
            && !prepared.data.transport_only_inputs
            && !is_checked_integer_scalar(T::TYPE)
            && !is_raw_float_type(T::TYPE)
        {
            return Err(RocmTensorExecutionError::UnsupportedScalarType(T::TYPE));
        }
        let view = prepared.view();
        validate_owned_scalar_profile::<T>(prepared.data.scalar_type)?;
        self.validate_execution_sources_view(&view, &[], inputs, pool)?;

        let mut bank = prepared
            .scratch
            .bind(self.session.tensor_runtime(), &view, pool, memory)?;
        if inputs.iter().any(|(_, input)| {
            bank.physical
                .iter()
                .any(|resource| resource.may_overlap(input.resource()))
        }) {
            return Err(RocmTensorExecutionError::ScratchMismatch);
        }
        let mut outputs = SmallVec::<[(ValueId, FreshTensorOutput<'_>); 1]>::new();
        outputs.reserve(view.outputs.len());
        for &value in &view.outputs {
            let shape = view.graph.shape(value)?;
            let resource = allocate_tensor_for::<T, P>(memory, pool, shape)?;
            if resource.pool() != pool
                || !resource.belongs_to_runtime(self.session.tensor_runtime())
                || resource.device_buffer().len() < byte_len_for::<T>(shape)?
                || !alignment_satisfies(resource.alignment_bytes(), scalar_layout(T::TYPE)?.1)
                || resource.access() != PcuMemoryAccess::ReadWrite
                || bank
                    .physical
                    .iter()
                    .any(|scratch| scratch.may_overlap(&resource))
                || inputs
                    .iter()
                    .any(|(_, input)| input.resource().may_overlap(&resource))
                || outputs
                    .iter()
                    .any(|(_, output): &(ValueId, FreshTensorOutput<'_>)| {
                        output.resource.may_overlap(&resource)
                    })
            {
                return Err(RocmTensorExecutionError::OutputResourceMismatch);
            }
            outputs.push((
                value,
                FreshTensorOutput {
                    shape,
                    resource,
                    scalar_type: T::TYPE,
                },
            ));
        }

        validate_owned_storage(&view, inputs, &outputs, &bank.resources)?;

        let bank_view = &mut *bank;
        let mut scratch_view = RocmExecutionScratch {
            resources: &bank_view.resources,
            statuses: Some(&mut bank_view.statuses),
            mse_squared: bank_view.mse_squared.as_ref(),
            outputs: &view.outputs,
            node_values: &view.node_values,
        };
        let mut timings = NoopNodeTiming;
        // Cold eligibility excludes numerical status gates and host-copy outputs. For a chain
        // of native GEMMs the selected stream orders dependencies; one final event proves
        // quiescence without a device-wide synchronization. The batch
        // retains every intermediate/library owner, including on an early operational error.
        let mut batch = prepared
            .data
            .batch_native_matmuls
            .then(|| HipCompletionBatch::new(&self.state().stream));
        let result = self.execute_prepared_schedule(
            &view,
            &[],
            inputs,
            pool,
            memory,
            Some(&mut scratch_view),
            None,
            Some(&outputs),
            &mut timings,
            batch.as_mut(),
            None,
        );
        // On error Drop establishes quiescence or quarantines owners before private outputs
        // can be released. On success the scheduler has already waited its final event.
        drop(batch);
        bank.finish(result.as_ref().err())?;
        let mut result = result?;
        drop(outputs);
        if result.len() != prepared.data.outputs.len() {
            return Err(RocmTensorExecutionError::OutputCountMismatch {
                expected: prepared.data.outputs.len(),
                actual: result.len(),
            });
        }
        result
            .drain(..)
            .zip(prepared.data.outputs.iter().copied())
            .map(|(output, value)| Ok((value, output.into_device_tensor::<T>()?)))
            .collect()
    }

    /// Execute a previously prepared output using new host inputs.
    ///
    /// Structural graph analysis is not repeated. Inputs are checked before the first device
    /// allocation, and the selected rocBLAS handle is checked before execution begins. The first
    /// Add of a particular flattened extent prepares its PCU executable; later calls reuse it.
    ///
    /// # Errors
    ///
    /// Returns an input error, memory-provider error, or a `ROCm` operation error.
    #[allow(clippy::too_many_lines)] // Keeps preflighted graph scheduling and resource release visible together.
    pub fn execute_prepared<P>(
        &self,
        prepared: &RocmPreparedTensorGraph<'_>,
        inputs: &[(ValueId, Tensor)],
        pool: PcuMemoryPoolId,
        memory: &mut P,
    ) -> Result<Tensor, RocmTensorExecutionError>
    where
        P: PcuMemoryProvider<Resource = RocmMemoryResource>,
    {
        let output = self.execute_prepared_resident(prepared, inputs, pool, memory)?;
        download_tensor(memory, &output)
    }

    /// Execute with HIP dispatch completions collected into final-event batches.
    ///
    /// HIP nodes, including rocBLAS SGEMM, remain ordered on this assessor's stream. A pending
    /// batch is completed before host/provider input transfers and synchronous rocBLAS operations.
    /// The default execution APIs remain synchronous per node.
    ///
    /// # Errors
    ///
    /// Returns the same graph, input, allocation, operation, or completion errors as the
    /// synchronous prepared execution path.
    pub fn execute_prepared_batched<P>(
        &self,
        prepared: &RocmPreparedTensorGraph<'_>,
        inputs: &[(ValueId, Tensor)],
        pool: PcuMemoryPoolId,
        memory: &mut P,
    ) -> Result<Tensor, RocmTensorExecutionError>
    where
        P: PcuMemoryProvider<Resource = RocmMemoryResource>,
    {
        let output = self.execute_prepared_resident_batched(prepared, inputs, pool, memory)?;
        download_tensor(memory, &output)
    }

    /// Execute with HIP dispatch batches and retain the output in device memory.
    ///
    /// # Errors
    ///
    /// Returns the same graph, input, allocation, operation, or completion errors as the
    /// synchronous resident execution path.
    pub fn execute_prepared_resident_batched<P>(
        &self,
        prepared: &RocmPreparedTensorGraph<'_>,
        inputs: &[(ValueId, Tensor)],
        pool: PcuMemoryPoolId,
        memory: &mut P,
    ) -> Result<RocmTensorInput<'session>, RocmTensorExecutionError>
    where
        P: PcuMemoryProvider<Resource = RocmMemoryResource>,
    {
        let mut batch = HipCompletionBatch::new(&self.state().stream);
        let mut outputs = self.execute_prepared_outputs_with_input_sources_and_scratch(
            prepared,
            inputs,
            &[] as &[(ValueId, &RocmTensorInput<'_>)],
            pool,
            memory,
            None,
            None,
            &mut NoopNodeTiming,
            Some(&mut batch),
        )?;
        Ok(outputs.remove(0))
    }

    /// Execute a prepared graph and retain its output in device memory.
    ///
    /// The returned resource is tied to this selected backend and can be supplied as an input to
    /// another prepared graph through [`Self::execute_prepared_with_resources_resident`]. This
    /// avoids both output readback and a subsequent host-to-device upload between graph stages.
    ///
    /// # Errors
    ///
    /// Returns an input error, memory-provider error, or a `ROCm` operation error.
    #[allow(clippy::too_many_lines)] // Keeps preflighted graph scheduling and resource release visible together.
    pub fn execute_prepared_resident<P>(
        &self,
        prepared: &RocmPreparedTensorGraph<'_>,
        inputs: &[(ValueId, Tensor)],
        pool: PcuMemoryPoolId,
        memory: &mut P,
    ) -> Result<RocmTensorInput<'session>, RocmTensorExecutionError>
    where
        P: PcuMemoryProvider<Resource = RocmMemoryResource>,
    {
        let mut timings = NoopNodeTiming;
        let mut outputs = self.execute_prepared_outputs_with_input_sources_and_scratch(
            prepared,
            inputs,
            &[] as &[(ValueId, &RocmTensorInput<'_>)],
            pool,
            memory,
            None,
            None,
            &mut timings,
            None,
        )?;
        Ok(outputs.remove(0))
    }

    /// Prepare persistent device storage for constants and non-output nodes in `prepared`.
    ///
    /// The scratch borrows both this assessor's selected session and the exact prepared plan.
    /// It is mutable during execution, which prevents overlapping use of its intermediate
    /// allocations. The requested output always gets a fresh allocation for each execution.
    ///
    /// # Errors
    ///
    /// Returns an allocation, upload, or selected-session resource mismatch error.
    pub fn prepare_scratch<'plan, 'graph, P>(
        &'session self,
        prepared: &'plan RocmPreparedTensorGraph<'graph>,
        pool: PcuMemoryPoolId,
        memory: &mut P,
    ) -> Result<RocmTensorScratch<'plan, 'graph, 'session>, RocmTensorExecutionError>
    where
        P: PcuMemoryProvider<Resource = RocmMemoryResource>,
    {
        require_f32_graph(&prepared.data)?;
        self.prepare_scratch_with_allocator(
            prepared,
            pool,
            memory,
            &mut |memory, pool, shape, upload| {
                if let Some(tensor) = upload {
                    upload_tensor(memory, pool, tensor)
                } else {
                    allocate_tensor(memory, pool, shape)
                }
            },
        )
    }

    pub(super) fn prepare_scratch_with_allocator<'plan, 'graph, P, A>(
        &'session self,
        prepared: &'plan RocmPreparedTensorGraph<'graph>,
        pool: PcuMemoryPoolId,
        memory: &mut P,
        allocate: &mut A,
    ) -> Result<RocmTensorScratch<'plan, 'graph, 'session>, RocmTensorExecutionError>
    where
        P: PcuMemoryProvider<Resource = RocmMemoryResource>,
        A: FnMut(
            &mut P,
            PcuMemoryPoolId,
            &[usize],
            Option<&Tensor>,
        ) -> Result<RocmMemoryResource, RocmTensorExecutionError>,
    {
        require_f32_graph(&prepared.data)?;
        let mut resources =
            self.allocate_planned_scratch_resources(prepared, pool, memory, allocate)?;
        let max_mse_count = mse_scratch_element_count(prepared.graph, &prepared.nodes)?;
        for (index, node) in prepared.nodes.iter().enumerate() {
            if resources[index].is_none()
                && !prepared.suppressed_adds.contains(&node.value)
                && scratch_stores_node(node.op, node.value, &prepared.outputs)
            {
                if is_scratch_computed_op(node.op) {
                    return Err(RocmTensorExecutionError::InvalidPlan(node.value));
                }
                resources[index] = match node.op {
                    OpDescriptor::Constant(tensor) => {
                        let tensor = tensor.as_typed::<f32>().map_err(|_| {
                            RocmTensorExecutionError::Unsupported {
                                value: node.value,
                                reason: TensorUnsupportedReason::ElementType,
                            }
                        })?;
                        Some(allocate(memory, pool, node.shape, Some(tensor))?)
                    }
                    OpDescriptor::Uniform { value } => {
                        let tensor = prepared.physical_layout(node.value)?.uniform_tensor(
                            node.shape,
                            value.as_typed::<f32>().map_err(|_| {
                                RocmTensorExecutionError::Unsupported {
                                    value: node.value,
                                    reason: TensorUnsupportedReason::ElementType,
                                }
                            })?,
                        )?;
                        Some(allocate(memory, pool, tensor.shape(), Some(&tensor))?)
                    }
                    OpDescriptor::Input => None,
                    _ => return Err(RocmTensorExecutionError::InvalidPlan(node.value)),
                };
            }
            if let Some(resource) = &resources[index]
                && (resource.pool() != pool
                    || !resource.belongs_to_runtime(self.session.tensor_runtime()))
            {
                return Err(RocmTensorExecutionError::ScratchMismatch);
            }
            if let Some(resource) = &resources[index] {
                physical_value_storage_requirement(prepared, node.value, node.op)?
                    .validate(resource)
                    .map_err(RocmTensorExecutionError::StorageConstraint)?;
            }
        }
        let mse_squared = if max_mse_count == 0 {
            None
        } else {
            let resource = allocate(memory, pool, &[max_mse_count], None)?;
            if resource.pool() != pool
                || !resource.belongs_to_runtime(self.session.tensor_runtime())
                || !mse_scratch_resource_fits(&resource, pool, max_mse_count)?
            {
                return Err(RocmTensorExecutionError::ScratchMismatch);
            }
            Some(resource)
        };
        Ok(RocmTensorScratch {
            prepared,
            session: self.session,
            pool,
            resources,
            mse_squared,
            poisoned: false,
        })
    }

    fn allocate_planned_scratch_resources<P, A>(
        &self,
        prepared: &RocmPreparedTensorGraph<'_>,
        pool: PcuMemoryPoolId,
        memory: &mut P,
        allocate: &mut A,
    ) -> Result<Vec<Option<RocmMemoryResource>>, RocmTensorExecutionError>
    where
        P: PcuMemoryProvider<Resource = RocmMemoryResource>,
        A: FnMut(
            &mut P,
            PcuMemoryPoolId,
            &[usize],
            Option<&Tensor>,
        ) -> Result<RocmMemoryResource, RocmTensorExecutionError>,
    {
        // The selected lowering plan measures inclusive lifetimes after rewrites and fusion.
        // Reuse is sound because this adapter submits operations in order on one stream, and
        // execution waits for completion before another call can reuse these slots.
        let storage_plan = selected_scratch_storage_plan(prepared)?;
        let mut slot_resources = Vec::with_capacity(storage_plan.slots().len());
        for slot in storage_plan.slots() {
            let element_bytes = size_of::<f32>();
            if !slot.capacity_bytes.is_multiple_of(element_bytes) {
                return Err(RocmTensorExecutionError::ScratchMismatch);
            }
            let elements = slot.capacity_bytes / element_bytes;
            let resource = allocate(memory, pool, &[elements], None)?;
            if resource.pool() != pool
                || !resource.belongs_to_runtime(self.session.tensor_runtime())
                || resource.size_bytes()
                    < u64::try_from(slot.capacity_bytes)
                        .map_err(|_| RocmTensorExecutionError::SizeOverflow)?
                || !alignment_satisfies(
                    resource.alignment_bytes(),
                    u64::try_from(slot.alignment_bytes)
                        .map_err(|_| RocmTensorExecutionError::SizeOverflow)?,
                )
                || resource.access() != PcuMemoryAccess::ReadWrite
                || !resource.supports(PcuMemoryResourceCapability::ReusableStorage)
            {
                return Err(RocmTensorExecutionError::ScratchMismatch);
            }
            slot_resources.push(resource);
        }

        let mut resources = (0..prepared.nodes.len()).map(|_| None).collect::<Vec<_>>();
        for assignment in storage_plan.assignments() {
            let index = prepared
                .index_of(assignment.value)
                .ok_or(RocmTensorExecutionError::InvalidPlan(assignment.value))?;
            let slot = slot_resources
                .get(assignment.slot)
                .ok_or(RocmTensorExecutionError::ScratchMismatch)?;
            if resources[index].is_some() {
                return Err(RocmTensorExecutionError::InvalidPlan(assignment.value));
            }
            physical_value_storage_requirement(
                prepared,
                assignment.value,
                prepared.nodes[index].op,
            )?
            .validate(slot)
            .map_err(RocmTensorExecutionError::StorageConstraint)?;
            resources[index] = Some(slot.clone_for_tensor_input());
        }
        Ok(resources)
    }

    /// Allocate reusable caller-owned storage for every output in the prepared plan.
    ///
    /// # Errors
    ///
    /// Returns an allocation, shape, or selected-session resource mismatch error.
    pub fn prepare_output_bank<'plan, 'graph, P>(
        &'session self,
        prepared: &'plan RocmPreparedTensorGraph<'graph>,
        pool: PcuMemoryPoolId,
        memory: &mut P,
    ) -> Result<RocmTensorOutputBank<'plan, 'graph, 'session>, RocmTensorExecutionError>
    where
        P: PcuMemoryProvider<Resource = RocmMemoryResource>,
    {
        self.prepare_output_bank_with_allocator(
            prepared,
            pool,
            memory,
            &mut |memory, pool, shape| allocate_tensor(memory, pool, shape),
        )
    }

    pub(super) fn prepare_output_bank_with_allocator<'plan, 'graph, P, A>(
        &'session self,
        prepared: &'plan RocmPreparedTensorGraph<'graph>,
        pool: PcuMemoryPoolId,
        memory: &mut P,
        allocate: &mut A,
    ) -> Result<RocmTensorOutputBank<'plan, 'graph, 'session>, RocmTensorExecutionError>
    where
        P: PcuMemoryProvider<Resource = RocmMemoryResource>,
        A: FnMut(
            &mut P,
            PcuMemoryPoolId,
            &[usize],
        ) -> Result<RocmMemoryResource, RocmTensorExecutionError>,
    {
        require_f32_graph(&prepared.data)?;
        let mut outputs = Vec::with_capacity(prepared.outputs.len());
        let mut requirements = Vec::with_capacity(prepared.outputs.len());
        for &value in &prepared.outputs {
            let shape = PcuOwnedShape::from_slice(prepared.graph.shape(value)?);
            let resource = allocate(memory, pool, shape.as_slice())?;
            if resource.pool() != pool
                || !resource.belongs_to_runtime(self.session.tensor_runtime())
                || resource.device_buffer().len() < byte_len(shape.as_slice())?
                || !alignment_satisfies(resource.alignment_bytes(), 4)
                || !matches!(
                    resource.access(),
                    PcuMemoryAccess::WriteOnly | PcuMemoryAccess::ReadWrite
                )
            {
                return Err(RocmTensorExecutionError::OutputResourceMismatch);
            }
            requirements.push(PcuMemoryMemberRequirement {
                pool,
                minimum_size_bytes: byte_len(shape.as_slice())? as u64,
                access: PcuMemoryAccess::ReadWrite,
                require_device_local: false,
            });
            outputs.push(RocmTensorInput {
                session: self.session,
                shape,
                resource,
                scalar_type: fusion_pcu::PcuScalarType::F32,
                updateable: true,
            });
        }
        validate_reusable_memory_bank_members_by(&outputs, &requirements, |output| {
            &output.resource
        })
        .map_err(|_| RocmTensorExecutionError::OutputResourceMismatch)?;
        Ok(RocmTensorOutputBank {
            prepared,
            session: self.session,
            pool,
            outputs,
            requirements,
            poisoned: false,
        })
    }

    /// Execute into reusable output storage. Outputs remain in the bank in prepared output order.
    ///
    /// # Errors
    ///
    /// Returns a plan, session, pool, shape, or allocation-alias mismatch, or an execution error.
    pub fn execute_prepared_outputs_into_bank<P>(
        &self,
        prepared: &RocmPreparedTensorGraph<'_>,
        inputs: &[(ValueId, &RocmTensorInput<'_>)],
        scratch: &mut RocmTensorScratch<'_, '_, 'session>,
        output_bank: &mut RocmTensorOutputBank<'_, '_, 'session>,
        memory: &mut P,
    ) -> Result<(), RocmTensorExecutionError>
    where
        P: PcuMemoryProvider<Resource = RocmMemoryResource>,
    {
        if output_bank.poisoned
            || !std::ptr::eq(output_bank.prepared, prepared)
            || !std::ptr::eq(output_bank.session, self.session)
            || output_bank.pool != scratch.pool
        {
            return Err(RocmTensorExecutionError::OutputResourceMismatch);
        }
        let result = self.execute_prepared_outputs_with_input_sources_and_scratch(
            prepared,
            &[],
            inputs,
            output_bank.pool,
            memory,
            Some(&mut *scratch),
            Some(output_bank),
            &mut NoopNodeTiming,
            None,
        );
        if result.is_err() {
            scratch.poisoned = true;
            output_bank.poisoned = true;
        }
        result.map(drop)
    }

    /// Diagnostic variant of [`Self::execute_prepared_outputs_into_bank`] that returns
    /// synchronous wall time for each prepared node in execution order.
    ///
    /// Timing collection is deliberately isolated to this entry point because reading the clock
    /// and retaining one record per node perturb short kernels.
    ///
    /// # Errors
    ///
    /// Returns the same validation, memory-provider, operation, or completion errors as the
    /// unprofiled output-bank path.
    pub fn execute_prepared_outputs_into_bank_profiled<P>(
        &self,
        prepared: &RocmPreparedTensorGraph<'_>,
        inputs: &[(ValueId, &RocmTensorInput<'_>)],
        scratch: &mut RocmTensorScratch<'_, '_, 'session>,
        output_bank: &mut RocmTensorOutputBank<'_, '_, 'session>,
        memory: &mut P,
    ) -> Result<Vec<RocmTensorNodeTiming>, RocmTensorExecutionError>
    where
        P: PcuMemoryProvider<Resource = RocmMemoryResource>,
    {
        if output_bank.poisoned
            || !std::ptr::eq(output_bank.prepared, prepared)
            || !std::ptr::eq(output_bank.session, self.session)
            || output_bank.pool != scratch.pool
        {
            return Err(RocmTensorExecutionError::OutputResourceMismatch);
        }
        let mut timings = CollectNodeTimings::default();
        let result = self.execute_prepared_outputs_with_input_sources_and_scratch(
            prepared,
            &[],
            inputs,
            output_bank.pool,
            memory,
            Some(&mut *scratch),
            Some(output_bank),
            &mut timings,
            None,
        );
        if result.is_err() {
            scratch.poisoned = true;
            output_bank.poisoned = true;
        }
        result.map(|_| timings.0)
    }

    /// Execute into a reusable output bank with HIP dispatch batching enabled.
    ///
    /// Batches flush at host-transfer and synchronous rocBLAS boundaries, and before this method
    /// returns. SGEMM nodes are enqueued into the graph batch on the assessor's stream.
    /// Scratch and bank storage are poisoned after an execution error, once the batch builder has
    /// synchronized or quarantined any queued work.
    ///
    /// # Errors
    ///
    /// Returns the same validation, memory-provider, operation, or completion errors as the
    /// synchronous output-bank path.
    pub fn execute_prepared_outputs_into_bank_batched<P>(
        &self,
        prepared: &RocmPreparedTensorGraph<'_>,
        inputs: &[(ValueId, &RocmTensorInput<'_>)],
        scratch: &mut RocmTensorScratch<'_, '_, 'session>,
        output_bank: &mut RocmTensorOutputBank<'_, '_, 'session>,
        memory: &mut P,
    ) -> Result<(), RocmTensorExecutionError>
    where
        P: PcuMemoryProvider<Resource = RocmMemoryResource>,
    {
        if output_bank.poisoned
            || !std::ptr::eq(output_bank.prepared, prepared)
            || !std::ptr::eq(output_bank.session, self.session)
            || output_bank.pool != scratch.pool
        {
            return Err(RocmTensorExecutionError::OutputResourceMismatch);
        }
        let mut batch = HipCompletionBatch::new(&self.state().stream);
        let result = self.execute_prepared_outputs_with_input_sources_and_scratch(
            prepared,
            &[],
            inputs,
            output_bank.pool,
            memory,
            Some(&mut *scratch),
            Some(output_bank),
            &mut NoopNodeTiming,
            Some(&mut batch),
        );
        drop(batch);
        if result.is_err() {
            scratch.poisoned = true;
            output_bank.poisoned = true;
        }
        result.map(drop)
    }

    fn execute_prepared_outputs_into_bank_batched_proven<P>(
        &self,
        prepared: &RocmPreparedTensorGraph<'_>,
        inputs: &[(ValueId, &RocmTensorInput<'_>)],
        scratch: &mut RocmTensorScratch<'_, '_, 'session>,
        output_bank: &mut RocmTensorOutputBank<'_, '_, 'session>,
        memory: &mut P,
        proof: &feedback_runtime::execution::ResidentFeedbackProof<'_, '_, '_>,
    ) -> Result<(), RocmTensorExecutionError>
    where
        P: PcuMemoryProvider<Resource = RocmMemoryResource>,
    {
        if output_bank.poisoned
            || scratch.poisoned
            || !proof.matches(prepared, self.session, proof.pool())
            || !std::ptr::eq(output_bank.prepared, prepared)
            || !std::ptr::eq(output_bank.session, self.session)
            || !std::ptr::eq(scratch.prepared, prepared)
            || !std::ptr::eq(scratch.session, self.session)
            || output_bank.pool != proof.pool()
            || scratch.pool != proof.pool()
        {
            return Err(RocmTensorExecutionError::OutputResourceMismatch);
        }
        let mut batch = HipCompletionBatch::new(&self.state().stream);
        let result = self.execute_prepared_outputs_with_input_sources_and_scratch_proven(
            prepared,
            &[],
            inputs,
            proof.pool(),
            memory,
            Some(&mut *scratch),
            Some(output_bank),
            &mut NoopNodeTiming,
            Some(&mut batch),
            Some(proof),
        );
        drop(batch);
        if result.is_err() {
            scratch.poisoned = true;
            output_bank.poisoned = true;
        }
        result.map(drop)
    }

    /// Diagnostic variant of [`Self::execute_prepared_outputs_into_bank_batched`] that returns
    /// per-node host wall time spent while executing prepared nodes in order.
    ///
    /// Timing collection is deliberately isolated to this entry point because reading the clock
    /// and retaining one record per node perturb short kernels. Batched nodes, including SGEMM,
    /// may only enqueue work; their deferred completion wait is charged to a later batch flush or
    /// occurs after the node loop when the final batch is dropped. SGEMM host phase timings are
    /// omitted in this asynchronous path. Batch completion is dropped
    /// before reusable storage is poisoned on error, so queued work is synchronized or
    /// quarantined before this method returns.
    ///
    /// # Errors
    ///
    /// Returns the same validation, memory-provider, operation, or completion errors as the
    /// batched output-bank path.
    pub fn execute_prepared_outputs_into_bank_batched_profiled<P>(
        &self,
        prepared: &RocmPreparedTensorGraph<'_>,
        inputs: &[(ValueId, &RocmTensorInput<'_>)],
        scratch: &mut RocmTensorScratch<'_, '_, 'session>,
        output_bank: &mut RocmTensorOutputBank<'_, '_, 'session>,
        memory: &mut P,
    ) -> Result<Vec<RocmTensorNodeTiming>, RocmTensorExecutionError>
    where
        P: PcuMemoryProvider<Resource = RocmMemoryResource>,
    {
        if output_bank.poisoned
            || !std::ptr::eq(output_bank.prepared, prepared)
            || !std::ptr::eq(output_bank.session, self.session)
            || output_bank.pool != scratch.pool
        {
            return Err(RocmTensorExecutionError::OutputResourceMismatch);
        }
        let mut timings = CollectNodeTimings::default();
        let mut batch = HipCompletionBatch::new(&self.state().stream);
        let result = self.execute_prepared_outputs_with_input_sources_and_scratch(
            prepared,
            &[],
            inputs,
            output_bank.pool,
            memory,
            Some(&mut *scratch),
            Some(output_bank),
            &mut timings,
            Some(&mut batch),
        );
        drop(batch);
        if result.is_err() {
            scratch.poisoned = true;
            output_bank.poisoned = true;
        }
        result.map(|_| timings.0)
    }

    /// Opt-in phase profile of batched prepared output-bank execution using caller raw ticks.
    ///
    /// The clock closure is invoked only by this diagnostic method. Supply a monotonic raw tick
    /// source and use its calibration to convert returned ticks. Batched `MatMul` records measure
    /// host scheduling/enqueue time, not GPU execution time.
    ///
    /// # Errors
    ///
    /// Returns the same validation, memory-provider, operation, or completion errors as batched
    /// output-bank execution.
    #[cfg(feature = "insights")]
    pub fn execute_prepared_outputs_into_bank_batched_insights_profiled<P, C>(
        &self,
        prepared: &RocmPreparedTensorGraph<'_>,
        inputs: &[(ValueId, &RocmTensorInput<'_>)],
        scratch: &mut RocmTensorScratch<'_, '_, 'session>,
        output_bank: &mut RocmTensorOutputBank<'_, '_, 'session>,
        memory: &mut P,
        clock: &mut C,
    ) -> Result<RocmTensorBatchedExecutionTiming, RocmTensorExecutionError>
    where
        P: PcuMemoryProvider<Resource = RocmMemoryResource>,
        C: fusion_pcu::insights::InsightClock,
    {
        let mut timings = CollectBatchedInsightsTiming::new(clock);
        let total_start = timings.stamp();
        let validation_start = timings.stamp();
        let valid = !output_bank.poisoned
            && std::ptr::eq(output_bank.prepared, prepared)
            && std::ptr::eq(output_bank.session, self.session)
            && output_bank.pool == scratch.pool;
        let validation_end = timings.stamp();
        timings.record_phase(
            InsightsPhase::EntryValidation,
            validation_start,
            validation_end,
        );
        if !valid {
            return Err(RocmTensorExecutionError::OutputResourceMismatch);
        }

        let mut batch = HipCompletionBatch::new(&self.state().stream);
        let result = self.execute_prepared_outputs_with_input_sources_and_scratch(
            prepared,
            &[],
            inputs,
            output_bank.pool,
            memory,
            Some(&mut *scratch),
            Some(output_bank),
            &mut timings,
            Some(&mut batch),
        );
        let cleanup_start = timings.stamp();
        drop(batch);
        let cleanup_end = timings.stamp();
        timings.record_phase(InsightsPhase::BatchDropCleanup, cleanup_start, cleanup_end);
        if result.is_err() {
            scratch.poisoned = true;
            output_bank.poisoned = true;
        }
        result?;

        let total_end = timings.stamp();
        timings.record_phase(InsightsPhase::Total, total_start, total_end);
        Ok(timings.into_profile())
    }

    /// Execute a prepared output-bank graph with opt-in per-batch HIP device timing.
    ///
    /// Returned durations are HIP stream-event intervals in milliseconds, in submission order.
    /// Each begins immediately before a batch's first launch and ends when the batch flushes.
    /// The interval excludes rocBLAS work but may include stream-idle time between launches or
    /// between the last launch and the flush marker. It is not a GPU-busy kernel-duration sum.
    /// The normal batched path creates no timing events.
    ///
    /// # Errors
    ///
    /// Returns the same validation, memory, launch, or completion errors as batched execution,
    /// plus timing-event failures. An execution failure poisons the reusable storage.
    pub fn execute_prepared_outputs_into_bank_batched_device_timed<P>(
        &self,
        prepared: &RocmPreparedTensorGraph<'_>,
        inputs: &[(ValueId, &RocmTensorInput<'_>)],
        scratch: &mut RocmTensorScratch<'_, '_, 'session>,
        output_bank: &mut RocmTensorOutputBank<'_, '_, 'session>,
        memory: &mut P,
    ) -> Result<Vec<f32>, RocmTensorExecutionError>
    where
        P: PcuMemoryProvider<Resource = RocmMemoryResource>,
    {
        if output_bank.poisoned
            || !std::ptr::eq(output_bank.prepared, prepared)
            || !std::ptr::eq(output_bank.session, self.session)
            || output_bank.pool != scratch.pool
        {
            return Err(RocmTensorExecutionError::OutputResourceMismatch);
        }
        let mut batch = HipCompletionBatch::new_timed(&self.state().stream);
        let result = self.execute_prepared_outputs_with_input_sources_and_scratch(
            prepared,
            &[],
            inputs,
            output_bank.pool,
            memory,
            Some(&mut *scratch),
            Some(output_bank),
            &mut NoopNodeTiming,
            Some(&mut batch),
        );
        match result {
            Ok(_) => Ok(batch.timings_ms()),
            Err(error) => {
                drop(batch);
                scratch.poisoned = true;
                output_bank.poisoned = true;
                Err(error)
            }
        }
    }

    /// Execute a prepared graph with caller-owned reusable constants and intermediate storage.
    ///
    /// # Errors
    ///
    /// Returns an input or scratch identity error, memory-provider error, or operation error.
    pub fn execute_prepared_with_scratch_resident<P>(
        &self,
        prepared: &RocmPreparedTensorGraph<'_>,
        inputs: &[(ValueId, Tensor)],
        scratch: &mut RocmTensorScratch<'_, '_, 'session>,
        memory: &mut P,
    ) -> Result<RocmTensorInput<'session>, RocmTensorExecutionError>
    where
        P: PcuMemoryProvider<Resource = RocmMemoryResource>,
    {
        if scratch.poisoned
            || !std::ptr::eq(scratch.prepared, prepared)
            || !std::ptr::eq(scratch.session, self.session)
        {
            return Err(RocmTensorExecutionError::ScratchMismatch);
        }
        let mut timings = NoopNodeTiming;
        let result = self.execute_prepared_outputs_with_input_sources_and_scratch(
            prepared,
            inputs,
            &[] as &[(ValueId, &RocmTensorInput<'_>)],
            scratch.pool,
            memory,
            Some(&mut *scratch),
            None,
            &mut timings,
            None,
        );
        if result.is_err() {
            scratch.poisoned = true;
        }
        result.map(|mut outputs| outputs.remove(0))
    }

    /// Execute with reusable scratch and HIP dispatch batching enabled.
    ///
    /// # Errors
    ///
    /// Returns the same input, scratch, memory-provider, operation, or completion errors as the
    /// synchronous scratch path.
    pub fn execute_prepared_with_scratch_resident_batched<P>(
        &self,
        prepared: &RocmPreparedTensorGraph<'_>,
        inputs: &[(ValueId, Tensor)],
        scratch: &mut RocmTensorScratch<'_, '_, 'session>,
        memory: &mut P,
    ) -> Result<RocmTensorInput<'session>, RocmTensorExecutionError>
    where
        P: PcuMemoryProvider<Resource = RocmMemoryResource>,
    {
        if scratch.poisoned
            || !std::ptr::eq(scratch.prepared, prepared)
            || !std::ptr::eq(scratch.session, self.session)
        {
            return Err(RocmTensorExecutionError::ScratchMismatch);
        }
        let mut batch = HipCompletionBatch::new(&self.state().stream);
        let result = self.execute_prepared_outputs_with_input_sources_and_scratch(
            prepared,
            inputs,
            &[] as &[(ValueId, &RocmTensorInput<'_>)],
            scratch.pool,
            memory,
            Some(&mut *scratch),
            None,
            &mut NoopNodeTiming,
            Some(&mut batch),
        );
        drop(batch);
        if result.is_err() {
            scratch.poisoned = true;
        }
        result.map(|mut outputs| outputs.remove(0))
    }

    /// Execute a prepared graph with reusable scratch and persistent device inputs.
    ///
    /// # Errors
    ///
    /// Returns an input or scratch identity error, memory-provider error, or operation error.
    pub fn execute_prepared_with_resources_and_scratch_resident<P>(
        &self,
        prepared: &RocmPreparedTensorGraph<'_>,
        inputs: &[(ValueId, &RocmTensorInput<'_>)],
        scratch: &mut RocmTensorScratch<'_, '_, 'session>,
        memory: &mut P,
    ) -> Result<RocmTensorInput<'session>, RocmTensorExecutionError>
    where
        P: PcuMemoryProvider<Resource = RocmMemoryResource>,
    {
        let mut outputs = self.execute_prepared_outputs_with_resources_and_scratch_resident(
            prepared, inputs, scratch, memory,
        )?;
        Ok(outputs.remove(0))
    }

    /// Execute from persistent device inputs with reusable scratch and HIP batching enabled.
    ///
    /// # Errors
    ///
    /// Returns the same input, scratch, memory-provider, operation, or completion errors as the
    /// synchronous resident path.
    pub fn execute_prepared_with_resources_and_scratch_resident_batched<P>(
        &self,
        prepared: &RocmPreparedTensorGraph<'_>,
        inputs: &[(ValueId, &RocmTensorInput<'_>)],
        scratch: &mut RocmTensorScratch<'_, '_, 'session>,
        memory: &mut P,
    ) -> Result<RocmTensorInput<'session>, RocmTensorExecutionError>
    where
        P: PcuMemoryProvider<Resource = RocmMemoryResource>,
    {
        let mut outputs = self
            .execute_prepared_outputs_with_resources_and_scratch_resident_batched(
                prepared, inputs, scratch, memory,
            )?;
        Ok(outputs.remove(0))
    }

    /// Execute a prepared multi-output graph with persistent inputs and reusable scratch.
    ///
    /// Returned outputs follow the order supplied to [`Self::prepare_graph_outputs`].
    ///
    /// # Errors
    ///
    /// Returns an input or scratch identity error, memory-provider error, or operation error.
    pub fn execute_prepared_outputs_with_resources_and_scratch_resident<P>(
        &self,
        prepared: &RocmPreparedTensorGraph<'_>,
        inputs: &[(ValueId, &RocmTensorInput<'_>)],
        scratch: &mut RocmTensorScratch<'_, '_, 'session>,
        memory: &mut P,
    ) -> Result<Vec<RocmTensorInput<'session>>, RocmTensorExecutionError>
    where
        P: PcuMemoryProvider<Resource = RocmMemoryResource>,
    {
        self.execute_prepared_outputs_with_resources_and_scratch_resident_timed(
            prepared,
            inputs,
            scratch,
            memory,
            &mut NoopNodeTiming,
        )
        .map(SmallVec::into_vec)
    }

    /// Execute multiple outputs from persistent inputs with reusable scratch and HIP batching.
    ///
    /// The final batch event completes before this method returns. On any error, the batch is
    /// dropped and establishes quiescence or quarantines its resources before scratch is poisoned.
    ///
    /// # Errors
    ///
    /// Returns the same validation, memory-provider, operation, or completion errors as the
    /// synchronous multi-output path.
    pub fn execute_prepared_outputs_with_resources_and_scratch_resident_batched<P>(
        &self,
        prepared: &RocmPreparedTensorGraph<'_>,
        inputs: &[(ValueId, &RocmTensorInput<'_>)],
        scratch: &mut RocmTensorScratch<'_, '_, 'session>,
        memory: &mut P,
    ) -> Result<Vec<RocmTensorInput<'session>>, RocmTensorExecutionError>
    where
        P: PcuMemoryProvider<Resource = RocmMemoryResource>,
    {
        if scratch.poisoned
            || !std::ptr::eq(scratch.prepared, prepared)
            || !std::ptr::eq(scratch.session, self.session)
        {
            return Err(RocmTensorExecutionError::ScratchMismatch);
        }
        let mut batch = HipCompletionBatch::new(&self.state().stream);
        let result = self.execute_prepared_outputs_with_input_sources_and_scratch(
            prepared,
            &[],
            inputs,
            scratch.pool,
            memory,
            Some(&mut *scratch),
            None,
            &mut NoopNodeTiming,
            Some(&mut batch),
        );
        drop(batch);
        if result.is_err() {
            scratch.poisoned = true;
        }
        result.map(SmallVec::into_vec)
    }

    /// Diagnostic variant of [`Self::execute_prepared_outputs_with_resources_and_scratch_resident`]
    /// that also returns synchronous wall time for each prepared node in execution order.
    ///
    /// Timing collection is deliberately isolated to this entry point because reading the clock
    /// and retaining one record per node perturb short kernels.
    ///
    /// # Errors
    ///
    /// Returns the same validation, memory-provider, or operation errors as the unprofiled
    /// execution entry point.
    pub fn execute_prepared_outputs_with_resources_and_scratch_resident_profiled<P>(
        &self,
        prepared: &RocmPreparedTensorGraph<'_>,
        inputs: &[(ValueId, &RocmTensorInput<'_>)],
        scratch: &mut RocmTensorScratch<'_, '_, 'session>,
        memory: &mut P,
    ) -> Result<(Vec<RocmTensorInput<'session>>, Vec<RocmTensorNodeTiming>), RocmTensorExecutionError>
    where
        P: PcuMemoryProvider<Resource = RocmMemoryResource>,
    {
        let mut timings = CollectNodeTimings::default();
        let outputs = self.execute_prepared_outputs_with_resources_and_scratch_resident_timed(
            prepared,
            inputs,
            scratch,
            memory,
            &mut timings,
        )?;
        Ok((outputs.into_vec(), timings.0))
    }

    fn execute_prepared_outputs_with_resources_and_scratch_resident_timed<P, T>(
        &self,
        prepared: &RocmPreparedTensorGraph<'_>,
        inputs: &[(ValueId, &RocmTensorInput<'_>)],
        scratch: &mut RocmTensorScratch<'_, '_, 'session>,
        memory: &mut P,
        timings: &mut T,
    ) -> Result<TensorExecutionOutputs<'session>, RocmTensorExecutionError>
    where
        P: PcuMemoryProvider<Resource = RocmMemoryResource>,
        T: NodeTimingSink,
    {
        if scratch.poisoned
            || !std::ptr::eq(scratch.prepared, prepared)
            || !std::ptr::eq(scratch.session, self.session)
        {
            return Err(RocmTensorExecutionError::ScratchMismatch);
        }
        let result = self.execute_prepared_outputs_with_input_sources_and_scratch(
            prepared,
            &[],
            inputs,
            scratch.pool,
            memory,
            Some(&mut *scratch),
            None,
            timings,
            None,
        );
        if result.is_err() {
            scratch.poisoned = true;
        }
        result
    }

    /// Upload a tensor once for reuse as an immutable graph input on this selected `ROCm` session.
    ///
    /// The returned input owns the allocation and is bound to this backend. Call
    /// [`Self::update_input`] to synchronously replace its contents before a later execution.
    ///
    /// # Errors
    ///
    /// Returns an allocation or host-to-device transfer error.
    pub fn upload_input<P>(
        &self,
        tensor: &Tensor,
        pool: PcuMemoryPoolId,
        memory: &mut P,
    ) -> Result<RocmTensorInput<'session>, RocmTensorExecutionError>
    where
        P: PcuMemoryProvider<Resource = RocmMemoryResource>,
    {
        let resource = upload_tensor(memory, pool, tensor)?;
        if !resource.belongs_to_runtime(self.session.tensor_runtime()) {
            return Err(RocmTensorExecutionError::InputResourceMismatch);
        }
        Ok(RocmTensorInput {
            session: self.session,
            shape: PcuOwnedShape::from_slice(tensor.shape()),
            resource,
            scalar_type: fusion_pcu::PcuScalarType::F32,
            updateable: true,
        })
    }

    /// Borrow an already resident typed tensor as an immutable input to this session's tensor
    /// graph without transferring its contents through host memory.
    ///
    /// The input retains a cloned provider lease while the caller keeps its original tensor.
    /// The graph treats inputs as read-only, and storage-constraint validation still requires
    /// writable outputs and scratch to be disjoint from this backing allocation.
    ///
    /// `pool` is the pool expected by the prepared graph. This method checks session/device,
    /// pool, dense shape, byte extent, and read permission before cloning the resource lease.
    /// The returned value borrows both the assessor and source tensor; the source cannot be
    /// moved or mutably accessed until the input view is dropped. Borrowed views also reject
    /// [`Self::update_input`].
    ///
    /// # Errors
    ///
    /// Returns a session, pool, extent, or access mismatch error when the resource cannot safely
    /// serve as a read-only graph input.
    ///
    /// ```
    /// use fusion_pcu_core::{PcuDeviceTensor, PcuMemoryPoolId};
    /// use fusion_pcu_rocm::{RocmMemoryResource, RocmTensorAssessor};
    ///
    /// fn release_after_use<'session>(
    ///     assessor: &RocmTensorAssessor<'session>,
    ///     owner: PcuDeviceTensor<f32, RocmMemoryResource>,
    ///     pool: PcuMemoryPoolId,
    /// ) {
    ///     let borrowed = assessor.borrow_device_input(&owner, pool);
    ///     drop(borrowed);
    ///     drop(owner);
    /// }
    /// ```
    ///
    /// ```compile_fail
    /// use fusion_pcu_core::{PcuDeviceTensor, PcuMemoryPoolId};
    /// use fusion_pcu_rocm::{RocmMemoryResource, RocmTensorAssessor};
    ///
    /// fn move_is_rejected_while_input_is_live<'session>(
    ///     assessor: &RocmTensorAssessor<'session>,
    ///     owner: PcuDeviceTensor<f32, RocmMemoryResource>,
    ///     pool: PcuMemoryPoolId,
    /// ) {
    ///     let Ok(input) = assessor.borrow_device_input(&owner, pool) else { return };
    ///     drop(owner);
    ///     drop(input);
    /// }
    /// ```
    pub fn borrow_device_input<'input, T: fusion_pcu::PcuScalar>(
        &'input self,
        tensor: &'input PcuDeviceTensor<T, RocmMemoryResource>,
        pool: PcuMemoryPoolId,
    ) -> Result<RocmTensorInput<'input>, RocmTensorExecutionError>
    where
        'session: 'input,
    {
        self.borrow_resource_input(tensor.buffer().resource(), tensor.shape(), T::TYPE, pool)
    }

    /// Validate and borrow a typed device tensor without allocating per-call input metadata.
    ///
    /// # Errors
    ///
    /// Returns a session, pool, extent, alignment, or access mismatch.
    pub fn borrow_device_input_ref<'input, T: fusion_pcu::PcuScalar>(
        &'input self,
        tensor: &'input PcuDeviceTensor<T, RocmMemoryResource>,
        pool: PcuMemoryPoolId,
    ) -> Result<RocmTensorInputRef<'input>, RocmTensorExecutionError>
    where
        'session: 'input,
    {
        self.borrow_resource_input_ref(tensor.buffer().resource(), tensor.shape(), T::TYPE, pool)
    }

    /// Borrow an erased resource as a typed graph input after validating its declared scalar
    /// layout against the selected session and physical allocation metadata.
    ///
    /// # Errors
    ///
    /// Returns a session, pool, extent, alignment, or access mismatch.
    pub fn borrow_resource_input<'input>(
        &'input self,
        resource: &'input RocmMemoryResource,
        shape: &'input [usize],
        scalar_type: fusion_pcu::PcuScalarType,
        pool: PcuMemoryPoolId,
    ) -> Result<RocmTensorInput<'input>, RocmTensorExecutionError>
    where
        'session: 'input,
    {
        let borrowed = self.borrow_resource_input_ref(resource, shape, scalar_type, pool)?;
        Ok(RocmTensorInput {
            session: borrowed.session,
            shape: PcuOwnedShape::from_slice(borrowed.shape),
            resource: borrowed.resource.clone_read_only_for_tensor_input(),
            scalar_type: borrowed.scalar_type,
            updateable: false,
        })
    }

    /// Validate and borrow an erased device resource without allocating descriptor storage.
    ///
    /// The caller retains ownership of both the resource and shape for the returned lifetime.
    ///
    /// # Errors
    ///
    /// Returns a session, pool, extent, alignment, or access mismatch.
    pub fn borrow_resource_input_ref<'input>(
        &'input self,
        resource: &'input RocmMemoryResource,
        shape: &'input [usize],
        scalar_type: fusion_pcu::PcuScalarType,
        pool: PcuMemoryPoolId,
    ) -> Result<RocmTensorInputRef<'input>, RocmTensorExecutionError>
    where
        'session: 'input,
    {
        let (element_size, alignment) = scalar_layout(scalar_type)?;
        if !resource.belongs_to_runtime(self.session.tensor_runtime()) {
            return Err(RocmTensorExecutionError::InputResourceMismatch);
        }
        if resource.pool() != pool {
            return Err(RocmTensorExecutionError::InputPoolMismatch);
        }
        let required_bytes = byte_len_for_size(shape, element_size)?;
        if u64::try_from(required_bytes).map_err(|_| RocmTensorExecutionError::SizeOverflow)?
            > resource.size_bytes()
            || resource.device_buffer().len() < required_bytes
            || !alignment_satisfies(resource.alignment_bytes(), alignment)
        {
            return Err(RocmTensorExecutionError::InputResourceMismatch);
        }
        if !matches!(
            resource.access(),
            PcuMemoryAccess::ReadOnly | PcuMemoryAccess::ReadWrite
        ) {
            return Err(RocmTensorExecutionError::InputAccessMismatch);
        }
        Ok(RocmTensorInputRef {
            session: self.session,
            shape,
            resource,
            scalar_type,
        })
    }

    /// Replace the contents of a reusable input with a synchronous host-to-device transfer.
    ///
    /// Shape changes are rejected; create a new input when the graph input shape changes.
    ///
    /// # Errors
    ///
    /// Returns a session mismatch, shape mismatch, or host-to-device transfer error.
    pub fn update_input<P>(
        &self,
        input: &mut RocmTensorInput<'_>,
        tensor: &Tensor,
        memory: &mut P,
    ) -> Result<(), RocmTensorExecutionError>
    where
        P: PcuMemoryProvider<Resource = RocmMemoryResource>,
    {
        if input.scalar_type != fusion_pcu::PcuScalarType::F32 {
            return Err(RocmTensorExecutionError::UnsupportedScalarType(
                input.scalar_type,
            ));
        }
        input.validate_session(self.session)?;
        if !input.updateable {
            return Err(RocmTensorExecutionError::BorrowedInputUpdate);
        }
        if tensor.shape() != input.shape.as_slice() {
            return Err(TensorError::ShapeMismatch {
                left: input.shape.as_slice().to_vec(),
                right: tensor.shape().to_vec(),
            }
            .into());
        }
        transfer_tensor(memory, &mut input.resource, tensor)
    }

    /// Copy a resident device tensor to a host [`Tensor`] after validating its session and pool.
    ///
    /// Keep outputs resident between graph stages and call this only when host access is needed.
    ///
    /// # Errors
    ///
    /// Returns a session, device, or pool mismatch, or a device-to-host transfer error.
    pub fn download_output<P>(
        &self,
        output: &RocmTensorInput<'_>,
        pool: PcuMemoryPoolId,
        memory: &mut P,
    ) -> Result<Tensor, RocmTensorExecutionError>
    where
        P: PcuMemoryProvider<Resource = RocmMemoryResource>,
    {
        output.validate_session(self.session)?;
        if output.resource.pool() != pool {
            return Err(RocmTensorExecutionError::InputPoolMismatch);
        }
        download_tensor(memory, output)
    }

    /// Execute a prepared graph using persistent device inputs, reusing their allocations.
    ///
    /// Every graph input must have exactly one matching resource. Resources must belong to this
    /// selected backend and match the graph shape. No host upload or backend fallback occurs.
    ///
    /// # Errors
    ///
    /// Returns an input validation error, resource identity error, or execution error.
    pub fn execute_prepared_with_resources<P>(
        &self,
        prepared: &RocmPreparedTensorGraph<'_>,
        inputs: &[(ValueId, &RocmTensorInput<'_>)],
        pool: PcuMemoryPoolId,
        memory: &mut P,
    ) -> Result<Tensor, RocmTensorExecutionError>
    where
        P: PcuMemoryProvider<Resource = RocmMemoryResource>,
    {
        let output =
            self.execute_prepared_with_resources_resident(prepared, inputs, pool, memory)?;
        download_tensor(memory, &output)
    }

    /// Execute a prepared graph from persistent inputs and return a reusable resident output.
    ///
    /// The output may be passed directly as an input to a later graph on this same selected
    /// backend and pool. The executor validates every input and the output's device identity
    /// before dispatch; it performs no host readback.
    ///
    /// # Errors
    ///
    /// Returns an input validation error, resource identity error, or execution error.
    pub fn execute_prepared_with_resources_resident<P>(
        &self,
        prepared: &RocmPreparedTensorGraph<'_>,
        inputs: &[(ValueId, &RocmTensorInput<'_>)],
        pool: PcuMemoryPoolId,
        memory: &mut P,
    ) -> Result<RocmTensorInput<'session>, RocmTensorExecutionError>
    where
        P: PcuMemoryProvider<Resource = RocmMemoryResource>,
    {
        let mut timings = NoopNodeTiming;
        let mut outputs = self.execute_prepared_outputs_with_input_sources_and_scratch(
            prepared,
            &[],
            inputs,
            pool,
            memory,
            None,
            None,
            &mut timings,
            None,
        )?;
        Ok(outputs.remove(0))
    }

    fn validate_execution_sources<I: RocmTensorInputDescriptor>(
        &self,
        prepared: &RocmPreparedTensorGraph<'_>,
        host_inputs: &[(ValueId, Tensor)],
        resource_inputs: &[(ValueId, &I)],
        pool: PcuMemoryPoolId,
        scratch: Option<&RocmTensorScratch<'_, '_, 'session>>,
        output_bank: Option<&RocmTensorOutputBank<'_, '_, 'session>>,
    ) -> Result<(), RocmTensorExecutionError> {
        require_f32_graph(&prepared.data)?;
        self.validate_execution_sources_view(&prepared.view(), host_inputs, resource_inputs, pool)?;
        if let Some(scratch) = scratch
            && (!std::ptr::eq(scratch.prepared, prepared)
                || !std::ptr::eq(scratch.session, self.session)
                || scratch.poisoned
                || scratch.pool != pool)
        {
            return Err(RocmTensorExecutionError::ScratchMismatch);
        }
        if let Some(bank) = output_bank {
            validate_output_bank(bank, prepared, self.session, pool, resource_inputs, scratch)?;
        }
        validate_prepared_storage_constraints(prepared, resource_inputs, scratch, output_bank)?;
        Ok(())
    }

    fn validate_execution_sources_view<I: RocmTensorInputDescriptor>(
        &self,
        prepared: &RocmPreparedGraphView<'_, '_>,
        host_inputs: &[(ValueId, Tensor)],
        resource_inputs: &[(ValueId, &I)],
        pool: PcuMemoryPoolId,
    ) -> Result<(), RocmTensorExecutionError> {
        validate_graph_input_sources(prepared, host_inputs, resource_inputs, self.session, pool)?;
        if prepared.requires_blas && !self.rocblas().map_err(RocmTensorError::from)?.is_usable() {
            return Err(RocmTensorExecutionError::Unsupported {
                value: prepared.output,
                reason: TensorUnsupportedReason::Other(
                    "rocBLAS completion is uncertain; selected handle cannot run more work".into(),
                ),
            });
        }
        Ok(())
    }

    fn validate_resident_proof(
        &self,
        proof: &feedback_runtime::execution::ResidentFeedbackProof<'_, '_, '_>,
        prepared: &RocmPreparedTensorGraph<'_>,
        pool: PcuMemoryPoolId,
        scratch: Option<&RocmTensorScratch<'_, '_, 'session>>,
        output_bank: Option<&RocmTensorOutputBank<'_, '_, 'session>>,
    ) -> Result<(), RocmTensorExecutionError> {
        let Some(scratch) = scratch else {
            return Err(RocmTensorExecutionError::ScratchMismatch);
        };
        let Some(bank) = output_bank else {
            return Err(RocmTensorExecutionError::OutputResourceMismatch);
        };
        if !proof.matches(prepared, self.session, pool)
            || scratch.poisoned
            || !std::ptr::eq(scratch.prepared, prepared)
            || !std::ptr::eq(scratch.session, self.session)
            || scratch.pool != pool
        {
            return Err(RocmTensorExecutionError::ScratchMismatch);
        }
        if bank.poisoned
            || !std::ptr::eq(bank.prepared, prepared)
            || !std::ptr::eq(bank.session, self.session)
            || bank.pool != pool
            || bank.outputs.len() != prepared.outputs.len()
        {
            return Err(RocmTensorExecutionError::OutputResourceMismatch);
        }
        if prepared.data.requires_blas
            && !self.rocblas().map_err(RocmTensorError::from)?.is_usable()
        {
            return Err(RocmTensorExecutionError::Unsupported {
                value: prepared.output,
                reason: TensorUnsupportedReason::Other(
                    "rocBLAS completion is uncertain; selected handle cannot run more work".into(),
                ),
            });
        }
        Ok(())
    }

    // Per-node scratch and completion borrows stay local to this scheduler loop; bundling them
    // would extend mutable lifetimes across iterations and obscure the device-ordering boundary.
    #[allow(
        clippy::cognitive_complexity,
        clippy::too_many_arguments,
        clippy::too_many_lines
    )]
    fn execute_prepared_outputs_with_input_sources_and_scratch<I: RocmTensorInputDescriptor, P>(
        &self,
        prepared: &RocmPreparedTensorGraph<'_>,
        host_inputs: &[(ValueId, Tensor)],
        resource_inputs: &[(ValueId, &I)],
        pool: PcuMemoryPoolId,
        memory: &mut P,
        scratch: Option<&mut RocmTensorScratch<'_, '_, 'session>>,
        output_bank: Option<&RocmTensorOutputBank<'_, '_, 'session>>,
        timings: &mut impl NodeTimingSink,
        batch: Option<&mut HipCompletionBatch>,
    ) -> Result<TensorExecutionOutputs<'session>, RocmTensorExecutionError>
    where
        P: PcuMemoryProvider<Resource = RocmMemoryResource>,
    {
        self.execute_prepared_outputs_with_input_sources_and_scratch_proven(
            prepared,
            host_inputs,
            resource_inputs,
            pool,
            memory,
            scratch,
            output_bank,
            timings,
            batch,
            None,
        )
    }

    #[allow(
        clippy::cognitive_complexity,
        clippy::too_many_arguments,
        clippy::too_many_lines
    )]
    fn execute_prepared_outputs_with_input_sources_and_scratch_proven<
        I: RocmTensorInputDescriptor,
        P,
    >(
        &self,
        prepared: &RocmPreparedTensorGraph<'_>,
        host_inputs: &[(ValueId, Tensor)],
        resource_inputs: &[(ValueId, &I)],
        pool: PcuMemoryPoolId,
        memory: &mut P,
        scratch: Option<&mut RocmTensorScratch<'_, '_, 'session>>,
        output_bank: Option<&RocmTensorOutputBank<'_, '_, 'session>>,
        timings: &mut impl NodeTimingSink,
        batch: Option<&mut HipCompletionBatch>,
        proof: Option<&feedback_runtime::execution::ResidentFeedbackProof<'_, '_, '_>>,
    ) -> Result<TensorExecutionOutputs<'session>, RocmTensorExecutionError>
    where
        P: PcuMemoryProvider<Resource = RocmMemoryResource>,
    {
        #[cfg(feature = "insights")]
        let validation_mark = timings.begin_scheduler_validation();
        let validation = proof.map_or_else(
            || {
                self.validate_execution_sources(
                    prepared,
                    host_inputs,
                    resource_inputs,
                    pool,
                    scratch.as_deref(),
                    output_bank,
                )
            },
            |proof| {
                self.validate_resident_proof(proof, prepared, pool, scratch.as_deref(), output_bank)
            },
        );
        #[cfg(feature = "insights")]
        timings.finish_scheduler_validation(validation_mark);
        validation?;
        let view = prepared.view();
        let mut scratch_view = scratch.map(|scratch| RocmExecutionScratch {
            statuses: None,
            resources: &scratch.resources,
            mse_squared: scratch.mse_squared.as_ref(),
            outputs: &scratch.prepared.outputs,
            node_values: &scratch.prepared.node_values,
        });
        self.execute_prepared_schedule(
            &view,
            host_inputs,
            resource_inputs,
            pool,
            memory,
            scratch_view.as_mut(),
            output_bank,
            None,
            timings,
            batch,
            proof,
        )
    }

    #[allow(
        clippy::cognitive_complexity,
        clippy::too_many_arguments,
        clippy::too_many_lines
    )]
    fn execute_prepared_schedule<I: RocmTensorInputDescriptor, P>(
        &self,
        prepared: &RocmPreparedGraphView<'_, '_>,
        host_inputs: &[(ValueId, Tensor)],
        resource_inputs: &[(ValueId, &I)],
        pool: PcuMemoryPoolId,
        memory: &mut P,
        mut scratch: Option<&mut RocmExecutionScratch<'_>>,
        output_bank: Option<&RocmTensorOutputBank<'_, '_, 'session>>,
        fresh_outputs: Option<&[(ValueId, FreshTensorOutput<'_>)]>,
        timings: &mut impl NodeTimingSink,
        mut batch: Option<&mut HipCompletionBatch>,
        proof: Option<&feedback_runtime::execution::ResidentFeedbackProof<'_, '_, '_>>,
    ) -> Result<TensorExecutionOutputs<'session>, RocmTensorExecutionError>
    where
        P: PcuMemoryProvider<Resource = RocmMemoryResource>,
    {
        let graph = prepared.graph;
        let plan = prepared;
        let outputs = &plan.outputs;
        let node_count = plan.node_values.len();
        let mut resources = execution_resource_slots(node_count);
        let mut remaining_uses = TensorExecutionUseCounts::from_slice(&plan.use_counts);

        for index in 0..node_count {
            let node = plan.node(index)?;
            if plan.suppressed_adds.contains(&node.value) {
                continue;
            }
            let timing_mark = timings.begin();
            let mut sgemm_host = None;
            let mut elementwise_host = None;
            match node.op {
                OpDescriptor::Input => {
                    if let Some(output) = tensor_output_view(output_bank, fresh_outputs, node.value)
                    {
                        if let Some(batch) = batch.as_deref_mut() {
                            tensor_flush_batch!(batch, timings, false)?;
                        }
                        let mut destination = output.resource.clone_for_tensor_input();
                        if let Some((_, input)) =
                            resource_inputs.iter().find(|(id, _)| *id == node.value)
                        {
                            let (element_size, _) = scalar_layout(output.scalar_type)?;
                            let bytes = byte_len_for_size(output.shape, element_size)?;
                            let binding = proof.map(|_| destination.clone_for_tensor_input());
                            let copied = memory.copy_resource(
                                &mut destination,
                                input.resource(),
                                bytes as u64,
                            );
                            if binding
                                .as_ref()
                                .is_some_and(|binding| !destination.same_binding(binding))
                            {
                                return Err(RocmTensorExecutionError::OutputResourceMismatch);
                            }
                            copied?;
                        } else {
                            let (_, tensor) = host_inputs
                                .iter()
                                .find(|(id, _)| *id == node.value)
                                .ok_or(TensorError::MissingInput(node.value))?;
                            let binding = proof.map(|_| destination.clone_for_tensor_input());
                            let transferred = transfer_tensor(memory, &mut destination, tensor);
                            if binding
                                .as_ref()
                                .is_some_and(|binding| !destination.same_binding(binding))
                            {
                                return Err(RocmTensorExecutionError::OutputResourceMismatch);
                            }
                            transferred?;
                        }
                        resources[index] = Some(destination);
                        timings.finish(timing_mark, &node);
                        continue;
                    }
                    if let Some((_, input)) =
                        resource_inputs.iter().find(|(id, _)| *id == node.value)
                    {
                        resources[index] = Some(input.resource().clone_for_tensor_input());
                        timings.finish(timing_mark, &node);
                        continue;
                    }
                    let (_, tensor) = host_inputs
                        .iter()
                        .find(|(id, _)| *id == node.value)
                        .ok_or(TensorError::MissingInput(node.value))?;
                    if let Some(batch) = batch.as_deref_mut() {
                        tensor_flush_batch!(batch, timings, false)?;
                    }
                    resources[index] = Some(upload_tensor(memory, pool, tensor)?);
                }
                OpDescriptor::Constant(tensor) => {
                    if node.scalar_type == fusion_pcu::PcuScalarType::F64
                        || is_low_float_type(node.scalar_type)
                        || is_raw_float_type(node.scalar_type)
                        || is_checked_integer_scalar(node.scalar_type)
                    {
                        if let Some(batch) = batch.as_deref_mut() {
                            tensor_flush_batch!(batch, timings, false)?;
                        }
                        resources[index] = Some(
                            if let Some(scratch) = scratch
                                .as_deref_mut()
                                .filter(|_| !outputs.contains(&node.value))
                            {
                                scratch.lease(index)?
                            } else {
                                literal::upload_dense(
                                    node,
                                    plan.physical_layout(node.value)?,
                                    tensor_output_view(output_bank, fresh_outputs, node.value)
                                        .map(|output| output.resource),
                                    pool,
                                    memory,
                                )?
                            },
                        );
                    } else {
                        let tensor = tensor.as_typed::<f32>().map_err(|_| {
                            RocmTensorExecutionError::Unsupported {
                                value: node.value,
                                reason: TensorUnsupportedReason::ElementType,
                            }
                        })?;
                        resources[index] = Some(
                            if let Some(output) =
                                tensor_output_view(output_bank, fresh_outputs, node.value)
                            {
                                if let Some(batch) = batch.as_deref_mut() {
                                    tensor_flush_batch!(batch, timings, false)?;
                                }
                                let mut resource = output.resource.clone_for_tensor_input();
                                let binding = proof.map(|_| resource.clone_for_tensor_input());
                                let transferred = transfer_tensor(memory, &mut resource, tensor);
                                if binding
                                    .as_ref()
                                    .is_some_and(|binding| !resource.same_binding(binding))
                                {
                                    return Err(RocmTensorExecutionError::OutputResourceMismatch);
                                }
                                transferred?;
                                resource
                            } else if outputs.contains(&node.value) {
                                if let Some(batch) = batch.as_deref_mut() {
                                    tensor_flush_batch!(batch, timings, false)?;
                                }
                                upload_tensor(memory, pool, tensor)?
                            } else if let Some(scratch) = scratch.as_deref_mut() {
                                scratch.lease(index)?
                            } else {
                                if let Some(batch) = batch.as_deref_mut() {
                                    tensor_flush_batch!(batch, timings, false)?;
                                }
                                upload_tensor(memory, pool, tensor)?
                            },
                        );
                    }
                }
                OpDescriptor::Uniform { value } => {
                    if node.scalar_type == fusion_pcu::PcuScalarType::F64
                        || is_low_float_type(node.scalar_type)
                        || is_raw_float_type(node.scalar_type)
                        || is_checked_integer_scalar(node.scalar_type)
                    {
                        if let Some(batch) = batch.as_deref_mut() {
                            tensor_flush_batch!(batch, timings, false)?;
                        }
                        resources[index] = Some(
                            if let Some(scratch) = scratch
                                .as_deref_mut()
                                .filter(|_| !outputs.contains(&node.value))
                            {
                                scratch.lease(index)?
                            } else {
                                literal::upload_dense(
                                    node,
                                    plan.physical_layout(node.value)?,
                                    tensor_output_view(output_bank, fresh_outputs, node.value)
                                        .map(|output| output.resource),
                                    pool,
                                    memory,
                                )?
                            },
                        );
                    } else {
                        // Scratch already owns the immutable cold initialization, including dense
                        // fallbacks. Materialize host data only for actual uploads/output copies;
                        // merely leasing retained storage must not allocate another dense tensor.
                        // Raw nonfinite encodings remain transport data until arithmetic consumes them.
                        let materialize = || {
                            prepared.physical_layout(node.value)?.uniform_tensor(
                                node.shape,
                                value.as_typed::<f32>().map_err(|_| {
                                    RocmTensorExecutionError::Unsupported {
                                        value: node.value,
                                        reason: TensorUnsupportedReason::ElementType,
                                    }
                                })?,
                            )
                        };
                        resources[index] = Some(
                            if let Some(output) =
                                tensor_output_view(output_bank, fresh_outputs, node.value)
                            {
                                let tensor = materialize()?;
                                if let Some(batch) = batch.as_deref_mut() {
                                    tensor_flush_batch!(batch, timings, false)?;
                                }
                                let mut resource = output.resource.clone_for_tensor_input();
                                let binding = proof.map(|_| resource.clone_for_tensor_input());
                                let transferred = transfer_tensor(memory, &mut resource, &tensor);
                                if binding
                                    .as_ref()
                                    .is_some_and(|binding| !resource.same_binding(binding))
                                {
                                    return Err(RocmTensorExecutionError::OutputResourceMismatch);
                                }
                                transferred?;
                                resource
                            } else if outputs.contains(&node.value) {
                                let tensor = materialize()?;
                                if let Some(batch) = batch.as_deref_mut() {
                                    tensor_flush_batch!(batch, timings, false)?;
                                }
                                upload_tensor(memory, pool, &tensor)?
                            } else if let Some(scratch) = scratch.as_deref_mut() {
                                scratch.lease(index)?
                            } else {
                                let tensor = materialize()?;
                                if let Some(batch) = batch.as_deref_mut() {
                                    tensor_flush_batch!(batch, timings, false)?;
                                }
                                upload_tensor(memory, pool, &tensor)?
                            },
                        );
                    }
                }
                OpDescriptor::MatMul {
                    left,
                    right,
                    transpose_left,
                    transpose_right,
                } => {
                    let operands = plan
                        .matmul_operands
                        .get(index)
                        .copied()
                        .flatten()
                        .ok_or(RocmTensorExecutionError::InvalidPlan(node.value))?;
                    let result = execution_resource(
                        memory,
                        pool,
                        node.shape,
                        node.scalar_type,
                        index,
                        node.value,
                        &mut scratch,
                        output_bank,
                        fresh_outputs,
                    )?;
                    let left_resource = resources[operands.left_index]
                        .as_ref()
                        .ok_or(RocmTensorExecutionError::MissingResource(left))?;
                    let right_resource = resources[operands.right_index]
                        .as_ref()
                        .ok_or(RocmTensorExecutionError::MissingResource(right))?;
                    if let Some(spec) = operands.strict {
                        if let Some(batch) = batch.as_deref_mut() {
                            tensor_flush_batch!(batch, timings, false)?;
                        }
                        self.execute_strict_matmul(
                            spec,
                            node.value,
                            left_resource,
                            right_resource,
                            &result,
                            scratch
                                .as_deref_mut()
                                .and_then(|scratch| scratch.status(index)),
                        )?;
                    } else if let Some(batch) = batch.as_deref_mut() {
                        self.execute_matmul_row_major_flags(
                            left_resource,
                            right_resource,
                            &result,
                            operands.left_shape,
                            operands.right_shape,
                            transpose_left,
                            transpose_right,
                            node.scalar_type,
                            None,
                            Some(batch),
                        )?;
                    } else if node.scalar_type == fusion_pcu::PcuScalarType::F32 {
                        sgemm_host = timings.with_sgemm_timing(|sgemm_timing| {
                            self.execute_matmul_row_major_flags(
                                left_resource,
                                right_resource,
                                &result,
                                operands.left_shape,
                                operands.right_shape,
                                transpose_left,
                                transpose_right,
                                node.scalar_type,
                                sgemm_timing,
                                None,
                            )
                            .map_err(Into::into)
                        })?;
                    } else {
                        self.execute_matmul_row_major_flags(
                            left_resource,
                            right_resource,
                            &result,
                            operands.left_shape,
                            operands.right_shape,
                            transpose_left,
                            transpose_right,
                            node.scalar_type,
                            None,
                            None,
                        )?;
                    }
                    resources[index] = Some(result);
                    release_after_read(&mut resources, &mut remaining_uses, operands.left_index)?;
                    release_after_read(&mut resources, &mut remaining_uses, operands.right_index)?;
                }
                OpDescriptor::Add { left, right }
                | OpDescriptor::Sub { left, right }
                | OpDescriptor::Mul { left, right }
                | OpDescriptor::Div { left, right } => {
                    if let Some(group) = plan.bounded_mul_by_output.get(&node.value) {
                        let output = execution_resource(
                            memory,
                            pool,
                            node.shape,
                            node.scalar_type,
                            index,
                            node.value,
                            &mut scratch,
                            output_bank,
                            fresh_outputs,
                        )?;
                        let mut leaf_resources = SmallVec::<[&RocmMemoryResource; 4]>::new();
                        let mut scalar_mask = 0_u8;
                        for (leaf_index, &leaf) in group.leaves.iter().enumerate() {
                            let leaf_index_in_plan = plan
                                .index_of(leaf)
                                .ok_or(RocmTensorExecutionError::InvalidPlan(leaf))?;
                            let resource = resources[leaf_index_in_plan]
                                .as_ref()
                                .ok_or(RocmTensorExecutionError::MissingResource(leaf))?;
                            if plan.physical_layout(leaf)?.representation
                                == RocmPhysicalRepresentation::UniformScalar
                            {
                                scalar_mask |= 1_u8 << leaf_index;
                            }
                            leaf_resources.push(resource);
                        }
                        self.execute_bounded_mul(
                            group,
                            &leaf_resources,
                            &output,
                            scalar_mask,
                            batch.as_deref_mut(),
                        )?;
                        drop(leaf_resources);
                        resources[index] = Some(output);
                        release_bounded_mul_leaves(
                            plan,
                            group,
                            &mut resources,
                            &mut remaining_uses,
                        )?;
                    } else if let Some(group) = plan
                        .bounded_pointwise_by_output
                        .get(&node.value)
                        .filter(|group| group.epilogue == TensorPointwiseEpilogue::Identity)
                    {
                        let output = execution_resource(
                            memory,
                            pool,
                            node.shape,
                            node.scalar_type,
                            index,
                            node.value,
                            &mut scratch,
                            output_bank,
                            fresh_outputs,
                        )?;
                        let mut leaf_resources = SmallVec::<[&RocmMemoryResource; 4]>::new();
                        let mut scalar_mask = 0_u8;
                        for (leaf_index, &leaf) in group.leaves.iter().enumerate() {
                            let leaf_plan_index = plan
                                .index_of(leaf)
                                .ok_or(RocmTensorExecutionError::InvalidPlan(leaf))?;
                            let resource = resources[leaf_plan_index]
                                .as_ref()
                                .ok_or(RocmTensorExecutionError::MissingResource(leaf))?;
                            if plan.physical_layout(leaf)?.representation
                                == RocmPhysicalRepresentation::UniformScalar
                            {
                                scalar_mask |= 1_u8 << leaf_index;
                            }
                            leaf_resources.push(resource);
                        }
                        self.execute_bounded_pointwise(
                            group,
                            &leaf_resources,
                            &output,
                            scalar_mask,
                            batch.as_deref_mut(),
                        )?;
                        drop(leaf_resources);
                        resources[index] = Some(output);
                        release_bounded_pointwise_leaves(
                            plan,
                            group,
                            &mut resources,
                            &mut remaining_uses,
                        )?;
                    } else {
                        let left_index = plan
                            .index_of(left)
                            .ok_or(RocmTensorExecutionError::InvalidPlan(left))?;
                        let right_index = plan
                            .index_of(right)
                            .ok_or(RocmTensorExecutionError::InvalidPlan(right))?;
                        let output = execution_resource(
                            memory,
                            pool,
                            node.shape,
                            node.scalar_type,
                            index,
                            node.value,
                            &mut scratch,
                            output_bank,
                            fresh_outputs,
                        )?;
                        let mut elementwise_timing = timings.begin_elementwise();
                        let checked_arithmetic = is_checked_integer_scalar(node.scalar_type)
                            || is_checked_float_binary_node(node);
                        if checked_arithmetic && let Some(batch) = batch.as_deref_mut() {
                            tensor_flush_batch!(batch, timings, false)?;
                        }
                        self.execute_elementwise(
                            plan.fixed_dispatch(index)?,
                            ElementwiseOperands {
                                left: resources[left_index]
                                    .as_ref()
                                    .ok_or(RocmTensorExecutionError::MissingResource(left))?,
                                right: Some(
                                    resources[right_index]
                                        .as_ref()
                                        .ok_or(RocmTensorExecutionError::MissingResource(right))?,
                                ),
                                output: &output,
                            },
                            if checked_arithmetic {
                                None
                            } else {
                                batch.as_deref_mut()
                            },
                            &mut elementwise_timing,
                            scratch
                                .as_deref_mut()
                                .and_then(|scratch| scratch.status(index)),
                        )?;
                        elementwise_host = timings.finish_elementwise(elementwise_timing);
                        resources[index] = Some(output);
                        release_after_read(&mut resources, &mut remaining_uses, left_index)?;
                        release_after_read(&mut resources, &mut remaining_uses, right_index)?;
                    }
                }
                OpDescriptor::Relu { input } => {
                    let output = execution_resource(
                        memory,
                        pool,
                        node.shape,
                        node.scalar_type,
                        index,
                        node.value,
                        &mut scratch,
                        output_bank,
                        fresh_outputs,
                    )?;
                    if let Some(group) = plan.bounded_pointwise_by_output.get(&node.value) {
                        let mut leaf_resources = SmallVec::<[&RocmMemoryResource; 4]>::new();
                        let mut scalar_mask = 0_u8;
                        for (leaf_index, &leaf) in group.leaves.iter().enumerate() {
                            let leaf_plan_index = plan
                                .index_of(leaf)
                                .ok_or(RocmTensorExecutionError::InvalidPlan(leaf))?;
                            let resource = resources[leaf_plan_index]
                                .as_ref()
                                .ok_or(RocmTensorExecutionError::MissingResource(leaf))?;
                            if plan.physical_layout(leaf)?.representation
                                == RocmPhysicalRepresentation::UniformScalar
                            {
                                scalar_mask |= 1_u8 << leaf_index;
                            }
                            leaf_resources.push(resource);
                        }
                        self.execute_bounded_pointwise(
                            group,
                            &leaf_resources,
                            &output,
                            scalar_mask,
                            batch.as_deref_mut(),
                        )?;
                        drop(leaf_resources);
                        resources[index] = Some(output);
                        release_bounded_pointwise_leaves(
                            plan,
                            group,
                            &mut resources,
                            &mut remaining_uses,
                        )?;
                    } else if let Some(&(left, right)) = plan.fused_add_by_relu.get(&node.value) {
                        let left_index = plan
                            .index_of(left)
                            .ok_or(RocmTensorExecutionError::InvalidPlan(left))?;
                        let right_index = plan
                            .index_of(right)
                            .ok_or(RocmTensorExecutionError::InvalidPlan(right))?;
                        self.execute_elementwise(
                            plan.fixed_dispatch(index)?,
                            ElementwiseOperands {
                                left: resources[left_index]
                                    .as_ref()
                                    .ok_or(RocmTensorExecutionError::MissingResource(left))?,
                                right: Some(
                                    resources[right_index]
                                        .as_ref()
                                        .ok_or(RocmTensorExecutionError::MissingResource(right))?,
                                ),
                                output: &output,
                            },
                            batch.as_deref_mut(),
                            &mut NoopElementwiseTiming,
                            scratch
                                .as_deref_mut()
                                .and_then(|scratch| scratch.status(index)),
                        )?;
                        resources[index] = Some(output);
                        release_after_read(&mut resources, &mut remaining_uses, left_index)?;
                        release_after_read(&mut resources, &mut remaining_uses, right_index)?;
                    } else {
                        let input_index = plan
                            .index_of(input)
                            .ok_or(RocmTensorExecutionError::InvalidPlan(input))?;
                        self.execute_relu(
                            plan.fixed_dispatch(index)?,
                            resources[input_index]
                                .as_ref()
                                .ok_or(RocmTensorExecutionError::MissingResource(input))?,
                            &output,
                            batch.as_deref_mut(),
                            scratch
                                .as_deref_mut()
                                .and_then(|scratch| scratch.status(index)),
                        )?;
                        resources[index] = Some(output);
                        release_after_read(&mut resources, &mut remaining_uses, input_index)?;
                    }
                }
                OpDescriptor::ReluBackward { input, upstream } => {
                    let input_index = plan
                        .index_of(input)
                        .ok_or(RocmTensorExecutionError::InvalidPlan(input))?;
                    let upstream_index = plan
                        .index_of(upstream)
                        .ok_or(RocmTensorExecutionError::InvalidPlan(upstream))?;
                    let output = execution_resource(
                        memory,
                        pool,
                        node.shape,
                        node.scalar_type,
                        index,
                        node.value,
                        &mut scratch,
                        output_bank,
                        fresh_outputs,
                    )?;
                    if let Some(batch) = batch.as_deref_mut() {
                        tensor_flush_batch!(batch, timings, false)?;
                    }
                    self.execute_admitted_relu_backward(
                        node,
                        resources[input_index]
                            .as_ref()
                            .ok_or(RocmTensorExecutionError::MissingResource(input))?,
                        resources[upstream_index]
                            .as_ref()
                            .ok_or(RocmTensorExecutionError::MissingResource(upstream))?,
                        &output,
                        scratch
                            .as_deref_mut()
                            .and_then(|scratch| scratch.status(index)),
                    )?;
                    resources[index] = Some(output);
                    release_after_read(&mut resources, &mut remaining_uses, input_index)?;
                    release_after_read(&mut resources, &mut remaining_uses, upstream_index)?;
                }
                OpDescriptor::SgdUpdate {
                    weights,
                    gradient,
                    learning_rate,
                } => {
                    let weights_index = plan
                        .index_of(weights)
                        .ok_or(RocmTensorExecutionError::InvalidPlan(weights))?;
                    let gradient_index = plan
                        .index_of(gradient)
                        .ok_or(RocmTensorExecutionError::InvalidPlan(gradient))?;
                    let output = execution_resource(
                        memory,
                        pool,
                        node.shape,
                        node.scalar_type,
                        index,
                        node.value,
                        &mut scratch,
                        output_bank,
                        fresh_outputs,
                    )?;
                    if let Some(spec) = plan.strict_sgd[index] {
                        if let Some(batch) = batch.as_deref_mut() {
                            tensor_flush_batch!(batch, timings, false)?;
                        }
                        self.execute_strict_sgd(
                            spec,
                            node.value,
                            resources[weights_index]
                                .as_ref()
                                .ok_or(RocmTensorExecutionError::MissingResource(weights))?,
                            resources[gradient_index]
                                .as_ref()
                                .ok_or(RocmTensorExecutionError::MissingResource(gradient))?,
                            &output,
                            scratch
                                .as_deref_mut()
                                .and_then(|scratch| scratch.status(index)),
                        )?;
                    } else {
                        self.execute_sgd_update(
                            node.shape,
                            resources[weights_index]
                                .as_ref()
                                .ok_or(RocmTensorExecutionError::MissingResource(weights))?,
                            resources[gradient_index]
                                .as_ref()
                                .ok_or(RocmTensorExecutionError::MissingResource(gradient))?,
                            SgdUpdateMode {
                                scalar: node.scalar_type,
                                learning_rate,
                                contracted: node.numerical_options.precision
                                    == fusion_pcu::PcuPrecisionPolicy::BackendOptimized
                                    || plan
                                        .rewrites
                                        .iter()
                                        .any(|rewrite| rewrite.output == node.value),
                            },
                            &output,
                            batch.as_deref_mut(),
                        )?;
                    }
                    resources[index] = Some(output);
                    release_after_read(&mut resources, &mut remaining_uses, weights_index)?;
                    release_after_read(&mut resources, &mut remaining_uses, gradient_index)?;
                }
                OpDescriptor::MeanSquaredError { prediction, target } => {
                    if let Some(batch) = batch.as_deref_mut() {
                        tensor_flush_batch!(batch, timings, false)?;
                    }
                    let prediction_index = plan
                        .index_of(prediction)
                        .ok_or(RocmTensorExecutionError::InvalidPlan(prediction))?;
                    let target_index = plan
                        .index_of(target)
                        .ok_or(RocmTensorExecutionError::InvalidPlan(target))?;
                    if node.numerical_mode == Some(fusion_pcu::PcuNumericalMode::Strict) {
                        let output = execution_resource(
                            memory,
                            pool,
                            node.shape,
                            node.scalar_type,
                            index,
                            node.value,
                            &mut scratch,
                            output_bank,
                            fresh_outputs,
                        )?;
                        self.execute_strict_mse(
                            graph,
                            node,
                            resources[prediction_index]
                                .as_ref()
                                .ok_or(RocmTensorExecutionError::MissingResource(prediction))?,
                            resources[target_index]
                                .as_ref()
                                .ok_or(RocmTensorExecutionError::MissingResource(target))?,
                            &output,
                            scratch
                                .as_deref_mut()
                                .and_then(|scratch| scratch.status(index)),
                        )?;
                        resources[index] = Some(output);
                        release_after_read(&mut resources, &mut remaining_uses, prediction_index)?;
                        release_after_read(&mut resources, &mut remaining_uses, target_index)?;
                    } else {
                        let shape = graph.shape(prediction)?;
                        let count = shape
                            .iter()
                            .try_fold(1usize, |n, d| n.checked_mul(*d))
                            .ok_or(RocmTensorExecutionError::SizeOverflow)?;
                        if count == 0 || count > i32::MAX as usize {
                            return Err(RocmTensorExecutionError::Unsupported {
                                value: node.value,
                                reason: TensorUnsupportedReason::Shape,
                            });
                        }
                        let squared = if let Some(scratch) = scratch.as_ref() {
                            let squared = scratch
                                .mse_squared
                                .as_ref()
                                .ok_or(RocmTensorExecutionError::ScratchMismatch)?
                                .clone_for_tensor_input();
                            if !mse_scratch_length_fits_scalar(
                                count,
                                node.scalar_type,
                                squared.device_buffer().len(),
                            )? {
                                return Err(RocmTensorExecutionError::ScratchMismatch);
                            }
                            squared
                        } else {
                            allocate_tensor_for_size(
                                memory,
                                pool,
                                shape,
                                usize::from(node.scalar_type.bit_width() / 8),
                                usize::from(node.scalar_type.bit_width() / 8),
                            )?
                        };
                        self.execute_native_mse_squared(
                            u32::try_from(count)
                                .map_err(|_| RocmTensorExecutionError::SizeOverflow)?,
                            node.scalar_type,
                            resources[prediction_index]
                                .as_ref()
                                .ok_or(RocmTensorExecutionError::MissingResource(prediction))?,
                            resources[target_index]
                                .as_ref()
                                .ok_or(RocmTensorExecutionError::MissingResource(target))?,
                            &squared,
                        )?;
                        let output = execution_resource(
                            memory,
                            pool,
                            node.shape,
                            node.scalar_type,
                            index,
                            node.value,
                            &mut scratch,
                            output_bank,
                            fresh_outputs,
                        )?;
                        let blas = self.rocblas().map_err(RocmTensorError::from)?;
                        match node.scalar_type {
                            fusion_pcu::PcuScalarType::F32 => blas.sasum_scaled(
                                count,
                                squared.device_buffer(),
                                1,
                                mse_scale(count)?,
                                output.device_buffer(),
                            ),
                            fusion_pcu::PcuScalarType::F64 => blas.dasum_scaled(
                                count,
                                squared.device_buffer(),
                                1,
                                mse_scale_f64(count)?,
                                output.device_buffer(),
                            ),
                            _ => {
                                return Err(RocmTensorExecutionError::UnsupportedScalarType(
                                    node.scalar_type,
                                ));
                            }
                        }
                        .map_err(RocmTensorError::from)?;
                        resources[index] = Some(output);
                        release_after_read(&mut resources, &mut remaining_uses, prediction_index)?;
                        release_after_read(&mut resources, &mut remaining_uses, target_index)?;
                    }
                }
            }
            if plan.bounded_pointwise_by_output.contains_key(&node.value) {
                timings.finish_fused_add_sub(
                    timing_mark,
                    &node,
                    plan.bounded_pointwise_by_output[&node.value].epilogue,
                );
            } else if plan.fused_add_by_relu.contains_key(&node.value) {
                timings.finish_fused_add_relu(timing_mark, &node);
            } else if matches!(node.op, OpDescriptor::MatMul { .. }) {
                timings.finish_matmul(timing_mark, &node, sgemm_host);
            } else if matches!(
                node.op,
                OpDescriptor::Add { .. }
                    | OpDescriptor::Sub { .. }
                    | OpDescriptor::Mul { .. }
                    | OpDescriptor::Div { .. }
            ) {
                timings.finish_fixed_elementwise(timing_mark, &node, elementwise_host);
            } else {
                timings.finish(timing_mark, &node);
            }
        }

        if let Some(batch) = batch.take() {
            tensor_flush_batch!(batch, timings, true)?;
        }

        if output_bank.is_some() {
            // The bank owns the output storage. Still consume and validate each scheduler
            // result so liveness and resource checks remain identical, but don't manufacture
            // public tensor wrappers (and their owned shapes) that the caller immediately drops.
            for &output in outputs {
                let output_index = plan
                    .index_of(output)
                    .ok_or(RocmTensorExecutionError::InvalidPlan(output))?;
                let resource = resources[output_index]
                    .take()
                    .ok_or(RocmTensorExecutionError::MissingResource(output))?;
                if resource.pool() != pool
                    || !resource.belongs_to_runtime(self.session.tensor_runtime())
                    || !matches!(
                        resource.access(),
                        PcuMemoryAccess::ReadOnly | PcuMemoryAccess::ReadWrite
                    )
                {
                    return Err(RocmTensorExecutionError::OutputResourceMismatch);
                }
                drop(resource);
            }
            Ok(SmallVec::new())
        } else {
            outputs
                .iter()
                .map(|&output| {
                    let output_index = plan
                        .index_of(output)
                        .ok_or(RocmTensorExecutionError::InvalidPlan(output))?;
                    let resource = resources[output_index]
                        .take()
                        .ok_or(RocmTensorExecutionError::MissingResource(output))?;
                    if resource.pool() != pool
                        || !resource.belongs_to_runtime(self.session.tensor_runtime())
                        || !matches!(
                            resource.access(),
                            PcuMemoryAccess::ReadOnly | PcuMemoryAccess::ReadWrite
                        )
                    {
                        return Err(RocmTensorExecutionError::OutputResourceMismatch);
                    }
                    Ok(RocmTensorInput {
                        session: self.session,
                        shape: PcuOwnedShape::from_slice(graph.shape(output)?),
                        resource,
                        scalar_type: graph.node(output)?.scalar_type,
                        updateable: true,
                    })
                })
                .collect()
        }
    }
    fn execute_relu(
        &self,
        dispatch: &PreparedFixedTensorDispatch,
        input: &RocmMemoryResource,
        output: &RocmMemoryResource,
        batch: Option<&mut HipCompletionBatch>,

        status: Option<&mut owned_scratch::Status>,
    ) -> Result<(), RocmTensorExecutionError> {
        self.execute_elementwise(
            dispatch,
            ElementwiseOperands {
                left: input,
                right: None,
                output,
            },
            batch,
            &mut NoopElementwiseTiming,
            status,
        )
    }

    fn ensure_strict_matmul_cached(
        &self,
        spec: strict_matmul::StrictMatMulSpec,
    ) -> Result<TensorDispatchCacheAdmission, RocmTensorExecutionError> {
        let key = TensorDispatchCacheKey::StrictMatMul(spec);
        let mut cache = self.state().add_dispatches.borrow_mut();
        if cache.iter().any(|(cached_key, _)| *cached_key == key) {
            return Ok(TensorDispatchCacheAdmission::default());
        }
        let prepared = self
            .session
            .prepare_strict_matmul_on_stream(spec, &self.state().stream)
            .map_err(RocmTensorExecutionError::Backend)?;
        let evicted = cache.len() == ADD_DISPATCH_CACHE_CAPACITY;
        if evicted {
            cache.pop_front();
        }
        cache.push_back((key, prepared));
        Ok(TensorDispatchCacheAdmission {
            compiled: true,
            evicted,
        })
    }

    fn execute_strict_matmul(
        &self,
        spec: strict_matmul::StrictMatMulSpec,
        value: ValueId,
        left: &RocmMemoryResource,
        right: &RocmMemoryResource,
        output: &RocmMemoryResource,

        mut status: Option<&mut owned_scratch::Status>,
    ) -> Result<(), RocmTensorExecutionError> {
        self.ensure_strict_matmul_cached(spec)?;
        let key = TensorDispatchCacheKey::StrictMatMul(spec);
        let cache = self.state().add_dispatches.borrow();
        let prepared = cache
            .iter()
            .find(|(cached_key, _)| *cached_key == key)
            .map(|(_, prepared)| prepared)
            .ok_or(RocmTensorExecutionError::InvalidPlan(value))?;
        let requirements = spec.requirements();
        let bindings = [left, right, output]
            .into_iter()
            .zip(requirements)
            .map(|(resource, requirement)| {
                self.session.bind(
                    requirement.target,
                    requirement.access,
                    requirement.binding_type,
                    resource,
                )
            })
            .collect::<Result<SmallVec<[_; 3]>, _>>()
            .map_err(RocmTensorExecutionError::Backend)?;
        let mut completion = owned_scratch::submit(prepared, &bindings, status.as_deref_mut())?;
        let outcome = completion
            .wait()
            .map_err(RocmTensorExecutionError::Completion)?;
        if let Some(status) = status {
            status.observe(outcome);
        }
        match outcome {
            PcuCompletionOutcome::Succeeded => Ok(()),
            PcuCompletionOutcome::Failed => Err(RocmTensorExecutionError::FailedCompletion),
            PcuCompletionOutcome::Fault(fault) => Err(spec.fault_error(value, fault)?.into()),
        }
    }

    #[allow(clippy::too_many_lines)] // Keeps dynamic IR construction, cache admission, and binding safety together.
    fn execute_bounded_pointwise(
        &self,
        group: &TensorBoundedPointwiseFusionGroup,
        leaves: &[&RocmMemoryResource],
        output: &RocmMemoryResource,
        scalar_mask: u8,
        batch: Option<&mut HipCompletionBatch>,
    ) -> Result<(), RocmTensorExecutionError> {
        if leaves.len() != group.leaves.len() {
            return Err(RocmTensorExecutionError::SizeOverflow);
        }
        let count = group
            .shape
            .iter()
            .try_fold(1usize, |count, dimension| count.checked_mul(*dimension))
            .ok_or(RocmTensorExecutionError::SizeOverflow)?;
        let invocations = u32::try_from(count)
            .ok()
            .and_then(NonZeroU32::new)
            .ok_or(RocmTensorExecutionError::SizeOverflow)?;
        let logical_count = invocations.get();
        let topology = bounded_pointwise_topology(group);
        self.ensure_bounded_pointwise_dispatch_cached(
            group,
            logical_count,
            scalar_mask,
            &topology,
        )?;

        let mut bindings = SmallVec::<[_; 5]>::new();
        for (index, resource) in leaves.iter().enumerate() {
            bindings.push(
                self.session
                    .bind(
                        PcuBindingRef::new(
                            0,
                            u32::try_from(index)
                                .map_err(|_| RocmTensorExecutionError::SizeOverflow)?,
                        ),
                        PcuBindingAccess::ReadOnly,
                        PcuBindingType::Value(PcuValueType::f32()),
                        resource,
                    )
                    .map_err(RocmTensorExecutionError::Backend)?,
            );
        }
        bindings.push(
            self.session
                .bind(
                    PcuBindingRef::new(
                        0,
                        u32::try_from(leaves.len())
                            .map_err(|_| RocmTensorExecutionError::SizeOverflow)?,
                    ),
                    PcuBindingAccess::WriteOnly,
                    PcuBindingType::Value(PcuValueType::f32()),
                    output,
                )
                .map_err(RocmTensorExecutionError::Backend)?,
        );

        let completion = {
            let cache = self.state().add_dispatches.borrow();
            let prepared = cache
                .iter()
                .find(|(key, _)| {
                    matches!(
                        key,
                        TensorDispatchCacheKey::Pointwise {
                            invocation_count,
                            scalar_mask: cached_scalar_mask,
                            topology: cached_topology,
                        } if *invocation_count == logical_count
                            && *cached_scalar_mask == scalar_mask
                            && *cached_topology == topology
                    )
                })
                .map(|(_, prepared)| prepared)
                .ok_or(RocmTensorExecutionError::SizeOverflow)?;
            if let Some(batch) = batch {
                prepared
                    .submit_into_batch(&bindings, batch)
                    .map_err(RocmTensorExecutionError::Backend)?;
                None
            } else {
                Some(
                    prepared
                        .submit(&bindings)
                        .map_err(RocmTensorExecutionError::Backend)?,
                )
            }
        };
        if let Some(mut completion) = completion {
            match completion
                .wait()
                .map_err(RocmTensorExecutionError::Completion)?
            {
                PcuCompletionOutcome::Succeeded => Ok(()),
                PcuCompletionOutcome::Failed => Err(RocmTensorExecutionError::FailedCompletion),
                PcuCompletionOutcome::Fault(fault) => {
                    Err(RocmTensorExecutionError::ExecutionFault(fault))
                }
            }
        } else {
            Ok(())
        }
    }

    fn execute_bounded_mul(
        &self,
        group: &TensorBoundedMulFusionGroup,
        leaves: &[&RocmMemoryResource],
        output: &RocmMemoryResource,
        scalar_mask: u8,
        batch: Option<&mut HipCompletionBatch>,
    ) -> Result<(), RocmTensorExecutionError> {
        if leaves.len() != group.leaves.len() {
            return Err(RocmTensorExecutionError::SizeOverflow);
        }
        let logical_count = flattened_invocation_count(&group.shape)?;
        let topology = bounded_mul_topology(group);
        self.ensure_bounded_mul_dispatch_cached(group, logical_count, scalar_mask, &topology)?;
        let mut bindings = SmallVec::<[_; 5]>::new();
        for (index, resource) in leaves.iter().enumerate() {
            bindings.push(
                self.session
                    .bind(
                        PcuBindingRef::new(
                            0,
                            u32::try_from(index)
                                .map_err(|_| RocmTensorExecutionError::SizeOverflow)?,
                        ),
                        PcuBindingAccess::ReadOnly,
                        PcuBindingType::Value(PcuValueType::f32()),
                        resource,
                    )
                    .map_err(RocmTensorExecutionError::Backend)?,
            );
        }
        bindings.push(
            self.session
                .bind(
                    PcuBindingRef::new(
                        0,
                        u32::try_from(leaves.len())
                            .map_err(|_| RocmTensorExecutionError::SizeOverflow)?,
                    ),
                    PcuBindingAccess::WriteOnly,
                    PcuBindingType::Value(PcuValueType::f32()),
                    output,
                )
                .map_err(RocmTensorExecutionError::Backend)?,
        );
        let completion = {
            let cache = self.state().add_dispatches.borrow();
            let prepared = cache.iter().find(|(key, _)| matches!(key,
                TensorDispatchCacheKey::Pointwise { invocation_count, scalar_mask: cached_mask, topology: cached_topology }
                    if *invocation_count == logical_count && *cached_mask == scalar_mask && *cached_topology == topology))
                .map(|(_, prepared)| prepared).ok_or(RocmTensorExecutionError::SizeOverflow)?;
            if let Some(batch) = batch {
                prepared
                    .submit_into_batch(&bindings, batch)
                    .map_err(RocmTensorExecutionError::Backend)?;
                None
            } else {
                Some(
                    prepared
                        .submit(&bindings)
                        .map_err(RocmTensorExecutionError::Backend)?,
                )
            }
        };
        if let Some(mut completion) = completion {
            match completion
                .wait()
                .map_err(RocmTensorExecutionError::Completion)?
            {
                PcuCompletionOutcome::Succeeded => Ok(()),
                PcuCompletionOutcome::Failed => Err(RocmTensorExecutionError::FailedCompletion),
                PcuCompletionOutcome::Fault(fault) => {
                    Err(RocmTensorExecutionError::ExecutionFault(fault))
                }
            }
        } else {
            Ok(())
        }
    }

    #[allow(clippy::too_many_lines)] // Keeps the cache, owned bindings, and completion lifetime explicit.
    fn execute_elementwise<T: ElementwiseTimingSink>(
        &self,
        dispatch: &PreparedFixedTensorDispatch,
        operands: ElementwiseOperands<'_>,
        batch: Option<&mut HipCompletionBatch>,
        timing: &mut T,
        mut status: Option<&mut owned_scratch::Status>,
    ) -> Result<(), RocmTensorExecutionError> {
        let ElementwiseOperands {
            left,
            right,
            output,
        } = operands;
        debug_assert_eq!(
            dispatch.value_type.scalar_type(),
            dispatch.scalar_type.scalar_type()
        );
        if right.is_some() != dispatch.right_binding.is_some() {
            return Err(RocmTensorExecutionError::InvalidPlan(dispatch.value));
        }
        let setup_mark = timing.begin(ElementwisePhase::CacheAndBind);
        self.ensure_dispatch_cached(
            dispatch.cache_key.clone(),
            dispatch.kernel,
            dispatch.invocation_shape,
            false,
        )?;
        let left_binding = self
            .session
            .bind(
                dispatch.left_binding,
                PcuBindingAccess::ReadOnly,
                PcuBindingType::Value(dispatch.value_type),
                left,
            )
            .map_err(RocmTensorExecutionError::Backend)?;
        let mut bindings = SmallVec::<[_; 3]>::new();
        bindings.push(left_binding);
        if let Some(right) = right {
            let right_binding = self
                .session
                .bind(
                    dispatch
                        .right_binding
                        .ok_or(RocmTensorExecutionError::SizeOverflow)?,
                    PcuBindingAccess::ReadOnly,
                    PcuBindingType::Value(dispatch.value_type),
                    right,
                )
                .map_err(RocmTensorExecutionError::Backend)?;
            bindings.push(right_binding);
        }
        let output_binding = self
            .session
            .bind(
                dispatch.output_binding,
                PcuBindingAccess::WriteOnly,
                PcuBindingType::Value(dispatch.value_type),
                output,
            )
            .map_err(RocmTensorExecutionError::Backend)?;
        bindings.push(output_binding);
        timing.finish(ElementwisePhase::CacheAndBind, setup_mark);
        let submit_mark = timing.begin(ElementwisePhase::Submit);
        let dispatch_result = {
            let cache = self.state().add_dispatches.borrow();
            let prepared = cache
                .iter()
                .find(|(key, _)| *key == dispatch.cache_key)
                .map(|(_, prepared)| prepared)
                .ok_or(RocmTensorExecutionError::SizeOverflow)?;
            if let Some(batch) = batch {
                prepared
                    .submit_into_batch(&bindings, batch)
                    .map_err(RocmTensorExecutionError::Backend)?;
                None
            } else {
                Some(owned_scratch::submit(
                    prepared,
                    &bindings,
                    status.as_deref_mut(),
                )?)
            }
        };
        timing.finish(ElementwisePhase::Submit, submit_mark);
        if let Some(mut completion) = dispatch_result {
            let wait_mark = timing.begin(ElementwisePhase::Wait);
            let outcome = completion
                .wait()
                .map_err(RocmTensorExecutionError::Completion)?;
            timing.finish(ElementwisePhase::Wait, wait_mark);
            if let Some(status) = status {
                status.observe(outcome);
            }
            match outcome {
                PcuCompletionOutcome::Succeeded => Ok(()),
                PcuCompletionOutcome::Failed => Err(RocmTensorExecutionError::FailedCompletion),
                PcuCompletionOutcome::Fault(fault) => {
                    Err(RocmTensorExecutionError::ExecutionFault(fault))
                }
            }
        } else {
            Ok(())
        }
    }

    #[allow(clippy::many_single_char_names)] // BLAS' A/B/C and m/n/k notation is standardized.
    fn execute_matmul_row_major(
        &self,
        a: &RocmMemoryResource,
        b: &RocmMemoryResource,
        c: &RocmMemoryResource,
        rows: usize,
        inner: usize,
        columns: usize,
    ) -> Result<(), RocmTensorError> {
        self.execute_matmul_row_major_flags(
            a,
            b,
            c,
            [rows, inner],
            [inner, columns],
            false,
            false,
            fusion_pcu::PcuScalarType::F32,
            None,
            None,
        )
    }

    /// Interpret row-major operands with optional logical transposition; rocBLAS consumes the
    /// same backing through its column-major view, swapping operand order and transpose flags.
    #[allow(clippy::too_many_arguments, clippy::many_single_char_names)]
    fn execute_matmul_row_major_flags(
        &self,
        a: &RocmMemoryResource,
        b: &RocmMemoryResource,
        c: &RocmMemoryResource,
        a_shape: [usize; 2],
        b_shape: [usize; 2],
        transpose_a: bool,
        transpose_b: bool,
        scalar_type: fusion_pcu::PcuScalarType,
        sgemm_timing: Option<&mut RocblasSgemmHostTiming>,
        batch: Option<&mut HipCompletionBatch>,
    ) -> Result<(), RocmTensorError> {
        let rows = a_shape[usize::from(transpose_a)];
        let inner = a_shape[usize::from(!transpose_a)];
        let right_inner = b_shape[usize::from(transpose_b)];
        let columns = b_shape[usize::from(!transpose_b)];
        if inner != right_inner {
            return Err(RocmTensorError::InvalidShape);
        }
        if rows == 0 || inner == 0 || columns == 0 {
            return Err(RocmTensorError::InvalidShape);
        }
        let (element_size, _) =
            scalar_layout(scalar_type).map_err(|_| RocmTensorError::InvalidShape)?;
        let a_bytes = a_shape[0]
            .checked_mul(a_shape[1])
            .and_then(|n| n.checked_mul(element_size));
        let b_bytes = b_shape[0]
            .checked_mul(b_shape[1])
            .and_then(|n| n.checked_mul(element_size));
        let c_bytes = rows
            .checked_mul(columns)
            .and_then(|n| n.checked_mul(element_size));
        if a_bytes.is_none() || b_bytes.is_none() || c_bytes.is_none() {
            return Err(RocmTensorError::DimensionOverflow);
        }
        if !matches!(
            a.access(),
            PcuMemoryAccess::ReadOnly | PcuMemoryAccess::ReadWrite
        ) || !matches!(
            b.access(),
            PcuMemoryAccess::ReadOnly | PcuMemoryAccess::ReadWrite
        ) || !matches!(
            c.access(),
            PcuMemoryAccess::WriteOnly | PcuMemoryAccess::ReadWrite
        ) {
            return Err(RocmTensorError::InvalidMemoryAccess);
        }
        let (m, n) = (columns, rows);
        let args = (
            transpose_b,
            transpose_a,
            m,
            n,
            inner,
            1.0,
            b.device_buffer(),
            b_shape[1],
            a.device_buffer(),
            a_shape[1],
            0.0,
            c.device_buffer(),
            m,
        );
        let rocblas = self.rocblas()?;
        match scalar_type {
            fusion_pcu::PcuScalarType::F32 => {
                if let Some(batch) = batch {
                    rocblas.sgemm_into_batch(
                        batch, args.0, args.1, args.2, args.3, args.4, args.5, args.6, args.7,
                        args.8, args.9, args.10, args.11, args.12,
                    )?;
                } else if let Some(timing) = sgemm_timing {
                    rocblas.sgemm_profiled(
                        args.0, args.1, args.2, args.3, args.4, args.5, args.6, args.7, args.8,
                        args.9, args.10, args.11, args.12, timing,
                    )?;
                } else {
                    rocblas.sgemm(
                        args.0, args.1, args.2, args.3, args.4, args.5, args.6, args.7, args.8,
                        args.9, args.10, args.11, args.12,
                    )?;
                }
            }
            fusion_pcu::PcuScalarType::F64 => {
                if let Some(batch) = batch {
                    rocblas.dgemm_into_batch(
                        batch, args.0, args.1, args.2, args.3, args.4, 1.0, args.6, args.7, args.8,
                        args.9, 0.0, args.11, args.12,
                    )?;
                } else {
                    rocblas.dgemm(
                        args.0, args.1, args.2, args.3, args.4, 1.0, args.6, args.7, args.8,
                        args.9, 0.0, args.11, args.12,
                    )?;
                }
            }
            _ => return Err(RocmTensorError::InvalidShape),
        }
        Ok(())
    }
}

fn flattened_invocation_count(shape: &[usize]) -> Result<u32, RocmTensorExecutionError> {
    let count = shape
        .iter()
        .try_fold(1usize, |count, dimension| count.checked_mul(*dimension))
        .ok_or(RocmTensorExecutionError::SizeOverflow)?;
    u32::try_from(count)
        .ok()
        .and_then(NonZeroU32::new)
        .map(NonZeroU32::get)
        .ok_or(RocmTensorExecutionError::SizeOverflow)
}

fn mse_scratch_length_fits_scalar(
    count: usize,
    scalar: fusion_pcu::PcuScalarType,
    squared_bytes: usize,
) -> Result<bool, RocmTensorExecutionError> {
    let required = count
        .checked_mul(usize::from(scalar.bit_width() / 8))
        .ok_or(RocmTensorExecutionError::SizeOverflow)?;
    Ok(squared_bytes >= required)
}

#[cfg(test)]
fn mse_scratch_length_fits(
    count: usize,
    squared_bytes: usize,
) -> Result<bool, RocmTensorExecutionError> {
    let required_bytes = count
        .checked_mul(size_of::<f32>())
        .ok_or(RocmTensorExecutionError::SizeOverflow)?;
    Ok(squared_bytes >= required_bytes)
}

fn mse_scratch_resource_fits<R: PcuMemoryResource>(
    resource: &R,
    pool: PcuMemoryPoolId,
    count: usize,
) -> Result<bool, RocmTensorExecutionError> {
    let required_bytes = count
        .checked_mul(size_of::<f32>())
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or(RocmTensorExecutionError::SizeOverflow)?;
    Ok(resource.pool() == pool
        && resource.size_bytes() >= required_bytes
        && resource.access() == PcuMemoryAccess::ReadWrite)
}

/// A structurally validated `ROCm` execution plan for one graph output.
///
/// The graph is borrowed and must remain alive and unchanged while the plan is used.
pub struct RocmPreparedTensorGraph<'graph> {
    graph: &'graph Graph,
    plan: TensorExecutionPlan<'graph>,
    lowering_plan: TensorSelectedLoweringPlan<'graph>,
    data: RocmPreparedGraphData,
    nodes: Vec<NodeDescriptor<'graph>>,
}

/// Lifetime-free backend facts shared by borrowed and graph-owning prepared schedules.
#[doc(hidden)]
pub struct RocmPreparedGraphData {
    scalar_type: Option<fusion_pcu::PcuScalarType>,
    requires_blas: bool,
    batch_native_matmuls: bool,
    transport_only_inputs: bool,
    consuming_action: Option<PreparedConsumingAction>,
    fixed_dispatches: Vec<Option<PreparedFixedTensorDispatch>>,
    node_values: Vec<ValueId>,
    output: ValueId,
    outputs: Vec<ValueId>,
    index_by_value: HashMap<ValueId, usize>,
    use_counts: Vec<usize>,
    fused_add_by_relu: HashMap<ValueId, (ValueId, ValueId)>,
    bounded_pointwise_by_output: HashMap<ValueId, TensorBoundedPointwiseFusionGroup>,
    bounded_mul_by_output: HashMap<ValueId, TensorBoundedMulFusionGroup>,
    suppressed_adds: HashSet<ValueId>,
    indexed_storage_constraints: Vec<PreparedStorageConstraint>,
    matmul_operands: Vec<Option<PreparedMatMulOperands>>,
    strict_sgd: Vec<Option<strict_sgd::StrictSgdSpec>>,
    physical_layouts: HashMap<ValueId, RocmPhysicalLayout>,
    rewrites: Vec<TensorSgdRewriteCandidate>,
    input_values: Vec<ValueId>,
}

fn homogeneous_scalar_type(nodes: &[NodeDescriptor<'_>]) -> Option<fusion_pcu::PcuScalarType> {
    let first = nodes.first()?.scalar_type;
    nodes
        .iter()
        .all(|node| node.scalar_type == first)
        .then_some(first)
}

fn nodes_require_blas(nodes: &[NodeDescriptor<'_>]) -> bool {
    nodes.iter().any(|node| {
        matches!(
            node.op,
            OpDescriptor::MatMul { .. } | OpDescriptor::MeanSquaredError { .. }
        ) && node.numerical_mode != Some(fusion_pcu::PcuNumericalMode::Strict)
    })
}

fn require_f32_graph(data: &RocmPreparedGraphData) -> Result<(), RocmTensorExecutionError> {
    if data.scalar_type != Some(fusion_pcu::PcuScalarType::F32) {
        return Err(RocmTensorExecutionError::UnsupportedScalarType(
            data.scalar_type.unwrap_or(fusion_pcu::PcuScalarType::F64),
        ));
    }
    Ok(())
}

fn validate_owned_scalar_profile<T: fusion_pcu::PcuScalar>(
    profile: Option<fusion_pcu::PcuScalarType>,
) -> Result<(), RocmTensorExecutionError> {
    match profile {
        Some(actual) if actual == T::TYPE => Ok(()),
        Some(actual) => Err(RocmTensorExecutionError::UnsupportedScalarType(actual)),
        None => Err(RocmTensorExecutionError::UnsupportedScalarType(T::TYPE)),
    }
}

fn validate_tensor_scalar_tag<T: fusion_pcu::PcuScalar>(
    scalar_type: fusion_pcu::PcuScalarType,
) -> Result<(), RocmTensorExecutionError> {
    if scalar_type == T::TYPE {
        Ok(())
    } else {
        Err(RocmTensorExecutionError::UnsupportedScalarType(scalar_type))
    }
}

/// Owning prepared tensor schedule. The selected graph and all backend indexes live together;
/// there are no references from the schedule back into its graph.
pub struct RocmOwnedPreparedTensorGraph {
    scratch: owned_scratch::State,
    program: Arc<fusion_pcu::dialect::tensor::TensorOwnedSelectedProgram>,
    data: RocmPreparedGraphData,
}

fn with_single_output_plan<R>(
    prepared: &RocmOwnedPreparedTensorGraph,
    execute: impl FnOnce() -> Result<R, RocmTensorExecutionError>,
) -> Result<R, RocmTensorExecutionError> {
    let actual = prepared.data.outputs.len();
    if actual != 1 {
        return Err(RocmTensorExecutionError::OutputCountMismatch {
            expected: 1,
            actual,
        });
    }
    execute()
}

enum SelectedPlanRef<'a, 'graph> {
    Borrowed(&'a TensorSelectedLoweringPlan<'graph>),
    Owned(&'a fusion_pcu::dialect::tensor::TensorOwnedSelectedProgram),
}

struct RocmPreparedGraphView<'a, 'graph> {
    graph: &'a Graph,
    data: &'a RocmPreparedGraphData,
    selected: SelectedPlanRef<'a, 'graph>,
}

impl RocmPreparedGraphView<'_, '_> {
    fn node(&self, index: usize) -> Result<NodeDescriptor<'_>, RocmTensorExecutionError> {
        let value = *self
            .data
            .node_values
            .get(index)
            .ok_or(RocmTensorExecutionError::InvalidPlan(self.data.output))?;
        self.graph.node(value).map_err(Into::into)
    }

    fn source_input_values(&self) -> &[ValueId] {
        &self.data.input_values
    }

    fn operation_index_of(&self, value: ValueId) -> Option<usize> {
        match self.selected {
            SelectedPlanRef::Borrowed(lowering) => lowering.operation_index_of(value),
            SelectedPlanRef::Owned(program) => program.operation_index_of(value),
        }
    }

    fn scratch_storage_plan(
        &self,
        eligible: &[ValueId],
    ) -> Result<TensorScratchStoragePlan, RocmTensorExecutionError> {
        match self.selected {
            SelectedPlanRef::Borrowed(lowering) => lowering.scratch_storage_plan(eligible),
            SelectedPlanRef::Owned(program) => program.scratch_storage_plan(eligible),
        }
        .map_err(Into::into)
    }

    fn index_of(&self, value: ValueId) -> Option<usize> {
        self.data.index_by_value.get(&value).copied()
    }

    fn physical_layout(
        &self,
        value: ValueId,
    ) -> Result<RocmPhysicalLayout, RocmTensorExecutionError> {
        self.data
            .physical_layouts
            .get(&value)
            .copied()
            .ok_or(RocmTensorExecutionError::InvalidPlan(value))
    }

    fn fixed_dispatch(
        &self,
        index: usize,
    ) -> Result<&PreparedFixedTensorDispatch, RocmTensorExecutionError> {
        self.data
            .fixed_dispatches
            .get(index)
            .and_then(Option::as_ref)
            .ok_or_else(|| {
                let value = self
                    .data
                    .node_values
                    .get(index)
                    .copied()
                    .unwrap_or(self.data.output);
                RocmTensorExecutionError::InvalidPlan(value)
            })
    }
}

impl Deref for RocmPreparedGraphView<'_, '_> {
    type Target = RocmPreparedGraphData;

    fn deref(&self) -> &Self::Target {
        self.data
    }
}

impl Deref for RocmPreparedTensorGraph<'_> {
    type Target = RocmPreparedGraphData;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl<'graph> RocmPreparedTensorGraph<'graph> {
    const fn view(&self) -> RocmPreparedGraphView<'_, 'graph> {
        RocmPreparedGraphView {
            graph: self.graph,
            data: &self.data,
            selected: SelectedPlanRef::Borrowed(&self.lowering_plan),
        }
    }
}

impl RocmOwnedPreparedTensorGraph {
    fn from_parts(
        program: Arc<fusion_pcu::dialect::tensor::TensorOwnedSelectedProgram>,
        data: RocmPreparedGraphData,
    ) -> Result<Self, RocmTensorExecutionError> {
        let view = RocmPreparedGraphView {
            graph: program.graph(),
            data: &data,
            selected: SelectedPlanRef::Owned(&program),
        };
        let scratch = owned_scratch::State::new(&view)?;
        Ok(Self {
            scratch,
            program,
            data,
        })
    }

    fn view(&self) -> RocmPreparedGraphView<'_, '_> {
        RocmPreparedGraphView {
            graph: self.program.graph(),
            data: &self.data,
            selected: SelectedPlanRef::Owned(&self.program),
        }
    }

    /// Selected graph outputs in their requested order.
    #[must_use]
    pub fn outputs(&self) -> &[ValueId] {
        &self.data.outputs
    }

    /// The graph owned by this prepared schedule.
    #[must_use]
    pub fn graph(&self) -> &Graph {
        self.program.graph()
    }

    /// Reusable core selected program backing this `ROCm` preparation.
    #[must_use]
    pub fn tensor_program(&self) -> &fusion_pcu::dialect::tensor::TensorOwnedSelectedProgram {
        &self.program
    }
}

#[derive(Clone, Copy, Debug)]
struct PreparedStorageConstraint {
    constraint: TensorStorageConstraint,
    left_index: usize,
    right_index: usize,
    left_output_index: Option<usize>,
    right_output_index: Option<usize>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PreparedMatMulOperands {
    left_index: usize,
    right_index: usize,
    left_shape: [usize; 2],
    right_shape: [usize; 2],
    strict: Option<strict_matmul::StrictMatMulSpec>,
}

/// Host wall time attributed to one prepared graph node in diagnostic mode.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RocmTensorNodeTiming {
    /// Graph value produced by this node.
    pub value: ValueId,
    /// Operation kind, such as `MatMul`, `ReluBackward`, or `Input`.
    pub operation: &'static str,
    /// Elapsed wall time while scheduling or executing this node. Synchronous execution includes
    /// device completion waits; batched dispatch and SGEMM execution may measure enqueue time
    /// only, with a deferred wait charged to a later flush or occurring after the node loop.
    pub elapsed: Duration,
    /// Synchronous SGEMM host phase durations for `MatMul` nodes. Batched execution omits them.
    pub sgemm_host: Option<RocblasSgemmHostTiming>,
    /// Fixed elementwise host phases for profiled `Add`, `Sub`, or `Mul` nodes.
    pub elementwise_host: Option<RocmTensorElementwiseHostTiming>,
}

/// Fixed aggregate for one opt-in host timing phase.
#[cfg(feature = "insights")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RocmTensorInsightRecord {
    pub hits: u64,
    /// `None` means samples were invalid or the aggregate overflowed.
    pub ticks: Option<u64>,
    pub invalid_samples: u64,
}

/// Fixed per-operation host execution aggregate.
#[cfg(feature = "insights")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RocmTensorOperationInsight {
    pub record: RocmTensorInsightRecord,
}

/// Opt-in host phase accounting for batched output-bank execution.
///
/// All tick values use the caller-provided `InsightClock`; convert them using its calibration.
/// They are not nanoseconds or GPU execution durations. Operation records aggregate by kind,
/// avoiding a graph-sized allocation. Node execution includes any batch flushes performed inside
/// nodes, so only the final flush is subtracted when deriving scheduler bookkeeping.
#[cfg(feature = "insights")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RocmTensorBatchedExecutionTiming {
    pub total: RocmTensorInsightRecord,
    pub entry_validation: RocmTensorInsightRecord,
    pub scheduler_validation: RocmTensorInsightRecord,
    pub node_execution: RocmTensorInsightRecord,
    pub operations: [RocmTensorOperationInsight; 15],
    pub final_batch_finish: RocmTensorInsightRecord,
    pub final_batch_wait: RocmTensorInsightRecord,
    pub boundary_batch_finish: RocmTensorInsightRecord,
    pub boundary_batch_wait: RocmTensorInsightRecord,
    pub batch_drop_cleanup: RocmTensorInsightRecord,
    /// Total less entry/scheduler validation, node execution, final finish/wait, and cleanup.
    /// Invalid or non-monotonic samples make this `None`; it includes probe costs.
    pub scheduler_bookkeeping_ticks: Option<u64>,
    pub invalid_samples: u64,
    pub counter_overflow: bool,
}

#[cfg(feature = "insights")]
impl RocmTensorBatchedExecutionTiming {
    /// Operation names corresponding to the fixed `operations` array indices.
    #[must_use]
    pub const fn operation_names() -> [&'static str; 15] {
        [
            "Input",
            "Constant",
            "Uniform",
            "MatMul",
            "Add",
            "Sub",
            "Mul",
            "Relu",
            "ReluBackward",
            "SgdUpdate",
            "MeanSquaredError",
            "FusedAddRelu",
            "FusedAddSub",
            "FusedAddSubRelu",
            "Div",
        ]
    }
}

/// Host durations for the fixed elementwise dispatch lifecycle.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct RocmTensorElementwiseHostTiming {
    /// Dispatch cache admission and binding construction.
    pub cache_and_bind: Duration,
    /// Prepared dispatch submission.
    pub submit: Duration,
    /// Synchronous completion wait; zero when completion is deferred to an enclosing batch.
    pub wait: Duration,
}

#[derive(Clone, Copy)]
enum ElementwisePhase {
    CacheAndBind,
    Submit,
    Wait,
}

trait ElementwiseTimingSink {
    type Mark;

    fn begin(&mut self, phase: ElementwisePhase) -> Self::Mark;
    fn finish(&mut self, phase: ElementwisePhase, mark: Self::Mark);
}

struct NoopElementwiseTiming;

impl ElementwiseTimingSink for NoopElementwiseTiming {
    type Mark = ();

    #[inline(always)]
    fn begin(&mut self, _: ElementwisePhase) {}

    #[inline(always)]
    fn finish(&mut self, _: ElementwisePhase, (): Self::Mark) {}
}

#[derive(Default)]
struct CollectElementwiseTiming(RocmTensorElementwiseHostTiming);

impl ElementwiseTimingSink for CollectElementwiseTiming {
    type Mark = Instant;

    fn begin(&mut self, _: ElementwisePhase) -> Self::Mark {
        Instant::now()
    }

    fn finish(&mut self, phase: ElementwisePhase, mark: Self::Mark) {
        let elapsed = mark.elapsed();
        match phase {
            ElementwisePhase::CacheAndBind => self.0.cache_and_bind = elapsed,
            ElementwisePhase::Submit => self.0.submit = elapsed,
            ElementwisePhase::Wait => self.0.wait = elapsed,
        }
    }
}

trait NodeTimingSink {
    type Mark;
    #[cfg(feature = "insights")]
    type PhaseMark;
    type ElementwiseTiming: ElementwiseTimingSink;

    fn begin(&mut self) -> Self::Mark;
    fn finish(&mut self, mark: Self::Mark, node: &NodeDescriptor<'_>);

    #[cfg(feature = "insights")]
    fn begin_scheduler_validation(&mut self) -> Self::PhaseMark;
    #[cfg(feature = "insights")]
    fn finish_scheduler_validation(&mut self, mark: Self::PhaseMark);
    #[cfg(feature = "insights")]
    fn begin_batch_finish(&mut self) -> Self::PhaseMark;
    #[cfg(feature = "insights")]
    fn finish_batch_finish(&mut self, mark: Self::PhaseMark, is_final: bool);
    #[cfg(feature = "insights")]
    fn begin_batch_wait(&mut self) -> Self::PhaseMark;
    #[cfg(feature = "insights")]
    fn finish_batch_wait(&mut self, mark: Self::PhaseMark, is_final: bool);

    fn begin_elementwise(&mut self) -> Self::ElementwiseTiming;
    fn finish_elementwise(
        &mut self,
        timing: Self::ElementwiseTiming,
    ) -> Option<RocmTensorElementwiseHostTiming>;

    fn finish_fixed_elementwise(
        &mut self,
        mark: Self::Mark,
        node: &NodeDescriptor<'_>,
        _timing: Option<RocmTensorElementwiseHostTiming>,
    ) {
        self.finish(mark, node);
    }

    fn with_sgemm_timing<F>(
        &mut self,
        operation: F,
    ) -> Result<Option<RocblasSgemmHostTiming>, RocmTensorExecutionError>
    where
        F: FnOnce(Option<&mut RocblasSgemmHostTiming>) -> Result<(), RocmTensorExecutionError>,
    {
        operation(None)?;
        Ok(None)
    }

    fn finish_matmul(
        &mut self,
        mark: Self::Mark,
        node: &NodeDescriptor<'_>,
        _sgemm_host: Option<RocblasSgemmHostTiming>,
    ) {
        self.finish(mark, node);
    }

    fn finish_fused_add_relu(&mut self, mark: Self::Mark, node: &NodeDescriptor<'_>) {
        self.finish(mark, node);
    }

    fn finish_fused_add_sub(
        &mut self,
        mark: Self::Mark,
        node: &NodeDescriptor<'_>,
        _epilogue: TensorPointwiseEpilogue,
    ) {
        self.finish(mark, node);
    }
}

struct NoopNodeTiming;

impl NodeTimingSink for NoopNodeTiming {
    type Mark = ();
    #[cfg(feature = "insights")]
    type PhaseMark = ();
    type ElementwiseTiming = NoopElementwiseTiming;

    #[inline(always)]
    fn begin(&mut self) {}

    #[inline(always)]
    fn finish(&mut self, (): Self::Mark, _: &NodeDescriptor<'_>) {}

    #[cfg(feature = "insights")]
    #[inline(always)]
    fn begin_scheduler_validation(&mut self) -> Self::PhaseMark {}

    #[cfg(feature = "insights")]
    #[inline(always)]
    fn finish_scheduler_validation(&mut self, (): Self::PhaseMark) {}

    #[cfg(feature = "insights")]
    #[inline(always)]
    fn begin_batch_finish(&mut self) -> Self::PhaseMark {}

    #[cfg(feature = "insights")]
    #[inline(always)]
    fn finish_batch_finish(&mut self, (): Self::PhaseMark, _: bool) {}

    #[cfg(feature = "insights")]
    #[inline(always)]
    fn begin_batch_wait(&mut self) -> Self::PhaseMark {}

    #[cfg(feature = "insights")]
    #[inline(always)]
    fn finish_batch_wait(&mut self, (): Self::PhaseMark, _: bool) {}

    fn begin_elementwise(&mut self) -> Self::ElementwiseTiming {
        NoopElementwiseTiming
    }

    fn finish_elementwise(
        &mut self,
        _: Self::ElementwiseTiming,
    ) -> Option<RocmTensorElementwiseHostTiming> {
        None
    }
}

#[derive(Default)]
struct CollectNodeTimings(Vec<RocmTensorNodeTiming>);

#[cfg(feature = "insights")]
#[derive(Clone, Copy)]
enum InsightsPhase {
    Total,
    EntryValidation,
    SchedulerValidation,
    NodeExecution,
    FinalBatchFinish,
    FinalBatchWait,
    BoundaryBatchFinish,
    BoundaryBatchWait,
    BatchDropCleanup,
}

#[cfg(feature = "insights")]
struct CollectBatchedInsightsTiming<'clock, C> {
    clock: &'clock mut C,
    profile: RocmTensorBatchedExecutionTiming,
}

#[cfg(feature = "insights")]
impl<'clock, C: fusion_pcu::insights::InsightClock> CollectBatchedInsightsTiming<'clock, C> {
    const fn new(clock: &'clock mut C) -> Self {
        let empty = RocmTensorInsightRecord {
            hits: 0,
            ticks: Some(0),
            invalid_samples: 0,
        };
        Self {
            clock,
            profile: RocmTensorBatchedExecutionTiming {
                total: empty,
                entry_validation: empty,
                scheduler_validation: empty,
                node_execution: empty,
                operations: [RocmTensorOperationInsight { record: empty }; 15],
                final_batch_finish: empty,
                final_batch_wait: empty,
                boundary_batch_finish: empty,
                boundary_batch_wait: empty,
                batch_drop_cleanup: empty,
                scheduler_bookkeeping_ticks: Some(0),
                invalid_samples: 0,
                counter_overflow: false,
            },
        }
    }

    fn stamp(&mut self) -> fusion_pcu::insights::InsightStamp {
        self.clock.stamp()
    }

    fn record_phase(
        &mut self,
        phase: InsightsPhase,
        start: fusion_pcu::insights::InsightStamp,
        end: fusion_pcu::insights::InsightStamp,
    ) {
        let record = match phase {
            InsightsPhase::Total => &mut self.profile.total,
            InsightsPhase::EntryValidation => &mut self.profile.entry_validation,
            InsightsPhase::SchedulerValidation => &mut self.profile.scheduler_validation,
            InsightsPhase::NodeExecution => &mut self.profile.node_execution,
            InsightsPhase::FinalBatchFinish => &mut self.profile.final_batch_finish,
            InsightsPhase::FinalBatchWait => &mut self.profile.final_batch_wait,
            InsightsPhase::BoundaryBatchFinish => &mut self.profile.boundary_batch_finish,
            InsightsPhase::BoundaryBatchWait => &mut self.profile.boundary_batch_wait,
            InsightsPhase::BatchDropCleanup => &mut self.profile.batch_drop_cleanup,
        };
        record_insight_sample(
            record,
            start,
            end,
            &mut self.profile.invalid_samples,
            &mut self.profile.counter_overflow,
        );
    }

    fn record_operation(
        &mut self,
        index: usize,
        start: fusion_pcu::insights::InsightStamp,
        end: fusion_pcu::insights::InsightStamp,
    ) {
        let record = &mut self.profile.operations[index].record;
        record_insight_sample(
            record,
            start,
            end,
            &mut self.profile.invalid_samples,
            &mut self.profile.counter_overflow,
        );
    }

    fn record_node(
        &mut self,
        start: fusion_pcu::insights::InsightStamp,
        end: fusion_pcu::insights::InsightStamp,
        operation: usize,
    ) {
        self.record_phase(InsightsPhase::NodeExecution, start, end);
        self.record_operation(operation, start, end);
    }

    fn into_profile(mut self) -> RocmTensorBatchedExecutionTiming {
        let accounted = self
            .profile
            .entry_validation
            .ticks
            .zip(self.profile.scheduler_validation.ticks)
            .zip(self.profile.node_execution.ticks)
            .zip(self.profile.final_batch_finish.ticks)
            .zip(self.profile.final_batch_wait.ticks)
            .zip(self.profile.batch_drop_cleanup.ticks)
            .and_then(
                |(((((entry, validation), nodes), finish), wait), cleanup)| {
                    entry
                        .checked_add(validation)?
                        .checked_add(nodes)?
                        .checked_add(finish)?
                        .checked_add(wait)?
                        .checked_add(cleanup)
                },
            );
        self.profile.scheduler_bookkeeping_ticks = self
            .profile
            .total
            .ticks
            .zip(accounted)
            .and_then(|(total, accounted)| total.checked_sub(accounted));
        if self.profile.scheduler_bookkeeping_ticks.is_none()
            && self.profile.total.ticks.is_some()
            && accounted.is_some()
        {
            if let Some(invalid) = self.profile.invalid_samples.checked_add(1) {
                self.profile.invalid_samples = invalid;
            } else {
                self.profile.counter_overflow = true;
            }
        }
        self.profile
    }
}

#[cfg(feature = "insights")]
fn record_insight_sample(
    record: &mut RocmTensorInsightRecord,
    start: fusion_pcu::insights::InsightStamp,
    end: fusion_pcu::insights::InsightStamp,
    invalid_samples: &mut u64,
    counter_overflow: &mut bool,
) {
    let elapsed = end
        .ticks
        .checked_sub(start.ticks)
        .filter(|_| end.context == start.context);
    if let Some(elapsed) = elapsed {
        if let Some(hits) = record.hits.checked_add(1) {
            record.hits = hits;
            if let Some(ticks) = record.ticks {
                if let Some(sum) = ticks.checked_add(elapsed) {
                    record.ticks = Some(sum);
                } else {
                    *counter_overflow = true;
                    record.ticks = None;
                }
            }
        } else {
            *counter_overflow = true;
            record.ticks = None;
        }
    } else {
        record.ticks = None;
        if let Some(invalid) = record.invalid_samples.checked_add(1) {
            record.invalid_samples = invalid;
        } else {
            *counter_overflow = true;
        }
        if let Some(invalid) = invalid_samples.checked_add(1) {
            *invalid_samples = invalid;
        } else {
            *counter_overflow = true;
        }
    }
}

#[cfg(feature = "insights")]
impl<C: fusion_pcu::insights::InsightClock> NodeTimingSink for CollectBatchedInsightsTiming<'_, C> {
    type Mark = fusion_pcu::insights::InsightStamp;
    type PhaseMark = fusion_pcu::insights::InsightStamp;
    type ElementwiseTiming = NoopElementwiseTiming;

    fn begin(&mut self) -> Self::Mark {
        self.stamp()
    }

    fn finish(&mut self, start: Self::Mark, node: &NodeDescriptor<'_>) {
        let end = self.stamp();
        self.record_node(start, end, operation_insight_index(node.op));
    }

    fn begin_scheduler_validation(&mut self) -> Self::PhaseMark {
        self.stamp()
    }

    fn finish_scheduler_validation(&mut self, start: Self::PhaseMark) {
        let end = self.stamp();
        self.record_phase(InsightsPhase::SchedulerValidation, start, end);
    }

    fn begin_batch_finish(&mut self) -> Self::PhaseMark {
        self.stamp()
    }

    fn finish_batch_finish(&mut self, start: Self::PhaseMark, is_final: bool) {
        let end = self.stamp();
        self.record_phase(
            if is_final {
                InsightsPhase::FinalBatchFinish
            } else {
                InsightsPhase::BoundaryBatchFinish
            },
            start,
            end,
        );
    }

    fn begin_batch_wait(&mut self) -> Self::PhaseMark {
        self.stamp()
    }

    fn finish_batch_wait(&mut self, start: Self::PhaseMark, is_final: bool) {
        let end = self.stamp();
        self.record_phase(
            if is_final {
                InsightsPhase::FinalBatchWait
            } else {
                InsightsPhase::BoundaryBatchWait
            },
            start,
            end,
        );
    }

    fn begin_elementwise(&mut self) -> Self::ElementwiseTiming {
        NoopElementwiseTiming
    }

    fn finish_elementwise(
        &mut self,
        _: Self::ElementwiseTiming,
    ) -> Option<RocmTensorElementwiseHostTiming> {
        None
    }

    fn finish_fused_add_relu(&mut self, start: Self::Mark, _: &NodeDescriptor<'_>) {
        let end = self.stamp();
        self.record_node(start, end, 11);
    }

    fn finish_fused_add_sub(
        &mut self,
        start: Self::Mark,
        _: &NodeDescriptor<'_>,
        epilogue: TensorPointwiseEpilogue,
    ) {
        let end = self.stamp();
        self.record_node(
            start,
            end,
            if epilogue == TensorPointwiseEpilogue::Relu {
                13
            } else {
                12
            },
        );
    }
}

#[cfg(feature = "insights")]
const fn operation_insight_index(op: OpDescriptor<'_>) -> usize {
    match op {
        OpDescriptor::Input => 0,
        OpDescriptor::Constant(_) => 1,
        OpDescriptor::Uniform { .. } => 2,
        OpDescriptor::MatMul { .. } => 3,
        OpDescriptor::Add { .. } => 4,
        OpDescriptor::Sub { .. } => 5,
        OpDescriptor::Mul { .. } => 6,
        OpDescriptor::Div { .. } => 14,
        OpDescriptor::Relu { .. } => 7,
        OpDescriptor::ReluBackward { .. } => 8,
        OpDescriptor::SgdUpdate { .. } => 9,
        OpDescriptor::MeanSquaredError { .. } => 10,
    }
}

impl NodeTimingSink for CollectNodeTimings {
    type Mark = Instant;
    #[cfg(feature = "insights")]
    type PhaseMark = ();
    type ElementwiseTiming = CollectElementwiseTiming;

    fn begin(&mut self) -> Self::Mark {
        Instant::now()
    }

    fn finish(&mut self, mark: Self::Mark, node: &NodeDescriptor<'_>) {
        self.push(mark, node, None, None);
    }

    #[cfg(feature = "insights")]
    fn begin_scheduler_validation(&mut self) -> Self::PhaseMark {}

    #[cfg(feature = "insights")]
    fn finish_scheduler_validation(&mut self, (): Self::PhaseMark) {}

    #[cfg(feature = "insights")]
    fn begin_batch_finish(&mut self) -> Self::PhaseMark {}

    #[cfg(feature = "insights")]
    fn finish_batch_finish(&mut self, (): Self::PhaseMark, _: bool) {}

    #[cfg(feature = "insights")]
    fn begin_batch_wait(&mut self) -> Self::PhaseMark {}

    #[cfg(feature = "insights")]
    fn finish_batch_wait(&mut self, (): Self::PhaseMark, _: bool) {}

    fn begin_elementwise(&mut self) -> Self::ElementwiseTiming {
        CollectElementwiseTiming::default()
    }

    fn finish_elementwise(
        &mut self,
        timing: Self::ElementwiseTiming,
    ) -> Option<RocmTensorElementwiseHostTiming> {
        Some(timing.0)
    }

    fn finish_fixed_elementwise(
        &mut self,
        mark: Self::Mark,
        node: &NodeDescriptor<'_>,
        elementwise_host: Option<RocmTensorElementwiseHostTiming>,
    ) {
        self.push(mark, node, None, elementwise_host);
    }

    fn with_sgemm_timing<F>(
        &mut self,
        operation: F,
    ) -> Result<Option<RocblasSgemmHostTiming>, RocmTensorExecutionError>
    where
        F: FnOnce(Option<&mut RocblasSgemmHostTiming>) -> Result<(), RocmTensorExecutionError>,
    {
        let mut timing = RocblasSgemmHostTiming::default();
        operation(Some(&mut timing))?;
        Ok(Some(timing))
    }

    fn finish_matmul(
        &mut self,
        mark: Self::Mark,
        node: &NodeDescriptor<'_>,
        sgemm_host: Option<RocblasSgemmHostTiming>,
    ) {
        self.push(mark, node, sgemm_host, None);
    }

    fn finish_fused_add_relu(&mut self, mark: Self::Mark, node: &NodeDescriptor<'_>) {
        self.0.push(RocmTensorNodeTiming {
            value: node.value,
            operation: "FusedAddRelu",
            elapsed: mark.elapsed(),
            sgemm_host: None,
            elementwise_host: None,
        });
    }

    fn finish_fused_add_sub(
        &mut self,
        mark: Self::Mark,
        node: &NodeDescriptor<'_>,
        epilogue: TensorPointwiseEpilogue,
    ) {
        self.0.push(RocmTensorNodeTiming {
            value: node.value,
            operation: match epilogue {
                TensorPointwiseEpilogue::Identity => "FusedAddSub",
                TensorPointwiseEpilogue::Relu => "FusedAddSubRelu",
            },
            elapsed: mark.elapsed(),
            sgemm_host: None,
            elementwise_host: None,
        });
    }
}

impl CollectNodeTimings {
    fn push(
        &mut self,
        mark: Instant,
        node: &NodeDescriptor<'_>,
        sgemm_host: Option<RocblasSgemmHostTiming>,
        elementwise_host: Option<RocmTensorElementwiseHostTiming>,
    ) {
        self.0.push(RocmTensorNodeTiming {
            value: node.value,
            operation: match node.op {
                OpDescriptor::Input => "Input",
                OpDescriptor::Constant(_) => "Constant",
                OpDescriptor::Uniform { .. } => "Uniform",
                OpDescriptor::MatMul { .. } => "MatMul",
                OpDescriptor::Add { .. } => "Add",
                OpDescriptor::Sub { .. } => "Sub",
                OpDescriptor::Mul { .. } => "Mul",
                OpDescriptor::Div { .. } => "Div",
                OpDescriptor::Relu { .. } => "Relu",
                OpDescriptor::ReluBackward { .. } => "ReluBackward",
                OpDescriptor::SgdUpdate { .. } => "SgdUpdate",
                OpDescriptor::MeanSquaredError { .. } => "MeanSquaredError",
            },
            elapsed: mark.elapsed(),
            sgemm_host,
            elementwise_host,
        });
    }
}

/// Reusable device allocations for one exact prepared graph, session, and pool.
///
/// Mutably borrowing this value for execution excludes concurrent reuse of its buffers.
pub struct RocmTensorScratch<'plan, 'graph, 'session> {
    prepared: &'plan RocmPreparedTensorGraph<'graph>,
    session: &'session RocmOwnedDispatchBackend,
    pool: PcuMemoryPoolId,
    resources: Vec<Option<RocmMemoryResource>>,
    mse_squared: Option<RocmMemoryResource>,
    poisoned: bool,
}

struct RocmExecutionScratch<'a> {
    resources: &'a [Option<RocmMemoryResource>],
    mse_squared: Option<&'a RocmMemoryResource>,
    statuses: Option<&'a mut [Option<owned_scratch::Status>]>,
    outputs: &'a [ValueId],
    node_values: &'a [ValueId],
}

impl RocmExecutionScratch<'_> {
    fn status(&mut self, index: usize) -> Option<&mut owned_scratch::Status> {
        self.statuses.as_deref_mut()?.get_mut(index)?.as_mut()
    }

    fn lease(&self, index: usize) -> Result<RocmMemoryResource, RocmTensorExecutionError> {
        self.resources
            .get(index)
            .and_then(Option::as_ref)
            .map(RocmMemoryResource::clone_for_tensor_input)
            .ok_or_else(|| RocmTensorExecutionError::MissingResource(self.node_values[index]))
    }
}

/// Caller-owned output allocations reusable for repeated execution of one exact prepared plan.
pub struct RocmTensorOutputBank<'plan, 'graph, 'session> {
    prepared: &'plan RocmPreparedTensorGraph<'graph>,
    session: &'session RocmOwnedDispatchBackend,
    pool: PcuMemoryPoolId,
    outputs: Vec<RocmTensorInput<'session>>,
    requirements: Vec<PcuMemoryMemberRequirement>,
    poisoned: bool,
}

impl<'session> RocmTensorOutputBank<'_, '_, 'session> {
    /// Outputs in the order requested when the graph was prepared.
    #[must_use]
    pub fn outputs(&self) -> &[RocmTensorInput<'session>] {
        &self.outputs
    }

    fn output(&self, value: ValueId) -> Option<&RocmTensorInput<'session>> {
        output_position(&self.prepared.outputs, value).and_then(|index| self.outputs.get(index))
    }
}

fn output_position(outputs: &[ValueId], value: ValueId) -> Option<usize> {
    outputs.iter().position(|&output| output == value)
}

fn validate_output_bank<I: RocmTensorInputDescriptor>(
    bank: &RocmTensorOutputBank<'_, '_, '_>,
    prepared: &RocmPreparedTensorGraph<'_>,
    session: &RocmOwnedDispatchBackend,
    pool: PcuMemoryPoolId,
    inputs: &[(ValueId, &I)],
    scratch: Option<&RocmTensorScratch<'_, '_, '_>>,
) -> Result<(), RocmTensorExecutionError> {
    if bank.poisoned
        || !std::ptr::eq(bank.prepared, prepared)
        || !std::ptr::eq(bank.session, session)
        || bank.pool != pool
        || bank.outputs.len() != prepared.outputs.len()
    {
        return Err(RocmTensorExecutionError::OutputResourceMismatch);
    }
    for (&value, output) in prepared.outputs.iter().zip(&bank.outputs) {
        let expected_shape = prepared.graph.shape(value)?;
        if !std::ptr::eq(output.session, session)
            || output.shape.as_slice() != expected_shape
            || output.resource.pool() != pool
            || !output.resource.belongs_to_runtime(session.tensor_runtime())
            || output.resource.device_buffer().len() < byte_len(expected_shape)?
            || output.resource.access() != PcuMemoryAccess::ReadWrite
            || !output
                .resource
                .supports(PcuMemoryResourceCapability::ReusableStorage)
        {
            return Err(RocmTensorExecutionError::OutputResourceMismatch);
        }
        if inputs
            .iter()
            .any(|(_, input)| input.resource().may_overlap(&output.resource))
            || scratch.is_some_and(|scratch| {
                scratch
                    .resources
                    .iter()
                    .flatten()
                    .any(|resource| resource.may_overlap(&output.resource))
                    || scratch
                        .mse_squared
                        .as_ref()
                        .is_some_and(|resource| resource.may_overlap(&output.resource))
            })
        {
            return Err(RocmTensorExecutionError::OutputResourceMismatch);
        }
    }
    validate_reusable_memory_bank_members_by(&bank.outputs, &bank.requirements, |output| {
        &output.resource
    })
    .map_err(|_| RocmTensorExecutionError::OutputResourceMismatch)?;
    Ok(())
}

fn validate_prepared_storage_constraints<I: RocmTensorInputDescriptor>(
    prepared: &RocmPreparedTensorGraph<'_>,
    inputs: &[(ValueId, &I)],
    scratch: Option<&RocmTensorScratch<'_, '_, '_>>,
    bank: Option<&RocmTensorOutputBank<'_, '_, '_>>,
) -> Result<(), RocmTensorExecutionError> {
    if let Some(scratch) = scratch
        && let Some(squared) = scratch.mse_squared.as_ref()
    {
        let input_resources = inputs.iter().map(|(_, input)| input.resource());
        let prepared_resources = scratch.resources.iter().flatten();
        if resource_may_overlap_any(squared, input_resources.chain(prepared_resources)) {
            return Err(RocmTensorExecutionError::ScratchMismatch);
        }
    }
    validate_indexed_storage_constraints(
        &prepared.indexed_storage_constraints,
        |value, index, output_index| {
            bank.and_then(|bank| {
                output_index
                    .and_then(|output_index| bank.outputs.get(output_index))
                    .map(|output| &output.resource)
            })
            .or_else(|| {
                inputs
                    .iter()
                    .find(|(id, _)| *id == value)
                    .map(|(_, input)| input.resource())
            })
            .or_else(|| {
                scratch
                    .and_then(|scratch| scratch.resources.get(index))
                    .and_then(Option::as_ref)
            })
        },
    )
}

fn validate_indexed_storage_constraints<'resource, R: PcuMemoryResource + 'resource>(
    constraints: &[PreparedStorageConstraint],
    mut resource_for: impl FnMut(ValueId, usize, Option<usize>) -> Option<&'resource R>,
) -> Result<(), RocmTensorExecutionError> {
    for indexed in constraints {
        if let (Some(left), Some(right)) = (
            resource_for(
                indexed.constraint.left,
                indexed.left_index,
                indexed.left_output_index,
            ),
            resource_for(
                indexed.constraint.right,
                indexed.right_index,
                indexed.right_output_index,
            ),
        ) {
            indexed
                .constraint
                .validate_resources(left, right)
                .map_err(RocmTensorExecutionError::StorageConstraint)?;
        }
    }
    Ok(())
}

fn resource_may_overlap_any<R: PcuMemoryResource>(
    resource: &R,
    others: impl IntoIterator<Item = impl std::borrow::Borrow<R>>,
) -> bool {
    others.into_iter().any(|other| {
        let other = other.borrow();
        resource.overlap(
            other,
            fusion_pcu::PcuMemoryRange {
                offset_bytes: 0,
                size_bytes: resource.size_bytes(),
            },
            fusion_pcu::PcuMemoryRange {
                offset_bytes: 0,
                size_bytes: other.size_bytes(),
            },
        ) != fusion_pcu::PcuMemoryOverlap::Disjoint
    })
}

fn input_storage_requirement(
    value: ValueId,
    shape: &[usize],
    scalar_type: fusion_pcu::PcuScalarType,
) -> Result<fusion_pcu::dialect::tensor::TensorValueStorageRequirement, RocmTensorExecutionError> {
    let (element_size, alignment) = scalar_layout(scalar_type)?;
    Ok(fusion_pcu::dialect::tensor::TensorValueStorageRequirement {
        value,
        scalar_type,
        alignment_bytes: alignment,
        output_bytes: u64::try_from(byte_len_for_size(shape, element_size)?)
            .map_err(|_| RocmTensorExecutionError::SizeOverflow)?,
        access: PcuMemoryAccess::ReadOnly,
    })
}

fn physical_value_storage_requirement(
    prepared: &RocmPreparedTensorGraph<'_>,
    value: ValueId,
    op: OpDescriptor<'_>,
) -> Result<fusion_pcu::dialect::tensor::TensorValueStorageRequirement, RocmTensorExecutionError> {
    Ok(fusion_pcu::dialect::tensor::TensorValueStorageRequirement {
        value,
        scalar_type: prepared.graph.node(value)?.scalar_type,
        alignment_bytes: scalar_layout(prepared.graph.node(value)?.scalar_type)?.1,
        output_bytes: prepared.physical_layout(value)?.physical_bytes,
        access: match op {
            OpDescriptor::Input | OpDescriptor::Constant(_) | OpDescriptor::Uniform { .. } => {
                PcuMemoryAccess::ReadOnly
            }
            _ => PcuMemoryAccess::ReadWrite,
        },
    })
}

// The destination may come from the persistent feedback bank, a fresh owned-result slot,
// selected scratch storage, or the provider; keeping routing at the allocation boundary avoids
// duplicating storage policy in every tensor operation branch.
#[allow(clippy::too_many_arguments)]
fn execution_resource<P: PcuMemoryProvider<Resource = RocmMemoryResource>>(
    memory: &mut P,
    pool: PcuMemoryPoolId,
    shape: &[usize],
    scalar_type: fusion_pcu::PcuScalarType,
    index: usize,
    value: ValueId,
    scratch: &mut Option<&mut RocmExecutionScratch<'_>>,
    output_bank: Option<&RocmTensorOutputBank<'_, '_, '_>>,
    fresh_outputs: Option<&[(ValueId, FreshTensorOutput<'_>)]>,
) -> Result<RocmMemoryResource, RocmTensorExecutionError> {
    if let Some(output) = tensor_output_view(output_bank, fresh_outputs, value) {
        return Ok(output.resource.clone_for_tensor_input());
    }
    if let Some(scratch) = scratch.as_deref_mut()
        && !scratch.outputs.contains(&value)
    {
        return scratch.lease(index);
    }
    let (element_size, alignment) = scalar_layout(scalar_type)?;
    allocate_tensor_for_size(
        memory,
        pool,
        shape,
        element_size,
        usize::try_from(alignment).map_err(|_| RocmTensorExecutionError::SizeOverflow)?,
    )
}

fn validate_owned_storage(
    view: &RocmPreparedGraphView<'_, '_>,
    inputs: &[(ValueId, &RocmTensorInputRef<'_>)],
    outputs: &[(ValueId, FreshTensorOutput<'_>)],
    resources: &[Option<RocmMemoryResource>],
) -> Result<(), RocmTensorExecutionError> {
    validate_indexed_storage_constraints(
        &view.indexed_storage_constraints,
        |value, index, _output_index| {
            fresh_output(Some(outputs), value)
                .map(|output| &output.resource)
                .or_else(|| {
                    inputs
                        .iter()
                        .find(|(candidate, _)| *candidate == value)
                        .map(|(_, input)| input.resource())
                })
                .or_else(|| resources.get(index).and_then(Option::as_ref))
        },
    )
}

fn fresh_output<'outputs, 'graph>(
    outputs: Option<&'outputs [(ValueId, FreshTensorOutput<'graph>)]>,
    value: ValueId,
) -> Option<&'outputs FreshTensorOutput<'graph>> {
    outputs?
        .iter()
        .find(|(candidate, _)| *candidate == value)
        .map(|(_, output)| output)
}

fn tensor_output_view<'output>(
    bank: Option<&'output RocmTensorOutputBank<'_, '_, '_>>,
    fresh_outputs: Option<&'output [(ValueId, FreshTensorOutput<'_>)]>,
    value: ValueId,
) -> Option<TensorOutputView<'output>> {
    bank.and_then(|bank| bank.output(value))
        .map(TensorOutputView::from)
        .or_else(|| fresh_output(fresh_outputs, value).map(TensorOutputView::from))
}

fn scratch_stores_node(op: OpDescriptor<'_>, value: ValueId, outputs: &[ValueId]) -> bool {
    !outputs.contains(&value) && !matches!(op, OpDescriptor::Input)
}

const fn is_scratch_computed_op(op: OpDescriptor<'_>) -> bool {
    matches!(
        op,
        OpDescriptor::MatMul { .. }
            | OpDescriptor::Add { .. }
            | OpDescriptor::Sub { .. }
            | OpDescriptor::Mul { .. }
            | OpDescriptor::Div { .. }
            | OpDescriptor::Relu { .. }
            | OpDescriptor::ReluBackward { .. }
            | OpDescriptor::SgdUpdate { .. }
            | OpDescriptor::MeanSquaredError { .. }
    )
}

fn selected_scratch_storage_plan(
    prepared: &RocmPreparedTensorGraph<'_>,
) -> Result<TensorScratchStoragePlan, RocmTensorExecutionError> {
    let view = prepared.view();
    let mut eligible = Vec::new();
    for index in 0..view.node_values.len() {
        let node = view.node(index)?;
        if !view.suppressed_adds.contains(&node.value)
            && scratch_stores_node(node.op, node.value, &view.outputs)
            && is_scratch_computed_op(node.op)
            && view.operation_index_of(node.value).is_some()
        {
            eligible.push(node.value);
        }
    }
    view.scratch_storage_plan(&eligible)
}

fn mse_scratch_element_count(
    graph: &Graph,
    nodes: &[NodeDescriptor<'_>],
) -> Result<usize, RocmTensorExecutionError> {
    nodes.iter().try_fold(0usize, |max_count, node| {
        let OpDescriptor::MeanSquaredError { prediction, .. } = node.op else {
            return Ok(max_count);
        };
        if node.numerical_mode == Some(fusion_pcu::PcuNumericalMode::Strict) {
            return Ok(max_count);
        }
        let shape = graph.shape(prediction)?;
        // The MSE output is scalar; size internal scratch from its prediction shape.
        let count = shape
            .iter()
            .try_fold(1usize, |n, d| n.checked_mul(*d))
            .ok_or(RocmTensorExecutionError::SizeOverflow)?;
        if count == 0 || count > i32::MAX as usize {
            return Err(RocmTensorExecutionError::Unsupported {
                value: node.value,
                reason: TensorUnsupportedReason::Shape,
            });
        }
        Ok(max_count.max(count))
    })
}

impl RocmPreparedTensorGraph<'_> {
    fn index_of(&self, value: ValueId) -> Option<usize> {
        self.index_by_value.get(&value).copied()
    }

    fn physical_layout(
        &self,
        value: ValueId,
    ) -> Result<RocmPhysicalLayout, RocmTensorExecutionError> {
        self.physical_layouts
            .get(&value)
            .copied()
            .ok_or(RocmTensorExecutionError::InvalidPlan(value))
    }

    fn is_compact_uniform(&self, value: ValueId) -> Result<bool, RocmTensorExecutionError> {
        let is_uniform_node = self
            .index_of(value)
            .and_then(|index| self.nodes.get(index))
            .is_some_and(|node| matches!(node.op, OpDescriptor::Uniform { .. }));
        Ok(is_uniform_node
            && self.physical_layout(value)?.representation
                == RocmPhysicalRepresentation::UniformScalar)
    }
}

#[allow(clippy::too_many_lines)] // Keeps per-node policy and cache-key identity assembled in one pass.
fn collect_dispatch_requests<'graph>(
    prepared: &'graph RocmPreparedTensorGraph<'_>,
) -> Result<Vec<TensorDispatchRequest<'graph>>, RocmTensorExecutionError> {
    let mut requests = Vec::new();
    for node in &prepared.nodes {
        if prepared.suppressed_adds.contains(&node.value) {
            continue;
        }
        let request = match node.op {
            OpDescriptor::MeanSquaredError { .. }
                if node.numerical_mode == Some(fusion_pcu::PcuNumericalMode::Strict) =>
            {
                Some(TensorDispatchRequest::StrictMse(
                    strict_mse::Profile::from_node(prepared.graph, *node).map_err(|reason| {
                        RocmTensorExecutionError::Unsupported {
                            value: node.value,
                            reason,
                        }
                    })?,
                ))
            }
            OpDescriptor::ReluBackward { .. } => Some(TensorDispatchRequest::ReluBackward(
                relu_backward::Profile::from_node(*node).map_err(|reason| {
                    RocmTensorExecutionError::Unsupported {
                        value: node.value,
                        reason,
                    }
                })?,
            )),
            OpDescriptor::SgdUpdate { .. }
                if node.numerical_mode == Some(fusion_pcu::PcuNumericalMode::Strict) =>
            {
                let index = prepared
                    .index_of(node.value)
                    .ok_or(RocmTensorExecutionError::InvalidPlan(node.value))?;
                Some(TensorDispatchRequest::StrictSgd(
                    prepared.strict_sgd[index]
                        .ok_or(RocmTensorExecutionError::InvalidPlan(node.value))?,
                ))
            }
            OpDescriptor::MatMul { .. }
                if node.numerical_mode == Some(fusion_pcu::PcuNumericalMode::Strict) =>
            {
                let index = prepared
                    .index_of(node.value)
                    .ok_or(RocmTensorExecutionError::InvalidPlan(node.value))?;
                let spec = prepared.matmul_operands[index]
                    .as_ref()
                    .and_then(|operands| operands.strict)
                    .ok_or(RocmTensorExecutionError::InvalidPlan(node.value))?;
                Some(TensorDispatchRequest::StrictMatMul(spec))
            }
            OpDescriptor::Add { left, right } => {
                if let Some(group) = prepared
                    .bounded_pointwise_by_output
                    .get(&node.value)
                    .filter(|group| group.epilogue == TensorPointwiseEpilogue::Identity)
                {
                    Some(bounded_pointwise_request(prepared, group)?)
                } else {
                    Some(fixed_binary_request(
                        prepared,
                        *node,
                        TensorDispatchKind::Add,
                        left,
                        right,
                    )?)
                }
            }
            OpDescriptor::Sub { left, right } => {
                if let Some(group) = prepared
                    .bounded_pointwise_by_output
                    .get(&node.value)
                    .filter(|group| group.epilogue == TensorPointwiseEpilogue::Identity)
                {
                    Some(bounded_pointwise_request(prepared, group)?)
                } else {
                    Some(fixed_binary_request(
                        prepared,
                        *node,
                        TensorDispatchKind::Sub,
                        left,
                        right,
                    )?)
                }
            }
            OpDescriptor::Mul { left, right } => {
                if let Some(group) = prepared.bounded_mul_by_output.get(&node.value) {
                    Some(bounded_mul_request(prepared, group)?)
                } else {
                    Some(fixed_binary_request(
                        prepared,
                        *node,
                        TensorDispatchKind::Mul,
                        left,
                        right,
                    )?)
                }
            }
            OpDescriptor::Div { left, right } => Some(fixed_binary_request(
                prepared,
                *node,
                TensorDispatchKind::CheckedFloatDiv,
                left,
                right,
            )?),
            OpDescriptor::Relu { .. } => {
                if let Some(group) = prepared.bounded_pointwise_by_output.get(&node.value) {
                    Some(bounded_pointwise_request(prepared, group)?)
                } else if let Some(&(left, right)) = prepared.fused_add_by_relu.get(&node.value) {
                    Some(TensorDispatchRequest::Fixed {
                        kind: TensorDispatchKind::AddRelu,
                        scalar_type: TensorPointwiseScalarType::try_from(node.scalar_type)?,
                        logical_count: flattened_invocation_count(node.shape)?,
                        scalar_mask: scalar_mask_for_operands(prepared, left, right)?,
                        numerical_requirements: fixed_numerical_requirements(*node),
                    })
                } else {
                    Some(TensorDispatchRequest::Fixed {
                        kind: TensorDispatchKind::Relu,
                        scalar_type: TensorPointwiseScalarType::try_from(node.scalar_type)?,
                        logical_count: flattened_invocation_count(node.shape)?,
                        scalar_mask: 0,
                        numerical_requirements: fixed_numerical_requirements(*node),
                    })
                }
            }
            OpDescriptor::Input
            | OpDescriptor::Constant(_)
            | OpDescriptor::Uniform { .. }
            | OpDescriptor::MatMul { .. }
            | OpDescriptor::MeanSquaredError { .. }
            | OpDescriptor::SgdUpdate { .. } => None,
        };
        if let Some(request) = request {
            let key = request.key();
            if !requests
                .iter()
                .any(|existing: &TensorDispatchRequest<'_>| existing.key() == key)
            {
                requests.push(request);
            }
        }
    }
    Ok(requests)
}

/// Scalar nodes deliberately omit compound exception granularity. Their canonical header
/// is Boundary: one checked primitive has identical Boundary/Strict semantics. None never
/// inherits mutable Graph defaults; captured options and underflow remain exact. Compound
/// nodes retain their explicit Some(mode). Graph Clamp has no node contract yet.
const fn fixed_numerical_requirements(
    node: NodeDescriptor<'_>,
) -> fusion_pcu::PcuImplementationRequirements {
    fusion_pcu::PcuImplementationRequirements {
        numerical_mode: match node.numerical_mode {
            Some(mode) => mode,
            None => fusion_pcu::PcuNumericalMode::Boundary,
        },
        numerical_options: node.numerical_options,
        float_underflow: match node.float_underflow_policy {
            Some(policy) => policy,
            None => PcuFloatUnderflowPolicy::IeeeAfterRounding,
        },
        range_policy: fusion_pcu::PcuRangePolicy::Reject,
    }
}

fn fixed_binary_request(
    prepared: &RocmPreparedTensorGraph<'_>,
    node: NodeDescriptor<'_>,
    float_kind: TensorDispatchKind,
    left: ValueId,
    right: ValueId,
) -> Result<TensorDispatchRequest<'static>, RocmTensorExecutionError> {
    let scalar_type = TensorPointwiseScalarType::try_from(node.scalar_type)?;
    Ok(TensorDispatchRequest::Fixed {
        kind: binary_dispatch_kind(float_kind, scalar_type),
        scalar_type,
        logical_count: flattened_invocation_count(node.shape)?,
        scalar_mask: scalar_mask_for_operands(prepared, left, right)?,
        numerical_requirements: fixed_numerical_requirements(node),
    })
}

fn bounded_pointwise_request<'graph>(
    prepared: &'graph RocmPreparedTensorGraph<'_>,
    group: &'graph TensorBoundedPointwiseFusionGroup,
) -> Result<TensorDispatchRequest<'graph>, RocmTensorExecutionError> {
    Ok(TensorDispatchRequest::BoundedPointwise {
        group,
        logical_count: flattened_invocation_count(&group.shape)?,
        scalar_mask: group_leaf_scalar_mask(prepared, group)?,
        topology: bounded_pointwise_topology(group),
    })
}

fn bounded_mul_request<'graph>(
    prepared: &'graph RocmPreparedTensorGraph<'_>,
    group: &'graph TensorBoundedMulFusionGroup,
) -> Result<TensorDispatchRequest<'graph>, RocmTensorExecutionError> {
    let scalar_mask = group
        .leaves
        .iter()
        .enumerate()
        .try_fold(0_u8, |mask, (index, &leaf)| {
            Ok::<_, RocmTensorExecutionError>(
                if prepared.physical_layout(leaf)?.representation
                    == RocmPhysicalRepresentation::UniformScalar
                {
                    mask | (1_u8 << index)
                } else {
                    mask
                },
            )
        })?;
    Ok(TensorDispatchRequest::BoundedMul {
        group,
        logical_count: flattened_invocation_count(&group.shape)?,
        scalar_mask,
        topology: bounded_mul_topology(group),
    })
}

fn retained_requested_key_count(
    requested: &[TensorDispatchCacheKey],
    cached: &[TensorDispatchCacheKey],
) -> usize {
    requested.iter().filter(|key| cached.contains(key)).count()
}

fn scalar_mask_for_operands(
    prepared: &RocmPreparedTensorGraph<'_>,
    left: ValueId,
    right: ValueId,
) -> Result<u8, RocmTensorExecutionError> {
    Ok(u8::from(prepared.is_compact_uniform(left)?)
        | (u8::from(prepared.is_compact_uniform(right)?) << 1))
}

fn group_leaf_scalar_mask(
    prepared: &RocmPreparedTensorGraph<'_>,
    group: &TensorBoundedPointwiseFusionGroup,
) -> Result<u8, RocmTensorExecutionError> {
    group
        .leaves
        .iter()
        .enumerate()
        .try_fold(0_u8, |mask, (index, &leaf)| {
            if prepared.physical_layout(leaf)?.representation
                == RocmPhysicalRepresentation::UniformScalar
            {
                Ok(mask | (1_u8 << index))
            } else {
                Ok(mask)
            }
        })
}

impl<'graph> RocmPreparedTensorGraph<'graph> {
    /// Backend-neutral selected-output schedule and storage facts used for this preparation.
    #[must_use]
    pub const fn tensor_plan(&self) -> &TensorExecutionPlan<'graph> {
        &self.plan
    }

    /// Backend-neutral selected lowering schedule used by `ROCm` preflight.
    #[must_use]
    pub const fn lowering_plan(&self) -> &TensorSelectedLoweringPlan<'graph> {
        &self.lowering_plan
    }
}

#[derive(Debug)]
struct GraphExecutionPreflight<'a> {
    tensor_plan: TensorExecutionPlan<'a>,
    lowering_plan: TensorSelectedLoweringPlan<'a>,
    nodes: Vec<NodeDescriptor<'a>>,
    index_by_value: HashMap<ValueId, usize>,
    use_counts: Vec<usize>,
    fused_add_by_relu: HashMap<ValueId, (ValueId, ValueId)>,
    bounded_pointwise_by_output: HashMap<ValueId, TensorBoundedPointwiseFusionGroup>,
    bounded_mul_by_output: HashMap<ValueId, TensorBoundedMulFusionGroup>,
    fixed_dispatches: Vec<Option<PreparedFixedTensorDispatch>>,
    suppressed_adds: HashSet<ValueId>,
    #[cfg(test)]
    storage_constraints: Vec<TensorStorageConstraint>,
    indexed_storage_constraints: Vec<PreparedStorageConstraint>,
    matmul_operands: Vec<Option<PreparedMatMulOperands>>,
    strict_sgd: Vec<Option<strict_sgd::StrictSgdSpec>>,
    physical_layouts: HashMap<ValueId, RocmPhysicalLayout>,
}

#[cfg(test)]
fn prepare_graph<'a, A: TensorOperationAssessor>(
    graph: &'a Graph,
    output: ValueId,
    assessor: &A,
) -> Result<GraphExecutionPreflight<'a>, RocmTensorExecutionError> {
    prepare_graph_outputs_plan(graph, &[output], assessor)
}

#[cfg(test)]
fn prepare_graph_outputs_plan<'a, A: TensorOperationAssessor>(
    graph: &'a Graph,
    outputs: &[ValueId],
    assessor: &A,
) -> Result<GraphExecutionPreflight<'a>, RocmTensorExecutionError> {
    prepare_graph_outputs_plan_with_arithmetic(
        graph,
        outputs,
        assessor,
        TensorArithmeticRewritePolicy::Disabled,
        TensorArithmeticCapability::Strict,
    )
}

#[cfg(test)]
fn prepare_graph_outputs_plan_with_arithmetic<'a, A: TensorOperationAssessor>(
    graph: &'a Graph,
    outputs: &[ValueId],
    assessor: &A,
    policy: TensorArithmeticRewritePolicy,
    arithmetic: TensorArithmeticCapability,
) -> Result<GraphExecutionPreflight<'a>, RocmTensorExecutionError> {
    prepare_graph_outputs_plan_with_policies(
        graph,
        outputs,
        assessor,
        policy,
        arithmetic,
        TensorPointwiseGroupingPolicy::Disabled,
    )
}

#[allow(clippy::too_many_lines)] // Keeps ordered capability, fusion, and storage preflight in one cold pass.
fn prepare_graph_outputs_plan_with_policies<'a, A: TensorOperationAssessor>(
    graph: &'a Graph,
    outputs: &[ValueId],
    assessor: &A,
    policy: TensorArithmeticRewritePolicy,
    arithmetic: TensorArithmeticCapability,
    grouping: TensorPointwiseGroupingPolicy,
) -> Result<GraphExecutionPreflight<'a>, RocmTensorExecutionError> {
    let plan = graph
        .execution_plan_for_outputs(outputs)
        .map_err(|error| match error {
            TensorError::EmptyOutputs => RocmTensorExecutionError::EmptyOutputs,
            TensorError::DuplicateOutput(value) => RocmTensorExecutionError::DuplicateOutput(value),
            other => RocmTensorExecutionError::Graph(other),
        })?;
    let lowering_plan = plan.select_lowering_with_grouping(policy, arithmetic, grouping);
    let nodes = lowering_plan.nodes().to_vec();
    for node in &nodes {
        let expected_route = match node.op {
            OpDescriptor::ReluBackward { .. }
                if relu_backward::Profile::from_node(*node)
                    .is_ok_and(relu_backward::Profile::checked) =>
            {
                TensorExecutionRoute::Synthesized
            }
            OpDescriptor::SgdUpdate { .. }
            | OpDescriptor::MatMul { .. }
            | OpDescriptor::MeanSquaredError { .. }
                if node.numerical_mode == Some(fusion_pcu::PcuNumericalMode::Strict) =>
            {
                TensorExecutionRoute::Synthesized
            }
            OpDescriptor::Input
            | OpDescriptor::Constant(_)
            | OpDescriptor::Uniform { .. }
            | OpDescriptor::ReluBackward { .. }
            | OpDescriptor::SgdUpdate { .. } => TensorExecutionRoute::Native,
            OpDescriptor::MatMul { .. } | OpDescriptor::MeanSquaredError { .. } => {
                TensorExecutionRoute::Library
            }
            OpDescriptor::Add { .. }
            | OpDescriptor::Sub { .. }
            | OpDescriptor::Mul { .. }
            | OpDescriptor::Div { .. }
            | OpDescriptor::Relu { .. } => TensorExecutionRoute::Synthesized,
        };
        match assessor.assess_node(graph, *node) {
            TensorOperationSupport::Supported { route, .. } if route == expected_route => {}
            TensorOperationSupport::Supported { .. } => {
                return Err(RocmTensorExecutionError::Unsupported {
                    value: node.value,
                    reason: TensorUnsupportedReason::Other(
                        "assessor selected a route this executor does not implement".into(),
                    ),
                });
            }
            TensorOperationSupport::Unsupported { reason } => {
                return Err(RocmTensorExecutionError::Unsupported {
                    value: node.value,
                    reason,
                });
            }
        }
        let (element_size, _) = scalar_layout(node.scalar_type)?;
        let _ = byte_len_for_size(node.shape, element_size)?;
    }
    let index_by_value = nodes
        .iter()
        .enumerate()
        .map(|(index, node)| (node.value, index))
        .collect();
    let use_counts = nodes
        .iter()
        .map(|node| {
            lowering_plan
                .operation_index_of(node.value)
                .map_or(0, |index| lowering_plan.operation_use_counts()[index])
        })
        .collect();
    let fusion_maps = selected_fusion_maps(&lowering_plan);
    let fused_add_by_relu = fusion_maps.fused_add_by_relu;
    let bounded_pointwise_by_output = fusion_maps.bounded_pointwise_by_output;
    let bounded_mul_by_output = fusion_maps.bounded_mul_by_output;
    if let Some(&value) = bounded_pointwise_by_output
        .keys()
        .chain(bounded_mul_by_output.keys())
        .next()
    {
        if is_checked_float_type(graph.node(value)?.scalar_type) {
            return Err(unchecked_float_fusion_error(value));
        }
        return Err(RocmTensorExecutionError::Unsupported {
            value,
            reason: TensorUnsupportedReason::ElementType,
        });
    }
    if let Some(group) = lowering_plan
        .pointwise_fusion_groups()
        .iter()
        .find(|group| {
            graph
                .node(group.add_output)
                .is_ok_and(|node| is_checked_float_type(node.scalar_type))
        })
    {
        return Err(unchecked_float_fusion_error(group.add_output));
    }
    let suppressed_adds = fusion_maps.suppressed;
    let compact_uniform_values = compact_uniform_candidates(
        graph,
        &nodes,
        outputs,
        assessor,
        &bounded_pointwise_by_output,
        &bounded_mul_by_output,
    );
    let physical_layouts = physical_layouts(&nodes, compact_uniform_values)?;
    let fixed_dispatches = prepare_fixed_dispatches(
        &nodes,
        &index_by_value,
        &fused_add_by_relu,
        &bounded_pointwise_by_output,
        &bounded_mul_by_output,
        &suppressed_adds,
        &physical_layouts,
    )?;
    let storage_constraints = physicalize_storage_constraints(
        lowering_plan.operation_storage_constraints()?,
        &physical_layouts,
    )?;
    #[cfg(test)]
    let test_storage_constraints = storage_constraints.clone();
    let indexed_storage_constraints =
        prepare_storage_constraint_indices(storage_constraints, &index_by_value, outputs)?;
    let matmul_operands = prepare_matmul_operands(graph, &nodes, &index_by_value)?;
    let strict_sgd = nodes
        .iter()
        .map(|node| strict_sgd::StrictSgdSpec::from_node(graph, *node))
        .collect();
    Ok(GraphExecutionPreflight {
        tensor_plan: plan,
        lowering_plan,
        nodes,
        index_by_value,
        use_counts,
        fused_add_by_relu,
        bounded_pointwise_by_output,
        bounded_mul_by_output,
        fixed_dispatches,
        suppressed_adds,
        #[cfg(test)]
        storage_constraints: test_storage_constraints,
        indexed_storage_constraints,
        matmul_operands,
        strict_sgd,
        physical_layouts,
    })
}

// Preparation performs one cold validation/projection pass, keeping all derived facts in a
// lifetime-free record so warm owned calls only walk the selected schedule.
#[allow(clippy::too_many_lines)]
fn prepare_owned_graph_data<A: TensorOperationAssessor>(
    program: &fusion_pcu::dialect::tensor::TensorOwnedSelectedProgram,
    assessor: &A,
) -> Result<RocmPreparedGraphData, RocmTensorExecutionError> {
    let graph = program.graph();
    let outputs = program.output_values();
    let nodes = program
        .selected_nodes()
        .iter()
        .map(|&value| graph.node(value).map_err(RocmTensorExecutionError::from))
        .collect::<Result<Vec<_>, _>>()?;
    let scalar_type = homogeneous_scalar_type(&nodes);
    let transport_only_inputs = !nodes.is_empty()
        && nodes
            .iter()
            .all(|node| matches!(node.op, OpDescriptor::Input))
        && scalar_type.is_some_and(is_transport_scalar);
    if nodes
        .iter()
        .all(|node| matches!(node.op, OpDescriptor::Input))
        && !nodes.is_empty()
        && scalar_type.is_none()
    {
        return Err(RocmTensorExecutionError::UnsupportedScalarType(
            nodes[0].scalar_type,
        ));
    }
    if let Some(scalar_type) = scalar_type {
        let integer_profile = is_checked_integer_scalar(scalar_type)
            && nodes.iter().all(|node| {
                matches!(
                    node.op,
                    OpDescriptor::Input
                        | OpDescriptor::Constant(_)
                        | OpDescriptor::Uniform { .. }
                        | OpDescriptor::Add { .. }
                        | OpDescriptor::Sub { .. }
                        | OpDescriptor::Mul { .. }
                )
            });
        let raw_float_profile = is_raw_float_type(scalar_type)
            && nodes.iter().all(|node| {
                matches!(
                    node.op,
                    OpDescriptor::Input | OpDescriptor::Constant(_) | OpDescriptor::Uniform { .. }
                )
            });
        if !is_checked_float_type(scalar_type)
            && !transport_only_inputs
            && !integer_profile
            && !raw_float_profile
        {
            return Err(RocmTensorExecutionError::UnsupportedScalarType(scalar_type));
        }
    }
    for node in &nodes {
        let expected_route = match node.op {
            OpDescriptor::ReluBackward { .. }
                if relu_backward::Profile::from_node(*node)
                    .is_ok_and(relu_backward::Profile::checked) =>
            {
                TensorExecutionRoute::Synthesized
            }
            OpDescriptor::SgdUpdate { .. }
            | OpDescriptor::MatMul { .. }
            | OpDescriptor::MeanSquaredError { .. }
                if node.numerical_mode == Some(fusion_pcu::PcuNumericalMode::Strict) =>
            {
                TensorExecutionRoute::Synthesized
            }
            OpDescriptor::Input
            | OpDescriptor::Constant(_)
            | OpDescriptor::Uniform { .. }
            | OpDescriptor::ReluBackward { .. }
            | OpDescriptor::SgdUpdate { .. } => TensorExecutionRoute::Native,
            OpDescriptor::MatMul { .. } | OpDescriptor::MeanSquaredError { .. } => {
                TensorExecutionRoute::Library
            }
            OpDescriptor::Add { .. }
            | OpDescriptor::Sub { .. }
            | OpDescriptor::Mul { .. }
            | OpDescriptor::Div { .. }
            | OpDescriptor::Relu { .. } => TensorExecutionRoute::Synthesized,
        };
        match assessor.assess_node(graph, *node) {
            TensorOperationSupport::Supported { route, .. } if route == expected_route => {}
            TensorOperationSupport::Supported { .. } => {
                return Err(RocmTensorExecutionError::Unsupported {
                    value: node.value,
                    reason: TensorUnsupportedReason::Other(
                        "assessor selected a route this executor does not implement".into(),
                    ),
                });
            }
            TensorOperationSupport::Unsupported { reason } => {
                return Err(RocmTensorExecutionError::Unsupported {
                    value: node.value,
                    reason,
                });
            }
        }
        let (element_size, _) = scalar_layout(node.scalar_type)?;
        let _ = byte_len_for_size(node.shape, element_size)?;
    }

    let index_by_value = nodes
        .iter()
        .enumerate()
        .map(|(index, node)| (node.value, index))
        .collect::<HashMap<_, _>>();
    let use_counts = nodes
        .iter()
        .map(|node| {
            program
                .operation_index_of(node.value)
                .map_or(0, |index| program.operation_use_counts()[index])
        })
        .collect();
    let mut fused_add_by_relu = HashMap::new();
    let bounded_pointwise_by_output = HashMap::new();
    let bounded_mul_by_output = HashMap::new();
    for operation in program.operations() {
        match operation {
            fusion_pcu::dialect::tensor::TensorOwnedSelectedOperation::FusedAddRelu {
                add_output,
                relu_output,
                left,
                right,
            } => {
                if is_checked_float_type(program.graph().node(*add_output)?.scalar_type) {
                    return Err(unchecked_float_fusion_error(*add_output));
                }
                fused_add_by_relu.insert(*relu_output, (*left, *right));
            }
            fusion_pcu::dialect::tensor::TensorOwnedSelectedOperation::FusedAddSub { group } => {
                if !is_checked_float_type(graph.node(group.output)?.scalar_type) {
                    return Err(RocmTensorExecutionError::Unsupported {
                        value: group.output,
                        reason: TensorUnsupportedReason::ElementType,
                    });
                }
                return Err(unchecked_float_fusion_error(group.output));
            }
            fusion_pcu::dialect::tensor::TensorOwnedSelectedOperation::FusedMul { group } => {
                if !is_checked_float_type(graph.node(group.output)?.scalar_type) {
                    return Err(RocmTensorExecutionError::Unsupported {
                        value: group.output,
                        reason: TensorUnsupportedReason::ElementType,
                    });
                }
                return Err(unchecked_float_fusion_error(group.output));
            }
            fusion_pcu::dialect::tensor::TensorOwnedSelectedOperation::Node { .. } => {}
        }
    }
    let suppressed_adds = program.suppressed_values().iter().copied().collect();
    let compact_uniform_values = compact_uniform_candidates(
        graph,
        &nodes,
        outputs,
        assessor,
        &bounded_pointwise_by_output,
        &bounded_mul_by_output,
    );
    let physical_layouts = physical_layouts(&nodes, compact_uniform_values)?;
    let fixed_dispatches = prepare_fixed_dispatches(
        &nodes,
        &index_by_value,
        &fused_add_by_relu,
        &bounded_pointwise_by_output,
        &bounded_mul_by_output,
        &suppressed_adds,
        &physical_layouts,
    )?;
    let storage_constraints = physicalize_storage_constraints(
        program.operation_storage_constraints().to_vec(),
        &physical_layouts,
    )?;
    let indexed_storage_constraints =
        prepare_storage_constraint_indices(storage_constraints, &index_by_value, outputs)?;
    let matmul_operands = prepare_matmul_operands(graph, &nodes, &index_by_value)?;
    let strict_sgd = nodes
        .iter()
        .map(|node| strict_sgd::StrictSgdSpec::from_node(graph, *node))
        .collect();
    Ok(RocmPreparedGraphData {
        scalar_type,
        requires_blas: nodes_require_blas(&nodes),
        batch_native_matmuls: native_execution::batch_native_matmuls(&nodes, outputs),
        transport_only_inputs,
        consuming_action: prepare_consuming_action(program, &nodes)?,
        node_values: program.selected_nodes().to_vec(),
        output: outputs[0],
        outputs: outputs.to_vec(),
        index_by_value,
        use_counts,
        fused_add_by_relu,
        bounded_pointwise_by_output,
        bounded_mul_by_output,
        fixed_dispatches,
        suppressed_adds,
        indexed_storage_constraints,
        matmul_operands,
        strict_sgd,
        physical_layouts,
        rewrites: program.rewrites().to_vec(),
        input_values: program.input_values().to_vec(),
    })
}

fn prepare_consuming_action(
    program: &fusion_pcu::dialect::tensor::TensorOwnedSelectedProgram,
    nodes: &[NodeDescriptor<'_>],
) -> Result<Option<PreparedConsumingAction>, RocmTensorExecutionError> {
    if nodes
        .iter()
        .any(|node| matches!(node.op, OpDescriptor::Relu { .. }))
    {
        // Checked ReLU can fault for non-finite operands or underflow policy violations. Keep its
        // destination fresh so a failed completion never leaves the caller's input overwritten.
        return Ok(None);
    }
    if nodes.iter().any(|node| {
        (is_checked_integer_scalar(node.scalar_type)
            && matches!(
                node.op,
                OpDescriptor::Add { .. }
                    | OpDescriptor::Sub { .. }
                    | OpDescriptor::Mul { .. }
                    | OpDescriptor::Div { .. }
            ))
            || is_checked_float_binary_node(*node)
    }) {
        return Ok(None);
    }
    if program.output_values().len() != 1 {
        return Ok(None);
    }
    if program.input_values().len() == 2 {
        return consuming::prepare_consuming_binary_action(program, nodes);
    }
    if program.input_values().len() != 1 {
        return Ok(None);
    }
    let input = program.input_values()[0];
    let output = program.output_values()[0];
    if input == output
        && program.selected_nodes().len() == 1
        && nodes.len() == 1
        && nodes[0].value == input
        && matches!(nodes[0].op, OpDescriptor::Input)
    {
        return Ok(Some(PreparedConsumingAction::IdentityTransfer(input)));
    }
    if program.selected_nodes().len() != 2 || nodes.len() != 2 {
        return Ok(None);
    }
    let Ok(proof) = program.prove_consumed_relu_reuse(input, output) else {
        return Ok(None);
    };
    let Some(input_node) = nodes.iter().find(|node| node.value == input) else {
        return Ok(None);
    };
    if input_node.scalar_type != proof.scalar_type() {
        return Ok(None);
    }
    let scalar_type = TensorPointwiseScalarType::try_from(proof.scalar_type())?;
    let logical_count = input_node.shape.iter().try_fold(1_u32, |count, &extent| {
        u32::try_from(extent)
            .ok()
            .and_then(|extent| count.checked_mul(extent))
    });
    let Some(logical_count) = logical_count.filter(|count| *count > 0) else {
        // A zero-element shape has no valid InvocationCount dispatch geometry. It keeps the
        // ordinary fresh-output behavior instead of defining a fake in-place no-op allocation.
        return Ok(None);
    };
    let invocation_count =
        NonZeroU32::new(logical_count).ok_or(RocmTensorExecutionError::SizeOverflow)?;
    Ok(Some(PreparedConsumingAction::TerminalRelu(
        PreparedConsumingRelu {
            proof,
            kernel: pointwise::consuming_relu_kernel(scalar_type, logical_count),
            invocation_shape: PcuInvocationShape::invocations(invocation_count),
            scalar_type,
            value_type: scalar_type.value_type(),
            binding: PcuBindingRef::new(0, 0),
            logical_count,
        },
    )))
}

fn prepare_storage_constraint_indices(
    constraints: Vec<TensorStorageConstraint>,
    index_by_value: &HashMap<ValueId, usize>,
    outputs: &[ValueId],
) -> Result<Vec<PreparedStorageConstraint>, RocmTensorExecutionError> {
    constraints
        .into_iter()
        .map(|constraint| {
            let left_index = index_by_value
                .get(&constraint.left)
                .copied()
                .ok_or(RocmTensorExecutionError::InvalidPlan(constraint.left))?;
            let right_index = index_by_value
                .get(&constraint.right)
                .copied()
                .ok_or(RocmTensorExecutionError::InvalidPlan(constraint.right))?;
            Ok(PreparedStorageConstraint {
                constraint,
                left_index,
                right_index,
                left_output_index: output_position(outputs, constraint.left),
                right_output_index: output_position(outputs, constraint.right),
            })
        })
        .collect()
}

fn prepare_matmul_operands(
    graph: &Graph,
    nodes: &[NodeDescriptor<'_>],
    index_by_value: &HashMap<ValueId, usize>,
) -> Result<Vec<Option<PreparedMatMulOperands>>, RocmTensorExecutionError> {
    nodes
        .iter()
        .map(|node| {
            let OpDescriptor::MatMul { left, right, .. } = node.op else {
                return Ok(None);
            };
            let left_index = index_by_value
                .get(&left)
                .copied()
                .ok_or(RocmTensorExecutionError::InvalidPlan(left))?;
            let right_index = index_by_value
                .get(&right)
                .copied()
                .ok_or(RocmTensorExecutionError::InvalidPlan(right))?;
            let left_shape = graph
                .shape(left)?
                .try_into()
                .map_err(|_| RocmTensorExecutionError::InvalidPlan(left))?;
            let right_shape = graph
                .shape(right)?
                .try_into()
                .map_err(|_| RocmTensorExecutionError::InvalidPlan(right))?;
            Ok(Some(PreparedMatMulOperands {
                left_index,
                right_index,
                left_shape,
                right_shape,
                strict: strict_matmul::StrictMatMulSpec::from_node(graph, *node),
            }))
        })
        .collect()
}

fn selected_fusion_maps(lowering_plan: &TensorSelectedLoweringPlan<'_>) -> SelectedFusionMaps {
    let fused_add_by_relu = lowering_plan
        .pointwise_fusion_groups()
        .iter()
        .map(|group| (group.relu_output, (group.left, group.right)))
        .collect();
    let bounded_pointwise_by_output = lowering_plan
        .bounded_pointwise_fusion_groups()
        .iter()
        .cloned()
        .map(|group| (group.output, group))
        .collect::<HashMap<_, _>>();
    let bounded_mul_by_output = lowering_plan
        .bounded_mul_fusion_groups()
        .iter()
        .cloned()
        .map(|group| (group.output, group))
        .collect::<HashMap<_, _>>();
    let mut suppressed: HashSet<ValueId> = lowering_plan
        .pointwise_fusion_groups()
        .iter()
        .map(|group| group.add_output)
        .collect();
    suppressed.extend(suppressed_bounded_pointwise_values(
        lowering_plan.bounded_pointwise_fusion_groups(),
    ));
    suppressed.extend(suppressed_mul_values(bounded_mul_by_output.values()));
    SelectedFusionMaps {
        fused_add_by_relu,
        bounded_pointwise_by_output,
        bounded_mul_by_output,
        suppressed,
    }
}

struct SelectedFusionMaps {
    fused_add_by_relu: HashMap<ValueId, (ValueId, ValueId)>,
    bounded_pointwise_by_output: HashMap<ValueId, TensorBoundedPointwiseFusionGroup>,
    bounded_mul_by_output: HashMap<ValueId, TensorBoundedMulFusionGroup>,
    suppressed: HashSet<ValueId>,
}

fn suppressed_bounded_pointwise_values(
    groups: &[TensorBoundedPointwiseFusionGroup],
) -> impl Iterator<Item = ValueId> + '_ {
    groups.iter().flat_map(|group| {
        group.steps.iter().filter_map(move |step| {
            (group.epilogue == TensorPointwiseEpilogue::Relu || step.output != group.output)
                .then_some(step.output)
        })
    })
}

fn suppressed_mul_values<'a>(
    groups: impl Iterator<Item = &'a TensorBoundedMulFusionGroup> + 'a,
) -> impl Iterator<Item = ValueId> + 'a {
    groups.flat_map(|group| {
        group
            .steps
            .iter()
            .filter_map(move |step| (step.output != group.output).then_some(step.output))
    })
}

fn compact_uniform_candidates<A: TensorOperationAssessor>(
    graph: &Graph,
    nodes: &[NodeDescriptor<'_>],
    outputs: &[ValueId],
    assessor: &A,
    bounded_groups: &HashMap<ValueId, TensorBoundedPointwiseFusionGroup>,
    bounded_mul_groups: &HashMap<ValueId, TensorBoundedMulFusionGroup>,
) -> HashSet<ValueId> {
    let source_nodes = graph.nodes().collect::<Vec<_>>();
    nodes
        .iter()
        .filter(|node| {
            matches!(node.op, OpDescriptor::Uniform { .. }) && !outputs.contains(&node.value)
        })
        .filter_map(|uniform| {
            let consumers_are_elementwise = nodes
                .iter()
                .filter(|node| node_consumes(node.op, uniform.value))
                .all(|consumer| {
                    assessor.supports_operand_representation(
                        graph,
                        *consumer,
                        uniform.value,
                        TensorOperandRepresentation::UniformScalar,
                    )
                })
                && bounded_groups.values().all(|group| {
                    group
                        .steps
                        .iter()
                        .filter(|step| pointwise_step_uses_leaf(group, step, uniform.value))
                        .all(|step| {
                            source_nodes
                                .iter()
                                .find(|node| node.value == step.output)
                                .is_some_and(|consumer| {
                                    assessor.supports_operand_representation(
                                        graph,
                                        *consumer,
                                        uniform.value,
                                        TensorOperandRepresentation::UniformScalar,
                                    )
                })
                && bounded_mul_groups.values().all(|group| group.steps.iter()
                    .filter(|step| matches!(step.left, TensorPointwiseOperand::Leaf(i) if group.leaves.get(i) == Some(&uniform.value))
                        || matches!(step.right, TensorPointwiseOperand::Leaf(i) if group.leaves.get(i) == Some(&uniform.value)))
                    .all(|step| source_nodes.iter().find(|node| node.value == step.output).is_some_and(|consumer|
                        assessor.supports_operand_representation(graph, *consumer, uniform.value, TensorOperandRepresentation::UniformScalar))))
                        })
                });
            consumers_are_elementwise.then_some(uniform.value)
        })
        .collect()
}

fn pointwise_step_uses_leaf(
    group: &TensorBoundedPointwiseFusionGroup,
    step: &fusion_pcu::dialect::tensor::TensorPointwiseStep,
    value: ValueId,
) -> bool {
    let is_leaf = |operand| matches!(operand, TensorPointwiseOperand::Leaf(index) if group.leaves.get(index) == Some(&value));
    is_leaf(step.left) || is_leaf(step.right)
}

fn node_consumes(op: OpDescriptor<'_>, value: ValueId) -> bool {
    match op {
        OpDescriptor::Add { left, right }
        | OpDescriptor::Sub { left, right }
        | OpDescriptor::Mul { left, right }
        | OpDescriptor::Div { left, right }
        | OpDescriptor::MatMul { left, right, .. }
        | OpDescriptor::MeanSquaredError {
            prediction: left,
            target: right,
        } => left == value || right == value,
        OpDescriptor::Relu { input } => input == value,
        OpDescriptor::ReluBackward { input, upstream } => input == value || upstream == value,
        OpDescriptor::SgdUpdate {
            weights, gradient, ..
        } => weights == value || gradient == value,
        OpDescriptor::Input | OpDescriptor::Constant(_) | OpDescriptor::Uniform { .. } => false,
    }
}

fn physicalize_storage_constraints(
    constraints: Vec<TensorStorageConstraint>,
    physical_layouts: &HashMap<ValueId, RocmPhysicalLayout>,
) -> Result<Vec<TensorStorageConstraint>, RocmTensorExecutionError> {
    let extent = |value: ValueId, logical_bytes: u64| -> Result<u64, RocmTensorExecutionError> {
        let layout = physical_layouts
            .get(&value)
            .ok_or(RocmTensorExecutionError::InvalidPlan(value))?;
        if layout.representation == RocmPhysicalRepresentation::Dense
            && layout.physical_bytes != logical_bytes
        {
            return Err(RocmTensorExecutionError::InvalidPlan(value));
        }
        Ok(layout.physical_bytes)
    };
    constraints
        .into_iter()
        .map(|constraint| {
            Ok(TensorStorageConstraint {
                left: constraint.left,
                right: constraint.right,
                left_bytes: extent(constraint.left, constraint.left_bytes)?,
                right_bytes: extent(constraint.right, constraint.right_bytes)?,
            })
        })
        .collect()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RocmPhysicalRepresentation {
    Dense,
    UniformScalar,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct RocmPhysicalLayout {
    representation: RocmPhysicalRepresentation,
    physical_bytes: u64,
}

impl RocmPhysicalLayout {
    const fn dense(physical_bytes: u64) -> Self {
        Self {
            representation: RocmPhysicalRepresentation::Dense,
            physical_bytes,
        }
    }

    const fn uniform_scalar() -> Self {
        Self {
            representation: RocmPhysicalRepresentation::UniformScalar,
            physical_bytes: size_of::<f32>() as u64,
        }
    }

    fn uniform_tensor(
        self,
        logical_shape: &[usize],
        value: f32,
    ) -> Result<Tensor, RocmTensorExecutionError> {
        match self.representation {
            RocmPhysicalRepresentation::Dense => {
                Tensor::splat(logical_shape.to_vec(), value).map_err(Into::into)
            }
            RocmPhysicalRepresentation::UniformScalar => Ok(Tensor::scalar(value)),
        }
    }
}

fn physical_layouts(
    nodes: &[NodeDescriptor<'_>],
    compact_uniform_values: impl IntoIterator<Item = ValueId>,
) -> Result<HashMap<ValueId, RocmPhysicalLayout>, RocmTensorExecutionError> {
    let compact = compact_uniform_values
        .into_iter()
        .collect::<std::collections::HashSet<_>>();
    nodes
        .iter()
        .map(|node| {
            let layout = if compact.contains(&node.value) {
                RocmPhysicalLayout::uniform_scalar()
            } else {
                let (element_size, _) = scalar_layout(node.scalar_type)?;
                let bytes = u64::try_from(byte_len_for_size(node.shape, element_size)?)
                    .map_err(|_| RocmTensorExecutionError::SizeOverflow)?;
                RocmPhysicalLayout::dense(bytes)
            };
            Ok((node.value, layout))
        })
        .collect()
}

#[allow(clippy::too_many_arguments, clippy::too_many_lines)] // All per-node dispatch facts are finalized together during cold graph preparation.
fn prepare_fixed_dispatches(
    nodes: &[NodeDescriptor<'_>],
    index_by_value: &HashMap<ValueId, usize>,
    fused_add_by_relu: &HashMap<ValueId, (ValueId, ValueId)>,
    bounded_pointwise_by_output: &HashMap<ValueId, TensorBoundedPointwiseFusionGroup>,
    bounded_mul_by_output: &HashMap<ValueId, TensorBoundedMulFusionGroup>,
    suppressed_adds: &HashSet<ValueId>,
    physical_layouts: &HashMap<ValueId, RocmPhysicalLayout>,
) -> Result<Vec<Option<PreparedFixedTensorDispatch>>, RocmTensorExecutionError> {
    nodes
        .iter()
        .map(|node| {
            if suppressed_adds.contains(&node.value) {
                return Ok(None);
            }
            let (kind, shape, scalar_mask) = match node.op {
                OpDescriptor::Add { left, right }
                    if !bounded_pointwise_by_output.contains_key(&node.value)
                        && !bounded_mul_by_output.contains_key(&node.value) =>
                {
                    (
                        binary_dispatch_kind(
                            TensorDispatchKind::Add,
                            TensorPointwiseScalarType::try_from(node.scalar_type)?,
                        ),
                        node.shape,
                        fixed_scalar_mask(left, right, nodes, index_by_value, physical_layouts)?,
                    )
                }
                OpDescriptor::Sub { left, right }
                    if !bounded_pointwise_by_output.contains_key(&node.value)
                        && !bounded_mul_by_output.contains_key(&node.value) =>
                {
                    (
                        binary_dispatch_kind(
                            TensorDispatchKind::Sub,
                            TensorPointwiseScalarType::try_from(node.scalar_type)?,
                        ),
                        node.shape,
                        fixed_scalar_mask(left, right, nodes, index_by_value, physical_layouts)?,
                    )
                }
                OpDescriptor::Mul { left, right }
                    if !bounded_pointwise_by_output.contains_key(&node.value)
                        && !bounded_mul_by_output.contains_key(&node.value) =>
                {
                    (
                        binary_dispatch_kind(
                            TensorDispatchKind::Mul,
                            TensorPointwiseScalarType::try_from(node.scalar_type)?,
                        ),
                        node.shape,
                        fixed_scalar_mask(left, right, nodes, index_by_value, physical_layouts)?,
                    )
                }
                OpDescriptor::Div { left, right } => (
                    TensorDispatchKind::CheckedFloatDiv,
                    node.shape,
                    fixed_scalar_mask(left, right, nodes, index_by_value, physical_layouts)?,
                ),
                OpDescriptor::Relu { .. }
                    if !bounded_pointwise_by_output.contains_key(&node.value) =>
                {
                    if let Some(&(left, right)) = fused_add_by_relu.get(&node.value) {
                        (
                            TensorDispatchKind::AddRelu,
                            node.shape,
                            fixed_scalar_mask(
                                left,
                                right,
                                nodes,
                                index_by_value,
                                physical_layouts,
                            )?,
                        )
                    } else {
                        (TensorDispatchKind::Relu, node.shape, 0)
                    }
                }
                _ => return Ok(None),
            };
            let logical_count = flattened_invocation_count(shape)?;
            let invocations =
                NonZeroU32::new(logical_count).ok_or(RocmTensorExecutionError::SizeOverflow)?;
            let scalar_type = TensorPointwiseScalarType::try_from(node.scalar_type)?;
            let numerical_requirements = fixed_numerical_requirements(*node);
            let cache_key = TensorDispatchCacheKey::Fixed(
                kind,
                scalar_type,
                logical_count,
                scalar_mask,
                numerical_requirements,
            );
            let kernel = PcuDispatchKernelIr {
                numerical_requirements,
                ..kind.kernel(
                    scalar_type,
                    logical_count,
                    scalar_mask,
                    Some(numerical_requirements.float_underflow),
                )?
            };
            let value_type = scalar_type.value_type();
            let (left_binding, right_binding, output_binding) = fixed_binding_refs(kind);
            Ok(Some(PreparedFixedTensorDispatch {
                value: node.value,
                cache_key,
                kernel,
                invocation_shape: PcuInvocationShape::invocations(invocations),
                scalar_type,
                value_type,
                left_binding,
                right_binding,
                output_binding,
            }))
        })
        .collect()
}

fn fixed_scalar_mask(
    left: ValueId,
    right: ValueId,
    nodes: &[NodeDescriptor<'_>],
    index_by_value: &HashMap<ValueId, usize>,
    physical_layouts: &HashMap<ValueId, RocmPhysicalLayout>,
) -> Result<u8, RocmTensorExecutionError> {
    let is_compact_uniform = |value| -> Result<bool, RocmTensorExecutionError> {
        let index = index_by_value
            .get(&value)
            .copied()
            .ok_or(RocmTensorExecutionError::InvalidPlan(value))?;
        let node = nodes
            .get(index)
            .ok_or(RocmTensorExecutionError::InvalidPlan(value))?;
        Ok(matches!(node.op, OpDescriptor::Uniform { .. })
            && physical_layouts.get(&value).is_some_and(|layout| {
                layout.representation == RocmPhysicalRepresentation::UniformScalar
            }))
    };
    Ok(u8::from(is_compact_uniform(left)?) | (u8::from(is_compact_uniform(right)?) << 1))
}

const fn fixed_binding_refs(
    kind: TensorDispatchKind,
) -> (PcuBindingRef, Option<PcuBindingRef>, PcuBindingRef) {
    match kind {
        TensorDispatchKind::Add
        | TensorDispatchKind::AddRelu
        | TensorDispatchKind::Sub
        | TensorDispatchKind::Mul
        | TensorDispatchKind::CheckedIntegerAdd
        | TensorDispatchKind::CheckedIntegerSub
        | TensorDispatchKind::CheckedIntegerMul
        | TensorDispatchKind::CheckedFloatAdd
        | TensorDispatchKind::CheckedFloatSub
        | TensorDispatchKind::CheckedFloatMul
        | TensorDispatchKind::CheckedFloatDiv => {
            (ADD_LEFT_REF, Some(ADD_RIGHT_REF), ADD_OUTPUT_REF)
        }
        TensorDispatchKind::Relu => (RELU_INPUT_REF, None, RELU_OUTPUT_REF),
    }
}

#[cfg(test)]
fn validate_graph_inputs(
    nodes: &[NodeDescriptor<'_>],
    inputs: &[(ValueId, Tensor)],
) -> Result<(), TensorError> {
    for (index, (id, tensor)) in inputs.iter().enumerate() {
        if inputs[..index].iter().any(|(previous, _)| previous == id) {
            return Err(TensorError::DuplicateInput(*id));
        }
        let Some(node) = nodes.iter().find(|node| node.value == *id) else {
            return Err(TensorError::ExtraInput(*id));
        };
        if !matches!(node.op, OpDescriptor::Input) {
            return Err(TensorError::ExtraInput(*id));
        }
        if tensor.shape() != node.shape {
            return Err(TensorError::ShapeMismatch {
                left: tensor.shape().to_vec(),
                right: node.shape.to_vec(),
            });
        }
    }
    for node in nodes {
        if matches!(node.op, OpDescriptor::Input) && !inputs.iter().any(|(id, _)| *id == node.value)
        {
            return Err(TensorError::MissingInput(node.value));
        }
    }
    Ok(())
}

fn validate_graph_input_sources<I: RocmTensorInputDescriptor>(
    prepared: &RocmPreparedGraphView<'_, '_>,
    host_inputs: &[(ValueId, Tensor)],
    resource_inputs: &[(ValueId, &I)],
    session: &RocmOwnedDispatchBackend,
    pool: PcuMemoryPoolId,
) -> Result<(), RocmTensorExecutionError> {
    for (index, (id, tensor)) in host_inputs.iter().enumerate() {
        if host_inputs[..index]
            .iter()
            .any(|(previous, _)| previous == id)
        {
            return Err(TensorError::DuplicateInput(*id).into());
        }
        let Some(index) = prepared.index_of(*id) else {
            return Err(TensorError::ExtraInput(*id).into());
        };
        let node = prepared.node(index)?;
        if !matches!(node.op, OpDescriptor::Input) {
            return Err(TensorError::ExtraInput(*id).into());
        }
        if node.scalar_type != fusion_pcu::PcuScalarType::F32 {
            return Err(RocmTensorExecutionError::UnsupportedScalarType(
                node.scalar_type,
            ));
        }
        if tensor.shape() != node.shape {
            return Err(TensorError::ShapeMismatch {
                left: tensor.shape().to_vec(),
                right: node.shape.to_vec(),
            }
            .into());
        }
    }
    for (index, (id, input)) in resource_inputs.iter().enumerate() {
        if resource_inputs[..index]
            .iter()
            .any(|(previous, _)| previous == id)
            || host_inputs.iter().any(|(previous, _)| previous == id)
        {
            return Err(TensorError::DuplicateInput(*id).into());
        }
        let Some(index) = prepared.index_of(*id) else {
            return Err(TensorError::ExtraInput(*id).into());
        };
        let node = prepared.node(index)?;
        if !matches!(node.op, OpDescriptor::Input) {
            return Err(TensorError::ExtraInput(*id).into());
        }
        if node.scalar_type != input.scalar_type() {
            return Err(RocmTensorExecutionError::UnsupportedScalarType(
                input.scalar_type(),
            ));
        }
        if node.shape != input.shape() {
            return Err(TensorError::ShapeMismatch {
                left: node.shape.to_vec(),
                right: input.shape().to_vec(),
            }
            .into());
        }
        if !alignment_satisfies(
            input.resource().alignment_bytes(),
            scalar_layout(node.scalar_type)?.1,
        ) {
            return Err(RocmTensorExecutionError::InputResourceMismatch);
        }
        // This path only accepts Input nodes; compact uniforms are materialized by the executor.
        input_storage_requirement(*id, node.shape, node.scalar_type)?
            .validate(input.resource())
            .map_err(RocmTensorExecutionError::StorageConstraint)?;
        if !std::ptr::eq(input.session(), session)
            || !input
                .resource()
                .belongs_to_runtime(session.tensor_runtime())
        {
            return Err(RocmTensorExecutionError::InputResourceMismatch);
        }
        if input.resource().pool() != pool {
            return Err(RocmTensorExecutionError::InputPoolMismatch);
        }
        if !matches!(
            input.resource().access(),
            PcuMemoryAccess::ReadOnly | PcuMemoryAccess::ReadWrite
        ) {
            return Err(RocmTensorExecutionError::InputResourceMismatch);
        }
    }
    for value in prepared.source_input_values() {
        if !host_inputs.iter().any(|(id, _)| id == value)
            && !resource_inputs.iter().any(|(id, _)| id == value)
        {
            return Err(TensorError::MissingInput(*value).into());
        }
    }
    Ok(())
}

fn byte_len(shape: &[usize]) -> Result<usize, RocmTensorExecutionError> {
    byte_len_for_size(shape, size_of::<f32>())
}

fn byte_len_for<T: fusion_pcu::PcuScalar>(
    shape: &[usize],
) -> Result<usize, RocmTensorExecutionError> {
    byte_len_for_size(shape, size_of::<T>())
}

fn byte_len_for_size(
    shape: &[usize],
    element_size: usize,
) -> Result<usize, RocmTensorExecutionError> {
    let bytes = shape.iter().try_fold(element_size, |bytes, dimension| {
        bytes
            .checked_mul(*dimension)
            .ok_or(RocmTensorExecutionError::SizeOverflow)
    })?;
    if bytes > isize::MAX as usize {
        return Err(RocmTensorExecutionError::SizeOverflow);
    }
    Ok(bytes)
}

const fn scalar_layout(
    scalar_type: fusion_pcu::PcuScalarType,
) -> Result<(usize, u64), RocmTensorExecutionError> {
    match scalar_type {
        fusion_pcu::PcuScalarType::I8
        | fusion_pcu::PcuScalarType::U8
        | fusion_pcu::PcuScalarType::F8E4M3FN
        | fusion_pcu::PcuScalarType::F8E5M2 => Ok((1, 1)),
        fusion_pcu::PcuScalarType::I16
        | fusion_pcu::PcuScalarType::U16
        | fusion_pcu::PcuScalarType::F16
        | fusion_pcu::PcuScalarType::BF16 => Ok((2, 2)),
        fusion_pcu::PcuScalarType::I32
        | fusion_pcu::PcuScalarType::U32
        | fusion_pcu::PcuScalarType::F32 => Ok((4, 4)),
        fusion_pcu::PcuScalarType::I64
        | fusion_pcu::PcuScalarType::U64
        | fusion_pcu::PcuScalarType::F64 => Ok((8, 8)),
        fusion_pcu::PcuScalarType::I128 => {
            Ok((size_of::<i128>(), core::mem::align_of::<i128>() as u64))
        }
        fusion_pcu::PcuScalarType::U128 => {
            Ok((size_of::<u128>(), core::mem::align_of::<u128>() as u64))
        }
        fusion_pcu::PcuScalarType::I256 => Ok((
            size_of::<fusion_pcu::PcuI256>(),
            core::mem::align_of::<fusion_pcu::PcuI256>() as u64,
        )),
        fusion_pcu::PcuScalarType::U256 => Ok((
            size_of::<fusion_pcu::PcuU256>(),
            core::mem::align_of::<fusion_pcu::PcuU256>() as u64,
        )),
        fusion_pcu::PcuScalarType::I512 => Ok((
            size_of::<fusion_pcu::PcuI512>(),
            core::mem::align_of::<fusion_pcu::PcuI512>() as u64,
        )),
        fusion_pcu::PcuScalarType::U512 => Ok((
            size_of::<fusion_pcu::PcuU512>(),
            core::mem::align_of::<fusion_pcu::PcuU512>() as u64,
        )),
        fusion_pcu::PcuScalarType::F128 => Ok((
            size_of::<fusion_pcu::PcuF128Bits>(),
            core::mem::align_of::<fusion_pcu::PcuF128Bits>() as u64,
        )),
        fusion_pcu::PcuScalarType::F256 => Ok((
            size_of::<fusion_pcu::PcuF256Bits>(),
            core::mem::align_of::<fusion_pcu::PcuF256Bits>() as u64,
        )),
        unsupported => Err(RocmTensorExecutionError::UnsupportedScalarType(unsupported)),
    }
}

const fn is_transport_scalar(scalar_type: fusion_pcu::PcuScalarType) -> bool {
    matches!(
        scalar_type,
        fusion_pcu::PcuScalarType::I8
            | fusion_pcu::PcuScalarType::U8
            | fusion_pcu::PcuScalarType::I16
            | fusion_pcu::PcuScalarType::U16
            | fusion_pcu::PcuScalarType::I32
            | fusion_pcu::PcuScalarType::U32
            | fusion_pcu::PcuScalarType::I64
            | fusion_pcu::PcuScalarType::U64
            | fusion_pcu::PcuScalarType::F16
            | fusion_pcu::PcuScalarType::BF16
            | fusion_pcu::PcuScalarType::F8E4M3FN
            | fusion_pcu::PcuScalarType::F8E5M2
            | fusion_pcu::PcuScalarType::F32
            | fusion_pcu::PcuScalarType::F64
            | fusion_pcu::PcuScalarType::I128
            | fusion_pcu::PcuScalarType::U128
            | fusion_pcu::PcuScalarType::I256
            | fusion_pcu::PcuScalarType::U256
            | fusion_pcu::PcuScalarType::I512
            | fusion_pcu::PcuScalarType::U512
            | fusion_pcu::PcuScalarType::F128
            | fusion_pcu::PcuScalarType::F256
    )
}

const fn is_checked_integer_scalar(scalar_type: fusion_pcu::PcuScalarType) -> bool {
    matches!(
        scalar_type,
        fusion_pcu::PcuScalarType::I8
            | fusion_pcu::PcuScalarType::U8
            | fusion_pcu::PcuScalarType::I16
            | fusion_pcu::PcuScalarType::U16
            | fusion_pcu::PcuScalarType::I32
            | fusion_pcu::PcuScalarType::U32
            | fusion_pcu::PcuScalarType::I64
            | fusion_pcu::PcuScalarType::U64
            | fusion_pcu::PcuScalarType::I128
            | fusion_pcu::PcuScalarType::U128
            | fusion_pcu::PcuScalarType::I256
            | fusion_pcu::PcuScalarType::U256
            | fusion_pcu::PcuScalarType::I512
            | fusion_pcu::PcuScalarType::U512
    )
}

const fn is_checked_float_binary_node(node: NodeDescriptor<'_>) -> bool {
    is_checked_float_type(node.scalar_type)
        && matches!(
            node.op,
            OpDescriptor::Add { .. }
                | OpDescriptor::Sub { .. }
                | OpDescriptor::Mul { .. }
                | OpDescriptor::Div { .. }
        )
}

// IEEE 754 binary128 and binary256 encodings are transport-only here.
// Admitting immutable bits does not authorize arithmetic, conversion or PortableV1.
const fn is_raw_float_type(scalar_type: fusion_pcu::PcuScalarType) -> bool {
    matches!(
        scalar_type,
        fusion_pcu::PcuScalarType::F128 | fusion_pcu::PcuScalarType::F256
    )
}

const fn is_low_float_type(scalar_type: fusion_pcu::PcuScalarType) -> bool {
    matches!(
        scalar_type,
        fusion_pcu::PcuScalarType::F16
            | fusion_pcu::PcuScalarType::BF16
            | fusion_pcu::PcuScalarType::F8E4M3FN
            | fusion_pcu::PcuScalarType::F8E5M2
    )
}

const fn is_checked_float_type(scalar_type: fusion_pcu::PcuScalarType) -> bool {
    matches!(
        scalar_type,
        fusion_pcu::PcuScalarType::F16
            | fusion_pcu::PcuScalarType::BF16
            | fusion_pcu::PcuScalarType::F8E4M3FN
            | fusion_pcu::PcuScalarType::F8E5M2
            | fusion_pcu::PcuScalarType::F32
            | fusion_pcu::PcuScalarType::F64
    )
}

fn unchecked_float_fusion_error(value: ValueId) -> RocmTensorExecutionError {
    RocmTensorExecutionError::Unsupported {
        value,
        reason: TensorUnsupportedReason::Other(
            "checked f32/f64 Add/Sub/Mul/Div cannot execute through unchecked fusion".into(),
        ),
    }
}

const fn alignment_satisfies(reported: u64, required: u64) -> bool {
    required != 0 && reported >= required && reported.is_multiple_of(required)
}

/// Formation is fixed nearest-even even when the permitted library reduction is native.
fn mse_scale(element_count: usize) -> Result<f32, RocmTensorExecutionError> {
    use fusion_pcu::PcuCheckedFloat;
    if element_count == 0 || element_count > i32::MAX as usize {
        return Err(RocmTensorExecutionError::SizeOverflow);
    }
    1.0_f32
        .pcu_checked_div(fusion_pcu::dialect::tensor::constants::count_f32(
            element_count,
        ))
        .map_err(|_| RocmTensorExecutionError::SizeOverflow)
}

/// Native F64 reduction uses an exact integer count (bounded by i32) and a F64 reciprocal.
fn mse_scale_f64(count: usize) -> Result<f64, RocmTensorExecutionError> {
    use fusion_pcu::PcuCheckedFloat;
    let count = u32::try_from(count)
        .ok()
        .filter(|n| *n > 0 && i32::try_from(*n).is_ok())
        .ok_or(RocmTensorExecutionError::SizeOverflow)?;
    1.0_f64
        .pcu_checked_div(fusion_pcu::dialect::tensor::constants::count_f64(
            count as usize,
        ))
        .map_err(|_| RocmTensorExecutionError::SizeOverflow)
}

fn allocate_tensor<P: PcuMemoryProvider<Resource = RocmMemoryResource>>(
    memory: &mut P,
    pool: PcuMemoryPoolId,
    shape: &[usize],
) -> Result<RocmMemoryResource, RocmTensorExecutionError> {
    allocate_tensor_for_size(memory, pool, shape, size_of::<f32>(), align_of::<f32>())
}

fn allocate_tensor_for<
    T: fusion_pcu::PcuScalar,
    P: PcuMemoryProvider<Resource = RocmMemoryResource>,
>(
    memory: &mut P,
    pool: PcuMemoryPoolId,
    shape: &[usize],
) -> Result<RocmMemoryResource, RocmTensorExecutionError> {
    allocate_tensor_for_size(memory, pool, shape, size_of::<T>(), align_of::<T>())
}

fn allocate_tensor_for_size<P: PcuMemoryProvider<Resource = RocmMemoryResource>>(
    memory: &mut P,
    pool: PcuMemoryPoolId,
    shape: &[usize],
    element_size: usize,
    alignment: usize,
) -> Result<RocmMemoryResource, RocmTensorExecutionError> {
    let size_bytes = u64::try_from(byte_len_for_size(shape, element_size)?)
        .map_err(|_| RocmTensorExecutionError::SizeOverflow)?;
    let size_bytes_usize =
        usize::try_from(size_bytes).map_err(|_| RocmTensorExecutionError::SizeOverflow)?;
    let alignment_bytes =
        u64::try_from(alignment).map_err(|_| RocmTensorExecutionError::SizeOverflow)?;
    let resource = memory.allocate(PcuMemoryAllocationRequest {
        pool,
        size_bytes,
        alignment_bytes,
        access: PcuMemoryAccess::ReadWrite,
        host_access: PcuMemoryHostAccess::TransferOnly,
        require_device_local: false,
    })?;
    if resource.pool() != pool
        || resource.size_bytes() < size_bytes
        || !alignment_satisfies(resource.alignment_bytes(), alignment_bytes)
        || resource.access() != PcuMemoryAccess::ReadWrite
        || resource.device_buffer().len() < size_bytes_usize
    {
        return Err(RocmTensorExecutionError::OutputResourceMismatch);
    }
    Ok(resource)
}

fn upload_tensor<P: PcuMemoryProvider<Resource = RocmMemoryResource>>(
    memory: &mut P,
    pool: PcuMemoryPoolId,
    tensor: &Tensor,
) -> Result<RocmMemoryResource, RocmTensorExecutionError> {
    let mut resource = allocate_tensor(memory, pool, tensor.shape())?;
    transfer_tensor(memory, &mut resource, tensor)?;
    Ok(resource)
}

fn download_tensor<P: PcuMemoryProvider<Resource = RocmMemoryResource>>(
    memory: &mut P,
    output: &RocmTensorInput<'_>,
) -> Result<Tensor, RocmTensorExecutionError> {
    let bytes_len = byte_len(output.shape.as_slice())?;
    let mut data = vec![0.0_f32; bytes_len / size_of::<f32>()];
    // SAFETY: `data` is initialized, contiguous f32 storage. Every f32 bit pattern is valid;
    // u8 has alignment one, and its byte length was checked from the output shape.
    let output_bytes =
        unsafe { std::slice::from_raw_parts_mut(data.as_mut_ptr().cast::<u8>(), bytes_len) };
    memory.transfer_from(&output.resource, 0, output_bytes)?;
    Ok(Tensor::new(output.shape.as_slice().to_vec(), data)?)
}

fn transfer_tensor<P: PcuMemoryProvider<Resource = RocmMemoryResource>>(
    memory: &mut P,
    resource: &mut RocmMemoryResource,
    tensor: &Tensor,
) -> Result<(), RocmTensorExecutionError> {
    let bytes_len = byte_len(tensor.shape())?;
    // SAFETY: Tensor construction checks that shape and initialized f32 data agree. f32 has
    // exactly four bytes with no padding, u8 has alignment one, and the byte extent is checked.
    // The provider's synchronous transfer ends before the Tensor can be released by this call.
    let bytes =
        unsafe { std::slice::from_raw_parts(tensor.data().as_ptr().cast::<u8>(), bytes_len) };
    memory.transfer_to(resource, 0, bytes)?;
    Ok(())
}

fn execution_resource_slots(node_count: usize) -> TensorExecutionResources {
    let mut resources = TensorExecutionResources::new();
    resources.resize_with(node_count, || None);
    resources
}

fn release_after_read(
    resources: &mut [Option<RocmMemoryResource>],
    remaining_uses: &mut [usize],
    index: usize,
) -> Result<(), RocmTensorExecutionError> {
    let Some(remaining) = remaining_uses.get_mut(index) else {
        return Err(RocmTensorExecutionError::SizeOverflow);
    };
    *remaining = remaining
        .checked_sub(1)
        .ok_or(RocmTensorExecutionError::SizeOverflow)?;
    if *remaining == 0 {
        resources[index] = None;
    }
    Ok(())
}

fn release_bounded_pointwise_leaves(
    plan: &RocmPreparedGraphView<'_, '_>,
    group: &TensorBoundedPointwiseFusionGroup,
    resources: &mut [Option<RocmMemoryResource>],
    remaining_uses: &mut [usize],
) -> Result<(), RocmTensorExecutionError> {
    for (leaf_index, &leaf) in group.leaves.iter().enumerate() {
        let leaf_plan_index = plan
            .index_of(leaf)
            .ok_or(RocmTensorExecutionError::InvalidPlan(leaf))?;
        let uses = group
            .steps
            .iter()
            .flat_map(|step| [step.left, step.right])
            .filter(|operand| *operand == TensorPointwiseOperand::Leaf(leaf_index))
            .count();
        for _ in 0..uses {
            release_after_read(resources, remaining_uses, leaf_plan_index)?;
        }
    }
    Ok(())
}

fn release_bounded_mul_leaves(
    plan: &RocmPreparedGraphView<'_, '_>,
    group: &TensorBoundedMulFusionGroup,
    resources: &mut [Option<RocmMemoryResource>],
    remaining_uses: &mut [usize],
) -> Result<(), RocmTensorExecutionError> {
    for (leaf_slot, &leaf) in group.leaves.iter().enumerate() {
        let plan_index = plan
            .index_of(leaf)
            .ok_or(RocmTensorExecutionError::InvalidPlan(leaf))?;
        // A repeated leaf can occur multiple times in one kernel. Consume every original graph
        // edge so its pooled allocation is released at the same point as unfused execution.
        let uses = group
            .steps
            .iter()
            .flat_map(|step| [step.left, step.right])
            .filter(|operand| *operand == TensorPointwiseOperand::Leaf(leaf_slot))
            .count();
        for _ in 0..uses {
            release_after_read(resources, remaining_uses, plan_index)?;
        }
    }
    Ok(())
}

// SAFETY: `Rocblas::sgemm` acquires shared exclusive allocation leases for A/B/C and performs a
// HIP device synchronization before returning. If synchronization fails, it poisons the handle
// and deliberately forgets all leases, retaining the allocations indefinitely. All validation
// errors occur before rocBLAS is called, so no device work can outlive a returned error.
unsafe impl TensorSynchronousF32MatMulBackend for RocmTensorAssessor<'_> {
    type Resource = RocmMemoryResource;
    type Error = RocmTensorError;

    fn matmul_row_major(
        &self,
        a: &Self::Resource,
        b: &Self::Resource,
        c: &Self::Resource,
        rows: usize,
        inner: usize,
        columns: usize,
    ) -> Result<(), Self::Error> {
        self.execute_matmul_row_major(a, b, c, rows, inner, columns)
    }
}

impl TensorOperationAssessor for RocmTensorAssessor<'_> {
    fn assess_node(&self, graph: &Graph, node: NodeDescriptor<'_>) -> TensorOperationSupport {
        // Pure numerical admission precedes lazy library initialization or device allocation.
        assess_tensor_node(graph, node)
    }

    fn supports_operand_representation(
        &self,
        _graph: &Graph,
        node: NodeDescriptor<'_>,
        _operand: ValueId,
        representation: TensorOperandRepresentation,
    ) -> bool {
        rocm_supports_operand_representation(node, representation)
    }
}

const fn rocm_supports_operand_representation(
    node: NodeDescriptor<'_>,
    representation: TensorOperandRepresentation,
) -> bool {
    if matches!(node.op, OpDescriptor::Input) {
        return matches!(representation, TensorOperandRepresentation::Dense)
            && is_transport_scalar(node.scalar_type);
    }
    if is_checked_integer_scalar(node.scalar_type) {
        return matches!(representation, TensorOperandRepresentation::Dense)
            && matches!(
                node.numerical_options.reproducibility,
                fusion_pcu::PcuReproducibility::Unspecified
            )
            && matches!(
                node.op,
                OpDescriptor::Constant(_)
                    | OpDescriptor::Uniform { .. }
                    | OpDescriptor::Add { .. }
                    | OpDescriptor::Sub { .. }
                    | OpDescriptor::Mul { .. }
            );
    }
    if is_raw_float_type(node.scalar_type) {
        return matches!(representation, TensorOperandRepresentation::Dense)
            && matches!(
                node.numerical_options.reproducibility,
                fusion_pcu::PcuReproducibility::Unspecified
            )
            && matches!(
                node.op,
                OpDescriptor::Constant(_) | OpDescriptor::Uniform { .. }
            );
    }
    if is_low_float_type(node.scalar_type) {
        return matches!(representation, TensorOperandRepresentation::Dense)
            && matches!(
                node.op,
                OpDescriptor::Constant(_)
                    | OpDescriptor::Uniform { .. }
                    | OpDescriptor::Add { .. }
                    | OpDescriptor::Sub { .. }
                    | OpDescriptor::Mul { .. }
                    | OpDescriptor::Div { .. }
                    | OpDescriptor::Relu { .. }
                    | OpDescriptor::ReluBackward { .. }
            );
    }
    let supported_scalar = matches!(
        node.scalar_type,
        fusion_pcu::PcuScalarType::F32 | fusion_pcu::PcuScalarType::F64
    );
    if !supported_scalar {
        return false;
    }
    match representation {
        TensorOperandRepresentation::Dense => matches!(
            node.op,
            OpDescriptor::Input
                | OpDescriptor::MatMul { .. }
                | OpDescriptor::MeanSquaredError { .. }
                | OpDescriptor::Add { .. }
                | OpDescriptor::Sub { .. }
                | OpDescriptor::Mul { .. }
                | OpDescriptor::Div { .. }
                | OpDescriptor::Relu { .. }
                | OpDescriptor::ReluBackward { .. }
                | OpDescriptor::Constant(_)
                | OpDescriptor::Uniform { .. }
        ),
        TensorOperandRepresentation::UniformScalar => {
            // Checked kernels load dense operands by invocation ID; they have no scalar binding profile.
            !is_checked_float_binary_node(node)
                && matches!(node.scalar_type, fusion_pcu::PcuScalarType::F32)
                && matches!(
                    node.op,
                    OpDescriptor::Add { .. }
                        | OpDescriptor::Sub { .. }
                        | OpDescriptor::Mul { .. }
                        | OpDescriptor::Div { .. }
                )
        }
    }
}

fn assess_float_matmul(graph: &Graph, node: NodeDescriptor<'_>) -> TensorOperationSupport {
    if graph.node(node.value).ok() != Some(node) {
        return TensorOperationSupport::Unsupported {
            reason: TensorUnsupportedReason::Operation,
        };
    }
    // A permitted native contract may use the stronger ordered Strict implementation.
    // Test this before the Boundary-only rocBLAS path; no native fast-path work changes.
    if strict_matmul::StrictMatMulSpec::from_node(graph, node).is_some() {
        return TensorOperationSupport::Supported {
            route: TensorExecutionRoute::Synthesized,
            workspace_bytes: Some(0),
        };
    }
    if node.numerical_options.compound_arithmetic
        == fusion_pcu::PcuCompoundArithmeticPolicy::BackendDefined
    {
        if let Err(reason) = native_policy::assess_underflow(node.float_underflow_policy) {
            return TensorOperationSupport::Unsupported { reason };
        }
        if node.numerical_mode != Some(fusion_pcu::PcuNumericalMode::Boundary) {
            return TensorOperationSupport::Unsupported {
                reason: TensorUnsupportedReason::NumericalPolicy {
                    requirement: fusion_pcu::PcuNumericalRequirement::CompoundArithmetic,
                    options: node.numerical_options,
                },
            };
        }
        // Explicit native compound arithmetic permits rocBLAS' reduction/contraction and
        // numerical exception behavior. Typed SGEMM/DGEMM retain the declared storage and
        // arithmetic precision; optimized permission may use this stronger implementation.
        // rocBLAS' typed GEMM table assigns SGEMM f32_r and DGEMM f64_r compute types;
        // mixed/HPA compute is a separate gemm_ex choice, not selected by this adapter:
        // https://rocm.docs.amd.com/projects/rocBLAS/en/latest/how-to/Programmers_Guide.html
        // No checked status kernel, strict synthesis or reproducibility restriction is added.
        // API failures, shape/extent validation, leases and terminal publication remain checked.
        return TensorOperationSupport::Supported {
            route: TensorExecutionRoute::Library,
            workspace_bytes: None,
        };
    }
    TensorOperationSupport::Unsupported {
        reason: TensorUnsupportedReason::Other(
            "checked result-boundary MatMul is not implemented; select strict checking or explicitly permit native compound arithmetic".into(),
        ),
    }
}

fn assess_tensor_node(graph: &Graph, node: NodeDescriptor<'_>) -> TensorOperationSupport {
    // The scalar dispatch Portable qualification does not certify an owned tensor program.
    if node.numerical_options.reproducibility != fusion_pcu::PcuReproducibility::Unspecified {
        return TensorOperationSupport::Unsupported {
            reason: TensorUnsupportedReason::NumericalPolicy {
                requirement: fusion_pcu::PcuNumericalRequirement::Reproducibility,
                options: node.numerical_options,
            },
        };
    }
    if matches!(node.op, OpDescriptor::MeanSquaredError { .. })
        && node.numerical_mode == Some(fusion_pcu::PcuNumericalMode::Strict)
    {
        return strict_mse::assess(graph, node);
    }
    if matches!(node.op, OpDescriptor::MeanSquaredError { .. }) {
        return native_loss::assess(graph, node);
    }
    if matches!(node.op, OpDescriptor::SgdUpdate { .. }) {
        return if strict_sgd::StrictSgdSpec::from_node(graph, node).is_some() {
            TensorOperationSupport::Supported {
                route: TensorExecutionRoute::Synthesized,
                workspace_bytes: Some(0),
            }
        } else {
            native_sgd::assess(graph, node)
        };
    }
    if matches!(node.op, OpDescriptor::ReluBackward { .. }) {
        return relu_backward::assess(graph, node);
    }
    if let OpDescriptor::MatMul {
        left,
        right,
        transpose_left,
        transpose_right,
    } = node.op
        && node.numerical_mode != Some(fusion_pcu::PcuNumericalMode::Strict)
        && !matmul_shape_supported(
            graph,
            node.shape,
            left,
            right,
            transpose_left,
            transpose_right,
        )
    {
        return TensorOperationSupport::Unsupported {
            reason: TensorUnsupportedReason::Shape,
        };
    }
    if matches!(node.op, OpDescriptor::MatMul { .. })
        && matches!(
            node.scalar_type,
            fusion_pcu::PcuScalarType::F32 | fusion_pcu::PcuScalarType::F64
        )
    {
        return assess_float_matmul(graph, node);
    }
    if matches!(node.op, OpDescriptor::Input) && is_transport_scalar(node.scalar_type) {
        return TensorOperationSupport::Supported {
            route: TensorExecutionRoute::Native,
            workspace_bytes: Some(0),
        };
    }
    if is_checked_integer_scalar(node.scalar_type) {
        return assess_checked_integer_node(graph, node);
    }
    if is_raw_float_type(node.scalar_type) {
        return literal::raw_float::assess(graph, node);
    }
    if is_low_float_type(node.scalar_type) {
        return assess_low_float_node(graph, node);
    }
    if node.scalar_type == fusion_pcu::PcuScalarType::F64 {
        return assess_f64_pointwise_or_literal_node(graph, node);
    }
    if !matches!(
        node.scalar_type,
        fusion_pcu::PcuScalarType::F32 | fusion_pcu::PcuScalarType::F64
    ) {
        return TensorOperationSupport::Unsupported {
            reason: TensorUnsupportedReason::ElementType,
        };
    }
    assess_f32_tensor_node(graph, node)
}

// F64 producers are exact dense transport; checked scalar consumers keep their existing law.
fn assess_f64_pointwise_or_literal_node(
    graph: &Graph,
    node: NodeDescriptor<'_>,
) -> TensorOperationSupport {
    if node.shape.contains(&0)
        && matches!(
            node.op,
            OpDescriptor::Constant(_) | OpDescriptor::Uniform { .. }
        )
    {
        return TensorOperationSupport::Unsupported {
            reason: TensorUnsupportedReason::Shape,
        };
    }
    if !matches!(
        node.op,
        OpDescriptor::Input
            | OpDescriptor::Constant(_)
            | OpDescriptor::Uniform { .. }
            | OpDescriptor::MatMul { .. }
            | OpDescriptor::Add { .. }
            | OpDescriptor::Sub { .. }
            | OpDescriptor::Mul { .. }
            | OpDescriptor::Div { .. }
            | OpDescriptor::Relu { .. }
    ) {
        return TensorOperationSupport::Unsupported {
            reason: TensorUnsupportedReason::ElementType,
        };
    }
    assess_f32_tensor_node(graph, node)
}

fn assess_low_float_node(graph: &Graph, node: NodeDescriptor<'_>) -> TensorOperationSupport {
    if matches!(
        node.op,
        OpDescriptor::Constant(_) | OpDescriptor::Uniform { .. }
    ) {
        let count = node
            .shape
            .iter()
            .try_fold(1_usize, |n, &d| n.checked_mul(d));
        if !count.is_some_and(|n| n > 0 && u32::try_from(n).is_ok()) {
            return TensorOperationSupport::Unsupported {
                reason: TensorUnsupportedReason::Shape,
            };
        }
        if graph.node(node.value).ok() != Some(node) {
            return TensorOperationSupport::Unsupported {
                reason: TensorUnsupportedReason::Operation,
            };
        }
        return TensorOperationSupport::Supported {
            route: TensorExecutionRoute::Native,
            workspace_bytes: Some(0),
        };
    }
    let operands = match node.op {
        OpDescriptor::Add { left, right }
        | OpDescriptor::Sub { left, right }
        | OpDescriptor::Mul { left, right }
        | OpDescriptor::Div { left, right } => [Some(left), Some(right)],
        OpDescriptor::Relu { input } => [Some(input), None],
        _ => {
            return TensorOperationSupport::Unsupported {
                reason: TensorUnsupportedReason::ElementType,
            };
        }
    };
    let count = node
        .shape
        .iter()
        .try_fold(1usize, |count, dim| count.checked_mul(*dim));
    if !count.is_some_and(|count| count > 0 && u32::try_from(count).is_ok()) {
        return TensorOperationSupport::Unsupported {
            reason: TensorUnsupportedReason::Shape,
        };
    }
    for value in operands.into_iter().flatten() {
        let Ok(operand) = graph.node(value) else {
            return TensorOperationSupport::Unsupported {
                reason: TensorUnsupportedReason::ElementType,
            };
        };
        if operand.scalar_type != node.scalar_type
            || operand.shape != node.shape
            || !matches!(
                operand.op,
                OpDescriptor::Input
                    | OpDescriptor::Constant(_)
                    | OpDescriptor::Uniform { .. }
                    | OpDescriptor::Add { .. }
                    | OpDescriptor::Sub { .. }
                    | OpDescriptor::Mul { .. }
                    | OpDescriptor::Div { .. }
                    | OpDescriptor::Relu { .. }
            )
        {
            return TensorOperationSupport::Unsupported {
                reason: TensorUnsupportedReason::ElementType,
            };
        }
    }
    TensorOperationSupport::Supported {
        route: TensorExecutionRoute::Synthesized,
        workspace_bytes: Some(0),
    }
}

fn assess_checked_integer_node(graph: &Graph, node: NodeDescriptor<'_>) -> TensorOperationSupport {
    if matches!(
        node.op,
        OpDescriptor::Constant(_) | OpDescriptor::Uniform { .. }
    ) {
        let count = node
            .shape
            .iter()
            .try_fold(1_usize, |n, &d| n.checked_mul(d));
        if !count.is_some_and(|n| n > 0 && u32::try_from(n).is_ok()) {
            return TensorOperationSupport::Unsupported {
                reason: TensorUnsupportedReason::Shape,
            };
        }
        if graph.node(node.value).ok() != Some(node) {
            return TensorOperationSupport::Unsupported {
                reason: TensorUnsupportedReason::Operation,
            };
        }
        return TensorOperationSupport::Supported {
            route: TensorExecutionRoute::Native,
            workspace_bytes: Some(0),
        };
    }
    let operands = match node.op {
        OpDescriptor::Add { left, right }
        | OpDescriptor::Sub { left, right }
        | OpDescriptor::Mul { left, right } => [left, right],
        OpDescriptor::Input => unreachable!("integer inputs are admitted by transport assessment"),
        _ => {
            return TensorOperationSupport::Unsupported {
                reason: TensorUnsupportedReason::ElementType,
            };
        }
    };
    let element_count = node
        .shape
        .iter()
        .try_fold(1usize, |count, dimension| count.checked_mul(*dimension));
    if !element_count.is_some_and(|count| count > 0 && u32::try_from(count).is_ok()) {
        return TensorOperationSupport::Unsupported {
            reason: TensorUnsupportedReason::Shape,
        };
    }
    for value in operands {
        let Ok(operand) = graph.node(value) else {
            return TensorOperationSupport::Unsupported {
                reason: TensorUnsupportedReason::ElementType,
            };
        };
        if operand.scalar_type != node.scalar_type
            || operand.shape != node.shape
            || !matches!(
                operand.op,
                OpDescriptor::Input
                    | OpDescriptor::Constant(_)
                    | OpDescriptor::Uniform { .. }
                    | OpDescriptor::Add { .. }
                    | OpDescriptor::Sub { .. }
                    | OpDescriptor::Mul { .. }
            )
        {
            return TensorOperationSupport::Unsupported {
                reason: TensorUnsupportedReason::ElementType,
            };
        }
    }
    TensorOperationSupport::Supported {
        route: TensorExecutionRoute::Synthesized,
        workspace_bytes: Some(0),
    }
}

fn assess_dense_binary_shape(
    graph: &Graph,
    shape: &[usize],
    left: ValueId,
    right: ValueId,
) -> TensorOperationSupport {
    match (graph.shape(left), graph.shape(right)) {
        (Ok(left), Ok(right))
            if left == right
                && left
                    .iter()
                    .try_fold(1usize, |count, dimension| count.checked_mul(*dimension))
                    .is_some_and(|count| count > 0 && u32::try_from(count).is_ok())
                && shape == left =>
        {
            TensorOperationSupport::Supported {
                route: TensorExecutionRoute::Synthesized,
                workspace_bytes: Some(0),
            }
        }
        _ => TensorOperationSupport::Unsupported {
            reason: TensorUnsupportedReason::Shape,
        },
    }
}

fn assess_f32_tensor_node(graph: &Graph, node: NodeDescriptor<'_>) -> TensorOperationSupport {
    match node.op {
        OpDescriptor::Uniform { .. } if node.shape.contains(&0) => {
            TensorOperationSupport::Unsupported {
                reason: TensorUnsupportedReason::Shape,
            }
        }
        OpDescriptor::Input | OpDescriptor::Constant(_) | OpDescriptor::Uniform { .. } => {
            TensorOperationSupport::Supported {
                route: TensorExecutionRoute::Native,
                workspace_bytes: Some(0),
            }
        }
        OpDescriptor::MatMul {
            left,
            right,
            transpose_left,
            transpose_right,
        } => {
            if matmul_shape_supported(
                graph,
                node.shape,
                left,
                right,
                transpose_left,
                transpose_right,
            ) {
                TensorOperationSupport::Supported {
                    route: TensorExecutionRoute::Library,
                    workspace_bytes: None,
                }
            } else {
                TensorOperationSupport::Unsupported {
                    reason: TensorUnsupportedReason::Shape,
                }
            }
        }
        OpDescriptor::Add { left, right }
        | OpDescriptor::Sub { left, right }
        | OpDescriptor::Mul { left, right }
        | OpDescriptor::Div { left, right } => {
            assess_dense_binary_shape(graph, node.shape, left, right)
        }
        OpDescriptor::Relu { input } => match graph.shape(input) {
            Ok(input_shape)
                if input_shape == node.shape
                    && input_shape
                        .iter()
                        .try_fold(1usize, |count, dimension| count.checked_mul(*dimension))
                        .is_some_and(|count| count > 0 && u32::try_from(count).is_ok()) =>
            {
                TensorOperationSupport::Supported {
                    route: TensorExecutionRoute::Synthesized,
                    workspace_bytes: Some(0),
                }
            }
            _ => TensorOperationSupport::Unsupported {
                reason: TensorUnsupportedReason::Shape,
            },
        },
        OpDescriptor::ReluBackward { input, upstream } => {
            assess_relu_backward(graph, node.shape, input, upstream)
        }
        OpDescriptor::SgdUpdate {
            weights, gradient, ..
        } => assess_sgd_update(graph, node.shape, weights, gradient),
        OpDescriptor::MeanSquaredError { prediction, target } => {
            match (graph.shape(prediction), graph.shape(target)) {
                (Ok(left), Ok(right)) if left == right && node.shape.is_empty() => {
                    let count = left.iter().try_fold(1usize, |n, d| n.checked_mul(*d));
                    if count.is_some_and(|n| n > 0 && i32::try_from(n).is_ok()) {
                        TensorOperationSupport::Supported {
                            route: TensorExecutionRoute::Library,
                            workspace_bytes: None,
                        }
                    } else {
                        TensorOperationSupport::Unsupported {
                            reason: TensorUnsupportedReason::Shape,
                        }
                    }
                }
                _ => TensorOperationSupport::Unsupported {
                    reason: TensorUnsupportedReason::Shape,
                },
            }
        }
    }
}

fn assess_relu_backward(
    graph: &Graph,
    output_shape: &[usize],
    input: ValueId,
    upstream: ValueId,
) -> TensorOperationSupport {
    match (graph.shape(input), graph.shape(upstream)) {
        (Ok(input_shape), Ok(upstream_shape))
            if input_shape == upstream_shape
                && input_shape == output_shape
                && input_shape
                    .iter()
                    .try_fold(1usize, |count, dimension| count.checked_mul(*dimension))
                    .is_some_and(|count| count > 0 && u32::try_from(count).is_ok()) =>
        {
            TensorOperationSupport::Supported {
                route: TensorExecutionRoute::Native,
                workspace_bytes: Some(0),
            }
        }
        _ => TensorOperationSupport::Unsupported {
            reason: TensorUnsupportedReason::Shape,
        },
    }
}

fn assess_sgd_update(
    graph: &Graph,
    output_shape: &[usize],
    weights: ValueId,
    gradient: ValueId,
) -> TensorOperationSupport {
    match (graph.shape(weights), graph.shape(gradient)) {
        (Ok(weights_shape), Ok(gradient_shape))
            if weights_shape == gradient_shape
                && weights_shape == output_shape
                && weights_shape
                    .iter()
                    .try_fold(1usize, |count, dimension| count.checked_mul(*dimension))
                    .is_some_and(|count| count > 0 && u32::try_from(count).is_ok()) =>
        {
            TensorOperationSupport::Supported {
                route: TensorExecutionRoute::Native,
                workspace_bytes: Some(0),
            }
        }
        _ => TensorOperationSupport::Unsupported {
            reason: TensorUnsupportedReason::Shape,
        },
    }
}

fn matmul_shape_supported(
    graph: &Graph,
    output_shape: &[usize],
    left: ValueId,
    right: ValueId,
    transpose_left: bool,
    transpose_right: bool,
) -> bool {
    let (Ok(left_node), Ok(right_node)) = (graph.node(left), graph.node(right)) else {
        return false;
    };
    let scalar_bytes = match (left_node.scalar_type, right_node.scalar_type) {
        (fusion_pcu::PcuScalarType::F32, fusion_pcu::PcuScalarType::F32) => 4,
        (fusion_pcu::PcuScalarType::F64, fusion_pcu::PcuScalarType::F64) => 8,
        _ => return false,
    };
    let (Ok([left_rows, left_columns]), Ok([right_rows, right_columns])) =
        (graph.shape(left), graph.shape(right))
    else {
        return false;
    };
    let rows = if transpose_left {
        *left_columns
    } else {
        *left_rows
    };
    let inner = if transpose_left {
        *left_rows
    } else {
        *left_columns
    };
    let right_inner = if transpose_right {
        *right_columns
    } else {
        *right_rows
    };
    let columns = if transpose_right {
        *right_rows
    } else {
        *right_columns
    };
    rows > 0
        && inner > 0
        && right_inner == inner
        && columns > 0
        && [
            *left_rows,
            *left_columns,
            *right_rows,
            *right_columns,
            rows,
            inner,
            columns,
        ]
        .into_iter()
        .all(|dimension| i32::try_from(dimension).is_ok())
        && [
            (*left_rows, *left_columns),
            (*right_rows, *right_columns),
            (rows, columns),
        ]
        .into_iter()
        .all(|(a, b)| {
            a.checked_mul(b)
                .and_then(|n| n.checked_mul(scalar_bytes))
                .is_some()
        })
        && output_shape == [rows, columns]
}

#[cfg(test)]
#[path = "tensor/numerical_tests.rs"]
mod numerical_tests;

#[cfg(all(test, any(target_arch = "x86_64", target_arch = "aarch64")))]
#[path = "tensor/cold_constants/cold_constants.rs"]
mod cold_constants;

#[cfg(test)]
mod tests {
    use super::native_execution;
    use fusion_pcu::PcuScalarType;
    use fusion_pcu::{PcuBf16Bits, PcuF16Bits, PcuScalar};
    use crate::RocmDiscovery;
    use std::cell::Cell;
    use std::sync::Arc;
    use std::time::Duration;

    #[rustfmt::skip]
    use fusion_pcu::{
        PcuDispatchOp,
        PcuDispatchDataOp,
        PcuDispatchKernelIr,
        PcuBindingType,
        PcuDeviceDescriptor,
        PcuDeviceTensor,
        PcuFloatUnderflowPolicy,
        PcuMemoryPoolId,
        PcuObjectKind,
        PcuObjectRef,
        PcuProviderDescriptor,
        PcuProviderId,
        PcuProviderReadiness,
        PcuProviderStatus,
        PcuRuntimeDiscovery,
        PcuTargetDescriptor,
        PcuValueType,
        PcuValueTypeCaps,
    };
    #[rustfmt::skip]
    use fusion_pcu::dialect::tensor::{
        Graph,
        NodeDescriptor,
        OpDescriptor,
        Tensor,
        TensorScalarValue,
        TensorValue,
        TensorError,
        TensorOperationAssessor,
        TensorOperationSupport,
        TensorStorageConstraint,
    };

    #[rustfmt::skip]
    use super::{
        ADD_LEFT_REF,
        ADD_OUTPUT_REF,
        ADD_RIGHT_REF,
        add_kernel,
        add_relu_kernel,
        assess_tensor_node,
        collect_dispatch_requests,
        byte_len_for,
        execution_resource_slots,
        homogeneous_scalar_type,
        is_transport_scalar,
        input_storage_requirement,
        validate_indexed_storage_constraints,
        retained_requested_key_count,
        mse_scratch_length_fits,
        mse_scratch_element_count,
        output_position,
        prepare_owned_graph_data,
        with_single_output_plan,
        prepare_consuming_action,
        prepare_graph,
        prepare_graph_outputs_plan,
        prepare_graph_outputs_plan_with_arithmetic,
        prepare_graph_outputs_plan_with_policies,
        relu_kernel,
        rocm_supports_operand_representation,
        scalar_layout,
        scratch_stores_node,
        selected_scratch_storage_plan,
        mse_scratch_resource_fits,
        nodes_require_blas,
        resource_may_overlap_any,
        validate_graph_inputs,
        validate_owned_scalar_profile,
        validate_tensor_scalar_tag,
        alignment_satisfies,
        CollectNodeTimings,
        CollectElementwiseTiming,
        ElementwisePhase,
        ElementwiseTimingSink,
        NodeTimingSink,
        RocmTensorExecutionError,
        PreparedConsumingAction,
        RocmPreparedGraphData,
        RocmPhysicalLayout,
        TensorExecutionRoute,
        TensorExecutionUseCounts,
        TensorDispatchCacheKey,
        TensorDispatchKind,
        TensorDispatchRequest,
        TensorPointwiseScalarType,
        binary_dispatch_kind,
        checked_float,
        pointwise,
        GraphExecutionPreflight,
        PreparedMatMulOperands,
        PreparedStorageConstraint,
        RocmPreparedTensorGraph,
        TENSOR_EXECUTION_INLINE_NODES,
        TensorArithmeticCapability,
        TensorArithmeticRewritePolicy,
        TensorOperandRepresentation,
        TensorPointwiseGroupingPolicy,
        TensorUnsupportedReason,
        ValueId,
        RocmOwnedDispatchBackend,
        RocmTensorAssessor,
    };

    const INVALID_ROCM_REFERENCE: PcuObjectRef = PcuObjectRef {
        provider: PcuProviderId(0),
        generation: 0,
        kind: PcuObjectKind::Target,
        id: 0,
    };

    const fn empty_test_provider() -> PcuProviderDescriptor<'static> {
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

    const fn empty_test_target() -> PcuTargetDescriptor<'static> {
        PcuTargetDescriptor {
            reference: INVALID_ROCM_REFERENCE,
            name: "",
            readiness: PcuProviderReadiness {
                status: PcuProviderStatus::Unavailable,
                reason: None,
            },
        }
    }

    const fn empty_test_device() -> PcuDeviceDescriptor<'static> {
        PcuDeviceDescriptor {
            reference: INVALID_ROCM_REFERENCE,
            target: INVALID_ROCM_REFERENCE,
            name: "",
            class: fusion_pcu::PcuDeviceClass::Other,
            vendor: None,
            architecture: None,
            generation: None,
            location: None,
        }
    }

    pub(super) fn rocm_test_session() -> (RocmDiscovery, RocmOwnedDispatchBackend) {
        let discovery = RocmDiscovery::new();
        let mut providers = [empty_test_provider()];
        assert_eq!(discovery.providers(&mut providers).unwrap(), 1);
        let mut targets = [empty_test_target()];
        assert_eq!(
            discovery
                .targets(providers[0].id, providers[0].generation, &mut targets)
                .unwrap(),
            1
        );
        let count = discovery.devices(targets[0].reference, &mut []).unwrap();
        let mut devices = vec![empty_test_device(); count];
        discovery
            .devices(targets[0].reference, &mut devices)
            .unwrap();
        let device = devices
            .first()
            .expect("ignored hardware test requires a visible ROCm device")
            .reference;
        let session = RocmOwnedDispatchBackend::open(&discovery, device, 64)
            .expect("open selected ROCm device");
        (discovery, session)
    }

    #[test]
    fn f32_and_f64_add_sub_mul_select_checked_fixed_dispatch_profiles() {
        for (operation, expected_kind) in [
            (TensorDispatchKind::Add, TensorDispatchKind::CheckedFloatAdd),
            (TensorDispatchKind::Sub, TensorDispatchKind::CheckedFloatSub),
            (TensorDispatchKind::Mul, TensorDispatchKind::CheckedFloatMul),
        ] {
            assert_eq!(
                binary_dispatch_kind(operation, TensorPointwiseScalarType::F32),
                expected_kind,
            );
            assert_eq!(
                binary_dispatch_kind(operation, TensorPointwiseScalarType::F64),
                expected_kind,
            );
            let float_op = match expected_kind {
                TensorDispatchKind::CheckedFloatAdd => fusion_pcu::PcuDispatchFloatBinaryOp::Add,
                TensorDispatchKind::CheckedFloatSub => fusion_pcu::PcuDispatchFloatBinaryOp::Sub,
                TensorDispatchKind::CheckedFloatMul => fusion_pcu::PcuDispatchFloatBinaryOp::Mul,
                _ => unreachable!(),
            };
            let kernel = checked_float::kernel(
                float_op,
                PcuScalarType::F32,
                PcuFloatUnderflowPolicy::default(),
                17,
            )
            .unwrap();
            assert!(kernel.entry.name.starts_with("tensor_checked_f32_"));
            assert_eq!(
                fusion_pcu::validate_checked_float_binary_kernel(
                    &kernel,
                    PcuValueType::f32(),
                    float_op,
                    fusion_pcu::PcuFloatUnderflowPolicy::IeeeAfterRounding,
                    PcuValueTypeCaps::FLOAT32,
                ),
                Ok(())
            );
            assert!(kernel.ops.iter().any(|instruction| matches!(
                instruction,
                PcuDispatchOp::Data(PcuDispatchDataOp::CheckedFloatBinary {
                    op,
                    underflow_policy: fusion_pcu::PcuFloatUnderflowPolicy::IeeeAfterRounding,
                    ..
                }) if *op == float_op
            )));
        }
    }

    #[test]
    fn consuming_action_is_limited_to_identity_and_keeps_checked_relu_fresh() {
        let make_program = |relu_depth: usize| {
            let mut graph = Graph::default();
            let input = graph.input([4], PcuScalarType::F32).unwrap();
            let mut output = input;
            for _ in 0..relu_depth {
                output = graph.relu(output).unwrap();
            }
            graph
                .into_selected_program(
                    &[output],
                    TensorArithmeticRewritePolicy::Disabled,
                    TensorArithmeticCapability::Strict,
                    TensorPointwiseGroupingPolicy::Disabled,
                )
                .unwrap()
        };

        let identity = make_program(0);
        let nodes = identity
            .selected_nodes()
            .iter()
            .map(|value| identity.graph().node(*value).unwrap())
            .collect::<Vec<_>>();
        assert!(matches!(
            prepare_consuming_action(&identity, &nodes).unwrap(),
            Some(PreparedConsumingAction::IdentityTransfer(value))
                if value == identity.input_values()[0]
        ));

        let terminal_relu = make_program(1);
        let nodes = terminal_relu
            .selected_nodes()
            .iter()
            .map(|value| terminal_relu.graph().node(*value).unwrap())
            .collect::<Vec<_>>();
        assert!(
            prepare_consuming_action(&terminal_relu, &nodes)
                .unwrap()
                .is_none()
        );

        let nonterminal_relu = make_program(2);
        let nodes = nonterminal_relu
            .selected_nodes()
            .iter()
            .map(|value| nonterminal_relu.graph().node(*value).unwrap())
            .collect::<Vec<_>>();
        assert!(
            prepare_consuming_action(&nonterminal_relu, &nodes)
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn integer_identity_preparation_transfers_all_widths_and_arithmetic_stays_fresh() {
        for scalar_type in [
            PcuScalarType::I8,
            PcuScalarType::U8,
            PcuScalarType::I16,
            PcuScalarType::U16,
            PcuScalarType::I32,
            PcuScalarType::U32,
            PcuScalarType::I64,
            PcuScalarType::U64,
        ] {
            let mut identity_graph = Graph::default();
            let input = identity_graph.input([4], scalar_type).unwrap();
            let identity = identity_graph
                .into_selected_program(
                    &[input],
                    TensorArithmeticRewritePolicy::Disabled,
                    TensorArithmeticCapability::Strict,
                    TensorPointwiseGroupingPolicy::Disabled,
                )
                .unwrap();
            let identity_nodes = identity
                .selected_nodes()
                .iter()
                .map(|value| identity.graph().node(*value).unwrap())
                .collect::<Vec<_>>();
            assert!(matches!(
                prepare_consuming_action(&identity, &identity_nodes).unwrap(),
                Some(PreparedConsumingAction::IdentityTransfer(value)) if value == input
            ));

            let mut arithmetic_graph = Graph::default();
            let left = arithmetic_graph.input([4], scalar_type).unwrap();
            let right = arithmetic_graph.input([4], scalar_type).unwrap();
            let output = arithmetic_graph.add(left, right).unwrap();
            let arithmetic = arithmetic_graph
                .into_selected_program(
                    &[output],
                    TensorArithmeticRewritePolicy::Disabled,
                    TensorArithmeticCapability::Strict,
                    TensorPointwiseGroupingPolicy::Disabled,
                )
                .unwrap();
            let arithmetic_nodes = arithmetic
                .selected_nodes()
                .iter()
                .map(|value| arithmetic.graph().node(*value).unwrap())
                .collect::<Vec<_>>();
            assert!(
                prepare_consuming_action(&arithmetic, &arithmetic_nodes)
                    .unwrap()
                    .is_none()
            );
        }
    }

    #[test]
    fn f32_checked_binary_uses_fresh_output_instead_of_donation() {
        let mut graph = Graph::default();
        let left = graph.input([4], PcuScalarType::F32).unwrap();
        let right = graph.input([4], PcuScalarType::F32).unwrap();
        let output = graph.add(left, right).unwrap();
        let program = graph
            .into_selected_program(
                &[output],
                TensorArithmeticRewritePolicy::Disabled,
                TensorArithmeticCapability::Strict,
                TensorPointwiseGroupingPolicy::Disabled,
            )
            .unwrap();
        let nodes = program
            .selected_nodes()
            .iter()
            .map(|value| program.graph().node(*value).unwrap())
            .collect::<Vec<_>>();
        assert!(
            prepare_consuming_action(&program, &nodes)
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn execution_scheduler_vectors_spill_without_truncating_large_graphs() {
        let inline_resources = execution_resource_slots(TENSOR_EXECUTION_INLINE_NODES);
        assert_eq!(inline_resources.len(), TENSOR_EXECUTION_INLINE_NODES);
        assert!(!inline_resources.spilled());

        let node_count = TENSOR_EXECUTION_INLINE_NODES + 1;
        let spilled_resources = execution_resource_slots(node_count);
        let mut spilled_use_counts = TensorExecutionUseCounts::new();
        spilled_use_counts.resize(node_count, 3);

        assert_eq!(spilled_resources.len(), node_count);
        assert_eq!(spilled_use_counts.len(), node_count);
        assert!(spilled_resources.spilled());
        assert!(spilled_use_counts.spilled());
    }

    #[test]
    fn selected_scratch_plan_preserves_fanout_lifetimes_and_reuses_expired_slots() {
        let mut graph = Graph::default();
        let left = graph.input([16], PcuScalarType::F32).unwrap();
        let right = graph.input([16], PcuScalarType::F32).unwrap();
        let fanout = graph.add(left, right).unwrap();
        let first_output = graph.relu(fanout).unwrap();
        let later = graph.mul(fanout, right).unwrap();
        let second_output = graph.relu(later).unwrap();
        let last_temporary = graph.sub(left, right).unwrap();
        let third_output = graph.relu(last_temporary).unwrap();
        let prepared = prepared_for_request_test(
            &graph,
            &[first_output, second_output, third_output],
            TensorPointwiseGroupingPolicy::Disabled,
        );

        let plan = selected_scratch_storage_plan(&prepared).unwrap();

        assert_eq!(plan.assignments().len(), 3);
        assert_eq!(plan.slot_for(fanout), Some(0));
        assert_eq!(plan.slot_for(later), Some(1));
        assert_eq!(plan.slot_for(last_temporary), Some(0));
        assert_eq!(plan.slots().len(), 2);
        assert_eq!(plan.total_bytes(), 2 * 16 * size_of::<f32>());
        assert_eq!(plan.slot_for(first_output), None);
        assert_eq!(plan.slot_for(second_output), None);
        assert_eq!(plan.slot_for(third_output), None);
    }

    #[test]
    fn selected_scratch_candidates_skip_uniforms_and_suppressed_fused_values() {
        let mut graph = Graph::default();
        let input = graph.input([16], PcuScalarType::F32).unwrap();
        let uniform = graph
            .uniform_value([16], TensorScalarValue::F32(2.0))
            .unwrap();
        let temporary = graph.add(input, uniform).unwrap();
        let output = graph.relu(temporary).unwrap();
        let prepared =
            prepared_for_request_test(&graph, &[output], TensorPointwiseGroupingPolicy::Disabled);
        let plan = selected_scratch_storage_plan(&prepared).unwrap();
        assert_eq!(plan.assignments().len(), 1);
        assert_eq!(plan.slot_for(temporary), Some(0));
        assert_eq!(plan.slot_for(uniform), None);

        let mut fused_graph = Graph::default();
        let left = fused_graph.input([16], PcuScalarType::F32).unwrap();
        let right = fused_graph.input([16], PcuScalarType::F32).unwrap();
        let suppressed = fused_graph.add(left, right).unwrap();
        let fused_output = fused_graph.sub(suppressed, right).unwrap();
        let fused = prepared_for_request_test(
            &fused_graph,
            &[fused_output],
            TensorPointwiseGroupingPolicy::BoundedAddSubIdentity,
        );
        assert!(!fused.suppressed_adds.contains(&suppressed));
        let fused_plan = selected_scratch_storage_plan(&fused).unwrap();
        assert_eq!(fused_plan.assignments().len(), 1);
        assert!(fused_plan.slot_for(suppressed).is_some());
        assert_eq!(fused_plan.slot_for(fused_output), None);
    }

    #[test]
    fn fixed_elementwise_timing_keeps_phase_values_independent() {
        let mut sink = CollectElementwiseTiming::default();
        let mark = sink.begin(ElementwisePhase::CacheAndBind);
        sink.finish(ElementwisePhase::CacheAndBind, mark);
        let cache_and_bind = sink.0.cache_and_bind;
        assert_eq!(sink.0.submit, Duration::ZERO);
        assert_eq!(sink.0.wait, Duration::ZERO);

        let mark = sink.begin(ElementwisePhase::Submit);
        sink.finish(ElementwisePhase::Submit, mark);
        let submit = sink.0.submit;
        assert_eq!(sink.0.cache_and_bind, cache_and_bind);
        assert_eq!(sink.0.wait, Duration::ZERO);

        let mark = sink.begin(ElementwisePhase::Wait);
        sink.finish(ElementwisePhase::Wait, mark);
        assert_eq!(sink.0.cache_and_bind, cache_and_bind);
        assert_eq!(sink.0.submit, submit);
    }

    #[test]
    fn diagnostic_node_timings_preserve_execution_order_and_count() {
        let mut graph = Graph::default();
        let left = graph.input([2, 2], PcuScalarType::F32).unwrap();
        let right = graph.input([2, 2], PcuScalarType::F32).unwrap();
        let sum = graph.add(left, right).unwrap();
        let left_node = NodeDescriptor {
            value: left,
            op: graph.nodes().find(|node| node.value == left).unwrap().op,
            shape: graph.shape(left).unwrap(),
            scalar_type: fusion_pcu::PcuScalarType::F32,
            numerical_mode: None,
            numerical_options: fusion_pcu::PcuNumericalOptions::default(),
            float_underflow_policy: None,
        };
        let sum_node = NodeDescriptor {
            value: sum,
            op: graph.nodes().find(|node| node.value == sum).unwrap().op,
            shape: graph.shape(sum).unwrap(),
            scalar_type: fusion_pcu::PcuScalarType::F32,
            numerical_mode: None,
            numerical_options: fusion_pcu::PcuNumericalOptions::default(),
            float_underflow_policy: graph.node(sum).unwrap().float_underflow_policy,
        };
        let mut timings = CollectNodeTimings::default();

        let mark = timings.begin();
        timings.finish(mark, &left_node);
        let mark = timings.begin();
        timings.finish(mark, &sum_node);

        assert_eq!(timings.0.len(), 2);
        assert_eq!(timings.0[0].value, left);
        assert_eq!(timings.0[0].operation, "Input");
        assert_eq!(timings.0[1].value, sum);
        assert_eq!(timings.0[1].operation, "Add");
    }

    #[test]
    fn prepared_matmul_metadata_keeps_original_operand_shapes_and_indices() {
        let mut graph = Graph::default();
        let left = graph.input([3, 2], PcuScalarType::F32).unwrap();
        let right = graph.input([3, 4], PcuScalarType::F32).unwrap();
        let output = graph.matmul_transposed(left, right, true, false).unwrap();
        let prepared = prepare_graph(&graph, output, &PureRocmAssessor).unwrap();
        let output_index = prepared.index_by_value[&output];

        assert_eq!(graph.shape(output).unwrap(), &[2, 4]);
        assert_eq!(
            prepared.matmul_operands[output_index],
            Some(PreparedMatMulOperands {
                left_index: prepared.index_by_value[&left],
                right_index: prepared.index_by_value[&right],
                left_shape: [3, 2],
                right_shape: [3, 4],
                strict: None,
            })
        );
        assert!(
            prepared.matmul_operands[..output_index]
                .iter()
                .all(Option::is_none)
        );
        assert!(matches!(
            prepared.nodes[output_index].op,
            OpDescriptor::MatMul {
                transpose_left: true,
                transpose_right: false,
                ..
            }
        ));
    }

    #[test]
    fn prepared_storage_constraints_resolve_node_and_output_slots() {
        let mut graph = Graph::default();
        let left = graph.input([4], PcuScalarType::F32).unwrap();
        let right = graph.input([4], PcuScalarType::F32).unwrap();
        let first_output = graph.add(left, right).unwrap();
        let second_output = graph.relu(first_output).unwrap();
        let outputs = [first_output, second_output];
        let prepared = prepare_graph_outputs_plan_with_policies(
            &graph,
            &outputs,
            &PureRocmAssessor,
            TensorArithmeticRewritePolicy::Disabled,
            TensorArithmeticCapability::Strict,
            TensorPointwiseGroupingPolicy::Disabled,
        )
        .unwrap();

        assert!(!prepared.indexed_storage_constraints.is_empty());
        for indexed in &prepared.indexed_storage_constraints {
            assert_eq!(
                prepared.nodes[indexed.left_index].value,
                indexed.constraint.left
            );
            assert_eq!(
                prepared.nodes[indexed.right_index].value,
                indexed.constraint.right
            );
            assert_eq!(
                indexed.left_output_index,
                outputs
                    .iter()
                    .position(|&value| value == indexed.constraint.left)
            );
            assert_eq!(
                indexed.right_output_index,
                outputs
                    .iter()
                    .position(|&value| value == indexed.constraint.right)
            );
        }
        assert!(prepared.indexed_storage_constraints.iter().any(|indexed| {
            indexed.left_output_index == Some(0) || indexed.right_output_index == Some(0)
        }));
    }

    #[test]
    fn legacy_raw_add_dispatch_kernel_is_rejected() {
        let kernel = add_kernel(17, 0);
        assert!(crate::lower_dispatch_to_hip_source(&kernel).is_err());
        assert_eq!(kernel.entry.logical_shape, [17, 1, 1]);
    }

    #[test]
    fn legacy_raw_add_relu_dispatch_kernel_is_rejected_for_dense_and_uniform_inputs() {
        let dense_kernel = add_relu_kernel(17, 0);
        assert!(crate::lower_dispatch_to_hip_source(&dense_kernel).is_err());
        assert_eq!(dense_kernel.entry.logical_shape, [17, 1, 1]);

        for scalar_mask in 1..=3 {
            let kernel = add_relu_kernel(17, scalar_mask);
            assert!(crate::lower_dispatch_to_hip_source(&kernel).is_err());
            assert_eq!(kernel.entry.logical_shape, [17, 1, 1]);
        }
    }

    #[test]
    fn bounded_add_sub_relu_dispatch_preserves_order_and_scalar_leaf_binding() {
        let mut graph = Graph::default();
        let dense_leaf = graph.input([17], PcuScalarType::F32).unwrap();
        let scalar_leaf = graph
            .uniform_value([17], TensorScalarValue::F32(-0.125))
            .unwrap();
        let sum = graph.add(dense_leaf, scalar_leaf).unwrap();
        let difference = graph.sub(dense_leaf, sum).unwrap();
        let output = graph.relu(difference).unwrap();
        let plan = graph.execution_plan_for_outputs(&[output]).unwrap();
        let selected = plan.select_lowering_with_grouping(
            fusion_pcu::dialect::tensor::TensorArithmeticRewritePolicy::Disabled,
            fusion_pcu::dialect::tensor::TensorArithmeticCapability::Strict,
            TensorPointwiseGroupingPolicy::BoundedAddSubRelu,
        );
        assert!(selected.bounded_pointwise_fusion_groups().is_empty());
        let prepared = prepared_for_request_test(
            &graph,
            &[output],
            TensorPointwiseGroupingPolicy::BoundedAddSubRelu,
        );
        let kinds = collect_dispatch_requests(&prepared)
            .unwrap()
            .into_iter()
            .filter_map(|request| match request {
                TensorDispatchRequest::Fixed { kind, .. } => Some(kind),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(
            kinds,
            [
                TensorDispatchKind::CheckedFloatAdd,
                TensorDispatchKind::CheckedFloatSub,
                TensorDispatchKind::Relu,
            ]
        );
    }

    #[test]
    fn bounded_add_sub_identity_dispatch_has_no_relu_epilogue_and_is_preparable() {
        let mut graph = Graph::default();
        let left = graph.input([17], PcuScalarType::F32).unwrap();
        let right = graph.input([17], PcuScalarType::F32).unwrap();
        let sum = graph.add(left, right).unwrap();
        let output = graph.sub(sum, left).unwrap();
        let selected = graph
            .execution_plan_for_outputs(&[output])
            .unwrap()
            .select_lowering_with_grouping(
                TensorArithmeticRewritePolicy::Disabled,
                TensorArithmeticCapability::Strict,
                TensorPointwiseGroupingPolicy::BoundedAddSubIdentity,
            );
        assert!(selected.bounded_pointwise_fusion_groups().is_empty());
        let prepared = prepared_for_request_test(
            &graph,
            &[output],
            TensorPointwiseGroupingPolicy::BoundedAddSubIdentity,
        );
        assert!(!prepared.suppressed_adds.contains(&sum));
        let requests = collect_dispatch_requests(&prepared).unwrap();
        assert_eq!(requests.len(), 2);
        assert!(matches!(
            requests[0],
            TensorDispatchRequest::Fixed {
                kind: TensorDispatchKind::CheckedFloatAdd,
                ..
            }
        ));
        assert!(matches!(
            requests[1],
            TensorDispatchRequest::Fixed {
                kind: TensorDispatchKind::CheckedFloatSub,
                ..
            }
        ));
    }

    #[test]
    fn low_four_owned_preparation_retains_all_checked_pointwise_dispatches() {
        for dtype in [
            PcuScalarType::F16,
            PcuScalarType::BF16,
            PcuScalarType::F8E4M3FN,
            PcuScalarType::F8E5M2,
        ] {
            for policy in [
                PcuFloatUnderflowPolicy::IeeeAfterRounding,
                PcuFloatUnderflowPolicy::RejectSubnormalResult,
                PcuFloatUnderflowPolicy::AllowGradualUnderflow,
            ] {
                let mut graph = Graph::default();
                graph.set_numerical_mode(fusion_pcu::PcuNumericalMode::Strict);
                let left = graph.input([65], dtype).unwrap();
                let right = graph.input([65], dtype).unwrap();
                let values = [
                    graph.add(left, right).unwrap(),
                    graph.sub(left, right).unwrap(),
                    graph.mul(left, right).unwrap(),
                    graph.div(left, right).unwrap(),
                    graph.relu(left).unwrap(),
                ];
                for value in values {
                    graph
                        .set_value_float_underflow_policy(value, policy)
                        .unwrap();
                }
                let program = graph
                    .into_selected_program(
                        &values,
                        TensorArithmeticRewritePolicy::Disabled,
                        TensorArithmeticCapability::Strict,
                        TensorPointwiseGroupingPolicy::Disabled,
                    )
                    .unwrap();
                let data = prepare_owned_graph_data(&program, &PureRocmAssessor).unwrap();
                assert_eq!(data.scalar_type, Some(dtype));
                assert!(!data.transport_only_inputs);
                let dispatches = data.fixed_dispatches.iter().flatten().collect::<Vec<_>>();
                assert_eq!(dispatches.len(), 5);
                for dispatch in dispatches {
                    let node = program.graph().node(dispatch.value).unwrap();
                    assert_eq!(
                        dispatch.kernel.numerical_requirements,
                        super::fixed_numerical_requirements(node)
                    );
                    assert_eq!(
                        dispatch.kernel.numerical_requirements.float_underflow,
                        policy
                    );
                }
            }
        }
    }

    #[test]
    fn fixed_captured_tuple_is_exact_and_scalar_none_is_canonical_boundary() {
        use fusion_pcu::{PcuCompoundArithmeticPolicy, PcuNumericalMode, PcuPrecisionPolicy};
        let mut graph = Graph::default();
        graph.set_numerical_mode(PcuNumericalMode::Strict);
        let input = graph.input([17], PcuScalarType::F16).unwrap();
        let value = graph.add(input, input).unwrap();
        let original = super::fixed_numerical_requirements(graph.node(value).unwrap());
        assert_eq!(graph.node(value).unwrap().numerical_mode, None);
        assert_eq!(original.numerical_mode, PcuNumericalMode::Boundary);
        graph.set_numerical_mode(PcuNumericalMode::Boundary);
        graph.set_numerical_options(fusion_pcu::PcuNumericalOptions {
            compound_arithmetic: PcuCompoundArithmeticPolicy::BackendDefined,
            precision: PcuPrecisionPolicy::BackendOptimized,
            ..Default::default()
        });
        // Defaults changed after capture cannot alter a scalar node's header or cache key.
        assert_eq!(
            super::fixed_numerical_requirements(graph.node(value).unwrap()),
            original
        );
        let node = graph.node(value).unwrap();
        let prepared =
            prepared_for_request_test(&graph, &[value], TensorPointwiseGroupingPolicy::Disabled);
        let requests = collect_dispatch_requests(&prepared).unwrap();
        assert_eq!(requests.len(), 1);
        assert_eq!(
            requests[0].key(),
            prepared
                .fixed_dispatches
                .iter()
                .flatten()
                .find(|dispatch| dispatch.value == value)
                .unwrap()
                .cache_key
        );
        assert_eq!(
            prepared
                .fixed_dispatches
                .iter()
                .flatten()
                .find(|dispatch| dispatch.value == value)
                .unwrap()
                .kernel
                .numerical_requirements,
            original
        );
        let mut distinct = Vec::new();
        for mode in [None, Some(PcuNumericalMode::Strict)] {
            for compound in [
                PcuCompoundArithmeticPolicy::Checked,
                PcuCompoundArithmeticPolicy::BackendDefined,
            ] {
                for precision in [
                    PcuPrecisionPolicy::Preserve,
                    PcuPrecisionPolicy::BackendOptimized,
                ] {
                    for underflow in [
                        PcuFloatUnderflowPolicy::IeeeAfterRounding,
                        PcuFloatUnderflowPolicy::RejectSubnormalResult,
                        PcuFloatUnderflowPolicy::AllowGradualUnderflow,
                    ] {
                        let mut profile = node;
                        profile.numerical_mode = mode;
                        profile.numerical_options.compound_arithmetic = compound;
                        profile.numerical_options.precision = precision;
                        profile.float_underflow_policy = Some(underflow);
                        let requirements = super::fixed_numerical_requirements(profile);
                        let key = TensorDispatchCacheKey::Fixed(
                            TensorDispatchKind::CheckedFloatAdd,
                            TensorPointwiseScalarType::F16,
                            17,
                            0,
                            requirements,
                        );
                        assert!(!distinct.contains(&key));
                        distinct.push(key);
                    }
                }
            }
        }
        assert_eq!(distinct.len(), 24);
    }

    #[test]
    fn mixed_float_policies_have_distinct_prewarm_keys_and_executables() {
        let mut graph = Graph::default();
        let left = graph.input([17], PcuScalarType::F32).unwrap();
        let right = graph.input([17], PcuScalarType::F32).unwrap();
        let ieee = graph.add(left, right).unwrap();
        let gradual = graph
            .add_with_underflow_policy(left, right, PcuFloatUnderflowPolicy::AllowGradualUnderflow)
            .unwrap();
        let prepared = prepared_for_request_test(
            &graph,
            &[ieee, gradual],
            TensorPointwiseGroupingPolicy::Disabled,
        );
        let requests = collect_dispatch_requests(&prepared).unwrap();
        assert_eq!(requests.len(), 2);
        let policies = requests
            .iter()
            .map(|request| match request {
                TensorDispatchRequest::Fixed {
                    kind: TensorDispatchKind::CheckedFloatAdd,
                    numerical_requirements,
                    ..
                } => numerical_requirements.float_underflow,
                other => panic!("unexpected request: {other:?}"),
            })
            .collect::<Vec<_>>();
        assert_eq!(
            policies,
            [
                PcuFloatUnderflowPolicy::IeeeAfterRounding,
                PcuFloatUnderflowPolicy::AllowGradualUnderflow,
            ]
        );
        assert_ne!(requests[0].key(), requests[1].key());

        let ieee_kernel = checked_float::kernel(
            fusion_pcu::PcuDispatchFloatBinaryOp::Add,
            PcuScalarType::F32,
            PcuFloatUnderflowPolicy::IeeeAfterRounding,
            17,
        )
        .unwrap();
        let gradual_kernel = checked_float::kernel(
            fusion_pcu::PcuDispatchFloatBinaryOp::Add,
            PcuScalarType::F32,
            PcuFloatUnderflowPolicy::AllowGradualUnderflow,
            17,
        )
        .unwrap();
        assert_ne!(ieee_kernel.id, gradual_kernel.id);
        let op_policy = |kernel: &PcuDispatchKernelIr<'_>| {
            kernel.ops.iter().find_map(|op| match op {
                PcuDispatchOp::Data(PcuDispatchDataOp::CheckedFloatBinary {
                    underflow_policy,
                    ..
                }) => Some(*underflow_policy),
                _ => None,
            })
        };
        assert_eq!(
            op_policy(&ieee_kernel),
            Some(PcuFloatUnderflowPolicy::IeeeAfterRounding)
        );
        assert_eq!(
            op_policy(&gradual_kernel),
            Some(PcuFloatUnderflowPolicy::AllowGradualUnderflow)
        );
    }

    #[test]
    fn checked_mul_identity_keeps_two_nodes_and_deduplicates_dense_dispatch_keys() {
        let mut graph = Graph::default();
        let input = graph.input([17], PcuScalarType::F32).unwrap();
        let factor = graph
            .uniform_value([17], TensorScalarValue::F32(2.0))
            .unwrap();
        let first = graph.mul(input, factor).unwrap();
        let output = graph.mul(first, input).unwrap();
        let prepared = prepared_for_request_test(
            &graph,
            &[output],
            TensorPointwiseGroupingPolicy::BoundedMulIdentity,
        );
        assert!(!prepared.suppressed_adds.contains(&first));
        assert_eq!(
            prepared
                .nodes
                .iter()
                .filter(|node| matches!(node.op, OpDescriptor::Mul { .. }))
                .count(),
            2
        );
        let requests = collect_dispatch_requests(&prepared).unwrap();
        // Both checked operations use the same dense ABI profile, so prewarm deduplicates their
        // identical fixed-dispatch key while the selected graph still retains both operations.
        assert_eq!(requests.len(), 1);
        assert!(requests.iter().all(|request| matches!(
            request,
            TensorDispatchRequest::Fixed {
                kind: TensorDispatchKind::CheckedFloatMul,
                logical_count: 17,
                scalar_mask: 0,
                ..
            }
        )));
    }

    #[test]
    fn legacy_raw_relu_dispatch_kernel_is_rejected() {
        let kernel = relu_kernel(17);
        assert!(crate::lower_dispatch_to_hip_source(&kernel).is_err());
        assert_eq!(kernel.entry.logical_shape, [17, 1, 1]);
    }

    struct PureRocmAssessor;

    struct DenseOnlyAssessor;

    impl TensorOperationAssessor for DenseOnlyAssessor {
        fn assess_node(&self, graph: &Graph, node: NodeDescriptor<'_>) -> TensorOperationSupport {
            assess_tensor_node(graph, node)
        }
    }

    impl TensorOperationAssessor for PureRocmAssessor {
        fn assess_node(&self, graph: &Graph, node: NodeDescriptor<'_>) -> TensorOperationSupport {
            // This test-only historical planner oracle never admits real device execution.
            // Preserve raw library route fixtures while production checked admission rejects it.
            if let OpDescriptor::MatMul {
                left,
                right,
                transpose_left,
                transpose_right,
            } = node.op
                && node.numerical_mode == Some(fusion_pcu::PcuNumericalMode::Boundary)
                && super::matmul_shape_supported(
                    graph,
                    node.shape,
                    left,
                    right,
                    transpose_left,
                    transpose_right,
                )
            {
                return TensorOperationSupport::Supported {
                    route: TensorExecutionRoute::Library,
                    workspace_bytes: None,
                };
            }
            // Raw historical training-route planning only. This assessor is never used to
            // create a production executable or certify floating exception handling.
            if node.scalar_type == PcuScalarType::F32
                && node.numerical_mode != Some(fusion_pcu::PcuNumericalMode::Strict)
                && matches!(
                    node.op,
                    OpDescriptor::MeanSquaredError { .. }
                        | OpDescriptor::SgdUpdate { .. }
                        | OpDescriptor::ReluBackward { .. }
                )
            {
                return super::assess_f32_tensor_node(graph, node);
            }
            assess_tensor_node(graph, node)
        }

        fn supports_operand_representation(
            &self,
            _graph: &Graph,
            node: NodeDescriptor<'_>,
            _operand: ValueId,
            representation: TensorOperandRepresentation,
        ) -> bool {
            rocm_supports_operand_representation(node, representation)
        }
    }

    #[test]
    fn single_output_plan_rejects_multi_output_before_provider_work() {
        let mut graph = Graph::default();
        let input = graph.input([4], PcuScalarType::F32).unwrap();
        let first = graph.relu(input).unwrap();
        let second = graph.add(input, input).unwrap();
        let program = graph
            .into_selected_program(
                &[first, second],
                TensorArithmeticRewritePolicy::Disabled,
                TensorArithmeticCapability::Strict,
                TensorPointwiseGroupingPolicy::Disabled,
            )
            .unwrap();
        let data = prepare_owned_graph_data(&program, &PureRocmAssessor).unwrap();
        let prepared =
            super::RocmOwnedPreparedTensorGraph::from_parts(Arc::new(program), data).unwrap();
        let provider_allocations = Cell::new(0);

        let error = with_single_output_plan(&prepared, || {
            provider_allocations.set(provider_allocations.get() + 1);
            Ok(())
        })
        .unwrap_err();

        assert!(matches!(
            error,
            super::RocmTensorExecutionError::OutputCountMismatch {
                expected: 1,
                actual: 2
            }
        ));
        assert_eq!(provider_allocations.get(), 0);
    }

    #[test]
    fn single_output_plan_runs_provider_work_for_one_output() {
        let mut graph = Graph::default();
        let input = graph.input([4], PcuScalarType::F32).unwrap();
        let output = graph.relu(input).unwrap();
        let program = graph
            .into_selected_program(
                &[output],
                TensorArithmeticRewritePolicy::Disabled,
                TensorArithmeticCapability::Strict,
                TensorPointwiseGroupingPolicy::Disabled,
            )
            .unwrap();
        let data = prepare_owned_graph_data(&program, &PureRocmAssessor).unwrap();
        let prepared =
            super::RocmOwnedPreparedTensorGraph::from_parts(Arc::new(program), data).unwrap();
        let provider_allocations = Cell::new(0);

        let result = with_single_output_plan(&prepared, || {
            provider_allocations.set(provider_allocations.get() + 1);
            Ok(17)
        });

        assert!(matches!(result, Ok(17)));
        assert_eq!(provider_allocations.get(), 1);
    }

    fn prepared_for_request_test<'graph>(
        graph: &'graph Graph,
        outputs: &[ValueId],
        grouping: TensorPointwiseGroupingPolicy,
    ) -> RocmPreparedTensorGraph<'graph> {
        let GraphExecutionPreflight {
            tensor_plan,
            lowering_plan,
            nodes,
            index_by_value,
            use_counts,
            fused_add_by_relu,
            bounded_pointwise_by_output,
            bounded_mul_by_output,
            fixed_dispatches,
            suppressed_adds,
            storage_constraints: _,
            indexed_storage_constraints,
            matmul_operands,
            strict_sgd,
            physical_layouts,
        } = prepare_graph_outputs_plan_with_policies(
            graph,
            outputs,
            &PureRocmAssessor,
            TensorArithmeticRewritePolicy::Disabled,
            TensorArithmeticCapability::Strict,
            grouping,
        )
        .unwrap();
        let data = RocmPreparedGraphData {
            scalar_type: homogeneous_scalar_type(&nodes),
            requires_blas: nodes_require_blas(&nodes),
            batch_native_matmuls: native_execution::batch_native_matmuls(&nodes, outputs),
            transport_only_inputs: false,
            consuming_action: None,
            node_values: nodes.iter().map(|node| node.value).collect(),
            output: outputs[0],
            outputs: outputs.to_vec(),
            index_by_value,
            use_counts,
            fused_add_by_relu,
            bounded_pointwise_by_output,
            bounded_mul_by_output,
            fixed_dispatches,
            suppressed_adds,
            indexed_storage_constraints,
            matmul_operands,
            strict_sgd,
            physical_layouts,
            rewrites: lowering_plan.rewrites().to_vec(),
            input_values: tensor_plan.input_values().to_vec(),
        };
        RocmPreparedTensorGraph {
            graph,
            plan: tensor_plan,
            lowering_plan,
            data,
            nodes,
        }
    }

    #[test]
    fn prepared_fixed_dispatch_binds_scalar_width_and_kernel_type_cold() {
        let mut kernel_ids = Vec::new();
        let mut cache_keys = Vec::new();
        for (scalar_type, expected, expected_value_type) in [
            (
                PcuScalarType::F32,
                TensorPointwiseScalarType::F32,
                PcuValueType::f32(),
            ),
            (
                PcuScalarType::F64,
                TensorPointwiseScalarType::F64,
                PcuValueType::f64(),
            ),
        ] {
            let mut graph = Graph::default();
            let left = graph.input([17], scalar_type).unwrap();
            let right = graph.input([17], scalar_type).unwrap();
            let output = graph.add(left, right).unwrap();
            let prepared = prepared_for_request_test(
                &graph,
                &[output],
                TensorPointwiseGroupingPolicy::Disabled,
            );
            let view = prepared.view();
            let index = view.index_of(output).unwrap();
            let dispatch = view.fixed_dispatch(index).unwrap();
            kernel_ids.push(dispatch.kernel.id);
            cache_keys.push(dispatch.cache_key.clone());

            assert_eq!(dispatch.scalar_type, expected);
            assert_eq!(dispatch.value_type, expected_value_type);
            assert_eq!(dispatch.left_binding, ADD_LEFT_REF);
            assert_eq!(dispatch.right_binding, Some(ADD_RIGHT_REF));
            assert_eq!(dispatch.output_binding, ADD_OUTPUT_REF);
            let expected_kind = TensorDispatchKind::CheckedFloatAdd;
            assert_eq!(
                dispatch.cache_key,
                TensorDispatchCacheKey::Fixed(
                    expected_kind,
                    expected,
                    17,
                    0,
                    PcuDispatchKernelIr::DEFAULT_REQUIREMENTS,
                )
            );
            assert!(dispatch.kernel.ops.iter().any(|op| matches!(
                op,
                PcuDispatchOp::Data(PcuDispatchDataOp::CheckedFloatBinary {
                    value_type: PcuValueType::Scalar(actual_scalar),
                    op: fusion_pcu::PcuDispatchFloatBinaryOp::Add,
                    underflow_policy: PcuFloatUnderflowPolicy::IeeeAfterRounding,
                    ..
                }) if *actual_scalar == scalar_type
            )));
        }
        assert_ne!(kernel_ids[0], kernel_ids[1]);
        assert_ne!(cache_keys[0], cache_keys[1]);
    }

    #[test]
    fn prepared_graph_caches_blas_requirement() {
        let mut add_graph = Graph::default();
        let left = add_graph.input([2, 2], PcuScalarType::F32).unwrap();
        let right = add_graph.input([2, 2], PcuScalarType::F32).unwrap();
        let add = add_graph.add(left, right).unwrap();
        let prepared_add =
            prepared_for_request_test(&add_graph, &[add], TensorPointwiseGroupingPolicy::Disabled);
        assert!(!prepared_add.data.requires_blas);

        let mut matmul_graph = Graph::default();
        let left = matmul_graph.input([2, 3], PcuScalarType::F32).unwrap();
        let right = matmul_graph.input([3, 2], PcuScalarType::F32).unwrap();
        let product = matmul_graph.matmul(left, right).unwrap();
        let prepared_matmul = prepared_for_request_test(
            &matmul_graph,
            &[product],
            TensorPointwiseGroupingPolicy::Disabled,
        );
        assert!(prepared_matmul.data.requires_blas);
    }

    #[test]
    fn owned_preparation_rejects_unsupported_integer_relu_precisely() {
        let mut graph = Graph::default();
        let input = graph.input([4], PcuScalarType::U32).unwrap();
        let sum = graph.relu(input).unwrap();
        let program = graph
            .into_selected_program(
                &[sum],
                TensorArithmeticRewritePolicy::Disabled,
                TensorArithmeticCapability::Strict,
                TensorPointwiseGroupingPolicy::Disabled,
            )
            .unwrap();

        assert!(matches!(
            prepare_owned_graph_data(&program, &PureRocmAssessor),
            Err(RocmTensorExecutionError::UnsupportedScalarType(
                PcuScalarType::U32
            ))
        ));
    }

    fn assert_transport_layout<T: PcuScalar>(scalar_type: PcuScalarType) {
        let (bytes, alignment) = scalar_layout(scalar_type).unwrap();
        assert_eq!(bytes, size_of::<T>());
        assert_eq!(alignment, align_of::<T>() as u64);
        assert_eq!(bytes, T::HOST_SIZE);
        assert_eq!(bytes, T::ENCODED_SIZE);
        assert_eq!(usize::try_from(alignment).unwrap(), T::HOST_ALIGNMENT);
        assert!(is_transport_scalar(scalar_type));
    }

    #[test]
    fn input_only_transport_profile_covers_all_sealed_scalar_layouts() {
        assert_transport_layout::<i8>(PcuScalarType::I8);
        assert_transport_layout::<u8>(PcuScalarType::U8);
        assert_transport_layout::<i16>(PcuScalarType::I16);
        assert_transport_layout::<u16>(PcuScalarType::U16);
        assert_transport_layout::<i32>(PcuScalarType::I32);
        assert_transport_layout::<u32>(PcuScalarType::U32);
        assert_transport_layout::<i64>(PcuScalarType::I64);
        assert_transport_layout::<u64>(PcuScalarType::U64);
        assert_transport_layout::<PcuF16Bits>(PcuScalarType::F16);
        assert_transport_layout::<PcuBf16Bits>(PcuScalarType::BF16);
        assert_transport_layout::<fusion_pcu::PcuF8E4M3FnBits>(PcuScalarType::F8E4M3FN);
        assert_transport_layout::<fusion_pcu::PcuF8E5M2Bits>(PcuScalarType::F8E5M2);
        assert_transport_layout::<f32>(PcuScalarType::F32);
        assert_transport_layout::<f64>(PcuScalarType::F64);

        assert_transport_layout::<i128>(PcuScalarType::I128);
        assert_transport_layout::<u128>(PcuScalarType::U128);
        assert_transport_layout::<fusion_pcu::PcuI256>(PcuScalarType::I256);
        assert_transport_layout::<fusion_pcu::PcuU256>(PcuScalarType::U256);
        assert_transport_layout::<fusion_pcu::PcuI512>(PcuScalarType::I512);
        assert_transport_layout::<fusion_pcu::PcuU512>(PcuScalarType::U512);
        assert_transport_layout::<fusion_pcu::PcuF128Bits>(PcuScalarType::F128);
        assert_transport_layout::<fusion_pcu::PcuF256Bits>(PcuScalarType::F256);

        for scalar_type in [PcuScalarType::Bool, PcuScalarType::I4, PcuScalarType::U4] {
            assert!(!is_transport_scalar(scalar_type));
            assert!(matches!(
                scalar_layout(scalar_type),
                Err(RocmTensorExecutionError::UnsupportedScalarType(actual)) if actual == scalar_type
            ));
        }
    }

    #[test]
    fn owned_input_only_preparation_caches_transport_profile_without_admitting_arithmetic() {
        for scalar_type in [
            PcuScalarType::I8,
            PcuScalarType::U8,
            PcuScalarType::I16,
            PcuScalarType::U16,
            PcuScalarType::I32,
            PcuScalarType::U32,
            PcuScalarType::I64,
            PcuScalarType::U64,
            PcuScalarType::F16,
            PcuScalarType::BF16,
            PcuScalarType::F32,
            PcuScalarType::F64,
        ] {
            let mut graph = Graph::default();
            let input = graph.input([4], scalar_type).unwrap();
            let program = graph
                .into_selected_program(
                    &[input],
                    TensorArithmeticRewritePolicy::Disabled,
                    TensorArithmeticCapability::Strict,
                    TensorPointwiseGroupingPolicy::Disabled,
                )
                .unwrap();
            let prepared = prepare_owned_graph_data(&program, &PureRocmAssessor).unwrap();
            assert_eq!(prepared.scalar_type, Some(scalar_type));
            assert!(prepared.transport_only_inputs);
            assert_eq!(prepared.node_values, [input]);
        }

        let mut graph = Graph::default();
        let input = graph.input([4], PcuScalarType::U32).unwrap();
        let sum = graph.add(input, input).unwrap();
        let program = graph
            .into_selected_program(
                &[sum],
                TensorArithmeticRewritePolicy::Disabled,
                TensorArithmeticCapability::Strict,
                TensorPointwiseGroupingPolicy::Disabled,
            )
            .unwrap();
        let prepared = prepare_owned_graph_data(&program, &PureRocmAssessor).unwrap();
        assert_eq!(prepared.scalar_type, Some(PcuScalarType::U32));
        let dispatch = prepared.fixed_dispatches[1].as_ref().unwrap();
        assert!(matches!(
            dispatch.kernel.ops[2],
            PcuDispatchOp::Data(PcuDispatchDataOp::CheckedIntegerBinary { .. })
        ));
    }

    #[test]
    fn owned_input_only_profile_rejects_mixed_and_unrepresented_scalars_before_allocation() {
        let mut mixed_graph = Graph::default();
        let integer = mixed_graph.input([4], PcuScalarType::U32).unwrap();
        let float = mixed_graph.input([4], PcuScalarType::F32).unwrap();
        let mixed_program = mixed_graph
            .into_selected_program(
                &[integer, float],
                TensorArithmeticRewritePolicy::Disabled,
                TensorArithmeticCapability::Strict,
                TensorPointwiseGroupingPolicy::Disabled,
            )
            .unwrap();
        assert!(matches!(
            prepare_owned_graph_data(&mixed_program, &PureRocmAssessor),
            Err(RocmTensorExecutionError::UnsupportedScalarType(_))
        ));

        for scalar_type in [PcuScalarType::Bool, PcuScalarType::I4, PcuScalarType::U4] {
            let mut graph = Graph::default();
            let input = graph.input([4], scalar_type).unwrap();
            let program = graph.into_selected_program(
                &[input],
                TensorArithmeticRewritePolicy::Disabled,
                TensorArithmeticCapability::Strict,
                TensorPointwiseGroupingPolicy::Disabled,
            );
            assert!(matches!(
                program,
                Err(TensorError::UnsupportedScalarType { scalar_type: actual, .. })
                    if actual == scalar_type
            ));
        }

        let mut overflow_graph = Graph::default();
        let huge_input = overflow_graph
            .input([usize::MAX], PcuScalarType::U64)
            .unwrap();
        let overflow_program = overflow_graph.into_selected_program(
            &[huge_input],
            TensorArithmeticRewritePolicy::Disabled,
            TensorArithmeticCapability::Strict,
            TensorPointwiseGroupingPolicy::Disabled,
        );
        assert!(matches!(overflow_program, Err(TensorError::ShapeOverflow)));
    }

    #[test]
    fn prewarm_strict_requests_deduplicate_and_preserve_full_profiles() {
        let mut scalar_keys = Vec::new();
        for scalar in [PcuScalarType::F32, PcuScalarType::F64] {
            let mut graph = Graph::default();
            graph.set_numerical_mode(fusion_pcu::PcuNumericalMode::Strict);
            let left = graph.input([2, 2], scalar).unwrap();
            let right = graph.input([2, 2], scalar).unwrap();
            let first = graph.matmul(left, right).unwrap();
            let duplicate = graph.matmul(left, right).unwrap();
            let gradual = graph.matmul(left, right).unwrap();
            graph
                .set_value_float_underflow_policy(
                    gradual,
                    PcuFloatUnderflowPolicy::AllowGradualUnderflow,
                )
                .unwrap();
            let transposed = graph.matmul_transposed(left, right, true, false).unwrap();
            let prepared = prepared_for_request_test(
                &graph,
                &[first, duplicate, gradual, transposed],
                TensorPointwiseGroupingPolicy::Disabled,
            );
            assert!(!prepared.requires_blas);
            let requests = collect_dispatch_requests(&prepared).unwrap();
            assert_eq!(requests.len(), 3);
            assert!(
                requests
                    .iter()
                    .all(|request| matches!(request, TensorDispatchRequest::StrictMatMul(_)))
            );
            let keys = requests
                .iter()
                .map(TensorDispatchRequest::key)
                .collect::<Vec<_>>();
            assert_ne!(keys[0], keys[1]);
            assert_ne!(keys[0], keys[2]);
            assert_ne!(keys[1], keys[2]);
            assert_eq!(retained_requested_key_count(&keys, &keys[..2]), 2);
            scalar_keys.push(keys[0].clone());

            let mut different_inner = Graph::default();
            different_inner.set_numerical_mode(fusion_pcu::PcuNumericalMode::Strict);
            let left = different_inner.input([2, 3], scalar).unwrap();
            let right = different_inner.input([3, 2], scalar).unwrap();
            let output = different_inner.matmul(left, right).unwrap();
            let other = prepared_for_request_test(
                &different_inner,
                &[output],
                TensorPointwiseGroupingPolicy::Disabled,
            );
            assert_ne!(collect_dispatch_requests(&other).unwrap()[0].key(), keys[0]);
        }
        assert_ne!(scalar_keys[0], scalar_keys[1]);
    }

    #[test]
    fn prewarm_requests_deduplicate_fixed_keys_and_keep_scalar_layout_mask() {
        let mut graph = Graph::default();
        let input = graph.input([16], PcuScalarType::F32).unwrap();
        let uniform = graph
            .uniform_value([16], TensorScalarValue::F32(2.0))
            .unwrap();
        let first = graph.add(input, uniform).unwrap();
        let second = graph.add(first, uniform).unwrap();
        let output = graph.relu(second).unwrap();
        let prepared =
            prepared_for_request_test(&graph, &[output], TensorPointwiseGroupingPolicy::Disabled);

        let requests = collect_dispatch_requests(&prepared).unwrap();

        assert_eq!(requests.len(), 2);
        assert!(matches!(
            requests[0],
            TensorDispatchRequest::Fixed {
                kind: TensorDispatchKind::CheckedFloatAdd,
                scalar_type: TensorPointwiseScalarType::F32,
                logical_count: 16,
                scalar_mask: 0,
                numerical_requirements: PcuDispatchKernelIr::DEFAULT_REQUIREMENTS,
            }
        ));
        assert!(matches!(
            requests[1],
            TensorDispatchRequest::Fixed {
                kind: TensorDispatchKind::Relu,
                scalar_type: TensorPointwiseScalarType::F32,
                logical_count: 16,
                scalar_mask: 0,
                numerical_requirements: PcuDispatchKernelIr::DEFAULT_REQUIREMENTS,
            }
        ));
    }

    #[test]
    fn f64_fixed_pointwise_factories_keep_bindings_values_and_cache_identity_typed() {
        let kinds = [
            TensorDispatchKind::Add,
            TensorDispatchKind::Sub,
            TensorDispatchKind::Mul,
            TensorDispatchKind::AddRelu,
            TensorDispatchKind::Relu,
        ];
        let mut ids = std::collections::HashSet::new();
        for kind in kinds {
            for mask in 0..=3 {
                if kind == TensorDispatchKind::Relu && mask != 0 {
                    continue;
                }
                let kernel = pointwise::kernel(kind, 17, mask).unwrap();
                assert!(ids.insert(kernel.id));
                assert_eq!(kernel.entry.logical_shape, [17, 1, 1]);
                assert!(kernel.type_caps.contains(PcuValueTypeCaps::FLOAT64));
                assert!(kernel.bindings.iter().all(|binding| {
                    binding.binding_type == PcuBindingType::Value(fusion_pcu::PcuValueType::f64())
                }));
                assert!(kernel.ops.iter().all(|op| match op {
                    PcuDispatchOp::Data(PcuDispatchDataOp::Alu { value_type, .. }) => {
                        *value_type == fusion_pcu::PcuValueType::f64()
                    }
                    _ => true,
                }));
                if kind == TensorDispatchKind::AddRelu || kind == TensorDispatchKind::Relu {
                    assert!(kernel.ops.iter().any(|op| {
                        matches!(op, PcuDispatchOp::Data(PcuDispatchDataOp::CheckedFloatUnary {
                            value_type,
                            op: fusion_pcu::PcuDispatchFloatUnaryOp::Relu,
                            ..
                        }) if *value_type == fusion_pcu::PcuValueType::f64())
                    }));
                }
            }
        }
        assert_eq!(ids.len(), 17);

        let f32_key = TensorDispatchCacheKey::Fixed(
            TensorDispatchKind::CheckedFloatAdd,
            TensorPointwiseScalarType::F32,
            17,
            0,
            PcuDispatchKernelIr::DEFAULT_REQUIREMENTS,
        );
        let f64_key = TensorDispatchCacheKey::Fixed(
            TensorDispatchKind::Add,
            TensorPointwiseScalarType::F64,
            17,
            0,
            PcuDispatchKernelIr::DEFAULT_REQUIREMENTS,
        );
        assert_ne!(f32_key, f64_key);
        assert_ne!(
            add_kernel(17, 0).id,
            pointwise::kernel(TensorDispatchKind::Add, 17, 0)
                .unwrap()
                .id
        );

        let mut graph = Graph::default();
        let left = graph.input([17], PcuScalarType::F64).unwrap();
        let right = graph.input([17], PcuScalarType::F64).unwrap();
        let output = graph.add(left, right).unwrap();
        let descriptor = graph.node(output).unwrap();
        assert_eq!(
            TensorPointwiseScalarType::try_from(descriptor.scalar_type).unwrap(),
            TensorPointwiseScalarType::F64
        );
    }

    #[test]
    fn f64_fixed_pointwise_factory_rejects_unsupported_profiles_and_uniform_masks() {
        assert!(matches!(
            pointwise::kernel(TensorDispatchKind::Relu, 17, 1),
            Err(RocmTensorExecutionError::InvalidPointwiseProfile)
        ));
        assert!(matches!(
            TensorPointwiseScalarType::try_from(PcuScalarType::I32),
            Ok(TensorPointwiseScalarType::I32)
        ));
        assert!(matches!(
            pointwise::kernel(TensorDispatchKind::CheckedIntegerAdd, 17, 0),
            Err(RocmTensorExecutionError::InvalidPointwiseProfile)
        ));
    }

    #[test]
    fn prewarm_request_collects_one_dynamic_key_for_bounded_group() {
        let mut graph = Graph::default();
        let left = graph.input([32], PcuScalarType::F32).unwrap();
        let right = graph.input([32], PcuScalarType::F32).unwrap();
        let sum = graph.add(left, right).unwrap();
        let difference = graph.sub(sum, right).unwrap();
        let output = graph.relu(difference).unwrap();
        let prepared = prepared_for_request_test(
            &graph,
            &[output],
            TensorPointwiseGroupingPolicy::BoundedAddSubRelu,
        );

        let requests = collect_dispatch_requests(&prepared).unwrap();

        assert_eq!(requests.len(), 3);
        let kinds = requests
            .iter()
            .filter_map(|request| match request {
                TensorDispatchRequest::Fixed { kind, .. } => Some(*kind),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(
            kinds,
            [
                TensorDispatchKind::CheckedFloatAdd,
                TensorDispatchKind::CheckedFloatSub,
                TensorDispatchKind::Relu,
            ]
        );
    }

    #[test]
    fn prewarm_request_collects_identity_group_at_the_terminal_arithmetic_node() {
        let mut graph = Graph::default();
        let left = graph.input([32], PcuScalarType::F32).unwrap();
        let right = graph.input([32], PcuScalarType::F32).unwrap();
        let sum = graph.add(left, right).unwrap();
        let output = graph.sub(sum, right).unwrap();
        let prepared = prepared_for_request_test(
            &graph,
            &[output],
            TensorPointwiseGroupingPolicy::BoundedAddSubIdentity,
        );

        assert!(!prepared.suppressed_adds.contains(&sum));
        assert!(!prepared.suppressed_adds.contains(&output));
        let requests = collect_dispatch_requests(&prepared).unwrap();
        assert_eq!(requests.len(), 2);
        assert!(matches!(
            requests[0],
            TensorDispatchRequest::Fixed {
                kind: TensorDispatchKind::CheckedFloatAdd,
                ..
            }
        ));
        assert!(matches!(
            requests[1],
            TensorDispatchRequest::Fixed {
                kind: TensorDispatchKind::CheckedFloatSub,
                ..
            }
        ));
    }

    #[test]
    fn prewarm_retained_count_tracks_fifo_capacity_truncation() {
        let requested = (0..35)
            .map(|count| {
                TensorDispatchCacheKey::Fixed(
                    TensorDispatchKind::CheckedFloatAdd,
                    TensorPointwiseScalarType::F32,
                    count,
                    0,
                    PcuDispatchKernelIr::DEFAULT_REQUIREMENTS,
                )
            })
            .collect::<Vec<_>>();
        let retained_fifo = requested[3..].to_vec();

        assert_eq!(retained_requested_key_count(&requested, &retained_fifo), 32);
        assert_eq!(
            retained_requested_key_count(&requested[..3], &retained_fifo),
            0
        );
    }

    #[test]
    fn checked_integer_binary_profiles_prepare_typed_fixed_kernels_for_all_widths() {
        for scalar in [
            PcuScalarType::I8,
            PcuScalarType::U8,
            PcuScalarType::I16,
            PcuScalarType::U16,
            PcuScalarType::I32,
            PcuScalarType::U32,
            PcuScalarType::I64,
            PcuScalarType::U64,
            PcuScalarType::I128,
            PcuScalarType::U128,
            PcuScalarType::I256,
            PcuScalarType::U256,
            PcuScalarType::I512,
            PcuScalarType::U512,
        ] {
            let mut graph = Graph::default();
            let left = graph.input([17], scalar).unwrap();
            let right = graph.input([17], scalar).unwrap();
            let output = graph.add(left, right).unwrap();
            let prepared = prepared_for_request_test(
                &graph,
                &[output],
                TensorPointwiseGroupingPolicy::Disabled,
            );
            let dispatch = prepared.fixed_dispatches[2].as_ref().unwrap();
            assert_eq!(
                dispatch.value_type,
                TensorPointwiseScalarType::try_from(scalar)
                    .unwrap()
                    .value_type()
            );
            assert!(matches!(
                dispatch.cache_key,
                TensorDispatchCacheKey::Fixed(TensorDispatchKind::CheckedIntegerAdd, _, 17, 0, _)
            ));
            assert!(matches!(
                dispatch.kernel.ops,
                [
                    PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad { .. }),
                    PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad { .. }),
                    PcuDispatchOp::Data(PcuDispatchDataOp::CheckedIntegerBinary {
                        op: fusion_pcu::PcuDispatchIntegerBinaryOp::Add,
                        ..
                    }),
                    PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore { .. }),
                    PcuDispatchOp::Control(fusion_pcu::PcuDispatchControlOp::Return),
                ]
            ));
            assert_eq!(
                fusion_pcu::validate_typed_dispatch_value_flow(&dispatch.kernel),
                Ok(())
            );
            assert_eq!(
                fusion_pcu::validate_integer_checked_binary_kernel(
                    &dispatch.kernel,
                    dispatch.value_type,
                    fusion_pcu::PcuDispatchIntegerBinaryOp::Add,
                    PcuValueTypeCaps::for_value_type(dispatch.value_type),
                ),
                Ok(())
            );
        }
    }

    #[test]
    fn preflight_selects_only_requested_output_dependencies() {
        let mut graph = Graph::default();
        let left = graph.input([2, 2], PcuScalarType::F32).unwrap();
        let right = graph.input([2, 2], PcuScalarType::F32).unwrap();
        let result = graph.matmul(left, right).unwrap();
        let unrelated_left = graph.input([2, 2], PcuScalarType::F64).unwrap();
        let unrelated_right = graph.input([2, 2], PcuScalarType::F64).unwrap();
        // Unselected pure inputs can be pruned; compound Result effects cannot.
        assert_ne!(unrelated_left, unrelated_right);
        let inputs = [
            (left, Tensor::new([2, 2], vec![1.0; 4]).unwrap()),
            (right, Tensor::new([2, 2], vec![2.0; 4]).unwrap()),
        ];

        let plan = prepare_graph(&graph, result, &PureRocmAssessor).unwrap();
        validate_graph_inputs(&plan.nodes, &inputs).unwrap();

        assert_eq!(plan.nodes.len(), 3);
        assert_eq!(plan.use_counts, [1, 1, 1]);
        assert!(plan.index_by_value.contains_key(&result));
    }

    #[test]
    fn discarded_checked_f64_work_keeps_required_inputs_in_preflight() {
        let mut graph = Graph::default();
        let input = graph.input([2], PcuScalarType::F32).unwrap();
        let result = graph.relu(input).unwrap();
        let checked_left = graph.input([2], PcuScalarType::F64).unwrap();
        let checked_right = graph.input([2], PcuScalarType::F64).unwrap();
        let discarded = graph.add(checked_left, checked_right).unwrap();
        let inputs = [(input, Tensor::new([2], vec![1.0; 2]).unwrap())];
        let plan = prepare_graph(&graph, result, &PureRocmAssessor).unwrap();
        assert!(plan.index_by_value.contains_key(&discarded));
        assert!(matches!(
            validate_graph_inputs(&plan.nodes, &inputs),
            Err(TensorError::MissingInput(value)) if value == checked_left
        ));
    }

    #[test]
    fn rocm_preflight_defaults_to_strict_unrewritten_selected_schedule() {
        let mut graph = Graph::default();
        let weights = graph.input([1], PcuScalarType::F32).unwrap();
        let gradient = graph.input([1], PcuScalarType::F32).unwrap();
        let rate = graph.constant_value(TensorValue::F32(Tensor::new([1], vec![0.125]).unwrap()));
        let scaled = graph.mul(rate, gradient).unwrap();
        let updated = graph.sub(weights, scaled).unwrap();
        let prepared = prepare_graph(&graph, updated, &PureRocmAssessor).unwrap();

        assert_eq!(prepared.lowering_plan.rewrites(), &[]);
        assert!(prepared.lowering_plan.suppressed_values().is_empty());
        assert_eq!(prepared.lowering_plan.output_values(), &[updated]);
        assert_eq!(
            prepared.nodes.len(),
            prepared.tensor_plan.node_order().len()
        );
        assert_eq!(prepared.use_counts, prepared.tensor_plan.use_counts());
        assert!(matches!(prepared.nodes[3].op, OpDescriptor::Mul { .. }));
        assert!(matches!(prepared.nodes[4].op, OpDescriptor::Sub { .. }));
    }

    #[test]
    fn checked_uniform_storage_is_dense_for_binary_and_unary_consumers() {
        let mut graph = Graph::default();
        let input = graph.input([4], PcuScalarType::F32).unwrap();
        let uniform = graph
            .uniform_value([4], TensorScalarValue::F32(0.25))
            .unwrap();
        let output = graph.add(input, uniform).unwrap();
        let prepared = prepare_graph_outputs_plan(&graph, &[output], &PureRocmAssessor).unwrap();

        assert_eq!(
            prepared.physical_layouts[&uniform],
            RocmPhysicalLayout::dense(16)
        );
        assert!(prepared.storage_constraints.iter().any(|constraint| {
            (constraint.left == uniform && constraint.left_bytes == 16)
                || (constraint.right == uniform && constraint.right_bytes == 16)
        }));

        let mut graph = Graph::default();
        let uniform = graph
            .uniform_value([4], TensorScalarValue::F32(0.25))
            .unwrap();
        let output = graph.relu(uniform).unwrap();
        let prepared = prepare_graph_outputs_plan(&graph, &[output], &PureRocmAssessor).unwrap();

        assert_eq!(
            prepared.physical_layouts[&uniform],
            RocmPhysicalLayout::dense(16)
        );
        assert!(prepared.storage_constraints.iter().any(|constraint| {
            (constraint.left == uniform && constraint.left_bytes == 16)
                || (constraint.right == uniform && constraint.right_bytes == 16)
        }));

        let mut graph = Graph::default();
        let input = graph.input([4], PcuScalarType::F32).unwrap();
        let uniform = graph
            .uniform_value([4], TensorScalarValue::F32(0.25))
            .unwrap();
        let output = graph.add(input, uniform).unwrap();
        let prepared = prepare_graph_outputs_plan(&graph, &[output], &DenseOnlyAssessor).unwrap();
        assert_eq!(
            prepared.physical_layouts[&uniform],
            RocmPhysicalLayout::dense(16)
        );
        assert!(prepared.storage_constraints.iter().any(|constraint| {
            (constraint.left == uniform && constraint.left_bytes == 16)
                || (constraint.right == uniform && constraint.right_bytes == 16)
        }));
    }

    #[test]
    fn checked_float_binary_uniform_operands_are_materialized_dense() {
        for (operation, expected_kind) in [
            ("add", TensorDispatchKind::CheckedFloatAdd),
            ("sub", TensorDispatchKind::CheckedFloatSub),
            ("mul", TensorDispatchKind::CheckedFloatMul),
            ("div", TensorDispatchKind::CheckedFloatDiv),
        ] {
            let mut graph = Graph::default();
            let input = graph.input([5], PcuScalarType::F32).unwrap();
            let uniform = graph
                .uniform_value([5], TensorScalarValue::F32(2.0))
                .unwrap();
            let output = match operation {
                "add" => graph.add(input, uniform).unwrap(),
                "sub" => graph.sub(input, uniform).unwrap(),
                "mul" => graph.mul(input, uniform).unwrap(),
                "div" => graph.div(input, uniform).unwrap(),
                _ => unreachable!(),
            };
            let prepared = prepared_for_request_test(
                &graph,
                &[output],
                TensorPointwiseGroupingPolicy::Disabled,
            );

            assert_eq!(
                prepared.data.physical_layouts[&uniform],
                RocmPhysicalLayout::dense(20),
                "checked {operation} must materialize uniform operands for per-element indexing"
            );
            let dispatch = prepared.fixed_dispatches[2]
                .as_ref()
                .expect("checked binary gets a fixed dispatch");
            assert!(matches!(
                dispatch.cache_key,
                TensorDispatchCacheKey::Fixed(kind, _, 5, 0, _) if kind == expected_kind
            ));
        }
    }

    #[test]
    #[ignore = "requires native device; private scratch exclusivity, ownership and terminal retry"]
    #[allow(clippy::cognitive_complexity)] // Exhaustive typed fault/ownership matrix retains its exact witnesses.
    fn owned_private_scratch_is_exclusive_and_retries_after_terminal_fault() {
        let (_discovery, session) = rocm_test_session();
        let pool = PcuMemoryPoolId(0x4352_0180);
        let assessor = RocmTensorAssessor::new(&session).unwrap();
        let mut memory = session.memory_provider(pool);
        macro_rules! check {
            ($ty:ty, $kind:ident) => {{
                let mut graph = Graph::default();
                let input = graph.input([65], PcuScalarType::$kind).unwrap();
                let literal = graph.uniform_typed([65], 0.25 as $ty).unwrap();
                let difference = graph.sub(input, literal.erase()).unwrap();
                let output = graph.mul(difference, difference).unwrap();
                let program = graph.into_selected_program(
                    &[output], TensorArithmeticRewritePolicy::Disabled,
                    TensorArithmeticCapability::Strict, TensorPointwiseGroupingPolicy::Disabled,
                ).unwrap();
                let prepared = assessor.prepare_owned_program(program).unwrap();
                let owner = PcuDeviceTensor::new([65], session.upload_buffer(pool, &[2.0 as $ty; 65]).unwrap()).unwrap();
                let bank = prepared.scratch.bind(session.tensor_runtime(), &prepared.view(), pool, &mut memory).unwrap();
                #[cfg(feature = "allocation-census")]
                crate::reset_rocm_api_census();
                assert!(matches!(assessor.execute_owned_program_outputs(&prepared, &[(input, &owner)], pool, &mut memory), Err(RocmTensorExecutionError::ScratchBusy)));
                #[cfg(feature = "allocation-census")]
                assert_eq!(crate::rocm_api_census().allocations, 0);
                drop(bank);
                let first = assessor.execute_owned_program_outputs(&prepared, &[(input, &owner)], pool, &mut memory).unwrap().pop().unwrap().1;
                let changed = PcuDeviceTensor::new([65], session.upload_buffer(pool, &[1.0 as $ty; 65]).unwrap()).unwrap();
                #[cfg(feature = "allocation-census")]
                crate::reset_rocm_api_census();
                let second = assessor.execute_owned_program_outputs(&prepared, &[(input, &changed)], pool, &mut memory).unwrap().pop().unwrap().1;
                #[cfg(feature = "allocation-census")]
                {
                    let api = crate::rocm_api_census();
                    // Only the escaped result is fresh. Private values and observed-sentinel
                    // faultwords remain retained under the same exclusive bank borrow.
                    assert_eq!(api.allocations, 1);
                    assert_eq!(api.host_to_device_copies, 0);
                    assert_eq!(api.symbol_resolutions, 0);
                    assert_eq!(api.module_loads, 0);
                }
                let mut bad = [2.0 as $ty; 65];
                bad[2] = <$ty>::INFINITY;
                let invalid = PcuDeviceTensor::new([65], session.upload_buffer(pool, &bad).unwrap()).unwrap();
                assert!(matches!(assessor.execute_owned_program_outputs(&prepared, &[(input, &invalid)], pool, &mut memory), Err(RocmTensorExecutionError::ExecutionFault(fault)) if fault.invocation_id == 2 && fault.kind == fusion_pcu::PcuExecutionFaultKind::InvalidFloatingOperand));
                let mut unavailable_pool_provider = session.memory_provider(PcuMemoryPoolId(0x4352_0181));
                assert!(matches!(assessor.execute_owned_program_outputs(&prepared, &[(input, &owner)], pool, &mut unavailable_pool_provider), Err(RocmTensorExecutionError::Memory(_))));
                #[cfg(feature = "allocation-census")]
                crate::reset_rocm_api_census();
                let retried = assessor.execute_owned_program_outputs(&prepared, &[(input, &owner)], pool, &mut memory).unwrap().pop().unwrap().1;
                #[cfg(feature = "allocation-census")]
                {
                    let api = crate::rocm_api_census();
                    assert_eq!(api.allocations, 1);
                    // The faulted subtraction resets; the unexecuted multiply still has MAX.
                    assert_eq!(api.host_to_device_copies, 1);
                }

                // Protocol injection, not a manufactured SDK failure: an unknown terminal
                // result must retire every private owner and refuse a later call before work.
                let mut bank = prepared.scratch.bind(session.tensor_runtime(), &prepared.view(), pool, &mut memory).unwrap();
                let leased = bank.physical.iter().find(|resource| resource.device_buffer().len() == 8).unwrap().clone_for_tensor_input();
                leased.device_buffer().with_access_lease_for_test(|| {
                    bank.finish(Some(&RocmTensorExecutionError::FailedCompletion)).unwrap();
                }).unwrap();
                drop(bank);
                assert!(matches!(assessor.execute_owned_program_outputs(&prepared, &[(input, &owner)], pool, &mut memory), Err(RocmTensorExecutionError::ScratchMismatch)));
                drop(prepared);
                for (escaped, expected) in [(first, 3.0625 as $ty), (second, 0.5625 as $ty), (retried, 3.0625 as $ty)] {
                    let mut actual = [99.0 as $ty; 66];
                    session.download_buffer(pool, escaped.buffer(), &mut actual[..65]).unwrap();
                    assert!(actual[..65].iter().all(|value| value.to_bits() == expected.to_bits()));
                    assert_eq!(actual[65].to_bits(), (99.0 as $ty).to_bits());
                }
            }};
        }
        check!(f32, F32);
        check!(f64, F64);
    }

    #[test]
    #[ignore = "requires authorized device; exact F64 Uniform selected-result underflow"]
    fn f64_uniform_consumers_preserve_all_underflow_policies() {
        let (_discovery, session) = rocm_test_session();
        let pool = PcuMemoryPoolId(0x4352_0173);
        let assessor = RocmTensorAssessor::new(&session).unwrap();
        let mut memory = session.memory_provider(pool);
        let owner =
            PcuDeviceTensor::new([3], session.upload_buffer(pool, &[2.0_f64; 3]).unwrap()).unwrap();
        for policy in [
            fusion_pcu::PcuFloatUnderflowPolicy::IeeeAfterRounding,
            fusion_pcu::PcuFloatUnderflowPolicy::RejectSubnormalResult,
            fusion_pcu::PcuFloatUnderflowPolicy::AllowGradualUnderflow,
        ] {
            let mut graph = Graph::default();
            let input = graph.input([3], PcuScalarType::F64).unwrap();
            let literal = graph
                .uniform_value([3], TensorScalarValue::F64(f64::from_bits(1)))
                .unwrap();
            let output = graph.mul(input, literal).unwrap();
            graph
                .set_value_float_underflow_policy(output, policy)
                .unwrap();
            let program = graph
                .into_selected_program(
                    &[output],
                    TensorArithmeticRewritePolicy::Disabled,
                    TensorArithmeticCapability::Strict,
                    TensorPointwiseGroupingPolicy::Disabled,
                )
                .unwrap();
            let prepared = assessor.prepare_owned_program(program).unwrap();
            for _ in 0..2 {
                let result = assessor.execute_owned_program_outputs(
                    &prepared,
                    &[(input, &owner)],
                    pool,
                    &mut memory,
                );
                if policy == fusion_pcu::PcuFloatUnderflowPolicy::RejectSubnormalResult {
                    assert!(
                        matches!(result,Err(RocmTensorExecutionError::ExecutionFault(fault))
                        if fault.kind == fusion_pcu::PcuExecutionFaultKind::ArithmeticUnderflow && fault.invocation_id == 0)
                    );
                } else {
                    let outputs = result.unwrap();
                    let mut actual = [99.0_f64; 4];
                    session
                        .download_buffer(pool, outputs[0].1.buffer(), &mut actual[..3])
                        .unwrap();
                    assert_eq!(actual.map(f64::to_bits), [2, 2, 2, 99.0_f64.to_bits()]);
                }
            }
            let mut unchanged = [0.0_f64; 3];
            session
                .download_buffer(pool, owner.buffer(), &mut unchanged)
                .unwrap();
            assert_eq!(unchanged.map(f64::to_bits), [2.0_f64.to_bits(); 3]);
        }
    }

    #[test]
    #[ignore = "requires authorized native device; exact F64 literal/consumer law"]
    #[allow(clippy::too_many_lines)] // Raw transport and the consuming fault share the same native owner proof.
    fn f64_literals_preserve_payloads_and_fault_only_when_consumed() {
        let (_discovery, session) = rocm_test_session();
        let pool = PcuMemoryPoolId(0x4352_0172);
        let assessor = RocmTensorAssessor::new(&session).unwrap();
        let mut memory = session.memory_provider(pool);
        let patterns = [
            0,
            0x8000_0000_0000_0000,
            0x7ff0_0000_0000_0000,
            0xfff0_0000_0000_0000,
            0x7ff0_0000_0000_1234,
            0xfff8_0000_0000_5678,
            1,
            0x8000_0000_0000_0001,
            0x3ff0_0000_0000_1000,
        ];
        for strict in [false, true] {
            for native in [false, true] {
                for optimized in [false, true] {
                    for uniform in [false, true] {
                        let mut graph = Graph::default();
                        if strict {
                            graph.set_numerical_mode(fusion_pcu::PcuNumericalMode::Strict);
                        }
                        graph.set_numerical_options(fusion_pcu::PcuNumericalOptions {
                            compound_arithmetic: if native {
                                fusion_pcu::PcuCompoundArithmeticPolicy::BackendDefined
                            } else {
                                fusion_pcu::PcuCompoundArithmeticPolicy::Checked
                            },
                            precision: if optimized {
                                fusion_pcu::PcuPrecisionPolicy::BackendOptimized
                            } else {
                                fusion_pcu::PcuPrecisionPolicy::Preserve
                            },
                            ..Default::default()
                        });
                        let values = (0..17)
                            .map(|i| f64::from_bits(patterns[i % patterns.len()]))
                            .collect::<Vec<_>>();
                        let literal = if uniform {
                            graph
                                .uniform_value([17], TensorScalarValue::F64(values[4]))
                                .unwrap()
                        } else {
                            graph.constant_value(TensorValue::F64(
                                Tensor::new([17], values.clone()).unwrap(),
                            ))
                        };
                        let expected = if uniform {
                            vec![patterns[4]; 17]
                        } else {
                            values.iter().map(|v| v.to_bits()).collect()
                        };
                        let program = graph
                            .into_selected_program(
                                &[literal],
                                TensorArithmeticRewritePolicy::Disabled,
                                TensorArithmeticCapability::Strict,
                                TensorPointwiseGroupingPolicy::Disabled,
                            )
                            .unwrap();
                        #[cfg(feature = "allocation-census")]
                        crate::reset_rocm_api_census();
                        let prepared = assessor.prepare_owned_program(program).unwrap();
                        #[cfg(feature = "allocation-census")]
                        {
                            let api = crate::rocm_api_census();
                            assert_eq!(api.allocations, 0);
                            assert_eq!(api.module_loads, 0);
                        }
                        let first = assessor
                            .execute_owned_program_output_from_inputs::<f64, _>(
                                &prepared,
                                &[],
                                pool,
                                &mut memory,
                            )
                            .unwrap();
                        let second = assessor
                            .execute_owned_program_output_from_inputs::<f64, _>(
                                &prepared,
                                &[],
                                pool,
                                &mut memory,
                            )
                            .unwrap();
                        drop(prepared);
                        for output in [first, second] {
                            let mut observed = [99.0_f64; 18];
                            session
                                .download_buffer(pool, output.buffer(), &mut observed[..17])
                                .unwrap();
                            assert_eq!(
                                observed[..17]
                                    .iter()
                                    .map(|v| v.to_bits())
                                    .collect::<Vec<_>>(),
                                expected
                            );
                            assert_eq!(observed[17].to_bits(), 99.0_f64.to_bits());
                        }
                    }
                }
            }
        }
        let mut graph = Graph::default();
        let input = graph.input([3], PcuScalarType::F64).unwrap();
        let constant = graph.constant_value(TensorValue::F64(
            Tensor::new([3], vec![1.0, 1.0, f64::from_bits(patterns[4])]).unwrap(),
        ));
        let output = graph.mul(input, constant).unwrap();
        let program = graph
            .into_selected_program(
                &[output],
                TensorArithmeticRewritePolicy::Disabled,
                TensorArithmeticCapability::Strict,
                TensorPointwiseGroupingPolicy::Disabled,
            )
            .unwrap();
        let prepared = assessor.prepare_owned_program(program).unwrap();
        let input_owner =
            PcuDeviceTensor::new([3], session.upload_buffer(pool, &[2.0_f64; 3]).unwrap()).unwrap();
        let result = assessor.execute_owned_program_outputs(
            &prepared,
            &[(input, &input_owner)],
            pool,
            &mut memory,
        );
        assert!(
            matches!(result,Err(RocmTensorExecutionError::ExecutionFault(fault))
            if fault.kind == fusion_pcu::PcuExecutionFaultKind::InvalidFloatingOperand && fault.invocation_id == 2)
        );
        let mut unchanged = [0.0_f64; 3];
        session
            .download_buffer(pool, input_owner.buffer(), &mut unchanged)
            .unwrap();
        assert_eq!(unchanged.map(f64::to_bits), [2.0_f64.to_bits(); 3]);
    }

    #[test]
    #[ignore = "requires authorized Rocm device"]
    fn selected_uniform_preserves_raw_bits_with_scratch_and_output_bank() {
        let (_discovery, session) = rocm_test_session();
        let pool = PcuMemoryPoolId(0x4352_0170);
        let assessor = RocmTensorAssessor::new(&session).unwrap();
        let mut memory = session.memory_provider(pool);
        for bits in [
            0x0000_0000,
            0x8000_0000,
            0x7f80_0000,
            0xff80_0000,
            0x7f80_1234,
            0x7fc0_5678,
            0x0000_0001,
            0x8000_0001,
        ] {
            let mut graph = Graph::default();
            let uniform = graph
                .uniform_value([17], TensorScalarValue::F32(f32::from_bits(bits)))
                .unwrap();
            let prepared = assessor.prepare_graph_outputs(&graph, &[uniform]).unwrap();
            let mut scratch = assessor
                .prepare_scratch(&prepared, pool, &mut memory)
                .unwrap();
            let mut bank = assessor
                .prepare_output_bank(&prepared, pool, &mut memory)
                .unwrap();
            for _ in 0..2 {
                assessor
                    .execute_prepared_outputs_into_bank(
                        &prepared,
                        &[],
                        &mut scratch,
                        &mut bank,
                        &mut memory,
                    )
                    .unwrap();
                let observed = assessor
                    .download_output(&bank.outputs()[0], pool, &mut memory)
                    .unwrap();
                assert!(observed.data().iter().all(|value| value.to_bits() == bits));
            }
        }
    }

    #[test]
    #[ignore = "requires a working ROCm device"]
    fn tensor_checked_float_uniform_operands_produce_exact_dense_outputs() {
        let (_discovery, session) = rocm_test_session();
        let pool = PcuMemoryPoolId(0x4352_0011);
        let assessor = RocmTensorAssessor::new(&session).expect("tensor assessor");
        let mut graph = Graph::default();
        let input = graph.input([5], PcuScalarType::F32).expect("input");
        let uniform = graph
            .uniform_value([5], TensorScalarValue::F32(2.0))
            .expect("uniform operand");
        let outputs = [
            graph.add(input, uniform).expect("checked add"),
            graph.sub(input, uniform).expect("checked sub"),
            graph.mul(input, uniform).expect("checked mul"),
            graph.div(input, uniform).expect("checked div"),
        ];
        let program = graph
            .into_selected_program(
                &outputs,
                TensorArithmeticRewritePolicy::Disabled,
                TensorArithmeticCapability::Strict,
                TensorPointwiseGroupingPolicy::Disabled,
            )
            .expect("selected program");
        let prepared = assessor
            .prepare_owned_program(program)
            .expect("backend preparation");
        let input_values = [-4.0_f32, -1.0, 0.5, 3.0, 10.0];
        let input_tensor = PcuDeviceTensor::new(
            [input_values.len()],
            session
                .upload_buffer(pool, &input_values)
                .expect("upload input"),
        )
        .expect("input tensor");
        let input_id = prepared.tensor_program().input_values()[0];
        let mut memory = session.memory_provider(pool);
        let observed_outputs = assessor
            .execute_owned_program_outputs(
                &prepared,
                &[(input_id, &input_tensor)],
                pool,
                &mut memory,
            )
            .expect("execute checked binary outputs");
        let expected = [
            [-2.0_f32, 1.0, 2.5, 5.0, 12.0],
            [-6.0_f32, -3.0, -1.5, 1.0, 8.0],
            [-8.0_f32, -2.0, 1.0, 6.0, 20.0],
            [-2.0_f32, -0.5, 0.25, 1.5, 5.0],
        ];
        assert_eq!(observed_outputs.len(), outputs.len());
        for (index, ((value, tensor), expected)) in
            observed_outputs.iter().zip(expected).enumerate()
        {
            assert_eq!(*value, outputs[index]);
            assert_eq!(tensor.shape(), &[5]);
            let mut observed = [0.0_f32; 5];
            session
                .download_buffer(pool, tensor.buffer(), &mut observed)
                .expect("download output");
            assert_eq!(
                observed.map(f32::to_bits),
                expected.map(f32::to_bits),
                "checked outputs must be exact for dense count > 1"
            );
        }
    }

    #[test]
    fn resource_input_requirement_covers_dense_input_extent() {
        let mut graph = Graph::default();
        let value = graph.input([16], PcuScalarType::F32).unwrap();
        let requirement = input_storage_requirement(value, &[16], PcuScalarType::F32).unwrap();
        assert_eq!(requirement.output_bytes, 64);
        assert_eq!(requirement.access, fusion_pcu::PcuMemoryAccess::ReadOnly);
        assert_eq!(graph.shape(value).unwrap(), &[16]);
    }

    #[test]
    fn owned_f64_storage_and_scalar_profiles_use_double_precision_extents() {
        assert_eq!(byte_len_for::<f64>(&[3, 5]).unwrap(), 120);
        let mut graph = Graph::default();
        let value = graph.input([3, 5], PcuScalarType::F64).unwrap();
        let requirement = input_storage_requirement(value, &[3, 5], PcuScalarType::F64).unwrap();
        assert_eq!(requirement.scalar_type, PcuScalarType::F64);
        assert_eq!(requirement.alignment_bytes, 8);
        assert_eq!(requirement.output_bytes, 120);
        assert!(alignment_satisfies(16, 8));
        assert!(!alignment_satisfies(12, 8));
        assert!(!alignment_satisfies(4, 8));
        assert!(validate_owned_scalar_profile::<f64>(Some(PcuScalarType::F64)).is_ok());
        assert!(matches!(
            validate_owned_scalar_profile::<f64>(Some(PcuScalarType::F32)),
            Err(RocmTensorExecutionError::UnsupportedScalarType(
                PcuScalarType::F32
            ))
        ));
        assert!(matches!(
            validate_owned_scalar_profile::<f32>(Some(PcuScalarType::F64)),
            Err(RocmTensorExecutionError::UnsupportedScalarType(
                PcuScalarType::F64
            ))
        ));
        assert!(matches!(
            validate_tensor_scalar_tag::<f64>(PcuScalarType::F32),
            Err(RocmTensorExecutionError::UnsupportedScalarType(
                PcuScalarType::F32
            ))
        ));
    }

    #[test]
    fn storage_constraints_check_available_pairs_and_defer_missing_values() {
        struct TestResource(u8);
        impl fusion_pcu::PcuMemoryResource for TestResource {
            fn pool(&self) -> fusion_pcu::PcuMemoryPoolId {
                fusion_pcu::PcuMemoryPoolId(0)
            }
            fn size_bytes(&self) -> u64 {
                4
            }
            fn alignment_bytes(&self) -> u64 {
                4
            }
            fn access(&self) -> fusion_pcu::PcuMemoryAccess {
                fusion_pcu::PcuMemoryAccess::ReadWrite
            }
            fn is_device_local(&self) -> Option<bool> {
                Some(true)
            }
            fn origin(&self) -> fusion_pcu::PcuMemoryResourceOrigin {
                fusion_pcu::PcuMemoryResourceOrigin::ProviderManaged
            }
            fn overlap(
                &self,
                other: &Self,
                _left: fusion_pcu::PcuMemoryRange,
                _right: fusion_pcu::PcuMemoryRange,
            ) -> fusion_pcu::PcuMemoryOverlap {
                if self.0 == u8::MAX || other.0 == u8::MAX {
                    fusion_pcu::PcuMemoryOverlap::Unknown
                } else if self.0 == other.0 {
                    fusion_pcu::PcuMemoryOverlap::Overlapping
                } else {
                    fusion_pcu::PcuMemoryOverlap::Disjoint
                }
            }
        }

        let mut graph = Graph::default();
        let left = graph.input([], PcuScalarType::F32).unwrap();
        let right = graph.input([], PcuScalarType::F32).unwrap();
        let constraints = [TensorStorageConstraint {
            left,
            right,
            left_bytes: 4,
            right_bytes: 4,
        }];
        let indexed_constraints = [PreparedStorageConstraint {
            constraint: constraints[0],
            left_index: 0,
            right_index: 1,
            left_output_index: Some(0),
            right_output_index: None,
        }];
        let input_resources = [(left, TestResource(1)), (right, TestResource(2))];
        validate_indexed_storage_constraints(&indexed_constraints, |_, index, _| {
            input_resources.get(index).map(|(_, resource)| resource)
        })
        .unwrap();
        let scratch_resources = [(left, TestResource(3)), (right, TestResource(4))];
        validate_indexed_storage_constraints(&indexed_constraints, |_, index, _| {
            scratch_resources.get(index).map(|(_, resource)| resource)
        })
        .unwrap();
        validate_indexed_storage_constraints(&indexed_constraints, |_, index, _| {
            (index == 0).then_some(&input_resources[0].1)
        })
        .unwrap();
        let aliased = [(left, TestResource(1)), (right, TestResource(1))];
        assert!(matches!(
            validate_indexed_storage_constraints(&indexed_constraints, |_, index, _| {
                aliased.get(index).map(|(_, resource)| resource)
            }),
            Err(RocmTensorExecutionError::StorageConstraint(
                fusion_pcu::dialect::tensor::TensorStorageValidationError::Overlapping { .. }
            ))
        ));
        let unknown = [(left, TestResource(u8::MAX)), (right, TestResource(2))];
        assert!(matches!(
            validate_indexed_storage_constraints(&indexed_constraints, |_, index, _| {
                unknown.get(index).map(|(_, resource)| resource)
            }),
            Err(RocmTensorExecutionError::StorageConstraint(
                fusion_pcu::dialect::tensor::TensorStorageValidationError::UnknownOverlap { .. }
            ))
        ));
        assert!(resource_may_overlap_any(
            &TestResource(1),
            [&TestResource(2), &TestResource(1)]
        ));
        assert!(!resource_may_overlap_any(
            &TestResource(1),
            [&TestResource(2), &TestResource(3)]
        ));
    }

    #[test]
    fn dense_checked_uniform_fans_out_across_selected_elementwise_outputs() {
        let mut graph = Graph::default();
        let input = graph.input([4], PcuScalarType::F32).unwrap();
        let uniform = graph
            .uniform_value([4], TensorScalarValue::F32(0.25))
            .unwrap();
        let added = graph.add(input, uniform).unwrap();
        let subtracted = graph.sub(added, uniform).unwrap();
        let multiplied = graph.mul(subtracted, uniform).unwrap();

        let prepared =
            prepare_graph_outputs_plan(&graph, &[subtracted, multiplied], &PureRocmAssessor)
                .unwrap();

        assert_eq!(
            prepared.physical_layouts[&uniform],
            RocmPhysicalLayout::dense(16)
        );
        assert_eq!(
            prepared.lowering_plan.output_values(),
            &[subtracted, multiplied]
        );

        let uniform_constraints = prepared
            .storage_constraints
            .iter()
            .filter(|constraint| constraint.left == uniform || constraint.right == uniform)
            .collect::<Vec<_>>();
        assert_eq!(uniform_constraints.len(), 3);
        assert!(uniform_constraints.iter().all(|constraint| {
            if constraint.left == uniform {
                constraint.left_bytes == 16
            } else {
                constraint.right_bytes == 16
            }
        }));
        assert!(uniform_constraints.iter().any(|constraint| {
            (constraint.left == uniform && constraint.right == added)
                || (constraint.right == uniform && constraint.left == added)
        }));
        assert!(uniform_constraints.iter().any(|constraint| {
            (constraint.left == uniform && constraint.right == subtracted)
                || (constraint.right == uniform && constraint.left == subtracted)
        }));
        assert!(uniform_constraints.iter().any(|constraint| {
            (constraint.left == uniform && constraint.right == multiplied)
                || (constraint.right == uniform && constraint.left == multiplied)
        }));
    }

    #[test]
    fn fused_add_relu_preflight_omits_add_resource_and_overlap_facts() {
        let mut graph = Graph::default();
        let left = graph.input([4], PcuScalarType::F32).unwrap();
        let right = graph.input([4], PcuScalarType::F32).unwrap();
        let added = graph.add(left, right).unwrap();
        let output = graph.relu(added).unwrap();
        let prepared = prepare_graph_outputs_plan_with_policies(
            &graph,
            &[output],
            &PureRocmAssessor,
            TensorArithmeticRewritePolicy::Disabled,
            TensorArithmeticCapability::Strict,
            TensorPointwiseGroupingPolicy::SingleUseAddRelu,
        )
        .unwrap();
        assert!(!prepared.suppressed_adds.contains(&added));
        assert!(!prepared.fused_add_by_relu.contains_key(&output));
        assert_eq!(prepared.use_counts[prepared.index_by_value[&added]], 1);
        assert!(
            prepared
                .storage_constraints
                .iter()
                .any(|constraint| { constraint.left == added || constraint.right == added })
        );
    }

    #[test]
    fn selected_uniform_output_stays_dense_and_mixed_fanout_stays_dense() {
        let mut graph = Graph::default();
        let uniform = graph
            .uniform_value([4], TensorScalarValue::F32(0.25))
            .unwrap();
        let prepared = prepare_graph_outputs_plan(&graph, &[uniform], &PureRocmAssessor).unwrap();
        assert_eq!(
            prepared.physical_layouts[&uniform],
            RocmPhysicalLayout::dense(16)
        );

        let mut graph = Graph::default();
        let input = graph.input([4], PcuScalarType::F32).unwrap();
        let uniform = graph
            .uniform_value([4], TensorScalarValue::F32(0.25))
            .unwrap();
        let added = graph.add(input, uniform).unwrap();
        let relu = graph.relu(uniform).unwrap();
        let prepared =
            prepare_graph_outputs_plan(&graph, &[added, relu], &PureRocmAssessor).unwrap();
        assert_eq!(
            prepared.physical_layouts[&uniform],
            RocmPhysicalLayout::dense(16)
        );
        assert!(prepared.storage_constraints.iter().any(|constraint| {
            (constraint.left == uniform && constraint.left_bytes == 16)
                || (constraint.right == uniform && constraint.right_bytes == 16)
        }));
    }

    #[test]
    fn empty_uniform_is_rejected_before_rocm_allocation() {
        let mut graph = Graph::default();
        let uniform = graph
            .uniform_value([0], TensorScalarValue::F32(0.25))
            .unwrap();
        assert!(matches!(
            prepare_graph_outputs_plan(&graph, &[uniform], &PureRocmAssessor),
            Err(RocmTensorExecutionError::Unsupported {
                value,
                reason: TensorUnsupportedReason::Shape,
            }) if value == uniform
        ));
    }

    #[test]
    fn rocm_preflight_only_selects_contracted_sgd_when_policy_is_explicit_and_candidate_is_safe() {
        let mut graph = Graph::default();
        let weights = graph.input([1], PcuScalarType::F32).unwrap();
        let gradient = graph.input([1], PcuScalarType::F32).unwrap();
        let rate = graph.constant_value(TensorValue::F32(Tensor::new([1], vec![0.125]).unwrap()));
        let scaled = graph.mul(rate, gradient).unwrap();
        let updated = graph.sub(weights, scaled).unwrap();
        let explicit = prepare_graph_outputs_plan_with_arithmetic(
            &graph,
            &[updated],
            &PureRocmAssessor,
            TensorArithmeticRewritePolicy::AllowContractedArithmetic,
            TensorArithmeticCapability::ContractedMultiplyAdd,
        )
        .unwrap();

        assert_eq!(explicit.lowering_plan.rewrites().len(), 1);
        assert_eq!(explicit.lowering_plan.rewrites()[0].output, updated);
        assert_eq!(explicit.nodes.len(), 3);
        assert!(explicit.storage_constraints.iter().all(|constraint| {
            explicit.lowering_plan.index_of(constraint.left).is_some()
                && explicit.lowering_plan.index_of(constraint.right).is_some()
        }));
        assert!(matches!(
            explicit.nodes[2].op,
            OpDescriptor::SgdUpdate { .. }
        ));

        // Contracted rounding opt-in does not grant an unchecked numerical fault contract.
        assert!(matches!(prepare_graph_outputs_plan_with_arithmetic(
            &graph, &[updated], &DenseOnlyAssessor,
            TensorArithmeticRewritePolicy::AllowContractedArithmetic,
            TensorArithmeticCapability::ContractedMultiplyAdd,
        ), Err(RocmTensorExecutionError::Unsupported { value, .. }) if value == updated));

        let shared = graph.add(scaled, weights).unwrap();
        let unsupported = prepare_graph_outputs_plan_with_arithmetic(
            &graph,
            &[updated, shared],
            &PureRocmAssessor,
            TensorArithmeticRewritePolicy::AllowContractedArithmetic,
            TensorArithmeticCapability::ContractedMultiplyAdd,
        )
        .unwrap();
        assert!(unsupported.lowering_plan.rewrites().is_empty());
        assert!(matches!(unsupported.nodes[3].op, OpDescriptor::Mul { .. }));
        assert!(matches!(unsupported.nodes[4].op, OpDescriptor::Sub { .. }));
    }

    #[test]
    fn multi_output_preflight_unions_dependencies_and_pins_requested_values() {
        let mut graph = Graph::default();
        let input = graph.input([2], PcuScalarType::F32).unwrap();
        let first_constant =
            graph.constant_value(TensorValue::F32(Tensor::new([2], vec![1.0; 2]).unwrap()));
        let first = graph.add(input, first_constant).unwrap();
        let second_constant =
            graph.constant_value(TensorValue::F32(Tensor::new([2], vec![2.0; 2]).unwrap()));
        let second = graph.mul(first, second_constant).unwrap();

        let plan = prepare_graph_outputs_plan(&graph, &[second, first], &PureRocmAssessor)
            .expect("both output dependency closures are supported");

        assert_eq!(plan.nodes.len(), 5);
        assert_eq!(plan.nodes.last().unwrap().value, second);
        let first_index = plan.index_by_value[&first];
        assert_eq!(plan.use_counts[first_index], 2);
        let output_order = [second, first];
        assert_eq!(output_position(&output_order, first), Some(1));
        assert_eq!(output_position(&output_order, second), Some(0));
        assert_eq!(output_position(&output_order, input), None);
    }

    #[test]
    fn multi_output_preflight_rejects_empty_and_duplicate_outputs() {
        let mut graph = Graph::default();
        let input = graph.input([2], PcuScalarType::F32).unwrap();

        assert!(matches!(
            prepare_graph_outputs_plan(&graph, &[], &PureRocmAssessor),
            Err(RocmTensorExecutionError::EmptyOutputs)
        ));
        assert!(matches!(
            prepare_graph_outputs_plan(&graph, &[input, input], &PureRocmAssessor),
            Err(RocmTensorExecutionError::DuplicateOutput(value)) if value == input
        ));
    }

    #[test]
    fn historical_mse_shape_planning_preserves_reduction_layout() {
        let mut graph = Graph::default();
        let left = graph.input([2, 2], PcuScalarType::F32).unwrap();
        let right = graph.input([2, 2], PcuScalarType::F32).unwrap();
        let output = graph.mean_squared_error(left, right).unwrap();

        let prepared = prepare_graph(&graph, output, &PureRocmAssessor).unwrap();
        assert_eq!(prepared.nodes.last().unwrap().value, output);
        assert!(prepared.nodes.last().unwrap().shape.is_empty());
    }

    #[test]
    fn mse_scratch_capacity_covers_largest_reduction_in_multi_output_plan() {
        let mut graph = Graph::default();
        let left = graph.input([2, 3], PcuScalarType::F32).unwrap();
        let right = graph.input([2, 3], PcuScalarType::F32).unwrap();
        let first_loss = graph.mean_squared_error(left, right).unwrap();
        let next_left = graph.input([4], PcuScalarType::F32).unwrap();
        let next_right = graph.input([4], PcuScalarType::F32).unwrap();
        let second_loss = graph.mean_squared_error(next_left, next_right).unwrap();
        let plan =
            prepare_graph_outputs_plan(&graph, &[first_loss, second_loss], &PureRocmAssessor)
                .unwrap();

        assert_eq!(mse_scratch_element_count(&graph, &plan.nodes).unwrap(), 6);
    }

    #[test]
    fn mse_scratch_validates_buffer_capacity_in_bytes() {
        assert!(mse_scratch_length_fits(3, 12).unwrap());
        assert!(!mse_scratch_length_fits(3, 11).unwrap());
        assert!(matches!(
            mse_scratch_length_fits(usize::MAX, usize::MAX),
            Err(RocmTensorExecutionError::SizeOverflow)
        ));
    }

    #[test]
    fn mse_scratch_preflight_requires_full_writable_allocation() {
        #[derive(Clone, Copy)]
        struct TestResource {
            pool: fusion_pcu::PcuMemoryPoolId,
            bytes: u64,
            access: fusion_pcu::PcuMemoryAccess,
        }
        impl fusion_pcu::PcuMemoryResource for TestResource {
            fn pool(&self) -> fusion_pcu::PcuMemoryPoolId {
                self.pool
            }
            fn size_bytes(&self) -> u64 {
                self.bytes
            }
            fn alignment_bytes(&self) -> u64 {
                4
            }
            fn access(&self) -> fusion_pcu::PcuMemoryAccess {
                self.access
            }
            fn is_device_local(&self) -> Option<bool> {
                Some(true)
            }
            fn origin(&self) -> fusion_pcu::PcuMemoryResourceOrigin {
                fusion_pcu::PcuMemoryResourceOrigin::ProviderManaged
            }
            fn overlap(
                &self,
                _other: &Self,
                _self_range: fusion_pcu::PcuMemoryRange,
                _other_range: fusion_pcu::PcuMemoryRange,
            ) -> fusion_pcu::PcuMemoryOverlap {
                fusion_pcu::PcuMemoryOverlap::Disjoint
            }
        }

        let pool = fusion_pcu::PcuMemoryPoolId(4);
        let valid = TestResource {
            pool,
            bytes: 12,
            access: fusion_pcu::PcuMemoryAccess::ReadWrite,
        };
        assert!(mse_scratch_resource_fits(&valid, pool, 3).unwrap());
        assert!(!mse_scratch_resource_fits(&TestResource { bytes: 11, ..valid }, pool, 3).unwrap());
        assert!(
            !mse_scratch_resource_fits(
                &TestResource {
                    access: fusion_pcu::PcuMemoryAccess::ReadOnly,
                    ..valid
                },
                pool,
                3
            )
            .unwrap()
        );
        assert!(
            !mse_scratch_resource_fits(
                &TestResource {
                    pool: fusion_pcu::PcuMemoryPoolId(5),
                    ..valid
                },
                pool,
                3
            )
            .unwrap()
        );
    }

    #[test]
    fn historical_sgd_planning_tracks_raw_operands_without_execution_admission() {
        let mut graph = Graph::default();
        let weights = graph.input([3], PcuScalarType::F32).unwrap();
        let gradient = graph.input([3], PcuScalarType::F32).unwrap();
        let updated = graph.sgd_update(weights, gradient, 0.25).unwrap();

        let plan = prepare_graph(&graph, updated, &PureRocmAssessor).unwrap();
        let node = plan.nodes.last().unwrap();
        assert!(
            matches!(node.op, fusion_pcu::dialect::tensor::OpDescriptor::SgdUpdate {
            weights: value_weights, gradient: value_gradient, learning_rate
        } if value_weights == weights
            && value_gradient == gradient
            && (learning_rate - 0.25).abs() < f32::EPSILON)
        );
        assert_eq!(plan.use_counts, [1, 1, 1]);
        assert_eq!(
            PureRocmAssessor.assess_node(&graph, *node),
            TensorOperationSupport::Supported {
                route: TensorExecutionRoute::Native,
                workspace_bytes: Some(0),
            }
        );
    }

    #[test]
    fn scratch_retains_constants_and_non_output_computations_only() {
        let mut graph = Graph::default();
        let input = graph.input([2], PcuScalarType::F32).unwrap();
        let constant =
            graph.constant_value(TensorValue::F32(Tensor::new([2], vec![1.0; 2]).unwrap()));
        let computed = graph.add(input, constant).unwrap();
        let input_node = graph.nodes().find(|node| node.value == input).unwrap();
        let constant_node = graph.nodes().find(|node| node.value == constant).unwrap();
        let computed_node = graph.nodes().find(|node| node.value == computed).unwrap();

        assert!(!scratch_stores_node(input_node.op, input, &[computed]));
        assert!(scratch_stores_node(constant_node.op, constant, &[computed]));
        assert!(!scratch_stores_node(
            computed_node.op,
            computed,
            &[computed]
        ));
        assert!(scratch_stores_node(computed_node.op, computed, &[input]));
    }

    #[test]
    fn preflight_checks_all_reachable_inputs_before_execution() {
        let mut graph = Graph::default();
        let left = graph.input([2, 2], PcuScalarType::F32).unwrap();
        let right = graph.input([2, 2], PcuScalarType::F32).unwrap();
        let output = graph.matmul(left, right).unwrap();
        let inputs = [(left, Tensor::new([2, 2], vec![1.0; 4]).unwrap())];

        assert!(matches!(
            prepare_graph(&graph, output, &PureRocmAssessor)
                .and_then(|plan| validate_graph_inputs(&plan.nodes, &inputs).map_err(Into::into)),
            Err(RocmTensorExecutionError::Graph(TensorError::MissingInput(id))) if id == right
        ));
    }

    #[test]
    fn preflight_rejects_wrong_shaped_input() {
        let mut graph = Graph::default();
        let input = graph.input([2, 2], PcuScalarType::F32).unwrap();
        let inputs = [(input, Tensor::new([4], vec![1.0; 4]).unwrap())];

        assert!(matches!(
            prepare_graph(&graph, input, &PureRocmAssessor).and_then(|plan| validate_graph_inputs(
                &plan.nodes,
                &inputs
            )
            .map_err(Into::into)),
            Err(RocmTensorExecutionError::Graph(
                TensorError::ShapeMismatch { .. }
            ))
        ));
    }

    #[test]
    fn assessor_rejects_unimplemented_tensor_scalar_representations() {
        let mut graph = Graph::default();
        let input = graph.input([2, 2], PcuScalarType::F32).unwrap();
        let sum = graph.add(input, input).unwrap();
        let product = graph.matmul(input, input).unwrap();
        for scalar_type in fusion_pcu::PcuScalarType::ALL {
            if matches!(scalar_type, PcuScalarType::F32 | PcuScalarType::F64) {
                continue;
            }
            for value in [input, sum, product] {
                let mut node = graph.nodes().find(|node| node.value == value).unwrap();
                node.scalar_type = scalar_type;
                let is_transport_input =
                    matches!(node.op, OpDescriptor::Input) && is_transport_scalar(scalar_type);
                assert_eq!(
                    rocm_supports_operand_representation(node, TensorOperandRepresentation::Dense),
                    is_transport_input
                        || (value == sum && super::is_checked_integer_scalar(scalar_type))
                        || (value == sum
                            && matches!(
                                scalar_type,
                                PcuScalarType::F16
                                    | PcuScalarType::BF16
                                    | PcuScalarType::F8E4M3FN
                                    | PcuScalarType::F8E5M2
                            ))
                );
                assert!(!rocm_supports_operand_representation(
                    node,
                    TensorOperandRepresentation::UniformScalar
                ));
                assert_eq!(
                    assess_tensor_node(&graph, node),
                    if is_transport_input {
                        TensorOperationSupport::Supported {
                            route: TensorExecutionRoute::Native,
                            workspace_bytes: Some(0),
                        }
                    } else {
                        TensorOperationSupport::Unsupported {
                            reason: TensorUnsupportedReason::ElementType,
                        }
                    }
                );
            }
        }
    }

    #[test]
    fn assessor_admits_dense_f64_literals_and_arithmetic() {
        let mut graph = Graph::default();
        graph.set_numerical_mode(fusion_pcu::PcuNumericalMode::Strict);
        let input = graph.input_typed::<f64>([2, 2]).unwrap();
        let constant = graph.constant_typed(Tensor::<f64>::splat([2, 2], 1.0).unwrap());
        let uniform = graph.uniform_typed([2, 2], 2.0_f64).unwrap();
        let sum = graph.add_typed(input, constant).unwrap();
        let product = graph.matmul_typed(sum, uniform).unwrap();
        for (value, expected) in [
            (
                input.erase(),
                TensorOperationSupport::Supported {
                    route: TensorExecutionRoute::Native,
                    workspace_bytes: Some(0),
                },
            ),
            (
                sum.erase(),
                TensorOperationSupport::Supported {
                    route: TensorExecutionRoute::Synthesized,
                    workspace_bytes: Some(0),
                },
            ),
            (
                product.erase(),
                TensorOperationSupport::Supported {
                    route: TensorExecutionRoute::Synthesized,
                    workspace_bytes: Some(0),
                },
            ),
        ] {
            let node = graph.nodes().find(|node| node.value == value).unwrap();
            assert_eq!(node.scalar_type, PcuScalarType::F64);
            assert_eq!(assess_tensor_node(&graph, node), expected);
            assert!(rocm_supports_operand_representation(
                node,
                TensorOperandRepresentation::Dense
            ));
            assert!(!rocm_supports_operand_representation(
                node,
                TensorOperandRepresentation::UniformScalar
            ));
        }
        for value in [constant.erase(), uniform.erase()] {
            let node = graph.nodes().find(|node| node.value == value).unwrap();
            assert_eq!(
                assess_tensor_node(&graph, node),
                TensorOperationSupport::Supported {
                    route: TensorExecutionRoute::Native,
                    workspace_bytes: Some(0),
                }
            );
        }
        assert!(prepare_graph(&graph, product.erase(), &PureRocmAssessor).is_ok());
    }

    #[test]
    fn assessor_requires_strict_synthesis_for_dense_rank_two_matmul() {
        let mut graph = Graph::default();
        let left = graph.input([2, 3], PcuScalarType::F32).unwrap();
        let right = graph.input([3, 4], PcuScalarType::F32).unwrap();
        let result = graph.matmul(left, right).unwrap();
        let node = graph.node(result).unwrap();
        assert!(matches!(
            assess_tensor_node(&graph, node),
            TensorOperationSupport::Unsupported { .. }
        ));
        graph
            .set_value_numerical_mode(result, fusion_pcu::PcuNumericalMode::Strict)
            .unwrap();
        assert_eq!(
            assess_tensor_node(&graph, graph.node(result).unwrap()),
            TensorOperationSupport::Supported {
                route: TensorExecutionRoute::Synthesized,
                workspace_bytes: Some(0),
            }
        );
    }

    #[test]
    fn unproved_compounds_reject_both_modes_before_storage_or_library_preparation() {
        for mode in [
            fusion_pcu::PcuNumericalMode::Boundary,
            fusion_pcu::PcuNumericalMode::Strict,
        ] {
            for operation in ["mse", "sgd", "relu_backward"] {
                let mut graph = Graph::default();
                graph.set_numerical_mode(mode);
                let first = graph.input([2], PcuScalarType::F32).unwrap();
                let second = graph.input([2], PcuScalarType::F32).unwrap();
                let output = match operation {
                    "mse" => graph.mean_squared_error(first, second).unwrap(),
                    "sgd" => graph.sgd_update(first, second, 0.25).unwrap(),
                    "relu_backward" => graph.relu_backward(first, second).unwrap(),
                    _ => unreachable!(),
                };
                if operation == "relu_backward"
                    || ((operation == "sgd" || operation == "mse")
                        && mode == fusion_pcu::PcuNumericalMode::Strict)
                {
                    assert!(matches!(
                        assess_tensor_node(&graph, graph.node(output).unwrap()),
                        TensorOperationSupport::Supported {
                            route: TensorExecutionRoute::Synthesized,
                            ..
                        }
                    ));
                    assert!(
                        prepare_graph_outputs_plan(&graph, &[output], &DenseOnlyAssessor).is_ok()
                    );
                    continue;
                }
                assert!(matches!(
                    assess_tensor_node(&graph, graph.node(output).unwrap()),
                    TensorOperationSupport::Unsupported { .. }
                ));
                assert!(
                    matches!(prepare_graph_outputs_plan(&graph, &[output], &DenseOnlyAssessor),
                    Err(RocmTensorExecutionError::Unsupported { value, .. }) if value == output)
                );
                let program = graph
                    .into_selected_program(
                        &[output],
                        TensorArithmeticRewritePolicy::Disabled,
                        TensorArithmeticCapability::Strict,
                        TensorPointwiseGroupingPolicy::Disabled,
                    )
                    .unwrap();
                assert!(
                    matches!(prepare_owned_graph_data(&program, &DenseOnlyAssessor),
                    Err(RocmTensorExecutionError::Unsupported { value, .. }) if value == output)
                );
            }
        }
    }

    #[test]
    fn assessor_synthesizes_checked_add_relu_and_rejects_raw_mse() {
        let mut graph = Graph::default();
        let first = graph.input([2, 2], PcuScalarType::F32).unwrap();
        let second = graph.input([2, 2], PcuScalarType::F32).unwrap();
        let add = graph.add(first, second).unwrap();
        let relu = graph.relu(first).unwrap();
        let loss = graph.mean_squared_error(first, second).unwrap();

        let add_node = graph.nodes().find(|node| node.value == add).unwrap();
        assert_eq!(
            assess_tensor_node(&graph, add_node),
            TensorOperationSupport::Supported {
                route: TensorExecutionRoute::Synthesized,
                workspace_bytes: Some(0),
            }
        );

        let relu_node = graph.nodes().find(|node| node.value == relu).unwrap();
        assert_eq!(
            assess_tensor_node(&graph, relu_node),
            TensorOperationSupport::Supported {
                route: TensorExecutionRoute::Synthesized,
                workspace_bytes: Some(0),
            }
        );

        let node = graph.nodes().find(|node| node.value == loss).unwrap();
        assert!(matches!(
            assess_tensor_node(&graph, node),
            TensorOperationSupport::Unsupported { .. }
        ));
    }

    #[test]
    fn assessor_rejects_zero_sized_matmul_output() {
        let mut graph = Graph::default();
        let left = graph.input([0, 3], PcuScalarType::F32).unwrap();
        let right = graph.input([3, 2], PcuScalarType::F32).unwrap();
        let result = graph.matmul(left, right).unwrap();
        let node = graph.nodes().find(|node| node.value == result).unwrap();

        assert_eq!(
            assess_tensor_node(&graph, node),
            TensorOperationSupport::Unsupported {
                reason: TensorUnsupportedReason::Shape,
            }
        );
    }

    #[test]
    fn assessor_rejects_zero_inner_dimension_even_with_nonzero_output() {
        let mut graph = Graph::default();
        let left = graph.input([2, 0], PcuScalarType::F32).unwrap();
        let right = graph.input([0, 3], PcuScalarType::F32).unwrap();
        let result = graph.matmul(left, right).unwrap();
        let node = graph.nodes().find(|node| node.value == result).unwrap();

        assert_eq!(
            assess_tensor_node(&graph, node),
            TensorOperationSupport::Unsupported {
                reason: TensorUnsupportedReason::Shape,
            }
        );
    }

    #[test]
    fn assessor_rejects_dimensions_outside_rocblas_integer_range() {
        let mut graph = Graph::default();
        let left = graph
            .input([i32::MAX as usize + 1, 1], PcuScalarType::F32)
            .unwrap();
        let right = graph.input([1, 1], PcuScalarType::F32).unwrap();
        let result = graph.matmul(left, right).unwrap();
        let node = graph.nodes().find(|node| node.value == result).unwrap();

        assert_eq!(
            assess_tensor_node(&graph, node),
            TensorOperationSupport::Unsupported {
                reason: TensorUnsupportedReason::Shape,
            }
        );
    }
}

#[cfg(test)]
#[path = "tensor/integer_literals/integer_literals.rs"]
mod integer_literals;
