//! Opt-in tensor operation assessment for an explicitly selected `ROCm` session.
//!
//! This adapter executes dense row-major f32 matrix multiplication through rocBLAS and elementwise
//! addition, `ReLU`, and explicitly selected bounded Mul chains through owned PCU Dispatch.
//! Other tensor operations remain unsupported.

mod feedback_runtime;
pub use feedback_runtime::{
    RocmAdmittedTensorFeedbackResources,
    RocmTensorFeedbackPrepareError,
    RocmTensorFeedbackReleaseError,
    RocmTensorFeedbackResources,
};

use std::{
    cell::RefCell,
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
    time::{
        Duration,
        Instant,
    },
};
use smallvec::SmallVec;

use fusion_pcu::{
    PcuBinding,
    PcuBindingAccess,
    PcuBindingRef,
    PcuBindingStorageClass,
    PcuBindingType,
    PcuCompletionOutcome,
    PcuDispatchControlOp,
    PcuDispatchDataOp,
    PcuDispatchEntryPoint,
    PcuDispatchAluOp,
    PcuDispatchFeatureCaps,
    PcuDispatchIndex,
    PcuDispatchKernelIr,
    PcuDispatchOp,
    PcuDispatchValueId,
    PcuInvocationShape,
    PcuMemoryAccess,
    PcuMemoryAllocationRequest,
    PcuMemoryHostAccess,
    PcuMemoryMemberRequirement,
    PcuMemoryResource,
    PcuMemoryProvider,
    PcuMemoryProviderError,
    PcuMemoryPoolId,
    PcuOwnedDispatchMemorySession,
    PcuOwnedCompletion,
    PcuParameterValue,
    PcuValueType,
    PcuValueTypeCaps,
    validate_reusable_memory_bank_members_by,
};
use std::num::NonZeroU32;
use fusion_pcu_tensor::{
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
    TensorStorageConstraint,
    TensorStorageValidationError,
    TensorUnsupportedReason,
    ValueId,
};

use crate::{
    HipCompletionBatch,
    HipKernel,
    HipKernelArgument,
    HipStreamHandle,
    HipTimingEventHandle,
    Rocblas,
    RocblasError,
    RocmMemoryResource,
    RocmOwnedDispatchBackend,
    RocmPreparedDispatch,
};

const ADD_DISPATCH_CACHE_CAPACITY: usize = 32;
const TENSOR_EXECUTION_INLINE_NODES: usize = 8;

type TensorExecutionResources =
    SmallVec<[Option<RocmMemoryResource>; TENSOR_EXECUTION_INLINE_NODES]>;
type TensorExecutionUseCounts = SmallVec<[usize; TENSOR_EXECUTION_INLINE_NODES]>;
type TensorExecutionOutputs<'session> = SmallVec<[RocmTensorInput<'session>; 4]>;
const RELU_BACKWARD_SOURCE: &str = r#"
extern "C" __global__ void tensor_relu_backward(
    const float *input, const float *upstream, float *output, unsigned int n) {
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) output[id] = input[id] > 0.0f ? upstream[id] : 0.0f;
}
"#;
const SGD_UPDATE_SOURCE: &str = r#"
extern "C" __global__ void tensor_sgd_update(
    const float *weights, const float *gradient, float *output,
    float learning_rate, unsigned int n) {
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) {
        // Keep the strict SgdUpdate contract: round the product before subtracting. HIPRTC's
        // default contraction fuses this expression and disagrees with the CPU reference.
        volatile float product = learning_rate * gradient[id];
        output[id] = weights[id] - product;
    }
}
"#;
const SGD_UPDATE_CONTRACTED_SOURCE: &str = r#"
extern "C" __global__ void tensor_sgd_update_contracted(
    const float *weights, const float *gradient, float *output,
    float learning_rate, unsigned int n) {
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) output[id] = __builtin_fmaf(-learning_rate, gradient[id], weights[id]);
}
"#;

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

const MSE_LEFT: PcuDispatchValueId = PcuDispatchValueId(1);
const MSE_RIGHT: PcuDispatchValueId = PcuDispatchValueId(2);
const MSE_DIFF: PcuDispatchValueId = PcuDispatchValueId(3);
const MSE_SQUARE: PcuDispatchValueId = PcuDispatchValueId(4);
const MSE_LEFT_REF: PcuBindingRef = PcuBindingRef::new(0, 0);
const MSE_RIGHT_REF: PcuBindingRef = PcuBindingRef::new(0, 1);
const MSE_OUTPUT_REF: PcuBindingRef = PcuBindingRef::new(0, 2);
const MSE_BINDINGS: &[PcuBinding<'static>] = &[
    PcuBinding::value(
        Some("prediction"),
        0,
        0,
        PcuBindingStorageClass::Storage,
        PcuBindingAccess::ReadOnly,
        PcuValueType::f32(),
    ),
    PcuBinding::value(
        Some("target"),
        0,
        1,
        PcuBindingStorageClass::Storage,
        PcuBindingAccess::ReadOnly,
        PcuValueType::f32(),
    ),
    PcuBinding::value(
        Some("squared_error"),
        0,
        2,
        PcuBindingStorageClass::Storage,
        PcuBindingAccess::WriteOnly,
        PcuValueType::f32(),
    ),
];
const MSE_OPS: &[PcuDispatchOp<'static>] = &[
    PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
        result: MSE_LEFT,
        binding: MSE_LEFT_REF,
        index: PcuDispatchIndex::InvocationId,
    }),
    PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
        result: MSE_RIGHT,
        binding: MSE_RIGHT_REF,
        index: PcuDispatchIndex::InvocationId,
    }),
    PcuDispatchOp::Data(PcuDispatchDataOp::Alu {
        value_type: fusion_pcu::PcuValueType::f32(),
        result: MSE_DIFF,
        op: fusion_pcu::PcuDispatchAluOp::Sub,
        lhs: MSE_LEFT,
        rhs: MSE_RIGHT,
    }),
    PcuDispatchOp::Data(PcuDispatchDataOp::Alu {
        value_type: fusion_pcu::PcuValueType::f32(),
        result: MSE_SQUARE,
        op: fusion_pcu::PcuDispatchAluOp::Mul,
        lhs: MSE_DIFF,
        rhs: MSE_DIFF,
    }),
    PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
        binding: MSE_OUTPUT_REF,
        index: PcuDispatchIndex::InvocationId,
        value: MSE_SQUARE,
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
    SquaredDifference,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum TensorDispatchCacheKey {
    Fixed(TensorDispatchKind, u32, u8),
    Pointwise {
        invocation_count: u32,
        scalar_mask: u8,
        topology: SmallVec<[u8; 64]>,
    },
}

type TensorDispatchCache = VecDeque<(TensorDispatchCacheKey, RocmPreparedDispatch)>;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct TensorDispatchCacheAdmission {
    compiled: bool,
    evicted: bool,
}

#[derive(Clone, Debug)]
enum TensorDispatchRequest<'graph> {
    Fixed {
        kind: TensorDispatchKind,
        logical_count: u32,
        scalar_mask: u8,
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
            Self::Fixed {
                kind,
                logical_count,
                scalar_mask,
            } => TensorDispatchCacheKey::Fixed(*kind, *logical_count, *scalar_mask),
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
    shape: &'a [usize],
    left: &'a RocmMemoryResource,
    right: Option<&'a RocmMemoryResource>,
    output: &'a RocmMemoryResource,
    scalar_mask: u8,
}

impl TensorDispatchKind {
    fn kernel(self, logical_count: u32, scalar_mask: u8) -> PcuDispatchKernelIr<'static> {
        match self {
            Self::Add => add_kernel(logical_count, scalar_mask),
            Self::AddRelu => add_relu_kernel(logical_count, scalar_mask),
            Self::Sub => sub_kernel(logical_count, scalar_mask),
            Self::Mul => mul_kernel(logical_count, scalar_mask),
            Self::Relu => relu_kernel(logical_count),
            Self::SquaredDifference => mse_kernel(logical_count),
        }
    }
}

fn mse_kernel(logical_count: u32) -> PcuDispatchKernelIr<'static> {
    PcuDispatchKernelIr {
        id: fusion_pcu::PcuKernelId(0x4d53_4521),
        entry: PcuDispatchEntryPoint {
            name: "tensor_mse_squared_difference",
            logical_shape: [logical_count, 1, 1],
        },
        bindings: MSE_BINDINGS,
        ports: &[],
        parameters: &[],
        ops: MSE_OPS,
        type_caps: PcuValueTypeCaps::FLOAT32.union(PcuValueTypeCaps::SCALAR_VALUES),
        feature_caps: PcuDispatchFeatureCaps::default(),
    }
}

fn add_kernel(logical_count: u32, scalar_mask: u8) -> PcuDispatchKernelIr<'static> {
    PcuDispatchKernelIr {
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
    InvalidPlan(ValueId),
    MissingResource(ValueId),
    InputResourceMismatch,
    InputPoolMismatch,
    OutputResourceMismatch,
    ScratchMismatch,
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

impl Error for RocmTensorExecutionError {}

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
/// Construction loads rocBLAS, creates a handle for the selected device, and verifies the SGEMM
/// entry point. Therefore a `MatMul` node is reported as `Library` only when the route is prepared.
pub struct RocmTensorAssessor<'session> {
    rocblas: Rocblas,
    session: &'session RocmOwnedDispatchBackend,
    stream: HipStreamHandle,
    add_dispatches: RefCell<TensorDispatchCache>,
    relu_backward: RefCell<Option<(HipKernel, HipStreamHandle)>>,
    sgd_update: RefCell<Option<(HipKernel, HipStreamHandle)>>,
    sgd_update_contracted: RefCell<Option<(HipKernel, HipStreamHandle)>>,
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
/// pool identity before launching work. Contents change only through [`RocmTensorAssessor::update_input`].
pub struct RocmTensorInput<'session> {
    session: &'session RocmOwnedDispatchBackend,
    shape: Vec<usize>,
    resource: RocmMemoryResource,
}

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

    fn resource_ref(&self) -> RocmMemoryResource {
        self.resource.clone_for_tensor_input()
    }
}

impl<'session> RocmTensorAssessor<'session> {
    fn ensure_dispatch_cached(
        &self,
        key: TensorDispatchCacheKey,
        kernel: PcuDispatchKernelIr<'_>,
        shape: PcuInvocationShape,
        dynamic_tensor_kernel: bool,
    ) -> Result<TensorDispatchCacheAdmission, RocmTensorExecutionError> {
        let mut cache = self.add_dispatches.borrow_mut();
        if cache.iter().any(|(cached_key, _)| *cached_key == key) {
            return Ok(TensorDispatchCacheAdmission::default());
        }
        let prepared = if dynamic_tensor_kernel {
            self.session
                .prepare_dynamic_tensor_kernel_on_stream(kernel, shape, &self.stream)
        } else {
            self.session
                .prepare_dispatch_owned_kernel_on_stream(kernel, shape, &self.stream)
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
        logical_count: u32,
        scalar_mask: u8,
    ) -> Result<TensorDispatchCacheAdmission, RocmTensorExecutionError> {
        let invocations =
            NonZeroU32::new(logical_count).ok_or(RocmTensorExecutionError::SizeOverflow)?;
        self.ensure_dispatch_cached(
            TensorDispatchCacheKey::Fixed(kind, logical_count, scalar_mask),
            kind.kernel(logical_count, scalar_mask),
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
            .add_dispatches
            .borrow()
            .iter()
            .any(|(cached_key, _)| *cached_key == key)
        {
            return Ok(TensorDispatchCacheAdmission::default());
        }
        let (kernel_bindings, kernel_ops) = bounded_pointwise_program(group, scalar_mask)?;
        let kernel = PcuDispatchKernelIr {
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
            .add_dispatches
            .borrow()
            .iter()
            .any(|(cached_key, _)| *cached_key == key)
        {
            return Ok(TensorDispatchCacheAdmission::default());
        }
        let (kernel_bindings, kernel_ops) = bounded_mul_program(group, scalar_mask)?;
        let kernel = PcuDispatchKernelIr {
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
        if let Err(error) = self.stream.record_timing(&event) {
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
        if let Err(error) = self.stream.record_timing(&end) {
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

    /// Prepare rocBLAS for the backend's already selected device.
    ///
    /// # Errors
    ///
    /// Returns an error if rocBLAS, its SGEMM symbol, or the selected-device handle is unavailable.
    pub fn new(session: &'session RocmOwnedDispatchBackend) -> Result<Self, RocblasError> {
        let stream = session.tensor_runtime().create_stream()?;
        let mut rocblas = Rocblas::new(session.tensor_runtime())?;
        rocblas.bind_stream(&stream)?;
        Ok(Self {
            rocblas,
            session,
            stream,
            add_dispatches: RefCell::new(VecDeque::new()),
            relu_backward: RefCell::new(None),
            sgd_update: RefCell::new(None),
            sgd_update_contracted: RefCell::new(None),
        })
    }

    /// Compile the PCU dispatches selected by a prepared graph on this assessor's stream.
    ///
    /// This is an explicit, idempotent warm-up operation: it compiles selected fixed and dynamic
    /// Dispatch kernels without allocating, binding, uploading, or requiring provider resources.
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
                TensorDispatchRequest::Fixed {
                    kind,
                    logical_count,
                    scalar_mask,
                } => self.ensure_fixed_dispatch_cached(*kind, *logical_count, *scalar_mask)?,
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

        let cache = self.add_dispatches.borrow();
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
    /// Inputs and constants use the supplied provider, dense `MatMul` nodes use rocBLAS, and
    /// same-shape nonempty `Add` and `ReLU` nodes use owned PCU Dispatch. Elementwise executables
    /// are cached by operation and flattened element count in a bounded per-assessor cache. The
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
        Ok(RocmPreparedTensorGraph {
            graph,
            plan: plan.tensor_plan,
            lowering_plan: plan.lowering_plan,
            output: outputs[0],
            outputs: outputs.to_vec(),
            nodes: plan.nodes,
            index_by_value: plan.index_by_value,
            use_counts: plan.use_counts,
            fused_add_by_relu: plan.fused_add_by_relu,
            bounded_pointwise_by_output: plan.bounded_pointwise_by_output,
            bounded_mul_by_output: plan.bounded_mul_by_output,
            suppressed_adds: plan.suppressed_adds,
            storage_constraints: plan.storage_constraints,
            physical_layouts: plan.physical_layouts,
        })
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
    /// HIP nodes remain ordered on this assessor's stream. A pending batch is completed before
    /// each rocBLAS operation and before host/provider input transfers. The default execution APIs
    /// remain synchronous per node.
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
        let mut batch = HipCompletionBatch::new(&self.stream);
        let mut outputs = self.execute_prepared_outputs_with_input_sources_and_scratch(
            prepared,
            inputs,
            &[],
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
            &[],
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
        let mut resources = Vec::with_capacity(prepared.nodes.len());
        let max_mse_count = mse_scratch_element_count(prepared.graph, &prepared.nodes)?;
        for node in &prepared.nodes {
            let resource = if !prepared.suppressed_adds.contains(&node.value)
                && scratch_stores_node(node.op, node.value, &prepared.outputs)
            {
                match node.op {
                    OpDescriptor::Constant(tensor) => {
                        Some(allocate(memory, pool, node.shape, Some(tensor))?)
                    }
                    OpDescriptor::Uniform { value } => {
                        let tensor = prepared
                            .physical_layout(node.value)?
                            .uniform_tensor(node.shape, value)?;
                        Some(allocate(memory, pool, tensor.shape(), Some(&tensor))?)
                    }
                    OpDescriptor::Input => None,
                    _ => Some(allocate(memory, pool, node.shape, None)?),
                }
            } else {
                None
            };
            if let Some(resource) = &resource
                && (resource.pool() != pool
                    || !resource.belongs_to_runtime(self.session.tensor_runtime()))
            {
                return Err(RocmTensorExecutionError::ScratchMismatch);
            }
            resources.push(resource);
        }
        let mse_squared = if max_mse_count == 0 {
            None
        } else {
            Some(allocate(memory, pool, &[max_mse_count], None)?)
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
        let mut outputs = Vec::with_capacity(prepared.outputs.len());
        let mut requirements = Vec::with_capacity(prepared.outputs.len());
        for &value in &prepared.outputs {
            let shape = prepared.graph.shape(value)?.to_vec();
            let resource = allocate(memory, pool, &shape)?;
            if resource.pool() != pool
                || !resource.belongs_to_runtime(self.session.tensor_runtime())
                || resource.device_buffer().len() < byte_len(&shape)?
                || !matches!(
                    resource.access(),
                    PcuMemoryAccess::WriteOnly | PcuMemoryAccess::ReadWrite
                )
            {
                return Err(RocmTensorExecutionError::OutputResourceMismatch);
            }
            requirements.push(PcuMemoryMemberRequirement {
                pool,
                minimum_size_bytes: byte_len(&shape)? as u64,
                access: PcuMemoryAccess::ReadWrite,
                require_device_local: false,
            });
            outputs.push(RocmTensorInput {
                session: self.session,
                shape,
                resource,
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

    /// Execute into a reusable output bank with HIP dispatch batching enabled.
    ///
    /// Batches flush at host-transfer and rocBLAS boundaries, and before this method returns.
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
        let mut batch = HipCompletionBatch::new(&self.stream);
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
            &[],
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
        let mut batch = HipCompletionBatch::new(&self.stream);
        let result = self.execute_prepared_outputs_with_input_sources_and_scratch(
            prepared,
            inputs,
            &[],
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
        let mut batch = HipCompletionBatch::new(&self.stream);
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
            shape: tensor.shape().to_vec(),
            resource,
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
        input: &mut RocmTensorInput<'session>,
        tensor: &Tensor,
        memory: &mut P,
    ) -> Result<(), RocmTensorExecutionError>
    where
        P: PcuMemoryProvider<Resource = RocmMemoryResource>,
    {
        input.validate_session(self.session)?;
        if tensor.shape() != input.shape {
            return Err(TensorError::ShapeMismatch {
                left: input.shape.clone(),
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

    fn validate_execution_sources(
        &self,
        prepared: &RocmPreparedTensorGraph<'_>,
        host_inputs: &[(ValueId, Tensor)],
        resource_inputs: &[(ValueId, &RocmTensorInput<'_>)],
        pool: PcuMemoryPoolId,
        scratch: Option<&RocmTensorScratch<'_, '_, 'session>>,
        output_bank: Option<&RocmTensorOutputBank<'_, '_, 'session>>,
    ) -> Result<(), RocmTensorExecutionError> {
        validate_graph_input_sources(
            &prepared.nodes,
            host_inputs,
            resource_inputs,
            self.session,
            pool,
        )?;
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
            if let Some(scratch) = scratch {
                validate_prepared_storage_constraints(prepared, resource_inputs, scratch, bank)?;
            }
        }
        if prepared.nodes.iter().any(|node| {
            matches!(
                node.op,
                OpDescriptor::MatMul { .. } | OpDescriptor::MeanSquaredError { .. }
            )
        }) && !self.rocblas.is_usable()
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
    fn execute_prepared_outputs_with_input_sources_and_scratch<P>(
        &self,
        prepared: &RocmPreparedTensorGraph<'_>,
        host_inputs: &[(ValueId, Tensor)],
        resource_inputs: &[(ValueId, &RocmTensorInput<'_>)],
        pool: PcuMemoryPoolId,
        memory: &mut P,
        mut scratch: Option<&mut RocmTensorScratch<'_, '_, 'session>>,
        output_bank: Option<&RocmTensorOutputBank<'_, '_, 'session>>,
        timings: &mut impl NodeTimingSink,
        mut batch: Option<&mut HipCompletionBatch>,
    ) -> Result<TensorExecutionOutputs<'session>, RocmTensorExecutionError>
    where
        P: PcuMemoryProvider<Resource = RocmMemoryResource>,
    {
        self.validate_execution_sources(
            prepared,
            host_inputs,
            resource_inputs,
            pool,
            scratch.as_deref(),
            output_bank,
        )?;
        let graph = prepared.graph;
        let plan = prepared;
        let outputs = &plan.outputs;
        let node_count = plan.nodes.len();
        let mut resources = execution_resource_slots(node_count);
        let mut remaining_uses = TensorExecutionUseCounts::from_slice(&plan.use_counts);

        for (index, node) in plan.nodes.iter().enumerate() {
            if plan.suppressed_adds.contains(&node.value) {
                continue;
            }
            let timing_mark = timings.begin();
            match node.op {
                OpDescriptor::Input => {
                    if let Some(output) = output_bank.and_then(|bank| bank.output(node.value)) {
                        if let Some(batch) = batch.as_deref_mut() {
                            flush_hip_batch(batch)?;
                        }
                        let mut destination = output.resource.clone_for_tensor_input();
                        if let Some((_, input)) =
                            resource_inputs.iter().find(|(id, _)| *id == node.value)
                        {
                            let bytes = byte_len(&output.shape)?;
                            memory.copy_resource(
                                &mut destination,
                                &input.resource,
                                bytes as u64,
                            )?;
                        } else {
                            let (_, tensor) = host_inputs
                                .iter()
                                .find(|(id, _)| *id == node.value)
                                .ok_or(TensorError::MissingInput(node.value))?;
                            transfer_tensor(memory, &mut destination, tensor)?;
                        }
                        resources[index] = Some(destination);
                        timings.finish(timing_mark, node);
                        continue;
                    }
                    if let Some((_, input)) =
                        resource_inputs.iter().find(|(id, _)| *id == node.value)
                    {
                        resources[index] = Some(input.resource_ref());
                        timings.finish(timing_mark, node);
                        continue;
                    }
                    let (_, tensor) = host_inputs
                        .iter()
                        .find(|(id, _)| *id == node.value)
                        .ok_or(TensorError::MissingInput(node.value))?;
                    if let Some(batch) = batch.as_deref_mut() {
                        flush_hip_batch(batch)?;
                    }
                    resources[index] = Some(upload_tensor(memory, pool, tensor)?);
                }
                OpDescriptor::Constant(tensor) => {
                    resources[index] = Some(
                        if let Some(output) = output_bank.and_then(|bank| bank.output(node.value)) {
                            if let Some(batch) = batch.as_deref_mut() {
                                flush_hip_batch(batch)?;
                            }
                            let mut resource = output.resource_ref();
                            transfer_tensor(memory, &mut resource, tensor)?;
                            resource
                        } else if outputs.contains(&node.value) {
                            if let Some(batch) = batch.as_deref_mut() {
                                flush_hip_batch(batch)?;
                            }
                            upload_tensor(memory, pool, tensor)?
                        } else if let Some(scratch) = scratch.as_deref_mut() {
                            scratch.lease(index)?
                        } else {
                            if let Some(batch) = batch.as_deref_mut() {
                                flush_hip_batch(batch)?;
                            }
                            upload_tensor(memory, pool, tensor)?
                        },
                    );
                }
                OpDescriptor::Uniform { value } => {
                    // Only elementwise consumers use scalar binding indices; every other route
                    // receives a dense fallback allocation during scratch preparation.
                    let tensor = prepared
                        .physical_layout(node.value)?
                        .uniform_tensor(node.shape, value)?;
                    resources[index] = Some(
                        if let Some(output) = output_bank.and_then(|bank| bank.output(node.value)) {
                            if let Some(batch) = batch.as_deref_mut() {
                                flush_hip_batch(batch)?;
                            }
                            let mut resource = output.resource_ref();
                            transfer_tensor(memory, &mut resource, &tensor)?;
                            resource
                        } else if outputs.contains(&node.value) {
                            if let Some(batch) = batch.as_deref_mut() {
                                flush_hip_batch(batch)?;
                            }
                            upload_tensor(memory, pool, &tensor)?
                        } else if let Some(scratch) = scratch.as_deref_mut() {
                            scratch.lease(index)?
                        } else {
                            if let Some(batch) = batch.as_deref_mut() {
                                flush_hip_batch(batch)?;
                            }
                            upload_tensor(memory, pool, &tensor)?
                        },
                    );
                }
                OpDescriptor::MatMul {
                    left,
                    right,
                    transpose_left,
                    transpose_right,
                } => {
                    if let Some(batch) = batch.as_deref_mut() {
                        flush_hip_batch(batch)?;
                    }
                    let left_index = plan
                        .index_of(left)
                        .ok_or(RocmTensorExecutionError::InvalidPlan(left))?;
                    let right_index = plan
                        .index_of(right)
                        .ok_or(RocmTensorExecutionError::InvalidPlan(right))?;
                    let left_shape: [usize; 2] = graph
                        .shape(left)?
                        .try_into()
                        .map_err(|_| RocmTensorExecutionError::InvalidPlan(left))?;
                    let right_shape: [usize; 2] = graph
                        .shape(right)?
                        .try_into()
                        .map_err(|_| RocmTensorExecutionError::InvalidPlan(right))?;
                    let result = execution_resource(
                        memory,
                        pool,
                        node.shape,
                        index,
                        node.value,
                        &mut scratch,
                        output_bank,
                    )?;
                    self.execute_matmul_row_major_flags(
                        resources[left_index]
                            .as_ref()
                            .ok_or(RocmTensorExecutionError::MissingResource(left))?,
                        resources[right_index]
                            .as_ref()
                            .ok_or(RocmTensorExecutionError::MissingResource(right))?,
                        &result,
                        left_shape,
                        right_shape,
                        transpose_left,
                        transpose_right,
                    )?;
                    resources[index] = Some(result);
                    release_after_read(&mut resources, &mut remaining_uses, left_index)?;
                    release_after_read(&mut resources, &mut remaining_uses, right_index)?;
                }
                OpDescriptor::Add { left, right }
                | OpDescriptor::Sub { left, right }
                | OpDescriptor::Mul { left, right } => {
                    if let Some(group) = plan.bounded_mul_by_output.get(&node.value) {
                        let output = execution_resource(
                            memory,
                            pool,
                            node.shape,
                            index,
                            node.value,
                            &mut scratch,
                            output_bank,
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
                            index,
                            node.value,
                            &mut scratch,
                            output_bank,
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
                        let kind = match node.op {
                            OpDescriptor::Add { .. } => TensorDispatchKind::Add,
                            OpDescriptor::Sub { .. } => TensorDispatchKind::Sub,
                            OpDescriptor::Mul { .. } => TensorDispatchKind::Mul,
                            _ => unreachable!("binary branch selects only Add, Sub, or Mul"),
                        };
                        let left_index = plan
                            .index_of(left)
                            .ok_or(RocmTensorExecutionError::InvalidPlan(left))?;
                        let right_index = plan
                            .index_of(right)
                            .ok_or(RocmTensorExecutionError::InvalidPlan(right))?;
                        let left_is_scalar =
                            matches!(plan.nodes[left_index].op, OpDescriptor::Uniform { .. })
                                && plan.physical_layout(left)?.representation
                                    == RocmPhysicalRepresentation::UniformScalar;
                        let right_is_scalar =
                            matches!(plan.nodes[right_index].op, OpDescriptor::Uniform { .. })
                                && plan.physical_layout(right)?.representation
                                    == RocmPhysicalRepresentation::UniformScalar;
                        let output = execution_resource(
                            memory,
                            pool,
                            node.shape,
                            index,
                            node.value,
                            &mut scratch,
                            output_bank,
                        )?;
                        self.execute_elementwise(
                            kind,
                            ElementwiseOperands {
                                shape: graph.shape(node.value)?,
                                left: resources[left_index]
                                    .as_ref()
                                    .ok_or(RocmTensorExecutionError::MissingResource(left))?,
                                right: Some(
                                    resources[right_index]
                                        .as_ref()
                                        .ok_or(RocmTensorExecutionError::MissingResource(right))?,
                                ),
                                output: &output,
                                scalar_mask: u8::from(left_is_scalar)
                                    | (u8::from(right_is_scalar) << 1),
                            },
                            batch.as_deref_mut(),
                        )?;
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
                        index,
                        node.value,
                        &mut scratch,
                        output_bank,
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
                        let left_is_scalar = plan.physical_layout(left)?.representation
                            == RocmPhysicalRepresentation::UniformScalar;
                        let right_is_scalar = plan.physical_layout(right)?.representation
                            == RocmPhysicalRepresentation::UniformScalar;
                        self.execute_elementwise(
                            TensorDispatchKind::AddRelu,
                            ElementwiseOperands {
                                shape: node.shape,
                                left: resources[left_index]
                                    .as_ref()
                                    .ok_or(RocmTensorExecutionError::MissingResource(left))?,
                                right: Some(
                                    resources[right_index]
                                        .as_ref()
                                        .ok_or(RocmTensorExecutionError::MissingResource(right))?,
                                ),
                                output: &output,
                                scalar_mask: u8::from(left_is_scalar)
                                    | (u8::from(right_is_scalar) << 1),
                            },
                            batch.as_deref_mut(),
                        )?;
                        resources[index] = Some(output);
                        release_after_read(&mut resources, &mut remaining_uses, left_index)?;
                        release_after_read(&mut resources, &mut remaining_uses, right_index)?;
                    } else {
                        let input_index = plan
                            .index_of(input)
                            .ok_or(RocmTensorExecutionError::InvalidPlan(input))?;
                        self.execute_relu(
                            node.shape,
                            resources[input_index]
                                .as_ref()
                                .ok_or(RocmTensorExecutionError::MissingResource(input))?,
                            &output,
                            batch.as_deref_mut(),
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
                        index,
                        node.value,
                        &mut scratch,
                        output_bank,
                    )?;
                    self.execute_relu_backward(
                        node.shape,
                        resources[input_index]
                            .as_ref()
                            .ok_or(RocmTensorExecutionError::MissingResource(input))?,
                        resources[upstream_index]
                            .as_ref()
                            .ok_or(RocmTensorExecutionError::MissingResource(upstream))?,
                        &output,
                        batch.as_deref_mut(),
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
                        index,
                        node.value,
                        &mut scratch,
                        output_bank,
                    )?;
                    self.execute_sgd_update(
                        node.shape,
                        resources[weights_index]
                            .as_ref()
                            .ok_or(RocmTensorExecutionError::MissingResource(weights))?,
                        resources[gradient_index]
                            .as_ref()
                            .ok_or(RocmTensorExecutionError::MissingResource(gradient))?,
                        SgdUpdateMode {
                            learning_rate,
                            contracted: plan
                                .lowering_plan
                                .rewrites()
                                .iter()
                                .any(|rewrite| rewrite.output == node.value),
                        },
                        &output,
                        batch.as_deref_mut(),
                    )?;
                    resources[index] = Some(output);
                    release_after_read(&mut resources, &mut remaining_uses, weights_index)?;
                    release_after_read(&mut resources, &mut remaining_uses, gradient_index)?;
                }
                OpDescriptor::MeanSquaredError { prediction, target } => {
                    if let Some(batch) = batch.as_deref_mut() {
                        flush_hip_batch(batch)?;
                    }
                    let prediction_index = plan
                        .index_of(prediction)
                        .ok_or(RocmTensorExecutionError::InvalidPlan(prediction))?;
                    let target_index = plan
                        .index_of(target)
                        .ok_or(RocmTensorExecutionError::InvalidPlan(target))?;
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
                        if !mse_scratch_length_fits(count, squared.device_buffer().len())? {
                            return Err(RocmTensorExecutionError::ScratchMismatch);
                        }
                        squared
                    } else {
                        allocate_tensor(memory, pool, shape)?
                    };
                    self.execute_elementwise(
                        TensorDispatchKind::SquaredDifference,
                        ElementwiseOperands {
                            shape,
                            left: resources[prediction_index]
                                .as_ref()
                                .ok_or(RocmTensorExecutionError::MissingResource(prediction))?,
                            right: Some(
                                resources[target_index]
                                    .as_ref()
                                    .ok_or(RocmTensorExecutionError::MissingResource(target))?,
                            ),
                            output: &squared,
                            scalar_mask: 0,
                        },
                        batch.as_deref_mut(),
                    )?;
                    if let Some(batch) = batch.as_deref_mut() {
                        flush_hip_batch(batch)?;
                    }
                    let output = execution_resource(
                        memory,
                        pool,
                        node.shape,
                        index,
                        node.value,
                        &mut scratch,
                        output_bank,
                    )?;
                    self.rocblas
                        .sasum_scaled(
                            count,
                            squared.device_buffer(),
                            1,
                            mse_scale(count),
                            output.device_buffer(),
                        )
                        .map_err(RocmTensorError::from)?;
                    resources[index] = Some(output);
                    release_after_read(&mut resources, &mut remaining_uses, prediction_index)?;
                    release_after_read(&mut resources, &mut remaining_uses, target_index)?;
                }
            }
            if plan.bounded_pointwise_by_output.contains_key(&node.value) {
                timings.finish_fused_add_sub(
                    timing_mark,
                    node,
                    plan.bounded_pointwise_by_output[&node.value].epilogue,
                );
            } else if plan.fused_add_by_relu.contains_key(&node.value) {
                timings.finish_fused_add_relu(timing_mark, node);
            } else {
                timings.finish(timing_mark, node);
            }
        }

        if let Some(batch) = batch.take() {
            flush_hip_batch(batch)?;
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
                        shape: graph.shape(output)?.to_vec(),
                        resource,
                    })
                })
                .collect()
        }
    }

    fn execute_relu(
        &self,
        shape: &[usize],
        input: &RocmMemoryResource,
        output: &RocmMemoryResource,
        batch: Option<&mut HipCompletionBatch>,
    ) -> Result<(), RocmTensorExecutionError> {
        self.execute_elementwise(
            TensorDispatchKind::Relu,
            ElementwiseOperands {
                shape,
                left: input,
                right: None,
                output,
                scalar_mask: 0,
            },
            batch,
        )
    }

    fn execute_relu_backward(
        &self,
        shape: &[usize],
        input: &RocmMemoryResource,
        upstream: &RocmMemoryResource,
        output: &RocmMemoryResource,
        batch: Option<&mut HipCompletionBatch>,
    ) -> Result<(), RocmTensorExecutionError> {
        let count = shape
            .iter()
            .try_fold(1usize, |count, dimension| count.checked_mul(*dimension))
            .ok_or(RocmTensorExecutionError::SizeOverflow)?;
        let count = u32::try_from(count)
            .ok()
            .filter(|count| *count > 0)
            .ok_or(RocmTensorExecutionError::SizeOverflow)?;
        if self.relu_backward.borrow().is_none() {
            let runtime = self.session.tensor_runtime();
            let image = self
                .session
                .compile_tensor_source(RELU_BACKWARD_SOURCE)
                .map_err(RocmTensorExecutionError::Backend)?;
            let module = runtime
                .load_module(&image)
                .map_err(RocmTensorExecutionError::Completion)?;
            let kernel = module
                .function(c"tensor_relu_backward")
                .map_err(RocmTensorExecutionError::Completion)?;
            let stream = self.stream.clone();
            *self.relu_backward.borrow_mut() = Some((kernel, stream));
        }
        let cache = self.relu_backward.borrow();
        let (kernel, stream) = cache
            .as_ref()
            .ok_or(RocmTensorExecutionError::SizeOverflow)?;
        let count_bytes = count.to_ne_bytes();
        let args = [
            HipKernelArgument::Buffer(input.device_buffer()),
            HipKernelArgument::Buffer(upstream.device_buffer()),
            HipKernelArgument::Buffer(output.device_buffer()),
            HipKernelArgument::Bytes(&count_bytes),
        ];
        // SAFETY: The compiled kernel takes three f32 buffers and a u32 count. Assessed shapes
        // and allocation sizes cover all guarded indices; the completion retains resources.
        #[allow(unsafe_code)]
        let mut completion =
            unsafe { kernel.launch(stream, [count.div_ceil(256), 1, 1], [256, 1, 1], 0, &args) }
                .map_err(RocmTensorExecutionError::Completion)?;
        if let Some(batch) = batch {
            batch
                .push(completion)
                .map_err(RocmTensorExecutionError::Completion)
        } else {
            completion
                .wait()
                .map_err(RocmTensorExecutionError::Completion)
        }
    }

    fn execute_sgd_update(
        &self,
        shape: &[usize],
        weights: &RocmMemoryResource,
        gradient: &RocmMemoryResource,
        mode: SgdUpdateMode,
        output: &RocmMemoryResource,
        batch: Option<&mut HipCompletionBatch>,
    ) -> Result<(), RocmTensorExecutionError> {
        let count = shape
            .iter()
            .try_fold(1usize, |count, dimension| count.checked_mul(*dimension))
            .ok_or(RocmTensorExecutionError::SizeOverflow)?;
        let count = u32::try_from(count)
            .ok()
            .filter(|count| *count > 0)
            .ok_or(RocmTensorExecutionError::SizeOverflow)?;
        let cache = if mode.contracted {
            &self.sgd_update_contracted
        } else {
            &self.sgd_update
        };
        if cache.borrow().is_none() {
            let runtime = self.session.tensor_runtime();
            let image = self
                .session
                .compile_tensor_source(if mode.contracted {
                    SGD_UPDATE_CONTRACTED_SOURCE
                } else {
                    SGD_UPDATE_SOURCE
                })
                .map_err(RocmTensorExecutionError::Backend)?;
            let module = runtime
                .load_module(&image)
                .map_err(RocmTensorExecutionError::Completion)?;
            let kernel = module
                .function(if mode.contracted {
                    c"tensor_sgd_update_contracted"
                } else {
                    c"tensor_sgd_update"
                })
                .map_err(RocmTensorExecutionError::Completion)?;
            let stream = self.stream.clone();
            *cache.borrow_mut() = Some((kernel, stream));
        }
        let cached = cache.borrow();
        let (kernel, stream) = cached
            .as_ref()
            .ok_or(RocmTensorExecutionError::SizeOverflow)?;
        let learning_rate_bytes = mode.learning_rate.to_ne_bytes();
        let count_bytes = count.to_ne_bytes();
        let args = [
            HipKernelArgument::Buffer(weights.device_buffer()),
            HipKernelArgument::Buffer(gradient.device_buffer()),
            HipKernelArgument::Buffer(output.device_buffer()),
            HipKernelArgument::Bytes(&learning_rate_bytes),
            HipKernelArgument::Bytes(&count_bytes),
        ];
        // SAFETY: assessed equal nonempty f32 shapes fit u32, and execution resources cover all
        // indices. The completion retains the buffers until the one launch has finished.
        #[allow(unsafe_code)]
        let mut completion =
            unsafe { kernel.launch(stream, [count.div_ceil(256), 1, 1], [256, 1, 1], 0, &args) }
                .map_err(RocmTensorExecutionError::Completion)?;
        if let Some(batch) = batch {
            batch
                .push(completion)
                .map_err(RocmTensorExecutionError::Completion)
        } else {
            completion
                .wait()
                .map_err(RocmTensorExecutionError::Completion)
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
            let cache = self.add_dispatches.borrow();
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
            let cache = self.add_dispatches.borrow();
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
    fn execute_elementwise(
        &self,
        kind: TensorDispatchKind,
        operands: ElementwiseOperands<'_>,
        batch: Option<&mut HipCompletionBatch>,
    ) -> Result<(), RocmTensorExecutionError> {
        let ElementwiseOperands {
            shape,
            left,
            right,
            output,
            scalar_mask,
        } = operands;
        let count = shape
            .iter()
            .try_fold(1usize, |count, dimension| count.checked_mul(*dimension))
            .ok_or(RocmTensorExecutionError::SizeOverflow)?;
        let invocations = u32::try_from(count)
            .ok()
            .and_then(NonZeroU32::new)
            .ok_or(RocmTensorExecutionError::SizeOverflow)?;
        let logical_count = invocations.get();
        let cache_key = TensorDispatchCacheKey::Fixed(kind, logical_count, scalar_mask);
        self.ensure_dispatch_cached(
            cache_key.clone(),
            kind.kernel(logical_count, scalar_mask),
            PcuInvocationShape::invocations(invocations),
            false,
        )?;
        let left_binding = self
            .session
            .bind(
                match kind {
                    TensorDispatchKind::Add
                    | TensorDispatchKind::AddRelu
                    | TensorDispatchKind::Sub
                    | TensorDispatchKind::Mul => ADD_LEFT_REF,
                    TensorDispatchKind::Relu => RELU_INPUT_REF,
                    TensorDispatchKind::SquaredDifference => MSE_LEFT_REF,
                },
                PcuBindingAccess::ReadOnly,
                PcuBindingType::Value(PcuValueType::f32()),
                left,
            )
            .map_err(RocmTensorExecutionError::Backend)?;
        let mut bindings = SmallVec::<[_; 3]>::new();
        bindings.push(left_binding);
        if let Some(right) = right {
            let right_binding = self
                .session
                .bind(
                    match kind {
                        TensorDispatchKind::Add
                        | TensorDispatchKind::AddRelu
                        | TensorDispatchKind::Sub
                        | TensorDispatchKind::Mul => ADD_RIGHT_REF,
                        TensorDispatchKind::Relu => {
                            return Err(RocmTensorExecutionError::SizeOverflow);
                        }
                        TensorDispatchKind::SquaredDifference => MSE_RIGHT_REF,
                    },
                    PcuBindingAccess::ReadOnly,
                    PcuBindingType::Value(PcuValueType::f32()),
                    right,
                )
                .map_err(RocmTensorExecutionError::Backend)?;
            bindings.push(right_binding);
        }
        let output_binding = self
            .session
            .bind(
                match kind {
                    TensorDispatchKind::Add
                    | TensorDispatchKind::AddRelu
                    | TensorDispatchKind::Sub
                    | TensorDispatchKind::Mul => ADD_OUTPUT_REF,
                    TensorDispatchKind::Relu => RELU_OUTPUT_REF,
                    TensorDispatchKind::SquaredDifference => MSE_OUTPUT_REF,
                },
                PcuBindingAccess::WriteOnly,
                PcuBindingType::Value(PcuValueType::f32()),
                output,
            )
            .map_err(RocmTensorExecutionError::Backend)?;
        bindings.push(output_binding);
        let dispatch_result = {
            let cache = self.add_dispatches.borrow();
            let prepared = cache
                .iter()
                .find(|(key, _)| *key == cache_key)
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
        if let Some(mut completion) = dispatch_result {
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
        self.execute_matmul_row_major_flags(a, b, c, [rows, inner], [inner, columns], false, false)
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
        let a_bytes = a_shape[0]
            .checked_mul(a_shape[1])
            .and_then(|n| n.checked_mul(4));
        let b_bytes = b_shape[0]
            .checked_mul(b_shape[1])
            .and_then(|n| n.checked_mul(4));
        let c_bytes = rows.checked_mul(columns).and_then(|n| n.checked_mul(4));
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
        self.rocblas.sgemm(
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
        )?;
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

fn mse_scratch_length_fits(
    count: usize,
    squared_bytes: usize,
) -> Result<bool, RocmTensorExecutionError> {
    let required_bytes = count
        .checked_mul(size_of::<f32>())
        .ok_or(RocmTensorExecutionError::SizeOverflow)?;
    Ok(squared_bytes >= required_bytes)
}

/// A structurally validated `ROCm` execution plan for one graph output.
///
/// The graph is borrowed and must remain alive and unchanged while the plan is used.
pub struct RocmPreparedTensorGraph<'graph> {
    graph: &'graph Graph,
    plan: TensorExecutionPlan<'graph>,
    lowering_plan: TensorSelectedLoweringPlan<'graph>,
    output: ValueId,
    outputs: Vec<ValueId>,
    nodes: Vec<NodeDescriptor<'graph>>,
    index_by_value: HashMap<ValueId, usize>,
    use_counts: Vec<usize>,
    fused_add_by_relu: HashMap<ValueId, (ValueId, ValueId)>,
    bounded_pointwise_by_output: HashMap<ValueId, TensorBoundedPointwiseFusionGroup>,
    bounded_mul_by_output: HashMap<ValueId, TensorBoundedMulFusionGroup>,
    suppressed_adds: HashSet<ValueId>,
    storage_constraints: Vec<TensorStorageConstraint>,
    physical_layouts: HashMap<ValueId, RocmPhysicalLayout>,
}

/// Synchronous wall time spent executing one prepared graph node in diagnostic mode.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RocmTensorNodeTiming {
    /// Graph value produced by this node.
    pub value: ValueId,
    /// Operation kind, such as `MatMul`, `ReluBackward`, or `Input`.
    pub operation: &'static str,
    /// Elapsed wall time, including synchronous device completion.
    pub elapsed: Duration,
}

trait NodeTimingSink {
    type Mark;

    fn begin(&mut self) -> Self::Mark;
    fn finish(&mut self, mark: Self::Mark, node: &NodeDescriptor<'_>);

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

    #[inline(always)]
    fn begin(&mut self) {}

    #[inline(always)]
    fn finish(&mut self, (): Self::Mark, _: &NodeDescriptor<'_>) {}
}

#[derive(Default)]
struct CollectNodeTimings(Vec<RocmTensorNodeTiming>);

#[derive(Clone, Copy)]
struct SgdUpdateMode {
    learning_rate: f32,
    contracted: bool,
}

impl NodeTimingSink for CollectNodeTimings {
    type Mark = Instant;

    fn begin(&mut self) -> Self::Mark {
        Instant::now()
    }

    fn finish(&mut self, mark: Self::Mark, node: &NodeDescriptor<'_>) {
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
                OpDescriptor::Relu { .. } => "Relu",
                OpDescriptor::ReluBackward { .. } => "ReluBackward",
                OpDescriptor::SgdUpdate { .. } => "SgdUpdate",
                OpDescriptor::MeanSquaredError { .. } => "MeanSquaredError",
            },
            elapsed: mark.elapsed(),
        });
    }

    fn finish_fused_add_relu(&mut self, mark: Self::Mark, node: &NodeDescriptor<'_>) {
        self.0.push(RocmTensorNodeTiming {
            value: node.value,
            operation: "FusedAddRelu",
            elapsed: mark.elapsed(),
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

/// Caller-owned output allocations reusable for repeated execution of one exact prepared plan.
pub struct RocmTensorOutputBank<'plan, 'graph, 'session> {
    prepared: &'plan RocmPreparedTensorGraph<'graph>,
    session: &'session RocmOwnedDispatchBackend,
    pool: PcuMemoryPoolId,
    outputs: Vec<RocmTensorInput<'session>>,
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

fn validate_output_bank(
    bank: &RocmTensorOutputBank<'_, '_, '_>,
    prepared: &RocmPreparedTensorGraph<'_>,
    session: &RocmOwnedDispatchBackend,
    pool: PcuMemoryPoolId,
    inputs: &[(ValueId, &RocmTensorInput<'_>)],
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
    let mut requirements = Vec::with_capacity(bank.outputs.len());
    for (&value, output) in prepared.outputs.iter().zip(&bank.outputs) {
        let expected_shape = prepared.graph.shape(value)?;
        if !std::ptr::eq(output.session, session)
            || output.shape != expected_shape
            || output.resource.pool() != pool
            || !output.resource.belongs_to_runtime(session.tensor_runtime())
            || output.resource.device_buffer().len() < byte_len(expected_shape)?
            || !matches!(
                output.resource.access(),
                PcuMemoryAccess::WriteOnly | PcuMemoryAccess::ReadWrite
            )
        {
            return Err(RocmTensorExecutionError::OutputResourceMismatch);
        }
        requirements.push(PcuMemoryMemberRequirement {
            pool,
            minimum_size_bytes: byte_len(expected_shape)? as u64,
            access: PcuMemoryAccess::ReadWrite,
            require_device_local: false,
        });
        if inputs
            .iter()
            .any(|(_, input)| input.resource.may_overlap(&output.resource))
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
    validate_reusable_memory_bank_members_by(&bank.outputs, &requirements, |output| {
        &output.resource
    })
    .map_err(|_| RocmTensorExecutionError::OutputResourceMismatch)?;
    Ok(())
}

fn validate_prepared_storage_constraints(
    prepared: &RocmPreparedTensorGraph<'_>,
    inputs: &[(ValueId, &RocmTensorInput<'_>)],
    scratch: &RocmTensorScratch<'_, '_, '_>,
    bank: &RocmTensorOutputBank<'_, '_, '_>,
) -> Result<(), RocmTensorExecutionError> {
    let resource_for = |value| {
        bank.output(value)
            .map(|output| &output.resource)
            .or_else(|| {
                inputs
                    .iter()
                    .find(|(id, _)| *id == value)
                    .map(|(_, input)| &input.resource)
            })
            .or_else(|| {
                prepared
                    .index_of(value)
                    .and_then(|index| scratch.resources.get(index))
                    .and_then(Option::as_ref)
            })
    };
    for constraint in &prepared.storage_constraints {
        let left =
            resource_for(constraint.left).ok_or(RocmTensorExecutionError::StorageConstraint(
                TensorStorageValidationError::MissingResource(constraint.left),
            ))?;
        let right =
            resource_for(constraint.right).ok_or(RocmTensorExecutionError::StorageConstraint(
                TensorStorageValidationError::MissingResource(constraint.right),
            ))?;
        constraint
            .validate_resources(left, right)
            .map_err(RocmTensorExecutionError::StorageConstraint)?;
    }
    Ok(())
}

impl RocmTensorScratch<'_, '_, '_> {
    fn lease(&self, index: usize) -> Result<RocmMemoryResource, RocmTensorExecutionError> {
        self.resources
            .get(index)
            .and_then(Option::as_ref)
            .map(RocmMemoryResource::clone_for_tensor_input)
            .ok_or_else(|| {
                RocmTensorExecutionError::MissingResource(
                    self.prepared
                        .nodes
                        .get(index)
                        .map_or(self.prepared.output, |node| node.value),
                )
            })
    }
}

fn execution_resource<P: PcuMemoryProvider<Resource = RocmMemoryResource>>(
    memory: &mut P,
    pool: PcuMemoryPoolId,
    shape: &[usize],
    index: usize,
    value: ValueId,
    scratch: &mut Option<&mut RocmTensorScratch<'_, '_, '_>>,
    output_bank: Option<&RocmTensorOutputBank<'_, '_, '_>>,
) -> Result<RocmMemoryResource, RocmTensorExecutionError> {
    if let Some(bank) = output_bank
        && let Some(output) = bank.output(value)
    {
        return Ok(output.resource_ref());
    }
    if let Some(scratch) = scratch.as_deref_mut()
        && !scratch.prepared.outputs.contains(&value)
    {
        return scratch.lease(index);
    }
    allocate_tensor(memory, pool, shape)
}

fn scratch_stores_node(op: OpDescriptor<'_>, value: ValueId, outputs: &[ValueId]) -> bool {
    !outputs.contains(&value) && !matches!(op, OpDescriptor::Input)
}

fn mse_scratch_element_count(
    graph: &Graph,
    nodes: &[NodeDescriptor<'_>],
) -> Result<usize, RocmTensorExecutionError> {
    nodes.iter().try_fold(0usize, |max_count, node| {
        let OpDescriptor::MeanSquaredError { prediction, .. } = node.op else {
            return Ok(max_count);
        };
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

fn collect_dispatch_requests<'graph>(
    prepared: &'graph RocmPreparedTensorGraph<'_>,
) -> Result<Vec<TensorDispatchRequest<'graph>>, RocmTensorExecutionError> {
    let mut requests = Vec::new();
    for node in &prepared.nodes {
        if prepared.suppressed_adds.contains(&node.value) {
            continue;
        }
        let request = match node.op {
            OpDescriptor::Add { left, right } => {
                if let Some(group) = prepared
                    .bounded_pointwise_by_output
                    .get(&node.value)
                    .filter(|group| group.epilogue == TensorPointwiseEpilogue::Identity)
                {
                    Some(bounded_pointwise_request(prepared, group)?)
                } else {
                    Some(TensorDispatchRequest::Fixed {
                        kind: TensorDispatchKind::Add,
                        logical_count: flattened_invocation_count(node.shape)?,
                        scalar_mask: scalar_mask_for_operands(prepared, left, right)?,
                    })
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
                    Some(TensorDispatchRequest::Fixed {
                        kind: TensorDispatchKind::Sub,
                        logical_count: flattened_invocation_count(node.shape)?,
                        scalar_mask: scalar_mask_for_operands(prepared, left, right)?,
                    })
                }
            }
            OpDescriptor::Mul { left, right } => {
                if let Some(group) = prepared.bounded_mul_by_output.get(&node.value) {
                    Some(bounded_mul_request(prepared, group)?)
                } else {
                    Some(TensorDispatchRequest::Fixed {
                        kind: TensorDispatchKind::Mul,
                        logical_count: flattened_invocation_count(node.shape)?,
                        scalar_mask: scalar_mask_for_operands(prepared, left, right)?,
                    })
                }
            }
            OpDescriptor::Relu { .. } => {
                if let Some(group) = prepared.bounded_pointwise_by_output.get(&node.value) {
                    Some(bounded_pointwise_request(prepared, group)?)
                } else if let Some(&(left, right)) = prepared.fused_add_by_relu.get(&node.value) {
                    Some(TensorDispatchRequest::Fixed {
                        kind: TensorDispatchKind::AddRelu,
                        logical_count: flattened_invocation_count(node.shape)?,
                        scalar_mask: scalar_mask_for_operands(prepared, left, right)?,
                    })
                } else {
                    Some(TensorDispatchRequest::Fixed {
                        kind: TensorDispatchKind::Relu,
                        logical_count: flattened_invocation_count(node.shape)?,
                        scalar_mask: 0,
                    })
                }
            }
            OpDescriptor::MeanSquaredError { prediction, .. } => {
                Some(TensorDispatchRequest::Fixed {
                    kind: TensorDispatchKind::SquaredDifference,
                    logical_count: flattened_invocation_count(prepared.graph.shape(prediction)?)?,
                    scalar_mask: 0,
                })
            }
            OpDescriptor::Input
            | OpDescriptor::Constant(_)
            | OpDescriptor::Uniform { .. }
            | OpDescriptor::MatMul { .. }
            | OpDescriptor::ReluBackward { .. }
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
    suppressed_adds: HashSet<ValueId>,
    storage_constraints: Vec<TensorStorageConstraint>,
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
            OpDescriptor::Input
            | OpDescriptor::Constant(_)
            | OpDescriptor::Uniform { .. }
            | OpDescriptor::ReluBackward { .. }
            | OpDescriptor::SgdUpdate { .. } => TensorExecutionRoute::Native,
            OpDescriptor::MatMul { .. } => TensorExecutionRoute::Library,
            OpDescriptor::Add { .. }
            | OpDescriptor::Sub { .. }
            | OpDescriptor::Mul { .. }
            | OpDescriptor::Relu { .. }
            | OpDescriptor::MeanSquaredError { .. } => TensorExecutionRoute::Synthesized,
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
        let _ = byte_len(node.shape)?;
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
    let storage_constraints = physicalize_storage_constraints(
        lowering_plan.operation_storage_constraints()?,
        &physical_layouts,
    )?;
    Ok(GraphExecutionPreflight {
        tensor_plan: plan,
        lowering_plan,
        nodes,
        index_by_value,
        use_counts,
        fused_add_by_relu,
        bounded_pointwise_by_output,
        bounded_mul_by_output,
        suppressed_adds,
        storage_constraints,
        physical_layouts,
    })
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
    step: &fusion_pcu_tensor::TensorPointwiseStep,
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
                let bytes = u64::try_from(byte_len(node.shape)?)
                    .map_err(|_| RocmTensorExecutionError::SizeOverflow)?;
                RocmPhysicalLayout::dense(bytes)
            };
            Ok((node.value, layout))
        })
        .collect()
}

#[cfg(test)]
fn validate_graph_inputs(
    nodes: &[NodeDescriptor<'_>],
    inputs: &[(ValueId, Tensor)],
) -> Result<(), TensorError> {
    validate_graph_inputs_inner(nodes, inputs, true)
}

fn validate_graph_inputs_inner(
    nodes: &[NodeDescriptor<'_>],
    inputs: &[(ValueId, Tensor)],
    require_all: bool,
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
        if require_all
            && matches!(node.op, OpDescriptor::Input)
            && !inputs.iter().any(|(id, _)| *id == node.value)
        {
            return Err(TensorError::MissingInput(node.value));
        }
    }
    Ok(())
}

fn validate_graph_input_sources(
    nodes: &[NodeDescriptor<'_>],
    host_inputs: &[(ValueId, Tensor)],
    resource_inputs: &[(ValueId, &RocmTensorInput<'_>)],
    session: &RocmOwnedDispatchBackend,
    pool: PcuMemoryPoolId,
) -> Result<(), RocmTensorExecutionError> {
    validate_graph_inputs_inner(nodes, host_inputs, false)?;
    for (index, (id, input)) in resource_inputs.iter().enumerate() {
        if resource_inputs[..index]
            .iter()
            .any(|(previous, _)| previous == id)
            || host_inputs.iter().any(|(previous, _)| previous == id)
        {
            return Err(TensorError::DuplicateInput(*id).into());
        }
        let Some(node) = nodes.iter().find(|node| node.value == *id) else {
            return Err(TensorError::ExtraInput(*id).into());
        };
        if !matches!(node.op, OpDescriptor::Input) {
            return Err(TensorError::ExtraInput(*id).into());
        }
        if node.shape != input.shape {
            return Err(TensorError::ShapeMismatch {
                left: node.shape.to_vec(),
                right: input.shape.clone(),
            }
            .into());
        }
        input.validate_session(session)?;
        if input.resource.pool() != pool {
            return Err(RocmTensorExecutionError::InputPoolMismatch);
        }
        if !matches!(
            input.resource.access(),
            PcuMemoryAccess::ReadOnly | PcuMemoryAccess::ReadWrite
        ) {
            return Err(RocmTensorExecutionError::InputResourceMismatch);
        }
    }
    for node in nodes {
        if matches!(node.op, OpDescriptor::Input)
            && !host_inputs.iter().any(|(id, _)| *id == node.value)
            && !resource_inputs.iter().any(|(id, _)| *id == node.value)
        {
            return Err(TensorError::MissingInput(node.value).into());
        }
    }
    Ok(())
}

fn byte_len(shape: &[usize]) -> Result<usize, RocmTensorExecutionError> {
    let bytes = shape
        .iter()
        .try_fold(size_of::<f32>(), |bytes, dimension| {
            bytes
                .checked_mul(*dimension)
                .ok_or(RocmTensorExecutionError::SizeOverflow)
        })?;
    if bytes > isize::MAX as usize {
        return Err(RocmTensorExecutionError::SizeOverflow);
    }
    Ok(bytes)
}

#[allow(clippy::cast_precision_loss)] // Matches the tensor dialect's f32 mean divisor semantics.
fn mse_scale(element_count: usize) -> f32 {
    1.0 / element_count as f32
}

fn allocate_tensor<P: PcuMemoryProvider<Resource = RocmMemoryResource>>(
    memory: &mut P,
    pool: PcuMemoryPoolId,
    shape: &[usize],
) -> Result<RocmMemoryResource, RocmTensorExecutionError> {
    let size_bytes =
        u64::try_from(byte_len(shape)?).map_err(|_| RocmTensorExecutionError::SizeOverflow)?;
    Ok(memory.allocate(PcuMemoryAllocationRequest {
        pool,
        size_bytes,
        alignment_bytes: u64::try_from(align_of::<f32>())
            .map_err(|_| RocmTensorExecutionError::SizeOverflow)?,
        access: PcuMemoryAccess::ReadWrite,
        host_access: PcuMemoryHostAccess::TransferOnly,
        require_device_local: false,
    })?)
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
    let bytes_len = byte_len(&output.shape)?;
    let mut data = vec![0.0_f32; bytes_len / size_of::<f32>()];
    // SAFETY: `data` is initialized, contiguous f32 storage. Every f32 bit pattern is valid;
    // u8 has alignment one, and its byte length was checked from the output shape.
    let output_bytes =
        unsafe { std::slice::from_raw_parts_mut(data.as_mut_ptr().cast::<u8>(), bytes_len) };
    memory.transfer_from(&output.resource, 0, output_bytes)?;
    Ok(Tensor::new(output.shape.clone(), data)?)
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
    plan: &RocmPreparedTensorGraph<'_>,
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
    plan: &RocmPreparedTensorGraph<'_>,
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
        if matches!(
            node.op,
            OpDescriptor::MatMul { .. } | OpDescriptor::MeanSquaredError { .. }
        ) && !self.rocblas.is_usable()
        {
            return TensorOperationSupport::Unsupported {
                reason: TensorUnsupportedReason::Other(
                    "rocBLAS completion is uncertain; selected handle cannot run more work".into(),
                ),
            };
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

const fn rocm_supports_operand_representation(
    node: NodeDescriptor<'_>,
    representation: TensorOperandRepresentation,
) -> bool {
    match representation {
        TensorOperandRepresentation::Dense => true,
        TensorOperandRepresentation::UniformScalar => matches!(
            node.op,
            OpDescriptor::Add { .. } | OpDescriptor::Sub { .. } | OpDescriptor::Mul { .. }
        ),
    }
}

fn assess_tensor_node(graph: &Graph, node: NodeDescriptor<'_>) -> TensorOperationSupport {
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
        | OpDescriptor::Mul { left, right } => match (graph.shape(left), graph.shape(right)) {
            (Ok(left), Ok(right))
                if left == right
                    && left
                        .iter()
                        .try_fold(1usize, |count, dimension| count.checked_mul(*dimension))
                        .is_some_and(|count| count > 0 && u32::try_from(count).is_ok())
                    && node.shape == left =>
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
                            route: TensorExecutionRoute::Synthesized,
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
        .all(|(a, b)| a.checked_mul(b).and_then(|n| n.checked_mul(4)).is_some())
        && output_shape == [rows, columns]
}

#[cfg(test)]
mod tests {
    use fusion_pcu::{
        PcuDispatchEntryPoint,
        PcuDispatchFeatureCaps,
        PcuDispatchKernelIr,
        PcuDispatchOp,
        PcuDispatchDataOp,
        PcuDispatchAluOp,
        PcuValueTypeCaps,
    };
    use fusion_pcu_tensor::{
        Graph,
        NodeDescriptor,
        OpDescriptor,
        Tensor,
        TensorError,
        TensorOperationAssessor,
        TensorOperationSupport,
        TensorPointwiseEpilogue,
    };

    use super::{
        add_kernel,
        add_relu_kernel,
        assess_tensor_node,
        bounded_pointwise_kernel_id,
        bounded_pointwise_program,
        bounded_pointwise_topology,
        bounded_mul_program,
        bounded_mul_topology,
        collect_dispatch_requests,
        execution_resource_slots,
        retained_requested_key_count,
        mse_kernel,
        mse_scratch_length_fits,
        mse_scratch_element_count,
        output_position,
        prepare_graph,
        prepare_graph_outputs_plan,
        prepare_graph_outputs_plan_with_arithmetic,
        prepare_graph_outputs_plan_with_policies,
        relu_kernel,
        rocm_supports_operand_representation,
        scratch_stores_node,
        validate_graph_inputs,
        CollectNodeTimings,
        NodeTimingSink,
        RocmTensorExecutionError,
        RocmPhysicalLayout,
        TensorExecutionRoute,
        TensorExecutionUseCounts,
        TensorDispatchCacheKey,
        TensorDispatchKind,
        TensorDispatchRequest,
        GraphExecutionPreflight,
        RocmPreparedTensorGraph,
        TENSOR_EXECUTION_INLINE_NODES,
        TensorArithmeticCapability,
        TensorArithmeticRewritePolicy,
        TensorOperandRepresentation,
        TensorPointwiseGroupingPolicy,
        TensorUnsupportedReason,
        ValueId,
    };

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
    fn diagnostic_node_timings_preserve_execution_order_and_count() {
        let mut graph = Graph::default();
        let left = graph.input([2, 2]).unwrap();
        let right = graph.input([2, 2]).unwrap();
        let sum = graph.add(left, right).unwrap();
        let left_node = NodeDescriptor {
            value: left,
            op: graph.nodes().find(|node| node.value == left).unwrap().op,
            shape: graph.shape(left).unwrap(),
        };
        let sum_node = NodeDescriptor {
            value: sum,
            op: graph.nodes().find(|node| node.value == sum).unwrap().op,
            shape: graph.shape(sum).unwrap(),
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
    fn add_dispatch_kernel_lowers_with_valid_ssa_values() {
        let kernel = add_kernel(17, 0);

        let source = crate::lower_dispatch_to_hip_source(&kernel).unwrap();

        assert!(source.contains("fusion_kernel"));
        assert_eq!(kernel.entry.logical_shape, [17, 1, 1]);
    }

    #[test]
    fn add_relu_dispatch_kernel_lowers_add_before_max_for_dense_and_uniform_inputs() {
        let dense_kernel = add_relu_kernel(17, 0);
        let dense_source = crate::lower_dispatch_to_hip_source(&dense_kernel).unwrap();

        let add_position = dense_source.find("float v3 = v1 + v2;").unwrap();
        let max_position = dense_source.find("float v5 = fmaxf(v3, v4);").unwrap();
        assert!(add_position < max_position);
        assert_eq!(dense_kernel.entry.logical_shape, [17, 1, 1]);

        for scalar_mask in 1..=3 {
            let kernel = add_relu_kernel(17, scalar_mask);
            let source = crate::lower_dispatch_to_hip_source(&kernel).unwrap();

            assert!(source.contains("[0]"));
            assert!(source.contains("fmaxf(v3, v4)"));
            assert_eq!(kernel.entry.logical_shape, [17, 1, 1]);
        }
    }

    #[test]
    fn bounded_add_sub_relu_dispatch_preserves_order_and_scalar_leaf_binding() {
        let mut graph = Graph::default();
        let dense_leaf = graph.input([17]).unwrap();
        let scalar_leaf = graph.uniform([17], -0.125).unwrap();
        let sum = graph.add(dense_leaf, scalar_leaf).unwrap();
        let difference = graph.sub(dense_leaf, sum).unwrap();
        let output = graph.relu(difference).unwrap();
        let plan = graph.execution_plan_for_outputs(&[output]).unwrap();
        let selected = plan.select_lowering_with_grouping(
            fusion_pcu_tensor::TensorArithmeticRewritePolicy::Disabled,
            fusion_pcu_tensor::TensorArithmeticCapability::Strict,
            TensorPointwiseGroupingPolicy::BoundedAddSubRelu,
        );
        let group = selected
            .bounded_pointwise_fusion_groups()
            .first()
            .expect("the bounded chain is selected");
        assert_eq!(
            bounded_pointwise_topology(group).as_slice(),
            [1, 2, 2, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0]
        );

        let (bindings, ops) = bounded_pointwise_program(group, 0b010).unwrap();
        let kernel = PcuDispatchKernelIr {
            id: bounded_pointwise_kernel_id(&bounded_pointwise_topology(group), 0b010),
            entry: PcuDispatchEntryPoint {
                name: "tensor_bounded_add_sub_relu",
                logical_shape: [17, 1, 1],
            },
            bindings: &bindings,
            ports: &[],
            parameters: &[],
            ops: &ops,
            type_caps: PcuValueTypeCaps::FLOAT32.union(PcuValueTypeCaps::SCALAR_VALUES),
            feature_caps: PcuDispatchFeatureCaps::default(),
        };
        let source = crate::lower_dispatch_to_hip_source(&kernel).unwrap();

        let add_position = source.find("float v3 = v1 + v2;").unwrap();
        let sub_position = source.find("float v4 = v1 - v3;").unwrap();
        let relu_position = source.find("float v6 = fmaxf(v4, v5);").unwrap();
        assert!(add_position < sub_position && sub_position < relu_position);
        assert!(source.contains("float v2 = binding_0_1[0];"));
        assert_eq!(kernel.entry.logical_shape, [17, 1, 1]);
    }

    #[test]
    fn bounded_add_sub_identity_dispatch_has_no_relu_epilogue_and_is_preparable() {
        let mut graph = Graph::default();
        let left = graph.input([17]).unwrap();
        let right = graph.input([17]).unwrap();
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
        let group = selected
            .bounded_pointwise_fusion_groups()
            .first()
            .expect("the two-step Add/Sub chain is selected");
        assert_eq!(group.output, output);
        assert_eq!(group.epilogue, TensorPointwiseEpilogue::Identity);
        assert_eq!(group.steps.len(), 2);

        let topology = bounded_pointwise_topology(group);
        let (bindings, ops) = bounded_pointwise_program(group, 0).unwrap();
        let kernel = PcuDispatchKernelIr {
            id: bounded_pointwise_kernel_id(&topology, 0),
            entry: PcuDispatchEntryPoint {
                name: "tensor_bounded_add_sub",
                logical_shape: [17, 1, 1],
            },
            bindings: &bindings,
            ports: &[],
            parameters: &[],
            ops: &ops,
            type_caps: PcuValueTypeCaps::FLOAT32.union(PcuValueTypeCaps::SCALAR_VALUES),
            feature_caps: PcuDispatchFeatureCaps::default(),
        };
        assert!(!ops.iter().any(|op| matches!(
            op,
            PcuDispatchOp::Data(PcuDispatchDataOp::Alu {
                value_type,
                op: PcuDispatchAluOp::Max,
                ..
            }) if *value_type == fusion_pcu::PcuValueType::f32()
        )));
        let source = crate::lower_dispatch_to_hip_source(&kernel).unwrap();
        assert!(source.contains("float v4 = v3 - v1;"));
        assert!(!source.contains("fmaxf"));
        assert!(matches!(
            selected.operations().last(),
            Some(fusion_pcu_tensor::TensorSelectedOperation::FusedAddSub { group: selected })
                if selected.output == output
                    && selected.epilogue == TensorPointwiseEpilogue::Identity
        ));
    }

    #[test]
    fn bounded_mul_identity_is_a_distinct_ordered_dispatch_and_prewarm_request() {
        let mut graph = Graph::default();
        let input = graph.input([17]).unwrap();
        let factor = graph.uniform([17], 2.0).unwrap();
        let first = graph.mul(input, factor).unwrap();
        let output = graph.mul(first, input).unwrap();
        let prepared = prepared_for_request_test(
            &graph,
            &[output],
            TensorPointwiseGroupingPolicy::BoundedMulIdentity,
        );
        assert!(prepared.suppressed_adds.contains(&first));
        let requests = collect_dispatch_requests(&prepared).unwrap();
        assert_eq!(requests.len(), 1);
        let TensorDispatchRequest::BoundedMul {
            group,
            logical_count,
            scalar_mask,
            topology,
        } = &requests[0]
        else {
            panic!("expected bounded Mul cache request");
        };
        assert_eq!((*logical_count, *scalar_mask), (17, 2));
        assert_eq!(group.output, output);
        assert_eq!(group.steps.len(), 2);
        assert_eq!(topology, &bounded_mul_topology(group));

        let (bindings, ops) = bounded_mul_program(group, 0).unwrap();
        let kernel = PcuDispatchKernelIr {
            id: bounded_pointwise_kernel_id(topology, 0),
            entry: PcuDispatchEntryPoint {
                name: "tensor_bounded_mul",
                logical_shape: [17, 1, 1],
            },
            bindings: &bindings,
            ports: &[],
            parameters: &[],
            ops: &ops,
            type_caps: PcuValueTypeCaps::FLOAT32.union(PcuValueTypeCaps::SCALAR_VALUES),
            feature_caps: PcuDispatchFeatureCaps::default(),
        };
        assert_eq!(
            ops.iter()
                .filter(|op| matches!(
                    op,
                    PcuDispatchOp::Data(PcuDispatchDataOp::Alu {
                        value_type,
                        op: PcuDispatchAluOp::Mul,
                        ..
                    }) if *value_type == fusion_pcu::PcuValueType::f32()
                ))
                .count(),
            2
        );
        assert!(!ops.iter().any(|op| matches!(
            op,
            PcuDispatchOp::Data(PcuDispatchDataOp::Alu {
                value_type,
                op: PcuDispatchAluOp::Add | PcuDispatchAluOp::Sub,
                ..
            }) if *value_type == fusion_pcu::PcuValueType::f32()
        )));
        assert!(
            crate::lower_dispatch_to_hip_source(&kernel)
                .unwrap()
                .contains("float v4 = v3 * v1;")
        );
        assert!(matches!(prepared.lowering_plan().operations().last(),
            Some(fusion_pcu_tensor::TensorSelectedOperation::FusedMul { group }) if group.output == output));
    }

    #[test]
    fn relu_dispatch_kernel_lowers_max_against_zero() {
        let kernel = relu_kernel(17);

        let source = crate::lower_dispatch_to_hip_source(&kernel).unwrap();

        assert!(source.contains("fmaxf(v2, v1)"));
        assert_eq!(kernel.entry.logical_shape, [17, 1, 1]);
    }

    #[test]
    fn mse_dispatch_kernel_lowers_squared_difference() {
        let kernel = mse_kernel(17);
        let source = crate::lower_dispatch_to_hip_source(&kernel).unwrap();
        assert!(source.contains(" - "));
        assert!(source.contains(" * "));
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
            suppressed_adds,
            storage_constraints,
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
        RocmPreparedTensorGraph {
            graph,
            plan: tensor_plan,
            lowering_plan,
            output: outputs[0],
            outputs: outputs.to_vec(),
            nodes,
            index_by_value,
            use_counts,
            fused_add_by_relu,
            bounded_pointwise_by_output,
            bounded_mul_by_output,
            suppressed_adds,
            storage_constraints,
            physical_layouts,
        }
    }

    #[test]
    fn prewarm_requests_deduplicate_fixed_keys_and_keep_scalar_layout_mask() {
        let mut graph = Graph::default();
        let input = graph.input([16]).unwrap();
        let uniform = graph.uniform([16], 2.0).unwrap();
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
                kind: TensorDispatchKind::Add,
                logical_count: 16,
                scalar_mask: 0b10,
            }
        ));
        assert!(matches!(
            requests[1],
            TensorDispatchRequest::Fixed {
                kind: TensorDispatchKind::Relu,
                logical_count: 16,
                scalar_mask: 0,
            }
        ));
    }

    #[test]
    fn prewarm_request_collects_one_dynamic_key_for_bounded_group() {
        let mut graph = Graph::default();
        let left = graph.input([32]).unwrap();
        let right = graph.input([32]).unwrap();
        let sum = graph.add(left, right).unwrap();
        let difference = graph.sub(sum, right).unwrap();
        let output = graph.relu(difference).unwrap();
        let prepared = prepared_for_request_test(
            &graph,
            &[output],
            TensorPointwiseGroupingPolicy::BoundedAddSubRelu,
        );

        let requests = collect_dispatch_requests(&prepared).unwrap();

        assert_eq!(requests.len(), 1);
        assert!(matches!(
            &requests[0],
            TensorDispatchRequest::BoundedPointwise {
                logical_count: 32,
                scalar_mask: 0,
                topology,
                ..
            } if topology == &bounded_pointwise_topology(
                prepared.bounded_pointwise_by_output.get(&output).unwrap()
            )
        ));
    }

    #[test]
    fn prewarm_request_collects_identity_group_at_the_terminal_arithmetic_node() {
        let mut graph = Graph::default();
        let left = graph.input([32]).unwrap();
        let right = graph.input([32]).unwrap();
        let sum = graph.add(left, right).unwrap();
        let output = graph.sub(sum, right).unwrap();
        let prepared = prepared_for_request_test(
            &graph,
            &[output],
            TensorPointwiseGroupingPolicy::BoundedAddSubIdentity,
        );

        assert!(prepared.suppressed_adds.contains(&sum));
        assert!(!prepared.suppressed_adds.contains(&output));
        let requests = collect_dispatch_requests(&prepared).unwrap();
        assert_eq!(requests.len(), 1);
        assert!(matches!(
            &requests[0],
            TensorDispatchRequest::BoundedPointwise {
                logical_count: 32,
                scalar_mask: 0,
                topology,
                group,
            } if group.output == output
                && group.epilogue == TensorPointwiseEpilogue::Identity
                && topology == &bounded_pointwise_topology(group)
        ));
    }

    #[test]
    fn prewarm_retained_count_tracks_fifo_capacity_truncation() {
        let requested = (0..35)
            .map(|count| TensorDispatchCacheKey::Fixed(TensorDispatchKind::Add, count, 0))
            .collect::<Vec<_>>();
        let retained_fifo = requested[3..].to_vec();

        assert_eq!(retained_requested_key_count(&requested, &retained_fifo), 32);
        assert_eq!(
            retained_requested_key_count(&requested[..3], &retained_fifo),
            0
        );
    }

    #[test]
    fn preflight_selects_only_requested_output_dependencies() {
        let mut graph = Graph::default();
        let left = graph.input([2, 2]).unwrap();
        let right = graph.input([2, 2]).unwrap();
        let result = graph.matmul(left, right).unwrap();
        let unrelated_left = graph.input([2, 2]).unwrap();
        let unrelated_right = graph.input([2, 2]).unwrap();
        let _unsupported_but_unreachable = graph.add(unrelated_left, unrelated_right).unwrap();
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
    fn rocm_preflight_defaults_to_strict_unrewritten_selected_schedule() {
        let mut graph = Graph::default();
        let weights = graph.input([1]).unwrap();
        let gradient = graph.input([1]).unwrap();
        let rate = graph.constant(Tensor::new([1], vec![0.125]).unwrap());
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
    fn compact_uniform_storage_is_limited_to_elementwise_consumers() {
        let mut graph = Graph::default();
        let input = graph.input([4]).unwrap();
        let uniform = graph.uniform([4], 0.25).unwrap();
        let output = graph.add(input, uniform).unwrap();
        let prepared = prepare_graph_outputs_plan(&graph, &[output], &PureRocmAssessor).unwrap();

        assert_eq!(
            prepared.physical_layouts[&uniform],
            RocmPhysicalLayout::uniform_scalar()
        );
        assert!(prepared.storage_constraints.iter().any(|constraint| {
            (constraint.left == uniform && constraint.left_bytes == 4)
                || (constraint.right == uniform && constraint.right_bytes == 4)
        }));

        let mut graph = Graph::default();
        let uniform = graph.uniform([4], 0.25).unwrap();
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
        let input = graph.input([4]).unwrap();
        let uniform = graph.uniform([4], 0.25).unwrap();
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
    fn compact_uniform_fans_out_across_selected_elementwise_outputs() {
        let mut graph = Graph::default();
        let input = graph.input([4]).unwrap();
        let uniform = graph.uniform([4], 0.25).unwrap();
        let added = graph.add(input, uniform).unwrap();
        let subtracted = graph.sub(added, uniform).unwrap();
        let multiplied = graph.mul(subtracted, uniform).unwrap();

        let prepared =
            prepare_graph_outputs_plan(&graph, &[subtracted, multiplied], &PureRocmAssessor)
                .unwrap();

        assert_eq!(
            prepared.physical_layouts[&uniform],
            RocmPhysicalLayout::uniform_scalar()
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
                constraint.left_bytes == 4
            } else {
                constraint.right_bytes == 4
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
        let left = graph.input([4]).unwrap();
        let right = graph.input([4]).unwrap();
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
        assert!(prepared.suppressed_adds.contains(&added));
        assert_eq!(
            prepared.fused_add_by_relu.get(&output),
            Some(&(left, right))
        );
        assert_eq!(prepared.use_counts[prepared.index_by_value[&added]], 0);
        assert!(
            prepared
                .storage_constraints
                .iter()
                .all(|constraint| constraint.left != added && constraint.right != added)
        );
    }

    #[test]
    fn selected_uniform_output_stays_dense_and_mixed_fanout_stays_dense() {
        let mut graph = Graph::default();
        let uniform = graph.uniform([4], 0.25).unwrap();
        let prepared = prepare_graph_outputs_plan(&graph, &[uniform], &PureRocmAssessor).unwrap();
        assert_eq!(
            prepared.physical_layouts[&uniform],
            RocmPhysicalLayout::dense(16)
        );

        let mut graph = Graph::default();
        let input = graph.input([4]).unwrap();
        let uniform = graph.uniform([4], 0.25).unwrap();
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
        let uniform = graph.uniform([0], 0.25).unwrap();
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
        let weights = graph.input([1]).unwrap();
        let gradient = graph.input([1]).unwrap();
        let rate = graph.constant(Tensor::new([1], vec![0.125]).unwrap());
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
        let input = graph.input([2]).unwrap();
        let first_constant = graph.constant(Tensor::new([2], vec![1.0; 2]).unwrap());
        let first = graph.add(input, first_constant).unwrap();
        let second_constant = graph.constant(Tensor::new([2], vec![2.0; 2]).unwrap());
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
        let input = graph.input([2]).unwrap();

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
    fn preflight_accepts_supported_mse_reduction() {
        let mut graph = Graph::default();
        let left = graph.input([2, 2]).unwrap();
        let right = graph.input([2, 2]).unwrap();
        let output = graph.mean_squared_error(left, right).unwrap();

        let prepared = prepare_graph(&graph, output, &PureRocmAssessor).unwrap();
        assert_eq!(prepared.nodes.last().unwrap().value, output);
        assert_eq!(prepared.nodes.last().unwrap().shape, &[]);
    }

    #[test]
    fn mse_scratch_capacity_covers_largest_reduction_in_multi_output_plan() {
        let mut graph = Graph::default();
        let left = graph.input([2, 3]).unwrap();
        let right = graph.input([2, 3]).unwrap();
        let first_loss = graph.mean_squared_error(left, right).unwrap();
        let next_left = graph.input([4]).unwrap();
        let next_right = graph.input([4]).unwrap();
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
    fn sgd_update_preflight_is_native_and_tracks_both_operands() {
        let mut graph = Graph::default();
        let weights = graph.input([3]).unwrap();
        let gradient = graph.input([3]).unwrap();
        let updated = graph.sgd_update(weights, gradient, 0.25).unwrap();

        let plan = prepare_graph(&graph, updated, &PureRocmAssessor).unwrap();
        let node = plan.nodes.last().unwrap();
        assert!(
            matches!(node.op, fusion_pcu_tensor::OpDescriptor::SgdUpdate {
            weights: value_weights, gradient: value_gradient, learning_rate
        } if value_weights == weights
            && value_gradient == gradient
            && (learning_rate - 0.25).abs() < f32::EPSILON)
        );
        assert_eq!(plan.use_counts, [1, 1, 1]);
        assert_eq!(
            assess_tensor_node(&graph, *node),
            TensorOperationSupport::Supported {
                route: TensorExecutionRoute::Native,
                workspace_bytes: Some(0),
            }
        );
    }

    #[test]
    fn scratch_retains_constants_and_non_output_computations_only() {
        let mut graph = Graph::default();
        let input = graph.input([2]).unwrap();
        let constant = graph.constant(Tensor::new([2], vec![1.0; 2]).unwrap());
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
        let left = graph.input([2, 2]).unwrap();
        let right = graph.input([2, 2]).unwrap();
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
        let input = graph.input([2, 2]).unwrap();
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
    fn assessor_selects_library_for_dense_rank_two_matmul() {
        let mut graph = Graph::default();
        let left = graph.input([2, 3]).unwrap();
        let right = graph.input([3, 4]).unwrap();
        let result = graph.matmul(left, right).unwrap();
        let node = graph.nodes().find(|node| node.value == result).unwrap();

        assert_eq!(
            assess_tensor_node(&graph, node),
            TensorOperationSupport::Supported {
                route: TensorExecutionRoute::Library,
                workspace_bytes: None,
            }
        );
    }

    #[test]
    fn assessor_synthesizes_add_relu_and_mse() {
        let mut graph = Graph::default();
        let first = graph.input([2, 2]).unwrap();
        let second = graph.input([2, 2]).unwrap();
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
        assert_eq!(
            assess_tensor_node(&graph, node),
            TensorOperationSupport::Supported {
                route: TensorExecutionRoute::Synthesized,
                workspace_bytes: None,
            }
        );
    }

    #[test]
    fn assessor_rejects_zero_sized_matmul_output() {
        let mut graph = Graph::default();
        let left = graph.input([0, 3]).unwrap();
        let right = graph.input([3, 2]).unwrap();
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
        let left = graph.input([2, 0]).unwrap();
        let right = graph.input([0, 3]).unwrap();
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
        let left = graph.input([i32::MAX as usize + 1, 1]).unwrap();
        let right = graph.input([1, 1]).unwrap();
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
