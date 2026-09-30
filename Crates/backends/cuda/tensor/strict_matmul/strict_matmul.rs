//! Ordered strict `MatMul` synthesis on the existing checked Dispatch lifetime spine.
//!
//! Each output folds increasing K from positive zero, rounding each multiply and add to the
//! destination width with the scalar integer-only checker. This PCU algorithm fixes operation
//! order and faults at constituent boundaries. IEEE Std 754-2019 4.3.1/4.3.3 specify RNE;
//! 7.4/7.5 govern range classification. Result/discard and finite-input rejection are PCU policy.

use std::fmt::Write as _;
#[rustfmt::skip]
use fusion_pcu::{
    PcuBindingAccess,
    PcuBindingRef,
    PcuBindingType,
    PcuExecutionFault,
    PcuFloatUnderflowPolicy,
    PcuNumericalMode,
    PcuOwnedBindingRequirement,
    PcuScalarType,
    PcuValueType,
};
#[rustfmt::skip]
use fusion_pcu::dialect::tensor::{
    Graph,
    NodeDescriptor,
    OpDescriptor,
    TensorArithmeticStep,
    TensorError,
    ValueId,
};
use super::CudaTensorExecutionError;

/// Full, collision-free cache identity and checked storage ABI, resolved during preparation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Profile {
    scalar: PcuScalarType,
    policy: PcuFloatUnderflowPolicy,
    rows: u32,
    inner: u32,
    columns: u32,
    left_columns: u32,
    right_columns: u32,
    transpose_left: bool,
    transpose_right: bool,
    byte_extents: [u64; 3],
}

impl Profile {
    #[allow(clippy::too_many_lines)] // One cold pass establishes type, shape, ABI and fault-index bounds.
    pub(super) fn from_node(
        graph: &Graph,
        node: NodeDescriptor<'_>,
    ) -> Result<Self, CudaTensorExecutionError> {
        let OpDescriptor::MatMul {
            left,
            right,
            transpose_left,
            transpose_right,
        } = node.op
        else {
            return Err(CudaTensorExecutionError::InvalidPlan(node.value));
        };
        if node.numerical_mode != Some(PcuNumericalMode::Strict) {
            return Err(CudaTensorExecutionError::InvalidPlan(node.value));
        }
        let width = match node.scalar_type {
            PcuScalarType::F32 => 4_u64,
            PcuScalarType::F64 => 8,
            scalar => return Err(CudaTensorExecutionError::UnsupportedScalarType(scalar)),
        };
        let left_node = graph.node(left)?;
        let right_node = graph.node(right)?;
        if left_node.scalar_type != node.scalar_type || right_node.scalar_type != node.scalar_type {
            return Err(CudaTensorExecutionError::InvalidPlan(node.value));
        }
        let left_shape: [usize; 2] = left_node
            .shape
            .try_into()
            .map_err(|_| CudaTensorExecutionError::SizeOverflow)?;
        let right_shape: [usize; 2] = right_node
            .shape
            .try_into()
            .map_err(|_| CudaTensorExecutionError::SizeOverflow)?;
        let dimensions = [
            left_shape[usize::from(transpose_left)],
            left_shape[usize::from(!transpose_left)],
            right_shape[usize::from(!transpose_right)],
        ];
        if dimensions.contains(&0)
            || dimensions[1] != right_shape[usize::from(transpose_right)]
            || node.shape != [dimensions[0], dimensions[2]]
        {
            return Err(CudaTensorExecutionError::SizeOverflow);
        }
        let [rows, inner, columns] = dimensions.map(u32::try_from);
        let rows = rows.map_err(|_| CudaTensorExecutionError::SizeOverflow)?;
        let inner = inner.map_err(|_| CudaTensorExecutionError::SizeOverflow)?;
        let columns = columns.map_err(|_| CudaTensorExecutionError::SizeOverflow)?;
        let output_count = rows
            .checked_mul(columns)
            .ok_or(CudaTensorExecutionError::SizeOverflow)?;
        // Three kind bits and one reserved recovery bit leave sixty event-index bits. Admission
        // bounds the entire output/K/step space before any launch or device allocation.
        u64::from(output_count)
            .checked_mul(u64::from(inner))
            .and_then(|count| count.checked_mul(2))
            .filter(|count| *count <= (1_u64 << 60))
            .ok_or(CudaTensorExecutionError::SizeOverflow)?;
        let elements = [
            left_shape[0].checked_mul(left_shape[1]),
            right_shape[0].checked_mul(right_shape[1]),
            usize::try_from(output_count).ok(),
        ];
        let mut byte_extents = [0; 3];
        for (extent, elements) in byte_extents.iter_mut().zip(elements) {
            *extent = u64::try_from(elements.ok_or(CudaTensorExecutionError::SizeOverflow)?)
                .ok()
                .and_then(|count| count.checked_mul(width))
                .ok_or(CudaTensorExecutionError::SizeOverflow)?;
        }
        Ok(Self {
            scalar: node.scalar_type,
            policy: node
                .float_underflow_policy
                .unwrap_or(PcuFloatUnderflowPolicy::IeeeAfterRounding),
            rows,
            inner,
            columns,
            left_columns: u32::try_from(left_shape[1])
                .map_err(|_| CudaTensorExecutionError::SizeOverflow)?,
            right_columns: u32::try_from(right_shape[1])
                .map_err(|_| CudaTensorExecutionError::SizeOverflow)?,
            transpose_left,
            transpose_right,
            byte_extents,
        })
    }

    pub const fn output_count(self) -> u32 {
        self.rows * self.columns
    }

    pub(super) const fn scalar(self) -> PcuScalarType {
        self.scalar
    }

    pub fn requirements(self) -> Vec<PcuOwnedBindingRequirement> {
        self.byte_extents
            .into_iter()
            .enumerate()
            .map(|(slot, min_required_bytes)| PcuOwnedBindingRequirement {
                target: PcuBindingRef::new(0, u32::try_from(slot).expect("three fixed bindings")),
                access: if slot == 2 {
                    PcuBindingAccess::WriteOnly
                } else {
                    PcuBindingAccess::ReadOnly
                },
                binding_type: PcuBindingType::Value(PcuValueType::Scalar(self.scalar)),
                min_required_bytes,
            })
            .collect()
    }

    pub(super) fn fault(
        self,
        value: ValueId,
        fault: PcuExecutionFault,
    ) -> Result<TensorError, CudaTensorExecutionError> {
        let event = fault.invocation_id;
        let pair = event / 2;
        let element_index = pair / u64::from(self.inner);
        if fault.recovered || element_index >= u64::from(self.output_count()) {
            return Err(CudaTensorExecutionError::InvalidPlan(value));
        }
        Ok(TensorError::CompoundArithmeticFault {
            value,
            element_index: usize::try_from(element_index)
                .map_err(|_| CudaTensorExecutionError::SizeOverflow)?,
            reduction_index: usize::try_from(pair % u64::from(self.inner))
                .map_err(|_| CudaTensorExecutionError::SizeOverflow)?,
            step: if event.is_multiple_of(2) {
                TensorArithmeticStep::Multiply
            } else {
                TensorArithmeticStep::Add
            },
            kind: fault.kind,
        })
    }

    pub fn source(self) -> String {
        let mut source = String::from("#pragma clang fp contract(off)\n");
        crate::codegen::lower::append_checked_float_helpers(&mut source, self.scalar);
        let (ty, bits, prefix, result) = match self.scalar {
            PcuScalarType::F32 => ("float", "unsigned int", "f32", "FusionF32CheckedResult"),
            PcuScalarType::F64 => (
                "double",
                "unsigned long long",
                "f64",
                "FusionF64CheckedResult",
            ),
            _ => unreachable!("profile construction admits only F32/F64"),
        };
        let policy = match self.policy {
            PcuFloatUnderflowPolicy::IeeeAfterRounding => 0,
            PcuFloatUnderflowPolicy::RejectSubnormalResult => 1,
            PcuFloatUnderflowPolicy::AllowGradualUnderflow => 2,
        };
        let left_index = if self.transpose_left {
            format!("k * {}ull + row", self.left_columns)
        } else {
            format!("row * {}ull + k", self.left_columns)
        };
        let right_index = if self.transpose_right {
            format!("column * {}ull + k", self.right_columns)
        } else {
            format!("k * {}ull + column", self.right_columns)
        };
        writeln!(source, "extern \"C\" __global__ void fusion_kernel(const {ty}* left, const {ty}* right, {ty}* output, unsigned long long* fusion_fault_word) {{
    const unsigned long long id = static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (id >= {}ull) return;
    const unsigned long long row = id / {}ull;
    const unsigned long long column = id % {}ull;
    {bits} accumulator = 0;
    for (unsigned long long k = 0; k < {}ull; ++k) {{
        const {result} product = fusion_checked_{prefix}_binary(__builtin_bit_cast({bits}, left[{left_index}]), __builtin_bit_cast({bits}, right[{right_index}]), 2u, {policy}u);
        const unsigned long long event = (id * {}ull + k) * 2ull;
        if (product.fault != 0u) {{ atomicMin(fusion_fault_word, (event << 3u) | product.fault); return; }}
        const {result} sum = fusion_checked_{prefix}_binary(accumulator, product.bits, 0u, {policy}u);
        if (sum.fault != 0u) {{ atomicMin(fusion_fault_word, ((event + 1ull) << 3u) | sum.fault); return; }}
        accumulator = sum.bits;
    }}
    output[id] = __builtin_bit_cast({ty}, accumulator);
}}", self.output_count(), self.columns, self.columns, self.inner, self.inner).expect("String formatting cannot fail");
        // CUDA intrinsics are available in both NVCC and standalone NVRTC.
        // The shared integer checker uses Clang spellings in its portable source.
        source
            .replace("__builtin_clzll(", "__clzll(")
            .replace("__builtin_clz(", "__clz(")
    }
}

impl super::CudaTensorAssessor<'_> {
    pub(super) fn ensure_strict_matmul_cached(
        &self,
        profile: Profile,
    ) -> Result<super::TensorDispatchCacheAdmission, CudaTensorExecutionError> {
        self.cache_prepared_dispatch(super::TensorDispatchCacheKey::StrictMatMul(profile), || {
            self.session
                .prepare_strict_matmul_dispatch(profile, &self.state().stream)
                .map_err(CudaTensorExecutionError::Backend)
        })
    }

    pub(super) fn execute_strict_matmul(
        &self,
        value: ValueId,
        profile: Profile,
        left: &crate::CudaMemoryResource,
        right: &crate::CudaMemoryResource,
        output: &crate::CudaMemoryResource,
    ) -> Result<(), CudaTensorExecutionError> {
        use fusion_pcu::PcuOwnedDispatchMemorySession;
        use fusion_pcu::PcuOwnedCompletion;
        self.ensure_strict_matmul_cached(profile)?;
        let bindings = [
            (0, left, PcuBindingAccess::ReadOnly),
            (1, right, PcuBindingAccess::ReadOnly),
            (2, output, PcuBindingAccess::WriteOnly),
        ]
        .into_iter()
        .map(|(slot, resource, access)| {
            self.session
                .bind(
                    PcuBindingRef::new(0, slot),
                    access,
                    PcuBindingType::Value(PcuValueType::Scalar(profile.scalar())),
                    resource,
                )
                .map_err(CudaTensorExecutionError::Backend)
        })
        .collect::<Result<smallvec::SmallVec<[_; 3]>, _>>()?;
        let mut completion = {
            let cache = self.state().add_dispatches.borrow();
            let prepared = cache
                .iter()
                .find(|(key, _)| *key == super::TensorDispatchCacheKey::StrictMatMul(profile))
                .map(|(_, prepared)| prepared)
                .ok_or(CudaTensorExecutionError::InvalidPlan(value))?;
            prepared
                .submit(&bindings)
                .map_err(CudaTensorExecutionError::Backend)?
        };
        match completion
            .wait()
            .map_err(CudaTensorExecutionError::Completion)?
        {
            fusion_pcu::PcuCompletionOutcome::Succeeded => Ok(()),
            fusion_pcu::PcuCompletionOutcome::Failed => {
                Err(CudaTensorExecutionError::FailedCompletion)
            }
            fusion_pcu::PcuCompletionOutcome::Fault(fault) => {
                Err(profile.fault(value, fault)?.into())
            }
        }
    }
}

/// Lower one explicitly strict F32/F64 dense `MatMul` to the same CUDA source used by execution.
///
/// `fusion_kernel` takes `(const T* left, const T* right, T* output, u64* status)` in that order.
/// A lane owns one row-major output cell; launch enough threads for rows*columns, with excess
/// threads guarded. Initialize private status to `u64::MAX`, wait terminal completion, then read
/// it before consuming output. Fatal status encodes `((cell*K+k)*2+step)<<3 | kind`, where
/// Multiply is step zero and Add step one. Kind tags use the existing checked CUDA ABI. No
/// failure publishes an initialized output. The generated integer-only arithmetic folds K in
/// increasing order with separate multiply/add rounding and never selects BLAS or CPU fallback.
///
/// # Errors
///
/// Returns graph, mode, scalar, shape, storage-size or deterministic fault-index admission errors.
pub fn lower_strict_matmul_to_cuda_source(
    graph: &Graph,
    value: ValueId,
) -> Result<String, CudaTensorExecutionError> {
    Ok(Profile::from_node(graph, graph.node(value)?)?.source())
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
