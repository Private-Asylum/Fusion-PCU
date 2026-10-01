//! Constituent-checked SGD: destination-width multiply followed by subtraction.
//!
//! The frozen finite F32 learning rate is widened exactly for F64 operands. Each lane checks
//! rate*gradient before weights-product, using the existing integer-only RNE float helpers.
//! This is an explicitly Strict contract; ordinary Boundary SGD remains separately admitted.

use std::fmt::Write as _;
#[rustfmt::skip]
use fusion_pcu::{
    PcuBindingAccess,
    PcuBindingRef,
    PcuBindingType,
    PcuCompoundArithmeticPolicy,
    PcuExecutionFault,
    PcuFloatUnderflowPolicy,
    PcuNumericalMode,
    PcuNumericalRequirement,
    PcuOwnedBindingRequirement,
    PcuPrecisionPolicy,
    PcuReproducibility,
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
    TensorUnsupportedReason,
    ValueId,
};
use super::CudaTensorExecutionError;

/// Cold resolved source, bounds, policy and rate-bit identity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Profile {
    scalar: PcuScalarType,
    policy: PcuFloatUnderflowPolicy,
    count: u32,
    rate_bits: u32,
}

pub(super) fn assess(
    graph: &Graph,
    node: NodeDescriptor<'_>,
) -> Result<Profile, TensorUnsupportedReason> {
    let unsupported = |requirement| TensorUnsupportedReason::NumericalPolicy {
        requirement,
        options: node.numerical_options,
    };
    if node.numerical_options.reproducibility != PcuReproducibility::Unspecified {
        return Err(unsupported(PcuNumericalRequirement::Reproducibility));
    }
    if node.numerical_mode != Some(PcuNumericalMode::Strict)
        || node.numerical_options.compound_arithmetic != PcuCompoundArithmeticPolicy::Checked
    {
        return Err(unsupported(PcuNumericalRequirement::CompoundArithmetic));
    }
    if node.numerical_options.precision != PcuPrecisionPolicy::Preserve {
        return Err(unsupported(PcuNumericalRequirement::Precision));
    }
    if !matches!(node.scalar_type, PcuScalarType::F32 | PcuScalarType::F64) {
        return Err(TensorUnsupportedReason::ElementType);
    }
    let OpDescriptor::SgdUpdate {
        weights,
        gradient,
        learning_rate,
    } = node.op
    else {
        return Err(TensorUnsupportedReason::Operation);
    };
    let original = graph
        .node(node.value)
        .map_err(|_| TensorUnsupportedReason::Operation)?;
    let OpDescriptor::SgdUpdate {
        learning_rate: original_rate,
        ..
    } = original.op
    else {
        return Err(TensorUnsupportedReason::Operation);
    };
    // Descriptor PartialEq aliases signed zeros; source/cache identity may not.
    if original != node
        || original_rate.to_bits() != learning_rate.to_bits()
        || !learning_rate.is_finite()
    {
        return Err(TensorUnsupportedReason::Operation);
    }
    for operand in [weights, gradient] {
        let input = graph
            .node(operand)
            .map_err(|_| TensorUnsupportedReason::Operation)?;
        if input.scalar_type != node.scalar_type {
            return Err(TensorUnsupportedReason::ElementType);
        }
        if input.shape != node.shape {
            return Err(TensorUnsupportedReason::Shape);
        }
    }
    let count = node
        .shape
        .iter()
        .try_fold(1_usize, |count, dimension| count.checked_mul(*dimension))
        .and_then(|count| u32::try_from(count).ok())
        .filter(|count| *count > 0)
        .ok_or(TensorUnsupportedReason::Shape)?;
    Ok(Profile {
        scalar: node.scalar_type,
        policy: node.float_underflow_policy.unwrap_or_default(),
        count,
        rate_bits: learning_rate.to_bits(),
    })
}

impl Profile {
    pub(crate) const fn count(self) -> u32 {
        self.count
    }

    pub(crate) fn requirements(self) -> Vec<PcuOwnedBindingRequirement> {
        let width = if self.scalar == PcuScalarType::F32 {
            4
        } else {
            8
        };
        (0..3)
            .map(|slot| PcuOwnedBindingRequirement {
                target: PcuBindingRef::new(0, slot),
                access: if slot == 2 {
                    PcuBindingAccess::WriteOnly
                } else {
                    PcuBindingAccess::ReadOnly
                },
                binding_type: PcuBindingType::Value(PcuValueType::Scalar(self.scalar)),
                min_required_bytes: u64::from(self.count) * width,
            })
            .collect()
    }

    pub fn source(self) -> String {
        let mut source = String::from("#pragma clang fp contract(off)\n");
        crate::codegen::lower::append_checked_float_helpers(&mut source, self.scalar);
        let (ty, bits, prefix, result, rate) = match self.scalar {
            PcuScalarType::F32 => (
                "float",
                "unsigned int",
                "f32",
                "FusionF32CheckedResult",
                u64::from(self.rate_bits),
            ),
            PcuScalarType::F64 => (
                "double",
                "unsigned long long",
                "f64",
                "FusionF64CheckedResult",
                f64::from(f32::from_bits(self.rate_bits)).to_bits(),
            ),
            _ => unreachable!("cold profile admits F32/F64 only"),
        };
        let policy = match self.policy {
            PcuFloatUnderflowPolicy::IeeeAfterRounding => 0,
            PcuFloatUnderflowPolicy::RejectSubnormalResult => 1,
            PcuFloatUnderflowPolicy::AllowGradualUnderflow => 2,
        };
        writeln!(source, "extern \"C\" __global__ void fusion_kernel(const {ty}* weights, const {ty}* gradient, {ty}* output, unsigned long long* fusion_fault_word) {{
    const unsigned long long id = static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (id >= {}ull) return;
    const {result} product = fusion_checked_{prefix}_binary(static_cast<{bits}>({rate}ull), __builtin_bit_cast({bits}, gradient[id]), 2u, {policy}u);
    const unsigned long long event = id * 2ull;
    if (product.fault != 0u) {{ atomicMin(fusion_fault_word, (event << 3u) | product.fault); return; }}
    const {result} difference = fusion_checked_{prefix}_binary(__builtin_bit_cast({bits}, weights[id]), product.bits, 1u, {policy}u);
    if (difference.fault != 0u) {{ atomicMin(fusion_fault_word, ((event + 1ull) << 3u) | difference.fault); return; }}
    output[id] = __builtin_bit_cast({ty}, difference.bits);
}}", self.count).expect("String formatting cannot fail");
        source
    }

    fn fault(
        self,
        value: ValueId,
        fault: PcuExecutionFault,
    ) -> Result<TensorError, CudaTensorExecutionError> {
        let index = fault.invocation_id / 2;
        if fault.recovered || index >= u64::from(self.count) {
            return Err(CudaTensorExecutionError::InvalidPlan(value));
        }
        Ok(TensorError::CompoundArithmeticFault {
            value,
            element_index: usize::try_from(index)
                .map_err(|_| CudaTensorExecutionError::SizeOverflow)?,
            reduction_index: 0,
            step: if fault.invocation_id.is_multiple_of(2) {
                TensorArithmeticStep::Multiply
            } else {
                TensorArithmeticStep::Subtract
            },
            kind: fault.kind,
        })
    }
}

impl super::CudaTensorAssessor<'_> {
    pub(super) fn ensure_strict_sgd_cached(
        &self,
        profile: Profile,
    ) -> Result<super::TensorDispatchCacheAdmission, CudaTensorExecutionError> {
        self.cache_prepared_dispatch(super::TensorDispatchCacheKey::StrictSgd(profile), || {
            self.session
                .prepare_strict_sgd_dispatch(profile, &self.state().stream)
                .map_err(CudaTensorExecutionError::Backend)
        })
    }

    pub(super) fn execute_strict_sgd(
        &self,
        value: ValueId,
        profile: Profile,
        weights: &crate::CudaMemoryResource,
        gradient: &crate::CudaMemoryResource,
        output: &crate::CudaMemoryResource,
    ) -> Result<(), CudaTensorExecutionError> {
        use fusion_pcu::PcuOwnedDispatchMemorySession;
        use fusion_pcu::PcuOwnedCompletion;
        self.ensure_strict_sgd_cached(profile)?;
        let bindings = [
            (0, weights, PcuBindingAccess::ReadOnly),
            (1, gradient, PcuBindingAccess::ReadOnly),
            (2, output, PcuBindingAccess::WriteOnly),
        ]
        .into_iter()
        .map(|(slot, resource, access)| {
            self.session
                .bind(
                    PcuBindingRef::new(0, slot),
                    access,
                    PcuBindingType::Value(PcuValueType::Scalar(profile.scalar)),
                    resource,
                )
                .map_err(CudaTensorExecutionError::Backend)
        })
        .collect::<Result<smallvec::SmallVec<[_; 3]>, _>>()?;
        let mut completion = {
            let cache = self.state().add_dispatches.borrow();
            let prepared = cache
                .iter()
                .find(|(key, _)| *key == super::TensorDispatchCacheKey::StrictSgd(profile))
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

/// Generate the exact admitted Strict F32/F64 SGD checker for an independent native control.
///
/// ABI: `(const T* weights, const T* gradient, T* output, u64* status)`. Initialize status to
/// `u64::MAX`, wait terminal completion and inspect status before using output. Events encode
/// `(lane*2+step)<<3 | kind`, with Multiply=0 and Subtract=1. A failed call publishes no output.
///
/// # Errors
/// Rejects unproved policy/type/shape/rate/provenance combinations before generating source.
pub fn lower_strict_sgd_to_cuda_source(
    graph: &Graph,
    value: ValueId,
) -> Result<String, CudaTensorExecutionError> {
    let node = graph.node(value)?;
    let profile = assess(graph, node)
        .map_err(|reason| CudaTensorExecutionError::Unsupported { value, reason })?;
    Ok(profile.source())
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
