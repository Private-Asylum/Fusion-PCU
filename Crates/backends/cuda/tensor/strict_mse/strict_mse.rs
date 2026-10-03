//! Ordered strict loss: checked subtraction, square, ascending sum, final division.
//!
//! Each prescribed operation uses the integer-only RNE/after-rounding IEEE-derived
//! helpers. Flattened reduction order and failure-before-publication are PCU contracts,
//! not a guarantee that other native loss algorithms produce identical results.
use std::fmt::Write as _;
#[rustfmt::skip]
use fusion_pcu::{
    PcuBindingAccess,
    PcuBindingRef,
    PcuBindingType,
    PcuExecutionFault,
    PcuExecutionFaultKind,
    PcuFloatUnderflowPolicy,
    PcuImplementationRequirements,
    PcuRangePolicy,
    PcuNumericalMode,
    PcuNumericalRequirement,
    PcuOwnedBindingRequirement,
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
    TensorExecutionRoute,
    TensorOperationSupport,
    TensorUnsupportedReason,
    ValueId,
};
use super::CudaTensorExecutionError;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Profile {
    scalar: PcuScalarType,
    policy: PcuFloatUnderflowPolicy,
    numerical_requirements: PcuImplementationRequirements,
    count: u32,
}
impl Profile {
    // Private factory admission establishes positive dimensions and packed event bounds.
    pub(crate) fn fault_domain(self) -> fusion_pcu::dialect::tensor::TensorStrictFaultDomain {
        fusion_pcu::dialect::tensor::TensorStrictFaultDomain::mse(
            self.scalar,
            u64::from(self.count),
            self.policy,
        )
        .expect("verified checked compound dimensions and format")
    }
    pub(crate) fn fault_extent(self) -> u64 {
        self.fault_domain().event_extent()
    }

    pub(super) fn from_node(
        graph: &Graph,
        node: NodeDescriptor<'_>,
    ) -> Result<Self, TensorUnsupportedReason> {
        let unsupported = |requirement| TensorUnsupportedReason::NumericalPolicy {
            requirement,
            options: node.numerical_options,
        };
        if graph.node(node.value).ok() != Some(node) {
            return Err(TensorUnsupportedReason::Operation);
        }
        if node.numerical_options.reproducibility != PcuReproducibility::Unspecified {
            return Err(unsupported(PcuNumericalRequirement::Reproducibility));
        }
        if node.numerical_mode != Some(PcuNumericalMode::Strict) {
            return Err(unsupported(PcuNumericalRequirement::CompoundArithmetic));
        }
        if !matches!(node.scalar_type, PcuScalarType::F32 | PcuScalarType::F64) {
            return Err(TensorUnsupportedReason::ElementType);
        }
        let OpDescriptor::MeanSquaredError { prediction, target } = node.op else {
            return Err(TensorUnsupportedReason::Operation);
        };
        let prediction = graph
            .node(prediction)
            .map_err(|_| TensorUnsupportedReason::Operation)?;
        let target = graph
            .node(target)
            .map_err(|_| TensorUnsupportedReason::Operation)?;
        if prediction.scalar_type != node.scalar_type || target.scalar_type != node.scalar_type {
            return Err(TensorUnsupportedReason::ElementType);
        }
        if prediction.shape != target.shape || !node.shape.is_empty() {
            return Err(TensorUnsupportedReason::Shape);
        }
        let count = prediction
            .shape
            .iter()
            .try_fold(1_usize, |n, d| n.checked_mul(*d))
            .and_then(|n| u32::try_from(n).ok())
            .filter(|n| *n > 0 && i32::try_from(*n).is_ok())
            .ok_or(TensorUnsupportedReason::Shape)?;
        Ok(Self {
            scalar: node.scalar_type,
            // Permissions may use this stronger ordered checker; retain the requested tuple.
            numerical_requirements: PcuImplementationRequirements {
                numerical_mode: PcuNumericalMode::Strict,
                numerical_options: node.numerical_options,
                float_underflow: node.float_underflow_policy.unwrap_or_default(),
                // Tensor nodes currently represent only Reject range.
                range_policy: PcuRangePolicy::Reject,
            },
            policy: node.float_underflow_policy.unwrap_or_default(),
            count,
        })
    }
    // One lane executes the prescribed ascending reduction; logical event IDs identify steps.
    #[allow(clippy::unused_self)] // Factory interface shared with pointwise checked dispatch profiles.
    pub(crate) const fn count(self) -> u32 {
        1
    }
    #[allow(clippy::unused_self)] // Factory interface shared with native/checked dispatch profiles.
    pub(crate) const fn checked(self) -> bool {
        true
    }
    pub(crate) fn requirements(self) -> [PcuOwnedBindingRequirement; 3] {
        let width = if self.scalar == PcuScalarType::F32 {
            4
        } else {
            8
        };
        [0_u32, 1, 2].map(|slot| PcuOwnedBindingRequirement {
            target: PcuBindingRef::new(0, slot),
            access: if slot == 2 {
                PcuBindingAccess::WriteOnly
            } else {
                PcuBindingAccess::ReadOnly
            },
            binding_type: PcuBindingType::Value(PcuValueType::Scalar(self.scalar)),
            min_required_bytes: if slot == 2 {
                width
            } else {
                u64::from(self.count) * width
            },
        })
    }
    pub(crate) fn source(self) -> String {
        let mut source = String::new();
        crate::codegen::lower::append_checked_float_helpers(&mut source, self.scalar);
        let (bits, prefix, result, divisor) = if self.scalar == PcuScalarType::F32 {
            (
                "unsigned int",
                "f32",
                "FusionF32CheckedResult",
                u64::from(
                    fusion_pcu::dialect::tensor::constants::count_f32(self.count as usize)
                        .to_bits(),
                ),
            )
        } else {
            (
                "unsigned long long",
                "f64",
                "FusionF64CheckedResult",
                fusion_pcu::dialect::tensor::constants::count_f64(self.count as usize).to_bits(),
            )
        };
        let policy = match self.policy {
            PcuFloatUnderflowPolicy::IeeeAfterRounding => 0,
            PcuFloatUnderflowPolicy::RejectSubnormalResult => 1,
            PcuFloatUnderflowPolicy::AllowGradualUnderflow => 2,
        };
        writeln!(source, r#"extern "C" __global__ void fusion_kernel(
    const {bits}* prediction, const {bits}* target, {bits}* output, unsigned long long* fault) {{
    if (blockIdx.x != 0u || threadIdx.x != 0u) return;
    {bits} sum = 0;
    for (unsigned long long i = 0ull; i < {}ull; ++i) {{
        const {result} difference = fusion_checked_{prefix}_binary(prediction[i], target[i], 1u, {policy}u);
        if (difference.fault != 0u) {{ atomicMin(fault, ((i * 3ull) << 3u) | difference.fault); return; }}
        const {result} square = fusion_checked_{prefix}_binary(difference.bits, difference.bits, 2u, {policy}u);
        if (square.fault != 0u) {{ atomicMin(fault, ((i * 3ull + 1ull) << 3u) | square.fault); return; }}
        const {result} addition = fusion_checked_{prefix}_binary(sum, square.bits, 0u, {policy}u);
        if (addition.fault != 0u) {{ atomicMin(fault, ((i * 3ull + 2ull) << 3u) | addition.fault); return; }}
        sum = addition.bits;
    }}
    const {result} mean = fusion_checked_{prefix}_binary(sum, static_cast<{bits}>({divisor}ull), 3u, {policy}u);
    if (mean.fault != 0u) {{ atomicMin(fault, (({}ull * 3ull) << 3u) | mean.fault); return; }}
    output[0] = mean.bits;
}}"#, self.count, self.count).expect("String formatting cannot fail");
        source
    }
    fn fault(
        self,
        value: ValueId,
        fault: PcuExecutionFault,
    ) -> Result<TensorError, CudaTensorExecutionError> {
        if fault.recovered
            || fault.invocation_id > u64::from(self.count) * 3
            || !matches!(
                fault.kind,
                PcuExecutionFaultKind::ArithmeticOverflow
                    | PcuExecutionFaultKind::ArithmeticUnderflow
                    | PcuExecutionFaultKind::InvalidFloatingOperand
            )
        {
            return Err(CudaTensorExecutionError::InvalidPlan(value));
        }
        let step = if fault.invocation_id == u64::from(self.count) * 3 {
            TensorArithmeticStep::Divide
        } else {
            match fault.invocation_id % 3 {
                0 => TensorArithmeticStep::Subtract,
                1 => TensorArithmeticStep::Multiply,
                _ => TensorArithmeticStep::Add,
            }
        };
        Ok(TensorError::CompoundArithmeticFault {
            value,
            element_index: 0,
            reduction_index: usize::try_from(fault.invocation_id / 3)
                .map_err(|_| CudaTensorExecutionError::SizeOverflow)?,
            step,
            kind: fault.kind,
        })
    }
}
pub(super) fn assess(graph: &Graph, node: NodeDescriptor<'_>) -> TensorOperationSupport {
    match Profile::from_node(graph, node) {
        Ok(_) => TensorOperationSupport::Supported {
            route: TensorExecutionRoute::Synthesized,
            workspace_bytes: Some(0),
        },
        Err(reason) => TensorOperationSupport::Unsupported { reason },
    }
}
impl super::CudaTensorAssessor<'_> {
    pub(super) fn execute_strict_mse(
        &self,
        graph: &Graph,
        node: NodeDescriptor<'_>,
        input: &crate::CudaMemoryResource,
        upstream: &crate::CudaMemoryResource,
        output: &crate::CudaMemoryResource,

        mut status: Option<&mut super::owned_scratch::Status>,
    ) -> Result<(), CudaTensorExecutionError> {
        use fusion_pcu::PcuOwnedDispatchMemorySession;
        use fusion_pcu::PcuOwnedCompletion;
        let profile = Profile::from_node(graph, node).map_err(|reason| {
            CudaTensorExecutionError::Unsupported {
                value: node.value,
                reason,
            }
        })?;
        self.ensure_strict_mse_cached(profile)?;
        let bindings = [input, upstream, output]
            .into_iter()
            .zip(profile.requirements())
            .map(|(resource, requirement)| {
                self.session
                    .bind(
                        requirement.target,
                        requirement.access,
                        requirement.binding_type,
                        resource,
                    )
                    .map_err(CudaTensorExecutionError::Backend)
            })
            .collect::<Result<smallvec::SmallVec<[_; 3]>, _>>()?;
        let mut completion = {
            let cache = self.state().add_dispatches.borrow();
            let prepared = cache
                .iter()
                .find(|(key, _)| *key == super::TensorDispatchCacheKey::StrictMse(profile))
                .map(|(_, prepared)| prepared)
                .ok_or(CudaTensorExecutionError::InvalidPlan(node.value))?;
            super::owned_scratch::submit(prepared, &bindings, status.as_deref_mut())?
        };
        let outcome = completion
            .wait()
            .map_err(CudaTensorExecutionError::Completion)?;
        if let Some(status) = status {
            status.observe(outcome);
        }
        match outcome {
            fusion_pcu::PcuCompletionOutcome::Succeeded => Ok(()),
            fusion_pcu::PcuCompletionOutcome::Failed => {
                Err(CudaTensorExecutionError::FailedCompletion)
            }
            fusion_pcu::PcuCompletionOutcome::Fault(fault) => {
                Err(profile.fault(node.value, fault)?.into())
            }
        }
    }
}

impl super::CudaTensorAssessor<'_> {
    pub(super) fn ensure_strict_mse_cached(
        &self,
        profile: Profile,
    ) -> Result<super::TensorDispatchCacheAdmission, CudaTensorExecutionError> {
        self.cache_prepared_dispatch(super::TensorDispatchCacheKey::StrictMse(profile), || {
            self.session
                .prepare_strict_mse_dispatch(profile, &self.state().stream)
                .map_err(CudaTensorExecutionError::Backend)
        })
    }
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;

/// Generate the exact ordered checker for an independent native comparison.
///
/// Three scalar storage pointers are followed by a u64 status pointer initialized to
/// `u64::MAX`. One active lane reduces all inputs in increasing flattened order. Wait,
/// inspect status, and discard every output on fault before publication.
///
/// # Errors
/// Rejects unsupported policy/type/provenance/shape combinations before source generation.
pub fn lower_strict_mse_to_cuda_source(
    graph: &Graph,
    value: ValueId,
) -> Result<String, CudaTensorExecutionError> {
    let node = graph.node(value)?;
    let profile = Profile::from_node(graph, node)
        .map_err(|reason| CudaTensorExecutionError::Unsupported { value, reason })?;
    Ok(profile.source())
}
