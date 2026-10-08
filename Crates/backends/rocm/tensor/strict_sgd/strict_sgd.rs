//! Strict F32/F64 SGD synthesis through the owned PCU Dispatch completion spine.
//!
//! IEEE 754-derived RNE and tininess-after-rounding rules are delegated to the integer-only
//! helpers. Checking multiply then subtract in logical element order is a PCU contract.
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
    PcuInvocationShape,
    PcuNumericalMode,
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
    ValueId,
};
use core::num::NonZeroU32;
use std::fmt::Write as _;
use super::RocmTensorExecutionError;

/// Only this cold factory may establish the safe generated-kernel binding/fault ABI.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[allow(clippy::redundant_pub_crate)] // Preserve the backend-private typed-factory boundary.
pub(crate) struct StrictSgdSpec {
    count: u32,
    scalar: PcuScalarType,
    policy: PcuFloatUnderflowPolicy,
    numerical_requirements: PcuImplementationRequirements,
    rate_bits: u32,
}

impl StrictSgdSpec {
    // Private factory admission establishes positive dimensions and packed event bounds.
    pub(crate) fn fault_domain(self) -> fusion_pcu::dialect::tensor::TensorStrictFaultDomain {
        fusion_pcu::dialect::tensor::TensorStrictFaultDomain::sgd(
            self.scalar,
            u64::from(self.count),
            self.policy,
        )
        .expect("verified checked compound dimensions and format")
    }
    pub(crate) fn fault_extent(self) -> u64 {
        self.fault_domain().event_extent()
    }

    pub(super) fn from_node(graph: &Graph, node: NodeDescriptor<'_>) -> Option<Self> {
        let OpDescriptor::SgdUpdate {
            weights,
            gradient,
            learning_rate,
        } = node.op
        else {
            return None;
        };
        if node.numerical_mode != Some(PcuNumericalMode::Strict)
            || node.numerical_options.reproducibility != PcuReproducibility::Unspecified
            || !matches!(node.scalar_type, PcuScalarType::F32 | PcuScalarType::F64)
            || !learning_rate.is_finite()
        {
            return None;
        }
        let actual = graph.node(node.value).ok()?;
        let OpDescriptor::SgdUpdate {
            learning_rate: actual_rate,
            ..
        } = actual.op
        else {
            return None;
        };
        // Descriptor equality aliases +0/-0; frozen executable identity must not.
        if actual != node || actual_rate.to_bits() != learning_rate.to_bits() {
            return None;
        }
        for input in [weights, gradient] {
            let input = graph.node(input).ok()?;
            if input.scalar_type != node.scalar_type || input.shape != node.shape {
                return None;
            }
        }
        let count = node
            .shape
            .iter()
            .try_fold(1_usize, |n, d| n.checked_mul(*d))?;
        let spec = Self {
            count: u32::try_from(count).ok().filter(|n| *n != 0)?,
            scalar: node.scalar_type,
            // Permissions may use this stronger ordered checker; retain the requested tuple.
            numerical_requirements: PcuImplementationRequirements {
                numerical_mode: PcuNumericalMode::Strict,
                numerical_options: node.numerical_options,
                float_underflow: node.float_underflow_policy.unwrap_or_default(),
                // Tensor nodes currently represent only Reject range.
                range_policy: PcuRangePolicy::Reject,
            },
            policy: node.float_underflow_policy?,
            rate_bits: learning_rate.to_bits(),
        };
        // Each logical lane has two fault events, packed below Dispatch's recovered bit.
        // The u32 launch extent already bounds this count; byte extents must fit the host too.
        usize::try_from(u64::from(spec.count) * spec.scalar_bytes()).ok()?;
        Some(spec)
    }

    const fn scalar_bytes(self) -> u64 {
        if matches!(self.scalar, PcuScalarType::F32) {
            4
        } else {
            8
        }
    }

    pub(crate) const fn shape(self) -> PcuInvocationShape {
        PcuInvocationShape::invocations(NonZeroU32::new(self.count).expect("validated count"))
    }

    pub(crate) fn requirements(self) -> [PcuOwnedBindingRequirement; 3] {
        [0_u32, 1, 2].map(|binding| PcuOwnedBindingRequirement {
            target: PcuBindingRef::new(0, binding),
            access: if binding == 2 {
                PcuBindingAccess::WriteOnly
            } else {
                PcuBindingAccess::ReadOnly
            },
            binding_type: PcuBindingType::Value(PcuValueType::Scalar(self.scalar)),
            min_required_bytes: u64::from(self.count) * self.scalar_bytes(),
        })
    }

    pub(crate) fn source(self) -> String {
        let (bits, prefix, result, rate) = match self.scalar {
            PcuScalarType::F32 => (
                "unsigned int",
                "f32",
                "FusionF32CheckedResult",
                u64::from(self.rate_bits),
            ),
            PcuScalarType::F64 => (
                "unsigned long long",
                "f64",
                "FusionF64CheckedResult",
                fusion_pcu::f32_bits_to_f64_bits(self.rate_bits),
            ),
            _ => unreachable!("private F32/F64 profile"),
        };
        let policy = match self.policy {
            PcuFloatUnderflowPolicy::IeeeAfterRounding => 0,
            PcuFloatUnderflowPolicy::RejectSubnormalResult => 1,
            PcuFloatUnderflowPolicy::AllowGradualUnderflow => 2,
        };
        let mut source = String::new();
        crate::codegen::lower::emit_compound_float_helpers(&mut source, self.scalar);
        writeln!(source, r#"
extern "C" __global__ void fusion_kernel(
    const {bits}* weights, const {bits}* gradient, {bits}* output,
    unsigned long long* fault) {{
    const unsigned long long id = static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (id >= {}ull) return;
    const unsigned long long event = id * 2ull;
    const {result} product = fusion_checked_{prefix}_binary(static_cast<{bits}>({rate}ull), gradient[id], 2u, {policy}u);
    if (product.fault != 0u) {{ atomicMin(fault, (event << 3u) | product.fault); return; }}
    const {result} difference = fusion_checked_{prefix}_binary(weights[id], product.bits, 1u, {policy}u);
    if (difference.fault != 0u) {{ atomicMin(fault, ((event + 1ull) << 3u) | difference.fault); return; }}
    output[id] = difference.bits;
}}
"#, self.count).expect("String formatting cannot fail");
        source
    }

    pub(super) fn fault_error(
        self,
        value: ValueId,
        fault: PcuExecutionFault,
    ) -> Result<TensorError, RocmTensorExecutionError> {
        let element = fault.invocation_id / 2;
        if fault.recovered
            || element >= u64::from(self.count)
            || !matches!(
                fault.kind,
                PcuExecutionFaultKind::ArithmeticOverflow
                    | PcuExecutionFaultKind::ArithmeticUnderflow
                    | PcuExecutionFaultKind::InvalidFloatingOperand
            )
        {
            return Err(RocmTensorExecutionError::InvalidPlan(value));
        }
        Ok(TensorError::CompoundArithmeticFault {
            value,
            element_index: usize::try_from(element)
                .map_err(|_| RocmTensorExecutionError::SizeOverflow)?,
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

#[rustfmt::skip]
use super::{
    ADD_DISPATCH_CACHE_CAPACITY,
    RocmTensorAssessor,
    TensorDispatchCacheAdmission,
    TensorDispatchCacheKey,
};
#[rustfmt::skip]
use fusion_pcu::{
    PcuCompletionOutcome,
    PcuOwnedCompletion,
    PcuOwnedDispatchMemorySession,
};
use crate::RocmMemoryResource;
use smallvec::SmallVec;

impl RocmTensorAssessor<'_> {
    pub(super) fn ensure_strict_sgd_cached(
        &self,
        spec: StrictSgdSpec,
    ) -> Result<TensorDispatchCacheAdmission, RocmTensorExecutionError> {
        let key = TensorDispatchCacheKey::StrictSgd(spec);
        let mut cache = self.state().add_dispatches.borrow_mut();
        if cache.iter().any(|(cached_key, _)| *cached_key == key) {
            return Ok(TensorDispatchCacheAdmission::default());
        }
        let prepared = self
            .session
            .prepare_strict_sgd_on_stream(spec, &self.state().stream)
            .map_err(RocmTensorExecutionError::Backend)?;
        let evicted = cache.len() == ADD_DISPATCH_CACHE_CAPACITY;
        if evicted {
            cache.pop_front();
        }
        cache.push_back((key, prepared.into()));
        Ok(TensorDispatchCacheAdmission {
            compiled: true,
            evicted,
        })
    }

    pub(super) fn execute_strict_sgd(
        &self,
        spec: StrictSgdSpec,
        value: ValueId,
        left: &RocmMemoryResource,
        right: &RocmMemoryResource,
        output: &RocmMemoryResource,

        mut status: Option<&mut super::owned_scratch::Status>,
    ) -> Result<(), RocmTensorExecutionError> {
        self.ensure_strict_sgd_cached(spec)?;
        let key = TensorDispatchCacheKey::StrictSgd(spec);
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
        let mut completion =
            super::owned_scratch::submit(prepared, &bindings, status.as_deref_mut())?;
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
}

#[cfg(test)]
mod tests;
