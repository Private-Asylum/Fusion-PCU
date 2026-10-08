//! Exact dense six-format checked `ReLU` derivative and explicitly native selection.
//!
//! Checked input finiteness, zero derivative at both signed zeros, and validation of the
//! inactive upstream are PCU policies. Selection copies bits without rounding, contraction
//! or inexactness: IEEE 754-2019 6.1/6.2 distinguish finite/subnormal encodings from NaNs
//! and infinities. `RejectSubnormalResult` is an additional PCU rule, not IEEE underflow.

#[rustfmt::skip]
use fusion_pcu::{
    PcuBindingAccess,
    PcuBindingRef,
    PcuBindingType,
    PcuCompoundArithmeticPolicy,
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
    TensorError,
    TensorExecutionRoute,
    TensorOperationSupport,
    TensorUnsupportedReason,
    ValueId,
};
use std::fmt::Write as _;

/// Factory-controlled storage/policy identity; never accepts arbitrary source.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Profile {
    scalar: PcuScalarType,
    policy: PcuFloatUnderflowPolicy,
    count: u32,
    checked: bool,
    numerical_requirements: PcuImplementationRequirements,
}

impl Profile {
    pub(super) fn from_node(node: NodeDescriptor<'_>) -> Result<Self, TensorUnsupportedReason> {
        let unsupported = |requirement| TensorUnsupportedReason::NumericalPolicy {
            requirement,
            options: node.numerical_options,
        };
        if !matches!(node.op, OpDescriptor::ReluBackward { .. }) {
            return Err(TensorUnsupportedReason::Operation);
        }
        if node.numerical_options.reproducibility != PcuReproducibility::Unspecified {
            return Err(unsupported(PcuNumericalRequirement::Reproducibility));
        }
        if !super::is_checked_float_type(node.scalar_type) {
            return Err(TensorUnsupportedReason::ElementType);
        }
        if !matches!(
            node.numerical_mode,
            Some(PcuNumericalMode::Boundary | PcuNumericalMode::Strict)
        ) {
            return Err(unsupported(PcuNumericalRequirement::CompoundArithmetic));
        }
        let policy = node.float_underflow_policy.unwrap_or_default();
        // BackendDefined permits a stronger checked implementation. Strict detection or
        // a non-IEEE underflow request requires it, even for F32/F64. Preserve the
        // existing status-free native Boundary/IEEE path when those checks are absent.
        let checked = super::is_low_float_type(node.scalar_type)
            || node.numerical_options.compound_arithmetic == PcuCompoundArithmeticPolicy::Checked
            || node.numerical_mode == Some(PcuNumericalMode::Strict)
            || policy != PcuFloatUnderflowPolicy::IeeeAfterRounding;
        let count = node
            .shape
            .iter()
            .try_fold(1_usize, |n, d| n.checked_mul(*d))
            .and_then(|n| u32::try_from(n).ok())
            .filter(|n| *n > 0)
            .ok_or(TensorUnsupportedReason::Shape)?;
        let width = u64::from(node.scalar_type.bit_width() / 8);
        usize::try_from(u64::from(count) * width).map_err(|_| TensorUnsupportedReason::Shape)?;
        Ok(Self {
            scalar: node.scalar_type,
            policy,
            count,
            checked,
            numerical_requirements: PcuImplementationRequirements {
                numerical_mode: node
                    .numerical_mode
                    .expect("admission verifies compound mode"),
                numerical_options: node.numerical_options,
                float_underflow: policy,
                // Tensor nodes have no range metadata: only Reject is represented here.
                range_policy: PcuRangePolicy::Reject,
            },
        })
    }

    pub(crate) const fn count(self) -> u32 {
        self.count
    }
    pub(crate) const fn fault_law(self) -> Option<fusion_pcu::PcuCheckedScalarFaultLaw> {
        if self.checked {
            fusion_pcu::PcuCheckedScalarFaultLaw::float_relu_backward(
                self.scalar,
                PcuRangePolicy::Reject,
                self.policy,
            )
        } else {
            None
        }
    }

    pub(crate) const fn checked(self) -> bool {
        self.checked
    }
    pub(crate) fn requirements(self) -> [PcuOwnedBindingRequirement; 3] {
        let width = u64::from(self.scalar.bit_width() / 8);
        [0_u32, 1, 2].map(|slot| PcuOwnedBindingRequirement {
            target: PcuBindingRef::new(0, slot),
            access: if slot == 2 {
                PcuBindingAccess::WriteOnly
            } else {
                PcuBindingAccess::ReadOnly
            },
            binding_type: PcuBindingType::Value(PcuValueType::Scalar(self.scalar)),
            min_required_bytes: u64::from(self.count) * width,
        })
    }

    pub(crate) fn source(self) -> String {
        let (bits, exponent, sign, magnitude, maximum, min_normal) = match self.scalar {
            PcuScalarType::F16 => (
                "unsigned short",
                "0x7c00u",
                "0x8000u",
                "0x7fffu",
                "0x7bffu",
                "0x400u",
            ),
            PcuScalarType::BF16 => (
                "unsigned short",
                "0x7f80u",
                "0x8000u",
                "0x7fffu",
                "0x7f7fu",
                "0x80u",
            ),
            PcuScalarType::F8E4M3FN => {
                ("unsigned char", "0x78u", "0x80u", "0x7fu", "0x7eu", "0x8u")
            }
            PcuScalarType::F8E5M2 => ("unsigned char", "0x7cu", "0x80u", "0x7fu", "0x7bu", "0x4u"),
            PcuScalarType::F32 => (
                "unsigned int",
                "0x7f800000u",
                "0x80000000u",
                "0x7fffffffu",
                "0x7f7fffffu",
                "0x800000u",
            ),
            PcuScalarType::F64 => (
                "unsigned long long",
                "0x7ff0000000000000ull",
                "0x8000000000000000ull",
                "0x7fffffffffffffffull",
                "0x7fefffffffffffffull",
                "0x10000000000000ull",
            ),
            _ => unreachable!("factory admits exactly six named floating encodings"),
        };
        let status_arg = if self.checked {
            ", unsigned long long* fault"
        } else {
            ""
        };
        let mut source = String::new();
        writeln!(
            source,
            "// Frozen PCU numerical requirements: {:?}",
            self.numerical_requirements
        )
        .expect("String formatting cannot fail");
        writeln!(source, r#"extern "C" __global__ void fusion_kernel(
    const {bits}* input, const {bits}* upstream, {bits}* output{status_arg}) {{
    const unsigned long long id = static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (id >= {}ull) return;
    const {bits} x = input[id];
    const {bits} dy = upstream[id];"#, self.count).expect("String formatting cannot fail");
        if self.checked {
            writeln!(
                source,
                r"    // PCU finite-input rule checks both operands, including the inactive branch.
    if ((x & {magnitude}) > {maximum} || (dy & {magnitude}) > {maximum}) {{
        atomicMin(fault, (id << 3u) | 5ull); return;
    }}"
            )
            .expect("String formatting cannot fail");
        }
        // Finiteness has already been proved for checked selection. E4M3FN's
        // all-ones exponent includes finite values, so exponent-only NaN tests are invalid.
        writeln!(
            source,
            "    const bool positive = (x & {sign}) == 0ull && (x & {magnitude}) != 0ull"
        )
        .expect("String formatting cannot fail");
        if !self.checked {
            // Preserve the explicit F32/F64 native contract: NaNs are unordered,
            // positive infinity selects upstream, and no finite scan/status is added.
            writeln!(source, "        && !((x & {exponent}) == {exponent} && (x & ~({sign} | {exponent})) != 0ull)")
                .expect("String formatting cannot fail");
        }
        writeln!(
            source,
            "        ;\n    const {bits} selected = positive ? dy : static_cast<{bits}>(0ull);"
        )
        .expect("String formatting cannot fail");
        if self.checked && self.policy == PcuFloatUnderflowPolicy::RejectSubnormalResult {
            writeln!(
                source,
                r"    if ((selected & {magnitude}) < {min_normal} && (selected & {magnitude}) != 0ull) {{
        atomicMin(fault, (id << 3u) | 4ull); return;
    }}"
            )
            .expect("String formatting cannot fail");
        }
        source.push_str("    output[id] = selected;\n}\n");
        source
    }

    fn fault(
        self,
        value: ValueId,
        fault: PcuExecutionFault,
    ) -> Result<TensorError, RocmTensorExecutionError> {
        if !self.checked
            || fault.recovered
            || fault.invocation_id >= u64::from(self.count)
            || !matches!(
                fault.kind,
                PcuExecutionFaultKind::InvalidFloatingOperand
                    | PcuExecutionFaultKind::ArithmeticUnderflow
            )
        {
            return Err(RocmTensorExecutionError::InvalidPlan(value));
        }
        Ok(TensorError::ArithmeticFault {
            value,
            element_index: usize::try_from(fault.invocation_id)
                .map_err(|_| RocmTensorExecutionError::SizeOverflow)?,
            kind: fault.kind,
        })
    }
}

pub(super) fn assess(graph: &Graph, node: NodeDescriptor<'_>) -> TensorOperationSupport {
    let result = (|| {
        if graph.node(node.value).ok() != Some(node) {
            return Err(TensorUnsupportedReason::Operation);
        }
        let profile = Profile::from_node(node)?;
        let OpDescriptor::ReluBackward { input, upstream } = node.op else {
            return Err(TensorUnsupportedReason::Operation);
        };
        for operand in [input, upstream] {
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
        Ok(profile)
    })();
    match result {
        Ok(profile) => TensorOperationSupport::Supported {
            route: if profile.checked {
                TensorExecutionRoute::Synthesized
            } else {
                TensorExecutionRoute::Native
            },
            workspace_bytes: Some(0),
        },
        Err(reason) => TensorOperationSupport::Unsupported { reason },
    }
}

use super::RocmTensorExecutionError;

impl super::RocmTensorAssessor<'_> {
    pub(super) fn execute_admitted_relu_backward(
        &self,
        node: NodeDescriptor<'_>,
        input: &crate::RocmMemoryResource,
        upstream: &crate::RocmMemoryResource,
        output: &crate::RocmMemoryResource,

        mut status: Option<&mut super::owned_scratch::Status>,
    ) -> Result<(), RocmTensorExecutionError> {
        use fusion_pcu::PcuOwnedDispatchMemorySession;
        use fusion_pcu::PcuOwnedCompletion;
        let profile =
            Profile::from_node(node).map_err(|reason| RocmTensorExecutionError::Unsupported {
                value: node.value,
                reason,
            })?;
        self.ensure_relu_backward_cached(profile)?;
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
                    .map_err(RocmTensorExecutionError::Backend)
            })
            .collect::<Result<smallvec::SmallVec<[_; 3]>, _>>()?;
        let mut completion = {
            let cache = self.state().add_dispatches.borrow();
            let prepared = cache
                .iter()
                .find(|(key, _)| *key == super::TensorDispatchCacheKey::ReluBackward(profile))
                .map(|(_, prepared)| prepared)
                .ok_or(RocmTensorExecutionError::InvalidPlan(node.value))?;
            super::owned_scratch::submit(prepared, &bindings, status.as_deref_mut())?
        };
        let outcome = completion
            .wait()
            .map_err(RocmTensorExecutionError::Completion)?;
        if let Some(status) = status {
            status.observe(outcome);
        }
        match outcome {
            fusion_pcu::PcuCompletionOutcome::Succeeded => Ok(()),
            fusion_pcu::PcuCompletionOutcome::Failed => {
                Err(RocmTensorExecutionError::FailedCompletion)
            }
            fusion_pcu::PcuCompletionOutcome::Fault(fault) => {
                Err(profile.fault(node.value, fault)?.into())
            }
        }
    }
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;

impl super::RocmTensorAssessor<'_> {
    pub(super) fn ensure_relu_backward_cached(
        &self,
        profile: Profile,
    ) -> Result<super::TensorDispatchCacheAdmission, RocmTensorExecutionError> {
        let key = super::TensorDispatchCacheKey::ReluBackward(profile);
        let mut cache = self.state().add_dispatches.borrow_mut();
        if cache.iter().any(|(existing, _)| *existing == key) {
            return Ok(super::TensorDispatchCacheAdmission::default());
        }
        let prepared = self
            .session
            .prepare_relu_backward_dispatch(profile, &self.state().stream)
            .map_err(RocmTensorExecutionError::Backend)?;
        let evicted = cache.len() == super::ADD_DISPATCH_CACHE_CAPACITY;
        if evicted {
            cache.pop_front();
        }
        cache.push_back((key, prepared.into()));
        Ok(super::TensorDispatchCacheAdmission {
            compiled: true,
            evicted,
        })
    }
}

/// Generate the exact admitted derivative kernel for an independent native control.
///
/// ABI: three scalar storage pointers; checked profiles append a u64 status pointer.
/// Initialize checked status to `u64::MAX`, inspect after terminal completion, and discard
/// all output on fault. Native profiles have no status argument or numerical classifier.
///
/// # Errors
/// Rejects unsupported policies, descriptor provenance and dense operand extents.
pub fn lower_relu_backward_to_hip_source(
    graph: &Graph,
    value: ValueId,
) -> Result<String, RocmTensorExecutionError> {
    let node = graph.node(value)?;
    if let TensorOperationSupport::Unsupported { reason } = assess(graph, node) {
        return Err(RocmTensorExecutionError::Unsupported { value, reason });
    }
    Ok(Profile::from_node(node)
        .map_err(|reason| RocmTensorExecutionError::Unsupported { value, reason })?
        .source())
}
