//! Structural admission checks for PCU submissions.

use crate::contract::PcuKernelIrContract;
use super::submission::PcuDispatchSubmission;
use crate::contract::{
    PcuBinding,
    PcuBindingRef,
    PcuDispatchOp,
    PcuError,
    PcuInvocationBindings,
    PcuInvocationTarget,
    PcuInvocationParameters,
    PcuInvocationTopology,
    PcuKernel,
    PcuKernelSignature,
    PcuPort,
};
use crate::contract::PcuBaseContract;

pub(super) fn validate_direct_kernel_support<B: PcuBaseContract + ?Sized>(
    backend: &B,
    kernel: PcuKernel<'_>,
) -> Result<(), PcuError> {
    let kernel_supported = backend.any_executor_supports_kernel_direct(kernel);

    if let PcuKernel::Dispatch(dispatch) = kernel
        && !kernel_supported
    {
        let any_structural = backend.executors().iter().copied().any(|descriptor| {
            descriptor
                .support
                .supports_dispatch_direct_structure(dispatch)
        });
        if any_structural {
            let any_typed = backend.executors().iter().copied().any(|descriptor| {
                descriptor
                    .support
                    .supports_dispatch_direct_structure(dispatch)
                    && descriptor
                        .support
                        .supports_value_types_direct(dispatch.required_type_support())
            });
            if any_typed {
                return Err(PcuError::unsupported_feature_support());
            }
            return Err(PcuError::unsupported_type_support());
        }
    }

    if kernel_supported {
        return Ok(());
    }
    Err(PcuError::unsupported())
}

/// Checks that invocation parameters match the declared slots and types.
///
/// # Errors
///
/// Returns `Invalid` for missing, extra, duplicate, or mistyped parameters.
pub fn validate_parameters(
    signature: PcuKernelSignature<'_>,
    parameters: PcuInvocationParameters<'_>,
) -> Result<(), PcuError> {
    if parameters.validate_against(signature.parameters) {
        Ok(())
    } else {
        Err(PcuError::invalid())
    }
}

/// Checks that each supplied binding target exists and appears only once.
///
/// This is a partial structural check: backends must also verify required targets, access,
/// layout, size, and residency before execution.
///
/// # Errors
///
/// Returns `Invalid` for duplicate or unknown targets.
pub fn validate_invocation_bindings(
    signature: PcuKernelSignature<'_>,
    bindings: PcuInvocationBindings<'_>,
) -> Result<(), PcuError> {
    for (index, binding) in bindings.bindings.iter().enumerate() {
        if bindings.bindings[..index]
            .iter()
            .any(|existing| existing.target == binding.target)
        {
            return Err(PcuError::invalid());
        }

        let target_exists = match binding.target {
            PcuInvocationTarget::Binding(reference) => {
                binding_exists(signature.bindings, reference)
            }
            PcuInvocationTarget::Port(name) => port_exists(signature.ports, name),
        };

        if !target_exists {
            return Err(PcuError::invalid());
        }
    }

    Ok(())
}

/// Checks that a dispatch submission's invocation count matches its logical shape.
///
/// # Errors
///
/// Returns `Invalid` for an unsupported topology, zero/overflowed extent, or mismatched count.
pub fn validate_dispatch_submission(submission: PcuDispatchSubmission<'_>) -> Result<(), PcuError> {
    for (index, binding) in submission.kernel.bindings.iter().enumerate() {
        if submission.kernel.bindings[..index]
            .iter()
            .any(|prior| prior.reference() == binding.reference())
        {
            // A binding address names exactly one declared resource. Accepting duplicates makes
            // lookup-based admission depend on declaration order, so the requested access/type
            // contract could disagree with the resource a backend actually selects.
            return Err(PcuError::invalid());
        }
    }
    if submission
        .kernel
        .ops
        .iter()
        .any(|op| matches!(op, PcuDispatchOp::GridStrideLoop { extent: 0, .. }))
    {
        return Err(PcuError::invalid());
    }
    let PcuInvocationTopology::Indexed { logical_shape } =
        submission.kernel.signature().invocation.topology
    else {
        return Err(PcuError::invalid());
    };

    let expected_invocations = logical_shape
        .into_iter()
        .try_fold(1_u64, |product, axis| product.checked_mul(u64::from(axis)))
        .ok_or_else(PcuError::invalid)?;

    if expected_invocations == 0
        || expected_invocations != u64::from(submission.shape.invocation_count().get())
    {
        return Err(PcuError::invalid());
    }

    Ok(())
}

fn binding_exists(bindings: &[PcuBinding<'_>], reference: PcuBindingRef) -> bool {
    bindings
        .iter()
        .copied()
        .any(|binding| binding.reference() == reference)
}

fn port_exists(ports: &[PcuPort<'_>], name: &str) -> bool {
    ports.iter().any(|port| port.name == Some(name))
}
