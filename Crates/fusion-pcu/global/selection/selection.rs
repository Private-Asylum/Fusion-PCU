//! Consumer-owned invocation ranking, evaluated only during cold preparation.

/// Concrete invocation context available to a consumer's cold device scorer.
///
/// Scores affect preference, never capability admission. Each selected backend still validates
/// the complete kernel before execution. Resident borrows pin their existing session and bypass
/// ranking. Tensor-region selection and cross-provider transitions are separate planner work.
pub struct PcuInvocationCandidate<'a> {
    pub device: crate::PcuDeviceDescriptor<'a>,
    /// Physical capacity, when directly reported; unknown is distinct from zero.
    pub total_memory_bytes: Option<u64>,
    pub facts: crate::PcuDeviceFacts,
    /// Full source-lowered work, including types, accesses, logical extent and instruction policy.
    pub kernel: &'a crate::PcuDispatchKernelIr<'a>,
    /// Global defaults; explicit instruction/function policies remain in the kernel itself.
    pub policy_defaults: crate::PcuImplementationRequirements,
}

/// Optional operation-aware consumer scoring, invoked once per candidate on a cold miss.
///
/// Higher scores rank first. A scorer may inspect workload, hardware observations and its own
/// previously calibrated evidence. It must not infer interop, support or correctness from a
/// score. Prepared warm calls never invoke it or query the physical facts again.
pub type PcuInvocationScorer = fn(&PcuInvocationCandidate<'_>) -> i128;

#[cfg(any(
    feature = "rocm",
    feature = "cuda",
    feature = "metal",
    feature = "vulkan",
    feature = "cpu",
    feature = "mlx"
))]
pub(super) fn score_candidate<E>(
    policy: super::PcuExecutionPolicy,
    descriptor: crate::PcuDeviceDescriptor<'_>,
    memory: Option<u64>,
    kernel: Option<&crate::PcuDispatchKernelIr<'_>>,
    facts: impl FnOnce() -> Result<crate::PcuDeviceFacts, E>,
) -> Result<i128, E> {
    if let (Some(score), Some(kernel)) = (policy.score_invocation, kernel) {
        let candidate = PcuInvocationCandidate {
            device: descriptor,
            total_memory_bytes: memory,
            facts: facts()?,
            kernel,
            policy_defaults: crate::PcuImplementationRequirements {
                numerical_mode: policy.numerical_mode,
                numerical_options: policy.numerical_options,
                float_underflow: policy.float_underflow,
                range_policy: policy.range_policy,
            },
        };
        return Ok(score(&candidate));
    }
    Ok((policy.score_device)(&descriptor, memory.unwrap_or(0)))
}

#[cfg(all(
    test,
    any(
        feature = "rocm",
        feature = "cuda",
        feature = "metal",
        feature = "vulkan",
        feature = "cpu",
        feature = "mlx"
    )
))]
mod tests;
