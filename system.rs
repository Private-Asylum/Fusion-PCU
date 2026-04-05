//! Generic fusion-pcu wrapper over the selected PCU executor backend.

use fusion_pal::sys::pcu::{
    PcuBaseContract,
    PcuControlContract,
    PcuError,
    PcuExecutorClaim,
    PcuExecutorClass,
    PcuExecutorDescriptor,
    PcuExecutorId,
    PcuSupport,
    PlatformPcu,
    system_pcu as pal_system_pcu,
};

#[cfg(all(target_os = "none", feature = "sys-cortex-m"))]
use fusion_pal::sys::soc::cortex_m::hal::soc::pio::PioBase;
#[cfg(all(target_os = "none", feature = "sys-cortex-m"))]
use super::PcuStreamPattern;
use super::{
    PcuBackendKind,
    PcuDispatchPlan,
    PcuDispatchPolicy,
    PcuInvocation,
    PcuInvocationBindings,
    PcuInvocationDescriptor,
    PcuInvocationHandle,
    PcuInvocationParameters,
    PcuPreparedKernel,
    PcuStreamKernelIr,
};

/// fusion-pcu wrapper around the selected generic PCU executor backend.
#[derive(Debug, Clone, Copy)]
pub struct PcuSystem {
    inner: PlatformPcu,
}

impl PcuSystem {
    /// Creates a wrapper for the selected generic coprocessor backend.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            inner: pal_system_pcu(),
        }
    }

    /// Reports the truthful generic PCU executor surface for the selected backend.
    #[must_use]
    pub fn support(&self) -> PcuSupport {
        PcuBaseContract::support(&self.inner)
    }

    /// Returns the surfaced generic PCU executors.
    #[must_use]
    pub fn executors(&self) -> &'static [PcuExecutorDescriptor] {
        PcuBaseContract::executors(&self.inner)
    }

    /// Returns one surfaced executor descriptor by stable id.
    #[must_use]
    pub fn executor(&self, executor: PcuExecutorId) -> Option<&'static PcuExecutorDescriptor> {
        self.executors()
            .iter()
            .find(|descriptor| descriptor.id == executor)
    }

    /// Back-compat alias while higher layers stop saying “device” when they mean “executor.”
    #[must_use]
    pub fn devices(&self) -> &'static [PcuExecutorDescriptor] {
        self.executors()
    }

    /// Claims one surfaced generic PCU executor.
    ///
    /// # Errors
    ///
    /// Returns any honest backend claim failure.
    pub fn claim_executor(&self, executor: PcuExecutorId) -> Result<PcuExecutorClaim, PcuError> {
        PcuControlContract::claim_executor(&self.inner, executor)
    }

    /// Back-compat alias while higher layers stop saying “device” when they mean “executor.”
    ///
    /// # Errors
    ///
    /// Returns any honest backend claim failure.
    pub fn claim_device(&self, device: PcuExecutorId) -> Result<PcuExecutorClaim, PcuError> {
        self.claim_executor(device)
    }

    /// Releases one previously claimed generic PCU executor.
    ///
    /// # Errors
    ///
    /// Returns any honest backend release failure.
    pub fn release_executor(&self, claim: PcuExecutorClaim) -> Result<(), PcuError> {
        PcuControlContract::release_executor(&self.inner, claim)
    }

    /// Back-compat alias while higher layers stop saying “device” when they mean “executor.”
    ///
    /// # Errors
    ///
    /// Returns any honest backend release failure.
    pub fn release_device(&self, claim: PcuExecutorClaim) -> Result<(), PcuError> {
        self.release_executor(claim)
    }

    fn stream_kernel_supports_pio(&self, kernel: PcuStreamKernelIr<'_>) -> bool {
        #[cfg(all(target_os = "none", feature = "sys-cortex-m"))]
        {
            use fusion_pal::sys::soc::cortex_m::hal::soc::pio::system_pio;

            if !kernel.simple_transform_patterns_are_valid()
                || !kernel.parameters.is_empty()
                || kernel.simple_transform_type() != Some(super::PcuStreamValueType::U32)
                || kernel.patterns.len() != 1
            {
                return false;
            }
            if !matches!(
                kernel.patterns[0],
                PcuStreamPattern::BitReverse
                    | PcuStreamPattern::BitInvert
                    | PcuStreamPattern::Increment
                    | PcuStreamPattern::ShiftLeft { .. }
                    | PcuStreamPattern::ShiftRight { .. }
                    | PcuStreamPattern::ExtractBits { .. }
                    | PcuStreamPattern::MaskLower { .. }
                    | PcuStreamPattern::ByteSwap32
            ) {
                return false;
            }

            let support = system_pio().support();
            return support.engine_count > 0;
        }
        #[cfg(not(all(target_os = "none", feature = "sys-cortex-m")))]
        {
            let _ = kernel;
            false
        }
    }

    fn has_cpu_executor(&self) -> bool {
        self.executors()
            .iter()
            .any(|descriptor| descriptor.class == PcuExecutorClass::Cpu)
    }

    fn select_pio_executor(&self) -> Option<PcuExecutorId> {
        self.executors()
            .iter()
            .find(|descriptor| descriptor.class == PcuExecutorClass::Io)
            .map(|descriptor| descriptor.id)
    }

    /// Returns the exact dispatch policy required to target one surfaced executor directly.
    ///
    /// # Errors
    ///
    /// Returns `Invalid` when the executor is unknown and `Unsupported` when the executor class
    /// does not yet map to one executable local backend.
    pub fn dispatch_policy_for_executor(
        &self,
        executor: PcuExecutorId,
    ) -> Result<PcuDispatchPolicy, PcuError> {
        let descriptor = self.executor(executor).ok_or_else(PcuError::invalid)?;
        match descriptor.class {
            PcuExecutorClass::Cpu => Ok(PcuDispatchPolicy::CpuOnly),
            PcuExecutorClass::Io => Ok(PcuDispatchPolicy::Require(PcuBackendKind::CortexMPio)),
            _ => Err(PcuError::unsupported()),
        }
    }

    fn select_backend(
        &self,
        invocation: PcuInvocationDescriptor<'_>,
    ) -> Result<PcuBackendKind, PcuError> {
        let kernel = invocation.kernel;
        if let Some(stream) = kernel.as_stream()
            && !stream.simple_transform_patterns_are_valid()
        {
            return Err(PcuError::invalid());
        }
        let pio_supported = kernel
            .as_stream()
            .is_some_and(|stream| self.stream_kernel_supports_pio(stream))
            && self.select_pio_executor().is_some();
        let cpu_supported = kernel.as_stream().is_some() && self.has_cpu_executor();

        match invocation.policy {
            PcuDispatchPolicy::CpuOnly => {
                if cpu_supported {
                    Ok(PcuBackendKind::Cpu)
                } else {
                    Err(PcuError::unsupported())
                }
            }
            PcuDispatchPolicy::Require(PcuBackendKind::Cpu) => {
                if cpu_supported {
                    Ok(PcuBackendKind::Cpu)
                } else {
                    Err(PcuError::unsupported())
                }
            }
            PcuDispatchPolicy::Require(PcuBackendKind::CortexMPio) => {
                if pio_supported {
                    Ok(PcuBackendKind::CortexMPio)
                } else {
                    Err(PcuError::unsupported())
                }
            }
            PcuDispatchPolicy::Prefer(PcuBackendKind::Cpu) => {
                if cpu_supported {
                    Ok(PcuBackendKind::Cpu)
                } else if pio_supported {
                    Ok(PcuBackendKind::CortexMPio)
                } else {
                    Err(PcuError::unsupported())
                }
            }
            PcuDispatchPolicy::Prefer(PcuBackendKind::CortexMPio) => {
                if pio_supported {
                    Ok(PcuBackendKind::CortexMPio)
                } else if cpu_supported {
                    Ok(PcuBackendKind::Cpu)
                } else {
                    Err(PcuError::unsupported())
                }
            }
            PcuDispatchPolicy::PreferHardwareAllowCpuFallback => {
                if pio_supported {
                    Ok(PcuBackendKind::CortexMPio)
                } else if cpu_supported {
                    Ok(PcuBackendKind::Cpu)
                } else {
                    Err(PcuError::unsupported())
                }
            }
        }
    }

    /// Plans one kernel invocation against the available backends and supplied policy.
    ///
    /// # Errors
    ///
    /// Returns any honest capability or policy-selection failure.
    pub fn plan<'a>(
        &self,
        invocation: PcuInvocationDescriptor<'a>,
    ) -> Result<PcuDispatchPlan<'a>, PcuError> {
        let backend = self.select_backend(invocation)?;
        Ok(PcuDispatchPlan {
            kernel: invocation.kernel,
            shape: invocation.shape,
            backend,
            executor: match backend {
                PcuBackendKind::Cpu => None,
                PcuBackendKind::CortexMPio => Some(
                    self.select_pio_executor()
                        .ok_or_else(PcuError::unsupported)?,
                ),
            },
        })
    }

    /// Plans one kernel invocation against one exact surfaced executor.
    ///
    /// # Errors
    ///
    /// Returns any honest executor-selection, capability, or policy-selection failure.
    pub fn plan_for_executor<'a>(
        &self,
        invocation: PcuInvocation<'a>,
        executor: PcuExecutorId,
    ) -> Result<PcuDispatchPlan<'a>, PcuError> {
        let mut plan = self.plan(PcuInvocationDescriptor {
            kernel: invocation.kernel,
            shape: invocation.shape,
            policy: self.dispatch_policy_for_executor(executor)?,
        })?;
        if matches!(plan.backend, PcuBackendKind::CortexMPio) {
            plan.executor = Some(executor);
        }
        Ok(plan)
    }

    /// Prepares one already-planned kernel for execution.
    ///
    /// # Errors
    ///
    /// Returns any honest lowering or preparation failure.
    pub fn prepare<'a>(
        &self,
        plan: PcuDispatchPlan<'a>,
    ) -> Result<PcuPreparedKernel<'a>, PcuError> {
        let _ = self;
        plan.prepare()
    }

    /// Plans, prepares, and dispatches one kernel invocation in one convenience call.
    ///
    /// # Errors
    ///
    /// Returns any honest planning, preparation, or dispatch failure.
    pub fn dispatch(
        &self,
        invocation: PcuInvocationDescriptor<'_>,
        bindings: PcuInvocationBindings<'_>,
    ) -> Result<impl PcuInvocationHandle, PcuError> {
        self.dispatch_with_parameters(invocation, bindings, PcuInvocationParameters::empty())
    }

    /// Plans, prepares, and dispatches one kernel invocation with explicit runtime parameters.
    ///
    /// # Errors
    ///
    /// Returns any honest planning, preparation, or dispatch failure.
    pub fn dispatch_with_parameters(
        &self,
        invocation: PcuInvocationDescriptor<'_>,
        bindings: PcuInvocationBindings<'_>,
        parameters: PcuInvocationParameters<'_>,
    ) -> Result<impl PcuInvocationHandle, PcuError> {
        let prepared = self.prepare(self.plan(invocation)?)?;
        prepared.dispatch_with_parameters(bindings, parameters)
    }

    /// Plans, prepares, and dispatches one kernel invocation against one exact surfaced executor.
    ///
    /// # Errors
    ///
    /// Returns any honest executor-selection, planning, preparation, or dispatch failure.
    pub fn dispatch_on_executor(
        &self,
        invocation: PcuInvocation<'_>,
        bindings: PcuInvocationBindings<'_>,
        executor: PcuExecutorId,
    ) -> Result<impl PcuInvocationHandle, PcuError> {
        self.dispatch_on_executor_with_parameters(
            invocation,
            bindings,
            PcuInvocationParameters::empty(),
            executor,
        )
    }

    /// Plans, prepares, and dispatches one kernel invocation against one exact surfaced executor
    /// with explicit runtime parameters.
    ///
    /// # Errors
    ///
    /// Returns any honest executor-selection, planning, preparation, or dispatch failure.
    pub fn dispatch_on_executor_with_parameters(
        &self,
        invocation: PcuInvocation<'_>,
        bindings: PcuInvocationBindings<'_>,
        parameters: PcuInvocationParameters<'_>,
        executor: PcuExecutorId,
    ) -> Result<impl PcuInvocationHandle, PcuError> {
        let prepared = self.prepare(self.plan_for_executor(invocation, executor)?)?;
        prepared.dispatch_with_parameters(bindings, parameters)
    }
}

impl Default for PcuSystem {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use core::num::NonZeroU32;

    use super::PcuSystem;
    use crate::{
        PcuByteStreamBindings,
        PcuDispatchPolicy,
        PcuExecutorClass,
        PcuInvocation,
        PcuInvocationBindings,
        PcuInvocationDescriptor,
        PcuInvocationHandle,
        PcuInvocationShape,
        PcuKernel,
        PcuKernelId,
        PcuPort,
        PcuStreamCapabilities,
        PcuStreamKernelIr,
        PcuStreamPattern,
        PcuStreamValueType,
        PcuWordStreamBindings,
    };

    #[test]
    fn generic_system_support_matches_device_inventory() {
        let system = PcuSystem::new();
        let support = system.support();
        assert_eq!(
            usize::from(support.executor_count),
            system.executors().len()
        );
    }

    #[test]
    fn host_stream_planning_falls_back_to_cpu() {
        let ports = [
            PcuPort::stream_input(Some("input"), PcuStreamValueType::U32.as_value_type()),
            PcuPort::stream_output(Some("output"), PcuStreamValueType::U32.as_value_type()),
        ];
        let patterns = [PcuStreamPattern::BitReverse];
        let kernel = PcuKernel::Stream(PcuStreamKernelIr {
            id: PcuKernelId(41),
            entry_point: "bit_reverse",
            bindings: &[],
            ports: &ports,
            parameters: &[],
            patterns: &patterns,
            capabilities: PcuStreamCapabilities::FIFO_INPUT
                | PcuStreamCapabilities::FIFO_OUTPUT
                | PcuStreamCapabilities::BIT_REVERSE,
        });
        let system = PcuSystem::new();
        let plan = system
            .plan(PcuInvocationDescriptor {
                kernel: &kernel,
                shape: PcuInvocationShape::threads(NonZeroU32::new(1).unwrap()),
                policy: PcuDispatchPolicy::PreferHardwareAllowCpuFallback,
            })
            .expect("host fallback planning should succeed");

        assert_eq!(plan.backend(), crate::PcuBackendKind::Cpu);
        assert_eq!(plan.device(), None);
    }

    #[test]
    fn host_prefer_cortex_m_pio_falls_back_to_cpu() {
        let ports = [
            PcuPort::stream_input(Some("input"), PcuStreamValueType::U32.as_value_type()),
            PcuPort::stream_output(Some("output"), PcuStreamValueType::U32.as_value_type()),
        ];
        let patterns = [PcuStreamPattern::BitReverse];
        let kernel = PcuKernel::Stream(PcuStreamKernelIr {
            id: PcuKernelId(46),
            entry_point: "bit_reverse",
            bindings: &[],
            ports: &ports,
            parameters: &[],
            patterns: &patterns,
            capabilities: PcuStreamCapabilities::FIFO_INPUT
                | PcuStreamCapabilities::FIFO_OUTPUT
                | PcuStreamCapabilities::BIT_REVERSE,
        });
        let system = PcuSystem::new();
        let plan = system
            .plan(PcuInvocationDescriptor {
                kernel: &kernel,
                shape: PcuInvocationShape::threads(NonZeroU32::new(1).unwrap()),
                policy: PcuDispatchPolicy::Prefer(crate::PcuBackendKind::CortexMPio),
            })
            .expect("prefer-pio planning should fall back to CPU on host");

        assert_eq!(plan.backend(), crate::PcuBackendKind::Cpu);
        assert_eq!(plan.device(), None);
    }

    #[test]
    fn host_stream_dispatch_executes_cpu_fallback() {
        let ports = [
            PcuPort::stream_input(Some("input"), PcuStreamValueType::U32.as_value_type()),
            PcuPort::stream_output(Some("output"), PcuStreamValueType::U32.as_value_type()),
        ];
        let patterns = [PcuStreamPattern::BitReverse];
        let kernel = PcuKernel::Stream(PcuStreamKernelIr {
            id: PcuKernelId(42),
            entry_point: "bit_reverse",
            bindings: &[],
            ports: &ports,
            parameters: &[],
            patterns: &patterns,
            capabilities: PcuStreamCapabilities::FIFO_INPUT
                | PcuStreamCapabilities::FIFO_OUTPUT
                | PcuStreamCapabilities::BIT_REVERSE,
        });
        let system = PcuSystem::new();
        let input = [0x0000_00f0, 0x8000_0001];
        let mut output = [0u32; 2];
        let handle = system
            .dispatch(
                PcuInvocationDescriptor {
                    kernel: &kernel,
                    shape: PcuInvocationShape::threads(NonZeroU32::new(1).unwrap()),
                    policy: PcuDispatchPolicy::PreferHardwareAllowCpuFallback,
                },
                PcuInvocationBindings::StreamWords(PcuWordStreamBindings {
                    input: &input,
                    output: &mut output,
                }),
            )
            .expect("CPU fallback dispatch should succeed");

        assert_eq!(handle.backend(), crate::PcuBackendKind::Cpu);
        handle
            .wait()
            .expect("completed CPU fallback should succeed");
        assert_eq!(output, [0x0f00_0000, 0x8000_0001]);
    }

    #[test]
    fn host_byte_stream_dispatch_executes_cpu_fallback() {
        let ports = [
            PcuPort::stream_input(Some("input"), PcuStreamValueType::U8.as_value_type()),
            PcuPort::stream_output(Some("output"), PcuStreamValueType::U8.as_value_type()),
        ];
        let patterns = [PcuStreamPattern::BitInvert];
        let kernel = PcuKernel::Stream(PcuStreamKernelIr {
            id: PcuKernelId(43),
            entry_point: "bit_invert",
            bindings: &[],
            ports: &ports,
            parameters: &[],
            patterns: &patterns,
            capabilities: PcuStreamCapabilities::FIFO_INPUT
                | PcuStreamCapabilities::FIFO_OUTPUT
                | PcuStreamCapabilities::BIT_INVERT,
        });
        let system = PcuSystem::new();
        let input = [0x0fu8, 0xa0];
        let mut output = [0u8; 2];
        let handle = system
            .dispatch(
                PcuInvocationDescriptor {
                    kernel: &kernel,
                    shape: PcuInvocationShape::threads(NonZeroU32::new(2).unwrap()),
                    policy: PcuDispatchPolicy::PreferHardwareAllowCpuFallback,
                },
                PcuInvocationBindings::StreamBytes(PcuByteStreamBindings {
                    input: &input,
                    output: &mut output,
                }),
            )
            .expect("byte CPU fallback dispatch should succeed");

        assert_eq!(handle.backend(), crate::PcuBackendKind::Cpu);
        handle
            .wait()
            .expect("completed byte CPU fallback should succeed");
        assert_eq!(output, [!0x0f, !0xa0]);
    }

    #[test]
    fn host_stream_planning_rejects_invalid_specialization() {
        let ports = [
            PcuPort::stream_input(Some("input"), PcuStreamValueType::U8.as_value_type()),
            PcuPort::stream_output(Some("output"), PcuStreamValueType::U8.as_value_type()),
        ];
        let patterns = [PcuStreamPattern::ByteSwap32];
        let kernel = PcuKernel::Stream(PcuStreamKernelIr {
            id: PcuKernelId(44),
            entry_point: "illegal_bswap32_u8",
            bindings: &[],
            ports: &ports,
            parameters: &[],
            patterns: &patterns,
            capabilities: PcuStreamCapabilities::FIFO_INPUT
                | PcuStreamCapabilities::FIFO_OUTPUT
                | PcuStreamCapabilities::BYTE_SWAP32,
        });
        let system = PcuSystem::new();
        let error = system
            .plan(PcuInvocationDescriptor {
                kernel: &kernel,
                shape: PcuInvocationShape::threads(NonZeroU32::new(1).unwrap()),
                policy: PcuDispatchPolicy::PreferHardwareAllowCpuFallback,
            })
            .expect_err("invalid pattern/type pairing should be rejected at planning time");

        assert_eq!(error.kind(), crate::PcuError::invalid().kind());
    }

    #[test]
    fn host_cpu_executor_maps_to_direct_cpu_policy() {
        let system = PcuSystem::new();
        let cpu = system
            .executors()
            .iter()
            .find(|descriptor| descriptor.class == PcuExecutorClass::Cpu)
            .expect("host should surface one cpu executor");

        assert_eq!(
            system
                .dispatch_policy_for_executor(cpu.id)
                .expect("cpu executor should map to one direct policy"),
            PcuDispatchPolicy::CpuOnly
        );
    }

    #[test]
    fn host_plan_for_executor_targets_cpu_directly() {
        let ports = [
            PcuPort::stream_input(Some("input"), PcuStreamValueType::U32.as_value_type()),
            PcuPort::stream_output(Some("output"), PcuStreamValueType::U32.as_value_type()),
        ];
        let patterns = [PcuStreamPattern::BitReverse];
        let kernel = PcuKernel::Stream(PcuStreamKernelIr {
            id: PcuKernelId(45),
            entry_point: "bit_reverse",
            bindings: &[],
            ports: &ports,
            parameters: &[],
            patterns: &patterns,
            capabilities: PcuStreamCapabilities::FIFO_INPUT
                | PcuStreamCapabilities::FIFO_OUTPUT
                | PcuStreamCapabilities::BIT_REVERSE,
        });
        let system = PcuSystem::new();
        let cpu = system
            .executors()
            .iter()
            .find(|descriptor| descriptor.class == PcuExecutorClass::Cpu)
            .expect("host should surface one cpu executor");

        let plan = system
            .plan_for_executor(
                PcuInvocation {
                    kernel: &kernel,
                    shape: PcuInvocationShape::threads(NonZeroU32::new(1).unwrap()),
                },
                cpu.id,
            )
            .expect("host cpu executor planning should succeed");

        assert_eq!(plan.backend(), crate::PcuBackendKind::Cpu);
        assert_eq!(plan.executor(), None);
    }
}
