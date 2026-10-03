//! Explicit mixed-host/native calls reuse exact existing map admission and stage every output.
#[path = "schema/schema.rs"]
mod schema;
#[rustfmt::skip]
use fusion_pcu::{
    PcuDispatchKernelIr,
    PcuHostKernelBackend,
};
#[rustfmt::skip]
use crate::{
    arguments::Kind,
    ffi::{VulkanMixedInput,VulkanMixedOutput,VulkanWriteState},
    PcuVulkanArgument,
    PcuVulkanBackend,
    PcuVulkanError,
    PcuVulkanPreparedHost,
    PcuVulkanPreparedMemoryRealizations,
};
#[rustfmt::skip]
use schema::{
    Role,
    Schema,
};

/// Frozen same-session mixed executable with private kernel outputs and retained public commits.
///
/// No caller handle is forged and no host pointer is exposed to Vulkan. A complete fatal scan
/// precedes every public write. Recoverable faults publish the useful prefix and return Err;
/// unknown native completion quarantines the complete retained logical-device root.
/// Ordinary host-only preparation remains the separate original ABI.
pub struct PcuVulkanPreparedMixed {
    plan: PcuVulkanPreparedHost,
    schema: Schema,
    state: VulkanWriteState,
}
impl PcuVulkanBackend {
    /// Cold-prepares existing exact scalar maps for actual host/native borrowed argument storage.
    /// Retains private upload/output/status buffers, native code and exact-byte commit resources.
    ///
    /// # Errors
    /// Returns existing profile/admission/device errors or native retained-resource failures.
    pub fn prepare_mixed_kernel(
        &self,
        kernel: &PcuDispatchKernelIr<'_>,
    ) -> Result<PcuVulkanPreparedMixed, PcuVulkanError> {
        let mut plan = self.prepare_host_kernel(kernel)?;
        let schema = schema::compile(&plan)?;
        if schema.input_count > 2 || schema.output_count > 2 {
            return Err(PcuVulkanError::InvalidArguments);
        }
        enable(&mut plan)?;
        Ok(PcuVulkanPreparedMixed {
            plan,
            schema,
            state: VulkanWriteState::default(),
        })
    }
}
impl PcuVulkanPreparedMixed {
    #[must_use]
    pub const fn argument_count(&self) -> usize {
        self.schema.bindings.len()
    }
    #[must_use]
    pub const fn last_call_may_have_written(&self) -> bool {
        self.state.may_have_written
    }
    #[must_use]
    pub const fn last_call_completion_uncertain(&self) -> bool {
        self.state.completion_uncertain
    }
    #[must_use]
    pub fn memory_realizations(&self) -> Option<PcuVulkanPreparedMemoryRealizations> {
        self.plan.memory_realizations()
    }
    /// Executes against exact cold declarations with stack-only operand/output projection.
    ///
    /// # Errors
    /// Rejects scalar/access/extent/session/alias mismatches before submission; fatal arithmetic
    /// preserves old public bytes, recovered arithmetic commits then returns Err, and uncertain
    /// native completion marks the retained root inaccessible before returning.
    pub fn call(&mut self, arguments: &mut [PcuVulkanArgument<'_>]) -> Result<(), PcuVulkanError> {
        self.state = VulkanWriteState::default();
        validate(arguments, &self.schema)?;
        let mut inputs = [VulkanMixedInput::Host(&[]); 2];
        let mut outputs = [
            VulkanMixedOutput::Host(&mut []),
            VulkanMixedOutput::Host(&mut []),
        ];
        for argument in arguments.iter_mut() {
            let binding = self
                .schema
                .bindings
                .iter()
                .find(|binding| binding.target == argument.target)
                .ok_or(PcuVulkanError::InvalidArguments)?;
            match (binding.role, &mut argument.kind) {
                (Role::Unused, _) => {}
                (Role::Input(index), Kind::Host(host)) => {
                    inputs[index] = VulkanMixedInput::Host(host.bytes());
                }
                (Role::Input(index), Kind::Read(owner)) => {
                    inputs[index] = VulkanMixedInput::Owned(owner);
                }
                (Role::Output(index), Kind::Host(host)) => {
                    outputs[index] = VulkanMixedOutput::Host(
                        host.bytes_mut().ok_or(PcuVulkanError::InvalidArguments)?,
                    );
                }
                (Role::Output(index), Kind::Write(owner)) => {
                    outputs[index] = VulkanMixedOutput::Owned(owner);
                }
                _ => return Err(PcuVulkanError::InvalidArguments),
            }
        }
        execute(
            &mut self.plan,
            &inputs[..self.schema.input_count],
            &mut outputs[..self.schema.output_count],
            &mut self.state,
        )
    }
}
fn validate(arguments: &[PcuVulkanArgument<'_>], schema: &Schema) -> Result<(), PcuVulkanError> {
    if arguments.len() != schema.bindings.len() {
        return Err(PcuVulkanError::InvalidArguments);
    }
    for binding in &schema.bindings {
        let mut matching = arguments
            .iter()
            .filter(|argument| argument.target() == binding.target);
        let argument = matching.next().ok_or(PcuVulkanError::InvalidArguments)?;
        if matching.next().is_some()
            || argument.scalar() != binding.scalar
            || argument.access() != binding.access
            || argument.byte_len() < binding.bytes
        {
            return Err(PcuVulkanError::InvalidArguments);
        }
    }
    Ok(())
}
fn enable(plan: &mut PcuVulkanPreparedHost) -> Result<(), PcuVulkanError> {
    match plan {
        PcuVulkanPreparedHost::BitMap(plan) => plan.native.enable_mixed(),
        PcuVulkanPreparedHost::Binary(plan) => plan.native.enable_mixed(),
        PcuVulkanPreparedHost::Integer(plan) => plan.native.enable_mixed(),
        PcuVulkanPreparedHost::DivRem(plan) => plan.native.enable_mixed(),
        PcuVulkanPreparedHost::Unary(plan) => plan.native.enable_mixed(),
        PcuVulkanPreparedHost::UnaryRoles(plan) => plan.enable_mixed(),
        PcuVulkanPreparedHost::Conversion(plan) => plan.native.enable_mixed(),
        PcuVulkanPreparedHost::ScalarTransport(plan) => plan.native.enable_mixed(),
        PcuVulkanPreparedHost::Composed(_) | PcuVulkanPreparedHost::OrderedTransport(_) => {
            Err(PcuVulkanError::UnsupportedPreparedProfile)
        }
    }
}
fn execute(
    plan: &mut PcuVulkanPreparedHost,
    inputs: &[VulkanMixedInput<'_>],
    outputs: &mut [VulkanMixedOutput<'_>],
    state: &mut VulkanWriteState,
) -> Result<(), PcuVulkanError> {
    match plan {
        PcuVulkanPreparedHost::BitMap(plan) => plan.native.call_mixed(inputs, outputs, state),
        PcuVulkanPreparedHost::Binary(plan) => plan.native.call_mixed(inputs, outputs, state),
        PcuVulkanPreparedHost::Integer(plan) => plan.native.call_mixed(inputs, outputs, state),
        PcuVulkanPreparedHost::DivRem(plan) => plan.native.call_mixed(inputs, outputs, state),
        PcuVulkanPreparedHost::Unary(plan) => plan.native.call_mixed(inputs, outputs, state),
        PcuVulkanPreparedHost::UnaryRoles(plan) => plan.call_mixed(inputs, outputs, state),
        PcuVulkanPreparedHost::Conversion(plan) => plan.native.call_mixed(inputs, outputs, state),
        PcuVulkanPreparedHost::ScalarTransport(plan) => {
            plan.native.call_mixed(inputs, outputs, state)
        }
        PcuVulkanPreparedHost::Composed(_) | PcuVulkanPreparedHost::OrderedTransport(_) => {
            Err(PcuVulkanError::UnsupportedPreparedProfile)
        }
    }
}
