//! Additive checked division with actual unique input resources and joint output publication.
#[path = "plan/plan.rs"]
mod plan;
pub use plan::MlxCheckedDivRemRolePlan;
#[rustfmt::skip]
use fusion_pcu::{
    PcuBindingAccess,
    PcuBindingRef,
    PcuDispatchKernelIr,
    PcuHostArgument,
    PcuHostDispatchError,
    PcuHostKernelBackend,
    PcuPreparedHostKernel,
    PcuScalarType,
};
#[rustfmt::skip]
use crate::{
    MlxBinaryInput,
    MlxEncodedArray,
    MlxError,
    MlxHostKernelError,
    MlxSession,
};
use super::MlxCheckedDivRemControl;

/// Cloning retains the same actual native session, without opening another device.
#[derive(Clone)]
pub struct MlxCheckedDivRemRoleBackend {
    session: MlxSession,
}
/// Cold role metadata and a retained primitive with private host readback scratch.
pub struct MlxPreparedDivRemRoleHostKernel {
    control: MlxCheckedDivRemControl,
    plan: MlxCheckedDivRemRolePlan,
    input_extents: [usize; 2],
}
impl MlxSession {
    /// Retains this session for the additive actual-input division profile.
    #[must_use]
    pub fn checked_div_rem_role_backend(&self) -> MlxCheckedDivRemRoleBackend {
        MlxCheckedDivRemRoleBackend {
            session: self.clone(),
        }
    }
}
impl PcuHostKernelBackend for MlxCheckedDivRemRoleBackend {
    type Prepared = MlxPreparedDivRemRoleHostKernel;
    type Error = MlxHostKernelError;
    fn prepare_host_kernel(
        &self,
        kernel: &PcuDispatchKernelIr<'_>,
    ) -> Result<Self::Prepared, Self::Error> {
        self.prepare_host_kernel_with_input_extents_internal(kernel, None)
    }
}
impl MlxCheckedDivRemRoleBackend {
    /// Separately freezes each actual unique resident capacity before native primitive priming.
    ///
    /// Input extents follow `MlxCheckedDivRemRolePlan::input_bindings` order. Host inputs retain
    /// their minimum IR span; resident inputs retain their full exact dense owner capacity.
    /// The ordinary constructor continues to require minimum input extents exactly.
    ///
    /// # Errors
    /// Rejects invalid IR, wrong input count, shorter capacities, physical extent overflow and native
    /// preparation failure. Execution rejects different full resident shapes before staging.
    pub fn prepare_host_kernel_with_input_extents(
        &self,
        kernel: &PcuDispatchKernelIr<'_>,
        input_extents: &[usize],
    ) -> Result<MlxPreparedDivRemRoleHostKernel, MlxHostKernelError> {
        self.prepare_host_kernel_with_input_extents_internal(kernel, Some(input_extents))
    }
    fn prepare_host_kernel_with_input_extents_internal(
        &self,
        kernel: &PcuDispatchKernelIr<'_>,
        input_extents: Option<&[usize]>,
    ) -> Result<MlxPreparedDivRemRoleHostKernel, MlxHostKernelError> {
        let plan =
            MlxCheckedDivRemRolePlan::assess(kernel).map_err(PcuHostDispatchError::Backend)?;
        let count = plan.input_bindings().len();
        let mut extents = plan.input_element_counts();
        if let Some(actual) = input_extents {
            if actual.len() != count
                || actual
                    .iter()
                    .zip(extents)
                    .any(|(&full, minimum)| full < minimum)
            {
                return Err(PcuHostDispatchError::Backend(MlxError::InvalidExtent));
            }
            extents[..count].copy_from_slice(actual);
        }
        let operands = plan.operand_inputs().map(|slot| extents[slot]);
        let control = if input_extents.is_some() {
            self.session
                .prepare_checked_div_rem_control_with_input_extents(
                    plan.scalar_type(),
                    plan.element_count(),
                    operands,
                    plan.operand_broadcast(),
                )
        } else {
            self.session.prepare_checked_div_rem_control(
                plan.scalar_type(),
                plan.element_count(),
                operands,
                plan.operand_broadcast(),
            )
        }
        .map_err(PcuHostDispatchError::Backend)?;
        Ok(MlxPreparedDivRemRoleHostKernel {
            control,
            plan,
            input_extents: extents,
        })
    }
}
impl MlxPreparedDivRemRoleHostKernel {
    #[must_use]
    pub fn argument_count(&self) -> usize {
        self.input_bindings().len() + 2
    }
    #[must_use]
    pub fn input_bindings(&self) -> &[PcuBindingRef] {
        self.plan.input_bindings()
    }
    /// Actual full native input capacities, in the unique binding order frozen during preparation.
    #[must_use]
    pub fn input_element_extents(&self) -> &[usize] {
        &self.input_extents[..self.input_bindings().len()]
    }
    #[must_use]
    pub const fn output_bindings(&self) -> [PcuBindingRef; 2] {
        self.plan.output_bindings()
    }
    #[must_use]
    pub const fn scalar_type(&self) -> PcuScalarType {
        self.plan.scalar_type()
    }
    #[must_use]
    pub const fn input_byte_lengths(&self) -> [usize; 2] {
        self.plan.input_byte_lengths()
    }
    #[must_use]
    pub const fn output_byte_lengths(&self) -> [usize; 2] {
        [self.plan.byte_len(); 2]
    }
    #[must_use]
    pub const fn output_element_count(&self) -> usize {
        self.plan.element_count()
    }
    fn validate_input(&self, slot: usize, input: MlxBinaryInput<'_>) -> Result<(), MlxError> {
        let expected = self.input_bindings()[slot];
        match input {
            MlxBinaryInput::HostBytes {
                target,
                scalar,
                bytes,
            } => {
                if target != expected {
                    return Err(MlxError::InvalidRequest(
                        "unexpected DivRem role input".into(),
                    ));
                }
                if scalar != self.scalar_type() {
                    return Err(MlxError::UnsupportedScalar(scalar));
                }
                if self.input_extents[slot] != self.plan.input_element_counts()[slot]
                    || bytes.len() < self.input_byte_lengths()[slot]
                {
                    return Err(MlxError::InvalidExtent);
                }
            }
            MlxBinaryInput::Resident { target, array } => {
                if target != expected {
                    return Err(MlxError::InvalidRequest(
                        "unexpected DivRem role input".into(),
                    ));
                }
                if array.scalar_type() != self.scalar_type() {
                    return Err(MlxError::UnsupportedScalar(array.scalar_type()));
                }
                if array.element_count() != self.input_extents[slot] {
                    return Err(MlxError::InvalidExtent);
                }
                if !array.same_session(&self.control.session) {
                    return Err(MlxError::ForeignSession);
                }
                array.validate_access_available()?;
            }
        }
        Ok(())
    }
    /// Preflights every actual read, stages each unique host prefix once, then completes both owners.
    ///
    /// # Errors
    /// Rejects missing/duplicate inputs, type/span/session mismatches before staging; checked/native
    /// failure publishes neither owner. Unknown completion retains actual pending native resources.
    pub fn execute_inputs(
        &mut self,
        inputs: &[MlxBinaryInput<'_>],
    ) -> Result<[MlxEncodedArray; 2], MlxError> {
        self.control.native.reset_write_fact();
        let count = self.input_bindings().len();
        if inputs.len() != count {
            return Err(MlxError::InvalidExtent);
        }
        let mut ordered = [None, None];
        for &input in inputs {
            let target = match input {
                MlxBinaryInput::HostBytes { target, .. }
                | MlxBinaryInput::Resident { target, .. } => target,
            };
            let slot = self
                .input_bindings()
                .iter()
                .position(|&binding| binding == target)
                .ok_or_else(|| MlxError::InvalidRequest("unexpected DivRem role input".into()))?;
            if ordered[slot].replace(input).is_some() {
                return Err(MlxError::InvalidExtent);
            }
            self.validate_input(slot, input)?;
        }
        let mut staged = [None, None];
        for slot in 0..count {
            if let Some(MlxBinaryInput::HostBytes { bytes, .. }) = ordered[slot] {
                staged[slot] = Some(self.control.session.upload_encoded_bytes(
                    self.scalar_type(),
                    self.plan.input_element_counts()[slot],
                    &bytes[..self.input_byte_lengths()[slot]],
                )?);
            }
        }
        let owner = |slot: usize| match ordered[slot] {
            Some(MlxBinaryInput::Resident { array, .. }) => Ok(array),
            Some(MlxBinaryInput::HostBytes { .. }) => {
                staged[slot].as_ref().ok_or(MlxError::InvalidExtent)
            }
            None => Err(MlxError::InvalidExtent),
        };
        let [left, right] = self.plan.operand_inputs();
        let result = self.control.execute_resident([owner(left)?, owner(right)?]);
        if self.last_call_completion_uncertain() {
            return result;
        }
        for owner in staged.into_iter().flatten() {
            owner.release()?;
        }
        result
    }
    #[must_use]
    pub const fn last_call_may_have_written(&self) -> bool {
        self.control.last_call_may_have_written()
    }
    /// Existing immutable output owners are replaced only after joint successful completion.
    #[must_use]
    pub const fn last_call_may_have_written_existing_encoded_owner(&self) -> bool {
        false
    }
    #[must_use]
    pub fn last_call_completion_uncertain(&self) -> bool {
        self.control.last_call_completion_uncertain()
    }
    fn positions(
        &self,
        arguments: &[PcuHostArgument<'_>],
    ) -> Result<[usize; 4], MlxHostKernelError> {
        let count = self.input_bindings().len();
        if self.input_extents != self.plan.input_element_counts() {
            return Err(PcuHostDispatchError::Backend(MlxError::InvalidExtent));
        }
        let outputs = self.output_bindings();
        let mut positions = [None; 4];
        for (index, argument) in arguments.iter().enumerate() {
            let target = argument.target();
            if arguments[..index]
                .iter()
                .any(|prior| prior.target() == target)
            {
                return Err(PcuHostDispatchError::Duplicate(target));
            }
            if self.plan.is_unread_declaration(target) {
                if argument.scalar() != self.scalar_type() {
                    return Err(PcuHostDispatchError::TypeMismatch(target));
                }
                if argument.access() != PcuBindingAccess::ReadOnly {
                    return Err(PcuHostDispatchError::AccessMismatch(target));
                }
                continue;
            }
            let slot = self
                .input_bindings()
                .iter()
                .position(|&binding| binding == target)
                .or_else(|| {
                    outputs
                        .iter()
                        .position(|&binding| binding == target)
                        .map(|slot| slot + count)
                })
                .ok_or(PcuHostDispatchError::Unexpected(target))?;
            if positions[slot].replace(index).is_some() {
                return Err(PcuHostDispatchError::Duplicate(target));
            }
            if argument.scalar() != self.scalar_type() {
                return Err(PcuHostDispatchError::TypeMismatch(target));
            }
            let access = if slot < count {
                PcuBindingAccess::ReadOnly
            } else {
                PcuBindingAccess::ReadWrite
            };
            if argument.access() != access {
                return Err(PcuHostDispatchError::AccessMismatch(target));
            }
            let bytes = if slot < count {
                self.input_byte_lengths()[slot]
            } else {
                self.plan.byte_len()
            };
            if argument.bytes().len() < bytes {
                return Err(PcuHostDispatchError::BufferTooSmall(target));
            }
        }
        let mut found = [0; 4];
        for slot in 0..count + 2 {
            let binding = if slot < count {
                self.input_bindings()[slot]
            } else {
                outputs[slot - count]
            };
            found[slot] = positions[slot].ok_or(PcuHostDispatchError::Missing(binding))?;
        }
        Ok(found)
    }
}
impl PcuPreparedHostKernel for MlxPreparedDivRemRoleHostKernel {
    type Error = MlxHostKernelError;
    fn call(&mut self, arguments: &mut [PcuHostArgument<'_>]) -> Result<(), Self::Error> {
        self.control.native.reset_write_fact();
        let positions = self.positions(arguments)?;
        let count = self.input_bindings().len();
        let low = positions[count].min(positions[count + 1]);
        let high = positions[count].max(positions[count + 1]);
        let (before, rest) = arguments.split_at_mut(low);
        let (low_output, rest) = rest.split_first_mut().unwrap();
        let (between, rest) = rest.split_at_mut(high - low - 1);
        let (high_output, after) = rest.split_first_mut().unwrap();
        let input = |position: usize| {
            if position < low {
                before[position].bytes()
            } else if position < high {
                between[position - low - 1].bytes()
            } else {
                after[position - high - 1].bytes()
            }
        };
        let inputs = std::array::from_fn::<_, 2, _>(|slot| {
            if slot < count {
                input(positions[slot])
            } else {
                &[]
            }
        });
        let bytes = self.plan.byte_len();
        let (q, r) = if positions[count] == low {
            (low_output, high_output)
        } else {
            (high_output, low_output)
        };
        let counts = self.plan.input_element_counts();
        self.control
            .native
            .execute_host_roles(
                &inputs[..count],
                &counts[..count],
                self.plan.operand_inputs(),
                [
                    &mut q.bytes_mut().unwrap()[..bytes],
                    &mut r.bytes_mut().unwrap()[..bytes],
                ],
            )
            .map_err(PcuHostDispatchError::Backend)
    }
}
