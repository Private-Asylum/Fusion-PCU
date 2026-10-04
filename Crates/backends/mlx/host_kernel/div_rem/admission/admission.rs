//! Exact four-argument preparation with two independently retained output bindings.
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
#[rustfmt::skip]
use super::{
    MlxCheckedDivRemControl,
    MlxCheckedDivRemPlan,
};
/// A clone retains the exact native session; it does not open or rebind a device.
#[derive(Clone)]
pub struct MlxCheckedDivRemBackend {
    session: MlxSession,
}
/// Prepared exact quotient/remainder map, with no single-output metadata fiction.
pub struct MlxPreparedDivRemHostKernel {
    control: MlxCheckedDivRemControl,
    plan: MlxCheckedDivRemPlan,
    input_extents: [usize; 2],
}
impl MlxSession {
    /// Retains this actual session for cold quotient/remainder source preparation.
    #[must_use]
    pub fn checked_div_rem_backend(&self) -> MlxCheckedDivRemBackend {
        MlxCheckedDivRemBackend {
            session: self.clone(),
        }
    }
}
impl PcuHostKernelBackend for MlxCheckedDivRemBackend {
    type Prepared = MlxPreparedDivRemHostKernel;
    type Error = MlxHostKernelError;
    fn prepare_host_kernel(
        &self,
        kernel: &PcuDispatchKernelIr<'_>,
    ) -> Result<Self::Prepared, Self::Error> {
        crate::dispatch_shape::require_non_nested(kernel)
            .map_err(fusion_pcu::PcuHostDispatchError::Backend)?;
        self.prepare_host_kernel_with_input_extents_internal(kernel, None)
    }
}
impl MlxCheckedDivRemBackend {
    /// Freezes exact full dense resident capacities without widening the bounded IR read span.
    ///
    /// Extents follow `MlxCheckedDivRemPlan::input_bindings`; host inputs keep the minimum IR span.
    /// The original factory and its exact input-span guards remain unchanged.
    ///
    /// # Errors
    /// Rejects invalid IR, wrong counts, undersized capacities, physical extent overflow and native
    /// preparation failures. Different full resident shapes reject before any staging.
    pub fn prepare_host_kernel_with_input_extents(
        &self,
        kernel: &PcuDispatchKernelIr<'_>,
        input_extents: &[usize],
    ) -> Result<MlxPreparedDivRemHostKernel, MlxHostKernelError> {
        crate::dispatch_shape::require_non_nested(kernel)
            .map_err(fusion_pcu::PcuHostDispatchError::Backend)?;
        self.prepare_host_kernel_with_input_extents_internal(kernel, Some(input_extents))
    }
    fn prepare_host_kernel_with_input_extents_internal(
        &self,
        kernel: &PcuDispatchKernelIr<'_>,
        input_extents: Option<&[usize]>,
    ) -> Result<MlxPreparedDivRemHostKernel, MlxHostKernelError> {
        crate::dispatch_shape::require_non_nested(kernel)
            .map_err(fusion_pcu::PcuHostDispatchError::Backend)?;
        let plan = MlxCheckedDivRemPlan::assess(kernel).map_err(PcuHostDispatchError::Backend)?;
        let mut extents = [plan.element_count(); 2];
        if let Some(actual) = input_extents {
            if actual.len() != 2 || actual.iter().any(|&full| full < plan.element_count()) {
                return Err(PcuHostDispatchError::Backend(MlxError::InvalidExtent));
            }
            extents.copy_from_slice(actual);
        }
        let operands = plan.operand_inputs().map(|slot| extents[slot]);
        let control = if input_extents.is_some() {
            self.session
                .prepare_checked_div_rem_control_with_input_extents(
                    plan.scalar_type(),
                    plan.element_count(),
                    operands,
                    [false; 2],
                )
        } else {
            self.session.prepare_checked_div_rem_control(
                plan.scalar_type(),
                plan.element_count(),
                operands,
                [false; 2],
            )
        }
        .map_err(PcuHostDispatchError::Backend)?;
        Ok(MlxPreparedDivRemHostKernel {
            control,
            plan,
            input_extents: extents,
        })
    }
}
impl MlxPreparedDivRemHostKernel {
    #[must_use]
    pub const fn argument_count(&self) -> usize {
        4
    }
    #[must_use]
    pub const fn input_bindings(&self) -> &[PcuBindingRef; 2] {
        self.plan.input_bindings()
    }
    /// Exact full native capacities frozen in source input binding order.
    #[must_use]
    pub const fn input_element_extents(&self) -> &[usize; 2] {
        &self.input_extents
    }
    #[must_use]
    pub const fn output_bindings(&self) -> &[PcuBindingRef; 2] {
        self.plan.output_bindings()
    }
    #[must_use]
    pub const fn scalar_type(&self) -> PcuScalarType {
        self.plan.scalar_type()
    }
    #[must_use]
    pub const fn input_byte_lengths(&self) -> [usize; 2] {
        [self.plan.byte_len(); 2]
    }
    #[must_use]
    pub const fn output_byte_lengths(&self) -> [usize; 2] {
        [self.plan.byte_len(); 2]
    }
    #[must_use]
    pub const fn output_element_count(&self) -> usize {
        self.plan.element_count()
    }
    /// Borrows exact-session source inputs in frozen load order, resolving actual SSA operands.
    ///
    /// # Errors
    /// Preflights both inputs before private GPU work; fatal execution publishes neither result.
    pub fn execute_resident(
        &mut self,
        inputs: [&MlxEncodedArray; 2],
    ) -> Result<[MlxEncodedArray; 2], MlxError> {
        self.control
            .execute_resident(self.plan.operand_inputs().map(|slot| inputs[slot]))
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
                        "unexpected MLX div/rem input".into(),
                    ));
                }
                if scalar != self.scalar_type() {
                    return Err(MlxError::UnsupportedScalar(scalar));
                }
                if self.input_extents[slot] != self.plan.element_count()
                    || bytes.len() < self.plan.byte_len()
                {
                    return Err(MlxError::InvalidExtent);
                }
            }
            MlxBinaryInput::Resident { target, array } => {
                if target != expected {
                    return Err(MlxError::InvalidRequest(
                        "unexpected MLX div/rem input".into(),
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
    /// Stages both used host prefixes once, with resident borrows retained on the actual session.
    ///
    /// # Errors
    /// Rejects duplicate/missing binding, type/span or affinity before staging either input;
    /// returns fatal arithmetic/native failures without publishing either new owner.
    pub fn execute_inputs(
        &mut self,
        inputs: &[MlxBinaryInput<'_>],
    ) -> Result<[MlxEncodedArray; 2], MlxError> {
        self.control.native.reset_write_fact();
        let [first, second] = inputs else {
            return Err(MlxError::InvalidExtent);
        };
        let target = |input: MlxBinaryInput<'_>| match input {
            MlxBinaryInput::HostBytes { target, .. } | MlxBinaryInput::Resident { target, .. } => {
                target
            }
        };
        let ordered = if target(*first) == self.input_bindings()[0] {
            [*first, *second]
        } else {
            [*second, *first]
        };
        for (slot, input) in ordered.into_iter().enumerate() {
            self.validate_input(slot, input)?;
        }
        let mut staged = [None, None];
        for (slot, input) in ordered.into_iter().enumerate() {
            if let MlxBinaryInput::HostBytes { bytes, .. } = input {
                staged[slot] = Some(self.control.session.upload_encoded_bytes(
                    self.scalar_type(),
                    self.plan.element_count(),
                    &bytes[..self.plan.byte_len()],
                )?);
            }
        }
        let owner = |slot: usize| match ordered[slot] {
            MlxBinaryInput::Resident { array, .. } => Ok(array),
            MlxBinaryInput::HostBytes { .. } => {
                staged[slot].as_ref().ok_or(MlxError::InvalidExtent)
            }
        };
        let result = self.execute_resident([owner(0)?, owner(1)?]);
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
    /// Private sibling arrays replace immutable owners only after joint successful completion.
    #[must_use]
    pub const fn last_call_may_have_written_existing_encoded_owner(&self) -> bool {
        false
    }
    #[must_use]
    pub fn last_call_completion_uncertain(&self) -> bool {
        self.control.last_call_completion_uncertain()
    }
    fn argument_positions(
        &self,
        arguments: &[PcuHostArgument<'_>],
    ) -> Result<[usize; 4], MlxHostKernelError> {
        if self.input_extents != [self.plan.element_count(); 2] {
            return Err(PcuHostDispatchError::Backend(MlxError::InvalidExtent));
        }
        let references = [
            self.input_bindings()[0],
            self.input_bindings()[1],
            self.output_bindings()[0],
            self.output_bindings()[1],
        ];
        let mut positions = [None; 4];
        for (index, argument) in arguments.iter().enumerate() {
            let target = argument.target();
            let slot = references
                .iter()
                .position(|&reference| reference == target)
                .ok_or(PcuHostDispatchError::Unexpected(target))?;
            if positions[slot].replace(index).is_some() {
                return Err(PcuHostDispatchError::Duplicate(target));
            }
            if argument.scalar() != self.scalar_type() {
                return Err(PcuHostDispatchError::TypeMismatch(target));
            }
            let access = if slot < 2 {
                PcuBindingAccess::ReadOnly
            } else {
                PcuBindingAccess::ReadWrite
            };
            if argument.access() != access {
                return Err(PcuHostDispatchError::AccessMismatch(target));
            }
            if argument.bytes().len() < self.plan.byte_len() {
                return Err(PcuHostDispatchError::BufferTooSmall(target));
            }
        }
        let mut found = [0; 4];
        for (slot, position) in positions.into_iter().enumerate() {
            found[slot] = position.ok_or(PcuHostDispatchError::Missing(references[slot]))?;
        }
        Ok(found)
    }
}
impl PcuPreparedHostKernel for MlxPreparedDivRemHostKernel {
    type Error = MlxHostKernelError;
    fn call(&mut self, arguments: &mut [PcuHostArgument<'_>]) -> Result<(), Self::Error> {
        self.control.native.reset_write_fact();
        let positions = self.argument_positions(arguments)?;
        let low = positions[2].min(positions[3]);
        let high = positions[2].max(positions[3]);
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
        let bytes = self.plan.byte_len();
        let inputs = self
            .plan
            .operand_inputs()
            .map(|slot| &input(positions[slot])[..bytes]);
        // Separate split_at_mut partitions retain both exclusive output leases.
        let (q, r) = if positions[2] == low {
            (low_output, high_output)
        } else {
            (high_output, low_output)
        };
        let [quotient, remainder] = *self.output_bindings();
        self.control
            .native
            .execute_host(
                inputs,
                [
                    &mut q
                        .bytes_mut()
                        .ok_or(PcuHostDispatchError::AccessMismatch(quotient))?[..bytes],
                    &mut r
                        .bytes_mut()
                        .ok_or(PcuHostDispatchError::AccessMismatch(remainder))?[..bytes],
                ],
            )
            .map_err(PcuHostDispatchError::Backend)
    }
}
