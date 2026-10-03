//! Detached fourteen-width integer admission and explicit exact-session preparation backend.
#[rustfmt::skip]
use fusion_pcu::{
    PcuBindingAccess,
    PcuBindingRef,
    PcuBindingType,
    PcuDispatchControlOp,
    PcuDispatchDataOp,
    PcuDispatchIndex,
    PcuDispatchKernelIr,
    PcuDispatchOp,
    PcuHostArgument,
    PcuHostDispatchError,
    PcuHostKernelBackend,
    PcuImplementationRequirements,
    PcuPreparedHostKernel,
    PcuReproducibility,
    PcuScalarType,
    PcuValueType,
    PcuValueTypeCaps,
    assess_checked_integer_binary_operands,
    PcuDispatchIntegerBinaryOp,
    PcuRangePolicy,
};
#[rustfmt::skip]
use crate::{
    MlxError,
    MlxSession,
    MlxCheckedIntegerControl,
    MlxEncodedArray,
    MlxEncodedCompletion,
    MlxBinaryInput,
};
use super::super::super::MlxHostKernelError;
/// Detached schema retaining actual SSA operands and unique input binding schemas.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MlxCheckedIntegerPlan {
    scalar: PcuScalarType,
    operation: PcuDispatchIntegerBinaryOp,
    range: PcuRangePolicy,
    count: usize,
    inputs: [PcuBindingRef; 2],
    input_count: usize,
    declarations: [PcuBindingRef; 2],
    declaration_count: usize,
    input_counts: [usize; 2],
    output: PcuBindingRef,
    roles: [usize; 2],
    broadcast: [bool; 2],
    requirements: PcuImplementationRequirements,
}
impl MlxCheckedIntegerPlan {
    /// Checks the shared typed integer schema and freezes actual SSA input roles cold.
    ///
    /// # Errors
    /// Rejects unsupported type/profile, Portable, policy mismatch, malformed schema or extent.
    #[allow(clippy::too_many_lines)] // One cold typed/schema/role admission; no native work occurs here.
    pub fn assess(kernel: &PcuDispatchKernelIr<'_>) -> Result<Self, MlxError> {
        let invalid = || MlxError::InvalidRequest("unsupported MLX checked integer profile".into());
        if cfg!(target_endian = "big")
            || kernel.entry.logical_shape[1..] != [1, 1]
            || kernel
                .numerical_requirements
                .numerical_options
                .reproducibility
                != PcuReproducibility::Unspecified
        {
            return Err(invalid());
        }
        let Some(PcuBindingType::Value(PcuValueType::Scalar(scalar))) =
            kernel.bindings.first().map(|binding| binding.binding_type)
        else {
            return Err(invalid());
        };
        let (body, count) = match kernel.ops {
            [
                PcuDispatchOp::GridStrideLoop { extent, body },
                PcuDispatchOp::Control(PcuDispatchControlOp::Return),
            ] => (*body, *extent),
            [
                body @ ..,
                PcuDispatchOp::Control(PcuDispatchControlOp::Return),
            ] => (body, kernel.entry.logical_shape[0]),
            _ => return Err(invalid()),
        };
        let [
            _,
            _,
            PcuDispatchOp::Data(PcuDispatchDataOp::CheckedIntegerBinary {
                op, range_policy, ..
            }),
            _,
        ] = body
        else {
            return Err(invalid());
        };
        let schema = assess_checked_integer_binary_operands(
            kernel,
            PcuValueType::Scalar(scalar),
            *op,
            PcuValueTypeCaps::for_scalar(scalar),
        )
        .map_err(|_| invalid())?;
        if count == 0 {
            return Err(MlxError::InvalidExtent);
        }
        let count = usize::try_from(count).map_err(|_| MlxError::InvalidExtent)?;
        let limbs = usize::from(scalar.bit_width()).div_ceil(32);
        i32::try_from(count.checked_mul(limbs).ok_or(MlxError::InvalidExtent)?)
            .map_err(|_| MlxError::InvalidExtent)?;
        if [usize::from(scalar.bit_width()) / 8, 4]
            .into_iter()
            .any(|width| {
                count
                    .checked_mul(width)
                    .is_none_or(|bytes| isize::try_from(bytes).is_err())
            })
        {
            return Err(MlxError::InvalidExtent);
        }
        let mut inputs = [schema.input_bindings()[0]; 2];
        let input_count = schema.input_bindings().len();
        inputs[..input_count].copy_from_slice(schema.input_bindings());
        let input_counts = schema.input_element_counts(count);
        let roles = schema.operand_inputs();
        let broadcast = schema
            .operand_indices()
            .map(|index| index == PcuDispatchIndex::BindingElementZero);
        let mut declarations = [inputs[0]; 2];
        let mut declaration_count = 0;
        for binding in kernel.bindings {
            if binding.access == PcuBindingAccess::ReadOnly {
                declarations[declaration_count] = binding.reference();
                declaration_count += 1;
            }
        }
        Ok(Self {
            scalar,
            operation: *op,
            range: *range_policy,
            count,
            inputs,
            input_count,
            declarations,
            declaration_count,
            input_counts,
            output: schema.output_binding(),
            roles,
            broadcast,
            requirements: kernel.numerical_requirements,
        })
    }
    pub(crate) fn implementation_local_id(&self) -> u32 {
        let operation = match self.operation {
            PcuDispatchIntegerBinaryOp::Add => 0,
            PcuDispatchIntegerBinaryOp::Sub => 1,
            PcuDispatchIntegerBinaryOp::Mul => 2,
        };
        0x2000
            + 128 * self.scalar as u32
            + operation
            + 4 * u32::from(self.range == PcuRangePolicy::Clamp)
            + 8 * u32::from(self.broadcast[0])
            + 16 * u32::from(self.broadcast[1])
            + 32 * u32::from(self.input_count == 1)
    }
    /// Exact admitted logical scalar representation.
    #[must_use]
    pub const fn scalar_type(&self) -> PcuScalarType {
        self.scalar
    }
    /// Frozen unique source input binding schemas; actual operation operands may repeat them.
    #[must_use]
    pub fn input_bindings(&self) -> &[PcuBindingRef] {
        &self.inputs[..self.input_count]
    }
    /// Minimum actual unique read spans; the unused metadata slot is zero.
    #[must_use]
    pub const fn input_element_counts(&self) -> [usize; 2] {
        self.input_counts
    }
    /// Complete admitted numerical tuple; scalar checking stays independent of permissions.
    #[must_use]
    pub const fn requirements(&self) -> PcuImplementationRequirements {
        self.requirements
    }
    /// Checks each actual binding's full capacity before runtime work or allocation.
    ///
    /// # Errors
    /// Rejects wrong arity, short spans or overflowing logical/physical carrier extents.
    pub fn assess_input_extents(&self, extents: &[usize]) -> Result<[usize; 2], MlxError> {
        if extents.len() != self.input_count {
            return Err(MlxError::InvalidExtent);
        }
        let width = usize::from(self.scalar.bit_width()) / 8;
        let mut full = [0; 2];
        for (slot, &extent) in extents.iter().enumerate() {
            let bytes = extent.checked_mul(width).ok_or(MlxError::InvalidExtent)?;
            if extent < self.input_counts[slot]
                || isize::try_from(bytes).is_err()
                || i32::try_from(bytes / width.min(4)).is_err()
            {
                return Err(MlxError::InvalidExtent);
            }
            full[slot] = extent;
        }
        Ok(full)
    }
}
/// Exact-session source preparation wrapper; cloning it does not open or rebind a device.
#[derive(Clone)]
pub struct MlxCheckedIntegerBackend {
    session: MlxSession,
}
/// Prepared fourteen-width binary host map with frozen operand/schema roles.
pub struct MlxPreparedIntegerHostKernel {
    control: MlxCheckedIntegerControl,
    plan: MlxCheckedIntegerPlan,
    input_bytes: [usize; 2],
    output_bytes: usize,
    input_extents: [usize; 2],
}
impl MlxSession {
    /// Retains this exact session for detached integer source preparation, without discovery.
    #[must_use]
    pub fn checked_integer_backend(&self) -> MlxCheckedIntegerBackend {
        MlxCheckedIntegerBackend {
            session: self.clone(),
        }
    }
}
impl PcuHostKernelBackend for MlxCheckedIntegerBackend {
    type Prepared = MlxPreparedIntegerHostKernel;
    type Error = MlxHostKernelError;
    fn prepare_host_kernel(
        &self,
        kernel: &PcuDispatchKernelIr<'_>,
    ) -> Result<Self::Prepared, Self::Error> {
        let plan = MlxCheckedIntegerPlan::assess(kernel).map_err(PcuHostDispatchError::Backend)?;
        self.prepare_plan(plan, None)
    }
}
impl MlxCheckedIntegerBackend {
    /// Prepares one/two actual input capacities cold, without fabricating an unused binding.
    /// Host-read minima and output/status extent remain unchanged; warm resident apply is exact.
    ///
    /// # Errors
    /// Rejects unsupported requests, wrong/short capacities or native preparation failure.
    pub fn prepare_host_kernel_with_input_extents(
        &self,
        kernel: &PcuDispatchKernelIr<'_>,
        extents: &[usize],
    ) -> Result<MlxPreparedIntegerHostKernel, MlxHostKernelError> {
        let plan = MlxCheckedIntegerPlan::assess(kernel).map_err(PcuHostDispatchError::Backend)?;
        let full = plan
            .assess_input_extents(extents)
            .map_err(PcuHostDispatchError::Backend)?;
        self.prepare_plan(plan, Some(full))
    }
    fn prepare_plan(
        &self,
        plan: MlxCheckedIntegerPlan,
        full: Option<[usize; 2]>,
    ) -> Result<MlxPreparedIntegerHostKernel, MlxHostKernelError> {
        let inputs = plan.roles.map(|role| plan.input_counts[role]);
        let control = full
            .map_or_else(
                || {
                    self.session.prepare_checked_integer_control(
                        plan.scalar,
                        plan.operation,
                        plan.range,
                        plan.count,
                        inputs,
                        plan.broadcast,
                    )
                },
                |full| {
                    self.session
                        .prepare_checked_integer_control_with_input_extents(
                            plan.scalar,
                            plan.operation,
                            plan.range,
                            plan.count,
                            inputs,
                            plan.broadcast,
                            plan.roles.map(|role| full[role]),
                        )
                },
            )
            .map_err(PcuHostDispatchError::Backend)?;
        let width = usize::from(plan.scalar.bit_width()) / 8;
        Ok(MlxPreparedIntegerHostKernel {
            control,
            plan,
            input_bytes: plan.input_counts.map(|count| count * width),
            output_bytes: plan.count * width,
            input_extents: full.unwrap_or(plan.input_counts),
        })
    }
}
impl MlxPreparedIntegerHostKernel {
    /// Exact full capacities of actual unique reads, in retained schema order.
    #[must_use]
    pub fn prepared_input_element_counts(&self) -> &[usize] {
        &self.input_extents[..self.plan.input_count]
    }
    /// Full admitted tuple; integer execution does not invent floating metadata.
    #[must_use]
    pub const fn requirements(&self) -> PcuImplementationRequirements {
        self.plan.requirements
    }
    /// Exact unique-input logical spans; the unused second metadata slot is zero.
    #[must_use]
    pub const fn input_byte_lengths(&self) -> [usize; 2] {
        self.input_bytes
    }
    #[must_use]
    pub const fn output_byte_len(&self) -> usize {
        self.output_bytes
    }
    /// Frozen unique input references, independent of Rust argument order or repeated operands.
    #[must_use]
    pub fn input_bindings(&self) -> &[PcuBindingRef] {
        self.plan.input_bindings()
    }
    #[must_use]
    pub const fn output_binding(&self) -> PcuBindingRef {
        self.plan.output
    }
    #[must_use]
    pub const fn scalar_type(&self) -> PcuScalarType {
        self.plan.scalar
    }
    #[must_use]
    pub const fn input_element_counts(&self) -> [usize; 2] {
        self.plan.input_counts
    }
    #[must_use]
    pub const fn output_element_count(&self) -> usize {
        self.plan.count
    }
    /// Borrows exact-session unique inputs and applies actual frozen SSA operand roles.
    ///
    /// # Errors
    /// Returns logical schema/session/extent, checked fatal or native terminal failure.
    pub fn execute_resident(
        &mut self,
        inputs: &[&MlxEncodedArray],
    ) -> Result<MlxEncodedCompletion, MlxError> {
        self.control.native.reset_write_fact();
        if inputs.len() != self.plan.input_count {
            return Err(MlxError::InvalidExtent);
        }
        for (slot, input) in inputs.iter().enumerate() {
            self.validate_resident(slot, input)?;
        }
        self.control
            .execute_resident(self.plan.roles.map(|role| inputs[role]))
    }
    fn validate_resident(&self, slot: usize, input: &MlxEncodedArray) -> Result<(), MlxError> {
        if input.scalar_type() != self.plan.scalar {
            return Err(MlxError::UnsupportedScalar(input.scalar_type()));
        }
        if input.element_count() != self.input_extents[slot] {
            return Err(MlxError::InvalidExtent);
        }
        if !input.same_session(&self.control.session) {
            return Err(MlxError::ForeignSession);
        }
        input.validate_access_available()
    }
    /// Stages each unique host prefix once and borrows each resident input without materialization.
    /// Argument order is independent from both frozen source bindings and actual SSA operands.
    ///
    /// # Errors
    /// Rejects the entire schema/type/span/session before staging; returns checked/native faults.
    pub fn execute_inputs(
        &mut self,
        inputs: &[MlxBinaryInput<'_>],
    ) -> Result<MlxEncodedCompletion, MlxError> {
        self.control.native.reset_write_fact();
        if inputs.len() != self.plan.input_count {
            return Err(MlxError::InvalidExtent);
        }
        let mut ordered = [None; 2];
        let width = usize::from(self.plan.scalar.bit_width()) / 8;
        for input in inputs {
            let slot = self
                .plan
                .input_bindings()
                .iter()
                .position(|&target| {
                    target
                        == match input {
                            MlxBinaryInput::HostBytes { target, .. }
                            | MlxBinaryInput::Resident { target, .. } => *target,
                        }
                })
                .ok_or_else(|| {
                    MlxError::InvalidRequest("unexpected MLX integer input binding".into())
                })?;
            if ordered[slot].replace(*input).is_some() {
                return Err(MlxError::InvalidRequest(
                    "duplicate MLX integer input binding".into(),
                ));
            }
            match input {
                MlxBinaryInput::HostBytes { scalar, bytes, .. } => {
                    if *scalar != self.plan.scalar {
                        return Err(MlxError::UnsupportedScalar(*scalar));
                    }
                    if bytes.len() < self.plan.input_counts[slot] * width {
                        return Err(MlxError::InvalidExtent);
                    }
                    if self.input_extents[slot] != self.plan.input_counts[slot] {
                        return Err(MlxError::InvalidExtent);
                    }
                }
                MlxBinaryInput::Resident { array, .. } => self.validate_resident(slot, array)?,
            }
        }
        let mut staged = [None, None];
        for (slot, input) in ordered.iter().enumerate().take(self.plan.input_count) {
            if let Some(MlxBinaryInput::HostBytes { bytes, .. }) = input {
                let count = self.plan.input_counts[slot];
                staged[slot] = Some(self.control.session.upload_encoded_bytes(
                    self.plan.scalar,
                    count,
                    &bytes[..count * width],
                )?);
            }
        }
        let mut owners = [None; 2];
        for slot in 0..self.plan.input_count {
            owners[slot] = match ordered[slot] {
                Some(MlxBinaryInput::Resident { array, .. }) => Some(array),
                Some(MlxBinaryInput::HostBytes { .. }) => staged[slot].as_ref(),
                None => {
                    return Err(MlxError::InvalidRequest(
                        "missing MLX integer input binding".into(),
                    ));
                }
            };
        }
        self.control.execute_resident([
            owners[self.plan.roles[0]].ok_or(MlxError::InvalidExtent)?,
            owners[self.plan.roles[1]].ok_or(MlxError::InvalidExtent)?,
        ])
    }
    #[must_use]
    pub const fn last_call_may_have_written(&self) -> bool {
        self.control.last_call_may_have_written()
    }
    #[must_use]
    pub fn last_call_completion_uncertain(&self) -> bool {
        self.control.last_call_completion_uncertain()
    }
}
impl PcuPreparedHostKernel for MlxPreparedIntegerHostKernel {
    type Error = MlxHostKernelError;
    fn call(&mut self, arguments: &mut [PcuHostArgument<'_>]) -> Result<(), Self::Error> {
        self.control.native.reset_write_fact();
        let width = usize::from(self.plan.scalar.bit_width()) / 8;
        let mut positions = [None; 3];
        for (index, argument) in arguments.iter().enumerate() {
            let target = argument.target();
            if arguments[..index]
                .iter()
                .any(|prior| prior.target() == target)
            {
                return Err(PcuHostDispatchError::Duplicate(target));
            }
            let input = self
                .plan
                .input_bindings()
                .iter()
                .position(|input| *input == target);
            let slot = if target == self.plan.output {
                Some(2)
            } else {
                input
            };
            if slot.is_none()
                && !self.plan.declarations[..self.plan.declaration_count].contains(&target)
            {
                return Err(PcuHostDispatchError::Unexpected(target));
            }
            if argument.scalar() != self.plan.scalar {
                return Err(PcuHostDispatchError::TypeMismatch(target));
            }
            let access = if slot == Some(2) {
                PcuBindingAccess::ReadWrite
            } else {
                PcuBindingAccess::ReadOnly
            };
            if argument.access() != access {
                return Err(PcuHostDispatchError::AccessMismatch(target));
            }
            if let Some(slot) = slot {
                positions[slot] = Some(index);
                let count = if slot == 2 {
                    self.plan.count
                } else {
                    self.plan.input_counts[slot]
                };
                if argument.bytes().len() < count * width {
                    return Err(PcuHostDispatchError::BufferTooSmall(target));
                }
            }
        }
        for (slot, input) in self.plan.input_bindings().iter().enumerate() {
            if positions[slot].is_none() {
                return Err(PcuHostDispatchError::Missing(*input));
            }
        }
        let output = positions[2].ok_or(PcuHostDispatchError::Missing(self.plan.output))?;
        let (before, after) = arguments.split_at_mut(output);
        let (output, after) = after.split_first_mut().unwrap();
        let input = |position: usize| {
            if position < before.len() {
                &before[position]
            } else {
                &after[position - before.len() - 1]
            }
        };
        let mut bytes = [&[][..]; 2];
        for (slot, bytes) in bytes.iter_mut().enumerate().take(self.plan.input_count) {
            *bytes = &input(
                positions[slot].ok_or(PcuHostDispatchError::Missing(self.plan.inputs[slot]))?,
            )
            .bytes()[..self.plan.input_counts[slot] * width];
        }
        self.control
            .native
            .execute_host_mapped(
                &bytes[..self.plan.input_count],
                self.plan.roles,
                &mut output
                    .bytes_mut()
                    .ok_or(PcuHostDispatchError::AccessMismatch(self.plan.output))?
                    [..self.plan.count * width],
            )
            .map_err(PcuHostDispatchError::Backend)
    }
}
