//! Exact scalar carrier maps through retained MLX integer primitive on an explicit GPU stream.
#[rustfmt::skip]
use fusion_pcu::{
    PcuBindingRef,
    PcuBindingType,
    PcuValueType,
    PcuBindingAccess,
    PcuScalarType,
    PcuDispatchKernelIr,
    PcuDispatchOp,
    PcuDispatchDataOp,
    PcuDispatchControlOp,
    PcuDispatchIndex,
    PcuReproducibility,
    PcuHostArgument,
    PcuHostDispatchError,
    PcuPreparedHostKernel,
    PcuScalar,
    validate_scalar_identity_kernel,
    validate_scalar_broadcast_kernel,
};
#[rustfmt::skip]
use crate::{
    ffi,
    MlxSession,
    MlxError,
    MlxEncodedArray,
    MlxEncodedCompletion,
};
use super::MlxHostKernelError;
/// Detached exact bit-transport profile; logical extents never masquerade as physical lane counts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MlxCarrierPlan {
    scalar: PcuScalarType,
    input: PcuBindingRef,
    output: PcuBindingRef,
    count: usize,
    broadcast: bool,
    input_bytes: usize,
    output_bytes: usize,
}
impl MlxCarrierPlan {
    /// Actual unique read binding, without a fabricated second input.
    #[must_use]
    pub const fn input_bindings(&self) -> &[PcuBindingRef] {
        std::slice::from_ref(&self.input)
    }
    /// Minimum logical read span; full resident capacity is assessed separately.
    #[must_use]
    pub const fn input_element_count(&self) -> usize {
        if self.broadcast { 1 } else { self.count }
    }
    #[must_use]
    pub const fn scalar_type(&self) -> PcuScalarType {
        self.scalar
    }
    /// Assesses the one actual input capacity without opening a runtime or creating a resource.
    /// Logical host read minima remain unchanged; the returned full count is for resident shape
    /// specialization and truthful full-size cold priming only.
    ///
    /// # Errors
    /// Rejects wrong input count, shorter capacity and oversized physical/byte extents.
    pub fn assess_input_extents(&self, extents: &[usize]) -> Result<usize, MlxError> {
        let [extent] = extents else {
            return Err(MlxError::InvalidExtent);
        };
        let minimum = if self.broadcast { 1 } else { self.count };
        let width = usize::from(self.scalar.bit_width()) / 8;
        let bytes = extent.checked_mul(width).ok_or(MlxError::InvalidExtent)?;
        if *extent < minimum
            || i32::try_from(bytes / width.min(4)).is_err()
            || isize::try_from(bytes).is_err()
        {
            return Err(MlxError::InvalidExtent);
        }
        Ok(*extent)
    }
    pub(crate) const fn implementation_local_id(self) -> u32 {
        0x500 + self.scalar as u32 + if self.broadcast { 32 } else { 0 }
    }
    /// Assesses actual canonical direct/grid SSA roles and the full request before native work.
    ///
    /// # Errors
    /// Rejects noncopy schemas, narrow packed types, Portable, unsafe geometry or oversized spans.
    pub fn assess(kernel: &PcuDispatchKernelIr<'_>) -> Result<Self, MlxError> {
        crate::dispatch_shape::require_non_nested(kernel)?;
        let unsupported = || MlxError::InvalidRequest("unsupported exact MLX carrier map".into());
        if kernel
            .numerical_requirements
            .numerical_options
            .reproducibility
            != PcuReproducibility::Unspecified
            || kernel.entry.logical_shape[1..] != [1, 1]
            || kernel.entry.logical_shape[0] == 0
        {
            return Err(unsupported());
        }
        let Some(binding) = kernel.bindings.first() else {
            return Err(unsupported());
        };
        let PcuBindingType::Value(PcuValueType::Scalar(scalar)) = binding.binding_type else {
            return Err(unsupported());
        };
        if matches!(
            scalar,
            PcuScalarType::Bool | PcuScalarType::I4 | PcuScalarType::U4
        ) {
            return Err(MlxError::UnsupportedScalar(scalar));
        }
        let (body, count) = match kernel.ops {
            [
                PcuDispatchOp::GridStrideLoop { extent, body },
                PcuDispatchOp::Control(PcuDispatchControlOp::Return),
            ] => (*body, *extent),
            [
                body @ ..,
                PcuDispatchOp::Control(PcuDispatchControlOp::Return),
            ] => (body, kernel.entry.logical_shape[0]),
            _ => return Err(unsupported()),
        };
        let [
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                binding: input,
                index,
                ..
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                binding: output, ..
            }),
        ] = body
        else {
            return Err(unsupported());
        };
        let broadcast = *index == PcuDispatchIndex::BindingElementZero;
        if broadcast {
            validate_scalar_broadcast_kernel(kernel, scalar)
        } else {
            validate_scalar_identity_kernel(kernel, scalar)
        }
        .map_err(|_| unsupported())?;
        let count = usize::try_from(count).map_err(|_| MlxError::InvalidExtent)?;
        let width = usize::from(scalar.bit_width()) / 8;
        let output_bytes = count.checked_mul(width).ok_or(MlxError::InvalidExtent)?;
        if count == 0
            || i32::try_from(output_bytes / width.min(4)).is_err()
            || isize::try_from(output_bytes).is_err()
        {
            return Err(MlxError::InvalidExtent);
        }
        Ok(Self {
            scalar,
            input: *input,
            output: *output,
            count,
            broadcast,
            input_bytes: if broadcast { width } else { output_bytes },
            output_bytes,
        })
    }
}
/// One exact retained-session integer carrier primitive plan with cold pipeline warmup.
pub struct MlxPreparedCarrierHostKernel {
    session: MlxSession,
    plan: MlxCarrierPlan,
    native: ffi::CarrierCopy,
    input_extent: usize,
    last_output: Option<ffi::EncodedArray>,
}
impl MlxSession {
    /// Prepares only a byte-aligned dense copy or scalar broadcast and warms real GPU completion.
    ///
    /// # Errors
    /// Returns detached profile rejection or actual staging, compilation, completion or cleanup error.
    pub fn prepare_carrier_host_kernel(
        &self,
        kernel: &PcuDispatchKernelIr<'_>,
    ) -> Result<MlxPreparedCarrierHostKernel, MlxHostKernelError> {
        crate::dispatch_shape::require_non_nested(kernel)
            .map_err(fusion_pcu::PcuHostDispatchError::Backend)?;
        let plan = MlxCarrierPlan::assess(kernel).map_err(PcuHostDispatchError::Backend)?;
        self.prepare_carrier_plan(plan)
            .map_err(PcuHostDispatchError::Backend)
    }
    /// Freezes the full resident input capacity separately from the minimum logical read span.
    /// The actual unique input list contains one entry. Its exact native shape is primed cold;
    /// warm calls borrow that full owner without a view, upload or shape reconstruction.
    ///
    /// # Errors
    /// Rejects invalid IR, wrong input count, short/oversized capacity or native preparation failure.
    pub fn prepare_carrier_host_kernel_with_input_extents(
        &self,
        kernel: &PcuDispatchKernelIr<'_>,
        input_extents: &[usize],
    ) -> Result<MlxPreparedCarrierHostKernel, MlxHostKernelError> {
        crate::dispatch_shape::require_non_nested(kernel)
            .map_err(fusion_pcu::PcuHostDispatchError::Backend)?;
        let plan = MlxCarrierPlan::assess(kernel).map_err(PcuHostDispatchError::Backend)?;
        let extent = plan
            .assess_input_extents(input_extents)
            .map_err(PcuHostDispatchError::Backend)?;
        self.prepare_carrier_plan_with_input_extent(plan, Some(extent))
            .map_err(PcuHostDispatchError::Backend)
    }
    fn prepare_carrier_plan(
        &self,
        plan: MlxCarrierPlan,
    ) -> Result<MlxPreparedCarrierHostKernel, MlxError> {
        self.prepare_carrier_plan_with_input_extent(plan, None)
    }
    fn prepare_carrier_plan_with_input_extent(
        &self,
        plan: MlxCarrierPlan,
        full_extent: Option<usize>,
    ) -> Result<MlxPreparedCarrierHostKernel, MlxError> {
        let minimum = if plan.broadcast { 1 } else { plan.count };
        let input_extent = full_extent.unwrap_or(minimum);
        plan.assess_input_extents(&[input_extent])?;
        let native = if full_extent.is_some() {
            self.prepare_carrier_native_with_input_extent(
                plan.scalar,
                plan.count,
                plan.broadcast,
                input_extent,
            )?
        } else {
            self.prepare_carrier_native(plan.scalar, plan.count, plan.broadcast)?
        };
        let mut prepared = MlxPreparedCarrierHostKernel {
            session: self.clone(),
            plan,
            native,
            input_extent,
            last_output: None,
        };
        let width = usize::from(plan.scalar.bit_width()) / 8;
        let input_bytes = input_extent
            .checked_mul(width)
            .ok_or(MlxError::InvalidExtent)?;
        let input = vec![0_u8; input_bytes];
        let mut output = vec![0_u8; plan.output_bytes];
        #[cfg(feature = "carrier-census")]
        ffi::CarrierCopy::record_prime_call();
        if input_extent == minimum {
            prepared.execute_host(&input, &mut output)?;
        } else {
            // Prime the actual full shape with truthful full-size synthetic storage. This is
            // cold transfer/memory cost, not an assertion of zero uploads or SDK allocations.
            let staged = self.upload_carrier_native(plan.scalar, input_extent, &input)?;
            let result = prepared.native.execute(&staged)?;
            staged.release()?;
            prepared.last_output = Some(result);
            prepared
                .last_output
                .as_ref()
                .ok_or_else(|| MlxError::Abi("missing carrier priming output".into()))?
                .read(&mut output)?;
        }
        prepared.native.reset_write_fact();
        Ok(prepared)
    }
}
/// Direct native integer primitive control, without source/neutral graph parsing or offer selection.
pub struct MlxCarrierControl {
    kernel: MlxPreparedCarrierHostKernel,
}
impl MlxSession {
    /// Prepares a matched exact integer-carrier control and warms its real GPU pipeline.
    ///
    /// # Errors
    /// Rejects packed types, zero/oversized logical and physical spans or native completion failure.
    pub fn prepare_carrier_control(
        &self,
        scalar: PcuScalarType,
        count: usize,
        broadcast: bool,
    ) -> Result<MlxCarrierControl, MlxError> {
        self.prepare_carrier_control_internal(scalar, count, broadcast, None)
    }
    /// Prepares the direct retained control with an exact full resident input shape.
    /// Cold priming stages that full capacity; warm execution reads only the selected prefix/tile.
    ///
    /// # Errors
    /// Rejects packed types, short/oversized capacities or native compilation/completion failure.
    pub fn prepare_carrier_control_with_input_extent(
        &self,
        scalar: PcuScalarType,
        count: usize,
        broadcast: bool,
        input_extent: usize,
    ) -> Result<MlxCarrierControl, MlxError> {
        self.prepare_carrier_control_internal(scalar, count, broadcast, Some(input_extent))
    }
    fn prepare_carrier_control_internal(
        &self,
        scalar: PcuScalarType,
        count: usize,
        broadcast: bool,
        input_extent: Option<usize>,
    ) -> Result<MlxCarrierControl, MlxError> {
        if matches!(
            scalar,
            PcuScalarType::Bool | PcuScalarType::I4 | PcuScalarType::U4
        ) {
            return Err(MlxError::UnsupportedScalar(scalar));
        }
        let width = usize::from(scalar.bit_width()) / 8;
        let output_bytes = count.checked_mul(width).ok_or(MlxError::InvalidExtent)?;
        if count == 0
            || i32::try_from(output_bytes / width.min(4)).is_err()
            || isize::try_from(output_bytes).is_err()
        {
            return Err(MlxError::InvalidExtent);
        }
        let plan = MlxCarrierPlan {
            scalar,
            input: PcuBindingRef::new(0, 0),
            output: PcuBindingRef::new(0, 1),
            count,
            broadcast,
            input_bytes: if broadcast { width } else { output_bytes },
            output_bytes,
        };
        Ok(MlxCarrierControl {
            kernel: self.prepare_carrier_plan_with_input_extent(plan, input_extent)?,
        })
    }
}
impl MlxCarrierControl {
    /// Fresh host input → real MLX integer primitive completion → transactional host prefix publication.
    ///
    /// # Errors
    /// Returns exact type/extent failure before staging or native terminal/cleanup failure.
    pub fn call<T: PcuScalar>(&mut self, input: &[T], output: &mut [T]) -> Result<(), MlxError> {
        self.kernel.reset_write_fact();
        if T::TYPE != self.kernel.scalar_type() {
            return Err(MlxError::UnsupportedScalar(T::TYPE));
        }
        let required = if self.kernel.plan.broadcast {
            1
        } else {
            self.kernel.plan.count
        };
        if input.len() < required || output.len() < self.kernel.plan.count {
            return Err(MlxError::InvalidExtent);
        }
        let source = PcuHostArgument::read(self.kernel.input_binding(), &input[..required]);
        let mut destination = PcuHostArgument::read_write(
            self.kernel.output_binding(),
            &mut output[..self.kernel.plan.count],
        );
        self.kernel.execute_host(
            source.bytes(),
            destination.bytes_mut().ok_or(MlxError::InvalidExtent)?,
        )
    }
    /// Borrows the real exact-session scalar or dense input and returns a fresh completed owner.
    ///
    /// # Errors
    /// Rejects schema, extent or session before native work, or returns native completion failure.
    pub fn execute_resident(
        &mut self,
        input: &MlxEncodedArray,
    ) -> Result<MlxEncodedCompletion, MlxError> {
        self.kernel.execute_resident(input)
    }
}
impl MlxPreparedCarrierHostKernel {
    pub(super) fn reset_write_fact(&self) {
        self.native.reset_write_fact();
    }
    #[must_use]
    pub const fn input_bindings(&self) -> &[PcuBindingRef] {
        std::slice::from_ref(&self.plan.input)
    }
    #[must_use]
    pub const fn input_binding(&self) -> PcuBindingRef {
        self.plan.input
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
    pub const fn input_byte_len(&self) -> usize {
        self.plan.input_bytes
    }
    /// Exact full logical resident capacity frozen for native shape matching.
    #[must_use]
    pub const fn prepared_input_element_count(&self) -> usize {
        self.input_extent
    }
    #[must_use]
    pub const fn output_byte_len(&self) -> usize {
        self.plan.output_bytes
    }
    #[must_use]
    pub const fn last_call_may_have_written(&self) -> bool {
        self.native.may_have_written()
    }
    #[must_use]
    pub fn last_call_completion_uncertain(&self) -> bool {
        self.session.validate_access_available().is_err()
    }
    /// Borrows an immutable exact logical input and publishes a fresh terminal carrier owner.
    ///
    /// # Errors
    /// Rejects dtype, extent or actual session before GPU work; unknown completion quarantines pending owners.
    pub fn execute_resident(
        &mut self,
        input: &MlxEncodedArray,
    ) -> Result<MlxEncodedCompletion, MlxError> {
        self.native.reset_write_fact();
        if input.scalar_type() != self.plan.scalar {
            return Err(MlxError::UnsupportedScalar(input.scalar_type()));
        }
        if input.element_count() != self.input_extent {
            return Err(MlxError::InvalidExtent);
        }
        if !input.same_session(&self.session) {
            return Err(MlxError::ForeignSession);
        }
        let output = self.native.execute(input.native())?;
        Ok(self.session.complete_encoded(output, None))
    }
    /// Stages exactly the admitted initialized byte prefix without reconstructing typed slices.
    ///
    /// # Errors
    /// Rejects type/span before staging or returns native terminal failure without an escaped output.
    pub fn execute_encoded_bytes(
        &mut self,
        scalar: PcuScalarType,
        input: &[u8],
    ) -> Result<MlxEncodedCompletion, MlxError> {
        self.native.reset_write_fact();
        if self.input_extent
            != if self.plan.broadcast {
                1
            } else {
                self.plan.count
            }
        {
            return Err(MlxError::InvalidExtent);
        }
        if scalar != self.plan.scalar {
            return Err(MlxError::UnsupportedScalar(scalar));
        }
        if input.len() < self.plan.input_bytes {
            return Err(MlxError::InvalidExtent);
        }
        let staged = self.session.upload_carrier_native(
            scalar,
            if self.plan.broadcast {
                1
            } else {
                self.plan.count
            },
            &input[..self.plan.input_bytes],
        )?;
        let output = self.native.execute(&staged)?;
        staged.release()?;
        Ok(self.session.complete_encoded(output, None))
    }
    fn execute_host(&mut self, input: &[u8], output: &mut [u8]) -> Result<(), MlxError> {
        self.native.reset_write_fact();
        if self.input_extent
            != if self.plan.broadcast {
                1
            } else {
                self.plan.count
            }
        {
            return Err(MlxError::InvalidExtent);
        }
        if input.len() != self.plan.input_bytes || output.len() != self.plan.output_bytes {
            return Err(MlxError::InvalidExtent);
        }
        if let Some(previous) = self.last_output.take() {
            previous.release()?;
        }
        let staged = self.session.upload_carrier_native(
            self.plan.scalar,
            if self.plan.broadcast {
                1
            } else {
                self.plan.count
            },
            input,
        )?;
        let result = self.native.execute(&staged)?;
        staged.release()?;
        self.last_output = Some(result);
        self.last_output
            .as_ref()
            .ok_or_else(|| MlxError::Abi("missing carrier output".into()))?
            .read(output)
    }
}
impl PcuPreparedHostKernel for MlxPreparedCarrierHostKernel {
    type Error = MlxHostKernelError;
    fn call(&mut self, arguments: &mut [PcuHostArgument<'_>]) -> Result<(), Self::Error> {
        self.native.reset_write_fact();
        for (position, argument) in arguments.iter().enumerate() {
            let target = argument.target();
            if arguments[..position]
                .iter()
                .any(|prior| prior.target() == target)
            {
                return Err(PcuHostDispatchError::Duplicate(target));
            }
            let (access, bytes) = if target == self.plan.input {
                (PcuBindingAccess::ReadOnly, self.plan.input_bytes)
            } else if target == self.plan.output {
                (PcuBindingAccess::ReadWrite, self.plan.output_bytes)
            } else {
                return Err(PcuHostDispatchError::Unexpected(target));
            };
            if argument.scalar() != self.plan.scalar {
                return Err(PcuHostDispatchError::TypeMismatch(target));
            }
            if argument.access() != access {
                return Err(PcuHostDispatchError::AccessMismatch(target));
            }
            if argument.bytes().len() < bytes {
                return Err(PcuHostDispatchError::BufferTooSmall(target));
            }
        }
        let input = arguments
            .iter()
            .position(|argument| argument.target() == self.plan.input)
            .ok_or(PcuHostDispatchError::Missing(self.plan.input))?;
        let output = arguments
            .iter()
            .position(|argument| argument.target() == self.plan.output)
            .ok_or(PcuHostDispatchError::Missing(self.plan.output))?;
        let (source, destination) = if input < output {
            let (left, right) = arguments.split_at_mut(output);
            (&left[input], &mut right[0])
        } else {
            let (left, right) = arguments.split_at_mut(input);
            (&right[0], &mut left[output])
        };
        let destination = destination
            .bytes_mut()
            .ok_or(PcuHostDispatchError::AccessMismatch(self.plan.output))?;
        self.execute_host(
            &source.bytes()[..self.plan.input_bytes],
            &mut destination[..self.plan.output_bytes],
        )
        .map_err(PcuHostDispatchError::Backend)
    }
}
