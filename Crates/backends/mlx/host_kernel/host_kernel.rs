//! Fixed checked six-format unary host maps with MLX-owned GPU scheduling and exact carrier bytes.
#[path = "div_rem/div_rem.rs"]
mod div_rem;
#[path = "unary/unary.rs"]
mod unary;
#[rustfmt::skip]
pub use div_rem::{
    MlxCheckedDivRemRolePlan,
    MlxCheckedDivRemRoleBackend,
    MlxPreparedDivRemRoleHostKernel,
    MlxCheckedDivRemPlan,
    MlxCheckedDivRemControl,
    MlxCheckedDivRemBackend,
    MlxPreparedDivRemHostKernel,
};
#[path = "integer/integer.rs"]
mod integer;
#[rustfmt::skip]
pub use integer::{
    MlxCheckedIntegerControl,
    MlxCheckedIntegerPlan,
    MlxCheckedIntegerBackend,
    MlxPreparedIntegerHostKernel,
};
#[path = "binary/binary.rs"]
mod binary;
#[path = "carrier/carrier.rs"]
mod carrier;
#[rustfmt::skip]
pub use carrier::{
    MlxCarrierPlan,
    MlxPreparedCarrierHostKernel,
    MlxCarrierControl,
};
#[rustfmt::skip]
pub use binary::{
    MlxCheckedBinaryControl,
    MlxCheckedBinaryPlan,
    MlxCheckedBinaryBackend,
    MlxPreparedBinaryHostKernel,
    MlxBinaryInput,
};
#[path = "aggregate/aggregate.rs"]
mod aggregate;
#[rustfmt::skip]
pub use aggregate::{
    MlxPreparedDispatchKernel,
    MlxDispatchOutputLayout,
    MlxDispatchCompletion,
};
#[rustfmt::skip]
use fusion_pcu::{
    PcuBindingAccess,
    PcuBindingRef,
    PcuDispatchKernelIr,
    PcuHostArgument,
    PcuHostDispatchError,
    PcuHostKernelBackend,
    PcuImplementationRequirements,
    PcuPreparedHostKernel,
    PcuReproducibility,
    PcuScalar,
    PcuScalarType,
};
#[rustfmt::skip]
use crate::{
    MlxError,
    MlxSession,
};
use crate::ffi;
/// Structured schema failure or checked/native MLX execution fault.
pub type MlxHostKernelError = PcuHostDispatchError<MlxError>;
/// One frozen checked six-format unary program; native `MatMul` permissions are independent.
pub struct MlxPreparedHostKernel {
    native: ffi::CheckedUnary,
    session: MlxSession,
    scalar: PcuScalarType,
    input: PcuBindingRef,
    output: PcuBindingRef,
    input_bytes: usize,
    output_bytes: usize,
    input_extent: usize,
    requirements: PcuImplementationRequirements,
    unused: Vec<PcuBindingRef>,
}
impl MlxPreparedHostKernel {
    /// Complete numerical request retained by cold preparation; scalar checking stays exact.
    #[must_use]
    pub const fn requirements(&self) -> PcuImplementationRequirements {
        self.requirements
    }
    #[must_use]
    pub const fn input_binding(&self) -> PcuBindingRef {
        self.input
    }
    /// Declared readonly bindings with no actual load in this frozen unary program.
    /// These retain type/access metadata and never become staged input resources.
    #[must_use]
    pub fn unused_bindings(&self) -> &[PcuBindingRef] {
        &self.unused
    }
    #[must_use]
    pub const fn output_binding(&self) -> PcuBindingRef {
        self.output
    }
    #[must_use]
    pub const fn scalar_type(&self) -> PcuScalarType {
        self.scalar
    }
    #[must_use]
    pub const fn input_byte_len(&self) -> usize {
        self.input_bytes
    }
    /// Exact full logical resident capacity frozen independently of the host read minimum.
    #[must_use]
    pub const fn prepared_input_element_count(&self) -> usize {
        self.input_extent
    }
    #[must_use]
    pub const fn output_byte_len(&self) -> usize {
        self.output_bytes
    }
    /// Executes the admitted program by borrowing an immutable exact-session encoded array.
    /// No host materialization, migration, discovery or policy rescoring occurs during this call.
    ///
    /// # Errors
    /// Rejects logical dtype, input extent or retained session mismatch before device work;
    /// returns fatal arithmetic/native errors without modifying any preexisting owner.
    pub fn execute_resident(
        &mut self,
        input: &crate::MlxEncodedArray,
    ) -> Result<crate::MlxEncodedCompletion, MlxError> {
        self.native.reset_write_fact();
        if !input.same_session(&self.session) {
            return Err(MlxError::ForeignSession);
        }
        let (output, recovered) = self.native.execute_encoded(input.native())?;
        Ok(self.session.complete_encoded(output, recovered))
    }
    /// Stages a current typed input and executes into a fresh completed private encoded owner.
    ///
    /// # Errors
    /// Returns unsupported dtype/extent, checked fatal arithmetic or native completion error.
    pub fn execute_encoded<T: PcuScalar>(
        &mut self,
        input: &[T],
    ) -> Result<crate::MlxEncodedCompletion, MlxError> {
        let source = PcuHostArgument::read(self.input, input);
        self.execute_encoded_bytes(source.scalar(), source.bytes())
    }
    /// Stages initialized logical input bytes under the frozen admitted scalar identity.
    /// This byte-oriented seam accepts previously validated host arguments without reconstructing
    /// typed Rust slices; exact logical and physical carrier checks remain in this backend.
    ///
    /// # Errors
    /// Returns scalar/extent mismatch before staging or checked fatal/native completion failure.
    pub fn execute_encoded_bytes(
        &mut self,
        scalar: PcuScalarType,
        input: &[u8],
    ) -> Result<crate::MlxEncodedCompletion, MlxError> {
        self.native.reset_write_fact();
        if self.input_extent != self.input_bytes / (usize::from(self.scalar.bit_width()) / 8) {
            return Err(MlxError::InvalidExtent);
        }
        if scalar != self.scalar {
            return Err(MlxError::UnsupportedScalar(scalar));
        }
        if input.len() < self.input_bytes {
            return Err(MlxError::InvalidExtent);
        }
        let width = usize::from(scalar.bit_width()) / 8;
        let staged = self.session.upload_encoded_bytes(
            scalar,
            self.input_bytes / width,
            &input[..self.input_bytes],
        )?;
        self.execute_resident(&staged)
    }
    /// These executions create private immutable results; no existing encoded owner is written.
    #[must_use]
    pub const fn last_call_may_have_written_existing_encoded_owner(&self) -> bool {
        false
    }
    /// Whether the latest call reached potentially output-writing MLX GPU evaluation.
    #[must_use]
    pub const fn last_call_may_have_written(&self) -> bool {
        self.native.may_have_written()
    }
    /// Whether actual stream/backing owners are quarantined after unknown completion.
    #[must_use]
    pub fn last_call_completion_uncertain(&self) -> bool {
        self.native.completion_uncertain()
    }
}
/// Detached exact low unary descriptor; assessment creates no runtime/session or executable.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MlxCheckedUnaryPlan {
    pub(crate) scalar: PcuScalarType,
    pub(crate) operation: fusion_pcu::PcuDispatchFloatUnaryOp,
    pub(crate) underflow: fusion_pcu::PcuFloatUnderflowPolicy,
    pub(crate) range: fusion_pcu::PcuRangePolicy,
    pub(crate) count: usize,
    pub(crate) broadcast: bool,
    input: PcuBindingRef,
    output: PcuBindingRef,
    input_bytes: usize,
    output_bytes: usize,
    requirements: PcuImplementationRequirements,
}
impl MlxCheckedUnaryPlan {
    /// Complete admitted header; independent permissions never erase scalar fault checking.
    #[must_use]
    pub const fn requirements(&self) -> PcuImplementationRequirements {
        self.requirements
    }
    /// The actual sole read role, never an unused declaration or fabricated resource.
    #[must_use]
    pub const fn input_bindings(&self) -> &[PcuBindingRef] {
        std::slice::from_ref(&self.input)
    }
    /// Minimum logical span loaded by the admitted operation.
    #[must_use]
    pub const fn input_element_count(&self) -> usize {
        if self.broadcast { 1 } else { self.count }
    }
    /// Assesses the exact full resident shape without opening a device or allocating a resource.
    ///
    /// # Errors
    /// Rejects wrong input arity, short capacity or overflowing byte/physical lane extent.
    pub fn assess_input_extents(&self, extents: &[usize]) -> Result<usize, MlxError> {
        let [extent] = extents else {
            return Err(MlxError::InvalidExtent);
        };
        let width = usize::from(self.scalar.bit_width()) / 8;
        let bytes = extent.checked_mul(width).ok_or(MlxError::InvalidExtent)?;
        if *extent < self.input_element_count()
            || i32::try_from(bytes / width.min(4)).is_err()
            || isize::try_from(bytes).is_err()
        {
            return Err(MlxError::InvalidExtent);
        }
        Ok(*extent)
    }
    /// Assesses actual SSA/load roles and the entire numerical header without native work.
    ///
    /// # Errors
    /// Rejects unproved types, operations, policies, shapes, interfaces or malformed value flow.
    pub fn assess(kernel: &PcuDispatchKernelIr<'_>) -> Result<Self, MlxError> {
        match kernel
            .numerical_requirements
            .numerical_options
            .reproducibility
        {
            PcuReproducibility::PortableV1 => unary::portable_plan(kernel),
            PcuReproducibility::Unspecified => unary::normal_plan(kernel),
        }
    }
}
impl PcuHostKernelBackend for MlxSession {
    type Prepared = MlxPreparedDispatchKernel;
    type Error = MlxHostKernelError;
    fn prepare_host_kernel(
        &self,
        kernel: &PcuDispatchKernelIr<'_>,
    ) -> Result<Self::Prepared, Self::Error> {
        if MlxCarrierPlan::assess(kernel).is_ok() {
            return self
                .prepare_carrier_host_kernel(kernel)
                .map(MlxPreparedDispatchKernel::Carrier);
        }
        if let Some(fusion_pcu::PcuBindingType::Value(fusion_pcu::PcuValueType::Scalar(scalar))) =
            kernel.bindings.first().map(|binding| binding.binding_type)
            && crate::MlxTransportPlan::assess(kernel, scalar).is_ok()
        {
            return self
                .transport_host_backend()
                .prepare_host_kernel(kernel)
                .map(MlxPreparedDispatchKernel::Transport);
        }
        if MlxCheckedBinaryPlan::assess(kernel).is_ok() {
            return self
                .checked_binary_backend()
                .prepare_host_kernel(kernel)
                .map(MlxPreparedDispatchKernel::Binary);
        }
        if MlxCheckedIntegerPlan::assess(kernel).is_ok() {
            return self
                .checked_integer_backend()
                .prepare_host_kernel(kernel)
                .map(MlxPreparedDispatchKernel::Integer);
        }
        if MlxCheckedDivRemPlan::assess(kernel).is_ok() {
            return self
                .checked_div_rem_backend()
                .prepare_host_kernel(kernel)
                .map(MlxPreparedDispatchKernel::DivRem);
        }
        if MlxCheckedDivRemRolePlan::assess(kernel).is_ok() {
            return self
                .checked_div_rem_role_backend()
                .prepare_host_kernel(kernel)
                .map(MlxPreparedDispatchKernel::DivRemRoles);
        }
        if let Some(fusion_pcu::PcuBindingType::Value(fusion_pcu::PcuValueType::Scalar(scalar))) =
            kernel.bindings.first().map(|binding| binding.binding_type)
            && crate::MlxCheckedMapPlan::assess(kernel, scalar).is_ok()
        {
            return self
                .composed_host_backend()
                .prepare_host_kernel(kernel)
                .map(MlxPreparedDispatchKernel::Composed);
        }
        self.prepare_unary_host_kernel(kernel)
            .map(MlxPreparedDispatchKernel::Unary)
    }
}
impl MlxSession {
    /// Prepares only an exact unary schema, retaining its concrete sole-input owner interface.
    ///
    /// # Errors
    /// Rejects non-unary/malformed policy or schema before runtime work, or returns native failure.
    pub fn prepare_unary_host_kernel(
        &self,
        kernel: &PcuDispatchKernelIr<'_>,
    ) -> Result<MlxPreparedHostKernel, MlxHostKernelError> {
        let plan = MlxCheckedUnaryPlan::assess(kernel).map_err(PcuHostDispatchError::Backend)?;
        self.prepare_unary_plan(plan, None, kernel.bindings)
    }
    /// Freezes a full resident input shape while retaining the original logical read/status domain.
    /// Cold priming stages the full synthetic capacity. Warm apply borrows that exact shape;
    /// it never creates an input view or widens the checked output/fault domain.
    ///
    /// # Errors
    /// Rejects unsupported tuples, wrong arity or short/oversized capacity before native work.
    pub fn prepare_unary_host_kernel_with_input_extents(
        &self,
        kernel: &PcuDispatchKernelIr<'_>,
        extents: &[usize],
    ) -> Result<MlxPreparedHostKernel, MlxHostKernelError> {
        let plan = MlxCheckedUnaryPlan::assess(kernel).map_err(PcuHostDispatchError::Backend)?;
        let full = plan
            .assess_input_extents(extents)
            .map_err(PcuHostDispatchError::Backend)?;
        self.prepare_unary_plan(plan, Some(full), kernel.bindings)
    }
    fn prepare_unary_plan(
        &self,
        plan: MlxCheckedUnaryPlan,
        full: Option<usize>,
        declarations: &[fusion_pcu::PcuBinding],
    ) -> Result<MlxPreparedHostKernel, MlxHostKernelError> {
        let input_extent = full.unwrap_or_else(|| plan.input_element_count());
        let native = if full.is_some() {
            self.prepare_checked_unary_native_with_input_extent(
                plan.scalar,
                plan.operation,
                plan.underflow,
                plan.range,
                plan.count,
                plan.broadcast,
                input_extent,
            )
        } else {
            self.prepare_checked_unary_native(
                plan.scalar,
                plan.operation,
                plan.underflow,
                plan.range,
                plan.count,
                plan.broadcast,
            )
        }
        .map_err(PcuHostDispatchError::Backend)?;
        Ok(MlxPreparedHostKernel {
            native,
            session: self.clone(),
            scalar: plan.scalar,
            input: plan.input,
            output: plan.output,
            input_bytes: plan.input_bytes,
            output_bytes: plan.output_bytes,
            input_extent,
            requirements: plan.requirements,
            unused: declarations
                .iter()
                .copied()
                .map(fusion_pcu::PcuBinding::reference)
                .filter(|target| *target != plan.input && *target != plan.output)
                .collect(),
        })
    }
}
impl PcuPreparedHostKernel for MlxPreparedHostKernel {
    type Error = MlxHostKernelError;
    fn call(&mut self, arguments: &mut [PcuHostArgument<'_>]) -> Result<(), Self::Error> {
        self.native.reset_write_fact();
        for (index, argument) in arguments.iter().enumerate() {
            let target = argument.target();
            if arguments[..index]
                .iter()
                .any(|prior| prior.target() == target)
            {
                return Err(PcuHostDispatchError::Duplicate(target));
            }
            let (access, bytes) = if target == self.input {
                (PcuBindingAccess::ReadOnly, self.input_bytes)
            } else if target == self.output {
                (PcuBindingAccess::ReadWrite, self.output_bytes)
            } else if self.unused.contains(&target) {
                (PcuBindingAccess::ReadOnly, 0)
            } else {
                return Err(PcuHostDispatchError::Unexpected(target));
            };
            if argument.scalar() != self.scalar {
                return Err(PcuHostDispatchError::TypeMismatch(target));
            }
            if argument.access() != access {
                return Err(PcuHostDispatchError::AccessMismatch(target));
            }
            if argument.bytes().len() < bytes {
                return Err(PcuHostDispatchError::BufferTooSmall(target));
            }
        }
        let source = arguments
            .iter()
            .position(|argument| argument.target() == self.input)
            .ok_or(PcuHostDispatchError::Missing(self.input))?;
        let destination = arguments
            .iter()
            .position(|argument| argument.target() == self.output)
            .ok_or(PcuHostDispatchError::Missing(self.output))?;
        // Disjoint typed borrows require no per-call argument collection or temporary data Vec.
        let (input, output) = if source < destination {
            let (left, right) = arguments.split_at_mut(destination);
            (&left[source], &mut right[0])
        } else {
            let (left, right) = arguments.split_at_mut(source);
            (&right[0], &mut left[destination])
        };
        let output = output
            .bytes_mut()
            .ok_or(PcuHostDispatchError::AccessMismatch(self.output))?;
        self.native
            .execute(
                &input.bytes()[..self.input_bytes],
                &mut output[..self.output_bytes],
            )
            .map_err(PcuHostDispatchError::Backend)
    }
}
/// Direct matched MLX-owned checked kernel control, without source/neutral graph admission.
pub struct MlxCheckedUnaryControl {
    native: ffi::CheckedUnary,
    session: MlxSession,
    scalar: PcuScalarType,
    count: usize,
    input_count: usize,
    input_extent: usize,
}
impl MlxCheckedUnaryControl {
    /// Executes a direct native control into a fresh private immutable encoded result.
    ///
    /// # Errors
    /// Returns dtype/extent/session mismatch or checked/native terminal failure.
    pub fn execute_resident(
        &mut self,
        input: &crate::MlxEncodedArray,
    ) -> Result<crate::MlxEncodedCompletion, MlxError> {
        self.native.reset_write_fact();
        if !input.same_session(&self.session) {
            return Err(MlxError::ForeignSession);
        }
        let (output, recovered) = self.native.execute_encoded(input.native())?;
        Ok(self.session.complete_encoded(output, recovered))
    }
    /// Stages a current typed prefix and runs the same private encoded control boundary.
    ///
    /// # Errors
    /// Returns unsupported scalar/extent before staging or checked/native terminal failure.
    pub fn execute_encoded<T: PcuScalar>(
        &mut self,
        input: &[T],
    ) -> Result<crate::MlxEncodedCompletion, MlxError> {
        self.native.reset_write_fact();
        if self.input_extent != self.input_count {
            return Err(MlxError::InvalidExtent);
        }
        if T::TYPE != self.scalar {
            return Err(MlxError::UnsupportedScalar(T::TYPE));
        }
        if input.len() < self.input_count {
            return Err(MlxError::InvalidExtent);
        }
        let staged = self.session.upload_encoded(&input[..self.input_count])?;
        self.execute_resident(&staged)
    }
    /// Runs fresh host input with checked transactional or recovered prefix publication.
    ///
    /// # Errors
    /// Returns scalar/extent, checked arithmetic or native completion failure.
    pub fn call<T: PcuScalar>(&mut self, input: &[T], output: &mut [T]) -> Result<(), MlxError> {
        self.native.reset_write_fact();
        if self.input_extent != self.input_count {
            return Err(MlxError::InvalidExtent);
        }
        if T::TYPE != self.scalar || input.len() < self.input_count || output.len() < self.count {
            return Err(MlxError::InvalidExtent);
        }
        let input = PcuHostArgument::read(PcuBindingRef::new(0, 0), &input[..self.input_count]);
        let mut output =
            PcuHostArgument::read_write(PcuBindingRef::new(0, 1), &mut output[..self.count]);
        self.native.execute(
            input.bytes(),
            output.bytes_mut().ok_or(MlxError::InvalidExtent)?,
        )
    }
}
impl MlxSession {
    /// Compiles a fixed low unary MLX-owned checker and completes a cold cache-warming run.
    ///
    /// # Errors
    /// Returns unsupported scalar, invalid extent, compilation or completion failure.
    #[allow(clippy::too_many_arguments)] // Independent frozen dtype, operation, policies and operand extent.
    pub fn prepare_checked_unary_control(
        &self,
        scalar: PcuScalarType,
        op: fusion_pcu::PcuDispatchFloatUnaryOp,
        policy: fusion_pcu::PcuFloatUnderflowPolicy,
        range: fusion_pcu::PcuRangePolicy,
        count: usize,
        broadcast: bool,
    ) -> Result<MlxCheckedUnaryControl, MlxError> {
        self.prepare_unary_control(scalar, op, policy, range, count, broadcast, None)
    }
    /// Direct matched unary control with a cold-frozen full resident input shape.
    /// The minimum logical read and output/status domain remain unchanged. Full synthetic
    /// priming is cold work; warm resident calls require exact shape/session and create no view.
    ///
    /// # Errors
    /// Rejects unsupported scalar, short/oversized shape or native preparation/completion failure.
    #[allow(clippy::too_many_arguments)] // Exact operation tuple plus independent full resident capacity.
    pub fn prepare_checked_unary_control_with_input_extent(
        &self,
        scalar: PcuScalarType,
        op: fusion_pcu::PcuDispatchFloatUnaryOp,
        policy: fusion_pcu::PcuFloatUnderflowPolicy,
        range: fusion_pcu::PcuRangePolicy,
        count: usize,
        broadcast: bool,
        input_extent: usize,
    ) -> Result<MlxCheckedUnaryControl, MlxError> {
        self.prepare_unary_control(
            scalar,
            op,
            policy,
            range,
            count,
            broadcast,
            Some(input_extent),
        )
    }
    #[allow(clippy::too_many_arguments)] // Shared exact versus full-capacity cold constructor, without policy projection.
    fn prepare_unary_control(
        &self,
        scalar: PcuScalarType,
        op: fusion_pcu::PcuDispatchFloatUnaryOp,
        policy: fusion_pcu::PcuFloatUnderflowPolicy,
        range: fusion_pcu::PcuRangePolicy,
        count: usize,
        broadcast: bool,
        full: Option<usize>,
    ) -> Result<MlxCheckedUnaryControl, MlxError> {
        let input_count = if broadcast { 1 } else { count };
        let native = full.map_or_else(
            || self.prepare_checked_unary_native(scalar, op, policy, range, count, broadcast),
            |input_extent| {
                self.prepare_checked_unary_native_with_input_extent(
                    scalar,
                    op,
                    policy,
                    range,
                    count,
                    broadcast,
                    input_extent,
                )
            },
        )?;
        Ok(MlxCheckedUnaryControl {
            session: self.clone(),
            native,
            scalar,
            count,
            input_count,
            input_extent: full.unwrap_or(input_count),
        })
    }
}
