//! Fourteen-width integer native control and independently admitted exact source preparation.
#[path = "admission/admission.rs"]
mod admission;
#[rustfmt::skip]
pub use admission::{
    MlxCheckedIntegerPlan,
    MlxCheckedIntegerBackend,
    MlxPreparedIntegerHostKernel,
};
#[rustfmt::skip]
use fusion_pcu::{
    PcuDispatchIntegerBinaryOp,
    PcuHostArgument,
    PcuBindingRef,
    PcuRangePolicy,
    PcuScalar,
    PcuScalarType,
};
#[rustfmt::skip]
use crate::{
    ffi,
    MlxEncodedArray,
    MlxEncodedCompletion,
    MlxError,
    MlxSession,
};
/// Retained independently proved MLX-owned U32-limb checked integer primitive.
pub struct MlxCheckedIntegerControl {
    native: ffi::CheckedInteger,
    session: MlxSession,
    scalar: PcuScalarType,
    count: usize,
    inputs: [usize; 2],
    full_inputs: [usize; 2],
}
impl MlxCheckedIntegerControl {
    /// Borrows exact-session inputs into a fresh completed immutable private output.
    ///
    /// # Errors
    /// Rejects scalar/span/session before device work; returns fatal checked or native failures.
    pub fn execute_resident(
        &mut self,
        inputs: [&MlxEncodedArray; 2],
    ) -> Result<MlxEncodedCompletion, MlxError> {
        self.native.reset_write_fact();
        for (input, count) in inputs.iter().zip(self.full_inputs) {
            if input.scalar_type() != self.scalar {
                return Err(MlxError::UnsupportedScalar(input.scalar_type()));
            }
            if input.element_count() != count {
                return Err(MlxError::InvalidExtent);
            }
            if !input.same_session(&self.session) {
                return Err(MlxError::ForeignSession);
            }
            input.validate_access_available()?;
        }
        let (output, recovered) = self
            .native
            .execute_encoded(inputs.map(MlxEncodedArray::native))?;
        Ok(self.session.complete_encoded(output, recovered))
    }
    /// Stages both preflighted typed prefixes once into the same retained native primitive.
    ///
    /// # Errors
    /// Returns unsupported logical type/extent before staging, or checked/native failure.
    pub fn execute_encoded<T: PcuScalar>(
        &mut self,
        inputs: [&[T]; 2],
    ) -> Result<MlxEncodedCompletion, MlxError> {
        self.native.reset_write_fact();
        if T::TYPE != self.scalar {
            return Err(MlxError::UnsupportedScalar(T::TYPE));
        }
        if self.inputs != self.full_inputs {
            return Err(MlxError::InvalidExtent);
        }
        if inputs
            .iter()
            .zip(self.inputs)
            .any(|(input, count)| input.len() < count)
        {
            return Err(MlxError::InvalidExtent);
        }
        let left = self.session.upload_encoded(&inputs[0][..self.inputs[0]])?;
        let right = self.session.upload_encoded(&inputs[1][..self.inputs[1]])?;
        self.execute_resident([&left, &right])
    }
    /// Publishes the terminal prefix and reports observable recovered Clamp faults.
    ///
    /// # Errors
    /// Returns unsupported scalar/span before launch; fatal/private failures preserve output.
    pub fn call<T: PcuScalar>(
        &mut self,
        inputs: [&[T]; 2],
        output: &mut [T],
    ) -> Result<(), MlxError> {
        self.native.reset_write_fact();
        if T::TYPE != self.scalar {
            return Err(MlxError::UnsupportedScalar(T::TYPE));
        }
        if output.len() < self.count
            || inputs
                .iter()
                .zip(self.inputs)
                .any(|(input, count)| input.len() < count)
        {
            return Err(MlxError::InvalidExtent);
        }
        let left = PcuHostArgument::read(PcuBindingRef::new(0, 0), &inputs[0][..self.inputs[0]]);
        let right = PcuHostArgument::read(PcuBindingRef::new(0, 1), &inputs[1][..self.inputs[1]]);
        let mut output =
            PcuHostArgument::read_write(PcuBindingRef::new(0, 2), &mut output[..self.count]);
        self.native.execute_host(
            [left.bytes(), right.bytes()],
            output.bytes_mut().ok_or(MlxError::InvalidExtent)?,
        )
    }
    /// Latest private payload work, distinct from writes to any preexisting immutable owner.
    #[must_use]
    pub const fn last_call_may_have_written(&self) -> bool {
        self.native.may_have_written()
    }
    #[must_use]
    pub const fn last_call_may_have_written_existing_encoded_owner(&self) -> bool {
        false
    }
    /// Actual pending resource/session quarantine after uncertain completion.
    #[must_use]
    pub fn last_call_completion_uncertain(&self) -> bool {
        self.native.completion_uncertain()
    }
}
impl MlxSession {
    /// Compiles the independently synthesized exact integer primitive and completes cold warming.
    /// No generic provider offer or source admission is inferred from this direct control API.
    ///
    /// # Errors
    /// Returns unsupported type/span, actual shader compilation or terminal completion failure.
    #[allow(clippy::too_many_arguments)] // Independent logical scalar, integer operation, range and both input roles.
    pub fn prepare_checked_integer_control(
        &self,
        scalar: PcuScalarType,
        operation: PcuDispatchIntegerBinaryOp,
        range: PcuRangePolicy,
        count: usize,
        inputs: [usize; 2],
        broadcast: [bool; 2],
    ) -> Result<MlxCheckedIntegerControl, MlxError> {
        self.prepare_integer_control(scalar, operation, range, count, inputs, broadcast, None)
    }
    /// Freezes full resident operand shapes independently of host minimum read spans.
    ///
    /// # Errors
    /// Rejects unsupported logical type, short/overflowing capacities or native preparation failure.
    #[allow(clippy::too_many_arguments)] // Operation/range, logical output and both exact native shapes are independent.
    pub fn prepare_checked_integer_control_with_input_extents(
        &self,
        scalar: PcuScalarType,
        operation: PcuDispatchIntegerBinaryOp,
        range: PcuRangePolicy,
        count: usize,
        inputs: [usize; 2],
        broadcast: [bool; 2],
        full: [usize; 2],
    ) -> Result<MlxCheckedIntegerControl, MlxError> {
        self.prepare_integer_control(
            scalar,
            operation,
            range,
            count,
            inputs,
            broadcast,
            Some(full),
        )
    }
    #[allow(clippy::too_many_arguments)] // Private shared constructor keeps exact and full-shape native lifecycles identical.
    fn prepare_integer_control(
        &self,
        scalar: PcuScalarType,
        operation: PcuDispatchIntegerBinaryOp,
        range: PcuRangePolicy,
        count: usize,
        inputs: [usize; 2],
        broadcast: [bool; 2],
        full: Option<[usize; 2]>,
    ) -> Result<MlxCheckedIntegerControl, MlxError> {
        let native = full.map_or_else(
            || {
                self.prepare_checked_integer_native(
                    scalar, operation, range, count, inputs, broadcast,
                )
            },
            |full| {
                self.prepare_checked_integer_native_with_input_extents(
                    scalar, operation, range, count, inputs, broadcast, full,
                )
            },
        )?;
        Ok(MlxCheckedIntegerControl {
            native,
            session: self.clone(),
            scalar,
            count,
            inputs,
            full_inputs: full.unwrap_or(inputs),
        })
    }
}
