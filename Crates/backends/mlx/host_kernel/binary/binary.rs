//! Direct checked six-format binary control with actual MLX-owned immutable array scheduling.
#[path = "admission/admission.rs"]
mod admission;
#[rustfmt::skip]
pub use admission::{
    MlxCheckedBinaryPlan,
    MlxCheckedBinaryBackend,
    MlxPreparedBinaryHostKernel,
    MlxBinaryInput,
};
#[rustfmt::skip]
use fusion_pcu::{
    PcuBindingRef,
    PcuDispatchFloatBinaryOp,
    PcuFloatUnderflowPolicy,
    PcuHostArgument,
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

/// Direct native checker control, independently qualified from source/neutral admission.
pub struct MlxCheckedBinaryControl {
    native: ffi::CheckedBinary,
    session: MlxSession,
    scalar: PcuScalarType,
    count: usize,
    inputs: [usize; 2],
    full_inputs: [usize; 2],
}
impl MlxCheckedBinaryControl {
    /// Borrows both exact-session carriers into a fresh completed private output.
    ///
    /// # Errors
    /// Rejects dtype, extent or session mismatch before device work; returns checked/native faults.
    pub fn execute_resident(
        &mut self,
        inputs: [&MlxEncodedArray; 2],
    ) -> Result<MlxEncodedCompletion, MlxError> {
        self.native.reset_write_fact();
        if inputs
            .iter()
            .any(|input| !input.same_session(&self.session))
        {
            return Err(MlxError::ForeignSession);
        }
        let (output, recovered) = self
            .native
            .execute_encoded(inputs.map(MlxEncodedArray::native))?;
        Ok(self.session.complete_encoded(output, recovered))
    }
    /// Stages typed input prefixes, then completes the same immutable native control boundary.
    ///
    /// # Errors
    /// Returns unsupported type/extent before staging or checked/native completion failure.
    pub fn execute_encoded<T: PcuScalar>(
        &mut self,
        inputs: [&[T]; 2],
    ) -> Result<MlxEncodedCompletion, MlxError> {
        self.native.reset_write_fact();
        if self.inputs != self.full_inputs {
            return Err(MlxError::InvalidExtent);
        }
        if T::TYPE != self.scalar {
            return Err(MlxError::UnsupportedScalar(T::TYPE));
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
    /// Publishes a completed host prefix, including observable recovered Clamp faults.
    ///
    /// # Errors
    /// Returns scalar/extent, checked arithmetic or native failure; fatal faults preserve output.
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
        self.native.execute(
            [left.bytes(), right.bytes()],
            output.bytes_mut().ok_or(MlxError::InvalidExtent)?,
        )
    }
    /// Whether private GPU work may have written during the latest call.
    #[must_use]
    pub const fn last_call_may_have_written(&self) -> bool {
        self.native.may_have_written()
    }
    /// Private immutable outputs never write any preexisting encoded owner.
    #[must_use]
    pub const fn last_call_may_have_written_existing_encoded_owner(&self) -> bool {
        false
    }
    /// Whether actual pending owners/session are quarantined after uncertain completion.
    #[must_use]
    pub fn last_call_completion_uncertain(&self) -> bool {
        self.native.completion_uncertain()
    }
}
impl MlxSession {
    /// Compiles exact six-format binary integer synthesis and completes its cold cache warming call.
    ///
    /// # Errors
    /// Returns unsupported type, extent, compilation or completion failure.
    #[allow(clippy::too_many_arguments)] // Independent frozen dtype, operation, policies and both operand roles.
    pub fn prepare_checked_binary_control(
        &self,
        scalar: PcuScalarType,
        operation: PcuDispatchFloatBinaryOp,
        underflow: PcuFloatUnderflowPolicy,
        range: PcuRangePolicy,
        count: usize,
        inputs: [usize; 2],
        broadcast: [bool; 2],
    ) -> Result<MlxCheckedBinaryControl, MlxError> {
        self.prepare_binary_control(
            scalar, operation, underflow, range, count, inputs, broadcast, None,
        )
    }
    /// Freezes exact full native operand shapes while retaining minimum host-read spans.
    /// Full-capacity synthetic priming is cold work; warm resident calls create no input view.
    ///
    /// # Errors
    /// Rejects unsupported tuple, short/overflowing capacity or native preparation failure.
    #[allow(clippy::too_many_arguments)] // Independent exact operation tuple and both full operand capacities.
    pub fn prepare_checked_binary_control_with_input_extents(
        &self,
        scalar: PcuScalarType,
        operation: PcuDispatchFloatBinaryOp,
        underflow: PcuFloatUnderflowPolicy,
        range: PcuRangePolicy,
        count: usize,
        inputs: [usize; 2],
        broadcast: [bool; 2],
        full: [usize; 2],
    ) -> Result<MlxCheckedBinaryControl, MlxError> {
        self.prepare_binary_control(
            scalar,
            operation,
            underflow,
            range,
            count,
            inputs,
            broadcast,
            Some(full),
        )
    }
    #[allow(clippy::too_many_arguments)] // Shared old-exact and additive full-shape cold construction.
    fn prepare_binary_control(
        &self,
        scalar: PcuScalarType,
        operation: PcuDispatchFloatBinaryOp,
        underflow: PcuFloatUnderflowPolicy,
        range: PcuRangePolicy,
        count: usize,
        inputs: [usize; 2],
        broadcast: [bool; 2],
        full: Option<[usize; 2]>,
    ) -> Result<MlxCheckedBinaryControl, MlxError> {
        let native = full.map_or_else(
            || {
                self.prepare_checked_binary_native(
                    scalar, operation, underflow, range, count, inputs, broadcast,
                )
            },
            |full| {
                self.prepare_checked_binary_native_with_input_extents(
                    scalar, operation, underflow, range, count, inputs, broadcast, full,
                )
            },
        )?;
        Ok(MlxCheckedBinaryControl {
            native,
            session: self.clone(),
            scalar,
            count,
            inputs,
            full_inputs: full.unwrap_or(inputs),
        })
    }
}
