//! Frozen exact binary graph execution through actual MLX-owned immutable carriers.
#[rustfmt::skip]
use std::sync::Arc;
#[rustfmt::skip]
use fusion_pcu::{
    PcuImplementationRequirements,
    PcuImplementationId,
};
#[rustfmt::skip]
use fusion_pcu::dialect::tensor::{
    TensorOwnedSelectedProgram,
    ValueId,
};
#[rustfmt::skip]
use crate::{
    ffi,
    MlxSession,
    MlxError,
    MlxEncodedArray,
    MlxCheckedProgramInput,
    MlxCheckedTensorIntegerPlan,
};
/// Authentic selected graph, unique actual input roles and retained native checker/session.
pub struct MlxPreparedTensorIntegerProgram {
    program: Arc<TensorOwnedSelectedProgram>,
    session: MlxSession,
    plan: MlxCheckedTensorIntegerPlan,
    native: ffi::CheckedInteger,
}
impl MlxSession {
    /// Freezes one checked binary effect and preserves an unused effect before identity output.
    ///
    /// # Errors
    /// Rejects unsupported graph/policy/extent or native preparation/compilation failure.
    pub fn prepare_tensor_integer_program(
        &self,
        program: Arc<TensorOwnedSelectedProgram>,
        requirements: PcuImplementationRequirements,
    ) -> Result<MlxPreparedTensorIntegerProgram, MlxError> {
        let plan = MlxCheckedTensorIntegerPlan::assess_program(&program, requirements)
            .map_err(MlxError::Unsupported)?;
        self.validate_access_available()?;
        let native = self.prepare_checked_integer_native(
            plan.scalar_type(),
            plan.operation(),
            requirements.range_policy,
            plan.element_count(),
            [plan.element_count(); 2],
            [false; 2],
        )?;
        Ok(MlxPreparedTensorIntegerProgram {
            program,
            session: self.clone(),
            plan,
            native,
        })
    }
}
impl MlxPreparedTensorIntegerProgram {
    #[must_use]
    pub fn program(&self) -> &TensorOwnedSelectedProgram {
        &self.program
    }
    #[must_use]
    pub const fn plan(&self) -> &MlxCheckedTensorIntegerPlan {
        &self.plan
    }
    #[must_use]
    pub const fn session(&self) -> &MlxSession {
        &self.session
    }
    #[must_use]
    pub fn implementation_id(&self) -> Option<PcuImplementationId> {
        self.session
            .identity()
            .map(|device| self.plan.implementation_id(device))
    }
    /// Evaluates exact frozen source inputs once each, then privately completes payload/status.
    ///
    /// # Errors
    /// Preflights every binding/type/span/actual session before host staging, then returns
    /// checked fatal or native failure. Unknown completion retains actual pending owners.
    pub fn execute_mixed(
        &self,
        inputs: &[(ValueId, MlxCheckedProgramInput<'_>)],
    ) -> Result<MlxEncodedArray, MlxError> {
        self.native.reset_write_fact();
        self.session.validate_access_available()?;
        let bindings = self.plan.input_values();
        if inputs.len() != bindings.len() {
            return Err(MlxError::InvalidExtent);
        }
        let mut ordered = [None; 2];
        for (id, input) in inputs {
            let slot = bindings
                .iter()
                .position(|binding| binding == id)
                .ok_or(MlxError::InvalidExtent)?;
            if ordered[slot].replace(input).is_some() {
                return Err(MlxError::InvalidExtent);
            }
            match input {
                MlxCheckedProgramInput::Host { scalar, bytes } => {
                    if *scalar != self.plan.scalar_type() {
                        return Err(MlxError::UnsupportedScalar(*scalar));
                    }
                    if bytes.len() != self.plan.byte_len() {
                        return Err(MlxError::InvalidExtent);
                    }
                }
                MlxCheckedProgramInput::Resident(array) => {
                    if array.scalar_type() != self.plan.scalar_type() {
                        return Err(MlxError::UnsupportedScalar(array.scalar_type()));
                    }
                    if array.element_count() != self.plan.element_count() {
                        return Err(MlxError::InvalidExtent);
                    }
                    if !array.same_session(&self.session) {
                        return Err(MlxError::ForeignSession);
                    }
                    array.validate_access_available()?;
                }
            }
        }
        let mut staged = [None, None];
        for slot in 0..bindings.len() {
            if let Some(MlxCheckedProgramInput::Host { scalar, bytes }) = ordered[slot] {
                staged[slot] = Some(self.session.upload_encoded_bytes(
                    *scalar,
                    self.plan.element_count(),
                    bytes,
                )?);
            }
        }
        let mut arrays = [None; 2];
        for slot in 0..bindings.len() {
            arrays[slot] = Some(match ordered[slot] {
                Some(MlxCheckedProgramInput::Host { .. }) => {
                    staged[slot].as_ref().ok_or(MlxError::InvalidExtent)?
                }
                Some(MlxCheckedProgramInput::Resident(array)) => *array,
                None => return Err(MlxError::InvalidExtent),
            });
        }
        let [left, right] = self.plan.operand_inputs();
        let (output, recovered) = self.native.execute_encoded([
            arrays[left].ok_or(MlxError::InvalidExtent)?.native(),
            arrays[right].ok_or(MlxError::InvalidExtent)?.native(),
        ])?;
        if let Some(fault) = recovered {
            return Err(MlxError::Arithmetic(fault));
        }
        if let Some(slot) = self.plan.identity_output() {
            // Checked unused work completed and was scanned; immutable input ownership can
            // now escape without a fabricated CarrierCopy or an unobserved fault.
            drop(output);
            return Ok(arrays[slot].ok_or(MlxError::InvalidExtent)?.clone());
        }
        Ok(self.session.wrap_encoded(output))
    }
}
