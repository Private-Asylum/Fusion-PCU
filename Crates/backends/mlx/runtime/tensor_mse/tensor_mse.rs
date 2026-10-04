//! Frozen exact mse graph execution through actual MLX-owned immutable carriers.
#[rustfmt::skip]
use std::sync::Arc;
#[rustfmt::skip]
use fusion_pcu::{
    PcuImplementationRequirements,

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
    MlxCheckedTensorMsePlan,
};
/// Authentic selected graph, unique actual input roles and retained native checker/session.
pub struct MlxPreparedTensorMseProgram {
    program: Arc<TensorOwnedSelectedProgram>,
    session: MlxSession,
    plan: MlxCheckedTensorMsePlan,
    native: ffi::StrictMse,
}
impl MlxSession {
    /// Freezes one checked mse effect and preserves an unused effect before identity output.
    ///
    /// # Errors
    /// Rejects unsupported graph/policy/extent or native preparation/compilation failure.
    pub fn prepare_tensor_mse_program(
        &self,
        program: Arc<TensorOwnedSelectedProgram>,
        requirements: PcuImplementationRequirements,
    ) -> Result<MlxPreparedTensorMseProgram, MlxError> {
        let plan = MlxCheckedTensorMsePlan::assess_program(&program, requirements)
            .map_err(MlxError::Unsupported)?;
        self.validate_access_available()?;
        let native = ffi::StrictMse::prepare(
            &self.0.native,
            plan.scalar_type(),
            requirements.float_underflow,
            plan.input_element_count(0).ok_or(MlxError::InvalidExtent)?,
        )?;
        Ok(MlxPreparedTensorMseProgram {
            program,
            session: self.clone(),
            plan,
            native,
        })
    }
}
impl MlxPreparedTensorMseProgram {
    #[must_use]
    pub fn program(&self) -> &TensorOwnedSelectedProgram {
        &self.program
    }
    #[must_use]
    pub const fn plan(&self) -> &MlxCheckedTensorMsePlan {
        &self.plan
    }
    #[must_use]
    pub const fn session(&self) -> &MlxSession {
        &self.session
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
                    if Some(bytes.len()) != self.plan.input_byte_len(slot) {
                        return Err(MlxError::InvalidExtent);
                    }
                }
                MlxCheckedProgramInput::Resident(array) => {
                    if array.scalar_type() != self.plan.scalar_type() {
                        return Err(MlxError::UnsupportedScalar(array.scalar_type()));
                    }
                    if Some(array.element_count()) != self.plan.input_element_count(slot) {
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
                staged[slot] = Some(
                    self.session.upload_encoded_bytes(
                        *scalar,
                        self.plan
                            .input_element_count(slot)
                            .ok_or(MlxError::InvalidExtent)?,
                        bytes,
                    )?,
                );
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
        let result = self.native.execute(
            arrays[left].ok_or(MlxError::InvalidExtent)?.native(),
            arrays[right].ok_or(MlxError::InvalidExtent)?.native(),
        );
        let (output, recovered) = match result {
            Ok(completed) => completed,
            Err(error) => {
                for array in staged.into_iter().flatten() {
                    array.release()?;
                }
                return Err(error);
            }
        };
        if let Some(fault) = recovered {
            output.release()?;
            return Err(MlxError::Arithmetic(fault));
        }
        let completed = if let Some(slot) = self.plan.identity_output() {
            output.release()?;
            arrays[slot].ok_or(MlxError::InvalidExtent)?.clone()
        } else {
            self.session.wrap_encoded(output)
        };
        for array in staged.into_iter().flatten() {
            array.release()?;
        }
        Ok(completed)
    }
}

#[cfg(test)]
#[path = "tests/tests.rs"]
mod tests;
