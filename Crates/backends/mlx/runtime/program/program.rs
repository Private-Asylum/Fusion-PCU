//! Safe opaque source binding: ordinary RAM stages automatically into the frozen session.

#[rustfmt::skip]
use fusion_pcu::{
    PcuScalar,
    PcuScalarType,
};
use fusion_pcu::dialect::tensor::ValueId;
#[rustfmt::skip]
use super::{
    MlxArray,
    MlxError,
    MlxPreparedProgram,
};

/// A copied ordinary input or an immutable retained array from the exact prepared session.
/// No raw buffer, borrowed native pointer or cross-provider import is exposed.
pub enum MlxProgramInput<'a, T: PcuScalar> {
    Host(&'a [T]),
    Resident(&'a MlxArray),
}

impl MlxPreparedProgram {
    /// Binds current RAM or opaque inputs to the retained authentic selected source program.
    /// Complete identities, scalar, shapes and quarantine checks precede every host upload.
    /// The returned array retains terminal initialized backing and its originating session.
    ///
    /// # Errors
    /// Rejects unsupported scalar, duplicate/missing inputs, short or oversized host extents,
    /// foreign opaque owners, changed shapes, quarantine and native execution failures.
    pub fn execute_mixed<T: PcuScalar>(
        &self,
        inputs: &[(ValueId, MlxProgramInput<'_, T>)],
    ) -> Result<MlxArray, MlxError> {
        let ordered = self.validate_inputs(inputs)?;
        let session = self.matmul.session();
        let shapes = [self.matmul.plan.left_shape, self.matmul.plan.right_shape];
        let stage = |input: &MlxProgramInput<'_, T>, shape| match input {
            MlxProgramInput::Host(values) => session.upload_typed(shape, values),
            MlxProgramInput::Resident(array) => Ok((*array).clone()),
        };
        let staged = [stage(ordered[0], shapes[0])?, stage(ordered[1], shapes[1])?];
        let [left, right] = self.matmul.plan.inputs();
        session.execute_program(self, &[(left, &staged[0]), (right, &staged[1])])
    }

    /// Executes ordinary input slices and explicitly materializes a complete host prefix.
    /// Destination type and extent reject before staging; all foreign work precedes publication.
    ///
    /// # Errors
    /// Returns binding/scalar/shape/quarantine/native failure, preserving destination on error.
    pub fn execute_host_into<T: PcuScalar>(
        &self,
        inputs: &[(ValueId, &[T])],
        output: &mut [T],
    ) -> Result<(), MlxError> {
        if T::TYPE != PcuScalarType::F32 {
            return Err(MlxError::UnsupportedScalar(T::TYPE));
        }
        let [rows, columns] = self.matmul.plan.shape();
        let count = rows.checked_mul(columns).ok_or(MlxError::InvalidExtent)?;
        if output.len() < count {
            return Err(MlxError::InvalidExtent);
        }
        let [first, second] = inputs else {
            return Err(MlxError::InvalidRequest(
                "expected two source input bindings".into(),
            ));
        };
        let arguments = [
            (first.0, MlxProgramInput::Host(first.1)),
            (second.0, MlxProgramInput::Host(second.1)),
        ];
        self.execute_mixed(&arguments)?.read_into_typed(output)
    }

    fn validate_inputs<'a, T: PcuScalar>(
        &self,
        inputs: &'a [(ValueId, MlxProgramInput<'_, T>)],
    ) -> Result<[&'a MlxProgramInput<'a, T>; 2], MlxError> {
        if T::TYPE != PcuScalarType::F32 {
            return Err(MlxError::UnsupportedScalar(T::TYPE));
        }
        let [first, second] = inputs else {
            return Err(MlxError::InvalidRequest(
                "expected two source input bindings".into(),
            ));
        };
        let [left, right] = self.matmul.plan.inputs();
        let ordered = if first.0 == left && second.0 == right {
            [&first.1, &second.1]
        } else if first.0 == right && second.0 == left {
            [&second.1, &first.1]
        } else {
            return Err(MlxError::InvalidRequest(
                "source input identities mismatch".into(),
            ));
        };
        let session = self.matmul.session();
        session.validate_access_available()?;
        for (input, shape) in ordered
            .iter()
            .zip([self.matmul.plan.left_shape, self.matmul.plan.right_shape])
        {
            match input {
                MlxProgramInput::Host(values) => {
                    if shape[0].checked_mul(shape[1]) != Some(values.len()) {
                        return Err(MlxError::InvalidExtent);
                    }
                }
                MlxProgramInput::Resident(array) => {
                    if !session.same_session(array.session()) {
                        return Err(MlxError::ForeignSession);
                    }
                    if array.shape() != shape {
                        return Err(MlxError::InvalidExtent);
                    }
                    array.validate_access_available()?;
                }
            }
        }
        Ok(ordered)
    }
}
