//! Real retained native execution; ordinary offers/facade publication remain ungranted.
#[rustfmt::skip]
use fusion_pcu::{
    PcuBindingRef,
    PcuExecutionFault,
    PcuScalarType,
};
#[rustfmt::skip]
use crate::{
    ffi,
    MlxCheckedMapPlan,
    MlxEncodedArray,
    MlxError,
    MlxSession,
};

/// One immutable MLX primitive, exact original plan and actual session.
pub struct MlxPreparedCheckedMapKernel {
    session: MlxSession,
    native: ffi::Composed,
}
/// Private terminal outputs plus an observable recovered notice, never a silent success.
pub struct MlxCheckedMapCompletion {
    outputs: [Option<MlxEncodedArray>; 2],
    fault: Option<PcuExecutionFault>,
}
impl MlxCheckedMapCompletion {
    #[must_use]
    pub fn into_outputs(self) -> ([Option<MlxEncodedArray>; 2], Option<PcuExecutionFault>) {
        (self.outputs, self.fault)
    }
}
impl MlxSession {
    /// Benchmark-only independent saved-stage workload body, sharing scalar primitives.
    /// The exact plan is used for cold metadata and status/terminal law, never body lowering.
    /// This is an orchestration control, not an independent numeric oracle or optimal kernel.
    /// Synthetic cold priming uploads and SDK allocation/JIT costs remain unobserved.
    /// # Errors
    /// Refuses a different workload, native shape/source or terminal cleanup failure.
    #[cfg(feature = "benchmark-control")]
    pub fn prepare_native_saved_stage_control(
        &self,
        plan: MlxCheckedMapPlan,
    ) -> Result<MlxPreparedCheckedMapKernel, MlxError> {
        let (header, body) = super::control::sources(&plan)?;
        let native = self.prepare_native_checked_map_control_internal(plan, header, body)?;
        let prepared = MlxPreparedCheckedMapKernel {
            session: self.clone(),
            native,
        };
        prepared.prime()?;
        Ok(prepared)
    }
    /// Prepare exact minimal physical input shapes and complete one genuine cold prime.
    ///
    /// Synthetic native inputs have proportional cold allocation/upload cost. A valid
    /// arithmetic fault from synthetic values is not an admission proof or caller result;
    /// it still requires terminal completion and checked private cleanup before return.
    /// # Errors
    /// Refuses unsupported source, shape/type limits, native failures or cleanup uncertainty.
    pub fn prepare_checked_map_plan(
        &self,
        plan: MlxCheckedMapPlan,
    ) -> Result<MlxPreparedCheckedMapKernel, MlxError> {
        let native = self.prepare_composed_native(plan)?;
        let prepared = MlxPreparedCheckedMapKernel {
            session: self.clone(),
            native,
        };
        prepared.prime()?;
        Ok(prepared)
    }
}
impl MlxPreparedCheckedMapKernel {
    #[must_use]
    pub const fn plan(&self) -> &MlxCheckedMapPlan {
        self.native.plan()
    }
    #[must_use]
    pub const fn session(&self) -> &MlxSession {
        &self.session
    }
    #[must_use]
    pub const fn input_bindings(&self) -> &[PcuBindingRef] {
        self.native.input_bindings()
    }
    #[must_use]
    pub const fn output_bindings(&self) -> &[PcuBindingRef] {
        self.native.output_bindings()
    }
    #[must_use]
    pub fn prepared_input_element_counts(&self) -> &[usize] {
        self.native.input_element_counts()
    }
    /// Replay only actual initial snapshots; store-supplied resources have no input owner.
    ///
    /// Every sibling completes, exact status validates, and fallible input/status cleanup
    /// precedes returned private owners. Fatal arithmetic returns no output; useful Clamp
    /// returns completed owners and a notice for a future transactional publisher.
    /// # Errors
    /// Refuses wrong type/exact span/actual session, native failure or fatal arithmetic.
    /// Unknown completion retains actual pending owners and poisons their session.
    pub fn execute(
        &self,
        inputs: &[&MlxEncodedArray],
    ) -> Result<MlxCheckedMapCompletion, MlxError> {
        if inputs.len() != self.input_bindings().len() {
            return Err(MlxError::InvalidExtent);
        }
        let first = inputs.first().ok_or(MlxError::InvalidExtent)?.native();
        let mut native = [first; 4];
        for (slot, input) in inputs.iter().enumerate() {
            native[slot] = input.native();
        }
        let (outputs, fault) = self.native.execute(&native[..inputs.len()])?;
        Ok(MlxCheckedMapCompletion {
            outputs: outputs
                .map(|array| array.map(|array| self.session.wrap_transport_output(array))),
            fault,
        })
    }
    fn prime(&self) -> Result<(), MlxError> {
        let scalar = self.plan().value_type().scalar_type();
        let width = usize::from(scalar.bit_width()) / 8;
        let mut word = [0; 16];
        match scalar {
            PcuScalarType::F32 => word[..4].copy_from_slice(&0x3f80_0000_u32.to_le_bytes()),
            PcuScalarType::F64 => {
                word[..8].copy_from_slice(&0x3ff0_0000_0000_0000_u64.to_le_bytes());
            }
            _ => word[0] = 1,
        }
        let mut inputs: [Option<MlxEncodedArray>; 4] = std::array::from_fn(|_| None);
        for (slot, &count) in self.prepared_input_element_counts().iter().enumerate() {
            let mut bytes = vec![0; count.checked_mul(width).ok_or(MlxError::InvalidExtent)?];
            for lane in bytes.chunks_exact_mut(width) {
                lane.copy_from_slice(&word[..width]);
            }
            inputs[slot] = Some(self.session.upload_transport_bytes(scalar, count, &bytes)?);
        }
        let first = inputs[0].as_ref().ok_or(MlxError::InvalidExtent)?;
        let mut refs = [first; 4];
        for (slot, input) in inputs.iter().enumerate() {
            if let Some(input) = input {
                refs[slot] = input;
            }
        }
        match self.execute(&refs[..self.input_bindings().len()]) {
            Ok(completed) => {
                for output in completed.into_outputs().0.into_iter().flatten() {
                    output.release()?;
                }
            }
            Err(MlxError::Arithmetic(_)) => {}
            Err(error) => return Err(error),
        }
        for input in inputs.into_iter().flatten() {
            input.release()?;
        }
        Ok(())
    }
}
