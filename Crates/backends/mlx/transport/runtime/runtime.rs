//! Explicit bounded transport primitive with authentic immutable MLX input/output ownership.
#[rustfmt::skip]
use crate::{
    ffi,
    MlxEncodedArray,
    MlxError,
    MlxSession,
    MlxTransportPlan,
};

/// One retained MLX custom primitive; no aggregate/facade admission is inferred.
pub struct MlxPreparedTransportKernel {
    session: MlxSession,
    plan: MlxTransportPlan,
    native: ffi::Transport,
}
/// Terminal private outputs in the exact plan writer order.
/// Missing array slots are absent outputs, not fabricated native resources.
pub struct MlxTransportCompletion {
    outputs: [Option<MlxEncodedArray>; 2],
    count: usize,
}
impl MlxTransportCompletion {
    #[must_use]
    pub const fn output_count(&self) -> usize {
        self.count
    }
    #[must_use]
    pub const fn outputs(&self) -> &[Option<MlxEncodedArray>; 2] {
        &self.outputs
    }
    #[must_use]
    pub fn into_outputs(self) -> [Option<MlxEncodedArray>; 2] {
        self.outputs
    }
}
impl MlxSession {
    /// Prepares a detached exact transport plan and completes one real cold prime.
    /// All prime input bytes and result arrays belong to MLX and are checked-released.
    /// # Errors
    /// Returns native construction, extent, completion or checked cleanup failure.
    pub fn prepare_transport_plan(
        &self,
        plan: MlxTransportPlan,
    ) -> Result<MlxPreparedTransportKernel, MlxError> {
        self.prepare_transport_layout(plan, None)
    }
    /// Freezes full physical input shapes while preserving the plan's logical read/output spans.
    /// Cold priming uploads full-size synthetic native inputs; this has a proportional cold
    /// host/device memory cost. Warm replay uses exact shapes and no input view construction.
    /// # Errors
    /// Rejects wrong arity, short or overflowing capacity before SDK construction.
    pub fn prepare_transport_plan_with_input_extents(
        &self,
        plan: MlxTransportPlan,
        extents: &[usize],
    ) -> Result<MlxPreparedTransportKernel, MlxError> {
        let full = plan.assess_input_extents(extents)?;
        self.prepare_transport_layout(plan, Some(full))
    }
    fn prepare_transport_layout(
        &self,
        plan: MlxTransportPlan,
        full: Option<[usize; 4]>,
    ) -> Result<MlxPreparedTransportKernel, MlxError> {
        let input_extents = full.unwrap_or_else(|| plan.input_element_counts());
        let native = match full {
            None => self.prepare_transport_native(&plan),
            Some(_) => self.prepare_transport_native_with_input_extents(
                &plan,
                &input_extents[..plan.inputs().len()],
            ),
        }?;
        let prepared = MlxPreparedTransportKernel {
            session: self.clone(),
            plan,
            native,
        };
        let mut inputs: [Option<MlxEncodedArray>; 4] = std::array::from_fn(|_| None);
        for (slot, &count) in prepared
            .prepared_input_element_counts()
            .iter()
            .take(prepared.plan.inputs().len())
            .enumerate()
        {
            let bytes = count
                .checked_mul(usize::from(prepared.plan.scalar_type().bit_width()) / 8)
                .ok_or(MlxError::InvalidExtent)?;
            inputs[slot] = Some(self.upload_transport_bytes(
                prepared.plan.scalar_type(),
                count,
                &vec![0; bytes],
            )?);
        }
        let first = inputs[0].as_ref().ok_or(MlxError::InvalidExtent)?;
        let mut refs = [first; 4];
        for (slot, input) in inputs.iter().enumerate() {
            if let Some(input) = input {
                refs[slot] = input;
            }
        }
        let completed = prepared.execute(&refs[..prepared.plan.inputs().len()])?;
        for output in completed.into_outputs().into_iter().flatten() {
            output.release()?;
        }
        for input in inputs.into_iter().flatten() {
            input.release()?;
        }
        Ok(prepared)
    }
}
impl MlxPreparedTransportKernel {
    #[must_use]
    pub const fn plan(&self) -> &MlxTransportPlan {
        &self.plan
    }
    #[must_use]
    pub const fn session(&self) -> &MlxSession {
        &self.session
    }
    /// Exact cold full native input capacities, in actual unique snapshot order.
    #[must_use]
    pub fn prepared_input_element_counts(&self) -> &[usize] {
        self.native.input_element_counts()
    }
    /// Replays actual unique original-input snapshots in the retained cold order.
    /// Input type, exact physical extent and authentic session validate before native work.
    /// # Errors
    /// Returns preflight, native or unknown completion; uncertainty quarantines actual owners.
    /// Previous immutable caller owners remain distinct from fresh private results.
    pub fn execute(&self, inputs: &[&MlxEncodedArray]) -> Result<MlxTransportCompletion, MlxError> {
        if inputs.len() != self.plan.inputs().len() {
            return Err(MlxError::InvalidExtent);
        }
        let first = inputs.first().ok_or(MlxError::InvalidExtent)?.native();
        let mut native = [first; 4];
        for (slot, input) in inputs.iter().enumerate() {
            native[slot] = input.native();
        }
        let outputs = self.native.execute(&native[..inputs.len()])?;
        Ok(MlxTransportCompletion {
            outputs: outputs
                .map(|output| output.map(|output| self.session.wrap_transport_output(output))),
            count: self.plan.outputs().len(),
        })
    }
}
