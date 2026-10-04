//! Cold selected graph decomposition retaining authoritative parent policies and liveness.
use std::{rc::Rc, sync::Arc};
use fusion_pcu::{PcuImplementationRequirements, PcuScalarType, PcuRangePolicy, PcuReproducibility};
use fusion_pcu::dialect::tensor::{
    OpDescriptor, TensorOwnedSelectedProgram, TensorUnsupportedReason, ValueId,
};
use crate::{MlxCheckedTensorPlan, MlxCheckedTensorBinaryPlan, MlxSelectedNumericalTensorPlan};

#[path = "producer.rs"]
mod producer;
use producer::Producer;

#[derive(Clone)]
enum Envelope {
    Unary(MlxCheckedTensorPlan),
    Binary(MlxCheckedTensorBinaryPlan),
    Numerical(MlxSelectedNumericalTensorPlan),
}
#[derive(Clone)]
struct Stage {
    effect: ValueId,
    operation_index: usize,
    bindings: Rc<[(ValueId, usize)]>,
    releases: Rc<[usize]>,
    program: Arc<TensorOwnedSelectedProgram>,
    envelope: Envelope,
}
#[derive(Clone)]
struct Input {
    value: ValueId,
    slot: usize,
    scalar: PcuScalarType,
    shape: Rc<[usize]>,
    count: usize,
    bytes: usize,
}
/// Structural graph eligibility only; no native compilation, execution or scratch reuse claim.
#[derive(Clone)]
pub struct MlxSelectedTensorGraphPlan {
    program: Arc<TensorOwnedSelectedProgram>,
    requirements: PcuImplementationRequirements,
    inputs: Rc<[Input]>,
    producers: Rc<[Producer]>,
    stages: Rc<[Stage]>,
    output: ValueId,
    output_shape: Rc<[usize]>,
    output_scalar: PcuScalarType,
    output_count: usize,
    output_bytes: usize,
    output_slot: usize,
    slots: usize,
}
impl MlxSelectedTensorGraphPlan {
    /// Freeze direct-node arithmetic leaves and eight-format immutable transport producers.
    /// Parent selected liveness controls release; isolated child output pins do not extend it.
    /// # Errors
    /// Refuses fused schedules, multiple outputs, malformed producers or any unsupported leaf.
    pub fn assess_program(
        program: Arc<TensorOwnedSelectedProgram>,
        requirements: PcuImplementationRequirements,
    ) -> Result<Self, TensorUnsupportedReason> {
        if requirements.range_policy != PcuRangePolicy::Reject {
            return Err(TensorUnsupportedReason::Operation);
        }
        let [output] = program.output_values() else {
            return Err(TensorUnsupportedReason::Operation);
        };
        let output = *output;
        let output_slot = slot(&program, output)?;
        validate_output_envelope(&program, output, requirements)?;
        let metadata = output_metadata(&program, output)?;
        let (output_shape, output_scalar, output_count, output_bytes) = metadata;
        let mut inputs = Vec::with_capacity(program.input_values().len());
        let producers = assess_producers(&program, requirements)?;
        for &value in program.input_values() {
            let node = program
                .graph()
                .node(value)
                .map_err(|_| TensorUnsupportedReason::Operation)?;
            if !matches!(node.op, OpDescriptor::Input) {
                return Err(TensorUnsupportedReason::Operation);
            }
            let count = node
                .shape
                .iter()
                .try_fold(1usize, |n, &d| n.checked_mul(d))
                .filter(|&n| n != 0)
                .ok_or(TensorUnsupportedReason::Shape)?;
            let bytes = count
                .checked_mul(usize::from(node.scalar_type.bit_width()) / 8)
                .ok_or(TensorUnsupportedReason::Shape)?;
            inputs.push(Input {
                value,
                slot: slot(&program, value)?,
                scalar: node.scalar_type,
                shape: Rc::from(node.shape),
                count,
                bytes,
            });
        }
        let fragments = program
            .operation_fragments()
            .map_err(|_| TensorUnsupportedReason::Operation)?;
        let mut stages = Vec::with_capacity(fragments.len());
        for fragment in fragments {
            let effect = fragment.parent_value();
            let operation_index = fragment.parent_operation_index();
            let bindings = fragment
                .inputs()
                .iter()
                .map(|input| Ok((input.child_value, slot(&program, input.parent_value)?)))
                .collect::<Result<Vec<_>, TensorUnsupportedReason>>()?;
            let child_requirements = fragment.implementation_requirements(requirements);
            let child = Arc::new(fragment.into_program());
            let envelope = assess_leaf(Arc::clone(&child), child_requirements)?;
            let releases = program
                .operation_liveness()
                .iter()
                .filter(|life| life.last_live_node == operation_index && life.value != output)
                .map(|life| slot(&program, life.value))
                .collect::<Result<Vec<_>, _>>()?;
            stages.push(Stage {
                effect,
                operation_index,
                bindings: Rc::from(bindings),
                releases: Rc::from(releases),
                program: child,
                envelope,
            });
        }
        // The first producer proof is pure transport. Constant-fed arithmetic needs its
        // own graph/native qualification before this assessment offers that closure.
        if !producers.is_empty() && (!inputs.is_empty() || !stages.is_empty()) {
            return Err(TensorUnsupportedReason::Operation);
        }
        // Preserve the separately qualified pure transport envelope. An
        // escaped Input with computed mandatory effects is assessed stagewise.
        if stages.is_empty()
            && requirements.numerical_options.reproducibility != PcuReproducibility::Unspecified
        {
            return Err(TensorUnsupportedReason::Operation);
        }
        if inputs.is_empty() && producers.is_empty() {
            return Err(TensorUnsupportedReason::Operation);
        }
        let slots = program.operations().len();
        Ok(Self {
            program,
            requirements,
            inputs: Rc::from(inputs),
            producers: Rc::from(producers),
            stages: Rc::from(stages),
            output,
            output_shape,
            output_scalar,
            output_count,
            output_bytes,
            output_slot,
            slots,
        })
    }
    #[must_use]
    pub fn program(&self) -> &TensorOwnedSelectedProgram {
        &self.program
    }
    #[must_use]
    pub fn program_owner(&self) -> Arc<TensorOwnedSelectedProgram> {
        Arc::clone(&self.program)
    }
    #[must_use]
    pub const fn requirements(&self) -> PcuImplementationRequirements {
        self.requirements
    }
    #[must_use]
    pub const fn output(&self) -> ValueId {
        self.output
    }
    #[must_use]
    pub fn shape(&self) -> &[usize] {
        &self.output_shape
    }
    #[must_use]
    pub const fn scalar_type(&self) -> PcuScalarType {
        self.output_scalar
    }
    #[must_use]
    pub const fn element_count(&self) -> usize {
        self.output_count
    }
    #[must_use]
    pub const fn byte_len(&self) -> usize {
        self.output_bytes
    }
    #[must_use]
    pub fn input_values(&self) -> &[ValueId] {
        self.program.input_values()
    }
    #[must_use]
    pub fn stage_count(&self) -> usize {
        self.stages.len()
    }
    /// Native graph family receipt; complete source, shapes and request remain cache identity.
    #[must_use]
    pub const fn implementation_id(
        &self,
        device: fusion_pcu::PcuDeviceIdentity,
    ) -> fusion_pcu::PcuImplementationId {
        fusion_pcu::PcuImplementationId {
            device,
            executor: fusion_pcu::PcuExecutorId(0),
            local_id: 0x7100,
            revision: 0x0000_0008_0000_0404,
        }
    }
    /// Original effect and parent schedule index, including discarded mandatory effects.
    #[must_use]
    pub fn stage_identity(&self, index: usize) -> Option<(ValueId, usize)> {
        self.stages
            .get(index)
            .map(|stage| (stage.effect, stage.operation_index))
    }
    /// Actual external input identity, shape, scalar and logical extent.
    #[must_use]
    pub fn input_metadata(
        &self,
        index: usize,
    ) -> Option<(ValueId, &[usize], PcuScalarType, usize, usize)> {
        self.inputs.get(index).map(|input| {
            (
                input.value,
                &*input.shape,
                input.scalar,
                input.count,
                input.bytes,
            )
        })
    }
}
fn slot(
    program: &TensorOwnedSelectedProgram,
    value: ValueId,
) -> Result<usize, TensorUnsupportedReason> {
    program
        .operation_index_of(value)
        .ok_or(TensorUnsupportedReason::Operation)
}
fn assess_leaf(
    program: Arc<TensorOwnedSelectedProgram>,
    requirements: PcuImplementationRequirements,
) -> Result<Envelope, TensorUnsupportedReason> {
    if let Ok(plan) = MlxCheckedTensorPlan::assess_program(&program, requirements) {
        return Ok(Envelope::Unary(plan));
    }
    if let Ok(plan) = MlxCheckedTensorBinaryPlan::assess_program(&program, requirements) {
        return Ok(Envelope::Binary(plan));
    }
    MlxSelectedNumericalTensorPlan::assess_program(program, requirements).map(Envelope::Numerical)
}

use crate::{
    MlxSession, MlxError, MlxEncodedArray, MlxCheckedProgramInput, MlxPreparedCheckedProgram,
    MlxPreparedTensorBinaryProgram, MlxPreparedSelectedNumericalTensorProgram,
};
/// Original parent effect provenance, preserving the existing raw leaf fault ordinal.
#[derive(Debug)]
pub struct MlxTensorGraphError {
    pub effect: Option<ValueId>,
    pub cause: MlxError,
}
impl std::fmt::Display for MlxTensorGraphError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for MlxTensorGraphError {}
enum Execution {
    Unary(MlxPreparedCheckedProgram),
    Binary(MlxPreparedTensorBinaryProgram),
    Numerical(Box<MlxPreparedSelectedNumericalTensorProgram>),
}
/// Cold direct-node leaf templates over authentic retained MLX resident carriers.
struct PreparedProducer {
    spec: Producer,
    seed: MlxEncodedArray,
    copy: crate::ffi::CarrierCopy,
}
pub struct MlxPreparedSelectedTensorGraph {
    plan: MlxSelectedTensorGraphPlan,
    session: MlxSession,
    stages: Vec<Execution>,
    producers: Vec<PreparedProducer>,
}
impl MlxSession {
    /// Freeze all native stages once; no parent graph walk occurs during replay.
    /// # Errors
    /// Returns exact native compilation/session failure before a graph executable escapes.
    pub fn prepare_selected_tensor_graph(
        &self,
        plan: MlxSelectedTensorGraphPlan,
    ) -> Result<MlxPreparedSelectedTensorGraph, MlxError> {
        let producers = plan
            .producers
            .iter()
            .map(|spec| {
                let bytes = spec
                    .bytes(&plan.program)
                    .map_err(|_| MlxError::InvalidExtent)?;
                let seed = self.upload_encoded_bytes(
                    spec.scalar,
                    if spec.broadcast { 1 } else { spec.count },
                    &bytes,
                )?;
                let copy = self.prepare_carrier_native(spec.scalar, spec.count, spec.broadcast)?;
                // MLX's lazy primitive is evaluated on the actual frozen seed/shape cold.
                // Checked terminal release occurs before any prepared graph can escape.
                copy.execute(seed.native())?.release()?;
                Ok(PreparedProducer {
                    spec: spec.clone(),
                    seed,
                    copy,
                })
            })
            .collect::<Result<Vec<_>, MlxError>>()?;
        let stages = plan
            .stages
            .iter()
            .map(|stage| match &stage.envelope {
                Envelope::Unary(leaf) => self
                    .prepare_checked_program(Arc::clone(&stage.program), leaf.requirements())
                    .map(Execution::Unary),
                Envelope::Binary(leaf) => self
                    .prepare_tensor_binary_program(Arc::clone(&stage.program), leaf.requirements())
                    .map(Execution::Binary),
                Envelope::Numerical(leaf) => self
                    .prepare_selected_numerical_tensor_program(leaf.clone())
                    .map(Box::new)
                    .map(Execution::Numerical),
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(MlxPreparedSelectedTensorGraph {
            plan,
            session: self.clone(),
            stages,
            producers,
        })
    }
}
impl MlxPreparedSelectedTensorGraph {
    #[must_use]
    pub const fn plan(&self) -> &MlxSelectedTensorGraphPlan {
        &self.plan
    }
    #[must_use]
    pub const fn session(&self) -> &MlxSession {
        &self.session
    }
    /// Receipt derives solely from the authentic retained discovery session.
    #[must_use]
    pub fn implementation_id(&self) -> Option<fusion_pcu::PcuImplementationId> {
        self.session
            .identity()
            .map(|device| self.plan.implementation_id(device))
    }
    /// Upload host inputs once; retain immutable resident inputs and every intermediate on device.
    /// All mandatory effects and checked last-use releases precede final owner publication.
    /// This bounded implementation allocates a Rust slot table per call; no allocation claim.
    /// # Errors
    /// Preflights all roles/types/extents/affinity before uploads/arithmetic. Error retains original
    /// effect identity; failed official cleanup also blocks publication and preserves quarantine.
    pub fn execute_mixed(
        &self,
        inputs: &[(ValueId, MlxCheckedProgramInput<'_>)],
    ) -> Result<MlxEncodedArray, MlxTensorGraphError> {
        self.preflight(inputs)
            .map_err(|cause| MlxTensorGraphError {
                effect: None,
                cause,
            })?;
        let mut values: Vec<Option<MlxEncodedArray>> = std::iter::repeat_with(|| None)
            .take(self.plan.slots)
            .collect();
        let result = self.execute_private(inputs, &mut values);
        let mut cleanup = None;
        for value in &mut values {
            if let Some(array) = value.take()
                && let Err(cause) = array.release()
            {
                cleanup.get_or_insert(cause);
            }
        }
        if let Some(cause) = cleanup {
            let effect = result.as_ref().err().and_then(|error| error.effect);
            if let Ok(output) = result {
                let _ = output.release();
            }
            return Err(MlxTensorGraphError { effect, cause });
        }
        result
    }
    fn execute_private(
        &self,
        inputs: &[(ValueId, MlxCheckedProgramInput<'_>)],
        values: &mut [Option<MlxEncodedArray>],
    ) -> Result<MlxEncodedArray, MlxTensorGraphError> {
        for producer in &self.producers {
            let output = producer
                .copy
                .execute(producer.seed.native())
                .map(|array| self.session.wrap_encoded(array));
            values[producer.spec.slot] = Some(output.map_err(|cause| MlxTensorGraphError {
                effect: None,
                cause,
            })?);
        }
        for spec in &*self.plan.inputs {
            let (_, input) = inputs
                .iter()
                .find(|(value, _)| *value == spec.value)
                .ok_or(MlxTensorGraphError {
                    effect: None,
                    cause: MlxError::InvalidExtent,
                })?;
            let result = match input {
                MlxCheckedProgramInput::Host { bytes, .. } => {
                    self.session
                        .upload_encoded_bytes(spec.scalar, spec.count, &bytes[..spec.bytes])
                }
                MlxCheckedProgramInput::Resident(array) => Ok((*array).clone()),
            };
            values[spec.slot] = Some(result.map_err(|cause| MlxTensorGraphError {
                effect: None,
                cause,
            })?);
        }
        for (stage, execution) in self.plan.stages.iter().zip(&self.stages) {
            let output =
                execute_stage(stage, execution, values).map_err(|cause| MlxTensorGraphError {
                    effect: Some(stage.effect),
                    cause,
                })?;
            values[stage.operation_index] = Some(output);
            for &index in &*stage.releases {
                if let Some(value) = values[index].take() {
                    value.release().map_err(|cause| MlxTensorGraphError {
                        effect: Some(stage.effect),
                        cause,
                    })?;
                }
            }
        }
        values[self.plan.output_slot]
            .take()
            .ok_or(MlxTensorGraphError {
                effect: None,
                cause: MlxError::InvalidExtent,
            })
    }
    fn preflight(&self, inputs: &[(ValueId, MlxCheckedProgramInput<'_>)]) -> Result<(), MlxError> {
        self.session.validate_access_available()?;
        if inputs.len() != self.plan.inputs.len() {
            return Err(MlxError::InvalidExtent);
        }
        for (position, (value, input)) in inputs.iter().enumerate() {
            if inputs[..position].iter().any(|(prior, _)| prior == value) {
                return Err(MlxError::InvalidExtent);
            }
            let spec = self
                .plan
                .inputs
                .iter()
                .find(|spec| spec.value == *value)
                .ok_or(MlxError::InvalidExtent)?;
            match input {
                MlxCheckedProgramInput::Host { scalar, bytes } => {
                    if *scalar != spec.scalar {
                        return Err(MlxError::UnsupportedScalar(*scalar));
                    }
                    if bytes.len() < spec.bytes {
                        return Err(MlxError::InvalidExtent);
                    }
                }
                MlxCheckedProgramInput::Resident(array) => {
                    array.validate_access_available()?;
                    if !array.same_session(&self.session) {
                        return Err(MlxError::ForeignSession);
                    }
                    if array.scalar_type() != spec.scalar {
                        return Err(MlxError::UnsupportedScalar(array.scalar_type()));
                    }
                    if array.element_count() != spec.count {
                        return Err(MlxError::InvalidExtent);
                    }
                }
            }
        }
        Ok(())
    }
}
fn execute_stage(
    stage: &Stage,
    execution: &Execution,
    values: &[Option<MlxEncodedArray>],
) -> Result<MlxEncodedArray, MlxError> {
    let binding = |index: usize| {
        let (value, slot) = *stage.bindings.get(index).ok_or(MlxError::InvalidExtent)?;
        let array = values
            .get(slot)
            .and_then(Option::as_ref)
            .ok_or(MlxError::InvalidExtent)?;
        Ok((value, MlxCheckedProgramInput::Resident(array)))
    };
    match execution {
        Execution::Unary(leaf) => leaf.execute_mixed(&[binding(0)?]),
        Execution::Binary(leaf) => {
            if stage.bindings.len() == 1 {
                leaf.execute_mixed(&[binding(0)?])
            } else {
                leaf.execute_mixed(&[binding(0)?, binding(1)?])
            }
        }
        Execution::Numerical(leaf) => {
            if stage.bindings.len() == 1 {
                leaf.execute_mixed(&[binding(0)?])
            } else {
                leaf.execute_mixed(&[binding(0)?, binding(1)?])
            }
        }
    }
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;

type OutputMetadata = (Rc<[usize]>, PcuScalarType, usize, usize);
fn output_metadata(
    program: &TensorOwnedSelectedProgram,
    output: ValueId,
) -> Result<OutputMetadata, TensorUnsupportedReason> {
    let node = program
        .graph()
        .node(output)
        .map_err(|_| TensorUnsupportedReason::Operation)?;
    let count = node
        .shape
        .iter()
        .try_fold(1usize, |n, &d| n.checked_mul(d))
        .filter(|&n| n != 0)
        .ok_or(TensorUnsupportedReason::Shape)?;
    let bytes = count
        .checked_mul(usize::from(node.scalar_type.bit_width()) / 8)
        .ok_or(TensorUnsupportedReason::Shape)?;
    Ok((Rc::from(node.shape), node.scalar_type, count, bytes))
}

#[cfg(test)]
#[path = "captured.rs"]
mod captured;

#[cfg(test)]
#[path = "offers.rs"]
mod offers;

fn assess_producers(
    program: &TensorOwnedSelectedProgram,
    requirements: PcuImplementationRequirements,
) -> Result<Vec<Producer>, TensorUnsupportedReason> {
    let mut producers = Vec::new();
    for &value in program.node_order() {
        if let Some(producer) = Producer::assess(program, value, requirements)? {
            producers.push(producer);
            continue;
        }
        let node = program
            .graph()
            .node(value)
            .map_err(|_| TensorUnsupportedReason::Operation)?;
        if !is_graph_float(node.scalar_type) {
            return Err(TensorUnsupportedReason::ElementType);
        }
    }
    Ok(producers)
}

const fn is_graph_float(scalar: PcuScalarType) -> bool {
    matches!(
        scalar,
        PcuScalarType::F16
            | PcuScalarType::BF16
            | PcuScalarType::F8E4M3FN
            | PcuScalarType::F8E5M2
            | PcuScalarType::F32
            | PcuScalarType::F64
    )
}

fn validate_output_envelope(
    program: &TensorOwnedSelectedProgram,
    output: ValueId,
    requirements: PcuImplementationRequirements,
) -> Result<(), TensorUnsupportedReason> {
    let node = program
        .graph()
        .node(output)
        .map_err(|_| TensorUnsupportedReason::Operation)?;
    // Transport has no arithmetic header. Each discarded computed effect is
    // independently admitted under its frozen local request below.
    if matches!(
        node.op,
        OpDescriptor::Input | OpDescriptor::Constant(_) | OpDescriptor::Uniform { .. }
    ) {
        return Ok(());
    }
    if node.numerical_options != requirements.numerical_options
        || node
            .numerical_mode
            .is_some_and(|mode| mode != requirements.numerical_mode)
        || node
            .float_underflow_policy
            .is_some_and(|policy| policy != requirements.float_underflow)
    {
        return Err(TensorUnsupportedReason::Operation);
    }
    Ok(())
}
