//! Cold selected graph decomposition retaining authoritative parent policies and liveness.
use std::{rc::Rc, sync::Arc};
use fusion_pcu::{PcuImplementationRequirements, PcuScalarType, PcuRangePolicy, PcuReproducibility};
use fusion_pcu::dialect::tensor::{
    OpDescriptor, TensorOwnedSelectedProgram, TensorUnsupportedReason, ValueId,
};
use crate::{MetalTensorPlan, MetalTensorBinaryPlan, MetalSelectedNumericalTensorPlan};

#[path = "producer.rs"]
mod producer;
use producer::Producer;

#[derive(Clone)]
enum Envelope {
    Unary(MetalTensorPlan),
    Binary(MetalTensorBinaryPlan),
    Numerical(MetalSelectedNumericalTensorPlan),
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
pub struct MetalSelectedTensorGraphPlan {
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
impl MetalSelectedTensorGraphPlan {
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
    /// Retained detached child for cold inspection; parent liveness remains authoritative.
    #[must_use]
    pub fn stage_program(&self, index: usize) -> Option<&TensorOwnedSelectedProgram> {
        self.stages.get(index).map(|stage| &*stage.program)
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
    if let Ok(plan) = MetalTensorPlan::assess_relu_program(&program, requirements) {
        return Ok(Envelope::Unary(plan));
    }
    if let Ok(plan) = MetalTensorBinaryPlan::assess_program(&program, requirements) {
        return Ok(Envelope::Binary(plan));
    }
    MetalSelectedNumericalTensorPlan::assess_program(program, requirements).map(Envelope::Numerical)
}

use crate::{
    MetalSession, MetalError, MetalTensorInput, MetalTensorOwner, MetalPreparedTensorProgram,
    MetalPreparedTensorBinaryProgram, MetalPreparedSelectedNumericalTensorProgram,
    MetalPreparedCarrierControl,
};
use fusion_pcu::{PcuMemoryPoolId, PcuMemoryResource, PcuMemoryAccess};
/// Parent effect provenance for a native/preflight failure; raw leaf ordinals remain unchanged.
#[derive(Debug)]
pub struct MetalTensorGraphError {
    pub effect: Option<ValueId>,
    pub cause: MetalError,
}
impl std::fmt::Display for MetalTensorGraphError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for MetalTensorGraphError {}
enum Execution {
    Unary(MetalPreparedTensorProgram),
    Binary(MetalPreparedTensorBinaryProgram),
    Numerical(Box<MetalPreparedSelectedNumericalTensorProgram>),
}
/// Cold leaf templates and parent-owned liveness; replay does not walk the source graph.
struct PreparedProducer {
    spec: Producer,
    seed: crate::MetalBuffer,
    copy: crate::MetalPreparedCarrierControl,
}
pub struct MetalPreparedSelectedTensorGraph {
    plan: MetalSelectedTensorGraphPlan,
    session: MetalSession,
    pool: PcuMemoryPoolId,
    inputs: Vec<MetalPreparedCarrierControl>,
    stages: Vec<Execution>,
    producers: Vec<PreparedProducer>,
}
impl MetalSession {
    /// Prepare all leaf stages and resident input copy controls exactly once.
    /// # Errors
    /// Returns native compilation/session failure without admitting a partial graph.
    pub fn prepare_selected_tensor_graph(
        &self,
        plan: MetalSelectedTensorGraphPlan,
        pool: PcuMemoryPoolId,
    ) -> Result<MetalPreparedSelectedTensorGraph, MetalError> {
        let inputs = plan
            .inputs
            .iter()
            .map(|input| self.prepare_carrier_control(input.scalar, input.count, false))
            .collect::<Result<Vec<_>, _>>()?;
        let producers = plan
            .producers
            .iter()
            .map(|spec| {
                let bytes = spec
                    .bytes(&plan.program)
                    .map_err(|_| MetalError::Unsupported)?;
                let seed = self.upload_bytes(&bytes)?;
                let copy = self.prepare_carrier_control(spec.scalar, spec.count, spec.broadcast)?;
                // Cold full-profile prime completes GPU access and every shape-specific setup.
                drop(copy.execute(&seed)?);
                Ok(PreparedProducer {
                    spec: spec.clone(),
                    seed,
                    copy,
                })
            })
            .collect::<Result<Vec<_>, MetalError>>()?;
        let stages = plan
            .stages
            .iter()
            .map(|stage| match &stage.envelope {
                Envelope::Unary(leaf) => self
                    .prepare_tensor_program(leaf.clone(), pool)
                    .map(Execution::Unary),
                Envelope::Binary(leaf) => self
                    .prepare_tensor_binary_program(leaf.clone(), pool)
                    .map(Execution::Binary),
                Envelope::Numerical(leaf) => self
                    .prepare_selected_numerical_tensor_program(leaf.clone(), pool)
                    .map(Box::new)
                    .map(Execution::Numerical),
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(MetalPreparedSelectedTensorGraph {
            plan,
            session: self.clone(),
            pool,
            inputs,
            stages,
            producers,
        })
    }
}
impl MetalPreparedSelectedTensorGraph {
    #[must_use]
    pub const fn plan(&self) -> &MetalSelectedTensorGraphPlan {
        &self.plan
    }
    #[must_use]
    pub const fn session(&self) -> &MetalSession {
        &self.session
    }
    /// Stage each host input once; complete every mandatory effect before publishing one owner.
    /// Intermediate payloads remain resident and release at parent-authorized last use.
    /// This bounded implementation allocates a Rust slot table per call; no allocation claim.
    /// # Errors
    /// Rejects all wrong input roles/types/extents/affinity before uploads or native arithmetic.
    /// Failure returns no output owner and retains original failing effect identity.
    pub fn execute_mixed(
        &self,
        inputs: &[(ValueId, MetalTensorInput<'_>)],
    ) -> Result<MetalTensorOwner, MetalTensorGraphError> {
        self.preflight(inputs)
            .map_err(|cause| MetalTensorGraphError {
                effect: None,
                cause,
            })?;
        let mut values: Vec<Option<MetalTensorOwner>> = std::iter::repeat_with(|| None)
            .take(self.plan.slots)
            .collect();
        for producer in &self.producers {
            let output = producer.copy.execute(&producer.seed).and_then(|buffer| {
                self.owner(
                    buffer,
                    producer.spec.scalar,
                    Rc::clone(&producer.spec.shape),
                    producer.spec.count,
                )
            });
            values[producer.spec.slot] = Some(output.map_err(|cause| MetalTensorGraphError {
                effect: None,
                cause,
            })?);
        }
        for (index, spec) in self.plan.inputs.iter().enumerate() {
            let (_, input) = inputs
                .iter()
                .find(|(value, _)| *value == spec.value)
                .ok_or(MetalTensorGraphError {
                    effect: None,
                    cause: MetalError::InvalidExtent,
                })?;
            let result = match input {
                MetalTensorInput::HostBytes { bytes, .. } => {
                    self.session.upload_bytes(&bytes[..spec.bytes])
                }
                MetalTensorInput::Resident { resource, .. } => {
                    let lease = resource.lease();
                    self.inputs[index].execute(&lease.borrow())
                }
            }
            .and_then(|buffer| self.owner(buffer, spec.scalar, Rc::clone(&spec.shape), spec.count));
            values[spec.slot] = Some(result.map_err(|cause| MetalTensorGraphError {
                effect: None,
                cause,
            })?);
        }
        for (stage, execution) in self.plan.stages.iter().zip(&self.stages) {
            let result = execute_stage(stage, execution, &values).map_err(|cause| {
                MetalTensorGraphError {
                    effect: Some(stage.effect),
                    cause,
                }
            })?;
            values[stage.operation_index] = Some(result);
            for &index in &*stage.releases {
                drop(values[index].take());
            }
        }
        values[self.plan.output_slot]
            .take()
            .ok_or(MetalTensorGraphError {
                effect: None,
                cause: MetalError::InvalidExtent,
            })
    }
    fn owner(
        &self,
        buffer: crate::MetalBuffer,
        scalar: PcuScalarType,
        shape: Rc<[usize]>,
        count: usize,
    ) -> Result<MetalTensorOwner, MetalError> {
        Ok(MetalTensorOwner {
            session: self.session.clone(),
            scalar,
            shape,
            count,
            resource: self
                .session
                .initialized_tensor_resource(self.pool, buffer)?,
        })
    }
    fn preflight(&self, inputs: &[(ValueId, MetalTensorInput<'_>)]) -> Result<(), MetalError> {
        self.session.ensure_quiescent()?;
        if inputs.len() != self.plan.inputs.len() {
            return Err(MetalError::InvalidExtent);
        }
        for (position, (value, input)) in inputs.iter().enumerate() {
            if inputs[..position].iter().any(|(prior, _)| prior == value) {
                return Err(MetalError::InvalidExtent);
            }
            let spec = self
                .plan
                .inputs
                .iter()
                .find(|spec| spec.value == *value)
                .ok_or(MetalError::InvalidExtent)?;
            let (scalar, elements) = match input {
                MetalTensorInput::HostBytes {
                    scalar,
                    elements,
                    bytes,
                } => {
                    if bytes.len() < spec.bytes {
                        return Err(MetalError::InvalidExtent);
                    }
                    (*scalar, *elements)
                }
                MetalTensorInput::Resident {
                    scalar,
                    elements,
                    resource,
                } => {
                    resource.validate_access_available()?;
                    if resource.access() == PcuMemoryAccess::WriteOnly {
                        return Err(MetalError::Unsupported);
                    }
                    if resource.size_bytes()
                        < u64::try_from(spec.bytes).map_err(|_| MetalError::InvalidExtent)?
                    {
                        return Err(MetalError::InvalidExtent);
                    }
                    if !self
                        .session
                        .same_session(resource.lease().borrow().session())
                    {
                        return Err(MetalError::ForeignSession);
                    }
                    (*scalar, *elements)
                }
            };
            if scalar != spec.scalar {
                return Err(MetalError::Unsupported);
            }
            if elements != spec.count {
                return Err(MetalError::InvalidExtent);
            }
        }
        Ok(())
    }
}
fn execute_stage(
    stage: &Stage,
    execution: &Execution,
    values: &[Option<MetalTensorOwner>],
) -> Result<MetalTensorOwner, MetalError> {
    let binding = |index: usize| {
        let (_, slot) = *stage.bindings.get(index).ok_or(MetalError::InvalidExtent)?;
        let value = values
            .get(slot)
            .and_then(Option::as_ref)
            .ok_or(MetalError::InvalidExtent)?;
        Ok(MetalTensorInput::Resident {
            scalar: value.scalar_type(),
            elements: value.element_count(),
            resource: value.resource(),
        })
    };
    match execution {
        Execution::Unary(leaf) => leaf.execute(binding(0)?),
        Execution::Binary(leaf) => {
            if stage.bindings.len() == 1 {
                leaf.execute(&[binding(0)?])
            } else {
                leaf.execute(&[binding(0)?, binding(1)?])
            }
        }
        Execution::Numerical(leaf) => {
            if stage.bindings.len() == 1 {
                leaf.execute(&[binding(0)?])
            } else {
                leaf.execute(&[binding(0)?, binding(1)?])
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

#[cfg(all(test, feature = "source-tensor"))]
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
