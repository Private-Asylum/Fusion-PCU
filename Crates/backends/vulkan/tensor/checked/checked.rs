//! Cold indexed pointwise table; warm execution never rebuilds or evaluates a graph on the CPU.
#[path = "compile/compile.rs"]
mod compile;
#[path = "compound/compound.rs"]
mod compound;
#[rustfmt::skip]
use std::rc::Rc;
#[rustfmt::skip]
use fusion_pcu::PcuScalar;
#[rustfmt::skip]
use fusion_pcu::dialect::tensor::{
    Graph,
    NodeDescriptor,
    OpDescriptor,
    TensorElement,
    TensorError,
    ValueId,
};
#[rustfmt::skip]
use crate::{
    ffi::{
        VulkanOwnedBuffer,
        VulkanPreparedOwnedCopy,
        VulkanPreparedTensorMap,
        TensorStatusPolicy,
    },
    owned::bytes,
    PcuVulkanBackend,
    PcuVulkanError,
    PcuVulkanOwnedBuffer,
    PcuVulkanTensorBinding,
    PcuVulkanTensorError,
    PcuVulkanTensorInput,
};

struct Slot {
    value: ValueId,
    count: usize,
    native: VulkanOwnedBuffer,
    input: Option<usize>,
}
enum Kernel {
    Unary(VulkanPreparedTensorMap<3>),
    Binary(VulkanPreparedTensorMap<4>),
    Compound(VulkanPreparedTensorMap<4>),
}
struct Step {
    kernel: Kernel,
    value: ValueId,
    inputs: [usize; 2],
    output: usize,
}

/// Frozen checked pointwise graph with private retained native storage and exact typed inputs.
///
/// Input/Constant/Uniform support all22 carriers. Add/Sub/Mul admit fourteen integers and six
/// checked floats; Div/ReLU and the `ReLU` derivative admit only the six floats. Strict
/// F32/F64 `MatMul`, MSE and SGD use ordered, independently checked constituents, retaining
/// reduction/step fault provenance. Every instruction retains its checked contract under
/// precision/compound permissions. Boundary compounds and Portable reject cold.
/// Graph execution is Reject-only; no result owner escapes after an arithmetic fault.
/// Warm execution allocates no Rust workspace and produces one fresh terminal native output.
pub struct PcuVulkanPreparedTensorGraph<T: TensorElement + PcuScalar> {
    device: Rc<crate::ffi::VulkanDevice>,
    slots: Vec<Slot>,
    steps: Vec<Step>,
    inputs: Vec<PcuVulkanTensorBinding>,
    output: PcuVulkanTensorBinding,
    output_slot: usize,
    copy: VulkanPreparedOwnedCopy,
    scalar: core::marker::PhantomData<T>,
}

impl<T: TensorElement + PcuScalar> PcuVulkanPreparedTensorGraph<T> {
    /// Assesses the complete selected closure without discovery or native activation.
    ///
    /// # Errors
    /// Rejects unsupported type/operation/Portable, shape overflow or invalid selected outputs.
    pub fn assess(graph: &Graph, outputs: &[ValueId]) -> Result<(), PcuVulkanTensorError> {
        if outputs.len() != 1 {
            return Err(PcuVulkanTensorError::OutputCount(outputs.len()));
        }
        if !cfg!(target_endian = "little") || T::HOST_SIZE != T::ENCODED_SIZE {
            return Err(PcuVulkanError::UnsupportedPreparedProfile.into());
        }
        for node in graph.execution_plan_for_outputs(outputs)?.nodes() {
            assess_node::<T>(node)?;
        }
        Ok(())
    }
    /// Freezes exact selected effects, indexed operations, shapes and native workspaces.
    ///
    /// # Errors
    /// Rejects unsupported nodes/type/Portable, selected output count, shape or native failures.
    pub fn prepare(
        backend: &PcuVulkanBackend,
        graph: &Graph,
        outputs: &[ValueId],
    ) -> Result<Self, PcuVulkanTensorError> {
        if outputs.len() != 1 {
            return Err(PcuVulkanTensorError::OutputCount(outputs.len()));
        }
        if !cfg!(target_endian = "little") || T::HOST_SIZE != T::ENCODED_SIZE {
            return Err(PcuVulkanError::UnsupportedPreparedProfile.into());
        }
        let plan = graph.execution_plan_for_outputs(outputs)?;
        let nodes: Vec<_> = plan.nodes().collect();
        // Assess the complete selected checked-effect closure before allocating native resources.
        for node in &nodes {
            assess_node::<T>(*node)?;
        }
        let mut slots = Vec::with_capacity(nodes.len());
        let mut inputs = Vec::new();
        for node in &nodes {
            let count = node
                .shape
                .iter()
                .try_fold(1_usize, |count, dimension| count.checked_mul(*dimension))
                .ok_or(TensorError::ShapeOverflow)?;
            let byte_count = count
                .checked_mul(T::HOST_SIZE)
                .ok_or(TensorError::ShapeOverflow)?;
            let mut native = VulkanOwnedBuffer::new(&backend.device, byte_count)?;
            let input = match node.op {
                OpDescriptor::Input => {
                    let index = inputs.len();
                    inputs.push(binding(*node, count));
                    Some(index)
                }
                OpDescriptor::Constant(value) => {
                    let data = T::as_value(value).ok_or(TensorError::UnsupportedScalarType {
                        value: node.value,
                        scalar_type: node.scalar_type,
                    })?;
                    native.write(bytes::read(data.data()))?;
                    None
                }
                OpDescriptor::Uniform { value } => {
                    let value = T::as_scalar(*value).ok_or(TensorError::UnsupportedScalarType {
                        value: node.value,
                        scalar_type: node.scalar_type,
                    })?;
                    native.write(bytes::read(&vec![value; count]))?;
                    None
                }
                _ => None,
            };
            slots.push(Slot {
                value: node.value,
                count,
                native,
                input,
            });
        }
        let mut steps = Vec::new();
        for (index, node) in nodes.iter().enumerate() {
            if let Some(step) = prepare_step(backend, *node, index, &slots, &nodes)? {
                steps.push(step);
            }
        }
        let output_slot = find(&slots, outputs[0])?;
        Ok(Self {
            device: Rc::clone(&backend.device),
            output: binding(nodes[output_slot], slots[output_slot].count),
            output_slot,
            slots,
            steps,
            inputs,
            copy: VulkanPreparedOwnedCopy::new(&backend.device)?,
            scalar: core::marker::PhantomData,
        })
    }

    /// Input order frozen by the selected graph plan, with exact logical extents.
    #[must_use]
    pub fn input_bindings(&self) -> &[PcuVulkanTensorBinding] {
        &self.inputs
    }

    /// Frozen immutable selected shape, shared without warm allocation.
    #[must_use]
    pub const fn output_shape(&self) -> &Rc<[usize]> {
        &self.output.shape
    }

    /// Executes retained native pointwise instructions and returns one detached terminal owner.
    ///
    /// # Errors
    /// Complete input preflight precedes uploads/submission; fatal arithmetic publishes no owner.
    /// Graph faults retain the failing node and first logical lane. Native uncertain completion
    /// quarantines every buffer in the originating session instead of freeing in-use storage.
    pub fn execute_owned(
        &mut self,
        inputs: &[PcuVulkanTensorInput<'_, T>],
    ) -> Result<PcuVulkanOwnedBuffer<T>, PcuVulkanTensorError> {
        preflight(inputs, &self.inputs, &self.device)?;
        for slot in &mut self.slots {
            if let Some(index) = slot.input
                && let PcuVulkanTensorInput::Host(data) = &inputs[index]
            {
                slot.native.write(bytes::read(&data[..slot.count]))?;
            }
        }
        for step in &mut self.steps {
            let left = resolve(&self.slots[step.inputs[0]], inputs);
            let right = resolve(&self.slots[step.inputs[1]], inputs);
            let output = &self.slots[step.output].native;
            let result = match &mut step.kernel {
                Kernel::Unary(kernel) => kernel.call(&[left, output]),
                Kernel::Binary(kernel) => kernel.call(&[left, right, output]),
                Kernel::Compound(kernel) => {
                    if let Some(fault) = kernel.call_compound(&[left, right, output])? {
                        return Err(TensorError::CompoundArithmeticFault {
                            value: step.value,
                            element_index: fault.element_index,
                            reduction_index: fault.reduction_index,
                            step: fault.step,
                            kind: fault.kind,
                        }
                        .into());
                    }
                    Ok(())
                }
            };
            if let Err(error) = result {
                return Err(match error {
                    PcuVulkanError::Fault(fault) => TensorError::ArithmeticFault {
                        value: step.value,
                        element_index: usize::try_from(fault.invocation_id)
                            .map_err(|_| PcuVulkanError::InvalidArguments)?,
                        kind: fault.kind,
                    }
                    .into(),
                    error => error.into(),
                });
            }
        }
        let native = self
            .copy
            .copy(resolve(&self.slots[self.output_slot], inputs))?;
        PcuVulkanOwnedBuffer::from_native(native, self.output.element_count).map_err(Into::into)
    }
}

fn assess_node<T: PcuScalar>(node: NodeDescriptor<'_>) -> Result<(), PcuVulkanTensorError> {
    if node.scalar_type != T::TYPE {
        return Err(TensorError::ScalarTypeMismatch {
            value: node.value,
            expected: T::TYPE,
            actual: node.scalar_type,
        }
        .into());
    }
    node.shape
        .iter()
        .try_fold(1_usize, |count, dimension| count.checked_mul(*dimension))
        .ok_or(TensorError::ShapeOverflow)?;
    compile::assess(node)
}

fn binding(node: NodeDescriptor<'_>, count: usize) -> PcuVulkanTensorBinding {
    PcuVulkanTensorBinding {
        value: node.value,
        shape: Rc::from(node.shape),
        element_count: count,
    }
}
fn find(slots: &[Slot], value: ValueId) -> Result<usize, TensorError> {
    slots
        .iter()
        .position(|slot| slot.value == value)
        .ok_or(TensorError::UnknownValue(value))
}
const fn resolve<'a, T: PcuScalar>(
    slot: &'a Slot,
    inputs: &'a [PcuVulkanTensorInput<'_, T>],
) -> &'a VulkanOwnedBuffer {
    if let Some(index) = slot.input
        && let PcuVulkanTensorInput::Owned(owner) = &inputs[index]
    {
        return owner.native();
    }
    &slot.native
}
fn preflight<T: PcuScalar>(
    inputs: &[PcuVulkanTensorInput<'_, T>],
    bindings: &[PcuVulkanTensorBinding],
    device: &Rc<crate::ffi::VulkanDevice>,
) -> Result<(), PcuVulkanTensorError> {
    if inputs.len() != bindings.len() {
        return Err(TensorError::DataLength {
            expected: bindings.len(),
            actual: inputs.len(),
        }
        .into());
    }
    for (input, binding) in inputs.iter().zip(bindings) {
        match input {
            PcuVulkanTensorInput::Host(data) if data.len() < binding.element_count => {
                return Err(TensorError::DataLength {
                    expected: binding.element_count,
                    actual: data.len(),
                }
                .into());
            }
            PcuVulkanTensorInput::Owned(owner) => {
                owner.validate_access_available()?;
                if owner.len() != binding.element_count || !owner.native().same_session(device) {
                    return Err(PcuVulkanError::InvalidArguments.into());
                }
            }
            PcuVulkanTensorInput::Host(_) => {}
        }
    }
    Ok(())
}

fn prepare_step(
    backend: &PcuVulkanBackend,
    node: NodeDescriptor<'_>,
    output: usize,
    slots: &[Slot],
    nodes: &[NodeDescriptor<'_>],
) -> Result<Option<Step>, PcuVulkanTensorError> {
    let (left, right) = match node.op {
        OpDescriptor::Input | OpDescriptor::Constant(_) | OpDescriptor::Uniform { .. } => {
            return Ok(None);
        }
        OpDescriptor::Add { left, right }
        | OpDescriptor::Sub { left, right }
        | OpDescriptor::Mul { left, right }
        | OpDescriptor::Div { left, right }
        | OpDescriptor::MatMul { left, right, .. } => (left, right),
        OpDescriptor::Relu { input } => (input, input),
        OpDescriptor::ReluBackward { input, upstream } => (input, upstream),
        OpDescriptor::SgdUpdate {
            weights, gradient, ..
        } => (weights, gradient),
        OpDescriptor::MeanSquaredError { prediction, target } => (prediction, target),
    };
    let inputs = [find(slots, left)?, find(slots, right)?];
    let count = slots[output].count;
    if count == 0 {
        return Ok(None);
    }
    if matches!(
        node.op,
        OpDescriptor::MatMul { .. }
            | OpDescriptor::SgdUpdate { .. }
            | OpDescriptor::MeanSquaredError { .. }
    ) {
        return compound::prepare(backend, node, output, inputs, slots, nodes).map(Some);
    }
    for input in inputs {
        if slots[input].count != count {
            return Err(PcuVulkanError::InvalidArguments.into());
        }
    }
    let extent = u32::try_from(count).map_err(|_| PcuVulkanError::BufferTooLarge)?;
    let lowered = compile::lower(node, extent)?;
    let bytes = slots[output].native.byte_len();
    let status = count.checked_mul(4).ok_or(PcuVulkanError::BufferTooLarge)?;
    let kernel = if lowered.unary {
        Kernel::Unary(VulkanPreparedTensorMap::new(
            &backend.device,
            &lowered.words,
            extent,
            lowered.dispatch_extent,
            lowered.local_size,
            [bytes, bytes, status],
            TensorStatusPolicy::scalar(lowered.fault_law)?,
        )?)
    } else {
        Kernel::Binary(VulkanPreparedTensorMap::new(
            &backend.device,
            &lowered.words,
            extent,
            lowered.dispatch_extent,
            lowered.local_size,
            [bytes, bytes, bytes, status],
            TensorStatusPolicy::scalar(lowered.fault_law)?,
        )?)
    };
    Ok(Some(Step {
        kernel,
        value: node.value,
        inputs,
        output,
    }))
}
