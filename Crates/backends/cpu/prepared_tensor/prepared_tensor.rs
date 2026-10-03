//! Cold-frozen checked tensor execution with reusable, liveness-planned host storage.
//!
//! Scalar rounding and faults use the core integer-significand contracts. Ordered compounds
//! preserve each separately rounded step. PCU's finite-input rejection and reduction order
//! are stricter contracts than IEEE 754 default exception results, not general IEEE certification.
#[rustfmt::skip]
use alloc::{
    vec,
    vec::Vec,
};
#[rustfmt::skip]
use fusion_pcu::{
    PcuCheckedFloat,
    PcuCheckedFloatWidening,
    PcuReproducibility,
    PcuScalarType,
};
#[rustfmt::skip]
use fusion_pcu::dialect::tensor::{
    constants::{
        count_f32,
        count_f64,
    },
    Graph,
    NodeDescriptor,
    OpDescriptor,
    Tensor,
    TensorCheckedReferenceAssessor,
    TensorElement,
    TensorError,
    TensorOperationAssessor,
    TensorOperationSupport,
    TensorScalarValue,
    TensorExecutionRoute,

    TensorUnsupportedReason,
    ValueId,
};
#[path = "execution/execution.rs"]
mod execution;
#[rustfmt::skip]
use execution::{
    Step,
    compile,
};

#[path = "integer/integer.rs"]
mod integer;
pub use integer::{PcuCpuPreparedIntegerTensorGraph, PcuCpuIntegerTensorAssessor};
#[path = "scalar/scalar.rs"]
mod scalar;
pub use scalar::{PcuCpuPreparedScalarTensorGraph, PcuCpuScalarTensorAssessor};

/// Six-format exact elementwise admission, with F32/F64-only ordered compounds.
/// Compound and precision permissions do not weaken a checked elementwise instruction.
pub struct PcuCpuTensorAssessor;
impl TensorOperationAssessor for PcuCpuTensorAssessor {
    fn assess_node(&self, graph: &Graph, node: NodeDescriptor<'_>) -> TensorOperationSupport {
        if !float_type(node.scalar_type) {
            return TensorOperationSupport::Unsupported {
                reason: TensorUnsupportedReason::ElementType,
            };
        }
        if elementwise(node.op)
            && node.numerical_options.reproducibility == PcuReproducibility::Unspecified
        {
            return TensorOperationSupport::Supported {
                route: TensorExecutionRoute::Synthesized,
                workspace_bytes: None,
            };
        }
        if !matches!(node.scalar_type, PcuScalarType::F32 | PcuScalarType::F64) {
            return TensorOperationSupport::Unsupported {
                reason: TensorUnsupportedReason::ElementType,
            };
        }
        TensorCheckedReferenceAssessor.assess_node(graph, node)
    }
}
const fn float_type(scalar: PcuScalarType) -> bool {
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
const fn elementwise(op: OpDescriptor<'_>) -> bool {
    matches!(
        op,
        OpDescriptor::Input
            | OpDescriptor::Constant(_)
            | OpDescriptor::Uniform { .. }
            | OpDescriptor::Add { .. }
            | OpDescriptor::Sub { .. }
            | OpDescriptor::Mul { .. }
            | OpDescriptor::Div { .. }
            | OpDescriptor::Relu { .. }
            | OpDescriptor::ReluBackward { .. }
    )
}

/// Frozen selected input/output schema, in selected-plan order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PcuCpuTensorBinding {
    pub value: ValueId,
    pub shape: Vec<usize>,
    pub element_count: usize,
    slot: usize,
}

/// Detached, cold-compiled six-format tensor plan with private reusable storage.
///
/// `execute` and `call` allocate nothing. Only `execute_owned` allocates the escaping output.
/// Failed execution invalidates every previous output view and publishes no caller output.
/// Selected constants are copied once at preparation; later graph changes cannot alter this plan.
pub struct PcuCpuPreparedTensorGraph<T: TensorElement + PcuCheckedFloat> {
    storage: Vec<Vec<T>>,
    steps: Vec<Step<T>>,
    inputs: Vec<PcuCpuTensorBinding>,
    outputs: Vec<PcuCpuTensorBinding>,
    scratch_slot_count: usize,
    ready: bool,
}

impl<T: TensorElement + PcuCheckedFloat> PcuCpuPreparedTensorGraph<T> {
    /// Freezes the selected dependency closure and allocates all executor workspace.
    ///
    /// # Errors
    /// Rejects unsupported scalar/policy/compound contracts or invalid selected outputs.
    pub fn prepare(graph: &Graph, outputs: &[ValueId]) -> Result<Self, TensorError> {
        let plan = graph.execution_plan_for_outputs(outputs)?;
        let nodes: Vec<_> = plan.nodes().collect();
        for node in &nodes {
            if !float_type(T::TYPE)
                || (!matches!(T::TYPE, PcuScalarType::F32 | PcuScalarType::F64)
                    && !elementwise(node.op))
            {
                return Err(TensorError::UnsupportedScalarType {
                    value: node.value,
                    scalar_type: T::TYPE,
                });
            }
            if node.scalar_type != T::TYPE {
                return Err(TensorError::ScalarTypeMismatch {
                    value: node.value,
                    expected: T::TYPE,
                    actual: node.scalar_type,
                });
            }
            if node.numerical_options.reproducibility != PcuReproducibility::Unspecified {
                return Err(TensorError::UnsupportedNumericalOptions {
                    value: node.value,
                    options: node.numerical_options,
                });
            }
        }
        let scratch = plan.scratch_storage_plan(plan.node_order())?;
        let zero = zero::<T>();
        let mut storage: Vec<Vec<T>> = scratch
            .slots()
            .iter()
            .map(|slot| vec![zero; slot.capacity_bytes / T::HOST_SIZE])
            .collect();
        let mut bindings = Vec::with_capacity(nodes.len());
        for node in &nodes {
            let count = node
                .shape
                .iter()
                .try_fold(1_usize, |count, &dimension| count.checked_mul(dimension))
                .ok_or(TensorError::ShapeOverflow)?;
            let slot = scratch.slot_for(node.value).unwrap_or_else(|| {
                let slot = storage.len();
                storage.push(vec![zero; count]);
                slot
            });
            match node.op {
                OpDescriptor::Constant(value) => {
                    let tensor = T::as_value(value).ok_or(TensorError::UnsupportedScalarType {
                        value: node.value,
                        scalar_type: node.scalar_type,
                    })?;
                    storage[slot][..count].copy_from_slice(tensor.data());
                }
                OpDescriptor::Uniform { value } => {
                    let value = T::as_scalar(*value).ok_or(TensorError::UnsupportedScalarType {
                        value: node.value,
                        scalar_type: node.scalar_type,
                    })?;
                    storage[slot][..count].fill(value);
                }
                _ => {}
            }
            bindings.push(PcuCpuTensorBinding {
                value: node.value,
                shape: node.shape.to_vec(),
                element_count: count,
                slot,
            });
        }
        let mut steps = Vec::new();
        for node in &nodes {
            if let Some(step) = compile(*node, &bindings)? {
                steps.push(step);
            }
        }
        let select = |values: &[ValueId]| -> Result<Vec<PcuCpuTensorBinding>, TensorError> {
            values
                .iter()
                .map(|value| {
                    bindings
                        .iter()
                        .find(|binding| binding.value == *value)
                        .cloned()
                        .ok_or(TensorError::UnknownValue(*value))
                })
                .collect()
        };
        Ok(Self {
            storage,
            steps,
            inputs: select(plan.input_values())?,
            outputs: select(plan.output_values())?,
            scratch_slot_count: scratch.slots().len(),
            ready: false,
        })
    }

    #[must_use]
    pub fn input_bindings(&self) -> &[PcuCpuTensorBinding] {
        &self.inputs
    }
    #[must_use]
    pub fn output_bindings(&self) -> &[PcuCpuTensorBinding] {
        &self.outputs
    }
    /// Number of actual reusable transient slots, excluding pinned leaves and outputs.
    #[must_use]
    pub const fn scratch_slot_count(&self) -> usize {
        self.scratch_slot_count
    }
    /// Total dense executor storage, including selected inputs, constants and pinned outputs.
    #[must_use]
    pub fn storage_bytes(&self) -> usize {
        self.storage
            .iter()
            .map(|slot| slot.len() * T::HOST_SIZE)
            .sum()
    }

    /// Executes frozen steps after complete input-schema validation, with no allocation.
    ///
    /// # Errors
    /// Returns schema mismatch or the first fault in selected node/element/reduction order.
    pub fn execute(&mut self, inputs: &[&[T]]) -> Result<(), TensorError> {
        self.ready = false;
        validate(inputs, &self.inputs)?;
        for (data, binding) in inputs.iter().zip(&self.inputs) {
            self.storage[binding.slot][..binding.element_count]
                .copy_from_slice(&data[..binding.element_count]);
        }
        for step in &self.steps {
            (step.run)(&mut self.storage, step)?;
        }
        self.ready = true;
        Ok(())
    }

    /// Executes and atomically publishes all selected outputs; caller tails remain unchanged.
    ///
    /// # Errors
    /// Validates every input and output before executing; any failure preserves caller outputs.
    pub fn call(&mut self, inputs: &[&[T]], outputs: &mut [&mut [T]]) -> Result<(), TensorError> {
        self.ready = false;
        validate(inputs, &self.inputs)?;
        if outputs.len() != self.outputs.len() {
            return Err(TensorError::DataLength {
                expected: self.outputs.len(),
                actual: outputs.len(),
            });
        }
        for (data, binding) in outputs.iter().zip(&self.outputs) {
            if data.len() < binding.element_count {
                return Err(TensorError::DataLength {
                    expected: binding.element_count,
                    actual: data.len(),
                });
            }
        }
        self.execute(inputs)?;
        for (data, binding) in outputs.iter_mut().zip(&self.outputs) {
            data[..binding.element_count]
                .copy_from_slice(&self.storage[binding.slot][..binding.element_count]);
        }
        Ok(())
    }

    /// Borrows a selected output only after a successful most recent execution.
    #[must_use]
    pub fn output(&self, index: usize) -> Option<&[T]> {
        if !self.ready {
            return None;
        }
        let binding = self.outputs.get(index)?;
        Some(&self.storage[binding.slot][..binding.element_count])
    }

    /// Executes one selected output and materializes fresh owned host storage.
    ///
    /// # Errors
    /// Rejects multiple outputs, invalid schemas, or terminal arithmetic faults.
    pub fn execute_owned(&mut self, inputs: &[&[T]]) -> Result<Tensor<T>, TensorError> {
        if self.outputs.len() != 1 {
            return Err(TensorError::DataLength {
                expected: 1,
                actual: self.outputs.len(),
            });
        }
        self.execute(inputs)?;
        let binding = &self.outputs[0];
        Tensor::new(
            binding.shape.clone(),
            self.storage[binding.slot][..binding.element_count].to_vec(),
        )
    }
}

fn validate<T>(data: &[&[T]], bindings: &[PcuCpuTensorBinding]) -> Result<(), TensorError> {
    if data.len() != bindings.len() {
        return Err(TensorError::DataLength {
            expected: bindings.len(),
            actual: data.len(),
        });
    }
    for (data, binding) in data.iter().zip(bindings) {
        if data.len() < binding.element_count {
            return Err(TensorError::DataLength {
                expected: binding.element_count,
                actual: data.len(),
            });
        }
    }
    Ok(())
}

fn scalar<T: TensorElement + PcuCheckedFloat>(value: f32) -> T {
    let value = if T::TYPE == PcuScalarType::F32 {
        TensorScalarValue::F32(value)
    } else {
        TensorScalarValue::F64(
            value
                .pcu_checked_to_f64()
                .expect("cold SGD admission proves a finite F32 rate"),
        )
    };
    T::as_scalar(value).expect("prepared tensor admission restricts scalars to F32 or F64")
}

// Representation-defined RN-even counts match the core MSE contract under ambient FP controls.
fn denominator<T: TensorElement + PcuCheckedFloat>(count: usize) -> T {
    let value = if T::TYPE == PcuScalarType::F32 {
        TensorScalarValue::F32(count_f32(count))
    } else {
        TensorScalarValue::F64(count_f64(count))
    };
    T::as_scalar(value).expect("prepared tensor admission restricts scalars to F32 or F64")
}

fn zero<T: TensorElement + PcuCheckedFloat>() -> T {
    let value = match T::TYPE {
        PcuScalarType::F16 => TensorScalarValue::F16(fusion_pcu::PcuF16Bits::from_bits(0)),
        PcuScalarType::BF16 => TensorScalarValue::Bf16(fusion_pcu::PcuBf16Bits::from_bits(0)),
        PcuScalarType::F8E4M3FN => {
            TensorScalarValue::F8E4M3Fn(fusion_pcu::PcuF8E4M3FnBits::from_bits(0))
        }
        PcuScalarType::F8E5M2 => TensorScalarValue::F8E5M2(fusion_pcu::PcuF8E5M2Bits::from_bits(0)),
        PcuScalarType::F32 => TensorScalarValue::F32(0.0),
        _ => TensorScalarValue::F64(0.0),
    };
    T::as_scalar(value)
        .expect("sealed checked-float type and cold six-format admission prove exact zero carrier")
}
