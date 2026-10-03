//! Cold indexed leaf-only carrier transport, without arithmetic or selection admission.
#[rustfmt::skip]
use alloc::{
    vec,
    vec::Vec,
};
#[rustfmt::skip]
use fusion_pcu::{
    PcuScalar,
    PcuReproducibility,
    PcuNumericalRequirement,
    PcuScalarType,
};
#[rustfmt::skip]
use fusion_pcu::dialect::tensor::{
    Graph,
    NodeDescriptor,
    OpDescriptor,
    Tensor,
    TensorExecutionRoute,
    TensorElement,
    TensorError,
    TensorOperationAssessor,
    TensorOperationSupport,
    TensorScalarValue,
    TensorUnsupportedReason,
    ValueId,
};
/// Exact twenty-two-carrier leaf storage admission; it never authorizes arithmetic.
pub struct PcuCpuScalarTensorAssessor;
impl TensorOperationAssessor for PcuCpuScalarTensorAssessor {
    fn assess_node(&self, _graph: &Graph, node: NodeDescriptor<'_>) -> TensorOperationSupport {
        if !carrier_type(node.scalar_type) {
            return TensorOperationSupport::Unsupported {
                reason: TensorUnsupportedReason::ElementType,
            };
        }
        if !matches!(
            node.op,
            OpDescriptor::Input | OpDescriptor::Constant(_) | OpDescriptor::Uniform { .. }
        ) {
            return TensorOperationSupport::Unsupported {
                reason: TensorUnsupportedReason::Operation,
            };
        }
        if node.numerical_options.reproducibility != PcuReproducibility::Unspecified {
            return TensorOperationSupport::Unsupported {
                reason: TensorUnsupportedReason::NumericalPolicy {
                    requirement: PcuNumericalRequirement::Reproducibility,
                    options: node.numerical_options,
                },
            };
        }
        // Compound/precision permissions do not change raw carrier transport.
        TensorOperationSupport::Supported {
            route: TensorExecutionRoute::Synthesized,
            workspace_bytes: None,
        }
    }
}

#[rustfmt::skip]
use super::PcuCpuTensorBinding;

/// Detached, cold-indexed twenty-two-carrier leaf tensor plan with private reusable storage.
///
/// `execute` and `call` allocate nothing. Only `execute_owned` allocates the escaping output.
/// Failed execution invalidates every previous output view and publishes no caller output.
/// Selected constants are copied once at preparation; later graph changes cannot alter this plan.
pub struct PcuCpuPreparedScalarTensorGraph<T: TensorElement + PcuScalar> {
    storage: Vec<Vec<T>>,
    inputs: Vec<PcuCpuTensorBinding>,
    outputs: Vec<PcuCpuTensorBinding>,
    scratch_slot_count: usize,
    ready: bool,
}

impl<T: TensorElement + PcuScalar> PcuCpuPreparedScalarTensorGraph<T> {
    /// Freezes the selected dependency closure and allocates all executor workspace.
    ///
    /// # Errors
    /// Rejects nonleaf operations, unsupported representations/policies or invalid selected outputs.
    #[allow(clippy::too_many_lines)] // Cold schema, liveness and detached constant ownership form one preparation transaction.
    pub fn prepare(graph: &Graph, outputs: &[ValueId]) -> Result<Self, TensorError> {
        let plan = graph.execution_plan_for_outputs(outputs)?;
        let nodes: Vec<_> = plan.nodes().collect();
        for node in &nodes {
            if !carrier_type(T::TYPE) {
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
        for node in &nodes {
            if matches!(
                PcuCpuScalarTensorAssessor.assess_node(graph, *node),
                TensorOperationSupport::Unsupported { .. }
            ) {
                return Err(TensorError::UnsupportedScalarType {
                    value: node.value,
                    scalar_type: node.scalar_type,
                });
            }
        }
        let scratch = plan.scratch_storage_plan(plan.node_order())?;
        let zero = zero::<T>().ok_or(TensorError::UnsupportedScalarType {
            value: nodes.first().ok_or(TensorError::EmptyOutputs)?.value,
            scalar_type: T::TYPE,
        })?;
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

    /// Copies indexed frozen inputs after complete schema validation, with no allocation.
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
    /// Rejects multiple outputs, invalid schemas, or unsupported profiles.
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

const fn carrier_type(ty: PcuScalarType) -> bool {
    matches!(
        ty,
        PcuScalarType::I8
            | PcuScalarType::U8
            | PcuScalarType::I16
            | PcuScalarType::U16
            | PcuScalarType::I32
            | PcuScalarType::U32
            | PcuScalarType::I64
            | PcuScalarType::U64
            | PcuScalarType::I128
            | PcuScalarType::U128
            | PcuScalarType::I256
            | PcuScalarType::U256
            | PcuScalarType::I512
            | PcuScalarType::U512
            | PcuScalarType::F16
            | PcuScalarType::BF16
            | PcuScalarType::F8E4M3FN
            | PcuScalarType::F8E5M2
            | PcuScalarType::F32
            | PcuScalarType::F64
            | PcuScalarType::F128
            | PcuScalarType::F256
    )
}
fn zero<T: TensorElement + PcuScalar>() -> Option<T> {
    use fusion_pcu::{PcuI256, PcuU256, PcuI512, PcuU512};
    let value = match T::TYPE {
        PcuScalarType::I8 => TensorScalarValue::I8(0),
        PcuScalarType::U8 => TensorScalarValue::U8(0),
        PcuScalarType::I16 => TensorScalarValue::I16(0),
        PcuScalarType::U16 => TensorScalarValue::U16(0),
        PcuScalarType::I32 => TensorScalarValue::I32(0),
        PcuScalarType::U32 => TensorScalarValue::U32(0),
        PcuScalarType::I64 => TensorScalarValue::I64(0),
        PcuScalarType::U64 => TensorScalarValue::U64(0),
        PcuScalarType::I128 => TensorScalarValue::I128(0),
        PcuScalarType::U128 => TensorScalarValue::U128(0),
        PcuScalarType::I256 => TensorScalarValue::I256(PcuI256::ZERO),
        PcuScalarType::U256 => TensorScalarValue::U256(PcuU256::ZERO),
        PcuScalarType::I512 => TensorScalarValue::I512(PcuI512::ZERO),
        PcuScalarType::U512 => TensorScalarValue::U512(PcuU512::ZERO),
        PcuScalarType::F16 => TensorScalarValue::F16(fusion_pcu::PcuF16Bits::from_bits(0)),
        PcuScalarType::BF16 => TensorScalarValue::Bf16(fusion_pcu::PcuBf16Bits::from_bits(0)),
        PcuScalarType::F8E4M3FN => {
            TensorScalarValue::F8E4M3Fn(fusion_pcu::PcuF8E4M3FnBits::from_bits(0))
        }
        PcuScalarType::F8E5M2 => TensorScalarValue::F8E5M2(fusion_pcu::PcuF8E5M2Bits::from_bits(0)),
        PcuScalarType::F32 => TensorScalarValue::F32(0.0),
        PcuScalarType::F64 => TensorScalarValue::F64(0.0),
        PcuScalarType::F128 => TensorScalarValue::F128(fusion_pcu::PcuF128Bits::decode_le([0; 16])),
        PcuScalarType::F256 => TensorScalarValue::F256(fusion_pcu::PcuF256Bits::decode_le([0; 32])),
        _ => return None,
    };
    T::as_scalar(value)
}
