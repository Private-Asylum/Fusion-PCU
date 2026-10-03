//! Cold leaf tensor selection backed by actual retained Vulkan carrier owners.
#[rustfmt::skip]
use core::fmt;
#[rustfmt::skip]
use std::rc::Rc;
#[rustfmt::skip]
use fusion_pcu::{
    PcuReproducibility,
    PcuScalar,
};
#[rustfmt::skip]
use fusion_pcu::dialect::tensor::{
    Graph,
    OpDescriptor,
    TensorElement,
    TensorError,
    ValueId,
};
#[rustfmt::skip]
use crate::{
    PcuVulkanBackend,
    PcuVulkanError,
    PcuVulkanOwnedBuffer,
    PcuVulkanPreparedCarrierCopy,
};

#[path = "checked/checked.rs"]
mod checked;
pub use checked::PcuVulkanPreparedTensorGraph;

/// Leaf planner failures retain graph metadata separately from native runtime failures.
#[derive(Debug)]
pub enum PcuVulkanTensorError {
    Graph(TensorError),
    Native(PcuVulkanError),
    UnsupportedNode(ValueId),
    OutputCount(usize),
}
impl fmt::Display for PcuVulkanTensorError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Graph(error) => write!(formatter, "Vulkan tensor graph: {error}"),
            Self::Native(error) => write!(formatter, "Vulkan tensor execution: {error}"),
            Self::UnsupportedNode(value) => {
                write!(formatter, "unsupported Vulkan tensor node: {value:?}")
            }
            Self::OutputCount(count) => write!(
                formatter,
                "Vulkan leaf planner requires one output, received {count}"
            ),
        }
    }
}
impl core::error::Error for PcuVulkanTensorError {}
impl From<TensorError> for PcuVulkanTensorError {
    fn from(error: TensorError) -> Self {
        Self::Graph(error)
    }
}
impl From<PcuVulkanError> for PcuVulkanTensorError {
    fn from(error: PcuVulkanError) -> Self {
        Self::Native(error)
    }
}

/// Frozen typed input schema; execution never searches or rebuilds the graph.
pub struct PcuVulkanTensorBinding {
    pub value: ValueId,
    pub shape: Rc<[usize]>,
    pub element_count: usize,
}

/// Explicit host upload or borrowed same-session native storage.
pub enum PcuVulkanTensorInput<'a, T: PcuScalar> {
    Host(&'a [T]),
    Owned(&'a PcuVulkanOwnedBuffer<T>),
}

/// One selected Input/Constant/Uniform leaf across all22 sealed carrier representations.
///
/// Input identity selects the graph input itself; no arithmetic, selection, compound or Portable
/// conformance is inferred. Preparation detaches constant bytes and immutable output shape.
/// Warm execution reuses upload/command/fence resources and allocates one fresh native output.
pub struct PcuVulkanPreparedScalarTensorGraph<T: TensorElement + PcuScalar> {
    copy: PcuVulkanPreparedCarrierCopy<T>,
    inputs: Vec<PcuVulkanTensorBinding>,
    output: PcuVulkanTensorBinding,
    constant: Option<PcuVulkanOwnedBuffer<T>>,
}

impl<T: TensorElement + PcuScalar> PcuVulkanPreparedScalarTensorGraph<T> {
    /// Freezes exactly one selected leaf and allocates its native transfer workspace.
    ///
    /// # Errors
    /// Rejects wrong graph/type, nonleaf dependencies, Portable, shape overflow or native errors.
    pub fn prepare(
        backend: &PcuVulkanBackend,
        graph: &Graph,
        outputs: &[ValueId],
    ) -> Result<Self, PcuVulkanTensorError> {
        if outputs.len() != 1 {
            return Err(PcuVulkanTensorError::OutputCount(outputs.len()));
        }
        let plan = graph.execution_plan_for_outputs(outputs)?;
        let mut nodes = plan.nodes();
        let node = nodes.next().ok_or(TensorError::EmptyOutputs)?;
        if let Some(extra) = nodes.next() {
            return Err(PcuVulkanTensorError::UnsupportedNode(extra.value));
        }
        if node.scalar_type != T::TYPE {
            return Err(TensorError::ScalarTypeMismatch {
                value: node.value,
                expected: T::TYPE,
                actual: node.scalar_type,
            }
            .into());
        }
        if node.numerical_options.reproducibility != PcuReproducibility::Unspecified {
            return Err(TensorError::UnsupportedNumericalOptions {
                value: node.value,
                options: node.numerical_options,
            }
            .into());
        }
        let element_count = node
            .shape
            .iter()
            .try_fold(1_usize, |count, dimension| count.checked_mul(*dimension))
            .ok_or(TensorError::ShapeOverflow)?;
        let shape: Rc<[usize]> = Rc::from(node.shape);
        let (constant, inputs) = match node.op {
            OpDescriptor::Input => (
                None,
                vec![PcuVulkanTensorBinding {
                    value: node.value,
                    shape: Rc::clone(&shape),
                    element_count,
                }],
            ),
            OpDescriptor::Constant(value) => {
                let value = T::as_value(value).ok_or(TensorError::UnsupportedScalarType {
                    value: node.value,
                    scalar_type: node.scalar_type,
                })?;
                (Some(backend.upload_owned(value.data())?), Vec::new())
            }
            OpDescriptor::Uniform { value } => {
                let value = T::as_scalar(*value).ok_or(TensorError::UnsupportedScalarType {
                    value: node.value,
                    scalar_type: node.scalar_type,
                })?;
                (
                    Some(backend.upload_owned(&vec![value; element_count])?),
                    Vec::new(),
                )
            }
            _ => return Err(PcuVulkanTensorError::UnsupportedNode(node.value)),
        };
        Ok(Self {
            copy: backend.prepare_carrier_copy(element_count)?,
            inputs,
            constant,
            output: PcuVulkanTensorBinding {
                value: node.value,
                shape,
                element_count,
            },
        })
    }

    /// Selected input bindings in cold frozen order (zero or one leaf).
    #[must_use]
    pub fn input_bindings(&self) -> &[PcuVulkanTensorBinding] {
        &self.inputs
    }

    /// Immutable selected shape, shared without a per-call shape allocation.
    #[must_use]
    pub const fn output_shape(&self) -> &Rc<[usize]> {
        &self.output.shape
    }

    /// Executes a leaf transfer into a fresh terminal owner without rebuilding the graph.
    ///
    /// # Errors
    /// Rejects input count/extent/session before submission; native failures publish no owner.
    pub fn execute_owned(
        &mut self,
        inputs: &[PcuVulkanTensorInput<'_, T>],
    ) -> Result<PcuVulkanOwnedBuffer<T>, PcuVulkanTensorError> {
        if inputs.len() != self.inputs.len() {
            return Err(TensorError::DataLength {
                expected: self.inputs.len(),
                actual: inputs.len(),
            }
            .into());
        }
        if let Some(constant) = &self.constant {
            return self.copy.copy_owned(constant).map_err(Into::into);
        }
        match &inputs[0] {
            PcuVulkanTensorInput::Host(input) => self.copy.copy_host(input).map_err(Into::into),
            PcuVulkanTensorInput::Owned(input) => self.copy.copy_owned(input).map_err(Into::into),
        }
    }
}
