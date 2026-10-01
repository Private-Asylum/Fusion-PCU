//! Cold admission for the first explicitly delegated MLX tensor primitive.

#[rustfmt::skip]
use fusion_pcu::{
    PcuCompoundArithmeticPolicy,
    PcuFloatUnderflowPolicy,
    PcuNumericalMode,
    PcuNumericalOptions,
    PcuNumericalRequirement,
    PcuPrecisionPolicy,
    PcuReproducibility,
    PcuScalarType,
};
#[rustfmt::skip]
use fusion_pcu::dialect::tensor::{
    Graph,
    NodeDescriptor,
    OpDescriptor,
    TensorExecutionRoute,
    TensorOperationAssessor,
    TensorOperationSupport,
    TensorOwnedSelectedOperation,
    TensorOwnedSelectedProgram,
    TensorUnsupportedReason,
    ValueId,
};

/// Immutable, graph-provenant dense F32 matrix-product contract.
///
/// MLX chooses intermediate precision, reduction order and contraction under explicit
/// `BackendDefined` and `BackendOptimized` permissions. This does not certify checked
/// faults, gradual underflow, portable bits or a native Metal allocation interchange.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MlxMatmulPlan {
    pub(crate) output: ValueId,
    pub(crate) left: ValueId,
    pub(crate) right: ValueId,
    pub(crate) left_shape: [usize; 2],
    pub(crate) right_shape: [usize; 2],
    pub(crate) output_shape: [usize; 2],
    pub(crate) options: PcuNumericalOptions,
}

impl MlxMatmulPlan {
    /// Admits the selected closure of an existing captured program without a second frontend.
    /// This bounded source slice requires two input values and one unrewritten matrix output.
    ///
    /// # Errors
    /// Returns unsupported topology or the complete matrix primitive's contract requirement.
    pub fn assess_program(
        program: &TensorOwnedSelectedProgram,
    ) -> Result<Self, TensorUnsupportedReason> {
        let [output] = program.output_values() else {
            return Err(TensorUnsupportedReason::Operation);
        };
        let graph = program.graph();
        let node = graph
            .node(*output)
            .map_err(|_| TensorUnsupportedReason::Operation)?;
        let plan = Self::assess(graph, node)?;
        if program.input_values().len() != 2
            || program.node_order().len() != 3
            || program.operations().len() != 3
            || plan.left == plan.right
            || !program.input_values().contains(&plan.left)
            || !program.input_values().contains(&plan.right)
        {
            return Err(TensorUnsupportedReason::Operation);
        }
        for &input in &[plan.left, plan.right] {
            if !matches!(
                graph.node(input).map(|node| node.op),
                Ok(OpDescriptor::Input)
            ) {
                return Err(TensorUnsupportedReason::Operation);
            }
        }
        if !program
            .operations()
            .iter()
            .zip(program.node_order())
            .all(|(operation, expected)| {
                matches!(operation, TensorOwnedSelectedOperation::Node { value } if value == expected)
            })
        {
            return Err(TensorUnsupportedReason::Operation);
        }
        Ok(plan)
    }

    /// Admits an authentic graph node without loading or activating MLX.
    ///
    /// # Errors
    /// Returns the exact unsupported operation, shape, scalar or numerical requirement.
    pub fn assess(
        graph: &Graph,
        node: NodeDescriptor<'_>,
    ) -> Result<Self, TensorUnsupportedReason> {
        let OpDescriptor::MatMul {
            left,
            right,
            transpose_left,
            transpose_right,
        } = node.op
        else {
            return Err(TensorUnsupportedReason::Operation);
        };
        if graph.node(node.value).ok() != Some(node) {
            return Err(TensorUnsupportedReason::Operation);
        }
        if node.numerical_options.reproducibility != PcuReproducibility::Unspecified {
            return Err(TensorUnsupportedReason::NumericalPolicy {
                requirement: PcuNumericalRequirement::Reproducibility,
                options: node.numerical_options,
            });
        }
        if node.numerical_mode != Some(PcuNumericalMode::Boundary)
            || node.numerical_options.compound_arithmetic
                != PcuCompoundArithmeticPolicy::BackendDefined
        {
            return Err(TensorUnsupportedReason::NumericalPolicy {
                requirement: PcuNumericalRequirement::CompoundArithmetic,
                options: node.numerical_options,
            });
        }
        if node.numerical_options.precision != PcuPrecisionPolicy::BackendOptimized {
            return Err(TensorUnsupportedReason::NumericalPolicy {
                requirement: PcuNumericalRequirement::Precision,
                options: node.numerical_options,
            });
        }
        match node.float_underflow_policy {
            Some(PcuFloatUnderflowPolicy::IeeeAfterRounding) => (),
            Some(policy) => return Err(TensorUnsupportedReason::UnderflowPolicy(policy)),
            None => return Err(TensorUnsupportedReason::Operation),
        }
        let a = graph
            .node(left)
            .map_err(|_| TensorUnsupportedReason::Operation)?;
        let b = graph
            .node(right)
            .map_err(|_| TensorUnsupportedReason::Operation)?;
        if [node.scalar_type, a.scalar_type, b.scalar_type] != [PcuScalarType::F32; 3] {
            return Err(TensorUnsupportedReason::ElementType);
        }
        if transpose_left || transpose_right {
            return Err(TensorUnsupportedReason::Layout);
        }
        let a = matrix_shape(a.shape)?;
        let b = matrix_shape(b.shape)?;
        let output = matrix_shape(node.shape)?;
        if a[1] != b[0] || output != [a[0], b[1]] {
            return Err(TensorUnsupportedReason::Shape);
        }
        Ok(Self {
            output: node.value,
            left,
            right,
            left_shape: a,
            right_shape: b,
            output_shape: output,
            options: node.numerical_options,
        })
    }

    /// Selected source output, retaining graph identity.
    #[must_use]
    pub const fn output(&self) -> ValueId {
        self.output
    }

    /// Selected source operands, retaining graph identity.
    #[must_use]
    pub const fn inputs(&self) -> [ValueId; 2] {
        [self.left, self.right]
    }

    /// Dense row-major output shape.
    #[must_use]
    pub const fn shape(&self) -> [usize; 2] {
        self.output_shape
    }

    /// Complete admitted numerical permissions frozen at preparation.
    #[must_use]
    pub const fn numerical_options(&self) -> PcuNumericalOptions {
        self.options
    }
}

pub fn matrix_shape(shape: &[usize]) -> Result<[usize; 2], TensorUnsupportedReason> {
    let [rows, columns] = shape else {
        return Err(TensorUnsupportedReason::Shape);
    };
    if *rows == 0
        || *columns == 0
        || i32::try_from(*rows).is_err()
        || i32::try_from(*columns).is_err()
        || rows
            .checked_mul(*columns)
            .and_then(|n| n.checked_mul(4))
            .is_none_or(|bytes| isize::try_from(bytes).is_err())
    {
        return Err(TensorUnsupportedReason::Shape);
    }
    Ok([*rows, *columns])
}

/// Pure numerical/shape assessor; runtime readiness is established separately.
#[derive(Clone, Copy, Debug, Default)]
pub struct MlxTensorAssessor;

impl TensorOperationAssessor for MlxTensorAssessor {
    fn assess_node(&self, graph: &Graph, node: NodeDescriptor<'_>) -> TensorOperationSupport {
        match MlxMatmulPlan::assess(graph, node) {
            Ok(_) => TensorOperationSupport::Supported {
                route: TensorExecutionRoute::Library,
                // Delegated MLX scheduling may materialize internal temporaries; zero is not proven.
                workspace_bytes: None,
            },
            Err(reason) => TensorOperationSupport::Unsupported { reason },
        }
    }
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
