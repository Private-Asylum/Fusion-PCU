//! Cold leaf decomposition of an already selected schedule; no execution or storage ownership.

#[rustfmt::skip]
use super::{
    Graph,
    Node,
    Op,
    TensorArithmeticCapability,
    TensorArithmeticRewritePolicy,
    TensorError,
    TensorOwnedSelectedOperation,
    TensorOwnedSelectedProgram,
    TensorPointwiseGroupingPolicy,
    ValueId,
    operands,
};
use alloc::vec::Vec;
use core::fmt;
use crate::PcuImplementationRequirements;

/// A unique leaf input and the parent value that must supply its device storage.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TensorFragmentInput {
    pub parent_value: ValueId,
    pub child_value: ValueId,
}

/// A selected schedule cannot be faithfully described by one unfused leaf.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TensorFragmentError {
    /// This value is foreign, suppressed, or outside the selected dependency closure.
    NotSelected(ValueId),
    /// Inputs, constants and uniforms are producer bindings, not computed leaf stages.
    NotComputed(ValueId),
    /// Decomposition would undo an explicitly selected arithmetic rewrite or fusion.
    SelectedFusion { operation_index: usize },
    /// The retained source node differs from an explicitly selected arithmetic replacement.
    SelectedRewrite { operation_index: usize },
    /// Child identity, shape, liveness or storage validation failed.
    Tensor(TensorError),
}

impl fmt::Display for TensorFragmentError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}
impl core::error::Error for TensorFragmentError {}
impl From<TensorError> for TensorFragmentError {
    fn from(error: TensorError) -> Self {
        Self::Tensor(error)
    }
}

/// One cold-prepared leaf with explicit parent provenance and operand roles.
///
/// All operands become child inputs, including parent constants and uniforms. The backend
/// retains the parent program and initializes those producers separately; this structure never
/// clones their payloads, uploads data, changes residency or authorizes storage aliasing.
/// Repeated operands share one child input while retaining every positional role.
///
/// Child output pinning enables ordinary leaf admission. Physical intermediate liveness and
/// scratch reuse remain governed by the *parent* program's selected operation liveness and
/// storage plan, not this child's isolated output lifetime. Backend preparation must freeze
/// bindings and leaf templates once; replay must not rebuild or traverse these fragments.
pub struct TensorOperationFragment {
    parent_value: ValueId,
    parent_operation_index: usize,
    inputs: Vec<TensorFragmentInput>,
    operand_input_indexes: Vec<usize>,
    child_output: ValueId,
    program: TensorOwnedSelectedProgram,
}

impl TensorOperationFragment {
    #[must_use]
    pub const fn parent_value(&self) -> ValueId {
        self.parent_value
    }

    /// Index into the original selected schedule and its aligned liveness/use-count tables.
    #[must_use]
    pub const fn parent_operation_index(&self) -> usize {
        self.parent_operation_index
    }

    /// Unique input bindings, in first operand occurrence order.
    #[must_use]
    pub fn inputs(&self) -> &[TensorFragmentInput] {
        &self.inputs
    }

    /// Original operand roles mapped into [`Self::inputs`]; duplicates are intentional.
    #[must_use]
    pub fn operand_input_indexes(&self) -> &[usize] {
        &self.operand_input_indexes
    }

    #[must_use]
    pub const fn child_output(&self) -> ValueId {
        self.child_output
    }

    #[must_use]
    pub const fn program(&self) -> &TensorOwnedSelectedProgram {
        &self.program
    }

    /// Resolve this computed leaf's exact captured numerical request during preparation.
    ///
    /// A nested function can have a different underflow or compound checking policy from
    /// its caller. The computed output's metadata is authoritative; synthetic operand
    /// inputs must not replace it with the policies of their earlier producers. Independent
    /// arithmetic/precision/reproducibility options are always retained from this node.
    ///
    /// The enclosing request supplies the range policy, which is not currently a per-node
    /// tensor property, and defaults for absent mode/underflow metadata. Scalar operations
    /// have no compound mode; integer operations have no floating underflow requirement.
    /// Those defaults do not introduce a compound stage or floating arithmetic. Backends
    /// still admit the resulting exact operation/type/shape tuple separately; this method
    /// neither grants capabilities nor relaxes unsupported requests. Freeze its result in
    /// the prepared leaf, rather than invoking it on replay.
    #[must_use]
    pub fn implementation_requirements(
        &self,
        enclosing: PcuImplementationRequirements,
    ) -> PcuImplementationRequirements {
        let node = &self.program.graph.nodes[self.child_output.index];
        PcuImplementationRequirements {
            numerical_mode: node.numerical_mode.unwrap_or(enclosing.numerical_mode),
            numerical_options: node.numerical_options,
            float_underflow: node
                .float_underflow_policy
                .unwrap_or(enclosing.float_underflow),
            range_policy: enclosing.range_policy,
        }
    }

    /// Transfer the child into a backend-owned immutable prepared template.
    /// Copy the cold binding/provenance tables first if they are needed alongside that owner.
    #[must_use]
    pub fn into_program(self) -> TensorOwnedSelectedProgram {
        self.program
    }

    /// Recover parent identity for a child fault or binding. Foreign IDs remain invalid.
    #[must_use]
    pub fn parent_value_for(&self, child: ValueId) -> Option<ValueId> {
        if child == self.child_output {
            Some(self.parent_value)
        } else {
            self.inputs
                .iter()
                .find(|input| input.child_value == child)
                .map(|input| input.parent_value)
        }
    }
}

impl TensorOwnedSelectedProgram {
    /// Describe one selected computed node as a cold leaf without changing its numerical law.
    ///
    /// Shape, scalar representation, mode, underflow and every independent numerical option
    /// are preserved exactly. The child uses disabled arithmetic rewriting and grouping.
    ///
    /// # Errors
    /// Returns a typed error for an unselected/foreign value, producer, selected fusion, or
    /// invalid child identity/storage extent. A fused schedule is never silently decomposed.
    pub fn operation_fragment(
        &self,
        value: ValueId,
    ) -> Result<TensorOperationFragment, TensorFragmentError> {
        let position = self
            .operation_index_of(value)
            .ok_or(TensorFragmentError::NotSelected(value))?;
        if self.rewrites.iter().any(|rewrite| rewrite.output == value) {
            return Err(TensorFragmentError::SelectedRewrite {
                operation_index: position,
            });
        }
        if !matches!(self.operations[position], TensorOwnedSelectedOperation::Node { value: actual } if actual == value)
        {
            return Err(TensorFragmentError::SelectedFusion {
                operation_index: position,
            });
        }
        let node = &self.graph.nodes[value.index];
        if matches!(node.op, Op::Input | Op::Constant(_) | Op::Uniform(_)) {
            return Err(TensorFragmentError::NotComputed(value));
        }
        let mut child = Graph::try_new()?;
        let mut inputs: Vec<TensorFragmentInput> = Vec::new();
        let mut roles = Vec::new();
        for parent in operands(&node.op) {
            let index = inputs
                .iter()
                .position(|input| input.parent_value == parent)
                .unwrap_or_else(|| {
                    let source = &self.graph.nodes[parent.index];
                    let child_value = append(&mut child, source, Op::Input);
                    inputs.push(TensorFragmentInput {
                        parent_value: parent,
                        child_value,
                    });
                    inputs.len() - 1
                });
            roles.push(index);
        }
        let operand = |role: usize| inputs[roles[role]].child_value;
        let operation = match node.op {
            Op::Add(..) => Op::Add(operand(0), operand(1)),
            Op::Sub(..) => Op::Sub(operand(0), operand(1)),
            Op::Mul(..) => Op::Mul(operand(0), operand(1)),
            Op::Div(..) => Op::Div(operand(0), operand(1)),
            Op::Relu(_) => Op::Relu(operand(0)),
            Op::ReluBackward(..) => Op::ReluBackward(operand(0), operand(1)),
            Op::MatMul(_, _, left, right) => Op::MatMul(operand(0), operand(1), left, right),
            Op::SgdUpdate(_, _, rate) => Op::SgdUpdate(operand(0), operand(1), rate),
            Op::MeanSquaredError(..) => Op::MeanSquaredError(operand(0), operand(1)),
            Op::Input | Op::Constant(_) | Op::Uniform(_) => {
                return Err(TensorFragmentError::NotComputed(value));
            }
        };
        let output = append(&mut child, node, operation);
        let program = child.into_selected_program(
            &[output],
            TensorArithmeticRewritePolicy::Disabled,
            TensorArithmeticCapability::Strict,
            TensorPointwiseGroupingPolicy::Disabled,
        )?;
        Ok(TensorOperationFragment {
            parent_value: value,
            parent_operation_index: position,
            inputs,
            operand_input_indexes: roles,
            child_output: output,
            program,
        })
    }

    /// Cold-decompose the complete selected computed schedule in its original order.
    ///
    /// Includes mandatory discarded checked effects. Parent inputs/constants/uniforms remain
    /// producer bindings; inspect the retained parent graph for their initialization. Final
    /// selected outputs may be inputs and are still taken from [`Self::output_values`].
    /// No execution, host transfer, resource lease or allocation-bank decision occurs here.
    ///
    /// # Errors
    /// Returns the first selected-fusion or child-validation error. Callers must not silently
    /// skip an unsupported stage or publish any output before all mandatory effects complete.
    pub fn operation_fragments(&self) -> Result<Vec<TensorOperationFragment>, TensorFragmentError> {
        let mut fragments = Vec::new();
        for (position, operation) in self.operations.iter().enumerate() {
            let TensorOwnedSelectedOperation::Node { value } = *operation else {
                return Err(TensorFragmentError::SelectedFusion {
                    operation_index: position,
                });
            };
            if !matches!(
                self.graph.nodes[value.index].op,
                Op::Input | Op::Constant(_) | Op::Uniform(_)
            ) {
                fragments.push(self.operation_fragment(value)?);
            }
        }
        Ok(fragments)
    }
}

/// Copy metadata only. A constant's large payload is never cloned into a leaf input.
fn append(graph: &mut Graph, source: &Node, op: Op) -> ValueId {
    let input = matches!(op, Op::Input);
    let value = ValueId {
        graph_id: graph.id,
        index: graph.nodes.len(),
    };
    graph.nodes.push(Node {
        op,
        shape: source.shape.clone(),
        scalar_type: source.scalar_type,
        // A synthetic Input is a binding, not a replay of its parent's producer arithmetic.
        float_underflow_policy: if input {
            None
        } else {
            source.float_underflow_policy
        },
        numerical_mode: if input { None } else { source.numerical_mode },
        numerical_options: source.numerical_options,
    });
    value
}

#[cfg(test)]
#[path = "tests/tests.rs"]
mod tests;
