//! Detached exact22 leaf ownership admission; no arithmetic or checked effects are erased.
#[rustfmt::skip]
use std::rc::Rc;
#[rustfmt::skip]
use fusion_pcu::{PcuImplementationRequirements,PcuScalarType,PcuRangePolicy,PcuReproducibility};
#[rustfmt::skip]
use fusion_pcu::dialect::tensor::{TensorOwnedSelectedProgram,TensorOwnedSelectedOperation,TensorUnsupportedReason,OpDescriptor,ValueId};
/// Frozen exact one-input leaf identity ownership plan.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MetalTensorPlan {
    input: ValueId,
    output: ValueId,
    relu: Option<ValueId>,
    scalar: PcuScalarType,
    shape: Rc<[usize]>,
    count: usize,
    bytes: usize,
    requirements: PcuImplementationRequirements,
}
impl MetalTensorPlan {
    /// Admits only a genuine selected Input leaf returning that same logical value.
    /// The explicit request is frozen; storage-only Input defaults cannot authorize arithmetic.
    /// # Errors
    /// Rejects arithmetic, unused checked effects, rewrites, extra selected inputs, packed types,
    /// zero/overflowing spans, graph Clamp and unqualified Portable requests.
    pub fn assess_program(
        program: &TensorOwnedSelectedProgram,
        requirements: PcuImplementationRequirements,
    ) -> Result<Self, TensorUnsupportedReason> {
        Self::assess_selected(program, requirements, false)
    }
    /// Admits one six-format checked `ReLU` effect on the sole Input, including an unused
    /// checked effect before an identity output. This entry is separate from leaf admission.
    /// # Errors
    /// Rejects mismatched scalar/shape/options/underflow, extra effects, Clamp or Portable.
    pub fn assess_relu_program(
        program: &TensorOwnedSelectedProgram,
        requirements: PcuImplementationRequirements,
    ) -> Result<Self, TensorUnsupportedReason> {
        let plan = Self::assess_selected(program, requirements, true)?;
        if plan.relu.is_none() {
            return Err(TensorUnsupportedReason::Operation);
        }
        Ok(plan)
    }
    fn assess_selected(
        program: &TensorOwnedSelectedProgram,
        requirements: PcuImplementationRequirements,
        allow_relu: bool,
    ) -> Result<Self, TensorUnsupportedReason> {
        if requirements.range_policy != PcuRangePolicy::Reject
            || requirements.numerical_options.reproducibility != PcuReproducibility::Unspecified
        {
            return Err(TensorUnsupportedReason::Operation);
        }
        let ([input], [output]) = (program.input_values(), program.output_values()) else {
            return Err(TensorUnsupportedReason::Operation);
        };
        if program.node_order().is_empty()
            || program.node_order()[0] != *input
            || program.node_order().len() > if allow_relu { 2 } else { 1 }
            || program.operations().len() != program.node_order().len()
            || !program.operations().iter().zip(program.node_order()).all(|(operation,value)|
                matches!(operation,TensorOwnedSelectedOperation::Node{value:actual} if actual==value))
        {
            return Err(TensorUnsupportedReason::Operation);
        }
        let node = program
            .graph()
            .node(*input)
            .map_err(|_| TensorUnsupportedReason::Operation)?;
        if !matches!(node.op, OpDescriptor::Input) {
            return Err(TensorUnsupportedReason::Operation);
        }
        if node.scalar_type.bit_width() < 8 {
            return Err(TensorUnsupportedReason::ElementType);
        }
        let relu = program.node_order().get(1).copied();
        if let Some(value) = relu {
            let effect = program
                .graph()
                .node(value)
                .map_err(|_| TensorUnsupportedReason::Operation)?;
            if !matches!(
                node.scalar_type,
                PcuScalarType::F16
                    | PcuScalarType::BF16
                    | PcuScalarType::F8E4M3FN
                    | PcuScalarType::F8E5M2
                    | PcuScalarType::F32
                    | PcuScalarType::F64
            ) {
                return Err(TensorUnsupportedReason::ElementType);
            }
            if effect.scalar_type != node.scalar_type
                || effect.shape != node.shape
                || !matches!(effect.op,OpDescriptor::Relu{input:operand} if operand==*input)
                || effect.numerical_mode.is_some()
                || effect.numerical_options != requirements.numerical_options
            {
                return Err(TensorUnsupportedReason::Operation);
            }
            if effect.float_underflow_policy != Some(requirements.float_underflow) {
                return Err(TensorUnsupportedReason::UnderflowPolicy(
                    requirements.float_underflow,
                ));
            }
        }
        if *output != *input && Some(*output) != relu {
            return Err(TensorUnsupportedReason::Operation);
        }
        let count = node
            .shape
            .iter()
            .try_fold(1_usize, |count, &dimension| count.checked_mul(dimension))
            .ok_or(TensorUnsupportedReason::Shape)?;
        let bytes = count
            .checked_mul(usize::from(node.scalar_type.bit_width()) / 8)
            .ok_or(TensorUnsupportedReason::Shape)?;
        if count == 0
            || u32::try_from(bytes).is_err()
            || count
                .checked_mul(4)
                .is_none_or(|status| u32::try_from(status).is_err())
        {
            return Err(TensorUnsupportedReason::Shape);
        }
        Ok(Self {
            input: *input,
            output: *output,
            relu,
            scalar: node.scalar_type,
            shape: Rc::from(node.shape),
            count,
            bytes,
            requirements,
        })
    }
    #[must_use]
    pub const fn input(&self) -> ValueId {
        self.input
    }
    #[must_use]
    pub const fn output(&self) -> ValueId {
        self.output
    }
    #[must_use]
    pub const fn relu_effect(&self) -> Option<ValueId> {
        self.relu
    }
    #[must_use]
    pub const fn scalar_type(&self) -> PcuScalarType {
        self.scalar
    }
    #[must_use]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
    #[must_use]
    pub fn shape_owner(&self) -> Rc<[usize]> {
        Rc::clone(&self.shape)
    }
    #[must_use]
    pub const fn element_count(&self) -> usize {
        self.count
    }
    #[must_use]
    pub const fn byte_len(&self) -> usize {
        self.bytes
    }
    #[must_use]
    pub const fn requirements(&self) -> PcuImplementationRequirements {
        self.requirements
    }
}
