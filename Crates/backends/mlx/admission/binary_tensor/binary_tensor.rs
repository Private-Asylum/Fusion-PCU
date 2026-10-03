//! Detached single checked binary effect with exact actually-selected input roles.
#[rustfmt::skip]
use std::rc::Rc;
#[rustfmt::skip]
use fusion_pcu::{
    PcuImplementationRequirements,
    PcuImplementationId,
    PcuDeviceIdentity,
    PcuScalarType,
    PcuRangePolicy,
    PcuReproducibility,
    PcuDispatchFloatBinaryOp,
};
#[rustfmt::skip]
use fusion_pcu::dialect::tensor::{
    TensorOwnedSelectedProgram,
    TensorOwnedSelectedOperation,
    TensorUnsupportedReason,
    OpDescriptor,
    ValueId,
};
/// One exact six-floating checked binary effect and fresh owner publication.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MlxCheckedTensorBinaryPlan {
    inputs: Rc<[ValueId]>,
    operands: [usize; 2],
    effect: ValueId,
    output: ValueId,
    identity_output: Option<usize>,
    operation: PcuDispatchFloatBinaryOp,
    scalar: PcuScalarType,
    shape: Rc<[usize]>,
    count: usize,
    bytes: usize,
    requirements: PcuImplementationRequirements,
}
impl MlxCheckedTensorBinaryPlan {
    /// Assesses one checked Add/Sub/Mul/Div (six floats).
    /// The selected output is the effect or an Input after the same checked unused effect.
    /// # Errors
    /// Rejects extra effects, mismatched policy/type/shape, Clamp/Portable and unsupported types.
    #[allow(clippy::too_many_lines)] // One cold detached closure assessment; each branch rejects a distinct schema defect.
    pub fn assess_program(
        program: &TensorOwnedSelectedProgram,
        requirements: PcuImplementationRequirements,
    ) -> Result<Self, TensorUnsupportedReason> {
        let inputs = program.input_values();
        let [output] = program.output_values() else {
            return Err(TensorUnsupportedReason::Operation);
        };
        if !(1..=2).contains(&inputs.len())
            || requirements.range_policy != PcuRangePolicy::Reject
            || requirements.numerical_options.reproducibility != PcuReproducibility::Unspecified
        {
            return Err(TensorUnsupportedReason::Operation);
        }
        let order = program.node_order();
        if order.len() != inputs.len() + 1
            || order[..inputs.len()] != *inputs
            || program.operations().len() != order.len()
            || !program
                .operations()
                .iter()
                .zip(order)
                .all(|(op, id)| matches!(op,TensorOwnedSelectedOperation::Node{value} if value==id))
        {
            return Err(TensorUnsupportedReason::Operation);
        }
        let first = program
            .graph()
            .node(inputs[0])
            .map_err(|_| TensorUnsupportedReason::Operation)?;
        for &input in inputs {
            let node = program
                .graph()
                .node(input)
                .map_err(|_| TensorUnsupportedReason::Operation)?;
            if !matches!(node.op, OpDescriptor::Input)
                || node.scalar_type != first.scalar_type
                || node.shape != first.shape
            {
                return Err(TensorUnsupportedReason::Operation);
            }
        }
        let effect = *order.last().ok_or(TensorUnsupportedReason::Operation)?;
        let node = program
            .graph()
            .node(effect)
            .map_err(|_| TensorUnsupportedReason::Operation)?;
        let (operation, left, right) = match node.op {
            OpDescriptor::Add { left, right } => (PcuDispatchFloatBinaryOp::Add, left, right),
            OpDescriptor::Sub { left, right } => (PcuDispatchFloatBinaryOp::Sub, left, right),
            OpDescriptor::Mul { left, right } => (PcuDispatchFloatBinaryOp::Mul, left, right),
            OpDescriptor::Div { left, right } => (PcuDispatchFloatBinaryOp::Div, left, right),
            _ => return Err(TensorUnsupportedReason::Operation),
        };
        let operands = [left, right].map(|id| {
            inputs
                .iter()
                .position(|&input| input == id)
                .ok_or(TensorUnsupportedReason::Operation)
        });
        let [left_slot, right_slot] = operands;
        let operands = [left_slot?, right_slot?];
        // No unused declaration is fabricated into a second runtime allocation.
        if (0..inputs.len()).any(|slot| !operands.contains(&slot)) {
            return Err(TensorUnsupportedReason::Operation);
        }
        if !matches!(
            first.scalar_type,
            PcuScalarType::F16
                | PcuScalarType::BF16
                | PcuScalarType::F8E4M3FN
                | PcuScalarType::F8E5M2
                | PcuScalarType::F32
                | PcuScalarType::F64
        ) {
            return Err(TensorUnsupportedReason::ElementType);
        }
        if node.shape != first.shape
            || node.scalar_type != first.scalar_type
            || node.numerical_mode.is_some()
            || node.numerical_options != requirements.numerical_options
        {
            return Err(TensorUnsupportedReason::Operation);
        }
        if node.float_underflow_policy != Some(requirements.float_underflow) {
            return Err(TensorUnsupportedReason::UnderflowPolicy(
                requirements.float_underflow,
            ));
        }
        let identity_output = if *output == effect {
            None
        } else {
            Some(
                inputs
                    .iter()
                    .position(|input| input == output)
                    .ok_or(TensorUnsupportedReason::Operation)?,
            )
        };
        let count = first
            .shape
            .iter()
            .try_fold(1_usize, |n, &d| n.checked_mul(d))
            .ok_or(TensorUnsupportedReason::Shape)?;
        let bytes = count
            .checked_mul(usize::from(first.scalar_type.bit_width()) / 8)
            .ok_or(TensorUnsupportedReason::Shape)?;
        if count == 0
            || i32::try_from(bytes / (usize::from(first.scalar_type.bit_width()) / 8).min(4))
                .is_err()
            || isize::try_from(bytes).is_err()
            || count
                .checked_mul(4)
                .is_none_or(|n| isize::try_from(n).is_err())
        {
            return Err(TensorUnsupportedReason::Shape);
        }
        Ok(Self {
            inputs: Rc::from(inputs),
            operands,
            effect,
            output: *output,
            identity_output,
            operation,
            scalar: first.scalar_type,
            shape: Rc::from(first.shape),
            count,
            bytes,
            requirements,
        })
    }
    #[must_use]
    pub fn input_values(&self) -> &[ValueId] {
        &self.inputs
    }
    #[must_use]
    pub const fn operand_inputs(&self) -> [usize; 2] {
        self.operands
    }
    #[must_use]
    pub const fn effect(&self) -> ValueId {
        self.effect
    }
    #[must_use]
    pub const fn output(&self) -> ValueId {
        self.output
    }
    #[must_use]
    pub const fn operation(&self) -> PcuDispatchFloatBinaryOp {
        self.operation
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
    /// Exact family identity; device provenance comes from actual retained discovery/session.
    #[must_use]
    pub fn implementation_id(&self, device: PcuDeviceIdentity) -> PcuImplementationId {
        let operation = match self.operation {
            PcuDispatchFloatBinaryOp::Add => 0,
            PcuDispatchFloatBinaryOp::Sub => 1,
            PcuDispatchFloatBinaryOp::Mul => 2,
            PcuDispatchFloatBinaryOp::Div => 3,
        };
        PcuImplementationId {
            device,
            executor: crate::MLX_EXECUTOR,
            local_id: 0x800
                + 32 * self.scalar as u32
                + operation
                + 4 * match self.identity_output {
                    None => 0,
                    Some(0) => 1,
                    Some(_) => 2,
                }
                + 16 * u32::from(self.inputs.len() == 1),
            revision: 0x0003_0020_0003_0800,
        }
    }
    pub(crate) const fn identity_output(&self) -> Option<usize> {
        self.identity_output
    }
}
