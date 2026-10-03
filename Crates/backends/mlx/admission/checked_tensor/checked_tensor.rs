//! Exact selected byte-aligned Identity and six-format `ReLU`, retaining unused checked effects.
#[rustfmt::skip]
use std::rc::Rc;
#[rustfmt::skip]
use fusion_pcu::{
    PcuScalarType,
    PcuImplementationRequirements,
    PcuRangePolicy,
    PcuReproducibility,
    PcuNumericalRequirement,
    PcuImplementationId,
    PcuDeviceIdentity,
};
#[rustfmt::skip]
use fusion_pcu::dialect::tensor::{
    TensorOwnedSelectedProgram,
    TensorOwnedSelectedOperation,
    OpDescriptor,
    ValueId,
    TensorUnsupportedReason,
};
/// Frozen one-input byte-aligned Identity or six-format `ReLU` with at most one checked effect.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MlxCheckedTensorPlan {
    input: ValueId,
    output: ValueId,
    relu: Option<ValueId>,
    scalar: PcuScalarType,
    shape: Rc<[usize]>,
    count: usize,
    requirements: PcuImplementationRequirements,
}
impl MlxCheckedTensorPlan {
    /// Admits the selected source closure and explicit immutable request without runtime work.
    /// Even an unused selected checked `ReLU` must execute before an identity result is published.
    ///
    /// # Errors
    /// Rejects extra inputs/effects, rewrites, unproved types, shape, node/request mismatch,
    /// Portable or graph Clamp (which has no stored node-level semantic contract).
    pub fn assess_program(
        program: &TensorOwnedSelectedProgram,
        requirements: PcuImplementationRequirements,
    ) -> Result<Self, TensorUnsupportedReason> {
        if requirements.range_policy != PcuRangePolicy::Reject {
            return Err(TensorUnsupportedReason::Other(
                "checked MLX graph range policy requires Reject".into(),
            ));
        }
        if requirements.numerical_options.reproducibility != PcuReproducibility::Unspecified {
            return Err(TensorUnsupportedReason::NumericalPolicy {
                requirement: PcuNumericalRequirement::Reproducibility,
                options: requirements.numerical_options,
            });
        }
        let ([input], [output]) = (program.input_values(), program.output_values()) else {
            return Err(TensorUnsupportedReason::Operation);
        };
        if program.node_order().is_empty() || program.node_order().len()>2 || program.node_order()[0]!=*input || program.operations().len()!=program.node_order().len() || !program.operations().iter().zip(program.node_order()).all(|(operation,value)|matches!(operation,TensorOwnedSelectedOperation::Node{value:actual} if actual==value)){return Err(TensorUnsupportedReason::Operation);}
        let graph = program.graph();
        let input_node = graph
            .node(*input)
            .map_err(|_| TensorUnsupportedReason::Operation)?;
        if !matches!(input_node.op, OpDescriptor::Input) {
            return Err(TensorUnsupportedReason::Operation);
        }
        let scalar = input_node.scalar_type;
        if matches!(
            scalar,
            PcuScalarType::Bool | PcuScalarType::I4 | PcuScalarType::U4
        ) {
            return Err(TensorUnsupportedReason::ElementType);
        }
        let count = input_node
            .shape
            .iter()
            .try_fold(1_usize, |count, &dimension| count.checked_mul(dimension))
            .ok_or(TensorUnsupportedReason::Shape)?;
        let width = usize::from(scalar.bit_width()) / 8;
        let bytes = count
            .checked_mul(width)
            .ok_or(TensorUnsupportedReason::Shape)?;
        if count == 0
            || i32::try_from(bytes / width.min(4)).is_err()
            || isize::try_from(bytes).is_err()
            || count
                .checked_mul(4)
                .is_none_or(|bytes| isize::try_from(bytes).is_err())
        {
            return Err(TensorUnsupportedReason::Shape);
        }
        let mut relu = None;
        for &value in program.node_order() {
            let node = graph
                .node(value)
                .map_err(|_| TensorUnsupportedReason::Operation)?;
            if node.scalar_type != scalar {
                return Err(TensorUnsupportedReason::ElementType);
            }
            if node.shape != input_node.shape {
                return Err(TensorUnsupportedReason::Shape);
            }
            if value == *input {
                // Input leaves transport storage; their pre-capture default metadata does
                // not authorize or constrain arithmetic. The explicit request stays frozen.
                continue;
            }
            if node.numerical_options != requirements.numerical_options {
                return Err(TensorUnsupportedReason::Other(
                    "checked MLX graph node/request options mismatch".into(),
                ));
            }
            if !checked_float(scalar) {
                return Err(TensorUnsupportedReason::ElementType);
            }
            if !matches!(node.op,OpDescriptor::Relu{input:operand} if operand==*input)
                || relu.is_some()
                || node.numerical_mode.is_some()
            {
                return Err(TensorUnsupportedReason::Operation);
            }
            if node.float_underflow_policy != Some(requirements.float_underflow) {
                return Err(TensorUnsupportedReason::UnderflowPolicy(
                    requirements.float_underflow,
                ));
            }
            relu = Some(value);
        }
        if *output != *input && Some(*output) != relu {
            return Err(TensorUnsupportedReason::Operation);
        }
        Ok(Self {
            input: *input,
            output: *output,
            relu,
            scalar,
            shape: Rc::from(input_node.shape),
            count,
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
    /// Retains the frozen immutable shape without allocating a fresh shape vector per call.
    #[must_use]
    pub fn shape_owner(&self) -> Rc<[usize]> {
        Rc::clone(&self.shape)
    }
    #[must_use]
    pub const fn element_count(&self) -> usize {
        self.count
    }
    #[must_use]
    pub const fn requirements(&self) -> PcuImplementationRequirements {
        self.requirements
    }
    /// Exact implementation family; activation provenance remains provided by actual discovery.
    #[must_use]
    pub fn implementation_id(&self, device: PcuDeviceIdentity) -> PcuImplementationId {
        let format = match self.scalar {
            PcuScalarType::F16 => 0,
            PcuScalarType::BF16 => 16,
            PcuScalarType::F8E4M3FN => 32,
            PcuScalarType::F8E5M2 => 48,
            PcuScalarType::F32 if self.relu.is_some() => 64,
            PcuScalarType::F64 if self.relu.is_some() => 80,
            _ => {
                return PcuImplementationId {
                    device,
                    executor: crate::MLX_EXECUTOR,
                    local_id: 0x400 + self.scalar as u32,
                    revision: 0x0003_0020_0003_0300,
                };
            }
        };
        PcuImplementationId {
            device,
            executor: crate::MLX_EXECUTOR,
            local_id: 0x200
                + format
                + u32::from(self.relu.is_some())
                + 2 * u32::from(self.output == self.input),
            revision: if format >= 64 {
                0x0003_0020_0003_0900
            } else {
                0x0003_0020_0003_0200
            },
        }
    }
}
#[cfg(test)]
#[path = "tests.rs"]
mod tests;

const fn checked_float(scalar: PcuScalarType) -> bool {
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
