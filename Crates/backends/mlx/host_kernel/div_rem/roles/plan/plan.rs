//! Detached actual-read roles for checked joint division; no native activation.
#[rustfmt::skip]
use fusion_pcu::{
    assess_checked_integer_div_rem_operands,
    CheckedIntegerDivRemOperandSchema,
    PcuBindingRef,
    PcuBindingType,
    PcuDispatchIndex,
    PcuDispatchKernelIr,
    PcuDispatchOp,
    PcuImplementationRequirements,
    PcuReproducibility,
    PcuScalarType,
    PcuValueType,
    PcuValueTypeCaps,
};
use crate::MlxError;
#[rustfmt::skip]
use super::super::plan::{
    checked_bytes,
    validate_reproducibility,
};

/// Immutable unique input spans and mathematical operand roles.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MlxCheckedDivRemRolePlan {
    scalar: PcuScalarType,
    count: usize,
    bytes: usize,
    inputs: [usize; 2],
    declarations: [PcuBindingRef; 4],
    declaration_count: usize,
    schema: CheckedIntegerDivRemOperandSchema,
    requirements: PcuImplementationRequirements,
}
impl MlxCheckedDivRemRolePlan {
    /// Assesses all fourteen integer widths, repeated or reversed operands and scalar reads.
    ///
    /// # Errors
    /// Rejects unsupported types, malformed SSA, extents, Clamp and ineligible Portable requests.
    pub fn assess(kernel: &PcuDispatchKernelIr<'_>) -> Result<Self, MlxError> {
        crate::dispatch_shape::require_non_nested(kernel)?;
        let invalid = || MlxError::InvalidRequest("unsupported MLX DivRem role profile".into());
        if cfg!(target_endian = "big") || kernel.entry.logical_shape[1..] != [1, 1] {
            return Err(invalid());
        }
        validate_reproducibility(kernel)?;
        let Some(PcuBindingType::Value(PcuValueType::Scalar(scalar))) =
            kernel.bindings.first().map(|binding| binding.binding_type)
        else {
            return Err(invalid());
        };
        let schema = assess_checked_integer_div_rem_operands(
            kernel,
            PcuValueType::Scalar(scalar),
            PcuValueTypeCaps::for_scalar(scalar),
        )
        .map_err(|_| invalid())?;
        let extent = match kernel.ops.first() {
            Some(PcuDispatchOp::GridStrideLoop { extent, .. }) => *extent,
            _ => kernel.entry.logical_shape[0],
        };
        let count = usize::try_from(extent).map_err(|_| MlxError::InvalidExtent)?;
        let bytes = checked_bytes(scalar, count)?;
        let inputs = schema.input_element_counts(count);
        for &extent in &inputs[..schema.input_bindings().len()] {
            checked_bytes(scalar, extent)?;
        }
        let mut declarations = [kernel.bindings[0].reference(); 4];
        for (slot, binding) in kernel.bindings.iter().enumerate() {
            declarations[slot] = binding.reference();
        }
        Ok(Self {
            scalar,
            count,
            bytes,
            inputs,
            declarations,
            declaration_count: kernel.bindings.len(),
            schema,
            requirements: kernel.numerical_requirements,
        })
    }
    #[must_use]
    pub const fn scalar_type(&self) -> PcuScalarType {
        self.scalar
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
    pub fn input_bindings(&self) -> &[PcuBindingRef] {
        self.schema.input_bindings()
    }
    #[must_use]
    pub const fn input_element_counts(&self) -> [usize; 2] {
        self.inputs
    }
    #[must_use]
    pub const fn input_byte_lengths(&self) -> [usize; 2] {
        let width = self.bytes / self.count;
        [self.inputs[0] * width, self.inputs[1] * width]
    }
    #[must_use]
    pub const fn operand_inputs(&self) -> [usize; 2] {
        self.schema.operand_inputs()
    }
    #[must_use]
    pub const fn operand_broadcast(&self) -> [bool; 2] {
        let indices = self.schema.operand_indices();
        [
            matches!(indices[0], PcuDispatchIndex::BindingElementZero),
            matches!(indices[1], PcuDispatchIndex::BindingElementZero),
        ]
    }
    #[must_use]
    pub const fn output_bindings(&self) -> [PcuBindingRef; 2] {
        self.schema.output_bindings()
    }
    #[must_use]
    pub const fn requirements(&self) -> PcuImplementationRequirements {
        self.requirements
    }
    /// Distinct implementation family for independently qualified actual-read division roles.
    #[must_use]
    pub const fn implementation_local_id(&self) -> u32 {
        let family = if matches!(
            self.requirements.numerical_options.reproducibility,
            PcuReproducibility::PortableV1
        ) {
            0x1600
        } else {
            0x600
        };
        family
            + match self.scalar {
                PcuScalarType::U8 => 0,
                PcuScalarType::I8 => 1,
                PcuScalarType::U16 => 2,
                PcuScalarType::I16 => 3,
                PcuScalarType::U32 => 4,
                PcuScalarType::I32 => 5,
                PcuScalarType::U64 => 6,
                PcuScalarType::I64 => 7,
                PcuScalarType::U128 => 8,
                PcuScalarType::I128 => 9,
                PcuScalarType::U256 => 10,
                PcuScalarType::I256 => 11,
                PcuScalarType::U512 => 12,
                PcuScalarType::I512 => 13,
                // Private construction is possible only after the central integer schema admits it.
                _ => return u32::MAX,
            }
    }
    pub(super) fn is_unread_declaration(&self, target: PcuBindingRef) -> bool {
        self.declarations[..self.declaration_count].contains(&target)
            && !self.input_bindings().contains(&target)
            && !self.output_bindings().contains(&target)
    }
}
