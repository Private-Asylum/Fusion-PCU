//! Detached fourteen-width quotient/remainder schema; no native work or capability claim.
#[rustfmt::skip]
use fusion_pcu::{
    PcuBindingRef,
    PcuBindingType,
    PcuDispatchControlOp,
    PcuDispatchDataOp,
    PcuDispatchKernelIr,
    PcuDispatchOp,
    PcuImplementationRequirements,
    PcuRangePolicy,
    PcuReproducibility,
    PcuScalarType,
    PcuValueType,
    PcuValueTypeCaps,
    validate_integer_checked_div_rem_kernel,
};
use crate::MlxError;
/// Frozen real two-input/two-output schema, assessed before native activation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MlxCheckedDivRemPlan {
    scalar: PcuScalarType,
    count: usize,
    bytes: usize,
    inputs: [PcuBindingRef; 2],
    outputs: [PcuBindingRef; 2],
    operands: [usize; 2],
    requirements: PcuImplementationRequirements,
}
impl MlxCheckedDivRemPlan {
    /// Admits only integer fourteen-width direct/canonical grid quotient/remainder.
    ///
    /// # Errors
    /// Rejects unsupported type/schema/extent, Clamp, ineligible Portable or repeated mathematical operands.
    pub fn assess(kernel: &PcuDispatchKernelIr<'_>) -> Result<Self, MlxError> {
        let invalid = || MlxError::InvalidRequest("unsupported MLX checked DivRem profile".into());
        if cfg!(target_endian = "big")
            || kernel.entry.logical_shape[1..] != [1, 1]
            || kernel.numerical_requirements.range_policy != PcuRangePolicy::Reject
        {
            return Err(invalid());
        }
        validate_reproducibility(kernel)?;
        let Some(PcuBindingType::Value(PcuValueType::Scalar(scalar))) =
            kernel.bindings.first().map(|binding| binding.binding_type)
        else {
            return Err(invalid());
        };
        if !matches!(
            scalar,
            PcuScalarType::U8
                | PcuScalarType::I8
                | PcuScalarType::U16
                | PcuScalarType::I16
                | PcuScalarType::U32
                | PcuScalarType::I32
                | PcuScalarType::U64
                | PcuScalarType::I64
                | PcuScalarType::U128
                | PcuScalarType::I128
                | PcuScalarType::U256
                | PcuScalarType::I256
                | PcuScalarType::U512
                | PcuScalarType::I512
        ) {
            return Err(MlxError::UnsupportedScalar(scalar));
        }
        validate_integer_checked_div_rem_kernel(
            kernel,
            PcuValueType::Scalar(scalar),
            PcuValueTypeCaps::for_scalar(scalar),
        )
        .map_err(|_| invalid())?;
        let (body, count) = match kernel.ops {
            [
                PcuDispatchOp::GridStrideLoop { body, extent },
                PcuDispatchOp::Control(PcuDispatchControlOp::Return),
            ] => (*body, *extent),
            [
                body @ ..,
                PcuDispatchOp::Control(PcuDispatchControlOp::Return),
            ] => (body, kernel.entry.logical_shape[0]),
            _ => return Err(invalid()),
        };
        let [
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result: first,
                binding: first_binding,
                ..
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result: second,
                binding: second_binding,
                ..
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::CheckedDivRem { lhs, rhs, .. }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                binding: quotient, ..
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                binding: remainder, ..
            }),
        ] = body
        else {
            return Err(invalid());
        };
        // This initial public profile does not stage an unused second binding for self-division.
        // A future central role schema can admit that profile without a fabricated resource.
        if lhs == rhs {
            return Err(invalid());
        }
        let slot = |value| {
            if value == first {
                Ok(0)
            } else if value == second {
                Ok(1)
            } else {
                Err(invalid())
            }
        };
        let count = usize::try_from(count).map_err(|_| MlxError::InvalidExtent)?;
        let bytes = checked_bytes(scalar, count)?;
        Ok(Self {
            scalar,
            count,
            bytes,
            inputs: [*first_binding, *second_binding],
            outputs: [*quotient, *remainder],
            operands: [slot(lhs)?, slot(rhs)?],
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
    pub const fn input_bindings(&self) -> &[PcuBindingRef; 2] {
        &self.inputs
    }
    #[must_use]
    pub const fn output_bindings(&self) -> &[PcuBindingRef; 2] {
        &self.outputs
    }
    #[must_use]
    pub const fn operand_inputs(&self) -> [usize; 2] {
        self.operands
    }
    #[must_use]
    pub const fn requirements(&self) -> PcuImplementationRequirements {
        self.requirements
    }
    /// Stable identity for this independently qualified integer fourteen-width profile.
    #[must_use]
    pub const fn implementation_local_id(&self) -> u32 {
        let family = if matches!(
            self.requirements.numerical_options.reproducibility,
            PcuReproducibility::PortableV1
        ) {
            0x1500
        } else {
            0x500
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
                _ => 14, // Unreachable from the private fields and exact cold assessor.
            }
    }
}

pub(super) fn checked_bytes(scalar: PcuScalarType, count: usize) -> Result<usize, MlxError> {
    let width = usize::from(scalar.bit_width()) / 8;
    count
        .checked_mul(width.div_ceil(4))
        .filter(|&n| n != 0 && i32::try_from(n).is_ok())
        .ok_or(MlxError::InvalidExtent)?;
    count
        .checked_mul(4)
        .filter(|&n| isize::try_from(n).is_ok())
        .ok_or(MlxError::InvalidExtent)?;
    count
        .checked_mul(width)
        .filter(|&n| isize::try_from(n).is_ok())
        .ok_or(MlxError::InvalidExtent)
}

/// Exact descriptor admission is separate from the scalar arithmetic permissions.
pub(super) fn validate_reproducibility(kernel: &PcuDispatchKernelIr<'_>) -> Result<(), MlxError> {
    if kernel
        .numerical_requirements
        .numerical_options
        .reproducibility
        == PcuReproducibility::PortableV1
    {
        fusion_pcu::describe_portable_v1_integer_div_rem_map(kernel).map_err(|_| {
            MlxError::InvalidRequest("unsupported MLX Portable joint integer division".into())
        })?;
    }
    Ok(())
}
