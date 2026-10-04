//! Bounded exact integer eligibility, independent of provider opt-in and conformance.
//!
//! For this profile Add/Sub/Mul use the mathematical integer result at the named width.
//! Reject reports an exceptional range and publishes no destination. Clamp completes with
//! signed MIN/MAX or unsigned zero/MAX and an observable recovered range fault. Lowest
//! logical invocation selection follows [`crate::PcuExecutionFault`], including grid-stride
//! scheduling: physical launch/workgroup order cannot determine the reported fault.
//!
//! No floating rounding, contraction, precision or tininess choice changes these explicit
//! integer instructions. Boundary/Strict and compound permissions remain independent.
//! A provider must prove exact values, fault selection, publication and physical resource
//! safety before opting in. This descriptor does not enable a provider, CPU fallback or
//! an arithmetic implementation. Wrapping, `DivRem`, reductions, tensors and RNG are separate.

#[path = "div_rem/div_rem.rs"]
mod div_rem;
#[path = "composed/composed.rs"]
mod composed;
#[rustfmt::skip]
pub use composed::{
    describe_portable_v1_checked_integer_composed_map,
    PcuPortableV1IntegerComposedMapError,
};
#[rustfmt::skip]
pub use div_rem::{
    describe_portable_v1_integer_div_rem_map,
    PcuPortableV1IntegerDivRemMapDescription,
};

#[rustfmt::skip]
use crate::{
    assess_checked_integer_binary_operands,
    CheckedIntegerBinaryOperandSchema,
    IntegerMapValidationError,
    PcuDispatchControlOp,
    PcuDispatchDataOp,
    PcuDispatchIntegerBinaryOp,
    PcuDispatchKernelIr,
    PcuDispatchOp,
    PcuRangePolicy,
    PcuReproducibility,
    PcuScalarType,
    PcuValueType,
    PcuValueTypeCaps,
};

/// Eligible detached integer map; backend/device execution remains separately admitted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PcuPortableV1IntegerMapDescription {
    pub scalar: PcuScalarType,
    pub operation: PcuDispatchIntegerBinaryOp,
    pub range_policy: PcuRangePolicy,
    pub submitted_invocations: u32,
    pub logical_extent: u32,
    /// Actual unique reads, mathematical operand slots/indices and writable destination.
    pub operands: CheckedIntegerBinaryOperandSchema,
}

/// Structural refusal, independent of a backend capability or device failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PcuPortableV1IntegerMapError {
    NotRequested,
    UnsupportedScalar,
    InvalidLogicalShape([u32; 3]),
    UnsupportedStructure,
    InvalidOperands(IntegerMapValidationError),
}

/// Describes one exact checked Add/Sub/Mul over the fourteen named integer widths.
///
/// Repeated reads, unused readonly declarations, reordered arguments and independent
/// invocation/index-zero loads use the shared operand schema. Range recovery remains
/// observable; a portable requirement never hides errors or implies Strict checking.
/// Floating underflow settings are irrelevant to this integer operation and retained in
/// the request envelope. Warm providers freeze the descriptor after independent admission.
///
/// # Errors
/// Rejects a missing portable request, other types/operations, malformed typed SSA,
/// binding roles, execution geometry, range metadata or extra effects.
pub fn describe_portable_v1_integer_map(
    kernel: &PcuDispatchKernelIr<'_>,
) -> Result<PcuPortableV1IntegerMapDescription, PcuPortableV1IntegerMapError> {
    use PcuPortableV1IntegerMapError as Error;
    if kernel
        .numerical_requirements
        .numerical_options
        .reproducibility
        != PcuReproducibility::PortableV1
    {
        return Err(Error::NotRequested);
    }
    let [submitted_invocations, 1, 1] = kernel.entry.logical_shape else {
        return Err(Error::InvalidLogicalShape(kernel.entry.logical_shape));
    };
    if submitted_invocations == 0 {
        return Err(Error::InvalidLogicalShape(kernel.entry.logical_shape));
    }
    let (body, logical_extent) = match kernel.ops {
        [
            PcuDispatchOp::GridStrideLoop { body, extent },
            PcuDispatchOp::Control(PcuDispatchControlOp::Return),
        ] if *extent != 0 => (*body, *extent),
        [
            body @ ..,
            PcuDispatchOp::Control(PcuDispatchControlOp::Return),
        ] => (body, submitted_invocations),
        _ => return Err(Error::UnsupportedStructure),
    };
    let [
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad { .. }),
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad { .. }),
        PcuDispatchOp::Data(PcuDispatchDataOp::CheckedIntegerBinary {
            value_type: PcuValueType::Scalar(scalar),
            op,
            ..
        }),
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore { .. }),
    ] = body
    else {
        return Err(Error::UnsupportedStructure);
    };
    if !matches!(
        scalar,
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
    ) {
        return Err(Error::UnsupportedScalar);
    }
    let operands = assess_checked_integer_binary_operands(
        kernel,
        PcuValueType::Scalar(*scalar),
        *op,
        PcuValueTypeCaps::for_scalar(*scalar),
    )
    .map_err(Error::InvalidOperands)?;
    Ok(PcuPortableV1IntegerMapDescription {
        scalar: *scalar,
        operation: *op,
        range_policy: kernel.numerical_requirements.range_policy,
        submitted_invocations,
        logical_extent,
        operands,
    })
}

#[cfg(test)]
#[path = "tests/tests.rs"]
mod tests;
