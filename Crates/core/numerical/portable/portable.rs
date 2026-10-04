//! Structural description of the first bounded `PortableV1` numerical map.
//!
//! Success establishes eligibility, never backend conformance or execution admission. This
//! profile specifies one destination-rounded checked operation over the four named low formats.
//! Integer synthesis avoids intermediate native floating rounding, contraction and FTZ. IEEE
//! nearest-even and after-rounding underflow references remain in the shared scalar contracts;
//! BF16/OFP8 retain their own encodings rather than being called IEEE basic formats.
//!
//! An admitted provider must independently prove exact encoded values, signed zeros, all selected
//! underflow dispositions and lowest fatal-fault identity. This initial profile excludes Clamp,
//! F32/F64/wide arithmetic, reductions, tensors, RNG and whole-model reproducibility. Its mere
//! presence never supplies a backend opt-in: independently unqualified profiles still reject.
//! The separately described checked-integer map has no floating rounding/tininess law.

#[path = "integer/integer.rs"]
mod integer;
#[path = "unary/unary.rs"]
mod unary;
#[rustfmt::skip]
pub use unary::{
    describe_portable_v1_unary_map,
    PcuPortableV1UnaryMapError,
};
#[rustfmt::skip]
pub use integer::{
    describe_portable_v1_checked_integer_composed_map,
    describe_portable_v1_integer_map,
    describe_portable_v1_integer_div_rem_map,
    PcuPortableV1IntegerDivRemMapDescription,
    PcuPortableV1IntegerMapDescription,
    PcuPortableV1IntegerMapError,
    PcuPortableV1IntegerComposedMapError,
};

#[rustfmt::skip]
use crate::{
    validate_checked_float_binary_kernel,
    validate_typed_dispatch_value_flow,
    CheckedFloatBinaryMapValidationError,
    PcuBindingRef,
    PcuDispatchControlOp,
    PcuDispatchDataOp,
    PcuDispatchFloatBinaryOp,
    PcuDispatchIndex,
    PcuDispatchKernelIr,
    PcuDispatchOp,
    PcuFloatUnderflowPolicy,
    PcuRangePolicy,
    PcuReproducibility,
    PcuScalarType,
    PcuTypedDispatchValidationError,
    PcuValueType,
    PcuValueTypeCaps,
};

/// An eligible map description; not a token of backend capability or numerical conformance.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PcuPortableV1MapDescription {
    pub scalar: PcuScalarType,
    pub operation: PcuDispatchFloatBinaryOp,
    pub underflow: PcuFloatUnderflowPolicy,
    pub submitted_invocations: u32,
    pub logical_extent: u32,
    /// Bindings in load order, which can differ from arithmetic operand order.
    pub load_bindings: [PcuBindingRef; 2],
    pub broadcast_loads: [bool; 2],
    /// For each arithmetic operand, its index in `load_bindings`; repeated operands are legal.
    pub operands: [u8; 2],
    pub output_binding: PcuBindingRef,
}

/// Unsupported profile or invalid structure, kept distinct from provider/device failures.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PcuPortableV1MapError {
    NotRequested,
    UnsupportedScalar,
    UnsupportedRange,
    UnderflowMismatch,
    InvalidLogicalShape([u32; 3]),
    UnsupportedStructure,
    InvalidValueFlow(PcuTypedDispatchValidationError),
    InvalidMap(CheckedFloatBinaryMapValidationError),
}

/// Describes the exact bounded map eligible for `PortableV1` numerical conformance.
///
/// Boundary/Strict both retain scalar checking; compound/precision permissions cannot weaken
/// these explicit operations. Backend admission, physical alias/layout/extent validation,
/// completion and publication are separate obligations. Warm executors retain the cold result.
///
/// # Errors
/// Rejects unsupported format/policy/geometry/effects or malformed typed SSA and binding schema.
pub fn describe_portable_v1_map(
    kernel: &PcuDispatchKernelIr<'_>,
) -> Result<PcuPortableV1MapDescription, PcuPortableV1MapError> {
    use PcuPortableV1MapError as Error;
    let requirements = kernel.numerical_requirements;
    if requirements.numerical_options.reproducibility != PcuReproducibility::PortableV1 {
        return Err(Error::NotRequested);
    }
    if requirements.range_policy != PcuRangePolicy::Reject {
        return Err(Error::UnsupportedRange);
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
    let description = extract_description(body, submitted_invocations, logical_extent)?;
    if description.underflow != requirements.float_underflow {
        return Err(Error::UnderflowMismatch);
    }
    validate_typed_dispatch_value_flow(kernel).map_err(Error::InvalidValueFlow)?;
    validate_checked_float_binary_kernel(
        kernel,
        PcuValueType::Scalar(description.scalar),
        description.operation,
        description.underflow,
        PcuValueTypeCaps::for_scalar(description.scalar),
    )
    .map_err(Error::InvalidMap)?;
    Ok(description)
}

fn extract_description(
    body: &[PcuDispatchOp<'_>],
    submitted_invocations: u32,
    logical_extent: u32,
) -> Result<PcuPortableV1MapDescription, PcuPortableV1MapError> {
    use PcuPortableV1MapError as Error;
    let [
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
            result: first,
            binding: first_binding,
            index: first_index,
        }),
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
            result: second,
            binding: second_binding,
            index: second_index,
        }),
        PcuDispatchOp::Data(PcuDispatchDataOp::CheckedFloatBinary {
            value_type,
            op,
            underflow_policy,
            range_policy,
            lhs,
            rhs,
            ..
        }),
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
            binding: output_binding,
            ..
        }),
    ] = body
    else {
        return Err(Error::UnsupportedStructure);
    };
    let PcuValueType::Scalar(
        scalar @ (PcuScalarType::F16
        | PcuScalarType::BF16
        | PcuScalarType::F8E4M3FN
        | PcuScalarType::F8E5M2),
    ) = value_type
    else {
        return Err(Error::UnsupportedScalar);
    };
    if *range_policy != PcuRangePolicy::Reject {
        return Err(Error::UnsupportedRange);
    }
    let operand = |value| {
        if value == first {
            Ok(0)
        } else if value == second {
            Ok(1)
        } else {
            Err(Error::UnsupportedStructure)
        }
    };
    Ok(PcuPortableV1MapDescription {
        scalar: *scalar,
        operation: *op,
        underflow: *underflow_policy,
        submitted_invocations,
        logical_extent,
        load_bindings: [*first_binding, *second_binding],
        broadcast_loads: [
            *first_index == PcuDispatchIndex::BindingElementZero,
            *second_index == PcuDispatchIndex::BindingElementZero,
        ],
        operands: [operand(lhs)?, operand(rhs)?],
        output_binding: *output_binding,
    })
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
