//! Cold eligibility for exact checked floating selection, separate from execution.
//!
//! Neg flips the sign of a finite value, including signed zero. `ReLU` selects +0
//! for negative values and either zero sign, otherwise retains the positive bits.
//! Nonfinite inputs remain fatal. Selection is exact: IEEE 754-2019 clause 7.5
//! tiny-and-inexact underflow cannot occur. PCU's tightened subnormal policy can
//! still reject an exact selected subnormal; explicit Clamp retains the specified
//! selection payload and reports its recovered range notice. Finite-input and
//! Result delivery are PCU contracts, not IEEE default exception handling.
//!
//! No native floating rounding, contraction, approximation or FTZ may change this
//! checked operation. BF16 and named OFP8 use their own encodings, not IEEE basic formats.
//! Providers independently prove values, fault arbitration, publication and
//! physical resource safety. This descriptor adds no implementation or fallback.

#[rustfmt::skip]
use crate::{
    validate_checked_float_map_kernel,
    CheckedFloatMapValidationError,
    PcuBindingAccess,
    PcuBindingRef,
    PcuDispatchControlOp,
    PcuDispatchDataOp,
    PcuDispatchFloatUnaryOp,
    PcuDispatchIndex,
    PcuDispatchKernelIr,
    PcuDispatchOp,
    PcuImplementationRequirements,
    PcuScalarType,
    PcuValueType,
    PcuValueTypeCaps,
};

/// One structurally eligible exact-selection map; never a conformance token.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PcuCheckedFloatUnaryMapDescription {
    pub scalar: PcuScalarType,
    pub operation: PcuDispatchFloatUnaryOp,
    /// Complete original tuple, including operation-irrelevant permissions.
    pub requirements: PcuImplementationRequirements,
    pub submitted_invocations: u32,
    pub logical_extent: u32,
    /// Actual read; unread declarations are not extra inputs or lifetime claims.
    pub input_binding: PcuBindingRef,
    pub broadcast_input: bool,
    pub output_binding: PcuBindingRef,
}

impl PcuCheckedFloatUnaryMapDescription {
    /// Minimum number of input elements; byte extent/alignment remain provider law.
    #[must_use]
    pub const fn input_extent(self) -> u32 {
        if self.broadcast_input {
            1
        } else {
            self.logical_extent
        }
    }
}

/// Structural eligibility refusal, distinct from runtime/backend failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PcuCheckedFloatUnaryMapError {
    UnsupportedScalar,
    InvalidLogicalShape([u32; 3]),
    UnsupportedStructure,
    RangeMismatch,
    UnderflowMismatch,
    InvalidAccess(PcuBindingRef),
    InvalidMap(CheckedFloatMapValidationError),
}

/// Describes one checked Neg/`ReLU` over F16/BF16/F32/F64 or named OFP8 formats.
///
/// Direct and canonical grid-stride maps, indexed/broadcast actual reads,
/// declaration permutation and unread declarations are eligible. The actual
/// input must be readonly and the output writable and distinct. No storage
/// alias/import permission is inferred. All three underflow policies and
/// Reject/observable Clamp keep their exact original request/header tuple.
/// The complete original numerical header is retained without rewriting its
/// reproducibility or other permissions. This detached matcher accepts ordinary
/// and requested Portable headers alike; it is not a reproducibility guarantee.
/// Boundary/Strict and native precision/compound permissions do not weaken this
/// explicitly checked, exact operation. Providers must separately admit it.
///
/// # Errors
/// Rejects unsupported types/effects/geometry,
/// header mismatches, malformed typed SSA, indices, or binding access.
pub fn describe_checked_float_unary_map(
    kernel: &PcuDispatchKernelIr<'_>,
) -> Result<PcuCheckedFloatUnaryMapDescription, PcuCheckedFloatUnaryMapError> {
    use PcuCheckedFloatUnaryMapError as Error;
    let requirements = kernel.numerical_requirements;
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
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
            result: loaded,
            binding: input,
            index,
        }),
        PcuDispatchOp::Data(PcuDispatchDataOp::CheckedFloatUnary {
            result,
            value,
            value_type,
            op,
            range_policy,
            underflow_policy,
        }),
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
            binding: output,
            value: stored,
            ..
        }),
    ] = body
    else {
        return Err(Error::UnsupportedStructure);
    };
    let PcuValueType::Scalar(
        scalar @ (PcuScalarType::F16
        | PcuScalarType::BF16
        | PcuScalarType::F32
        | PcuScalarType::F64
        | PcuScalarType::F8E4M3FN
        | PcuScalarType::F8E5M2),
    ) = value_type
    else {
        return Err(Error::UnsupportedScalar);
    };
    if *range_policy != requirements.range_policy {
        return Err(Error::RangeMismatch);
    }
    if *underflow_policy != requirements.float_underflow {
        return Err(Error::UnderflowMismatch);
    }
    if value != loaded || stored != result {
        return Err(Error::UnsupportedStructure);
    }
    validate_checked_float_map_kernel(kernel, *value_type, PcuValueTypeCaps::for_scalar(*scalar))
        .map_err(Error::InvalidMap)?;
    if !kernel.bindings.iter().any(|binding| {
        binding.reference() == *input && binding.access == PcuBindingAccess::ReadOnly
    }) {
        return Err(Error::InvalidAccess(*input));
    }
    Ok(PcuCheckedFloatUnaryMapDescription {
        scalar: *scalar,
        operation: *op,
        requirements,
        submitted_invocations,
        logical_extent,
        input_binding: *input,
        broadcast_input: *index == PcuDispatchIndex::BindingElementZero,
        output_binding: *output,
    })
}
