//! Semantic admission for decoded checked-scalar fault records.
//!
//! Providers own physical status encodings and terminal completion. This law is
//! retained during preparation and excludes fault classes or recovery dispositions
//! that the admitted operation cannot report. It does not prove the actual inputs
//! justify a fault, certify result bits, or extend scalar laws to compound kernels.

#[rustfmt::skip]
use crate::{
    PcuExecutionFault,
    PcuExecutionFaultKind,
    PcuFloatUnderflowPolicy,
    PcuRangePolicy,
    PcuScalarType,
    model::{
        PcuDispatchCheckedFloatConversion,
        PcuDispatchFloatBinaryOp,
        PcuDispatchFloatUnaryOp,
        PcuDispatchIntegerBinaryOp,
    },
};

const ZERO: u8 = 1;
const OVERFLOW: u8 = 2;
const UNDERFLOW: u8 = 4;
const SIGNED_DIVISION: u8 = 8;
const INVALID_FLOAT: u8 = 16;

/// Cold-retained admissible fault classes for one bounded checked scalar operation.
///
/// Fatal operand/domain faults never become Clamp notices. Range faults must use
/// the exact admitted Reject/Clamp disposition. Floating underflow permission
/// remains independent of range recovery. This is PCU error delivery over the
/// classifications described by IEEE Std 754-2019 clauses 7.4/7.5; it is not
/// IEEE default status handling or a complete IEEE exception mask.
///
/// Validate **every** nonzero physical record before selecting the first fatal
/// fault or recovered notice. A later malformed record must not be concealed by
/// an earlier valid fault. Family-wide status whitelists alone are insufficient.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PcuCheckedScalarFaultLaw {
    fatal: u8,
    recovered: u8,
}

impl PcuCheckedScalarFaultLaw {
    /// Combines constituent laws for a validated composed scalar map.
    ///
    /// A lane-only status record identifies no arithmetic step, so a candidate
    /// must be legal for at least one constituent. This preserves each fatal and
    /// recovered disposition; it does not turn every range fault into recovery.
    /// Validate structure, types, original policies and physical recovery encoding
    /// before folding cold. The union does not establish source attribution or
    /// permit continuation after a fatal predecessor. Step-addressed compounds
    /// retain their ordered domain instead of replacing it with this union.
    #[must_use]
    pub const fn union(self, other: Self) -> Self {
        Self {
            fatal: self.fatal | other.fatal,
            recovered: self.recovered | other.recovered,
        }
    }

    /// Law for the fourteen supported checked integer Add/Sub/Mul representations.
    /// Returns `None` for an unsupported scalar; it never invents execution support.
    #[must_use]
    pub const fn integer_binary(
        scalar: PcuScalarType,
        op: PcuDispatchIntegerBinaryOp,
        range: PcuRangePolicy,
    ) -> Option<Self> {
        let Some(signed) = integer_signedness(scalar) else {
            return None;
        };
        let kinds = if signed {
            OVERFLOW | UNDERFLOW
        } else {
            match op {
                PcuDispatchIntegerBinaryOp::Add | PcuDispatchIntegerBinaryOp::Mul => OVERFLOW,
                PcuDispatchIntegerBinaryOp::Sub => UNDERFLOW,
            }
        };
        Some(Self::with_range(0, kinds, range))
    }

    /// Reject-only quotient/remainder law, shared by joint and scalar references.
    /// Signed MIN/-1 is legal as a fault only for signed representations.
    /// This does not admit a single-output dispatch profile or clamped division.
    #[must_use]
    pub const fn integer_div_rem(scalar: PcuScalarType) -> Option<Self> {
        let Some(signed) = integer_signedness(scalar) else {
            return None;
        };
        Some(Self {
            fatal: if signed { ZERO | SIGNED_DIVISION } else { ZERO },
            recovered: 0,
        })
    }

    /// Law for the six supported finite-input checked floating binary formats.
    /// Add/Sub/Mul cannot report a zero-divisor fault. Permitting gradual underflow
    /// forbids an underflow error, including an otherwise recovered Clamp notice.
    /// Same-format Add/Sub cannot have a tiny inexact result: all finite inputs
    /// are integer multiples of the destination's smallest subnormal, and every
    /// tiny exact sum/difference is representable. The tightened subnormal policy
    /// can still reject it. Mul/Div can lose information below that quantum.
    #[must_use]
    pub const fn float_binary(
        scalar: PcuScalarType,
        op: PcuDispatchFloatBinaryOp,
        range: PcuRangePolicy,
        underflow: PcuFloatUnderflowPolicy,
    ) -> Option<Self> {
        if !checked_float(scalar) {
            return None;
        }
        let (domain, range_kinds) = match op {
            PcuDispatchFloatBinaryOp::Add | PcuDispatchFloatBinaryOp::Sub => {
                let kinds = match underflow {
                    PcuFloatUnderflowPolicy::RejectSubnormalResult => OVERFLOW | UNDERFLOW,
                    PcuFloatUnderflowPolicy::IeeeAfterRounding
                    | PcuFloatUnderflowPolicy::AllowGradualUnderflow => OVERFLOW,
                };
                (0, kinds)
            }
            PcuDispatchFloatBinaryOp::Mul => (0, float_range(underflow)),
            PcuDispatchFloatBinaryOp::Div => (ZERO, float_range(underflow)),
        };
        Some(Self::with_range(INVALID_FLOAT | domain, range_kinds, range))
    }

    /// Law for exact finite-input Neg/ReLU selection in the six checked formats.
    /// These operations do not overflow or round. A selected subnormal is an
    /// error only under PCU's explicitly tightened `RejectSubnormalResult` policy;
    /// exact tiny results do not signal IEEE clause 7.5 default underflow.
    #[must_use]
    pub const fn float_unary(
        scalar: PcuScalarType,
        op: PcuDispatchFloatUnaryOp,
        range: PcuRangePolicy,
        underflow: PcuFloatUnderflowPolicy,
    ) -> Option<Self> {
        if !checked_float(scalar) {
            return None;
        }
        let kinds = match (op, underflow) {
            (
                PcuDispatchFloatUnaryOp::Neg | PcuDispatchFloatUnaryOp::Relu,
                PcuFloatUnderflowPolicy::RejectSubnormalResult,
            ) => UNDERFLOW,
            (
                PcuDispatchFloatUnaryOp::Neg | PcuDispatchFloatUnaryOp::Relu,
                PcuFloatUnderflowPolicy::IeeeAfterRounding
                | PcuFloatUnderflowPolicy::AllowGradualUnderflow,
            ) => 0,
        };
        Some(Self::with_range(INVALID_FLOAT, kinds, range))
    }

    /// Law for finite-input checked ReLU-backward selection without arithmetic.
    /// Both inputs must be finite, including a masked-out upstream gradient.
    /// The selected value has the same exact-tiny rule as unary selection.
    #[must_use]
    pub const fn float_relu_backward(
        scalar: PcuScalarType,
        range: PcuRangePolicy,
        underflow: PcuFloatUnderflowPolicy,
    ) -> Option<Self> {
        Self::float_unary(scalar, PcuDispatchFloatUnaryOp::Relu, range, underflow)
    }

    /// Law for the declared F64-to-F32 narrowing and exact F32-to-F64 widening.
    /// Every finite F32 value widens to a normal F64 value or zero; widening has
    /// no range fault even under the tightened destination-subnormal policy.
    /// IEEE format conversion/rounding references: clauses 5.4.2 and 4.3.1.
    #[must_use]
    pub const fn float_conversion(
        conversion: PcuDispatchCheckedFloatConversion,
        range: PcuRangePolicy,
        underflow: PcuFloatUnderflowPolicy,
    ) -> Self {
        let kinds = match conversion {
            PcuDispatchCheckedFloatConversion::F64ToF32 => float_range(underflow),
            PcuDispatchCheckedFloatConversion::F32ToF64 => 0,
        };
        Self::with_range(INVALID_FLOAT, kinds, range)
    }

    /// Whether a decoded candidate has an admitted fault class and disposition.
    /// The provider must separately check raw encoding, logical extent, complete
    /// sibling status, actual completion and publication/resource obligations.
    #[must_use]
    pub const fn allows(self, kind: PcuExecutionFaultKind, recovered: bool) -> bool {
        let mask = if recovered {
            self.recovered
        } else {
            self.fatal
        };
        mask & kind_bit(kind) != 0
    }

    /// Adds the actual logical domain check to [`Self::allows`].
    /// Grid-stride maps supply visited extent, not launched invocation count.
    #[must_use]
    pub const fn accepts(self, fault: PcuExecutionFault, logical_extent: u64) -> bool {
        fault.is_within_logical_extent(logical_extent) && self.allows(fault.kind, fault.recovered)
    }

    const fn with_range(fatal: u8, range_kinds: u8, range: PcuRangePolicy) -> Self {
        match range {
            PcuRangePolicy::Reject => Self {
                fatal: fatal | range_kinds,
                recovered: 0,
            },
            PcuRangePolicy::Clamp => Self {
                fatal,
                recovered: range_kinds,
            },
        }
    }
}

const fn float_range(underflow: PcuFloatUnderflowPolicy) -> u8 {
    match underflow {
        PcuFloatUnderflowPolicy::IeeeAfterRounding
        | PcuFloatUnderflowPolicy::RejectSubnormalResult => OVERFLOW | UNDERFLOW,
        PcuFloatUnderflowPolicy::AllowGradualUnderflow => OVERFLOW,
    }
}

const fn kind_bit(kind: PcuExecutionFaultKind) -> u8 {
    match kind {
        PcuExecutionFaultKind::DivideByZero => ZERO,
        PcuExecutionFaultKind::ArithmeticOverflow => OVERFLOW,
        PcuExecutionFaultKind::ArithmeticUnderflow => UNDERFLOW,
        PcuExecutionFaultKind::SignedDivisionOverflow => SIGNED_DIVISION,
        PcuExecutionFaultKind::InvalidFloatingOperand => INVALID_FLOAT,
    }
}

const fn integer_signedness(scalar: PcuScalarType) -> Option<bool> {
    match scalar {
        PcuScalarType::I8
        | PcuScalarType::I16
        | PcuScalarType::I32
        | PcuScalarType::I64
        | PcuScalarType::I128
        | PcuScalarType::I256
        | PcuScalarType::I512 => Some(true),
        PcuScalarType::U8
        | PcuScalarType::U16
        | PcuScalarType::U32
        | PcuScalarType::U64
        | PcuScalarType::U128
        | PcuScalarType::U256
        | PcuScalarType::U512 => Some(false),
        _ => None,
    }
}

const fn checked_float(scalar: PcuScalarType) -> bool {
    matches!(
        scalar,
        PcuScalarType::F16
            | PcuScalarType::BF16
            | PcuScalarType::F32
            | PcuScalarType::F64
            | PcuScalarType::F8E4M3FN
            | PcuScalarType::F8E5M2
    )
}

#[cfg(test)]
#[path = "tests/law.rs"]
mod tests;
