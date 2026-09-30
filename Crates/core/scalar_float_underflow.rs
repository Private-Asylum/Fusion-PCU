//! Floating-point arithmetic underflow policy vocabulary.
//!
//! The classification follows IEEE Std 754-2019, clause 7.5 (Underflow), choosing
//! tininess after rounding consistently for binary operations and conversions. Under IEEE
//! default handling, an exact tiny result raises no underflow flag; a tiny inexact result
//! raises underflow and inexact. PCU maps that classification to a `Result` error by default,
//! rather than IEEE's default delivery of a rounded value with status flags.
//! See the [IEEE working-group exception notes](https://grouper.ieee.org/groups/msc/ANSI_IEEE-Std-754-2019/background/exceptions.txt).
//!
//! These policies describe decisions a scalar execution route may apply when it has a
//! trustworthy classification of an arithmetic result. The classification distinguishes
//! destination-precision rounding with an unbounded exponent range from the final exponent-
//! limited result. They do not detect underflow or
//! enforce behavior for existing host or device arithmetic routes. Classification must be
//! supplied by the operation implementation; the rounded `f32` or `f64` value alone cannot
//! establish whether a result was exact.
//!
//! Flush-to-zero (FTZ) is a separate execution capability. None of these policies enables
//! FTZ or implies that a backend supports it. Transport and stored bit patterns are unaffected.

/// Facts about an arithmetic result needed to apply a floating underflow policy.
///
/// Callers provide these independently from the final floating-point value. In particular,
/// `is_inexact` must reflect the exact operation and rounding, rather than an inference from
/// whether the rounded result is zero or subnormal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PcuFloatUnderflowClassification {
    /// Whether the nonzero result rounded to destination precision with an unbounded
    /// exponent range is smaller in magnitude than the smallest normal value. This is
    /// IEEE Std 754-2019, clause 7.5(a), tininess-after-rounding; not a test of stored bits.
    /// It remains true when destination exponent limits subsequently round the result to zero.
    pub is_tiny_after_rounding: bool,
    /// Whether rounding discarded nonzero information from the exact result.
    pub is_inexact: bool,
    /// Whether the rounded arithmetic result is a nonzero subnormal value.
    pub result_is_subnormal: bool,
}

/// Policy for deciding whether classified floating-point arithmetic underflow is an error.
///
/// The default is [`Self::IeeeAfterRounding`]: underflow is an error only when the result is
/// tiny after rounding and inexact. Therefore an exact subnormal result succeeds. The stricter
/// [`Self::RejectSubnormalResult`] also rejects exact subnormals while retaining the default's
/// rejection of tiny inexact results rounded to zero. This is a semantic contract only; current
/// unchecked host and GPU arithmetic is not made checked by selecting this policy.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PcuFloatUnderflowPolicy {
    /// Use IEEE Std 754-2019, clause 7.5(a), classification with PCU error handling.
    #[default]
    IeeeAfterRounding,
    /// Reject every nonzero subnormal result and every tiny inexact result rounded to zero.
    /// Exact subnormal results are rejected as well; this is a stricter PCU policy.
    RejectSubnormalResult,
    /// Preserve gradual underflow, accepting rounded subnormal and tiny inexact results.
    AllowGradualUnderflow,
}

impl PcuFloatUnderflowPolicy {
    /// Returns whether the supplied arithmetic classification violates this policy.
    #[must_use]
    pub const fn rejects(self, classification: PcuFloatUnderflowClassification) -> bool {
        match self {
            Self::IeeeAfterRounding => {
                classification.is_tiny_after_rounding && classification.is_inexact
            }
            Self::RejectSubnormalResult => {
                classification.result_is_subnormal
                    || (classification.is_tiny_after_rounding && classification.is_inexact)
            }
            Self::AllowGradualUnderflow => false,
        }
    }
}

#[cfg(test)]
mod tests {
    #[rustfmt::skip]
    use super::{
        PcuFloatUnderflowClassification,
        PcuFloatUnderflowPolicy,
    };

    const EXACT_SUBNORMAL: PcuFloatUnderflowClassification = PcuFloatUnderflowClassification {
        is_tiny_after_rounding: true,
        is_inexact: false,
        result_is_subnormal: true,
    };

    const TINY_INEXACT_ROUNDED_ZERO: PcuFloatUnderflowClassification =
        PcuFloatUnderflowClassification {
            is_tiny_after_rounding: true,
            is_inexact: true,
            result_is_subnormal: false,
        };

    const TINY_INEXACT_ROUNDED_SUBNORMAL: PcuFloatUnderflowClassification =
        PcuFloatUnderflowClassification {
            is_tiny_after_rounding: true,
            is_inexact: true,
            result_is_subnormal: true,
        };

    const EXACT_ZERO: PcuFloatUnderflowClassification = PcuFloatUnderflowClassification {
        is_tiny_after_rounding: false,
        is_inexact: false,
        result_is_subnormal: false,
    };

    #[test]
    fn default_policy_is_ieee_tininess_after_rounding() {
        assert_eq!(
            PcuFloatUnderflowPolicy::default(),
            PcuFloatUnderflowPolicy::IeeeAfterRounding
        );
        assert!(!PcuFloatUnderflowPolicy::default().rejects(EXACT_SUBNORMAL));
        assert!(PcuFloatUnderflowPolicy::default().rejects(TINY_INEXACT_ROUNDED_ZERO));
        assert!(PcuFloatUnderflowPolicy::default().rejects(TINY_INEXACT_ROUNDED_SUBNORMAL));
    }

    #[test]
    fn tighter_policy_rejects_exact_and_inexact_subnormal_results() {
        let policy = PcuFloatUnderflowPolicy::RejectSubnormalResult;
        assert!(policy.rejects(EXACT_SUBNORMAL));
        assert!(policy.rejects(TINY_INEXACT_ROUNDED_SUBNORMAL));
        assert!(policy.rejects(TINY_INEXACT_ROUNDED_ZERO));
        assert!(!policy.rejects(EXACT_ZERO));
    }

    #[test]
    fn gradual_policy_accepts_tiny_rounded_values() {
        let policy = PcuFloatUnderflowPolicy::AllowGradualUnderflow;
        assert!(!policy.rejects(EXACT_SUBNORMAL));
        assert!(!policy.rejects(TINY_INEXACT_ROUNDED_SUBNORMAL));
        assert!(!policy.rejects(TINY_INEXACT_ROUNDED_ZERO));
    }
}
