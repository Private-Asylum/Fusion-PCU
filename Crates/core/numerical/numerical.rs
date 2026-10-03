//! Independent arithmetic, checking, precision and reproducibility requirements.
//!
//! These are semantic requirements, not selections of a vendor library or compiler switch.
//! Backends admit an implementation for the complete combination during preparation. Enabling
//! one option must not silently enable or disable another option.

/// Determines where compound arithmetic classifies exceptions.
///
/// This does not select underflow handling, precision, contraction, or error disposition.
/// Explicit checked scalar operations retain their contracts in either mode. A backend must
/// prove the selected compound contract before admitting it; boundary mode is not unchecked.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum PcuNumericalMode {
    /// Classify exceptions at the declared compound primitive boundary.
    #[default]
    Boundary,
    /// Check each constituent operation at its declared precision and ordering.
    Strict,
}

/// Numerical exception behavior of a named compound primitive.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum PcuCompoundArithmeticPolicy {
    /// Report the primitive's declared numerical faults as execution errors.
    #[default]
    Checked,
    /// Explicitly accept the selected implementation's documented compound arithmetic.
    ///
    /// This permits its reduction/contraction and numerical exception behavior, including
    /// nonfinite outputs and underflow. It does not permit memory unsafety, unchecked dimensions,
    /// suppressed library/API errors, incomplete outputs or weaker resource lifetime guarantees.
    /// Precision changes need separate permission. Ordinary scalar instructions remain checked.
    BackendDefined,
}

/// Precision freedoms available to an implementation of a compound primitive.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum PcuPrecisionPolicy {
    /// Preserve the declared input, output and arithmetic precision.
    ///
    /// This does not prescribe a reduction order or promise cross-vendor bit identity. A library
    /// implementation must exclude unrequested reduced-precision/emulation configuration.
    #[default]
    Preserve,
    /// Permit documented backend-selected intermediate precision and approximations.
    ///
    /// Logical storage types remain unchanged. The selected implementation must expose its
    /// precision constraints; enabling this permission does not relax arithmetic fault handling
    /// or request determinism. More restrictive named format profiles can be added independently.
    BackendOptimized,
}

/// Required scope of numerical reproducibility, independently of checking granularity.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum PcuReproducibility {
    /// No numerical repeatability guarantee beyond the other selected contracts.
    #[default]
    Unspecified,
    /// Require the version-one portable numerical contract.
    ///
    /// This is an admission requirement, not a claim that every existing operation implements
    /// the profile. Unproved combinations must reject before device work. It neither enables
    /// strict checking nor changes numerical exception disposition.
    PortableV1,
}

/// Independent numerical requirements captured by an operation and its prepared cache identity.
///
/// [`PcuNumericalMode`] remains the separate checking-granularity axis. Default options retain
/// checked compound errors and declared precision without requiring portable reproduction.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct PcuNumericalOptions {
    pub compound_arithmetic: PcuCompoundArithmeticPolicy,
    pub precision: PcuPrecisionPolicy,
    pub reproducibility: PcuReproducibility,
}

/// Function-local overrides; an absent field inherits its caller's effective setting.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct PcuNumericalOverrides {
    pub compound_arithmetic: Option<PcuCompoundArithmeticPolicy>,
    pub precision: Option<PcuPrecisionPolicy>,
    pub reproducibility: Option<PcuReproducibility>,
}

impl PcuNumericalOptions {
    /// Applies only explicitly selected fields, preserving unrelated requirements.
    #[must_use]
    pub const fn with_overrides(self, overrides: PcuNumericalOverrides) -> Self {
        Self {
            compound_arithmetic: match overrides.compound_arithmetic {
                Some(policy) => policy,
                None => self.compound_arithmetic,
            },
            precision: match overrides.precision {
                Some(policy) => policy,
                None => self.precision,
            },
            reproducibility: match overrides.reproducibility {
                Some(policy) => policy,
                None => self.reproducibility,
            },
        }
    }
}

/// Which numerical requirement prevented an implementation from being admitted.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PcuNumericalRequirement {
    CompoundArithmetic,
    Precision,
    Reproducibility,
}

#[cfg(test)]
mod tests;

#[path = "portable/portable.rs"]
mod portable;
pub use portable::*;
