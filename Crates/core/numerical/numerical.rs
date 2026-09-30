//! Independent compound-arithmetic exception detection granularity.

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
