//! Representation-defined constants shared by tensor capture and backend preparation.
//!
//! IEEE Std 754-2019 clauses 4.3.1 and 5.4.2 specify nearest, ties-to-even
//! integer-to-floating conversion. These element-count conversions implement
//! that rounding with integer operations, without consulting or changing the
//! host thread's floating environment. They do not imply device arithmetic support.

#[rustfmt::skip]
use crate::{
    u64_to_f32_nearest_even,
    u64_to_f64_nearest_even,
};

/// Rounds a tensor element count to binary32, nearest with ties to even.
///
/// Zero maps to positive zero. Every supported `usize` count has a finite result.
#[must_use]
pub const fn count_f32(count: usize) -> f32 {
    u64_to_f32_nearest_even(count as u64)
}

/// Rounds a tensor element count to binary64, nearest with ties to even.
///
/// Zero maps to positive zero. Every supported `usize` count has a finite result.
#[must_use]
pub const fn count_f64(count: usize) -> f64 {
    u64_to_f64_nearest_even(count as u64)
}

#[cfg(test)]
mod tests;
