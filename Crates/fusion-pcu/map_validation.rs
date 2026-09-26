//! Structural admission profiles for bounded scalar maps.
//!
//! The public per-profile modules remain available at the crate root for compatibility, while
//! their implementation files live together under `map_validation/`.

pub mod f32_map_validation;
pub mod f64_map_validation;
pub mod i16_map_validation;
pub mod i32_map_validation;
pub mod i64_map_validation;
pub mod i8_map_validation;
mod integer_map_validation;
pub mod u16_map_validation;
pub mod u32_identity_validation;
pub mod u32_map_validation;
pub mod u64_identity_validation;
pub mod u64_map_validation;
pub mod u8_map_validation;

pub use f32_map_validation::*;
pub use f64_map_validation::*;
pub use i16_map_validation::*;
pub use i32_map_validation::*;
pub use i64_map_validation::*;
pub use i8_map_validation::*;
pub use u16_map_validation::*;
pub use u32_identity_validation::*;
pub use u32_map_validation::*;
pub use u64_identity_validation::*;
pub use u64_map_validation::*;
pub use u8_map_validation::*;
