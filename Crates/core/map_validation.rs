//! Structural admission profiles for bounded scalar maps.
//!
//! The public per-profile modules remain available at the crate root for compatibility, while
//! their implementation files live together under `map_validation/`.

mod checked_float_binary_validation;
mod checked_float_conversion_map_validation;
mod checked_float_map_validation;
#[path = "map_validation/checked_integer_map_validation/checked_integer_map_validation.rs"]
mod checked_integer_map_validation;
#[path = "map_validation/checked_scalar_resources/checked_scalar_resources.rs"]
mod checked_scalar_resources;
pub mod f16_bf16_identity_validation;
pub mod f32_map_validation;
pub mod f64_map_validation;
pub mod i16_map_validation;
pub mod i32_map_validation;
pub mod i64_map_validation;
pub mod i8_map_validation;
mod integer_map_validation;
#[rustfmt::skip]
pub use integer_map_validation::{
    assess_checked_integer_binary_operands,
    assess_checked_integer_div_rem_operands,
    validate_integer_checked_binary_kernel,
    validate_integer_checked_div_rem_kernel,
    CheckedIntegerBinaryOperandSchema,
    CheckedIntegerDivRemOperandSchema,
    IntegerMapValidationError,
};
pub mod scalar_identity_validation;
#[path = "map_validation/scalar_transport/scalar_transport.rs"]
mod scalar_transport;
pub mod typed_dispatch;
pub mod u16_map_validation;
pub mod u32_identity_validation;
pub mod u32_map_validation;
pub mod u64_identity_validation;
pub mod u64_map_validation;
pub mod u8_map_validation;

pub use f32_map_validation::*;
pub use checked_float_binary_validation::*;
pub use checked_float_map_validation::*;
pub use checked_integer_map_validation::*;
pub use checked_scalar_resources::*;
pub use checked_float_conversion_map_validation::*;
pub use f64_map_validation::*;
pub use f16_bf16_identity_validation::*;
pub use i16_map_validation::*;
pub use i32_map_validation::*;
pub use i64_map_validation::*;
pub use i8_map_validation::*;
pub use scalar_identity_validation::*;
pub use scalar_transport::*;
pub use u16_map_validation::*;
pub use u32_identity_validation::*;
pub use u32_map_validation::*;
pub use u64_identity_validation::*;
pub use u64_map_validation::*;
pub use u8_map_validation::*;
pub use typed_dispatch::*;
