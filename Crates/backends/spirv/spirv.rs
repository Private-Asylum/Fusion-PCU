//! SPIR-V lowering backend for PCU IR.
//!
//! This module is a compiler target, not a device runner. Vulkan, `OpenCL`, or any other runtime
//! may consume the generated words elsewhere; this backend only lowers PCU dispatch IR into
//! backend-neutral SPIR-V module bytes.

#![no_std]

extern crate fusion_pcu_core as fusion_pcu;
extern crate alloc;

#[path = "ordered_transport/ordered_transport.rs"]
mod ordered_transport;
#[rustfmt::skip]
pub use ordered_transport::{
    lower_ordered_scalar_transport_to_spirv,
    validate_ordered_scalar_transport_map,
    PcuSpirvOrderedTransportProfile,
};

#[path = "composed/composed.rs"]
mod composed;
#[rustfmt::skip]
pub use composed::{
    lower_composed_float_to_spirv,
    lower_one_effect_float_to_spirv,
    validate_composed_float_map,
    validate_one_effect_float_map,
    PcuSpirvComposedFloatProfile,
    PcuSpirvComposedResource,
};

#[cfg(test)]
extern crate std;

#[path = "bit_map/bit_map.rs"]
mod bit_map;
#[path = "checked_binary/checked_binary.rs"]
mod checked_binary;
#[path = "checked_unary/checked_unary.rs"]
mod checked_unary;
#[path = "error/error.rs"]
pub mod error;
#[path = "lower/lower.rs"]
pub mod lower;
#[path = "module/module.rs"]
pub mod module;
#[path = "sink/sink.rs"]
pub mod sink;
#[path = "types/types.rs"]
pub mod types;

pub use error::*;
pub use lower::*;
pub use module::*;
pub use sink::*;
pub use types::*;
pub use bit_map::*;
pub use checked_binary::*;
pub use checked_unary::*;

#[path = "scalar_transport/scalar_transport.rs"]
mod scalar_transport;
pub use scalar_transport::{
    PcuSpirvScalarTransportProfile, validate_scalar_transport_map, lower_scalar_transport_to_spirv,
};

#[path = "checked_integer/checked_integer.rs"]
mod checked_integer;
pub use checked_integer::{
    PcuSpirvCheckedIntegerProfile, validate_checked_integer_map, lower_checked_integer_to_spirv,
};

#[path = "checked_div_rem/checked_div_rem.rs"]
mod checked_div_rem;
#[rustfmt::skip]
pub use checked_div_rem::{PcuSpirvCheckedDivRemProfile,validate_checked_div_rem_map,lower_checked_div_rem_to_spirv};

#[path = "checked_conversion/checked_conversion.rs"]
mod checked_conversion;
pub use checked_conversion::{
    lower_checked_float_conversion_to_spirv, validate_checked_float_conversion_map,
    PcuSpirvCheckedConversionProfile,
};

#[path = "checked_backward/checked_backward.rs"]
mod checked_backward;
pub use checked_backward::lower_checked_relu_backward_to_spirv;
#[path = "checked_compound/checked_compound.rs"]
mod checked_compound;
#[rustfmt::skip]
pub use checked_compound::{
    lower_checked_compound_to_spirv,
    PcuSpirvCompoundOperation,
    PcuSpirvCompoundProfile,
};
