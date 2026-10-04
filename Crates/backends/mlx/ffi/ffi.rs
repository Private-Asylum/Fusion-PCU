//! Entry to the isolated Rust foreign-runtime boundary.

#[path = "rust/rust.rs"]
mod rust;

#[rustfmt::skip]
pub use rust::{
    Api,
    Array,
    Session,
    CheckedUnary,
    CheckedBinary,
    CheckedInteger,
    CheckedDivRem,
    EncodedArray,
    CarrierCopy,
    Transport,
    Composed,
    Conversion,
    ReluBackward,
    as_f32,
    as_f32_mut,
};
#[cfg(feature = "tensor")]
pub use rust::PreparedMatmul;

#[cfg(feature = "view-census")]
#[rustfmt::skip]
pub use rust::{
    MlxViewCallCensus,
    view_call_census,
    reset_view_call_census,
};

#[cfg(feature = "division-census")]
#[rustfmt::skip]
pub use rust::{
    MlxDivRemCallCensus,
    div_rem_call_census,
    reset_div_rem_call_census,
};

#[cfg(feature = "carrier-census")]
#[rustfmt::skip]
pub use rust::{
    MlxCarrierCallCensus,
    carrier_call_census,
    reset_carrier_call_census,
};

#[cfg(feature = "binary-census")]
#[rustfmt::skip]
pub use rust::{
    MlxBinaryCallCensus,
    binary_call_census,
    reset_binary_call_census,
};

#[cfg(feature = "integer-census")]
#[rustfmt::skip]
pub use rust::{
    MlxIntegerCallCensus,
    integer_call_census,
    reset_integer_call_census,
};

#[cfg(feature = "tensor")]
pub use rust::StrictMatMul;

#[cfg(feature = "tensor")]
pub use rust::StrictSgd;

#[cfg(feature="tensor")]
pub use rust::StrictMse;
