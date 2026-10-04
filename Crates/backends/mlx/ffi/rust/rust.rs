//! Direct upstream C tables, owned opaque handles, scoped errors and foreign copies.

use crate::MlxError;
const ERROR_BYTES: usize = 2048;
const SDK_VERSION: &str = "0.32.3";

#[path = "c_api/c_api.rs"]
mod c_api;
#[rustfmt::skip]
pub use c_api::{
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
};
#[cfg(feature = "tensor")]
pub use c_api::PreparedMatmul;
#[path = "typed/typed.rs"]
mod typed;
#[rustfmt::skip]
pub use typed::{
    as_f32,
    as_f32_mut,
};

fn dimensions(shape: [usize; 2]) -> Result<[i32; 2], MlxError> {
    let count = shape[0]
        .checked_mul(shape[1])
        .and_then(|count| count.checked_mul(size_of::<f32>()))
        .ok_or(MlxError::InvalidExtent)?;
    if count == 0 || isize::try_from(count).is_err() {
        return Err(MlxError::InvalidExtent);
    }
    Ok([
        i32::try_from(shape[0]).map_err(|_| MlxError::InvalidExtent)?,
        i32::try_from(shape[1]).map_err(|_| MlxError::InvalidExtent)?,
    ])
}
fn text_value(bytes: &[u8]) -> Result<String, MlxError> {
    let end = bytes
        .iter()
        .position(|&byte| byte == 0)
        .ok_or_else(|| MlxError::Abi("unterminated C ABI text".into()))?;
    String::from_utf8(bytes[..end].to_vec()).map_err(|_| MlxError::Abi("invalid C ABI UTF8".into()))
}
const fn require_apple_silicon() -> Result<(), MlxError> {
    if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
        Ok(())
    } else {
        Err(MlxError::UnsupportedPlatform)
    }
}

#[cfg(feature = "view-census")]
#[rustfmt::skip]
pub use c_api::{
    MlxViewCallCensus,
    view_call_census,
    reset_view_call_census,
};

#[cfg(feature = "division-census")]
#[rustfmt::skip]
pub use c_api::{
    MlxDivRemCallCensus,
    div_rem_call_census,
    reset_div_rem_call_census,
};

#[cfg(feature = "carrier-census")]
#[rustfmt::skip]
pub use c_api::{
    MlxCarrierCallCensus,
    carrier_call_census,
    reset_carrier_call_census,
};

#[cfg(feature = "binary-census")]
#[rustfmt::skip]
pub use c_api::{
    MlxBinaryCallCensus,
    binary_call_census,
    reset_binary_call_census,
};

#[cfg(feature = "integer-census")]
#[rustfmt::skip]
pub use c_api::{
    MlxIntegerCallCensus,
    integer_call_census,
    reset_integer_call_census,
};

#[cfg(feature = "tensor")]
pub use c_api::StrictMatMul;

#[cfg(feature = "tensor")]
pub use c_api::StrictSgd;

#[cfg(feature="tensor")]
pub use c_api::StrictMse;
