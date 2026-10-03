//! Routing for the audited direct-C FFI; every foreign concern remains contained here.

#[path = "abi/abi.rs"]
mod abi;
#[path = "admission/admission.rs"]
mod admission;
#[path = "api/api.rs"]
mod api;
#[path = "array/array.rs"]
mod array;
#[path = "binary/binary.rs"]
mod binary;
#[cfg(feature = "tensor")]
#[path = "callback/callback.rs"]
mod callback;
#[path = "calls/calls.rs"]
mod calls;
#[path = "carrier/carrier.rs"]
mod carrier;
#[path = "checked/checked.rs"]
mod checked;
#[path = "composed/composed.rs"]
mod composed;
#[path = "device/device.rs"]
mod device;
#[path = "div_rem/div_rem.rs"]
mod div_rem;
#[path = "encoded/encoded.rs"]
mod encoded;
#[path = "fault/fault.rs"]
mod fault;
#[path = "integer/integer.rs"]
mod integer;
#[path = "loading/loading.rs"]
mod loading;
#[path = "owner/owner.rs"]
mod owner;
#[cfg(feature = "tensor")]
#[path = "prepared/prepared.rs"]
mod prepared;
#[path = "session/session.rs"]
mod session;
#[path = "text/text.rs"]
mod text;
#[path = "transport/transport.rs"]
mod transport;
#[path = "view/view.rs"]
mod view;

#[rustfmt::skip]
pub use {
    api::Api,
    array::Array,
    session::Session,
    checked::CheckedUnary,
    binary::CheckedBinary,
    integer::CheckedInteger,
    div_rem::CheckedDivRem,
    encoded::EncodedArray,
    carrier::CarrierCopy,
    transport::Transport,
    composed::Composed,
};
#[cfg(feature = "tensor")]
pub use prepared::PreparedMatmul;

#[cfg(all(test, feature = "tensor"))]
#[path = "tests/tests.rs"]
mod tests;

#[cfg(feature = "view-census")]
#[rustfmt::skip]
pub use view::census::{
    MlxViewCallCensus,
    view_call_census,
    reset_view_call_census,
};

#[cfg(feature = "division-census")]
#[rustfmt::skip]
pub use div_rem::census::{
    MlxDivRemCallCensus,
    div_rem_call_census,
    reset_div_rem_call_census,
};

#[cfg(feature = "carrier-census")]
#[rustfmt::skip]
pub use carrier::census::{
    MlxCarrierCallCensus,
    carrier_call_census,
    reset_carrier_call_census,
};

#[cfg(feature = "binary-census")]
#[rustfmt::skip]
pub use binary::census::{
    MlxBinaryCallCensus,
    binary_call_census,
    reset_binary_call_census,
};

#[cfg(feature = "integer-census")]
#[rustfmt::skip]
pub use integer::census::{
    MlxIntegerCallCensus,
    integer_call_census,
    reset_integer_call_census,
};
