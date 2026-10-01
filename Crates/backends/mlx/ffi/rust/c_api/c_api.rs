//! Routing for the audited direct-C FFI; every foreign concern remains contained here.

#[path = "abi/abi.rs"]
mod abi;
#[path = "admission/admission.rs"]
mod admission;
#[path = "api/api.rs"]
mod api;
#[path = "array/array.rs"]
mod array;
#[cfg(feature = "tensor")]
#[path = "callback/callback.rs"]
mod callback;
#[path = "calls/calls.rs"]
mod calls;
#[path = "device/device.rs"]
mod device;
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

#[rustfmt::skip]
pub use {
    api::Api,
    array::Array,
    session::Session,
};
#[cfg(feature = "tensor")]
pub use prepared::PreparedMatmul;

#[cfg(all(test, feature = "tensor"))]
#[path = "tests/tests.rs"]
mod tests;
