//! Typed cold function table and retained loaded-image lease.

#[rustfmt::skip]
use std::ffi::{
    c_char,
    c_void,
};
use libloading::Library;
#[rustfmt::skip]
use super::abi::{
    CArray,
    CCheckedUnary,
    CCarrier,
    CarrierNew,
    CarrierPrefixNew,
    CarrierApply,
    CCheckedBinary,
    BinaryNew,
    BinaryApply,
    CheckedNew,
    CheckedPrefixNew,
    CheckedApply,
    CheckedUpload,
    CDevice,
    CDeviceInfo,
    CStream,
    Free,
    ScopeBegin,
    ScopeEnd,
};
#[cfg(feature = "tensor")]
#[rustfmt::skip]
use super::abi::{
    Callback,
    CClosure,
    CVector,
    PayloadDrop,
    ReplayApply,
    ReplayFree,
    ReplayNew,
};

#[rustfmt::skip]
use super::abi::{
    CCheckedInteger,
    IntegerNew,
    IntegerApply,
};
#[rustfmt::skip]
use super::abi::{
    CCheckedDivRem,
    DivRemNew,
    DivRemApply,
};
#[rustfmt::skip]
use super::abi::{
    CTransport,
    TransportNew,
    TransportApply,
    CComposed,
    ComposedNew,
    ComposedApply,
};
// Every operation is resolved once during load. Warm dispatch is through these typed C
// function pointers; the owning Library remains retained through every owner and callback.
pub struct Api {
    pub(super) composed_new: ComposedNew,
    pub(super) composed_apply: ComposedApply,
    pub(super) composed_free: Free<CComposed>,
    pub(super) transport_new: TransportNew,
    pub(super) transport_apply: TransportApply,
    pub(super) transport_free: Free<CTransport>,
    #[cfg_attr(
        not(all(test, feature = "tensor")),
        allow(
            dead_code,
            reason = "Retain the loaded native image until every opaque owner and callback is released."
        )
    )]
    pub(super) library: Library,
    pub(super) version: String,
    pub(super) begin: ScopeBegin,
    pub(super) end: ScopeEnd,
    pub(super) device_count: unsafe extern "C" fn(*mut i32, i32) -> i32,
    pub(super) device_new: unsafe extern "C" fn(i32, i32) -> CDevice,
    pub(super) device_available: unsafe extern "C" fn(*mut bool, CDevice) -> i32,
    pub(super) device_free: Free<CDevice>,
    pub(super) info_get: unsafe extern "C" fn(*mut CDeviceInfo, CDevice) -> i32,
    pub(super) info_has: unsafe extern "C" fn(*mut bool, CDeviceInfo, *const c_char) -> i32,
    pub(super) info_string:
        unsafe extern "C" fn(*mut *const c_char, CDeviceInfo, *const c_char) -> i32,
    pub(super) info_free: Free<CDeviceInfo>,
    pub(super) metal_available: unsafe extern "C" fn(*mut bool) -> i32,
    pub(super) stream_new: unsafe extern "C" fn(CDevice) -> CStream,
    pub(super) stream_free: Free<CStream>,
    pub(super) synchronize: unsafe extern "C" fn(CStream) -> i32,
    pub(super) array_set_data:
        unsafe extern "C" fn(*mut CArray, *const c_void, *const i32, i32, i32) -> i32,
    pub(super) array_set: unsafe extern "C" fn(*mut CArray, CArray) -> i32,
    pub(super) array_free: Free<CArray>,
    pub(super) array_dtype: unsafe extern "C" fn(CArray) -> i32,
    pub(super) array_ndim: unsafe extern "C" fn(CArray) -> usize,
    pub(super) array_dim: unsafe extern "C" fn(CArray, i32) -> i32,
    pub(super) array_size: unsafe extern "C" fn(CArray) -> usize,
    pub(super) array_nbytes: unsafe extern "C" fn(CArray) -> usize,
    pub(super) array_strides: unsafe extern "C" fn(CArray) -> *const usize,
    pub(super) array_data: unsafe extern "C" fn(CArray) -> *const f32,
    pub(super) array_data_u8: unsafe extern "C" fn(CArray) -> *const u8,
    pub(super) array_data_u16: unsafe extern "C" fn(CArray) -> *const u16,
    pub(super) array_data_u32: unsafe extern "C" fn(CArray) -> *const u32,
    pub(super) checked_new: CheckedNew,
    pub(super) checked_prefix_new: CheckedPrefixNew,
    pub(super) checked_apply: CheckedApply,
    pub(super) checked_upload: CheckedUpload,
    pub(super) checked_free: Free<CCheckedUnary>,
    pub(super) binary_new: BinaryNew,
    pub(super) binary_prefix_new: BinaryNew,
    pub(super) binary_apply: BinaryApply,
    pub(super) binary_free: Free<CCheckedBinary>,
    pub(super) integer_new: IntegerNew,
    pub(super) integer_prefix_new: IntegerNew,
    pub(super) integer_apply: IntegerApply,
    pub(super) integer_free: Free<CCheckedInteger>,
    pub(super) div_rem_new: DivRemNew,
    pub(super) div_rem_prefix_new: DivRemNew,
    pub(super) div_rem_apply: DivRemApply,
    pub(super) div_rem_free: Free<CCheckedDivRem>,
    pub(super) f32_view_validate_shared: unsafe extern "C" fn(CArray, CArray) -> i32,
    pub(super) f32_view: unsafe extern "C" fn(*mut CArray, CArray, CStream, i32, i32, i32) -> i32,
    pub(super) encoded_prefix: unsafe extern "C" fn(*mut CArray, CArray, CArray, CStream) -> i32,
    pub(super) carrier_new: CarrierNew,
    pub(super) carrier_prefix_new: CarrierPrefixNew,
    pub(super) carrier_apply: CarrierApply,
    pub(super) carrier_free: Free<CCarrier>,
    pub(super) array_eval: unsafe extern "C" fn(CArray) -> i32,
    pub(super) array_wait: unsafe extern "C" fn(CArray) -> i32,
    pub(super) array_available: unsafe extern "C" fn(*mut bool, CArray) -> i32,
    pub(super) array_contiguous: unsafe extern "C" fn(*mut bool, CArray) -> i32,
    #[cfg(feature = "tensor")]
    pub(super) matmul: unsafe extern "C" fn(*mut CArray, CArray, CArray, CStream) -> i32,
    #[cfg(feature = "tensor")]
    pub(super) vector_set_data: unsafe extern "C" fn(*mut CVector, *const CArray, usize) -> i32,
    #[cfg(feature = "tensor")]
    pub(super) vector_get: unsafe extern "C" fn(*mut CArray, CVector, usize) -> i32,
    #[cfg(feature = "tensor")]
    pub(super) vector_set_value: unsafe extern "C" fn(*mut CVector, CArray) -> i32,
    #[cfg(feature = "tensor")]
    pub(super) vector_size: unsafe extern "C" fn(CVector) -> usize,
    #[cfg(feature = "tensor")]
    pub(super) vector_free: Free<CVector>,
    #[cfg(feature = "tensor")]
    pub(super) closure_new: unsafe extern "C" fn(Callback, *mut c_void, PayloadDrop) -> CClosure,
    #[cfg(feature = "tensor")]
    pub(super) closure_apply: unsafe extern "C" fn(*mut CVector, CClosure, CVector) -> i32,
    #[cfg(feature = "tensor")]
    pub(super) closure_free: Free<CClosure>,
    #[cfg(feature = "tensor")]
    pub(super) compile: unsafe extern "C" fn(*mut CClosure, CClosure, bool) -> i32,
    #[cfg(feature = "tensor")]
    pub(super) replay_new: ReplayNew,
    #[cfg(feature = "tensor")]
    pub(super) replay_apply: ReplayApply,
    #[cfg(feature = "tensor")]
    pub(super) replay_free: ReplayFree,
}

impl Api {
    pub fn version(&self) -> &str {
        &self.version
    }
}
