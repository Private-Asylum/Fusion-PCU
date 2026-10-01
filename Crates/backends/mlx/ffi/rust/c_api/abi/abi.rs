//! Exact selected C handle, manifest, scope and callback representations.

#[rustfmt::skip]
use std::ffi::{
    c_char,
    c_void,
};

macro_rules! handle {
    ($($name:ident),+ $(,)?) => {$(
        #[repr(C)]
        #[derive(Clone, Copy)]
        pub(super) struct $name { ctx: *mut c_void }
        impl Opaque for $name {
            fn empty() -> Self { Self { ctx: std::ptr::null_mut() } }
            fn is_empty(self) -> bool { self.ctx.is_null() }
        }
    )+};
}
pub(super) trait Opaque: Copy {
    fn empty() -> Self;
    fn is_empty(self) -> bool;
}
handle!(CArray, CDevice, CStream, CString, CDeviceInfo);
#[cfg(feature = "tensor")]
handle!(CClosure, CVector);

#[repr(C)]
pub(super) struct ErrorScope {
    pub(super) previous: *mut Self,
    pub(super) message: *mut c_char,
    pub(super) capacity: usize,
    pub(super) failed: i32,
    pub(super) active: i32,
}
#[repr(C)]
pub(super) struct Manifest {
    pub(super) safety_abi: u32,
    pub(super) native_version_numeric: u32,
    pub(super) array_handle_size: u32,
    pub(super) device_handle_size: u32,
    pub(super) stream_handle_size: u32,
    pub(super) closure_handle_size: u32,
    pub(super) vector_array_handle_size: u32,
    pub(super) string_handle_size: u32,
    pub(super) float32_dtype: i32,
    pub(super) flags: u32,
    pub(super) native_header_version: [u8; 32],
    pub(super) c_source_revision: [u8; 64],
}
pub(super) type ScopeBegin = unsafe extern "C" fn(*mut ErrorScope, *mut c_char, usize) -> i32;
pub(super) type ScopeEnd = unsafe extern "C" fn(*mut ErrorScope) -> i32;
pub(super) type GetManifest = unsafe extern "C" fn(*mut Manifest) -> i32;
pub(super) type SafetyAbi = unsafe extern "C" fn() -> u32;
pub(super) type Free<H> = unsafe extern "C" fn(H) -> i32;
#[cfg(feature = "tensor")]
pub(super) type Callback = unsafe extern "C" fn(*mut CVector, CVector, *mut c_void) -> i32;
#[cfg(feature = "tensor")]
pub(super) type PayloadDrop = unsafe extern "C" fn(*mut c_void);
#[cfg(feature = "tensor")]
pub(super) type ReplayNew =
    unsafe extern "C" fn(CArray, CArray, CArray, CStream, *mut *mut c_void) -> i32;
#[cfg(feature = "tensor")]
pub(super) type ReplayApply = unsafe extern "C" fn(*mut CArray, *mut c_void, CArray, CArray) -> i32;
#[cfg(feature = "tensor")]
pub(super) type ReplayFree = unsafe extern "C" fn(*mut c_void) -> i32;
