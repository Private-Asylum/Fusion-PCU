//! Cold library loading, source admission, typed entrypoint resolution and version ownership.

#[rustfmt::skip]
use std::{
    ffi::{
        c_char,
        c_void,
    },
    path::Path,
    rc::Rc,
};
use libloading::Library;
use crate::MlxError;
#[rustfmt::skip]
use super::{
    admission::admit,
    api::Api,
    abi::{
        CCarrier,
        CarrierNew,
        CarrierPrefixNew,
        CarrierApply,
        CCheckedUnary,
        CCheckedBinary,
        BinaryNew,
        BinaryApply,
        CheckedNew,
        CheckedPrefixNew,
        CheckedApply,
        CheckedUpload,
        CArray,
        CDevice,
        CDeviceInfo,
        CString,
        CStream,
        Free,
        ScopeBegin,
        ScopeEnd,
    },
    owner::Owner,
    text::bounded_text,
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
use super::super::{
    require_apple_silicon,
    SDK_VERSION,
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
impl Api {
    #[allow(
        clippy::too_many_lines,
        reason = "Keep the cold audited C table and ABI admission together so signatures and lifetime cannot drift across helpers."
    )]
    pub fn load(path: &Path) -> Result<Rc<Self>, MlxError> {
        require_apple_silicon()?;
        // SAFETY: the caller selects the trusted full upstream C image built by c_api's
        // exact-source consumer. Extension manifest gates precede any native object call.
        let library = unsafe { Library::new(path) }
            .map_err(|error| MlxError::Unavailable(error.to_string()))?;
        macro_rules! symbol {
            ($name:literal, $ty:ty) => {{
                #[cfg(feature = "division-census")]
                super::div_rem::census::record(super::div_rem::census::Call::TableSymbol);
                #[cfg(feature = "carrier-census")]
                super::carrier::census::record(super::carrier::census::Call::TableSymbol);
                #[cfg(feature = "binary-census")]
                super::binary::census::record(super::binary::census::Call::TableSymbol);
                #[cfg(feature = "integer-census")]
                super::integer::census::record(super::integer::census::Call::TableSymbol);
                // SAFETY: names/signatures match audited exact-source upstream C headers
                // and safety.h. Copied pointers remain bound to the retained Library.
                unsafe { library.get::<$ty>(concat!($name, "\0").as_bytes()) }
                    .map(|symbol| *symbol)
                    .map_err(|error| MlxError::Abi(error.to_string()))?
            }};
        }
        admit(&library)?;
        let version_function = symbol!("mlx_version", unsafe extern "C" fn(*mut CString) -> i32);
        let string_data = symbol!(
            "mlx_string_data",
            unsafe extern "C" fn(CString) -> *const c_char
        );
        let string_free = symbol!("mlx_string_free", Free<CString>);
        let api = Rc::new(Self {
            composed_new: symbol!("pcu_mlx_c_composed_new", ComposedNew),
            composed_apply: symbol!("pcu_mlx_c_composed_apply", ComposedApply),
            composed_free: symbol!("pcu_mlx_c_composed_free", Free<CComposed>),
            transport_new: symbol!("pcu_mlx_c_transport_new", TransportNew),
            transport_apply: symbol!("pcu_mlx_c_transport_apply", TransportApply),
            transport_free: symbol!("pcu_mlx_c_transport_free", Free<CTransport>),
            version: SDK_VERSION.into(),
            begin: symbol!("pcu_mlx_c_error_scope_begin", ScopeBegin),
            end: symbol!("pcu_mlx_c_error_scope_end", ScopeEnd),
            device_count: symbol!(
                "mlx_device_count",
                unsafe extern "C" fn(*mut i32, i32) -> i32
            ),
            device_new: symbol!(
                "mlx_device_new_type",
                unsafe extern "C" fn(i32, i32) -> CDevice
            ),
            device_available: symbol!(
                "mlx_device_is_available",
                unsafe extern "C" fn(*mut bool, CDevice) -> i32
            ),
            device_free: symbol!("mlx_device_free", Free<CDevice>),
            info_get: symbol!(
                "mlx_device_info_get",
                unsafe extern "C" fn(*mut CDeviceInfo, CDevice) -> i32
            ),
            info_has: symbol!(
                "mlx_device_info_has_key",
                unsafe extern "C" fn(*mut bool, CDeviceInfo, *const c_char) -> i32
            ),
            info_string: symbol!(
                "mlx_device_info_get_string",
                unsafe extern "C" fn(*mut *const c_char, CDeviceInfo, *const c_char) -> i32
            ),
            info_free: symbol!("mlx_device_info_free", Free<CDeviceInfo>),
            metal_available: symbol!(
                "mlx_metal_is_available",
                unsafe extern "C" fn(*mut bool) -> i32
            ),
            stream_new: symbol!(
                "mlx_stream_new_device",
                unsafe extern "C" fn(CDevice) -> CStream
            ),
            stream_free: symbol!("mlx_stream_free", Free<CStream>),
            synchronize: symbol!("mlx_synchronize", unsafe extern "C" fn(CStream) -> i32),
            array_set_data: symbol!(
                "mlx_array_set_data",
                unsafe extern "C" fn(*mut CArray, *const c_void, *const i32, i32, i32) -> i32
            ),
            array_set: symbol!(
                "mlx_array_set",
                unsafe extern "C" fn(*mut CArray, CArray) -> i32
            ),
            array_free: symbol!("mlx_array_free", Free<CArray>),
            array_dtype: symbol!("mlx_array_dtype", unsafe extern "C" fn(CArray) -> i32),
            array_ndim: symbol!("mlx_array_ndim", unsafe extern "C" fn(CArray) -> usize),
            array_dim: symbol!("mlx_array_dim", unsafe extern "C" fn(CArray, i32) -> i32),
            array_size: symbol!("mlx_array_size", unsafe extern "C" fn(CArray) -> usize),
            array_nbytes: symbol!("mlx_array_nbytes", unsafe extern "C" fn(CArray) -> usize),
            array_strides: symbol!(
                "mlx_array_strides",
                unsafe extern "C" fn(CArray) -> *const usize
            ),
            array_data: symbol!(
                "mlx_array_data_float32",
                unsafe extern "C" fn(CArray) -> *const f32
            ),
            array_data_u8: symbol!(
                "mlx_array_data_uint8",
                unsafe extern "C" fn(CArray) -> *const u8
            ),
            array_data_u16: symbol!(
                "mlx_array_data_uint16",
                unsafe extern "C" fn(CArray) -> *const u16
            ),
            array_data_u32: symbol!(
                "mlx_array_data_uint32",
                unsafe extern "C" fn(CArray) -> *const u32
            ),
            checked_new: symbol!("pcu_mlx_c_checked_unary_new", CheckedNew),
            checked_prefix_new: symbol!("pcu_mlx_c_checked_unary_prefix_new", CheckedPrefixNew),
            checked_apply: symbol!("pcu_mlx_c_checked_unary_apply", CheckedApply),
            checked_upload: symbol!("pcu_mlx_c_checked_upload", CheckedUpload),
            checked_free: symbol!("pcu_mlx_c_checked_unary_free", Free<CCheckedUnary>),
            binary_new: symbol!("pcu_mlx_c_checked_binary_new", BinaryNew),
            binary_prefix_new: symbol!("pcu_mlx_c_checked_binary_prefix_new", BinaryNew),
            binary_apply: symbol!("pcu_mlx_c_checked_binary_apply", BinaryApply),
            binary_free: symbol!("pcu_mlx_c_checked_binary_free", Free<CCheckedBinary>),
            integer_new: symbol!("pcu_mlx_c_checked_integer_new", IntegerNew),
            integer_prefix_new: symbol!("pcu_mlx_c_checked_integer_prefix_new", IntegerNew),
            integer_apply: symbol!("pcu_mlx_c_checked_integer_apply", IntegerApply),
            integer_free: symbol!("pcu_mlx_c_checked_integer_free", Free<CCheckedInteger>),
            div_rem_new: symbol!("pcu_mlx_c_checked_div_rem_new", DivRemNew),
            div_rem_prefix_new: symbol!("pcu_mlx_c_checked_div_rem_prefix_new", DivRemNew),
            div_rem_apply: symbol!("pcu_mlx_c_checked_div_rem_apply", DivRemApply),
            div_rem_free: symbol!("pcu_mlx_c_checked_div_rem_free", Free<CCheckedDivRem>),
            f32_view_validate_shared: symbol!(
                "pcu_mlx_c_f32_view_validate_shared",
                unsafe extern "C" fn(CArray, CArray) -> i32
            ),
            f32_view: symbol!(
                "pcu_mlx_c_f32_view",
                unsafe extern "C" fn(*mut CArray, CArray, CStream, i32, i32, i32) -> i32
            ),
            encoded_prefix: symbol!(
                "pcu_mlx_c_encoded_prefix_merge",
                unsafe extern "C" fn(*mut CArray, CArray, CArray, CStream) -> i32
            ),
            carrier_new: symbol!("pcu_mlx_c_carrier_new", CarrierNew),
            carrier_prefix_new: symbol!("pcu_mlx_c_carrier_prefix_new", CarrierPrefixNew),
            carrier_apply: symbol!("pcu_mlx_c_carrier_apply", CarrierApply),
            carrier_free: symbol!("pcu_mlx_c_carrier_free", Free<CCarrier>),
            array_eval: symbol!("mlx_array_eval", unsafe extern "C" fn(CArray) -> i32),
            array_wait: symbol!("_mlx_array_wait", unsafe extern "C" fn(CArray) -> i32),
            array_available: symbol!(
                "_mlx_array_is_available",
                unsafe extern "C" fn(*mut bool, CArray) -> i32
            ),
            array_contiguous: symbol!(
                "_mlx_array_is_row_contiguous",
                unsafe extern "C" fn(*mut bool, CArray) -> i32
            ),
            #[cfg(feature = "tensor")]
            matmul: symbol!(
                "mlx_matmul",
                unsafe extern "C" fn(*mut CArray, CArray, CArray, CStream) -> i32
            ),
            #[cfg(feature = "tensor")]
            vector_set_data: symbol!(
                "mlx_vector_array_set_data",
                unsafe extern "C" fn(*mut CVector, *const CArray, usize) -> i32
            ),
            #[cfg(feature = "tensor")]
            vector_get: symbol!(
                "mlx_vector_array_get",
                unsafe extern "C" fn(*mut CArray, CVector, usize) -> i32
            ),
            #[cfg(feature = "tensor")]
            vector_set_value: symbol!(
                "mlx_vector_array_set_value",
                unsafe extern "C" fn(*mut CVector, CArray) -> i32
            ),
            #[cfg(feature = "tensor")]
            vector_size: symbol!(
                "mlx_vector_array_size",
                unsafe extern "C" fn(CVector) -> usize
            ),
            #[cfg(feature = "tensor")]
            vector_free: symbol!("mlx_vector_array_free", Free<CVector>),
            #[cfg(feature = "tensor")]
            closure_new: symbol!(
                "mlx_closure_new_func_payload",
                unsafe extern "C" fn(Callback, *mut c_void, PayloadDrop) -> CClosure
            ),
            #[cfg(feature = "tensor")]
            closure_apply: symbol!(
                "mlx_closure_apply",
                unsafe extern "C" fn(*mut CVector, CClosure, CVector) -> i32
            ),
            #[cfg(feature = "tensor")]
            closure_free: symbol!("mlx_closure_free", Free<CClosure>),
            #[cfg(feature = "tensor")]
            compile: symbol!(
                "mlx_compile",
                unsafe extern "C" fn(*mut CClosure, CClosure, bool) -> i32
            ),
            #[cfg(feature = "tensor")]
            replay_new: symbol!("pcu_mlx_c_replay_new", ReplayNew),
            #[cfg(feature = "tensor")]
            replay_apply: symbol!("pcu_mlx_c_replay_apply", ReplayApply),
            #[cfg(feature = "tensor")]
            replay_free: symbol!("pcu_mlx_c_replay_free", ReplayFree),
            library,
        });
        let mut native_version = Owner::empty(Rc::clone(&api), string_free);
        // SAFETY: official string output starts with the valid empty sentinel.
        api.status(|| unsafe { version_function(&raw mut native_version.raw) })?;
        // SAFETY: successful version owns a valid initialized C string.
        let data = api.guarded(|| unsafe { string_data(native_version.raw) })?;
        let actual = bounded_text(data, 32)?;
        native_version.release()?;
        if actual != SDK_VERSION {
            return Err(MlxError::Abi(format!("unsupported C runtime{actual}")));
        }
        Ok(api)
    }

    /// Select the Cargo-built runtime, allowing a process-local explicit override.
    pub fn load_default() -> Result<Rc<Self>, MlxError> {
        require_apple_silicon()?;
        let path = std::env::var_os("PCU_MLX_LIBRARY")
            .or_else(|| {
                option_env!("PCU_MLX_DEFAULT_LIBRARY")
                    .filter(|path| !path.is_empty())
                    .map(Into::into)
            })
            .ok_or_else(|| {
                MlxError::Unavailable("Cargo-built MLX runtime path is unavailable".into())
            })?;
        Self::load(Path::new(&path))
    }
}
