//! Reject unknown safety/source/header/handle families before native owner creation.

use libloading::Library;
use crate::MlxError;
#[rustfmt::skip]
use super::abi::{
    GetManifest,
    Manifest,
    SafetyAbi,
};
#[rustfmt::skip]
use super::super::{
    text_value,
    SDK_VERSION,
};
use std::ffi::c_void;

pub(super) fn admit(library: &Library) -> Result<(), MlxError> {
    // SAFETY: trusted C image; this stable zero-argument query has no versioned layout.
    let safety_abi = unsafe { library.get::<SafetyAbi>(b"pcu_mlx_c_safety_abi\0") }
        .map(|symbol| *symbol)
        .map_err(|error| MlxError::Abi(error.to_string()))?;
    // SAFETY: this stable zero-argument extension query precedes every layout-dependent
    // call. An unknown future manifest/scope layout is rejected before any pointer write.
    if unsafe { safety_abi() } != 1 {
        return Err(MlxError::Abi("unsupported C safety ABI family".into()));
    }
    // SAFETY: ABI1 was admitted before resolving this exact manifest signature.
    let get_manifest = unsafe { library.get::<GetManifest>(b"pcu_mlx_c_manifest_get\0") }
        .map(|symbol| *symbol)
        .map_err(|error| MlxError::Abi(error.to_string()))?;
    let mut manifest = Manifest {
        safety_abi: 0,
        native_version_numeric: 0,
        array_handle_size: 0,
        device_handle_size: 0,
        stream_handle_size: 0,
        closure_handle_size: 0,
        vector_array_handle_size: 0,
        string_handle_size: 0,
        float32_dtype: 0,
        flags: 0,
        native_header_version: [0; 32],
        c_source_revision: [0; 64],
    };
    // SAFETY: exact repr(C) ABI1 writable manifest and no handle/default-state inputs.
    if unsafe { get_manifest(&raw mut manifest) } != 0 {
        return Err(MlxError::Abi("C safety manifest unavailable".into()));
    }
    let pointer_size = u32::try_from(size_of::<*mut c_void>())
        .map_err(|_| MlxError::Abi("pointer size exceeds manifest".into()))?;
    if manifest.safety_abi != 1
        || manifest.native_version_numeric != 32_003
        || manifest.float32_dtype != 10
        || manifest.flags != 1
        || [
            manifest.array_handle_size,
            manifest.device_handle_size,
            manifest.stream_handle_size,
            manifest.closure_handle_size,
            manifest.vector_array_handle_size,
            manifest.string_handle_size,
        ]
        .iter()
        .any(|&size| size != pointer_size)
        || text_value(&manifest.native_header_version)? != SDK_VERSION
    {
        return Err(MlxError::Abi(
            "unsupported C safety/header/handle family".into(),
        ));
    }
    let source = text_value(&manifest.c_source_revision)?;
    if source != "a341b4925024b88b2c593468f16e12f5e17315da" {
        return Err(MlxError::Abi("invalid pinned C source revision".into()));
    }
    Ok(())
}
