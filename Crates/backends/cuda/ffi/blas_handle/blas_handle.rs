//! Cold immutable handle/DOT entries retained with the originating cuBLAS library.
#[rustfmt::skip]
use super::{
    cublas,
    Library,
};
/// Provider pointers; the safe owner retains the library through completion or quarantine.
pub struct BlasHandleFunctions {
    pub destroy: cublas::DestroyHandle,
    pub set_stream: cublas::SetStream,
    pub sdot: cublas::Sdot,
}
/// Resolve each required handle entry before creating a handle.
///
/// # Errors
/// Returns the exact missing entry without constructing a partially usable handle.
pub fn retain(library: &Library) -> Result<BlasHandleFunctions, (&'static str, libloading::Error)> {
    Ok(BlasHandleFunctions {
        destroy: *super::resolve_cublas_destroy_v2(library)
            .map_err(|error| ("cublasDestroy_v2", error))?,
        set_stream: *super::resolve_cublas_set_stream_v2(library)
            .map_err(|error| ("cublasSetStream_v2", error))?,
        sdot: *super::resolve_cublas_sdot_v2(library).map_err(|error| ("cublasSdot_v2", error))?,
    })
}
