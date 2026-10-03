//! Immutable cold handle/GEMM/DOT entry retention beside the originating library owner.
#[rustfmt::skip]
use super::{
    rocblas,
    Library,
};
/// Typed provider pointers; the handle owner retains the originating library through every use.
pub struct BlasHandleFunctions {
    pub destroy: rocblas::DestroyHandle,
    pub set_stream: rocblas::SetStream,
    pub sgemm: rocblas::Sgemm,
    pub sdot: rocblas::Sdot,
}
/// Resolve the required handle operations before creating the handle.
///
/// # Safety
/// The provider must export the matching public rocBLAS ABI and remain loaded through
/// all invocations, terminal completion, destruction or quarantine of its handle.
pub unsafe fn retain(
    library: &Library,
) -> Result<BlasHandleFunctions, (&'static str, libloading::Error)> {
    Ok(BlasHandleFunctions {
        destroy: *unsafe { super::require_rocblas_destroy_handle(library) }
            .map_err(|error| ("rocblas_destroy_handle", error))?,
        set_stream: *unsafe { super::require_rocblas_set_stream(library) }
            .map_err(|error| ("rocblas_set_stream", error))?,
        sgemm: *unsafe { super::require_rocblas_sgemm(library) }
            .map_err(|error| ("rocblas_sgemm", error))?,
        sdot: *unsafe { super::require_rocblas_sdot(library) }
            .map_err(|error| ("rocblas_sdot", error))?,
    })
}
