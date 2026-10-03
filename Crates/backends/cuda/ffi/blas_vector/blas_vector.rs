//! Cold typed ASUM/SCAL/mode entry retention; the safe handle owner retains the SDK library.
#[rustfmt::skip]
use super::{
    cublas,
    Library,
};
/// Exact SDK function pointers; every use requires the originating library to remain loaded.
pub struct VectorFunctions {
    pub get_pointer_mode: cublas::GetPointerMode,
    pub set_pointer_mode: cublas::SetPointerMode,
    pub sasum: cublas::Sasum,
    pub sscal: cublas::Sscal,
    pub dasum: cublas::Dasum,
    pub dscal: cublas::Dscal,
}
/// Resolve each vector entry once before creating a handle that offers native loss.
///
/// # Safety
/// The library must be a compatible SDK provider and must remain retained by the handle
/// owner through every function invocation and terminal completion or quarantine.
pub unsafe fn retain(
    library: &Library,
) -> Result<VectorFunctions, (&'static str, libloading::Error)> {
    Ok(VectorFunctions {
        get_pointer_mode: *super::resolve_cublas_get_pointer_mode_v2(library)
            .map_err(|error| ("cublasGetPointerMode_v2", error))?,
        set_pointer_mode: *super::resolve_cublas_set_pointer_mode_v2(library)
            .map_err(|error| ("cublasSetPointerMode_v2", error))?,
        sasum: *super::resolve_cublas_sasum_v2(library)
            .map_err(|error| ("cublasSasum_v2", error))?,
        sscal: *super::resolve_cublas_sscal_v2(library)
            .map_err(|error| ("cublasSscal_v2", error))?,
        dasum: *super::resolve_cublas_dasum_v2(library)
            .map_err(|error| ("cublasDasum_v2", error))?,
        dscal: *super::resolve_cublas_dscal_v2(library)
            .map_err(|error| ("cublasDscal_v2", error))?,
    })
}
