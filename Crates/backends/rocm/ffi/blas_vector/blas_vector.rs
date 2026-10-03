//! Cold typed ASUM/SCAL/mode entry retention; the safe handle owner retains the SDK library.
#[rustfmt::skip]
use super::{
    rocblas,
    Library,
};
/// Exact SDK function pointers; every use requires the originating library to remain loaded.
pub struct VectorFunctions {
    pub get_pointer_mode: rocblas::GetPointerMode,
    pub set_pointer_mode: rocblas::SetPointerMode,
    pub sasum: rocblas::Sasum,
    pub sscal: rocblas::Sscal,
    pub dasum: rocblas::Dasum,
    pub dscal: rocblas::Dscal,
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
        get_pointer_mode: *unsafe { super::require_rocblas_get_pointer_mode(library) }
            .map_err(|error| ("rocblas_get_pointer_mode", error))?,
        set_pointer_mode: *unsafe { super::require_rocblas_set_pointer_mode(library) }
            .map_err(|error| ("rocblas_set_pointer_mode", error))?,
        sasum: *unsafe { super::require_rocblas_sasum(library) }
            .map_err(|error| ("rocblas_sasum", error))?,
        sscal: *unsafe { super::require_rocblas_sscal(library) }
            .map_err(|error| ("rocblas_sscal", error))?,
        dasum: *unsafe { super::require_rocblas_dasum(library) }
            .map_err(|error| ("rocblas_dasum", error))?,
        dscal: *unsafe { super::require_rocblas_dscal(library) }
            .map_err(|error| ("rocblas_dscal", error))?,
    })
}
