//! Established cuBLASLt C ABI, loading and call boundary.
#[rustfmt::skip]
use std::{
    ffi::{
        c_int,
        c_void,
        OsStr,
    },
    sync::Arc,
};
use super::Library;
use crate::CublasError;
pub const CUDA_R_32F: c_int = 0;
pub const CUDA_R_64F: c_int = 1;
pub const COMPUTE_32F: c_int = 68;
pub const COMPUTE_64F: c_int = 70;
pub const COMPUTE_32F_FAST_TF32: c_int = 77;
pub const DESC_POINTER_MODE: c_int = 2;
pub const DESC_TRANSA: c_int = 3;
pub const DESC_TRANSB: c_int = 4;
pub const POINTER_MODE_HOST: c_int = 0;
pub const LAYOUT_ORDER: c_int = 1;
pub const ORDER_ROW: c_int = 1;
pub const PREF_MAX_WORKSPACE: c_int = 1;
pub const PREF_ALIGNMENTS: [c_int; 4] = [5, 6, 7, 8];
pub const CAP_ALIGNMENTS: [c_int; 4] = [16, 17, 18, 19];
pub const CONFIG_ID: c_int = 0;

/// Optional caller-thread API census; driver and allocation counts are separate.
#[cfg(feature = "allocation-census")]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ApiCensus {
    pub loads: usize,
    pub heuristics: usize,
    pub matmuls: usize,
}
#[cfg(feature = "allocation-census")]
std::thread_local! { static CENSUS: std::cell::Cell<ApiCensus> = const { std::cell::Cell::new(ApiCensus { loads: 0, heuristics: 0, matmuls: 0 }) }; }
#[cfg(feature = "allocation-census")]
#[must_use]
pub fn api_census() -> ApiCensus {
    CENSUS.get()
}
#[cfg(feature = "allocation-census")]
pub fn reset_api_census() {
    CENSUS.set(ApiCensus::default());
}
pub type Handle = *mut c_void;
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct Algorithm {
    pub data: [u64; 8],
}
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct Heuristic {
    pub algorithm: Algorithm,
    pub workspace: usize,
    pub status: c_int,
    pub waves: f32,
    pub reserved: [c_int; 4],
}
pub struct Api {
    _library: Arc<Library>,
    create: unsafe extern "C" fn(*mut Handle) -> c_int,
    destroy: unsafe extern "C" fn(Handle) -> c_int,
    property: unsafe extern "C" fn(c_int, *mut c_int) -> c_int,
    desc_create: unsafe extern "C" fn(*mut Handle, c_int, c_int) -> c_int,
    desc_destroy: unsafe extern "C" fn(Handle) -> c_int,
    desc_set: unsafe extern "C" fn(Handle, c_int, *const c_void, usize) -> c_int,
    layout_create: unsafe extern "C" fn(*mut Handle, c_int, u64, u64, i64) -> c_int,
    layout_destroy: unsafe extern "C" fn(Handle) -> c_int,
    layout_set: unsafe extern "C" fn(Handle, c_int, *const c_void, usize) -> c_int,
    preference_create: unsafe extern "C" fn(*mut Handle) -> c_int,
    preference_destroy: unsafe extern "C" fn(Handle) -> c_int,
    preference_set: unsafe extern "C" fn(Handle, c_int, *const c_void, usize) -> c_int,
    heuristic: unsafe extern "C" fn(
        Handle,
        Handle,
        Handle,
        Handle,
        Handle,
        Handle,
        Handle,
        c_int,
        *mut Heuristic,
        *mut c_int,
    ) -> c_int,
    config: unsafe extern "C" fn(*const Algorithm, c_int, *mut c_void, usize, *mut usize) -> c_int,
    capability:
        unsafe extern "C" fn(*const Algorithm, c_int, *mut c_void, usize, *mut usize) -> c_int,
    matmul: unsafe extern "C" fn(
        Handle,
        Handle,
        *const c_void,
        *const c_void,
        Handle,
        *const c_void,
        Handle,
        *const c_void,
        *const c_void,
        Handle,
        *mut c_void,
        Handle,
        *const Algorithm,
        *mut c_void,
        usize,
        *mut c_void,
    ) -> c_int,
}
impl Api {
    #[allow(clippy::too_many_lines)] // Resolve the complete established ABI table atomically.
    pub fn load() -> Result<Self, CublasError> {
        #[cfg(feature = "allocation-census")]
        CENSUS.with(|cell| {
            let mut census = cell.get();
            census.loads += 1;
            cell.set(census);
        });
        let mut failures = Vec::new();
        for name in ["libcublasLt.so.13", "libcublasLt.so"] {
            let library = match super::load_library(OsStr::new(name)) {
                Ok(value) => value,
                Err(error) => {
                    failures.push(error);
                    continue;
                }
            };
            // SAFETY: every field matches the established SDK signature; the exact library is retained.
            return unsafe {
                Ok(Self {
                    create: *super::symbol(&library, b"cublasLtCreate\0").map_err(|error| {
                        CublasError::MissingSymbol {
                            symbol: "cublasLtCreate",
                            detail: error.to_string(),
                        }
                    })?,
                    destroy: *super::symbol(&library, b"cublasLtDestroy\0").map_err(|error| {
                        CublasError::MissingSymbol {
                            symbol: "cublasLtDestroy",
                            detail: error.to_string(),
                        }
                    })?,
                    property: *super::symbol(&library, b"cublasLtGetProperty\0").map_err(
                        |error| CublasError::MissingSymbol {
                            symbol: "cublasLtGetProperty",
                            detail: error.to_string(),
                        },
                    )?,
                    desc_create: *super::symbol(&library, b"cublasLtMatmulDescCreate\0").map_err(
                        |error| CublasError::MissingSymbol {
                            symbol: "cublasLtMatmulDescCreate",
                            detail: error.to_string(),
                        },
                    )?,
                    desc_destroy: *super::symbol(&library, b"cublasLtMatmulDescDestroy\0")
                        .map_err(|error| CublasError::MissingSymbol {
                            symbol: "cublasLtMatmulDescDestroy",
                            detail: error.to_string(),
                        })?,
                    desc_set: *super::symbol(&library, b"cublasLtMatmulDescSetAttribute\0")
                        .map_err(|error| CublasError::MissingSymbol {
                            symbol: "cublasLtMatmulDescSetAttribute",
                            detail: error.to_string(),
                        })?,
                    layout_create: *super::symbol(&library, b"cublasLtMatrixLayoutCreate\0")
                        .map_err(|error| CublasError::MissingSymbol {
                            symbol: "cublasLtMatrixLayoutCreate",
                            detail: error.to_string(),
                        })?,
                    layout_destroy: *super::symbol(&library, b"cublasLtMatrixLayoutDestroy\0")
                        .map_err(|error| CublasError::MissingSymbol {
                            symbol: "cublasLtMatrixLayoutDestroy",
                            detail: error.to_string(),
                        })?,
                    layout_set: *super::symbol(&library, b"cublasLtMatrixLayoutSetAttribute\0")
                        .map_err(|error| CublasError::MissingSymbol {
                            symbol: "cublasLtMatrixLayoutSetAttribute",
                            detail: error.to_string(),
                        })?,
                    preference_create: *super::symbol(
                        &library,
                        b"cublasLtMatmulPreferenceCreate\0",
                    )
                    .map_err(|error| CublasError::MissingSymbol {
                        symbol: "cublasLtMatmulPreferenceCreate",
                        detail: error.to_string(),
                    })?,
                    preference_destroy: *super::symbol(
                        &library,
                        b"cublasLtMatmulPreferenceDestroy\0",
                    )
                    .map_err(|error| CublasError::MissingSymbol {
                        symbol: "cublasLtMatmulPreferenceDestroy",
                        detail: error.to_string(),
                    })?,
                    preference_set: *super::symbol(
                        &library,
                        b"cublasLtMatmulPreferenceSetAttribute\0",
                    )
                    .map_err(|error| CublasError::MissingSymbol {
                        symbol: "cublasLtMatmulPreferenceSetAttribute",
                        detail: error.to_string(),
                    })?,
                    heuristic: *super::symbol(&library, b"cublasLtMatmulAlgoGetHeuristic\0")
                        .map_err(|error| CublasError::MissingSymbol {
                            symbol: "cublasLtMatmulAlgoGetHeuristic",
                            detail: error.to_string(),
                        })?,
                    config: *super::symbol(&library, b"cublasLtMatmulAlgoConfigGetAttribute\0")
                        .map_err(|error| CublasError::MissingSymbol {
                            symbol: "cublasLtMatmulAlgoConfigGetAttribute",
                            detail: error.to_string(),
                        })?,
                    capability: *super::symbol(&library, b"cublasLtMatmulAlgoCapGetAttribute\0")
                        .map_err(|error| CublasError::MissingSymbol {
                            symbol: "cublasLtMatmulAlgoCapGetAttribute",
                            detail: error.to_string(),
                        })?,
                    matmul: *super::symbol(&library, b"cublasLtMatmul\0").map_err(|error| {
                        CublasError::MissingSymbol {
                            symbol: "cublasLtMatmul",
                            detail: error.to_string(),
                        }
                    })?,
                    _library: library,
                })
            };
        }
        Err(CublasError::LibraryUnavailable(failures.join("; ")))
    }
    /// # Safety
    /// Pointers and handles must match their SDK roles and remain live through queued work.
    #[allow(clippy::too_many_arguments)] // Preserve the established SDK boundary.
    pub unsafe fn create(&self, out: *mut Handle) -> Result<(), CublasError> {
        // SAFETY: the caller supplies live, correctly typed SDK objects.
        let code = unsafe { (self.create)(out) };
        if code == 0 {
            Ok(())
        } else {
            Err(CublasError::Status {
                operation: "cublasLtCreate",
                code,
            })
        }
    }
    /// # Safety
    /// Pointers and handles must match their SDK roles and remain live through queued work.
    #[allow(clippy::too_many_arguments)] // Preserve the established SDK boundary.
    pub unsafe fn destroy(&self, handle: Handle) -> Result<(), CublasError> {
        // SAFETY: the caller supplies live, correctly typed SDK objects.
        let code = unsafe { (self.destroy)(handle) };
        if code == 0 {
            Ok(())
        } else {
            Err(CublasError::Status {
                operation: "cublasLtDestroy",
                code,
            })
        }
    }
    /// # Safety
    /// Pointers and handles must match their SDK roles and remain live through queued work.
    #[allow(clippy::too_many_arguments)] // Preserve the established SDK boundary.
    pub unsafe fn property(&self, attribute: c_int, out: *mut c_int) -> Result<(), CublasError> {
        // SAFETY: the caller supplies live, correctly typed SDK objects.
        let code = unsafe { (self.property)(attribute, out) };
        if code == 0 {
            Ok(())
        } else {
            Err(CublasError::Status {
                operation: "cublasLtGetProperty",
                code,
            })
        }
    }
    /// # Safety
    /// Pointers and handles must match their SDK roles and remain live through queued work.
    #[allow(clippy::too_many_arguments)] // Preserve the established SDK boundary.
    pub unsafe fn desc_create(
        &self,
        out: *mut Handle,
        compute: c_int,
        scalar: c_int,
    ) -> Result<(), CublasError> {
        // SAFETY: the caller supplies live, correctly typed SDK objects.
        let code = unsafe { (self.desc_create)(out, compute, scalar) };
        if code == 0 {
            Ok(())
        } else {
            Err(CublasError::Status {
                operation: "cublasLtMatmulDescCreate",
                code,
            })
        }
    }
    /// # Safety
    /// Pointers and handles must match their SDK roles and remain live through queued work.
    #[allow(clippy::too_many_arguments)] // Preserve the established SDK boundary.
    pub unsafe fn desc_destroy(&self, handle: Handle) -> Result<(), CublasError> {
        // SAFETY: the caller supplies live, correctly typed SDK objects.
        let code = unsafe { (self.desc_destroy)(handle) };
        if code == 0 {
            Ok(())
        } else {
            Err(CublasError::Status {
                operation: "cublasLtMatmulDescDestroy",
                code,
            })
        }
    }
    /// # Safety
    /// Pointers and handles must match their SDK roles and remain live through queued work.
    #[allow(clippy::too_many_arguments)] // Preserve the established SDK boundary.
    pub unsafe fn desc_set(
        &self,
        handle: Handle,
        attribute: c_int,
        value: *const c_void,
        bytes: usize,
    ) -> Result<(), CublasError> {
        // SAFETY: the caller supplies live, correctly typed SDK objects.
        let code = unsafe { (self.desc_set)(handle, attribute, value, bytes) };
        if code == 0 {
            Ok(())
        } else {
            Err(CublasError::Status {
                operation: "cublasLtMatmulDescSetAttribute",
                code,
            })
        }
    }
    /// # Safety
    /// Pointers and handles must match their SDK roles and remain live through queued work.
    #[allow(clippy::too_many_arguments)] // Preserve the established SDK boundary.
    pub unsafe fn layout_create(
        &self,
        out: *mut Handle,
        scalar: c_int,
        rows: u64,
        columns: u64,
        leading: i64,
    ) -> Result<(), CublasError> {
        // SAFETY: the caller supplies live, correctly typed SDK objects.
        let code = unsafe { (self.layout_create)(out, scalar, rows, columns, leading) };
        if code == 0 {
            Ok(())
        } else {
            Err(CublasError::Status {
                operation: "cublasLtMatrixLayoutCreate",
                code,
            })
        }
    }
    /// # Safety
    /// Pointers and handles must match their SDK roles and remain live through queued work.
    #[allow(clippy::too_many_arguments)] // Preserve the established SDK boundary.
    pub unsafe fn layout_destroy(&self, handle: Handle) -> Result<(), CublasError> {
        // SAFETY: the caller supplies live, correctly typed SDK objects.
        let code = unsafe { (self.layout_destroy)(handle) };
        if code == 0 {
            Ok(())
        } else {
            Err(CublasError::Status {
                operation: "cublasLtMatrixLayoutDestroy",
                code,
            })
        }
    }
    /// # Safety
    /// Pointers and handles must match their SDK roles and remain live through queued work.
    #[allow(clippy::too_many_arguments)] // Preserve the established SDK boundary.
    pub unsafe fn layout_set(
        &self,
        handle: Handle,
        attribute: c_int,
        value: *const c_void,
        bytes: usize,
    ) -> Result<(), CublasError> {
        // SAFETY: the caller supplies live, correctly typed SDK objects.
        let code = unsafe { (self.layout_set)(handle, attribute, value, bytes) };
        if code == 0 {
            Ok(())
        } else {
            Err(CublasError::Status {
                operation: "cublasLtMatrixLayoutSetAttribute",
                code,
            })
        }
    }
    /// # Safety
    /// Pointers and handles must match their SDK roles and remain live through queued work.
    #[allow(clippy::too_many_arguments)] // Preserve the established SDK boundary.
    pub unsafe fn preference_create(&self, out: *mut Handle) -> Result<(), CublasError> {
        // SAFETY: the caller supplies live, correctly typed SDK objects.
        let code = unsafe { (self.preference_create)(out) };
        if code == 0 {
            Ok(())
        } else {
            Err(CublasError::Status {
                operation: "cublasLtMatmulPreferenceCreate",
                code,
            })
        }
    }
    /// # Safety
    /// Pointers and handles must match their SDK roles and remain live through queued work.
    #[allow(clippy::too_many_arguments)] // Preserve the established SDK boundary.
    pub unsafe fn preference_destroy(&self, handle: Handle) -> Result<(), CublasError> {
        // SAFETY: the caller supplies live, correctly typed SDK objects.
        let code = unsafe { (self.preference_destroy)(handle) };
        if code == 0 {
            Ok(())
        } else {
            Err(CublasError::Status {
                operation: "cublasLtMatmulPreferenceDestroy",
                code,
            })
        }
    }
    /// # Safety
    /// Pointers and handles must match their SDK roles and remain live through queued work.
    #[allow(clippy::too_many_arguments)] // Preserve the established SDK boundary.
    pub unsafe fn preference_set(
        &self,
        handle: Handle,
        attribute: c_int,
        value: *const c_void,
        bytes: usize,
    ) -> Result<(), CublasError> {
        // SAFETY: the caller supplies live, correctly typed SDK objects.
        let code = unsafe { (self.preference_set)(handle, attribute, value, bytes) };
        if code == 0 {
            Ok(())
        } else {
            Err(CublasError::Status {
                operation: "cublasLtMatmulPreferenceSetAttribute",
                code,
            })
        }
    }
    /// # Safety
    /// Pointers and handles must match their SDK roles and remain live through queued work.
    #[allow(clippy::too_many_arguments)] // Preserve the established SDK boundary.
    pub unsafe fn heuristic(
        &self,
        handle: Handle,
        desc: Handle,
        a: Handle,
        b: Handle,
        c: Handle,
        d: Handle,
        preference: Handle,
        count: c_int,
        results: *mut Heuristic,
        returned: *mut c_int,
    ) -> Result<(), CublasError> {
        // SAFETY: the caller supplies live, correctly typed SDK objects.
        #[cfg(feature = "allocation-census")]
        CENSUS.with(|cell| {
            let mut census = cell.get();
            census.heuristics += 1;
            cell.set(census);
        });
        let code = unsafe {
            (self.heuristic)(
                handle, desc, a, b, c, d, preference, count, results, returned,
            )
        };
        if code == 0 {
            Ok(())
        } else {
            Err(CublasError::Status {
                operation: "cublasLtMatmulAlgoGetHeuristic",
                code,
            })
        }
    }
    /// # Safety
    /// Pointers and handles must match their SDK roles and remain live through queued work.
    #[allow(clippy::too_many_arguments)] // Preserve the established SDK boundary.
    pub unsafe fn config(
        &self,
        algorithm: *const Algorithm,
        attribute: c_int,
        value: *mut c_void,
        bytes: usize,
        written: *mut usize,
    ) -> Result<(), CublasError> {
        // SAFETY: the caller supplies live, correctly typed SDK objects.
        let code = unsafe { (self.config)(algorithm, attribute, value, bytes, written) };
        if code == 0 {
            Ok(())
        } else {
            Err(CublasError::Status {
                operation: "cublasLtMatmulAlgoConfigGetAttribute",
                code,
            })
        }
    }
    /// # Safety
    /// Pointers and handles must match their SDK roles and remain live through queued work.
    #[allow(clippy::too_many_arguments)] // Preserve the established SDK boundary.
    pub unsafe fn capability(
        &self,
        algorithm: *const Algorithm,
        attribute: c_int,
        value: *mut c_void,
        bytes: usize,
        written: *mut usize,
    ) -> Result<(), CublasError> {
        // SAFETY: the caller supplies live, correctly typed SDK objects.
        let code = unsafe { (self.capability)(algorithm, attribute, value, bytes, written) };
        if code == 0 {
            Ok(())
        } else {
            Err(CublasError::Status {
                operation: "cublasLtMatmulAlgoCapGetAttribute",
                code,
            })
        }
    }
    /// # Safety
    /// Pointers and handles must match their SDK roles and remain live through queued work.
    #[allow(clippy::too_many_arguments)] // Preserve the established SDK boundary.
    pub unsafe fn matmul(
        &self,
        handle: Handle,
        desc: Handle,
        alpha: *const c_void,
        a: *const c_void,
        a_layout: Handle,
        b: *const c_void,
        b_layout: Handle,
        beta: *const c_void,
        c: *const c_void,
        c_layout: Handle,
        d: *mut c_void,
        d_layout: Handle,
        algorithm: *const Algorithm,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> Result<(), CublasError> {
        // SAFETY: the caller supplies live, correctly typed SDK objects.
        #[cfg(feature = "allocation-census")]
        CENSUS.with(|cell| {
            let mut census = cell.get();
            census.matmuls += 1;
            cell.set(census);
        });
        #[cfg(feature = "allocation-census")]
        super::census::lt_matmul();
        let code = unsafe {
            (self.matmul)(
                handle,
                desc,
                alpha,
                a,
                a_layout,
                b,
                b_layout,
                beta,
                c,
                c_layout,
                d,
                d_layout,
                algorithm,
                workspace,
                workspace_bytes,
                stream,
            )
        };
        if code == 0 {
            Ok(())
        } else {
            Err(CublasError::Status {
                operation: "cublasLtMatmul",
                code,
            })
        }
    }
}
#[cfg(test)]
mod tests {
    use super::{Algorithm, Heuristic};
    #[test]
    fn installed_x86_64_heuristic_abi() {
        assert_eq!(std::mem::size_of::<Algorithm>(), 64);
        assert_eq!(std::mem::align_of::<Algorithm>(), 8);
        assert_eq!(std::mem::size_of::<Heuristic>(), 96);
        assert_eq!(std::mem::align_of::<Heuristic>(), 8);
        assert_eq!(std::mem::offset_of!(Heuristic, workspace), 64);
        assert_eq!(std::mem::offset_of!(Heuristic, status), 72);
        assert_eq!(std::mem::offset_of!(Heuristic, waves), 76);
        assert_eq!(std::mem::offset_of!(Heuristic, reserved), 80);
    }
}
