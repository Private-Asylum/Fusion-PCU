//! Backend-private CUDA ABI declarations and dynamic symbol loading.
//! Safe execution and resource ownership live outside this module.

#[rustfmt::skip]
pub use libloading::{
    Library,
    Symbol,
};

/// Resolve a symbol while preserving its borrowed library lifetime.
///
/// # Safety
///
/// `T` must match the named symbol's exact ABI and signature. The symbol must not outlive `library`.
pub unsafe fn symbol<'library, T>(
    library: &'library Library,
    name: &[u8],
) -> Result<Symbol<'library, T>, libloading::Error> {
    // SAFETY: the caller establishes the requested ABI and retained-library lifetime.
    unsafe { library.get(name) }
}

#[path = "loader.rs"]
mod loader;
#[rustfmt::skip]
pub use loader::{
    load_library,
    load_uncached_library,
};

pub mod cublas;
pub mod driver;
pub mod nvrtc;
pub mod runtime;
