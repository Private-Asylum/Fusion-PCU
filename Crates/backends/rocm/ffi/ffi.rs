//! Foreign ABI declarations and dynamic loading for `ROCm` provider libraries.

#[rustfmt::skip]
pub use libloading::{
    Library,
    Symbol,
};

/// Resolves one symbol from a retained provider library.
///
/// # Safety
///
/// `T` must exactly match the ABI and signature of the named symbol, and the returned symbol
/// must not outlive `library`.
pub unsafe fn symbol<'library, T>(
    library: &'library Library,
    name: &[u8],
) -> Result<Symbol<'library, T>, libloading::Error> {
    // SAFETY: the caller upholds the symbol name/type and library lifetime contract above.
    unsafe { library.get(name) }
}

#[path = "hip.rs"]
pub mod hip;
#[path = "hiprtc.rs"]
pub mod hiprtc;
#[path = "loader.rs"]
mod loader;
#[path = "rocblas.rs"]
pub mod rocblas;
pub use loader::load_library;
pub use loader::load_uncached_library;
