//! Process-lifetime dynamic loader state for the HIP runtime.

use std::{
    collections::HashMap,
    ffi::OsString,
    sync::{
        Arc,
        Mutex,
        OnceLock,
    },
};

use libloading::Library;

type LibraryCache = Mutex<HashMap<OsString, Arc<Library>>>;

static HIP_LIBRARIES: OnceLock<LibraryCache> = OnceLock::new();

/// Loads a library once per exact candidate and retains it for the process lifetime.
///
/// HIP has process-wide runtime state. Unloading `libamdhip64` after a probe can leave that state
/// unusable when a later caller reloads the library, so even probe-only handles remain cached.
#[allow(clippy::redundant_pub_crate)] // The runtime loader is internal to the backend root.
pub(super) fn load_library(candidate: &std::ffi::OsStr) -> Result<Arc<Library>, String> {
    let cache = HIP_LIBRARIES.get_or_init(|| Mutex::new(HashMap::new()));
    // A poisoned cache lock does not invalidate the libraries it already retains. Recover the
    // map so later callers can still reuse those handles and avoid unloading HIP.
    let mut cache = cache
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some(library) = cache.get(candidate) {
        return Ok(Arc::clone(library));
    }

    // SAFETY: callers resolve and invoke symbols only while holding an Arc to this Library.
    // Successful loads are also retained by the process-wide cache, preventing HIP teardown.
    let library = unsafe { Library::new(candidate) }.map_err(|error| error.to_string())?;
    let library = Arc::new(library);
    cache.insert(candidate.to_os_string(), Arc::clone(&library));
    drop(cache);
    Ok(library)
}

#[cfg(all(test, target_os = "linux"))]
mod tests {
    use std::{
        ffi::OsStr,
        sync::Arc,
    };

    use super::load_library;

    #[test]
    fn successful_library_load_is_shared_for_the_exact_candidate() {
        let first = load_library(OsStr::new("libc.so.6")).expect("system C library loads");
        let second = load_library(OsStr::new("libc.so.6")).expect("cached C library loads");
        assert!(Arc::ptr_eq(&first, &second));
    }
}
