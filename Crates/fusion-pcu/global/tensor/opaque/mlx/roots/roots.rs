//! Cold weak session reuse across source specializations and escaped Rust owners.
//!
//! Code-cache eviction does not mean device-context destruction. Weak roots let
//! differently shaped host preparations share a live exact MLX session without
//! extending an owner's lifetime. Warm execution never consults this registry.
#[rustfmt::skip]
use std::{
    cell::RefCell,
    ffi::OsString,
    rc::{
        Rc,
        Weak,
    },
};
use crate::global::arguments::MlxSourceRoot;
use crate::global::PcuExecutionError;

struct Entry {
    library: Option<OsString>,
    device: usize,
    root: Weak<MlxSourceRoot>,
}

std::thread_local! {
    static ROOTS: RefCell<Vec<Entry>> = const { RefCell::new(Vec::new()) };
}

pub(super) fn retained(device: usize) -> Result<Option<Rc<MlxSourceRoot>>, PcuExecutionError> {
    let library = std::env::var_os("PCU_MLX_LIBRARY");
    ROOTS
        .try_with(|roots| {
            let mut roots = roots
                .try_borrow_mut()
                .map_err(|_| PcuExecutionError::ReentrantCall)?;
            roots.retain(|entry| entry.root.strong_count() != 0);
            let root = roots
                .iter()
                .find(|entry| entry.library == library && entry.device == device)
                .and_then(|entry| entry.root.upgrade());
            if let Some(root) = &root {
                // Reopening a quarantined context would silently evade its terminal-lifetime
                // obligations. Preserve the actual session failure for the consuming API.
                root.session
                    .validate_access_available()
                    .map_err(PcuExecutionError::MlxExecution)?;
            }
            Ok(root)
        })
        .map_err(|_| PcuExecutionError::ThreadUnavailable)?
}

pub(super) fn remember(root: &Rc<MlxSourceRoot>) -> Result<(), PcuExecutionError> {
    let library = std::env::var_os("PCU_MLX_LIBRARY");
    ROOTS
        .try_with(|roots| {
            let mut roots = roots
                .try_borrow_mut()
                .map_err(|_| PcuExecutionError::ReentrantCall)?;
            roots.retain(|entry| entry.root.strong_count() != 0);
            if !roots
                .iter()
                .any(|entry| entry.library == library && entry.root.ptr_eq(&Rc::downgrade(root)))
            {
                roots.push(Entry {
                    library,
                    device: root.session.facts().index,
                    root: Rc::downgrade(root),
                });
            }
            Ok(())
        })
        .map_err(|_| PcuExecutionError::ThreadUnavailable)?
}
