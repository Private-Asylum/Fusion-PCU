//! Cold weak logical-device roots across shapes and escaped owners, never warm discovery.
#[rustfmt::skip]
use std::{
    cell::RefCell,
    ffi::OsString,
    rc::{Rc, Weak},
};
use fusion_pcu_vulkan::PcuVulkanBackend;
use crate::PcuExecutionError;

struct Entry {
    driver: Option<OsString>,
    legacy_driver: Option<OsString>,
    ordinal: u32,
    root: Weak<PcuVulkanBackend>,
}
std::thread_local! {
    static ROOTS: RefCell<Vec<Entry>> = const { RefCell::new(Vec::new()) };
}

pub(super) fn retained(ordinal: u32) -> Result<Option<Rc<PcuVulkanBackend>>, PcuExecutionError> {
    let driver = std::env::var_os("VK_DRIVER_FILES");
    let legacy_driver = std::env::var_os("VK_ICD_FILENAMES");
    ROOTS
        .try_with(|roots| {
            let mut roots = roots
                .try_borrow_mut()
                .map_err(|_| PcuExecutionError::ReentrantCall)?;
            roots.retain(|entry| entry.root.strong_count() != 0);
            Ok(roots
                .iter()
                .find(|entry| {
                    entry.ordinal == ordinal
                        && entry.driver == driver
                        && entry.legacy_driver == legacy_driver
                })
                .and_then(|entry| entry.root.upgrade()))
        })
        .map_err(|_| PcuExecutionError::ThreadUnavailable)?
}

pub(super) fn remember(ordinal: u32, root: &Rc<PcuVulkanBackend>) -> Result<(), PcuExecutionError> {
    ROOTS
        .try_with(|roots| {
            let mut roots = roots
                .try_borrow_mut()
                .map_err(|_| PcuExecutionError::ReentrantCall)?;
            roots.retain(|entry| entry.root.strong_count() != 0);
            if !roots
                .iter()
                .any(|entry| entry.root.ptr_eq(&Rc::downgrade(root)))
            {
                roots.push(Entry {
                    ordinal,
                    driver: std::env::var_os("VK_DRIVER_FILES"),
                    legacy_driver: std::env::var_os("VK_ICD_FILENAMES"),
                    root: Rc::downgrade(root),
                });
            }
            Ok(())
        })
        .map_err(|_| PcuExecutionError::ThreadUnavailable)?
}
