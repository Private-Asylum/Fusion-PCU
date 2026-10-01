//! Explicit GPU facts and device owners; descriptive identity remains bounded.

#[rustfmt::skip]
use std::{
    ffi::CStr,
    rc::Rc,
};
#[rustfmt::skip]
use crate::{
    MlxDeviceFacts,
    MlxError,
    MlxGpuBackend,
};
#[rustfmt::skip]
use super::{
    abi::CDevice,
    api::Api,
    owner::Owner,
    text::bounded_text,
};

impl Api {
    pub fn devices(self: &Rc<Self>) -> Result<Vec<MlxDeviceFacts>, MlxError> {
        let mut available = false;
        // SAFETY: writable bool and selected catch-all Metal query; no workload is submitted.
        self.status(|| unsafe { (self.metal_available)(&raw mut available) })?;
        if !available {
            return Err(MlxError::Unavailable("MLX Metal GPU unavailable".into()));
        }
        let mut count = 0;
        // SAFETY: GPU enum value1 is frozen by the selected upstream header family.
        self.status(|| unsafe { (self.device_count)(&raw mut count, 1) })?;
        let count =
            usize::try_from(count).map_err(|_| MlxError::Abi("negative GPU count".into()))?;
        (0..count).map(|index| self.device(index)).collect()
    }

    pub(super) fn device_owner(self: &Rc<Self>, index: usize) -> Result<Owner<CDevice>, MlxError> {
        let index = i32::try_from(index).map_err(|_| MlxError::InvalidExtent)?;
        let mut owner = Owner::empty(Rc::clone(self), self.device_free);
        // SAFETY: native constructor validates explicit GPU index; nonempty is required for
        // this constructor, unlike allocation-free empty sentinels used for output setters.
        self.guarded(|| {
            // SAFETY: explicit GPU index is checked by the contained native constructor.
            owner.raw = unsafe { (self.device_new)(1, index) };
        })?;
        owner.require_live()?;
        let mut available = false;
        // SAFETY: retained valid device and exact writable bool.
        self.status(|| unsafe { (self.device_available)(&raw mut available, owner.raw) })?;
        if !available {
            return Err(MlxError::Unavailable("explicit MLX GPU unavailable".into()));
        }
        Ok(owner)
    }
    pub fn device(self: &Rc<Self>, index: usize) -> Result<MlxDeviceFacts, MlxError> {
        let device = self.device_owner(index)?;
        let mut info = Owner::empty(Rc::clone(self), self.info_free);
        // SAFETY: empty official output sentinel and valid device; result retains own map.
        self.status(|| unsafe { (self.info_get)(&raw mut info.raw, device.raw) })?;
        info.require_live()?;
        let read = |key: &CStr| -> Result<String, MlxError> {
            let mut present = false;
            // SAFETY: live map and terminated fixed key, exact bool output.
            self.status(|| unsafe { (self.info_has)(&raw mut present, info.raw, key.as_ptr()) })?;
            if !present {
                return Ok(String::new());
            }
            let mut text = std::ptr::null();
            // SAFETY: actual string-valued installed device keys; status2 rejects type mismatch.
            self.status(|| unsafe { (self.info_string)(&raw mut text, info.raw, key.as_ptr()) })?;
            bounded_text(text, 256)
        };
        Ok(MlxDeviceFacts {
            index,
            backend: MlxGpuBackend::Metal,
            name: read(c"device_name")?,
            architecture: read(c"architecture")?,
        })
    }
}
