//! Stable CUDA Driver ABI and native ownership isolated from PCU implementation.
#![allow(unsafe_code)] // Foreign calls and exact byte pointers remain in this module.
#[rustfmt::skip]
use std::{
    cell::Cell,
    ffi::c_void,
    rc::Rc,
};
use libloading::Library;
type Handle = *mut c_void;
type Init = unsafe extern "C" fn(u32) -> i32;
type Device = unsafe extern "C" fn(*mut i32, i32) -> i32;
type Retain = unsafe extern "C" fn(*mut Handle, i32) -> i32;
type Release = unsafe extern "C" fn(i32) -> i32;
type Select = unsafe extern "C" fn(Handle) -> i32;
type Allocate = unsafe extern "C" fn(*mut u64, usize) -> i32;
type Free = unsafe extern "C" fn(u64) -> i32;
type Upload = unsafe extern "C" fn(u64, *const c_void, usize) -> i32;
type Download = unsafe extern "C" fn(*mut c_void, u64, usize) -> i32;
type Wait = unsafe extern "C" fn(Handle) -> i32;

#[derive(Clone, Copy, Debug, Default)]
pub struct Api {
    pub allocations: u64,
    pub frees: u64,
    pub uploads: u64,
    pub downloads: u64,
    pub selections: u64,
    pub uploaded_bytes: u64,
    pub downloaded_bytes: u64,
    pub stream_waits: u64,
}
impl Api {
    pub const fn delta(self, before: Self) -> Self {
        Self {
            allocations: self.allocations - before.allocations,
            frees: self.frees - before.frees,
            uploads: self.uploads - before.uploads,
            downloads: self.downloads - before.downloads,
            selections: self.selections - before.selections,
            uploaded_bytes: self.uploaded_bytes - before.uploaded_bytes,
            downloaded_bytes: self.downloaded_bytes - before.downloaded_bytes,
            stream_waits: self.stream_waits - before.stream_waits,
        }
    }
}
struct Context {
    handle: Handle,
    device: i32,
    release: Release,
    select: Select,
    allocate: Allocate,
    free: Free,
    upload: Upload,
    download: Download,
    wait: Wait,
    api: Cell<Api>,
    _library: Library,
}
impl Context {
    fn select(&self) -> i32 {
        // SAFETY: primary-context reference and loaded library outlive every owner.
        let status = unsafe { (self.select)(self.handle) };
        let mut api = self.api.get();
        api.selections += 1;
        self.api.set(api);
        status
    }
}
impl Drop for Context {
    fn drop(&mut self) {
        // SAFETY: all synchronous owners have been destroyed, or retain this whole context.
        assert_eq!(unsafe { (self.release)(self.device) }, 0);
    }
}
pub struct Control {
    context: Rc<Context>,
}
pub struct Owner {
    context: Rc<Context>,
    pointer: u64,
    bytes: usize,
    poisoned: bool,
    pending_upload: bool,
}
impl Control {
    pub fn new(ordinal: i32) -> Self {
        // SAFETY: fixed official Driver symbol names and ABI types; retained library lifetime.
        let library = unsafe { Library::new("libcuda.so.1") }.unwrap();
        // SAFETY: every symbol is a documented stable CUDA Driver function.
        let (init, device_get, retain, release, select, allocate, free, upload, download, wait) = unsafe {
            (
                *library.get::<Init>(b"cuInit\0").unwrap(),
                *library.get::<Device>(b"cuDeviceGet\0").unwrap(),
                *library
                    .get::<Retain>(b"cuDevicePrimaryCtxRetain\0")
                    .unwrap(),
                *library
                    .get::<Release>(b"cuDevicePrimaryCtxRelease_v2\0")
                    .unwrap(),
                *library.get::<Select>(b"cuCtxSetCurrent\0").unwrap(),
                *library.get::<Allocate>(b"cuMemAlloc_v2\0").unwrap(),
                *library.get::<Free>(b"cuMemFree_v2\0").unwrap(),
                *library.get::<Upload>(b"cuMemcpyHtoD_v2\0").unwrap(),
                *library.get::<Download>(b"cuMemcpyDtoH_v2\0").unwrap(),
                *library.get::<Wait>(b"cuStreamSynchronize\0").unwrap(),
            )
        };
        let mut device = 0;
        let mut handle = std::ptr::null_mut();
        // SAFETY: valid output pointers and fixed selected ordinal; calls retain the context.
        unsafe {
            assert_eq!(init(0), 0);
            assert_eq!(device_get(&raw mut device, ordinal), 0);
            assert_eq!(retain(&raw mut handle, device), 0);
        }
        Self {
            context: Rc::new(Context {
                handle,
                device,
                release,
                select,
                allocate,
                free,
                upload,
                download,
                wait,
                api: Cell::new(Api::default()),
                _library: library,
            }),
        }
    }
    pub fn upload(&self, bytes: &[u8]) -> Owner {
        assert!(!bytes.is_empty());
        assert_eq!(self.context.select(), 0);
        let mut pointer = 0;
        // SAFETY: exact allocation geometry and current retained context.
        assert_eq!(
            unsafe { (self.context.allocate)(&raw mut pointer, bytes.len()) },
            0
        );
        let mut api = self.context.api.get();
        api.allocations += 1;
        self.context.api.set(api);
        let mut owner = Owner {
            context: self.context.clone(),
            pointer,
            bytes: bytes.len(),
            poisoned: false,
            pending_upload: false,
        };
        owner.write(bytes);
        owner
    }
    pub fn api(&self) -> Api {
        self.context.api.get()
    }
}
impl Owner {
    pub fn write(&mut self, bytes: &[u8]) {
        assert!(!self.poisoned);
        assert_eq!(bytes.len(), self.bytes);
        let selection = self.context.select();
        self.poisoned |= selection != 0;
        assert_eq!(selection, 0);
        // SAFETY: exact owned destination; pageable source is staged before this returns.
        // DMA may remain pending; the owner retains the destination until readback or Drop wait.
        let status =
            unsafe { (self.context.upload)(self.pointer, bytes.as_ptr().cast(), bytes.len()) };
        self.poisoned |= status != 0;
        assert_eq!(status, 0);
        self.pending_upload = true;
        let mut api = self.context.api.get();
        api.uploads += 1;
        api.uploaded_bytes += u64::try_from(bytes.len()).unwrap();
        self.context.api.set(api);
    }
    pub fn read(&mut self, output: &mut [u8]) {
        assert!(self.try_read(output));
    }
    pub fn try_read(&mut self, output: &mut [u8]) -> bool {
        assert!(!self.poisoned);
        if output.len() < self.bytes {
            return false;
        }
        let selection = self.context.select();
        self.poisoned |= selection != 0;
        assert_eq!(selection, 0);
        // SAFETY: exact device source and exclusive host prefix; completion is synchronous.
        let status = unsafe {
            (self.context.download)(output.as_mut_ptr().cast(), self.pointer, self.bytes)
        };
        self.poisoned |= status != 0;
        assert_eq!(status, 0);
        self.pending_upload = false;
        let mut api = self.context.api.get();
        api.downloads += 1;
        api.downloaded_bytes += u64::try_from(self.bytes).unwrap();
        self.context.api.set(api);
        true
    }
}
impl Drop for Owner {
    fn drop(&mut self) {
        if self.poisoned {
            // Retain allocation/context/library after any uncertain copy failure.
            std::mem::forget(self.context.clone());
            return;
        }
        if self.context.select() != 0 {
            std::mem::forget(self.context.clone());
            return;
        }
        if self.pending_upload {
            // SAFETY: unsuffixed synchronous copies use legacy default stream; pageable H2D
            // may return after staging. Fence that stream before destroying an unread output.
            let status = unsafe { (self.context.wait)(std::ptr::null_mut()) };
            if status != 0 {
                std::mem::forget(self.context.clone());
                return;
            }
            let mut api = self.context.api.get();
            api.stream_waits += 1;
            self.context.api.set(api);
        }
        // SAFETY: successful D2H completion or the explicit pending-upload wait is terminal.
        let status = unsafe { (self.context.free)(self.pointer) };
        if status != 0 {
            std::mem::forget(self.context.clone());
        }
        assert_eq!(status, 0);
        let mut api = self.context.api.get();
        api.frees += 1;
        self.context.api.set(api);
    }
}
