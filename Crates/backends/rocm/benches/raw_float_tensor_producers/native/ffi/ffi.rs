//! Stable HIP Runtime byte-copy ABI and independent retained stream ownership.
#![allow(unsafe_code)] // Foreign calls and exact initialized byte extents are isolated here.
#[rustfmt::skip]
use std::{
    cell::Cell,
    ffi::c_void,
    rc::Rc,
};
use libloading::Library;
type Handle = *mut c_void;
type Select = unsafe extern "C" fn(i32) -> i32;
type Create = unsafe extern "C" fn(*mut Handle, u32) -> i32;
type Stream = unsafe extern "C" fn(Handle) -> i32;
type Allocate = unsafe extern "C" fn(*mut Handle, usize) -> i32;
type Free = unsafe extern "C" fn(Handle) -> i32;
type Copy = unsafe extern "C" fn(Handle, *const c_void, usize, i32, Handle) -> i32;

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
    stream: Handle,
    device: i32,
    select: Select,
    allocate: Allocate,
    free: Free,
    copy: Copy,
    wait: Stream,
    destroy: Stream,
    api: Cell<Api>,
    _library: Library,
}
impl Context {
    fn select(&self) -> i32 {
        // SAFETY: the selected device and loaded Runtime remain valid for this retained owner.
        let code = unsafe { (self.select)(self.device) };
        let mut api = self.api.get();
        api.selections += 1;
        self.api.set(api);
        code
    }
    fn complete_copy(&self, code: i32) {
        // SAFETY: the independently retained stream orders exactly the submitted copy. Waiting
        // on this stream covers DMA even if a pageable copy returned after staging only.
        let completion = unsafe { (self.wait)(self.stream) };
        let mut api = self.api.get();
        api.stream_waits += 1;
        self.api.set(api);
        if completion != 0 {
            // A benchmark cannot retain an arbitrary borrowed host slice after unknown DMA
            // completion. Stop the process before unwinding could release that host storage.
            std::process::abort();
        }
        assert_eq!(code, 0, "HIP copy failed after terminal stream completion");
    }
}
impl Drop for Context {
    fn drop(&mut self) {
        // SAFETY: every copy has been fenced before its borrowed host extent is released;
        // the final Rc is destroyed only after every independently owned allocation is gone.
        assert_eq!(unsafe { (self.destroy)(self.stream) }, 0);
    }
}
pub struct Control {
    context: Rc<Context>,
}
pub struct Owner {
    context: Rc<Context>,
    pointer: Handle,
    bytes: usize,
    poisoned: bool,
}
impl Control {
    pub fn new(device: i32) -> Self {
        // SAFETY: fixed installed HIP Runtime ABI; this owner retains the loaded library.
        let library = unsafe { Library::new("/opt/rocm/lib/libamdhip64.so") }.unwrap();
        // SAFETY: signatures and constants match hip/hip_runtime_api.h, without PCU wrappers.
        let (select, create, allocate, free, copy, wait, destroy) = unsafe {
            (
                *library.get::<Select>(b"hipSetDevice\0").unwrap(),
                *library
                    .get::<Create>(b"hipStreamCreateWithFlags\0")
                    .unwrap(),
                *library.get::<Allocate>(b"hipMalloc\0").unwrap(),
                *library.get::<Free>(b"hipFree\0").unwrap(),
                *library.get::<Copy>(b"hipMemcpyAsync\0").unwrap(),
                *library.get::<Stream>(b"hipStreamSynchronize\0").unwrap(),
                *library.get::<Stream>(b"hipStreamDestroy\0").unwrap(),
            )
        };
        let mut stream = std::ptr::null_mut();
        // SAFETY: fixed device ordinal and valid stream output; flag 1 is hipStreamNonBlocking.
        unsafe {
            assert_eq!(select(device), 0);
            assert_eq!(create(&raw mut stream, 1), 0);
        }
        Self {
            context: Rc::new(Context {
                stream,
                device,
                select,
                allocate,
                free,
                copy,
                wait,
                destroy,
                api: Cell::new(Api::default()),
                _library: library,
            }),
        }
    }
    pub fn upload(&self, bytes: &[u8]) -> Owner {
        assert!(!bytes.is_empty());
        assert_eq!(self.context.select(), 0);
        let mut pointer = std::ptr::null_mut();
        // SAFETY: valid pointer output, nonempty exact allocation extent and selected device.
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
        // SAFETY: owned exact destination, initialized exclusive source and retained stream.
        // The source borrow remains live until complete_copy proves terminal quiescence.
        let code = unsafe {
            (self.context.copy)(
                self.pointer,
                bytes.as_ptr().cast(),
                self.bytes,
                1,
                self.context.stream,
            )
        };
        self.context.complete_copy(code);
        let mut api = self.context.api.get();
        api.uploads += 1;
        api.uploaded_bytes += u64::try_from(self.bytes).unwrap();
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
        // SAFETY: exact initialized device source and exclusive host prefix; tails untouched.
        // Host output remains borrowed until complete_copy fences the explicit stream.
        let code = unsafe {
            (self.context.copy)(
                output.as_mut_ptr().cast(),
                self.pointer,
                self.bytes,
                2,
                self.context.stream,
            )
        };
        self.context.complete_copy(code);
        let mut api = self.context.api.get();
        api.downloads += 1;
        api.downloaded_bytes += u64::try_from(self.bytes).unwrap();
        self.context.api.set(api);
        true
    }
}
impl Drop for Owner {
    fn drop(&mut self) {
        if self.poisoned || self.context.select() != 0 {
            // Retain allocation, stream and library if selecting its device becomes uncertain.
            std::mem::forget(self.context.clone());
            return;
        }
        // SAFETY: both upload and readback have independently established stream quiescence.
        let code = unsafe { (self.context.free)(self.pointer) };
        if code != 0 {
            std::mem::forget(self.context.clone());
        }
        assert_eq!(code, 0);
        let mut api = self.context.api.get();
        api.frees += 1;
        self.context.api.set(api);
    }
}
