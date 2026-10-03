//! Benchmark-private direct SDK endpoints; caller RAM is never submitted.
#[rustfmt::skip]
use std::ffi::{
    c_int,
    c_void,
};
#[rustfmt::skip]
use fusion_pcu_cuda::{
    CudaKernel,
    CudaStreamHandle,
    CudaError,
    CudaCompletion,
    CudaKernelArgument,
};
type Allocate = unsafe extern "C" fn(*mut *mut c_void, usize) -> c_int;
type Release = unsafe extern "C" fn(*mut c_void) -> c_int;
type Copy = unsafe extern "C" fn(*mut c_void, *const c_void, usize, c_int) -> c_int;
type Select = unsafe extern "C" fn(c_int) -> c_int;
#[derive(Clone, Copy)]
pub struct Pointer {
    raw: *mut c_void,
    bytes: usize,
}
pub struct Sdk {
    allocate: Allocate,
    release: Release,
    copy: Copy,
    _library: libloading::Library,
}
impl Sdk {
    pub fn new(device: u32) -> Self {
        // SAFETY: the selected benchmark SDK exports these established runtime ABI signatures.
        let path = std::env::var_os("PCU_NATIVE_RUNTIME_LIBRARY").map_or_else(
            || std::path::PathBuf::from("/usr/lib/x86_64-linux-gnu/libcudart.so.12"),
            std::path::PathBuf::from,
        );
        let path = std::fs::canonicalize(path).unwrap();
        eprintln!(
            "cold/native_owned_minimum: runtime_library={} selected_device={device}",
            path.display()
        );
        let library = unsafe { libloading::Library::new(path) }.unwrap();
        let allocate = unsafe { *library.get::<Allocate>(b"cudaMalloc\0").unwrap() };
        let release = unsafe { *library.get::<Release>(b"cudaFree\0").unwrap() };
        let copy = unsafe { *library.get::<Copy>(b"cudaMemcpy\0").unwrap() };
        let select = unsafe { *library.get::<Select>(b"cudaSetDevice\0").unwrap() };
        assert_eq!(unsafe { select(c_int::try_from(device).unwrap()) }, 0);
        Self {
            allocate,
            release,
            copy,
            _library: library,
        }
    }
    pub fn allocate(&self, bytes: usize) -> Pointer {
        let mut pointer = std::ptr::null_mut();
        // SAFETY: exact output pointer and admitted nonzero native allocation size.
        assert_eq!(unsafe { (self.allocate)(&raw mut pointer, bytes) }, 0);
        assert!(!pointer.is_null());
        Pointer {
            raw: pointer,
            bytes,
        }
    }
    pub fn upload(&self, destination: Pointer, source: &[u8]) -> c_int {
        // SAFETY: private owner keeps source RAM and selected-device allocation stable until
        // known completion; its error path retires the entire RAM/code/stream/SDK state.
        assert!(source.len() <= destination.bytes);
        unsafe { (self.copy)(destination.raw, source.as_ptr().cast(), source.len(), 1) }
    }
    pub fn read(&self, destination: &mut [u8], source: Pointer) -> c_int {
        // SAFETY: destination is exclusively owned retained RAM, never a caller borrow.
        assert!(destination.len() <= source.bytes);
        unsafe {
            (self.copy)(
                destination.as_mut_ptr().cast(),
                source.raw,
                destination.len(),
                2,
            )
        }
    }
    pub fn release(&self, pointer: Pointer) {
        // SAFETY: healthy owner only drops after its final completion and synchronous copies.
        let _ = unsafe { (self.release)(pointer.raw) };
    }
}
pub fn launch(
    kernel: &CudaKernel,
    stream: &CudaStreamHandle,
    grid: [u32; 3],
    block: [u32; 3],
    pointers: [Pointer; 4],
) -> Result<CudaCompletion, CudaError> {
    assert_eq!(
        size_of::<usize>(),
        8,
        "captured SDK storage-pointer ABI is64bit"
    );
    let words = pointers.map(|pointer| (pointer.raw as usize).to_ne_bytes());
    let arguments = words.each_ref().map(|word| CudaKernelArgument::Bytes(word));
    // SAFETY: cold generated ABI froze three pointer roles then the private U64 fault pointer.
    // Private raw owner holds all pointers, RAM, code and stream through wait or retirement.
    unsafe { kernel.launch(stream, grid, block, 0, &arguments) }
}

const LIBRARY: &str = "libcuda.so.1";
const LOAD: &[u8] = b"cuModuleLoadData\0";
const GET: &[u8] = b"cuModuleGetFunction\0";
const CREATE_STREAM: &[u8] = b"cuStreamCreate\0";
const CREATE_EVENT: &[u8] = b"cuEventCreate\0";
const LAUNCH: &[u8] = b"cuLaunchKernel\0";
const RECORD: &[u8] = b"cuEventRecord\0";
const WAIT: &[u8] = b"cuEventSynchronize\0";
const DESTROY_EVENT: &[u8] = b"cuEventDestroy_v2\0";
const DESTROY_STREAM: &[u8] = b"cuStreamDestroy_v2\0";
const UNLOAD: &[u8] = b"cuModuleUnload\0";

// Independent retained SDK handles: no PCU launch/completion adapter in the warm path.
type Load = unsafe extern "C" fn(*mut *mut c_void, *const c_void) -> c_int;
type Function = unsafe extern "C" fn(*mut *mut c_void, *mut c_void, *const i8) -> c_int;
type Create = unsafe extern "C" fn(*mut *mut c_void, u32) -> c_int;
type Destroy = unsafe extern "C" fn(*mut c_void) -> c_int;
type Record = unsafe extern "C" fn(*mut c_void, *mut c_void) -> c_int;
type Launch = unsafe extern "C" fn(
    *mut c_void,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    *mut c_void,
    *mut *mut c_void,
    *mut *mut c_void,
) -> c_int;
pub struct Program {
    module: *mut c_void,
    function: *mut c_void,
    stream: *mut c_void,
    event: *mut c_void,
    launch: Launch,
    record: Record,
    wait: Destroy,
    destroy_event: Destroy,
    destroy_stream: Destroy,
    unload: Destroy,
    allocate_host: AllocateHost,
    release_host: Destroy,
    upload: UploadAsync,
    read: ReadAsync,
    library: std::sync::Arc<libloading::Library>,
}
impl Program {
    pub fn new(image: &[u8]) -> Self {
        let library = std::sync::Arc::new(unsafe { libloading::Library::new(LIBRARY) }.unwrap());
        // SAFETY: exact documented selected-device SDK module/stream/event signatures;
        // all symbol pointers remain owned by this program through terminal or retirement.
        let load = unsafe { *library.get::<Load>(LOAD).unwrap() };
        let get = unsafe { *library.get::<Function>(GET).unwrap() };
        let create_stream = unsafe { *library.get::<Create>(CREATE_STREAM).unwrap() };
        let create_event = unsafe { *library.get::<Create>(CREATE_EVENT).unwrap() };
        let mut module = std::ptr::null_mut();
        let mut function = std::ptr::null_mut();
        let mut stream = std::ptr::null_mut();
        let mut event = std::ptr::null_mut();
        assert_eq!(unsafe { load(&raw mut module, image.as_ptr().cast()) }, 0);
        assert_eq!(unsafe { get(&raw mut function, module, c"native_helpers".as_ptr()) }, 0);
        assert_eq!(unsafe { create_stream(&raw mut stream, 1) }, 0);
        assert_eq!(unsafe { create_event(&raw mut event, 2) }, 0);
        Self {
            module,
            function,
            stream,
            event,
            launch: unsafe { *library.get::<Launch>(LAUNCH).unwrap() },
            record: unsafe { *library.get::<Record>(RECORD).unwrap() },
            wait: unsafe { *library.get::<Destroy>(WAIT).unwrap() },
            destroy_event: unsafe { *library.get::<Destroy>(DESTROY_EVENT).unwrap() },
            destroy_stream: unsafe { *library.get::<Destroy>(DESTROY_STREAM).unwrap() },
            unload: unsafe { *library.get::<Destroy>(UNLOAD).unwrap() },
            allocate_host: unsafe { *library.get::<AllocateHost>(ALLOCATE_HOST).unwrap() },
            release_host: unsafe { *library.get::<Destroy>(RELEASE_HOST).unwrap() },
            upload: unsafe { *library.get::<UploadAsync>(UPLOAD_ASYNC).unwrap() },
            read: unsafe { *library.get::<ReadAsync>(READ_ASYNC).unwrap() },
            library,
        }
    }

    fn upload_owned(&self, destination: Pointer, source: &Host) -> c_int {
        assert!(source.bytes <= destination.bytes);
        // SAFETY: SDK-pinned source and allocation remain owned through this retained stream's
        // terminal event or whole-state quarantine. No caller RAM or default stream is used.
        unsafe {
            (self.upload)(
                destination.raw as u64,
                source.pointer.cast_const(),
                source.bytes,
                self.stream,
            )
        }
    }
    fn read_owned(&self, destination: &mut Host, source: Pointer, bytes: usize) -> c_int {
        assert!(bytes <= destination.bytes && bytes <= source.bytes);
        // SAFETY: private SDK-pinned destination, exact bounds, exclusive stream-owned state.
        unsafe { (self.read)(destination.pointer, source.raw as u64, bytes, self.stream) }
    }
    fn finish(&self) -> c_int {
        // SAFETY: event/stream handles belong to this retained selected runtime; reuse follows
        // terminal completion, and error paths retain code/queue/endpoint roots together.
        let record = unsafe { (self.record)(self.event, self.stream) };
        if record != 0 {
            return record;
        }
        unsafe { (self.wait)(self.event) }
    }
    pub fn initialize_status(&self, ram: &Endpoints, status: Pointer) -> c_int {
        let code = self.upload_owned(status, &ram.status);
        if code != 0 {
            return code;
        }
        self.finish()
    }
    pub fn submit(
        &self,
        ram: &mut Endpoints,
        bank: usize,
        pointers: [Pointer; 4],
        grid: [u32; 3],
        block: [u32; 3],
        prefix: usize,
    ) -> c_int {
        let upload = self.upload_owned(pointers[2], &ram.banks[bank]);
        if upload != 0 {
            return upload;
        }
        let mut words = pointers.map(|pointer| pointer.raw);
        let mut parameters = words.each_mut().map(|word| std::ptr::from_mut(word).cast());
        // SAFETY: driver consumes pointer arguments during launch. H2D, kernel, faultword
        // and both output readbacks use this SAME retained stream and owned pinned endpoints.
        let launch = unsafe {
            (self.launch)(
                self.function,
                grid[0],
                grid[1],
                grid[2],
                block[0],
                block[1],
                block[2],
                0,
                self.stream,
                parameters.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };
        if launch != 0 {
            return launch;
        }
        let status = self.read_owned(&mut ram.status, pointers[3], 8);
        if status != 0 {
            return status;
        }
        for (destination, source) in ram.readbacks.iter_mut().zip(pointers) {
            let read = self.read_owned(destination, source, prefix);
            if read != 0 {
                return read;
            }
        }
        // Both outputs remain private until this wait succeeds and the faultword is decoded.
        self.finish()
    }
}
impl Drop for Program {
    fn drop(&mut self) {
        // SAFETY: healthy owner only reaches Drop after terminal wait for all queued kernels and transfers;
        // errors leak the complete boxed owner, so none of these handles is destroyed early.
        unsafe {
            let _ = (self.destroy_event)(self.event);
            let _ = (self.destroy_stream)(self.stream);
            let _ = (self.unload)(self.module);
        }
    }
}

type UploadAsync = unsafe extern "C" fn(u64, *const c_void, usize, *mut c_void) -> c_int;
type ReadAsync = unsafe extern "C" fn(*mut c_void, u64, usize, *mut c_void) -> c_int;
const ALLOCATE_HOST: &[u8] = b"cuMemHostAlloc\0";
const RELEASE_HOST: &[u8] = b"cuMemFreeHost\0";
const UPLOAD_ASYNC: &[u8] = b"cuMemcpyHtoDAsync_v2\0";
const READ_ASYNC: &[u8] = b"cuMemcpyDtoHAsync_v2\0";

type AllocateHost = unsafe extern "C" fn(*mut *mut c_void, usize, u32) -> c_int;
struct Host {
    pointer: *mut c_void,
    bytes: usize,
    release: Destroy,
    _library: std::sync::Arc<libloading::Library>,
}
impl Host {
    fn new(program: &Program, bytes: &[u8]) -> Self {
        let mut pointer = std::ptr::null_mut();
        // SAFETY: cold exact pinned host allocation; no native work uses the buffer yet.
        assert_eq!(
            unsafe { (program.allocate_host)(&raw mut pointer, bytes.len(), 0) },
            0
        );
        assert!(!pointer.is_null());
        let mut owner = Self {
            pointer,
            bytes: bytes.len(),
            release: program.release_host,
            _library: program.library.clone(),
        };
        owner.bytes_mut().copy_from_slice(bytes);
        owner
    }
    const fn bytes(&self) -> &[u8] {
        // SAFETY: this owner retains the exact SDK-pinned allocation; callers inspect only
        // after a successful terminal wait, and exclusive state prevents overlapping submit.
        unsafe { std::slice::from_raw_parts(self.pointer.cast(), self.bytes) }
    }
    const fn bytes_mut(&mut self) -> &mut [u8] {
        // SAFETY: exclusive owned endpoint; mutation/growth occurs only before submission or
        // after proven terminal completion. Unknown paths forget the whole enclosing owner.
        unsafe { std::slice::from_raw_parts_mut(self.pointer.cast(), self.bytes) }
    }
}
impl Drop for Host {
    fn drop(&mut self) {
        // SAFETY: healthy enclosing state is terminal, its selected runtime remains alive.
        // Unknown completion paths retain this allocation and loader without invoking Drop.
        let _ = unsafe { (self.release)(self.pointer) };
    }
}
pub struct Endpoints {
    banks: [Host; 2],
    readbacks: [Host; 2],
    status: Host,
}
impl Endpoints {
    pub fn new(program: &Program, banks: [&[u8]; 2], initial_output: &[u8]) -> Self {
        Self {
            banks: banks.map(|bytes| Host::new(program, bytes)),
            readbacks: std::array::from_fn(|_| Host::new(program, initial_output)),
            status: Host::new(program, &u64::MAX.to_le_bytes()),
        }
    }
    pub const fn bank(&self, index: usize) -> &[u8] {
        self.banks[index].bytes()
    }
    pub const fn replace_bank(&mut self, index: usize, bytes: &[u8]) {
        self.banks[index].bytes_mut().copy_from_slice(bytes);
    }
    pub fn readbacks(&self) -> [&[u8]; 2] {
        self.readbacks.each_ref().map(Host::bytes)
    }
    pub fn word(&self) -> u64 {
        u64::from_le_bytes(self.status.bytes().try_into().unwrap())
    }
    pub const fn status_address(&self) -> *const u8 {
        self.status.pointer.cast()
    }
    pub const fn reset(&mut self) {
        self.status
            .bytes_mut()
            .copy_from_slice(&u64::MAX.to_le_bytes());
    }
}
