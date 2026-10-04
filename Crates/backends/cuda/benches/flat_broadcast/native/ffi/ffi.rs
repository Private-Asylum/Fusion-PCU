//! Isolated stable CUDA Driver ABI for the independent pinned-endpoint control.
#![allow(unsafe_code)] // All direct ABI declarations, foreign calls and pointer views stay here.
#[rustfmt::skip]
use std::{
    ffi::{c_int,c_void},
    sync::{Arc,atomic::{AtomicUsize,Ordering}},
};
use libloading::Library;
type Handle = *mut c_void;
type Device = u64;
type Load = unsafe extern "C" fn(*mut Handle, *const c_void) -> c_int;
type Function = unsafe extern "C" fn(*mut Handle, Handle, *const i8) -> c_int;
type Create = unsafe extern "C" fn(*mut Handle, u32) -> c_int;
type Destroy = unsafe extern "C" fn(Handle) -> c_int;
type Record = unsafe extern "C" fn(Handle, Handle) -> c_int;
type Allocate = unsafe extern "C" fn(*mut Device, usize) -> c_int;
type Free = unsafe extern "C" fn(Device) -> c_int;
type AllocateHost = unsafe extern "C" fn(*mut Handle, usize, u32) -> c_int;
type Upload = unsafe extern "C" fn(Device, *const c_void, usize, Handle) -> c_int;
type Download = unsafe extern "C" fn(*mut c_void, Device, usize, Handle) -> c_int;
type Launch = unsafe extern "C" fn(
    Handle,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    Handle,
    *mut Handle,
    *mut Handle,
) -> c_int;
#[cfg(feature = "allocation-census")]
#[derive(Debug, Default, Clone, Copy)]
pub struct Api {
    pub uploads: u64,
    pub downloads: u64,
    pub launches: u64,
    pub event_records: u64,
    pub event_waits: u64,
    pub uploaded_bytes: u64,
    pub downloaded_bytes: u64,
}
#[cfg(feature = "allocation-census")]
impl Api {
    pub const fn delta(self, before: Self) -> Self {
        Self {
            uploads: self.uploads - before.uploads,
            downloads: self.downloads - before.downloads,
            launches: self.launches - before.launches,
            event_records: self.event_records - before.event_records,
            event_waits: self.event_waits - before.event_waits,
            uploaded_bytes: self.uploaded_bytes - before.uploaded_bytes,
            downloaded_bytes: self.downloaded_bytes - before.downloaded_bytes,
        }
    }
}
struct Host {
    pointer: Handle,
    bytes: usize,
    free: Destroy,
    _library: Arc<Library>,
}
impl Host {
    fn new(program: &Program, initial: &[u8]) -> Self {
        let mut pointer = std::ptr::null_mut();
        // SAFETY: exact cold pinned allocation, before any asynchronous operation.
        assert_eq!(
            unsafe { (program.allocate_host)(&raw mut pointer, initial.len(), 0) },
            0
        );
        assert!(!pointer.is_null());
        let mut owner = Self {
            pointer,
            bytes: initial.len(),
            free: program.free_host,
            _library: program.library.clone(),
        };
        owner.bytes_mut().copy_from_slice(initial);
        owner
    }
    const fn bytes(&self) -> &[u8] {
        // SAFETY: owner retains the exact pinned allocation; public views require terminal wait.
        unsafe { std::slice::from_raw_parts(self.pointer.cast(), self.bytes) }
    }
    const fn bytes_mut(&mut self) -> &mut [u8] {
        // SAFETY: exclusive cold endpoint initialization; no operation is in flight.
        unsafe { std::slice::from_raw_parts_mut(self.pointer.cast(), self.bytes) }
    }
}
impl Drop for Host {
    fn drop(&mut self) {
        // SAFETY: state only drops after known terminal completion. Unknown paths retain it whole.
        let _ = unsafe { (self.free)(self.pointer) };
    }
}
struct DeviceOwner {
    pointer: Device,
    bytes: usize,
    free: Free,
    _library: Arc<Library>,
}
impl DeviceOwner {
    fn new(program: &Program, bytes: usize) -> Self {
        let mut pointer = 0;
        // SAFETY: exact cold selected-context allocation; no asynchronous use exists yet.
        assert_eq!(unsafe { (program.allocate)(&raw mut pointer, bytes) }, 0);
        assert_ne!(pointer, 0);
        Self {
            pointer,
            bytes,
            free: program.free,
            _library: program.library.clone(),
        }
    }
}
impl Drop for DeviceOwner {
    fn drop(&mut self) {
        // SAFETY: final successful wait precedes healthy Drop; unknown paths retain all roots.
        let _ = unsafe { (self.free)(self.pointer) };
    }
}
struct Program {
    library: Arc<Library>,
    module: Handle,
    function: Handle,
    stream: Handle,
    event: Handle,
    unload: Destroy,
    destroy_stream: Destroy,
    destroy_event: Destroy,
    record: Record,
    wait: Destroy,
    launch: Launch,
    allocate: Allocate,
    free: Free,
    allocate_host: AllocateHost,
    free_host: Destroy,
    upload: Upload,
    download: Download,
}
impl Program {
    fn new(image: &[u8]) -> Self {
        // SAFETY: established selected-device Driver ABI signatures; Arc retains symbol owners.
        let library = Arc::new(unsafe { Library::new("libcuda.so.1") }.unwrap());
        let load = unsafe { *library.get::<Load>(b"cuModuleLoadData\0").unwrap() };
        let function = unsafe { *library.get::<Function>(b"cuModuleGetFunction\0").unwrap() };
        let create_stream = unsafe { *library.get::<Create>(b"cuStreamCreate\0").unwrap() };
        let create_event = unsafe { *library.get::<Create>(b"cuEventCreate\0").unwrap() };
        let mut module = std::ptr::null_mut();
        let mut kernel = std::ptr::null_mut();
        let mut stream = std::ptr::null_mut();
        let mut event = std::ptr::null_mut();
        assert_eq!(unsafe { load(&raw mut module, image.as_ptr().cast()) }, 0);
        assert_eq!(
            unsafe { function(&raw mut kernel, module, c"native_flat".as_ptr()) },
            0
        );
        assert_eq!(unsafe { create_stream(&raw mut stream, 1) }, 0);
        assert_eq!(unsafe { create_event(&raw mut event, 2) }, 0);
        Self {
            module,
            function: kernel,
            stream,
            event,
            unload: unsafe { *library.get::<Destroy>(b"cuModuleUnload\0").unwrap() },
            destroy_stream: unsafe { *library.get::<Destroy>(b"cuStreamDestroy_v2\0").unwrap() },
            destroy_event: unsafe { *library.get::<Destroy>(b"cuEventDestroy_v2\0").unwrap() },
            record: unsafe { *library.get::<Record>(b"cuEventRecord\0").unwrap() },
            wait: unsafe { *library.get::<Destroy>(b"cuEventSynchronize\0").unwrap() },
            launch: unsafe { *library.get::<Launch>(b"cuLaunchKernel\0").unwrap() },
            allocate: unsafe { *library.get::<Allocate>(b"cuMemAlloc_v2\0").unwrap() },
            free: unsafe { *library.get::<Free>(b"cuMemFree_v2\0").unwrap() },
            allocate_host: unsafe { *library.get::<AllocateHost>(b"cuMemHostAlloc\0").unwrap() },
            free_host: unsafe { *library.get::<Destroy>(b"cuMemFreeHost\0").unwrap() },
            upload: unsafe { *library.get::<Upload>(b"cuMemcpyHtoDAsync_v2\0").unwrap() },
            download: unsafe { *library.get::<Download>(b"cuMemcpyDtoHAsync_v2\0").unwrap() },
            library,
        }
    }
}
impl Drop for Program {
    fn drop(&mut self) {
        // SAFETY: known final quiescence. On unknown completion the entire state is forgotten.
        unsafe {
            let _ = (self.destroy_event)(self.event);
            let _ = (self.destroy_stream)(self.stream);
            let _ = (self.unload)(self.module);
        }
    }
}
static DROPS: AtomicUsize = AtomicUsize::new(0);
struct State {
    program: Program,
    banks: [Host; 2],
    output: Host,
    input_device: DeviceOwner,
    output_device: DeviceOwner,
    grid: [u32; 3],
    block: [u32; 3],
    prefix: usize,
    #[cfg(feature = "allocation-census")]
    api: std::rc::Rc<std::cell::Cell<Api>>,
    _runtime: fusion_pcu_cuda::CudaRuntime,
}
impl Drop for State {
    fn drop(&mut self) {
        DROPS.fetch_add(1, Ordering::Relaxed);
    }
}
pub struct Owner {
    state: Option<Box<State>>,
}
impl Owner {
    pub fn new(
        runtime: fusion_pcu_cuda::CudaRuntime,
        image: &[u8],
        banks: [&[u8]; 2],
        initial: &[u8],
        prefix: usize,
        grid: [u32; 3],
        block: [u32; 3],
    ) -> Self {
        assert!(
            banks[0].len() == banks[1].len() && !banks[0].is_empty() && prefix <= initial.len()
        );
        let program = Program::new(image);
        let banks = banks.map(|bank| Host::new(&program, bank));
        let output = Host::new(&program, initial);
        let input_device = DeviceOwner::new(&program, banks[0].bytes);
        let output_device = DeviceOwner::new(&program, prefix);
        Self {
            state: Some(Box::new(State {
                program,
                banks,
                output,
                input_device,
                output_device,
                grid,
                block,
                prefix,
                #[cfg(feature = "allocation-census")]
                api: std::rc::Rc::new(std::cell::Cell::new(Api::default())),
                _runtime: runtime,
            })),
        }
    }
    fn retire(&mut self) -> ! {
        self.retain_complete_state();
        panic!("flat broadcast native owner retired after unproven SDK completion");
    }
    const fn retain_complete_state(&mut self) {
        let state = self.state.take().unwrap();
        std::mem::forget(state);
    }
    pub fn call(&mut self, bank: usize) {
        let state = self.state.as_mut().unwrap();
        let p = &state.program;
        assert!(
            state.banks[bank].bytes <= state.input_device.bytes
                && state.prefix <= state.output_device.bytes
        );
        #[cfg(feature = "allocation-census")]
        {
            let mut api = state.api.get();
            api.uploads += 1;
            api.uploaded_bytes += u64::try_from(state.banks[bank].bytes).unwrap();
            state.api.set(api);
        }
        // SAFETY: SDK-pinned input RAM, selected allocation and exact explicit stream survive wait/retirement.
        let upload = unsafe {
            (p.upload)(
                state.input_device.pointer,
                state.banks[bank].pointer.cast_const(),
                state.banks[bank].bytes,
                p.stream,
            )
        };
        if upload != 0 {
            self.retire();
        }
        let state = self.state.as_mut().unwrap();
        let p = &state.program;
        let mut words = [state.input_device.pointer, state.output_device.pointer];
        let mut parameters = words.each_mut().map(|word| std::ptr::from_mut(word).cast());
        #[cfg(feature = "allocation-census")]
        {
            let mut api = state.api.get();
            api.launches += 1;
            state.api.set(api);
        }
        // SAFETY: exact two-pointer compiled ABI; arguments are consumed during the SDK call.
        let launch = unsafe {
            (p.launch)(
                p.function,
                state.grid[0],
                state.grid[1],
                state.grid[2],
                state.block[0],
                state.block[1],
                state.block[2],
                0,
                p.stream,
                parameters.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };
        if launch != 0 {
            self.retire();
        }
        let state = self.state.as_mut().unwrap();
        let p = &state.program;
        #[cfg(feature = "allocation-census")]
        {
            let mut api = state.api.get();
            api.downloads += 1;
            api.downloaded_bytes += u64::try_from(state.prefix).unwrap();
            state.api.set(api);
        }
        // SAFETY: private SDK-pinned output remains exclusive until successful same-stream terminal wait.
        let read = unsafe {
            (p.download)(
                state.output.pointer,
                state.output_device.pointer,
                state.prefix,
                p.stream,
            )
        };
        if read != 0 {
            self.retire();
        }
        let state = self.state.as_mut().unwrap();
        let p = &state.program;
        #[cfg(feature = "allocation-census")]
        {
            let mut api = state.api.get();
            api.event_records += 1;
            state.api.set(api);
        }
        // SAFETY: retained event marks the same stream after upload/kernel/readback.
        let record = unsafe { (p.record)(p.event, p.stream) };
        if record != 0 {
            self.retire();
        }
        let state = self.state.as_mut().unwrap();
        let p = &state.program;
        #[cfg(feature = "allocation-census")]
        {
            let mut api = state.api.get();
            api.event_waits += 1;
            state.api.set(api);
        }
        // SAFETY: this event's success proves all private RAM and device endpoints terminal.
        if unsafe { (p.wait)(p.event) } != 0 {
            self.retire();
        }
    }
    pub fn output(&self) -> &[u8] {
        self.state.as_ref().unwrap().output.bytes()
    }
    #[cfg(feature = "allocation-census")]
    pub fn counter(&self) -> std::rc::Rc<std::cell::Cell<Api>> {
        self.state.as_ref().unwrap().api.clone()
    }
    pub fn known_terminal_retirement_witness(mut self) {
        self.call(0);
        let drops = DROPS.load(Ordering::Relaxed);
        self.retain_complete_state();
        assert_eq!(DROPS.load(Ordering::Relaxed), drops);
        eprintln!(
            "known-terminal-retirement: full raw allocation/pinned RAM/code/event/stream/runtime/Driver owner retained; no driver-loss simulation"
        );
    }
}
