//! Isolated stable CUDA Driver ABI for the independent pinned-endpoint control.
#![allow(unsafe_code)] // All direct ABI declarations, foreign calls and pointer views stay here.
#[rustfmt::skip]
use std::{
    ffi::{c_int,c_void},
    sync::Arc,
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
type Reset = unsafe extern "C" fn(Device, u8, usize, Handle) -> c_int;
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
    pub allocations: u64,
    pub frees: u64,
    pub uploads: u64,
    pub downloads: u64,
    pub resets: u64,
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
            allocations: self.allocations - before.allocations,
            frees: self.frees - before.frees,
            uploads: self.uploads - before.uploads,
            downloads: self.downloads - before.downloads,
            resets: self.resets - before.resets,
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
    functions: [Handle; 2],
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
    reset: Reset,
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
        let mut kernels = [std::ptr::null_mut(); 2];
        let mut stream = std::ptr::null_mut();
        let mut event = std::ptr::null_mut();
        assert_eq!(unsafe { load(&raw mut module, image.as_ptr().cast()) }, 0);
        for (kernel, name) in kernels
            .iter_mut()
            .zip([c"independent_literal_add", c"independent_literal_mul"])
        {
            assert_eq!(
                unsafe { function(std::ptr::from_mut(kernel), module, name.as_ptr()) },
                0
            );
        }
        assert_eq!(unsafe { create_stream(&raw mut stream, 1) }, 0);
        assert_eq!(unsafe { create_event(&raw mut event, 2) }, 0);
        Self {
            module,
            functions: kernels,
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
            reset: unsafe { *library.get::<Reset>(b"cuMemsetD8Async\0").unwrap() },
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
struct Endpoint {
    host: Host,
    device: DeviceOwner,
}
impl Endpoint {
    fn new(program: &Program, bytes: usize) -> Self {
        Self {
            host: Host::new(program, &vec![0; bytes]),
            device: DeviceOwner::new(program, bytes),
        }
    }
}
struct State {
    program: Program,
    input: Endpoint,
    seed: Endpoint,
    stage: Option<Endpoint>,
    output: Endpoint,
    status: Endpoint,
    fresh_output: Option<DeviceOwner>,
    grid: [u32; 3],
    block: [u32; 3],
    extent: u32,
    invocations: u32,
    clamp: u32,
    dead: u32,
    terminal: bool,
    #[cfg(feature = "allocation-census")]
    api: std::rc::Rc<std::cell::Cell<Api>>,
    _runtime: fusion_pcu_cuda::CudaRuntime,
}
impl State {
    /// Read private pinned shadows, then prove all queued work terminal before host access.
    fn readback<const PAYLOAD: bool, const STATUS: bool>(&mut self) -> bool {
        let s = self;
        let mut failed = false;
        for (enabled, e) in [(PAYLOAD, &s.output), (STATUS, &s.status)] {
            if !enabled {
                continue;
            }
            #[cfg(feature = "allocation-census")]
            {
                let mut a = s.api.get();
                a.downloads += 1;
                a.downloaded_bytes += u64::try_from(e.device.bytes).unwrap();
                s.api.set(a);
            }
            // SAFETY: private pinned shadows remain inaccessible until the following event completes.
            if unsafe {
                (s.program.download)(
                    e.host.pointer,
                    if PAYLOAD && std::ptr::eq(e, &raw const s.output) {
                        s.fresh_output.as_ref().unwrap_or(&e.device).pointer
                    } else {
                        e.device.pointer
                    },
                    e.device.bytes,
                    s.program.stream,
                )
            } != 0
            {
                failed = true;
                break;
            }
        }
        if failed {
            return false;
        }
        #[cfg(feature = "allocation-census")]
        {
            let mut a = s.api.get();
            a.event_records += 1;
            s.api.set(a);
        }
        // SAFETY: retained event records after all uploads/kernel/private readbacks on this stream.
        if unsafe { (s.program.record)(s.program.event, s.program.stream) } != 0 {
            return false;
        }
        #[cfg(feature = "allocation-census")]
        {
            let mut a = s.api.get();
            a.event_waits += 1;
            s.api.set(a);
        }
        // SAFETY: success proves the complete private resource set is terminal.
        if unsafe { (s.program.wait)(s.program.event) } != 0 {
            return false;
        }
        true
    }
}
pub struct Owner {
    state: Option<Box<State>>,
}
impl Owner {
    #[allow(clippy::too_many_arguments)] // The independent compiled ABI has explicit shape, policy and geometry.
    pub fn new(
        runtime: fusion_pcu_cuda::CudaRuntime,
        image: &[u8],
        bytes: usize,
        extent: u32,
        invocations: u32,
        clamp: bool,
        dead: bool,
        grid: [u32; 3],
        block: [u32; 3],
    ) -> Self {
        let program = Program::new(image);
        let state = State {
            input: Endpoint::new(&program, bytes),
            seed: Endpoint::new(&program, bytes),
            stage: Some(Endpoint::new(&program, bytes)),
            output: Endpoint::new(&program, bytes),
            status: Endpoint::new(&program, 16),
            fresh_output: None,
            grid,
            block,
            extent,
            invocations,
            clamp: u32::from(clamp),
            dead: u32::from(dead),
            terminal: true,
            #[cfg(feature = "allocation-census")]
            api: std::rc::Rc::new(std::cell::Cell::new(Api::default())),
            _runtime: runtime,
            program,
        };
        Self {
            state: Some(Box::new(state)),
        }
    }
    fn retire(&mut self) -> ! {
        // CUDA did not establish terminal state: every pinned/device/code/event/context root survives.
        std::mem::forget(self.state.take().unwrap());
        panic!("independent arithmetic owner retained after uncertain SDK completion");
    }
    pub fn call(
        &mut self,
        input: &[u8],
        seed: &[u8],
        uniform: &[u8],
        output: &mut [u8],
    ) -> Result<Option<(u32, u32)>, ()> {
        self.call_inner::<false, false>(input, seed, uniform, output)
    }
    #[allow(dead_code)] // Shared by sibling producer benches; only the physical-work witness uses this variant.
    pub fn call_fresh_output(
        &mut self,
        input: &[u8],
        seed: &[u8],
        uniform: &[u8],
        output: &mut [u8],
    ) -> Result<Option<(u32, u32)>, ()> {
        self.call_inner::<true, false>(input, seed, uniform, output)
    }
    #[allow(dead_code)] // Shared by sibling producer benches; only the physical-work witness uses this variant.
    pub fn call_split_completion(
        &mut self,
        input: &[u8],
        seed: &[u8],
        uniform: &[u8],
        output: &mut [u8],
    ) -> Result<Option<(u32, u32)>, ()> {
        self.call_inner::<false, true>(input, seed, uniform, output)
    }
    #[allow(dead_code)] // Shared by sibling producer benches; only the physical-work witness uses this variant.
    pub fn call_fresh_split_completion(
        &mut self,
        input: &[u8],
        seed: &[u8],
        uniform: &[u8],
        output: &mut [u8],
    ) -> Result<Option<(u32, u32)>, ()> {
        self.call_inner::<true, true>(input, seed, uniform, output)
    }
    /// Synchronously publish useful prefixes only after terminal status validation.
    #[allow(clippy::too_many_lines)] // Enqueue failures and whole-owner retention stay visible together.
    fn call_inner<const FRESH_OUTPUT: bool, const SPLIT_COMPLETION: bool>(
        &mut self,
        input: &[u8],
        seed: &[u8],
        uniform: &[u8],
        output: &mut [u8],
    ) -> Result<Option<(u32, u32)>, ()> {
        let s = self.state.as_mut().unwrap();
        if input.len() < s.input.device.bytes
            || seed.len() != s.seed.device.bytes
            || output.len() < s.output.device.bytes
            || uniform.len() != s.stage.as_ref().unwrap().device.bytes
        {
            return Err(());
        }
        if FRESH_OUTPUT {
            assert!(s.terminal && s.fresh_output.is_none());
            s.fresh_output = Some(DeviceOwner::new(&s.program, s.output.device.bytes));
            #[cfg(feature = "allocation-census")]
            {
                let mut a = s.api.get();
                a.allocations += 1;
                s.api.set(a);
            }
        }
        s.input
            .host
            .bytes_mut()
            .copy_from_slice(&input[..s.input.device.bytes]);
        s.seed.host.bytes_mut().copy_from_slice(seed);
        s.stage
            .as_mut()
            .unwrap()
            .host
            .bytes_mut()
            .copy_from_slice(uniform);
        s.terminal = false;
        let mut failed = false;
        for e in [&s.input, &s.seed, s.stage.as_ref().unwrap()] {
            #[cfg(feature = "allocation-census")]
            {
                let mut a = s.api.get();
                a.uploads += 1;
                a.uploaded_bytes += u64::try_from(e.device.bytes).unwrap();
                s.api.set(a);
            }
            // SAFETY: exact SDK-pinned initialized endpoint and selected device allocation survive terminal wait.
            if unsafe {
                (s.program.upload)(
                    e.device.pointer,
                    e.host.pointer.cast_const(),
                    e.device.bytes,
                    s.program.stream,
                )
            } != 0
            {
                failed = true;
                break;
            }
        }
        if failed {
            self.retire();
        }
        let s = self.state.as_mut().unwrap();
        #[cfg(feature = "allocation-census")]
        {
            let mut a = s.api.get();
            a.resets += 1;
            s.api.set(a);
        }
        // SAFETY: all retained per-phase status bytes reset before this stream's kernel observes them.
        if unsafe {
            (s.program.reset)(
                s.status.device.pointer,
                255,
                s.status.device.bytes,
                s.program.stream,
            )
        } != 0
        {
            self.retire();
        }
        let s = self.state.as_mut().unwrap();
        let mut pointers = [
            s.input.device.pointer,
            s.seed.device.pointer,
            s.stage.as_ref().map_or(0, |e| e.device.pointer),
            s.fresh_output.as_ref().unwrap_or(&s.output.device).pointer,
            s.status.device.pointer,
        ];
        let mut scalars = [s.extent, s.invocations, s.clamp, s.dead];
        let mut args = [std::ptr::null_mut(); 9];
        for (i, v) in pointers.iter_mut().enumerate() {
            args[i] = std::ptr::from_mut(v).cast();
        }
        for (i, v) in scalars.iter_mut().enumerate() {
            args[5 + i] = std::ptr::from_mut(v).cast();
        }
        let mut failed = false;
        for kernel in s.program.functions {
            #[cfg(feature = "allocation-census")]
            {
                let mut a = s.api.get();
                a.launches += 1;
                s.api.set(a);
            }
            // SAFETY: independent fixed five-pointer/four-u32 ABI; ordered same-stream phases
            // retain every endpoint and separate status words until the final event.
            let launch = unsafe {
                (s.program.launch)(
                    kernel,
                    s.grid[0],
                    s.grid[1],
                    s.grid[2],
                    s.block[0],
                    s.block[1],
                    s.block[2],
                    0,
                    s.program.stream,
                    args.as_mut_ptr(),
                    std::ptr::null_mut(),
                )
            };
            if launch != 0 {
                failed = true;
                break;
            }
        }
        if failed {
            self.retire();
        }
        let observed = if SPLIT_COMPLETION {
            self.state.as_mut().unwrap().readback::<false, true>()
        } else {
            self.state.as_mut().unwrap().readback::<true, true>()
        };
        if !observed {
            self.retire();
        }
        let s = self.state.as_mut().unwrap();
        s.terminal = true;
        let status = s.status.host.bytes();
        let add = u64::from_le_bytes(status[..8].try_into().unwrap());
        let mul = u64::from_le_bytes(status[8..].try_into().unwrap());
        let word = if add == u64::MAX { mul } else { add };
        let fault = (word != u64::MAX).then(|| {
            (
                u32::try_from(word >> 32).unwrap(),
                u32::try_from(word & u64::from(u32::MAX)).unwrap(),
            )
        });
        let publish = fault.is_none() || s.clamp != 0;
        if SPLIT_COMPLETION && publish {
            s.terminal = false;
            if !s.readback::<true, false>() {
                self.retire();
            }
        }
        let s = self.state.as_mut().unwrap();
        s.terminal = true;
        if publish {
            output[..s.output.device.bytes].copy_from_slice(s.output.host.bytes());
        }
        if FRESH_OUTPUT {
            drop(s.fresh_output.take().unwrap());
            #[cfg(feature = "allocation-census")]
            {
                let mut a = s.api.get();
                a.frees += 1;
                s.api.set(a);
            }
        }
        Ok(fault)
    }
    #[cfg(feature = "allocation-census")]
    pub fn counter(&self) -> std::rc::Rc<std::cell::Cell<Api>> {
        self.state.as_ref().unwrap().api.clone()
    }
}
