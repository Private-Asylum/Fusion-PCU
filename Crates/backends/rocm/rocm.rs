//! Dynamically loaded HIP runtime support for Fusion PCU.
//!
//! The bounded PCU Dispatch path lowers scalar f32 programs to HIP source, compiles code objects,
//! and submits them through module launches. Runtime resources and completion ownership are kept
//! here; unsupported PCU operations are rejected by the lowerer.

use std::{
    cell::Cell,
    ffi::{
        CStr,
        c_char,
        c_int,
        c_void,
    },
    ptr,
    rc::Rc,
    sync::Arc,
};
#[cfg(target_os = "linux")]
use std::fs;

use libloading::Library;
use fusion_pcu::{
    PcuMemoryPoolId,
    PcuMemoryPoolSnapshot,
    PcuMemoryUsage,
};

mod blas;
mod codegen;
mod discovery;
mod dispatch;
mod error;
mod memory;
mod owned_dispatch;
#[cfg(feature = "tensor")]
mod tensor;

pub use blas::{
    Rocblas,
    RocblasError,
};
pub use codegen::compiler::{
    HipCompileError,
    compile_hip_source,
};
pub use dispatch::{
    RocmDispatchBinding,
    RocmDispatchError,
    execute_pcu_dispatch,
};
pub use discovery::RocmDiscovery;
pub use error::*;
pub use codegen::lower::{
    RocmLowerError,
    lower_dispatch_to_hip_rtc_source,
    lower_dispatch_to_hip_source,
};
pub use memory::{
    RocmImportDescriptor,
    RocmMemoryMapping,
    RocmMemoryProvider,
    RocmMemoryResource,
};
pub use owned_dispatch::{
    RocmOwnedCompletion,
    RocmOwnedDispatchBackend,
    RocmOwnedDispatchError,
    RocmPreparedDispatch,
};
pub use codegen::rtc::{
    HipRtcError,
    compile_hip_source_for_device,
};
#[cfg(feature = "tensor")]
pub use tensor::{
    RocmAdmittedTensorFeedbackResources,
    RocmPreparedTensorGraph,
    RocmTensorAssessor,
    RocmTensorError,
    RocmTensorExecutionError,
    RocmTensorFeedbackPrepareError,
    RocmTensorFeedbackReleaseError,
    RocmTensorFeedbackResources,
    RocmTensorInput,
    RocmTensorOutputBank,
    RocmTensorPrewarmReport,
    RocmTensorScratch,
};

type HipResult = c_int;
type HipDevice = c_int;
type HipStream = *mut c_void;
type HipEvent = *mut c_void;

const HIP_SUCCESS: HipResult = 0;
const HIP_EVENT_DEFAULT: c_int = 0;
const HIP_EVENT_DISABLE_TIMING: c_int = 2;
const HIP_MEMCPY_HOST_TO_DEVICE: c_int = 1;
const HIP_MEMCPY_DEVICE_TO_HOST: c_int = 2;
const HIP_MEMCPY_DEVICE_TO_DEVICE: c_int = 3;

type GetDeviceCount = unsafe extern "C" fn(*mut c_int) -> HipResult;
type GetDevice = unsafe extern "C" fn(*mut HipDevice, c_int) -> HipResult;
type SetDevice = unsafe extern "C" fn(HipDevice) -> HipResult;
type GetDeviceName = unsafe extern "C" fn(*mut c_char, c_int, HipDevice) -> HipResult;
type GetDevicePciBusId = unsafe extern "C" fn(*mut c_char, c_int, c_int) -> HipResult;
type DeviceTotalMem = unsafe extern "C" fn(*mut usize, HipDevice) -> HipResult;
type MemGetInfo = unsafe extern "C" fn(*mut usize, *mut usize) -> HipResult;
type GetErrorString = unsafe extern "C" fn(HipResult) -> *const c_char;
type Malloc = unsafe extern "C" fn(*mut *mut c_void, usize) -> HipResult;
type Free = unsafe extern "C" fn(*mut c_void) -> HipResult;
type Memcpy = unsafe extern "C" fn(*mut c_void, *const c_void, usize, c_int) -> HipResult;
type StreamCreate = unsafe extern "C" fn(*mut HipStream) -> HipResult;
type StreamDestroy = unsafe extern "C" fn(HipStream) -> HipResult;
type StreamSynchronize = unsafe extern "C" fn(HipStream) -> HipResult;
type EventCreate = unsafe extern "C" fn(*mut HipEvent, c_int) -> HipResult;
type EventDestroy = unsafe extern "C" fn(HipEvent) -> HipResult;
type EventRecord = unsafe extern "C" fn(HipEvent, HipStream) -> HipResult;
type EventSynchronize = unsafe extern "C" fn(HipEvent) -> HipResult;
type EventElapsedTime = unsafe extern "C" fn(*mut f32, HipEvent, HipEvent) -> HipResult;
type ModuleHandle = *mut c_void;
type KernelHandle = *mut c_void;
type ModuleLoadData = unsafe extern "C" fn(*mut ModuleHandle, *const c_void) -> HipResult;
type ModuleGetFunction =
    unsafe extern "C" fn(*mut KernelHandle, ModuleHandle, *const c_char) -> HipResult;
type ModuleUnload = unsafe extern "C" fn(ModuleHandle) -> HipResult;
type ModuleLaunchKernel = unsafe extern "C" fn(
    KernelHandle,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    HipStream,
    *mut *mut c_void,
    *mut *mut c_void,
) -> HipResult;

/// Dynamically loaded HIP runtime and the selected device.
#[derive(Clone)]
pub struct HipRuntime(Arc<RuntimeInner>);

/// HIP runtime availability and visible-device count, queried without selecting a device.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HipRuntimeProbe {
    pub device_count: u32,
    pub runtime_library: String,
}

struct RuntimeInner {
    library: Arc<Library>,
    device: HipDevice,
    name: String,
}

impl HipRuntime {
    /// Load HIP and report visible devices without selecting one. Zero devices is a valid result.
    ///
    /// # Errors
    ///
    /// Returns an error when a runtime library cannot be loaded, a required symbol is missing,
    /// or HIP reports a failure.
    pub fn probe() -> Result<HipRuntimeProbe, HipError> {
        let candidates = std::env::var_os("HIP_RUNTIME_LIBRARY").map_or_else(
            || vec!["libamdhip64.so".into(), "libamdhip64.so.6".into()],
            |path| vec![path],
        );
        let mut last_error = None;
        for candidate in candidates {
            let path = candidate.to_string_lossy().into_owned();
            // SAFETY: HIP exports the documented C ABI, and the library stays alive through query.
            let library = match unsafe { Library::new(&candidate) } {
                Ok(library) => library,
                Err(error) => {
                    last_error = Some(error.to_string());
                    continue;
                }
            };
            let get_count = unsafe { library.get::<GetDeviceCount>(b"hipGetDeviceCount\0") }
                .map_err(|error| HipError::MissingSymbol {
                    symbol: "hipGetDeviceCount",
                    detail: error.to_string(),
                })?;
            let mut count = 0;
            let status = unsafe { get_count(ptr::from_mut(&mut count)) };
            if status != HIP_SUCCESS {
                return Err(raw_hip_error(&library, "hipGetDeviceCount", status));
            }
            return Ok(HipRuntimeProbe {
                device_count: count.max(0).unsigned_abs(),
                runtime_library: path,
            });
        }
        Err(HipError::RuntimeUnavailable(
            last_error.unwrap_or_else(|| "no runtime library candidates".into()),
        ))
    }

    /// Enumerate visible devices without selecting one, querying only stable HIP identity facts.
    ///
    /// # Errors
    ///
    /// Returns an error when a runtime library cannot be loaded, a required symbol is missing,
    /// or HIP fails to enumerate a device.
    pub fn enumerate_devices() -> Result<Vec<HipDeviceInfo>, HipError> {
        let candidates = std::env::var_os("HIP_RUNTIME_LIBRARY").map_or_else(
            || vec!["libamdhip64.so".into(), "libamdhip64.so.6".into()],
            |path| vec![path],
        );
        let mut last_error = None;
        for candidate in candidates {
            // SAFETY: symbols are used while this library remains in scope.
            let library = match unsafe { Library::new(&candidate) } {
                Ok(library) => library,
                Err(error) => {
                    last_error = Some(error.to_string());
                    continue;
                }
            };
            let get_count = unsafe { library.get::<GetDeviceCount>(b"hipGetDeviceCount\0") }
                .map_err(|error| HipError::MissingSymbol {
                    symbol: "hipGetDeviceCount",
                    detail: error.to_string(),
                })?;
            let get_device =
                unsafe { library.get::<GetDevice>(b"hipDeviceGet\0") }.map_err(|error| {
                    HipError::MissingSymbol {
                        symbol: "hipDeviceGet",
                        detail: error.to_string(),
                    }
                })?;
            let get_name = unsafe { library.get::<GetDeviceName>(b"hipDeviceGetName\0") }.map_err(
                |error| HipError::MissingSymbol {
                    symbol: "hipDeviceGetName",
                    detail: error.to_string(),
                },
            )?;
            let total_mem = unsafe { library.get::<DeviceTotalMem>(b"hipDeviceTotalMem\0") }
                .map_err(|error| HipError::MissingSymbol {
                    symbol: "hipDeviceTotalMem",
                    detail: error.to_string(),
                })?;
            let mut count = 0;
            let status = unsafe { get_count(ptr::from_mut(&mut count)) };
            if status != HIP_SUCCESS {
                return Err(raw_hip_error(&library, "hipGetDeviceCount", status));
            }
            let mut devices = Vec::new();
            for index in 0..count.max(0) {
                let mut device = 0;
                let status = unsafe { get_device(ptr::from_mut(&mut device), index) };
                if status != HIP_SUCCESS {
                    return Err(raw_hip_error(&library, "hipDeviceGet", status));
                }
                let mut name = [0_i8; 256];
                let status = unsafe { get_name(name.as_mut_ptr(), 256, device) };
                if status != HIP_SUCCESS {
                    return Err(raw_hip_error(&library, "hipDeviceGetName", status));
                }
                let name = bounded_device_name(&name);
                let mut total = 0_usize;
                let status = unsafe { total_mem(&raw mut total, device) };
                if status != HIP_SUCCESS {
                    return Err(raw_hip_error(&library, "hipDeviceTotalMem", status));
                }
                let pci_bus_id = query_pci_bus_id(&library, index);
                devices.push(HipDeviceInfo {
                    index,
                    name,
                    vendor: "AMD".into(),
                    architecture: pci_bus_id.as_deref().and_then(query_architecture),
                    generation: None,
                    pci_bus_id,
                    total_memory: total as u64,
                });
            }
            return Ok(devices);
        }
        Err(HipError::RuntimeUnavailable(
            last_error.unwrap_or_else(|| "no runtime library candidates".into()),
        ))
    }

    /// Load `libamdhip64.so` (or `HIP_RUNTIME_LIBRARY`) and select `device_index`.
    ///
    /// # Errors
    ///
    /// Returns an error when HIP cannot load, enumerate, or select the requested device.
    pub fn new(device_index: u32) -> Result<Self, HipError> {
        let candidates = std::env::var_os("HIP_RUNTIME_LIBRARY").map_or_else(
            || vec!["libamdhip64.so".into(), "libamdhip64.so.6".into()],
            |path| vec![path],
        );
        let mut last_error = None;
        for candidate in candidates {
            // SAFETY: HIP runtime exports C ABI symbols; retaining the library in RuntimeInner
            // keeps all loaded function pointers valid for the lifetime of every resource.
            let library = match unsafe { Library::new(&candidate) } {
                Ok(library) => library,
                Err(error) => {
                    last_error = Some(error.to_string());
                    continue;
                }
            };
            // Query availability before selecting a device. hipSetDevice(0) can fail when no GPU
            // is visible, hiding the useful zero-device result.
            let get_count = unsafe { library.get::<GetDeviceCount>(b"hipGetDeviceCount\0") }
                .map_err(|error| HipError::MissingSymbol {
                    symbol: "hipGetDeviceCount",
                    detail: error.to_string(),
                })?;
            let mut count: c_int = 0;
            let status = unsafe { get_count(ptr::from_mut(&mut count)) };
            let runtime = Self(Arc::new(RuntimeInner {
                library: Arc::new(library),
                device: 0,
                name: String::new(),
            }));
            if status != HIP_SUCCESS {
                return Err(runtime.error("hipGetDeviceCount", status));
            }
            let count = count.max(0).unsigned_abs();
            if device_index >= count {
                return Err(HipError::DeviceIndexOutOfRange {
                    requested: device_index,
                    count,
                });
            }
            let device =
                runtime.hip_get_device(c_int::try_from(device_index).map_err(|_| {
                    HipError::DeviceIndexOutOfRange {
                        requested: device_index,
                        count,
                    }
                })?)?;
            runtime.hip_set_device(device)?;
            let name = runtime.device_name(device)?;
            return Ok(Self(Arc::new(RuntimeInner {
                library: runtime.0.library.clone(),
                device,
                name,
            })));
        }
        Err(HipError::RuntimeUnavailable(
            last_error.unwrap_or_else(|| "no runtime library candidates".into()),
        ))
    }

    /// Return visible HIP device count.
    ///
    /// # Errors
    ///
    /// Returns an error when HIP cannot query the visible device count.
    pub fn device_count(&self) -> Result<u32, HipError> {
        let mut count = 0;
        self.call("hipGetDeviceCount", |f: GetDeviceCount| unsafe {
            f(&raw mut count)
        })?;
        Ok(count.max(0).unsigned_abs())
    }

    /// Return the selected device's index, name, and physical memory capacity.
    ///
    /// # Errors
    ///
    /// Returns an error when HIP cannot query device memory information.
    pub fn device_info(&self) -> Result<HipDeviceInfo, HipError> {
        let mut total = 0_usize;
        self.call("hipDeviceTotalMem", |f: DeviceTotalMem| unsafe {
            f(&raw mut total, self.0.device)
        })?;
        let pci_bus_id = query_pci_bus_id(&self.0.library, self.0.device);
        Ok(HipDeviceInfo {
            index: self.0.device,
            name: self.0.name.clone(),
            vendor: "AMD".into(),
            architecture: pci_bus_id.as_deref().and_then(query_architecture),
            generation: None,
            pci_bus_id,
            total_memory: total as u64,
        })
    }

    /// Query the device's current free and total physical memory in bytes.
    ///
    /// This is system-wide free capacity at a point in time, not an allocation reservation or
    /// process-attributable usage. Admission can race with other GPU clients.
    ///
    /// # Errors
    ///
    /// Returns an error when HIP cannot query device memory information.
    pub fn memory_info(&self) -> Result<HipMemoryInfo, HipError> {
        let mut free = 0_usize;
        let mut total = 0_usize;
        self.call("hipMemGetInfo", |f: MemGetInfo| unsafe {
            f(&raw mut free, &raw mut total)
        })?;
        Ok(HipMemoryInfo {
            free_bytes: free as u64,
            total_bytes: total as u64,
        })
    }

    /// Query a point-in-time snapshot of this HIP device's physical memory pool.
    ///
    /// `pool` is assigned by the caller so it can remain unique across provider types. HIP's
    /// free/total query describes device-wide availability; it does not attribute memory use to
    /// this process, so process usage is reported as unknown. The result is telemetry only and
    /// does not reserve memory or prevent another client from allocating concurrently.
    ///
    /// # Errors
    ///
    /// Returns an error when the underlying HIP memory query fails.
    pub fn memory_pool_snapshot(
        &self,
        pool: PcuMemoryPoolId,
    ) -> Result<PcuMemoryPoolSnapshot, HipError> {
        self.memory_info()
            .map(|info| hip_memory_pool_snapshot(pool, info))
    }

    /// Bind a caller-assigned abstract pool identity to this selected HIP device.
    #[must_use]
    pub fn memory_provider(&self, pool: PcuMemoryPoolId) -> RocmMemoryProvider {
        RocmMemoryProvider::new(self.clone(), pool)
    }

    /// Allocate device memory. The allocation is released when the returned buffer is dropped.
    ///
    /// # Errors
    ///
    /// Returns the HIP allocation error if the device cannot allocate the requested size.
    pub fn allocate(&self, bytes: usize) -> Result<DeviceBuffer, HipError> {
        let mut pointer = ptr::null_mut();
        self.call("hipMalloc", |f: Malloc| unsafe {
            f(&raw mut pointer, bytes)
        })?;
        Ok(DeviceBuffer {
            allocation: Rc::new(DeviceAllocation {
                runtime: self.clone(),
                pointer,
                bytes,
                access: Rc::new(AllocationAccess {
                    state: Cell::new(AllocationAccessState::Idle),
                }),
            }),
        })
    }

    /// Load a HIP code object (HSACO) into this device context.
    ///
    /// # Errors
    ///
    /// Returns an error when HIP cannot load the module.
    pub fn load_module(&self, image: &[u8]) -> Result<HipModule, HipError> {
        let mut raw = ptr::null_mut();
        self.call("hipModuleLoadData", |f: ModuleLoadData| unsafe {
            f(&raw mut raw, image.as_ptr().cast())
        })?;
        Ok(HipModule {
            inner: Rc::new(ModuleInner {
                runtime: self.clone(),
                raw,
            }),
        })
    }

    /// Create a stream for ordered asynchronous operations.
    ///
    /// The initial crate surface does not enqueue work onto this stream; it is exposed so later
    /// submission support can build on a real HIP stream without changing the resource type.
    ///
    /// # Errors
    ///
    /// Returns an error when HIP cannot create the stream.
    pub fn create_stream(&self) -> Result<HipStreamHandle, HipError> {
        let mut stream = ptr::null_mut();
        self.call("hipStreamCreate", |f: StreamCreate| unsafe {
            f(&raw mut stream)
        })?;
        Ok(HipStreamHandle {
            inner: Rc::new(StreamInner {
                runtime: self.clone(),
                raw: stream,
            }),
        })
    }

    /// Create a timing-disabled event.
    ///
    /// Events can be recorded on streams and synchronized, but are not yet linked to submissions.
    ///
    /// # Errors
    ///
    /// Returns an error when HIP cannot create the event.
    pub fn create_event(&self) -> Result<HipEventHandle, HipError> {
        let mut event = ptr::null_mut();
        self.call("hipEventCreateWithFlags", |f: EventCreate| unsafe {
            f(&raw mut event, HIP_EVENT_DISABLE_TIMING)
        })?;
        Ok(HipEventHandle {
            inner: Rc::new(EventInner {
                runtime: self.clone(),
                raw: event,
            }),
        })
    }

    /// Create an event with timing enabled for device elapsed-time measurements.
    ///
    /// Timing events are a separate type so timing-disabled completion events keep their
    /// existing behavior and cannot accidentally be used for elapsed-time queries.
    ///
    /// # Errors
    ///
    /// Returns an error when HIP cannot create the event.
    pub fn create_timing_event(&self) -> Result<HipTimingEventHandle, HipError> {
        let mut event = ptr::null_mut();
        self.call("hipEventCreateWithFlags", |f: EventCreate| unsafe {
            f(&raw mut event, HIP_EVENT_DEFAULT)
        })?;
        Ok(HipTimingEventHandle {
            inner: Rc::new(EventInner {
                runtime: self.clone(),
                raw: event,
            }),
        })
    }

    /// Return device elapsed time between two completed timing events, in milliseconds.
    ///
    /// Both events are synchronized before querying elapsed time. If HIP cannot confirm either
    /// event completed, this returns that error and does not report a possibly invalid duration.
    ///
    /// # Errors
    ///
    /// Returns an error if events belong to different runtime/device pairs, either event cannot
    /// be confirmed complete, or HIP rejects the elapsed-time query.
    pub fn elapsed_time_ms(
        &self,
        start: &HipTimingEventHandle,
        end: &HipTimingEventHandle,
    ) -> Result<f32, HipError> {
        self.ensure_same_runtime(&start.inner.runtime)?;
        self.ensure_same_runtime(&end.inner.runtime)?;
        start.synchronize()?;
        end.synchronize()?;
        let mut milliseconds = 0.0_f32;
        self.call("hipEventElapsedTime", |f: EventElapsedTime| unsafe {
            f(&raw mut milliseconds, start.inner.raw, end.inner.raw)
        })?;
        Ok(milliseconds)
    }

    fn device_name(&self, device: HipDevice) -> Result<String, HipError> {
        let mut name = [0_i8; 256];
        self.call("hipDeviceGetName", |f: GetDeviceName| unsafe {
            f(
                name.as_mut_ptr(),
                c_int::try_from(name.len()).expect("fixed name buffer fits c_int"),
                device,
            )
        })?;
        Ok(bounded_device_name(&name))
    }

    fn hip_get_device(&self, index: c_int) -> Result<HipDevice, HipError> {
        let mut device = 0;
        self.call("hipDeviceGet", |f: GetDevice| unsafe {
            f(ptr::from_mut(&mut device), index)
        })?;
        Ok(device)
    }

    fn hip_set_device(&self, device: HipDevice) -> Result<(), HipError> {
        self.call("hipSetDevice", |f: SetDevice| unsafe { f(device) })
    }

    fn call<T: Copy>(
        &self,
        symbol: &'static str,
        invoke: impl FnOnce(T) -> HipResult,
    ) -> Result<(), HipError> {
        if symbol != "hipSetDevice" {
            // HIP's current device is thread-local, so select this runtime's device before each
            // operation. This keeps cloned handles valid when used from another host thread.
            let setter =
                unsafe { self.0.library.get::<SetDevice>(b"hipSetDevice\0") }.map_err(|error| {
                    HipError::MissingSymbol {
                        symbol: "hipSetDevice",
                        detail: error.to_string(),
                    }
                })?;
            let status = unsafe { setter(self.0.device) };
            if status != HIP_SUCCESS {
                return Err(self.error("hipSetDevice", status));
            }
        }
        // libloading allocates a CString when the supplied symbol lacks a trailing NUL. All
        // ordinary HIP symbols fit in this stack buffer; retain an overflow path for future ABI
        // names rather than making symbol length an undocumented runtime limit.
        let mut inline_symbol = [0_u8; 64];
        let mut overflow_symbol = Vec::new();
        let symbol_bytes = if symbol.len() < inline_symbol.len() {
            inline_symbol[..symbol.len()].copy_from_slice(symbol.as_bytes());
            &inline_symbol[..=symbol.len()]
        } else {
            overflow_symbol.extend_from_slice(symbol.as_bytes());
            overflow_symbol.push(0);
            &overflow_symbol
        };
        // SAFETY: `symbol` is loaded from the retained HIP runtime and `T` matches the named C ABI.
        let function = unsafe { self.0.library.get::<T>(symbol_bytes) }.map_err(|error| {
            HipError::MissingSymbol {
                symbol,
                detail: error.to_string(),
            }
        })?;
        let status = invoke(*function);
        if status == HIP_SUCCESS {
            Ok(())
        } else {
            Err(self.error(symbol, status))
        }
    }

    fn error(&self, operation: &'static str, code: HipResult) -> HipError {
        // Error-string lookup is optional; preserve numeric status even if the symbol is absent.
        let detail = unsafe { self.0.library.get::<GetErrorString>(b"hipGetErrorString\0") }
            .ok()
            .map(|f| unsafe { f(code) })
            .filter(|p| !p.is_null())
            .map(|p| unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned());
        HipError::Runtime {
            operation,
            code,
            detail,
        }
    }

    fn ensure_same_runtime(&self, other: &Self) -> Result<(), HipError> {
        if Arc::ptr_eq(&self.0.library, &other.0.library) && self.0.device == other.0.device {
            Ok(())
        } else {
            Err(HipError::DifferentRuntime)
        }
    }
}

/// Basic physical-device facts queried without depending on versioned HIP property layouts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HipDeviceInfo {
    pub index: i32,
    pub name: String,
    pub vendor: String,
    /// Base `gfx` target matched from HIP's PCI identity to KFD topology on Linux when exposed.
    /// KFD's numeric target does not report optional target feature suffixes.
    pub architecture: Option<String>,
    /// GPU generation is not inferred from the marketing name.
    pub generation: Option<String>,
    /// PCI bus ID from the documented HIP `hipDeviceGetPCIBusId` API, when supported.
    /// HIP exposes `hipDeviceGetUuid`, but the HIP header marks it Beta, so stable discovery omits
    /// it pending a stable HIP or ROCr/HSA identity API.
    pub pci_bus_id: Option<String>,
    pub total_memory: u64,
}

/// Query PCI location through HIP's standalone C API, which avoids `hipDeviceProp_t` ABI layout.
/// Older runtimes may not export this symbol; discovery remains useful without the location.
fn query_pci_bus_id(library: &Library, ordinal: c_int) -> Option<String> {
    let function = unsafe { library.get::<GetDevicePciBusId>(b"hipDeviceGetPCIBusId\0") }.ok()?;
    let mut buffer = [0_i8; 64];
    let capacity = c_int::try_from(buffer.len()).expect("fixed PCI bus buffer fits c_int");
    let status = unsafe { function(buffer.as_mut_ptr(), capacity, ordinal) };
    if status != HIP_SUCCESS {
        return None;
    }
    let value = bounded_device_name(&buffer);
    (!value.is_empty()).then_some(value)
}

/// Resolve the base AMD target reported by Linux KFD for the HIP device's PCI function.
/// HIP keeps `gcnArchName` in an ABI-versioned properties struct, while KFD exposes the same
/// target as `gfx_target_version`. Matching the DRM render node to HIP's PCI ID avoids relying on
/// the changing HIP struct layout or guessing an ISA from a marketing name.
#[cfg(target_os = "linux")]
fn query_architecture(pci_bus_id: &str) -> Option<String> {
    let expected = canonical_pci_bus_id(pci_bus_id);
    let nodes = fs::read_dir("/sys/class/kfd/kfd/topology/nodes").ok()?;
    for node in nodes.flatten() {
        let Ok(properties) = fs::read_to_string(node.path().join("properties")) else {
            continue;
        };
        let mut render_minor = None;
        let mut target_version = None;
        for line in properties.lines() {
            let Some((key, value)) = line.split_once(char::is_whitespace) else {
                continue;
            };
            match key {
                "drm_render_minor" => render_minor = value.trim().parse::<u32>().ok(),
                "gfx_target_version" => target_version = value.trim().parse::<u32>().ok(),
                _ => {}
            }
        }
        let (Some(render_minor), Some(target_version)) = (render_minor, target_version) else {
            continue;
        };
        if target_version == 0 {
            continue;
        }
        let render_device = format!("/sys/class/drm/renderD{render_minor}/device");
        let Ok(device_path) = fs::canonicalize(render_device) else {
            continue;
        };
        let Some(actual) = device_path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        if canonical_pci_bus_id(actual) == expected {
            return format_gfx_target(target_version);
        }
    }
    None
}

#[cfg(not(target_os = "linux"))]
fn query_architecture(_pci_bus_id: &str) -> Option<String> {
    // No portable, safely versioned HIP property query is wired into this backend yet.
    None
}

#[cfg(target_os = "linux")]
fn canonical_pci_bus_id(value: &str) -> String {
    value.trim().to_ascii_lowercase()
}

#[cfg(any(target_os = "linux", test))]
fn format_gfx_target(target_version: u32) -> Option<String> {
    let major = target_version / 10_000;
    let minor = (target_version / 100) % 100;
    let stepping = target_version % 100;
    if major == 0 || major > 99 || minor > 9 || stepping > 15 {
        return None;
    }
    let stepping = char::from_digit(stepping, 16)?;
    Some(format!("gfx{major}{minor}{stepping}"))
}

fn bounded_device_name(buffer: &[c_char]) -> String {
    let bytes: Vec<u8> = buffer
        .iter()
        .take_while(|&&byte| byte != 0)
        .map(|&byte| byte.to_ne_bytes()[0])
        .collect();
    String::from_utf8_lossy(&bytes).into_owned()
}

fn raw_hip_error(library: &Library, operation: &'static str, code: HipResult) -> HipError {
    let detail = unsafe { library.get::<GetErrorString>(b"hipGetErrorString\0") }
        .ok()
        .map(|function| unsafe { function(code) })
        .filter(|pointer| !pointer.is_null())
        .map(|pointer| {
            unsafe { CStr::from_ptr(pointer) }
                .to_string_lossy()
                .into_owned()
        });
    HipError::Runtime {
        operation,
        code,
        detail,
    }
}

/// Point-in-time device memory telemetry from HIP.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HipMemoryInfo {
    pub free_bytes: u64,
    pub total_bytes: u64,
}

fn hip_memory_pool_snapshot(id: PcuMemoryPoolId, info: HipMemoryInfo) -> PcuMemoryPoolSnapshot {
    let capacity_bytes = (info.total_bytes != 0).then_some(info.total_bytes);
    let system_used_bytes = match (
        capacity_bytes,
        info.total_bytes.checked_sub(info.free_bytes),
    ) {
        (Some(_), Some(used)) => PcuMemoryUsage::Known(used),
        _ => PcuMemoryUsage::Unknown,
    };
    PcuMemoryPoolSnapshot {
        id,
        capacity_bytes,
        system_used_bytes,
        process_used_bytes: PcuMemoryUsage::Unknown,
        system_ledger_reserved_bytes: 0,
        process_ledger_reserved_bytes: 0,
    }
}

/// Shared owner for one HIP device allocation.
struct DeviceAllocation {
    runtime: HipRuntime,
    pointer: *mut c_void,
    bytes: usize,
    access: Rc<AllocationAccess>,
}

/// Shared single-operation gate consulted through every cloned device-buffer handle.
struct AllocationAccess {
    state: Cell<AllocationAccessState>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AllocationAccessState {
    Idle,
    Exclusive,
    Stream { identity: usize, leases: usize },
    Poisoned,
}

#[derive(Clone, Copy)]
enum AllocationAccessKind {
    Exclusive,
    Stream(usize),
}

impl AllocationAccess {
    fn acquire(self: &Rc<Self>) -> Result<AllocationAccessGuard, ()> {
        if self.state.get() != AllocationAccessState::Idle {
            return Err(());
        }
        self.state.set(AllocationAccessState::Exclusive);
        Ok(AllocationAccessGuard {
            access: Rc::clone(self),
            kind: AllocationAccessKind::Exclusive,
        })
    }

    fn acquire_stream(self: &Rc<Self>, identity: usize) -> Result<AllocationAccessGuard, ()> {
        match self.state.get() {
            AllocationAccessState::Idle => {
                self.state.set(AllocationAccessState::Stream {
                    identity,
                    leases: 1,
                });
            }
            AllocationAccessState::Stream {
                identity: active,
                leases,
            } if active == identity => {
                let leases = leases.checked_add(1).ok_or(())?;
                self.state
                    .set(AllocationAccessState::Stream { identity, leases });
            }
            AllocationAccessState::Exclusive
            | AllocationAccessState::Stream { .. }
            | AllocationAccessState::Poisoned => {
                return Err(());
            }
        }
        Ok(AllocationAccessGuard {
            access: Rc::clone(self),
            kind: AllocationAccessKind::Stream(identity),
        })
    }
}

struct AllocationAccessGuard {
    access: Rc<AllocationAccess>,
    kind: AllocationAccessKind,
}

impl Drop for AllocationAccessGuard {
    fn drop(&mut self) {
        let next = match (self.kind, self.access.state.get()) {
            (AllocationAccessKind::Exclusive, AllocationAccessState::Exclusive) => {
                AllocationAccessState::Idle
            }
            (
                AllocationAccessKind::Stream(identity),
                AllocationAccessState::Stream {
                    identity: active,
                    leases,
                },
            ) if identity == active && leases > 1 => AllocationAccessState::Stream {
                identity,
                leases: leases - 1,
            },
            (
                AllocationAccessKind::Stream(identity),
                AllocationAccessState::Stream {
                    identity: active,
                    leases: 1,
                },
            ) if identity == active => AllocationAccessState::Idle,
            // A gate is never reset after a mismatched or forgotten lease. Staying busy is the
            // safe failure mode if internal ownership invariants are ever violated.
            _ => return,
        };
        self.access.state.set(next);
    }
}

impl AllocationAccessGuard {
    fn quarantine_stream(&self) {
        if let AllocationAccessKind::Stream(identity) = self.kind
            && matches!(
                self.access.state.get(),
                AllocationAccessState::Stream {
                    identity: active,
                    ..
                } if active == identity
            )
        {
            self.access.state.set(AllocationAccessState::Poisoned);
        }
    }
}

/// A live access lease pins both the native allocation and its shared busy gate.
struct DeviceAccessLease {
    allocation: Rc<DeviceAllocation>,
    guard: AllocationAccessGuard,
}
impl Drop for DeviceAllocation {
    fn drop(&mut self) {
        let _ = self
            .runtime
            .call("hipFree", |f: Free| unsafe { f(self.pointer) });
    }
}

/// Owned HIP device-memory allocation. Clones share the same allocation.
#[derive(Clone)]
pub struct DeviceBuffer {
    allocation: Rc<DeviceAllocation>,
}
impl DeviceBuffer {
    #[must_use]
    pub fn len(&self) -> usize {
        self.allocation.bytes
    }
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.allocation.bytes == 0
    }
    /// Copy bytes from host memory into this allocation.
    ///
    /// # Errors
    ///
    /// Returns an error when the range is invalid, another operation holds the allocation, or HIP
    /// reports a copy or completion failure.
    pub fn copy_from(&mut self, source: &[u8]) -> Result<(), HipError> {
        self.copy_from_at(0, source)
    }
    /// Copy bytes from host memory into a checked byte range of this allocation.
    ///
    /// # Errors
    ///
    /// Returns an error when the range is invalid, another operation holds the allocation, or HIP
    /// reports a copy or completion failure.
    pub fn copy_from_at(&mut self, offset: usize, source: &[u8]) -> Result<(), HipError> {
        self.check_range(offset, source.len())?;
        if source.is_empty() {
            return Ok(());
        }
        let lease = self.acquire_access()?;
        let result = self
            .allocation
            .runtime
            .call("hipMemcpy", |f: Memcpy| unsafe {
                f(
                    (self.allocation.pointer.cast::<u8>().wrapping_add(offset)).cast(),
                    source.as_ptr().cast(),
                    source.len(),
                    HIP_MEMCPY_HOST_TO_DEVICE,
                )
            });
        lease.finish_synchronous(result)
    }
    /// Copy this allocation into host memory.
    ///
    /// # Errors
    ///
    /// Returns an error when the range is invalid, another operation holds the allocation, or HIP
    /// reports a copy or completion failure.
    pub fn copy_to(&self, destination: &mut [u8]) -> Result<(), HipError> {
        self.copy_to_at(0, destination)
    }
    /// Copy a checked byte range of this allocation into host memory.
    ///
    /// # Errors
    ///
    /// Returns an error when the range is invalid, another operation holds the allocation, or HIP
    /// reports a copy or completion failure.
    pub fn copy_to_at(&self, offset: usize, destination: &mut [u8]) -> Result<(), HipError> {
        self.check_range(offset, destination.len())?;
        if destination.is_empty() {
            return Ok(());
        }
        let lease = self.acquire_access()?;
        let result = self
            .allocation
            .runtime
            .call("hipMemcpy", |f: Memcpy| unsafe {
                f(
                    destination.as_mut_ptr().cast(),
                    (self.allocation.pointer.cast::<u8>().wrapping_add(offset)).cast(),
                    destination.len(),
                    HIP_MEMCPY_DEVICE_TO_HOST,
                )
            });
        lease.finish_synchronous(result)
    }
    /// Copy bytes from another device allocation.
    /// Copy bytes from a different allocation belonging to the same HIP runtime and device.
    ///
    /// # Errors
    ///
    /// Returns an error when the runtimes differ, either buffer is too small or busy, or HIP
    /// reports a copy or completion failure.
    pub fn copy_from_device(&mut self, source: &Self, bytes: usize) -> Result<(), HipError> {
        self.allocation
            .runtime
            .ensure_same_runtime(&source.allocation.runtime)?;
        self.check_length(bytes)?;
        source.check_length(bytes)?;
        if Rc::ptr_eq(&self.allocation, &source.allocation) {
            return Err(HipError::Busy);
        }
        let destination_lease = self.acquire_access()?;
        let source_lease = source.acquire_access()?;
        let result = self
            .allocation
            .runtime
            .call("hipMemcpy", |f: Memcpy| unsafe {
                f(
                    self.allocation.pointer,
                    source.allocation.pointer,
                    bytes,
                    HIP_MEMCPY_DEVICE_TO_DEVICE,
                )
            });
        destination_lease.finish_synchronous_with(source_lease, result)
    }
    fn check_length(&self, bytes: usize) -> Result<(), HipError> {
        if bytes > self.allocation.bytes {
            Err(HipError::BufferTooSmall {
                allocation: self.allocation.bytes,
                requested: bytes,
            })
        } else {
            Ok(())
        }
    }

    fn check_range(&self, offset: usize, bytes: usize) -> Result<(), HipError> {
        if offset
            .checked_add(bytes)
            .is_none_or(|end| end > self.allocation.bytes)
        {
            Err(HipError::BufferTooSmall {
                allocation: self.allocation.bytes.saturating_sub(offset),
                requested: bytes,
            })
        } else {
            Ok(())
        }
    }

    fn acquire_access(&self) -> Result<DeviceAccessLease, HipError> {
        let guard = self
            .allocation
            .access
            .acquire()
            .map_err(|()| HipError::Busy)?;
        Ok(DeviceAccessLease {
            allocation: Rc::clone(&self.allocation),
            guard,
        })
    }

    fn acquire_stream_access(
        &self,
        stream: &HipStreamHandle,
    ) -> Result<DeviceAccessLease, HipError> {
        let identity = Rc::as_ptr(&stream.inner) as usize;
        let guard = self
            .allocation
            .access
            .acquire_stream(identity)
            .map_err(|()| HipError::Busy)?;
        Ok(DeviceAccessLease {
            allocation: Rc::clone(&self.allocation),
            guard,
        })
    }
}

impl DeviceAccessLease {
    fn quarantine(&self) {
        self.guard.quarantine_stream();
    }

    fn finish_synchronous(self, result: Result<(), HipError>) -> Result<(), HipError> {
        match result {
            Ok(()) => Ok(()),
            Err(error) => {
                if self
                    .allocation
                    .runtime
                    .call(
                        "hipDeviceSynchronize",
                        |f: unsafe extern "C" fn() -> c_int| unsafe { f() },
                    )
                    .is_err()
                {
                    std::mem::forget(self);
                }
                Err(error)
            }
        }
    }

    fn finish_synchronous_with(
        self,
        other: Self,
        result: Result<(), HipError>,
    ) -> Result<(), HipError> {
        match result {
            Ok(()) => Ok(()),
            Err(error) => {
                if self
                    .allocation
                    .runtime
                    .call(
                        "hipDeviceSynchronize",
                        |f: unsafe extern "C" fn() -> c_int| unsafe { f() },
                    )
                    .is_err()
                {
                    std::mem::forget(self);
                    std::mem::forget(other);
                }
                Err(error)
            }
        }
    }
}

struct StreamInner {
    runtime: HipRuntime,
    raw: HipStream,
}
impl Drop for StreamInner {
    fn drop(&mut self) {
        let _ = self
            .runtime
            .call("hipStreamDestroy", |f: StreamDestroy| unsafe {
                f(self.raw)
            });
    }
}
/// Shared owner for a HIP stream.
#[derive(Clone)]
pub struct HipStreamHandle {
    inner: Rc<StreamInner>,
}
impl HipStreamHandle {
    /// Return the raw stream pointer for internal backend FFI integration.
    #[cfg(feature = "tensor")]
    pub(crate) fn raw_stream(&self) -> *mut c_void {
        self.inner.raw
    }

    /// Check whether this stream belongs to the supplied runtime/device pair.
    #[cfg(feature = "tensor")]
    pub(crate) fn belongs_to_runtime(&self, runtime: &HipRuntime) -> bool {
        runtime.ensure_same_runtime(&self.inner.runtime).is_ok()
    }

    /// Wait until all operations queued on this stream have completed.
    ///
    /// # Errors
    ///
    /// Returns the HIP synchronization error, if any.
    pub fn synchronize(&self) -> Result<(), HipError> {
        self.inner
            .runtime
            .call("hipStreamSynchronize", |f: StreamSynchronize| unsafe {
                f(self.inner.raw)
            })
    }
    /// Record an event after all work already queued on this stream.
    ///
    /// # Errors
    ///
    /// Returns an error when the event belongs to another runtime or HIP fails to record it.
    pub fn record(&self, event: &HipEventHandle) -> Result<(), HipError> {
        self.inner
            .runtime
            .ensure_same_runtime(&event.inner.runtime)?;
        self.inner
            .runtime
            .call("hipEventRecord", |f: EventRecord| unsafe {
                f(event.inner.raw, self.inner.raw)
            })
    }

    /// Record a timing-enabled event after all work already queued on this stream.
    ///
    /// # Errors
    ///
    /// Returns an error when the event belongs to another runtime or HIP fails to record it.
    pub fn record_timing(&self, event: &HipTimingEventHandle) -> Result<(), HipError> {
        self.inner
            .runtime
            .ensure_same_runtime(&event.inner.runtime)?;
        self.inner
            .runtime
            .call("hipEventRecord", |f: EventRecord| unsafe {
                f(event.inner.raw, self.inner.raw)
            })
    }
}

struct EventInner {
    runtime: HipRuntime,
    raw: HipEvent,
}
impl Drop for EventInner {
    fn drop(&mut self) {
        let _ = self
            .runtime
            .call("hipEventDestroy", |f: EventDestroy| unsafe { f(self.raw) });
    }
}
/// Shared owner for a HIP event.
#[derive(Clone)]
pub struct HipEventHandle {
    inner: Rc<EventInner>,
}
impl HipEventHandle {
    /// Wait until the event has completed.
    ///
    /// # Errors
    ///
    /// Returns the HIP synchronization error, if any.
    pub fn synchronize(&self) -> Result<(), HipError> {
        self.inner
            .runtime
            .call("hipEventSynchronize", |f: EventSynchronize| unsafe {
                f(self.inner.raw)
            })
    }
}

/// Shared owner for a HIP event created with timing enabled.
#[derive(Clone)]
pub struct HipTimingEventHandle {
    inner: Rc<EventInner>,
}
impl HipTimingEventHandle {
    /// Wait until the event has completed.
    ///
    /// # Errors
    ///
    /// Returns the HIP synchronization error, if any.
    pub fn synchronize(&self) -> Result<(), HipError> {
        self.inner
            .runtime
            .call("hipEventSynchronize", |f: EventSynchronize| unsafe {
                f(self.inner.raw)
            })
    }
}

struct ModuleInner {
    runtime: HipRuntime,
    raw: ModuleHandle,
}
impl Drop for ModuleInner {
    fn drop(&mut self) {
        let _ = self
            .runtime
            .call("hipModuleUnload", |f: ModuleUnload| unsafe { f(self.raw) });
    }
}
/// Loaded HIP code object.
#[derive(Clone)]
pub struct HipModule {
    inner: Rc<ModuleInner>,
}
impl HipModule {
    /// Resolve one named kernel function from this module.
    ///
    /// # Errors
    ///
    /// Returns an error when HIP cannot resolve the named function.
    pub fn function(&self, name: &CStr) -> Result<HipKernel, HipError> {
        let mut raw = ptr::null_mut();
        self.inner
            .runtime
            .call("hipModuleGetFunction", |f: ModuleGetFunction| unsafe {
                f(&raw mut raw, self.inner.raw, name.as_ptr())
            })?;
        Ok(HipKernel {
            module: self.inner.clone(),
            raw,
        })
    }
}

/// A named function that keeps its containing HIP module loaded.
#[derive(Clone)]
pub struct HipKernel {
    module: Rc<ModuleInner>,
    raw: KernelHandle,
}

/// One kernel argument supplied as host bytes or a device allocation.
///
/// `Bytes` is copied into naturally aligned storage before launch. The kernel's actual parameter
/// types and order remain the caller's responsibility.
pub enum HipKernelArgument<'a> {
    Bytes(&'a [u8]),
    Buffer(&'a DeviceBuffer),
}

#[repr(align(16))]
struct AlignedKernelWord {
    _word: usize,
}

const INLINE_KERNEL_PARAMETERS: usize = 8;

const fn ensure_batch_open(failed: bool) -> Result<(), HipError> {
    if failed {
        Err(HipError::BatchPoisoned)
    } else {
        Ok(())
    }
}

struct LaunchAccessLeases {
    inline: [Option<DeviceAccessLease>; INLINE_KERNEL_PARAMETERS],
    overflow: Vec<DeviceAccessLease>,
    inline_len: usize,
}

impl LaunchAccessLeases {
    fn new() -> Self {
        Self {
            inline: std::array::from_fn(|_| None),
            overflow: Vec::new(),
            inline_len: 0,
        }
    }

    fn contains(&self, allocation: &Rc<DeviceAllocation>) -> bool {
        self.inline[..self.inline_len]
            .iter()
            .flatten()
            .chain(self.overflow.iter())
            .any(|lease| Rc::ptr_eq(&lease.allocation, allocation))
    }

    fn push(&mut self, lease: DeviceAccessLease) {
        if self.inline_len < self.inline.len() {
            self.inline[self.inline_len] = Some(lease);
            self.inline_len += 1;
        } else {
            self.overflow.push(lease);
        }
    }

    fn quarantine(&self) {
        self.inline[..self.inline_len]
            .iter()
            .flatten()
            .chain(self.overflow.iter())
            .for_each(DeviceAccessLease::quarantine);
    }
}

fn can_inline_kernel_parameters(arguments: &[HipKernelArgument<'_>]) -> bool {
    arguments.len() <= INLINE_KERNEL_PARAMETERS
        && arguments.iter().all(|argument| match argument {
            HipKernelArgument::Bytes(bytes) => bytes.len() <= size_of::<AlignedKernelWord>(),
            HipKernelArgument::Buffer(_) => {
                size_of::<*mut c_void>() <= size_of::<AlignedKernelWord>()
            }
        })
}

fn copy_inline_kernel_parameter(destination: &mut AlignedKernelWord, bytes: &[u8]) -> *mut c_void {
    assert!(bytes.len() <= size_of::<AlignedKernelWord>());
    let pointer = ptr::from_mut(destination).cast::<c_void>();
    // SAFETY: the bound above keeps the write within one aligned word; both pointers are valid
    // for `bytes.len()` bytes, and the source slice cannot alias the local destination word.
    unsafe { ptr::copy_nonoverlapping(bytes.as_ptr(), pointer.cast(), bytes.len()) };
    pointer
}

struct LaunchResources {
    _module: Rc<ModuleInner>,
    // Each unique lease owns the allocation as well as its busy gate until completion.
    access_leases: LaunchAccessLeases,
    stream: HipStreamHandle,
}
impl LaunchResources {
    fn quarantine(&self) {
        self.access_leases.quarantine();
    }

    fn belongs_to_stream(&self, stream: &HipStreamHandle) -> bool {
        Rc::ptr_eq(&self.stream.inner, &stream.inner)
    }
}

/// Completion token for a kernel launch. Dropping it waits for completion; if HIP cannot confirm
/// completion, the backend leaks retained resources rather than freeing memory still in use.
pub struct HipCompletion {
    event: Option<HipEventHandle>,
    resources: Option<LaunchResources>,
}
impl HipCompletion {
    /// Wait for the launch to complete. On an error the token retains its resources and can be
    /// retried or dropped (drop retries and leaks resources if HIP still cannot confirm completion).
    ///
    /// # Errors
    ///
    /// Returns the HIP synchronization error while retaining the launch resources for a retry.
    pub fn wait(&mut self) -> Result<(), HipError> {
        if let Some(event) = &self.event {
            event.synchronize()?;
        } else {
            return Ok(());
        }
        self.resources.take();
        self.event.take();
        Ok(())
    }
}
impl Drop for HipCompletion {
    fn drop(&mut self) {
        if self.resources.is_none() {
            return;
        }
        let completed = self
            .event
            .as_ref()
            .is_some_and(|event| event.synchronize().is_ok());
        if completed {
            self.resources.take();
            self.event.take();
        } else {
            // HIP reported an error and did not confirm that the queued kernel has stopped using
            // these resources. Leak the owners rather than risk a device use-after-free.
            if let Some(resources) = self.resources.take() {
                resources.quarantine();
                std::mem::forget(resources);
            }
            if let Some(event) = self.event.take() {
                std::mem::forget(event);
            }
        }
    }
}

/// Completion for several launches queued in order on one HIP stream.
///
/// Every launch's resources remain retained until the final event completes. Use
/// [`HipCompletionBatch::new`] before submitting nodes, push each launch completion, then call
/// [`HipCompletionBatch::finish`] after the final launch has been queued.
pub struct HipCompletionBatch {
    stream: HipStreamHandle,
    launch_events: Vec<HipEventHandle>,
    resources: Vec<LaunchResources>,
    failed: bool,
}

impl HipCompletionBatch {
    /// Start collecting completions for launches queued on `stream`.
    #[must_use]
    pub fn new(stream: &HipStreamHandle) -> Self {
        Self {
            stream: stream.clone(),
            launch_events: Vec::new(),
            resources: Vec::new(),
            failed: false,
        }
    }

    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.resources.is_empty()
    }

    /// Add one launch completion to this batch.
    ///
    /// The launch must have been submitted to the batch stream. Its event and resources are
    /// retained until the batch's final event is synchronized.
    ///
    /// # Errors
    ///
    /// Returns [`HipError::DifferentStream`] when the completion belongs to another stream.
    pub fn push(&mut self, mut completion: HipCompletion) -> Result<(), HipError> {
        ensure_batch_open(self.failed)?;
        if completion
            .resources
            .as_ref()
            .is_some_and(|resources| !resources.belongs_to_stream(&self.stream))
        {
            return Err(HipError::DifferentStream);
        }
        if let Some(event) = completion.event.take() {
            self.launch_events.push(event);
        }
        if let Some(resources) = completion.resources.take() {
            self.resources.push(resources);
        }
        Ok(())
    }

    /// Record the final event after all launches already queued on this stream.
    ///
    /// If event creation or recording fails, the stream is synchronized before resources are
    /// released. When synchronization cannot establish quiescence, the complete bundle is
    /// quarantined and intentionally leaked.
    ///
    /// # Errors
    ///
    /// Returns the event or stream synchronization error.
    pub fn finish(&mut self) -> Result<HipBatchCompletion, HipError> {
        ensure_batch_open(self.failed)?;
        let event = match self.stream.inner.runtime.create_event() {
            Ok(event) => event,
            Err(error) => {
                self.release_after_stream_sync();
                return Err(error);
            }
        };
        if let Err(error) = self.stream.record(&event) {
            if self.stream.synchronize().is_err() {
                self.failed = true;
                self.quarantine_and_forget();
                std::mem::forget(event);
            } else {
                // Stream quiescence is proven; the failed event is not needed to release owners.
                self.resources.clear();
                self.launch_events.clear();
            }
            return Err(error);
        }
        Ok(HipBatchCompletion {
            final_event: Some(event),
            launch_events: std::mem::take(&mut self.launch_events),
            resources: std::mem::take(&mut self.resources),
        })
    }

    fn release_after_stream_sync(&mut self) {
        if self.stream.synchronize().is_err() {
            self.failed = true;
            self.quarantine_and_forget();
        } else {
            self.resources.clear();
            self.launch_events.clear();
        }
    }

    fn quarantine_and_forget(&mut self) {
        for resources in &self.resources {
            resources.quarantine();
        }
        for resources in self.resources.drain(..) {
            std::mem::forget(resources);
        }
        for event in self.launch_events.drain(..) {
            std::mem::forget(event);
        }
    }
}

impl Drop for HipCompletionBatch {
    fn drop(&mut self) {
        if self.resources.is_empty() {
            return;
        }
        if self.stream.synchronize().is_err() {
            self.quarantine_and_forget();
        }
    }
}

/// Final-event completion token retaining every launch in a [`HipCompletionBatch`].
pub struct HipBatchCompletion {
    final_event: Option<HipEventHandle>,
    launch_events: Vec<HipEventHandle>,
    resources: Vec<LaunchResources>,
}

impl HipBatchCompletion {
    /// Wait for the batch's final event. On failure the token retains resources for retry.
    ///
    /// # Errors
    ///
    /// Returns the HIP event synchronization error.
    pub fn wait(&mut self) -> Result<(), HipError> {
        if let Some(event) = &self.final_event {
            event.synchronize()?;
        }
        self.resources.clear();
        self.launch_events.clear();
        self.final_event.take();
        Ok(())
    }
}

impl Drop for HipBatchCompletion {
    fn drop(&mut self) {
        if self.resources.is_empty() {
            return;
        }
        let completed = self
            .final_event
            .as_ref()
            .is_some_and(|event| event.synchronize().is_ok());
        if completed {
            self.resources.clear();
            self.launch_events.clear();
            self.final_event.take();
        } else {
            for resources in &self.resources {
                resources.quarantine();
            }
            for resources in self.resources.drain(..) {
                std::mem::forget(resources);
            }
            for event in self.launch_events.drain(..) {
                std::mem::forget(event);
            }
            if let Some(event) = self.final_event.take() {
                std::mem::forget(event);
            }
        }
    }
}

impl HipKernel {
    /// Launch this function on `stream` and return a completion token that retains its module,
    /// stream, and all referenced buffers until the queued work finishes.
    ///
    /// # Safety
    /// The caller must provide arguments in the exact ABI order and representation expected by
    /// the loaded kernel. Each `Bytes` slice must contain a valid value of its corresponding
    /// parameter type; each `Buffer` must be used only in ways valid for that allocation's size.
    /// The caller must not perform conflicting accesses to referenced buffers until the returned
    /// completion token has synchronized. The kernel must not retain argument pointers after it
    /// returns. GPU execution errors remain possible and are reported when the completion token is
    /// synchronized.
    /// Launch this kernel with the supplied argument storage and retain resources until completion.
    ///
    /// # Safety
    ///
    /// The caller must ensure the argument count, byte layouts, ordering, and pointer types match
    /// the kernel's actual ABI. Device buffers must be valid for the kernel's accesses.
    ///
    /// # Errors
    ///
    /// Returns a HIP launch error or an error while preparing stream/event resources.
    #[allow(clippy::too_many_lines)] // Keep argument storage, launch, and uncertain-completion retention visible together.
    pub unsafe fn launch<'a>(
        &'a self,
        stream: &'a HipStreamHandle,
        grid: [u32; 3],
        block: [u32; 3],
        shared_memory_bytes: u32,
        arguments: &'a [HipKernelArgument<'a>],
    ) -> Result<HipCompletion, HipError> {
        // SAFETY: the caller upholds the kernel argument and buffer-access contract documented
        // above; the helper performs the same enqueue while retaining completion resources.
        unsafe { self.launch_inner(stream, grid, block, shared_memory_bytes, arguments, None) }
            .and_then(|completion| completion.ok_or(HipError::BatchPoisoned))
    }

    /// Queue this kernel into an ordered completion batch without creating a per-launch event.
    /// The batch must be finished after all launches; it owns all referenced allocations and the
    /// module until the final event completes. A launch error poisons the batch, which must then
    /// be dropped so its drop handler synchronizes the stream or quarantines every resource.
    ///
    /// # Safety
    /// The caller must provide arguments in the exact ABI order and representation expected by
    /// the loaded kernel. Each buffer must be valid for the kernel's accesses, and the kernel must
    /// not retain argument pointers after returning. All accesses sharing resources with other
    /// queued batch launches must be ordered on this same stream.
    ///
    /// # Errors
    /// Returns a HIP launch error, a stream/runtime mismatch, or [`HipError::BatchPoisoned`] if a
    /// prior launch failed.
    pub unsafe fn launch_into_batch(
        &self,
        batch: &mut HipCompletionBatch,
        grid: [u32; 3],
        block: [u32; 3],
        shared_memory_bytes: u32,
        arguments: &[HipKernelArgument<'_>],
    ) -> Result<(), HipError> {
        ensure_batch_open(batch.failed)?;
        // Clone first so the mutable batch borrow can be passed alongside its stream identity.
        let stream = batch.stream.clone();
        // SAFETY: the caller upholds the ABI and buffer-validity requirements. The batch has
        // ownership of every successfully attempted launch before the HIP enqueue is invoked.
        unsafe {
            self.launch_inner(
                &stream,
                grid,
                block,
                shared_memory_bytes,
                arguments,
                Some(batch),
            )
        }?;
        Ok(())
    }

    #[allow(clippy::too_many_lines)]
    unsafe fn launch_inner(
        &self,
        stream: &HipStreamHandle,
        grid: [u32; 3],
        block: [u32; 3],
        shared_memory_bytes: u32,
        arguments: &[HipKernelArgument<'_>],
        mut batch: Option<&mut HipCompletionBatch>,
    ) -> Result<Option<HipCompletion>, HipError> {
        self.module
            .runtime
            .ensure_same_runtime(&stream.inner.runtime)?;
        if let Some(batch) = batch.as_deref() {
            if !Rc::ptr_eq(&batch.stream.inner, &stream.inner) {
                return Err(HipError::DifferentStream);
            }
            ensure_batch_open(batch.failed)?;
        }
        for axis in 0..3 {
            if grid[axis] == 0 || block[axis] == 0 {
                return Err(HipError::InvalidLaunchDimensions);
            }
        }

        let inline = can_inline_kernel_parameters(arguments);
        let mut inline_storage: [AlignedKernelWord; INLINE_KERNEL_PARAMETERS] =
            std::array::from_fn(|_| AlignedKernelWord { _word: 0 });
        let mut inline_params = [ptr::null_mut(); INLINE_KERNEL_PARAMETERS];
        let mut storage = if inline {
            Vec::<Vec<AlignedKernelWord>>::new()
        } else {
            Vec::<Vec<AlignedKernelWord>>::with_capacity(arguments.len())
        };
        let mut access_leases = LaunchAccessLeases::new();
        for (index, argument) in arguments.iter().enumerate() {
            let bytes = match argument {
                HipKernelArgument::Bytes(bytes) => *bytes,
                HipKernelArgument::Buffer(buffer) => {
                    self.module
                        .runtime
                        .ensure_same_runtime(&buffer.allocation.runtime)?;
                    if !access_leases.contains(&buffer.allocation) {
                        access_leases.push(buffer.acquire_stream_access(stream)?);
                    }
                    // HIP kernel parameters receive a device pointer value, not its host address.
                    unsafe {
                        std::slice::from_raw_parts(
                            (&raw const buffer.allocation.pointer).cast(),
                            size_of::<*mut c_void>(),
                        )
                    }
                }
            };
            if inline {
                // Each word stays at a stable stack address until HIP consumes the pointer table.
                inline_params[index] =
                    copy_inline_kernel_parameter(&mut inline_storage[index], bytes);
            } else {
                let words = bytes.len().div_ceil(size_of::<AlignedKernelWord>()).max(1);
                let mut aligned = (0..words)
                    .map(|_| AlignedKernelWord { _word: 0 })
                    .collect::<Vec<_>>();
                // SAFETY: `aligned` has enough writable bytes and each argument region starts at
                // a 16-byte-aligned address. `bytes` is a live argument value or device pointer.
                unsafe {
                    ptr::copy_nonoverlapping(
                        bytes.as_ptr(),
                        aligned.as_mut_ptr().cast(),
                        bytes.len(),
                    );
                }
                storage.push(aligned);
            }
        }
        let mut kernel_params: Vec<*mut c_void> = if inline {
            Vec::new()
        } else {
            storage
                .iter_mut()
                .map(|arg| arg.as_mut_ptr().cast())
                .collect()
        };
        let params = if arguments.is_empty() {
            ptr::null_mut()
        } else if inline {
            inline_params.as_mut_ptr()
        } else {
            kernel_params.as_mut_ptr()
        };
        let mut resources = Some(LaunchResources {
            _module: self.module.clone(),
            access_leases,
            stream: stream.clone(),
        });
        let direct_batch = batch.is_some();
        if let Some(batch) = batch.as_deref_mut() {
            // Retain before the launch call: HIP may report an error after work has entered the
            // stream, so dropping these owners on the error path would be unsafe.
            batch
                .resources
                .push(resources.take().expect("resources retained once"));
        }
        let completion_event = if direct_batch {
            None
        } else {
            Some(self.module.runtime.create_event()?)
        };
        let launch_result =
            self.module
                .runtime
                .call("hipModuleLaunchKernel", |f: ModuleLaunchKernel| unsafe {
                    f(
                        self.raw,
                        grid[0],
                        grid[1],
                        grid[2],
                        block[0],
                        block[1],
                        block[2],
                        shared_memory_bytes,
                        stream.inner.raw,
                        params,
                        ptr::null_mut(),
                    )
                });
        if let Err(error) = launch_result {
            if let Some(batch) = batch {
                batch.failed = true;
            } else if stream.synchronize().is_err() {
                // HIP can surface an earlier asynchronous fault from a later API call. If the
                // stream cannot confirm quiescence, retain the launch resources conservatively.
                let Some(resources) = resources.take() else {
                    return Err(HipError::BatchPoisoned);
                };
                resources.quarantine();
                std::mem::forget(resources);
            }
            return Err(error);
        }
        if let Some(completion_event) = completion_event {
            if let Err(error) = stream.record(&completion_event) {
                if stream.synchronize().is_err() {
                    let Some(resources) = resources.take() else {
                        return Err(HipError::BatchPoisoned);
                    };
                    resources.quarantine();
                    std::mem::forget(resources);
                    std::mem::forget(completion_event);
                }
                return Err(error);
            }
            // Keep args borrowed through this call to tie the launch's completion token lifetime
            // to the argument descriptors as well as its owned resource references.
            let _ = arguments;
            return Ok(Some(HipCompletion {
                event: Some(completion_event),
                resources,
            }));
        }
        // Direct batch launches are covered by the final event recorded after all queued work.
        // Keep arguments borrowed through this synchronous enqueue call for ABI pointer validity.
        let _ = arguments;
        Ok(None)
    }
}

#[cfg(test)]
mod memory_snapshot_tests {
    use super::*;

    #[test]
    fn timing_events_keep_distinct_flags_from_completion_events() {
        assert_eq!(HIP_EVENT_DEFAULT, 0);
        assert_eq!(HIP_EVENT_DISABLE_TIMING, 2);
        assert_ne!(HIP_EVENT_DEFAULT, HIP_EVENT_DISABLE_TIMING);
    }

    #[test]
    fn kfd_target_version_is_formatted_as_an_exact_gfx_target() {
        assert_eq!(format_gfx_target(100_300).as_deref(), Some("gfx1030"));
        assert_eq!(format_gfx_target(110_000).as_deref(), Some("gfx1100"));
        assert_eq!(format_gfx_target(90_010).as_deref(), Some("gfx90a"));
        assert_eq!(format_gfx_target(0), None);
        assert_eq!(format_gfx_target(101_010), None);
    }

    #[test]
    fn inline_kernel_parameter_is_aligned_and_copies_exact_bytes() {
        let mut word = AlignedKernelWord { _word: 0 };
        let bytes = [0x12_u8, 0x34, 0x56, 0x78, 0x9a];
        let pointer = copy_inline_kernel_parameter(&mut word, &bytes);
        assert_eq!((pointer as usize) % 16, 0);
        // SAFETY: the pointer refers to the live aligned word and five initialized bytes.
        let copied = unsafe { std::slice::from_raw_parts(pointer.cast::<u8>(), bytes.len()) };
        assert_eq!(copied, bytes);
        assert!(!copy_inline_kernel_parameter(&mut word, &[]).is_null());
    }

    #[test]
    fn inline_kernel_arguments_fall_back_for_large_arity_or_values() {
        let empty = HipKernelArgument::Bytes(&[]);
        let full_word = HipKernelArgument::Bytes(&[0_u8; 16]);
        let oversized = HipKernelArgument::Bytes(&[0_u8; 17]);
        assert!(can_inline_kernel_parameters(&[empty, full_word]));
        assert!(!can_inline_kernel_parameters(&[oversized]));
        let nine: [HipKernelArgument<'_>; 9] =
            std::array::from_fn(|_| HipKernelArgument::Bytes(&[]));
        assert!(!can_inline_kernel_parameters(&nine));
    }

    #[test]
    fn cloned_allocation_gate_rejects_busy_access_and_releases_with_lease() {
        let state = Rc::new(AllocationAccess {
            state: Cell::new(AllocationAccessState::Idle),
        });
        let clone = Rc::clone(&state);
        let lease = state.acquire().unwrap();

        assert!(clone.acquire().is_err());
        drop(lease);
        assert!(clone.acquire().is_ok());
    }

    #[test]
    fn forgotten_gate_lease_stays_busy_after_unconfirmed_completion() {
        let state = Rc::new(AllocationAccess {
            state: Cell::new(AllocationAccessState::Idle),
        });
        let clone = Rc::clone(&state);
        let lease = state.acquire().unwrap();

        std::mem::forget(lease);
        assert!(clone.acquire().is_err());
    }

    #[test]
    fn allocation_gate_allows_ordered_leases_only_for_the_same_stream() {
        let state = Rc::new(AllocationAccess {
            state: Cell::new(AllocationAccessState::Idle),
        });
        let first = state.acquire_stream(11).unwrap();
        let second = state.acquire_stream(11).unwrap();

        assert!(state.acquire().is_err());
        assert!(state.acquire_stream(12).is_err());
        drop(first);
        assert!(state.acquire().is_err());
        assert!(state.acquire_stream(11).is_ok());
        drop(second);
        assert!(state.acquire().is_ok());
    }

    #[test]
    fn queued_node_leases_remain_exclusive_until_the_batch_releases_them() {
        let state = Rc::new(AllocationAccess {
            state: Cell::new(AllocationAccessState::Idle),
        });
        let mut retained_by_batch = vec![
            state.acquire_stream(11).unwrap(),
            state.acquire_stream(11).unwrap(),
        ];

        drop(retained_by_batch.pop());
        assert!(state.acquire().is_err());
        drop(retained_by_batch);
        assert!(state.acquire().is_ok());
    }

    #[test]
    fn exclusive_gate_blocks_stream_leases_until_release() {
        let state = Rc::new(AllocationAccess {
            state: Cell::new(AllocationAccessState::Idle),
        });
        let exclusive = state.acquire().unwrap();

        assert!(state.acquire_stream(11).is_err());
        drop(exclusive);
        assert!(state.acquire_stream(11).is_ok());
    }

    #[test]
    fn quarantined_stream_gate_rejects_even_the_same_stream() {
        let state = Rc::new(AllocationAccess {
            state: Cell::new(AllocationAccessState::Idle),
        });
        let lease = state.acquire_stream(11).unwrap();

        lease.quarantine_stream();
        assert!(state.acquire_stream(11).is_err());
        assert!(state.acquire().is_err());
        std::mem::forget(lease);
    }

    #[test]
    fn poisoned_batch_rejects_more_work_and_finish() {
        assert_eq!(ensure_batch_open(false), Ok(()));
        assert_eq!(ensure_batch_open(true), Err(HipError::BatchPoisoned));
    }

    #[test]
    #[ignore = "requires RX 6900 XT and ROCm device access"]
    fn rx_6900_xt_two_dependent_launches_complete_through_final_batch_event() {
        let runtime = HipRuntime::new(0).expect("RX 6900 XT HIP runtime");
        let info = runtime.device_info().expect("selected HIP device info");
        assert!(
            info.name.contains("6900 XT"),
            "unexpected GPU: {}",
            info.name
        );

        let image = crate::compile_hip_source_for_device(
            &runtime,
            r#"
extern "C" __global__ void increment(float* value) {
    if (blockIdx.x == 0 && threadIdx.x == 0) value[0] += 1.0f;
}
"#,
        )
        .expect("compile increment kernel with HIPRTC");
        let module = runtime.load_module(&image).expect("load increment kernel");
        let kernel = module
            .function(c"increment")
            .expect("resolve increment kernel");
        let stream = runtime.create_stream().expect("create HIP stream");
        let mut buffer = runtime
            .allocate(size_of::<f32>())
            .expect("allocate device value");
        buffer
            .copy_from(&0.0_f32.to_ne_bytes())
            .expect("initialize device value");
        let arguments = [HipKernelArgument::Buffer(&buffer)];

        let mut batch = HipCompletionBatch::new(&stream);
        // SAFETY: HIPRTC compiled one f32 pointer parameter, and the one-element allocation is
        // valid for both ordered launches. The batch retains it; the final event is the only wait.
        unsafe {
            kernel
                .launch_into_batch(&mut batch, [1, 1, 1], [1, 1, 1], 0, &arguments)
                .expect("queue first increment directly into batch");
            kernel
                .launch_into_batch(&mut batch, [1, 1, 1], [1, 1, 1], 0, &arguments)
                .expect("queue dependent increment directly into batch");
        }
        batch
            .finish()
            .expect("record final batch event")
            .wait()
            .expect("wait for final batch event");

        let mut actual = [0_u8; size_of::<f32>()];
        buffer
            .copy_to(&mut actual)
            .expect("read completed device value");
        assert_eq!(f32::from_ne_bytes(actual).to_bits(), 2.0_f32.to_bits());
    }

    #[test]
    fn translates_physical_pool_memory_without_claiming_process_attribution() {
        let snapshot = hip_memory_pool_snapshot(
            PcuMemoryPoolId(17),
            HipMemoryInfo {
                free_bytes: 30,
                total_bytes: 100,
            },
        );
        assert_eq!(snapshot.id, PcuMemoryPoolId(17));
        assert_eq!(snapshot.capacity_bytes, Some(100));
        assert_eq!(snapshot.system_used_bytes, PcuMemoryUsage::Known(70));
        assert_eq!(snapshot.process_used_bytes, PcuMemoryUsage::Unknown);
        assert_eq!(snapshot.system_ledger_reserved_bytes, 0);
        assert_eq!(snapshot.process_ledger_reserved_bytes, 0);
    }

    #[test]
    fn malformed_or_zero_capacity_telemetry_falls_back_to_unknown() {
        let malformed = hip_memory_pool_snapshot(
            PcuMemoryPoolId(1),
            HipMemoryInfo {
                free_bytes: 101,
                total_bytes: 100,
            },
        );
        assert_eq!(malformed.capacity_bytes, Some(100));
        assert_eq!(malformed.system_used_bytes, PcuMemoryUsage::Unknown);
        assert_eq!(malformed.process_used_bytes, PcuMemoryUsage::Unknown);

        let zero_capacity = hip_memory_pool_snapshot(
            PcuMemoryPoolId(2),
            HipMemoryInfo {
                free_bytes: 0,
                total_bytes: 0,
            },
        );
        assert_eq!(zero_capacity.capacity_bytes, None);
        assert_eq!(zero_capacity.system_used_bytes, PcuMemoryUsage::Unknown);
    }

    #[test]
    fn runtime_probe_is_safe_without_selecting_a_device() {
        match HipRuntime::probe() {
            Ok(probe) => assert!(!probe.runtime_library.is_empty()),
            Err(
                HipError::RuntimeUnavailable(_)
                | HipError::Runtime {
                    operation: "hipGetDeviceCount",
                    code: 100,
                    ..
                },
            ) => {}
            Err(error) => panic!("unexpected HIP probe error: {error}"),
        }
    }
}
