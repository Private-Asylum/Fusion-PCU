//! Dynamically loaded CUDA runtime support for Fusion PCU.
//!
//! The bounded PCU Dispatch path lowers checked scalar programs to CUDA source and compiles images,
//! and submits them through module launches. Runtime resources and completion owners are kept
//! here; unsupported PCU operations are rejected by the lowerer.

extern crate fusion_pcu_core as fusion_pcu;
#[rustfmt::skip]
use crate::ffi::{
    query_pci_bus_id,
    driver_architecture,
    enumerate_driver_devices,
    raw_cuda_error,
    raw_driver_error,
};

#[rustfmt::skip]
use std::{
    any::Any,
    cell::{
        Cell,
        RefCell,
    },
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

use crate::ffi::Library;
use smallvec::SmallVec;
#[rustfmt::skip]
use crate::ffi::{
    driver::{
        KernelHandle,
        ModuleHandle,
    },
    runtime::{
        CUDA_EVENT_DEFAULT,
        CUDA_EVENT_DISABLE_TIMING,
        CUDA_MEMCPY_DEVICE_TO_DEVICE,
        CUDA_MEMCPY_DEVICE_TO_HOST,
        CUDA_MEMCPY_HOST_TO_DEVICE,
        CUDA_SUCCESS,
        CudaDevice,
        CudaEvent,
        CudaStream,
    },
};
#[rustfmt::skip]
use fusion_pcu::{
    PcuMemoryPoolId,
    PcuMemoryPoolSnapshot,
    PcuMemoryUsage,
};

#[path = "admission/admission.rs"]
mod admission;
mod blas;
#[path = "blas_policy/blas_policy.rs"]
mod blas_policy;
#[path = "codegen/codegen.rs"]
mod codegen;
#[path = "control/control.rs"]
mod control;
#[path = "device_facts/device_facts.rs"]
mod device_facts;
mod device_kernel;
mod discovery;
mod dispatch;
mod error;
#[path = "ffi/ffi.rs"]
mod ffi;
#[path = "graph/graph.rs"]
mod graph;
mod host_kernel;
mod memory;
#[path = "owned_dispatch/owned_dispatch.rs"]
mod owned_dispatch;
#[cfg(feature = "tensor")]
#[path = "tensor/tensor.rs"]
mod tensor;
#[path = "transfer/transfer.rs"]
pub mod transfer;

#[rustfmt::skip]
pub use blas::{
    Cublas,
    CublasError,
    CublasSgemmHostTiming,
};
#[rustfmt::skip]
pub use blas_policy::{
    CublasConfiguredModes,
    CublasEnvironmentSnapshot,
    CublasNumericalConfig,
};
#[rustfmt::skip]
pub use codegen::compiler::{
    CudaCompileError,
    compile_cuda_source,
};
#[rustfmt::skip]
pub use dispatch::{
    CudaDispatchBinding,
    CudaDispatchError,
    execute_pcu_dispatch,
};
#[rustfmt::skip]
pub use control::{
    CudaReadiness,
    CudaStreamMode,
    CudaStreamOptions,
    CudaStreamPriorityRange,
};
#[rustfmt::skip]
pub use device_facts::{
    CudaDeviceUuid,
    CudaPhysicalDeviceFacts,
    CudaPhysicalDeviceFlags,
};
pub use discovery::CudaDiscovery;
#[rustfmt::skip]
pub use graph::{
    CudaGraphDispatch,
    CudaNativeGraph,
    CudaNativeGraphError,
};
#[rustfmt::skip]
pub use device_kernel::{
    CudaDeviceKernelError,
    CudaPreparedDeviceKernel,
};
pub use error::*;
#[rustfmt::skip]
pub use host_kernel::{
    CudaHostKernelError,
    CudaMixedHostArgument,
    CudaPreparedHostKernel,
};
#[rustfmt::skip]
pub use codegen::lower::{
    CudaLowerError,
    lower_dispatch_to_cuda_rtc_source,
    lower_dispatch_to_cuda_source,
};
#[rustfmt::skip]
pub use memory::{
    CudaImportDescriptor,
    CudaMemoryMapping,
    CudaMemoryProvider,
    CudaMemoryResource,
};
#[rustfmt::skip]
pub use owned_dispatch::{
    CudaCheckedBatchCompletion,
    CudaCheckedDispatchBatch,
    CudaExecutionStep,
    CudaOwnedCompletion,
    CudaOwnedDispatchBackend,
    CudaOwnedDispatchError,
    CudaOwnedExecution,
    CudaOwnedExecutionError,
    CudaOwnedExecutionNode,
    CudaOwnedExecutionOperation,
    CudaOwnedExecutionTwoSlot,
    CudaPreparedDispatch,
    CudaSequentialCheckedDispatch,
    CudaTwoSlotExecutionStep,
};
#[rustfmt::skip]
pub use codegen::rtc::{
    CudaRtcError,
    compile_cuda_source_for_device,
};
#[cfg(feature = "tensor")]
#[rustfmt::skip]
pub use tensor::{
    CudaAdmittedTensorFeedbackResources,
    CudaPreparedTensorGraph,
    CudaTensorExecution,
    CudaTensorAssessor,
    CudaOwnedTensorAssessor,
    CudaOwnedPreparedTensorGraph,
    CudaTensorElementwiseHostTiming,
    CudaTensorError,
    CudaTensorExecutionError,
    CudaTensorFeedbackPrepareError,
    CudaTensorFeedbackReleaseError,
    CudaTensorFeedbackResources,
    CudaTensorInput,
    CudaTensorInputRef,
    CudaTensorNodeTiming,
    CudaTensorOutputBank,
    CudaTensorOwnedOutput,
    CudaTensorPrewarmReport,
    lower_strict_matmul_to_cuda_source,
    CudaTensorScratch,
};

#[cfg(all(feature = "tensor", feature = "insights"))]
#[rustfmt::skip]
pub use tensor::{
    CudaTensorBatchedExecutionTiming,
    CudaTensorInsightRecord,
    CudaTensorOperationInsight,
};

/// Dynamically loaded CUDA runtime and the selected device.
#[derive(Clone)]
pub struct CudaRuntime(Arc<RuntimeInner>);

/// CUDA runtime availability and visible-device count, queried without selecting a device.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CudaRuntimeProbe {
    pub device_count: u32,
    pub runtime_library: String,
}

struct RuntimeInner {
    library: Arc<Library>,
    driver: Arc<Library>,
    ordinal: c_int,
    device: CudaDevice,
    name: String,
}

impl CudaRuntime {
    /// Load CUDA and report visible devices without selecting one. Zero devices is a valid result.
    ///
    /// # Errors
    ///
    /// Returns an error when a runtime library cannot be loaded, a required symbol is missing,
    /// or CUDA reports a failure.
    pub fn probe() -> Result<CudaRuntimeProbe, CudaError> {
        let candidates = std::env::var_os("CUDA_RUNTIME_LIBRARY").map_or_else(
            || {
                vec![
                    "/usr/local/cuda/lib64/libcudart.so".into(),
                    "libcudart.so".into(),
                    "libcudart.so.13".into(),
                    "libcudart.so.12".into(),
                ]
            },
            |path| vec![path],
        );
        let mut last_error = None;
        for candidate in candidates {
            let path = candidate.to_string_lossy().into_owned();
            let library = match ffi::load_library(&candidate) {
                Ok(library) => library,
                Err(error) => {
                    last_error = Some(error);
                    continue;
                }
            };
            let get_count =
                crate::ffi::resolve_cuda_get_device_count(&library).map_err(|error| {
                    CudaError::MissingSymbol {
                        symbol: "cudaGetDeviceCount",
                        detail: error.to_string(),
                    }
                })?;
            let mut count = 0;
            let status = unsafe {
                crate::ffi::call_runtime_GetDeviceCount(*get_count, ptr::from_mut(&mut count))
            };
            if status != CUDA_SUCCESS {
                return Err(raw_cuda_error(&library, "cudaGetDeviceCount", status));
            }
            return Ok(CudaRuntimeProbe {
                device_count: count.max(0).unsigned_abs(),
                runtime_library: path,
            });
        }
        Err(CudaError::RuntimeUnavailable(
            last_error.unwrap_or_else(|| "no runtime library candidates".into()),
        ))
    }

    /// Enumerate visible devices without selecting one, querying only stable CUDA identity facts.
    ///
    /// # Errors
    ///
    /// Returns an error when a runtime library cannot be loaded, a required symbol is missing,
    /// or CUDA fails to enumerate a device.
    pub fn enumerate_devices() -> Result<Vec<CudaDeviceInfo>, CudaError> {
        let candidates = std::env::var_os("CUDA_RUNTIME_LIBRARY").map_or_else(
            || {
                vec![
                    "/usr/local/cuda/lib64/libcudart.so".into(),
                    "libcudart.so".into(),
                    "libcudart.so.13".into(),
                    "libcudart.so.12".into(),
                ]
            },
            |path| vec![path],
        );
        let mut last_error = None;
        for candidate in candidates {
            let library = match ffi::load_library(&candidate) {
                Ok(library) => library,
                Err(error) => {
                    last_error = Some(error);
                    continue;
                }
            };
            let get_count =
                crate::ffi::resolve_cuda_get_device_count(&library).map_err(|error| {
                    CudaError::MissingSymbol {
                        symbol: "cudaGetDeviceCount",
                        detail: error.to_string(),
                    }
                })?;
            let mut count = 0;
            let status = unsafe {
                crate::ffi::call_runtime_GetDeviceCount(*get_count, ptr::from_mut(&mut count))
            };
            if status != CUDA_SUCCESS {
                return Err(raw_cuda_error(&library, "cudaGetDeviceCount", status));
            }
            let driver_candidate =
                std::env::var_os("CUDA_DRIVER_LIBRARY").unwrap_or_else(|| "libcuda.so.1".into());
            let driver =
                ffi::load_library(&driver_candidate).map_err(CudaError::RuntimeUnavailable)?;
            let cu_init =
                crate::ffi::resolve_cu_init(&driver).map_err(|error| CudaError::MissingSymbol {
                    symbol: "cuInit",
                    detail: error.to_string(),
                })?;
            let status = unsafe { crate::ffi::call_driver_DriverInit(*cu_init, 0) };
            if status != CUDA_SUCCESS {
                return Err(raw_driver_error(&driver, "cuInit", status));
            }
            let get_device = crate::ffi::resolve_cu_device_get(&driver).map_err(|error| {
                CudaError::MissingSymbol {
                    symbol: "cuDeviceGet",
                    detail: error.to_string(),
                }
            })?;
            let get_name = crate::ffi::resolve_cu_device_get_name(&driver).map_err(|error| {
                CudaError::MissingSymbol {
                    symbol: "cuDeviceGetName",
                    detail: error.to_string(),
                }
            })?;
            let total_mem =
                crate::ffi::resolve_cu_device_total_mem_v2(&driver).map_err(|error| {
                    CudaError::MissingSymbol {
                        symbol: "cuDeviceTotalMem_v2",
                        detail: error.to_string(),
                    }
                })?;
            return enumerate_driver_devices(&driver, count, &get_device, &get_name, &total_mem);
        }
        Err(CudaError::RuntimeUnavailable(
            last_error.unwrap_or_else(|| "no runtime library candidates".into()),
        ))
    }

    /// Load `libcudart` and `libcuda` and select `device_index`.
    ///
    /// # Errors
    ///
    /// Returns an error when CUDA cannot load, enumerate, or select the requested device.
    pub fn new(device_index: u32) -> Result<Self, CudaError> {
        let candidates = std::env::var_os("CUDA_RUNTIME_LIBRARY").map_or_else(
            || {
                vec![
                    "/usr/local/cuda/lib64/libcudart.so".into(),
                    "libcudart.so".into(),
                    "libcudart.so.13".into(),
                    "libcudart.so.12".into(),
                ]
            },
            |path| vec![path],
        );
        let mut last_error = None;
        for candidate in candidates {
            let library = match ffi::load_library(&candidate) {
                Ok(library) => library,
                Err(error) => {
                    last_error = Some(error);
                    continue;
                }
            };
            // Query availability before selecting a device. cudaSetDevice(0) can fail when no GPU
            // is visible, hiding the useful zero-device result.
            let get_count =
                crate::ffi::resolve_cuda_get_device_count(&library).map_err(|error| {
                    CudaError::MissingSymbol {
                        symbol: "cudaGetDeviceCount",
                        detail: error.to_string(),
                    }
                })?;
            let mut count: c_int = 0;
            let status = unsafe {
                crate::ffi::call_runtime_GetDeviceCount(*get_count, ptr::from_mut(&mut count))
            };
            let driver_candidate =
                std::env::var_os("CUDA_DRIVER_LIBRARY").unwrap_or_else(|| "libcuda.so.1".into());
            let driver =
                ffi::load_library(&driver_candidate).map_err(CudaError::RuntimeUnavailable)?;
            let runtime = Self(Arc::new(RuntimeInner {
                library,
                driver,
                ordinal: c_int::try_from(device_index).unwrap_or(0),
                device: 0,
                name: String::new(),
            }));
            if status != CUDA_SUCCESS {
                return Err(runtime.error("cudaGetDeviceCount", status));
            }
            let count = count.max(0).unsigned_abs();
            if device_index >= count {
                return Err(CudaError::DeviceIndexOutOfRange {
                    requested: device_index,
                    count,
                });
            }
            let init = crate::ffi::resolve_cu_init(&runtime.0.driver).map_err(|error| {
                CudaError::MissingSymbol {
                    symbol: "cuInit",
                    detail: error.to_string(),
                }
            })?;
            let status = unsafe { crate::ffi::call_driver_DriverInit(*init, 0) };
            if status != CUDA_SUCCESS {
                return Err(raw_driver_error(&runtime.0.driver, "cuInit", status));
            }
            let ordinal =
                c_int::try_from(device_index).map_err(|_| CudaError::DeviceIndexOutOfRange {
                    requested: device_index,
                    count,
                })?;
            runtime.cuda_set_device(ordinal)?;
            // cudaSetDevice establishes the device's primary context; cudaFree(NULL) forces
            // initialization before any driver API module operation uses that context.
            unsafe { crate::ffi::invoke_cudaFree(&runtime, ptr::null_mut()) }?;
            let mut device = 0;
            unsafe { crate::ffi::invoke_cuDeviceGet(&runtime, &raw mut device, ordinal) }?;
            let name = runtime.device_name(device)?;
            return Ok(Self(Arc::new(RuntimeInner {
                library: runtime.0.library.clone(),
                driver: runtime.0.driver.clone(),
                ordinal,
                device,
                name,
            })));
        }
        Err(CudaError::RuntimeUnavailable(
            last_error.unwrap_or_else(|| "no runtime library candidates".into()),
        ))
    }

    /// Return visible CUDA device count.
    ///
    /// # Errors
    ///
    /// Returns an error when CUDA cannot query the visible device count.
    pub fn device_count(&self) -> Result<u32, CudaError> {
        let mut count = 0;
        unsafe { crate::ffi::invoke_cudaGetDeviceCount(self, &raw mut count) }?;
        Ok(count.max(0).unsigned_abs())
    }

    /// Return the selected device's index, name, and physical memory capacity.
    ///
    /// # Errors
    ///
    /// Returns an error when CUDA cannot query device memory information.
    pub fn device_info(&self) -> Result<CudaDeviceInfo, CudaError> {
        let mut total = 0_usize;
        unsafe { crate::ffi::invoke_cuDeviceTotalMem_v2(self, &raw mut total, self.0.device) }?;
        let pci_bus_id = query_pci_bus_id(&self.0.driver, self.0.device);
        Ok(CudaDeviceInfo {
            index: self.0.ordinal,
            name: self.0.name.clone(),
            vendor: "NVIDIA".into(),
            architecture: driver_architecture(&self.0.driver, self.0.device),
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
    /// Returns an error when CUDA cannot query device memory information.
    pub fn memory_info(&self) -> Result<CudaMemoryInfo, CudaError> {
        let mut free = 0_usize;
        let mut total = 0_usize;
        unsafe { crate::ffi::invoke_cudaMemGetInfo(self, &raw mut free, &raw mut total) }?;
        Ok(CudaMemoryInfo {
            free_bytes: free as u64,
            total_bytes: total as u64,
        })
    }

    /// Query a point-in-time snapshot of this CUDA device's physical memory pool.
    ///
    /// `pool` is assigned by the caller so it can remain unique across provider types. CUDA's
    /// free/total query describes device-wide availability; it does not attribute memory use to
    /// this process, so process usage is reported as unknown. The result is telemetry only and
    /// does not reserve memory or prevent another client from allocating concurrently.
    ///
    /// # Errors
    ///
    /// Returns an error when the underlying CUDA memory query fails.
    pub fn memory_pool_snapshot(
        &self,
        pool: PcuMemoryPoolId,
    ) -> Result<PcuMemoryPoolSnapshot, CudaError> {
        self.memory_info()
            .map(|info| cuda_memory_pool_snapshot(pool, info))
    }

    /// Bind a caller-assigned abstract pool identity to this selected CUDA device.
    #[must_use]
    pub fn memory_provider(&self, pool: PcuMemoryPoolId) -> CudaMemoryProvider {
        CudaMemoryProvider::new(self.clone(), pool)
    }

    /// Allocate device memory. The allocation is released when the returned buffer is dropped.
    ///
    /// # Errors
    ///
    /// Returns the CUDA allocation error if the device cannot allocate the requested size.
    pub fn allocate(&self, bytes: usize) -> Result<DeviceBuffer, CudaError> {
        let mut pointer = ptr::null_mut();
        unsafe { crate::ffi::invoke_cudaMalloc(self, &raw mut pointer, bytes) }?;
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

    /// Load a CUDA code object (cubin or PTX) into this device context.
    ///
    /// # Errors
    ///
    /// Returns an error when CUDA cannot load the module.
    pub fn load_module(&self, image: &[u8]) -> Result<CudaModule, CudaError> {
        if image.is_empty() {
            return Err(CudaError::InvalidModuleImage);
        }
        // Driver API module loading accepts a NUL-terminated PTX string or a binary
        // image. nvcc PTX files omit the terminator; append one before the FFI read.
        let mut terminated = image.to_vec();
        terminated.push(0);
        let mut raw = ptr::null_mut();
        unsafe {
            crate::ffi::invoke_cuModuleLoadData(self, &raw mut raw, terminated.as_ptr().cast())
        }?;
        Ok(CudaModule {
            inner: Rc::new(ModuleInner {
                runtime: self.clone(),
                raw,
            }),
        })
    }

    /// Create a stream for ordered asynchronous operations.
    ///
    /// The initial crate surface does not enqueue work onto this stream; it is exposed so later
    /// submission support can build on a real CUDA stream without changing the resource type.
    ///
    /// # Errors
    ///
    /// Returns an error when CUDA cannot create the stream.
    pub fn create_stream(&self) -> Result<CudaStreamHandle, CudaError> {
        let mut stream = ptr::null_mut();
        unsafe { crate::ffi::invoke_cudaStreamCreate(self, &raw mut stream) }?;
        Ok(CudaStreamHandle {
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
    /// Returns an error when CUDA cannot create the event.
    pub fn create_event(&self) -> Result<CudaEventHandle, CudaError> {
        let mut event = ptr::null_mut();
        unsafe {
            crate::ffi::invoke_cudaEventCreateWithFlags(
                self,
                &raw mut event,
                CUDA_EVENT_DISABLE_TIMING,
            )
        }?;
        Ok(CudaEventHandle {
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
    /// Returns an error when CUDA cannot create the event.
    pub fn create_timing_event(&self) -> Result<CudaTimingEventHandle, CudaError> {
        let mut event = ptr::null_mut();
        unsafe {
            crate::ffi::invoke_cudaEventCreateWithFlags(self, &raw mut event, CUDA_EVENT_DEFAULT)
        }?;
        Ok(CudaTimingEventHandle {
            inner: Rc::new(EventInner {
                runtime: self.clone(),
                raw: event,
            }),
        })
    }

    /// Return device elapsed time between two completed timing events, in milliseconds.
    ///
    /// Both events are synchronized before querying elapsed time. If CUDA cannot confirm either
    /// event completed, this returns that error and does not report a possibly invalid duration.
    ///
    /// # Errors
    ///
    /// Returns an error if events belong to different runtime/device pairs, either event cannot
    /// be confirmed complete, or CUDA rejects the elapsed-time query.
    pub fn elapsed_time_ms(
        &self,
        start: &CudaTimingEventHandle,
        end: &CudaTimingEventHandle,
    ) -> Result<f32, CudaError> {
        self.ensure_same_runtime(&start.inner.runtime)?;
        self.ensure_same_runtime(&end.inner.runtime)?;
        start.synchronize()?;
        end.synchronize()?;
        let mut milliseconds = 0.0_f32;
        unsafe {
            crate::ffi::invoke_cudaEventElapsedTime(
                self,
                &raw mut milliseconds,
                start.inner.raw,
                end.inner.raw,
            )
        }?;
        Ok(milliseconds)
    }

    fn device_name(&self, device: CudaDevice) -> Result<String, CudaError> {
        let mut name = [0_i8; 256];
        unsafe {
            crate::ffi::invoke_cuDeviceGetName(
                self,
                name.as_mut_ptr(),
                c_int::try_from(name.len()).expect("fixed name buffer fits c_int"),
                device,
            )
        }?;
        Ok(bounded_device_name(&name))
    }

    fn cuda_set_device(&self, device: CudaDevice) -> Result<(), CudaError> {
        crate::ffi::invoke_cudaSetDevice(self, device)
    }

    fn ensure_same_runtime(&self, other: &Self) -> Result<(), CudaError> {
        if Arc::ptr_eq(&self.0.library, &other.0.library)
            && Arc::ptr_eq(&self.0.driver, &other.0.driver)
            && self.0.ordinal == other.0.ordinal
            && self.0.device == other.0.device
        {
            Ok(())
        } else {
            Err(CudaError::DifferentRuntime)
        }
    }
}

/// Basic physical-device facts queried without depending on versioned CUDA property layouts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CudaDeviceInfo {
    pub index: i32,
    pub name: String,
    pub vendor: String,
    /// CUDA compute architecture, expressed as `sm_XX`.
    pub architecture: Option<String>,
    /// GPU generation is not inferred from the marketing name.
    pub generation: Option<String>,
    /// PCI bus ID from CUDA driver API, when supported.
    /// This existing activation check uses PCI location. Additional UUID and hardware attributes
    /// are available through the explicitly requested Driver API physical-device facts.
    pub pci_bus_id: Option<String>,
    pub total_memory: u64,
}

fn bounded_device_name(buffer: &[c_char]) -> String {
    let bytes: Vec<u8> = buffer
        .iter()
        .take_while(|&&byte| byte != 0)
        .map(|&byte| byte.to_ne_bytes()[0])
        .collect();
    String::from_utf8_lossy(&bytes).into_owned()
}

/// Point-in-time device memory telemetry from CUDA.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CudaMemoryInfo {
    pub free_bytes: u64,
    pub total_bytes: u64,
}

fn cuda_memory_pool_snapshot(id: PcuMemoryPoolId, info: CudaMemoryInfo) -> PcuMemoryPoolSnapshot {
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

/// Shared owner for one CUDA device allocation.
struct DeviceAllocation {
    runtime: CudaRuntime,
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

#[derive(Clone, Copy, PartialEq, Eq)]
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

fn reassign_allocation_access_guards(
    guards: &mut [&mut AllocationAccessGuard],
    from: usize,
    to: usize,
) -> Result<(), ()> {
    // Preflight the whole set before changing any shared gate. This is all-or-nothing because
    // these Rc<Cell<_>> gates are accessed on the current host thread and cannot race here.
    let read_guards = guards.iter().map(|guard| &**guard).collect::<Vec<_>>();
    if !allocation_access_guards_can_reassign(&read_guards, from, to) {
        return Err(());
    }
    let mut groups: Vec<(Rc<AllocationAccess>, usize)> = Vec::new();
    for guard in guards.iter() {
        if let Some((_, count)) = groups
            .iter_mut()
            .find(|(access, _)| Rc::ptr_eq(access, &guard.access))
        {
            *count += 1;
        } else {
            groups.push((Rc::clone(&guard.access), 1));
        }
    }
    for (access, count) in groups {
        access.state.set(AllocationAccessState::Stream {
            identity: to,
            leases: count,
        });
    }
    for guard in guards.iter_mut() {
        guard.kind = AllocationAccessKind::Stream(to);
    }
    Ok(())
}

fn allocation_access_guards_can_reassign(
    guards: &[&AllocationAccessGuard],
    from: usize,
    to: usize,
) -> bool {
    if from == to
        || guards
            .iter()
            .any(|guard| guard.kind != AllocationAccessKind::Stream(from))
    {
        return false;
    }
    let mut groups: Vec<(&Rc<AllocationAccess>, usize)> = Vec::new();
    for guard in guards {
        if let Some((_, count)) = groups
            .iter_mut()
            .find(|(access, _)| Rc::ptr_eq(access, &guard.access))
        {
            *count += 1;
        } else {
            groups.push((&guard.access, 1));
        }
    }
    groups.iter().all(|(access, count)| {
        access.state.get()
            == (AllocationAccessState::Stream {
                identity: from,
                leases: *count,
            })
    })
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
            // safe failure mode if internal ownerscuda invariants are ever violated.
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

/// Opaque identifier for a host readback owned by one completion batch.
///
/// Readback IDs are neither transferable across batches nor usable after their bytes are taken.
#[derive(Clone)]
pub struct CudaReadbackId {
    identity: Rc<()>,
    index: usize,
}

/// Fixed, privately owned host destinations for asynchronous device-to-host copies.
struct CudaReadbackStorage {
    identity: Rc<()>,
    buffers: Vec<Option<CudaReadbackBuffer>>,
}

struct CudaReadbackBuffer {
    bytes: Box<[u8]>,
    queued: bool,
}

const fn readback_is_complete(
    final_event_present: bool,
    resources_present: bool,
    dependencies_present: bool,
) -> bool {
    !final_event_present && !resources_present && !dependencies_present
}

impl CudaReadbackStorage {
    fn new() -> Self {
        Self {
            identity: Rc::new(()),
            buffers: Vec::new(),
        }
    }

    fn allocate(&mut self, bytes: usize) -> CudaReadbackId {
        let index = self.buffers.len();
        self.buffers.push(Some(CudaReadbackBuffer {
            bytes: vec![0; bytes].into_boxed_slice(),
            queued: false,
        }));
        CudaReadbackId {
            identity: Rc::clone(&self.identity),
            index,
        }
    }

    fn prepare_destination(
        &mut self,
        id: &CudaReadbackId,
        offset: usize,
        bytes: usize,
    ) -> Result<*mut u8, CudaError> {
        let buffer = self.buffer(id)?;
        if buffer.queued {
            return Err(CudaError::ReadbackAlreadyQueued);
        }
        validate_buffer_range(buffer.bytes.len(), offset, bytes)?;
        let buffer = self.buffer_mut(id)?;
        buffer.queued = true;
        Ok(buffer.bytes.as_mut_ptr().wrapping_add(offset))
    }

    fn validate_destination(
        &self,
        id: &CudaReadbackId,
        offset: usize,
        bytes: usize,
    ) -> Result<(), CudaError> {
        let buffer = self.buffer(id)?;
        if buffer.queued {
            return Err(CudaError::ReadbackAlreadyQueued);
        }
        validate_buffer_range(buffer.bytes.len(), offset, bytes)
    }

    fn read(&self, id: &CudaReadbackId) -> Result<&[u8], CudaError> {
        let buffer = self.buffer(id)?;
        if !buffer.queued && !buffer.bytes.is_empty() {
            return Err(CudaError::ReadbackNotQueued);
        }
        Ok(&buffer.bytes)
    }

    fn take(&mut self, id: &CudaReadbackId) -> Result<Box<[u8]>, CudaError> {
        let buffer = self.buffer(id)?;
        if !buffer.queued && !buffer.bytes.is_empty() {
            return Err(CudaError::ReadbackNotQueued);
        }
        self.buffers
            .get_mut(id.index)
            .filter(|_| Rc::ptr_eq(&self.identity, &id.identity))
            .and_then(Option::take)
            .map(|buffer| buffer.bytes)
            .ok_or(CudaError::InvalidReadbackId)
    }

    fn buffer(&self, id: &CudaReadbackId) -> Result<&CudaReadbackBuffer, CudaError> {
        self.buffers
            .get(id.index)
            .filter(|_| Rc::ptr_eq(&self.identity, &id.identity))
            .and_then(Option::as_ref)
            .ok_or(CudaError::InvalidReadbackId)
    }

    fn buffer_mut(&mut self, id: &CudaReadbackId) -> Result<&mut CudaReadbackBuffer, CudaError> {
        self.buffers
            .get_mut(id.index)
            .filter(|_| Rc::ptr_eq(&self.identity, &id.identity))
            .and_then(Option::as_mut)
            .ok_or(CudaError::InvalidReadbackId)
    }

    fn quarantine_and_forget(&mut self) {
        for buffer in self.buffers.drain(..).flatten() {
            std::mem::forget(buffer);
        }
    }

    fn has_live_readbacks(&self) -> bool {
        self.buffers.iter().any(Option::is_some)
    }
}

impl Drop for DeviceAllocation {
    fn drop(&mut self) {
        let _ = unsafe { crate::ffi::invoke_cudaFree(&self.runtime, self.pointer) };
    }
}

/// Owned CUDA device-memory allocation. Clones share the same allocation.
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
    /// Returns an error when the range is invalid, another operation holds the allocation, or CUDA
    /// reports a copy or completion failure.
    pub fn copy_from(&mut self, source: &[u8]) -> Result<(), CudaError> {
        self.copy_from_at(0, source)
    }
    /// Copy bytes from host memory into a checked byte range of this allocation.
    ///
    /// # Errors
    ///
    /// Returns an error when the range is invalid, another operation holds the allocation, or CUDA
    /// reports a copy or completion failure.
    pub fn copy_from_at(&mut self, offset: usize, source: &[u8]) -> Result<(), CudaError> {
        self.check_range(offset, source.len())?;
        if source.is_empty() {
            return Ok(());
        }
        let lease = self.acquire_access()?;
        let result = unsafe {
            crate::ffi::invoke_cudaMemcpy(
                &self.allocation.runtime,
                (self.allocation.pointer.cast::<u8>().wrapping_add(offset)).cast(),
                source.as_ptr().cast(),
                source.len(),
                CUDA_MEMCPY_HOST_TO_DEVICE,
            )
        };
        lease.finish_synchronous(result)
    }
    /// Copy this allocation into host memory.
    ///
    /// # Errors
    ///
    /// Returns an error when the range is invalid, another operation holds the allocation, or CUDA
    /// reports a copy or completion failure.
    pub fn copy_to(&self, destination: &mut [u8]) -> Result<(), CudaError> {
        self.copy_to_at(0, destination)
    }
    /// Copy a checked byte range of this allocation into host memory.
    ///
    /// # Errors
    ///
    /// Returns an error when the range is invalid, another operation holds the allocation, or CUDA
    /// reports a copy or completion failure.
    pub fn copy_to_at(&self, offset: usize, destination: &mut [u8]) -> Result<(), CudaError> {
        self.check_range(offset, destination.len())?;
        if destination.is_empty() {
            return Ok(());
        }
        let lease = self.acquire_access()?;
        let result = unsafe {
            crate::ffi::invoke_cudaMemcpy(
                &self.allocation.runtime,
                destination.as_mut_ptr().cast(),
                (self.allocation.pointer.cast::<u8>().wrapping_add(offset)).cast(),
                destination.len(),
                CUDA_MEMCPY_DEVICE_TO_HOST,
            )
        };
        lease.finish_synchronous(result)
    }
    /// Copy bytes from another device allocation.
    /// Copy bytes from a different allocation belonging to the same CUDA runtime and device.
    ///
    /// # Errors
    ///
    /// Returns an error when the runtimes differ, either buffer is too small or busy, or CUDA
    /// reports a copy or completion failure.
    pub fn copy_from_device(&mut self, source: &Self, bytes: usize) -> Result<(), CudaError> {
        self.allocation
            .runtime
            .ensure_same_runtime(&source.allocation.runtime)?;
        self.check_length(bytes)?;
        source.check_length(bytes)?;
        if Rc::ptr_eq(&self.allocation, &source.allocation) {
            return Err(CudaError::Busy);
        }
        let destination_lease = self.acquire_access()?;
        let source_lease = source.acquire_access()?;
        let result = unsafe {
            crate::ffi::invoke_cudaMemcpy(
                &self.allocation.runtime,
                self.allocation.pointer,
                source.allocation.pointer,
                bytes,
                CUDA_MEMCPY_DEVICE_TO_DEVICE,
            )
        };
        destination_lease.finish_synchronous_with(source_lease, result)
    }
    fn check_length(&self, bytes: usize) -> Result<(), CudaError> {
        if bytes > self.allocation.bytes {
            Err(CudaError::BufferTooSmall {
                allocation: self.allocation.bytes,
                requested: bytes,
            })
        } else {
            Ok(())
        }
    }

    fn check_range(&self, offset: usize, bytes: usize) -> Result<(), CudaError> {
        validate_buffer_range(self.allocation.bytes, offset, bytes)
    }

    fn acquire_access(&self) -> Result<DeviceAccessLease, CudaError> {
        let guard = self
            .allocation
            .access
            .acquire()
            .map_err(|()| CudaError::Busy)?;
        Ok(DeviceAccessLease {
            allocation: Rc::clone(&self.allocation),
            guard,
        })
    }

    #[cfg(test)]
    pub(crate) fn with_access_lease_for_test<R>(
        &self,
        inspect: impl FnOnce() -> R,
    ) -> Result<R, CudaError> {
        let lease = self.acquire_access()?;
        let result = inspect();
        drop(lease);
        Ok(result)
    }

    pub(crate) fn validate_access_available(&self) -> Result<(), CudaError> {
        if self.allocation.access.state.get() == AllocationAccessState::Idle {
            Ok(())
        } else {
            Err(CudaError::Busy)
        }
    }

    fn acquire_stream_access(
        &self,
        stream: &CudaStreamHandle,
    ) -> Result<DeviceAccessLease, CudaError> {
        let identity = Rc::as_ptr(&stream.inner) as usize;
        let guard = self
            .allocation
            .access
            .acquire_stream(identity)
            .map_err(|()| CudaError::Busy)?;
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

    fn finish_synchronous(self, result: Result<(), CudaError>) -> Result<(), CudaError> {
        match result {
            Ok(()) => Ok(()),
            Err(error) => {
                if crate::ffi::invoke_cudaDeviceSynchronize(&self.allocation.runtime).is_err() {
                    std::mem::forget(self);
                }
                Err(error)
            }
        }
    }

    fn finish_synchronous_with(
        self,
        other: Self,
        result: Result<(), CudaError>,
    ) -> Result<(), CudaError> {
        match result {
            Ok(()) => Ok(()),
            Err(error) => {
                if crate::ffi::invoke_cudaDeviceSynchronize(&self.allocation.runtime).is_err() {
                    std::mem::forget(self);
                    std::mem::forget(other);
                }
                Err(error)
            }
        }
    }
}

struct StreamInner {
    runtime: CudaRuntime,
    raw: CudaStream,
}
impl Drop for StreamInner {
    fn drop(&mut self) {
        let _ = unsafe { crate::ffi::invoke_cudaStreamDestroy(&self.runtime, self.raw) };
    }
}
/// Shared owner for a CUDA stream.
#[derive(Clone)]
pub struct CudaStreamHandle {
    inner: Rc<StreamInner>,
}
impl CudaStreamHandle {
    /// Return the raw stream pointer for internal backend FFI integration.
    #[allow(dead_code)] // Used by asynchronous library integrations in sibling modules.
    pub(crate) fn raw_stream(&self) -> *mut c_void {
        self.inner.raw
    }

    /// Check whether this stream belongs to the supplied runtime/device pair.
    pub(crate) fn belongs_to_runtime(&self, runtime: &CudaRuntime) -> bool {
        runtime.ensure_same_runtime(&self.inner.runtime).is_ok()
    }

    /// Wait until all operations queued on this stream have completed.
    ///
    /// # Errors
    ///
    /// Returns the CUDA synchronization error, if any.
    pub fn synchronize(&self) -> Result<(), CudaError> {
        unsafe { crate::ffi::invoke_cudaStreamSynchronize(&self.inner.runtime, self.inner.raw) }
    }
    /// Record an event after all work already queued on this stream.
    ///
    /// # Errors
    ///
    /// Returns an error when the event belongs to another runtime or CUDA fails to record it.
    pub fn record(&self, event: &CudaEventHandle) -> Result<(), CudaError> {
        self.inner
            .runtime
            .ensure_same_runtime(&event.inner.runtime)?;
        unsafe {
            crate::ffi::invoke_cudaEventRecord(&self.inner.runtime, event.inner.raw, self.inner.raw)
        }
    }

    /// Record a timing-enabled event after all work already queued on this stream.
    ///
    /// # Errors
    ///
    /// Returns an error when the event belongs to another runtime or CUDA fails to record it.
    pub fn record_timing(&self, event: &CudaTimingEventHandle) -> Result<(), CudaError> {
        self.inner
            .runtime
            .ensure_same_runtime(&event.inner.runtime)?;
        unsafe {
            crate::ffi::invoke_cudaEventRecord(&self.inner.runtime, event.inner.raw, self.inner.raw)
        }
    }
}

struct EventInner {
    runtime: CudaRuntime,
    raw: CudaEvent,
}
impl Drop for EventInner {
    fn drop(&mut self) {
        let _ = unsafe { crate::ffi::invoke_cudaEventDestroy(&self.runtime, self.raw) };
    }
}
/// Shared owner for a CUDA event.
#[derive(Clone)]
pub struct CudaEventHandle {
    inner: Rc<EventInner>,
}
impl CudaEventHandle {
    /// Wait until the event has completed.
    ///
    /// # Errors
    ///
    /// Returns the CUDA synchronization error, if any.
    pub fn synchronize(&self) -> Result<(), CudaError> {
        unsafe { crate::ffi::invoke_cudaEventSynchronize(&self.inner.runtime, self.inner.raw) }
    }
}

/// Shared owner for a CUDA event created with timing enabled.
#[derive(Clone)]
pub struct CudaTimingEventHandle {
    inner: Rc<EventInner>,
}
impl CudaTimingEventHandle {
    /// Wait until the event has completed.
    ///
    /// # Errors
    ///
    /// Returns the CUDA synchronization error, if any.
    pub fn synchronize(&self) -> Result<(), CudaError> {
        unsafe { crate::ffi::invoke_cudaEventSynchronize(&self.inner.runtime, self.inner.raw) }
    }
}

struct ModuleInner {
    runtime: CudaRuntime,
    raw: ModuleHandle,
}
impl Drop for ModuleInner {
    fn drop(&mut self) {
        let _ = unsafe { crate::ffi::invoke_cuModuleUnload(&self.runtime, self.raw) };
    }
}
/// Loaded CUDA code object.
#[derive(Clone)]
pub struct CudaModule {
    inner: Rc<ModuleInner>,
}
impl CudaModule {
    /// Resolve one named kernel function from this module.
    ///
    /// # Errors
    ///
    /// Returns an error when CUDA cannot resolve the named function.
    pub fn function(&self, name: &CStr) -> Result<CudaKernel, CudaError> {
        let mut raw = ptr::null_mut();
        unsafe {
            crate::ffi::invoke_cuModuleGetFunction(
                &self.inner.runtime,
                &raw mut raw,
                self.inner.raw,
                name.as_ptr(),
            )
        }?;
        Ok(CudaKernel {
            module: self.inner.clone(),
            raw,
        })
    }
}

/// A named function that keeps its containing CUDA module loaded.
#[derive(Clone)]
pub struct CudaKernel {
    module: Rc<ModuleInner>,
    raw: KernelHandle,
}

/// One kernel argument supplied as host bytes or a device allocation.
///
/// `Bytes` is copied into naturally aligned storage before launch. The kernel's actual parameter
/// types and order remain the caller's responsibility.
pub enum CudaKernelArgument<'a> {
    Bytes(&'a [u8]),
    Buffer(&'a DeviceBuffer),
}

#[repr(align(16))]
struct AlignedKernelWord {
    _word: usize,
}

const INLINE_KERNEL_PARAMETERS: usize = 8;

const fn ensure_batch_open(failed: bool) -> Result<(), CudaError> {
    if failed {
        Err(CudaError::BatchPoisoned)
    } else {
        Ok(())
    }
}

fn validate_device_copy(
    destination_bytes: usize,
    source_bytes: usize,
    bytes: usize,
    same_allocation: bool,
) -> Result<(), CudaError> {
    for allocation in [destination_bytes, source_bytes] {
        if bytes > allocation {
            return Err(CudaError::BufferTooSmall {
                allocation,
                requested: bytes,
            });
        }
    }
    if same_allocation && bytes != 0 {
        return Err(CudaError::Busy);
    }
    Ok(())
}

fn validate_buffer_range(
    allocation_bytes: usize,
    offset: usize,
    bytes: usize,
) -> Result<(), CudaError> {
    if offset
        .checked_add(bytes)
        .is_none_or(|end| end > allocation_bytes)
    {
        Err(CudaError::BufferTooSmall {
            allocation: allocation_bytes.saturating_sub(offset),
            requested: bytes,
        })
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

    fn can_handoff_stream(&self, from: usize, to: usize) -> bool {
        let guards = self.inline[..self.inline_len]
            .iter()
            .flatten()
            .chain(self.overflow.iter())
            .map(|lease| &lease.guard)
            .collect::<Vec<_>>();
        allocation_access_guards_can_reassign(&guards, from, to)
    }
}

fn can_inline_kernel_parameters(arguments: &[CudaKernelArgument<'_>]) -> bool {
    arguments.len() <= INLINE_KERNEL_PARAMETERS
        && arguments.iter().all(|argument| match argument {
            CudaKernelArgument::Bytes(bytes) => bytes.len() <= size_of::<AlignedKernelWord>(),
            CudaKernelArgument::Buffer(_) => {
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
    _module: Option<Rc<ModuleInner>>,
    _external_owner: Option<Rc<dyn Any>>,
    // Each unique lease owns the allocation as well as its busy gate until completion.
    access_leases: LaunchAccessLeases,
    stream: CudaStreamHandle,
}
impl LaunchResources {
    fn quarantine(&self) {
        self.access_leases.quarantine();
    }

    fn belongs_to_stream(&self, stream: &CudaStreamHandle) -> bool {
        Rc::ptr_eq(&self.stream.inner, &stream.inner)
    }

    fn handoff_accesses_to_stream(&mut self, stream: &CudaStreamHandle) -> Result<(), ()> {
        let from = Rc::as_ptr(&self.stream.inner) as usize;
        let to = Rc::as_ptr(&stream.inner) as usize;
        reassign_launch_resource_accesses(std::slice::from_mut(self), from, to)
    }

    fn can_handoff_accesses_to_stream(&self, stream: &CudaStreamHandle) -> bool {
        let from = Rc::as_ptr(&self.stream.inner) as usize;
        let to = Rc::as_ptr(&stream.inner) as usize;
        self.access_leases.can_handoff_stream(from, to)
    }
}

fn reassign_launch_resource_accesses(
    resources: &mut [LaunchResources],
    from: usize,
    to: usize,
) -> Result<(), ()> {
    let mut guards = Vec::new();
    append_launch_resource_access_guards(resources, &mut guards);
    reassign_allocation_access_guards(&mut guards, from, to)
}

fn append_launch_resource_access_guards<'a>(
    resources: &'a mut [LaunchResources],
    guards: &mut Vec<&'a mut AllocationAccessGuard>,
) {
    for resource in resources {
        guards.extend(
            resource.access_leases.inline[..resource.access_leases.inline_len]
                .iter_mut()
                .flatten()
                .chain(resource.access_leases.overflow.iter_mut())
                .map(|lease| &mut lease.guard),
        );
    }
}

fn append_launch_resource_access_guards_readonly<'a>(
    resources: &'a [LaunchResources],
    guards: &mut Vec<&'a AllocationAccessGuard>,
) {
    for resource in resources {
        guards.extend(
            resource.access_leases.inline[..resource.access_leases.inline_len]
                .iter()
                .flatten()
                .chain(resource.access_leases.overflow.iter())
                .map(|lease| &lease.guard),
        );
    }
}

/// Completion token for a kernel launch. Dropping it waits for completion; if CUDA cannot confirm
/// completion, the backend leaks retained resources rather than freeing memory still in use.
pub struct CudaCompletion {
    event: Option<CudaEventHandle>,
    resources: Option<LaunchResources>,
}
impl CudaCompletion {
    /// Wait for the launch to complete. On an error the token retains its resources and can be
    /// retried or dropped (drop retries and leaks resources if CUDA still cannot confirm completion).
    ///
    /// # Errors
    ///
    /// Returns the CUDA synchronization error while retaining the launch resources for a retry.
    pub fn wait(&mut self) -> Result<(), CudaError> {
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
impl Drop for CudaCompletion {
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
            // CUDA reported an error and did not confirm that the queued kernel has stopped using
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

/// Completion for several launches queued in order on one CUDA stream.
///
/// Every launch's resources remain retained until the final event completes. Use
/// [`CudaCompletionBatch::new`] before submitting nodes, push each launch completion, then call
/// [`CudaCompletionBatch::finish`] after the final launch has been queued.
pub struct CudaCompletionBatch {
    stream: CudaStreamHandle,
    launch_events: Vec<CudaEventHandle>,
    resources: Vec<LaunchResources>,
    dependencies: Vec<CudaProducerCompletion>,
    readbacks: CudaReadbackStorage,
    queue_markers: SmallVec<[Rc<dyn Any>; 1]>,
    failed: bool,
    timing: Option<CudaBatchTiming>,
}

#[allow(clippy::large_enum_variant)] // Keep launch tokens inline; they are moved infrequently.
enum CudaProducerCompletion {
    Launch(CudaCompletion),
    Batch(CudaBatchCompletion),
}

struct CudaBatchTiming {
    start: Option<CudaTimingEventHandle>,
    end: Option<CudaTimingEventHandle>,
    segments_ms: Rc<RefCell<Vec<f32>>>,
}

struct CudaBatchTimingSegment {
    start: CudaTimingEventHandle,
    end: CudaTimingEventHandle,
    segments_ms: Rc<RefCell<Vec<f32>>>,
}

impl CudaCompletionBatch {
    /// Start collecting completions for launches queued on `stream`.
    #[must_use]
    pub fn new(stream: &CudaStreamHandle) -> Self {
        Self {
            stream: stream.clone(),
            launch_events: Vec::new(),
            resources: Vec::new(),
            dependencies: Vec::new(),
            readbacks: CudaReadbackStorage::new(),
            queue_markers: SmallVec::new(),
            failed: false,
            timing: None,
        }
    }

    /// Start collecting completions and device elapsed-time measurements for each segment.
    /// Timing events are only created when a launch is submitted.
    #[must_use]
    pub fn new_timed(stream: &CudaStreamHandle) -> Self {
        Self {
            stream: stream.clone(),
            launch_events: Vec::new(),
            resources: Vec::new(),
            dependencies: Vec::new(),
            readbacks: CudaReadbackStorage::new(),
            queue_markers: SmallVec::new(),
            failed: false,
            timing: Some(CudaBatchTiming {
                start: None,
                end: None,
                segments_ms: Rc::new(RefCell::new(Vec::new())),
            }),
        }
    }

    /// Return the stream used by this batch for internal asynchronous-library integration.
    pub(crate) const fn stream_handle(&self) -> &CudaStreamHandle {
        &self.stream
    }

    pub(crate) fn has_queue_marker(&self, marker: &Rc<dyn Any>) -> bool {
        has_queue_marker(&self.queue_markers, marker)
    }

    pub(crate) fn register_queue_marker(&mut self, marker: Rc<dyn Any>) -> Result<(), CudaError> {
        ensure_batch_open(self.failed)?;
        register_queue_marker(&mut self.queue_markers, marker);
        Ok(())
    }

    /// Retain an external asynchronous operation's buffers and owner through this batch's final
    /// event. Call this before enqueueing the operation, and ensure the external library is bound
    /// to [`Self::stream_handle`]. Repeated buffer references are deduplicated. If the external
    /// call reports an error after possibly enqueueing work, keep this batch and finish or drop it
    /// normally so its final event or stream synchronization proves quiescence.
    ///
    /// # Errors
    ///
    /// Returns a runtime mismatch, busy allocation, or [`CudaError::BatchPoisoned`].
    pub(crate) fn retain_external_operation(
        &mut self,
        buffers: &[&DeviceBuffer],
        owner: Rc<dyn Any>,
    ) -> Result<(), CudaError> {
        ensure_batch_open(self.failed)?;
        for buffer in buffers {
            self.stream
                .inner
                .runtime
                .ensure_same_runtime(&buffer.allocation.runtime)?;
        }

        let mut access_leases = LaunchAccessLeases::new();
        let mut unique = SmallVec::<[&DeviceBuffer; 8]>::with_capacity(buffers.len());
        for buffer in buffers {
            if unique
                .iter()
                .any(|known| Rc::ptr_eq(&known.allocation, &buffer.allocation))
            {
                continue;
            }
            // If an acquisition fails, already acquired leases drop here and restore their gates.
            access_leases.push(buffer.acquire_stream_access(&self.stream)?);
            unique.push(buffer);
        }
        self.resources.push(LaunchResources {
            _module: None,
            _external_owner: Some(owner),
            access_leases,
            stream: self.stream.clone(),
        });
        Ok(())
    }

    /// Return device elapsed times for segments whose completion tokens have been waited.
    #[must_use]
    pub fn timings_ms(&self) -> Vec<f32> {
        self.timing
            .as_ref()
            .map_or_else(Vec::new, |timing| timing.segments_ms.borrow().clone())
    }

    /// Allocate a fixed, batch-owned host destination for an asynchronous device-to-host copy.
    ///
    /// The returned identifier is valid only with this batch and its final completion. The
    /// storage remains private and cannot be read until the completion's complete dependency
    /// chain has succeeded. Zero-byte results are valid and require no device operation.
    ///
    /// # Errors
    ///
    /// Returns [`CudaError::BatchPoisoned`] if this batch has already failed.
    pub fn allocate_readback(&mut self, bytes: usize) -> Result<CudaReadbackId, CudaError> {
        ensure_batch_open(self.failed)?;
        Ok(self.readbacks.allocate(bytes))
    }

    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.resources.is_empty() && self.launch_events.is_empty() && self.dependencies.is_empty()
    }

    /// Add one launch completion to this batch.
    ///
    /// The launch must have been submitted to the batch stream. Its event and resources are
    /// retained until the batch's final event is synchronized.
    ///
    /// # Errors
    ///
    /// Returns [`CudaError::DifferentStream`] when the completion belongs to another stream.
    pub fn push(&mut self, mut completion: CudaCompletion) -> Result<(), CudaError> {
        ensure_batch_open(self.failed)?;
        if completion
            .resources
            .as_ref()
            .is_some_and(|resources| !resources.belongs_to_stream(&self.stream))
        {
            return Err(CudaError::DifferentStream);
        }
        if let Some(event) = completion.event.take() {
            self.launch_events.push(event);
        }
        if let Some(resources) = completion.resources.take() {
            self.resources.push(resources);
        }
        Ok(())
    }

    /// Queue a device-to-device copy on this batch's stream.
    ///
    /// Both allocations remain leased and owned until the batch's final event completes. If CUDA
    /// reports an enqueue error, the stream is synchronized before releasing the leases; when
    /// quiescence cannot be established, the allocations and their gates are quarantined.
    ///
    /// # Errors
    ///
    /// Returns a runtime mismatch, invalid byte count, busy allocation, CUDA enqueue error, or
    /// [`CudaError::BatchPoisoned`] if this batch has already failed.
    pub fn copy_device_to_device(
        &mut self,
        destination: &DeviceBuffer,
        source: &DeviceBuffer,
        bytes: usize,
    ) -> Result<(), CudaError> {
        ensure_batch_open(self.failed)?;
        self.stream
            .inner
            .runtime
            .ensure_same_runtime(&destination.allocation.runtime)?;
        self.stream
            .inner
            .runtime
            .ensure_same_runtime(&source.allocation.runtime)?;
        validate_device_copy(
            destination.allocation.bytes,
            source.allocation.bytes,
            bytes,
            Rc::ptr_eq(&destination.allocation, &source.allocation),
        )?;
        if bytes == 0 {
            return Ok(());
        }

        let mut access_leases = LaunchAccessLeases::new();
        access_leases.push(destination.acquire_stream_access(&self.stream)?);
        access_leases.push(source.acquire_stream_access(&self.stream)?);
        // Retain both owners before enqueue. CUDA can report an error after submitting work, so
        // the same batch synchronization/quarantine path used by failed launches must own them.
        self.resources.push(LaunchResources {
            _module: None,
            _external_owner: None,
            access_leases,
            stream: self.stream.clone(),
        });
        let enqueue = unsafe {
            crate::ffi::invoke_cudaMemcpyAsync(
                &self.stream.inner.runtime,
                destination.allocation.pointer,
                source.allocation.pointer,
                bytes,
                CUDA_MEMCPY_DEVICE_TO_DEVICE,
                self.stream.inner.raw,
            )
        };
        if let Err(error) = enqueue {
            self.failed = true;
            self.release_after_stream_sync();
            return Err(error);
        }
        Ok(())
    }

    /// Queue an asynchronous copy of a device allocation into a batch-owned readback.
    ///
    /// Use [`Self::copy_device_to_host_at`] to select source and destination offsets. The
    /// readback bytes are available from the finished completion only after its final event and
    /// every recursively retained dependency have all waited successfully.
    ///
    /// # Errors
    ///
    /// Returns a runtime mismatch, invalid range or readback ID, busy source allocation, CUDA
    /// enqueue error, or [`CudaError::BatchPoisoned`] if this batch has already failed.
    pub fn copy_device_to_host(
        &mut self,
        source: &DeviceBuffer,
        readback: &CudaReadbackId,
        bytes: usize,
    ) -> Result<(), CudaError> {
        self.copy_device_to_host_at(source, 0, readback, 0, bytes)
    }

    /// Queue an asynchronous copy from a checked device range into a checked batch readback range.
    ///
    /// The source allocation stays leased through the batch's final event. Host storage has a
    /// fixed address for the duration of the copy, and is only exposed after a successful full
    /// wait. Empty ranges are validated no-ops and do not acquire the source allocation.
    ///
    /// # Errors
    ///
    /// Returns a runtime mismatch, invalid source/destination range or readback ID, busy source
    /// allocation, CUDA enqueue error, or [`CudaError::BatchPoisoned`] if this batch has already
    /// failed.
    pub fn copy_device_to_host_at(
        &mut self,
        source: &DeviceBuffer,
        source_offset: usize,
        readback: &CudaReadbackId,
        destination_offset: usize,
        bytes: usize,
    ) -> Result<(), CudaError> {
        ensure_batch_open(self.failed)?;
        self.stream
            .inner
            .runtime
            .ensure_same_runtime(&source.allocation.runtime)?;
        source.check_range(source_offset, bytes)?;
        self.readbacks
            .validate_destination(readback, destination_offset, bytes)?;
        if bytes == 0 {
            return Ok(());
        }

        let mut access_leases = LaunchAccessLeases::new();
        access_leases.push(source.acquire_stream_access(&self.stream)?);
        let destination =
            self.readbacks
                .prepare_destination(readback, destination_offset, bytes)?;
        self.resources.push(LaunchResources {
            _module: None,
            _external_owner: None,
            access_leases,
            stream: self.stream.clone(),
        });
        let enqueue = unsafe {
            crate::ffi::invoke_cudaMemcpyAsync(
                &self.stream.inner.runtime,
                destination.cast(),
                source
                    .allocation
                    .pointer
                    .cast::<u8>()
                    .wrapping_add(source_offset)
                    .cast(),
                bytes,
                CUDA_MEMCPY_DEVICE_TO_HOST,
                self.stream.inner.raw,
            )
        };
        if let Err(error) = enqueue {
            self.failed = true;
            self.release_after_stream_sync();
            return Err(error);
        }
        Ok(())
    }

    /// Queue an owned host-to-device copy on this batch's stream.
    ///
    /// The immutable host payload and destination allocation remain retained until the batch's
    /// final event completes. The payload can be shared with graph nodes by cloning its `Arc`;
    /// the batch keeps its own owner until CUDA confirms completion. An empty payload is a
    /// validated no-op and does not acquire the destination's busy gate.
    ///
    /// # Errors
    ///
    /// Returns a runtime mismatch, invalid destination range, busy destination, CUDA enqueue
    /// error, or [`CudaError::BatchPoisoned`] if this batch has already failed.
    pub fn copy_host_to_device(
        &mut self,
        destination: &DeviceBuffer,
        source: Arc<[u8]>,
    ) -> Result<(), CudaError> {
        self.copy_host_to_device_at(destination, 0, source)
    }

    /// Queue an owned host-to-device copy at `offset` on this batch's stream.
    ///
    /// The byte range is checked before acquiring the allocation lease or submitting CUDA work.
    /// The host source must be represented by an immutable `Arc<[u8]>`; the batch retains that
    /// owner through its final event, so callers may release their own handle immediately after
    /// this method returns.
    ///
    /// # Errors
    ///
    /// Returns a runtime mismatch, invalid destination range, busy destination, CUDA enqueue
    /// error, or [`CudaError::BatchPoisoned`] if this batch has already failed.
    pub fn copy_host_to_device_at(
        &mut self,
        destination: &DeviceBuffer,
        offset: usize,
        source: Arc<[u8]>,
    ) -> Result<(), CudaError> {
        ensure_batch_open(self.failed)?;
        self.stream
            .inner
            .runtime
            .ensure_same_runtime(&destination.allocation.runtime)?;
        validate_buffer_range(destination.allocation.bytes, offset, source.len())?;
        if source.is_empty() {
            return Ok(());
        }

        let bytes = source.len();
        let source_pointer = source.as_ptr().cast::<c_void>();
        let owner: Rc<dyn Any> = Rc::new(source);
        self.retain_external_operation(&[destination], owner)?;
        let enqueue = unsafe {
            crate::ffi::invoke_cudaMemcpyAsync(
                &self.stream.inner.runtime,
                destination
                    .allocation
                    .pointer
                    .cast::<u8>()
                    .wrapping_add(offset)
                    .cast(),
                source_pointer,
                bytes,
                CUDA_MEMCPY_HOST_TO_DEVICE,
                self.stream.inner.raw,
            )
        };
        if let Err(error) = enqueue {
            self.failed = true;
            self.release_after_stream_sync();
            return Err(error);
        }
        Ok(())
    }

    /// Check whether an unconsumed launch completion can be attached as a dependency.
    ///
    /// This performs no CUDA calls and does not change the completion or access gates. The result
    /// remains valid until another operation changes one of the allocation gates; these handles
    /// are thread-local (`Rc<Cell<_>>`), so no other thread can race this check.
    pub(crate) fn can_wait_for(&self, completion: &CudaCompletion) -> Result<bool, CudaError> {
        ensure_batch_open(self.failed)?;
        let producer_runtime = completion
            .event
            .as_ref()
            .map(|event| &event.inner.runtime)
            .or_else(|| {
                completion
                    .resources
                    .as_ref()
                    .map(|resources| &resources.stream.inner.runtime)
            });
        if let Some(runtime) = producer_runtime {
            self.stream.inner.runtime.ensure_same_runtime(runtime)?;
        }
        if completion.event.is_none()
            || completion
                .resources
                .as_ref()
                .is_none_or(|resources| resources.belongs_to_stream(&self.stream))
        {
            return Ok(true);
        }
        Ok(completion
            .resources
            .as_ref()
            .is_none_or(|resources| resources.can_handoff_accesses_to_stream(&self.stream)))
    }

    /// Queue this batch after a producer completion from another stream.
    ///
    /// The producer token, including its event and access leases, remains owned until this
    /// consumer batch's final event completes. Its buffer access leases transfer to the consumer
    /// stream only when each buffer has exactly one active stream lease; otherwise this returns
    /// [`CudaError::Busy`] before queuing a wait. This bounded handoff accepts one launch completion,
    /// not a [`CudaBatchCompletion`]. Each completion event is consumed once and is never
    /// re-recorded through this API.
    ///
    /// # Errors
    ///
    /// Returns a runtime/device mismatch, [`CudaError::Busy`] when a producer buffer has other
    /// active leases, a CUDA wait error, or [`CudaError::BatchPoisoned`].
    pub fn wait_for(&mut self, completion: CudaCompletion) -> Result<(), CudaError> {
        if !self.can_wait_for(&completion)? {
            return Err(CudaError::Busy);
        }
        self.wait_for_dependency(CudaProducerCompletion::Launch(completion))
    }

    /// Queue this batch after a finished producer batch from another stream.
    ///
    /// All producer batch buffer leases transfer atomically, allocation by allocation, only when
    /// the producer batch owns every active stream lease for those allocations. If another token
    /// or producer operation also leases any allocation, this returns [`CudaError::Busy`] before
    /// queuing a wait. Producer resources remain retained through this consumer batch's final
    /// event. Same-stream batches use stream order directly. The producer token is borrowed so a
    /// preflight rejection leaves it available to the caller; after acceptance it becomes inert.
    ///
    /// # Errors
    ///
    /// Returns a runtime/device mismatch, [`CudaError::Busy`] for shared producer leases or live
    /// readbacks, a CUDA wait error, or [`CudaError::BatchPoisoned`].
    pub fn wait_for_batch(
        &mut self,
        completion: &mut CudaBatchCompletion,
    ) -> Result<(), CudaError> {
        if !self.can_wait_for_batch(completion)? {
            return Err(CudaError::Busy);
        }
        let inherited_markers = completion.queue_markers.clone();
        let completion = completion.take_for_dependency()?;
        self.wait_for_dependency(CudaProducerCompletion::Batch(completion))?;
        for marker in inherited_markers {
            self.register_queue_marker(marker)?;
        }
        Ok(())
    }

    /// Check whether an unconsumed producer batch can be attached as a dependency.
    ///
    /// This checks runtime/device identity and every direct or transitive buffer lease without
    /// making CUDA calls or changing ownerscuda. The result remains valid until another operation
    /// changes an allocation gate; `Rc<Cell<_>>` keeps these resources on the current thread.
    pub(crate) fn can_wait_for_batch(
        &self,
        completion: &CudaBatchCompletion,
    ) -> Result<bool, CudaError> {
        ensure_batch_open(self.failed)?;
        self.stream
            .inner
            .runtime
            .ensure_same_runtime(&completion.stream.inner.runtime)?;
        if !batch_handoff_is_available(
            completion.handed_off,
            completion.wait_error_observed,
            completion.readbacks.has_live_readbacks(),
        ) {
            return Ok(false);
        }
        if let Some(event) = &completion.final_event {
            self.stream
                .inner
                .runtime
                .ensure_same_runtime(&event.inner.runtime)?;
        }
        if completion.final_event.is_none()
            || Rc::ptr_eq(&completion.stream.inner, &self.stream.inner)
        {
            return Ok(true);
        }
        Ok(completion.can_handoff_accesses_to_stream(&self.stream))
    }

    fn wait_for_dependency(
        &mut self,
        mut completion: CudaProducerCompletion,
    ) -> Result<(), CudaError> {
        ensure_batch_open(self.failed)?;
        let producer_runtime = completion.runtime();
        if let Some(runtime) = producer_runtime {
            self.stream.inner.runtime.ensure_same_runtime(runtime)?;
        }

        if completion
            .stream()
            .is_some_and(|stream| Rc::ptr_eq(&stream.inner, &self.stream.inner))
        {
            self.dependencies.push(completion);
            return Ok(());
        }

        let Some(raw_event) = completion.event().map(|event| event.inner.raw) else {
            // A completion with no event has already been waited successfully.
            return Ok(());
        };
        completion
            .handoff_accesses_to_stream(&self.stream)
            .map_err(|()| CudaError::Busy)?;
        self.dependencies.push(completion);
        if let Err(error) = unsafe {
            crate::ffi::invoke_cudaStreamWaitEvent(
                &self.stream.inner.runtime,
                self.stream.inner.raw,
                raw_event,
                0,
            )
        } {
            // Even a failed enqueue may leave the consumer's queue state uncertain. Establish
            // consumer quiescence before releasing the retained producer token.
            if self.stream.synchronize().is_err() {
                self.failed = true;
                self.quarantine_and_forget();
            } else {
                self.release_after_stream_sync();
            }
            return Err(error);
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
    pub fn finish(&mut self) -> Result<CudaBatchCompletion, CudaError> {
        ensure_batch_open(self.failed)?;
        if self
            .timing
            .as_ref()
            .is_some_and(|timing| timing.start.is_some())
        {
            let end = match self.stream.inner.runtime.create_timing_event() {
                Ok(event) => event,
                Err(error) => {
                    self.release_after_stream_sync();
                    return Err(error);
                }
            };
            if let Err(error) = self.stream.record_timing(&end) {
                if self.stream.synchronize().is_err() {
                    self.failed = true;
                    self.quarantine_and_forget();
                    std::mem::forget(end);
                } else {
                    self.resources.clear();
                    self.launch_events.clear();
                    self.dependencies.clear();
                    self.queue_markers.clear();
                    if let Some(timing) = &mut self.timing {
                        timing.start.take();
                    }
                }
                return Err(error);
            }
            if let Some(timing) = &mut self.timing {
                timing.end = Some(end);
            }
        }
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
                self.dependencies.clear();
                self.queue_markers.clear();
                if let Some(timing) = &mut self.timing {
                    timing.start.take();
                    timing.end.take();
                }
            }
            return Err(error);
        }
        let timing_events = self.timing.as_mut().and_then(|timing| {
            Some(CudaBatchTimingSegment {
                start: timing.start.take()?,
                end: timing.end.take()?,
                segments_ms: Rc::clone(&timing.segments_ms),
            })
        });
        Ok(CudaBatchCompletion {
            stream: self.stream.clone(),
            final_event: Some(event),
            launch_events: std::mem::take(&mut self.launch_events),
            resources: std::mem::take(&mut self.resources),
            dependencies: std::mem::take(&mut self.dependencies),
            readbacks: std::mem::replace(&mut self.readbacks, CudaReadbackStorage::new()),
            queue_markers: std::mem::take(&mut self.queue_markers),
            timing_events,
            handed_off: false,
            wait_error_observed: false,
        })
    }

    fn release_after_stream_sync(&mut self) {
        if self.stream.synchronize().is_err() {
            self.failed = true;
            self.quarantine_and_forget();
        } else {
            self.resources.clear();
            self.launch_events.clear();
            self.dependencies.clear();
            self.queue_markers.clear();
            if let Some(timing) = &mut self.timing {
                timing.start.take();
                timing.end.take();
            }
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
        for mut dependency in self.dependencies.drain(..) {
            dependency.quarantine_and_forget();
            std::mem::forget(dependency);
        }
        if let Some(timing) = &mut self.timing {
            if let Some(event) = timing.start.take() {
                std::mem::forget(event);
            }
            if let Some(event) = timing.end.take() {
                std::mem::forget(event);
            }
        }
        for marker in self.queue_markers.drain(..) {
            std::mem::forget(marker);
        }
        self.readbacks.quarantine_and_forget();
        std::mem::forget(self.stream.clone());
    }
}

impl Drop for CudaCompletionBatch {
    fn drop(&mut self) {
        if self.resources.is_empty()
            && self.launch_events.is_empty()
            && self.dependencies.is_empty()
            && self
                .timing
                .as_ref()
                .is_none_or(|timing| timing.start.is_none() && timing.end.is_none())
        {
            return;
        }
        if self.stream.synchronize().is_err() {
            self.quarantine_and_forget();
        }
    }
}

/// Final-event completion token retaining every launch in a [`CudaCompletionBatch`].
pub struct CudaBatchCompletion {
    stream: CudaStreamHandle,
    final_event: Option<CudaEventHandle>,
    launch_events: Vec<CudaEventHandle>,
    resources: Vec<LaunchResources>,
    dependencies: Vec<CudaProducerCompletion>,
    readbacks: CudaReadbackStorage,
    queue_markers: SmallVec<[Rc<dyn Any>; 1]>,
    timing_events: Option<CudaBatchTimingSegment>,
    handed_off: bool,
    wait_error_observed: bool,
}

impl CudaBatchCompletion {
    /// Wait for the batch's final event. On failure the token retains resources for retry.
    ///
    /// # Errors
    ///
    /// Returns the CUDA event synchronization error.
    pub fn wait(&mut self) -> Result<(), CudaError> {
        let result = self.wait_inner();
        track_wait_error(&mut self.wait_error_observed, result)
    }

    fn wait_inner(&mut self) -> Result<(), CudaError> {
        if let Some(event) = &self.final_event {
            event.synchronize()?;
        }
        if let Some(timing) = &self.timing_events {
            let milliseconds = timing
                .start
                .inner
                .runtime
                .elapsed_time_ms(&timing.start, &timing.end)?;
            timing.segments_ms.borrow_mut().push(milliseconds);
            self.timing_events.take();
        }
        wait_completion_dependencies(&mut self.dependencies, CudaProducerCompletion::wait)?;
        self.resources.clear();
        self.launch_events.clear();
        self.queue_markers.clear();
        self.final_event.take();
        Ok(())
    }

    pub(crate) fn has_queue_marker(&self, marker: &Rc<dyn Any>) -> bool {
        completion_has_queue_marker(&self.queue_markers, self.wait_error_observed, marker)
    }

    pub(crate) fn uses_stream(&self, stream: &CudaStreamHandle) -> bool {
        !self.wait_error_observed && Rc::ptr_eq(&self.stream.inner, &stream.inner)
    }

    /// Borrow completed host bytes for one readback ID.
    ///
    /// Bytes remain unavailable until this completion's event and every recursively retained
    /// dependency have all waited successfully.
    ///
    /// # Errors
    ///
    /// Returns [`CudaError::BatchNotComplete`] before a full successful wait, or
    /// [`CudaError::InvalidReadbackId`] for an ID from another batch or whose bytes were taken.
    pub fn readback(&self, id: &CudaReadbackId) -> Result<&[u8], CudaError> {
        if !readback_is_complete(
            self.final_event.is_some(),
            !self.resources.is_empty(),
            !self.dependencies.is_empty(),
        ) {
            return Err(CudaError::BatchNotComplete);
        }
        self.readbacks.read(id)
    }

    /// Move completed host bytes out of this completion without copying.
    ///
    /// The readback is single-use. The bytes remain unavailable until this completion's event
    /// and every recursively retained dependency have all waited successfully.
    ///
    /// # Errors
    ///
    /// Returns [`CudaError::BatchNotComplete`] before a full successful wait, or
    /// [`CudaError::InvalidReadbackId`] for an ID from another batch or whose bytes were already
    /// taken.
    pub fn take_readback(&mut self, id: &CudaReadbackId) -> Result<Box<[u8]>, CudaError> {
        if !readback_is_complete(
            self.final_event.is_some(),
            !self.resources.is_empty(),
            !self.dependencies.is_empty(),
        ) {
            return Err(CudaError::BatchNotComplete);
        }
        self.readbacks.take(id)
    }

    fn take_for_dependency(&mut self) -> Result<Self, CudaError> {
        if !batch_handoff_is_available(
            self.handed_off,
            self.wait_error_observed,
            self.readbacks.has_live_readbacks(),
        ) {
            return Err(CudaError::Busy);
        }
        let empty_readbacks = CudaReadbackStorage::new();
        self.handed_off = true;
        Ok(Self {
            stream: self.stream.clone(),
            final_event: self.final_event.take(),
            launch_events: std::mem::take(&mut self.launch_events),
            resources: std::mem::take(&mut self.resources),
            dependencies: std::mem::take(&mut self.dependencies),
            readbacks: std::mem::replace(&mut self.readbacks, empty_readbacks),
            queue_markers: std::mem::take(&mut self.queue_markers),
            timing_events: self.timing_events.take(),
            handed_off: false,
            wait_error_observed: false,
        })
    }

    fn can_handoff_accesses_to_stream(&self, stream: &CudaStreamHandle) -> bool {
        let from = Rc::as_ptr(&self.stream.inner) as usize;
        let to = Rc::as_ptr(&stream.inner) as usize;
        let mut guards = Vec::new();
        append_launch_resource_access_guards_readonly(&self.resources, &mut guards);
        for dependency in &self.dependencies {
            dependency.append_access_guards_readonly(&mut guards);
        }
        allocation_access_guards_can_reassign(&guards, from, to)
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
        for mut dependency in self.dependencies.drain(..) {
            dependency.quarantine_and_forget();
            std::mem::forget(dependency);
        }
        if let Some(event) = self.final_event.take() {
            std::mem::forget(event);
        }
        if let Some(timing) = self.timing_events.take() {
            std::mem::forget(timing.start);
            std::mem::forget(timing.end);
        }
        for marker in self.queue_markers.drain(..) {
            std::mem::forget(marker);
        }
        self.readbacks.quarantine_and_forget();
        std::mem::forget(self.stream.clone());
    }
}

const fn track_wait_error<T, E>(sticky_error: &mut bool, result: Result<T, E>) -> Result<T, E> {
    *sticky_error = result.is_err();
    result
}

fn has_queue_marker(markers: &[Rc<dyn Any>], marker: &Rc<dyn Any>) -> bool {
    markers.iter().any(|known| Rc::ptr_eq(known, marker))
}

fn register_queue_marker(markers: &mut SmallVec<[Rc<dyn Any>; 1]>, marker: Rc<dyn Any>) {
    if !has_queue_marker(markers, &marker) {
        markers.push(marker);
    }
}

fn completion_has_queue_marker(
    markers: &[Rc<dyn Any>],
    wait_error_observed: bool,
    marker: &Rc<dyn Any>,
) -> bool {
    !wait_error_observed && has_queue_marker(markers, marker)
}

const fn batch_handoff_is_available(
    handed_off: bool,
    wait_error_observed: bool,
    has_live_readbacks: bool,
) -> bool {
    !handed_off && !wait_error_observed && !has_live_readbacks
}

impl Drop for CudaBatchCompletion {
    fn drop(&mut self) {
        if self.resources.is_empty()
            && self.launch_events.is_empty()
            && self.dependencies.is_empty()
            && self.final_event.is_none()
            && self.timing_events.is_none()
        {
            return;
        }
        let completed = self
            .final_event
            .as_ref()
            .is_some_and(|event| event.synchronize().is_ok());
        if completed {
            self.resources.clear();
            self.launch_events.clear();
            self.dependencies.clear();
            self.final_event.take();
            self.timing_events.take();
        } else {
            self.quarantine_and_forget();
        }
    }
}

impl CudaProducerCompletion {
    fn wait(&mut self) -> Result<(), CudaError> {
        match self {
            Self::Launch(completion) => completion.wait(),
            Self::Batch(completion) => completion.wait(),
        }
    }

    const fn event(&self) -> Option<&CudaEventHandle> {
        match self {
            Self::Launch(completion) => completion.event.as_ref(),
            Self::Batch(completion) => completion.final_event.as_ref(),
        }
    }

    fn stream(&self) -> Option<&CudaStreamHandle> {
        match self {
            Self::Launch(completion) => completion
                .resources
                .as_ref()
                .map(|resources| &resources.stream),
            Self::Batch(completion) => Some(&completion.stream),
        }
    }

    fn runtime(&self) -> Option<&CudaRuntime> {
        self.event()
            .map(|event| &event.inner.runtime)
            .or_else(|| self.stream().map(|stream| &stream.inner.runtime))
    }

    fn handoff_accesses_to_stream(&mut self, stream: &CudaStreamHandle) -> Result<(), ()> {
        match self {
            Self::Launch(completion) => completion.resources.as_mut().map_or(Ok(()), |resources| {
                resources.handoff_accesses_to_stream(stream)
            }),
            Self::Batch(completion) => {
                let from = Rc::as_ptr(&completion.stream.inner) as usize;
                let to = Rc::as_ptr(&stream.inner) as usize;
                let mut guards = Vec::new();
                append_launch_resource_access_guards(&mut completion.resources, &mut guards);
                for dependency in &mut completion.dependencies {
                    dependency.append_access_guards(&mut guards);
                }
                reassign_allocation_access_guards(&mut guards, from, to)
            }
        }
    }

    fn append_access_guards<'a>(&'a mut self, guards: &mut Vec<&'a mut AllocationAccessGuard>) {
        match self {
            Self::Launch(completion) => {
                if let Some(resources) = &mut completion.resources {
                    append_launch_resource_access_guards(std::slice::from_mut(resources), guards);
                }
            }
            Self::Batch(completion) => {
                append_launch_resource_access_guards(&mut completion.resources, guards);
                for dependency in &mut completion.dependencies {
                    dependency.append_access_guards(guards);
                }
            }
        }
    }

    fn append_access_guards_readonly<'a>(&'a self, guards: &mut Vec<&'a AllocationAccessGuard>) {
        match self {
            Self::Launch(completion) => {
                if let Some(resources) = &completion.resources {
                    append_launch_resource_access_guards_readonly(
                        std::slice::from_ref(resources),
                        guards,
                    );
                }
            }
            Self::Batch(completion) => {
                append_launch_resource_access_guards_readonly(&completion.resources, guards);
                for dependency in &completion.dependencies {
                    dependency.append_access_guards_readonly(guards);
                }
            }
        }
    }

    fn quarantine_and_forget(&mut self) {
        match self {
            Self::Launch(completion) => {
                if let Some(resources) = completion.resources.take() {
                    resources.quarantine();
                    std::mem::forget(resources);
                }
                if let Some(event) = completion.event.take() {
                    std::mem::forget(event);
                }
            }
            Self::Batch(completion) => completion.quarantine_and_forget(),
        }
    }
}

fn wait_completion_dependencies<T>(
    dependencies: &mut Vec<T>,
    mut wait: impl FnMut(&mut T) -> Result<(), CudaError>,
) -> Result<(), CudaError> {
    let index = 0;
    while index < dependencies.len() {
        match wait(&mut dependencies[index]) {
            Ok(()) => {
                dependencies.remove(index);
            }
            Err(error) => return Err(error),
        }
    }
    Ok(())
}

impl CudaKernel {
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
    /// Returns a CUDA launch error or an error while preparing stream/event resources.
    #[allow(clippy::too_many_lines)] // Keep argument storage, launch, and uncertain-completion retention visible together.
    pub unsafe fn launch<'a>(
        &'a self,
        stream: &'a CudaStreamHandle,
        grid: [u32; 3],
        block: [u32; 3],
        shared_memory_bytes: u32,
        arguments: &'a [CudaKernelArgument<'a>],
    ) -> Result<CudaCompletion, CudaError> {
        // SAFETY: the caller upholds the kernel argument and buffer-access contract documented
        // above; the helper performs the same enqueue while retaining completion resources.
        unsafe { self.launch_inner(stream, grid, block, shared_memory_bytes, arguments, None) }
            .and_then(|completion| completion.ok_or(CudaError::BatchPoisoned))
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
    /// Returns a CUDA launch error, a stream/runtime mismatch, or [`CudaError::BatchPoisoned`] if a
    /// prior launch failed.
    pub unsafe fn launch_into_batch(
        &self,
        batch: &mut CudaCompletionBatch,
        grid: [u32; 3],
        block: [u32; 3],
        shared_memory_bytes: u32,
        arguments: &[CudaKernelArgument<'_>],
    ) -> Result<(), CudaError> {
        ensure_batch_open(batch.failed)?;
        // Clone first so the mutable batch borrow can be passed alongside its stream identity.
        let stream = batch.stream.clone();
        // SAFETY: the caller upholds the ABI and buffer-validity requirements. The batch has
        // ownerscuda of every successfully attempted launch before the CUDA enqueue is invoked.
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
        stream: &CudaStreamHandle,
        grid: [u32; 3],
        block: [u32; 3],
        shared_memory_bytes: u32,
        arguments: &[CudaKernelArgument<'_>],
        mut batch: Option<&mut CudaCompletionBatch>,
    ) -> Result<Option<CudaCompletion>, CudaError> {
        self.module
            .runtime
            .ensure_same_runtime(&stream.inner.runtime)?;
        if let Some(batch) = batch.as_deref() {
            if !Rc::ptr_eq(&batch.stream.inner, &stream.inner) {
                return Err(CudaError::DifferentStream);
            }
            ensure_batch_open(batch.failed)?;
        }
        for axis in 0..3 {
            if grid[axis] == 0 || block[axis] == 0 {
                return Err(CudaError::InvalidLaunchDimensions);
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
                CudaKernelArgument::Bytes(bytes) => *bytes,
                CudaKernelArgument::Buffer(buffer) => {
                    self.module
                        .runtime
                        .ensure_same_runtime(&buffer.allocation.runtime)?;
                    if !access_leases.contains(&buffer.allocation) {
                        access_leases.push(buffer.acquire_stream_access(stream)?);
                    }
                    // CUDA kernel parameters receive a device pointer value, not its host address.
                    unsafe {
                        std::slice::from_raw_parts(
                            (&raw const buffer.allocation.pointer).cast(),
                            size_of::<*mut c_void>(),
                        )
                    }
                }
            };
            if inline {
                // Each word stays at a stable stack address until CUDA consumes the pointer table.
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
            _module: Some(self.module.clone()),
            _external_owner: None,
            access_leases,
            stream: stream.clone(),
        });
        let direct_batch = batch.is_some();
        if let Some(batch) = batch.as_deref_mut() {
            // Retain before the launch call: CUDA may report an error after work has entered the
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
        if let Some(batch) = batch.as_deref_mut()
            && let Some(timing) = &mut batch.timing
            && timing.start.is_none()
        {
            let start = match self.module.runtime.create_timing_event() {
                Ok(event) => event,
                Err(error) => {
                    batch.failed = true;
                    batch.release_after_stream_sync();
                    return Err(error);
                }
            };
            if let Err(error) = stream.record_timing(&start) {
                batch.failed = true;
                if stream.synchronize().is_err() {
                    batch.quarantine_and_forget();
                    std::mem::forget(start);
                } else {
                    batch.resources.clear();
                    batch.launch_events.clear();
                }
                return Err(error);
            }
            timing.start = Some(start);
        }
        let launch_result = unsafe {
            crate::ffi::invoke_cuLaunchKernel(
                &self.module.runtime,
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
        };
        if let Err(error) = launch_result {
            if let Some(batch) = batch {
                batch.failed = true;
            } else if stream.synchronize().is_err() {
                // CUDA can surface an earlier asynchronous fault from a later API call. If the
                // stream cannot confirm quiescence, retain the launch resources conservatively.
                let Some(resources) = resources.take() else {
                    return Err(CudaError::BatchPoisoned);
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
                        return Err(CudaError::BatchPoisoned);
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
            return Ok(Some(CudaCompletion {
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
    fn batch_wait_error_stays_sticky_until_a_full_wait_succeeds() {
        let mut observed = false;
        assert_eq!(track_wait_error(&mut observed, Ok::<_, ()>(())), Ok(()));
        assert!(!observed);

        assert_eq!(
            track_wait_error(&mut observed, Err::<(), _>("event")),
            Err("event")
        );
        assert!(observed);
        assert_eq!(
            track_wait_error(&mut observed, Err::<(), _>("dependency")),
            Err("dependency")
        );
        assert!(observed);

        assert_eq!(track_wait_error(&mut observed, Ok::<_, ()>(())), Ok(()));
        assert!(!observed);
    }

    #[test]
    fn failed_wait_preflight_preserves_the_producer_token_for_retry() {
        let handed_off = false;
        let wait_error_observed = true;
        let has_live_readbacks = false;
        assert!(!batch_handoff_is_available(
            handed_off,
            wait_error_observed,
            has_live_readbacks,
        ));
        assert!(!handed_off);
        assert!(wait_error_observed);
        assert!(!has_live_readbacks);

        assert!(batch_handoff_is_available(false, false, false));
    }

    #[test]
    fn queue_markers_follow_identity_and_are_hidden_after_a_wait_error() {
        let marker = Rc::new(()) as Rc<dyn Any>;
        let marker_clone = Rc::clone(&marker);
        let unrelated = Rc::new(()) as Rc<dyn Any>;
        let mut markers = SmallVec::new();
        register_queue_marker(&mut markers, Rc::clone(&marker));
        register_queue_marker(&mut markers, marker_clone);

        assert_eq!(markers.len(), 1);
        assert!(completion_has_queue_marker(&markers, false, &marker));
        assert!(!completion_has_queue_marker(&markers, true, &marker));
        assert!(!completion_has_queue_marker(&markers, false, &unrelated));
    }

    #[test]
    fn device_copy_preflight_checks_both_ranges_and_aliasing() {
        assert_eq!(
            validate_device_copy(16, 8, 9, false),
            Err(CudaError::BufferTooSmall {
                allocation: 8,
                requested: 9,
            })
        );
        assert_eq!(
            validate_device_copy(8, 16, 9, false),
            Err(CudaError::BufferTooSmall {
                allocation: 8,
                requested: 9,
            })
        );
        assert_eq!(validate_device_copy(8, 8, 8, true), Err(CudaError::Busy));
        assert_eq!(validate_device_copy(8, 8, 0, true), Ok(()));
        assert_eq!(validate_device_copy(8, 16, 8, false), Ok(()));
    }

    #[test]
    fn host_upload_preflight_checks_bounds_overflow_and_empty_end_ranges() {
        assert_eq!(validate_buffer_range(16, 4, 12), Ok(()));
        assert_eq!(validate_buffer_range(16, 16, 0), Ok(()));
        assert_eq!(
            validate_buffer_range(16, 4, 13),
            Err(CudaError::BufferTooSmall {
                allocation: 12,
                requested: 13,
            })
        );
        assert_eq!(
            validate_buffer_range(16, usize::MAX, 1),
            Err(CudaError::BufferTooSmall {
                allocation: 0,
                requested: 1,
            })
        );
    }

    #[test]
    fn readback_storage_checks_batch_identity_ranges_and_single_queue() {
        let mut storage = CudaReadbackStorage::new();
        let id = storage.allocate(5);
        assert!(storage.has_live_readbacks());
        assert_eq!(storage.validate_destination(&id, 2, 3), Ok(()));
        assert_eq!(
            storage.validate_destination(&id, 4, 2),
            Err(CudaError::BufferTooSmall {
                allocation: 1,
                requested: 2,
            })
        );

        let mut foreign_storage = CudaReadbackStorage::new();
        let foreign_id = foreign_storage.allocate(5);
        assert_eq!(
            storage.validate_destination(&foreign_id, 0, 1),
            Err(CudaError::InvalidReadbackId)
        );

        let destination = storage.prepare_destination(&id, 1, 3).unwrap();
        // SAFETY: the storage allocated five writable bytes and the checked range [1, 4) fits.
        // The pointer is the only access to those bytes until completion in production.
        unsafe { ptr::copy_nonoverlapping(b"abc".as_ptr(), destination, 3) };
        assert_eq!(
            storage.validate_destination(&id, 0, 1),
            Err(CudaError::ReadbackAlreadyQueued)
        );
        assert_eq!(storage.read(&id).unwrap(), &[0, b'a', b'b', b'c', 0]);

        assert_eq!(
            storage.take(&id).unwrap().as_ref(),
            &[0, b'a', b'b', b'c', 0]
        );
        assert_eq!(storage.take(&id), Err(CudaError::InvalidReadbackId));
        assert!(!storage.has_live_readbacks());
    }

    #[test]
    fn empty_readback_has_an_owned_identity_and_no_copy_range() {
        let mut storage = CudaReadbackStorage::new();
        let empty = storage.allocate(0);
        assert_eq!(storage.validate_destination(&empty, 0, 0), Ok(()));
        assert!(storage.read(&empty).unwrap().is_empty());
        assert!(storage.take(&empty).unwrap().as_ref().is_empty());
        assert_eq!(storage.read(&empty), Err(CudaError::InvalidReadbackId));
    }

    #[test]
    fn readback_waits_for_the_final_event_resources_and_recursive_dependencies() {
        assert!(!readback_is_complete(false, false, true));
        assert!(!readback_is_complete(true, false, false));
        assert!(!readback_is_complete(false, true, false));
        assert!(readback_is_complete(false, false, false));
    }

    #[test]
    fn timing_events_keep_distinct_flags_from_completion_events() {
        assert_eq!(CUDA_EVENT_DEFAULT, 0);
        assert_eq!(CUDA_EVENT_DISABLE_TIMING, 2);
        assert_ne!(CUDA_EVENT_DEFAULT, CUDA_EVENT_DISABLE_TIMING);
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
        let empty = CudaKernelArgument::Bytes(&[]);
        let full_word = CudaKernelArgument::Bytes(&[0_u8; 16]);
        let oversized = CudaKernelArgument::Bytes(&[0_u8; 17]);
        assert!(can_inline_kernel_parameters(&[empty, full_word]));
        assert!(!can_inline_kernel_parameters(&[oversized]));
        let nine: [CudaKernelArgument<'_>; 9] =
            std::array::from_fn(|_| CudaKernelArgument::Bytes(&[]));
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
    fn allocation_gate_handoff_moves_a_sole_lease_to_the_consumer_stream() {
        let state = Rc::new(AllocationAccess {
            state: Cell::new(AllocationAccessState::Idle),
        });
        let mut producer_lease = state.acquire_stream(11).unwrap();
        let eligibility = [&producer_lease];
        assert!(allocation_access_guards_can_reassign(&eligibility, 11, 12));
        let mut guards = [&mut producer_lease];

        reassign_allocation_access_guards(&mut guards, 11, 12).unwrap();
        assert_eq!(
            state.state.get(),
            AllocationAccessState::Stream {
                identity: 12,
                leases: 1,
            }
        );
        assert!(state.acquire_stream(11).is_err());
        let consumer_lease = state.acquire_stream(12).unwrap();
        drop(producer_lease);
        assert_eq!(
            state.state.get(),
            AllocationAccessState::Stream {
                identity: 12,
                leases: 1,
            }
        );
        drop(consumer_lease);
        assert_eq!(state.state.get(), AllocationAccessState::Idle);
    }

    #[test]
    fn allocation_gate_handoff_preflights_every_lease_before_mutating_any() {
        let sole = Rc::new(AllocationAccess {
            state: Cell::new(AllocationAccessState::Idle),
        });
        let shared = Rc::new(AllocationAccess {
            state: Cell::new(AllocationAccessState::Idle),
        });
        let mut sole_lease = sole.acquire_stream(11).unwrap();
        let mut shared_lease = shared.acquire_stream(11).unwrap();
        let other_shared_lease = shared.acquire_stream(11).unwrap();
        let sole_guard = [&sole_lease];
        let shared_guard = [&shared_lease];
        let whole_batch_guards = [&sole_lease, &shared_lease];
        assert!(allocation_access_guards_can_reassign(&sole_guard, 11, 12));
        assert!(!allocation_access_guards_can_reassign(
            &shared_guard,
            11,
            12
        ));
        assert!(!allocation_access_guards_can_reassign(
            &whole_batch_guards,
            11,
            12
        ));
        let mut guards = [&mut sole_lease, &mut shared_lease];

        assert!(reassign_allocation_access_guards(&mut guards, 11, 12).is_err());
        assert_eq!(
            sole.state.get(),
            AllocationAccessState::Stream {
                identity: 11,
                leases: 1,
            }
        );
        assert_eq!(
            shared.state.get(),
            AllocationAccessState::Stream {
                identity: 11,
                leases: 2,
            }
        );

        drop(other_shared_lease);
        drop(shared_lease);
        drop(sole_lease);
        assert_eq!(sole.state.get(), AllocationAccessState::Idle);
        assert_eq!(shared.state.get(), AllocationAccessState::Idle);
    }

    #[test]
    fn allocation_gate_handoff_moves_all_leases_owned_by_one_batch() {
        let shared = Rc::new(AllocationAccess {
            state: Cell::new(AllocationAccessState::Idle),
        });
        let mut first_lease = shared.acquire_stream(11).unwrap();
        let mut second_lease = shared.acquire_stream(11).unwrap();
        let batch_guards = [&first_lease, &second_lease];
        assert!(allocation_access_guards_can_reassign(&batch_guards, 11, 12));
        let mut guards = [&mut first_lease, &mut second_lease];

        reassign_allocation_access_guards(&mut guards, 11, 12).unwrap();
        assert_eq!(
            shared.state.get(),
            AllocationAccessState::Stream {
                identity: 12,
                leases: 2,
            }
        );
        assert!(shared.acquire_stream(11).is_err());
        let third_lease = shared.acquire_stream(12).unwrap();
        drop(first_lease);
        drop(second_lease);
        assert_eq!(
            shared.state.get(),
            AllocationAccessState::Stream {
                identity: 12,
                leases: 1,
            }
        );
        drop(third_lease);
        assert_eq!(shared.state.get(), AllocationAccessState::Idle);
    }

    #[test]
    fn handed_off_allocation_is_quarantined_under_the_consumer_stream_identity() {
        let state = Rc::new(AllocationAccess {
            state: Cell::new(AllocationAccessState::Idle),
        });
        let mut producer_lease = state.acquire_stream(11).unwrap();
        let mut guards = [&mut producer_lease];

        reassign_allocation_access_guards(&mut guards, 11, 12).unwrap();
        producer_lease.quarantine_stream();
        drop(producer_lease);

        assert_eq!(state.state.get(), AllocationAccessState::Poisoned);
        assert!(state.acquire_stream(11).is_err());
        assert!(state.acquire_stream(12).is_err());
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
        assert_eq!(ensure_batch_open(true), Err(CudaError::BatchPoisoned));
    }

    #[test]
    fn dependency_wait_removes_successes_and_retains_failed_and_unvisited_tokens() {
        #[derive(Debug, PartialEq, Eq)]
        struct WaitFixture {
            id: u8,
            failures_left: u8,
            attempts: u8,
        }

        let mut dependencies = vec![
            WaitFixture {
                id: 1,
                failures_left: 0,
                attempts: 0,
            },
            WaitFixture {
                id: 2,
                failures_left: 1,
                attempts: 0,
            },
            WaitFixture {
                id: 3,
                failures_left: 0,
                attempts: 0,
            },
        ];
        let first_wait = wait_completion_dependencies(&mut dependencies, |dependency| {
            dependency.attempts += 1;
            if dependency.failures_left > 0 {
                dependency.failures_left -= 1;
                Err(CudaError::BatchPoisoned)
            } else {
                Ok(())
            }
        });

        assert_eq!(first_wait, Err(CudaError::BatchPoisoned));
        assert_eq!(
            dependencies,
            vec![
                WaitFixture {
                    id: 2,
                    failures_left: 0,
                    attempts: 1,
                },
                WaitFixture {
                    id: 3,
                    failures_left: 0,
                    attempts: 0,
                },
            ]
        );

        assert_eq!(
            wait_completion_dependencies(&mut dependencies, |dependency| {
                dependency.attempts += 1;
                Ok(())
            }),
            Ok(())
        );
        assert!(dependencies.is_empty());
    }

    #[test]
    #[ignore = "requires CUDA GPU and CUDA device access"]
    fn cuda_gpu_two_dependent_launches_complete_through_final_batch_event() {
        let runtime = CudaRuntime::new(0).expect("CUDA GPU CUDA runtime");
        let info = runtime.device_info().expect("selected CUDA device info");
        assert!(!info.name.is_empty(), "unexpected GPU: {}", info.name);

        let image = crate::compile_cuda_source_for_device(
            &runtime,
            r#"
extern "C" __global__ void increment(float* value) {
    if (blockIdx.x == 0 && threadIdx.x == 0) value[0] += 1.0f;
}
"#,
        )
        .expect("compile increment kernel with CUDARTC");
        let module = runtime.load_module(&image).expect("load increment kernel");
        let kernel = module
            .function(c"increment")
            .expect("resolve increment kernel");
        let stream = runtime.create_stream().expect("create CUDA stream");
        let mut buffer = runtime
            .allocate(size_of::<f32>())
            .expect("allocate device value");
        buffer
            .copy_from(&0.0_f32.to_ne_bytes())
            .expect("initialize device value");
        let arguments = [CudaKernelArgument::Buffer(&buffer)];

        let mut batch = CudaCompletionBatch::new(&stream);
        // SAFETY: CUDARTC compiled one f32 pointer parameter, and the one-element allocation is
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
    #[ignore = "requires CUDA GPU and CUDA device access"]
    fn cuda_gpu_cross_stream_completion_handoff_transfers_buffer_lease() {
        let runtime = CudaRuntime::new(0).expect("CUDA GPU CUDA runtime");
        let info = runtime.device_info().expect("selected CUDA device info");
        assert!(!info.name.is_empty(), "unexpected GPU: {}", info.name);

        let image = crate::compile_cuda_source_for_device(
            &runtime,
            r#"
extern "C" __global__ void produce(unsigned int* value) {
    if (blockIdx.x == 0 && threadIdx.x == 0) value[0] = 40u;
}
extern "C" __global__ void consume(unsigned int* value) {
    if (blockIdx.x == 0 && threadIdx.x == 0) value[0] += 2u;
}
"#,
        )
        .expect("compile cross-stream producer and consumer with CUDARTC");
        let module = runtime.load_module(&image).expect("load CUDA module");
        let producer = module
            .function(c"produce")
            .expect("resolve producer kernel");
        let consumer = module
            .function(c"consume")
            .expect("resolve consumer kernel");
        let producer_stream = runtime.create_stream().expect("create producer stream");
        let consumer_stream = runtime.create_stream().expect("create consumer stream");
        let buffer = runtime
            .allocate(size_of::<u32>())
            .expect("allocate shared u32 buffer");
        let arguments = [CudaKernelArgument::Buffer(&buffer)];

        // SAFETY: both CUDARTC kernels take one u32 pointer and access only the allocated word.
        let producer_completion = unsafe {
            producer
                .launch(&producer_stream, [1, 1, 1], [1, 1, 1], 0, &arguments)
                .expect("launch producer kernel")
        };
        let mut consumer_batch = CudaCompletionBatch::new(&consumer_stream);
        consumer_batch
            .wait_for(producer_completion)
            .expect("enqueue cross-stream dependency and hand off buffer lease");
        // SAFETY: wait_for orders this consumer after the producer event and transfers the
        // producer's sole buffer lease to the consumer stream before this launch acquires it.
        unsafe {
            consumer
                .launch_into_batch(&mut consumer_batch, [1, 1, 1], [1, 1, 1], 0, &arguments)
                .expect("launch consumer kernel on dependent stream");
        }
        let mut completion = consumer_batch
            .finish()
            .expect("record consumer final event");
        completion.wait().expect("wait for consumer completion");

        let mut actual = [0_u8; size_of::<u32>()];
        buffer
            .copy_to(&mut actual)
            .expect("read completed device value");
        assert_eq!(u32::from_ne_bytes(actual), 42);
    }

    #[test]
    #[ignore = "requires CUDA GPU and CUDA device access"]
    fn cuda_gpu_cross_stream_batch_handoff_transfers_shared_buffer_leases() {
        let runtime = CudaRuntime::new(0).expect("CUDA GPU CUDA runtime");
        let info = runtime.device_info().expect("selected CUDA device info");
        assert!(!info.name.is_empty(), "unexpected GPU: {}", info.name);

        let image = crate::compile_cuda_source_for_device(
            &runtime,
            r#"
extern "C" __global__ void produce(unsigned int* value) {
    if (blockIdx.x == 0 && threadIdx.x == 0) value[0] = 20u;
}
extern "C" __global__ void accumulate(unsigned int* value) {
    if (blockIdx.x == 0 && threadIdx.x == 0) value[0] += 20u;
}
extern "C" __global__ void consume(unsigned int* value) {
    if (blockIdx.x == 0 && threadIdx.x == 0) value[0] += 2u;
}
"#,
        )
        .expect("compile batch producer and consumer kernels with CUDARTC");
        let module = runtime.load_module(&image).expect("load CUDA module");
        let producer = module
            .function(c"produce")
            .expect("resolve producer kernel");
        let accumulate = module
            .function(c"accumulate")
            .expect("resolve accumulation kernel");
        let consumer = module
            .function(c"consume")
            .expect("resolve consumer kernel");
        let producer_stream = runtime.create_stream().expect("create producer stream");
        let consumer_stream = runtime.create_stream().expect("create consumer stream");
        let buffer = runtime
            .allocate(size_of::<u32>())
            .expect("allocate shared u32 buffer");
        let arguments = [CudaKernelArgument::Buffer(&buffer)];

        let mut producer_batch = CudaCompletionBatch::new(&producer_stream);
        // SAFETY: all kernels take one u32 pointer and access only the allocated word. Both
        // producer launches are ordered in one batch, whose final event protects the two leases.
        unsafe {
            producer
                .launch_into_batch(&mut producer_batch, [1, 1, 1], [1, 1, 1], 0, &arguments)
                .expect("queue first producer launch");
            accumulate
                .launch_into_batch(&mut producer_batch, [1, 1, 1], [1, 1, 1], 0, &arguments)
                .expect("queue second producer launch");
        }
        let mut producer_completion = producer_batch
            .finish()
            .expect("record producer batch final event");

        let mut consumer_batch = CudaCompletionBatch::new(&consumer_stream);
        consumer_batch
            .wait_for_batch(&mut producer_completion)
            .expect("enqueue batch dependency and transfer both buffer leases");
        // SAFETY: wait_for_batch orders this consumer after both producer launches and transfers
        // all producer-owned leases to the consumer stream before this launch acquires the word.
        unsafe {
            consumer
                .launch_into_batch(&mut consumer_batch, [1, 1, 1], [1, 1, 1], 0, &arguments)
                .expect("launch consumer after producer batch");
        }
        let mut completion = consumer_batch
            .finish()
            .expect("record consumer batch final event");
        completion
            .wait()
            .expect("wait for consumer batch completion");

        let mut actual = [0_u8; size_of::<u32>()];
        buffer
            .copy_to(&mut actual)
            .expect("read completed device value");
        assert_eq!(u32::from_ne_bytes(actual), 42);
    }

    #[test]
    #[ignore = "requires CUDA GPU and CUDA device access"]
    fn cuda_gpu_batch_device_copy_retains_buffers_until_final_event() {
        let runtime = CudaRuntime::new(0).expect("CUDA GPU CUDA runtime");
        let info = runtime.device_info().expect("selected CUDA device info");
        assert!(!info.name.is_empty(), "unexpected GPU: {}", info.name);
        let stream = runtime.create_stream().expect("create CUDA stream");
        let mut source = runtime
            .allocate(size_of::<u32>())
            .expect("allocate source word");
        let destination = runtime
            .allocate(size_of::<u32>())
            .expect("allocate destination word");
        source
            .copy_from(&0xA1B2_C3D4_u32.to_ne_bytes())
            .expect("initialize source word");

        let mut batch = CudaCompletionBatch::new(&stream);
        batch
            .copy_device_to_device(&destination, &source, size_of::<u32>())
            .expect("enqueue asynchronous batch copy");
        assert!(matches!(destination.acquire_access(), Err(CudaError::Busy)));
        assert!(matches!(source.acquire_access(), Err(CudaError::Busy)));
        batch
            .finish()
            .expect("record final batch event")
            .wait()
            .expect("wait for copied data");

        let mut actual = [0_u8; size_of::<u32>()];
        destination
            .copy_to(&mut actual)
            .expect("read copied destination word");
        assert_eq!(u32::from_ne_bytes(actual), 0xA1B2_C3D4);
    }

    #[test]
    #[ignore = "requires CUDA GPU and CUDA device access"]
    fn cuda_gpu_batch_host_upload_retains_payload_and_destination_until_final_event() {
        let runtime = CudaRuntime::new(0).expect("CUDA GPU CUDA runtime");
        let info = runtime.device_info().expect("selected CUDA device info");
        assert!(!info.name.is_empty(), "unexpected GPU: {}", info.name);
        let stream = runtime.create_stream().expect("create CUDA stream");
        let mut destination = runtime.allocate(8).expect("allocate destination bytes");
        destination
            .copy_from(&[0xA5; 8])
            .expect("initialize bytes outside the upload range");
        let payload: Arc<[u8]> = Arc::from([0x10_u8, 0x20, 0x30, 0x40]);
        let payload_weak = Arc::downgrade(&payload);

        let mut batch = CudaCompletionBatch::new(&stream);
        batch
            .copy_host_to_device_at(&destination, 2, Arc::clone(&payload))
            .expect("enqueue asynchronous host upload");
        drop(payload);
        assert!(payload_weak.upgrade().is_some());
        assert!(matches!(destination.acquire_access(), Err(CudaError::Busy)));

        batch
            .finish()
            .expect("record final batch event")
            .wait()
            .expect("wait for uploaded data");
        assert!(payload_weak.upgrade().is_none());

        let mut actual = [0_u8; 8];
        destination
            .copy_to(&mut actual)
            .expect("read uploaded destination bytes");
        assert_eq!(actual, [0xA5, 0xA5, 0x10, 0x20, 0x30, 0x40, 0xA5, 0xA5]);
    }

    #[test]
    #[ignore = "requires CUDA GPU and CUDA device access"]
    fn cuda_gpu_batch_device_readback_is_hidden_until_full_completion() {
        let runtime = CudaRuntime::new(0).expect("CUDA GPU CUDA runtime");
        let info = runtime.device_info().expect("selected CUDA device info");
        assert!(!info.name.is_empty(), "unexpected GPU: {}", info.name);
        let stream = runtime.create_stream().expect("create CUDA stream");
        let mut source = runtime.allocate(8).expect("allocate source bytes");
        source
            .copy_from(&[0x10_u8, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70, 0x80])
            .expect("initialize source bytes");

        let mut batch = CudaCompletionBatch::new(&stream);
        let id = batch
            .allocate_readback(6)
            .expect("allocate readback storage");
        batch
            .copy_device_to_host_at(&source, 2, &id, 1, 4)
            .expect("enqueue asynchronous readback from checked ranges");
        assert_eq!(
            batch.copy_device_to_host_at(&source, 0, &id, 0, 1),
            Err(CudaError::ReadbackAlreadyQueued)
        );
        assert!(matches!(source.acquire_access(), Err(CudaError::Busy)));

        let mut completion = batch.finish().expect("record final readback event");
        let mut consumer_batch =
            CudaCompletionBatch::new(&runtime.create_stream().expect("create consumer stream"));
        assert_eq!(
            consumer_batch.wait_for_batch(&mut completion),
            Err(CudaError::Busy)
        );
        assert_eq!(completion.readback(&id), Err(CudaError::BatchNotComplete));
        completion.wait().expect("wait for asynchronous readback");
        assert_eq!(
            completion.readback(&id).expect("borrow completed bytes"),
            &[0, 0x30, 0x40, 0x50, 0x60, 0]
        );
        assert_eq!(
            completion.take_readback(&id).unwrap().as_ref(),
            &[0, 0x30, 0x40, 0x50, 0x60, 0]
        );
        assert_eq!(
            completion.take_readback(&id),
            Err(CudaError::InvalidReadbackId)
        );
        assert!(source.acquire_access().is_ok());
    }

    #[test]
    fn translates_physical_pool_memory_without_claiming_process_attribution() {
        let snapshot = cuda_memory_pool_snapshot(
            PcuMemoryPoolId(17),
            CudaMemoryInfo {
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
        let malformed = cuda_memory_pool_snapshot(
            PcuMemoryPoolId(1),
            CudaMemoryInfo {
                free_bytes: 101,
                total_bytes: 100,
            },
        );
        assert_eq!(malformed.capacity_bytes, Some(100));
        assert_eq!(malformed.system_used_bytes, PcuMemoryUsage::Unknown);
        assert_eq!(malformed.process_used_bytes, PcuMemoryUsage::Unknown);

        let zero_capacity = cuda_memory_pool_snapshot(
            PcuMemoryPoolId(2),
            CudaMemoryInfo {
                free_bytes: 0,
                total_bytes: 0,
            },
        );
        assert_eq!(zero_capacity.capacity_bytes, None);
        assert_eq!(zero_capacity.system_used_bytes, PcuMemoryUsage::Unknown);
    }

    #[test]
    fn runtime_probe_is_safe_without_selecting_a_device() {
        match CudaRuntime::probe() {
            Ok(probe) => assert!(!probe.runtime_library.is_empty()),
            Err(
                CudaError::RuntimeUnavailable(_)
                | CudaError::Runtime {
                    operation: "cudaGetDeviceCount",
                    code: 100,
                    ..
                },
            ) => {}
            Err(error) => panic!("unexpected CUDA probe error: {error}"),
        }
    }
}
