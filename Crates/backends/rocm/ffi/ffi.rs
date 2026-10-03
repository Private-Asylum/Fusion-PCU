//! Foreign ABI declarations and dynamic loading for `ROCm` provider libraries.

#[rustfmt::skip]
pub use libloading::{
    Library,
    Symbol,
};

/// Resolves one symbol from a retained provider library.
///
/// # Safety
///
/// `T` must exactly match the ABI and signature of the named symbol, and the returned symbol
/// must not outlive `library`.
pub unsafe fn symbol<'library, T>(
    library: &'library Library,
    name: &[u8],
) -> Result<Symbol<'library, T>, libloading::Error> {
    #[cfg(feature = "allocation-census")]
    census::symbol();
    // SAFETY: the caller upholds the symbol name/type and library lifetime contract above.
    unsafe { library.get(name) }
}

#[cfg(feature = "allocation-census")]
#[path = "census/census.rs"]
mod census;
#[cfg(feature = "allocation-census")]
pub use census::{RocmApiCensus, rocm_api_census, reset_rocm_api_census};
#[path = "retained/retained.rs"]
mod retained;
pub use retained::RetainedApi;

#[path = "hip.rs"]
pub mod hip;
#[path = "hiprtc.rs"]
pub mod hiprtc;
#[path = "loader.rs"]
mod loader;
#[path = "rocblas.rs"]
pub mod rocblas;
pub use loader::load_library;
pub use loader::load_uncached_library;

#[rustfmt::skip]
use std::ffi::{
    c_char,
    c_int,
    c_void,
};

/// Resolve the exact `rocblas_create_handle` ABI while retaining the library borrow.
///
/// # Safety
/// The library must be a compatible rocBLAS provider and remain loaded through every use.
pub unsafe fn require_rocblas_create_handle(
    library: &Library,
) -> Result<Symbol<'_, rocblas::CreateHandle>, libloading::Error> {
    // SAFETY: SDK name and ABI are paired here; caller retains the compatible library.
    unsafe { symbol(library, b"rocblas_create_handle\0") }
}
/// Invoke `rocblas_create_handle` with the caller's validated selected-device resources.
///
/// # Safety
/// The function must originate from a retained compatible library. Handles must be live on
/// the selected device. Pointers must cover validated typed extents and appropriate host/device
/// pointer mode; retained leases must prevent conflicting access through terminal completion.
#[inline]
pub unsafe fn invoke_rocblas_create_handle(
    function: &rocblas::CreateHandle,
    handle: *mut rocblas::RocblasHandle,
) -> rocblas::RocblasStatus {
    // SAFETY: caller supplies the validated pointer, handle, library and lease contract above.
    unsafe { function(handle) }
}

/// Resolve the exact `rocblas_destroy_handle` ABI while retaining the library borrow.
///
/// # Safety
/// The library must be a compatible rocBLAS provider and remain loaded through every use.
pub unsafe fn require_rocblas_destroy_handle(
    library: &Library,
) -> Result<Symbol<'_, rocblas::DestroyHandle>, libloading::Error> {
    // SAFETY: SDK name and ABI are paired here; caller retains the compatible library.
    unsafe { symbol(library, b"rocblas_destroy_handle\0") }
}
/// Invoke `rocblas_destroy_handle` with the caller's validated selected-device resources.
///
/// # Safety
/// The function must originate from a retained compatible library. Handles must be live on
/// the selected device. Pointers must cover validated typed extents and appropriate host/device
/// pointer mode; retained leases must prevent conflicting access through terminal completion.
#[inline]
pub unsafe fn invoke_rocblas_destroy_handle(
    function: &rocblas::DestroyHandle,
    handle: rocblas::RocblasHandle,
) -> rocblas::RocblasStatus {
    // SAFETY: caller supplies the validated pointer, handle, library and lease contract above.
    unsafe { function(handle) }
}

/// Resolve the exact `rocblas_sgemm` ABI while retaining the library borrow.
///
/// # Safety
/// The library must be a compatible rocBLAS provider and remain loaded through every use.
pub unsafe fn require_rocblas_sgemm(
    library: &Library,
) -> Result<Symbol<'_, rocblas::Sgemm>, libloading::Error> {
    // SAFETY: SDK name and ABI are paired here; caller retains the compatible library.
    unsafe { symbol(library, b"rocblas_sgemm\0") }
}
/// Invoke `rocblas_sgemm` with the caller's validated selected-device resources.
///
/// # Safety
/// The function must originate from a retained compatible library. Handles must be live on
/// the selected device. Pointers must cover validated typed extents and appropriate host/device
/// pointer mode; retained leases must prevent conflicting access through terminal completion.
#[inline]
#[allow(clippy::too_many_arguments)] // Mirror the private typed SDK ABI.
pub unsafe fn invoke_rocblas_sgemm(
    function: &rocblas::Sgemm,
    handle: rocblas::RocblasHandle,
    transpose_left: c_int,
    transpose_right: c_int,
    rows: c_int,
    columns: c_int,
    inner: c_int,
    alpha: *const f32,
    left: *const f32,
    left_stride: c_int,
    right: *const f32,
    right_stride: c_int,
    beta: *const f32,
    output: *mut f32,
    output_stride: c_int,
) -> rocblas::RocblasStatus {
    // SAFETY: caller supplies the validated pointer, handle, library and lease contract above.
    unsafe {
        function(
            handle,
            transpose_left,
            transpose_right,
            rows,
            columns,
            inner,
            alpha,
            left,
            left_stride,
            right,
            right_stride,
            beta,
            output,
            output_stride,
        )
    }
}

/// Resolve the exact `rocblas_dgemm` ABI while retaining the library borrow.
///
/// # Safety
/// The library must be a compatible rocBLAS provider and remain loaded through every use.
pub unsafe fn require_rocblas_dgemm(
    library: &Library,
) -> Result<Symbol<'_, rocblas::Dgemm>, libloading::Error> {
    // SAFETY: SDK name and ABI are paired here; caller retains the compatible library.
    unsafe { symbol(library, b"rocblas_dgemm\0") }
}
/// Invoke `rocblas_dgemm` with the caller's validated selected-device resources.
///
/// # Safety
/// The function must originate from a retained compatible library. Handles must be live on
/// the selected device. Pointers must cover validated typed extents and appropriate host/device
/// pointer mode; retained leases must prevent conflicting access through terminal completion.
#[inline]
#[allow(clippy::too_many_arguments)] // Mirror the private typed SDK ABI.
pub unsafe fn invoke_rocblas_dgemm(
    function: &rocblas::Dgemm,
    handle: rocblas::RocblasHandle,
    transpose_left: c_int,
    transpose_right: c_int,
    rows: c_int,
    columns: c_int,
    inner: c_int,
    alpha: *const f64,
    left: *const f64,
    left_stride: c_int,
    right: *const f64,
    right_stride: c_int,
    beta: *const f64,
    output: *mut f64,
    output_stride: c_int,
) -> rocblas::RocblasStatus {
    // SAFETY: caller supplies the validated pointer, handle, library and lease contract above.
    unsafe {
        function(
            handle,
            transpose_left,
            transpose_right,
            rows,
            columns,
            inner,
            alpha,
            left,
            left_stride,
            right,
            right_stride,
            beta,
            output,
            output_stride,
        )
    }
}

/// Resolve the exact `rocblas_sdot` ABI while retaining the library borrow.
///
/// # Safety
/// The library must be a compatible rocBLAS provider and remain loaded through every use.
pub unsafe fn require_rocblas_sdot(
    library: &Library,
) -> Result<Symbol<'_, rocblas::Sdot>, libloading::Error> {
    // SAFETY: SDK name and ABI are paired here; caller retains the compatible library.
    unsafe { symbol(library, b"rocblas_sdot\0") }
}
/// Invoke `rocblas_sdot` with the caller's validated selected-device resources.
///
/// # Safety
/// The function must originate from a retained compatible library. Handles must be live on
/// the selected device. Pointers must cover validated typed extents and appropriate host/device
/// pointer mode; retained leases must prevent conflicting access through terminal completion.
#[inline]
#[allow(clippy::too_many_arguments)] // Exact private SDK parameter ordering.
pub unsafe fn invoke_rocblas_sdot(
    function: &rocblas::Sdot,
    handle: rocblas::RocblasHandle,
    length: c_int,
    left: *const f32,
    left_stride: c_int,
    right: *const f32,
    right_stride: c_int,
    result: *mut f32,
) -> rocblas::RocblasStatus {
    // SAFETY: caller supplies the validated pointer, handle, library and lease contract above.
    unsafe {
        function(
            handle,
            length,
            left,
            left_stride,
            right,
            right_stride,
            result,
        )
    }
}

/// Resolve the exact `rocblas_sasum` ABI while retaining the library borrow.
///
/// # Safety
/// The library must be a compatible rocBLAS provider and remain loaded through every use.
pub unsafe fn require_rocblas_sasum(
    library: &Library,
) -> Result<Symbol<'_, rocblas::Sasum>, libloading::Error> {
    // SAFETY: SDK name and ABI are paired here; caller retains the compatible library.
    unsafe { symbol(library, b"rocblas_sasum\0") }
}
/// Invoke `rocblas_sasum` with the caller's validated selected-device resources.
///
/// # Safety
/// The function must originate from a retained compatible library. Handles must be live on
/// the selected device. Pointers must cover validated typed extents and appropriate host/device
/// pointer mode; retained leases must prevent conflicting access through terminal completion.
#[inline]
pub unsafe fn invoke_rocblas_sasum(
    function: &rocblas::Sasum,
    handle: rocblas::RocblasHandle,
    length: c_int,
    input: *const f32,
    stride: c_int,
    result: *mut f32,
) -> rocblas::RocblasStatus {
    // SAFETY: caller supplies the validated pointer, handle, library and lease contract above.
    unsafe { function(handle, length, input, stride, result) }
}

/// Resolve the exact `rocblas_sscal` ABI while retaining the library borrow.
///
/// # Safety
/// The library must be a compatible rocBLAS provider and remain loaded through every use.
pub unsafe fn require_rocblas_sscal(
    library: &Library,
) -> Result<Symbol<'_, rocblas::Sscal>, libloading::Error> {
    // SAFETY: SDK name and ABI are paired here; caller retains the compatible library.
    unsafe { symbol(library, b"rocblas_sscal\0") }
}
/// Invoke `rocblas_sscal` with the caller's validated selected-device resources.
///
/// # Safety
/// The function must originate from a retained compatible library. Handles must be live on
/// the selected device. Pointers must cover validated typed extents and appropriate host/device
/// pointer mode; retained leases must prevent conflicting access through terminal completion.
#[inline]
pub unsafe fn invoke_rocblas_sscal(
    function: &rocblas::Sscal,
    handle: rocblas::RocblasHandle,
    length: c_int,
    alpha: *const f32,
    values: *mut f32,
    stride: c_int,
) -> rocblas::RocblasStatus {
    // SAFETY: caller supplies the validated pointer, handle, library and lease contract above.
    unsafe { function(handle, length, alpha, values, stride) }
}

/// Resolve the exact `rocblas_get_pointer_mode` ABI while retaining the library borrow.
///
/// # Safety
/// The library must be a compatible rocBLAS provider and remain loaded through every use.
pub unsafe fn require_rocblas_get_pointer_mode(
    library: &Library,
) -> Result<Symbol<'_, rocblas::GetPointerMode>, libloading::Error> {
    // SAFETY: SDK name and ABI are paired here; caller retains the compatible library.
    unsafe { symbol(library, b"rocblas_get_pointer_mode\0") }
}
/// Invoke `rocblas_get_pointer_mode` with the caller's validated selected-device resources.
///
/// # Safety
/// The function must originate from a retained compatible library. Handles must be live on
/// the selected device. Pointers must cover validated typed extents and appropriate host/device
/// pointer mode; retained leases must prevent conflicting access through terminal completion.
#[inline]
pub unsafe fn invoke_rocblas_get_pointer_mode(
    function: &rocblas::GetPointerMode,
    handle: rocblas::RocblasHandle,
    mode: *mut c_int,
) -> rocblas::RocblasStatus {
    // SAFETY: caller supplies the validated pointer, handle, library and lease contract above.
    unsafe { function(handle, mode) }
}

/// Resolve the exact `rocblas_set_pointer_mode` ABI while retaining the library borrow.
///
/// # Safety
/// The library must be a compatible rocBLAS provider and remain loaded through every use.
pub unsafe fn require_rocblas_set_pointer_mode(
    library: &Library,
) -> Result<Symbol<'_, rocblas::SetPointerMode>, libloading::Error> {
    // SAFETY: SDK name and ABI are paired here; caller retains the compatible library.
    unsafe { symbol(library, b"rocblas_set_pointer_mode\0") }
}
/// Invoke `rocblas_set_pointer_mode` with the caller's validated selected-device resources.
///
/// # Safety
/// The function must originate from a retained compatible library. Handles must be live on
/// the selected device. Pointers must cover validated typed extents and appropriate host/device
/// pointer mode; retained leases must prevent conflicting access through terminal completion.
#[inline]
pub unsafe fn invoke_rocblas_set_pointer_mode(
    function: &rocblas::SetPointerMode,
    handle: rocblas::RocblasHandle,
    mode: c_int,
) -> rocblas::RocblasStatus {
    // SAFETY: caller supplies the validated pointer, handle, library and lease contract above.
    unsafe { function(handle, mode) }
}

/// Resolve the exact `rocblas_set_stream` ABI while retaining the library borrow.
///
/// # Safety
/// The library must be a compatible rocBLAS provider and remain loaded through every use.
pub unsafe fn require_rocblas_set_stream(
    library: &Library,
) -> Result<Symbol<'_, rocblas::SetStream>, libloading::Error> {
    // SAFETY: SDK name and ABI are paired here; caller retains the compatible library.
    unsafe { symbol(library, b"rocblas_set_stream\0") }
}
/// Invoke `rocblas_set_stream` with the caller's validated selected-device resources.
///
/// # Safety
/// The function must originate from a retained compatible library. Handles must be live on
/// the selected device. Pointers must cover validated typed extents and appropriate host/device
/// pointer mode; retained leases must prevent conflicting access through terminal completion.
#[inline]
pub unsafe fn invoke_rocblas_set_stream(
    function: &rocblas::SetStream,
    handle: rocblas::RocblasHandle,
    stream: *mut c_void,
) -> rocblas::RocblasStatus {
    // SAFETY: caller supplies the validated pointer, handle, library and lease contract above.
    unsafe { function(handle, stream) }
}

/// Open a provider library; callers retain it for every resolved function use.
///
/// # Safety
/// Loading runs foreign initializers; the candidate must be a trusted compatible provider.
pub unsafe fn open_library(candidate: &std::ffi::OsStr) -> Result<Library, libloading::Error> {
    // SAFETY: caller selects a trusted provider and owns the returned library lifetime.
    unsafe { Library::new(candidate) }
}

#[rustfmt::skip]
use crate::{
    HipRuntime,
    HipError,
};
use std::ffi::CStr;
fn hip_call<T: Copy>(
    runtime: &HipRuntime,
    symbol: &'static str,
    retained: &Result<T, HipError>,
    invoke: impl FnOnce(T) -> hip::HipResult,
) -> Result<(), HipError> {
    // Resolve failures stay cold-owned but retain the same operation-specific error when used.
    let function = *retained.as_ref().map_err(Clone::clone)?;
    if symbol != "hipSetDevice" {
        // HIP current-device state remains thread-local. Pointer retention removes resolution,
        // never this per-operation selection required by cloned runtimes on another thread.
        let setter = runtime.0.api.set_device.as_ref().map_err(Clone::clone)?;
        #[cfg(feature = "allocation-census")]
        census::call("hipSetDevice");
        // SAFETY: RuntimeInner retains this exact HIP library and selected device ordinal.
        let status = unsafe { setter(runtime.0.device) };
        if status != hip::HIP_SUCCESS {
            return Err(runtime.error("hipSetDevice", status));
        }
    }
    #[cfg(feature = "allocation-census")]
    census::call(symbol);
    let status = invoke(function);
    if status == hip::HIP_SUCCESS {
        Ok(())
    } else {
        Err(runtime.error(symbol, status))
    }
}

pub fn hip_error(runtime: &HipRuntime, operation: &'static str, code: hip::HipResult) -> HipError {
    // Error-string lookup is optional; preserve numeric status even if the symbol is absent.
    let detail = runtime
        .0
        .api
        .get_error_string
        .as_ref()
        .ok()
        .map(|f| {
            #[cfg(feature = "allocation-census")]
            census::call("hipGetErrorString");
            // SAFETY: exact retained HIP error-string entry; HIP owns the returned static string.
            unsafe { f(code) }
        })
        .filter(|p| !p.is_null())
        .map(|p| unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned());
    HipError::Runtime {
        operation,
        code,
        detail,
    }
}

/// Invoke `hipGetDeviceCount` on the retained selected HIP runtime.
///
/// # Safety
/// Handles and allocations must belong to this runtime's device and remain live through
/// terminal completion. Pointer outputs must address writable storage; transfer buffers must
/// cover the exact declared bytes, with correct direction and retained nonconflicting leases.
#[inline]
#[allow(non_snake_case)] // Preserve private SDK names at the foreign boundary.
pub unsafe fn invoke_hipGetDeviceCount(
    runtime: &HipRuntime,
    count: *mut c_int,
) -> Result<(), HipError> {
    hip_call(
        runtime,
        "hipGetDeviceCount",
        &runtime.0.api.get_device_count,
        |f: hip::GetDeviceCount| unsafe { f(count) },
    )
}

/// Invoke `hipDeviceTotalMem` on the retained selected HIP runtime.
///
/// # Safety
/// Handles and allocations must belong to this runtime's device and remain live through
/// terminal completion. Pointer outputs must address writable storage; transfer buffers must
/// cover the exact declared bytes, with correct direction and retained nonconflicting leases.
#[inline]
#[allow(non_snake_case)] // Preserve private SDK names at the foreign boundary.
pub unsafe fn invoke_hipDeviceTotalMem(
    runtime: &HipRuntime,
    bytes: *mut usize,
    device: hip::HipDevice,
) -> Result<(), HipError> {
    hip_call(
        runtime,
        "hipDeviceTotalMem",
        &runtime.0.api.device_total_mem,
        |f: hip::DeviceTotalMem| unsafe { f(bytes, device) },
    )
}

/// Invoke `hipMemGetInfo` on the retained selected HIP runtime.
///
/// # Safety
/// Handles and allocations must belong to this runtime's device and remain live through
/// terminal completion. Pointer outputs must address writable storage; transfer buffers must
/// cover the exact declared bytes, with correct direction and retained nonconflicting leases.
#[inline]
#[allow(non_snake_case)] // Preserve private SDK names at the foreign boundary.
pub unsafe fn invoke_hipMemGetInfo(
    runtime: &HipRuntime,
    free: *mut usize,
    total: *mut usize,
) -> Result<(), HipError> {
    hip_call(
        runtime,
        "hipMemGetInfo",
        &runtime.0.api.mem_get_info,
        |f: hip::MemGetInfo| unsafe { f(free, total) },
    )
}

/// Invoke `hipMalloc` on the retained selected HIP runtime.
///
/// # Safety
/// Handles and allocations must belong to this runtime's device and remain live through
/// terminal completion. Pointer outputs must address writable storage; transfer buffers must
/// cover the exact declared bytes, with correct direction and retained nonconflicting leases.
#[inline]
#[allow(non_snake_case)] // Preserve private SDK names at the foreign boundary.
pub unsafe fn invoke_hipMalloc(
    runtime: &HipRuntime,
    allocation: *mut *mut c_void,
    bytes: usize,
) -> Result<(), HipError> {
    hip_call(
        runtime,
        "hipMalloc",
        &runtime.0.api.malloc,
        |f: hip::Malloc| unsafe { f(allocation, bytes) },
    )
}

/// Invoke `hipModuleLoadData` on the retained selected HIP runtime.
///
/// # Safety
/// Handles and allocations must belong to this runtime's device and remain live through
/// terminal completion. Pointer outputs must address writable storage; transfer buffers must
/// cover the exact declared bytes, with correct direction and retained nonconflicting leases.
#[inline]
#[allow(non_snake_case)] // Preserve private SDK names at the foreign boundary.
pub unsafe fn invoke_hipModuleLoadData(
    runtime: &HipRuntime,
    module: *mut hip::ModuleHandle,
    image: *const c_void,
) -> Result<(), HipError> {
    hip_call(
        runtime,
        "hipModuleLoadData",
        &runtime.0.api.module_load_data,
        |f: hip::ModuleLoadData| unsafe { f(module, image) },
    )
}

/// Invoke `hipStreamCreate` on the retained selected HIP runtime.
///
/// # Safety
/// Handles and allocations must belong to this runtime's device and remain live through
/// terminal completion. Pointer outputs must address writable storage; transfer buffers must
/// cover the exact declared bytes, with correct direction and retained nonconflicting leases.
#[inline]
#[allow(non_snake_case)] // Preserve private SDK names at the foreign boundary.
pub unsafe fn invoke_hipStreamCreate(
    runtime: &HipRuntime,
    stream: *mut hip::HipStream,
) -> Result<(), HipError> {
    hip_call(
        runtime,
        "hipStreamCreate",
        &runtime.0.api.stream_create,
        |f: hip::StreamCreate| unsafe { f(stream) },
    )
}

/// Invoke `hipEventCreateWithFlags` on the retained selected HIP runtime.
///
/// # Safety
/// Handles and allocations must belong to this runtime's device and remain live through
/// terminal completion. Pointer outputs must address writable storage; transfer buffers must
/// cover the exact declared bytes, with correct direction and retained nonconflicting leases.
#[inline]
#[allow(non_snake_case)] // Preserve private SDK names at the foreign boundary.
pub unsafe fn invoke_hipEventCreateWithFlags(
    runtime: &HipRuntime,
    event: *mut hip::HipEvent,
    flags: c_int,
) -> Result<(), HipError> {
    hip_call(
        runtime,
        "hipEventCreateWithFlags",
        &runtime.0.api.event_create_with_flags,
        |f: hip::EventCreate| unsafe { f(event, flags) },
    )
}

/// Invoke `hipEventElapsedTime` on the retained selected HIP runtime.
///
/// # Safety
/// Handles and allocations must belong to this runtime's device and remain live through
/// terminal completion. Pointer outputs must address writable storage; transfer buffers must
/// cover the exact declared bytes, with correct direction and retained nonconflicting leases.
#[inline]
#[allow(non_snake_case)] // Preserve private SDK names at the foreign boundary.
pub unsafe fn invoke_hipEventElapsedTime(
    runtime: &HipRuntime,
    milliseconds: *mut f32,
    start: hip::HipEvent,
    end: hip::HipEvent,
) -> Result<(), HipError> {
    hip_call(
        runtime,
        "hipEventElapsedTime",
        &runtime.0.api.event_elapsed_time,
        |f: hip::EventElapsedTime| unsafe { f(milliseconds, start, end) },
    )
}

/// Invoke `hipDeviceGetName` on the retained selected HIP runtime.
///
/// # Safety
/// Handles and allocations must belong to this runtime's device and remain live through
/// terminal completion. Pointer outputs must address writable storage; transfer buffers must
/// cover the exact declared bytes, with correct direction and retained nonconflicting leases.
#[inline]
#[allow(non_snake_case)] // Preserve private SDK names at the foreign boundary.
pub unsafe fn invoke_hipDeviceGetName(
    runtime: &HipRuntime,
    name: *mut c_char,
    capacity: c_int,
    device: hip::HipDevice,
) -> Result<(), HipError> {
    hip_call(
        runtime,
        "hipDeviceGetName",
        &runtime.0.api.device_get_name,
        |f: hip::GetDeviceName| unsafe { f(name, capacity, device) },
    )
}

/// Invoke the optional exact PCI-location ABI retained by this runtime.
///
/// # Safety
/// The buffer must cover `capacity` writable bytes and ordinal must refer to this provider.
#[inline]
#[allow(non_snake_case)] // Exact private SDK boundary.
pub unsafe fn invoke_hipDeviceGetPCIBusId(
    runtime: &HipRuntime,
    buffer: *mut c_char,
    capacity: c_int,
    ordinal: c_int,
) -> Result<(), HipError> {
    hip_call(
        runtime,
        "hipDeviceGetPCIBusId",
        &runtime.0.api.device_get_pci_bus_id,
        |function: hip::GetDevicePciBusId| {
            // SAFETY: caller validates the exact capacity and retained provider ordinal.
            unsafe { function(buffer, capacity, ordinal) }
        },
    )
}

/// Invoke `hipDeviceGet` on the retained selected HIP runtime.
///
/// # Safety
/// Handles and allocations must belong to this runtime's device and remain live through
/// terminal completion. Pointer outputs must address writable storage; transfer buffers must
/// cover the exact declared bytes, with correct direction and retained nonconflicting leases.
#[inline]
#[allow(non_snake_case)] // Preserve private SDK names at the foreign boundary.
pub unsafe fn invoke_hipDeviceGet(
    runtime: &HipRuntime,
    device: *mut hip::HipDevice,
    ordinal: c_int,
) -> Result<(), HipError> {
    hip_call(
        runtime,
        "hipDeviceGet",
        &runtime.0.api.device_get,
        |f: hip::GetDevice| unsafe { f(device, ordinal) },
    )
}

/// Invoke `hipSetDevice` on the retained selected HIP runtime.
///
/// # Safety
/// Handles and allocations must belong to this runtime's device and remain live through
/// terminal completion. Pointer outputs must address writable storage; transfer buffers must
/// cover the exact declared bytes, with correct direction and retained nonconflicting leases.
#[inline]
#[allow(non_snake_case)] // Preserve private SDK names at the foreign boundary.
pub unsafe fn invoke_hipSetDevice(
    runtime: &HipRuntime,
    device: hip::HipDevice,
) -> Result<(), HipError> {
    hip_call(
        runtime,
        "hipSetDevice",
        &runtime.0.api.set_device,
        |f: hip::SetDevice| unsafe { f(device) },
    )
}

/// Invoke `hipFree` on the retained selected HIP runtime.
///
/// # Safety
/// Handles and allocations must belong to this runtime's device and remain live through
/// terminal completion. Pointer outputs must address writable storage; transfer buffers must
/// cover the exact declared bytes, with correct direction and retained nonconflicting leases.
#[inline]
#[allow(non_snake_case)] // Preserve private SDK names at the foreign boundary.
pub unsafe fn invoke_hipFree(
    runtime: &HipRuntime,
    allocation: *mut c_void,
) -> Result<(), HipError> {
    hip_call(
        runtime,
        "hipFree",
        &runtime.0.api.free,
        |f: hip::Free| unsafe { f(allocation) },
    )
}

/// Invoke `hipMemcpy` on the retained selected HIP runtime.
///
/// # Safety
/// Handles and allocations must belong to this runtime's device and remain live through
/// terminal completion. Pointer outputs must address writable storage; transfer buffers must
/// cover the exact declared bytes, with correct direction and retained nonconflicting leases.
#[inline]
#[allow(non_snake_case)] // Preserve private SDK names at the foreign boundary.
pub unsafe fn invoke_hipMemcpy(
    runtime: &HipRuntime,
    destination: *mut c_void,
    source: *const c_void,
    bytes: usize,
    direction: c_int,
) -> Result<(), HipError> {
    hip_call(
        runtime,
        "hipMemcpy",
        &runtime.0.api.memcpy,
        |f: hip::Memcpy| unsafe {
            #[cfg(feature = "allocation-census")]
            census::copy(direction);
            f(destination, source, bytes, direction)
        },
    )
}

/// Invoke `hipDeviceSynchronize` on the retained selected HIP runtime.
///
/// # Safety
/// Handles and allocations must belong to this runtime's device and remain live through
/// terminal completion. Pointer outputs must address writable storage; transfer buffers must
/// cover the exact declared bytes, with correct direction and retained nonconflicting leases.
#[inline]
#[allow(non_snake_case)] // Preserve private SDK names at the foreign boundary.
pub unsafe fn invoke_hipDeviceSynchronize(runtime: &HipRuntime) -> Result<(), HipError> {
    hip_call(
        runtime,
        "hipDeviceSynchronize",
        &runtime.0.api.device_synchronize,
        |f: hip::HipNoArgStatus| unsafe { f() },
    )
}

/// Invoke `hipStreamDestroy` on the retained selected HIP runtime.
///
/// # Safety
/// Handles and allocations must belong to this runtime's device and remain live through
/// terminal completion. Pointer outputs must address writable storage; transfer buffers must
/// cover the exact declared bytes, with correct direction and retained nonconflicting leases.
#[inline]
#[allow(non_snake_case)] // Preserve private SDK names at the foreign boundary.
pub unsafe fn invoke_hipStreamDestroy(
    runtime: &HipRuntime,
    stream: hip::HipStream,
) -> Result<(), HipError> {
    hip_call(
        runtime,
        "hipStreamDestroy",
        &runtime.0.api.stream_destroy,
        |f: hip::StreamDestroy| unsafe { f(stream) },
    )
}

/// Invoke `hipStreamSynchronize` on the retained selected HIP runtime.
///
/// # Safety
/// Handles and allocations must belong to this runtime's device and remain live through
/// terminal completion. Pointer outputs must address writable storage; transfer buffers must
/// cover the exact declared bytes, with correct direction and retained nonconflicting leases.
#[inline]
#[allow(non_snake_case)] // Preserve private SDK names at the foreign boundary.
pub unsafe fn invoke_hipStreamSynchronize(
    runtime: &HipRuntime,
    stream: hip::HipStream,
) -> Result<(), HipError> {
    hip_call(
        runtime,
        "hipStreamSynchronize",
        &runtime.0.api.stream_synchronize,
        |f: hip::StreamSynchronize| unsafe { f(stream) },
    )
}

/// Invoke `hipEventRecord` on the retained selected HIP runtime.
///
/// # Safety
/// Handles and allocations must belong to this runtime's device and remain live through
/// terminal completion. Pointer outputs must address writable storage; transfer buffers must
/// cover the exact declared bytes, with correct direction and retained nonconflicting leases.
#[inline]
#[allow(non_snake_case)] // Preserve private SDK names at the foreign boundary.
pub unsafe fn invoke_hipEventRecord(
    runtime: &HipRuntime,
    event: hip::HipEvent,
    stream: hip::HipStream,
) -> Result<(), HipError> {
    hip_call(
        runtime,
        "hipEventRecord",
        &runtime.0.api.event_record,
        |f: hip::EventRecord| unsafe { f(event, stream) },
    )
}

/// Invoke `hipEventDestroy` on the retained selected HIP runtime.
///
/// # Safety
/// Handles and allocations must belong to this runtime's device and remain live through
/// terminal completion. Pointer outputs must address writable storage; transfer buffers must
/// cover the exact declared bytes, with correct direction and retained nonconflicting leases.
#[inline]
#[allow(non_snake_case)] // Preserve private SDK names at the foreign boundary.
pub unsafe fn invoke_hipEventDestroy(
    runtime: &HipRuntime,
    event: hip::HipEvent,
) -> Result<(), HipError> {
    hip_call(
        runtime,
        "hipEventDestroy",
        &runtime.0.api.event_destroy,
        |f: hip::EventDestroy| unsafe { f(event) },
    )
}

/// Invoke `hipEventSynchronize` on the retained selected HIP runtime.
///
/// # Safety
/// Handles and allocations must belong to this runtime's device and remain live through
/// terminal completion. Pointer outputs must address writable storage; transfer buffers must
/// cover the exact declared bytes, with correct direction and retained nonconflicting leases.
#[inline]
#[allow(non_snake_case)] // Preserve private SDK names at the foreign boundary.
pub unsafe fn invoke_hipEventSynchronize(
    runtime: &HipRuntime,
    event: hip::HipEvent,
) -> Result<(), HipError> {
    hip_call(
        runtime,
        "hipEventSynchronize",
        &runtime.0.api.event_synchronize,
        |f: hip::EventSynchronize| unsafe { f(event) },
    )
}

/// Invoke `hipModuleUnload` on the retained selected HIP runtime.
///
/// # Safety
/// Handles and allocations must belong to this runtime's device and remain live through
/// terminal completion. Pointer outputs must address writable storage; transfer buffers must
/// cover the exact declared bytes, with correct direction and retained nonconflicting leases.
#[inline]
#[allow(non_snake_case)] // Preserve private SDK names at the foreign boundary.
pub unsafe fn invoke_hipModuleUnload(
    runtime: &HipRuntime,
    module: hip::ModuleHandle,
) -> Result<(), HipError> {
    hip_call(
        runtime,
        "hipModuleUnload",
        &runtime.0.api.module_unload,
        |f: hip::ModuleUnload| unsafe { f(module) },
    )
}

/// Invoke `hipModuleGetFunction` on the retained selected HIP runtime.
///
/// # Safety
/// Handles and allocations must belong to this runtime's device and remain live through
/// terminal completion. Pointer outputs must address writable storage; transfer buffers must
/// cover the exact declared bytes, with correct direction and retained nonconflicting leases.
#[inline]
#[allow(non_snake_case)] // Preserve private SDK names at the foreign boundary.
pub unsafe fn invoke_hipModuleGetFunction(
    runtime: &HipRuntime,
    kernel: *mut hip::KernelHandle,
    module: hip::ModuleHandle,
    name: *const c_char,
) -> Result<(), HipError> {
    hip_call(
        runtime,
        "hipModuleGetFunction",
        &runtime.0.api.module_get_function,
        |f: hip::ModuleGetFunction| unsafe { f(kernel, module, name) },
    )
}

/// Invoke `hipMemcpyAsync` on the retained selected HIP runtime.
///
/// # Safety
/// Handles and allocations must belong to this runtime's device and remain live through
/// terminal completion. Pointer outputs must address writable storage; transfer buffers must
/// cover the exact declared bytes, with correct direction and retained nonconflicting leases.
#[inline]
#[allow(non_snake_case)] // Preserve private SDK names at the foreign boundary.
pub unsafe fn invoke_hipMemcpyAsync(
    runtime: &HipRuntime,
    destination: *mut c_void,
    source: *const c_void,
    bytes: usize,
    direction: c_int,
    stream: hip::HipStream,
) -> Result<(), HipError> {
    hip_call(
        runtime,
        "hipMemcpyAsync",
        &runtime.0.api.memcpy_async,
        |f: hip::MemcpyAsync| unsafe {
            #[cfg(feature = "allocation-census")]
            census::copy(direction);
            f(destination, source, bytes, direction, stream)
        },
    )
}

/// Invoke `hipStreamWaitEvent` on the retained selected HIP runtime.
///
/// # Safety
/// Handles and allocations must belong to this runtime's device and remain live through
/// terminal completion. Pointer outputs must address writable storage; transfer buffers must
/// cover the exact declared bytes, with correct direction and retained nonconflicting leases.
#[inline]
#[allow(non_snake_case)] // Preserve private SDK names at the foreign boundary.
pub unsafe fn invoke_hipStreamWaitEvent(
    runtime: &HipRuntime,
    stream: hip::HipStream,
    event: hip::HipEvent,
    flags: u32,
) -> Result<(), HipError> {
    hip_call(
        runtime,
        "hipStreamWaitEvent",
        &runtime.0.api.stream_wait_event,
        |f: hip::StreamWaitEvent| unsafe { f(stream, event, flags) },
    )
}

/// Invoke `hipModuleLaunchKernel` on the retained selected HIP runtime.
///
/// # Safety
/// Handles and allocations must belong to this runtime's device and remain live through
/// terminal completion. Pointer outputs must address writable storage; transfer buffers must
/// cover the exact declared bytes, with correct direction and retained nonconflicting leases.
#[inline]
#[allow(clippy::too_many_arguments)] // Mirror the exact private SDK ABI.
#[allow(non_snake_case)] // Preserve private SDK names at the foreign boundary.
pub unsafe fn invoke_hipModuleLaunchKernel(
    runtime: &HipRuntime,
    kernel: hip::KernelHandle,
    grid_x: u32,
    grid_y: u32,
    grid_z: u32,
    block_x: u32,
    block_y: u32,
    block_z: u32,
    shared_bytes: u32,
    stream: hip::HipStream,
    parameters: *mut *mut c_void,
    extra: *mut *mut c_void,
) -> Result<(), HipError> {
    hip_call(
        runtime,
        "hipModuleLaunchKernel",
        &runtime.0.api.module_launch_kernel,
        |f: hip::ModuleLaunchKernel| unsafe {
            f(
                kernel,
                grid_x,
                grid_y,
                grid_z,
                block_x,
                block_y,
                block_z,
                shared_bytes,
                stream,
                parameters,
                extra,
            )
        },
    )
}

/// Resolve the documented `hipGetDeviceCount` ABI from a retained compatible HIP library.
///
/// # Safety
/// The supplied library must be a compatible HIP provider, retained through all function uses.
#[allow(non_snake_case)] // Preserve private SDK names at the foreign boundary.
pub unsafe fn require_hipGetDeviceCount(
    library: &Library,
) -> Result<Symbol<'_, hip::GetDeviceCount>, libloading::Error> {
    // SAFETY: SDK name and signature are paired here; caller retains the provider.
    unsafe { symbol(library, b"hipGetDeviceCount\0") }
}
/// Invoke `hipGetDeviceCount` during discovery using validated ordinal and exact output storage.
///
/// # Safety
/// The function's library must remain loaded, output pointers must cover their exact declared
/// types/capacities, and device ordinals must come from discovery of the same provider.
#[inline]
#[allow(non_snake_case)] // Preserve private SDK names at the foreign boundary.
pub unsafe fn raw_hipGetDeviceCount(
    function: &hip::GetDeviceCount,
    count: *mut c_int,
) -> hip::HipResult {
    // SAFETY: caller provides validated outputs and retained provider/device identity.
    unsafe { function(count) }
}

/// Resolve the documented `hipDeviceGet` ABI from a retained compatible HIP library.
///
/// # Safety
/// The supplied library must be a compatible HIP provider, retained through all function uses.
#[allow(non_snake_case)] // Preserve private SDK names at the foreign boundary.
pub unsafe fn require_hipDeviceGet(
    library: &Library,
) -> Result<Symbol<'_, hip::GetDevice>, libloading::Error> {
    // SAFETY: SDK name and signature are paired here; caller retains the provider.
    unsafe { symbol(library, b"hipDeviceGet\0") }
}
/// Invoke `hipDeviceGet` during discovery using validated ordinal and exact output storage.
///
/// # Safety
/// The function's library must remain loaded, output pointers must cover their exact declared
/// types/capacities, and device ordinals must come from discovery of the same provider.
#[inline]
#[allow(non_snake_case)] // Preserve private SDK names at the foreign boundary.
pub unsafe fn raw_hipDeviceGet(
    function: &hip::GetDevice,
    device: *mut hip::HipDevice,
    ordinal: c_int,
) -> hip::HipResult {
    // SAFETY: caller provides validated outputs and retained provider/device identity.
    unsafe { function(device, ordinal) }
}

/// Resolve the documented `hipDeviceGetName` ABI from a retained compatible HIP library.
///
/// # Safety
/// The supplied library must be a compatible HIP provider, retained through all function uses.
#[allow(non_snake_case)] // Preserve private SDK names at the foreign boundary.
pub unsafe fn require_hipDeviceGetName(
    library: &Library,
) -> Result<Symbol<'_, hip::GetDeviceName>, libloading::Error> {
    // SAFETY: SDK name and signature are paired here; caller retains the provider.
    unsafe { symbol(library, b"hipDeviceGetName\0") }
}
/// Invoke `hipDeviceGetName` during discovery using validated ordinal and exact output storage.
///
/// # Safety
/// The function's library must remain loaded, output pointers must cover their exact declared
/// types/capacities, and device ordinals must come from discovery of the same provider.
#[inline]
#[allow(non_snake_case)] // Preserve private SDK names at the foreign boundary.
pub unsafe fn raw_hipDeviceGetName(
    function: &hip::GetDeviceName,
    name: *mut c_char,
    capacity: c_int,
    device: hip::HipDevice,
) -> hip::HipResult {
    // SAFETY: caller provides validated outputs and retained provider/device identity.
    unsafe { function(name, capacity, device) }
}

/// Resolve the documented `hipDeviceTotalMem` ABI from a retained compatible HIP library.
///
/// # Safety
/// The supplied library must be a compatible HIP provider, retained through all function uses.
#[allow(non_snake_case)] // Preserve private SDK names at the foreign boundary.
pub unsafe fn require_hipDeviceTotalMem(
    library: &Library,
) -> Result<Symbol<'_, hip::DeviceTotalMem>, libloading::Error> {
    // SAFETY: SDK name and signature are paired here; caller retains the provider.
    unsafe { symbol(library, b"hipDeviceTotalMem\0") }
}
/// Invoke `hipDeviceTotalMem` during discovery using validated ordinal and exact output storage.
///
/// # Safety
/// The function's library must remain loaded, output pointers must cover their exact declared
/// types/capacities, and device ordinals must come from discovery of the same provider.
#[inline]
#[allow(non_snake_case)] // Preserve private SDK names at the foreign boundary.
pub unsafe fn raw_hipDeviceTotalMem(
    function: &hip::DeviceTotalMem,
    bytes: *mut usize,
    device: hip::HipDevice,
) -> hip::HipResult {
    // SAFETY: caller provides validated outputs and retained provider/device identity.
    unsafe { function(bytes, device) }
}

/// Resolve the documented `hipDeviceGetPCIBusId` ABI from a retained compatible HIP library.
///
/// # Safety
/// The supplied library must be a compatible HIP provider, retained through all function uses.
#[allow(non_snake_case)] // Preserve private SDK names at the foreign boundary.
pub unsafe fn require_hipDeviceGetPCIBusId(
    library: &Library,
) -> Result<Symbol<'_, hip::GetDevicePciBusId>, libloading::Error> {
    // SAFETY: SDK name and signature are paired here; caller retains the provider.
    unsafe { symbol(library, b"hipDeviceGetPCIBusId\0") }
}
/// Invoke `hipDeviceGetPCIBusId` during discovery using validated ordinal and exact output storage.
///
/// # Safety
/// The function's library must remain loaded, output pointers must cover their exact declared
/// types/capacities, and device ordinals must come from discovery of the same provider.
#[inline]
#[allow(non_snake_case)] // Preserve private SDK names at the foreign boundary.
pub unsafe fn raw_hipDeviceGetPCIBusId(
    function: &hip::GetDevicePciBusId,
    bus_id: *mut c_char,
    capacity: c_int,
    ordinal: c_int,
) -> hip::HipResult {
    // SAFETY: caller provides validated outputs and retained provider/device identity.
    unsafe { function(bus_id, capacity, ordinal) }
}

/// Resolve the documented `hipDeviceGetAttribute` ABI from a retained compatible HIP library.
///
/// # Safety
/// The supplied library must be a compatible HIP provider, retained through all function uses.
#[allow(non_snake_case)] // Preserve private SDK names at the foreign boundary.
pub unsafe fn require_hipDeviceGetAttribute(
    library: &Library,
) -> Result<Symbol<'_, hip::DeviceGetAttribute>, libloading::Error> {
    // SAFETY: SDK name and signature are paired here; caller retains the provider.
    unsafe { symbol(library, b"hipDeviceGetAttribute\0") }
}
/// Invoke `hipDeviceGetAttribute` during discovery using validated ordinal and exact output storage.
///
/// # Safety
/// The function's library must remain loaded, output pointers must cover their exact declared
/// types/capacities, and device ordinals must come from discovery of the same provider.
#[inline]
#[allow(non_snake_case)] // Preserve private SDK names at the foreign boundary.
pub unsafe fn raw_hipDeviceGetAttribute(
    function: &hip::DeviceGetAttribute,
    value: *mut c_int,
    attribute: c_int,
    ordinal: c_int,
) -> hip::HipResult {
    // SAFETY: caller provides validated outputs and retained provider/device identity.
    unsafe { function(value, attribute, ordinal) }
}
#[allow(non_snake_case)] // Preserve private SDK names at the foreign boundary.
pub fn raw_hip_error(library: &Library, operation: &'static str, code: hip::HipResult) -> HipError {
    let detail =
        unsafe { crate::ffi::symbol::<hip::GetErrorString>(library, b"hipGetErrorString\0") }
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
/// Load the optional stable attribute query without extending ordinary runtime initialization.
pub fn device_get_attribute(library: &Library) -> Option<hip::DeviceGetAttribute> {
    // SAFETY: HIP documents this exact C signature; the caller retains the loaded library.
    unsafe { require_hipDeviceGetAttribute(library) }
        .ok()
        .map(|function| *function)
}

use crate::HipRtcError;

/// Resolve the exact `hiprtcCreateProgram` ABI, retaining the compatible provider for its use.
#[allow(non_snake_case)] // Preserve private SDK names at the foreign boundary.
pub fn require_hiprtcCreateProgram(
    library: &Library,
) -> Result<hiprtc::CreateProgram, HipRtcError> {
    // SAFETY: SDK name and signature are paired here; compilation retains the library.
    unsafe { symbol::<hiprtc::CreateProgram>(library, b"hiprtcCreateProgram\0") }
        .map(|function| *function)
        .map_err(|error| HipRtcError::MissingSymbol {
            symbol: "hiprtcCreateProgram",
            detail: error.to_string(),
        })
}
/// Invoke `hiprtcCreateProgram` using a live retained compiler program and exact output storage.
///
/// # Safety
/// The provider must remain loaded. Program handles must be live and belong to that provider.
/// Input strings/options must be NUL terminated; output pointers must cover the compiler's
/// reported size, and writable program handles must have exclusive access.
#[inline]
#[allow(non_snake_case)] // Preserve private SDK names at the foreign boundary.
pub unsafe fn invoke_hiprtcCreateProgram(
    function: &hiprtc::CreateProgram,
    program: *mut hiprtc::Program,
    source: *const c_char,
    name: *const c_char,
    header_count: c_int,
    headers: *const *const c_char,
    include_names: *const *const c_char,
) -> hiprtc::ResultCode {
    // SAFETY: caller preserves compiler library, program ownership and exact buffer extents.
    unsafe { function(program, source, name, header_count, headers, include_names) }
}

/// Resolve the exact `hiprtcDestroyProgram` ABI, retaining the compatible provider for its use.
#[allow(non_snake_case)] // Preserve private SDK names at the foreign boundary.
pub fn require_hiprtcDestroyProgram(
    library: &Library,
) -> Result<hiprtc::DestroyProgram, HipRtcError> {
    // SAFETY: SDK name and signature are paired here; compilation retains the library.
    unsafe { symbol::<hiprtc::DestroyProgram>(library, b"hiprtcDestroyProgram\0") }
        .map(|function| *function)
        .map_err(|error| HipRtcError::MissingSymbol {
            symbol: "hiprtcDestroyProgram",
            detail: error.to_string(),
        })
}
/// Invoke `hiprtcDestroyProgram` using a live retained compiler program and exact output storage.
///
/// # Safety
/// The provider must remain loaded. Program handles must be live and belong to that provider.
/// Input strings/options must be NUL terminated; output pointers must cover the compiler's
/// reported size, and writable program handles must have exclusive access.
#[inline]
#[allow(non_snake_case)] // Preserve private SDK names at the foreign boundary.
pub unsafe fn invoke_hiprtcDestroyProgram(
    function: &hiprtc::DestroyProgram,
    program: *mut hiprtc::Program,
) -> hiprtc::ResultCode {
    // SAFETY: caller preserves compiler library, program ownership and exact buffer extents.
    unsafe { function(program) }
}

/// Resolve the exact `hiprtcCompileProgram` ABI, retaining the compatible provider for its use.
#[allow(non_snake_case)] // Preserve private SDK names at the foreign boundary.
pub fn require_hiprtcCompileProgram(
    library: &Library,
) -> Result<hiprtc::CompileProgram, HipRtcError> {
    // SAFETY: SDK name and signature are paired here; compilation retains the library.
    unsafe { symbol::<hiprtc::CompileProgram>(library, b"hiprtcCompileProgram\0") }
        .map(|function| *function)
        .map_err(|error| HipRtcError::MissingSymbol {
            symbol: "hiprtcCompileProgram",
            detail: error.to_string(),
        })
}
/// Invoke `hiprtcCompileProgram` using a live retained compiler program and exact output storage.
///
/// # Safety
/// The provider must remain loaded. Program handles must be live and belong to that provider.
/// Input strings/options must be NUL terminated; output pointers must cover the compiler's
/// reported size, and writable program handles must have exclusive access.
#[inline]
#[allow(non_snake_case)] // Preserve private SDK names at the foreign boundary.
pub unsafe fn invoke_hiprtcCompileProgram(
    function: &hiprtc::CompileProgram,
    program: hiprtc::Program,
    option_count: c_int,
    options: *const *const c_char,
) -> hiprtc::ResultCode {
    // SAFETY: caller preserves compiler library, program ownership and exact buffer extents.
    unsafe { function(program, option_count, options) }
}

/// Resolve the exact `hiprtcGetProgramLogSize` ABI, retaining the compatible provider for its use.
#[allow(non_snake_case)] // Preserve private SDK names at the foreign boundary.
pub fn require_hiprtcGetProgramLogSize(
    library: &Library,
) -> Result<hiprtc::GetProgramLogSize, HipRtcError> {
    // SAFETY: SDK name and signature are paired here; compilation retains the library.
    unsafe { symbol::<hiprtc::GetProgramLogSize>(library, b"hiprtcGetProgramLogSize\0") }
        .map(|function| *function)
        .map_err(|error| HipRtcError::MissingSymbol {
            symbol: "hiprtcGetProgramLogSize",
            detail: error.to_string(),
        })
}
/// Invoke `hiprtcGetProgramLogSize` using a live retained compiler program and exact output storage.
///
/// # Safety
/// The provider must remain loaded. Program handles must be live and belong to that provider.
/// Input strings/options must be NUL terminated; output pointers must cover the compiler's
/// reported size, and writable program handles must have exclusive access.
#[inline]
#[allow(non_snake_case)] // Preserve private SDK names at the foreign boundary.
pub unsafe fn invoke_hiprtcGetProgramLogSize(
    function: &hiprtc::GetProgramLogSize,
    program: hiprtc::Program,
    bytes: *mut usize,
) -> hiprtc::ResultCode {
    // SAFETY: caller preserves compiler library, program ownership and exact buffer extents.
    unsafe { function(program, bytes) }
}

/// Resolve the exact `hiprtcGetProgramLog` ABI, retaining the compatible provider for its use.
#[allow(non_snake_case)] // Preserve private SDK names at the foreign boundary.
pub fn require_hiprtcGetProgramLog(
    library: &Library,
) -> Result<hiprtc::GetProgramLog, HipRtcError> {
    // SAFETY: SDK name and signature are paired here; compilation retains the library.
    unsafe { symbol::<hiprtc::GetProgramLog>(library, b"hiprtcGetProgramLog\0") }
        .map(|function| *function)
        .map_err(|error| HipRtcError::MissingSymbol {
            symbol: "hiprtcGetProgramLog",
            detail: error.to_string(),
        })
}
/// Invoke `hiprtcGetProgramLog` using a live retained compiler program and exact output storage.
///
/// # Safety
/// The provider must remain loaded. Program handles must be live and belong to that provider.
/// Input strings/options must be NUL terminated; output pointers must cover the compiler's
/// reported size, and writable program handles must have exclusive access.
#[inline]
#[allow(non_snake_case)] // Preserve private SDK names at the foreign boundary.
pub unsafe fn invoke_hiprtcGetProgramLog(
    function: &hiprtc::GetProgramLog,
    program: hiprtc::Program,
    log: *mut c_char,
) -> hiprtc::ResultCode {
    // SAFETY: caller preserves compiler library, program ownership and exact buffer extents.
    unsafe { function(program, log) }
}

/// Resolve the exact `hiprtcGetCodeSize` ABI, retaining the compatible provider for its use.
#[allow(non_snake_case)] // Preserve private SDK names at the foreign boundary.
pub fn require_hiprtcGetCodeSize(library: &Library) -> Result<hiprtc::GetCodeSize, HipRtcError> {
    // SAFETY: SDK name and signature are paired here; compilation retains the library.
    unsafe { symbol::<hiprtc::GetCodeSize>(library, b"hiprtcGetCodeSize\0") }
        .map(|function| *function)
        .map_err(|error| HipRtcError::MissingSymbol {
            symbol: "hiprtcGetCodeSize",
            detail: error.to_string(),
        })
}
/// Invoke `hiprtcGetCodeSize` using a live retained compiler program and exact output storage.
///
/// # Safety
/// The provider must remain loaded. Program handles must be live and belong to that provider.
/// Input strings/options must be NUL terminated; output pointers must cover the compiler's
/// reported size, and writable program handles must have exclusive access.
#[inline]
#[allow(non_snake_case)] // Preserve private SDK names at the foreign boundary.
pub unsafe fn invoke_hiprtcGetCodeSize(
    function: &hiprtc::GetCodeSize,
    program: hiprtc::Program,
    bytes: *mut usize,
) -> hiprtc::ResultCode {
    // SAFETY: caller preserves compiler library, program ownership and exact buffer extents.
    unsafe { function(program, bytes) }
}

/// Resolve the exact `hiprtcGetCode` ABI, retaining the compatible provider for its use.
#[allow(non_snake_case)] // Preserve private SDK names at the foreign boundary.
pub fn require_hiprtcGetCode(library: &Library) -> Result<hiprtc::GetCode, HipRtcError> {
    // SAFETY: SDK name and signature are paired here; compilation retains the library.
    unsafe { symbol::<hiprtc::GetCode>(library, b"hiprtcGetCode\0") }
        .map(|function| *function)
        .map_err(|error| HipRtcError::MissingSymbol {
            symbol: "hiprtcGetCode",
            detail: error.to_string(),
        })
}
/// Invoke `hiprtcGetCode` using a live retained compiler program and exact output storage.
///
/// # Safety
/// The provider must remain loaded. Program handles must be live and belong to that provider.
/// Input strings/options must be NUL terminated; output pointers must cover the compiler's
/// reported size, and writable program handles must have exclusive access.
#[inline]
#[allow(non_snake_case)] // Preserve private SDK names at the foreign boundary.
pub unsafe fn invoke_hiprtcGetCode(
    function: &hiprtc::GetCode,
    program: hiprtc::Program,
    code: *mut c_char,
) -> hiprtc::ResultCode {
    // SAFETY: caller preserves compiler library, program ownership and exact buffer extents.
    unsafe { function(program, code) }
}

/// Resolve the exact `hiprtcGetErrorString` ABI, retaining the compatible provider for its use.
#[allow(non_snake_case)] // Preserve private SDK names at the foreign boundary.
pub fn require_hiprtcGetErrorString(
    library: &Library,
) -> Result<hiprtc::GetErrorString, HipRtcError> {
    // SAFETY: SDK name and signature are paired here; compilation retains the library.
    unsafe { symbol::<hiprtc::GetErrorString>(library, b"hiprtcGetErrorString\0") }
        .map(|function| *function)
        .map_err(|error| HipRtcError::MissingSymbol {
            symbol: "hiprtcGetErrorString",
            detail: error.to_string(),
        })
}
pub fn check_hiprtc(
    operation: &'static str,
    code: hiprtc::ResultCode,
    error_string: hiprtc::GetErrorString,
) -> Result<(), HipRtcError> {
    if code == 0 {
        return Ok(());
    }
    let detail = unsafe {
        let pointer = error_string(code);
        if pointer.is_null() {
            "unknown HIPRTC error".into()
        } else {
            CStr::from_ptr(pointer).to_string_lossy().into_owned()
        }
    };
    Err(HipRtcError::Api {
        operation,
        code,
        detail,
    })
}

/// Resolve the exact `rocblas_dasum` ABI while retaining the library borrow.
///
/// # Safety
/// The library must be a compatible rocBLAS provider and remain loaded through every use.
pub unsafe fn require_rocblas_dasum(
    library: &Library,
) -> Result<Symbol<'_, rocblas::Dasum>, libloading::Error> {
    // SAFETY: SDK name and ABI are paired here; caller retains the compatible library.
    unsafe { symbol(library, b"rocblas_dasum\0") }
}
/// Invoke `rocblas_dasum` with the caller's validated selected-device resources.
///
/// # Safety
/// The function must originate from a retained compatible library. Handles must be live on
/// the selected device. Pointers must cover validated typed extents and appropriate host/device
/// pointer mode; retained leases must prevent conflicting access through terminal completion.
#[inline]
pub unsafe fn invoke_rocblas_dasum(
    function: &rocblas::Dasum,
    handle: rocblas::RocblasHandle,
    length: c_int,
    input: *const f64,
    stride: c_int,
    result: *mut f64,
) -> rocblas::RocblasStatus {
    // SAFETY: caller supplies the validated pointer, handle, library and lease contract above.
    unsafe { function(handle, length, input, stride, result) }
}

/// Resolve the exact `rocblas_dscal` ABI while retaining the library borrow.
///
/// # Safety
/// The library must be a compatible rocBLAS provider and remain loaded through every use.
pub unsafe fn require_rocblas_dscal(
    library: &Library,
) -> Result<Symbol<'_, rocblas::Dscal>, libloading::Error> {
    // SAFETY: SDK name and ABI are paired here; caller retains the compatible library.
    unsafe { symbol(library, b"rocblas_dscal\0") }
}
/// Invoke `rocblas_dscal` with the caller's validated selected-device resources.
///
/// # Safety
/// The function must originate from a retained compatible library. Handles must be live on
/// the selected device. Pointers must cover validated typed extents and appropriate host/device
/// pointer mode; retained leases must prevent conflicting access through terminal completion.
#[inline]
pub unsafe fn invoke_rocblas_dscal(
    function: &rocblas::Dscal,
    handle: rocblas::RocblasHandle,
    length: c_int,
    alpha: *const f64,
    values: *mut f64,
    stride: c_int,
) -> rocblas::RocblasStatus {
    // SAFETY: caller supplies the validated pointer, handle, library and lease contract above.
    unsafe { function(handle, length, alpha, values, stride) }
}

#[path = "blas_vector/blas_vector.rs"]
mod blas_vector;
#[rustfmt::skip]
pub use blas_vector::{
    VectorFunctions,
    retain as retain_blas_vector_functions,
};

#[path = "blas_handle/blas_handle.rs"]
mod blas_handle;
#[rustfmt::skip]
pub use blas_handle::{
    BlasHandleFunctions,
    retain as retain_blas_handle_functions,
};
