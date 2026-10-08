//! Caller-thread foreign-boundary census, compiled out of primary execution.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CudaApiCensus {
    pub symbol_resolutions: u64,
    pub runtime_driver_calls: u64,
    pub device_selections: u64,
    pub cublas_calls: u64,
    pub cublaslt_matmul_calls: u64,
    pub allocations: u64,
    pub frees: u64,
    pub host_to_device_copies: u64,
    pub device_to_host_copies: u64,
    pub device_to_device_copies: u64,
    pub kernel_launches: u64,
    pub guarded_chain_submissions: u64,
    pub guarded_kernel_launches: u64,
    pub event_creates: u64,
    pub event_records: u64,
    pub event_waits: u64,
    pub event_destroys: u64,
    pub module_loads: u64,
    pub device_synchronizations: u64,
    pub stream_synchronizations: u64,
    /// Explicit device/stream wait time, measured only in the separate instrumentation build.
    pub completion_wait_nanoseconds: u64,
}
std::thread_local! { static CENSUS: std::cell::Cell<CudaApiCensus> = const { std::cell::Cell::new(CudaApiCensus { symbol_resolutions: 0, runtime_driver_calls: 0, device_selections: 0, cublas_calls: 0, cublaslt_matmul_calls: 0, allocations: 0, frees: 0, host_to_device_copies: 0, device_to_host_copies: 0, device_to_device_copies: 0, kernel_launches: 0, guarded_chain_submissions: 0, guarded_kernel_launches: 0, event_creates: 0, event_records: 0, event_waits: 0, event_destroys: 0, module_loads: 0, device_synchronizations: 0, stream_synchronizations: 0, completion_wait_nanoseconds: 0 }) }; }
#[must_use]
pub fn cuda_api_census() -> CudaApiCensus {
    CENSUS.get()
}
pub fn reset_cuda_api_census() {
    CENSUS.set(CudaApiCensus::default());
}
pub(super) fn symbol() {
    CENSUS.with(|cell| {
        let mut census = cell.get();
        census.symbol_resolutions += 1;
        cell.set(census);
    });
}
pub(super) fn call(name: &str, copy_direction: Option<i32>) {
    CENSUS.with(|cell| {
        let mut census = cell.get();
        census.runtime_driver_calls += 1;
        match name {
            "cudaSetDevice" => census.device_selections += 1,
            "cudaDeviceSynchronize" => census.device_synchronizations += 1,
            "cudaStreamSynchronize" => census.stream_synchronizations += 1,
            "cudaMalloc" => census.allocations += 1,
            "cudaFree" => census.frees += 1,
            "cuLaunchKernel" => census.kernel_launches += 1,
            "cudaEventCreateWithFlags" => census.event_creates += 1,
            "cudaEventRecord" => census.event_records += 1,
            "cudaEventSynchronize" => census.event_waits += 1,
            "cudaEventDestroy" => census.event_destroys += 1,
            "cuModuleLoadData" => census.module_loads += 1,
            "cudaMemcpy" | "cudaMemcpyAsync" => match copy_direction {
                Some(1) => census.host_to_device_copies += 1,
                Some(2) => census.device_to_host_copies += 1,
                Some(3) => census.device_to_device_copies += 1,
                _ => (),
            },
            _ => (),
        }
        cell.set(census);
    });
}

pub(super) fn completion_time(duration: std::time::Duration) {
    CENSUS.with(|cell| {
        let mut value = cell.get();
        value.completion_wait_nanoseconds = value
            .completion_wait_nanoseconds
            .saturating_add(u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX));
        cell.set(value);
    });
}

pub(super) fn blas() {
    CENSUS.with(|cell| {
        let mut value = cell.get();
        value.cublas_calls += 1;
        cell.set(value);
    });
}
pub(super) fn lt_matmul() {
    CENSUS.with(|cell| {
        let mut value = cell.get();
        value.cublaslt_matmul_calls += 1;
        cell.set(value);
    });
}

#[cfg(feature = "tensor")]
pub fn guarded_chain() {
    CENSUS.with(|cell| {
        let mut value = cell.get();
        value.guarded_chain_submissions += 1;
        cell.set(value);
    });
}
#[cfg(feature = "tensor")]
pub fn guarded_kernel() {
    CENSUS.with(|cell| {
        let mut value = cell.get();
        value.guarded_kernel_launches += 1;
        cell.set(value);
    });
}
