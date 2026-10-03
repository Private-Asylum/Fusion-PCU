//! Caller-thread HIP boundary census, compiled out of primary execution.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RocmApiCensus {
    pub symbol_resolutions: u64,
    pub runtime_calls: u64,
    pub device_selections: u64,
    pub allocations: u64,
    pub frees: u64,
    pub host_to_device_copies: u64,
    pub device_to_host_copies: u64,
    pub kernel_launches: u64,
    pub event_creates: u64,
    pub event_records: u64,
    pub event_waits: u64,
    pub event_destroys: u64,
    pub module_loads: u64,
}
std::thread_local! { static CENSUS: std::cell::Cell<RocmApiCensus> = const { std::cell::Cell::new(RocmApiCensus { symbol_resolutions: 0, runtime_calls: 0, device_selections: 0, allocations: 0, frees: 0, host_to_device_copies: 0, device_to_host_copies: 0, kernel_launches: 0, event_creates: 0, event_records: 0, event_waits: 0, event_destroys: 0, module_loads: 0 }) }; }
#[must_use]
pub fn rocm_api_census() -> RocmApiCensus {
    CENSUS.get()
}
pub fn reset_rocm_api_census() {
    CENSUS.set(RocmApiCensus::default());
}
pub(super) fn symbol() {
    CENSUS.with(|cell| {
        let mut value = cell.get();
        value.symbol_resolutions += 1;
        cell.set(value);
    });
}
pub(super) fn copy(direction: i32) {
    CENSUS.with(|cell| {
        let mut value = cell.get();
        match direction {
            1 => value.host_to_device_copies += 1,
            2 => value.device_to_host_copies += 1,
            _ => (),
        }
        cell.set(value);
    });
}
pub(super) fn call(name: &str) {
    CENSUS.with(|cell| {
        let mut value = cell.get();
        value.runtime_calls += 1;
        match name {
            "hipSetDevice" => value.device_selections += 1,
            "hipMalloc" => value.allocations += 1,
            "hipFree" => value.frees += 1,
            "hipModuleLaunchKernel" => value.kernel_launches += 1,
            "hipEventCreateWithFlags" => value.event_creates += 1,
            "hipEventRecord" => value.event_records += 1,
            "hipEventSynchronize" => value.event_waits += 1,
            "hipEventDestroy" => value.event_destroys += 1,
            "hipModuleLoadData" => value.module_loads += 1,
            _ => (),
        }
        cell.set(value);
    });
}
