//! Genuine raw producer source, explicit graph and independent byte-copy owners.
extern crate pcu_facade as fusion_pcu;
#[cfg(feature = "insights")]
#[path = "census/census.rs"]
mod census;
#[path = "../../tests/scalar_transport/device/device.rs"]
mod device;
#[path = "driver/driver.rs"]
mod driver;
#[path = "native/ffi/ffi.rs"]
#[allow(unsafe_code, dead_code)]
// Independent lifetime/transfer owner; pointwise extension is a separate target.
mod ffi;
#[cfg(feature = "insights")]
#[path = "native/heap/heap.rs"]
#[allow(unsafe_code, dead_code)]
mod heap;
#[path = "source/source.rs"]
mod source;
#[cfg(feature = "insights")]
#[global_allocator]
static ALLOCATOR: heap::CountingAllocator = heap::CountingAllocator;
fn main() {
    #[cfg(feature = "insights")]
    census::run(driver::run);
    #[cfg(not(feature = "insights"))]
    driver::run();
}
