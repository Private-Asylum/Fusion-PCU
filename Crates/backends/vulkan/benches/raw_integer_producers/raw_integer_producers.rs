//! Genuine raw integer producer source, explicit graph and independent byte-copy owners.
extern crate pcu_facade as fusion_pcu;
#[cfg(feature = "insights")]
#[path = "census/census.rs"]
mod census;
#[path = "../../tests/scalar_transport/device/device.rs"]
mod device;
#[path = "driver/driver.rs"]
mod driver;
#[path = "../raw_float_producers/native/ffi/ffi.rs"]
#[allow(unsafe_code, dead_code)]
// Independent lifetime/transfer owner; pointwise extension is a separate target.
mod ffi;
#[cfg(feature = "insights")]
#[path = "../raw_float_producers/native/heap/heap.rs"]
#[allow(unsafe_code, dead_code)]
mod heap;
#[path = "../../../rocm/benches/raw_integer_producers/source/source.rs"]
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
