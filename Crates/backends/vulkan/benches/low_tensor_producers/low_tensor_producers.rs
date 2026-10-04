//! Genuine immutable producers, typed graph and independent packed GLSL control.
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
// Handwritten SDK lifetime owner retains additional cold utilities.
mod ffi;
#[path = "../../../rocm/benches/low_tensor_producers/oracle/oracle.rs"]
#[allow(dead_code)] // Independent midpoint oracle also serves the full boundary corpus.
mod oracle;
#[path = "../../../rocm/benches/low_tensor_producers/source/source.rs"]
#[allow(dead_code)] // Genuine source authoring keeps the reciprocal ROCm fixture provenance.
mod source;
#[global_allocator]
static ALLOCATOR: ffi::CountingAllocator = ffi::CountingAllocator;
use criterion::{Criterion, criterion_group, criterion_main};
fn producers(c: &mut Criterion) {
    driver::run(c);
}
criterion_group!(benches, producers);
criterion_main!(benches);
