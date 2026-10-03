//! Four genuine caller boundaries with matched unique inputs, terminal status and dual publication.
extern crate pcu_facade as fusion_pcu;
#[path = "cases/cases.rs"]
mod cases;
#[path = "../../tests/scalar_transport/device/device.rs"]
mod device;
#[path = "../checked_div_rem/ffi/ffi.rs"]
#[allow(dead_code)]
// Role peers use the independent role constructor; old canonical controls retain the same owner.
mod ffi;
#[path = "../../../cpu/tests/wide_div_rem/oracle/oracle.rs"]
#[allow(dead_code)] // Complete byte oracle; text golden fixture is qualified separately.
mod oracle;
#[path = "../../../cpu/tests/div_rem_roles/source/source.rs"]
mod source;
#[path = "support/support.rs"]
mod support;
#[global_allocator]
static ALLOCATOR: ffi::CountingAllocator = ffi::CountingAllocator;
#[rustfmt::skip]
use std::sync::atomic::{AtomicUsize,Ordering};
#[rustfmt::skip]
use criterion::{Criterion,criterion_group,criterion_main};
use pcu_facade::global;
static SCORES: AtomicUsize = AtomicUsize::new(0);
fn score(_: &pcu_facade::PcuDeviceDescriptor<'_>, _: u64) -> i128 {
    SCORES.fetch_add(1, Ordering::Relaxed);
    0
}
fn benchmarks(c: &mut Criterion) {
    let (backend, identity) = device::selected();
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Vulkan,
        score_device: score,
        ..Default::default()
    })
    .unwrap();
    macro_rules! widths {($($ty:ty),+)=>{$(cases::repeated::<$ty>(c,&backend,identity);cases::unused::<$ty>(c,&backend,identity);cases::reordered::<$ty>(c,&backend,identity);cases::mixed::<$ty>(c,&backend,identity);cases::grid::<$ty>(c,&backend,identity);cases::scalar::<$ty>(c,&backend,identity);)+};}
    widths!(
        i8,
        u8,
        i16,
        u16,
        i32,
        u32,
        i64,
        u64,
        i128,
        u128,
        pcu_facade::PcuI256,
        pcu_facade::PcuU256,
        pcu_facade::PcuI512,
        pcu_facade::PcuU512
    );
    global::clear_thread_cache().unwrap();
    global::use_defaults().unwrap();
}
criterion_group!(benches, benchmarks);
criterion_main!(benches);
