//! Actual annotated resident source, detached prepared graph and independent native dual publisher.
#[path = "../../tests/scalar_transport/device/device.rs"]
mod device;
#[path = "../checked_div_rem/ffi/ffi.rs"]
#[allow(dead_code)] // This cohort selects the resident native owner and shared heap counter only.
mod ffi;
#[path = "../../../cpu/tests/wide_div_rem/oracle/oracle.rs"]
#[allow(dead_code)]
// Full bit/domain text goldens remain independently qualified in source fixtures.
mod oracle;
#[path = "../../../cpu/tests/scalar_tensor/source/source.rs"]
#[allow(dead_code)]
mod owned;
#[path = "../../../cpu/tests/div_rem_roles/source/source.rs"]
#[allow(dead_code)] // Scalar Rust borrows are covered by the separate explicit resident fixture.
mod source;
#[path = "support/support.rs"]
mod support;
#[global_allocator]
static ALLOCATOR: ffi::CountingAllocator = ffi::CountingAllocator;
#[rustfmt::skip]
use criterion::{Criterion,criterion_group,criterion_main};
#[rustfmt::skip]
use pcu_facade::{global,PcuNumericalMode};
#[rustfmt::skip]
use std::sync::atomic::{AtomicUsize,Ordering};
static SCORES: AtomicUsize = AtomicUsize::new(0);
fn score(_: &global::PcuInvocationCandidate<'_>) -> i128 {
    SCORES.fetch_add(1, Ordering::Relaxed);
    0
}
fn benchmarks(criterion: &mut Criterion) {
    let (backend, identity) = device::selected();
    for mode in [PcuNumericalMode::Boundary, PcuNumericalMode::Strict] {
        global::configure(global::PcuExecutionPolicy {
            backend: global::PcuBackendChoice::Vulkan,
            numerical_mode: mode,
            score_invocation: Some(score),
            ..Default::default()
        })
        .unwrap();
        macro_rules! widths {($($ty:ty),+)=>{$(for kind in 0..5 {support::compare::<$ty>(criterion,&backend,identity,kind,mode);})+};}
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
    }
    global::clear_thread_cache().unwrap();
    global::use_defaults().unwrap();
}
criterion_group!(benches, benchmarks);
criterion_main!(benches);
