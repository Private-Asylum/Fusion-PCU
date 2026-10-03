//! Genuine six-profile joint division with independent exact-byte checked controls.
#[rustfmt::skip]
use std::sync::atomic::{
    AtomicUsize,
    Ordering,
};
#[rustfmt::skip]
use criterion::{
    Criterion,
    criterion_group,
    criterion_main,
};
use pcu_facade::global;
#[path = "../div_rem_roles/cases/cases.rs"]
mod cases;
#[path = "../checked_div_rem/ffi/ffi.rs"]
mod ffi;
#[path = "../../tests/wide_div_rem/oracle/oracle.rs"]
#[allow(dead_code)] // Golden/parser helpers qualify the independent oracle in separate tests.
mod oracle;
#[path = "../../tests/portable_div_rem/source/source.rs"]
mod source;
#[path = "support/support.rs"]
mod support;
#[global_allocator]
static ALLOCATOR: ffi::CountingAllocator = ffi::CountingAllocator;
static SCORES: AtomicUsize = AtomicUsize::new(0);
fn score(_: &global::PcuInvocationCandidate<'_>) -> i128 {
    SCORES.fetch_add(1, Ordering::Relaxed);
    0
}
fn width<T: oracle::Wide + pcu_facade::PcuCheckedIntegerDivision>(criterion: &mut Criterion) {
    cases::repeated::<T>(criterion);
    cases::unused::<T>(criterion);
    cases::reordered::<T>(criterion);
    cases::mixed::<T>(criterion);
    cases::grid::<T>(criterion);
    cases::scalar::<T>(criterion);
}
fn benchmarks(criterion: &mut Criterion) {
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Cpu,
        score_invocation: Some(score),
        ..Default::default()
    })
    .unwrap();
    global::clear_thread_cache().unwrap();
    width::<i8>(criterion);
    width::<u8>(criterion);
    width::<i16>(criterion);
    width::<u16>(criterion);
    width::<i32>(criterion);
    width::<u32>(criterion);
    width::<i64>(criterion);
    width::<u64>(criterion);
    width::<i128>(criterion);
    width::<u128>(criterion);
    width::<pcu_facade::PcuI256>(criterion);
    width::<pcu_facade::PcuU256>(criterion);
    width::<pcu_facade::PcuI512>(criterion);
    width::<pcu_facade::PcuU512>(criterion);
}
criterion_group!(benches, benchmarks);
criterion_main!(benches);
