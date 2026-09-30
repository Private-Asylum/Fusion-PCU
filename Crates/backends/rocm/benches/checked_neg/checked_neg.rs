//! Executed source, explicit IR and native peers for exact checked F32/F64 negation.
extern crate pcu_facade as fusion_pcu;

mod activity;
#[cfg(feature = "allocation-census")]
#[allow(unsafe_code)] // Optional allocator forwards caller layouts to System.
mod allocations;
mod discovery;
mod driver;
mod graph;
mod native;
mod oracle;
mod source;

#[rustfmt::skip]
use criterion::{
    Criterion,
    criterion_group,
    criterion_main,
};

fn checked_neg(criterion: &mut Criterion) {
    let wall = std::time::Instant::now();
    activity::activity_guard();
    fusion_pcu::global::use_defaults().unwrap();
    eprintln!(
        "Checked Neg: exact finite sign inversion. Timings include input upload, launch, terminal completion, status readback, and output download. Retained success status resets after faults. Compilation, input generation, and oracle checks are outside timing. No CPU fallback."
    );
    driver::profile::<f32, 65>(criterion);
    driver::profile::<f32, 1_048_576>(criterion);
    driver::profile::<f64, 65>(criterion);
    driver::profile::<f64, 1_048_576>(criterion);
    fusion_pcu::global::clear_thread_cache().unwrap();
    eprintln!(
        "diagnostic/whole_checked_neg/process_wall={:?}",
        wall.elapsed()
    );
}

criterion_group!(benches, checked_neg);
criterion_main!(benches);
