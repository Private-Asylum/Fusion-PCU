//! Executed source, explicit IR and native peers for exact checked F32/F64 negation.
extern crate pcu_facade as fusion_pcu;

#[path = "../support/activity/activity.rs"]
mod activity;
#[cfg(feature = "allocation-census")]
#[path = "../support/allocations/allocations.rs"]
mod allocations;
#[path = "../support/discovery/discovery.rs"]
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
        "Checked Neg: exact finite sign inversion; inputupload/launch/terminalcompletion/statusreadback/outputdownload included. Retained success sentinel/resetafterfault match. Oracles/compilation/inputgeneration excluded; no CPUfallback."
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
