//! Pageable byte identity, including both host transfers and completed device work.
extern crate pcu_facade as fusion_pcu;

#[path = "../strict_matmul/activity.rs"]
mod activity;
#[path = "driver/driver.rs"]
mod driver;
#[path = "native/native.rs"]
mod native;
#[path = "oracle/oracle.rs"]
mod oracle;
#[path = "source/source.rs"]
mod source;
#[path = "../support/support.rs"]
mod support;

#[rustfmt::skip]
use criterion::{
    criterion_group,
    criterion_main,
    Criterion,
};

fn bench(criterion: &mut Criterion) {
    driver::run(criterion).expect("ROCm pageable identity roundtrip failed");
}

criterion_group! { name = benches; config = support::criterion_config(); targets = bench }
criterion_main!(benches);
