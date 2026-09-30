//! Compare checked F32/F64 addition through PCU dispatch and its identical generated HIP source.

#[path = "support/owned_checked_float.rs"]
mod owned_checked_float_support;
mod support;

#[rustfmt::skip]
use criterion::{
    Criterion,
    criterion_group,
    criterion_main,
};

fn owned_checked_float(criterion: &mut Criterion) {
    owned_checked_float_support::run(criterion).expect("owned checked float benchmark failed");
}

criterion_group! {
    name = benches;
    config = support::criterion_config();
    targets = owned_checked_float
}
criterion_main!(benches);
