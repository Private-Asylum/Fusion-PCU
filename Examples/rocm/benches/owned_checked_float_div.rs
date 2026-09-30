//! Compare checked F32/F64 division through PCU dispatch and its identical generated HIP source.

#[path = "support/owned_checked_float_div.rs"]
mod owned_checked_float_div_support;
mod support;

#[rustfmt::skip]
use criterion::{
    Criterion,
    criterion_group,
    criterion_main,
};

fn owned_checked_float_div(criterion: &mut Criterion) {
    owned_checked_float_div_support::run(criterion).expect("owned checked float benchmark failed");
}

criterion_group! {
    name = benches;
    config = support::criterion_config();
    targets = owned_checked_float_div
}
criterion_main!(benches);
