//! Compare checked F32/F64 division through PCU dispatch and its identical generated HIP source.

extern crate pcu_facade as fusion_pcu;

#[path = "../support/owned_checked_float_div.rs"]
mod owned_checked_float_div_support;
#[path = "../support/support.rs"]
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
