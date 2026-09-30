//! Checked binary64-to-binary32 conversion benchmark composition.

#[path = "support/checked_float_convert.rs"]
mod checked_float_convert_support;
mod support;

#[rustfmt::skip]
use criterion::{
    Criterion,
    criterion_group,
    criterion_main,
};

fn checked_float_convert(criterion: &mut Criterion) {
    if let Err(error) = checked_float_convert_support::run(criterion) {
        panic!("checked float conversion benchmark setup failed: {error}");
    }
}

criterion_group! {
    name = benches;
    config = support::criterion_config();
    targets = checked_float_convert
}
criterion_main!(benches);
