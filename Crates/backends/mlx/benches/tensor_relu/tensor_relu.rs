//! Fresh host input, completed immutable `ReLU` owner, caller readback and Drop.
#[rustfmt::skip]
use criterion::{
    criterion_group,
    criterion_main,
};
#[path = "support/support.rs"]
mod support;
criterion_group!(benches, support::run);
criterion_main!(benches);
