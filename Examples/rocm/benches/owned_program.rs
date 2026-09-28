//! Fresh escaping owned-program execution against the same native storage lifecycle.

#[path = "support/owned_program.rs"]
mod owned_program_support;
mod support;

#[rustfmt::skip]
use criterion::{
    Criterion,
    criterion_group,
    criterion_main,
};

fn owned_program(criterion: &mut Criterion) {
    owned_program_support::run(criterion).expect("owned-program benchmark failed");
}

criterion_group! {
    name = benches;
    config = support::criterion_config();
    targets = owned_program
}
criterion_main!(benches);
