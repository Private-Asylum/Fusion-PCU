//! Compare typed source `MatMul`, an owned graph, and direct rocBLAS SGEMM.

extern crate pcu_facade as fusion_pcu;

#[path = "../support/owned_matmul_common.rs"]
mod owned_matmul_common;
#[path = "../support/owned_matmul_f64.rs"]
mod owned_matmul_f64_support;
#[path = "../support/owned_matmul.rs"]
mod owned_matmul_support;
#[path = "../support/support.rs"]
mod support;

#[rustfmt::skip]
use criterion::{
    Criterion,
    criterion_group,
    criterion_main,
};

fn owned_matmul(criterion: &mut Criterion) {
    owned_matmul_support::run(criterion).expect("owned MatMul benchmark failed");
}

fn owned_matmul_f64(criterion: &mut Criterion) {
    owned_matmul_f64_support::run(criterion).expect("owned f64 MatMul benchmark failed");
}

criterion_group! {
    name = benches;
    config = support::criterion_config();
    targets = owned_matmul, owned_matmul_f64
}
criterion_main!(benches);
