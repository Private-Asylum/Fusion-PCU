//! Authentic F64 Neg source, neutral graph and same integer checker with equal publication.
//! Two U32 limbs represent each F64; no native floating ALU or arbitrary F64 arithmetic claim.
#[rustfmt::skip]
use std::{
    hint::black_box,
    time::Duration,
};
#[rustfmt::skip]
use criterion::{
    BenchmarkId,
    Criterion,
    criterion_group,
    criterion_main,
};
use fusion_pcu_metal::MetalSession;
use pcu_facade::PcuFloatUnderflowPolicy;
#[path = "../support/support.rs"]
mod shared;
#[path = "source.rs"]
mod source;
#[path = "support/support.rs"]
mod support;
fn cases<const N: usize>(criterion: &mut Criterion) {
    let session = MetalSession::open(0).unwrap();
    let mut source = source::negate_prepare::<N, _>(&session).unwrap();
    let graph = support::graph::<N>(&session);
    let native = session
        .prepare_f64_neg(PcuFloatUnderflowPolicy::IeeeAfterRounding)
        .unwrap();
    support::validate::<N>(&session, &graph, &native, &mut source);
    let mut input = vec![1.0_f64; N];
    let mut output = vec![91.0_f64; N + 1];
    let mut group = criterion.benchmark_group("host_f64_neg");
    group.bench_function(BenchmarkId::new("source", N), |bench| {
        bench.iter(|| {
            input[0] = f64::from_bits(input[0].to_bits() ^ 1);
            source(black_box(&input), black_box(&mut output)).unwrap();
            black_box(&output);
        });
    });
    group.bench_function(BenchmarkId::new("graph", N), |bench| {
        bench.iter(|| {
            input[0] = f64::from_bits(input[0].to_bits() ^ 1);
            support::host_call(&session, &graph, black_box(&input), black_box(&mut output))
                .unwrap();
            black_box(&output);
        });
    });
    group.bench_function(BenchmarkId::new("native_checker", N), |bench| {
        bench.iter(|| {
            input[0] = f64::from_bits(input[0].to_bits() ^ 1);
            support::host_call(&session, &native, black_box(&input), black_box(&mut output))
                .unwrap();
            black_box(&output);
        });
    });
    group.finish();
}
fn checked_neg_f64(criterion: &mut Criterion) {
    if !cfg!(target_os = "macos") {
        println!("SKIP: paired-limb F64 Metal requires actual hardware");
        return;
    }
    shared::guard();
    cases::<257>(criterion);
    cases::<65_536>(criterion);
}
criterion_group! {name = benches; config = Criterion::default().sample_size(30).confidence_level(0.95).warm_up_time(Duration::from_millis(250)).measurement_time(Duration::from_secs(1)); targets = checked_neg_f64}
criterion_main!(benches);
