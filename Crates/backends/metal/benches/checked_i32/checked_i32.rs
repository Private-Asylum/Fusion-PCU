//! Authentic source, graph and native checked I32 maps with identical host publication.
//! Every route stages two inputs, owns fresh output/status, waits, checks all lanes and copies.
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
#[rustfmt::skip]
use fusion_pcu_metal::{
    MetalIntegerOp,
    MetalSession,
};
#[rustfmt::skip]
use pcu_facade::{
    PcuDispatchIntegerBinaryOp,
};
#[path = "source.rs"]
mod source;
#[path = "../support/integer/integer.rs"]
mod support;

fn operation<const N: usize>(
    criterion: &mut Criterion,
    session: &MetalSession,
    name: &str,
    op: PcuDispatchIntegerBinaryOp,
    native_op: MetalIntegerOp,
    source: &mut impl FnMut(
        &[i32],
        &[i32],
        &mut [i32],
    ) -> Result<(), fusion_pcu_metal::MetalHostKernelError>,
) {
    let graph = support::graph::<N, i32>(session, op);
    let native = session.prepare_i32_map(native_op).unwrap();
    let (mut left, right, mut output) = support::validate::<N, _>(
        session,
        &graph,
        &native,
        op,
        source,
        support::Fixture {
            phases: [-31, -30],
            right: 17,
            fault: if op == PcuDispatchIntegerBinaryOp::Sub {
                i32::MIN
            } else {
                i32::MAX
            },
            sentinel: 91,
        },
    );
    let mut group = criterion.benchmark_group(format!("host_i32_{name}"));
    group.bench_function(BenchmarkId::new("source", N), |bench| {
        bench.iter(|| {
            left[0] ^= 1;
            source(black_box(&left), black_box(&right), black_box(&mut output)).unwrap();
            black_box(&output);
        });
    });
    group.bench_function(BenchmarkId::new("graph", N), |bench| {
        bench.iter(|| {
            left[0] ^= 1;
            support::host_call(
                session,
                &graph,
                black_box(&left),
                black_box(&right),
                black_box(&mut output),
            )
            .unwrap();
            black_box(&output);
        });
    });
    group.bench_function(BenchmarkId::new("native_checker", N), |bench| {
        bench.iter(|| {
            left[0] ^= 1;
            support::host_call(
                session,
                &native,
                black_box(&left),
                black_box(&right),
                black_box(&mut output),
            )
            .unwrap();
            black_box(&output);
        });
    });
    group.finish();
}
fn cases<const N: usize>(criterion: &mut Criterion) {
    let session = MetalSession::open(0).unwrap();
    operation::<N>(
        criterion,
        &session,
        "add",
        PcuDispatchIntegerBinaryOp::Add,
        MetalIntegerOp::Add,
        &mut source::add_prepare::<N, _>(&session).unwrap(),
    );
    operation::<N>(
        criterion,
        &session,
        "sub",
        PcuDispatchIntegerBinaryOp::Sub,
        MetalIntegerOp::Subtract,
        &mut source::sub_prepare::<N, _>(&session).unwrap(),
    );
    operation::<N>(
        criterion,
        &session,
        "mul",
        PcuDispatchIntegerBinaryOp::Mul,
        MetalIntegerOp::Multiply,
        &mut source::mul_prepare::<N, _>(&session).unwrap(),
    );
}
fn checked_i32(criterion: &mut Criterion) {
    if !cfg!(target_os = "macos") {
        println!("SKIP: signed Metal benchmark requires hardware");
        return;
    }
    support::guard();
    cases::<257>(criterion);
    cases::<65_536>(criterion);
}
criterion_group! { name = benches; config = Criterion::default().sample_size(30).confidence_level(0.95).warm_up_time(Duration::from_millis(250)).measurement_time(Duration::from_secs(1)); targets = checked_i32 }
criterion_main!(benches);
