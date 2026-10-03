//! Actual ordinary/prepared source, authentic neutral graph and native integer checker peers.
#[rustfmt::skip]
use criterion::{
    BenchmarkId,
    Criterion,
    criterion_group,
    criterion_main,
};
#[rustfmt::skip]
use std::{
    hint::black_box,
    time::Duration,
};
#[rustfmt::skip]
use fusion_pcu_metal::{
    MetalHostKernelError,
    MetalSession,
};
#[rustfmt::skip]
use pcu_facade::{
    PcuDispatchFloatBinaryOp as Op,
    PcuExecutionError,
    PcuFloatUnderflowPolicy,
};
#[path = "../support/activity/activity.rs"]
mod activity;
#[path = "source.rs"]
mod source;
#[path = "../support/float_binary/float_binary.rs"]
mod support;

#[allow(clippy::suboptimal_flops)] // Exact dyadic input-bank arithmetic is outside samples; keep the peers' separate checked-operation boundary explicit.
fn operation<const N: usize>(
    criterion: &mut Criterion,
    session: &MetalSession,
    op: Op,
    source: &mut impl FnMut(&[f64], &[f64], &mut [f64]) -> Result<(), MetalHostKernelError>,
    ordinary: impl FnMut(&[f64], &[f64], &mut [f64]) -> Result<(), PcuExecutionError>,
) {
    let graph = support::graph::<N, f64>(session, op);
    let native = session
        .prepare_f64_binary(op, PcuFloatUnderflowPolicy::IeeeAfterRounding)
        .unwrap();
    let banks = [1.0, 2.0, 3.0].map(|phase| {
        let left: Vec<_> = (0..N)
            .map(|index| phase + f64::from(u8::try_from(index % 8).unwrap()) * 0.25)
            .collect();
        let right: Vec<_> = (0..N)
            .map(|index| phase + 2.0 + f64::from(u8::try_from(index % 4).unwrap()) * 0.5)
            .collect();
        let expected = support::expected(op, &left, &right, 91.0_f64);
        (left, right, expected)
    });
    let mut output = vec![91.0_f64; N + 1];
    let mut ordinary = ordinary;
    let mut execute = |route, bank: usize, verify| {
        let (left, right, expected) = &banks[bank];
        match route {
            0 => source(black_box(left), black_box(right), black_box(&mut output)).unwrap(),
            1 => ordinary(black_box(left), black_box(right), black_box(&mut output)).unwrap(),
            2 => support::host_call(
                session,
                &graph,
                black_box(left),
                black_box(right),
                black_box(&mut output),
            )
            .unwrap(),
            3 => support::host_call(
                session,
                &native,
                black_box(left),
                black_box(right),
                black_box(&mut output),
            )
            .unwrap(),
            _ => unreachable!("registered exact checker peer"),
        }
        if verify {
            assert_eq!(
                pcu_facade::PcuHostArgument::read(pcu_facade::PcuBindingRef::new(0, 0), &output)
                    .bytes(),
                expected
            );
        }
        black_box(&output);
    };
    let mut group = criterion.benchmark_group(format!("host_f64_{op:?}_copied_terminal_readback"));
    for (route, name) in [
        "source_prepared",
        "ordinary_pcu",
        "neutral_graph",
        "native_checker",
    ]
    .into_iter()
    .enumerate()
    {
        for bank in 0..banks.len() {
            execute(route, bank, true);
        }
        // A filtered registration still starts with a fully checked current input bank.
        execute(route, 0, true);
        let mut bank = 0;
        group.bench_function(BenchmarkId::new(name, N), |bench| {
            bench.iter(|| {
                bank = (bank + 1) % banks.len();
                execute(route, bank, false);
            });
        });
        for bank in 0..banks.len() {
            execute(route, bank, true);
        }
    }
    group.finish();
}
fn cases<const N: usize>(criterion: &mut Criterion) {
    let session = MetalSession::open(0).unwrap();
    operation::<N>(
        criterion,
        &session,
        Op::Add,
        &mut source::add_prepare::<N, _>(&session).unwrap(),
        source::add::<N>,
    );
    operation::<N>(
        criterion,
        &session,
        Op::Sub,
        &mut source::sub_prepare::<N, _>(&session).unwrap(),
        source::sub::<N>,
    );
    operation::<N>(
        criterion,
        &session,
        Op::Mul,
        &mut source::mul_prepare::<N, _>(&session).unwrap(),
        source::mul::<N>,
    );
    operation::<N>(
        criterion,
        &session,
        Op::Div,
        &mut source::div_prepare::<N, _>(&session).unwrap(),
        source::div::<N>,
    );
}
fn checked_f64(criterion: &mut Criterion) {
    if !cfg!(target_os = "macos") {
        println!("SKIP: exact checked F64 Metal requires hardware");
        return;
    }
    activity::guard();
    pcu_facade::global::configure(pcu_facade::global::PcuExecutionPolicy {
        backend: pcu_facade::global::PcuBackendChoice::Metal,
        ..pcu_facade::global::PcuExecutionPolicy::default()
    })
    .unwrap();
    cases::<257>(criterion);
    cases::<65_536>(criterion);
    pcu_facade::global::clear_thread_cache().unwrap();
    activity::guard();
}
criterion_group! {name = benches; config = Criterion::default().sample_size(30).confidence_level(0.95).warm_up_time(Duration::from_millis(250)).measurement_time(Duration::from_secs(1)); targets = checked_f64}
criterion_main!(benches);
