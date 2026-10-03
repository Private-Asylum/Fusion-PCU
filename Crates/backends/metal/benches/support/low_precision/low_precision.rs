//! Actual ordinary/prepared source, authentic neutral graph and native integer checker peers.
#[rustfmt::skip]
use criterion::{
    Criterion,
};
#[rustfmt::skip]
#[cfg(not(feature = "allocation-census"))]
use criterion::BenchmarkId;
#[rustfmt::skip]
use std::{
    hint::black_box,
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
    PcuCheckedFloat,
    PcuF16Bits,
    PcuBf16Bits,
    PcuF8E4M3FnBits,
    PcuF8E5M2Bits,
};
#[path = "../activity/activity.rs"]
mod activity;
#[cfg(feature = "allocation-census")]
#[path = "../../checked_neg/census/census.rs"]
mod census;
use crate::source;
#[path = "../float_binary/float_binary.rs"]
mod support;

trait Sample: PcuCheckedFloat {
    fn value(value: f32) -> Self;
}
macro_rules! sample {
    ($ty:ty) => {
        impl Sample for $ty {
            fn value(value: f32) -> Self {
                Self::pcu_checked_from_f32(value).unwrap()
            }
        }
    };
}
sample!(PcuF16Bits);
sample!(PcuBf16Bits);
sample!(PcuF8E4M3FnBits);
sample!(PcuF8E5M2Bits);
#[allow(clippy::too_many_lines)] // Frozen four-peer boundary, bank validation and census share one harness.
#[allow(clippy::suboptimal_flops)]
// Exact dyadic input-bank arithmetic is outside samples; source/graph/native checked operations retain separate rounding.
#[cfg_attr(feature = "allocation-census", allow(clippy::needless_pass_by_ref_mut))] // Criterion mutability is required by the uninstrumented peer with this shared entry.
fn operation<const N: usize, T: Sample>(
    criterion: &mut Criterion,
    session: &MetalSession,
    op: Op,
    portable: bool,
    source: &mut impl FnMut(&[T], &[T], &mut [T]) -> Result<(), MetalHostKernelError>,
    ordinary: impl FnMut(&[T], &[T], &mut [T]) -> Result<(), PcuExecutionError>,
) {
    let graph = if portable {
        support::graph_with_reproducibility::<N, T>(
            session,
            op,
            pcu_facade::PcuReproducibility::PortableV1,
        )
    } else {
        support::graph::<N, T>(session, op)
    };
    let native = session
        .prepare_low_precision_binary(T::TYPE, op, PcuFloatUnderflowPolicy::IeeeAfterRounding)
        .unwrap();
    let banks = [1.0, 2.0, 3.0].map(|phase| {
        let left: Vec<_> = (0..N)
            .map(|index| T::value(phase + f32::from(u8::try_from(index % 8).unwrap()) * 0.25))
            .collect();
        let right: Vec<_> = (0..N)
            .map(|index| T::value(phase + 2.0 + f32::from(u8::try_from(index % 4).unwrap()) * 0.5))
            .collect();
        let expected = support::expected(op, &left, &right, T::value(91.0));
        (left, right, expected)
    });
    let mut output = vec![T::value(91.0); N + 1];
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
    #[cfg(feature = "allocation-census")]
    let _ = criterion;
    #[cfg(not(feature = "allocation-census"))]
    let mut group = criterion.benchmark_group(format!(
        "host_{}_{:?}_{op:?}_copied_terminal_readback",
        if portable {
            "PortableV1"
        } else {
            "Unspecified"
        },
        T::TYPE
    ));
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
        #[cfg(feature = "allocation-census")]
        census::census(&format!("{:?}/{op:?}/{name}/{N}", T::TYPE), || {
            execute(route, 1, false);
        });
        #[cfg(not(feature = "allocation-census"))]
        {
            let mut bank = 0;
            group.bench_function(BenchmarkId::new(name, N), |bench| {
                bench.iter(|| {
                    bank = (bank + 1) % banks.len();
                    execute(route, bank, false);
                });
            });
        }
        for bank in 0..banks.len() {
            execute(route, bank, true);
        }
    }
    #[cfg(not(feature = "allocation-census"))]
    group.finish();
}
fn cases<const N: usize, T: Sample>(criterion: &mut Criterion, portable: bool) {
    let session = MetalSession::open(0).unwrap();
    operation::<N, T>(
        criterion,
        &session,
        Op::Add,
        portable,
        &mut source::add_prepare::<T, N, _>(&session).unwrap(),
        source::add::<T, N>,
    );
    operation::<N, T>(
        criterion,
        &session,
        Op::Sub,
        portable,
        &mut source::sub_prepare::<T, N, _>(&session).unwrap(),
        source::sub::<T, N>,
    );
    operation::<N, T>(
        criterion,
        &session,
        Op::Mul,
        portable,
        &mut source::mul_prepare::<T, N, _>(&session).unwrap(),
        source::mul::<T, N>,
    );
    operation::<N, T>(
        criterion,
        &session,
        Op::Div,
        portable,
        &mut source::div_prepare::<T, N, _>(&session).unwrap(),
        source::div::<T, N>,
    );
}
pub fn run(criterion: &mut Criterion, portable: bool) {
    if !cfg!(target_os = "macos") {
        println!("SKIP: exact checked low-precision Metal requires hardware");
        return;
    }
    activity::guard();
    pcu_facade::global::configure(pcu_facade::global::PcuExecutionPolicy {
        backend: pcu_facade::global::PcuBackendChoice::Metal,
        ..pcu_facade::global::PcuExecutionPolicy::default()
    })
    .unwrap();
    cases::<257, PcuF16Bits>(criterion, portable);
    cases::<65_537, PcuF16Bits>(criterion, portable);
    cases::<257, PcuBf16Bits>(criterion, portable);
    cases::<65_537, PcuBf16Bits>(criterion, portable);
    cases::<257, PcuF8E4M3FnBits>(criterion, portable);
    cases::<65_537, PcuF8E4M3FnBits>(criterion, portable);
    cases::<257, PcuF8E5M2Bits>(criterion, portable);
    cases::<65_537, PcuF8E5M2Bits>(criterion, portable);
    pcu_facade::global::clear_thread_cache().unwrap();
    activity::guard();
}
