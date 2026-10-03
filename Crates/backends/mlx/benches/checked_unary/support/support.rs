//! Matched MLX-owned staging, sibling output/status, terminal prefix publication and retained output peers.
#[rustfmt::skip]
use fusion_pcu_mlx::{
    MlxRuntime,
    MlxSession,
    MlxHostKernelError,
};
#[rustfmt::skip]
use pcu_facade::{
    PcuCheckedFloat,
    PcuExecutionError,
    PcuFloatUnderflowPolicy as Policy,
    PcuDispatchFloatUnaryOp as Op,
    PcuHostArgument,
    PcuBindingRef,
    PcuHostKernelBackend,
    PcuPreparedHostKernel,
    PcuRangePolicy as Range,
    PcuF16Bits,
    PcuBf16Bits,
    PcuF8E4M3FnBits,
    PcuF8E5M2Bits,
};
#[cfg(not(feature = "allocation-census"))]
use criterion::BenchmarkId;
use criterion::Criterion;
use std::hint::black_box;
#[path = "activity.rs"]
mod activity;
#[cfg(feature = "allocation-census")]
#[path = "../census/census.rs"]
mod census;
#[path = "../graph/graph.rs"]
mod graph;
#[path = "../source/source.rs"]
mod source;
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
impl Sample for f32 {
    fn value(value: f32) -> Self {
        value
    }
}
impl Sample for f64 {
    fn value(value: f32) -> Self {
        Self::from(value)
    }
}

#[allow(clippy::too_many_lines)] // One matching four-peer boundary retains bank/tail checks and separate census.
#[cfg_attr(feature = "allocation-census", allow(clippy::needless_pass_by_ref_mut))] // Criterion requires mutability in the noninstrumented peer.
fn operation<T: Sample, const N: usize>(
    criterion: &mut Criterion,
    session: &MlxSession,
    op: Op,
    policy: Policy,
    range: Range,
    mut prepared: impl FnMut(&[T], &mut [T]) -> Result<(), MlxHostKernelError>,
    mut ordinary: impl FnMut(&[T], &mut [T]) -> Result<(), PcuExecutionError>,
) {
    let prepare =
        |kernel: &pcu_facade::PcuDispatchKernelIr<'_>| session.prepare_host_kernel(kernel).unwrap();
    let count = u32::try_from(N).unwrap();
    let mut map = if range == Range::Reject {
        graph::fixture::<T, _>(count, op, policy, false, prepare)
    } else {
        graph::fixture_profile::<T, _>(count, op, policy, range, false, false, prepare)
    };
    let mut native = session
        .prepare_checked_unary_control(T::TYPE, op, policy, range, N, false)
        .unwrap();
    let sentinel = T::value(91.0);
    let banks = [1.0, 2.0, 3.0].map(|phase| {
        let input: Vec<T> = (0..N)
            .map(|i| {
                T::value(match i % 4 {
                    0 => -phase,
                    1 => phase,
                    2 => 0.0,
                    _ => -0.0,
                })
            })
            .collect();
        let mut expected: Vec<T> = input
            .iter()
            .map(|&value| {
                match op {
                    Op::Neg => value.pcu_checked_neg_with_policy(policy),
                    Op::Relu => value.pcu_checked_relu_with_policy(policy),
                }
                .unwrap()
            })
            .collect();
        expected.push(sentinel);
        let bytes = PcuHostArgument::read(PcuBindingRef::new(0, 1), &expected)
            .bytes()
            .to_vec();
        (input, bytes)
    });
    let mut output = vec![sentinel; N + 1];
    let mut execute = |route, bank: usize, verify| {
        let (input, expected) = &banks[bank];
        match route {
            0 => prepared(black_box(input), black_box(&mut output)).unwrap(),
            1 => ordinary(black_box(input), black_box(&mut output)).unwrap(),
            2 => map
                .call(&mut [
                    PcuHostArgument::read(PcuBindingRef::new(2, 3), black_box(input)),
                    PcuHostArgument::read_write(PcuBindingRef::new(4, 1), black_box(&mut output)),
                ])
                .unwrap(),
            3 => native
                .call(black_box(input), black_box(&mut output))
                .unwrap(),
            _ => unreachable!("registered unary peer"),
        }
        if verify {
            assert_eq!(
                PcuHostArgument::read(PcuBindingRef::new(0, 1), &output).bytes(),
                expected
            );
        }
        black_box(&output);
    };
    #[cfg(feature = "allocation-census")]
    let _ = criterion;
    #[cfg(not(feature = "allocation-census"))]
    let mut group = criterion.benchmark_group(format!(
        "host_Unspecified_{:?}_{op:?}_{policy:?}_{range:?}_copied_terminal_readback",
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
        // Every filtered registration gets its own initialized and checked input bank.
        execute(route, 0, true);
        #[cfg(feature = "allocation-census")]
        census::census(
            &format!("{:?}/{op:?}/{policy:?}/{range:?}/{name}/{N}", T::TYPE),
            || {
                execute(route, 1, false);
            },
        );
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
fn cases<T: Sample, const N: usize>(criterion: &mut Criterion) {
    let runtime = MlxRuntime::load_default().unwrap();
    let session = runtime.open_gpu(0).unwrap();
    macro_rules! run {
        ($op:ident,$policy:ident,$range:ident,$source:ident,$prepare:ident) => {
            operation::<T, N>(
                criterion,
                &session,
                Op::$op,
                Policy::$policy,
                Range::$range,
                source::$prepare::<T, N, _>(&session).unwrap(),
                source::$source::<T, N>,
            );
        };
    }
    run!(Neg, IeeeAfterRounding, Reject, negate, negate_prepare);
    run!(Relu, IeeeAfterRounding, Reject, relu, relu_prepare);
    run!(
        Neg,
        AllowGradualUnderflow,
        Reject,
        negate_gradual,
        negate_gradual_prepare
    );
    run!(
        Relu,
        AllowGradualUnderflow,
        Reject,
        relu_gradual,
        relu_gradual_prepare
    );
    run!(
        Neg,
        RejectSubnormalResult,
        Reject,
        negate_tight,
        negate_tight_prepare
    );
    run!(
        Relu,
        RejectSubnormalResult,
        Reject,
        relu_tight,
        relu_tight_prepare
    );
    run!(
        Neg,
        IeeeAfterRounding,
        Clamp,
        negate_clamp,
        negate_clamp_prepare
    );
    run!(
        Relu,
        IeeeAfterRounding,
        Clamp,
        relu_clamp,
        relu_clamp_prepare
    );
    run!(
        Neg,
        AllowGradualUnderflow,
        Clamp,
        negate_gradual_clamp,
        negate_gradual_clamp_prepare
    );
    run!(
        Relu,
        AllowGradualUnderflow,
        Clamp,
        relu_gradual_clamp,
        relu_gradual_clamp_prepare
    );
    run!(
        Neg,
        RejectSubnormalResult,
        Clamp,
        negate_tight_clamp,
        negate_tight_clamp_prepare
    );
    run!(
        Relu,
        RejectSubnormalResult,
        Clamp,
        relu_tight_clamp,
        relu_tight_clamp_prepare
    );
}
pub fn run(criterion: &mut Criterion) {
    if !cfg!(target_os = "macos") {
        println!("SKIP: checked unary MLX needs actual pinned hardware");
        return;
    }
    activity::guard();
    pcu_facade::global::configure(pcu_facade::global::PcuExecutionPolicy {
        backend: pcu_facade::global::PcuBackendChoice::Mlx,
        ..pcu_facade::global::PcuExecutionPolicy::default()
    })
    .unwrap();
    macro_rules! formats {
        ($ty:ty) => {
            cases::<$ty, 257>(criterion);
            cases::<$ty, 65_537>(criterion);
        };
    }
    formats!(PcuF16Bits);
    formats!(PcuBf16Bits);
    formats!(PcuF8E4M3FnBits);
    formats!(PcuF8E5M2Bits);
    formats!(f32);
    formats!(f64);
    pcu_facade::global::clear_thread_cache().unwrap();
    activity::guard();
}
