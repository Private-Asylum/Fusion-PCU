//! Fresh three-bank checks surround every independently filterable semantic or census peer.
#[rustfmt::skip]
use pcu_facade::{PcuCheckedInteger,PcuScalar,PcuDispatchIntegerBinaryOp as Op,PcuRangePolicy as Range,
    PcuPreparedHostKernel,PcuHostKernelBackend,PcuHostArgument,PcuBindingRef,PcuU256,PcuI256,PcuU512,PcuI512};
#[rustfmt::skip]
use fusion_pcu_metal::{MetalSession,MetalIntegerOp,MetalHostKernelError};
use criterion::Criterion;
#[cfg(not(feature = "allocation-census"))]
use criterion::BenchmarkId;
use std::hint::black_box;
#[path = "../../support/activity/activity.rs"]
mod activity;
#[cfg(feature = "allocation-census")]
#[path = "../../checked_neg/census/census.rs"]
mod census;
#[path = "../graph/graph.rs"]
mod graph;
#[path = "../source/source.rs"]
mod source;
trait Sample: PcuCheckedInteger {
    fn small(value: u8) -> Self;
}
macro_rules! samples {($($ty:ty=>$width:literal;)+)=>{$(
    impl Sample for $ty {fn small(value:u8)->Self {let mut bytes=[0;$width];bytes[0]=value;Self::decode_le(bytes)}}
)+};}
samples! {u8=>1;i8=>1;u16=>2;i16=>2;u32=>4;i32=>4;u64=>8;i64=>8;u128=>16;i128=>16;
PcuU256=>32;PcuI256=>32;PcuU512=>64;PcuI512=>64;}
#[allow(clippy::too_many_lines)] // One matched boundary keeps per-peer bank/tail validation beside its census.
#[cfg_attr(feature = "allocation-census", allow(clippy::needless_pass_by_ref_mut))] // Criterion's uninstru­mented registration requires this same mutable parameter.
fn operation<T: Sample, const N: usize>(
    criterion: &mut Criterion,
    session: &MetalSession,
    op: Op,
    range: Range,
    mut source: impl FnMut(&[T], &[T], &mut [T]) -> Result<(), MetalHostKernelError>,
) {
    let mut graph = graph::fixture_profile::<T, _>(
        u32::try_from(N).unwrap(),
        op,
        range,
        false,
        [false; 2],
        |ir| session.prepare_host_kernel(ir).unwrap(),
    );
    let native_op = match op {
        Op::Add => MetalIntegerOp::Add,
        Op::Sub => MetalIntegerOp::Subtract,
        Op::Mul => MetalIntegerOp::Multiply,
    };
    let mut native = session
        .prepare_checked_integer_control(T::TYPE, native_op, range, N, [false; 2])
        .unwrap();
    let sentinel = T::small(91);
    let banks = [0_u8, 1, 2].map(|phase| {
        let a: Vec<T> = (0..N)
            .map(|i| T::small(5 + phase + u8::try_from(i % 4).unwrap()))
            .collect();
        let b: Vec<T> = (0..N)
            .map(|i| T::small(1 + u8::try_from(i % 3).unwrap()))
            .collect();
        let mut expected: Vec<T> = a
            .iter()
            .zip(&b)
            .map(|(&a, &b)| {
                match op {
                    Op::Add => a.pcu_checked_add(b),
                    Op::Sub => a.pcu_checked_sub(b),
                    Op::Mul => a.pcu_checked_mul(b),
                }
                .unwrap()
            })
            .collect();
        expected.extend([sentinel; 3]);
        let bytes = PcuHostArgument::read(PcuBindingRef::new(0, 0), &expected)
            .bytes()
            .to_vec();
        (a, b, bytes)
    });
    let mut output = vec![sentinel; N + 3];
    let mut execute = |route, bank: usize, verify| {
        let (a, b, expected) = &banks[bank];
        match route {
            0 => source(black_box(a), black_box(b), black_box(&mut output)).unwrap(),
            1 => graph
                .call(&mut [
                    PcuHostArgument::read_write(PcuBindingRef::new(4, 1), black_box(&mut output)),
                    PcuHostArgument::read(PcuBindingRef::new(2, 7), black_box(b)),
                    PcuHostArgument::read(PcuBindingRef::new(2, 3), black_box(a)),
                ])
                .unwrap(),
            2 => native
                .call(black_box(a), black_box(b), black_box(&mut output))
                .unwrap(),
            3 => match (op, range) {
                (Op::Add, Range::Reject) => {
                    source::add::<T, N>(black_box(a), black_box(b), black_box(&mut output))
                }
                (Op::Sub, Range::Reject) => {
                    source::sub::<T, N>(black_box(a), black_box(b), black_box(&mut output))
                }
                (Op::Mul, Range::Reject) => {
                    source::mul::<T, N>(black_box(a), black_box(b), black_box(&mut output))
                }
                (Op::Add, Range::Clamp) => {
                    source::add_clamp::<T, N>(black_box(a), black_box(b), black_box(&mut output))
                }
                (Op::Sub, Range::Clamp) => {
                    source::sub_clamp::<T, N>(black_box(a), black_box(b), black_box(&mut output))
                }
                (Op::Mul, Range::Clamp) => {
                    source::mul_clamp::<T, N>(black_box(a), black_box(b), black_box(&mut output))
                }
            }
            .unwrap(),
            _ => unreachable!("registered peer"),
        }
        if verify {
            assert_eq!(
                PcuHostArgument::read(PcuBindingRef::new(0, 0), &output).bytes(),
                expected
            );
        }
        black_box(&output);
    };
    #[cfg(feature = "allocation-census")]
    let _ = criterion;
    #[cfg(not(feature = "allocation-census"))]
    let mut group = criterion.benchmark_group(format!(
        "host_{:?}_{op:?}_{range:?}_fresh_terminal_prefix",
        T::TYPE
    ));
    for (route, name) in [
        "source_prepared",
        "neutral_graph",
        "native_checker",
        "ordinary_source",
    ]
    .into_iter()
    .enumerate()
    {
        for bank in 0..banks.len() {
            execute(route, bank, true);
        }
        execute(route, 0, true);
        #[cfg(feature = "allocation-census")]
        census::census(
            &format!("{:?}/{op:?}/{range:?}/{name}/{N}", T::TYPE),
            || execute(route, 1, false),
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
fn cases<T: Sample, const N: usize>(criterion: &mut Criterion, session: &MetalSession) {
    operation::<T, N>(
        criterion,
        session,
        Op::Add,
        Range::Reject,
        source::add_prepare::<T, N, _>(session).unwrap(),
    );
    operation::<T, N>(
        criterion,
        session,
        Op::Sub,
        Range::Reject,
        source::sub_prepare::<T, N, _>(session).unwrap(),
    );
    operation::<T, N>(
        criterion,
        session,
        Op::Mul,
        Range::Reject,
        source::mul_prepare::<T, N, _>(session).unwrap(),
    );
    operation::<T, N>(
        criterion,
        session,
        Op::Add,
        Range::Clamp,
        source::add_clamp_prepare::<T, N, _>(session).unwrap(),
    );
    operation::<T, N>(
        criterion,
        session,
        Op::Sub,
        Range::Clamp,
        source::sub_clamp_prepare::<T, N, _>(session).unwrap(),
    );
    operation::<T, N>(
        criterion,
        session,
        Op::Mul,
        Range::Clamp,
        source::mul_clamp_prepare::<T, N, _>(session).unwrap(),
    );
}
pub fn run(criterion: &mut Criterion) {
    if !cfg!(target_os = "macos") {
        println!("SKIP: requires actual Metal GPU");
        return;
    }
    activity::guard();
    pcu_facade::global::configure(pcu_facade::global::PcuExecutionPolicy {
        backend: pcu_facade::global::PcuBackendChoice::Metal,
        ..pcu_facade::global::PcuExecutionPolicy::default()
    })
    .unwrap();
    let session = MetalSession::open(0).unwrap();
    macro_rules! all {($($ty:ty),+)=>{$(cases::<$ty,65>(criterion,&session);cases::<$ty,4096>(criterion,&session);)+};}
    all!(
        u8, i8, u16, i16, u32, i32, u64, i64, u128, i128, PcuU256, PcuI256, PcuU512, PcuI512
    );
}
