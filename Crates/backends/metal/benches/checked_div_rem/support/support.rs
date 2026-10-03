//! Fresh changed banks, joint publication and exact tails surround every filtered peer.
#[rustfmt::skip]
use pcu_facade::{PcuCheckedIntegerDivision,PcuScalar,PcuHostKernelBackend,PcuPreparedHostKernel,PcuHostArgument,PcuBindingRef};
#[rustfmt::skip]
use fusion_pcu_metal::{MetalSession,MetalHostKernelError};
use criterion::Criterion;
#[cfg(not(feature = "allocation-census"))]
use criterion::BenchmarkId;
use std::hint::black_box;
#[cfg(feature = "allocation-census")]
#[path = "../../checked_neg/census/census.rs"]
mod census;
#[path = "../graph/graph.rs"]
mod graph;
pub trait Sample: PcuCheckedIntegerDivision {
    fn small(value: u8) -> Self;
}
macro_rules! samples{($($ty:ty=>$width:literal;)+)=>{$(impl Sample for $ty{fn small(value:u8)->Self{let mut bytes=[0;$width];bytes[0]=value;Self::decode_le(bytes)}})+};}
samples! {u8=>1;i8=>1;u16=>2;i16=>2;u32=>4;i32=>4;u64=>8;i64=>8;u128=>16;i128=>16;pcu_facade::PcuU256=>32;pcu_facade::PcuI256=>32;pcu_facade::PcuU512=>64;pcu_facade::PcuI512=>64;}
#[allow(clippy::too_many_lines)] // Keep cold matched routes and per-filter fresh-bank proof next to the census.
#[cfg_attr(feature = "allocation-census", allow(clippy::needless_pass_by_ref_mut))] // Criterion registration requires the same mutable interface in normal builds.
pub fn compare<T: Sample, const N: usize>(
    criterion: &mut Criterion,
    session: &MetalSession,
    profile: usize,
    mut source: impl FnMut(&[T], &[T], &mut [T], &mut [T]) -> Result<(), MetalHostKernelError>,
    mut ordinary: impl FnMut(
        &[T],
        &[T],
        &mut [T],
        &mut [T],
    ) -> Result<(), pcu_facade::PcuExecutionError>,
) {
    let backend = session.checked_div_rem_backend();
    let mut graph = graph::fixture::<T, _>(
        u32::try_from(N).unwrap(),
        profile == 2,
        profile == 1,
        |ir| backend.prepare_host_kernel(ir).unwrap(),
    );
    let mut native = session.prepare_checked_div_rem_control(T::TYPE, N).unwrap();
    let sentinel = T::small(91);
    let banks = [0_u8, 1, 2].map(|phase| {
        let left = (0..N)
            .map(|i| T::small(7 + phase + u8::try_from(i % 4).unwrap()))
            .collect::<Vec<_>>();
        let right = (0..N)
            .map(|i| T::small(2 + u8::try_from(i % 3).unwrap()))
            .collect::<Vec<_>>();
        let mut q = left
            .iter()
            .zip(&right)
            .map(|(&a, &b)| a.pcu_checked_div(b).unwrap())
            .collect::<Vec<_>>();
        q.extend([sentinel; 3]);
        let mut r = left
            .iter()
            .zip(&right)
            .map(|(&a, &b)| a.pcu_checked_rem(b).unwrap())
            .collect::<Vec<_>>();
        r.extend([sentinel; 3]);
        let qb = PcuHostArgument::read(PcuBindingRef::new(0, 0), &q)
            .bytes()
            .to_vec();
        let rb = PcuHostArgument::read(PcuBindingRef::new(0, 0), &r)
            .bytes()
            .to_vec();
        (left, right, qb, rb)
    });
    let mut q = vec![sentinel; N + 3];
    let mut r = q.clone();
    let mut execute = |route, bank: usize, verify| {
        let (a, b, eq, er) = &banks[bank];
        match route {
            0 => source(
                black_box(a),
                black_box(b),
                black_box(&mut q),
                black_box(&mut r),
            )
            .unwrap(),
            1 => graph
                .call(&mut [
                    PcuHostArgument::read_write(PcuBindingRef::new(4, 6), black_box(&mut r)),
                    PcuHostArgument::read(PcuBindingRef::new(2, 7), black_box(b)),
                    PcuHostArgument::read_write(PcuBindingRef::new(4, 1), black_box(&mut q)),
                    PcuHostArgument::read(PcuBindingRef::new(2, 3), black_box(a)),
                ])
                .unwrap(),
            2 => native
                .call(
                    black_box(a),
                    black_box(b),
                    black_box(&mut q),
                    black_box(&mut r),
                )
                .unwrap(),
            3 => ordinary(
                black_box(a),
                black_box(b),
                black_box(&mut q),
                black_box(&mut r),
            )
            .unwrap(),
            _ => unreachable!("registered peer"),
        }
        if verify {
            assert_eq!(
                PcuHostArgument::read(PcuBindingRef::new(0, 0), &q).bytes(),
                eq
            );
            assert_eq!(
                PcuHostArgument::read(PcuBindingRef::new(0, 0), &r).bytes(),
                er
            );
        }
        black_box((&q, &r));
    };
    #[cfg(feature = "allocation-census")]
    let _ = criterion;
    #[cfg(not(feature = "allocation-census"))]
    let mut group = criterion.benchmark_group(format!(
        "host_{:?}_DivRem_profile{profile}_joint_terminal_prefix",
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
        {
            let before = super::SCORES.load(std::sync::atomic::Ordering::Relaxed);
            census::census(
                &format!("{:?}/DivRem/profile{profile}/{name}/{N}", T::TYPE),
                || execute(route, 1, false),
            );
            let scored = super::SCORES.load(std::sync::atomic::Ordering::Relaxed) - before;
            assert_eq!(scored, 0, "warm DivRem peer rescored candidates");
            eprintln!(
                "Warm invocation score census/{:?}/DivRem/profile{profile}/{name}/{N}: calls=1 callbacks={scored}",
                T::TYPE
            );
        }
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
