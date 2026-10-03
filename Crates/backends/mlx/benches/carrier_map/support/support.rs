//! Cold retained source/IR/native integer primitives; exact fresh-bank host publication peers.
#[rustfmt::skip]
use std::hint::black_box;
#[rustfmt::skip]
use pcu_facade::{PcuHostArgument,PcuBindingRef,PcuPreparedHostKernel,PcuF16Bits,PcuBf16Bits,
    PcuF8E4M3FnBits,PcuF8E5M2Bits,PcuF128Bits,PcuF256Bits,PcuU256,PcuI256,PcuU512,PcuI512};
use fusion_pcu_mlx::{MlxRuntime, MlxSession};
use criterion::Criterion;
#[cfg(not(feature = "allocation-census"))]
use criterion::BenchmarkId;
#[path = "../../checked_unary/support/activity.rs"]
mod activity;
#[cfg(feature = "allocation-census")]
#[path = "../../checked_unary/census/census.rs"]
mod census;
#[path = "../../../tests/carrier_map/graph/graph.rs"]
mod graph;
#[path = "../../../tests/encoded_carrier/support/support.rs"]
mod oracle;
#[path = "../../../tests/carrier_map/source/source.rs"]
mod source;
use oracle::{Sample, compare};
#[cfg_attr(feature = "allocation-census", allow(clippy::needless_pass_by_ref_mut))] // Same Criterion registration interface; census records no timing samples.
#[allow(clippy::too_many_lines)] // Keep all three matched peer boundaries and fresh-bank/tail checks visible together.
fn format<T: Sample, const N: usize>(criterion: &mut Criterion, session: &MlxSession) {
    #[cfg(feature = "allocation-census")]
    let _ = criterion;
    let mut dense = source::copy_prepare::<T, N, _>(session).unwrap();
    let mut grid = source::grid_prepare::<T, N, _>(session).unwrap();
    let mut broadcast = source::broadcast_prepare::<T, N, _>(session).unwrap();
    let banks: [Vec<T>; 3] = std::array::from_fn(|bank| {
        (0..N)
            .map(|index| {
                T::sample(
                    u8::try_from(index % 256)
                        .unwrap()
                        .wrapping_add(u8::try_from(bank * 23).unwrap()),
                )
            })
            .collect()
    });
    let sentinel = T::sample(91);
    let mut output = vec![sentinel; N + 3];
    for (profile, profile_name) in ["dense", "grid", "scalar-broadcast"]
        .into_iter()
        .enumerate()
    {
        let scalar = profile == 2;
        let mut explicit =
            graph::fixture::<T, _>(u32::try_from(N).unwrap(), scalar, profile == 1, |ir| {
                session.prepare_carrier_host_kernel(ir)
            })
            .unwrap();
        let mut native = session.prepare_carrier_control(T::TYPE, N, scalar).unwrap();
        let mut execute = |route, bank: usize, verify| {
            if verify {
                output.fill(sentinel);
            }
            let input = if scalar {
                &banks[bank][..1]
            } else {
                &banks[bank][..]
            };
            match route {
                0 => match profile {
                    0 => dense(black_box(input), black_box(&mut output)).unwrap(),
                    1 => grid(black_box(input), black_box(&mut output)).unwrap(),
                    2 => broadcast(black_box(&input[0]), black_box(&mut output)).unwrap(),
                    _ => unreachable!("three retained source profiles"),
                },
                1 => explicit
                    .call(&mut [
                        PcuHostArgument::read_write(
                            PcuBindingRef::new(4, 1),
                            black_box(&mut output),
                        ),
                        PcuHostArgument::read(PcuBindingRef::new(2, 3), black_box(input)),
                    ])
                    .unwrap(),
                2 => native
                    .call(black_box(input), black_box(&mut output))
                    .unwrap(),
                _ => unreachable!("three registered host publication peers"),
            }
            if verify {
                if scalar {
                    for value in &output[..N] {
                        compare(std::slice::from_ref(value), &input[..1]);
                    }
                } else {
                    compare(&output[..N], input);
                }
                compare(&output[N..], &[sentinel; 3]);
            }
            black_box(&output);
        };
        #[cfg(not(feature = "allocation-census"))]
        let mut group =
            criterion.benchmark_group(format!("mlx_carrier_map/{:?}/{profile_name}", T::TYPE));
        for (route, name) in ["source-prepared", "explicit-graph", "direct-native"]
            .into_iter()
            .enumerate()
        {
            // Fresh actual input banks and output tails precede and follow EACH peer, even when filtered.
            for bank in 0..3 {
                execute(route, bank, true);
            }
            #[cfg(feature = "allocation-census")]
            census::census(
                &format!("mlx-carrier-map/{:?}/{profile_name}/{N}/{name}", T::TYPE),
                || execute(route, 1, false),
            );
            #[cfg(not(feature = "allocation-census"))]
            group.bench_with_input(BenchmarkId::new(name, N), &N, |bench, _| {
                bench.iter(|| execute(route, 1, false));
            });
            for bank in 0..3 {
                execute(route, bank, true);
            }
        }
        #[cfg(not(feature = "allocation-census"))]
        group.finish();
    }
}
pub fn run(criterion: &mut Criterion) {
    activity::guard();
    let runtime = MlxRuntime::load_default().unwrap();
    let session = runtime.open_gpu(0).unwrap();
    macro_rules! all {($n:expr;$($ty:ty),+)=>{$(format::<$ty,$n>(criterion,&session);)+};}
    all!(257;u8,i8,u16,i16,u32,i32,u64,i64,u128,i128,PcuU256,PcuI256,PcuU512,PcuI512,
        PcuF16Bits,PcuBf16Bits,PcuF8E4M3FnBits,PcuF8E5M2Bits,f32,f64,PcuF128Bits,PcuF256Bits);
    all!(65537;u8,i8,u16,i16,u32,i32,u64,i64,u128,i128,PcuU256,PcuI256,PcuU512,PcuI512,
        PcuF16Bits,PcuBf16Bits,PcuF8E4M3FnBits,PcuF8E5M2Bits,f32,f64,PcuF128Bits,PcuF256Bits);
}
