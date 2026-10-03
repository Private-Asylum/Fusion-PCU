//! Source composition and execution only; retained native banks and fault oracles live separately.
use std::hint::black_box;
#[rustfmt::skip]
use criterion::{
    BenchmarkId,
    Criterion,
    Throughput,
};
use fusion_pcu_cpu::PcuCpuHostBackend;
use super::native::composition::Composed;
#[rustfmt::skip]
use super::{
    oracle,
    setup,
    source,
};

#[path = "integer/integer.rs"]
mod integer;

fn compare<const N: usize, const ORDERED: bool, const HELPER: bool>(criterion: &mut Criterion) {
    let backend = PcuCpuHostBackend::detect();
    let mut locals = source::composed_locals_prepare::<N, _>(&backend).unwrap();
    let mut stores = source::ordered_stores_prepare::<N, _>(&backend).unwrap();
    let mut helper = source::composed_helper_prepare::<N, _>(&backend).unwrap();
    let mut control = Composed::new(N, ORDERED);
    let mut global = |input: &[f32; N], stage: &mut [f32], output: &mut [f32]| {
        if ORDERED {
            source::ordered_stores(input, stage, output)
        } else if HELPER {
            source::composed_helper(input, output)
        } else {
            source::composed_locals(input, output)
        }
        .map_err(|error| setup::global_error(&error))
    };
    let mut prepared = |input: &[f32; N], stage: &mut [f32], output: &mut [f32]| {
        if ORDERED {
            stores(input, stage, output)
        } else if HELPER {
            helper(input, output)
        } else {
            locals(input, output)
        }
        .map_err(setup::prepared_error)
    };
    let mut native = |input: &[f32; N], stage: &mut [f32], output: &mut [f32]| {
        control.call(input, stage, output)
    };
    oracle::composition::<N, ORDERED>(&mut global);
    oracle::composition::<N, ORDERED>(&mut prepared);
    oracle::composition::<N, ORDERED>(&mut native);
    #[cfg(feature = "cpu-benchmark-control")]
    {
        super::census::composed::<N, ORDERED>(
            if HELPER {
                "helper_global_warm"
            } else {
                "global_warm"
            },
            &mut global,
        );
        super::census::composed::<N, ORDERED>(
            if HELPER {
                "helper_prepared_host"
            } else {
                "prepared_host"
            },
            &mut prepared,
        );
        super::census::composed::<N, ORDERED>(
            if HELPER {
                "helper_native_checked"
            } else {
                "native_checked"
            },
            &mut native,
        );
    }
    let name = if ORDERED {
        "cpu_source/F32OrderedStores"
    } else if HELPER {
        "cpu_source/F32HelperLocals"
    } else {
        "cpu_source/F32Locals"
    };
    let mut group = criterion.benchmark_group(name);
    group.throughput(Throughput::Elements(u64::try_from(N).unwrap()));
    macro_rules! measure {
        ($label:literal, $call:expr) => {{
            let mut input = setup::array::<f32, N>(1.0);
            let mut stage = vec![0.0; N];
            let mut output = vec![0.0; N];
            group.bench_function(BenchmarkId::new($label, N), |bencher| {
                bencher.iter(|| {
                    input[0] = if input[0].to_bits() == 1.0_f32.to_bits() {
                        2.0
                    } else {
                        1.0
                    };
                    $call(
                        black_box(&input),
                        black_box(&mut stage),
                        black_box(&mut output),
                    )
                    .unwrap();
                    black_box((&stage, &output));
                });
            });
        }};
    }
    measure!("global_warm", global);
    measure!("prepared_host", prepared);
    measure!("native_checked", native);
    group.finish();
}

pub fn comparisons(criterion: &mut Criterion) {
    compare::<17, false, false>(criterion);
    compare::<4096, false, false>(criterion);
    compare::<1_048_576, false, false>(criterion);
    compare::<17, true, false>(criterion);
    compare::<4096, true, false>(criterion);
    compare::<1_048_576, true, false>(criterion);
    compare::<17, false, true>(criterion);
    compare::<4096, false, true>(criterion);
    compare::<1_048_576, false, true>(criterion);
    integer::compare::<17>(criterion);
    integer::compare::<4096>(criterion);
    integer::compare::<1_048_576>(criterion);
}
