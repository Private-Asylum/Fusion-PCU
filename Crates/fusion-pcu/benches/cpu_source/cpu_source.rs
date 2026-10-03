//! Production global-source, explicit prepared-source and checked native CPU boundaries.
use std::hint::black_box;
#[rustfmt::skip]
use criterion::{
    BenchmarkId,
    Criterion,
    Throughput,
    criterion_group,
    criterion_main,
};
use fusion_pcu_cpu::PcuCpuHostBackend;
#[path = "composition/composition.rs"]
mod composition;
#[path = "native/native.rs"]
mod native;
#[path = "oracle/oracle.rs"]
mod oracle;
#[path = "setup/setup.rs"]
mod setup;
#[path = "source/source.rs"]
mod source;

#[cfg(feature = "cpu-benchmark-control")]
#[path = "census/census.rs"]
mod census;
#[cfg(feature = "cpu-benchmark-control")]
#[global_allocator]
static ALLOCATOR: census::CountingAllocator = census::CountingAllocator;

fn neg_case<const N: usize>(criterion: &mut Criterion) {
    let backend = PcuCpuHostBackend::detect();
    let mut prepared = source::negate_prepare::<N, _>(&backend).unwrap();
    let mut global = |input: &[f32; N], output: &mut [f32]| {
        source::negate(input, output).map_err(|error| setup::global_error(&error))
    };
    let mut prepared_call = |input: &[f32; N], output: &mut [f32]| {
        prepared(input, output).map_err(setup::prepared_error)
    };
    oracle::negate::<N>(&mut global);
    oracle::negate::<N>(&mut prepared_call);
    oracle::negate::<N>(native::negate::<N>);
    let mut group = criterion.benchmark_group("cpu_source/F32Neg");
    group.throughput(Throughput::Elements(u64::try_from(N).unwrap()));
    macro_rules! measure {
        ($label:literal, $call:expr) => {{
            let mut input = setup::array::<f32, N>(1.0);
            let mut output = std::vec![0.0; N];
            group.bench_function(BenchmarkId::new($label, N), |bencher| {
                bencher.iter(|| {
                    input[0] = if input[0].to_bits() == 1.0_f32.to_bits() { 2.0 } else { 1.0 };
                    $call(black_box(&input), black_box(&mut output)).unwrap();
                    black_box(&output);
                });
            });
        }};
    }
    measure!("global_warm", global);
    measure!("prepared_host", prepared_call);
    measure!("native_checked", native::negate::<N>);
    group.finish();
}
fn integer_case<const N: usize, const MUL: bool>(criterion: &mut Criterion) {
    let backend = PcuCpuHostBackend::detect();
    let mut prepared_add = source::add_prepare::<N, _>(&backend).unwrap();
    let mut prepared_mul = source::mul_prepare::<N, _>(&backend).unwrap();
    let mut global = |lhs: &[u64; N], rhs: &[u64; N], output: &mut [u64]| {
        if MUL {
            source::mul(lhs, rhs, output)
        } else {
            source::add(lhs, rhs, output)
        }
        .map_err(|error| setup::global_error(&error))
    };
    let mut prepared_call = |lhs: &[u64; N], rhs: &[u64; N], output: &mut [u64]| {
        if MUL {
            prepared_mul(lhs, rhs, output)
        } else {
            prepared_add(lhs, rhs, output)
        }
        .map_err(setup::prepared_error)
    };
    oracle::integer::<N, MUL>(&mut global);
    oracle::integer::<N, MUL>(&mut prepared_call);
    oracle::integer::<N, MUL>(native::integer::<N, MUL>);
    let mut group = criterion.benchmark_group(if MUL {
        "cpu_source/U64Mul"
    } else {
        "cpu_source/U64Add"
    });
    group.throughput(Throughput::Elements(u64::try_from(N).unwrap()));
    macro_rules! measure {
        ($label:literal, $call:expr) => {{
            let mut lhs = setup::array::<u64, N>(6);
            let rhs = setup::array::<u64, N>(2);
            let mut output = std::vec![0; N];
            group.bench_function(BenchmarkId::new($label, N), |bencher| {
                bencher.iter(|| {
                    lhs[0] = if lhs[0] == 6 { 7 } else { 6 };
                    $call(black_box(&lhs), black_box(&rhs), black_box(&mut output)).unwrap();
                    black_box(&output);
                });
            });
        }};
    }
    measure!("global_warm", global);
    measure!("prepared_host", prepared_call);
    measure!("native_checked", native::integer::<N, MUL>);
    group.finish();
}
fn comparisons(criterion: &mut Criterion) {
    setup::configure();
    neg_case::<17>(criterion);
    neg_case::<4096>(criterion);
    neg_case::<1_048_576>(criterion);
    integer_case::<17, false>(criterion);
    integer_case::<4096, false>(criterion);
    integer_case::<1_048_576, false>(criterion);
    integer_case::<17, true>(criterion);
    integer_case::<4096, true>(criterion);
    integer_case::<1_048_576, true>(criterion);
    composition::comparisons(criterion);
}
criterion_group!(benches, comparisons);
criterion_main!(benches);
