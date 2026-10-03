//! Genuine helper source and independent native checked execution, with matching boundaries.
#[rustfmt::skip]
use super::{
    black_box,
    BenchmarkId,
    Criterion,
    oracle,
    PcuCpuHostBackend,
    setup,
    source,
    Throughput,
};
use super::super::native::helper_integer::Helper;

pub(super) fn compare<const N: usize>(criterion: &mut Criterion) {
    let backend = PcuCpuHostBackend::detect();
    let mut kernel = source::composed_integer_helper_prepare::<N, _>(&backend).unwrap();
    let mut control = Helper::new(N);
    let mut global = |input: &[u64; N], seed: &u64, output: &mut [u64]| {
        source::composed_integer_helper(input, seed, output)
            .map_err(|error| setup::global_error(&error))
    };
    let mut prepared = |input: &[u64; N], seed: &u64, output: &mut [u64]| {
        kernel(input, seed, output).map_err(setup::prepared_error)
    };
    let mut native =
        |input: &[u64; N], seed: &u64, output: &mut [u64]| control.call(input, seed, output);
    oracle::helper_integer::verify::<N>(&mut global);
    oracle::helper_integer::verify::<N>(&mut prepared);
    oracle::helper_integer::verify::<N>(&mut native);
    #[cfg(feature = "cpu-benchmark-control")]
    {
        super::super::census::helper_integer::<N>("integer_helper_global_warm", &mut global);
        super::super::census::helper_integer::<N>("integer_helper_prepared_host", &mut prepared);
        super::super::census::helper_integer::<N>("integer_helper_native_checked", &mut native);
    }
    let mut group = criterion.benchmark_group("cpu_source/U64HelperLocals");
    group.throughput(Throughput::Elements(u64::try_from(N).unwrap()));
    macro_rules! measure {
        ($label:literal, $call:expr) => {{
            let mut input = setup::array::<u64, N>(1);
            let mut output = vec![0; N];
            group.bench_function(BenchmarkId::new($label, N), |bencher| {
                bencher.iter(|| {
                    // Fresh input on every measured call, well below the U64 squared limit.
                    input[0] = input[0].checked_add(1).unwrap();
                    $call(black_box(&input), black_box(&1), black_box(&mut output)).unwrap();
                    black_box(&output);
                });
            });
        }};
    }
    measure!("global_warm", global);
    measure!("prepared_host", prepared);
    measure!("native_checked", native);
    group.finish();
}
