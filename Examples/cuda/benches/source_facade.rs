//! Matched checked F32 source-facade and manual CUDA boundaries.
#[path = "support/support.rs"]
mod support;

#[rustfmt::skip]
use criterion::{
    Criterion,
    Throughput,
    criterion_group,
    criterion_main,
};
#[rustfmt::skip]
use fusion_pcu::{
    global,
};
use support::source::transform;

fn case<const N: usize>(criterion: &mut Criterion) {
    support::activity::activity_guard();
    let (_discovery, backend) = support::cold("device_open", support::discovery::selected_device);
    let mut prepared = support::cold("source_prepare", || {
        support::source::transform_prepare::<N, _>(&backend).unwrap()
    });
    let mut native = support::fixture::host::<N>(&backend, &mut prepared);
    {
        let mut group = criterion.benchmark_group("cuda_source_checked_f32_host");
        group.sample_size(20);
        group.throughput(Throughput::Elements(u64::try_from(N).unwrap()));
        group.warm_up_time(std::time::Duration::from_secs(1));
        group.measurement_time(std::time::Duration::from_secs(2));
        group.bench_function(format!("ordinary_auto/{N}"), |bench| {
            bench.iter_custom(|iterations| {
                support::fixture::measure_host::<N>(iterations, |input, output| {
                    transform::<N>(input, output).unwrap();
                })
            });
        });
        group.bench_function(format!("prepared_source/{N}"), |bench| {
            bench.iter_custom(|iterations| {
                support::fixture::measure_host::<N>(iterations, |input, output| {
                    prepared(input, output).unwrap();
                })
            });
        });
        group.bench_function(format!("native_same_lowering/{N}"), |bench| {
            bench.iter_custom(|iterations| {
                support::fixture::measure_native::<N>(iterations, &mut native)
            });
        });
        group.finish();
    }
    support::activity::activity_guard();
    let mut resident = support::fixture::Resident::new::<N>(&backend);
    let last_pcu_bank;
    let last_native_bank;
    {
        let mut group = criterion.benchmark_group("cuda_source_checked_f32_resident");
        group.sample_size(20);
        group.throughput(Throughput::Elements(u64::try_from(N).unwrap()));
        group.warm_up_time(std::time::Duration::from_secs(1));
        group.measurement_time(std::time::Duration::from_secs(2));
        let mut bank = 0;
        group.bench_function(format!("ordinary_borrowed_mut/{N}"), |bench| {
            bench.iter(|| {
                bank ^= 1;
                transform::<N>(&resident.banks[bank], &mut resident.destination).unwrap();
            });
        });
        last_pcu_bank = bank;
        group.bench_function(format!("native_same_lowering/{N}"), |bench| {
            bench.iter(|| {
                bank ^= 1;
                assert_eq!(resident.native.submit_bank(bank), u64::MAX);
            });
        });
        last_native_bank = bank;
        group.finish();
    }
    resident.verify(last_pcu_bank, last_native_bank);
}

fn source_facade(criterion: &mut Criterion) {
    let wall = std::time::Instant::now();
    support::activity::activity_guard();
    global::use_defaults().unwrap();
    eprintln!(
        "Pinned CUDA device with automatic facade staging; separate checked multiply and add. Host: upload, terminal wait/status, output download. Resident: launch, terminal wait/status. Retained success sentinel, reset after fault, setup/compilation/oracles excluded."
    );
    case::<65>(criterion);
    case::<1_048_576>(criterion);
    global::clear_thread_cache().unwrap();
    eprintln!(
        "diagnostic/whole_benchmark/process_wall={:?}",
        wall.elapsed()
    );
}

criterion_group!(benches, source_facade);
criterion_main!(benches);
