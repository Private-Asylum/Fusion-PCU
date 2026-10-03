//! Genuine identity roundtrip plus explicitly distinct copy-only staging diagnostics.
extern crate pcu_facade as fusion_pcu;

#[path = "support/activity.rs"]
mod activity;
#[cfg(feature = "allocation-census")]
#[path = "support/allocations/allocations.rs"]
mod allocations;
#[path = "pinned_transfer/driver/driver.rs"]
mod driver;
#[path = "pinned_transfer/native/native.rs"]
mod native;
#[path = "support/discovery/discovery.rs"]
mod selection;
#[path = "pinned_transfer/source/source.rs"]
mod source;

#[rustfmt::skip]
use criterion::{
    Criterion,
    Throughput,
    criterion_group,
    criterion_main,
};
use fusion_pcu_cuda::CudaRuntime;
use std::hint::black_box;

fn pinned_transfer(criterion: &mut Criterion) {
    activity::activity_guard();
    let (_discovery, backend) = selection::selected_device();
    driver::configure(&backend);
    let runtime = CudaRuntime::new(0).expect("selected CUDA device");
    driver::case::<4096>(criterion, &backend, &runtime);
    driver::case::<4_194_304>(criterion, &backend, &runtime);
    let stream = runtime.create_stream().expect("transfer stream");
    let mut group = criterion.benchmark_group("cuda_copy_only_host_device_roundtrip_diagnostic");
    group.sample_size(20);
    group.measurement_time(std::time::Duration::from_secs(3));
    for bytes in [4096_usize, 4 * 1024 * 1024] {
        activity::compute_owner_guard();
        let mut device = runtime.allocate(bytes).unwrap();
        let mut pageable_source = vec![0; bytes];
        let mut pageable_output = vec![0; bytes];
        let mut upload = Some(runtime.allocate_pinned(bytes).unwrap());
        let mut download = Some(runtime.allocate_pinned(bytes).unwrap());
        // Mutating the full source and checking the full destination is outside timed loops.
        for marker in [17_u8, 239] {
            pageable_source.fill(marker);
            device.copy_from(&pageable_source).unwrap();
            device.copy_to(&mut pageable_output).unwrap();
            assert_eq!(pageable_output, pageable_source);
            let mut staging = upload.take().unwrap();
            staging.as_bytes_mut().copy_from_slice(&pageable_source);
            upload = Some(
                staging
                    .upload_reusable(&device, &stream)
                    .unwrap()
                    .finish()
                    .unwrap(),
            );
            let returned = download
                .take()
                .unwrap()
                .download(&device, &stream)
                .unwrap()
                .finish()
                .unwrap();
            assert_eq!(returned.as_bytes(), pageable_source);
            download = Some(returned);
        }
        // Allocations persist across both routes. Pinned timing includes each transfer's event
        // completion and its safe staging reclaim; pageable timing includes synchronous copies.
        group.throughput(Throughput::Bytes(u64::try_from(bytes * 2).unwrap()));
        group.bench_function(format!("pageable/{bytes}"), |bench| {
            bench.iter(|| {
                device.copy_from(black_box(&pageable_source)).unwrap();
                device.copy_to(black_box(&mut pageable_output)).unwrap();
            });
        });
        group.bench_function(format!("pinned/{bytes}"), |bench| {
            bench.iter(|| {
                upload = Some(
                    upload
                        .take()
                        .unwrap()
                        .upload_reusable(&device, &stream)
                        .unwrap()
                        .finish()
                        .unwrap(),
                );
                download = Some(
                    download
                        .take()
                        .unwrap()
                        .download(&device, &stream)
                        .unwrap()
                        .finish()
                        .unwrap(),
                );
                black_box(download.as_ref().unwrap().as_bytes());
            });
        });
    }
    group.finish();
}

criterion_group!(benches, pinned_transfer);
criterion_main!(benches);
