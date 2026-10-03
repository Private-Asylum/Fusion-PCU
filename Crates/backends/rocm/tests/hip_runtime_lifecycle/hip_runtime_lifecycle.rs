//! Hardware regression for HIP runtime handles across short-lived threads.
//!
//! Run explicitly on a `ROCm` host with:
//! `cargo test -p fusion-pcu-rocm --features tensor --test hip_runtime_lifecycle -- --ignored --nocapture`

extern crate pcu_facade as fusion_pcu;

use fusion_pcu_rocm::HipRuntime;

fn query_runtime_lifecycle() -> (u32, usize, u32) {
    let probe = HipRuntime::probe().expect("HIP probe succeeds");
    assert!(
        probe.device_count > 0,
        "this regression requires a visible ROCm device"
    );

    let devices = HipRuntime::enumerate_devices().expect("HIP enumeration succeeds");
    assert!(
        !devices.is_empty(),
        "the probed ROCm device remains enumerable"
    );

    let runtime = HipRuntime::new(0).expect("the first visible ROCm device can be selected");
    let runtime_count = runtime
        .device_count()
        .expect("selected runtime remains usable");
    drop(runtime);

    (probe.device_count, devices.len(), runtime_count)
}

#[test]
#[ignore = "requires a visible ROCm device; run explicitly on ROCm hardware"]
fn hip_runtime_remains_usable_after_short_lived_threads_exit() {
    let first = std::thread::spawn(query_runtime_lifecycle)
        .join()
        .expect("first HIP thread exits cleanly");
    let second = std::thread::spawn(query_runtime_lifecycle)
        .join()
        .expect("second HIP thread exits cleanly");
    let third = std::thread::spawn(query_runtime_lifecycle)
        .join()
        .expect("third HIP thread exits cleanly");

    assert_eq!(
        second, first,
        "HIP discovery survives the first thread exit"
    );
    assert_eq!(third, first, "HIP discovery survives repeated thread exits");
}

#[cfg(feature = "allocation-census")]
#[test]
#[ignore = "requires authorized ROCm hardware; correctness/census only"]
fn retained_runtime_entries_select_device_on_another_thread_without_resolving() {
    let runtime = HipRuntime::new(0).unwrap();
    std::thread::spawn(move || {
        fusion_pcu_rocm::reset_rocm_api_census();
        assert!(runtime.device_count().unwrap() > 0);
        assert!(runtime.memory_info().unwrap().total_bytes > 0);
        let info = runtime.device_info().unwrap();
        assert!(!info.name.is_empty());
        let mut buffer = runtime.allocate(16).unwrap();
        buffer.copy_from(&[3_u8; 16]).unwrap();
        let mut output = [0_u8; 16];
        buffer.copy_to(&mut output).unwrap();
        assert_eq!(output, [3_u8; 16]);
        let stream = runtime.create_stream().unwrap();
        stream.synchronize().unwrap();
        drop(stream);
        drop(buffer);
        let api = fusion_pcu_rocm::rocm_api_census();
        assert_eq!(api.symbol_resolutions, 0);
        assert_eq!(api.allocations, 1);
        assert_eq!(api.frees, 1);
        assert_eq!(api.host_to_device_copies, 1);
        assert_eq!(api.device_to_host_copies, 1);
        assert_eq!(api.device_selections * 2, api.runtime_calls);
        assert!(api.device_selections >= 10);
        eprintln!("retained cross-thread HIP: {api:?}");
    })
    .join()
    .unwrap();
}
