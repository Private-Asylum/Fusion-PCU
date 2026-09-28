//! Hardware regression for HIP runtime handles across short-lived threads.
//!
//! Run explicitly on a `ROCm` host with:
//! `cargo test -p fusion-pcu-example-rocm --test hip_runtime_lifecycle -- --ignored --nocapture`

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
