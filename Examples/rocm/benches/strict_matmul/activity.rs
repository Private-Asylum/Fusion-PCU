//! Read-only utilization gate before setup and each profile; no GPU workload is launched here.
#[rustfmt::skip]
use std::{
    fs,
    thread,
    time::Duration,
};

pub fn guard() {
    for attempt in 0..30 {
        let utilization: Vec<u32> = fs::read_dir("/sys/class/drm")
            .expect("DRM activity inventory")
            .filter_map(Result::ok)
            .filter_map(|entry| {
                fs::read_to_string(entry.path().join("device/gpu_busy_percent")).ok()
            })
            .map(|value| value.trim().parse().expect("GPU activity percentage"))
            .collect();
        assert!(
            !utilization.is_empty(),
            "AMD GPU activity guard requires readable gpu_busy_percent"
        );
        if utilization.iter().all(|&value| value <= 10) {
            eprintln!(
                "AMD activity gate passed: utilization <=10%; external lab ownership gate is required"
            );
            return;
        }
        if attempt == 0 {
            eprintln!("AMD activity gate deferred: waiting for preceding GPU work to settle");
        }
        thread::sleep(Duration::from_secs(1));
    }
    panic!("AMD GPU remains above 10 percent after 30 seconds");
}
