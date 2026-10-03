//! One whole-cohort idle preflight; the physical host coordinator owns the AMD window.
use std::fs;
pub fn foreign_owners() {
    if std::env::args().any(|argument| argument == "--test") {
        return;
    }
    // Only accessible KFD descriptors can be checked locally. Statistical acceptance also
    // requires the independently recorded physical-host ownership preflight.
    for entry in fs::read_dir("/proc").unwrap().filter_map(Result::ok) {
        let Ok(pid) = entry.file_name().to_string_lossy().parse::<u32>() else {
            continue;
        };
        if pid == std::process::id() {
            continue;
        }
        let Ok(descriptors) = fs::read_dir(entry.path().join("fd")) else {
            continue;
        };
        for descriptor in descriptors.filter_map(Result::ok) {
            if fs::read_link(descriptor.path())
                .is_ok_and(|path| path == std::path::Path::new("/dev/kfd"))
            {
                panic!("foreign KFD owner entered cohort: {pid}");
            }
        }
    }
}
pub fn guard() {
    if std::env::args().any(|argument| argument == "--test") {
        return;
    }
    foreign_owners();
    let utilization: Vec<u32> = fs::read_dir("/sys/class/drm")
        .unwrap()
        .filter_map(Result::ok)
        .filter_map(|entry| fs::read_to_string(entry.path().join("device/gpu_busy_percent")).ok())
        .map(|value| value.trim().parse().unwrap())
        .collect();
    assert!(!utilization.is_empty());
    assert!(
        utilization.iter().all(|&value| value <= 10),
        "whole AMD benchmark cohort did not enter idle"
    );
    eprintln!(
        "whole-cohort entry idle <=10%; independent physical-host ownership certificate required"
    );
}
