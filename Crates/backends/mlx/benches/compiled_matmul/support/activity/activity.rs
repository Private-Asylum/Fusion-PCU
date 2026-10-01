//! Independent hardware activity guard for Apple silicon MLX benchmark hosts.

use std::process::Command;

pub fn gpu_idle_guard() {
    activity(true);
}

pub fn gpu_post_guard() {
    activity(false);
}

fn activity(require_idle: bool) {
    if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
        metal(require_idle);
    } else {
        panic!("Fusion PCU MLX benchmarks require Apple silicon macOS");
    }
}

fn metal(require_idle: bool) {
    let result = Command::new("/usr/sbin/ioreg")
        .args(["-r", "-d", "1", "-c", "AGXAccelerator"])
        .output()
        .expect("Mac AGX telemetry is required before this benchmark");
    assert!(result.status.success(), "AGX telemetry failed");
    let text = String::from_utf8(result.stdout).unwrap();
    let stats = text
        .lines()
        .find(|line| line.contains("\"PerformanceStatistics\""))
        .expect("AGX activity is unavailable");
    for field in [
        "Device Utilization %",
        "Renderer Utilization %",
        "Tiler Utilization %",
    ] {
        let key = format!("\"{field}\"=");
        let value: u32 = stats
            .split(&key)
            .nth(1)
            .expect("AGX activity field unavailable")
            .split(|character: char| !character.is_ascii_digit())
            .next()
            .unwrap()
            .parse()
            .unwrap();
        assert!(!require_idle || value <= 5, "GPU busy: {field}={value}%");
        eprintln!("MLX_GPU_ACTIVITY {field}={value}%");
    }
}
