//! Non-disruptive activity gate for dedicated hardware benchmarks.
use std::process::Command;

pub fn activity_guard() {
    let result = Command::new("nvidia-smi")
        .args([
            "--query-gpu=utilization.gpu",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .expect("nvidia-smi activity guard must be available");
    assert!(result.status.success(), "nvidia-smi activity guard failed");
    let utilization = String::from_utf8(result.stdout).expect("GPU utilization text");
    assert!(
        utilization
            .lines()
            .all(|line| line.trim().parse::<u32>().is_ok_and(|value| value <= 10)),
        "GPU activity exceeds 10 percent"
    );
    let applications = Command::new("nvidia-smi")
        .args(["--query-compute-apps=pid", "--format=csv,noheader,nounits"])
        .output()
        .expect("compute activity guard must be available");
    assert!(
        applications.status.success()
            && String::from_utf8(applications.stdout)
                .expect("compute PID text")
                .lines()
                .filter(|line| !line.trim().is_empty())
                .all(|line| line
                    .trim()
                    .parse::<u32>()
                    .is_ok_and(|pid| pid == std::process::id())),
        "another compute application owns the device"
    );
}
