//! Non-disruptive activity gate, allowing preceding benchmark work to settle outside timing.
use std::process::Command;

pub fn guard() {
    for attempt in 0..30 {
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
        let result = Command::new("nvidia-smi")
            .args([
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
            ])
            .output()
            .expect("nvidia-smi activity guard must be available");
        assert!(result.status.success(), "nvidia-smi activity guard failed");
        let utilization = String::from_utf8(result.stdout).expect("GPU utilization text");
        if utilization
            .lines()
            .all(|line| line.trim().parse::<u32>().is_ok_and(|value| value <= 10))
        {
            eprintln!("activity gate passed: utilization <=10%, no foreign compute PID");
            return;
        }
        if attempt == 0 {
            eprintln!("activity gate deferred: waiting for preceding GPU work to settle");
        }
        std::thread::sleep(std::time::Duration::from_secs(1));
    }
    panic!("GPU activity remains above 10 percent after 30 seconds");
}
