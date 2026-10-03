//! One whole-cohort idle preflight; foreign compute owners checked between phases.
use std::process::Command;
pub fn foreign_owners() {
    if std::env::args().any(|argument| argument == "--test") {
        return;
    }
    let result = Command::new("nvidia-smi")
        .args(["--query-compute-apps=pid", "--format=csv,noheader,nounits"])
        .output()
        .expect("native ownership preflight");
    assert!(result.status.success());
    assert!(
        String::from_utf8(result.stdout)
            .unwrap()
            .lines()
            .filter(|row| !row.trim().is_empty())
            .all(|row| row
                .trim()
                .parse::<u32>()
                .is_ok_and(|pid| pid == std::process::id())),
        "foreign compute owner entered the cohort"
    );
}
pub fn guard() {
    if std::env::args().any(|argument| argument == "--test") {
        return;
    }
    foreign_owners();
    let result = Command::new("nvidia-smi")
        .args([
            "--query-gpu=utilization.gpu",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .expect("native idle preflight");
    assert!(result.status.success());
    assert!(
        String::from_utf8(result.stdout)
            .unwrap()
            .lines()
            .all(|row| row.trim().parse::<u32>().is_ok_and(|value| value <= 10)),
        "whole benchmark cohort did not enter idle"
    );
    eprintln!("whole-cohort entry idle <=10%; foreign-owner checks remain between phases");
}
