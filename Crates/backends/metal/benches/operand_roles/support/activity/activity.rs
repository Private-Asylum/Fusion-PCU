//! Statistical timing requires idle activity; correctness-only modes record actual load.
pub fn guard() {
    let result = std::process::Command::new("/usr/sbin/ioreg")
        .args(["-r", "-c", "AGXAccelerator", "-l"])
        .output()
        .unwrap();
    assert!(result.status.success());
    let inventory = String::from_utf8(result.stdout).unwrap();
    let activity: u32 = inventory
        .split("\"Device Utilization %\"=")
        .nth(1)
        .expect("GPU activity counter unavailable")
        .chars()
        .take_while(char::is_ascii_digit)
        .collect::<String>()
        .parse()
        .unwrap();
    println!("Metal activity guard: {activity}%");
    let correctness = cfg!(feature = "allocation-census")
        || std::env::args().any(|argument| argument == "--test");
    if correctness {
        println!(
            "Correctness/census only; observed activity {activity}%, no idle or statistical claim"
        );
    } else {
        assert!(activity <= 5, "GPU busy: refusing statistical timings");
    }
}
