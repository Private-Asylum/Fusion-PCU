//! Fail closed on observed GPU activity; external exclusive ownership is still required.
pub fn guard() {
    let activity = std::fs::read_to_string("/sys/class/drm/card1/device/gpu_busy_percent")
        .expect("card1 GPU activity must be readable");
    let percentage: u32 = activity.trim().parse().expect("GPU activity percentage");
    assert_eq!(
        percentage, 0,
        "GPU active; native SGD run is provisional and must stop"
    );
}
