//! Ordinary full-width Vulkan integer maps and useful observable saturation.
#[rustfmt::skip]
use pcu_facade::{pcu,global,PcuCheckedInteger,PcuU512};
#[pcu(invocations=3,crate_path=::pcu_facade,flag(clamp_range))]
fn add<T: PcuCheckedInteger>(left: &[T], right: &[T], output: &mut [T]) {
    let id = context.global_invocation_id;
    output[id] = left[id] + right[id];
}
fn main() {
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Vulkan,
        ..Default::default()
    })
    .unwrap();
    let left = [
        PcuU512::from_limbs_le([u64::MAX; 8]),
        PcuU512::from_limbs_le([0, 0, 0, 0, 0, 0, 0, 1]),
        PcuU512::ZERO,
    ];
    let right = [PcuU512::from_limbs_le([1, 0, 0, 0, 0, 0, 0, 0]); 3];
    let sentinel = PcuU512::from_limbs_le([17; 8]);
    let mut output = [sentinel; 5];
    let error = add(&left, &right, &mut output).unwrap_err();
    let fault = error.arithmetic_fault().unwrap();
    assert!(fault.recovered);
    assert_eq!(fault.invocation_id, 0);
    assert_eq!(output[0], left[0]);
    assert_eq!(output[1], PcuU512::from_limbs_le([1, 0, 0, 0, 0, 0, 0, 1]));
    assert_eq!(output[2], right[0]);
    assert_eq!(output[3..], [sentinel; 2]);
    println!(
        "Vulkan U512 addition publishes exact complete saturation plus an observable recovered error"
    );
    global::clear_thread_cache().unwrap();
    global::use_defaults().unwrap();
}
