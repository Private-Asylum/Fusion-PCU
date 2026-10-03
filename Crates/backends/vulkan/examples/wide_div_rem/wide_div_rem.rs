//! Ordinary Vulkan 512-bit joint division, preserving tails and both outputs on failure.
#[path = "../../../cpu/tests/wide_div_rem/source/source.rs"]
#[allow(dead_code)] // Example uses direct source; source fixtures also qualify grid/Strict.
mod source;
use pcu_facade::{global, PcuU512};
fn main() {
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Vulkan,
        ..Default::default()
    })
    .unwrap();
    let numerator = PcuU512::from_limbs_le([7, 0, 0, 0, 0, 0, 0, 1]);
    let three = PcuU512::from_limbs_le([3, 0, 0, 0, 0, 0, 0, 0]);
    let sentinel = PcuU512::from_limbs_le([99; 8]);
    let mut q = [sentinel; 2];
    let mut r = q;
    source::direct::<PcuU512, 1>(&[numerator], &[three], &mut q, &mut r).unwrap();
    assert_eq!((q[1], r[1]), (sentinel, sentinel));
    let before = (q, r);
    assert!(
        source::direct::<PcuU512, 1>(
            &[numerator],
            &[PcuU512::from_limbs_le([0; 8])],
            &mut q,
            &mut r
        )
        .is_err()
    );
    assert_eq!((q, r), before);
    println!(
        "Vulkan512 quotient={:?}, remainder={:?}; zero divisor preserves both outputs",
        q[0], r[0]
    );
}
