//! Ordinary 512-bit quotient/remainder with retained tails and atomic failed-map rollback.
#[path = "../../tests/wide_div_rem/source/source.rs"]
#[allow(dead_code)] // The example demonstrates direct source; tests also qualify grid/Strict.
mod source;
use pcu_facade::{global, PcuU512};
fn main() {
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Cpu,
        ..Default::default()
    })
    .unwrap();
    let numerator = PcuU512::from_limbs_le([7, 0, 0, 0, 0, 0, 0, 1]);
    let three = PcuU512::from_limbs_le([3, 0, 0, 0, 0, 0, 0, 0]);
    let tail = PcuU512::from_limbs_le([99; 8]);
    let mut q = [tail; 2];
    let mut r = q;
    source::direct::<PcuU512, 1>(&[numerator], &[three], &mut q, &mut r).unwrap();
    assert_eq!((q[1], r[1]), (tail, tail));
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
        "512-bit quotient={:?}, remainder={:?}; zero divisor preserves both outputs",
        q[0], r[0]
    );
}
