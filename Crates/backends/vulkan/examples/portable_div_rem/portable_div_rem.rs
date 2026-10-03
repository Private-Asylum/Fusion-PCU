//! Ordinary 512-bit broadcast/indexed joint division with two useful outputs and rollback.
#[rustfmt::skip]
use pcu_facade::{
    global,
    PcuU512,
};
#[path = "../../../cpu/tests/portable_div_rem/source/source.rs"]
#[allow(dead_code)]
// The example executes the grid profile; all source profiles have independent tests.
mod source;
fn main() {
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Vulkan,
        ..Default::default()
    })
    .unwrap();
    let seven = PcuU512::from_limbs_le([7, 0, 0, 0, 0, 0, 0, 0]);
    let three = PcuU512::from_limbs_le([3, 0, 0, 0, 0, 0, 0, 0]);
    let sentinel = PcuU512::from_limbs_le([99, 0, 0, 0, 0, 0, 0, 0]);
    let mut input = [three; 5];
    input[0] = seven;
    let mut q = [sentinel; 8];
    let mut r = q;
    source::grid::<PcuU512, 5>(&mut r, &[], &mut q, &input).unwrap();
    assert_eq!(q[1].to_limbs_le(), [2, 0, 0, 0, 0, 0, 0, 0]);
    assert_eq!(r[1].to_limbs_le(), [1, 0, 0, 0, 0, 0, 0, 0]);
    assert_eq!((&q[5..], &r[5..]), (&[sentinel; 3][..], &[sentinel; 3][..]));
    let before = (q, r);
    input[2] = PcuU512::from_limbs_le([0; 8]);
    let fault = source::grid::<PcuU512, 5>(&mut r, &[], &mut q, &input)
        .unwrap_err()
        .arithmetic_fault()
        .unwrap();
    assert_eq!(fault.invocation_id, 2);
    assert_eq!((q, r), before);
    input[2] = three;
    source::grid::<PcuU512, 5>(&mut r, &[], &mut q, &input).unwrap();
    println!(
        "Portable 512-bit joint division uses the original scalar input and preserves both outputs on a fatal lane"
    );
}
