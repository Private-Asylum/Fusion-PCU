//! Genuine 512-bit source roles borrow retained native owners and publish both exact prefixes.
#[path = "../../../cpu/tests/wide_div_rem/oracle/oracle.rs"]
#[allow(dead_code)]
mod oracle;
#[path = "../../../cpu/tests/scalar_tensor/source/source.rs"]
#[allow(dead_code)]
mod owned;
#[path = "../../../cpu/tests/div_rem_roles/source/source.rs"]
#[allow(dead_code)]
mod source;
#[rustfmt::skip]
use pcu_facade::{global,PcuU512};
fn main() {
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Vulkan,
        ..Default::default()
    })
    .unwrap();
    let sentinel = oracle::small::<PcuU512>(99);
    let mut input = [oracle::small::<PcuU512>(7); 8];
    input[2] = oracle::maximum();
    let right = [oracle::small::<PcuU512>(3); 8];
    let left = owned::identity(&input).unwrap();
    let rhs = owned::identity(&right).unwrap();
    let mut quotient = owned::identity(&[sentinel; 8]).unwrap();
    let mut remainder = owned::identity(&[sentinel; 8]).unwrap();
    global::clear_thread_cache().unwrap();
    source::reordered::<PcuU512, 5>(&rhs, &mut remainder, &left, &mut quotient).unwrap();
    let mut q = [sentinel; 8];
    let mut r = q;
    quotient.read_into(&mut q).unwrap();
    remainder.read_into(&mut r).unwrap();
    for lane in 0..5 {
        assert_eq!(
            (q[lane], r[lane]),
            oracle::evaluate(input[lane], right[lane]).unwrap()
        );
    }
    assert_eq!((&q[5..], &r[5..]), (&[sentinel; 3][..], &[sentinel; 3][..]));
    println!(
        "Vulkan512-bit genuine borrowed source: both native owners, exact results and unchanged tails; originating session survives cache clear"
    );
    global::use_defaults().unwrap();
}
