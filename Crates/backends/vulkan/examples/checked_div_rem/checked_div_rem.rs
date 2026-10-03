//! Ordinary U32-synthesized signed division and transactional publication of both results.
#[rustfmt::skip]
use pcu_facade::{pcu,global,PcuExecutionFaultKind};
#[pcu(invocations=3,crate_path=::pcu_facade)]
fn divide(left: &[i64], right: &[i64], quotient: &mut [i64], remainder: &mut [i64]) {
    let id = context.global_invocation_id;
    let (q, r) = pcu::checked_div_rem(left[id], right[id]);
    quotient[id] = q;
    remainder[id] = r;
}
fn main() {
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Vulkan,
        ..Default::default()
    })
    .unwrap();
    let mut quotient = [27; 5];
    let mut remainder = [29; 5];
    divide(
        &[-7, i64::MIN, 13],
        &[3, 1, -5],
        &mut quotient,
        &mut remainder,
    )
    .unwrap();
    assert_eq!(quotient, [-2, i64::MIN, -2, 27, 27]);
    assert_eq!(remainder, [-1, 0, 3, 29, 29]);
    let saved = (quotient, remainder);
    let fault = divide(
        &[-7, i64::MIN, 13],
        &[3, -1, 0],
        &mut quotient,
        &mut remainder,
    )
    .unwrap_err()
    .arithmetic_fault()
    .unwrap();
    assert_eq!(fault.kind, PcuExecutionFaultKind::SignedDivisionOverflow);
    assert_eq!(fault.invocation_id, 1);
    assert!(!fault.recovered);
    assert_eq!((quotient, remainder), saved);
    divide(&[7, 8, 9], &[2, 3, 4], &mut quotient, &mut remainder).unwrap();
    assert_eq!(quotient, [3, 2, 2, 27, 27]);
    assert_eq!(remainder, [1, 2, 1, 29, 29]);
    global::clear_thread_cache().unwrap();
    global::use_defaults().unwrap();
    println!(
        "Vulkan checked i64 division publishes both exact results; fatal arithmetic preserves both outputs and retry succeeds"
    );
}
