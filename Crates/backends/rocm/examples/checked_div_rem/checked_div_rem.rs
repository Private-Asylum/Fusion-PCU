//! Ordinary integer quotient/remainder with two-output fault rollback.
extern crate pcu_facade as fusion_pcu;
use fusion_pcu::pcu;
#[pcu(invocations = N)]
fn divide<const N: usize>(lhs: &[i64], rhs: &[i64], quotient: &mut [i64], remainder: &mut [i64]) {
    let id = pcu::context::global_invocation_id();
    let (q, r) = pcu::checked_div_rem(lhs[id], rhs[id]);
    quotient[id] = q;
    remainder[id] = r;
}
fn main() {
    fusion_pcu::global::configure(fusion_pcu::global::PcuExecutionPolicy {
        backend: fusion_pcu::global::PcuBackendChoice::Rocm,
        ..Default::default()
    })
    .unwrap();
    let mut q = [99; 4];
    let mut r = q;
    divide::<3>(&[-7, 7, -7], &[3, -3, -3], &mut q, &mut r).unwrap();
    assert_eq!(q, [-2, -2, 2, 99]);
    assert_eq!(r, [-1, 1, -1, 99]);
    let before = (q, r);
    assert!(
        matches!(divide::<3>(&[7, i64::MIN, 9], &[3, -1, 2], &mut q, &mut r),
        Err(fusion_pcu::PcuExecutionError::ArithmeticFault(fault)) if fault.invocation_id == 1
            && fault.kind == fusion_pcu::PcuExecutionFaultKind::SignedDivisionOverflow)
    );
    assert_eq!((q, r), before);
    println!("quotient={q:?}; remainder={r:?}; both host outputs preserved on overflow");
    fusion_pcu::global::clear_thread_cache().unwrap();
}
