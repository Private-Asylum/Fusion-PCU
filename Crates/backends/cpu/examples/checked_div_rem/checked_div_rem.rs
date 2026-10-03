//! Exact signed quotient/remainder, terminal-fault rollback and retained caller tails.
use fusion_pcu_cpu::PcuCpuHostBackend;
use pcu_facade::pcu;
#[pcu(invocations = N, crate_path = ::pcu_facade)]
fn divide<const N: usize>(lhs: &[i64], rhs: &[i64], quotient: &mut [i64], remainder: &mut [i64]) {
    let id = context.global_invocation_id;
    let (q, r) = pcu::checked_div_rem(lhs[id], rhs[id]);
    quotient[id] = q;
    remainder[id] = r;
}
fn main() {
    let mut call = divide_prepare::<3, _>(&PcuCpuHostBackend::scalar()).unwrap();
    let mut q = [99; 4];
    let mut r = q;
    call(&[-7, 7, -7], &[3, -3, -3], &mut q, &mut r).unwrap();
    assert_eq!(q, [-2, -2, 2, 99]);
    assert_eq!(r, [-1, 1, -1, 99]);
    let before = (q, r);
    let fault = call(&[-7, i64::MIN, -7], &[3, -1, -3], &mut q, &mut r).unwrap_err();
    assert_eq!(fault.fault().unwrap().invocation_id, 1);
    assert_eq!((q, r), before);
    println!("quotient={q:?}; remainder={r:?}; signed overflow preserves both outputs");
}
