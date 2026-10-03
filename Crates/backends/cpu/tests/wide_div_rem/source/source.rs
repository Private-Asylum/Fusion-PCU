//! Genuine generic checked integer division, preserving two public outputs.
use pcu_facade::{pcu, PcuCheckedIntegerDivision};
#[pcu(invocations = N, crate_path = ::pcu_facade)]
pub fn direct<T: PcuCheckedIntegerDivision, const N: usize>(
    lhs: &[T],
    rhs: &[T],
    quotient: &mut [T],
    remainder: &mut [T],
) {
    let id = context.global_invocation_id;
    let (q, r) = pcu::checked_div_rem(lhs[id], rhs[id]);
    quotient[id] = q;
    remainder[id] = r;
}
#[pcu(invocations = 3, flag(strict), crate_path = ::pcu_facade)]
pub fn grid<T: PcuCheckedIntegerDivision, const N: usize>(
    lhs: &[T],
    rhs: &[T],
    quotient: &mut [T],
    remainder: &mut [T],
) {
    let mut id = context.global_invocation_id;
    let stride = context.invocation_count;
    while id < N {
        let (q, r) = pcu::checked_div_rem(lhs[id], rhs[id]);
        quotient[id] = q;
        remainder[id] = r;
        id += stride;
    }
}
