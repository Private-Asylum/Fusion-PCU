//! Genuine three/four-declaration joint division with independent mathematical roles.
#[rustfmt::skip]
use pcu_facade::{
    pcu,
    PcuCheckedIntegerDivision,
};
#[pcu(invocations = N, crate_path = ::pcu_facade)]
pub fn repeated<T: PcuCheckedIntegerDivision, const N: usize>(
    q: &mut [T],
    r: &mut [T],
    input: &[T],
) {
    let id = context.global_invocation_id;
    let (quotient, remainder) = pcu::checked_div_rem(input[id], input[id]);
    r[id] = remainder;
    q[id] = quotient;
}
#[pcu(invocations = N, crate_path = ::pcu_facade)]
pub fn unused<T: PcuCheckedIntegerDivision, const N: usize>(
    unused: &[T],
    input: &[T],
    r: &mut [T],
    q: &mut [T],
) {
    let id = context.global_invocation_id;
    let (quotient, remainder) = pcu::checked_div_rem(input[id], input[id]);
    r[id] = remainder;
    q[id] = quotient;
}
#[pcu(invocations = N, crate_path = ::pcu_facade)]
pub fn reordered<T: PcuCheckedIntegerDivision, const N: usize>(
    right: &[T],
    r: &mut [T],
    left: &[T],
    q: &mut [T],
) {
    let id = context.global_invocation_id;
    let (quotient, remainder) = pcu::checked_div_rem(left[id], right[id]);
    r[id] = remainder;
    q[id] = quotient;
}
#[pcu(invocations = N, crate_path = ::pcu_facade)]
pub fn mixed<T: PcuCheckedIntegerDivision, const N: usize>(q: &mut [T], input: &[T], r: &mut [T]) {
    let id = context.global_invocation_id;
    let (quotient, remainder) = pcu::checked_div_rem(input[id], input[0]);
    q[id] = quotient;
    r[id] = remainder;
}
#[pcu(invocations = 3, flag(strict), crate_path = ::pcu_facade)]
pub fn grid<T: PcuCheckedIntegerDivision, const N: usize>(
    r: &mut [T],
    unused: &[T],
    q: &mut [T],
    input: &[T],
) {
    let mut id = context.global_invocation_id;
    let stride = context.invocation_count;
    while id < N {
        let (quotient, remainder) = pcu::checked_div_rem(input[0], input[id]);
        r[id] = remainder;
        q[id] = quotient;
        id += stride;
    }
}
#[pcu(invocations = N, crate_path = ::pcu_facade)]
pub fn scalar<T: PcuCheckedIntegerDivision, const N: usize>(
    left: &T,
    q: &mut [T],
    right: &T,
    r: &mut [T],
) {
    let id = context.global_invocation_id;
    let (quotient, remainder) = pcu::checked_div_rem(*left, *right);
    q[id] = quotient;
    r[id] = remainder;
}
