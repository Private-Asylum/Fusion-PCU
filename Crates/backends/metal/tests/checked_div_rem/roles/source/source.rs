//! Genuine source roles; unused declarations carry no resource or affinity requirement.
#[rustfmt::skip]
use pcu_facade::{
    pcu,
    PcuCheckedIntegerDivision,
};
#[pcu(invocations=N,crate_path=::pcu_facade)]
pub fn repeated<T: PcuCheckedIntegerDivision, const N: usize>(
    q: &mut [T],
    input: &[T],
    r: &mut [T],
) {
    let id = pcu::context::global_invocation_id();
    let (quotient, remainder) = pcu::checked_div_rem(input[id], input[id]);
    r[id] = remainder;
    q[id] = quotient;
}
#[pcu(invocations=N,crate_path=::pcu_facade)]
pub fn unread<T: PcuCheckedIntegerDivision, const N: usize>(
    _unused: &[T],
    r: &mut [T],
    input: &[T],
    q: &mut [T],
) {
    let id = pcu::context::global_invocation_id();
    let (quotient, remainder) = pcu::checked_div_rem(input[id], input[0]);
    q[id] = quotient;
    r[id] = remainder;
}
#[pcu(invocations=N,crate_path=::pcu_facade)]
pub fn reordered<T: PcuCheckedIntegerDivision, const N: usize>(
    right: &[T],
    r: &mut [T],
    left: &[T],
    q: &mut [T],
) {
    let id = pcu::context::global_invocation_id();
    let (quotient, remainder) = pcu::checked_div_rem(left[id], right[id]);
    r[id] = remainder;
    q[id] = quotient;
}
#[pcu(invocations=3,flag(strict),crate_path=::pcu_facade)]
pub fn grid<T: PcuCheckedIntegerDivision, const N: usize>(
    _unused: &[T],
    r: &mut [T],
    input: &[T],
    q: &mut [T],
) {
    let mut id = pcu::context::global_invocation_id();
    let stride = pcu::context::invocation_count();
    while id < N {
        let (quotient, remainder) = pcu::checked_div_rem(input[0], input[id]);
        r[id] = remainder;
        q[id] = quotient;
        id += stride;
    }
}
#[pcu(invocations=N,crate_path=::pcu_facade)]
pub fn scalar_divisor<T: PcuCheckedIntegerDivision, const N: usize>(
    q: &mut [T],
    divisor: &T,
    r: &mut [T],
    input: &[T],
) {
    let id = pcu::context::global_invocation_id();
    let (quotient, remainder) = pcu::checked_div_rem(input[id], *divisor);
    q[id] = quotient;
    r[id] = remainder;
}
#[pcu(invocations=3,crate_path=::pcu_facade)]
pub fn scalar_grid<T: PcuCheckedIntegerDivision, const N: usize>(
    input: &T,
    q: &mut [T],
    r: &mut [T],
) {
    let mut id = pcu::context::global_invocation_id();
    let stride = pcu::context::invocation_count();
    while id < N {
        let (quotient, remainder) = pcu::checked_div_rem(*input, *input);
        q[id] = quotient;
        r[id] = remainder;
        id += stride;
    }
}
