//! Genuine joint division; the full request belongs to the invocation, not a copied header.
#[rustfmt::skip]
use pcu_facade::{
    pcu,
    PcuCheckedIntegerDivision,
};
#[pcu(crate_path=::pcu_facade,invocations=N)]
pub fn repeated<T: PcuCheckedIntegerDivision, const N: usize>(q: &mut [T], r: &mut [T], a: &[T]) {
    let id = pcu::context::global_invocation_id();
    let (quotient, remainder) = pcu::checked_div_rem(a[id], a[id]);
    q[id] = quotient;
    r[id] = remainder;
}
#[pcu(crate_path=::pcu_facade,invocations=N)]
pub fn unread<T: PcuCheckedIntegerDivision, const N: usize>(
    unused: &[T],
    r: &mut [T],
    a: &[T],
    q: &mut [T],
) {
    let id = pcu::context::global_invocation_id();
    let (quotient, remainder) = pcu::checked_div_rem(a[id], a[id]);
    q[id] = quotient;
    r[id] = remainder;
}
#[pcu(crate_path=::pcu_facade,invocations=N)]
pub fn reordered<T: PcuCheckedIntegerDivision, const N: usize>(
    b: &[T],
    r: &mut [T],
    a: &[T],
    q: &mut [T],
) {
    let id = pcu::context::global_invocation_id();
    let (quotient, remainder) = pcu::checked_div_rem(a[id], b[id]);
    r[id] = remainder;
    q[id] = quotient;
}
#[pcu(crate_path=::pcu_facade,invocations=N)]
pub fn indexed_zero<T: PcuCheckedIntegerDivision, const N: usize>(
    q: &mut [T],
    r: &mut [T],
    a: &[T],
) {
    let id = pcu::context::global_invocation_id();
    let (quotient, remainder) = pcu::checked_div_rem(a[id], a[0usize]);
    q[id] = quotient;
    r[id] = remainder;
}
#[pcu(crate_path=::pcu_facade,invocations=17)]
pub fn grid_zero<T: PcuCheckedIntegerDivision, const N: usize>(
    unused: &[T],
    r: &mut [T],
    a: &[T],
    q: &mut [T],
) {
    let mut id = pcu::context::global_invocation_id();
    let stride = pcu::context::invocation_count();
    while id < N {
        let (quotient, remainder) = pcu::checked_div_rem(a[0], a[id]);
        r[id] = remainder;
        q[id] = quotient;
        id += stride;
    }
}

#[pcu(crate_path=::pcu_facade)]
pub fn identity<T: pcu_facade::PcuScalar>(
    input: &[T],
) -> Result<pcu_facade::PcuTensor<T>, pcu_facade::PcuExecutionError> {
    pcu::identity(input)
}
#[pcu(crate_path=::pcu_facade)]
pub fn fresh_zero<T: pcu_facade::PcuScalar>(
    input: &[T],
) -> Result<pcu_facade::PcuTensor<T>, pcu_facade::PcuExecutionError> {
    pcu::sub(input, input)
}
