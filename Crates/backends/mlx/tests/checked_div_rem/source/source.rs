//! Genuine checked quotient/remainder source; preparation specializes the sealed scalar cold.
#[rustfmt::skip]
use pcu_facade::{
    pcu,
    PcuCheckedIntegerDivision,
};
#[pcu(invocations=N,crate_path=::pcu_facade)]
pub fn direct<T: PcuCheckedIntegerDivision, const N: usize>(
    left: &[T],
    right: &[T],
    quotient: &mut [T],
    remainder: &mut [T],
) {
    let id = pcu::context::global_invocation_id();
    let (q, r) = pcu::checked_div_rem(left[id], right[id]);
    quotient[id] = q;
    remainder[id] = r;
}
#[pcu(invocations=1,flag(strict),crate_path=::pcu_facade)]
pub fn grid<T: PcuCheckedIntegerDivision, const N: usize>(
    left: &[T],
    right: &[T],
    quotient: &mut [T],
    remainder: &mut [T],
) {
    let mut id = pcu::context::global_invocation_id();
    let stride = pcu::context::invocation_count();
    while id < N {
        let (q, r) = pcu::checked_div_rem(left[id], right[id]);
        quotient[id] = q;
        remainder[id] = r;
        id += stride;
    }
}
