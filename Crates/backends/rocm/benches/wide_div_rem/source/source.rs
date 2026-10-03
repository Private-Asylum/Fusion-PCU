//! Genuine checked joint quotient/remainder source; policy belongs to each invocation.
#[rustfmt::skip]
use fusion_pcu::{
    pcu,
    PcuCheckedIntegerDivision,
    PcuExecutionError,
    PcuScalar,
    PcuTensor,
};
#[pcu(invocations = N)]
pub fn direct<T: PcuCheckedIntegerDivision, const N: usize>(
    lhs: &[T],
    rhs: &[T],
    quotient: &mut [T],
    remainder: &mut [T],
) {
    let id = pcu::context::global_invocation_id();
    let (q, r) = pcu::checked_div_rem(lhs[id], rhs[id]);
    quotient[id] = q;
    remainder[id] = r;
}
#[pcu(invocations = 3)]
pub fn grid<T: PcuCheckedIntegerDivision, const N: usize>(
    lhs: &[T],
    rhs: &[T],
    quotient: &mut [T],
    remainder: &mut [T],
) {
    let mut id = pcu::context::global_invocation_id();
    let stride = pcu::context::invocation_count();
    while id < N {
        let (q, r) = pcu::checked_div_rem(lhs[id], rhs[id]);
        quotient[id] = q;
        remainder[id] = r;
        id += stride;
    }
}
#[pcu]
pub fn identity<T: PcuScalar>(input: &[T]) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::identity(input)
}
