//! Genuine ordinary integer invocation selected cold from exact source roles.
#[rustfmt::skip]
use pcu_facade::{
    PcuCheckedInteger,
    PcuDispatchIntegerBinaryOp as Op,
    PcuRangePolicy as Range,
    PcuTensor,
    PcuExecutionError,
};
use super::source;
pub type Invocation<T> =
    fn(&PcuTensor<T>, &PcuTensor<T>, &mut [T]) -> Result<(), PcuExecutionError>;
pub fn select<T: PcuCheckedInteger, const N: usize>(
    op: Op,
    range: Range,
    profile: usize,
) -> Invocation<T> {
    match (op, range, profile) {
        (Op::Add, Range::Reject, 0) => source::add_reject_0::<T, N>,
        (Op::Add, Range::Reject, 1) => source::add_reject_1::<T, N>,
        (Op::Add, Range::Reject, 2) => source::add_reject_2::<T, N>,
        (Op::Add, Range::Clamp, 0) => source::add_clamp_0::<T, N>,
        (Op::Add, Range::Clamp, 1) => source::add_clamp_1::<T, N>,
        (Op::Add, Range::Clamp, 2) => source::add_clamp_2::<T, N>,
        (Op::Sub, Range::Reject, 0) => source::sub_reject_0::<T, N>,
        (Op::Sub, Range::Reject, 1) => source::sub_reject_1::<T, N>,
        (Op::Sub, Range::Reject, 2) => source::sub_reject_2::<T, N>,
        (Op::Sub, Range::Clamp, 0) => source::sub_clamp_0::<T, N>,
        (Op::Sub, Range::Clamp, 1) => source::sub_clamp_1::<T, N>,
        (Op::Sub, Range::Clamp, 2) => source::sub_clamp_2::<T, N>,
        (Op::Mul, Range::Reject, 0) => source::mul_reject_0::<T, N>,
        (Op::Mul, Range::Reject, 1) => source::mul_reject_1::<T, N>,
        (Op::Mul, Range::Reject, 2) => source::mul_reject_2::<T, N>,
        (Op::Mul, Range::Clamp, 0) => source::mul_clamp_0::<T, N>,
        (Op::Mul, Range::Clamp, 1) => source::mul_clamp_1::<T, N>,
        (Op::Mul, Range::Clamp, 2) => source::mul_clamp_2::<T, N>,
        _ => unreachable!("three actual integer source roles"),
    }
}
