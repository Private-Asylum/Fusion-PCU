//! Real ordinary source function pointer selected cold; no warm lowering or dynamic dispatch.
#[rustfmt::skip]
use pcu_facade::{
    PcuCheckedFloat,
    PcuDispatchFloatBinaryOp as Op,
    PcuFloatUnderflowPolicy as Policy,
    PcuRangePolicy as Range,
    PcuTensor,
    PcuExecutionError,
};
use super::source;
pub type Invocation<T> =
    fn(&PcuTensor<T>, &PcuTensor<T>, &mut [T]) -> Result<(), PcuExecutionError>;
pub fn select<T: PcuCheckedFloat, const N: usize>(
    op: Op,
    policy: Policy,
    range: Range,
    profile: usize,
) -> Invocation<T> {
    match profile {
        0 => dense::<T, N>(op, policy, range),
        1 => grid::<T, N>(op, policy, range),
        2 => repeated::<T, N>(op, policy, range),
        _ => unreachable!("three ordinary binary source roles"),
    }
}
fn dense<T: PcuCheckedFloat, const N: usize>(
    op: Op,
    policy: Policy,
    range: Range,
) -> Invocation<T> {
    match (op, policy, range) {
        (Op::Add, Policy::IeeeAfterRounding, Range::Reject) => source::add_ieee_reject_0::<T, N>,
        (Op::Add, Policy::IeeeAfterRounding, Range::Clamp) => source::add_ieee_clamp_0::<T, N>,
        (Op::Add, Policy::RejectSubnormalResult, Range::Reject) => {
            source::add_tight_reject_0::<T, N>
        }
        (Op::Add, Policy::RejectSubnormalResult, Range::Clamp) => source::add_tight_clamp_0::<T, N>,
        (Op::Add, Policy::AllowGradualUnderflow, Range::Reject) => {
            source::add_gradual_reject_0::<T, N>
        }
        (Op::Add, Policy::AllowGradualUnderflow, Range::Clamp) => {
            source::add_gradual_clamp_0::<T, N>
        }
        (Op::Sub, Policy::IeeeAfterRounding, Range::Reject) => source::sub_ieee_reject_0::<T, N>,
        (Op::Sub, Policy::IeeeAfterRounding, Range::Clamp) => source::sub_ieee_clamp_0::<T, N>,
        (Op::Sub, Policy::RejectSubnormalResult, Range::Reject) => {
            source::sub_tight_reject_0::<T, N>
        }
        (Op::Sub, Policy::RejectSubnormalResult, Range::Clamp) => source::sub_tight_clamp_0::<T, N>,
        (Op::Sub, Policy::AllowGradualUnderflow, Range::Reject) => {
            source::sub_gradual_reject_0::<T, N>
        }
        (Op::Sub, Policy::AllowGradualUnderflow, Range::Clamp) => {
            source::sub_gradual_clamp_0::<T, N>
        }
        (Op::Mul, Policy::IeeeAfterRounding, Range::Reject) => source::mul_ieee_reject_0::<T, N>,
        (Op::Mul, Policy::IeeeAfterRounding, Range::Clamp) => source::mul_ieee_clamp_0::<T, N>,
        (Op::Mul, Policy::RejectSubnormalResult, Range::Reject) => {
            source::mul_tight_reject_0::<T, N>
        }
        (Op::Mul, Policy::RejectSubnormalResult, Range::Clamp) => source::mul_tight_clamp_0::<T, N>,
        (Op::Mul, Policy::AllowGradualUnderflow, Range::Reject) => {
            source::mul_gradual_reject_0::<T, N>
        }
        (Op::Mul, Policy::AllowGradualUnderflow, Range::Clamp) => {
            source::mul_gradual_clamp_0::<T, N>
        }
        (Op::Div, Policy::IeeeAfterRounding, Range::Reject) => source::div_ieee_reject_0::<T, N>,
        (Op::Div, Policy::IeeeAfterRounding, Range::Clamp) => source::div_ieee_clamp_0::<T, N>,
        (Op::Div, Policy::RejectSubnormalResult, Range::Reject) => {
            source::div_tight_reject_0::<T, N>
        }
        (Op::Div, Policy::RejectSubnormalResult, Range::Clamp) => source::div_tight_clamp_0::<T, N>,
        (Op::Div, Policy::AllowGradualUnderflow, Range::Reject) => {
            source::div_gradual_reject_0::<T, N>
        }
        (Op::Div, Policy::AllowGradualUnderflow, Range::Clamp) => {
            source::div_gradual_clamp_0::<T, N>
        }
    }
}
fn grid<T: PcuCheckedFloat, const N: usize>(op: Op, policy: Policy, range: Range) -> Invocation<T> {
    match (op, policy, range) {
        (Op::Add, Policy::IeeeAfterRounding, Range::Reject) => source::add_ieee_reject_1::<T, N>,
        (Op::Add, Policy::IeeeAfterRounding, Range::Clamp) => source::add_ieee_clamp_1::<T, N>,
        (Op::Add, Policy::RejectSubnormalResult, Range::Reject) => {
            source::add_tight_reject_1::<T, N>
        }
        (Op::Add, Policy::RejectSubnormalResult, Range::Clamp) => source::add_tight_clamp_1::<T, N>,
        (Op::Add, Policy::AllowGradualUnderflow, Range::Reject) => {
            source::add_gradual_reject_1::<T, N>
        }
        (Op::Add, Policy::AllowGradualUnderflow, Range::Clamp) => {
            source::add_gradual_clamp_1::<T, N>
        }
        (Op::Sub, Policy::IeeeAfterRounding, Range::Reject) => source::sub_ieee_reject_1::<T, N>,
        (Op::Sub, Policy::IeeeAfterRounding, Range::Clamp) => source::sub_ieee_clamp_1::<T, N>,
        (Op::Sub, Policy::RejectSubnormalResult, Range::Reject) => {
            source::sub_tight_reject_1::<T, N>
        }
        (Op::Sub, Policy::RejectSubnormalResult, Range::Clamp) => source::sub_tight_clamp_1::<T, N>,
        (Op::Sub, Policy::AllowGradualUnderflow, Range::Reject) => {
            source::sub_gradual_reject_1::<T, N>
        }
        (Op::Sub, Policy::AllowGradualUnderflow, Range::Clamp) => {
            source::sub_gradual_clamp_1::<T, N>
        }
        (Op::Mul, Policy::IeeeAfterRounding, Range::Reject) => source::mul_ieee_reject_1::<T, N>,
        (Op::Mul, Policy::IeeeAfterRounding, Range::Clamp) => source::mul_ieee_clamp_1::<T, N>,
        (Op::Mul, Policy::RejectSubnormalResult, Range::Reject) => {
            source::mul_tight_reject_1::<T, N>
        }
        (Op::Mul, Policy::RejectSubnormalResult, Range::Clamp) => source::mul_tight_clamp_1::<T, N>,
        (Op::Mul, Policy::AllowGradualUnderflow, Range::Reject) => {
            source::mul_gradual_reject_1::<T, N>
        }
        (Op::Mul, Policy::AllowGradualUnderflow, Range::Clamp) => {
            source::mul_gradual_clamp_1::<T, N>
        }
        (Op::Div, Policy::IeeeAfterRounding, Range::Reject) => source::div_ieee_reject_1::<T, N>,
        (Op::Div, Policy::IeeeAfterRounding, Range::Clamp) => source::div_ieee_clamp_1::<T, N>,
        (Op::Div, Policy::RejectSubnormalResult, Range::Reject) => {
            source::div_tight_reject_1::<T, N>
        }
        (Op::Div, Policy::RejectSubnormalResult, Range::Clamp) => source::div_tight_clamp_1::<T, N>,
        (Op::Div, Policy::AllowGradualUnderflow, Range::Reject) => {
            source::div_gradual_reject_1::<T, N>
        }
        (Op::Div, Policy::AllowGradualUnderflow, Range::Clamp) => {
            source::div_gradual_clamp_1::<T, N>
        }
    }
}
fn repeated<T: PcuCheckedFloat, const N: usize>(
    op: Op,
    policy: Policy,
    range: Range,
) -> Invocation<T> {
    match (op, policy, range) {
        (Op::Add, Policy::IeeeAfterRounding, Range::Reject) => source::add_ieee_reject_2::<T, N>,
        (Op::Add, Policy::IeeeAfterRounding, Range::Clamp) => source::add_ieee_clamp_2::<T, N>,
        (Op::Add, Policy::RejectSubnormalResult, Range::Reject) => {
            source::add_tight_reject_2::<T, N>
        }
        (Op::Add, Policy::RejectSubnormalResult, Range::Clamp) => source::add_tight_clamp_2::<T, N>,
        (Op::Add, Policy::AllowGradualUnderflow, Range::Reject) => {
            source::add_gradual_reject_2::<T, N>
        }
        (Op::Add, Policy::AllowGradualUnderflow, Range::Clamp) => {
            source::add_gradual_clamp_2::<T, N>
        }
        (Op::Sub, Policy::IeeeAfterRounding, Range::Reject) => source::sub_ieee_reject_2::<T, N>,
        (Op::Sub, Policy::IeeeAfterRounding, Range::Clamp) => source::sub_ieee_clamp_2::<T, N>,
        (Op::Sub, Policy::RejectSubnormalResult, Range::Reject) => {
            source::sub_tight_reject_2::<T, N>
        }
        (Op::Sub, Policy::RejectSubnormalResult, Range::Clamp) => source::sub_tight_clamp_2::<T, N>,
        (Op::Sub, Policy::AllowGradualUnderflow, Range::Reject) => {
            source::sub_gradual_reject_2::<T, N>
        }
        (Op::Sub, Policy::AllowGradualUnderflow, Range::Clamp) => {
            source::sub_gradual_clamp_2::<T, N>
        }
        (Op::Mul, Policy::IeeeAfterRounding, Range::Reject) => source::mul_ieee_reject_2::<T, N>,
        (Op::Mul, Policy::IeeeAfterRounding, Range::Clamp) => source::mul_ieee_clamp_2::<T, N>,
        (Op::Mul, Policy::RejectSubnormalResult, Range::Reject) => {
            source::mul_tight_reject_2::<T, N>
        }
        (Op::Mul, Policy::RejectSubnormalResult, Range::Clamp) => source::mul_tight_clamp_2::<T, N>,
        (Op::Mul, Policy::AllowGradualUnderflow, Range::Reject) => {
            source::mul_gradual_reject_2::<T, N>
        }
        (Op::Mul, Policy::AllowGradualUnderflow, Range::Clamp) => {
            source::mul_gradual_clamp_2::<T, N>
        }
        (Op::Div, Policy::IeeeAfterRounding, Range::Reject) => source::div_ieee_reject_2::<T, N>,
        (Op::Div, Policy::IeeeAfterRounding, Range::Clamp) => source::div_ieee_clamp_2::<T, N>,
        (Op::Div, Policy::RejectSubnormalResult, Range::Reject) => {
            source::div_tight_reject_2::<T, N>
        }
        (Op::Div, Policy::RejectSubnormalResult, Range::Clamp) => source::div_tight_clamp_2::<T, N>,
        (Op::Div, Policy::AllowGradualUnderflow, Range::Reject) => {
            source::div_gradual_reject_2::<T, N>
        }
        (Op::Div, Policy::AllowGradualUnderflow, Range::Clamp) => {
            source::div_gradual_clamp_2::<T, N>
        }
    }
}
