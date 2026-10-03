//! Actual ordinary source function pointer is selected cold; warm calls perform no source lowering.
#[rustfmt::skip]
use pcu_facade::{
    PcuCheckedFloat,
    PcuDispatchFloatUnaryOp as Op,
    PcuFloatUnderflowPolicy as Policy,
    PcuRangePolicy as Range,
    PcuTensor,
    PcuExecutionError,
};
use super::source;
pub type Invocation<T> = fn(&PcuTensor<T>, &mut [T]) -> Result<(), PcuExecutionError>;
pub fn select<T: PcuCheckedFloat, const N: usize>(
    op: Op,
    policy: Policy,
    range: Range,
    profile: usize,
) -> Invocation<T> {
    match profile {
        0 => dense::<T, N>(op, policy, range),
        1 => grid::<T, N>(op, policy, range),
        2 => broadcast::<T, N>(op, policy, range),
        _ => unreachable!("three ordinary unary source roles"),
    }
}
fn dense<T: PcuCheckedFloat, const N: usize>(
    op: Op,
    policy: Policy,
    range: Range,
) -> Invocation<T> {
    match (op, policy, range) {
        (Op::Neg, Policy::IeeeAfterRounding, Range::Reject) => {
            source::neg_ieeeafterrounding_reject_0::<T, N>
        }
        (Op::Neg, Policy::IeeeAfterRounding, Range::Clamp) => {
            source::neg_ieeeafterrounding_clamp_0::<T, N>
        }
        (Op::Neg, Policy::AllowGradualUnderflow, Range::Reject) => {
            source::neg_allowgradualunderflow_reject_0::<T, N>
        }
        (Op::Neg, Policy::AllowGradualUnderflow, Range::Clamp) => {
            source::neg_allowgradualunderflow_clamp_0::<T, N>
        }
        (Op::Neg, Policy::RejectSubnormalResult, Range::Reject) => {
            source::neg_rejectsubnormalresult_reject_0::<T, N>
        }
        (Op::Neg, Policy::RejectSubnormalResult, Range::Clamp) => {
            source::neg_rejectsubnormalresult_clamp_0::<T, N>
        }
        (Op::Relu, Policy::IeeeAfterRounding, Range::Reject) => {
            source::relu_ieeeafterrounding_reject_0::<T, N>
        }
        (Op::Relu, Policy::IeeeAfterRounding, Range::Clamp) => {
            source::relu_ieeeafterrounding_clamp_0::<T, N>
        }
        (Op::Relu, Policy::AllowGradualUnderflow, Range::Reject) => {
            source::relu_allowgradualunderflow_reject_0::<T, N>
        }
        (Op::Relu, Policy::AllowGradualUnderflow, Range::Clamp) => {
            source::relu_allowgradualunderflow_clamp_0::<T, N>
        }
        (Op::Relu, Policy::RejectSubnormalResult, Range::Reject) => {
            source::relu_rejectsubnormalresult_reject_0::<T, N>
        }
        (Op::Relu, Policy::RejectSubnormalResult, Range::Clamp) => {
            source::relu_rejectsubnormalresult_clamp_0::<T, N>
        }
    }
}
fn grid<T: PcuCheckedFloat, const N: usize>(op: Op, policy: Policy, range: Range) -> Invocation<T> {
    match (op, policy, range) {
        (Op::Neg, Policy::IeeeAfterRounding, Range::Reject) => {
            source::neg_ieeeafterrounding_reject_1::<T, N>
        }
        (Op::Neg, Policy::IeeeAfterRounding, Range::Clamp) => {
            source::neg_ieeeafterrounding_clamp_1::<T, N>
        }
        (Op::Neg, Policy::AllowGradualUnderflow, Range::Reject) => {
            source::neg_allowgradualunderflow_reject_1::<T, N>
        }
        (Op::Neg, Policy::AllowGradualUnderflow, Range::Clamp) => {
            source::neg_allowgradualunderflow_clamp_1::<T, N>
        }
        (Op::Neg, Policy::RejectSubnormalResult, Range::Reject) => {
            source::neg_rejectsubnormalresult_reject_1::<T, N>
        }
        (Op::Neg, Policy::RejectSubnormalResult, Range::Clamp) => {
            source::neg_rejectsubnormalresult_clamp_1::<T, N>
        }
        (Op::Relu, Policy::IeeeAfterRounding, Range::Reject) => {
            source::relu_ieeeafterrounding_reject_1::<T, N>
        }
        (Op::Relu, Policy::IeeeAfterRounding, Range::Clamp) => {
            source::relu_ieeeafterrounding_clamp_1::<T, N>
        }
        (Op::Relu, Policy::AllowGradualUnderflow, Range::Reject) => {
            source::relu_allowgradualunderflow_reject_1::<T, N>
        }
        (Op::Relu, Policy::AllowGradualUnderflow, Range::Clamp) => {
            source::relu_allowgradualunderflow_clamp_1::<T, N>
        }
        (Op::Relu, Policy::RejectSubnormalResult, Range::Reject) => {
            source::relu_rejectsubnormalresult_reject_1::<T, N>
        }
        (Op::Relu, Policy::RejectSubnormalResult, Range::Clamp) => {
            source::relu_rejectsubnormalresult_clamp_1::<T, N>
        }
    }
}
fn broadcast<T: PcuCheckedFloat, const N: usize>(
    op: Op,
    policy: Policy,
    range: Range,
) -> Invocation<T> {
    match (op, policy, range) {
        (Op::Neg, Policy::IeeeAfterRounding, Range::Reject) => {
            source::neg_ieeeafterrounding_reject_2::<T, N>
        }
        (Op::Neg, Policy::IeeeAfterRounding, Range::Clamp) => {
            source::neg_ieeeafterrounding_clamp_2::<T, N>
        }
        (Op::Neg, Policy::AllowGradualUnderflow, Range::Reject) => {
            source::neg_allowgradualunderflow_reject_2::<T, N>
        }
        (Op::Neg, Policy::AllowGradualUnderflow, Range::Clamp) => {
            source::neg_allowgradualunderflow_clamp_2::<T, N>
        }
        (Op::Neg, Policy::RejectSubnormalResult, Range::Reject) => {
            source::neg_rejectsubnormalresult_reject_2::<T, N>
        }
        (Op::Neg, Policy::RejectSubnormalResult, Range::Clamp) => {
            source::neg_rejectsubnormalresult_clamp_2::<T, N>
        }
        (Op::Relu, Policy::IeeeAfterRounding, Range::Reject) => {
            source::relu_ieeeafterrounding_reject_2::<T, N>
        }
        (Op::Relu, Policy::IeeeAfterRounding, Range::Clamp) => {
            source::relu_ieeeafterrounding_clamp_2::<T, N>
        }
        (Op::Relu, Policy::AllowGradualUnderflow, Range::Reject) => {
            source::relu_allowgradualunderflow_reject_2::<T, N>
        }
        (Op::Relu, Policy::AllowGradualUnderflow, Range::Clamp) => {
            source::relu_allowgradualunderflow_clamp_2::<T, N>
        }
        (Op::Relu, Policy::RejectSubnormalResult, Range::Reject) => {
            source::relu_rejectsubnormalresult_reject_2::<T, N>
        }
        (Op::Relu, Policy::RejectSubnormalResult, Range::Clamp) => {
            source::relu_rejectsubnormalresult_clamp_2::<T, N>
        }
    }
}
