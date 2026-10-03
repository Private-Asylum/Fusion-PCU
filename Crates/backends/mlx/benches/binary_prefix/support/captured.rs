//! Source IR capture remains cold, with only actual unique resident shapes retained.
#[rustfmt::skip]
use fusion_pcu_mlx::{
    MlxPreparedBinaryHostKernel,
    MlxSession,
};
#[rustfmt::skip]
use pcu_facade::{
    PcuCheckedFloat,
    PcuDispatchFloatBinaryOp as Op,
    PcuDispatchKernelIr,
    PcuFloatUnderflowPolicy as Policy,
    PcuRangePolicy as Range,
};
use super::source;
pub fn prepare<T: PcuCheckedFloat, const N: usize>(
    session: &MlxSession,
    op: Op,
    policy: Policy,
    range: Range,
    profile: usize,
    full: &[usize],
) -> MlxPreparedBinaryHostKernel {
    match profile {
        0 => dense::<T, N>(session, op, policy, range, full),
        1 => grid::<T, N>(session, op, policy, range, full),
        2 => repeated::<T, N>(session, op, policy, range, full),
        _ => unreachable!("three exact binary prefix source schemas"),
    }
}
#[allow(clippy::too_many_lines)] // Exhaustive cold source table retains independent operation/policy specialization.
fn dense<T: PcuCheckedFloat, const N: usize>(
    session: &MlxSession,
    op: Op,
    policy: Policy,
    range: Range,
    full: &[usize],
) -> MlxPreparedBinaryHostKernel {
    let prepare = |ir: &PcuDispatchKernelIr<'_>| {
        session
            .checked_binary_backend()
            .prepare_host_kernel_with_input_extents(ir, full)
    };
    match (op, policy, range) {
        (Op::Add, Policy::IeeeAfterRounding, Range::Reject) => {
            source::add_ieee_reject_0_ir::<T, N>(&source::add_ieee_reject_0_bindings::<T>())
                .unwrap()
                .with_ir(prepare)
        }
        (Op::Add, Policy::IeeeAfterRounding, Range::Clamp) => {
            source::add_ieee_clamp_0_ir::<T, N>(&source::add_ieee_clamp_0_bindings::<T>())
                .unwrap()
                .with_ir(prepare)
        }
        (Op::Add, Policy::RejectSubnormalResult, Range::Reject) => {
            source::add_tight_reject_0_ir::<T, N>(&source::add_tight_reject_0_bindings::<T>())
                .unwrap()
                .with_ir(prepare)
        }
        (Op::Add, Policy::RejectSubnormalResult, Range::Clamp) => {
            source::add_tight_clamp_0_ir::<T, N>(&source::add_tight_clamp_0_bindings::<T>())
                .unwrap()
                .with_ir(prepare)
        }
        (Op::Add, Policy::AllowGradualUnderflow, Range::Reject) => {
            source::add_gradual_reject_0_ir::<T, N>(&source::add_gradual_reject_0_bindings::<T>())
                .unwrap()
                .with_ir(prepare)
        }
        (Op::Add, Policy::AllowGradualUnderflow, Range::Clamp) => {
            source::add_gradual_clamp_0_ir::<T, N>(&source::add_gradual_clamp_0_bindings::<T>())
                .unwrap()
                .with_ir(prepare)
        }
        (Op::Sub, Policy::IeeeAfterRounding, Range::Reject) => {
            source::sub_ieee_reject_0_ir::<T, N>(&source::sub_ieee_reject_0_bindings::<T>())
                .unwrap()
                .with_ir(prepare)
        }
        (Op::Sub, Policy::IeeeAfterRounding, Range::Clamp) => {
            source::sub_ieee_clamp_0_ir::<T, N>(&source::sub_ieee_clamp_0_bindings::<T>())
                .unwrap()
                .with_ir(prepare)
        }
        (Op::Sub, Policy::RejectSubnormalResult, Range::Reject) => {
            source::sub_tight_reject_0_ir::<T, N>(&source::sub_tight_reject_0_bindings::<T>())
                .unwrap()
                .with_ir(prepare)
        }
        (Op::Sub, Policy::RejectSubnormalResult, Range::Clamp) => {
            source::sub_tight_clamp_0_ir::<T, N>(&source::sub_tight_clamp_0_bindings::<T>())
                .unwrap()
                .with_ir(prepare)
        }
        (Op::Sub, Policy::AllowGradualUnderflow, Range::Reject) => {
            source::sub_gradual_reject_0_ir::<T, N>(&source::sub_gradual_reject_0_bindings::<T>())
                .unwrap()
                .with_ir(prepare)
        }
        (Op::Sub, Policy::AllowGradualUnderflow, Range::Clamp) => {
            source::sub_gradual_clamp_0_ir::<T, N>(&source::sub_gradual_clamp_0_bindings::<T>())
                .unwrap()
                .with_ir(prepare)
        }
        (Op::Mul, Policy::IeeeAfterRounding, Range::Reject) => {
            source::mul_ieee_reject_0_ir::<T, N>(&source::mul_ieee_reject_0_bindings::<T>())
                .unwrap()
                .with_ir(prepare)
        }
        (Op::Mul, Policy::IeeeAfterRounding, Range::Clamp) => {
            source::mul_ieee_clamp_0_ir::<T, N>(&source::mul_ieee_clamp_0_bindings::<T>())
                .unwrap()
                .with_ir(prepare)
        }
        (Op::Mul, Policy::RejectSubnormalResult, Range::Reject) => {
            source::mul_tight_reject_0_ir::<T, N>(&source::mul_tight_reject_0_bindings::<T>())
                .unwrap()
                .with_ir(prepare)
        }
        (Op::Mul, Policy::RejectSubnormalResult, Range::Clamp) => {
            source::mul_tight_clamp_0_ir::<T, N>(&source::mul_tight_clamp_0_bindings::<T>())
                .unwrap()
                .with_ir(prepare)
        }
        (Op::Mul, Policy::AllowGradualUnderflow, Range::Reject) => {
            source::mul_gradual_reject_0_ir::<T, N>(&source::mul_gradual_reject_0_bindings::<T>())
                .unwrap()
                .with_ir(prepare)
        }
        (Op::Mul, Policy::AllowGradualUnderflow, Range::Clamp) => {
            source::mul_gradual_clamp_0_ir::<T, N>(&source::mul_gradual_clamp_0_bindings::<T>())
                .unwrap()
                .with_ir(prepare)
        }
        (Op::Div, Policy::IeeeAfterRounding, Range::Reject) => {
            source::div_ieee_reject_0_ir::<T, N>(&source::div_ieee_reject_0_bindings::<T>())
                .unwrap()
                .with_ir(prepare)
        }
        (Op::Div, Policy::IeeeAfterRounding, Range::Clamp) => {
            source::div_ieee_clamp_0_ir::<T, N>(&source::div_ieee_clamp_0_bindings::<T>())
                .unwrap()
                .with_ir(prepare)
        }
        (Op::Div, Policy::RejectSubnormalResult, Range::Reject) => {
            source::div_tight_reject_0_ir::<T, N>(&source::div_tight_reject_0_bindings::<T>())
                .unwrap()
                .with_ir(prepare)
        }
        (Op::Div, Policy::RejectSubnormalResult, Range::Clamp) => {
            source::div_tight_clamp_0_ir::<T, N>(&source::div_tight_clamp_0_bindings::<T>())
                .unwrap()
                .with_ir(prepare)
        }
        (Op::Div, Policy::AllowGradualUnderflow, Range::Reject) => {
            source::div_gradual_reject_0_ir::<T, N>(&source::div_gradual_reject_0_bindings::<T>())
                .unwrap()
                .with_ir(prepare)
        }
        (Op::Div, Policy::AllowGradualUnderflow, Range::Clamp) => {
            source::div_gradual_clamp_0_ir::<T, N>(&source::div_gradual_clamp_0_bindings::<T>())
                .unwrap()
                .with_ir(prepare)
        }
    }
    .unwrap()
}
#[allow(clippy::too_many_lines)] // Exhaustive cold source table retains independent operation/policy specialization.
fn grid<T: PcuCheckedFloat, const N: usize>(
    session: &MlxSession,
    op: Op,
    policy: Policy,
    range: Range,
    full: &[usize],
) -> MlxPreparedBinaryHostKernel {
    let prepare = |ir: &PcuDispatchKernelIr<'_>| {
        session
            .checked_binary_backend()
            .prepare_host_kernel_with_input_extents(ir, full)
    };
    match (op, policy, range) {
        (Op::Add, Policy::IeeeAfterRounding, Range::Reject) => {
            source::add_ieee_reject_1_ir::<T, N>(&source::add_ieee_reject_1_bindings::<T>())
                .unwrap()
                .with_ir(prepare)
        }
        (Op::Add, Policy::IeeeAfterRounding, Range::Clamp) => {
            source::add_ieee_clamp_1_ir::<T, N>(&source::add_ieee_clamp_1_bindings::<T>())
                .unwrap()
                .with_ir(prepare)
        }
        (Op::Add, Policy::RejectSubnormalResult, Range::Reject) => {
            source::add_tight_reject_1_ir::<T, N>(&source::add_tight_reject_1_bindings::<T>())
                .unwrap()
                .with_ir(prepare)
        }
        (Op::Add, Policy::RejectSubnormalResult, Range::Clamp) => {
            source::add_tight_clamp_1_ir::<T, N>(&source::add_tight_clamp_1_bindings::<T>())
                .unwrap()
                .with_ir(prepare)
        }
        (Op::Add, Policy::AllowGradualUnderflow, Range::Reject) => {
            source::add_gradual_reject_1_ir::<T, N>(&source::add_gradual_reject_1_bindings::<T>())
                .unwrap()
                .with_ir(prepare)
        }
        (Op::Add, Policy::AllowGradualUnderflow, Range::Clamp) => {
            source::add_gradual_clamp_1_ir::<T, N>(&source::add_gradual_clamp_1_bindings::<T>())
                .unwrap()
                .with_ir(prepare)
        }
        (Op::Sub, Policy::IeeeAfterRounding, Range::Reject) => {
            source::sub_ieee_reject_1_ir::<T, N>(&source::sub_ieee_reject_1_bindings::<T>())
                .unwrap()
                .with_ir(prepare)
        }
        (Op::Sub, Policy::IeeeAfterRounding, Range::Clamp) => {
            source::sub_ieee_clamp_1_ir::<T, N>(&source::sub_ieee_clamp_1_bindings::<T>())
                .unwrap()
                .with_ir(prepare)
        }
        (Op::Sub, Policy::RejectSubnormalResult, Range::Reject) => {
            source::sub_tight_reject_1_ir::<T, N>(&source::sub_tight_reject_1_bindings::<T>())
                .unwrap()
                .with_ir(prepare)
        }
        (Op::Sub, Policy::RejectSubnormalResult, Range::Clamp) => {
            source::sub_tight_clamp_1_ir::<T, N>(&source::sub_tight_clamp_1_bindings::<T>())
                .unwrap()
                .with_ir(prepare)
        }
        (Op::Sub, Policy::AllowGradualUnderflow, Range::Reject) => {
            source::sub_gradual_reject_1_ir::<T, N>(&source::sub_gradual_reject_1_bindings::<T>())
                .unwrap()
                .with_ir(prepare)
        }
        (Op::Sub, Policy::AllowGradualUnderflow, Range::Clamp) => {
            source::sub_gradual_clamp_1_ir::<T, N>(&source::sub_gradual_clamp_1_bindings::<T>())
                .unwrap()
                .with_ir(prepare)
        }
        (Op::Mul, Policy::IeeeAfterRounding, Range::Reject) => {
            source::mul_ieee_reject_1_ir::<T, N>(&source::mul_ieee_reject_1_bindings::<T>())
                .unwrap()
                .with_ir(prepare)
        }
        (Op::Mul, Policy::IeeeAfterRounding, Range::Clamp) => {
            source::mul_ieee_clamp_1_ir::<T, N>(&source::mul_ieee_clamp_1_bindings::<T>())
                .unwrap()
                .with_ir(prepare)
        }
        (Op::Mul, Policy::RejectSubnormalResult, Range::Reject) => {
            source::mul_tight_reject_1_ir::<T, N>(&source::mul_tight_reject_1_bindings::<T>())
                .unwrap()
                .with_ir(prepare)
        }
        (Op::Mul, Policy::RejectSubnormalResult, Range::Clamp) => {
            source::mul_tight_clamp_1_ir::<T, N>(&source::mul_tight_clamp_1_bindings::<T>())
                .unwrap()
                .with_ir(prepare)
        }
        (Op::Mul, Policy::AllowGradualUnderflow, Range::Reject) => {
            source::mul_gradual_reject_1_ir::<T, N>(&source::mul_gradual_reject_1_bindings::<T>())
                .unwrap()
                .with_ir(prepare)
        }
        (Op::Mul, Policy::AllowGradualUnderflow, Range::Clamp) => {
            source::mul_gradual_clamp_1_ir::<T, N>(&source::mul_gradual_clamp_1_bindings::<T>())
                .unwrap()
                .with_ir(prepare)
        }
        (Op::Div, Policy::IeeeAfterRounding, Range::Reject) => {
            source::div_ieee_reject_1_ir::<T, N>(&source::div_ieee_reject_1_bindings::<T>())
                .unwrap()
                .with_ir(prepare)
        }
        (Op::Div, Policy::IeeeAfterRounding, Range::Clamp) => {
            source::div_ieee_clamp_1_ir::<T, N>(&source::div_ieee_clamp_1_bindings::<T>())
                .unwrap()
                .with_ir(prepare)
        }
        (Op::Div, Policy::RejectSubnormalResult, Range::Reject) => {
            source::div_tight_reject_1_ir::<T, N>(&source::div_tight_reject_1_bindings::<T>())
                .unwrap()
                .with_ir(prepare)
        }
        (Op::Div, Policy::RejectSubnormalResult, Range::Clamp) => {
            source::div_tight_clamp_1_ir::<T, N>(&source::div_tight_clamp_1_bindings::<T>())
                .unwrap()
                .with_ir(prepare)
        }
        (Op::Div, Policy::AllowGradualUnderflow, Range::Reject) => {
            source::div_gradual_reject_1_ir::<T, N>(&source::div_gradual_reject_1_bindings::<T>())
                .unwrap()
                .with_ir(prepare)
        }
        (Op::Div, Policy::AllowGradualUnderflow, Range::Clamp) => {
            source::div_gradual_clamp_1_ir::<T, N>(&source::div_gradual_clamp_1_bindings::<T>())
                .unwrap()
                .with_ir(prepare)
        }
    }
    .unwrap()
}
#[allow(clippy::too_many_lines)] // Exhaustive cold source table retains independent operation/policy specialization.
fn repeated<T: PcuCheckedFloat, const N: usize>(
    session: &MlxSession,
    op: Op,
    policy: Policy,
    range: Range,
    full: &[usize],
) -> MlxPreparedBinaryHostKernel {
    let prepare = |ir: &PcuDispatchKernelIr<'_>| {
        session
            .checked_binary_backend()
            .prepare_host_kernel_with_input_extents(ir, full)
    };
    match (op, policy, range) {
        (Op::Add, Policy::IeeeAfterRounding, Range::Reject) => {
            source::add_ieee_reject_2_ir::<T, N>(&source::add_ieee_reject_2_bindings::<T>())
                .unwrap()
                .with_ir(prepare)
        }
        (Op::Add, Policy::IeeeAfterRounding, Range::Clamp) => {
            source::add_ieee_clamp_2_ir::<T, N>(&source::add_ieee_clamp_2_bindings::<T>())
                .unwrap()
                .with_ir(prepare)
        }
        (Op::Add, Policy::RejectSubnormalResult, Range::Reject) => {
            source::add_tight_reject_2_ir::<T, N>(&source::add_tight_reject_2_bindings::<T>())
                .unwrap()
                .with_ir(prepare)
        }
        (Op::Add, Policy::RejectSubnormalResult, Range::Clamp) => {
            source::add_tight_clamp_2_ir::<T, N>(&source::add_tight_clamp_2_bindings::<T>())
                .unwrap()
                .with_ir(prepare)
        }
        (Op::Add, Policy::AllowGradualUnderflow, Range::Reject) => {
            source::add_gradual_reject_2_ir::<T, N>(&source::add_gradual_reject_2_bindings::<T>())
                .unwrap()
                .with_ir(prepare)
        }
        (Op::Add, Policy::AllowGradualUnderflow, Range::Clamp) => {
            source::add_gradual_clamp_2_ir::<T, N>(&source::add_gradual_clamp_2_bindings::<T>())
                .unwrap()
                .with_ir(prepare)
        }
        (Op::Sub, Policy::IeeeAfterRounding, Range::Reject) => {
            source::sub_ieee_reject_2_ir::<T, N>(&source::sub_ieee_reject_2_bindings::<T>())
                .unwrap()
                .with_ir(prepare)
        }
        (Op::Sub, Policy::IeeeAfterRounding, Range::Clamp) => {
            source::sub_ieee_clamp_2_ir::<T, N>(&source::sub_ieee_clamp_2_bindings::<T>())
                .unwrap()
                .with_ir(prepare)
        }
        (Op::Sub, Policy::RejectSubnormalResult, Range::Reject) => {
            source::sub_tight_reject_2_ir::<T, N>(&source::sub_tight_reject_2_bindings::<T>())
                .unwrap()
                .with_ir(prepare)
        }
        (Op::Sub, Policy::RejectSubnormalResult, Range::Clamp) => {
            source::sub_tight_clamp_2_ir::<T, N>(&source::sub_tight_clamp_2_bindings::<T>())
                .unwrap()
                .with_ir(prepare)
        }
        (Op::Sub, Policy::AllowGradualUnderflow, Range::Reject) => {
            source::sub_gradual_reject_2_ir::<T, N>(&source::sub_gradual_reject_2_bindings::<T>())
                .unwrap()
                .with_ir(prepare)
        }
        (Op::Sub, Policy::AllowGradualUnderflow, Range::Clamp) => {
            source::sub_gradual_clamp_2_ir::<T, N>(&source::sub_gradual_clamp_2_bindings::<T>())
                .unwrap()
                .with_ir(prepare)
        }
        (Op::Mul, Policy::IeeeAfterRounding, Range::Reject) => {
            source::mul_ieee_reject_2_ir::<T, N>(&source::mul_ieee_reject_2_bindings::<T>())
                .unwrap()
                .with_ir(prepare)
        }
        (Op::Mul, Policy::IeeeAfterRounding, Range::Clamp) => {
            source::mul_ieee_clamp_2_ir::<T, N>(&source::mul_ieee_clamp_2_bindings::<T>())
                .unwrap()
                .with_ir(prepare)
        }
        (Op::Mul, Policy::RejectSubnormalResult, Range::Reject) => {
            source::mul_tight_reject_2_ir::<T, N>(&source::mul_tight_reject_2_bindings::<T>())
                .unwrap()
                .with_ir(prepare)
        }
        (Op::Mul, Policy::RejectSubnormalResult, Range::Clamp) => {
            source::mul_tight_clamp_2_ir::<T, N>(&source::mul_tight_clamp_2_bindings::<T>())
                .unwrap()
                .with_ir(prepare)
        }
        (Op::Mul, Policy::AllowGradualUnderflow, Range::Reject) => {
            source::mul_gradual_reject_2_ir::<T, N>(&source::mul_gradual_reject_2_bindings::<T>())
                .unwrap()
                .with_ir(prepare)
        }
        (Op::Mul, Policy::AllowGradualUnderflow, Range::Clamp) => {
            source::mul_gradual_clamp_2_ir::<T, N>(&source::mul_gradual_clamp_2_bindings::<T>())
                .unwrap()
                .with_ir(prepare)
        }
        (Op::Div, Policy::IeeeAfterRounding, Range::Reject) => {
            source::div_ieee_reject_2_ir::<T, N>(&source::div_ieee_reject_2_bindings::<T>())
                .unwrap()
                .with_ir(prepare)
        }
        (Op::Div, Policy::IeeeAfterRounding, Range::Clamp) => {
            source::div_ieee_clamp_2_ir::<T, N>(&source::div_ieee_clamp_2_bindings::<T>())
                .unwrap()
                .with_ir(prepare)
        }
        (Op::Div, Policy::RejectSubnormalResult, Range::Reject) => {
            source::div_tight_reject_2_ir::<T, N>(&source::div_tight_reject_2_bindings::<T>())
                .unwrap()
                .with_ir(prepare)
        }
        (Op::Div, Policy::RejectSubnormalResult, Range::Clamp) => {
            source::div_tight_clamp_2_ir::<T, N>(&source::div_tight_clamp_2_bindings::<T>())
                .unwrap()
                .with_ir(prepare)
        }
        (Op::Div, Policy::AllowGradualUnderflow, Range::Reject) => {
            source::div_gradual_reject_2_ir::<T, N>(&source::div_gradual_reject_2_bindings::<T>())
                .unwrap()
                .with_ir(prepare)
        }
        (Op::Div, Policy::AllowGradualUnderflow, Range::Clamp) => {
            source::div_gradual_clamp_2_ir::<T, N>(&source::div_gradual_clamp_2_bindings::<T>())
                .unwrap()
                .with_ir(prepare)
        }
    }
    .unwrap()
}
