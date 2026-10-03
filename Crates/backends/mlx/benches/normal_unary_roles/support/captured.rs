//! Exact source specialization: no warm lowering or request mutation.
#[rustfmt::skip]
use fusion_pcu_mlx::{
    MlxPreparedHostKernel,
    MlxSession,
};
#[rustfmt::skip]
use pcu_facade::{
    PcuCheckedFloat,
    PcuDispatchFloatUnaryOp as Op,
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
    full: usize,
) -> MlxPreparedHostKernel {
    match profile {
        0 => dense::<T, N>(session, op, policy, range, full),
        1 => grid::<T, N>(session, op, policy, range, full),
        2 => broadcast::<T, N>(session, op, policy, range, full),
        _ => unreachable!("three exact unary prefix source schemas"),
    }
}

fn dense<T: PcuCheckedFloat, const N: usize>(
    session: &MlxSession,
    op: Op,
    policy: Policy,
    range: Range,
    full: usize,
) -> MlxPreparedHostKernel {
    let prepare = |ir: &PcuDispatchKernelIr<'_>| {
        let request = *ir;

        session.prepare_unary_host_kernel_with_input_extents(&request, &[full])
    };
    match (op, policy, range) {
        (Op::Neg, Policy::IeeeAfterRounding, Range::Reject) => {
            source::neg_ieeeafterrounding_reject_0_ir::<T, N>(
                &source::neg_ieeeafterrounding_reject_0_bindings::<T>(),
            )
            .unwrap()
            .with_ir(prepare)
        }
        (Op::Neg, Policy::IeeeAfterRounding, Range::Clamp) => {
            source::neg_ieeeafterrounding_clamp_0_ir::<T, N>(
                &source::neg_ieeeafterrounding_clamp_0_bindings::<T>(),
            )
            .unwrap()
            .with_ir(prepare)
        }
        (Op::Neg, Policy::AllowGradualUnderflow, Range::Reject) => {
            source::neg_allowgradualunderflow_reject_0_ir::<T, N>(
                &source::neg_allowgradualunderflow_reject_0_bindings::<T>(),
            )
            .unwrap()
            .with_ir(prepare)
        }
        (Op::Neg, Policy::AllowGradualUnderflow, Range::Clamp) => {
            source::neg_allowgradualunderflow_clamp_0_ir::<T, N>(
                &source::neg_allowgradualunderflow_clamp_0_bindings::<T>(),
            )
            .unwrap()
            .with_ir(prepare)
        }
        (Op::Neg, Policy::RejectSubnormalResult, Range::Reject) => {
            source::neg_rejectsubnormalresult_reject_0_ir::<T, N>(
                &source::neg_rejectsubnormalresult_reject_0_bindings::<T>(),
            )
            .unwrap()
            .with_ir(prepare)
        }
        (Op::Neg, Policy::RejectSubnormalResult, Range::Clamp) => {
            source::neg_rejectsubnormalresult_clamp_0_ir::<T, N>(
                &source::neg_rejectsubnormalresult_clamp_0_bindings::<T>(),
            )
            .unwrap()
            .with_ir(prepare)
        }
        (Op::Relu, Policy::IeeeAfterRounding, Range::Reject) => {
            source::relu_ieeeafterrounding_reject_0_ir::<T, N>(
                &source::relu_ieeeafterrounding_reject_0_bindings::<T>(),
            )
            .unwrap()
            .with_ir(prepare)
        }
        (Op::Relu, Policy::IeeeAfterRounding, Range::Clamp) => {
            source::relu_ieeeafterrounding_clamp_0_ir::<T, N>(
                &source::relu_ieeeafterrounding_clamp_0_bindings::<T>(),
            )
            .unwrap()
            .with_ir(prepare)
        }
        (Op::Relu, Policy::AllowGradualUnderflow, Range::Reject) => {
            source::relu_allowgradualunderflow_reject_0_ir::<T, N>(
                &source::relu_allowgradualunderflow_reject_0_bindings::<T>(),
            )
            .unwrap()
            .with_ir(prepare)
        }
        (Op::Relu, Policy::AllowGradualUnderflow, Range::Clamp) => {
            source::relu_allowgradualunderflow_clamp_0_ir::<T, N>(
                &source::relu_allowgradualunderflow_clamp_0_bindings::<T>(),
            )
            .unwrap()
            .with_ir(prepare)
        }
        (Op::Relu, Policy::RejectSubnormalResult, Range::Reject) => {
            source::relu_rejectsubnormalresult_reject_0_ir::<T, N>(
                &source::relu_rejectsubnormalresult_reject_0_bindings::<T>(),
            )
            .unwrap()
            .with_ir(prepare)
        }
        (Op::Relu, Policy::RejectSubnormalResult, Range::Clamp) => {
            source::relu_rejectsubnormalresult_clamp_0_ir::<T, N>(
                &source::relu_rejectsubnormalresult_clamp_0_bindings::<T>(),
            )
            .unwrap()
            .with_ir(prepare)
        }
    }
    .unwrap()
}

fn grid<T: PcuCheckedFloat, const N: usize>(
    session: &MlxSession,
    op: Op,
    policy: Policy,
    range: Range,
    full: usize,
) -> MlxPreparedHostKernel {
    let prepare = |ir: &PcuDispatchKernelIr<'_>| {
        let request = *ir;

        session.prepare_unary_host_kernel_with_input_extents(&request, &[full])
    };
    match (op, policy, range) {
        (Op::Neg, Policy::IeeeAfterRounding, Range::Reject) => {
            source::neg_ieeeafterrounding_reject_1_ir::<T, N>(
                &source::neg_ieeeafterrounding_reject_1_bindings::<T>(),
            )
            .unwrap()
            .with_ir(prepare)
        }
        (Op::Neg, Policy::IeeeAfterRounding, Range::Clamp) => {
            source::neg_ieeeafterrounding_clamp_1_ir::<T, N>(
                &source::neg_ieeeafterrounding_clamp_1_bindings::<T>(),
            )
            .unwrap()
            .with_ir(prepare)
        }
        (Op::Neg, Policy::AllowGradualUnderflow, Range::Reject) => {
            source::neg_allowgradualunderflow_reject_1_ir::<T, N>(
                &source::neg_allowgradualunderflow_reject_1_bindings::<T>(),
            )
            .unwrap()
            .with_ir(prepare)
        }
        (Op::Neg, Policy::AllowGradualUnderflow, Range::Clamp) => {
            source::neg_allowgradualunderflow_clamp_1_ir::<T, N>(
                &source::neg_allowgradualunderflow_clamp_1_bindings::<T>(),
            )
            .unwrap()
            .with_ir(prepare)
        }
        (Op::Neg, Policy::RejectSubnormalResult, Range::Reject) => {
            source::neg_rejectsubnormalresult_reject_1_ir::<T, N>(
                &source::neg_rejectsubnormalresult_reject_1_bindings::<T>(),
            )
            .unwrap()
            .with_ir(prepare)
        }
        (Op::Neg, Policy::RejectSubnormalResult, Range::Clamp) => {
            source::neg_rejectsubnormalresult_clamp_1_ir::<T, N>(
                &source::neg_rejectsubnormalresult_clamp_1_bindings::<T>(),
            )
            .unwrap()
            .with_ir(prepare)
        }
        (Op::Relu, Policy::IeeeAfterRounding, Range::Reject) => {
            source::relu_ieeeafterrounding_reject_1_ir::<T, N>(
                &source::relu_ieeeafterrounding_reject_1_bindings::<T>(),
            )
            .unwrap()
            .with_ir(prepare)
        }
        (Op::Relu, Policy::IeeeAfterRounding, Range::Clamp) => {
            source::relu_ieeeafterrounding_clamp_1_ir::<T, N>(
                &source::relu_ieeeafterrounding_clamp_1_bindings::<T>(),
            )
            .unwrap()
            .with_ir(prepare)
        }
        (Op::Relu, Policy::AllowGradualUnderflow, Range::Reject) => {
            source::relu_allowgradualunderflow_reject_1_ir::<T, N>(
                &source::relu_allowgradualunderflow_reject_1_bindings::<T>(),
            )
            .unwrap()
            .with_ir(prepare)
        }
        (Op::Relu, Policy::AllowGradualUnderflow, Range::Clamp) => {
            source::relu_allowgradualunderflow_clamp_1_ir::<T, N>(
                &source::relu_allowgradualunderflow_clamp_1_bindings::<T>(),
            )
            .unwrap()
            .with_ir(prepare)
        }
        (Op::Relu, Policy::RejectSubnormalResult, Range::Reject) => {
            source::relu_rejectsubnormalresult_reject_1_ir::<T, N>(
                &source::relu_rejectsubnormalresult_reject_1_bindings::<T>(),
            )
            .unwrap()
            .with_ir(prepare)
        }
        (Op::Relu, Policy::RejectSubnormalResult, Range::Clamp) => {
            source::relu_rejectsubnormalresult_clamp_1_ir::<T, N>(
                &source::relu_rejectsubnormalresult_clamp_1_bindings::<T>(),
            )
            .unwrap()
            .with_ir(prepare)
        }
    }
    .unwrap()
}

fn broadcast<T: PcuCheckedFloat, const N: usize>(
    session: &MlxSession,
    op: Op,
    policy: Policy,
    range: Range,
    full: usize,
) -> MlxPreparedHostKernel {
    let prepare = |ir: &PcuDispatchKernelIr<'_>| {
        let request = *ir;

        session.prepare_unary_host_kernel_with_input_extents(&request, &[full])
    };
    match (op, policy, range) {
        (Op::Neg, Policy::IeeeAfterRounding, Range::Reject) => {
            source::neg_ieeeafterrounding_reject_2_ir::<T, N>(
                &source::neg_ieeeafterrounding_reject_2_bindings::<T>(),
            )
            .unwrap()
            .with_ir(prepare)
        }
        (Op::Neg, Policy::IeeeAfterRounding, Range::Clamp) => {
            source::neg_ieeeafterrounding_clamp_2_ir::<T, N>(
                &source::neg_ieeeafterrounding_clamp_2_bindings::<T>(),
            )
            .unwrap()
            .with_ir(prepare)
        }
        (Op::Neg, Policy::AllowGradualUnderflow, Range::Reject) => {
            source::neg_allowgradualunderflow_reject_2_ir::<T, N>(
                &source::neg_allowgradualunderflow_reject_2_bindings::<T>(),
            )
            .unwrap()
            .with_ir(prepare)
        }
        (Op::Neg, Policy::AllowGradualUnderflow, Range::Clamp) => {
            source::neg_allowgradualunderflow_clamp_2_ir::<T, N>(
                &source::neg_allowgradualunderflow_clamp_2_bindings::<T>(),
            )
            .unwrap()
            .with_ir(prepare)
        }
        (Op::Neg, Policy::RejectSubnormalResult, Range::Reject) => {
            source::neg_rejectsubnormalresult_reject_2_ir::<T, N>(
                &source::neg_rejectsubnormalresult_reject_2_bindings::<T>(),
            )
            .unwrap()
            .with_ir(prepare)
        }
        (Op::Neg, Policy::RejectSubnormalResult, Range::Clamp) => {
            source::neg_rejectsubnormalresult_clamp_2_ir::<T, N>(
                &source::neg_rejectsubnormalresult_clamp_2_bindings::<T>(),
            )
            .unwrap()
            .with_ir(prepare)
        }
        (Op::Relu, Policy::IeeeAfterRounding, Range::Reject) => {
            source::relu_ieeeafterrounding_reject_2_ir::<T, N>(
                &source::relu_ieeeafterrounding_reject_2_bindings::<T>(),
            )
            .unwrap()
            .with_ir(prepare)
        }
        (Op::Relu, Policy::IeeeAfterRounding, Range::Clamp) => {
            source::relu_ieeeafterrounding_clamp_2_ir::<T, N>(
                &source::relu_ieeeafterrounding_clamp_2_bindings::<T>(),
            )
            .unwrap()
            .with_ir(prepare)
        }
        (Op::Relu, Policy::AllowGradualUnderflow, Range::Reject) => {
            source::relu_allowgradualunderflow_reject_2_ir::<T, N>(
                &source::relu_allowgradualunderflow_reject_2_bindings::<T>(),
            )
            .unwrap()
            .with_ir(prepare)
        }
        (Op::Relu, Policy::AllowGradualUnderflow, Range::Clamp) => {
            source::relu_allowgradualunderflow_clamp_2_ir::<T, N>(
                &source::relu_allowgradualunderflow_clamp_2_bindings::<T>(),
            )
            .unwrap()
            .with_ir(prepare)
        }
        (Op::Relu, Policy::RejectSubnormalResult, Range::Reject) => {
            source::relu_rejectsubnormalresult_reject_2_ir::<T, N>(
                &source::relu_rejectsubnormalresult_reject_2_bindings::<T>(),
            )
            .unwrap()
            .with_ir(prepare)
        }
        (Op::Relu, Policy::RejectSubnormalResult, Range::Clamp) => {
            source::relu_rejectsubnormalresult_clamp_2_ir::<T, N>(
                &source::relu_rejectsubnormalresult_clamp_2_bindings::<T>(),
            )
            .unwrap()
            .with_ir(prepare)
        }
    }
    .unwrap()
}
