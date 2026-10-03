//! Exact integer source specialization is detached and completed cold.
#[rustfmt::skip]
use fusion_pcu_mlx::{
    MlxPreparedIntegerHostKernel,
    MlxSession,
};
#[rustfmt::skip]
use pcu_facade::{
    PcuCheckedInteger,
    PcuDispatchIntegerBinaryOp as Op,
    PcuDispatchKernelIr,
    PcuRangePolicy as Range,
};
use super::source;
pub fn prepare<T: PcuCheckedInteger, const N: usize>(
    session: &MlxSession,
    op: Op,
    range: Range,
    profile: usize,
    full: &[usize],
) -> MlxPreparedIntegerHostKernel {
    let prepare = |ir: &PcuDispatchKernelIr<'_>| {
        session
            .checked_integer_backend()
            .prepare_host_kernel_with_input_extents(ir, full)
    };
    match (profile, op, range) {
        (0, Op::Add, Range::Reject) => {
            source::add_reject_0_ir::<T, N>(&source::add_reject_0_bindings::<T>())
                .unwrap()
                .with_ir(prepare)
        }
        (0, Op::Add, Range::Clamp) => {
            source::add_clamp_0_ir::<T, N>(&source::add_clamp_0_bindings::<T>())
                .unwrap()
                .with_ir(prepare)
        }
        (0, Op::Sub, Range::Reject) => {
            source::sub_reject_0_ir::<T, N>(&source::sub_reject_0_bindings::<T>())
                .unwrap()
                .with_ir(prepare)
        }
        (0, Op::Sub, Range::Clamp) => {
            source::sub_clamp_0_ir::<T, N>(&source::sub_clamp_0_bindings::<T>())
                .unwrap()
                .with_ir(prepare)
        }
        (0, Op::Mul, Range::Reject) => {
            source::mul_reject_0_ir::<T, N>(&source::mul_reject_0_bindings::<T>())
                .unwrap()
                .with_ir(prepare)
        }
        (0, Op::Mul, Range::Clamp) => {
            source::mul_clamp_0_ir::<T, N>(&source::mul_clamp_0_bindings::<T>())
                .unwrap()
                .with_ir(prepare)
        }
        (1, Op::Add, Range::Reject) => {
            source::add_reject_1_ir::<T, N>(&source::add_reject_1_bindings::<T>())
                .unwrap()
                .with_ir(prepare)
        }
        (1, Op::Add, Range::Clamp) => {
            source::add_clamp_1_ir::<T, N>(&source::add_clamp_1_bindings::<T>())
                .unwrap()
                .with_ir(prepare)
        }
        (1, Op::Sub, Range::Reject) => {
            source::sub_reject_1_ir::<T, N>(&source::sub_reject_1_bindings::<T>())
                .unwrap()
                .with_ir(prepare)
        }
        (1, Op::Sub, Range::Clamp) => {
            source::sub_clamp_1_ir::<T, N>(&source::sub_clamp_1_bindings::<T>())
                .unwrap()
                .with_ir(prepare)
        }
        (1, Op::Mul, Range::Reject) => {
            source::mul_reject_1_ir::<T, N>(&source::mul_reject_1_bindings::<T>())
                .unwrap()
                .with_ir(prepare)
        }
        (1, Op::Mul, Range::Clamp) => {
            source::mul_clamp_1_ir::<T, N>(&source::mul_clamp_1_bindings::<T>())
                .unwrap()
                .with_ir(prepare)
        }
        (2, Op::Add, Range::Reject) => {
            source::add_reject_2_ir::<T, N>(&source::add_reject_2_bindings::<T>())
                .unwrap()
                .with_ir(prepare)
        }
        (2, Op::Add, Range::Clamp) => {
            source::add_clamp_2_ir::<T, N>(&source::add_clamp_2_bindings::<T>())
                .unwrap()
                .with_ir(prepare)
        }
        (2, Op::Sub, Range::Reject) => {
            source::sub_reject_2_ir::<T, N>(&source::sub_reject_2_bindings::<T>())
                .unwrap()
                .with_ir(prepare)
        }
        (2, Op::Sub, Range::Clamp) => {
            source::sub_clamp_2_ir::<T, N>(&source::sub_clamp_2_bindings::<T>())
                .unwrap()
                .with_ir(prepare)
        }
        (2, Op::Mul, Range::Reject) => {
            source::mul_reject_2_ir::<T, N>(&source::mul_reject_2_bindings::<T>())
                .unwrap()
                .with_ir(prepare)
        }
        (2, Op::Mul, Range::Clamp) => {
            source::mul_clamp_2_ir::<T, N>(&source::mul_clamp_2_bindings::<T>())
                .unwrap()
                .with_ir(prepare)
        }
        _ => unreachable!("three exact integer prefix source schemas"),
    }
    .unwrap()
}
