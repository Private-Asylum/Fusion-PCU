//! Aggregate effects must stay checked, including discarded operations.
#[rustfmt::skip]
use crate::{
    MlxError,
    MlxHostKernelError,
    MlxPreparedDispatchKernel,
    MlxRuntime,
};
use super::super::source;
#[rustfmt::skip]
use fusion_pcu::{
    PcuBindingRef,
    PcuDispatchDataOp,
    PcuDispatchKernelIr,
    PcuDispatchOp,
    PcuExecutionFaultKind,
    PcuHostArgument,
    PcuHostDispatchError,
    PcuHostKernelBackend,
    PcuPreparedHostKernel,
    PcuRangePolicy,
};
fn call_integer(
    kernel: &mut MlxPreparedDispatchKernel,
    targets: [PcuBindingRef; 5],
    input: &[i32],
    stage: &mut [i32],
    output: &mut [i32],
) -> Result<(), MlxHostKernelError> {
    // Caller order is unrelated to declarations or native first-access slots.
    let mut ghost: [i32; 0] = [];
    kernel.call(&mut [
        PcuHostArgument::read_write(targets[4], output),
        PcuHostArgument::read_scalar(targets[2], &2_i32),
        PcuHostArgument::read_write(targets[0], &mut ghost),
        PcuHostArgument::read(targets[1], input),
        PcuHostArgument::read_write(targets[3], stage),
    ])
}

#[test]
#[cfg_attr(
    not(all(target_os = "macos", target_arch = "aarch64")),
    ignore = "requires authentic Mlx atomic host publication"
)]
fn session_aggregate_keeps_dead_checked_fault_and_local_clamp_joint_publication() {
    let runtime = MlxRuntime::load_default().unwrap();
    let session = runtime.open_gpu(0).unwrap();
    let backend = &session;
    source::integer_ir::<i32, 7>(&source::integer_bindings::<i32>()).unwrap().with_ir(|ir| {
        let targets = core::array::from_fn(|slot| ir.bindings[slot].reference());
        let mut reject = backend.prepare_host_kernel(ir).unwrap();

        let MlxPreparedDispatchKernel::Composed(concrete) = &reject else {
            panic!("expected static composed selection")
        };
        assert_eq!(concrete.argument_count(), 5);
        assert_eq!(concrete.plan().requirements(), ir.numerical_requirements);
        let bad = [i32::MAX - 3, 1, 2, 3, 4, 5, 6];
        let good = [1, 2, 3, 4, 5, 6, 7];
        let mut stage = [19; 10];
        let mut output = [19; 11];
        assert!(matches!(call_integer(&mut reject, targets, &bad, &mut stage, &mut output),
            Err(PcuHostDispatchError::Backend(MlxError::Arithmetic(fault)))
            if fault.kind == PcuExecutionFaultKind::ArithmeticOverflow && fault.invocation_id == 0 && !fault.recovered));
        assert!(!reject.last_call_may_have_written());
        assert!(!reject.last_call_completion_uncertain());
        assert_eq!(stage, [19; 10]);
        assert_eq!(output, [19; 11]);
        call_integer(&mut reject, targets, &good, &mut stage, &mut output).unwrap();
        assert!(reject.last_call_may_have_written());
        let mut ops = ir.ops.to_vec();
        for op in &mut ops {
            if let PcuDispatchOp::Data(PcuDispatchDataOp::CheckedIntegerBinary { range_policy, .. }) = op {
                *range_policy = PcuRangePolicy::Clamp;
            }
        }
        let local = PcuDispatchKernelIr { ops: &ops, ..*ir };
        let mut clamp = backend.prepare_host_kernel(&local).unwrap();
        let MlxPreparedDispatchKernel::Composed(concrete) = &clamp else {
            panic!("expected static composed selection")
        };
        assert_eq!(concrete.plan().requirements().range_policy, PcuRangePolicy::Reject);
        assert!(matches!(call_integer(&mut clamp, targets, &bad, &mut stage, &mut output),
            Err(PcuHostDispatchError::Backend(MlxError::Arithmetic(fault)))
            if fault.kind == PcuExecutionFaultKind::ArithmeticOverflow && fault.invocation_id == 0 && fault.recovered));
        assert!(clamp.last_call_may_have_written());
        assert_eq!(&stage[..7], &bad);
        assert_eq!(&stage[7..], &[19; 3]);
        assert_eq!(output[0], 3);
        assert_eq!(&output[7..], &[19; 4]);
        let before_stage = stage;
        let mut short = [19; 6];
        assert!(matches!(call_integer(&mut clamp, targets, &good, &mut stage, &mut short),
            Err(PcuHostDispatchError::BufferTooSmall(_))));
        assert!(!clamp.last_call_may_have_written());
        assert_eq!(stage, before_stage);
        assert_eq!(short, [19; 6]);
        call_integer(&mut clamp, targets, &good, &mut stage, &mut output).unwrap();
        assert_eq!(&output[..7], &[2, 6, 12, 20, 30, 42, 56]);
    });
}
