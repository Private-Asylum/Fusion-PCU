//! A resident writer keeps possible-write disposition while a host sibling stays atomic.
#[rustfmt::skip]
use crate::{
    MetalComposedHostBackend,
    MetalError,
    MetalHostKernelError,
    MetalMemoryResource,
    MetalMixedHostArgument,
    MetalPreparedCheckedMapHostKernel,
    MetalSession,
};
#[rustfmt::skip]
use fusion_pcu::{
    PcuBindingRef,
    PcuDeviceArgument,
    PcuDeviceBuffer,
    PcuDispatchDataOp,
    PcuDispatchKernelIr,
    PcuDispatchOp,
    PcuExecutionFaultKind,
    PcuHostArgument,
    PcuHostDispatchError,
    PcuHostKernelBackend,
    PcuMemoryPoolId,
    PcuRangePolicy,
};
#[rustfmt::skip]
use super::{
    bytes,
    owner,
    read,
    source,
};
fn prepare(
    backend: &MetalComposedHostBackend,
    ir: &PcuDispatchKernelIr<'_>,
    clamp: bool,
) -> MetalPreparedCheckedMapHostKernel {
    let mut ops = ir.ops.to_vec();
    for op in &mut ops {
        if clamp
            && let PcuDispatchOp::Data(PcuDispatchDataOp::CheckedIntegerBinary {
                range_policy, ..
            }) = op
        {
            *range_policy = PcuRangePolicy::Clamp;
        }
    }
    backend
        .prepare_host_kernel(&PcuDispatchKernelIr { ops: &ops, ..*ir })
        .unwrap()
}
fn call(
    kernel: &mut MetalPreparedCheckedMapHostKernel,
    targets: [PcuBindingRef; 5],
    input: &PcuDeviceBuffer<i32, MetalMemoryResource>,
    stage: &mut PcuDeviceBuffer<i32, MetalMemoryResource>,
    output: &mut [i32],
) -> Result<(), MetalHostKernelError> {
    let mut ghost: [i32; 0] = [];
    kernel.call_mixed(&mut [
        MetalMixedHostArgument::Host(PcuHostArgument::read_write(targets[4], output)),
        MetalMixedHostArgument::Host(PcuHostArgument::read_scalar(targets[2], &2_i32)),
        MetalMixedHostArgument::Host(PcuHostArgument::read_write(targets[0], &mut ghost)),
        MetalMixedHostArgument::Resident(PcuDeviceArgument::read(targets[1], input)),
        MetalMixedHostArgument::Resident(PcuDeviceArgument::read_write(targets[3], stage)),
    ])
}
#[test]
#[cfg_attr(
    not(target_os = "macos"),
    ignore = "requires authentic Metal checked mixed disposition"
)]
fn fatal_possible_resident_write_keeps_host_sibling_unchanged_and_clamp_publishes_useful_pair() {
    let session = MetalSession::open(0).unwrap();
    let foreign = MetalSession::open(0).unwrap();
    let backend = session.composed_host_backend();
    source::integer_ir::<i32, 7>(&source::integer_bindings::<i32>()).unwrap().with_ir(|ir| {
        let targets = core::array::from_fn(|slot| ir.bindings[slot].reference());
        let mut provider = session.memory_provider(PcuMemoryPoolId(121));
        let mut other = foreign.memory_provider(PcuMemoryPoolId(121));
        let bad = [i32::MAX - 3, 1, 2, 3, 4, 5, 6];
        let bad_owner = owner(&mut provider, &bad);
        let good_owner = owner(&mut provider, &[1, 2, 3, 4, 5, 6, 7]);
        let foreign_owner = owner(&mut other, &[1, 2, 3, 4, 5, 6, 7]);
        let mut stage = owner(&mut provider, &[19; 10]);
        let mut output = [19; 11];
        let mut reject = prepare(&backend, ir, false);
        assert!(matches!(call(&mut reject, targets, &bad_owner, &mut stage, &mut output),
            Err(PcuHostDispatchError::Backend(MetalError::Arithmetic(fault)))
            if !fault.recovered && fault.kind == PcuExecutionFaultKind::ArithmeticOverflow && fault.invocation_id == 0));
        assert!(reject.last_call_may_have_written());
        assert!(!reject.last_call_completion_uncertain());
        assert_eq!(output, [19; 11]);
        // No resident rollback assumption: this is a raw, known-terminal possible-write resource.
        call(&mut reject, targets, &good_owner, &mut stage, &mut output).unwrap();
        assert_eq!(&output[..7], &[2, 6, 12, 20, 30, 42, 56]);
        assert_eq!(&output[7..], &[19; 4]);
        let mut clamp = prepare(&backend, ir, true);
        assert!(matches!(call(&mut clamp, targets, &bad_owner, &mut stage, &mut output),
            Err(PcuHostDispatchError::Backend(MetalError::Arithmetic(fault)))
            if fault.recovered && fault.kind == PcuExecutionFaultKind::ArithmeticOverflow && fault.invocation_id == 0));
        assert!(clamp.last_call_may_have_written());
        assert_eq!(output[0], 3);
        assert_eq!(&output[7..], &[19; 4]);
        let got_stage = read(&mut provider, &stage);
        assert_eq!(&got_stage[..28], &bytes(&bad));
        assert_eq!(&got_stage[28..], &bytes(&[19_i32; 3]));
        let old_output = output;
        assert!(matches!(call(&mut reject, targets, &foreign_owner, &mut stage, &mut output),
            Err(PcuHostDispatchError::Backend(MetalError::ForeignSession))));
        assert!(!reject.last_call_may_have_written());
        assert_eq!(output, old_output);
        assert_eq!(read(&mut provider, &stage), got_stage);
        let mut short = [19; 6];
        assert!(matches!(call(&mut reject, targets, &good_owner, &mut stage, &mut short),
            Err(PcuHostDispatchError::BufferTooSmall(_))));
        assert!(!reject.last_call_may_have_written());
        assert_eq!(read(&mut provider, &stage), got_stage);
        assert_eq!(short, [19; 6]);
        call(&mut reject, targets, &good_owner, &mut stage, &mut output).unwrap();
        assert_eq!(&output[..7], &[2, 6, 12, 20, 30, 42, 56]);
    });
}
