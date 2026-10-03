//! Joint owned validation and terminal checked outcomes precede caller interpretation.
use super::open;
use super::source;
#[rustfmt::skip]
use crate::{
    MetalMemoryResource,
    MetalOwnedDispatchBackend,
    MetalOwnedResource,
};
#[rustfmt::skip]
use fusion_pcu::{
    PcuCompletionOutcome,
    PcuDispatchDataOp,
    PcuDispatchKernelIr,
    PcuDispatchOp,
    PcuDispatchSubmission,
    PcuExecutionFaultKind,
    PcuInvocationParameters,
    PcuInvocationShape,
    PcuMemoryPoolId,
    PcuOwnedBinding,
    PcuOwnedCompletion,
    PcuOwnedDispatchBackend,
    PcuOwnedDispatchMemorySession,
    PcuPreparedOwnedDispatch,
    PcuRangePolicy,
};
fn bindings(
    backend: &MetalOwnedDispatchBackend,
    ir: &PcuDispatchKernelIr<'_>,
    actual: [&MetalMemoryResource; 4],
) -> Vec<PcuOwnedBinding<MetalOwnedResource>> {
    actual
        .iter()
        .enumerate()
        .map(|(slot, resource)| {
            let declared = &ir.bindings[slot + 1];
            backend
                .bind(
                    declared.reference(),
                    declared.access,
                    declared.binding_type,
                    resource,
                )
                .unwrap()
        })
        .collect()
}
#[test]
#[cfg_attr(
    not(target_os = "macos"),
    ignore = "requires real owned joint preflight and terminal fault outcomes"
)]
fn dead_fault_recovered_outputs_short_second_rollback_and_retry_remain_explicit() {
    let backend = open();
    let pool = PcuMemoryPoolId(121);
    source::integer_ir::<i32, 7>(&source::integer_bindings::<i32>()).unwrap().with_ir(|ir| {
        let submission = |kernel| PcuDispatchSubmission {
            kernel,
            shape: PcuInvocationShape::invocations(core::num::NonZeroU32::new(7).unwrap()),
        };
        let reject = backend.prepare_dispatch_owned_direct(submission(ir), PcuInvocationParameters::empty()).unwrap();
        let bad = backend.upload_buffer(pool, &[i32::MAX - 3, 1, 2, 3, 4, 5, 6]).unwrap();
        let good = backend.upload_buffer(pool, &[1, 2, 3, 4, 5, 6, 7]).unwrap();
        let seed = backend.upload_buffer(pool, &[2_i32]).unwrap();
        let stage = backend.upload_buffer(pool, &[19_i32; 10]).unwrap();
        let output = backend.upload_buffer(pool, &[19_i32; 11]).unwrap();
        let actual = |input: &MetalMemoryResource, destination: &MetalMemoryResource| bindings(&backend, ir, [input, seed.resource(), stage.resource(), destination]);
        let mut fatal = reject.submit_owned_direct(actual(bad.resource(), output.resource())).unwrap();
        assert!(matches!(fatal.wait().unwrap(), PcuCompletionOutcome::Fault(fault)
            if fault.kind == PcuExecutionFaultKind::ArithmeticOverflow && fault.invocation_id == 0 && !fault.recovered));
        // Existing resident contents may have been written on fatal completion; no rollback promise.
        let mut retry = reject.submit_owned_direct(actual(good.resource(), output.resource())).unwrap();
        assert_eq!(retry.wait().unwrap(), PcuCompletionOutcome::Succeeded);
        let mut stage_host = [19; 10];
        let mut output_host = [19; 11];
        backend.download_buffer(pool, &stage, &mut stage_host).unwrap();
        backend.download_buffer(pool, &output, &mut output_host).unwrap();
        assert_eq!(stage_host, [1, 2, 3, 4, 5, 6, 7, 19, 19, 19]);
        assert_eq!(output_host, [2, 6, 12, 20, 30, 42, 56, 19, 19, 19, 19]);
        let short = backend.upload_buffer(pool, &[19_i32; 6]).unwrap();
        assert!(reject.submit_owned_direct(actual(good.resource(), short.resource())).is_err());
        let mut after = [19; 10];
        backend.download_buffer(pool, &stage, &mut after).unwrap();
        assert_eq!(after, stage_host);
        let mut ops = ir.ops.to_vec();
        for instruction in &mut ops {
            if let PcuDispatchOp::Data(PcuDispatchDataOp::CheckedIntegerBinary { range_policy, .. }) = instruction {
                *range_policy = PcuRangePolicy::Clamp;
            }
        }
        let scoped = PcuDispatchKernelIr { ops: &ops, ..*ir };
        assert_eq!(scoped.numerical_requirements.range_policy, PcuRangePolicy::Reject);
        let clamp = backend.prepare_dispatch_owned_direct(submission(&scoped), PcuInvocationParameters::empty()).unwrap();
        let mut recovered = clamp.submit_owned_direct(actual(bad.resource(), output.resource())).unwrap();
        assert!(matches!(recovered.wait().unwrap(), PcuCompletionOutcome::Fault(fault)
            if fault.kind == PcuExecutionFaultKind::ArithmeticOverflow && fault.invocation_id == 0 && fault.recovered));
        backend.download_buffer(pool, &output, &mut output_host).unwrap();
        assert_eq!(output_host[0], 3);
        assert_eq!(&output_host[7..], &[19; 4]);
    });
}
