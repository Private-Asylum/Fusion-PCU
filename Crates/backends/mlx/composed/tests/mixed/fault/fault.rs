//! Failure preflight and useful recovery never mutate an existing immutable input owner.
#[rustfmt::skip]
use crate::{
    MlxCheckedMapInput,
    MlxError,
    MlxRuntime,
};
#[rustfmt::skip]
use fusion_pcu::{
    PcuDispatchDataOp,
    PcuDispatchKernelIr,
    PcuDispatchOp,
    PcuExecutionFaultKind,
    PcuHostDispatchError,
    PcuHostKernelBackend,
    PcuRangePolicy,
    PcuScalarType,
};
#[rustfmt::skip]
use super::{
    bytes,
    source,
};
#[test]
#[cfg_attr(
    not(all(target_os = "macos", target_arch = "aarch64")),
    ignore = "requires actual MLX mixed preflight and immutable recovery"
)]
fn dead_overflow_recovery_and_foreign_short_duplicate_inputs_keep_old_owners_valid() {
    let runtime = MlxRuntime::load_default().unwrap();
    let session = runtime.open_gpu(0).unwrap();
    let foreign = runtime.open_gpu(0).unwrap();
    source::integer_ir::<i32, 7>(&source::integer_bindings::<i32>()).unwrap().with_ir(|ir| {
        let backend = session.composed_host_backend();
        let mut reject = backend.prepare_host_kernel(ir).unwrap();
        let targets: [_; 2] = core::array::from_fn(|slot| reject.input_bindings()[slot]);
        let bad = [i32::MAX - 3, 1, 2, 3, 4, 5, 6];
        let old = session.upload_transport_bytes(PcuScalarType::I32, 7, &bytes(&bad)).unwrap();
        let wrong = foreign.upload_transport_bytes(PcuScalarType::I32, 7, &bytes(&bad)).unwrap();
        let short = session.upload_transport_bytes(PcuScalarType::I32, 6, &bytes(&bad[..6])).unwrap();
        let seed = bytes(&[2_i32]);
        let seed_arg = MlxCheckedMapInput::HostBytes { target: targets[1], scalar: PcuScalarType::I32, bytes: &seed };
        let input_arg = MlxCheckedMapInput::Resident { target: targets[0], array: &old };
        assert!(matches!(reject.execute_inputs(&[input_arg, seed_arg]),
            Err(PcuHostDispatchError::Backend(MlxError::Arithmetic(fault)))
            if !fault.recovered && fault.kind == PcuExecutionFaultKind::ArithmeticOverflow && fault.invocation_id == 0));
        assert!(!reject.last_call_may_have_written());
        assert!(!reject.last_call_completion_uncertain());
        assert!(matches!(reject.execute_inputs(&[MlxCheckedMapInput::Resident { target: targets[0], array: &wrong }, seed_arg]),
            Err(PcuHostDispatchError::Backend(MlxError::ForeignSession))));
        assert!(matches!(reject.execute_inputs(&[MlxCheckedMapInput::Resident { target: targets[0], array: &short }, seed_arg]),
            Err(PcuHostDispatchError::BufferTooSmall(_))));
        assert!(matches!(reject.execute_inputs(&[input_arg, input_arg, seed_arg]), Err(PcuHostDispatchError::Duplicate(_))));
        assert!(matches!(reject.execute_inputs(&[input_arg]), Err(PcuHostDispatchError::Missing(_))));
        let mut ops = ir.ops.to_vec();
        for op in &mut ops {
            if let PcuDispatchOp::Data(PcuDispatchDataOp::CheckedIntegerBinary { range_policy, .. }) = op { *range_policy = PcuRangePolicy::Clamp; }
        }
        let local = PcuDispatchKernelIr { ops: &ops, ..*ir };
        let mut clamp = backend.prepare_host_kernel(&local).unwrap();
        let (outputs, notice) = clamp.execute_inputs(&[seed_arg, input_arg]).unwrap().into_outputs();
        let fault = notice.unwrap();
        assert!(fault.recovered);
        assert_eq!(fault.kind, PcuExecutionFaultKind::ArithmeticOverflow);
        assert_eq!(fault.invocation_id, 0);
        assert!(!clamp.last_call_may_have_written());
        let [Some(stage), Some(output)] = outputs else { panic!("missing useful private siblings") };
        let mut actual = [0; 28];
        stage.read_bytes_into(&mut actual).unwrap();
        assert_eq!(&actual, bytes(&bad).as_slice());
        output.read_bytes_into(&mut actual).unwrap();
        assert_eq!(&actual[..4], &3_i32.to_le_bytes());
        let mut preserved = [0; 28];
        old.read_bytes_into(&mut preserved).unwrap();
        assert_eq!(&preserved, bytes(&bad).as_slice());
        stage.release().unwrap();
        output.release().unwrap();
        let good = bytes(&[1_i32, 2, 3, 4, 5, 6, 7]);
        let host_good = MlxCheckedMapInput::HostBytes { target: targets[0], scalar: PcuScalarType::I32, bytes: &good };
        let (retry, notice) = reject.execute_inputs(&[seed_arg, host_good]).unwrap().into_outputs();
        assert!(notice.is_none());
        let retry_output = retry[1].as_ref().unwrap();
        retry_output.read_bytes_into(&mut actual).unwrap();
        assert_eq!(actual.as_slice(), bytes(&[2_i32, 6, 12, 20, 30, 42, 56]));
        for array in retry.into_iter().flatten() { array.release().unwrap(); }
        old.release().unwrap();
        wrong.release().unwrap();
        short.release().unwrap();
    });
}
