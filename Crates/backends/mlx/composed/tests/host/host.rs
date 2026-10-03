//! Separate public host boundary: private checked effects precede joint commit.
#[rustfmt::skip]
use crate::{
    MlxError,
    MlxHostKernelError,
    MlxPreparedCheckedMapHostKernel,
    MlxRuntime,
};
use super::source;
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

#[test]
#[cfg_attr(
    not(all(target_os = "macos", target_arch = "aarch64")),
    ignore = "requires authentic Mlx host composition"
)]
fn ten_integer_two_float_host_composition_publishes_both_prefixes_and_preserves_tails() {
    let runtime = MlxRuntime::load_default().unwrap();
    let session = runtime.open_gpu(0).unwrap();
    let backend = session.composed_host_backend();
    macro_rules! integer { ($($ty:ty),*) => { $(
        let mut run = source::integer_prepare::<$ty, 7, _>(&backend).unwrap();
        for phase in [0_u8, 1, 2] {
            let input: [$ty; 9] = core::array::from_fn(|index|
                <$ty>::try_from(1_u8 + u8::try_from(index).unwrap() + phase).unwrap());
            let seed = <$ty>::try_from(2_u8).unwrap();
            let sentinel = <$ty>::try_from(19_u8).unwrap();
            let mut stage = [sentinel; 10];
            let mut output = [sentinel; 11];
            run(&mut [], &input, &seed, &mut stage, &mut output).unwrap();
            assert_eq!(&stage[..7], &input[..7]);
            assert_eq!(&stage[7..], &[sentinel; 3]);
            let expected: [$ty; 7] = core::array::from_fn(|lane|
                (input[lane] + 2) * input[lane] - input[lane]);
            assert_eq!(&output[..7], &expected);
            assert_eq!(&output[7..], &[sentinel; 4]);
            let mut short = [sentinel; 6];
            let old_stage = stage;
            assert!(matches!(run(&mut [], &input, &seed, &mut stage, &mut short),
                Err(PcuHostDispatchError::BufferTooSmall(_))));
            assert_eq!(stage, old_stage);
            assert_eq!(short, [sentinel; 6]);
        }
    )* }; }
    integer!(i8, u8, i16, u16, i32, u32, i64, u64, i128, u128);
    macro_rules! floating { ($($ty:ty),*) => { $(
        let mut run = source::floating_prepare::<$ty, 7, _>(&backend).unwrap();
        for phase in [0_u8, 1, 2] {
            let input: [$ty; 9] = core::array::from_fn(|index|
                <$ty>::from(1_u8 + u8::try_from(index).unwrap() + phase));
            let seed = <$ty>::from(2_u8);
            let sentinel = <$ty>::from(19_u8);
            let mut stage = [sentinel; 10];
            let mut output = [sentinel; 11];
            run(&mut [], &input, &seed, &mut stage, &mut output).unwrap();
            assert_eq!(&stage.map(<$ty>::to_bits)[..7], &input.map(<$ty>::to_bits)[..7]);
            assert_eq!(&stage.map(<$ty>::to_bits)[7..], &[sentinel.to_bits(); 3]);
            let expected: [$ty; 7] = core::array::from_fn(|lane| {
                let value = u16::from(1_u8 + u8::try_from(lane).unwrap() + phase);
                <$ty>::from((value + 2) * value + value)
            });
            assert_eq!(&output.map(<$ty>::to_bits)[..7], &expected.map(<$ty>::to_bits));
            assert_eq!(&output.map(<$ty>::to_bits)[7..], &[sentinel.to_bits(); 4]);
            let old_stage = stage;
            let old_output = output;
            assert!(matches!(run(&mut [], &input, &0.0, &mut stage, &mut output),
                Err(PcuHostDispatchError::Backend(MlxError::Arithmetic(fault)))
                if fault.kind == PcuExecutionFaultKind::DivideByZero && !fault.recovered));
            assert_eq!(stage.map(<$ty>::to_bits), old_stage.map(<$ty>::to_bits));
            assert_eq!(output.map(<$ty>::to_bits), old_output.map(<$ty>::to_bits));
            run(&mut [], &input, &seed, &mut stage, &mut output).unwrap();
            assert_eq!(&output.map(<$ty>::to_bits)[..7], &expected.map(<$ty>::to_bits));
        }
    )* }; }
    floating!(f32, f64);
}

fn call_integer(
    kernel: &mut MlxPreparedCheckedMapHostKernel,
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
fn discarded_checked_fatal_preserves_all_host_outputs_and_local_clamp_publishes_with_notice() {
    let runtime = MlxRuntime::load_default().unwrap();
    let session = runtime.open_gpu(0).unwrap();
    let backend = session.composed_host_backend();
    source::integer_ir::<i32, 7>(&source::integer_bindings::<i32>()).unwrap().with_ir(|ir| {
        let targets = core::array::from_fn(|slot| ir.bindings[slot].reference());
        let mut reject = backend.prepare_host_kernel(ir).unwrap();
        assert_eq!(reject.argument_count(), 5);
        assert_eq!(reject.plan().requirements(), ir.numerical_requirements);
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
        assert_eq!(clamp.plan().requirements().range_policy, PcuRangePolicy::Reject);
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
