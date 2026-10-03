//! Handwritten workload body; checked primitives and terminal ABI are shared explicitly.
#[rustfmt::skip]
use crate::{
    MetalBuffer,
    MetalCheckedMapPlan,
    MetalError,
    MetalSession,
};
use super::source;
#[path = "pure/pure.rs"]
mod pure;
#[rustfmt::skip]
use fusion_pcu::{
    PcuDispatchDataOp,
    PcuDispatchKernelIr,
    PcuDispatchOp,
    PcuExecutionFaultKind,
    PcuRangePolicy,
    PcuScalarType,
};

fn execute(
    session: &MetalSession,
    plan: MetalCheckedMapPlan,
    input: &[u8],
    seed: &[u8],
    stage: &[u8],
    output: &[u8],
) -> (Vec<u8>, Vec<u8>) {
    // The genuine source first accesses input, stage, seed, then output.
    assert_eq!(plan.resources().len(), 4);
    let buffers: [MetalBuffer; 4] =
        [input, stage, seed, output].map(|bytes| session.upload_bytes(bytes).unwrap());
    let mut kernel = session.prepare_native_saved_stage_control(plan).unwrap();
    let resources = buffers.each_ref();
    kernel.execute_into(&resources).unwrap();
    assert!(kernel.last_call_may_have_written());
    let mut got_stage = vec![0; stage.len()];
    let mut got_output = vec![0; output.len()];
    buffers[1].read_into_bytes(&mut got_stage).unwrap();
    buffers[3].read_into_bytes(&mut got_output).unwrap();
    (got_stage, got_output)
}

#[test]
#[cfg_attr(
    not(target_os = "macos"),
    ignore = "requires authentic Metal compiler and terminal execution"
)]
fn handwritten_body_keeps_ten_integer_two_float_saved_values_and_tails() {
    let session = MetalSession::open(0).unwrap();
    macro_rules! integer { ($($ty:ty),*) => { $(
        for phase in [0_u8, 1, 2] {
            let values: [$ty; 7] = core::array::from_fn(|index|
                <$ty>::try_from(1_u8 + u8::try_from(index).unwrap() + phase).unwrap());
            let input: Vec<u8> = values.iter().flat_map(|value| value.to_le_bytes()).collect();
            let seed = <$ty>::try_from(2_u8).unwrap().to_le_bytes();
            let sentinel = <$ty>::try_from(19_u8).unwrap().to_le_bytes();
            let stage = sentinel.repeat(10);
            let output = sentinel.repeat(11);
            let expected: Vec<u8> = values.iter().flat_map(|value|
                ((value + 2) * value - value).to_le_bytes()).collect();
            source::integer_ir::<$ty, 7>(&source::integer_bindings::<$ty>()).unwrap().with_ir(|ir| {
                let plan = MetalCheckedMapPlan::assess(ir, <$ty as pcu_facade::PcuScalar>::TYPE).unwrap();
                let (actual_stage, actual_output) = execute(&session, plan, &input, &seed, &stage, &output);
                assert_eq!(&actual_stage[..input.len()], &input);
                assert_eq!(&actual_stage[input.len()..], &stage[input.len()..]);
                assert_eq!(&actual_output[..expected.len()], &expected);
                assert_eq!(&actual_output[expected.len()..], &output[expected.len()..]);
            });
        }
    )* }; }
    integer!(i8, u8, i16, u16, i32, u32, i64, u64, i128, u128);
    macro_rules! floating { ($($ty:ty),*) => { $(
        for phase in [0_u8, 1, 2] {
        let values: [$ty; 7] = core::array::from_fn(|index| <$ty>::from(1_u8 + u8::try_from(index).unwrap() + phase));
        let input: Vec<u8> = values.iter().flat_map(|value| value.to_le_bytes()).collect();
        let seed = <$ty>::from(2_u8).to_le_bytes();
        let sentinel = <$ty>::from(19_u8).to_le_bytes();
        let stage = sentinel.repeat(10);
        let output = sentinel.repeat(11);
        let expected: Vec<u8> = values.iter().flat_map(|value|
            ((value + 2.0) * value + value).to_le_bytes()).collect();
        source::floating_ir::<$ty, 7>(&source::floating_bindings::<$ty>()).unwrap().with_ir(|ir| {
            let plan = MetalCheckedMapPlan::assess(ir, <$ty as pcu_facade::PcuScalar>::TYPE).unwrap();
            let (actual_stage, actual_output) = execute(&session, plan, &input, &seed, &stage, &output);
            assert_eq!(&actual_stage[..input.len()], &input);
            assert_eq!(&actual_stage[input.len()..], &stage[input.len()..]);
            assert_eq!(&actual_output[..expected.len()], &expected);
            assert_eq!(&actual_output[expected.len()..], &output[expected.len()..]);
        });
    } )* }; }
    floating!(f32, f64);
    // The prototype is intentionally not an ordinary facade admission.
    assert!(matches!(
        MetalSession::open(usize::MAX),
        Err(MetalError::Unsupported | MetalError::Runtime(_))
    ));
}

#[test]
#[cfg_attr(
    not(target_os = "macos"),
    ignore = "requires authentic Metal checked-effect execution"
)]
fn handwritten_body_keeps_discarded_faults_local_recovery_and_actual_resources() {
    let session = MetalSession::open(0).unwrap();
    source::integer_ir::<i32, 7>(&source::integer_bindings::<i32>())
        .unwrap()
        .with_ir(|ir| {
            let input = [i32::MAX - 3, 1, 2, 3, 4, 5, 6];
            let bytes: Vec<u8> = input.iter().flat_map(|v| v.to_le_bytes()).collect();
            let input = session.upload_bytes(&bytes).unwrap();
            let stage = session
                .upload_bytes(&19_i32.to_le_bytes().repeat(10))
                .unwrap();
            let seed = session.upload_bytes(&2_i32.to_le_bytes()).unwrap();
            let output = session
                .upload_bytes(&19_i32.to_le_bytes().repeat(11))
                .unwrap();
            let resources = [&input, &stage, &seed, &output];
            let plan = MetalCheckedMapPlan::assess(ir, PcuScalarType::I32).unwrap();
            let mut kernel = session.prepare_native_saved_stage_control(plan).unwrap();
            assert!(
                matches!(kernel.execute_into(&resources), Err(MetalError::Arithmetic(fault))
                if fault.kind == PcuExecutionFaultKind::ArithmeticOverflow
                    && fault.invocation_id == 0 && !fault.recovered)
            );
            assert!(kernel.last_call_may_have_written());
            let short = session
                .upload_bytes(&19_i32.to_le_bytes().repeat(6))
                .unwrap();
            assert_eq!(
                kernel.execute_into(&[&input, &stage, &seed, &short]),
                Err(MetalError::InvalidExtent)
            );
            assert!(!kernel.last_call_may_have_written());
            let mut untouched = vec![0; short.byte_len()];
            short.read_into_bytes(&mut untouched).unwrap();
            assert_eq!(untouched, 19_i32.to_le_bytes().repeat(6));
            assert_eq!(
                kernel.execute_into(&[&input, &input, &seed, &output]),
                Err(MetalError::Unsupported)
            );
            assert!(!kernel.last_call_may_have_written());
            let foreign_session = MetalSession::open(0).unwrap();
            let foreign = foreign_session.upload_bytes(&bytes).unwrap();
            assert_eq!(
                kernel.execute_into(&[&foreign, &stage, &seed, &output]),
                Err(MetalError::ForeignSession)
            );
            assert!(!kernel.last_call_may_have_written());

            let mut ops = ir.ops.to_vec();
            for op in &mut ops {
                if let PcuDispatchOp::Data(PcuDispatchDataOp::CheckedIntegerBinary {
                    range_policy,
                    ..
                }) = op
                {
                    *range_policy = PcuRangePolicy::Clamp;
                }
            }
            let scoped = PcuDispatchKernelIr { ops: &ops, ..*ir };
            let plan = MetalCheckedMapPlan::assess(&scoped, PcuScalarType::I32).unwrap();
            let mut recovered = session.prepare_native_saved_stage_control(plan).unwrap();
            assert!(
                matches!(recovered.execute_into(&resources), Err(MetalError::Arithmetic(fault))
                if fault.kind == PcuExecutionFaultKind::ArithmeticOverflow
                    && fault.invocation_id == 0 && fault.recovered)
            );
            let mut got = vec![0; output.byte_len()];
            output.read_into_bytes(&mut got).unwrap();
            assert_eq!(&got[..4], &3_i32.to_le_bytes());
            assert_eq!(&got[28..], &19_i32.to_le_bytes().repeat(4));
        });
    source::floating_ir::<f64, 7>(&source::floating_bindings::<f64>())
        .unwrap().with_ir(|ir| {
            let input = session.upload_bytes(&2_f64.to_le_bytes().repeat(7)).unwrap();
            let stage = session.upload_bytes(&19_f64.to_le_bytes().repeat(10)).unwrap();
            let zero = session.upload_bytes(&0_f64.to_le_bytes()).unwrap();
            let seed = session.upload_bytes(&2_f64.to_le_bytes()).unwrap();
            let output = session.upload_bytes(&19_f64.to_le_bytes().repeat(11)).unwrap();
            let plan = MetalCheckedMapPlan::assess(ir, PcuScalarType::F64).unwrap();
            let mut kernel = session.prepare_native_saved_stage_control(plan).unwrap();
            assert!(matches!(kernel.execute_into(&[&input, &stage, &zero, &output]), Err(MetalError::Arithmetic(fault))
                if fault.kind == PcuExecutionFaultKind::DivideByZero && fault.invocation_id == 0 && !fault.recovered));
            kernel.execute_into(&[&input, &stage, &seed, &output]).unwrap();
            let mut got = vec![0; output.byte_len()];
            output.read_into_bytes(&mut got).unwrap();
            assert_eq!(&got[..56], &10_f64.to_le_bytes().repeat(7));
            assert_eq!(&got[56..], &19_f64.to_le_bytes().repeat(4));
        });
}
