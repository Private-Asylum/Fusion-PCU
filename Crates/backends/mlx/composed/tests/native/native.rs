//! Authentic three-sibling prototype; no ordinary factory/host publication inference.
#[rustfmt::skip]
use crate::{
    MlxCheckedMapPlan,
    MlxError,
    MlxRuntime,
    MlxSession,
};
use super::source;
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
    session: &MlxSession,
    plan: MlxCheckedMapPlan,
    input: &[u8],
    seed: &[u8],
    stage: &[u8],
    output: &[u8],
) -> (Vec<u8>, Vec<u8>) {
    let scalar = plan.value_type().scalar_type();
    let prepared = session.prepare_checked_map_plan(plan).unwrap();
    assert_eq!(prepared.prepared_input_element_counts(), &[7, 1]);
    assert_eq!(prepared.output_bindings().len(), 2);
    let input_owner = session.upload_transport_bytes(scalar, 7, input).unwrap();
    let seed_owner = session.upload_transport_bytes(scalar, 1, seed).unwrap();
    let (outputs, fault) = prepared
        .execute(&[&input_owner, &seed_owner])
        .unwrap()
        .into_outputs();
    assert!(fault.is_none());
    let [Some(stage_owner), Some(output_owner)] = outputs else {
        panic!("missing actual private sibling")
    };
    let mut got_stage = stage.to_vec();
    let mut got_output = output.to_vec();
    stage_owner
        .read_bytes_into(&mut got_stage[..input.len()])
        .unwrap();
    output_owner
        .read_bytes_into(&mut got_output[..input.len()])
        .unwrap();
    let mut preserved = vec![0; input.len()];
    input_owner.read_bytes_into(&mut preserved).unwrap();
    assert_eq!(preserved, input);
    stage_owner.release().unwrap();
    output_owner.release().unwrap();
    input_owner.release().unwrap();
    seed_owner.release().unwrap();
    (got_stage, got_output)
}
#[test]
#[ignore = "requires actual MLX three-sibling kernel and checked terminal cleanup"]
fn ten_integer_two_float_native_composition_keeps_saved_values_and_tails() {
    let runtime = MlxRuntime::load_default().unwrap();
    let session = runtime.open_gpu(0).unwrap();
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
                let plan = MlxCheckedMapPlan::assess(ir, <$ty as pcu_facade::PcuScalar>::TYPE).unwrap();
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
        let values: [$ty; 7] = core::array::from_fn(|index| <$ty>::from(1_u8 + u8::try_from(index).unwrap()));
        let input: Vec<u8> = values.iter().flat_map(|value| value.to_le_bytes()).collect();
        let seed = <$ty>::from(2_u8).to_le_bytes();
        let sentinel = <$ty>::from(19_u8).to_le_bytes();
        let stage = sentinel.repeat(10);
        let output = sentinel.repeat(11);
        let expected: Vec<u8> = values.iter().flat_map(|value|
            ((value + 2.0) * value + value).to_le_bytes()).collect();
        source::floating_ir::<$ty, 7>(&source::floating_bindings::<$ty>()).unwrap().with_ir(|ir| {
            let plan = MlxCheckedMapPlan::assess(ir, <$ty as pcu_facade::PcuScalar>::TYPE).unwrap();
            let (actual_stage, actual_output) = execute(&session, plan, &input, &seed, &stage, &output);
            assert_eq!(&actual_stage[..input.len()], &input);
            assert_eq!(&actual_stage[input.len()..], &stage[input.len()..]);
            assert_eq!(&actual_output[..expected.len()], &expected);
            assert_eq!(&actual_output[expected.len()..], &output[expected.len()..]);
        });
    )* }; }
    floating!(f32, f64);
}

#[test]
#[ignore = "requires genuine three-sibling MLX terminal fault and immutable owner proof"]
fn checked_discarded_effects_recovery_and_actual_input_preflight() {
    let runtime = MlxRuntime::load_default().unwrap();
    let session = runtime.open_gpu(0).unwrap();
    let foreign = runtime.open_gpu(0).unwrap();
    source::integer_ir::<i32, 7>(&source::integer_bindings::<i32>())
        .unwrap().with_ir(|ir| {
            let bytes: Vec<u8> = [i32::MAX-3,1,2,3,4,5,6].iter().flat_map(|v|v.to_le_bytes()).collect();
            let input=session.upload_transport_bytes(PcuScalarType::I32,7,&bytes).unwrap();
            let seed=session.upload_transport_bytes(PcuScalarType::I32,1,&2_i32.to_le_bytes()).unwrap();
            let plan=MlxCheckedMapPlan::assess(ir,PcuScalarType::I32).unwrap();
            let prepared=session.prepare_checked_map_plan(plan).unwrap();
            assert!(matches!(prepared.execute(&[&input,&seed]),Err(MlxError::Arithmetic(fault))
                if fault.kind==PcuExecutionFaultKind::ArithmeticOverflow && fault.invocation_id==0 && !fault.recovered));
            let mut retained=vec![0;bytes.len()];
            input.read_bytes_into(&mut retained).unwrap();assert_eq!(retained,bytes);
            let short=session.upload_transport_bytes(PcuScalarType::I32,6,&bytes[..24]).unwrap();
            assert!(matches!(prepared.execute(&[&short,&seed]),Err(MlxError::InvalidExtent)));
            let other=foreign.upload_transport_bytes(PcuScalarType::I32,7,&bytes).unwrap();
            assert!(matches!(prepared.execute(&[&other,&seed]),Err(MlxError::ForeignSession)));
            let mut ops=ir.ops.to_vec();
            for op in &mut ops {
                if let PcuDispatchOp::Data(PcuDispatchDataOp::CheckedIntegerBinary{range_policy,..})=op { *range_policy=PcuRangePolicy::Clamp; }
            }
            let scoped=PcuDispatchKernelIr{ops:&ops,..*ir};
            let plan=MlxCheckedMapPlan::assess(&scoped,PcuScalarType::I32).unwrap();
            let prepared=session.prepare_checked_map_plan(plan).unwrap();
            let (outputs,fault)=prepared.execute(&[&input,&seed]).unwrap().into_outputs();
            assert!(matches!(fault,Some(fault) if fault.kind==PcuExecutionFaultKind::ArithmeticOverflow && fault.invocation_id==0 && fault.recovered));
            let [Some(stage),Some(output)]=outputs else {panic!("missing useful private siblings")};
            let mut got=vec![0;28];output.read_bytes_into(&mut got).unwrap();
            assert_eq!(&got[..4],&3_i32.to_le_bytes());
            input.read_bytes_into(&mut retained).unwrap();assert_eq!(retained,bytes);
            stage.release().unwrap();output.release().unwrap();input.release().unwrap();seed.release().unwrap();short.release().unwrap();other.release().unwrap();
        });
    source::floating_ir::<f64,7>(&source::floating_bindings::<f64>())
        .unwrap().with_ir(|ir| {
            let input=session.upload_transport_bytes(PcuScalarType::F64,7,&2_f64.to_le_bytes().repeat(7)).unwrap();
            let zero=session.upload_transport_bytes(PcuScalarType::F64,1,&0_f64.to_le_bytes()).unwrap();
            let seed=session.upload_transport_bytes(PcuScalarType::F64,1,&2_f64.to_le_bytes()).unwrap();
            let plan=MlxCheckedMapPlan::assess(ir,PcuScalarType::F64).unwrap();
            let prepared=session.prepare_checked_map_plan(plan).unwrap();
            assert!(matches!(prepared.execute(&[&input,&zero]),Err(MlxError::Arithmetic(fault))
                if fault.kind==PcuExecutionFaultKind::DivideByZero && fault.invocation_id==0 && !fault.recovered));
            let (outputs,fault)=prepared.execute(&[&input,&seed]).unwrap().into_outputs();assert!(fault.is_none());
            let [Some(stage),Some(output)]=outputs else {panic!("missing useful private siblings")};
            let mut got=vec![0;56];output.read_bytes_into(&mut got).unwrap();assert_eq!(got,10_f64.to_le_bytes().repeat(7));
            stage.release().unwrap();output.release().unwrap();input.release().unwrap();zero.release().unwrap();seed.release().unwrap();
        });
}
