//! Independent unique-input spans and actual checked source roles.
#[path = "offers/offers.rs"]
mod offers;
#[path = "portable/portable.rs"]
mod portable;
#[path = "source/source.rs"]
mod source;
#[path = "spans/spans.rs"]
mod spans;
#[rustfmt::skip]
use super::{
    graph,
    same,
    Sample,
};
#[rustfmt::skip]
use fusion_pcu_mlx::{
    MlxCheckedDivRemRolePlan,
    MlxBinaryInput,
    MlxError,
    MlxRuntime,
    MlxSession,
};
#[rustfmt::skip]
use pcu_facade::{
    PcuDispatchDataOp,
    PcuDispatchIndex,
    PcuDispatchKernelIr,
    PcuDispatchOp,
    PcuExecutionFaultKind,
    PcuHostDispatchError,
    PcuHostArgument,
    PcuHostKernelBackend,
    PcuImplementationRequirements,
    PcuI256,
    PcuI512,
    PcuRangePolicy,
    PcuReproducibility,
    PcuU256,
    PcuU512,
};
fn cold<T: Sample>() {
    graph::fixture::<T, _>(
        65,
        false,
        false,
        PcuImplementationRequirements::default(),
        |original| {
            for scalar in [false, true] {
                let mut ops = original.ops.to_vec();
                let PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad { binding, index, .. }) =
                    &mut ops[1]
                else {
                    panic!("load fixture");
                };
                *binding = original.bindings[0].reference();
                if scalar {
                    *index = PcuDispatchIndex::BindingElementZero;
                }
                let ir = PcuDispatchKernelIr {
                    ops: &ops,
                    ..*original
                };
                let plan = MlxCheckedDivRemRolePlan::assess(&ir).unwrap();
                assert_eq!(plan.input_bindings(), &[original.bindings[0].reference()]);
                assert_eq!(plan.input_element_counts(), [65, 0]);
                assert_eq!(plan.operand_inputs(), [0, 0]);
                assert_eq!(plan.operand_broadcast(), [false, scalar]);
                assert_eq!(
                    plan.output_bindings(),
                    [
                        original.bindings[2].reference(),
                        original.bindings[3].reference()
                    ]
                );
            }
            {
                let ir = PcuDispatchKernelIr {
                    numerical_requirements: PcuImplementationRequirements {
                        range_policy: PcuRangePolicy::Clamp,
                        ..Default::default()
                    },
                    ..*original
                };
                assert!(MlxCheckedDivRemRolePlan::assess(&ir).is_err());
            }
            let mut request = PcuImplementationRequirements::default();
            request.numerical_options.reproducibility = PcuReproducibility::PortableV1;
            let plan = MlxCheckedDivRemRolePlan::assess(&PcuDispatchKernelIr {
                numerical_requirements: request,
                ..*original
            })
            .unwrap();
            assert_eq!(plan.requirements(), request);
            assert_eq!(
                plan.implementation_local_id(),
                MlxCheckedDivRemRolePlan::assess(original)
                    .unwrap()
                    .implementation_local_id()
                    + 0x1000
            );
        },
    );
}
fn resident<T: Sample>(session: &MlxSession, foreign: &MlxSession) {
    graph::fixture::<T, _>(
        5,
        false,
        false,
        PcuImplementationRequirements::default(),
        |original| {
            let mut ops = original.ops.to_vec();
            let PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad { binding, index, .. }) =
                &mut ops[1]
            else {
                panic!("load fixture");
            };
            *binding = original.bindings[0].reference();
            *index = PcuDispatchIndex::BindingElementZero;
            let ir = PcuDispatchKernelIr {
                ops: &ops,
                ..*original
            };
            let mut prepared = session
                .checked_div_rem_role_backend()
                .prepare_host_kernel(&ir)
                .unwrap();
            assert_eq!(prepared.argument_count(), 3);
            let target = prepared.input_bindings()[0];
            let data = [T::raw(3), T::raw(17), T::raw(23), T::minimum(), T::raw(31)];
            let input = session.upload_encoded(&data).unwrap();
            let owner = MlxBinaryInput::Resident {
                target,
                array: &input,
            };
            let [q, r] = prepared.execute_inputs(&[owner]).unwrap();
            let sentinel = T::raw(91);
            let mut q_read = [sentinel; 7];
            let mut r_read = [sentinel; 7];
            q.read_into(&mut q_read).unwrap();
            r.read_into(&mut r_read).unwrap();
            for lane in 0..5 {
                same(
                    &q_read[lane..=lane],
                    &[data[lane].pcu_checked_div(data[0]).unwrap()],
                );
                same(
                    &r_read[lane..=lane],
                    &[data[lane].pcu_checked_rem(data[0]).unwrap()],
                );
            }
            same(&q_read[5..], &[sentinel; 2]);
            same(&r_read[5..], &[sentinel; 2]);
            q.release().unwrap();
            r.release().unwrap();
            let bytes = PcuHostArgument::read(target, &data);
            let [q, r] = prepared
                .execute_inputs(&[MlxBinaryInput::HostBytes {
                    target,
                    scalar: T::TYPE,
                    bytes: bytes.bytes(),
                }])
                .unwrap();
            q.read_into(&mut q_read).unwrap();
            r.read_into(&mut r_read).unwrap();
            check_divisor(&data, data[0], &q_read, &r_read);
            q.release().unwrap();
            r.release().unwrap();
            let other = foreign.upload_encoded(&data).unwrap();
            assert!(matches!(
                prepared.execute_inputs(&[MlxBinaryInput::Resident {
                    target,
                    array: &other
                }]),
                Err(MlxError::ForeignSession)
            ));
            assert!(!prepared.last_call_may_have_written());
            assert!(prepared.execute_inputs(&[owner, owner]).is_err());
            assert!(!prepared.last_call_may_have_written());
            let mut bad = data;
            bad[0] = T::raw(0);
            let bad_owner = session.upload_encoded(&bad).unwrap();
            assert!(
                matches!(prepared.execute_inputs(&[MlxBinaryInput::Resident{target,array:&bad_owner}]),Err(MlxError::Arithmetic(fault)) if fault.invocation_id==0 && fault.kind==PcuExecutionFaultKind::DivideByZero)
            );
            input.read_into(&mut q_read).unwrap();
            same(&q_read[..5], &data);
            bad_owner.release().unwrap();
            other.release().unwrap();
            input.release().unwrap();
        },
    );
}
#[test]
fn fourteen_width_detached_unique_read_roles_and_rejected_policies() {
    macro_rules! types {($($ty:ty),+) => {$(cold::<$ty>();)+};}
    types!(
        u8, i8, u16, i16, u32, i32, u64, i64, u128, i128, PcuU256, PcuI256, PcuU512, PcuI512
    );
}
fn verify<T: Sample>(session: &MlxSession) {
    let backend = session.checked_div_rem_role_backend();
    let mut repeated = source::repeated_prepare::<T, 65, _>(&backend).unwrap();
    let mut unread = source::unread_prepare::<T, 65, _>(&backend).unwrap();
    let mut reordered = source::reordered_prepare::<T, 65, _>(&backend).unwrap();
    let mut grid = source::grid_prepare::<T, 65, _>(&backend).unwrap();
    let mut scalar = source::scalar_divisor_prepare::<T, 65, _>(&backend).unwrap();
    let mut scalar_grid = source::scalar_grid_prepare::<T, 65, _>(&backend).unwrap();
    let sentinel = T::raw(91);
    let mut q = [sentinel; 67];
    let mut r = [sentinel; 67];
    for phase in 1..=3 {
        let mut input = std::array::from_fn::<_, 65, _>(|lane| {
            T::raw(u64::try_from(lane % 13).unwrap() + phase)
        });
        input[0] = T::raw(4);
        let divisor = T::raw(3);
        repeated(&mut q, &input, &mut r).unwrap();
        same(&q[..65], &[T::raw(1); 65]);
        same(&r[..65], &[T::raw(0); 65]);
        unread(&[], &mut r, &input, &mut q).unwrap();
        for lane in 0..65 {
            same(
                &q[lane..=lane],
                &[input[lane].pcu_checked_div(input[0]).unwrap()],
            );
            same(
                &r[lane..=lane],
                &[input[lane].pcu_checked_rem(input[0]).unwrap()],
            );
        }
        grid(&[], &mut r, &input, &mut q).unwrap();
        for lane in 0..65 {
            same(
                &q[lane..=lane],
                &[input[0].pcu_checked_div(input[lane]).unwrap()],
            );
            same(
                &r[lane..=lane],
                &[input[0].pcu_checked_rem(input[lane]).unwrap()],
            );
        }
        reordered(&[divisor; 65], &mut r, &input, &mut q).unwrap();
        check_divisor(&input, divisor, &q, &r);
        scalar(&mut q, &divisor, &mut r, &input).unwrap();
        check_divisor(&input, divisor, &q, &r);
        scalar_grid(&input[0], &mut q, &mut r).unwrap();
        same(&q[..65], &[T::raw(1); 65]);
        same(&r[..65], &[T::raw(0); 65]);
        same(&q[65..], &[sentinel; 2]);
        same(&r[65..], &[sentinel; 2]);
        let mut bad = input;
        bad[2] = T::raw(0);
        bad[6] = T::raw(0);
        q.fill(sentinel);
        r.fill(sentinel);
        let failure = repeated(&mut q, &bad, &mut r).unwrap_err();
        assert!(
            matches!(failure,PcuHostDispatchError::Backend(MlxError::Arithmetic(fault)) if fault.kind==PcuExecutionFaultKind::DivideByZero && fault.invocation_id==2 && !fault.recovered)
        );
        same(&q, &[sentinel; 67]);
        same(&r, &[sentinel; 67]);
        let mut short = [sentinel; 64];
        assert!(scalar(&mut q, &divisor, &mut short, &input).is_err());
        same(&q, &[sentinel; 67]);
        same(&short, &[sentinel; 64]);
        repeated(&mut q, &input, &mut r).unwrap();
        same(&q[..65], &[T::raw(1); 65]);
    }
}
fn check_divisor<T: Sample>(input: &[T], divisor: T, q: &[T], r: &[T]) {
    for (lane, &value) in input.iter().enumerate() {
        same(&q[lane..=lane], &[value.pcu_checked_div(divisor).unwrap()]);
        same(&r[lane..=lane], &[value.pcu_checked_rem(divisor).unwrap()]);
    }
}
#[test]
#[ignore = "required real MLX GPU fourteen-width selected-input source qualification"]
fn fourteen_width_source_repeated_broadcast_reordered_and_joint_faults() {
    let runtime = MlxRuntime::load_default().unwrap();
    let session = runtime.open_gpu(0).unwrap();
    let foreign = runtime.open_gpu(0).unwrap();
    macro_rules! types {($($ty:ty),+) => {$(verify::<$ty>(&session);)+};}
    types!(
        u8, i8, u16, i16, u32, i32, u64, i64, u128, i128, PcuU256, PcuI256, PcuU512, PcuI512
    );
    macro_rules! residents {($($ty:ty),+) => {$(resident::<$ty>(&session,&foreign);)+};}
    residents!(
        u8, i8, u16, i16, u32, i32, u64, i64, u128, i128, PcuU256, PcuI256, PcuU512, PcuI512
    );
}
