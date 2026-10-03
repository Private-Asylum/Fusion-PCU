//! Fourteen-width public preparation, exact joint publication and actual-session immutable owners.
#[path = "graph/graph.rs"]
mod graph;
#[path = "offers/offers.rs"]
mod offers;
#[path = "ordinary/ordinary.rs"]
mod ordinary;
#[path = "roles/roles.rs"]
mod roles;
#[path = "source/source.rs"]
mod source;
#[rustfmt::skip]
use fusion_pcu_mlx::{
    MlxRuntime,
    MlxSession,
    MlxError,
    MlxCheckedDivRemPlan,
    MlxBinaryInput,
};
#[rustfmt::skip]
use pcu_facade::{
    PcuCheckedIntegerDivision,
    PcuScalar,
    PcuHostKernelBackend,
    PcuPreparedHostKernel,
    PcuHostArgument,
    PcuBindingRef,
    PcuHostDispatchError,
    PcuImplementationRequirements,
    PcuNumericalMode,
    PcuCompoundArithmeticPolicy,
    PcuPrecisionPolicy,
    PcuFloatUnderflowPolicy,
    PcuRangePolicy,
    PcuReproducibility,
    PcuExecutionFault,
    PcuExecutionFaultKind,
    PcuI256,
    PcuU256,
    PcuI512,
    PcuU512,
};
trait Sample: PcuCheckedIntegerDivision {
    fn raw(value: u64) -> Self;
    fn minimum() -> Self;
    fn negative_one() -> Self;
}
macro_rules! samples {
    ($($ty:ty=>$width:literal;)+) => {$(impl Sample for $ty {
        fn raw(value:u64)->Self {let mut bytes=[0;$width];let n=bytes.len().min(8);bytes[..n].copy_from_slice(&value.to_le_bytes()[..n]);Self::decode_le(bytes)}
        fn minimum()->Self {let mut bytes=[0;$width];bytes[$width-1]=0x80;Self::decode_le(bytes)}
        fn negative_one()->Self {Self::decode_le([u8::MAX;$width])}
    })+};
}
samples! {u8=>1;i8=>1;u16=>2;i16=>2;u32=>4;i32=>4;u64=>8;i64=>8;u128=>16;i128=>16;PcuU256=>32;PcuI256=>32;PcuU512=>64;PcuI512=>64;}
fn same<T: PcuScalar>(actual: &[T], expected: &[T]) {
    assert_eq!(
        PcuHostArgument::read(PcuBindingRef::new(0, 0), actual).bytes(),
        PcuHostArgument::read(PcuBindingRef::new(0, 0), expected).bytes()
    );
}
fn cold<T: Sample>() {
    for mode in [PcuNumericalMode::Boundary, PcuNumericalMode::Strict] {
        for compound in [
            PcuCompoundArithmeticPolicy::Checked,
            PcuCompoundArithmeticPolicy::BackendDefined,
        ] {
            for precision in [
                PcuPrecisionPolicy::Preserve,
                PcuPrecisionPolicy::BackendOptimized,
            ] {
                for underflow in [
                    PcuFloatUnderflowPolicy::AllowGradualUnderflow,
                    PcuFloatUnderflowPolicy::IeeeAfterRounding,
                    PcuFloatUnderflowPolicy::RejectSubnormalResult,
                ] {
                    let request = PcuImplementationRequirements {
                        numerical_mode: mode,
                        float_underflow: underflow,
                        numerical_options: pcu_facade::PcuNumericalOptions {
                            compound_arithmetic: compound,
                            precision,
                            ..Default::default()
                        },
                        ..Default::default()
                    };
                    for grid in [false, true] {
                        for reverse in [false, true] {
                            graph::fixture::<T, _>(65, grid, reverse, request, |ir| {
                                let plan = MlxCheckedDivRemPlan::assess(ir).unwrap();
                                assert_eq!(plan.scalar_type(), T::TYPE);
                                assert_eq!(plan.element_count(), 65);
                                assert_eq!(plan.requirements(), request);
                                assert_eq!(
                                    plan.operand_inputs(),
                                    if reverse { [1, 0] } else { [0, 1] }
                                );
                                assert_eq!(
                                    plan.output_bindings(),
                                    &[PcuBindingRef::new(2, 9), PcuBindingRef::new(2, 1)]
                                );
                            });
                        }
                    }
                }
            }
        }
    }
}
#[test]
fn fourteen_width_detached_admission_and_negative_policy_schema() {
    cold::<u8>();
    cold::<i8>();
    cold::<u16>();
    cold::<i16>();
    cold::<u32>();
    cold::<i32>();
    cold::<u64>();
    cold::<i64>();
    cold::<u128>();
    cold::<i128>();
    cold::<PcuU256>();
    cold::<PcuI256>();
    cold::<PcuU512>();
    cold::<PcuI512>();
    let clamp = PcuImplementationRequirements {
        range_policy: PcuRangePolicy::Clamp,
        ..PcuImplementationRequirements::default()
    };
    let mut portable = PcuImplementationRequirements::default();
    portable.numerical_options.reproducibility = PcuReproducibility::PortableV1;
    graph::fixture::<i64, _>(65, false, false, clamp, |ir| {
        assert!(MlxCheckedDivRemPlan::assess(ir).is_err());
    });
    graph::fixture::<i64, _>(65, false, false, portable, |ir| {
        let plan = MlxCheckedDivRemPlan::assess(ir).unwrap();
        assert_eq!(plan.requirements(), portable);
        assert_eq!(plan.implementation_local_id(), 0x1507);
    });
    macro_rules! negative {($($ty:ty),+) => {$(graph::fixture::<$ty,_>(65,false,false,Default::default(),|ir| assert!(MlxCheckedDivRemPlan::assess(ir).is_err()));)+};}
    negative!(f32, f64);
    graph::fixture::<i64, _>(
        0,
        false,
        false,
        PcuImplementationRequirements::default(),
        |ir| {
            assert!(MlxCheckedDivRemPlan::assess(ir).is_err());
        },
    );
}
fn checked<T: Sample>(left: &[T], right: &[T], q: &[T], r: &[T], sentinel: T) {
    let count = left.len();
    for index in 0..count {
        same(
            &q[index..=index],
            &[left[index].pcu_checked_div(right[index]).unwrap()],
        );
        same(
            &r[index..=index],
            &[left[index].pcu_checked_rem(right[index]).unwrap()],
        );
    }
    same(&q[count..], &[sentinel; 2]);
    same(&r[count..], &[sentinel; 2]);
}
fn source_calls<T: Sample>(session: &MlxSession) {
    let backend = session.checked_div_rem_backend();
    let mut direct = source::direct_prepare::<T, 65, _>(&backend).unwrap();
    let mut grid = source::grid_prepare::<T, 65, _>(&backend).unwrap();
    let sentinel = T::raw(17);
    let mut q = [sentinel; 67];
    let mut r = [sentinel; 67];
    for phase in 1..=3 {
        let left =
            std::array::from_fn::<_, 65, _>(|lane| T::raw(u64::try_from(lane).unwrap() + phase));
        let right = [T::raw(3); 65];
        direct(&left, &right, &mut q, &mut r).unwrap();
        checked(&left, &right, &q, &r, sentinel);
        grid(&left, &right, &mut q, &mut r).unwrap();
        checked(&left, &right, &q, &r, sentinel);
        let mut short = [sentinel; 64];
        q.fill(sentinel);
        r.fill(sentinel);
        assert!(direct(&left, &right, &mut q, &mut short).is_err());
        same(&q, &[sentinel; 67]);
        same(&short, &[sentinel; 64]);
        let mut bad = right;
        bad[1] = T::raw(0);
        bad[3] = T::raw(0);
        let error = direct(&left, &bad, &mut q, &mut r).unwrap_err();
        assert_eq!(
            error,
            PcuHostDispatchError::Backend(MlxError::Arithmetic(PcuExecutionFault {
                kind: PcuExecutionFaultKind::DivideByZero,
                invocation_id: 1,
                recovered: false
            }))
        );
        same(&q, &[sentinel; 67]);
        same(&r, &[sentinel; 67]);
        direct(&left, &right, &mut q, &mut r).unwrap();
        checked(&left, &right, &q, &r, sentinel);
    }
}
fn requested<T: Sample>(session: &MlxSession) {
    for mode in [PcuNumericalMode::Boundary, PcuNumericalMode::Strict] {
        for compound in [
            PcuCompoundArithmeticPolicy::Checked,
            PcuCompoundArithmeticPolicy::BackendDefined,
        ] {
            for precision in [
                PcuPrecisionPolicy::Preserve,
                PcuPrecisionPolicy::BackendOptimized,
            ] {
                for underflow in [
                    PcuFloatUnderflowPolicy::IeeeAfterRounding,
                    PcuFloatUnderflowPolicy::AllowGradualUnderflow,
                    PcuFloatUnderflowPolicy::RejectSubnormalResult,
                ] {
                    let request = PcuImplementationRequirements {
                        numerical_mode: mode,
                        float_underflow: underflow,
                        numerical_options: pcu_facade::PcuNumericalOptions {
                            compound_arithmetic: compound,
                            precision,
                            ..Default::default()
                        },
                        ..Default::default()
                    };
                    for grid in [false, true] {
                        let mut kernel = graph::fixture::<T, _>(5, grid, false, request, |ir| {
                            session.checked_div_rem_backend().prepare_host_kernel(ir)
                        })
                        .unwrap();
                        let sentinel = T::raw(19);
                        let left = [T::raw(23); 5];
                        let right = [T::raw(5); 5];
                        let input = *kernel.input_bindings();
                        let output = *kernel.output_bindings();
                        let mut q = [sentinel; 7];
                        let mut r = [sentinel; 7];
                        let mut call = |right: &[T], q: &mut [T], r: &mut [T]| {
                            kernel.call(&mut [
                                PcuHostArgument::read_write(output[1], r),
                                PcuHostArgument::read(input[0], &left),
                                PcuHostArgument::read_write(output[0], q),
                                PcuHostArgument::read(input[1], right),
                            ])
                        };
                        call(&right, &mut q, &mut r).unwrap();
                        checked(&left, &right, &q, &r, sentinel);
                        q.fill(sentinel);
                        r.fill(sentinel);
                        let mut bad = right;
                        bad[1] = T::raw(0);
                        bad[3] = T::raw(0);
                        assert_eq!(
                            call(&bad, &mut q, &mut r).unwrap_err(),
                            PcuHostDispatchError::Backend(MlxError::Arithmetic(
                                PcuExecutionFault {
                                    kind: PcuExecutionFaultKind::DivideByZero,
                                    invocation_id: 1,
                                    recovered: false
                                }
                            ))
                        );
                        same(&q, &[sentinel; 7]);
                        same(&r, &[sentinel; 7]);
                        call(&right, &mut q, &mut r).unwrap();
                        checked(&left, &right, &q, &r, sentinel);
                    }
                }
            }
        }
    }
}
fn direct_broadcast<T: Sample>(session: &MlxSession) {
    let sentinel = T::raw(19);
    for broadcast in [[false, false], [true, false], [false, true]] {
        let a = [T::raw(23); 5];
        let b = [T::raw(5); 5];
        let left = if broadcast[0] { &a[..1] } else { &a };
        let right = if broadcast[1] { &b[..1] } else { &b };
        let mut control = session
            .prepare_checked_div_rem_control(T::TYPE, 5, [left.len(), right.len()], broadcast)
            .unwrap();
        let mut q = [sentinel; 7];
        let mut r = [sentinel; 7];
        control.call([left, right], [&mut q, &mut r]).unwrap();
        checked(&a, &b, &q, &r, sentinel);
        let [qo, ro] = control.execute_encoded([left, right]).unwrap();
        qo.read_into(&mut q).unwrap();
        ro.read_into(&mut r).unwrap();
        checked(&a, &b, &q, &r, sentinel);
    }
}
#[allow(clippy::too_many_lines)] // Joint source/schema/session/output preflight, terminal faults and escaped siblings in one oracle.
fn owners<T: Sample>(session: &MlxSession, foreign: &MlxSession) {
    let left = [T::raw(23); 5];
    let right = [T::raw(5); 5];
    let sentinel = T::raw(17);
    let a = session.upload_encoded(&left).unwrap();
    let b = session.upload_encoded(&right).unwrap();
    let other = foreign.upload_encoded(&right).unwrap();
    for grid in [false, true] {
        for reverse in [false, true] {
            let mut kernel = graph::fixture::<T, _>(
                5,
                grid,
                reverse,
                PcuImplementationRequirements::default(),
                |ir| session.checked_div_rem_backend().prepare_host_kernel(ir),
            )
            .unwrap();
            assert_eq!(kernel.argument_count(), 4);
            assert_eq!(kernel.input_byte_lengths(), [5 * T::ENCODED_SIZE; 2]);
            let (lhs, rhs) = if reverse {
                (&right, &left)
            } else {
                (&left, &right)
            };
            let [q, r] = kernel.execute_resident([&a, &b]).unwrap();
            let mut oq = [sentinel; 7];
            let mut or = [sentinel; 7];
            q.read_into(&mut oq).unwrap();
            r.read_into(&mut or).unwrap();
            checked(lhs, rhs, &oq, &or, sentinel);
            let host = PcuHostArgument::read(kernel.input_bindings()[0], &left);
            let [mq, mr] = kernel
                .execute_inputs(&[
                    MlxBinaryInput::Resident {
                        target: kernel.input_bindings()[1],
                        array: &b,
                    },
                    MlxBinaryInput::HostBytes {
                        target: host.target(),
                        scalar: T::TYPE,
                        bytes: host.bytes(),
                    },
                ])
                .unwrap();
            mq.read_into(&mut oq).unwrap();
            mr.read_into(&mut or).unwrap();
            checked(lhs, rhs, &oq, &or, sentinel);
            assert!(matches!(
                kernel.execute_inputs(&[
                    MlxBinaryInput::HostBytes {
                        target: host.target(),
                        scalar: T::TYPE,
                        bytes: host.bytes()
                    },
                    MlxBinaryInput::Resident {
                        target: kernel.input_bindings()[1],
                        array: &other
                    },
                ]),
                Err(MlxError::ForeignSession)
            ));
            assert!(!kernel.last_call_may_have_written());
            oq.fill(sentinel);
            or.fill(sentinel);
            let refs = *kernel.output_bindings();
            kernel
                .call(&mut [
                    PcuHostArgument::read_write(refs[1], &mut or),
                    PcuHostArgument::read(kernel.input_bindings()[1], &right),
                    PcuHostArgument::read_write(refs[0], &mut oq),
                    PcuHostArgument::read(kernel.input_bindings()[0], &left),
                ])
                .unwrap();
            checked(lhs, rhs, &oq, &or, sentinel);
            mq.release().unwrap();
            mr.release().unwrap();
            let shared = q.clone();
            shared.release().unwrap();
            drop(kernel);
            q.read_into(&mut oq).unwrap();
            r.read_into(&mut or).unwrap();
            checked(lhs, rhs, &oq, &or, sentinel);
            q.release().unwrap();
            r.release().unwrap();
        }
    }
    let signed = matches!(
        T::TYPE,
        pcu_facade::PcuScalarType::I8
            | pcu_facade::PcuScalarType::I16
            | pcu_facade::PcuScalarType::I32
            | pcu_facade::PcuScalarType::I64
            | pcu_facade::PcuScalarType::I128
            | pcu_facade::PcuScalarType::I256
            | pcu_facade::PcuScalarType::I512
    );
    let mut control = session
        .prepare_checked_div_rem_control(T::TYPE, 5, [5; 2], [false; 2])
        .unwrap();
    let bad_left = [
        T::raw(23),
        T::minimum(),
        T::raw(23),
        T::minimum(),
        T::raw(23),
    ];
    let bad_right = [
        T::raw(5),
        T::negative_one(),
        T::raw(5),
        T::raw(0),
        T::raw(5),
    ];
    let mut q = [sentinel; 7];
    let mut r = [sentinel; 7];
    let error = control
        .call([&bad_left, &bad_right], [&mut q, &mut r])
        .unwrap_err();
    assert_eq!(
        error,
        MlxError::Arithmetic(PcuExecutionFault {
            kind: if signed {
                PcuExecutionFaultKind::SignedDivisionOverflow
            } else {
                PcuExecutionFaultKind::DivideByZero
            },
            invocation_id: if signed { 1 } else { 3 },
            recovered: false
        })
    );
    same(&q, &[sentinel; 7]);
    same(&r, &[sentinel; 7]);
    assert!(!control.last_call_may_have_written_existing_encoded_owner());
    control.call([&left, &right], [&mut q, &mut r]).unwrap();
    checked(&left, &right, &q, &r, sentinel);
    let minimum = [T::minimum(); 5];
    let ones = [T::raw(1); 5];
    control.call([&minimum, &ones], [&mut q, &mut r]).unwrap();
    checked(&minimum, &ones, &q, &r, sentinel);
}
#[test]
#[ignore = "Required authentic MLX fourteen-width public source/graph/native joint publication on Apple GPU."]
fn fourteen_width_public_source_joint_publication_and_owners() {
    let runtime = MlxRuntime::load_default().unwrap();
    let session = runtime.open_gpu(0).unwrap();
    let foreign = runtime.open_gpu(0).unwrap();
    macro_rules! run {($($ty:ty),+) => {$(source_calls::<$ty>(&session);requested::<$ty>(&session);direct_broadcast::<$ty>(&session);owners::<$ty>(&session,&foreign);)+};}
    run!(
        u8, i8, u16, i16, u32, i32, u64, i64, u128, i128, PcuU256, PcuI256, PcuU512, PcuI512
    );
}
