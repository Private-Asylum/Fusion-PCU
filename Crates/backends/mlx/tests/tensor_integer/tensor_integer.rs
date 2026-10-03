//! Detached and authentic fourteen-width owned checked integer graph qualification.
#[path = "source/source.rs"]
mod source;
#[path = "../checked_integer/support/support.rs"]
mod support;
#[rustfmt::skip]
use std::sync::Arc;
#[rustfmt::skip]
use pcu_facade::{
    global,
    PcuScalarType,
    PcuImplementationRequirements,
    PcuNumericalMode,
    PcuNumericalOptions,
    PcuCompoundArithmeticPolicy,
    PcuPrecisionPolicy,
    PcuFloatUnderflowPolicy,
    PcuRangePolicy,
    PcuReproducibility,
    PcuHostArgument,
    PcuBindingRef,
    PcuExecutionFault,
    PcuExecutionFaultKind,
    PcuDispatchIntegerBinaryOp,
    PcuI256,
    PcuU256,
    PcuI512,
    PcuU512,
};
#[rustfmt::skip]
use pcu_facade::dialect::tensor::{
    Graph,
    TensorOwnedSelectedProgram,
    TensorArithmeticRewritePolicy,
    TensorArithmeticCapability,
    TensorPointwiseGroupingPolicy,
};
#[rustfmt::skip]
use fusion_pcu_mlx::{
    MlxCheckedTensorIntegerPlan,
    MlxPreparedTensorIntegerProgram,
    MlxCheckedProgramInput,
    MlxRuntime,
    MlxSession,
    MlxError,
    MlxEncodedArray,
};
#[rustfmt::skip]
use support::{
    Sample,
    same,
};
fn program(
    scalar: PcuScalarType,
    request: PcuImplementationRequirements,
    operation: PcuDispatchIntegerBinaryOp,
    repeated: bool,
    unused: bool,
) -> Arc<TensorOwnedSelectedProgram> {
    let mut graph = Graph::try_new().unwrap();
    let a = graph.input([5], scalar).unwrap();
    let b = if repeated {
        a
    } else {
        graph.input([5], scalar).unwrap()
    };
    graph.set_numerical_options(request.numerical_options);
    let effect = match operation {
        PcuDispatchIntegerBinaryOp::Add => graph.add(a, b),
        PcuDispatchIntegerBinaryOp::Sub => graph.sub(b, a),
        PcuDispatchIntegerBinaryOp::Mul => graph.mul(a, b),
    }
    .unwrap();
    Arc::new(
        graph
            .into_selected_program(
                &[if unused { a } else { effect }],
                TensorArithmeticRewritePolicy::Disabled,
                TensorArithmeticCapability::Strict,
                TensorPointwiseGroupingPolicy::Disabled,
            )
            .unwrap(),
    )
}
fn requests() -> Vec<PcuImplementationRequirements> {
    let mut requests = Vec::new();
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
                    PcuFloatUnderflowPolicy::RejectSubnormalResult,
                    PcuFloatUnderflowPolicy::AllowGradualUnderflow,
                ] {
                    requests.push(PcuImplementationRequirements {
                        numerical_mode: mode,
                        numerical_options: PcuNumericalOptions {
                            compound_arithmetic: compound,
                            precision,
                            reproducibility: PcuReproducibility::Unspecified,
                        },
                        float_underflow: underflow,
                        range_policy: PcuRangePolicy::Reject,
                    });
                }
            }
        }
    }
    requests
}
#[test]
fn exact_integer_selected_effect_and_policy_admission() {
    for scalar in [
        PcuScalarType::U8,
        PcuScalarType::I8,
        PcuScalarType::U16,
        PcuScalarType::I16,
        PcuScalarType::U32,
        PcuScalarType::I32,
        PcuScalarType::U64,
        PcuScalarType::I64,
        PcuScalarType::U128,
        PcuScalarType::I128,
        PcuScalarType::U256,
        PcuScalarType::I256,
        PcuScalarType::U512,
        PcuScalarType::I512,
    ] {
        for request in requests() {
            for operation in [
                PcuDispatchIntegerBinaryOp::Add,
                PcuDispatchIntegerBinaryOp::Sub,
                PcuDispatchIntegerBinaryOp::Mul,
            ] {
                for repeated in [false, true] {
                    for unused in [false, true] {
                        let graph = program(scalar, request, operation, repeated, unused);
                        let plan =
                            MlxCheckedTensorIntegerPlan::assess_program(&graph, request).unwrap();
                        assert_eq!(plan.scalar_type(), scalar);
                        assert_eq!(plan.input_values().len(), if repeated { 1 } else { 2 });
                        assert_eq!(plan.requirements(), request);
                        let mut rejected = request;
                        rejected.range_policy = PcuRangePolicy::Clamp;
                        assert!(
                            MlxCheckedTensorIntegerPlan::assess_program(&graph, rejected).is_err()
                        );
                        rejected = request;
                        rejected.numerical_options.reproducibility = PcuReproducibility::PortableV1;
                        assert!(
                            MlxCheckedTensorIntegerPlan::assess_program(&graph, rejected).is_err()
                        );
                    }
                }
            }
        }
    }
}
fn capture<T: Sample>(
    request: PcuImplementationRequirements,
    profile: u8,
) -> (Arc<TensorOwnedSelectedProgram>, Vec<usize>) {
    let captured = global::__pcu_capture_tensor_program::<T, 2, _>(
        [global::PcuSourceShape::Slice { length: 5 }; 2],
        request.float_underflow,
        request.numerical_mode,
        request.numerical_options,
        |capture, inputs| match profile {
            0 => source::add::__pcu_capture_entry::<T>(capture, inputs),
            1 => source::sub::__pcu_capture_entry::<T>(capture, inputs),
            2 => source::mul::__pcu_capture_entry::<T>(capture, inputs),
            3 => source::repeated::__pcu_capture_entry::<T>(capture, inputs),
            _ => source::checked_unused::__pcu_capture_entry::<T>(capture, inputs),
        },
    )
    .unwrap();
    (
        Arc::clone(captured.program()),
        captured.argument_indices().to_vec(),
    )
}
fn source_admission<T: Sample>() {
    for request in requests() {
        for profile in 0..5 {
            let (graph, roles) = capture::<T>(request, profile);
            let plan = MlxCheckedTensorIntegerPlan::assess_program(&graph, request).unwrap();
            assert_eq!(roles, if profile == 3 { vec![1] } else { vec![0, 1] });
            assert_eq!(plan.input_values().len(), roles.len());
            assert_eq!(plan.scalar_type(), T::TYPE);
            assert_eq!(plan.requirements(), request);
        }
    }
}
#[test]
fn actual_source_fourteen_width_unique_roles_and_policies() {
    source_admission::<u8>();
    source_admission::<i8>();
    source_admission::<u16>();
    source_admission::<i16>();
    source_admission::<u32>();
    source_admission::<i32>();
    source_admission::<u64>();
    source_admission::<i64>();
    source_admission::<u128>();
    source_admission::<i128>();
    source_admission::<PcuU256>();
    source_admission::<PcuI256>();
    source_admission::<PcuU512>();
    source_admission::<PcuI512>();
}
fn read<T: Sample>(owner: &MlxEncodedArray, wanted: &[T]) {
    let sentinel = T::small(7);
    let mut stack = [sentinel; 7];
    owner.read_into(&mut stack).unwrap();
    same(&stack[..5], wanted);
    same(&stack[5..], &[sentinel; 2]);
    let mut short = [sentinel; 4];
    assert!(owner.read_into(&mut short).is_err());
    same(&short, &[sentinel; 4]);
}
fn expected_profile<T: Sample>(data: &[[T; 5]; 2], profile: u8) -> Vec<T> {
    let op = match profile {
        0 => PcuDispatchIntegerBinaryOp::Add,
        1 => PcuDispatchIntegerBinaryOp::Sub,
        _ => PcuDispatchIntegerBinaryOp::Mul,
    };
    let operands = match profile {
        1 => [1, 0],
        3 => [1, 1],
        _ => [0, 1],
    };
    let (wanted, fault) = support::expected(
        &data[operands[0]],
        &data[operands[1]],
        op,
        PcuRangePolicy::Reject,
        [false; 2],
    );
    assert_eq!(fault, None);
    wanted
}
fn fail<T: Sample>(prepared: &MlxPreparedTensorIntegerProgram, profile: u8, roles: &[usize]) {
    let (data, operands, op) = match profile {
        0 => (
            [[T::max(); 5], [T::small(1); 5]],
            [0, 1],
            PcuDispatchIntegerBinaryOp::Add,
        ),
        1 => (
            [[T::small(1); 5], [T::min(); 5]],
            [1, 0],
            PcuDispatchIntegerBinaryOp::Sub,
        ),
        2 => (
            [[T::max(); 5], [T::small(2); 5]],
            [0, 1],
            PcuDispatchIntegerBinaryOp::Mul,
        ),
        _ => (
            [[T::zero(); 5], [T::max(); 5]],
            [1, 1],
            PcuDispatchIntegerBinaryOp::Mul,
        ),
    };
    let (_, fault) = support::expected(
        &data[operands[0]],
        &data[operands[1]],
        op,
        PcuRangePolicy::Reject,
        [false; 2],
    );
    let fault = fault.expect("independent lane0 range fault");
    assert_eq!(fault.invocation_id, 0);
    assert!(!fault.recovered);
    let arrays = data.map(|values| prepared.session().upload_encoded(&values).unwrap());
    let inputs: Vec<_> = prepared
        .plan()
        .input_values()
        .iter()
        .zip(roles)
        .map(|(&id, &role)| (id, MlxCheckedProgramInput::Resident(&arrays[role])))
        .collect();
    assert_eq!(
        prepared.execute_mixed(&inputs).err(),
        Some(MlxError::Arithmetic(fault))
    );
}
fn used<T: Sample>(session: &MlxSession, foreign: &MlxSession) {
    let data = [
        [
            T::zero(),
            T::small(2),
            T::small(3),
            T::small(4),
            T::small(5),
        ],
        [
            T::small(1),
            T::small(3),
            T::small(4),
            T::small(5),
            T::small(6),
        ],
    ];
    let residents = data.map(|values| session.upload_encoded(&values).unwrap());
    let other = foreign.upload_encoded(&data[1]).unwrap();
    for request in requests() {
        for profile in 0..4 {
            let (graph, roles) = capture::<T>(request, profile);
            let prepared = session
                .prepare_tensor_integer_program(graph, request)
                .unwrap();
            let ids = prepared.plan().input_values();
            assert_eq!(roles.len(), ids.len());
            assert_eq!(roles, if profile == 3 { vec![1] } else { vec![0, 1] });
            let wanted = expected_profile(&data, profile);
            let bytes = data.map(|values| {
                PcuHostArgument::read(PcuBindingRef::new(0, 0), &values)
                    .bytes()
                    .to_vec()
            });
            let inputs: Vec<_> = ids
                .iter()
                .zip(&roles)
                .map(|(&id, &role)| {
                    (
                        id,
                        MlxCheckedProgramInput::Host {
                            scalar: T::TYPE,
                            bytes: &bytes[role],
                        },
                    )
                })
                .collect();
            let owner = prepared.execute_mixed(&inputs).unwrap();
            read(&owner, &wanted);
            let mixed: Vec<_> = ids
                .iter()
                .zip(&roles)
                .rev()
                .enumerate()
                .map(|(slot, (&id, &role))| {
                    (
                        id,
                        if slot == 0 {
                            MlxCheckedProgramInput::Resident(&residents[role])
                        } else {
                            MlxCheckedProgramInput::Host {
                                scalar: T::TYPE,
                                bytes: &bytes[role],
                            }
                        },
                    )
                })
                .collect();
            let next = prepared.execute_mixed(&mixed).unwrap();
            read(&next, &wanted);
            let all: Vec<_> = ids
                .iter()
                .zip(&roles)
                .map(|(&id, &role)| (id, MlxCheckedProgramInput::Resident(&residents[role])))
                .collect();
            read(&prepared.execute_mixed(&all).unwrap(), &wanted);
            let mut bad = all;
            bad[roles.len() - 1].1 = MlxCheckedProgramInput::Resident(&other);
            assert!(matches!(
                prepared.execute_mixed(&bad),
                Err(MlxError::ForeignSession)
            ));
            assert!(prepared.execute_mixed(&[]).is_err());
            fail::<T>(&prepared, profile, &roles);
            read(&owner, &wanted);
            read(&prepared.execute_mixed(&inputs).unwrap(), &wanted);
            drop(prepared);
            drop(next);
            read(&owner, &wanted);
        }
    }
}
fn unused<T: Sample>(session: &MlxSession) {
    let left = [T::zero(), T::small(2), T::min(), T::max(), T::small(3)];
    let right = [T::small(1), T::small(3), T::zero(), T::zero(), T::small(4)];
    let a = session.upload_encoded(&left).unwrap();
    let b = session.upload_encoded(&right).unwrap();
    let overflow = session.upload_encoded(&[T::max(); 5]).unwrap();
    let ones = session.upload_encoded(&[T::small(1); 5]).unwrap();
    for request in requests() {
        let (graph, roles) = capture::<T>(request, 4);
        assert_eq!(roles, [0, 1]);
        let prepared = session
            .prepare_tensor_integer_program(graph, request)
            .unwrap();
        let ids = prepared.plan().input_values();
        let owner = prepared
            .execute_mixed(&[
                (ids[0], MlxCheckedProgramInput::Resident(&a)),
                (ids[1], MlxCheckedProgramInput::Resident(&b)),
            ])
            .unwrap();
        read(&owner, &left);
        assert_eq!(
            prepared
                .execute_mixed(&[
                    (ids[0], MlxCheckedProgramInput::Resident(&overflow)),
                    (ids[1], MlxCheckedProgramInput::Resident(&ones))
                ])
                .err(),
            Some(MlxError::Arithmetic(PcuExecutionFault {
                invocation_id: 0,
                kind: PcuExecutionFaultKind::ArithmeticOverflow,
                recovered: false
            }))
        );
        read(&owner, &left);
        read(
            &prepared
                .execute_mixed(&[
                    (ids[0], MlxCheckedProgramInput::Resident(&a)),
                    (ids[1], MlxCheckedProgramInput::Resident(&b)),
                ])
                .unwrap(),
            &left,
        );
        drop(prepared);
        read(&owner, &left);
    }
}
fn native<T: Sample>(session: &MlxSession, foreign: &MlxSession) {
    used::<T>(session, foreign);
    unused::<T>(session);
}
#[test]
#[ignore = "Requires actual fourteen-width checked integer selected owner/native lifetime qualification."]
fn authentic_source_integer_owners_and_unused_fault_lifetimes() {
    let session = MlxRuntime::load_default().unwrap().open_gpu(0).unwrap();
    let foreign = MlxRuntime::load_default().unwrap().open_gpu(0).unwrap();
    native::<u8>(&session, &foreign);
    native::<i8>(&session, &foreign);
    native::<u16>(&session, &foreign);
    native::<i16>(&session, &foreign);
    native::<u32>(&session, &foreign);
    native::<i32>(&session, &foreign);
    native::<u64>(&session, &foreign);
    native::<i64>(&session, &foreign);
    native::<u128>(&session, &foreign);
    native::<i128>(&session, &foreign);
    native::<PcuU256>(&session, &foreign);
    native::<PcuI256>(&session, &foreign);
    native::<PcuU512>(&session, &foreign);
    native::<PcuI512>(&session, &foreign);
}
