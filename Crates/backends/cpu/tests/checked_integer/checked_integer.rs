//! Full-width source and detached/reference checked integer execution.

use core::fmt::Debug;
#[rustfmt::skip]
use fusion_pcu_cpu::{
    PcuCheckedIntegerReference,
    PcuCpuCheckedInteger,
    PcuCpuCheckedIntegerError,
    PcuCpuPreparedInteger,
    PcuCpuIntegerOfferError,
    PcuCpuIntegerOffers,
};
#[rustfmt::skip]
use pcu_facade::{
    PcuBinding,
    PcuBindingAccess,
    PcuBindingRef,
    PcuBindingStorageClass,
    PcuCheckedInteger,
    PcuDispatchControlOp,
    PcuDispatchDataOp,
    PcuDispatchEntryPoint,
    PcuDispatchFeatureCaps,
    PcuDispatchIndex,
    PcuDispatchIntegerBinaryOp,
    PcuDispatchKernelIr,
    PcuDispatchOp,
    PcuDispatchSubmission,
    PcuDispatchValueId,
    PcuExecutionFault,
    PcuExecutionFaultKind,
    PcuHostArgument,
    PcuHostKernelBackend,
    PcuHostScalarBinding,
    PcuHostScalarSlice,
    PcuInvocationParameters,
    PcuInvocationShape,
    PcuKernelId,
    PcuPreparedHostKernel,
    PcuSynchronousHostDispatchBackend,
    PcuValueType,
    PcuValueTypeCaps,
    PcuCostBoundary,
    PcuDeviceIdentity,
    PcuExecutorId,
    PcuImplementationCost,
    PcuImplementationOffers,
    PcuImplementationRequest,
    PcuImplementationRequirements,
    PcuNumericalMode,
    PcuObjectKind,
    PcuObjectRef,
    PcuProviderId,
    PcuRangePolicy,
    PcuReproducibility,
};

#[path = "source/source.rs"]
mod source;

fn native<T: PcuCheckedInteger>(
    op: PcuDispatchIntegerBinaryOp,
    lhs: T,
    rhs: T,
) -> Result<T, PcuExecutionFaultKind> {
    match op {
        PcuDispatchIntegerBinaryOp::Add => lhs.pcu_checked_add(rhs),
        PcuDispatchIntegerBinaryOp::Sub => lhs.pcu_checked_sub(rhs),
        PcuDispatchIntegerBinaryOp::Mul => lhs.pcu_checked_mul(rhs),
    }
}

fn reference_call<T: PcuCheckedInteger>(
    kernel: &PcuDispatchKernelIr<'_>,
    lhs: &[T],
    rhs: &[T],
    output: &mut [T],
) -> Result<(), PcuCpuCheckedIntegerError> {
    let submission = PcuDispatchSubmission {
        kernel,
        shape: PcuInvocationShape::invocations(
            core::num::NonZeroU32::new(kernel.entry.logical_shape[0]).unwrap(),
        ),
    };
    PcuCheckedIntegerReference::<T>::new().run_host_direct(
        submission,
        &mut [
            PcuHostScalarBinding {
                target: PcuBindingRef::new(0, 0),
                slice: PcuHostScalarSlice::Read(lhs),
            },
            PcuHostScalarBinding {
                target: PcuBindingRef::new(0, 1),
                slice: PcuHostScalarSlice::Read(rhs),
            },
            PcuHostScalarBinding {
                target: PcuBindingRef::new(0, 2),
                slice: PcuHostScalarSlice::ReadWrite(output),
            },
        ],
        PcuInvocationParameters::empty(),
    )
}

fn check_source<T: PcuCheckedInteger + Debug + PartialEq>(
    mut call: impl FnMut(&[T], &[T], &mut [T]) -> Result<(), PcuCpuCheckedIntegerError>,
    op: PcuDispatchIntegerBinaryOp,
    zero: T,
    one: T,
    maximum: T,
    minimum: T,
    random: impl Fn(u64) -> T,
) {
    let mut state = 0x243f_6a88_85a3_08d3_u64;
    for iteration in 0..256 {
        let mut lhs = [zero; 17];
        let mut rhs = [zero; 17];
        for (index, (left, right)) in lhs.iter_mut().zip(&mut rhs).enumerate() {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            *left = random(state);
            *right = if iteration % 2 == 0 { zero } else { one };
            if index == 0 {
                *left = one;
            }
        }
        let mut output = [maximum; 19];
        let mut expected = [maximum; 19];
        let mut first = None;
        for index in 0..17 {
            match native(op, lhs[index], rhs[index]) {
                Ok(value) => expected[index] = value,
                Err(kind) => {
                    first = Some((index, kind));
                    break;
                }
            }
        }
        let outcome = call(&lhs, &rhs, &mut output);
        let mut reference_output = [maximum; 19];
        let reference_outcome = with_kernel::<T, _>(profile(op), |kernel| {
            reference_call(kernel, &lhs, &rhs, &mut reference_output)
        });
        assert_eq!(reference_outcome, outcome);
        assert_eq!(reference_output, output);
        if let Some((invocation, kind)) = first {
            assert_eq!(
                outcome,
                Err(PcuCpuCheckedIntegerError::Fault(PcuExecutionFault {
                    recovered: false,
                    kind,
                    invocation_id: u64::try_from(invocation).unwrap(),
                }))
            );
            assert_eq!(output, [maximum; 19]);
        } else {
            outcome.unwrap();
            assert_eq!(output, expected);
        }
    }
    let mut lhs = [one; 17];
    let mut rhs = [one; 17];
    lhs[5] = match op {
        PcuDispatchIntegerBinaryOp::Sub => minimum,
        _ => maximum,
    };
    if op == PcuDispatchIntegerBinaryOp::Mul {
        rhs[5] = maximum;
    }
    let kind = native(op, lhs[5], rhs[5]).unwrap_err();
    let mut output = [maximum; 19];
    assert_eq!(
        call(&lhs, &rhs, &mut output),
        Err(PcuCpuCheckedIntegerError::Fault(PcuExecutionFault {
            recovered: false,
            kind,
            invocation_id: 5,
        }))
    );
    assert_eq!(output, [maximum; 19]);
    lhs.fill(one);
    rhs.fill(one);
    call(&lhs, &rhs, &mut output).unwrap();
    assert!(
        output[..17]
            .iter()
            .all(|value| *value == native(op, one, one).unwrap())
    );
    assert_eq!(output[17..], [maximum; 2]);
    assert_eq!(
        call(&lhs[..16], &rhs, &mut output),
        Err(PcuCpuCheckedIntegerError::InvalidArguments)
    );
}

macro_rules! source_cases {
    ($name:ident, $module:ident, $ty:ty) => {
        mod $name {
            use super::*;

            fn random(bits: u64) -> $ty {
                <$ty>::from_le_bytes(
                    bits.to_le_bytes()[..core::mem::size_of::<$ty>()]
                        .try_into()
                        .unwrap(),
                )
            }

            #[test]
            fn add_source() {
                let backend = PcuCpuCheckedInteger::<$ty>::new();
                check_source(
                    source::$module::add_prepare::<17, _>(&backend).unwrap(),
                    PcuDispatchIntegerBinaryOp::Add,
                    0,
                    1,
                    <$ty>::MAX,
                    <$ty>::MIN,
                    random,
                );
            }
            #[test]
            fn sub_source() {
                let backend = PcuCpuCheckedInteger::<$ty>::new();
                check_source(
                    source::$module::sub_prepare::<17, _>(&backend).unwrap(),
                    PcuDispatchIntegerBinaryOp::Sub,
                    0,
                    1,
                    <$ty>::MAX,
                    <$ty>::MIN,
                    random,
                );
            }
            #[test]
            fn mul_source() {
                let backend = PcuCpuCheckedInteger::<$ty>::new();
                check_source(
                    source::$module::mul_prepare::<17, _>(&backend).unwrap(),
                    PcuDispatchIntegerBinaryOp::Mul,
                    0,
                    1,
                    <$ty>::MAX,
                    <$ty>::MIN,
                    random,
                );
            }
        }
    };
}

source_cases!(i8_cases, i8_source, i8);
source_cases!(u8_cases, u8_source, u8);
source_cases!(i16_cases, i16_source, i16);
source_cases!(u16_cases, u16_source, u16);
source_cases!(i32_cases, i32_source, i32);
source_cases!(u32_cases, u32_source, u32);
source_cases!(i64_cases, i64_source, i64);
source_cases!(u64_cases, u64_source, u64);

#[derive(Clone, Copy)]
struct Profile {
    op: PcuDispatchIntegerBinaryOp,
    grid: bool,
    broadcast: [bool; 2],
    order: [usize; 2],
    operands: [usize; 2],
}

fn with_kernel<T: PcuCheckedInteger, R>(
    profile: Profile,
    callback: impl FnOnce(&PcuDispatchKernelIr<'_>) -> R,
) -> R {
    let bindings = [
        PcuBinding::scalar::<T>(
            Some("first"),
            0,
            0,
            PcuBindingStorageClass::Storage,
            PcuBindingAccess::ReadOnly,
        ),
        PcuBinding::scalar::<T>(
            Some("second"),
            0,
            1,
            PcuBindingStorageClass::Storage,
            PcuBindingAccess::ReadOnly,
        ),
        PcuBinding::scalar::<T>(
            Some("output"),
            0,
            2,
            PcuBindingStorageClass::Storage,
            PcuBindingAccess::WriteOnly,
        ),
    ];
    let index = if profile.grid {
        PcuDispatchIndex::GridStrideId
    } else {
        PcuDispatchIndex::InvocationId
    };
    let values = [PcuDispatchValueId(1), PcuDispatchValueId(2)];
    let load = |slot: usize| {
        let binding = profile.order[slot];
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
            result: values[slot],
            binding: bindings[binding].reference(),
            index: if profile.broadcast[binding] {
                PcuDispatchIndex::BindingElementZero
            } else {
                index
            },
        })
    };
    let body = [
        load(0),
        load(1),
        PcuDispatchOp::Data(PcuDispatchDataOp::CheckedIntegerBinary {
            range_policy: pcu_facade::PcuRangePolicy::Reject,
            value_type: PcuValueType::Scalar(T::TYPE),
            op: profile.op,
            result: PcuDispatchValueId(3),
            lhs: values[profile.operands[0]],
            rhs: values[profile.operands[1]],
        }),
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
            binding: bindings[2].reference(),
            index,
            value: PcuDispatchValueId(3),
        }),
    ];
    let direct = [
        body[0],
        body[1],
        body[2],
        body[3],
        PcuDispatchOp::Control(PcuDispatchControlOp::Return),
    ];
    let grid = [
        PcuDispatchOp::GridStrideLoop {
            extent: 17,
            body: &body,
        },
        PcuDispatchOp::Control(PcuDispatchControlOp::Return),
    ];
    callback(&PcuDispatchKernelIr {
        numerical_requirements: PcuDispatchKernelIr::DEFAULT_REQUIREMENTS,
        id: PcuKernelId(80),
        entry: PcuDispatchEntryPoint {
            name: "integer",
            logical_shape: [if profile.grid { 2 } else { 17 }, 1, 1],
        },
        bindings: &bindings,
        ports: &[],
        parameters: &[],
        ops: if profile.grid { &grid } else { &direct },
        type_caps: PcuValueTypeCaps::for_scalar(T::TYPE),
        feature_caps: PcuDispatchFeatureCaps::default(),
    })
}

const fn profile(op: PcuDispatchIntegerBinaryOp) -> Profile {
    Profile {
        op,
        grid: false,
        broadcast: [false; 2],
        order: [0, 1],
        operands: [0, 1],
    }
}

fn prepare<T: PcuCheckedInteger>(profile: Profile) -> PcuCpuPreparedInteger<T> {
    with_kernel::<T, _>(profile, |kernel| {
        PcuCpuCheckedInteger::<T>::new()
            .prepare_host_kernel(kernel)
            .unwrap()
    })
}

fn call<T: PcuCheckedInteger>(
    prepared: &mut PcuCpuPreparedInteger<T>,
    lhs: &[T],
    rhs: &[T],
    output: &mut [T],
) -> Result<(), PcuCpuCheckedIntegerError> {
    prepared.call(&mut [
        PcuHostArgument::read(PcuBindingRef::new(0, 0), lhs),
        PcuHostArgument::read(PcuBindingRef::new(0, 1), rhs),
        PcuHostArgument::read_write(PcuBindingRef::new(0, 2), output),
    ])
}

#[test]
fn detached_grid_broadcasts_and_actual_swapped_repeated_operands() {
    for grid in [false, true] {
        for order in [[0, 1], [1, 0]] {
            for operands in [[0, 1], [1, 0], [0, 0], [1, 1]] {
                for broadcast in [[false, false], [true, false], [false, true], [true, true]] {
                    let profile = Profile {
                        op: PcuDispatchIntegerBinaryOp::Sub,
                        grid,
                        broadcast,
                        order,
                        operands,
                    };
                    let mut prepared = prepare::<i64>(profile);
                    let lhs: [i64; 17] =
                        core::array::from_fn(|index| 90 + i64::try_from(index).unwrap());
                    let rhs: [i64; 17] =
                        core::array::from_fn(|index| 3 + i64::try_from(index).unwrap());
                    let inputs = [
                        &lhs[..if broadcast[0] { 1 } else { 17 }],
                        &rhs[..if broadcast[1] { 1 } else { 17 }],
                    ];
                    let mut output = [41_i64; 19];
                    call(&mut prepared, inputs[0], inputs[1], &mut output).unwrap();
                    let expected: [i64; 17] = core::array::from_fn(|invocation| {
                        let operand = |slot: usize| {
                            let binding = order[operands[slot]];
                            inputs[binding][if broadcast[binding] { 0 } else { invocation }]
                        };
                        operand(0) - operand(1)
                    });
                    assert_eq!(&output[..17], &expected);
                    assert_eq!(&output[17..], &[41; 2]);
                }
            }
        }
    }
}

#[test]
fn grid_first_logical_fault_and_typed_reference_retry() {
    let mut definition = profile(PcuDispatchIntegerBinaryOp::Add);
    definition.grid = true;
    with_kernel::<u64, _>(definition, |kernel| {
        let mut lhs = [1_u64; 17];
        lhs[3] = u64::MAX;
        lhs[4] = u64::MAX;
        let rhs = [1_u64; 17];
        let mut output = [55_u64; 19];
        let mut prepared = PcuCpuCheckedInteger::<u64>::new()
            .prepare_host_kernel(kernel)
            .unwrap();
        assert_eq!(
            call(&mut prepared, &lhs, &rhs, &mut output),
            Err(PcuCpuCheckedIntegerError::Fault(PcuExecutionFault {
                recovered: false,
                kind: PcuExecutionFaultKind::ArithmeticOverflow,
                invocation_id: 3
            }))
        );
        assert_eq!(output, [55; 19]);
        let submission = PcuDispatchSubmission {
            kernel,
            shape: PcuInvocationShape::invocations(core::num::NonZeroU32::new(2).unwrap()),
        };
        let reference = PcuCheckedIntegerReference::<u64>::new();
        let run = |lhs: &[u64], output: &mut [u64]| {
            reference.run_host_direct(
                submission,
                &mut [
                    PcuHostScalarBinding {
                        target: PcuBindingRef::new(0, 0),
                        slice: PcuHostScalarSlice::Read(lhs),
                    },
                    PcuHostScalarBinding {
                        target: PcuBindingRef::new(0, 1),
                        slice: PcuHostScalarSlice::Read(&rhs),
                    },
                    PcuHostScalarBinding {
                        target: PcuBindingRef::new(0, 2),
                        slice: PcuHostScalarSlice::ReadWrite(output),
                    },
                ],
                PcuInvocationParameters::empty(),
            )
        };
        assert_eq!(
            run(&lhs, &mut output),
            Err(PcuCpuCheckedIntegerError::Fault(PcuExecutionFault {
                recovered: false,
                kind: PcuExecutionFaultKind::ArithmeticOverflow,
                invocation_id: 3
            }))
        );
        assert_eq!(output, [55; 19]);
        lhs.fill(1);
        run(&lhs, &mut output).unwrap();
        assert_eq!(&output[..17], &[2; 17]);
        assert_eq!(&output[17..], &[55; 2]);
    });
}

#[test]
fn signed_minimum_times_minus_one_and_exact_wide_bits() {
    macro_rules! signed {
        ($ty:ty) => {{
            let mut prepared = prepare::<$ty>(profile(PcuDispatchIntegerBinaryOp::Mul));
            assert_eq!(
                call(&mut prepared, &[<$ty>::MIN; 17], &[-1; 17], &mut [0; 17]),
                Err(PcuCpuCheckedIntegerError::Fault(PcuExecutionFault {
                    recovered: false,
                    kind: PcuExecutionFaultKind::ArithmeticOverflow,
                    invocation_id: 0
                }))
            );
        }};
    }
    signed!(i8);
    signed!(i16);
    signed!(i32);
    signed!(i64);
    let mut prepared = prepare::<u64>(profile(PcuDispatchIntegerBinaryOp::Sub));
    let mut output = [0_u64; 19];
    call(
        &mut prepared,
        &[u64::MAX; 17],
        &[0x1234_5678_9abc_def0; 17],
        &mut output,
    )
    .unwrap();
    assert_eq!(&output[..17], &[u64::MAX - 0x1234_5678_9abc_def0; 17]);
    let mut prepared = prepare::<i64>(profile(PcuDispatchIntegerBinaryOp::Add));
    let mut output = [0_i64; 17];
    call(
        &mut prepared,
        &[i64::MIN; 17],
        &[0x1234_5678_9abc_def0; 17],
        &mut output,
    )
    .unwrap();
    assert_eq!(output, [i64::MIN + 0x1234_5678_9abc_def0; 17]);
}

#[test]
fn all_schemas_validate_before_writes_and_reordered_arguments_work() {
    let mut prepared = prepare::<u32>(profile(PcuDispatchIntegerBinaryOp::Add));
    let lhs = [1_u32; 17];
    let rhs = [2_u32; 17];
    let mut output = [71_u32; 19];
    for (left, right, output_len) in [
        (&lhs[..16], &rhs[..], 19),
        (&lhs[..], &rhs[..16], 19),
        (&lhs[..], &rhs[..], 16),
    ] {
        assert_eq!(
            call(&mut prepared, left, right, &mut output[..output_len]),
            Err(PcuCpuCheckedIntegerError::InvalidArguments)
        );
        assert_eq!(output, [71; 19]);
    }
    assert_eq!(
        prepared.call(&mut [
            PcuHostArgument::read(PcuBindingRef::new(0, 0), &lhs),
            PcuHostArgument::read(PcuBindingRef::new(0, 0), &rhs),
            PcuHostArgument::read_write(PcuBindingRef::new(0, 2), &mut output),
        ]),
        Err(PcuCpuCheckedIntegerError::InvalidArguments)
    );
    assert_eq!(
        prepared.call(&mut [
            PcuHostArgument::read(PcuBindingRef::new(0, 0), &[1_i32; 17]),
            PcuHostArgument::read(PcuBindingRef::new(0, 1), &rhs),
            PcuHostArgument::read_write(PcuBindingRef::new(0, 2), &mut output),
        ]),
        Err(PcuCpuCheckedIntegerError::InvalidArguments)
    );
    assert_eq!(
        prepared.call(&mut [
            PcuHostArgument::read(PcuBindingRef::new(0, 0), &lhs),
            PcuHostArgument::read(PcuBindingRef::new(0, 1), &rhs),
            PcuHostArgument::read(PcuBindingRef::new(0, 2), &lhs),
        ]),
        Err(PcuCpuCheckedIntegerError::InvalidArguments)
    );
    assert_eq!(output, [71; 19]);
    let mut mutable_lhs = lhs;
    assert_eq!(
        prepared.call(&mut [
            PcuHostArgument::read_write(PcuBindingRef::new(0, 0), &mut mutable_lhs),
            PcuHostArgument::read(PcuBindingRef::new(0, 1), &rhs),
            PcuHostArgument::read_write(PcuBindingRef::new(0, 2), &mut output),
        ]),
        Err(PcuCpuCheckedIntegerError::InvalidArguments)
    );
    assert_eq!(output, [71; 19]);
    prepared
        .call(&mut [
            PcuHostArgument::read_write(PcuBindingRef::new(0, 2), &mut output),
            PcuHostArgument::read(PcuBindingRef::new(0, 1), &rhs),
            PcuHostArgument::read(PcuBindingRef::new(0, 0), &lhs),
        ])
        .unwrap();
    assert_eq!(&output[..17], &[3; 17]);
    assert_eq!(&output[17..], &[71; 2]);
}

#[test]
fn signed_all_width_underflow_overflow_are_distinct_and_transactional() {
    macro_rules! cases {
        ($ty:ty) => {{
            for (op, lhs, rhs, kind) in [
                (
                    PcuDispatchIntegerBinaryOp::Add,
                    <$ty>::MIN,
                    -1,
                    PcuExecutionFaultKind::ArithmeticUnderflow,
                ),
                (
                    PcuDispatchIntegerBinaryOp::Sub,
                    <$ty>::MAX,
                    -1,
                    PcuExecutionFaultKind::ArithmeticOverflow,
                ),
                (
                    PcuDispatchIntegerBinaryOp::Mul,
                    <$ty>::MIN,
                    2,
                    PcuExecutionFaultKind::ArithmeticUnderflow,
                ),
            ] {
                let mut prepared = prepare::<$ty>(profile(op));
                let mut left = [1; 17];
                let mut right = [1; 17];
                left[9] = lhs;
                right[9] = rhs;
                let mut output = [17; 19];
                assert_eq!(
                    call(&mut prepared, &left, &right, &mut output),
                    Err(PcuCpuCheckedIntegerError::Fault(PcuExecutionFault {
                        recovered: false,
                        kind,
                        invocation_id: 9
                    }))
                );
                assert_eq!(output, [17; 19]);
                left.fill(1);
                right.fill(1);
                call(&mut prepared, &left, &right, &mut output).unwrap();
                assert_eq!(&output[..17], &[native::<$ty>(op, 1, 1).unwrap(); 17]);
                assert_eq!(&output[17..], &[17; 2]);
            }
        }};
    }
    cases!(i8);
    cases!(i16);
    cases!(i32);
    cases!(i64);
}

#[test]
fn cold_admission_rejects_malformed_ssa_and_schemas() {
    with_kernel::<u32, _>(profile(PcuDispatchIntegerBinaryOp::Sub), |kernel| {
        let backend = PcuCpuCheckedInteger::<u32>::new();
        let mut operations = kernel.ops.to_vec();
        let PcuDispatchOp::Data(PcuDispatchDataOp::CheckedIntegerBinary { rhs, .. }) =
            &mut operations[2]
        else {
            unreachable!()
        };
        *rhs = PcuDispatchValueId(99);
        let invalid = PcuDispatchKernelIr {
            ops: &operations,
            ..*kernel
        };
        assert!(backend.prepare_host_kernel(&invalid).is_err());
        let mut bindings = kernel.bindings.to_vec();
        bindings[1] = bindings[0];
        let invalid = PcuDispatchKernelIr {
            bindings: &bindings,
            ..*kernel
        };
        assert!(backend.prepare_host_kernel(&invalid).is_err());
        bindings.copy_from_slice(kernel.bindings);
        bindings[0].access = PcuBindingAccess::ReadWrite;
        let invalid = PcuDispatchKernelIr {
            bindings: &bindings,
            ..*kernel
        };
        assert!(backend.prepare_host_kernel(&invalid).is_err());
        let mut invalid = *kernel;
        invalid.entry.logical_shape = [17, 2, 1];
        assert!(backend.prepare_host_kernel(&invalid).is_err());
        assert!(
            PcuCpuCheckedInteger::<i32>::new()
                .prepare_host_kernel(kernel)
                .is_err()
        );
    });
    let mut definition = profile(PcuDispatchIntegerBinaryOp::Add);
    definition.operands = [0, 0];
    let mut prepared = prepare::<u64>(definition);
    let mut output = [17; 19];
    // Both input loads remain part of the admitted schema even if arithmetic repeats one value.
    assert_eq!(
        call(&mut prepared, &[1; 17], &[], &mut output),
        Err(PcuCpuCheckedIntegerError::InvalidArguments)
    );
    assert_eq!(output, [17; 19]);
}

const fn device(generation: u64) -> PcuDeviceIdentity {
    PcuDeviceIdentity::from_device_ref(PcuObjectRef {
        provider: PcuProviderId(3),
        generation,
        kind: PcuObjectKind::Device,
        id: 0,
    })
    .unwrap()
}

fn check_offers<T: PcuCheckedInteger>(ids: &mut std::vec::Vec<u32>) {
    for op in [
        PcuDispatchIntegerBinaryOp::Add,
        PcuDispatchIntegerBinaryOp::Sub,
        PcuDispatchIntegerBinaryOp::Mul,
    ] {
        with_kernel::<T, _>(profile(op), |kernel| {
            let offers = PcuCpuIntegerOffers::<T>::new(device(1), PcuExecutorId(0));
            let mut request = PcuImplementationRequest {
                device: device(1),
                executor: PcuExecutorId(0),
                requirements: PcuImplementationRequirements::default(),
                boundary: PcuCostBoundary::Host,
                operation: kernel,
            };
            for mode in [PcuNumericalMode::Boundary, PcuNumericalMode::Strict] {
                request.requirements.numerical_mode = mode;
                let kernel = PcuDispatchKernelIr {
                    numerical_requirements: request.requirements,
                    ..*request.operation
                };
                let request = PcuImplementationRequest {
                    operation: &kernel,
                    ..request
                };
                let mut output = [None; 2];
                assert_eq!(offers.implementation_offers(&request, &mut output), Ok(1));
                let offer = output[0].unwrap();
                assert_eq!(offer.requirements, request.requirements);
                assert_eq!(offer.workspace_bytes, Some(0));
                assert_eq!(
                    offer.cost,
                    PcuImplementationCost::unknown(PcuCostBoundary::Host)
                );
                assert_eq!(output[1], None);
                if mode == PcuNumericalMode::Boundary {
                    ids.push(offer.implementation.local_id);
                }
            }
            request.requirements.numerical_mode = PcuNumericalMode::Boundary;
            assert_eq!(offers.implementation_offers(&request, &mut []), Ok(1));
            let mut output = [None];
            request.boundary = PcuCostBoundary::Resident;
            assert_eq!(offers.implementation_offers(&request, &mut output), Ok(0));
            request.boundary = PcuCostBoundary::Host;
            request.requirements.range_policy = PcuRangePolicy::Clamp;
            assert_eq!(offers.implementation_offers(&request, &mut output), Ok(0));
            request.requirements.range_policy = PcuRangePolicy::Reject;
            request.requirements.numerical_options.reproducibility = PcuReproducibility::PortableV1;
            assert_eq!(offers.implementation_offers(&request, &mut output), Ok(0));
            request.requirements = PcuImplementationRequirements::default();
            request.device = device(2);
            assert_eq!(
                offers.implementation_offers(&request, &mut output),
                Err(PcuCpuIntegerOfferError::DeviceMismatch)
            );
            request.device = device(1);
            request.executor = PcuExecutorId(1);
            assert_eq!(
                offers.implementation_offers(&request, &mut output),
                Err(PcuCpuIntegerOfferError::ExecutorMismatch)
            );
            assert_eq!(output, [None]);
            request.executor = PcuExecutorId(0);
            let mut operations = kernel.ops.to_vec();
            let PcuDispatchOp::Data(PcuDispatchDataOp::CheckedIntegerBinary { rhs, .. }) =
                &mut operations[2]
            else {
                unreachable!()
            };
            *rhs = PcuDispatchValueId(99);
            let malformed = PcuDispatchKernelIr {
                ops: &operations,
                ..*kernel
            };
            request.operation = &malformed;
            assert!(matches!(
                offers.implementation_offers(&request, &mut output),
                Err(PcuCpuIntegerOfferError::Provider(_))
            ));
            assert_eq!(output, [None]);
        });
    }
}

#[test]
fn integer_offers_have_distinct_ids_exact_snapshots_and_unknown_host_costs() {
    let mut ids = std::vec::Vec::new();
    check_offers::<i8>(&mut ids);
    check_offers::<u8>(&mut ids);
    check_offers::<i16>(&mut ids);
    check_offers::<u16>(&mut ids);
    check_offers::<i32>(&mut ids);
    check_offers::<u32>(&mut ids);
    check_offers::<i64>(&mut ids);
    check_offers::<u64>(&mut ids);
    assert_eq!(ids, (4..28).collect::<std::vec::Vec<_>>());
}
