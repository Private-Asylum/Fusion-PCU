use super::*;
#[rustfmt::skip]
use fusion_pcu::{
    PcuBinding,
    PcuBindingAccess,
    PcuBindingRef,
    PcuBindingStorageClass,
    PcuBindingType,
    PcuCompletionOutcome,
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
    PcuExecutionFaultKind,
    PcuInvocationShape,
    PcuOwnedCompletion,
    PcuParameter,
    PcuPort,
    PcuCheckedInteger,
    PcuScalarType,
    PcuValueType,
    PcuValueTypeCaps,
    PcuProviderDescriptor,
    PcuProviderId,
    PcuProviderReadiness,
    PcuProviderStatus,
    PcuTargetDescriptor,
    PcuDeviceDescriptor,
    PcuObjectRef,
    PcuObjectKind,
    PcuDeviceClass,
    PcuRuntimeDiscovery,
};
use std::num::NonZeroU32;

trait TestInteger: PcuCheckedInteger + Copy + PartialEq + core::fmt::Debug {
    const SCALAR: PcuScalarType;
    const MIN: Self;
    const MAX: Self;
    const ZERO: Self;
    const ONE: Self;
    const TWO: Self;
    const NEGATIVE_ONE: Option<Self>;
    fn append_ne_bytes(self, output: &mut Vec<u8>);
    fn decode_ne_bytes(input: &[u8]) -> Vec<Self>;
}

macro_rules! impl_test_integer {
    ($ty:ty, $scalar:ident, $negative_one:expr) => {
        impl TestInteger for $ty {
            const SCALAR: PcuScalarType = PcuScalarType::$scalar;
            const MIN: Self = <$ty>::MIN;
            const MAX: Self = <$ty>::MAX;
            const ZERO: Self = 0;
            const ONE: Self = 1;
            const TWO: Self = 2;
            const NEGATIVE_ONE: Option<Self> = $negative_one;

            fn append_ne_bytes(self, output: &mut Vec<u8>) {
                output.extend_from_slice(&self.to_ne_bytes());
            }

            fn decode_ne_bytes(input: &[u8]) -> Vec<Self> {
                input
                    .chunks_exact(core::mem::size_of::<Self>())
                    .map(|chunk| {
                        Self::from_ne_bytes(chunk.try_into().expect("one complete scalar"))
                    })
                    .collect()
            }
        }
    };
}

impl_test_integer!(u8, U8, None);
impl_test_integer!(u16, U16, None);
impl_test_integer!(u32, U32, None);
impl_test_integer!(u64, U64, None);
impl_test_integer!(i8, I8, Some(-1));
impl_test_integer!(i16, I16, Some(-1));
impl_test_integer!(i32, I32, Some(-1));
impl_test_integer!(i64, I64, Some(-1));

pub(super) fn selected_device() -> (RocmDiscovery, RocmOwnedDispatchBackend) {
    let discovery = RocmDiscovery::new();
    let invalid = PcuObjectRef {
        provider: PcuProviderId(0),
        generation: 0,
        kind: PcuObjectKind::Device,
        id: 0,
    };
    let mut providers = [PcuProviderDescriptor {
        id: PcuProviderId(0),
        generation: 0,
        backend: "",
        readiness: PcuProviderReadiness {
            status: PcuProviderStatus::Unavailable,
            reason: None,
        },
    }];
    assert_eq!(discovery.providers(&mut providers).unwrap(), 1);
    let mut targets = [PcuTargetDescriptor {
        reference: invalid,
        name: "",
        readiness: PcuProviderReadiness {
            status: PcuProviderStatus::Unavailable,
            reason: None,
        },
    }];
    assert_eq!(
        discovery
            .targets(providers[0].id, providers[0].generation, &mut targets)
            .unwrap(),
        1
    );
    let count = discovery.devices(targets[0].reference, &mut []).unwrap();
    assert!(count > 0, "test requires a visible ROCm device");
    let mut devices = vec![
        PcuDeviceDescriptor {
            reference: invalid,
            target: invalid,
            name: "",
            class: PcuDeviceClass::Other,
            vendor: None,
            architecture: None,
            generation: None,
            location: None,
        };
        count
    ];
    discovery
        .devices(targets[0].reference, &mut devices)
        .unwrap();
    let session = RocmOwnedDispatchBackend::open(&discovery, devices[0].reference, 2)
        .expect("open selected ROCm device");
    (discovery, session)
}

fn encoded<T: TestInteger>(values: &[T]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(core::mem::size_of_val(values));
    for &value in values {
        value.append_ne_bytes(&mut bytes);
    }
    bytes
}

fn prepare<T: TestInteger>(
    session: &RocmOwnedDispatchBackend,
    operation: PcuDispatchIntegerBinaryOp,
    extent: u32,
    grid_stride: bool,
) -> RocmPreparedDispatch {
    const LEFT: PcuBindingRef = PcuBindingRef::new(0, 0);
    const RIGHT: PcuBindingRef = PcuBindingRef::new(0, 1);
    const OUTPUT: PcuBindingRef = PcuBindingRef::new(0, 2);
    const LEFT_VALUE: PcuDispatchValueId = PcuDispatchValueId(1);
    const RIGHT_VALUE: PcuDispatchValueId = PcuDispatchValueId(2);
    const RESULT: PcuDispatchValueId = PcuDispatchValueId(3);
    let scalar_type = PcuValueType::Scalar(T::SCALAR);
    let bindings = [
        PcuBinding::value(
            Some("left"),
            0,
            0,
            PcuBindingStorageClass::Storage,
            PcuBindingAccess::ReadOnly,
            scalar_type,
        ),
        PcuBinding::value(
            Some("right"),
            0,
            1,
            PcuBindingStorageClass::Storage,
            PcuBindingAccess::ReadOnly,
            scalar_type,
        ),
        PcuBinding::value(
            Some("output"),
            0,
            2,
            PcuBindingStorageClass::Storage,
            PcuBindingAccess::WriteOnly,
            scalar_type,
        ),
    ];
    let index = if grid_stride {
        PcuDispatchIndex::GridStrideId
    } else {
        PcuDispatchIndex::InvocationId
    };
    let body = [
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
            result: LEFT_VALUE,
            binding: LEFT,
            index,
        }),
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
            result: RIGHT_VALUE,
            binding: RIGHT,
            index: PcuDispatchIndex::BindingElementZero,
        }),
        PcuDispatchOp::Data(PcuDispatchDataOp::CheckedIntegerBinary {
            value_type: scalar_type,
            op: operation,
            result: RESULT,
            lhs: LEFT_VALUE,
            rhs: RIGHT_VALUE,
        }),
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
            binding: OUTPUT,
            index,
            value: RESULT,
        }),
        PcuDispatchOp::Control(PcuDispatchControlOp::Return),
    ];
    let grid_loop = [
        PcuDispatchOp::GridStrideLoop {
            extent,
            body: &body[..4],
        },
        PcuDispatchOp::Control(PcuDispatchControlOp::Return),
    ];
    let ops = if grid_stride {
        &grid_loop[..]
    } else {
        &body[..]
    };
    let kernel = PcuDispatchKernelIr {
        id: fusion_pcu::PcuKernelId(91),
        entry: PcuDispatchEntryPoint {
            name: "checked_integer_map",
            logical_shape: [if grid_stride { 2 } else { extent }, 1, 1],
        },
        bindings: &bindings,
        ports: &[] as &[PcuPort<'_>],
        parameters: &[] as &[PcuParameter<'_>],
        ops,
        type_caps: PcuValueTypeCaps::for_scalar(T::SCALAR),
        feature_caps: PcuDispatchFeatureCaps::MUTABLE_RESOURCES
            .union(PcuDispatchFeatureCaps::READ_ONLY_RESOURCES),
    };
    let invocations =
        NonZeroU32::new(if grid_stride { 2 } else { extent }).expect("positive test extent");
    session
        .prepare_dispatch(PcuDispatchSubmission {
            kernel: &kernel,
            shape: PcuInvocationShape::invocations(invocations),
        })
        .expect("prepare checked integer kernel")
}

fn buffers<T: TestInteger>(
    session: &RocmOwnedDispatchBackend,
    left: &[T],
    right: T,
) -> (DeviceBuffer, DeviceBuffer, DeviceBuffer) {
    let mut lhs = session
        .allocate(core::mem::size_of_val(left))
        .expect("allocate left values");
    lhs.copy_from(&encoded(left)).expect("upload left values");
    let mut rhs = session
        .allocate(core::mem::size_of::<T>())
        .expect("allocate broadcast right value");
    rhs.copy_from(&encoded(&[right]))
        .expect("upload right value");
    let output = session
        .allocate(core::mem::size_of_val(left))
        .expect("allocate output values");
    (lhs, rhs, output)
}

fn bindings<T: TestInteger>(
    session: &RocmOwnedDispatchBackend,
    values: &(DeviceBuffer, DeviceBuffer, DeviceBuffer),
) -> [PcuOwnedBinding<DeviceBuffer>; 3] {
    [
        session
            .binding(
                PcuBindingRef::new(0, 0),
                PcuBindingAccess::ReadOnly,
                PcuBindingType::Value(PcuValueType::Scalar(T::SCALAR)),
                values.0.clone(),
            )
            .unwrap(),
        session
            .binding(
                PcuBindingRef::new(0, 1),
                PcuBindingAccess::ReadOnly,
                PcuBindingType::Value(PcuValueType::Scalar(T::SCALAR)),
                values.1.clone(),
            )
            .unwrap(),
        session
            .binding(
                PcuBindingRef::new(0, 2),
                PcuBindingAccess::WriteOnly,
                PcuBindingType::Value(PcuValueType::Scalar(T::SCALAR)),
                values.2.clone(),
            )
            .unwrap(),
    ]
}

fn submit<T: TestInteger>(
    session: &RocmOwnedDispatchBackend,
    prepared: &RocmPreparedDispatch,
    values: &(DeviceBuffer, DeviceBuffer, DeviceBuffer),
    fault_word: &mut DeviceBuffer,
) -> PcuCompletionOutcome {
    let bindings = bindings::<T>(session, values);
    let mut completion = prepared
        .submit_with_fault_word(&bindings, fault_word)
        .expect("submit checked integer kernel");
    completion.wait().expect("wait for checked completion")
}

fn assert_fault<T: TestInteger>(
    session: &RocmOwnedDispatchBackend,
    operation: PcuDispatchIntegerBinaryOp,
    grid_stride: bool,
    left: &[T],
    right: T,
    expected_kind: PcuExecutionFaultKind,
    expected_invocation: u64,
) {
    let extent = u32::try_from(left.len()).expect("test extent fits the ROCm dispatch ABI");
    let prepared = prepare::<T>(session, operation, extent, grid_stride);
    let values = buffers(session, left, right);
    let mut fault_word = session.allocate(core::mem::size_of::<u64>()).unwrap();
    assert_eq!(
        submit::<T>(session, &prepared, &values, &mut fault_word),
        PcuCompletionOutcome::Fault(fusion_pcu::PcuExecutionFault {
            kind: expected_kind,
            invocation_id: expected_invocation,
            recovered: false,
        })
    );
}

fn exercise_valid_boundaries<T: TestInteger>(
    session: &RocmOwnedDispatchBackend,
) -> RocmPreparedDispatch {
    let extent = 5;
    let boundary_values = [T::MIN, T::MAX, T::ZERO, T::ONE, T::MAX];
    let prepared = prepare::<T>(session, PcuDispatchIntegerBinaryOp::Add, extent, false);
    let values = buffers(session, &boundary_values, T::ZERO);
    let mut fault_word = session.allocate(core::mem::size_of::<u64>()).unwrap();
    assert_eq!(
        submit::<T>(session, &prepared, &values, &mut fault_word),
        PcuCompletionOutcome::Succeeded
    );
    let mut output_bytes = vec![0; core::mem::size_of_val(&boundary_values)];
    values.2.copy_to(&mut output_bytes).unwrap();
    assert_eq!(T::decode_ne_bytes(&output_bytes), boundary_values);

    for operation in [
        PcuDispatchIntegerBinaryOp::Sub,
        PcuDispatchIntegerBinaryOp::Mul,
    ] {
        let rhs = if operation == PcuDispatchIntegerBinaryOp::Mul {
            T::ONE
        } else {
            T::ZERO
        };
        let executable = prepare::<T>(session, operation, extent, true);
        let resources = buffers(session, &boundary_values, rhs);
        assert_eq!(
            submit::<T>(session, &executable, &resources, &mut fault_word),
            PcuCompletionOutcome::Succeeded
        );
        resources.2.copy_to(&mut output_bytes).unwrap();
        assert_eq!(T::decode_ne_bytes(&output_bytes), boundary_values);
    }

    prepared
}

fn exercise_integer<T: TestInteger>(session: &RocmOwnedDispatchBackend) {
    let prepared = exercise_valid_boundaries::<T>(session);

    let mut overflow = [T::ZERO; 5];
    overflow[2] = T::MAX;
    overflow[4] = T::MAX;
    assert_fault(
        session,
        PcuDispatchIntegerBinaryOp::Add,
        true,
        &overflow,
        T::ONE,
        PcuExecutionFaultKind::ArithmeticOverflow,
        2,
    );

    let mut underflow = [T::ONE; 5];
    underflow[1] = T::MIN;
    underflow[3] = T::MIN;
    assert_fault(
        session,
        PcuDispatchIntegerBinaryOp::Sub,
        true,
        &underflow,
        T::ONE,
        PcuExecutionFaultKind::ArithmeticUnderflow,
        1,
    );

    let mut multiply_overflow = [T::ZERO; 5];
    multiply_overflow[1] = T::MAX;
    assert_fault(
        session,
        PcuDispatchIntegerBinaryOp::Mul,
        false,
        &multiply_overflow,
        T::TWO,
        PcuExecutionFaultKind::ArithmeticOverflow,
        1,
    );

    if let Some(negative_one) = T::NEGATIVE_ONE {
        let mut signed_overflow = [T::ZERO; 5];
        signed_overflow[2] = T::MIN;
        assert_fault(
            session,
            PcuDispatchIntegerBinaryOp::Add,
            false,
            &signed_overflow,
            negative_one,
            PcuExecutionFaultKind::ArithmeticUnderflow,
            2,
        );
        assert_fault(
            session,
            PcuDispatchIntegerBinaryOp::Mul,
            false,
            &signed_overflow,
            negative_one,
            PcuExecutionFaultKind::ArithmeticOverflow,
            2,
        );
        assert_fault(
            session,
            PcuDispatchIntegerBinaryOp::Sub,
            true,
            &overflow,
            negative_one,
            PcuExecutionFaultKind::ArithmeticOverflow,
            2,
        );
        assert_fault(
            session,
            PcuDispatchIntegerBinaryOp::Mul,
            true,
            &multiply_overflow.map(|value| if value == T::MAX { T::MIN } else { value }),
            T::TWO,
            PcuExecutionFaultKind::ArithmeticUnderflow,
            1,
        );
    }

    exercise_retry::<T>(session, &prepared);
}

fn exercise_retry<T: TestInteger>(
    session: &RocmOwnedDispatchBackend,
    prepared: &RocmPreparedDispatch,
) {
    // A faulted completion must not poison a prepared executable or its caller-owned status word.
    let mut retry_left = [T::ZERO; 5];
    retry_left[0] = T::MAX;
    let retry_rhs = T::ONE;
    let mut retry_values = buffers(session, &retry_left, retry_rhs);
    let mut reusable_fault_word = session.allocate(core::mem::size_of::<u64>()).unwrap();
    assert_eq!(
        submit::<T>(session, prepared, &retry_values, &mut reusable_fault_word),
        PcuCompletionOutcome::Fault(fusion_pcu::PcuExecutionFault {
            kind: PcuExecutionFaultKind::ArithmeticOverflow,
            invocation_id: 0,
            recovered: false,
        })
    );
    retry_left.fill(T::ZERO);
    retry_values.0.copy_from(&encoded(&retry_left)).unwrap();
    assert_eq!(
        submit::<T>(session, prepared, &retry_values, &mut reusable_fault_word),
        PcuCompletionOutcome::Succeeded
    );
    let mut retried_output = vec![0; core::mem::size_of_val(&retry_left)];
    retry_values.2.copy_to(&mut retried_output).unwrap();
    assert_eq!(T::decode_ne_bytes(&retried_output), vec![T::ONE; 5]);

    // The sequential owner reuses the terminal success sentinel, then resets after a fault.
    let mut sequential = prepared.sequential_checked().unwrap();
    retry_left[0] = T::MAX;
    retry_values.0.copy_from(&encoded(&retry_left)).unwrap();
    assert!(matches!(
        sequential.submit_and_wait(&bindings::<T>(session, &retry_values)),
        Ok(PcuCompletionOutcome::Fault(_))
    ));
    retry_left.fill(T::ZERO);
    retry_values.0.copy_from(&encoded(&retry_left)).unwrap();
    assert_eq!(
        sequential
            .submit_and_wait(&bindings::<T>(session, &retry_values))
            .unwrap(),
        PcuCompletionOutcome::Succeeded
    );
}

#[test]
#[ignore = "requires an explicitly visible ROCm device and HIPRTC"]
fn checked_integer_widths_report_exact_results_faults_and_retry_on_a_real_device() {
    let (_discovery, session) = selected_device();
    exercise_integer::<u8>(&session);
    exercise_integer::<u16>(&session);
    exercise_integer::<u32>(&session);
    exercise_integer::<u64>(&session);
    exercise_integer::<i8>(&session);
    exercise_integer::<i16>(&session);
    exercise_integer::<i32>(&session);
    exercise_integer::<i64>(&session);
}
