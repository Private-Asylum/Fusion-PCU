//! Actual requested `PortableV1` header, exact-byte broadcasts and logical publication.
#[rustfmt::skip]
use pcu_facade::{
    pcu,
    PcuCheckedFloat,
    PcuF16Bits,
    PcuBf16Bits,
    PcuF8E4M3FnBits,
    PcuF8E5M2Bits,
    PcuTensor,
    PcuMemoryPoolId,
    PcuExecutionError,
    PcuExecutionFaultKind,
    PcuArgumentError,
};
use fusion_pcu_metal::MetalSession;
use super::low_precision::{Sample, assert_bits, open_backend};
#[path = "../../../benches/checked_portable/source.rs"]
mod source;
#[pcu(invocations=3,flag(deterministic),crate_path=::pcu_facade)]
fn broadcast_left<T: PcuCheckedFloat>(left: &T, right: &[T], output: &mut [T]) {
    let id = pcu::context::global_invocation_id();
    output[id] = *left / right[id];
}
#[pcu(invocations=3,flag(deterministic),flag(strict),flag(native_compound),flag(backend_precision),crate_path=::pcu_facade)]
fn broadcast_right<T: PcuCheckedFloat>(left: &[T], right: &T, output: &mut [T]) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[id] - *right;
}
#[pcu(invocations=3,flag(deterministic),crate_path=::pcu_facade)]
fn both_broadcast<T: PcuCheckedFloat>(left: &T, right: &T, output: &mut [T]) {
    let id = pcu::context::global_invocation_id();
    output[id] = *right / *left;
}
#[pcu(invocations=3,flag(deterministic),flag(allow_gradual_underflow),crate_path=::pcu_facade)]
fn gradual<T: PcuCheckedFloat>(left: &[T], right: &[T], output: &mut [T]) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[id] / right[id];
}
#[pcu(invocations=3,flag(deterministic),flag(reject_subnormal_result),crate_path=::pcu_facade)]
fn tight<T: PcuCheckedFloat>(left: &[T], right: &[T], output: &mut [T]) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[id] + right[id];
}
#[pcu(invocations=3,flag(deterministic),flag(clamp_range),crate_path=::pcu_facade)]
fn clamp<T: PcuCheckedFloat>(left: &[T], right: &[T], output: &mut [T]) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[id] * right[id];
}
#[pcu(invocations=3,flag(deterministic),crate_path=::pcu_facade)]
fn chain<T: PcuCheckedFloat>(left: &[T], right: &[T], output: &mut [T]) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[id] * right[id] + right[id];
}
#[allow(clippy::too_many_lines)] // Full requested-header lifecycle has distinct operation/policy/refusal cohorts.
fn host<T: Sample>() {
    let session = MetalSession::open(0).unwrap();
    let left = [-0.0, 1.0, -1.0].map(T::value);
    let right = [1.0, 2.0, 2.0].map(T::value);
    let sentinel = T::value(91.0);
    let mut output = [sentinel; 5];
    let mut add = source::add_prepare::<T, 3, _>(&session).unwrap();
    let mut sub = source::sub_prepare::<T, 3, _>(&session).unwrap();
    let mut mul = source::mul_prepare::<T, 3, _>(&session).unwrap();
    let mut div = source::div_prepare::<T, 3, _>(&session).unwrap();
    pcu_facade::global::configure(pcu_facade::global::PcuExecutionPolicy {
        backend: pcu_facade::global::PcuBackendChoice::Metal,
        ..pcu_facade::global::PcuExecutionPolicy::default()
    })
    .unwrap();
    for op in 0..4 {
        match op {
            0 => add(&left, &right, &mut output),
            1 => sub(&left, &right, &mut output),
            2 => mul(&left, &right, &mut output),
            _ => div(&left, &right, &mut output),
        }
        .unwrap();
        let mut expected = [sentinel; 5];
        for index in 0..3 {
            expected[index] = match op {
                0 => left[index].pcu_checked_add(right[index]),
                1 => left[index].pcu_checked_sub(right[index]),
                2 => left[index].pcu_checked_mul(right[index]),
                _ => left[index].pcu_checked_div(right[index]),
            }
            .unwrap();
        }
        assert_bits(&output, &expected);
        for _ in 0..3 {
            match op {
                0 => source::add::<T, 3>(&left, &right, &mut output),
                1 => source::sub::<T, 3>(&left, &right, &mut output),
                2 => source::mul::<T, 3>(&left, &right, &mut output),
                _ => source::div::<T, 3>(&left, &right, &mut output),
            }
            .unwrap();
            assert_bits(&output, &expected);
        }
    }
    let before = output;
    assert!(
        matches!(source::div::<T,3>(&left,&[T::value(1.0),T::value(0.0),T::raw(T::NAN)],&mut output),
        Err(PcuExecutionError::ArithmeticFault(fault)) if fault.invocation_id==1 && fault.kind==PcuExecutionFaultKind::DivideByZero)
    );
    assert_bits(&output, &before);
    assert!(matches!(div(&[T::raw(1);3],&[T::value(2.0);3],&mut output),
        Err(pcu_facade::PcuHostDispatchError::Backend(fusion_pcu_metal::MetalError::Arithmetic(fault))) if fault.kind==PcuExecutionFaultKind::ArithmeticUnderflow));
    assert_bits(&output, &before);
    gradual_prepare::<T, _>(&session).unwrap()(&[T::raw(1); 3], &[T::value(2.0); 3], &mut output)
        .unwrap();
    assert_bits(
        &output,
        &[T::raw(0), T::raw(0), T::raw(0), sentinel, sentinel],
    );
    assert!(
        tight_prepare::<T, _>(&session).unwrap()(&[T::raw(1); 3], &[T::raw(0); 3], &mut output)
            .is_err()
    );
    assert!(clamp_prepare::<T, _>(&session).is_err());
    assert!(chain_prepare::<T, _>(&session).is_err());
    assert!(source::add_prepare::<f32, 3, _>(&session).is_err());
    assert!(source::add_prepare::<f64, 3, _>(&session).is_err());
    broadcast_left_prepare::<T, _>(&session).unwrap()(&T::value(2.0), &right, &mut output).unwrap();
    assert_bits(
        &output,
        &[
            T::value(2.0),
            T::value(1.0),
            T::value(1.0),
            sentinel,
            sentinel,
        ],
    );
    broadcast_right_prepare::<T, _>(&session).unwrap()(&left, &T::value(1.0), &mut output).unwrap();
    assert_bits(
        &output,
        &[
            T::value(-1.0),
            T::value(0.0),
            T::value(-2.0),
            sentinel,
            sentinel,
        ],
    );
    both_broadcast_prepare::<T, _>(&session).unwrap()(&T::value(2.0), &T::value(1.0), &mut output)
        .unwrap();
    assert_bits(
        &output,
        &[
            T::value(0.5),
            T::value(0.5),
            T::value(0.5),
            sentinel,
            sentinel,
        ],
    );
    broadcast_left::<T>(&T::value(2.0), &right, &mut output).unwrap();
    both_broadcast::<T>(&T::value(2.0), &T::value(1.0), &mut output).unwrap();
    pcu_facade::global::use_defaults().unwrap();
}
#[allow(clippy::too_many_lines)] // One retained owner lifecycle verifies preflight, write/discard, fresh retry and foreign affinity.
fn resident<T: Sample>() {
    let backend = open_backend();
    let pool = PcuMemoryPoolId(33);
    let left = backend
        .upload_buffer(pool, &[T::value(1.0), T::value(2.0), T::value(3.0)])
        .unwrap();
    let right = backend
        .upload_buffer(pool, &[T::value(2.0), T::value(4.0), T::value(6.0)])
        .unwrap();
    let scalar = backend.upload_buffer(pool, &[T::value(2.0)]).unwrap();
    let sentinel = T::value(91.0);
    let mut output = backend.upload_buffer(pool, &[sentinel; 5]).unwrap();
    let mut values = [T::raw(0); 5];
    source::div_prepare_device::<T, 3, _>(&backend).unwrap()(&left, &right, &mut output).unwrap();
    backend.download_buffer(pool, &output, &mut values).unwrap();
    assert_bits(
        &values,
        &[
            T::value(0.5),
            T::value(0.5),
            T::value(0.5),
            sentinel,
            sentinel,
        ],
    );
    broadcast_left_prepare_device::<T, _>(&backend).unwrap()(&scalar, &right, &mut output).unwrap();
    backend.download_buffer(pool, &output, &mut values).unwrap();
    assert_bits(
        &values,
        &[
            T::value(1.0),
            T::value(0.5),
            T::value(2.0).pcu_checked_div(T::value(6.0)).unwrap(),
            sentinel,
            sentinel,
        ],
    );
    let left = PcuTensor::from_device_buffer(backend.clone(), left, &[3]).unwrap();
    let scalar = PcuTensor::from_device_buffer(backend.clone(), scalar, &[]).unwrap();
    let mut output = PcuTensor::from_device_buffer(backend.clone(), output, &[5]).unwrap();
    pcu_facade::global::configure(pcu_facade::global::PcuExecutionPolicy {
        backend: pcu_facade::global::PcuBackendChoice::Metal,
        ..pcu_facade::global::PcuExecutionPolicy::default()
    })
    .unwrap();
    source::div::<T, 3>(
        &left,
        &[T::value(2.0), T::value(4.0), T::value(6.0)],
        &mut output,
    )
    .unwrap();
    output.read_into(&mut values).unwrap();
    assert_bits(
        &values,
        &[
            T::value(0.5),
            T::value(0.5),
            T::value(0.5),
            sentinel,
            sentinel,
        ],
    );
    broadcast_left::<T>(
        &scalar,
        &[T::value(1.0), T::value(2.0), T::value(4.0)],
        &mut output,
    )
    .unwrap();
    output.read_into(&mut values).unwrap();
    assert_bits(
        &values,
        &[
            T::value(2.0),
            T::value(1.0),
            T::value(0.5),
            sentinel,
            sentinel,
        ],
    );
    let before = values;
    assert!(broadcast_left::<T>(&scalar, &[T::value(1.0)], &mut output).is_err());
    output.read_into(&mut values).unwrap();
    assert_bits(&values, &before);
    assert!(
        matches!(broadcast_left::<T>(&scalar,&[T::value(1.0),T::value(0.0),T::raw(T::NAN)],&mut output),Err(PcuExecutionError::ArithmeticFault(fault)) if fault.invocation_id==1)
    );
    assert!(matches!(
        output.read_into(&mut values),
        Err(PcuExecutionError::Argument(
            PcuArgumentError::ResidentValueDiscarded
        ))
    ));
    assert!(matches!(
        source::div::<T, 3>(&output, &[T::value(1.0); 3], &mut values),
        Err(PcuExecutionError::Argument(
            PcuArgumentError::ResidentValueDiscarded
        ))
    ));
    let fresh = backend.upload_buffer(pool, &[sentinel; 5]).unwrap();
    let mut fresh = PcuTensor::from_device_buffer(backend.clone(), fresh, &[5]).unwrap();
    broadcast_left::<T>(&scalar, &[T::value(1.0); 3], &mut fresh).unwrap();
    fresh.read_into(&mut values).unwrap();
    assert_bits(
        &values,
        &[
            T::value(2.0),
            T::value(2.0),
            T::value(2.0),
            sentinel,
            sentinel,
        ],
    );
    let foreign_backend = open_backend();
    let foreign_buffer = foreign_backend.upload_buffer(pool, &[sentinel; 5]).unwrap();
    let mut foreign = PcuTensor::from_device_buffer(foreign_backend, foreign_buffer, &[5]).unwrap();
    assert!(matches!(
        broadcast_left::<T>(&scalar, &[T::value(1.0); 3], &mut foreign),
        Err(PcuExecutionError::Argument(
            PcuArgumentError::SessionMismatch
        ))
    ));
    pcu_facade::global::use_defaults().unwrap();
}
#[test]
#[ignore = "Requires actual Metal requested Portable low-format source and exact broadcasts."]
fn requested_portable_host_policy_broadcast_bits_faults_and_negative_profiles() {
    let _policy_guard = crate::source_policy_guard();
    host::<PcuF16Bits>();
    host::<PcuBf16Bits>();
    host::<PcuF8E4M3FnBits>();
    host::<PcuF8E5M2Bits>();
}
#[test]
#[ignore = "Requires actual Metal Portable resident/mixed publication and exact session proof."]
fn requested_portable_resident_mixed_broadcast_discard_and_retry() {
    let _policy_guard = crate::source_policy_guard();
    resident::<PcuF16Bits>();
    resident::<PcuBf16Bits>();
    resident::<PcuF8E4M3FnBits>();
    resident::<PcuF8E5M2Bits>();
}
