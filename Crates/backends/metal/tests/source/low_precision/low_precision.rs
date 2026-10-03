//! Actual cold source specialization, exact byte extents and logical resident publication.
#[rustfmt::skip]
use pcu_facade::{
    pcu,
    PcuCheckedFloat,
    PcuF16Bits,
    PcuBf16Bits,
    PcuF8E4M3FnBits,
    PcuF8E5M2Bits,
    PcuHostArgument,
    PcuBindingRef,
    PcuExecutionError,
    PcuExecutionFaultKind,
    PcuArgumentError,
    PcuRuntimeDiscovery,
    PcuMemoryPoolId,
    PcuTensor,
};
use fusion_pcu_metal::MetalSession;
#[path = "../../../benches/checked_low_precision/source.rs"]
mod source;
pub trait Sample: PcuCheckedFloat {
    fn value(value: f32) -> Self;
    fn raw(bits: u16) -> Self;
    const NAN: u16;
    const MAX: u16;
}
macro_rules! sample {
    ($ty:ty,$word:ty,$nan:expr,$max:expr) => {
        impl Sample for $ty {
            fn value(value: f32) -> Self {
                Self::pcu_checked_from_f32(value).unwrap()
            }
            fn raw(bits: u16) -> Self {
                Self::from_bits(<$word>::try_from(bits).unwrap())
            }
            const NAN: u16 = $nan;
            const MAX: u16 = $max;
        }
    };
}
sample!(PcuF16Bits, u16, 0x7e00, 0x7bff);
sample!(PcuBf16Bits, u16, 0x7fc0, 0x7f7f);
sample!(PcuF8E4M3FnBits, u8, 0x7f, 0x7e);
sample!(PcuF8E5M2Bits, u8, 0x7e, 0x7b);
pub fn assert_bits<T: PcuCheckedFloat>(actual: &[T], expected: &[T]) {
    assert_eq!(
        PcuHostArgument::read(PcuBindingRef::new(0, 0), actual).bytes(),
        PcuHostArgument::read(PcuBindingRef::new(0, 0), expected).bytes()
    );
}
#[pcu(invocations=3,flag(allow_gradual_underflow),crate_path=::pcu_facade)]
fn gradual<T: PcuCheckedFloat>(left: &[T], right: &[T], output: &mut [T]) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[id] / right[id];
}
#[pcu(invocations=3,flag(reject_subnormal_result),crate_path=::pcu_facade)]
fn tight<T: PcuCheckedFloat>(left: &[T], right: &[T], output: &mut [T]) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[id] + right[id];
}
#[pcu(invocations=3,flag(clamp_range),crate_path=::pcu_facade)]
fn clamp<T: PcuCheckedFloat>(left: &[T], right: &[T], output: &mut [T]) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[id] * right[id];
}
#[pcu(invocations=3,flag(deterministic),crate_path=::pcu_facade)]
fn portable<T: PcuCheckedFloat>(left: &[T], right: &[T], output: &mut [T]) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[id] * right[id];
}
#[pcu(invocations=3,flag(native_compound),flag(backend_precision),crate_path=::pcu_facade)]
fn permissions<T: PcuCheckedFloat>(left: &[T], right: &[T], output: &mut [T]) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[id] * right[id];
}
#[allow(clippy::too_many_lines)] // One cold source lifecycle verifies four operations and independent policy protections.
fn host<T: Sample>() {
    let session = MetalSession::open(0).unwrap();
    let left = [1.0, 2.0, 3.0].map(T::value);
    let right = [2.0, 4.0, 6.0].map(T::value);
    let sentinel = T::value(91.0);
    let mut output = [sentinel; 5];
    let mut add = source::add_prepare::<T, 3, _>(&session).unwrap();
    let mut sub = source::sub_prepare::<T, 3, _>(&session).unwrap();
    let mut mul = source::mul_prepare::<T, 3, _>(&session).unwrap();
    let mut div = source::div_prepare::<T, 3, _>(&session).unwrap();
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
    }
    let before = output;
    assert!(
        div(
            &left,
            &[T::value(2.0), T::value(0.0), T::raw(T::NAN)],
            &mut output
        )
        .is_err()
    );
    assert_bits(&output, &before);
    assert!(matches!(div(&[T::raw(1);3],&[T::value(2.0);3],&mut output),
        Err(pcu_facade::PcuHostDispatchError::Backend(fusion_pcu_metal::MetalError::Arithmetic(fault)))
        if fault.invocation_id==0 && fault.kind==PcuExecutionFaultKind::ArithmeticUnderflow));
    assert_bits(&output, &before);
    assert!(div(&left[..1], &right, &mut output).is_err());
    assert_bits(&output, &before);
    div(&left, &right, &mut output).unwrap();
    let mut gradual = gradual_prepare::<T, _>(&session).unwrap();
    gradual(&[T::raw(1); 3], &[T::value(2.0); 3], &mut output).unwrap();
    assert_bits(
        &output,
        &[T::raw(0), T::raw(0), T::raw(0), sentinel, sentinel],
    );
    let mut tight = tight_prepare::<T, _>(&session).unwrap();
    assert!(tight(&[T::raw(1); 3], &[T::raw(0); 3], &mut output).is_err());
    let mut clamped = clamp_prepare::<T, _>(&session).unwrap();
    assert!(
        matches!(clamped(&[T::raw(T::MAX);3],&[T::value(2.0);3],&mut output),
        Err(pcu_facade::PcuHostDispatchError::Backend(fusion_pcu_metal::MetalError::Arithmetic(fault)))
        if fault.recovered&&fault.invocation_id==0&&fault.kind==PcuExecutionFaultKind::ArithmeticOverflow)
    );
    assert_bits(
        &output,
        &[
            T::raw(T::MAX),
            T::raw(T::MAX),
            T::raw(T::MAX),
            sentinel,
            sentinel,
        ],
    );
    portable_prepare::<T, _>(&session).unwrap()(&left, &right, &mut output).unwrap();
    let mut permissions = permissions_prepare::<T, _>(&session).unwrap();
    permissions(&left, &right, &mut output).unwrap();
    let before = output;
    assert!(
        matches!(permissions(&[T::raw(T::MAX);3],&[T::value(2.0);3],&mut output),
        Err(pcu_facade::PcuHostDispatchError::Backend(fusion_pcu_metal::MetalError::Arithmetic(fault)))
        if fault.invocation_id==0 && fault.kind==PcuExecutionFaultKind::ArithmeticOverflow)
    );
    assert_bits(&output, &before);
    div(&left, &right, &mut output).unwrap();
    let before = output;
    pcu_facade::global::configure(pcu_facade::global::PcuExecutionPolicy {
        backend: pcu_facade::global::PcuBackendChoice::Metal,
        ..pcu_facade::global::PcuExecutionPolicy::default()
    })
    .unwrap();
    source::div::<T, 3>(&left, &right, &mut output).unwrap();
    assert_bits(&output, &before);
    pcu_facade::global::configure(pcu_facade::global::PcuExecutionPolicy {
        backend: pcu_facade::global::PcuBackendChoice::Metal,
        numerical_options: pcu_facade::PcuNumericalOptions {
            reproducibility: pcu_facade::PcuReproducibility::PortableV1,
            ..pcu_facade::PcuNumericalOptions::default()
        },
        ..pcu_facade::global::PcuExecutionPolicy::default()
    })
    .unwrap();
    source::div::<T, 3>(&left, &right, &mut output).unwrap();
    assert_bits(&output, &before);
    pcu_facade::global::use_defaults().unwrap();
}
pub fn open_backend() -> fusion_pcu_metal::MetalOwnedDispatchBackend {
    let discovery = fusion_pcu_metal::MetalDiscovery::discover().unwrap();
    let mut providers = [pcu_facade::PcuProviderDescriptor {
        id: pcu_facade::PcuProviderId(0),
        generation: 0,
        backend: "",
        readiness: pcu_facade::PcuProviderReadiness {
            status: pcu_facade::PcuProviderStatus::Unavailable,
            reason: None,
        },
    }];
    discovery.providers(&mut providers).unwrap();
    discovery
        .open_owned_device(pcu_facade::PcuObjectRef {
            provider: providers[0].id,
            generation: providers[0].generation,
            kind: pcu_facade::PcuObjectKind::Device,
            id: 0,
        })
        .unwrap()
}
#[allow(clippy::too_many_lines)] // One native lifecycle distinguishes same/foreign affinity, preflight and discarded publication.
fn resident<T: Sample>() {
    let backend = open_backend();
    let pool = PcuMemoryPoolId(19);
    let sentinel = T::value(91.0);
    let buffer = backend
        .upload_buffer(pool, &[1.0, 2.0, 3.0, 81.0, 82.0].map(T::value))
        .unwrap();
    let right_buffer = backend
        .upload_buffer(pool, &[2.0, 4.0, 6.0, 83.0, 84.0].map(T::value))
        .unwrap();
    let output_buffer = backend.upload_buffer(pool, &[sentinel; 5]).unwrap();
    let mut direct = source::div_prepare_device::<T, 3, _>(&backend).unwrap();
    let mut raw_output = backend.upload_buffer(pool, &[sentinel; 5]).unwrap();
    direct(&buffer, &right_buffer, &mut raw_output).unwrap();
    let mut raw_result = [T::raw(0); 5];
    backend
        .download_buffer(pool, &raw_output, &mut raw_result)
        .unwrap();
    assert_bits(
        &raw_result,
        &[
            T::value(0.5),
            T::value(0.5),
            T::value(0.5),
            sentinel,
            sentinel,
        ],
    );
    let mut same_output =
        PcuTensor::from_device_buffer(backend.clone(), output_buffer, &[5]).unwrap();
    let same_right = PcuTensor::from_device_buffer(backend.clone(), right_buffer, &[5]).unwrap();
    let input = PcuTensor::from_device_buffer(backend, buffer, &[5]).unwrap();
    let mut copied = [T::raw(0); 5];
    input.read_into(&mut copied).unwrap();
    assert_bits(&copied, &[1.0, 2.0, 3.0, 81.0, 82.0].map(T::value));
    let backend = open_backend();
    let output_buffer = backend.upload_buffer(pool, &[sentinel; 5]).unwrap();
    let mut output = PcuTensor::from_device_buffer(backend, output_buffer, &[5]).unwrap();
    pcu_facade::global::configure(pcu_facade::global::PcuExecutionPolicy {
        backend: pcu_facade::global::PcuBackendChoice::Metal,
        ..pcu_facade::global::PcuExecutionPolicy::default()
    })
    .unwrap();
    let right = [2.0, 4.0, 6.0].map(T::value);
    source::div::<T, 3>(&input, &right, &mut copied).unwrap();
    assert_bits(
        &copied,
        &[
            T::value(0.5),
            T::value(0.5),
            T::value(0.5),
            T::value(81.0),
            T::value(82.0),
        ],
    );
    source::div::<T, 3>(&input, &same_right, &mut same_output).unwrap();
    same_output.read_into(&mut copied).unwrap();
    assert_bits(
        &copied,
        &[
            T::value(0.5),
            T::value(0.5),
            T::value(0.5),
            sentinel,
            sentinel,
        ],
    );
    // A separate independently opened output owner cannot migrate into the input session.
    assert!(source::div::<T, 3>(&input, &right, &mut output).is_err());
    output.read_into(&mut copied).unwrap();
    assert_bits(&copied, &[sentinel; 5]);
    source::div::<T, 3>(&[1.0, 2.0, 3.0].map(T::value), &right, &mut output).unwrap();
    output.read_into(&mut copied).unwrap();
    assert_bits(
        &copied,
        &[
            T::value(0.5),
            T::value(0.5),
            T::value(0.5),
            sentinel,
            sentinel,
        ],
    );
    assert!(source::div::<T, 3>(&[T::value(1.0); 1], &right, &mut output).is_err());
    output.read_into(&mut copied).unwrap();
    assert!(
        matches!(source::div::<T,3>(&[T::value(1.0);3],&[T::value(1.0),T::value(0.0),T::raw(T::NAN)],&mut output),Err(PcuExecutionError::ArithmeticFault(fault)) if fault.invocation_id==1 && fault.kind==PcuExecutionFaultKind::DivideByZero)
    );
    assert!(matches!(
        output.read_into(&mut copied),
        Err(PcuExecutionError::Argument(
            PcuArgumentError::ResidentValueDiscarded
        ))
    ));
    assert!(matches!(
        source::div::<T, 3>(&output, &right, &mut copied),
        Err(PcuExecutionError::Argument(
            PcuArgumentError::ResidentValueDiscarded
        ))
    ));
    let backend = open_backend();
    let fresh_buffer = backend.upload_buffer(pool, &[sentinel; 5]).unwrap();
    let mut fresh = PcuTensor::from_device_buffer(backend, fresh_buffer, &[5]).unwrap();
    source::div::<T, 3>(&[1.0, 2.0, 3.0].map(T::value), &right, &mut fresh).unwrap();
    fresh.read_into(&mut copied).unwrap();
    assert_bits(
        &copied,
        &[
            T::value(0.5),
            T::value(0.5),
            T::value(0.5),
            sentinel,
            sentinel,
        ],
    );
    pcu_facade::global::use_defaults().unwrap();
}
#[test]
#[ignore = "Requires actual Metal half/BF16/OFP8 ordinary/prepared source policy proof."]
fn all_low_formats_source_host_odd_extent_fault_policy_and_retry() {
    let _policy_guard = crate::source_policy_guard();
    host::<PcuF16Bits>();
    host::<PcuBf16Bits>();
    host::<PcuF8E4M3FnBits>();
    host::<PcuF8E5M2Bits>();
}
#[test]
#[ignore = "Requires actual Metal low-format typed resident/mixed source and discard proof."]
fn all_low_formats_resident_readback_mixed_source_affinity_and_discard() {
    let _policy_guard = crate::source_policy_guard();
    resident::<PcuF16Bits>();
    resident::<PcuBf16Bits>();
    resident::<PcuF8E4M3FnBits>();
    resident::<PcuF8E5M2Bits>();
}
